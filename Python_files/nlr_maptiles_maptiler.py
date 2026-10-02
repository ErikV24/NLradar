# Copyright (C) 2016-2024 Bram van 't Veen, bramvtveen94@hotmail.com
# Additions (c) 2026: Stadia Maps live basemap support with Mercator -> equirectangular reprojection.
# Distributed under the GNU General Public License version 3, see <https://www.gnu.org/licenses/>.

"""
Live, scrollable basemap support via a web tile provider, as an alternative to the bundled, pre-rendered
local tiles handled by nlr_maptiles.py. Three providers are supported, chosen via Settings -> Map -> "Live map
provider" in the GUI (self.gui.basemap_source_maptiler_provider):
  - 'esri' (default): Esri's "Dark Gray Canvas" basemap, publicly accessible, no API key required.
  - 'esri_street': Esri's "World Street Map" basemap, publicly accessible, no API key required.
  - 'stadia': Stadia Maps' "Alidade Smooth Dark" style, requires a free personal API key.

Key technical point: all three providers (like virtually all web map tile providers) serve tiles in the Web
Mercator projection (EPSG:3857), in the standard XYZ tile scheme (Esri's URL puts the z/y/x path segments in
a different order than Stadia, but the underlying tile indices are identical). The bundled local tiles
handled by nlr_maptiles.py,
on the other hand, are stored as a plain equirectangular grid -- i.e. row/column position in self.map_data is
directly proportional to latitude/longitude, with no projection baked in. NLradar's existing rendering
pipeline (Plotting.draw_map_tiles, the 'map' visual's transform chain, and
nlr_customvispy.LatLon_to_Azimuthal_Equidistant_Transform) takes that equirectangular self.map_data and
projects it to the Azimuthal Equidistant (AEQD) projection centered on the current radar, *on the GPU*,
every time the view is drawn.

This module must therefore hand over data in that same plain equirectangular format -- NOT pre-projected to
AEQD -- so that the existing GPU transform chain can do the AEQD projection exactly once.

So the only reprojection actually needed here is Mercator -> equirectangular: for every pixel of the *output*
lat/lon grid, compute its position within the downloaded Mercator tile mosaic, and sample (bilinearly) the
colour at that position. This is done efficiently with cv2.remap, not a per-pixel Python loop.

This is a deliberately minimal version -- no black-point/gamma/sharpening post-processing -- kept simple so
it's easy to confirm the core pipeline (fetch, cache, reproject) works before layering visual tuning back on.
"""

import os
import threading
import numpy as np
import cv2
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from PyQt5.QtCore import QThread, pyqtSignal

import nlr_globalvars as gv

opa = os.path.abspath


def latlon_to_mercator_tile_xy(lat, lon, zoom):
    """Returns fractional (x, y) tile coordinates (not necessarily integer) for a lat/lon at a given zoom level,
    using the standard slippy-map / Web Mercator XYZ tile scheme."""
    lat = np.asarray(lat, dtype='float64')
    lon = np.asarray(lon, dtype='float64')
    lat, lon = np.broadcast_arrays(lat, lon)
    lat = np.clip(lat, -85.0511, 85.0511)  # Web Mercator is undefined beyond this latitude
    lat_rad = np.radians(lat)
    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - np.log(np.tan(lat_rad) + 1.0 / np.cos(lat_rad)) / np.pi) / 2.0 * n
    return x, y


def choose_zoom_level(lat_min, lat_max, lon_min, lon_max, pixel_width, pixel_height, tile_size=256, max_zoom=18):
    """Pick the smallest zoom level (i.e. least detailed/fewest tiles) for which the resulting mosaic still has
    at least as many pixels as the requested output raster, in both dimensions."""
    for zoom in range(0, max_zoom + 1):
        x0, y0 = latlon_to_mercator_tile_xy(lat_max, lon_min, zoom)  # top-left (lat_max = north = top)
        x1, y1 = latlon_to_mercator_tile_xy(lat_min, lon_max, zoom)  # bottom-right
        mosaic_w = (x1 - x0) * tile_size
        mosaic_h = (y1 - y0) * tile_size
        if mosaic_w >= pixel_width and mosaic_h >= pixel_height:
            return zoom
    return max_zoom


class MapTilesMapTiler(QThread):
    """Drop-in alternative to nlr_maptiles.MapTiles for use when self.gui.basemap_source == 'MapTiler'. Exposes
    the same public interface (set_radar_and_mapbounds, run, run_outside_thread, finished_signal) so that
    Plotting.update_map_tiles can use either tile source interchangeably."""

    finished_signal = pyqtSignal(np.ndarray, dict)

    TILE_SIZE = 256
    REQUEST_TIMEOUT_S = 3
    MAX_TILES_PER_AXIS = 12
    MAX_PARALLEL_FETCHES = 24

    @property
    def TILE_PROVIDER(self):
        # Reads live from the GUI setting (Settings -> Map -> "Live map provider"), so switching between
        # Esri and Stadia in the GUI takes effect on the next tile fetch without needing to edit this file.
        provider = getattr(self.gui, 'basemap_source_maptiler_provider', 'esri')
        if gv.frozen and provider == 'stadia':
            return 'esri'  # gedeelde versie: geen kaarten met API-key
        return provider

    # Stadia Maps' "Alidade Smooth Dark" style. Free for non-commercial/evaluation use, requires a personal
    # API key -- sign up at https://client.stadiamaps.com/signup/ (no credit card needed), create a Property,
    # add an API key under "Authentication Configuration", and paste it below.
    STADIA_STYLE = 'alidade_smooth_dark'
    STADIA_API_KEY = ''  # <-- vul hier je eigen, gratis Stadia Maps API key in

    # Esri's "Dark Gray Canvas" basemap. Publiek toegankelijk, geen API key nodig. Legacy ArcGIS Online
    # service -- werkt al jaren stabiel voor licht/matig gebruik, maar zonder garantie tegen toekomstige
    # wijzigingen door Esri. Let op de {z}/{y}/{x}-padvolgorde hieronder: dat wijkt af van de gebruikelijke
    # {z}/{x}/{y}-volgorde die Stadia/OSM/vrijwel alle andere providers gebruiken -- de tegelindices zelf
    # (tx/ty) zijn wel gewoon de standaard Web Mercator XYZ-indices, alleen de URL zet ze in een andere
    # volgorde neer.
    ESRI_DARK_GRAY_BASE_PATH = 'Canvas/World_Dark_Gray_Base'
    ESRI_DARK_GRAY_REFERENCE_PATH = 'Canvas/World_Dark_Gray_Reference'  # labels+roads, transparant PNG, bovenop base
    ESRI_STREET_MAP_PATH = 'World_Street_Map'  # kleurrijke straatkaart, alles-in-1 (geen aparte reference-laag)
    # Extra key-loze Esri-kaarten (zelfde openbare server, geen account/key):
    ESRI_LIGHT_GRAY_BASE_PATH = 'Canvas/World_Light_Gray_Base'
    ESRI_LIGHT_GRAY_REFERENCE_PATH = 'Canvas/World_Light_Gray_Reference'  # labels, transparant PNG
    ESRI_IMAGERY_PATH = 'World_Imagery'  # satellietbeeld
    ESRI_IMAGERY_REFERENCE_PATH = 'Reference/World_Boundaries_and_Places'  # grenzen+plaatsnamen, transparant
    ESRI_TOPO_PATH = 'World_Topo_Map'  # topografische kaart met reliëf, alles-in-1
    # Per provider: (lagen, max zoom). De eerste laag is de basis, eventuele tweede laag een transparante overlay.
    ESRI_LAYERS = {'esri':         ((ESRI_DARK_GRAY_BASE_PATH, ESRI_DARK_GRAY_REFERENCE_PATH), 16),
                   'esri_street':  ((ESRI_STREET_MAP_PATH,), 18),
                   'esri_light':   ((ESRI_LIGHT_GRAY_BASE_PATH, ESRI_LIGHT_GRAY_REFERENCE_PATH), 16),
                   'esri_imagery': ((ESRI_IMAGERY_PATH, ESRI_IMAGERY_REFERENCE_PATH), 18),
                   'esri_topo':    ((ESRI_TOPO_PATH,), 18)}
    # Gedeelde versie: externe kaartbronnen verwijderd (zie nlr.py, basemap_source altijd 'Local').
    ESRI_URL_TEMPLATE = ''

    def __init__(self, pb_class):
        QThread.__init__(self)
        self.pb = pb_class
        self.gui = pb_class.gui

        self.radar = None
        self.lat_min = self.lat_max = self.lon_min = self.lon_max = None
        self.starting = True
        self.last_error = None  # Set to a human-readable string on failure, so the GUI can surface it.

        self.cache_dir = opa(gv.userdir + '/Generated_files/stadia_tile_cache')
        os.makedirs(self.cache_dir, exist_ok=True)

        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=self.MAX_PARALLEL_FETCHES,
                                                 pool_maxsize=self.MAX_PARALLEL_FETCHES)
        self.session.mount('https://', adapter)

        self.tilebounds = None
        self.filenames_tilegrid_before = None
        self.map = None
        self.map_before = None
        # Laatst opgehaalde Mercator-mozaïek + de kaartuitsnede waarvoor self.map gemaakt is. Nodig om bij
        # zoomen/pannen binnen dezelfde tegels opnieuw te projecteren i.p.v. een oud beeld te houden.
        self._mosaic = None
        self._render_key_before = None
        # BUGFIX 28 sep 2026: voorkomt dat de achtergrond-thread en de hoofd-thread tegelijk get_tiles draaien.
        self._lock = threading.Lock()

        self.finished_signal.connect(self.pb.draw_map_tiles)

    def set_radar_and_mapbounds(self, radar, lat_min, lat_max, lon_min, lon_max):
        self.radar = radar
        self.lat_min, self.lat_max, self.lon_min, self.lon_max = lat_min, lat_max, lon_min, lon_max

    def run(self):
        try:
            # BUGFIX 28 sep 2026: als de gebruiker zoomt/schuift terwijl er nog gedownload wordt, wijzigt de
            # uitsnede halverwege. Vroeger werden de tegels dan voor de oude uitsnede opgehaald maar op de nieuwe
            # geprojecteerd (zwarte, 'afgesneden' randen). Nu werkt get_tiles met een vaste momentopname van de
            # uitsnede, en wordt hier opnieuw opgehaald zolang de uitsnede tijdens het ophalen veranderd is.
            for _ in range(5):
                bounds_before = (self.lat_min, self.lat_max, self.lon_min, self.lon_max)
                map_bounds, tiles_changed = self.get_tiles()
                if tiles_changed:
                    self.finished_signal.emit(self.map.copy(), map_bounds)
                if (self.lat_min, self.lat_max, self.lon_min, self.lon_max) == bounds_before:
                    break
        except Exception as e:
            self.last_error = str(e)
            print('MapTilesMapTiler.run failed:', e)

    def run_outside_thread(self):
        try:
            map_bounds, _ = self.get_tiles()
        except Exception as e:
            self.last_error = str(e)
            print('MapTilesMapTiler.run_outside_thread failed:', e)
            map_bounds = {'lat': [self.lat_min, self.lat_max], 'lon': [self.lon_min, self.lon_max]}
            if self.map is None:
                self.map = np.zeros((1, 1, 3), dtype='uint8')
        return self.map, map_bounds

    # ------------------------------------------------------------------
    # Mercator tile fetching
    # ------------------------------------------------------------------
    def _tile_cache_path(self, zoom, tx, ty, path_or_style):
        safe_key = path_or_style.replace('/', '_')
        return opa(self.cache_dir + f'/{self.TILE_PROVIDER}_{safe_key}_{zoom}_{tx}_{ty}.png')

    def _tile_url(self, zoom, tx, ty, path_or_style):
        if self.TILE_PROVIDER in self.ESRI_LAYERS:
            return self.ESRI_URL_TEMPLATE.format(path=path_or_style, z=zoom, x=tx, y=ty)
        return (f'{path_or_style}/{zoom}/{tx}/{ty}.png'
                f'?api_key={self.STADIA_API_KEY}')

    def _fetch_tile(self, zoom, tx, ty, path_or_style, want_alpha=False):
        """Returns a (TILE_SIZE, TILE_SIZE, 3 or 4) uint8 array for the given XYZ tile, using an on-disk cache
        to avoid re-downloading the same tile repeatedly during a session (or across sessions). want_alpha=True
        preserves the alpha channel (used for Esri's transparent labels/roads reference layer)."""
        cache_path = self._tile_cache_path(zoom, tx, ty, path_or_style)
        n_channels = 4 if want_alpha else 3
        if os.path.exists(cache_path):
            img = cv2.imread(cache_path, cv2.IMREAD_UNCHANGED if want_alpha else cv2.IMREAD_COLOR)
            if img is not None and img.shape[:2] == (self.TILE_SIZE, self.TILE_SIZE) and img.shape[2] == n_channels:
                return img[:, :, [2, 1, 0, 3][:n_channels]]  # BGR(A) -> RGB(A)

        provider_label = 'Esri' if self.TILE_PROVIDER in self.ESRI_LAYERS else 'Stadia Maps'

        if self.TILE_PROVIDER == 'stadia' and (not self.STADIA_API_KEY or self.STADIA_API_KEY == 'JOUW_API_KEY_HIER'):
            raise Exception("Geen Stadia Maps API key ingevuld -- zet je (gratis) key in STADIA_API_KEY "
                             "bovenin nlr_maptiles_maptiler.py. Aanmelden: "
                             "https://client.stadiamaps.com/signup/")

        url = self._tile_url(zoom, tx, ty, path_or_style)
        response = self.session.get(url, timeout=self.REQUEST_TIMEOUT_S)
        if response.status_code != 200:
            if response.status_code in (401, 403):
                raise Exception(f"{provider_label} wees het verzoek af (HTTP {response.status_code})"
                                 + ("" if self.TILE_PROVIDER != 'stadia' else
                                    " -- controleer of STADIA_API_KEY correct is ingevuld."))
            elif response.status_code == 429:
                raise Exception(f"{provider_label}: limiet bereikt (HTTP 429).")
            elif response.status_code == 404:
                raise Exception(f"Tegel niet gevonden (HTTP 404) bij {provider_label}, stijl '{path_or_style}', "
                                 f"zoom={zoom}, tile=({tx},{ty}).")
            else:
                raise Exception(f"{provider_label}-verzoek mislukt met HTTP {response.status_code}: "
                                 f"{response.text[:200]}")

        img_array = np.frombuffer(response.content, dtype='uint8')
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED if want_alpha else cv2.IMREAD_COLOR)
        if img is None or img.shape[:2] != (self.TILE_SIZE, self.TILE_SIZE) or img.shape[2] != n_channels:
            raise Exception(f"{provider_label} gaf een onleesbare of verkeerd-formaat afbeelding terug voor "
                             f"stijl '{path_or_style}' bij zoom={zoom}, tile=({tx},{ty}). "
                             f"Response begon met: {response.content[:50]!r}")

        cv2.imwrite(cache_path, img)
        return img[:, :, [2, 1, 0, 3][:n_channels]]  # BGR(A) -> RGB(A)

    def _fetch_mercator_mosaic(self, zoom, tx_min, tx_max, ty_min, ty_max, style, want_alpha=False):
        """Downloads (or reads from cache) all tiles in the given XYZ range and assembles them into one mosaic
        image. tx_max/ty_max are inclusive."""
        n_channels = 4 if want_alpha else 3
        n_tiles_side = 2 ** zoom
        ntx = tx_max - tx_min + 1
        nty = ty_max - ty_min + 1
        mosaic = np.zeros((nty * self.TILE_SIZE, ntx * self.TILE_SIZE, n_channels), dtype='uint8')

        jobs = []
        for i, tx in enumerate(range(tx_min, tx_max + 1)):
            tx_wrapped = tx % n_tiles_side  # Longitude wraps around the date line; XYZ tile x must too.
            for j, ty in enumerate(range(ty_min, ty_max + 1)):
                if ty < 0 or ty >= n_tiles_side:
                    continue  # No valid tiles beyond the poles.
                jobs.append((i, j, tx_wrapped, ty))

        with ThreadPoolExecutor(max_workers=self.MAX_PARALLEL_FETCHES) as executor:
            future_to_job = {
                executor.submit(self._fetch_tile, zoom, tx_wrapped, ty, style, want_alpha): (i, j, tx_wrapped, ty)
                for (i, j, tx_wrapped, ty) in jobs
            }
            for future in as_completed(future_to_job):
                i, j, tx_wrapped, ty = future_to_job[future]
                try:
                    tile = future.result()
                    mosaic[j * self.TILE_SIZE:(j + 1) * self.TILE_SIZE, i * self.TILE_SIZE:(i + 1) * self.TILE_SIZE] = tile
                except Exception as e:
                    print(f'MapTilesMapTiler: tile fetch failed for zoom={zoom} tile=({tx_wrapped},{ty}), '
                          f'leaving that area blank: {e}')
        return mosaic

    def _composite_alpha(self, base_rgb, overlay_rgba):
        """Alpha-blends overlay_rgba (e.g. Esri's transparent labels/roads reference layer) on top of
        base_rgb, both (H, W, ...) uint8 mosaics of identical size. Returns a 3-channel uint8 mosaic."""
        alpha = overlay_rgba[:, :, 3:4].astype('float32') / 255.0
        overlay_rgb = overlay_rgba[:, :, :3].astype('float32')
        base = base_rgb.astype('float32')
        blended = base * (1 - alpha) + overlay_rgb * alpha
        return np.clip(blended, 0, 255).astype('uint8')

    # ------------------------------------------------------------------
    # Mercator -> simple equirectangular lat/lon grid
    # ------------------------------------------------------------------
    def _reproject_to_latlon_grid(self, mosaic, zoom, tx_min, ty_min, out_width, out_height, bounds):
        """Inverse-warps the Mercator tile mosaic into a plain equirectangular lat/lon raster covering
        self.lat_min/max/lon_min/max, with shape (out_height, out_width, 3) -- i.e. row/column position is
        directly proportional to latitude/longitude, with NO further projection applied here."""
        lat_min, lat_max, lon_min, lon_max = bounds
        lat_grid = np.linspace(lat_max, lat_min, out_height).reshape(-1, 1) * np.ones((1, out_width))
        lon_grid = np.linspace(lon_min, lon_max, out_width).reshape(1, -1) * np.ones((out_height, 1))

        tile_x, tile_y = latlon_to_mercator_tile_xy(lat_grid, lon_grid, zoom)
        map_x = ((tile_x - tx_min) * self.TILE_SIZE).astype('float32')
        map_y = ((tile_y - ty_min) * self.TILE_SIZE).astype('float32')

        reprojected = cv2.remap(mosaic, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        return reprojected

    # ------------------------------------------------------------------
    # Public entry point, mirroring nlr_maptiles.MapTiles.get_tiles
    # ------------------------------------------------------------------
    def get_tiles(self):
        with self._lock:
            return self._get_tiles_for(self.lat_min, self.lat_max, self.lon_min, self.lon_max)

    # Extra marge rond de zichtbare uitsnede (fractie van de breedte/hoogte per kant). BUGFIX 28 sep 2026:
    # zonder marge eindigde de kaart precies op de schermrand, waardoor kleine afwijkingen in schaal of de
    # kromming van de radarprojectie smalle zwarte randjes in de hoeken gaven. Met marge valt de kaartrand
    # buiten beeld. Bijkomend voordeel: kleine schuifbewegingen vragen geen nieuwe download.
    MAP_MARGIN = 0.08

    def _get_tiles_for(self, lat_min, lat_max, lon_min, lon_max):
        if None in (lat_min, lat_max, lon_min, lon_max, self.radar):
            raise Exception('MapTilesMapTiler.get_tiles called before set_radar_and_mapbounds.')
        dlat = (lat_max - lat_min) * self.MAP_MARGIN; dlon = (lon_max - lon_min) * self.MAP_MARGIN
        lat_min, lat_max = max(lat_min - dlat, -85.0), min(lat_max + dlat, 85.0)
        lon_min, lon_max = lon_min - dlon, lon_max + dlon

        screen_width = max(int(self.pb.wsize['main'][0]), 1)
        screen_height = max(int(self.pb.wsize['main'][1]), 1)
        lat_span = lat_max - lat_min
        lon_span = lon_max - lon_min
        if lon_span > 0 and lat_span > 0:
            height_from_width = max(int(round(screen_width * lat_span / lon_span)), 1)
            width_from_height = max(int(round(screen_height * lon_span / lat_span)), 1)
            if screen_width * height_from_width >= width_from_height * screen_height:
                out_width, out_height = screen_width, height_from_width
            else:
                out_width, out_height = width_from_height, screen_height
        else:
            out_width, out_height = screen_width, screen_height

        provider_max_zoom = self.ESRI_LAYERS[self.TILE_PROVIDER][1] if self.TILE_PROVIDER in self.ESRI_LAYERS else 18
        zoom = choose_zoom_level(lat_min, lat_max, lon_min, lon_max,
                                  out_width, out_height, self.TILE_SIZE, max_zoom=provider_max_zoom)

        tx0, ty0 = latlon_to_mercator_tile_xy(lat_max, lon_min, zoom)
        tx1, ty1 = latlon_to_mercator_tile_xy(lat_min, lon_max, zoom)
        tx_min, tx_max = int(np.floor(tx0)), int(np.floor(tx1))
        ty_min, ty_max = int(np.floor(ty0)), int(np.floor(ty1))

        if (tx_max - tx_min + 1) > self.MAX_TILES_PER_AXIS or (ty_max - ty_min + 1) > self.MAX_TILES_PER_AXIS:
            while zoom > 0 and ((tx_max - tx_min + 1) > self.MAX_TILES_PER_AXIS or
                                 (ty_max - ty_min + 1) > self.MAX_TILES_PER_AXIS):
                zoom -= 1
                tx0, ty0 = latlon_to_mercator_tile_xy(lat_max, lon_min, zoom)
                tx1, ty1 = latlon_to_mercator_tile_xy(lat_min, lon_max, zoom)
                tx_min, tx_max = int(np.floor(tx0)), int(np.floor(tx1))
                ty_min, ty_max = int(np.floor(ty0)), int(np.floor(ty1))

        if self.TILE_PROVIDER in self.ESRI_LAYERS:
            style_key = self.ESRI_LAYERS[self.TILE_PROVIDER][0]
        else:
            style_key = (self.STADIA_STYLE,)
        tilegrid_key = (zoom, tx_min, tx_max, ty_min, ty_max, self.TILE_PROVIDER, style_key)
        map_bounds = {'lat': [lat_min, lat_max], 'lon': [lon_min, lon_max]}
        render_key = (lat_min, lat_max, lon_min, lon_max, out_width, out_height)
        if not self.starting and tilegrid_key == self.filenames_tilegrid_before and self._mosaic is not None:
            # BUGFIX 28 sep 2026: zelfde tegels, maar de uitsnede kan veranderd zijn (zoomen/pannen). Vroeger werd
            # hier niets gedaan, waardoor het oude beeld bleef staan en er na zoomen zwarte stukken aan de randen
            # ontstonden. Nu opnieuw projecteren vanuit de bewaarde mozaïek (snel, geen downloads).
            if render_key == self._render_key_before:
                return map_bounds, False
            self.map = self._reproject_to_latlon_grid(self._mosaic, zoom, tx_min, ty_min, out_width, out_height,
                                                   (lat_min, lat_max, lon_min, lon_max))
            self._render_key_before = render_key
            return map_bounds, True

        if self.TILE_PROVIDER in self.ESRI_LAYERS:
            # Basislaag, plus eventueel een transparante overlay (labels/grenzen) erbovenop.
            layers = self.ESRI_LAYERS[self.TILE_PROVIDER][0]
            mosaic = self._fetch_mercator_mosaic(zoom, tx_min, tx_max, ty_min, ty_max, layers[0])
            if len(layers) > 1:
                reference_mosaic = self._fetch_mercator_mosaic(zoom, tx_min, tx_max, ty_min, ty_max,
                                                                 layers[1], want_alpha=True)
                mosaic = self._composite_alpha(mosaic, reference_mosaic)
        else:
            mosaic = self._fetch_mercator_mosaic(zoom, tx_min, tx_max, ty_min, ty_max, self.STADIA_STYLE)
        self.map = self._reproject_to_latlon_grid(mosaic, zoom, tx_min, ty_min, out_width, out_height,
                                                   (lat_min, lat_max, lon_min, lon_max))
        self._mosaic = mosaic
        self._render_key_before = render_key

        self.filenames_tilegrid_before = tilegrid_key
        self.starting = False
        self.last_error = None

        map_bounds = {'lat': [lat_min, lat_max], 'lon': [lon_min, lon_max]}
        return map_bounds, True
