# Copyright (C) 2016-2024 Bram van 't Veen, bramvtveen94@hotmail.com
# Distributed under the GNU General Public License version 3, see <https://www.gnu.org/licenses/>.

# Installeerbare versie: werkmap meteen op Python_files zetten, vóór alle andere imports (Brams code leest
# Tables/_data/gifsicle relatief t.o.v. de werkmap). Bij 'python nlr.py' verandert hier niets.
import sys as _sys, os as _os
if getattr(_sys, 'frozen', False):
    _os.chdir(_os.path.join(_os.path.dirname(_sys.executable), 'Python_files'))

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
# When setting QtCore.Qt.AA_EnableHighDpiScaling to True, the variable screen.devicePixelRatio() would be needed in the plotting code
# QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
# QApplication.setAttribute(QtCore.Qt.HighDpiScaleFactorRoundingPolicy, Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

from vispy import gloo

import sys
import numpy as np
import traceback
from numpy import array, float32 # For use of eval
import os
# os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1" #I haven't yet seen this doing anything
opa=os.path.abspath #opa should always be used when specifying a pathname
import subprocess
import glob
import time as pytime
from PIL import Image
import av
from fractions import Fraction
import shutil
import pickle
import copy
import bisect
import pyperclip

from nlr_plotting import Plotting
from nlr_changedata import Change_RadarData
from nlr_currentdata import AutomaticDownload, DownloadOlderData, CurrentData
from VWP.nlr_vwp import GUI_VWP
import nlr_background as bg
import nlr_functions as ft
import nlr_globalvars as gv
import nlr_meltinglevels as nlr_ml



"""
General structure of the code:

The class GUI (gui) contains all the GUI-related code. Plotting (pb) performs all basic plotting actions (in VisPy), 
and Change_RadarData (crd) contains the part of the code that processes changes in time, radar or products. CurrentData (cd)
checks whether a download is needed in the case of current data, and downloads it when that's the case.
DataSource_Specific (dsg) is the class that handles all things that are specific to a particular data source, like the import of radar data
DerivedProducts (dp) contains the functions in which derived products are created.

There is interaction between the different classes, and in a particular class another class is referred to as 
self.[classabbreviation]. 
All variables and methods used in the code 'live' in a particular class, and when it is used in another class it is therefore
referred to as self.[classabbreviation].[variablename/methodname]. Most of the changes to a particular variable take place in its
living class, but there are some exceptions.
Variables that are predefined here below (and which are saved to the settings.pkl file) all have GUI as their living class,
with the exception of a number of variables that belong to/live in the crd class. These variables are indicated below.

nlr_functions contains small functions that can be used in general, also outside the code for this program.
nlr_background contains larger functions or functions that are more specific for use in this program.
nlr_globalvars contains global parameters that don't change within the application. 
"""



#Initialization of variables, mostly saved from a previous run of the application. 
#At first, each variable is assigned its default value (defined in nlr_globalvars), but if they are saved from a previous run this value is
#subsequently overwritten.

radar_basedir = gv.default_basedir
radarsources_dirs = {}
radardirs_additional = {}
radardata_dirs={}
radardata_dirs_indices={}
derivedproducts_dir=gv.derivedproducts_dir_Default
colortables_dirs_filenames={}
savefig_filename=gv.userdir+'/'
savefig_include_menubar = False
animation_filename=gv.userdir+'/'
ani_delay_ref = 'frame'
ani_delay = 25
ani_delay_end = 50
ani_sort_files = True
ani_group_datasets = False
ani_quality = 5

radardata_product_versions={} # Sometimes more than 1 version of a product is available (as determined in self.dsg.get_product_versions).
# This dictionary contains for each radar-dataset a string that identifies the currently selected product version.
selected_product_versions_ordered=[] # Ordered list of product versions that have been selected within the function 
# self.crd.change_product_version, with the most recently selected placed at the end. It contains each product version at most once, which means
# that earlier occurrences are removed from the list. It is used to ensure that a switch of product version persists when
# switching from radar/dataset. It is needed, because a switch of product version within self.crd.change_product_version only affects a single
# radar-dataset.
derivedproducts_filename_version=None


movefiles_parameters={'startdate':'','starttime':'','enddate':'','endtime':'','oldstructure':'','newstructure':''}
      
"""The variables in this section belong to/live in the crd class."""  
radar='Herwijnen'
scan_selection_mode='scanangle'
date='c'
time='c'
current_case_list_name=None
current_case_list=None
current_case=None
cases_offset_minutes=0
cases_looping_speed=0.5
cases_animation_window=np.array([-30, 15])
cases_use_case_zoom=True
cases_loop_subset=False
cases_loop_subset_ncases=10
dataset='V'
products=['z','z','z','z','z','v','v','v','v','v']
productunfiltered={j:False for j in range(10)}
polarization={j:'H' for j in range(10)}
apply_dealiasing = {j:True for j in range(10)}
dealiasing_setting = gv.velocity_dealiasing_settings[-1]
dealiasing_max_nyquist_vel = 100/gv.scale_factors_velocities['kts']
dealiasing_dualprf_n_it = 50
# MESH-kalibratie-instelling (24 juli, zie select_mesh_calibration_settings/change_mesh_calibration_setting
# hieronder) - zelfde opslag-/laadpatroon als dealiasing_setting hierboven (module-level default, wordt
# bij het laden van stored_settings.pkl overschreven via variables_names_raw/exec, zie hoger in dit bestand).
mesh_calibration_setting = gv.mesh_calibration_settings[0]
# Handmatige stationskeuze voor het Wyoming-sounding-archief (25 juli, op Eriks verzoek) -
# terugvaloptie naast de automatische, dichtstbijzijnde-tijd-keuze in nlr_meltinglevels.py
# (_get_melting_levels_from_wyoming). Zelfde opslag-/laadpatroon als mesh_calibration_setting
# hierboven. melting_levels_manual_date is een string 'YYYY-MM-DD' (leeg = nog niet ingevuld).
melting_levels_manual_override = False
melting_levels_manual_station = list(nlr_ml.WYOMING_STATIONS.keys())[0]
melting_levels_manual_date = ''
melting_levels_manual_hour = 0
# ZPHI-verzwakkingscorrectie aan/uit (28 juli 2026, op Eriks verzoek, zie select_attenuation_correction_settings/
# change_attenuation_correction_enabled hieronder) - zelfde opslag-/laadpatroon als mesh_calibration_setting/
# melting_levels_manual_override hierboven. Standaard AAN.
attenuation_correction_enabled = True
cartesian_product_res = 1.
cartesian_product_maxrange = 200.
scans=[1,2,3,4,5,1,2,3,4,5]
plot_mode='Single'

animation_duration=60
animation_speed_minpsec=90
animation_hold_lastframe=0.5
desired_timestep_minutes=0
max_timestep_minutes=180 # Value of 1e10 corresponds to empty input in menu widget
maxspeed_minpsec=1e10 # Value of 1e10 corresponds to empty input in menu widget
networktimeout=60
minimum_downloadspeed=0.01
api_keys = {} # Default values are assigned below when necessary
pos_markers_latlons = []
pos_markers_positions = []
pos_markers_latlons_save = []
# Optionele tekst-labels bij positiemarkers (7 juli 2026), bv. voor gebruik als "waar bevind ik mij"-markering
# in de 3D volume-viewer. Losse, parallelle lijst t.o.v. pos_markers_latlons/positions (zelfde index=zelfde marker).
pos_markers_labels = []
stormmotion_save = {}
use_storm_following_view = False
view_nearest_radar = False
radar_bands_view_nearest_radar = ['S', 'C', 'X']
data_selected_startazimuth = 0 #Selected (desired) start azimuth for the scans for radars in gv.radars_with_adjustable_startazimuth
show_vwp = False
include_sfcobs_vwp = True
vwp_manual_sfcobs = {} 
vwp_manual_axlim = [] 
vvp_range_limits = [2, 25]
vvp_height_limits = [0.1, 11.9]
vvp_vmin_mps = 3.
vwp_sigmamax_mps = None
vwp_shear_layers = {1:[0,1], 2:[0,3], 3:[0,6], 4:[1,6]}
vwp_vorticity_layers = {1:[0,0.5], 2:[0,1]}
vwp_srh_layers = {1:[0,1], 2:[0,2], 3:[0,3]}
vwp_sm_display = {j:True for j in gv.vwp_sm_names if not j == 'SM'}

base_url_obs = 'https://kachelmannwetter.com/de/messwerte/18584bea4ec779beb796d3770ed37f8e/'

cmaps_minvalues={j:'' for j in gv.products_all}
for p in gv.products_all:
    if gv.i_p[p] == 'z':
        cmaps_minvalues[p] = -20
cmaps_maxvalues={j:'' for j in gv.products_all}

# Default parameters for the polarimetric RGB composite (product 'g'). See DataSource_General._calculate_polrgb
# for how these are used. Exposed in Settings -> PolRGB so they can be tuned without editing code.
polrgb_params_default = {
    'Z_MIN':-10.0, 'Z_MAX':60.0,
    'CC_MIN':70.0, 'CC_MAX':100.0,
    'ZDR_MIN':0.0, 'ZDR_MAX':3.0,
    'Z_FADE_LO':-15.0, 'Z_FADE_HI':10.0,
    'ALPHA_GAMMA':0.6,
    'CC_FALLBACK':97.0, 'ZDR_FALLBACK':0.5,
    'Z_GAMMA':2.0,  # >1: lage dBZ blijft donker, hoge dBZ snel rood (ESSL-stijl). 1.0=lineair (origineel).
    'ESSL_MODE':False,  # True: vaste (instelbare, zie hieronder) ESSL-tabel i.p.v. bovenstaande doorlopende
    # passthrough-parameters -- die worden dan genegeerd (zie DataSource_General._calculate_polrgb/
    # _essl_polrgb_channels). Uit = exact het oude, doorlopende gedrag.
    # ESSL-tabel zelf (16 september 2026, op Eriks verzoek instelbaar gemaakt -- Bram blijkt deze getallen
    # per publicatie/sessie te varieren, zie de posterbijlage vs. de conferentie-abstract vs. de live ESSL-
    # viewer). Standaardwaarden zijn EXPLICIET de posterbijlage (Van 't Veen/Groenemeijer/Pucik, ECSS 2025
    # Utrecht) -- ook wat je terugkrijgt bij 'Reset to defaults', per Eriks expliciete eis.
    'ESSL_Z_MIN':30.0, 'ESSL_Z_MAX':60.0,
    'ESSL_CC_MIN':70.0, 'ESSL_CC_MAX':100.0,
    'ESSL_ZDR_MIN':0.0, 'ESSL_ZDR_MAX':4.0,
    # 11-punts piecewise-lineaire alpha(Z)-curve, rechtstreeks uit de posterbijlage. Twee losse lijsten
    # (i.p.v. lijst van paren) zodat elke lijst zich als 1 los "veld" laat behandelen in de UI-tabel
    # hieronder -- Z moet strikt oplopend blijven (np.interp-eis), zie change_polrgb_essl_alpha.
    'ESSL_ALPHA_Z':[-10.0, 0.0, 10.0, 15.0, 20.0, 24.0, 28.0, 31.0, 34.0, 37.0, 40.0],
    'ESSL_ALPHA_V':[0.05, 0.12, 0.22, 0.29, 0.39, 0.48, 0.58, 0.67, 0.77, 0.88, 1.00],
}
polrgb_params = dict(polrgb_params_default)
polrgb_cc_hide_above=None # 2D PolRGB (product 'g'): pixels met CC BOVEN deze grens (%) worden helemaal niet
#getekend (volledig doorzichtig), ongeacht Z/ZDR (15 augustus 2026, op Eriks verzoek: hetzelfde als het al
#bestaande 3D-only CC-filter (volume3d_polrgb_cc_max), nu ook voor de gewone 2D-weergave). Los van/
#onafhankelijk van de CC_MIN/CC_MAX-kleurschaalinstellingen hierboven (die bepalen alleen HOE fel het groene
#kanaal kleurt, niet OF een pixel getekend wordt). None/leeg = geen grens (oorspronkelijk gedrag).

PP_parameter_values={}
PP_parameter_values['e']={1:18.5,2:30.,3:40.,4:50.}
PP_parameter_values['a']={1:0.3,2:0.5,3:1.5,4:3.}
PP_parameter_values['m']={1:0.5,2:1.5,3:5.,4:10.}
PP_parameter_values['h']={1:[True, 1], 2:[True, 3], 3:[True, 5], 4:[True, 10]}
PP_parameter_values['l']={1:[True, 0], 2:[False, 0], 3:[False, 0], 4:[False, 0]}
PP_parameters_panels={j:1 for j in range(10)}

max_radardata_in_memory_GBs=2
sleeptime_after_plotting=0.01

# Multiplier applied to the cross-section raster's base size (150 distance bins x 80 height bins, see
# show_cross_section in nlr_plotting.py) -- e.g. 2.0 gives a 300x160 raster. Exposed in Settings ->
# Miscellaneous (see settings_tabmiscellaneous) so it can be tuned live, without editing code, the same way
# polrgb_params is. 3.0 (450x240) was chosen as the shipped default: with the current 2000 line-samples per
# elevation scan (see get_cross_section in nlr_datasourcegeneral.py), that keeps a comfortable fill fraction
# even for radars with relatively few elevation scans -- going noticeably higher risks approaching the very
# sparse, speckled-looking raster that caused the original "stray horizontal dashes" bug (see the comment at
# the top of show_cross_section's raster-binning step for that history).
cross_section_resolution_factor=3.0

# Interpolation mode for the cross-section's ImageVisual (see nlr_plotting.py __init__, where visuals['cross_
# section'][i] is created) -- how the raster's individual bins get blended together when stretched up to the
# on-screen image size. 'bicubic' (the shipped default) looks the smoothest, at the cost of a small risk of a
# faint light/dark "overshoot" fringe right at hard edges (e.g. the boundary between real data and empty
# space); 'bilinear' is a safer, slightly less smooth middle ground; 'nearest' shows the raw raster bins as
# hard-edged squares (vispy's own default, and what this cross-section originally shipped with). Exposed in
# Settings -> Miscellaneous, same live-tunable pattern as cross_section_resolution_factor above.
cross_section_interpolation_mode='bicubic'

# Number of evenly-spaced sample points taken along the A/B line for EACH elevation scan (see get_cross_section
# in nlr_datasourcegeneral.py, called from show_cross_section in nlr_plotting.py). More samples fill a given
# raster more densely -- relevant mainly if cross_section_resolution_factor above is increased -- at the cost
# of a bit more time spent recomputing the cross-section. Exposed in Settings -> Miscellaneous, same live-
# tunable pattern as the other cross_section_* settings.
cross_section_n_samples=2000

# Extra vertical headroom above the 99th-percentile echo top, as a percentage, when picking the cross-section's
# vertical scale (see max_height_km in show_cross_section: a flat percentage margin gives a tall storm's
# overshooting top some empty space above it, rather than the top few rows of the raster). 15 means the scale's
# top sits 15% higher than that percentile height (with an unconditional floor of 8 km either way -- see
# show_cross_section). Higher shows more empty sky above the echo (more "zoomed out" vertically); lower zooms
# in more tightly on the echo itself. Exposed in Settings -> Miscellaneous, same live-tunable pattern.
cross_section_height_headroom_percent=15.0

# Shipped defaults for the 4 tunable cross-section settings above, kept together in one place so the
# Settings -> Miscellaneous "Reset to defaults" button (see reset_cross_section_settings) has a single source
# of truth to reset to, the same "defaults dict alongside the live values" pattern used for polrgb_params.
cross_section_settings_default = {
    'cross_section_resolution_factor': 3.0,
    'cross_section_interpolation_mode': 'bicubic',
    'cross_section_n_samples': 2000,
    'cross_section_height_headroom_percent': 15.0,
}

# Tunable settings for the 3D volume viewer (CTRL+SHIFT+4, show_volume_3d_viewer), same live-tunable
# pattern as the cross_section_* settings above -- exposed in Settings -> Miscellaneous. Unlike the
# cross-section (which is a persistent visual within the main app and re-renders immediately on change),
# the 3D viewer is a standalone window opened fresh each time, so these take effect the NEXT time it's
# opened rather than on any already-open window.
volume3d_grid_res_km=0.5 # Horizontal grid resolution (get_volume_grid) -- finer shows more detail but is
#slower to compute and render; coarser is quicker but blockier (partly offset by volume3d_smoothing_sigma).
volume3d_z_max_km=15.0 # Maximum height (km) included in the reconstructed grid.
volume3d_z_res_km=0.25 # Vertical grid resolution (km).
volume3d_vertical_exaggeration=5.0 # How many times taller the height axis is stretched on screen, since a
#storm's real height (a few km) would otherwise look almost flat next to its horizontal extent (tens of km).
volume3d_smoothing_sigma=1.2 # Strength of the horizontal (x/y) smoothing between neighbouring grid columns
#that turns the raw, blocky ("Minecraft") reconstruction into a smoother cloud shape -- see
#_smooth_volume_grid_horizontally in nlr_datasourcegeneral.py. 0 disables smoothing entirely (raw/blocky).
volume3d_tick_interval_km=10.0 # Spacing (km) between the ruler tick marks along the 4 base edges of the
#reference box.
volume3d_height_tick_interval_km=2.0 # Spacing (km, REAL height, before the exaggeration above) between the
#ruler tick marks on the vertical height ruler.
volume3d_gamma=1.0 # Gamma-correctie op de kleur-intensiteit van de 3D-volumedata (7 juli 2026, op Eriks
#verzoek voor een visueel "dieper"/contrastrijker beeld). 1.0 = geen aanpassing (oorspronkelijk gedrag).
#Lager dan 1 maakt middenwaarden feller/contrastrijker, hoger dan 1 dooft alles behalve de hoogste waarden.
#Puur een visuele kleurintensiteit-vertaalslag; de onderliggende data/waarden blijven ongewijzigd.
volume3d_pointcloud_stride=3 # Puntenwolk-weergave (8 juli 2026, op Eriks verzoek: "een dichte pixelweergave
#waar ik doorheen kan kijken", i.p.v. MIP/translucent die last hebben van opstapelende
#(on)doorzichtigheid). In plaats van een aaneengesloten oppervlak worden alleen losse roosterpunten
#getekend, met LEGE ruimte ertussen -- dat maakt echt "erdoorheen kijken" mogelijk. Deze waarde bepaalt
#de dichtheid: 1 = elk roosterpunt (dicht, traag), hoger = every-Nth-punt in elke richting (ijler, sneller).
volume3d_pointcloud_point_size=4.0 # Grootte (in beeldschermpixels) van elk punt in de puntenwolk-weergave.
volume3d_min_value=None # 3D-viewer: waarden ONDER deze grens worden helemaal niet getekend (in geen enkele
#weergavemodus -- MIP, translucent, of puntenwolk), i.p.v. alleen anders gekleurd (8 juli 2026, op Eriks
#verzoek: "de 3D weergave heeft duidelijk last van de lage dBZ waarden", net als de bestaande 2D-instelling
#voor kleurtabel-minimum/maximum). None/leeg = geen grens (oorspronkelijk gedrag). Eenheid is dezelfde als
#het product dat je bekijkt (bv. dBZ voor Z, m/s voor V).
volume3d_circular_area=False # 3D-viewer: als aangevinkt, wordt het gebied binnen de getekende rechthoek
#(CTRL+SHIFT+slepen) behandeld als een ELLIPS die precies in die rechthoek past (middelpunt = midden van de
#rechthoek, halve-assen = halve breedte/hoogte), i.p.v. de volle rechthoek zelf -- data buiten die ellips
#wordt weggemaskeerd (8 juli 2026, op Eriks verzoek: "zou dat ook een cirkel kunnen zijn"). Zowel de 2D-
#voorvertoning (de getekende rechthoek op de kaart zelf) als de 3D-data volgen deze instelling, zodat je
#vooraf al ziet welk gebied je krijgt. Het kader/de assen/tick-streepjes in 3D blijven altijd de volle,
#rechthoekige omvang tonen -- alleen de gekleurde data zelf wordt rond weggesneden.
volume3d_polrgb_cc_max=None # PolRGB-specifiek (15 augustus 2026, op Eriks verzoek: "dat groen van de regen
#wil ik kwijt" -- gewone regen heeft een hoge CC, ongeacht Z, dus volume3d_min_value (dat op Z filtert) helpt
#daar niet tegen): voxels met CC BOVEN deze grens (%) worden helemaal niet getekend, in GEEN enkele
#PolRGB-3D-weergavemodus (RGB-MIP of puntenwolk) -- ongeacht hun Z/ZDR. Omdat hagel per definitie een lagere
#CC heeft dan gewone regen, blijft het hagelgebied hierdoor onaangeroerd terwijl "zeker gewone regen"
#verdwijnt. Los van/onafhankelijk van volume3d_min_value hierboven (dat blijft op Z filteren, voor élk
#product incl. PolRGB). None/leeg = geen grens (oorspronkelijk gedrag). Alleen van toepassing op PolRGB
#('g'); voor elk ander product zonder effect.

volume3d_settings_default = {
    'volume3d_grid_res_km': 0.5,
    'volume3d_z_max_km': 15.0,
    'volume3d_z_res_km': 0.25,
    'volume3d_vertical_exaggeration': 5.0,
    'volume3d_smoothing_sigma': 1.2,
    'volume3d_tick_interval_km': 10.0,
    'volume3d_height_tick_interval_km': 2.0,
    'volume3d_gamma': 1.0,
    'volume3d_pointcloud_stride': 3,
    'volume3d_pointcloud_point_size': 4.0,
    'volume3d_min_value': None,
    'volume3d_circular_area': False,
    'volume3d_polrgb_cc_max': None,
}
"""All variables that represent the state of a QCheckbox should take on values 0 or 2, where a state of 2 means checked! 
"""
use_scissor=2

dimensions_main={'height':0.32,'width':0.8} #Dimensions of the areas of the screen that are occupied by the color bars and titles
fontsizes_main={'titles':8,'cbars_labels':7,'cbars_ticks':6}
bgcolor=0.92*np.array([255,255,255])
panelbdscolor=np.array([75,75,75])

bgmapcolor=np.array([0,0,0])
mapvisibility=False
mapcolorfilter=(1.0,1.0,1.0,0.975) #Color display can differ per OS
radardata_colorfilter=(1.0,1.0,1.0,1.0) #Color filter applied to the radar data itself (not the basemap). The
#alpha component lets the basemap (streets/place names) show through the radar data, e.g. for product 'g' or
#any other product, uniformly regardless of echo intensity.
maptiles_update_time = 0.1 #In seconds
basemap_source = 'Local' #'Local' (the bundled, pre-rendered satellite-like tiles) or 'MapTiler' (live, scrollable
#vector-rendered-to-raster basemap, requires a free MapTiler API key, see Settings -> Download -> API keys)
basemap_source_maptiler_style = 'dataviz-v4-dark' #MapTiler map ID/style, see https://cloud.maptiler.com/maps/
basemap_source_maptiler_provider = 'esri' #Which live tile provider to use when basemap_source == 'MapTiler':
#'esri' (Esri Dark Gray Canvas, publicly accessible, no API key needed) or 'stadia' (Stadia Maps Alidade
#Smooth Dark, requires a free API key, see Settings -> Download -> API keys)
radar_markersize=7.5
radar_colors={'Default':np.array([0,255,255]),'Selected':np.array([255,0,0]),'Automatic download':np.array([255,255,0]),'Automatic download + selected':np.array([255,128,0])}
lines_names=['countries','provinces','rivers','grid','heightrings']
lines_colors={'countries':np.array([255,255,255,255]),'provinces':np.array([255,255,0,150]),'rivers':np.array([0,255,255,130]),'grid':np.array([74,74,74,170]),'heightrings':np.array([74,74,74,220])}
lines_width=1.35
lines_antialias=True
show_heightrings_derivedproducts={j:not j in gv.plain_products_show_max_elevations for j in gv.plain_products}
showgridheightrings_panzoom=False
showgridheightrings_panzoom_time=0.6
gridheightrings_fontcolor={'bottom':np.array([0,0,0]),'top':np.array([255,255,255])}
gridheightrings_fontsize=10.8
grid_showtext=True 
lines_show=[j for j in lines_names if not j in ('rivers',)]
ghtext_names=['grid','heightrings']
ghtext_show=[j for j in ghtext_names]

reset_volume_attributes = True #Gets set to False in nlr_datasourcegeneral.py



variables_names_raw=['variables_resettodefault_version','reset_volume_attributes','radar_basedir','radarsources_dirs','radardirs_additional','radardata_dirs','radardata_dirs_indices','derivedproducts_dir','derivedproducts_filename_version','radardata_product_versions','selected_product_versions_ordered','movefiles_parameters','radar','scan_selection_mode','date','time','current_case_list_name','current_case','cases_offset_minutes','cases_looping_speed','cases_animation_window','cases_use_case_zoom','cases_loop_subset','cases_loop_subset_ncases','animation_duration','animation_speed_minpsec','animation_hold_lastframe','desired_timestep_minutes','max_timestep_minutes','maxspeed_minpsec','dataset','products','productunfiltered','polarization','apply_dealiasing','dealiasing_setting','dealiasing_max_nyquist_vel','dealiasing_dualprf_n_it','mesh_calibration_setting','melting_levels_manual_override','melting_levels_manual_station','melting_levels_manual_date','melting_levels_manual_hour','attenuation_correction_enabled','cartesian_product_res','cartesian_product_maxrange','scans','plot_mode','savefig_filename','savefig_include_menubar','animation_filename','ani_delay_ref','ani_delay','ani_delay_end','ani_sort_files','ani_group_datasets','ani_quality','networktimeout','minimum_downloadspeed','api_keys','stormmotion_save','pos_markers_latlons','pos_markers_labels','pos_markers_latlons_save','use_storm_following_view','view_nearest_radar','radar_bands_view_nearest_radar','data_selected_startazimuth','show_vwp','include_sfcobs_vwp','vwp_manual_sfcobs','vwp_manual_axlim','vvp_range_limits','vvp_height_limits','vvp_vmin_mps','vwp_sigmamax_mps','vwp_shear_layers','vwp_vorticity_layers','vwp_srh_layers','vwp_sm_display','base_url_obs','cmaps_minvalues','cmaps_maxvalues','polrgb_params','polrgb_cc_hide_above','PP_parameter_values','PP_parameters_panels','max_radardata_in_memory_GBs','sleeptime_after_plotting','cross_section_resolution_factor','cross_section_interpolation_mode','cross_section_n_samples','cross_section_height_headroom_percent','volume3d_grid_res_km','volume3d_z_max_km','volume3d_z_res_km','volume3d_vertical_exaggeration','volume3d_smoothing_sigma','volume3d_tick_interval_km','volume3d_height_tick_interval_km','volume3d_gamma','volume3d_pointcloud_stride','volume3d_pointcloud_point_size','volume3d_min_value','volume3d_circular_area','volume3d_polrgb_cc_max','use_scissor','colortables_dirs_filenames','dimensions_main','fontsizes_main','bgcolor','panelbdscolor','bgmapcolor','mapvisibility','mapcolorfilter','radardata_colorfilter','maptiles_update_time','basemap_source','basemap_source_maptiler_style','basemap_source_maptiler_provider','radar_markersize','radar_colors','lines_colors','lines_show','lines_width','lines_antialias','ghtext_show','grid_showtext','show_heightrings_derivedproducts','showgridheightrings_panzoom','showgridheightrings_panzoom_time','gridheightrings_fontcolor','gridheightrings_fontsize','grid_showtext']
variables_names_withclassreference=['variables_resettodefault_version','self.reset_volume_attributes','self.radar_basedir','self.radarsources_dirs','self.radardirs_additional','self.radardata_dirs','self.radardata_dirs_indices','self.derivedproducts_dir','self.derivedproducts_filename_version','self.radardata_product_versions','self.selected_product_versions_ordered','self.movefiles_parameters','self.crd.radar','self.crd.scan_selection_mode','self.crd.date','self.crd.time','self.current_case_list_name','self.current_case','self.cases_offset_minutes','self.cases_looping_speed','self.cases_animation_window','self.cases_use_case_zoom','self.cases_loop_subset','self.cases_loop_subset_ncases','self.animation_duration','self.animation_speed_minpsec','self.animation_hold_lastframe','self.desired_timestep_minutes','self.max_timestep_minutes','self.maxspeed_minpsec','self.crd.dataset','self.crd.products','self.crd.productunfiltered','self.crd.polarization','self.crd.apply_dealiasing','self.dealiasing_setting','self.dealiasing_max_nyquist_vel','self.dealiasing_dualprf_n_it','self.mesh_calibration_setting','self.melting_levels_manual_override','self.melting_levels_manual_station','self.melting_levels_manual_date','self.melting_levels_manual_hour','self.attenuation_correction_enabled','self.cartesian_product_res','self.cartesian_product_maxrange','self.crd.scans','self.crd.plot_mode','self.savefig_filename','self.savefig_include_menubar','self.animation_filename','self.ani_delay_ref','self.ani_delay','self.ani_delay_end','self.ani_sort_files','self.ani_group_datasets','self.ani_quality','self.networktimeout','self.minimum_downloadspeed','self.api_keys','self.stormmotion_save','self.pos_markers_latlons','self.pos_markers_labels','self.pos_markers_latlons_save','self.use_storm_following_view','self.view_nearest_radar','self.radar_bands_view_nearest_radar','self.data_selected_startazimuth','self.show_vwp','self.include_sfcobs_vwp','self.vwp_manual_sfcobs','self.vwp_manual_axlim','self.vvp_range_limits','self.vvp_height_limits','self.vvp_vmin_mps','self.vwp_sigmamax_mps','self.vwp_shear_layers','self.vwp_vorticity_layers','self.vwp_srh_layers','self.vwp_sm_display','self.base_url_obs','self.cmaps_minvalues','self.cmaps_maxvalues','self.polrgb_params','self.polrgb_cc_hide_above','self.PP_parameter_values','self.PP_parameters_panels','self.max_radardata_in_memory_GBs','self.sleeptime_after_plotting','self.cross_section_resolution_factor','self.cross_section_interpolation_mode','self.cross_section_n_samples','self.cross_section_height_headroom_percent','self.volume3d_grid_res_km','self.volume3d_z_max_km','self.volume3d_z_res_km','self.volume3d_vertical_exaggeration','self.volume3d_smoothing_sigma','self.volume3d_tick_interval_km','self.volume3d_height_tick_interval_km','self.volume3d_gamma','self.volume3d_pointcloud_stride','self.volume3d_pointcloud_point_size','self.volume3d_min_value','self.volume3d_circular_area','self.volume3d_polrgb_cc_max','self.use_scissor','self.colortables_dirs_filenames','self.dimensions_main','self.fontsizes_main','self.bgcolor','self.panelbdscolor','self.bgmapcolor','self.mapvisibility','self.mapcolorfilter','self.radardata_colorfilter','self.maptiles_update_time','self.basemap_source','self.basemap_source_maptiler_style','self.basemap_source_maptiler_provider','self.radar_markersize','self.radar_colors','self.lines_colors','self.lines_show','self.lines_width','self.lines_antialias','self.ghtext_show','self.grid_showtext','self.show_heightrings_derivedproducts','self.showgridheightrings_panzoom','self.showgridheightrings_panzoom_time','self.gridheightrings_fontcolor','self.gridheightrings_fontsize','self.grid_showtext']

#Variables that are reset to their default for the next update. Needs to be updated before every new update, 
#and 'variables_resettodefault_version' should always be included!!!!! reset_volume_attributes maybe too.
variables_resettodefault_forupdate=['variables_resettodefault_version', 'stormmotion_save']
variables_resettodefault_version = 10 #Version for variables_resettodefault_forupdate. Number needs to be increased by 1 before every new update!!!!!

try:
    #pickle.load appears to be incompatible with changes in pyqt version, i.e. when the file is saved while using pyqt5, then it also needs pyqt5 for loading the file.
    settings_filename=opa(os.path.join(gv.userdir+'/Generated_files','stored_settings.pkl'))
    if os.path.exists(settings_filename):
        with open(settings_filename,'rb') as f:
            settings=pickle.load(f)
    elif os.path.exists(opa(gv.programdir+'/Input_files/shared_default_settings.txt')):
        # Gedeelde versie, eerste start: Eriks weergave-instellingen (layout, lijnen, kleuren, PolRGB, 3D, VWP...)
        # als startwaarden, zonder keys/paden. Tekstformaat i.p.v. pickle, zodat er geen key in mee kan liften.
        with open(opa(gv.programdir+'/Input_files/shared_default_settings.txt'), encoding='utf-8') as f:
            settings = eval(f.read(), {'__builtins__': {}, 'A': lambda l, dt: np.array(l, dtype=dt)})
    else: settings={}
    
    # Deal with some variable name changings
    if 'KNMI_apikeys' in settings:
        api_keys['KNMI'] = settings['KNMI_apikeys']
    if 'bgcolor' in settings and not 'bgmapcolor' in settings:
        bgmapcolor = settings['bgcolor']
        del settings['bgcolor']
    
    resettodefault = not 'variables_resettodefault_version' in settings or variables_resettodefault_version != settings['variables_resettodefault_version']
        
    for name in variables_names_raw:
        try:
            if resettodefault and name in variables_resettodefault_forupdate:
                continue #Don't update these variables, since they are reset to their default value            
            elif name in settings:
                exec(name+"=settings[name]")
        except Exception:
            pass

    # Fill in any keys missing from an older stored settings file (e.g. after adding a new tunable parameter)
    # with their current defaults, rather than risking a KeyError later when that key is looked up.
    for key, default_value in polrgb_params_default.items():
        if key not in polrgb_params:
            polrgb_params[key] = default_value

    # BUGFIX (23 juli 2026, na Eriks crash bij Settings -> Map): show_heightrings_derivedproducts is net als
    # polrgb_params hierboven een dict die per product wordt opgeslagen/geladen (zie variables_names_raw) - een
    # ouder stored_settings.pkl (van voor MESH's toevoeging, 22 juli) mist dus de sleutel 'o', wat een KeyError
    # gaf in settings_tabmap zodra die dict werd doorgelopen. Zelfde oplossing als bij polrgb_params: ontbrekende
    # sleutels aanvullen met de actuele standaardwaarde (dezelfde formule als de oorspronkelijke definitie
    # hierboven), in plaats van te vertrouwen op wat er toevallig in het oude, opgeslagen bestand stond.
    for _product in gv.plain_products:
        if _product not in show_heightrings_derivedproducts:
            show_heightrings_derivedproducts[_product] = not _product in gv.plain_products_show_max_elevations

    # BUGFIX (27 juli 2026, na Eriks crash bij opstarten): een opgeslagen 'products'-lijst (welk
    # product elk paneel toont, zie variables_names_raw) kan een productcode bevatten die inmiddels
    # niet meer bestaat - bv. 'hd' (HDR), dat deze sessie weer is verwijderd nadat Erik het al had
    # bekeken/actief had staan in een paneel. Zonder deze check crasht de opstart met een KeyError
    # zodra nlr_plotting.py voor zo'n paneel een kleurenschaal probeert op te bouwen (self.cm1['hd']
    # bestaat dan niet meer). Zelfde soort vangnet als hierboven voor show_heightrings_derivedproducts:
    # een ongeldige/verwijderde code wordt stilzwijgend vervangen door 'z' (Reflectivity), i.p.v. te
    # crashen op een productcode die niet meer geregistreerd is.
    for _panel_idx in range(len(products)):
        if products[_panel_idx] not in gv.products_all:
            products[_panel_idx] = 'z'
except Exception:
    pass


if radar not in gv.radars_all:
    # Can happen when a radar has been removed from the radar meta files
    radar = gv.radars_all[0]

#Some variables are treated separately, because these are dictionaries for which the number of items (products) might vary between
#different versions of the application.   

for source in gv.radars:
    for i in gv.radars[source]:
        datasets = ('Z', 'V') if i in gv.radars_with_datasets else ('',)
        for j in datasets:
            radar_dataset = i+f'_{j}'*len(j)
            source_dataset = source+f'_{j}'*len(j)
            if not source_dataset in radarsources_dirs:
                radarsources_dirs[source_dataset] = gv.radarsources_dirs_Default[source_dataset]
            if not radar_dataset in radardata_dirs:
                radardata_dirs[radar_dataset] = radarsources_dirs[source_dataset]
                radardata_dirs_indices[radar_dataset] = 0
                radardata_product_versions[radar_dataset] = None

# Jabbeke (KMI, via MeteoGate) is the only radar for which 2 separate DBZH volumes are published per
# timestep -- a long-range one (starts at elevation 0.3 degrees, ~299 km) and a short-range one (starts at
# 0.5 degrees, ~150 km); see Source_MeteoGate in nlr_currentdata.py for the full explanation. Rather than
# using a single directory plus a Settings toggle to pick which one gets downloaded/displayed (an earlier
# approach that turned out to be prone to timing/race issues -- the displayed range could end up out of
# sync with the setting after quick navigation), 'Jabbeke_Z' is given TWO directory strings here, separated
# by ';' -- the existing multi-directory mechanism that NLradar already has for e.g. user-added alternative
# data locations (see dirstring_to_dirlist in nlr_background.py, and self.gui.radardata_dirs_indices, which
# tracks which of the 2 is currently selected for display). Both directories get downloaded into
# unconditionally (see Source_MeteoGate.get_urls_and_savenames_downloadfile in nlr_currentdata.py, which
# saves the long-range file under the normal 'Jabbeke_Z' directory and the short-range file -- when
# available -- under 'Jabbeke_Z_short'), and the EXISTING, already wired up CTRL+D shortcut
# (self.crd.change_dir_index) switches which of the 2 is displayed -- exactly the same mechanism already
# used for any other radar/dataset with multiple configured directories, so no new settings, shortcuts, or
# per-read range comparisons are needed at all.
#
# 'radardata_dirs' was already loaded from stored_settings.pkl (see the try-block near the top of this
# file) by the time we get here, so 'Jabbeke_Z' may already hold a value from a PREVIOUS session/version --
# possibly one that differs from the pure, freshly-computed default above (e.g. because an earlier version
# of this feature, or a manual edit, set something slightly different). Comparing against the default with
# '==' would then incorrectly conclude "the user customized this, leave it alone" and skip adding the short
# directory -- which is exactly what happened in practice. So instead, just check whether the short
# directory is ALREADY one of the configured directory strings (via dirstring_to_dirlist, the same parser
# used everywhere else for this), and append it if it's missing -- regardless of what else is in there. This
# still never discards or overwrites anything the user (or an earlier version of this feature) already
# configured; it only ever adds the short directory if it's not already present.
if 'Jabbeke_Z' in radardata_dirs:
    _jabbeke_z_short_dirstring = radarsources_dirs.get('KMI_Z', gv.radarsources_dirs_Default.get('KMI_Z', ''))+'_short'
    _jabbeke_z_existing_dirlist = bg.dirstring_to_dirlist(radardata_dirs['Jabbeke_Z'])
    if _jabbeke_z_short_dirstring not in _jabbeke_z_existing_dirlist:
        radardata_dirs['Jabbeke_Z'] = radardata_dirs['Jabbeke_Z'].rstrip().rstrip(';')+'; '+_jabbeke_z_short_dirstring
    del _jabbeke_z_short_dirstring, _jabbeke_z_existing_dirlist
                  
                        
for i in gv.radars_all:                              
    #Remove files that have '.crdownload' as extension, as these are likely unfinished downloads that weren't removed when the program exited.
    #It is also possibly that these are files that are currently being downloaded, but in this case a PermissionError is raised, and the file
    #won't be deleted.
    for k in ('','_Z','_V'):
        try:
            download_directory=bg.get_download_directory(radardata_dirs[i+k])
            if os.path.exists(download_directory):
                files=os.listdir(download_directory)
                for file in files:
                    try:
                        os.remove(os.path.join(download_directory,file))
                    except Exception: pass
        except Exception: pass


for datasource in gv.api_keys:
    if not datasource in api_keys:
        api_keys[datasource] = {}
    for key in gv.api_keys[datasource]:
        if not key in api_keys[datasource]:
            api_keys[datasource][key] = ''
# Installeerbare (gedeelde) versie: geen achtergrondkaarten die een API-key vereisen (MapTiler/Stadia).
# Alleen de meegeleverde lokale kaart en de key-loze Esri-kaarten blijven beschikbaar.
if gv.frozen:
    api_keys.pop('MapTiler', None)
    if basemap_source_maptiler_provider == 'stadia':
        basemap_source_maptiler_provider = 'esri'
# Gedeelde versie (installer/GitHub, 1 okt 2026): ALLEEN de meegeleverde lokale kaart. Live kaarten (Esri, Stadia,
# MapTiler) zijn verwijderd: Esri's voorwaarden vereisen een abonnement/Esri-software, Stadia/MapTiler een key.
basemap_source = 'Local'
            
            
for j in gv.colortables_dirs_filenames_Default:
    if j not in colortables_dirs_filenames or not os.path.exists(colortables_dirs_filenames[j]):
        colortables_dirs_filenames[j]=gv.colortables_dirs_filenames_Default[j]
for j in gv.products_all:
    if j not in cmaps_minvalues:
        cmaps_minvalues[j]=''
for j in gv.products_all:
    if j not in cmaps_maxvalues:
        cmaps_maxvalues[j]=''

        




cases_lists_filename = opa(os.path.join(gv.userdir+'/Generated_files','cases_lists.pkl'))
if os.path.exists(cases_lists_filename):
    with open(cases_lists_filename, 'rb') as f:
        cases_lists=pickle.load(f)
else:
    cases_lists = {}

# current_case_list_name=None
# current_case=None
# PP_parameter_values['l']={1:[True, 0], 2:[False, 0], 3:[False, 0], 4:[False, 0]}
# PP_parameter_values['h']={1:[True, 1], 2:[True, 3], 3:[True, 5], 4:[True, 10]}
# vwp_sm_display = {j: True for j in ('MW', 'LM', 'RM')}
# vwp_shear_layers = {1:[0,1], 2:[0,3], 3:[0,6], 4:[1,6]}



class ListWidgetItem(QListWidgetItem):
    def __lt__(self, other):
        listwidget = self.listWidget()
        qlabel = listwidget.itemWidget(self)
        qlabel_other = listwidget.itemWidget(other)
        descr, other_descr = qlabel.text(), qlabel_other.text()
        s = sorted([descr, other_descr])
        return s[0] == descr
    
previous_screen_DPI = None
def screen_DPI(screen):
    global previous_screen_DPI
    try:
        screen_DPI = screen.physicalDotsPerInch()
    except RuntimeError:
        # Can happen when the monitor is disconnected
        screen_DPI = previous_screen_DPI if previous_screen_DPI else 96
    previous_screen_DPI = screen_DPI
    return screen_DPI

class GUI(QWidget):
    def __init__(self, parent=None):
        super(GUI, self).__init__(parent) 
        self.changing_fullscreen=False
        self.showMaximized()
        
        self.setWindowTitle('NLradar')
        self.setWindowIcon(QIcon(gv.programdir+'/NLradar.ico'))
        
        # self.screen_DPI is a function! This is done to let it update automatically when the DPI changes after startup of the program.
        self.screen_DPI = lambda: screen_DPI(self.screen())
        self.screen_pixel_ratio = lambda: self.screen().devicePixelRatio()
        self.screen_size = lambda: np.array([self.screen().size().width(), self.screen().size().height()])
        print('screen_size=',self.screen_size())
        print('screen_DPI=',self.screen_DPI())        
        self.screen_physicalsize = lambda: self.screen_size()/self.screen_DPI()*2.54
        print('screen_physicalsize=',self.screen_physicalsize())
        print('scale_fac=',self.logicalDpiX() / 96.0)
        self.ref_screen_size=np.array([1920.,1080])
        self.ref_screen_DPI=141.58475185806762
        self.ref_screen_physicalsize=self.ref_screen_size/self.ref_screen_DPI*2.54

        
        #Variables with gui as their living class are assigned to gui here.
        self.radar_basedir = radar_basedir
        self.radarsources_dirs = radarsources_dirs
        self.radardirs_additional = radardirs_additional
        self.radardata_dirs=radardata_dirs
        self.radardata_dirs_indices=radardata_dirs_indices
        self.derivedproducts_dir=derivedproducts_dir
        self.derivedproducts_filename_version=derivedproducts_filename_version
        self.radardata_product_versions=radardata_product_versions
        self.selected_product_versions_ordered=selected_product_versions_ordered
        self.movefiles_parameters=movefiles_parameters
        self.animation_duration=animation_duration
        self.animation_speed_minpsec=animation_speed_minpsec
        self.animation_hold_lastframe=animation_hold_lastframe
        self.desired_timestep_minutes=desired_timestep_minutes
        self.max_timestep_minutes=max_timestep_minutes
        self.maxspeed_minpsec=maxspeed_minpsec
        self.savefig_filename=savefig_filename
        self.savefig_include_menubar = savefig_include_menubar
        self.animation_filename=animation_filename
        self.ani_delay_ref = ani_delay_ref
        self.ani_delay = ani_delay
        self.ani_delay_end = ani_delay_end
        self.ani_sort_files = ani_sort_files
        self.ani_group_datasets = ani_group_datasets
        self.ani_quality = ani_quality
        self.dealiasing_setting = dealiasing_setting
        self.dealiasing_max_nyquist_vel = dealiasing_max_nyquist_vel
        self.dealiasing_dualprf_n_it = dealiasing_dualprf_n_it
        self.mesh_calibration_setting = mesh_calibration_setting
        self.melting_levels_manual_override = melting_levels_manual_override
        self.melting_levels_manual_station = melting_levels_manual_station
        self.melting_levels_manual_date = melting_levels_manual_date
        self.melting_levels_manual_hour = melting_levels_manual_hour
        self.attenuation_correction_enabled = attenuation_correction_enabled
        self.cartesian_product_res = cartesian_product_res
        self.cartesian_product_maxrange = cartesian_product_maxrange
        self.networktimeout=networktimeout
        self.minimum_downloadspeed=minimum_downloadspeed
        self.api_keys = api_keys
        self.pos_markers_latlons = pos_markers_latlons
        self.pos_markers_latlons_save = pos_markers_latlons_save
        self.pos_markers_labels = pos_markers_labels
        self.stormmotion_save = stormmotion_save
        self.stormmotion = np.array([0,0], dtype='float32') #Don't use the saved storm motion vector, always start with no storm motion.
        self.use_storm_following_view = False
        self.view_nearest_radar = view_nearest_radar
        self.radar_bands_view_nearest_radar = radar_bands_view_nearest_radar
        self.data_selected_startazimuth = data_selected_startazimuth
        self.show_vwp = show_vwp
        self.include_sfcobs_vwp = include_sfcobs_vwp
        self.vwp_manual_sfcobs = vwp_manual_sfcobs
        self.vwp_manual_axlim = vwp_manual_axlim
        self.vvp_range_limits = vvp_range_limits
        self.vvp_height_limits = vvp_height_limits
        self.vvp_vmin_mps = vvp_vmin_mps
        self.vwp_sigmamax_mps = vwp_sigmamax_mps
        self.vwp_shear_layers = vwp_shear_layers
        self.vwp_vorticity_layers = vwp_vorticity_layers
        self.vwp_srh_layers = vwp_srh_layers
        self.vwp_sm_display = vwp_sm_display
        self.base_url_obs = base_url_obs
        self.cmaps_minvalues=cmaps_minvalues
        self.cmaps_maxvalues=cmaps_maxvalues
        self.polrgb_params=polrgb_params
        self.polrgb_cc_hide_above=polrgb_cc_hide_above
        self.PP_parameter_values=PP_parameter_values
        self.PP_parameters_panels=PP_parameters_panels
        self.max_radardata_in_memory_GBs=max_radardata_in_memory_GBs
        self.sleeptime_after_plotting=sleeptime_after_plotting
        self.cross_section_resolution_factor=cross_section_resolution_factor
        self.cross_section_interpolation_mode=cross_section_interpolation_mode
        self.cross_section_n_samples=cross_section_n_samples
        self.cross_section_height_headroom_percent=cross_section_height_headroom_percent
        self.volume3d_grid_res_km=volume3d_grid_res_km
        self.volume3d_z_max_km=volume3d_z_max_km
        self.volume3d_z_res_km=volume3d_z_res_km
        self.volume3d_vertical_exaggeration=volume3d_vertical_exaggeration
        self.volume3d_smoothing_sigma=volume3d_smoothing_sigma
        self.volume3d_tick_interval_km=volume3d_tick_interval_km
        self.volume3d_height_tick_interval_km=volume3d_height_tick_interval_km
        self.volume3d_gamma=volume3d_gamma
        self.volume3d_pointcloud_stride=volume3d_pointcloud_stride
        self.volume3d_pointcloud_point_size=volume3d_pointcloud_point_size
        self.volume3d_min_value=volume3d_min_value
        self.volume3d_circular_area=volume3d_circular_area
        self.volume3d_polrgb_cc_max=volume3d_polrgb_cc_max
        self.use_scissor=use_scissor
        self.colortables_dirs_filenames=colortables_dirs_filenames
        self.dimensions_main=dimensions_main
        self.fontsizes_main=fontsizes_main
        self.bgcolor=bgcolor
        self.panelbdscolor=panelbdscolor
        self.bgmapcolor=bgmapcolor
        self.mapvisibility=mapvisibility
        self.mapcolorfilter=mapcolorfilter
        self.radardata_colorfilter=radardata_colorfilter
        self.maptiles_update_time = maptiles_update_time
        self.basemap_source = basemap_source
        self.basemap_source_maptiler_style = basemap_source_maptiler_style
        self.basemap_source_maptiler_provider = basemap_source_maptiler_provider
        self.radar_markersize=radar_markersize
        self.radar_colors=radar_colors
        self.lines_names=lines_names
        self.lines_colors=lines_colors
        self.lines_show=lines_show    
        self.lines_width=lines_width
        self.lines_antialias=lines_antialias
        self.ghtext_names=ghtext_names
        self.ghtext_show=ghtext_show    
        self.show_heightrings_derivedproducts=show_heightrings_derivedproducts
        self.showgridheightrings_panzoom=showgridheightrings_panzoom
        self.showgridheightrings_panzoom_time=showgridheightrings_panzoom_time
        self.gridheightrings_fontcolor=gridheightrings_fontcolor
        self.gridheightrings_fontsize=gridheightrings_fontsize
        self.grid_showtext=grid_showtext
        
        self.reset_volume_attributes = reset_volume_attributes
        
        self.current_case_list = cases_lists.get(current_case_list_name, None)
        self.current_case_list_name = current_case_list_name if self.current_case_list else None
        self.current_case = current_case if self.get_case_index(self.current_case_list, current_case) != None else None
        self.previous_case_list_name = None
        self.previous_case = None
        self.cases_lists = cases_lists
        self.cases_offset_minutes = cases_offset_minutes
        self.cases_looping_speed = cases_looping_speed
        self.cases_animation_window = cases_animation_window
        self.cases_use_case_zoom = cases_use_case_zoom
        self.cases_loop_subset = cases_loop_subset
        self.cases_loop_subset_ncases = cases_loop_subset_ncases
        
        
                    
        self.pos_markers_positions = []
        self.sm_marker_present = False
        self.sm_marker_position = None
        self.sm_marker_latlon = None
        self.sm_marker_scantime = None
        self.sm_marker_scandatetime = None
        
        self.time_set_textbar_new=0
        self.time_last_removal_volumeattributes = 0
        self.switch_to_case_running=False
        self.move_to_next_case_call_ID=None
        self.move_to_next_case_running=False
        self.fullscreen=False
        self.need_rightclickmenu=False
        self.setting_saved_choice=False
        self.continue_savefig=False
        self.creating_animation = False
        self.exit=False #Set to True when exitting the program
        self.radars_automatic_download=[] #Radars for which currently data is automatically being downloaded.
        self.radars_download_older_data=[] #Radars for which older data is currently being downloaded.      
        #TODO:
   
        # First perform 'empty init' of self.pb, which allows use of self.pb during subsequent initialisation of classes, without getting issues
        # due to these other classes not being defined yet when referenced in self.pb.__init__.
        self.pb=Plotting(gui_class=self, empty_init=True)
        self.crd=Change_RadarData(gui_class=self, radar=radar, scan_selection_mode=scan_selection_mode, date=date, time=time, products=products, dataset=dataset, productunfiltered=productunfiltered, polarization=polarization, apply_dealiasing = apply_dealiasing, scans=scans, plot_mode=plot_mode)  
        self.dsg=self.crd.dsg
        self.dp=self.dsg.dp
        self.ani=self.crd.ani
        self.ad={}; self.dod={}; self.cd={}
        for j in gv.radars_all:
            self.ad[j]=AutomaticDownload(gui_class=self,radar=j)
            self.dod[j]=DownloadOlderData(gui_class=self,radar=j)
            self.cd[j]=CurrentData(gui_class=self,radar=j)
        self.cds = self.cd[gv.radars_all[0]].cds #is the same for all radars
        #Enable self.crd to also use self.dod
        self.crd.dod=self.dod
        
        # Perform full init of self.pb
        self.pb.__init__(gui_class=self)
        self.vwp = self.pb.vwp
        self.gui_vwp = GUI_VWP(gui_class=self)
        
        

        self.datew=QLineEdit(self.crd.selected_date)
        self.datew.setToolTip('Date (YYYYMMDD or c, if the time is also c)')
        self.timew=QLineEdit(self.crd.selected_time)
        self.timew.setToolTip('Time (HHMM or c, if the date is also c)')
        
        self.download_startstopw=QPushButton('Start', autoDefault=True)
        self.download_startstopw.setToolTip('Start/Stop download of data, for the time range specified in the widget to the right')
        self.download_timerangew=QLineEdit('Download')
        self.download_timerangew.setToolTip('Time range (minutes) for which data is downloaded when clicking Start. Download starts at input date and time, and continues backward until time range is spanned.')
                
        self.animation_settingsw=QPushButton('Ani', autoDefault=True)
        self.animation_settingsw.setToolTip('Animation settings')
        self.desired_timestep_minutesw=QLineEdit(str(self.desired_timestep_minutes))
        self.desired_timestep_minutesw.setToolTip("Desired timestep (minutes) when pressing LEFT/RIGHT, can be set to 'V' for moving by one full radar volume")
        self.max_timestep_minutesw=QLineEdit(str(ft.rifdot0(self.max_timestep_minutes))*(self.max_timestep_minutes < 1e10))
        self.max_timestep_minutesw.setToolTip('Maximum allowed timestep (minutes) when pressing LEFT/RIGHT. Can be left empty for no maximum.') 
        self.maxspeed_minpsecw=QLineEdit(str(ft.rifdot0(self.maxspeed_minpsec))*(self.maxspeed_minpsec < 1e10))
        self.maxspeed_minpsecw.setToolTip('Maximum speed (minutes/second). Can be left empty for no maximum.')
          
        self.textbar=QLineEdit()
        self.textbar.setReadOnly(True)
        self.textbar.setToolTip('Shows messages, or shows (latitude, longitude), (x, y), distance to radar or marker, beam elevation, product value for mouse cursor/min/max values for product within view.')
        
        self.hodow=QPushButton('VWP', autoDefault=True)
        self.hodow.setToolTip('Display/hide radar-derived vertical wind profile')        

        self.casesw=QPushButton('Cases', autoDefault=True)
        self.casesw.setToolTip('Switch to one of the cases stored in your list(s)')
        
        self.help_font = QFont()
        
        self.savefig_include_menubarw = QCheckBox()
        self.savefig_include_menubarw.setTristate(False)
        self.savefig_include_menubarw.setCheckState(2*self.savefig_include_menubar)
        self.savefig_include_menubarw.setToolTip('Whether to include this menu bar when saving a figure')
    
        self.extraw=QPushButton('Extra', autoDefault=True)
        self.settingsw=QPushButton('Settings', autoDefault=True)
        self.helpw=QPushButton('Help', autoDefault=True)
             
        self.widgets = ('datew', 'timew', 'casesw', 'download_timerangew', 'download_startstopw', 'animation_settingsw', 'desired_timestep_minutesw', 'max_timestep_minutesw', 'maxspeed_minpsecw', 'textbar', 'hodow', 'savefig_include_menubarw', 'extraw', 'settingsw', 'helpw')
        self.f1 = QFont('Times')
        f2 = QFont('Consolas') # Use a monospace font for the textbar
        for w in self.widgets:
            widget = getattr(self, w)
            widget.setMinimumWidth(1)
            # setFixedHeight is currently disabled because it can lead to issues when moving the app to a different screen with different size, resolution etc
            # widget.setFixedHeight(QFontMetrics(f1).height())
            # widget.setStyleSheet( "margin: 0px;" )
            # widget.setMinimumHeight(int(round(self.pb.scale_pixelsize(24))))
            widget.setFont(f2 if w == 'textbar' else self.f1)
        hbox=QHBoxLayout()
        hbox.addWidget(self.datew,8)
        hbox.addWidget(self.timew,5)
        hbox.addWidget(self.download_startstopw,5)
        hbox.addWidget(self.download_timerangew,6)
        hbox.addWidget(self.animation_settingsw,4)
        hbox.addWidget(self.desired_timestep_minutesw,4)
        hbox.addWidget(self.max_timestep_minutesw,4)
        hbox.addWidget(self.maxspeed_minpsecw,4)
        # De 0C/-20C-melting-level-balk is verwijderd (22 juli, op Eriks verzoek): die toonde alleen 1
        # vast punt (de radarlocatie zelf), terwijl de HCLASS-tooltip inmiddels al de nauwkeurigere
        # per-pixel-roosterwaarde toont (zie set_hclass_legend/update_data_readout in nlr_plotting.py).
        # textbar krijgt de vrijgekomen ruimte terug (63+18=81, was voor de balk werd toegevoegd 75).
        hbox.addWidget(self.textbar,81)
        hbox.addWidget(self.hodow,5)
        hbox.addWidget(self.casesw,5)
        hbox.addStretch(12)
        hbox.addWidget(self.savefig_include_menubarw,2)
        hbox.addWidget(self.extraw,5)
        hbox.addWidget(self.settingsw,6)
        hbox.addWidget(self.helpw,5)
        
        self.layout=QVBoxLayout()
        self.layout.addLayout(hbox)
        self.plotwidget=QWidget()
        self.plotwidget_layout=QHBoxLayout()
        self.plotwidget_layout.addWidget(self.pb.native)
        self.plotwidget_layout.setSpacing(0)
        self.plotwidget_layout.setContentsMargins(0,0,0,0)
        self.plotwidget.setLayout(self.plotwidget_layout)
        self.layout.addWidget(self.plotwidget)
        self.layout.setSpacing(1)
        self.layout.setContentsMargins(1,1,1,1)
        self.setLayout(self.layout)
        
        self.datew.returnPressed.connect(self.crd.process_datetimeinput)
        self.timew.returnPressed.connect(self.crd.process_datetimeinput)
        self.casesw.clicked.connect(self.cases_menu)
        self.download_startstopw.clicked.connect(self.startstop_download_oldercurrentdata)
        self.animation_settingsw.clicked.connect(self.change_animation_settings)
        self.desired_timestep_minutesw.editingFinished.connect(self.change_desired_timestep_minutes)
        self.max_timestep_minutesw.editingFinished.connect(self.change_max_timestep_minutes)
        self.maxspeed_minpsecw.editingFinished.connect(self.change_maxspeed_minpsec)
        self.hodow.clicked.connect(self.change_show_vwp)
        self.savefig_include_menubarw.stateChanged.connect(self.change_savefig_include_menubar)
        self.extraw.clicked.connect(self.extra)
        self.settingsw.clicked.connect(self.settings)
        self.helpw.clicked.connect(self.helpwidget)
        
        
                    
        
        #At the start (self.firstplot_performed=False), try to find a file with as date self.crd.selected_date.
        k = QShortcut(QKeySequence('Return'),self.pb.native,lambda: self.crd.process_datetimeinput())
        # Only enable when self.pb.native is in focus, in order to allow for clicking menu buttons by using tab and enter
        k.setContext(Qt.WidgetShortcut)
        QShortcut(QKeySequence('Backspace'),self,lambda: self.crd.back_to_previous_plot(False))
        QShortcut(QKeySequence('SHIFT+Backspace'),self,lambda: self.crd.back_to_previous_plot(True))
        QShortcut(QKeySequence('LEFT'),self,lambda: self.crd.process_keyboardinput(-1,0,0,'0',None,False))
        QShortcut(QKeySequence('RIGHT'),self,lambda: self.crd.process_keyboardinput(1,0,0,'0',None,False))
        QShortcut(QKeySequence('SHIFT+LEFT'),self,lambda: self.crd.process_keyboardinput(-12,0,0,'0',None,False))
        QShortcut(QKeySequence('SHIFT+RIGHT'),self,lambda: self.crd.process_keyboardinput(12,0,0,'0',None,False))
        QShortcut(QKeySequence(','),self,lambda: self.ani.change_continue_type('leftright',-1))
        QShortcut(QKeySequence('.'),self,lambda: self.ani.change_continue_type('leftright',1))
        QShortcut(QKeySequence('SPACE'),self,lambda: self.ani.change_continue_type('ani',0))
        QShortcut(QKeySequence('END'),self,lambda: self.crd.plot_mostrecent_data(True))
        QShortcut(QKeySequence('SHIFT+END'),self,lambda: self.crd.plot_mostrecent_data(False))
        QShortcut(QKeySequence('CTRL+Return'),self,lambda: self.move_to_next_case(None,0))
        QShortcut(QKeySequence('CTRL+LEFT'),self,lambda: self.move_to_next_case(None,-1))
        QShortcut(QKeySequence('CTRL+RIGHT'),self,lambda: self.move_to_next_case(None,1))
        QShortcut(QKeySequence('CTRL+BACKSPACE'),self,self.back_to_previous_case)
        QShortcut(QKeySequence('CTRL+SPACE'),self,lambda: self.ani.change_continue_type('cases',0))
        QShortcut(QKeySequence('SHIFT+SPACE'),self,lambda: self.ani.change_continue_type('ani_case',0))
        QShortcut(QKeySequence('CTRL+SHIFT+SPACE'),self,lambda: self.ani.change_continue_type('ani_cases',0))
        QShortcut(QKeySequence('DOWN'),self,lambda: self.crd.process_keyboardinput(0,-1,0,'0',None,False))
        QShortcut(QKeySequence('UP'),self,lambda: self.crd.process_keyboardinput(0,1,0,'0',None,False))
        QShortcut(QKeySequence('SHIFT+DOWN'),self,lambda: self.crd.process_keyboardinput(0,-1.1,0,'0',None,False))
        QShortcut(QKeySequence('SHIFT+UP'),self,lambda: self.crd.process_keyboardinput(0,1.1,0,'0',None,False))
        
        QShortcut(QKeySequence('HOME'),self,self.pb.reset_panel_view)      
        QShortcut(QKeySequence('SHIFT+HOME'),self,lambda: self.pb.reset_panel_view(False))
        QShortcut(QKeySequence('CTRL+HOME'),self,lambda: self.pb.reset_panel_view(True, False))
        QShortcut(QKeySequence('F'),self,self.change_use_storm_following_view)
        QShortcut(QKeySequence('F2'),self,lambda: self.pb.toggle_cross_sections_for_ab_line())
        QShortcut(QKeySequence('CTRL+SHIFT+4'),self,self.show_volume_3d_viewer)
        QShortcut(QKeySequence('Delete'),self,lambda: self.pb.clear_ab_line())
        QShortcut(QKeySequence('CTRL+SHIFT+Delete'),self,lambda: self.pb.clear_volume3d_rect())
        
        for product in gv.products_all:
            # Echo base ('eb', 24 juli) bewust overslaan hier: 'e' EN 'b' zijn beide al zelfstandige
            # 1-letter-sneltoetsen (ETH resp. POSH), dus een automatische 'E,B'-tweestapsreeks zou daarmee
            # kunnen interfereren. 'eb' is daarom alleen bereikbaar via de losse ALT+E-QShortcut verderop.
            if product in gv.products_alt_only_shortcuts:
                continue
            # BUGFIX (24 juli, na Eriks melding "dat kan dus niet, dan krijg ik CMH"): voor een
            # productcode van meer dan 1 teken (zoals 'uv', 'uh') gaf QKeySequence(product.upper())
            # -- dus bv. QKeySequence('UH') zonder komma -- GEEN geldige twee-staps-toetsreeks. Qt
            # parseert 'UH' als EEN ongeldige toets (lege/nooit-matchende QKeySequence), niet als
            # "eerst U, dan H". Empirisch geverifieerd (PyQt5, QT_QPA_PLATFORM=offscreen):
            # QKeySequence('UH').toString() geeft '' (0 stappen die ooit matchen), terwijl
            # QKeySequence('U,H').toString() echt 'U, H' geeft (2 stappen, werkt wel). Dit trof dus
            # OOK al 'uv' (unfiltered velocity) - die sneltoets deed vermoedelijk ook al nooit iets.
            # Fix: voor elk teken in de productcode een aparte, komma-gescheiden stap opbouwen.
            key_sequence = ','.join(list(product.upper()))
            QShortcut(QKeySequence(key_sequence),self,lambda product=product: self.crd.process_keyboardinput(0,0,0,product,None,False))
        
        QShortcut(QKeySequence('SHIFT+U'),self,self.crd.change_productunfiltered)
        QShortcut(QKeySequence('SHIFT+P'),self,self.crd.change_polarization)
        QShortcut(QKeySequence('SHIFT+V'),self,self.crd.change_apply_dealiasing)
        QShortcut(QKeySequence('Alt+V'),self,self.select_dealiasing_settings)
        # ALT+M (24 juli, op Eriks verzoek na het zien van iRadar's MESH-kalibratie-dropdown): M staat
        # hier vrij als ALT-combinatie (ALT+A/F1/N/P/V zijn al bezet, zie hierboven; M zelf is als LOSSE
        # letter al MESH's eigen product-sneltoets, maar dat is een andere combinatie dan ALT+M).
        QShortcut(QKeySequence('Alt+M'),self,self.select_mesh_calibration_settings)
        # ALT+W (25 juli, op Eriks verzoek): handmatige terugvaloptie voor de 0C/-20C-temperatuurbron
        # bij het Wyoming-sounding-archief (oude cases), naast de automatische stationskeuze.
        QShortcut(QKeySequence('Alt+W'),self,self.select_melting_levels_override)
        QShortcut(QKeySequence('Alt+C'),self,self.select_attenuation_correction_settings)
        # ALT+E (24 juli 2026, op Eriks verzoek - "Echo top is nu E, kun je ALT+E niet ook doen?" voor
        # Echo base): in tegenstelling tot ALT+V/ALT+M hierboven (die een INSTELLINGEN-dialoog openen voor
        # het huidige product) selecteert dit een heel ANDER, apart product ('eb') - Erik gaf aan niet te
        # geven om die inconsistentie met het bestaande ALT-patroon, simpelweg omdat er geen losse letters
        # meer over zijn (zie products_alt_only_shortcuts in nlr_globalvars.py).
        QShortcut(QKeySequence('Alt+E'),self,lambda: self.crd.process_keyboardinput(0,0,0,'eb',None,False))
        # ALT+L (24 juli 2026, op Eriks verzoek voor VILD = VIL/ETH): zelfde soort uitzondering als ALT+E
        # hierboven - selecteert een heel ANDER product ('vd'), i.p.v. een instellingen-dialoog voor het
        # huidige product. Erik gaf al eerder aan niet te geven om die inconsistentie met het ALT-patroon.
        QShortcut(QKeySequence('Alt+L'),self,lambda: self.crd.process_keyboardinput(0,0,0,'vd',None,False))
        # ALT+S (27 juli 2026, SHI): zelfde soort uitzondering als ALT+E/ALT+L hierboven - 'S' kan niet
        # via de automatische letter-sneltoetsenlus (product 'si' staat daarom in
        # gv.products_alt_only_shortcuts), omdat 'S' zelf al een losse 1-letter-sneltoets is (SRV) en
        # dat exact de Qt-shortcut-ambiguiteit zou geven die ook 'eb' (E+B) al uitsloot van die lus.
        QShortcut(QKeySequence('Alt+S'),self,lambda: self.crd.process_keyboardinput(0,0,0,'si',None,False))
        # ALT+Z (27 juli 2026, ZDR-kolomdiepte): zelfde soort uitzondering - 'Z' is al een losse
        # 1-letter-sneltoets (Reflectivity), dus 'Z,C' zou dezelfde Qt-shortcut-ambiguiteit geven.
        QShortcut(QKeySequence('Alt+Z'),self,lambda: self.crd.process_keyboardinput(0,0,0,'zc',None,False))
        QShortcut(QKeySequence('SHIFT+Q'),self,self.change_plainproducts_parameters)
        QShortcut(QKeySequence('SHIFT+I'),self,self.pb.change_interpolation)
        QShortcut(QKeySequence('SHIFT+Z'),self,self.pb.change_radarimage_visibility)
        
        for j in range(1, 16):
            if j < 10: k = str(j)
            elif j == 10: k = '0'
            else: k = f'SHIFT+{j-10}'
            QShortcut(QKeySequence(k),self,lambda j=j: self.crd.process_keyboardinput(0,0,j,'0',None,False))
                
        for j in (1, 2, 3, 4, 6, 8, 10):
            k = j if not j == 10 else 0
            QShortcut(QKeySequence(f'ALT+{k}'),self,lambda j=j: self.pb.change_panels(j))
                        
        QShortcut(QKeySequence('SHIFT+D'),self,self.crd.change_dataset)
        QShortcut(QKeySequence('CTRL+D'),self,self.crd.change_dir_index)
        QShortcut(QKeySequence('CTRL+P'),self,self.crd.change_product_version)
        QShortcut(QKeySequence('SHIFT+S'),self,lambda: self.crd.change_scan_selection_mode('scan'))
        QShortcut(QKeySequence('SHIFT+E'),self,lambda: self.crd.change_scan_selection_mode('scanangle'))
        QShortcut(QKeySequence('SHIFT+H'),self,lambda: self.crd.change_scan_selection_mode('height'))
        
        QShortcut(QKeySequence('SHIFT+N'),self,lambda: self.change_plot_mode('Single'))
        QShortcut(QKeySequence('SHIFT+A'),self,lambda: self.change_plot_mode('All'))
        QShortcut(QKeySequence('SHIFT+R'),self,lambda: self.change_plot_mode('Row'))
        QShortcut(QKeySequence('SHIFT+C'),self,lambda: self.change_plot_mode('Column'))
                
        for j in range(1, 13):
            QShortcut(QKeySequence(f'SHIFT+F{j}'),self,lambda j=j: self.save_choice(j))
            QShortcut(QKeySequence(f'F{j}'),self,lambda j=j: self.set_choice(j))
        
        for j in range(1, 11):
            k = j if not j == 10 else 0
            QShortcut(QKeySequence(f'CTRL+{k}'),self,lambda j=j: self.crd.switch_to_nearby_radar(j))
            QShortcut(QKeySequence(f'CTRL+ALT+{k}'),self,lambda j=j: self.crd.switch_to_nearby_radar(j, False))
        QShortcut(QKeySequence('N'),self,self.change_view_nearest_radar)
        QShortcut(QKeySequence('ALT+N'),self,self.select_radar_bands_for_view_nearest_radar)

        QShortcut(QKeySequence('ALT+F1'),self,self.view_choices)
        QShortcut(QKeySequence('ALT+A'),self,self.show_archiveddays)
        QShortcut(QKeySequence('ALT+P'),self,self.show_scans_properties)
        
        QShortcut(QKeySequence('SHIFT+F'),self,self.show_fullscreen)
        QShortcut(QKeySequence('CTRL+S'),self,self.savefig)
        QShortcut(QKeySequence('CTRL+ALT+S'),self,self.change_continue_savefig)
        QShortcut(QKeySequence('CTRL+SHIFT+S'),self,lambda: self.change_continue_savefig(True))
        
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showrightclickMenu)
                         
    
        
        
        
    def set_textbar(self,text=None,color=None,minimum_display_time=None):
        if pytime.time()>self.time_set_textbar_new:
            
            if text is None or color is None:
                error_info = ''
                for radar in [self.crd.selected_radar]+list(self.cd):
                    if self.cd[radar].cd_message_type == 'Error_info':
                        error_info = self.cd[radar].cd_message+f' ({radar})'
                        break
                
                if error_info:
                    color = 'red'
                elif self.cd[self.crd.selected_radar].cd_message==None: 
                    color='black'
                elif self.cd[self.crd.selected_radar].cd_message_type in ('Progress_info','Download_info'): 
                    color='green'
                if self.continue_savefig and color!='red':
                    color='blue'
                self.textbar.setStyleSheet('QLineEdit {color:'+color+'}')
                
                textbar_text=error_info
                if self.pb.datareadout_text!=None and not error_info: 
                    textbar_text+=self.pb.datareadout_text
                if self.pb.radar_mouse_selected!=None and not error_info: 
                    textbar_text+=', '+self.pb.radar_mouse_selected
                if self.cd[self.crd.selected_radar].cd_message!=None and not error_info:
                    textbar_text+=', '+self.cd[self.crd.selected_radar].cd_message
                if self.ad[self.crd.selected_radar].timer_info!=None and not error_info:
                    textbar_text+=', '+self.ad[self.crd.selected_radar].timer_info
                if self.current_case_shown():
                    case = self.current_case
                    label = case['label']
                    if 'extra_datetimes' in case:
                        # Also for extra datetimes a label can have been provided. In this case use the label (excluding empty strings)
                        # for which the scantime is closest to the current scantime
                        case_st = case.get('scantime', case['datetime'][8:10]+':'+case['datetime'][-2:]+':00')
                        extra_sts = [j.get('scantime', i[8:10]+':'+i[-2:]+':00') for i,j in case['extra_datetimes'].items() if j['label']]
                        sts = extra_sts+[case_st]
                        time = self.crd.time
                        ref_st = time[:2]+':'+time[-2:]+':00' if (self.pb.data_empty[0] or self.pb.data_isold[0]) else self.pb.data_attr['scantime'][0]
                        i_nearest_st = np.argmin([abs(ft.scantimediff_s(j, ref_st)) for j in sts])
                        if i_nearest_st != len(sts)-1: # Otherwise the scantime of the case is nearest, so the case label remains used
                            label = list(case['extra_datetimes'].values())[i_nearest_st]['label']
                    textbar_text += '. '+label
                        
                if textbar_text[:2] == ', ': 
                    textbar_text=textbar_text[2:]
            else:
                textbar_text=text
                self.textbar.setStyleSheet('QLineEdit {color:'+color+'}')
                
            if not minimum_display_time is None:
                #No new text will be displayed for a time of minimum_display_time seconds.
                self.time_set_textbar_new=pytime.time()+minimum_display_time
            
            self.textbar.setText(textbar_text)
            self.textbar.setCursorPosition(0)
            self.textbar.repaint()
        
                
    def change_animation_settings(self):
        self.animation_settings=QWidget();
        self.animation_settings.setWindowTitle('Change animation settings')
        animation_settings_layout=QFormLayout()
        self.animation_durationw=QLineEdit(str(ft.rifdot0(ft.r1dec(self.animation_duration))))
        self.animation_speed_minpsecw=QLineEdit(str(ft.rifdot0(ft.r1dec(self.animation_speed_minpsec))))
        self.animation_hold_lastframew=QLineEdit(str(ft.rifdot0(ft.rndec(self.animation_hold_lastframe,2))))
        animation_settings_layout.addRow(QLabel('Duration (minutes)'),self.animation_durationw)
        animation_settings_layout.addRow(QLabel('Speed (minutes/second)'),self.animation_speed_minpsecw)
        animation_settings_layout.addRow(QLabel('Hold last frame (seconds)'),self.animation_hold_lastframew)
        self.animation_durationw.editingFinished.connect(self.change_animation_duration)
        self.animation_speed_minpsecw.editingFinished.connect(self.change_animation_speed_minpsec)
        self.animation_hold_lastframew.editingFinished.connect(self.change_animation_hold_lastframe)
        self.animation_settings.setLayout(animation_settings_layout)
        self.animation_settings.resize(self.animation_settings.sizeHint())
        self.animation_settings.show()
        
    def change_animation_duration(self):
        input_duration=self.animation_durationw.text()
        number=ft.to_number(input_duration)
        if not number is None and number>0:
            self.animation_duration=int(number)
        else: self.animation_durationw.setText(str(ft.rifdot0(self.animation_duration)))
    def change_animation_speed_minpsec(self):
        input_speed=self.animation_speed_minpsecw.text()
        number=ft.to_number(input_speed)
        if not number is None and number>0:
            self.animation_speed_minpsec=number
        else: self.animation_speed_minpsecw.setText(str(ft.rifdot0(self.animation_speed_minpsec)))
    def change_animation_hold_lastframe(self):
        input_hold_lastframe=self.animation_hold_lastframew.text()
        number=ft.to_number(input_hold_lastframe)
        if not number is None and number>0:
            self.animation_hold_lastframe=number
        self.animation_hold_lastframew.setText(str(ft.rifdot0(self.animation_hold_lastframe)))
            
            
    def change_desired_timestep_minutes(self):
        input_desired_timestep_minutes=self.desired_timestep_minutesw.text()
        number=ft.to_number(input_desired_timestep_minutes)
        if input_desired_timestep_minutes.upper() == 'V':
            self.desired_timestep_minutes = 'V'
        elif not number is None and number >= 0.:
            self.desired_timestep_minutes=number
        else:
            self.desired_timestep_minutesw.setText(str(self.desired_timestep_minutes))
            
    def change_max_timestep_minutes(self):
        input_max_timestep_minutes=self.max_timestep_minutesw.text()
        if input_max_timestep_minutes == '':
            input_max_timestep_minutes = '1e10'
        number=ft.to_number(input_max_timestep_minutes)
        if not number is None and number>0:
            self.max_timestep_minutes=float(number)
        else: 
            self.max_timestep_minutesw.setText(str(ft.rifdot0(self.max_timestep_minutes))*(self.max_timestep_minutes < 1e10))        
                            
    def change_maxspeed_minpsec(self):
        input_maxspeed_minpsec=self.maxspeed_minpsecw.text()
        if input_maxspeed_minpsec == '':
            input_maxspeed_minpsec = '1e10'
        number=ft.to_number(input_maxspeed_minpsec)
        if not number is None and number>0:
            self.maxspeed_minpsec=number
        else:
            self.maxspeed_minpsecw.setText(str(ft.rifdot0(self.maxspeed_minpsec))*(self.maxspeed_minpsec < 1e10))
                    
            
    def change_use_storm_following_view(self):
        self.use_storm_following_view = not self.use_storm_following_view
        
    def change_view_nearest_radar(self):
        self.view_nearest_radar = not self.view_nearest_radar
        
    def select_radar_bands_for_view_nearest_radar(self):
        self.radar_bands_for_view_nearest_radar = QWidget()
        self.radar_bands_for_view_nearest_radar.setWindowTitle('Radar wavelength bands')
                
        self.radar_bands_view_nearest_radarw = {}
        hbox = QHBoxLayout()
        for j in ('S', 'C', 'X'):
            self.radar_bands_view_nearest_radarw[j] = QCheckBox(j)
            self.radar_bands_view_nearest_radarw[j].setTristate(False)
            self.radar_bands_view_nearest_radarw[j].setCheckState(2*(j in self.radar_bands_view_nearest_radar))
            self.radar_bands_view_nearest_radarw[j].stateChanged.connect(self.change_radar_bands_view_nearest_radar)
            hbox.addWidget(self.radar_bands_view_nearest_radarw[j])
            
        layout = QVBoxLayout()
        layout.addWidget(QLabel('Select which radar wavelength bands to include in search for nearest radar'))
        layout.addLayout(hbox)       
        self.radar_bands_for_view_nearest_radar.setLayout(layout)
        self.radar_bands_for_view_nearest_radar.resize(self.radar_bands_for_view_nearest_radar.sizeHint())
        self.radar_bands_for_view_nearest_radar.show()
        
    def change_radar_bands_view_nearest_radar(self):
        self.radar_bands_view_nearest_radar = [i for i,j in self.radar_bands_view_nearest_radarw.items() if j.checkState() == 2]
        
        
    def select_dealiasing_settings(self):
        self.select_dealiasing_settings = QWidget()
        self.select_dealiasing_settings.setWindowTitle('Select dealiasing settings')
        
        layout = QVBoxLayout()
        layout.addWidget(QLabel('Setting used for velocity dealiasing'))
        
        self.dealiasing_settingsw = {}
        group = QButtonGroup()
        hbox = QHBoxLayout()
        for j in gv.velocity_dealiasing_settings:            
            self.dealiasing_settingsw[j] = QRadioButton(j)
            self.dealiasing_settingsw[j].setChecked(j == self.dealiasing_setting)
            self.dealiasing_settingsw[j].toggled.connect(lambda state, j=j: self.change_dealiasing_setting(j))
            group.addButton(self.dealiasing_settingsw[j])
            hbox.addWidget(self.dealiasing_settingsw[j])
        layout.addLayout(hbox)
        self.dealiasing_max_nyquist_velw = QLineEdit(str(ft.rifdot0(ft.r1dec(self.dealiasing_max_nyquist_vel*self.pb.scale_factors['v']))))
        self.dealiasing_max_nyquist_velw.editingFinished.connect(self.change_dealiasing_max_nyquist_vel)
        layout.addWidget(QLabel("Select maximum Nyquist velocity for which to apply the Unet VDA model. For higher Nyquist velocities"))
        layout.addWidget(QLabel("the model will not be applied, since aliasing becomes unlikely. You can request Nyquist velocities for"))
        layout.addWidget(QLabel(f"the current radar volume by pressing ALT+P. Unit: {self.pb.productunits['v']}."))
        layout.addWidget(self.dealiasing_max_nyquist_velw)
        
        self.dualprfdealiasing_n_itw = QLineEdit(str(self.dealiasing_dualprf_n_it))
        self.dualprfdealiasing_n_itw.editingFinished.connect(self.change_dualprfdealiasing_n_it)
        layout.addWidget(QLabel("Maximum number of iterations for dual-PRF dealiasing. Dealiasing continues until either convergence or"))
        layout.addWidget(QLabel("this number of iterations is reached."))
        layout.addWidget(self.dualprfdealiasing_n_itw)
            
        self.select_dealiasing_settings.setLayout(layout)
        self.select_dealiasing_settings.resize(self.select_dealiasing_settings.sizeHint())
        self.select_dealiasing_settings.show()
            
    def change_dealiasing_setting(self, setting):
        self.dealiasing_setting = setting
        
        if self.pb.firstplot_performed:
            panels_update = [j for j in self.pb.panellist if gv.i_p[self.crd.products[j]] == 'v']
            self.pb.set_newdata(panels_update)
            
    def change_dealiasing_max_nyquist_vel(self):
        input_val = self.dealiasing_max_nyquist_velw.text()
        number = ft.to_number(input_val)
        if not number is None and number > 0:
            self.dealiasing_max_nyquist_vel = number/self.pb.scale_factors['v']
        else:
            self.dealiasing_max_nyquist_velw.setText(str(ft.rifdot0(ft.r1dec(self.dealiasing_max_nyquist_vel*self.pb.scale_factors['v']))))

    def select_mesh_calibration_settings(self):
        """Kalibratie-keuzevenster voor MESH (product 'o'), analoog aan select_dealiasing_settings
        hierboven. Kiest tussen Witt (1998, de standaard) en de twee Murillo & Homeyer (2019)-hertijkingen
        (P75/P95) - zie nlr_mesh.py voor de volledige formules/bronvermelding. Verandert alleen MESH zelf;
        POSH ('b'), SHI ('si', 27 juli 2026) en POH ('uh') blijven ongemoeid: POSH/SHI gebruiken altijd
        Witt's eigen (C-band-gecorrigeerde) SHI-opbouw (Murillo & Homeyer herzagen geen POSH/SHI-
        equivalent), en POH is een compleet andere, niet-SHI-gebaseerde berekening (Waldvogel/Holleman).
        """
        self.select_mesh_calibration_settings = QWidget()
        self.select_mesh_calibration_settings.setWindowTitle('Select MESH calibration')

        layout = QVBoxLayout()
        layout.addWidget(QLabel('Calibration used for MESH (Maximum Estimated Size of Hail)'))

        self.mesh_calibration_settingsw = {}
        group = QButtonGroup()
        vbox = QVBoxLayout()
        for j in gv.mesh_calibration_settings:
            self.mesh_calibration_settingsw[j] = QRadioButton(j)
            self.mesh_calibration_settingsw[j].setChecked(j == self.mesh_calibration_setting)
            self.mesh_calibration_settingsw[j].toggled.connect(lambda state, j=j: self.change_mesh_calibration_setting(j))
            group.addButton(self.mesh_calibration_settingsw[j])
            vbox.addWidget(self.mesh_calibration_settingsw[j])
        layout.addLayout(vbox)

        self.select_mesh_calibration_settings.setLayout(layout)
        self.select_mesh_calibration_settings.resize(self.select_mesh_calibration_settings.sizeHint())
        self.select_mesh_calibration_settings.show()

    def change_mesh_calibration_setting(self, setting):
        self.mesh_calibration_setting = setting

        # GEFIXT (24 juli): MESH's schijf-cache-sleutel bevat nu de kalibratienaam (zie
        # get_dataset_name in nlr_derived_plain.py) - elke kalibratie krijgt zijn eigen cache-slot,
        # dus wisselen van kalibratie kan geen verouderd resultaat van een ANDERE kalibratie meer
        # tonen. set_newdata hieronder forceert de hertekening/herberekening voor de zichtbare
        # MESH-panelen.
        if self.pb.firstplot_performed:
            panels_update = [j for j in self.pb.panellist if self.crd.products[j] == 'o']
            self.pb.set_newdata(panels_update)

    def select_melting_levels_override(self):
        """Handmatige terugvaloptie (25 juli, op Eriks verzoek) voor de 0C/-20C-temperatuurbron
        bij oude cases die buiten Open-Meteo's venster vallen en dus op het University of
        Wyoming radiosonde-archief terugvallen (zie nlr_meltinglevels.py). Analoog aan
        select_mesh_calibration_settings hierboven qua opzet (QWidget met QRadioButtons/checkbox).

        LET OP - status (25 juli): dit venster laat je zelf station+datum+uur kiezen en direct
        testen (via de 'Test ophalen'-knop hieronder, die nlr_meltinglevels.get_melting_levels_wyoming_manual
        rechtstreeks aanroept), maar de eigenlijke HCLASS/MESH/POSH/POH-berekening in
        nlr_derived_plain.py gebruikt deze instelling nog NIET - dat vergt een aanpassing in dat
        bestand zelf (nog niet aangeleverd door Erik), die zou moeten controleren of
        self.gui.melting_levels_manual_override aan staat en zo ja get_melting_levels_wyoming_manual
        met deze 3 instellingen aanroepen i.p.v. de automatische keuze.
        """
        self.select_melting_levels_override = QWidget()
        self.select_melting_levels_override.setWindowTitle('Select melting level source (Wyoming archive)')

        layout = QVBoxLayout()
        layout.addWidget(QLabel('Handmatige stationskeuze voor het Wyoming-sounding-archief (oude cases)'))
        layout.addWidget(QLabel('Terugvaloptie naast de automatische, dichtstbijzijnde-tijd-keuze - alleen actief als onderstaand vakje is aangevinkt.'))

        self.melting_levels_manual_overridew = QCheckBox('Gebruik handmatige keuze i.p.v. automatische keuze')
        self.melting_levels_manual_overridew.setChecked(self.melting_levels_manual_override)
        self.melting_levels_manual_overridew.toggled.connect(self.change_melting_levels_manual_override)
        layout.addWidget(self.melting_levels_manual_overridew)

        hbox_station = QHBoxLayout()
        hbox_station.addWidget(QLabel('Station:'))
        self.melting_levels_manual_stationw = QComboBox()
        self.melting_levels_manual_stationw.addItems(list(nlr_ml.WYOMING_STATIONS.keys()))
        self.melting_levels_manual_stationw.setCurrentText(self.melting_levels_manual_station)
        self.melting_levels_manual_stationw.currentTextChanged.connect(self.change_melting_levels_manual_station)
        hbox_station.addWidget(self.melting_levels_manual_stationw)
        layout.addLayout(hbox_station)

        hbox_datetime = QHBoxLayout()
        hbox_datetime.addWidget(QLabel('Datum (JJJJ-MM-DD):'))
        self.melting_levels_manual_datew = QLineEdit(self.melting_levels_manual_date)
        self.melting_levels_manual_datew.editingFinished.connect(self.change_melting_levels_manual_date)
        hbox_datetime.addWidget(self.melting_levels_manual_datew)
        hbox_datetime.addWidget(QLabel('Uur (UTC):'))
        self.melting_levels_manual_hourw = QComboBox()
        self.melting_levels_manual_hourw.addItems(['0', '12'])
        self.melting_levels_manual_hourw.setCurrentText(str(self.melting_levels_manual_hour))
        self.melting_levels_manual_hourw.currentTextChanged.connect(self.change_melting_levels_manual_hour)
        hbox_datetime.addWidget(self.melting_levels_manual_hourw)
        layout.addLayout(hbox_datetime)
        layout.addWidget(QLabel('De Bilt lanceert alleen om 0 UTC; Meppen/Essen/Norderney om 0 en 12 UTC.'))

        self.melting_levels_manual_testbutton = QPushButton('Test ophalen')
        self.melting_levels_manual_testbutton.clicked.connect(self.test_melting_levels_manual_fetch)
        layout.addWidget(self.melting_levels_manual_testbutton)
        self.melting_levels_manual_resultw = QLabel('')
        layout.addWidget(self.melting_levels_manual_resultw)

        self.select_melting_levels_override.setLayout(layout)
        self.select_melting_levels_override.resize(self.select_melting_levels_override.sizeHint())
        self.select_melting_levels_override.show()

    def change_melting_levels_manual_override(self, state):
        self.melting_levels_manual_override = state
        self._redraw_melting_level_dependent_panels()

    def change_melting_levels_manual_station(self, station):
        self.melting_levels_manual_station = station
        self._redraw_melting_level_dependent_panels()

    def change_melting_levels_manual_date(self):
        self.melting_levels_manual_date = self.melting_levels_manual_datew.text().strip()
        self._redraw_melting_level_dependent_panels()

    def change_melting_levels_manual_hour(self, hour_text):
        self.melting_levels_manual_hour = int(hour_text)
        self._redraw_melting_level_dependent_panels()

    def _redraw_melting_level_dependent_panels(self):
        """Forceert herberekening van de zichtbare panelen die van het gedeelde temperatuurrooster
        afhangen (HCLASS 'j', MESH 'o', POSH 'b', POH 'uh', en sinds 27 juli 2026 ook SHI 'si' zelf)
        - analoog aan change_mesh_calibration_setting hierboven, maar voor alle producten die
        ensure_melting_level_grid_current gebruiken i.p.v. alleen MESH. Nodig zodat aan/uitzetten van
        de handmatige stationskeuze, of het station/datum/uur daarvan wijzigen, meteen zichtbaar wordt
        i.p.v. pas bij een toevallige volgende herberekening - de cache-sleutel in
        ensure_melting_level_grid_current (nlr_datasourcegeneral.py) neemt de override-instellingen al
        mee, dus deze aanroep hoeft alleen de herberekening zelf te triggeren.

        BUGFIX (25 juli, na Eriks melding dat het radarbeeld waarop hij de stationswissel deed werd
        overgeslagen): deze functie wordt aangeroepen vanuit 4 losse widget-signalen (checkbox +
        3 dropdowns/invoerveld), die kort na elkaar kunnen vuren. self.pb.set_newdata is niet
        re-entrant-veilig (het deelt muterende status zoals self.data_attr_before/self.scans_before
        tussen aanroepen) - ELKE andere plek in de code die op vergelijkbare wijze een instelling
        wijzigt en set_newdata handmatig aanroept (change_polarization/change_apply_dealiasing/
        change_productunfiltered in nlr_changedata.py) beschermt zich hiertegen met een tijdslot via
        self.crd.end_time/self.sleeptime_after_plotting - die guard ontbrak hier, wat overlappende
        aanroepen kon toelaten. Nu hetzelfde patroon toegepast.
        """
        if pytime.time() - self.crd.end_time < self.sleeptime_after_plotting:
            return
        if self.pb.firstplot_performed:
            panels_update = [j for j in self.pb.panellist if self.crd.products[j] in ('j', 'o', 'b', 'uh', 'si', 'zc')]
            if panels_update:
                self.pb.set_newdata(panels_update)
        self.crd.end_time = pytime.time()

    def select_attenuation_correction_settings(self):
        """Aan/uit-instelling (28 juli 2026, op Eriks verzoek na de ZPHI-implementatie) voor de
        C-band-verzwakkingscorrectie (nlr_attenuation.py) op MESH/POSH/POH/SHI/HCLASS. Standaard AAN.
        Analoog aan select_melting_levels_override hierboven qua opzet - 1 checkbox, direct effect
        op de zichtbare panelen via _redraw_hail_attenuation_dependent_panels hieronder. Bedoeld om
        oud (uit) en nieuw (aan) gedrag naast elkaar te kunnen vergelijken, i.p.v. alleen de eerdere
        cijfers uit een vorige sessie erbij te moeten pakken.
        """
        self.select_attenuation_correction_settings = QWidget()
        self.select_attenuation_correction_settings.setWindowTitle('ZPHI attenuation correction')

        layout = QVBoxLayout()
        layout.addWidget(QLabel('C-band-verzwakkingscorrectie (ZPHI, Bringi 2001/Gou 2019)'))
        layout.addWidget(QLabel('Geldt voor HCLASS/MESH/POSH/POH/SHI. Uit = exact het oude, ongecorrigeerde gedrag.'))

        self.attenuation_correction_enabledw = QCheckBox('Verzwakkingscorrectie toepassen')
        self.attenuation_correction_enabledw.setChecked(self.attenuation_correction_enabled)
        self.attenuation_correction_enabledw.toggled.connect(self.change_attenuation_correction_enabled)
        layout.addWidget(self.attenuation_correction_enabledw)

        self.select_attenuation_correction_settings.setLayout(layout)
        self.select_attenuation_correction_settings.resize(self.select_attenuation_correction_settings.sizeHint())
        self.select_attenuation_correction_settings.show()

    def change_attenuation_correction_enabled(self, state):
        self.attenuation_correction_enabled = state
        print(f"change_attenuation_correction_enabled: state={state}")
        self._redraw_hail_attenuation_dependent_panels()

    def _redraw_hail_attenuation_dependent_panels(self):
        """Forceert herberekening van de zichtbare HCLASS/MESH/POSH/POH/SHI-panelen na het aan/uit
        zetten van de verzwakkingscorrectie - zelfde opzet (incl. dezelfde debounce-guard) als
        _redraw_melting_level_dependent_panels hierboven. LET OP: 'zc' (ZDR-kolomdiepte) hoort hier
        NIET bij, in tegenstelling tot bij de temperatuurrooster-variant hierboven - zc gebruikt geen
        gecorrigeerde Z/ZDR (zie nlr_attenuation.py/get_hail_corrected_data_all: scope is uitdrukkelijk
        beperkt tot j/o/b/uh/si).
        """
        if pytime.time() - self.crd.end_time < self.sleeptime_after_plotting:
            return
        if self.pb.firstplot_performed:
            panels_update = [j for j in self.pb.panellist if self.crd.products[j] in ('j', 'o', 'b', 'uh', 'si')]
            if panels_update:
                self.pb.set_newdata(panels_update)
                self.pb.set_draw_action('plotting')
                self.pb.update()
        self.crd.end_time = pytime.time()

    def test_melting_levels_manual_fetch(self):
        """Haalt direct de sounding op voor de op dit moment ingestelde station+datum+uur-combinatie
        en toont het resultaat (of de foutmelding) in het label onder de knop - zo kun je een
        station+tijdstip controleren zonder dat de HCLASS/MESH/POSH/POH-berekening zelf al aan
        deze instelling gekoppeld hoeft te zijn (zie kanttekening bovenaan select_melting_levels_override)."""
        try:
            year, month, day = (int(x) for x in self.melting_levels_manual_date.split('-'))
        except (ValueError, AttributeError):
            self.melting_levels_manual_resultw.setText('Ongeldige datum - gebruik JJJJ-MM-DD')
            return
        try:
            result = nlr_ml.get_melting_levels_wyoming_manual(
                self.melting_levels_manual_station, year, month, day, self.melting_levels_manual_hour)
            self.melting_levels_manual_resultw.setText(
                'OK - 0C: {} m, -20C: {} m (sounding: {})'.format(
                    result['h0_m'], result['h_minus20_m'], result['datetime_used']))
        except (ValueError, nlr_ml.MeltingLevelError) as e:
            self.melting_levels_manual_resultw.setText('Fout: {}'.format(e))

    def change_dualprfdealiasing_n_it(self):
        number = ft.to_number(self.dualprfdealiasing_n_itw.text())
        if not number is None and number > 0:
            self.dealiasing_dualprf_n_it = int(number)
        self.dualprfdealiasing_n_itw.setText(str(self.dealiasing_dualprf_n_it))
        if any([self.crd.products[j] in ('v','s') for j in self.pb.panellist]) and self.crd.apply_dealiasing:
            self.pb.set_newdata(self.pb.panellist)
        
          
    def change_plainproducts_parameters(self):
        product=self.crd.products[self.pb.panel]
        if product in gv.plain_products_with_parameters:
            self.plainproducts_parameters=QWidget()
            self.plainproducts_parameters.setWindowTitle('Change product parameters')
            plainproducts_parameters_layout=QFormLayout()
            
            use_list = isinstance(gv.plain_products_parameter_description[product], list)
            hbox = QHBoxLayout() if not use_list else [QHBoxLayout() for j in gv.plain_products_parameter_description[product]]
            self.parameter_valuesw = {}
            for j in range(1,5):
                if use_list:
                    self.parameter_valuesw[j] = []
                    if product in ('h', 'l'):
                        # When more products use more than one parameter or use something different than a QLineEdit, then a better way of handling
                        # this is probably needed
                        self.parameter_valuesw[j] += [QCheckBox()]
                        self.parameter_valuesw[j][0].setTristate(False)
                        self.parameter_valuesw[j][0].setCheckState(2 if self.PP_parameter_values[product][j][0] else 0)
                        self.parameter_valuesw[j][0].stateChanged.connect(lambda state, j=j: self.change_parameter_value('stateChanged', j, 0))
                        self.parameter_valuesw[j] += [QLineEdit(str(ft.rifdot0(self.PP_parameter_values[product][j][1])))]
                        self.parameter_valuesw[j][1].editingFinished.connect(lambda j=j: self.change_parameter_value('editingFinished', j, 1))
                        self.parameter_valuesw[j][1].returnPressed.connect(lambda j=j: self.change_parameter_value('returnPressed', j, 1)) 
                    for i in range(len(hbox)):
                        hbox[i].addWidget(self.parameter_valuesw[j][i])
                else:
                    self.parameter_valuesw[j] = QLineEdit(str(ft.rifdot0(self.PP_parameter_values[product][j])))
                    self.parameter_valuesw[j].editingFinished.connect(lambda j=j: self.change_parameter_value('editingFinished', j))
                    self.parameter_valuesw[j].returnPressed.connect(lambda j=j: self.change_parameter_value('returnPressed', j)) 
                    hbox.addWidget(self.parameter_valuesw[j])
                    
            if use_list:
                for i in range(len(hbox)):
                    plainproducts_parameters_layout.addRow(QLabel(''), QLabel(gv.plain_products_parameter_description[product][i]))
                    plainproducts_parameters_layout.addRow(QLabel(''), hbox[i])
            else:
                plainproducts_parameters_layout.addRow(QLabel(''), QLabel(gv.plain_products_parameter_description[product]))
                plainproducts_parameters_layout.addRow(QLabel(''), hbox)
            self.plainproducts_parameters.setLayout(plainproducts_parameters_layout)
            self.plainproducts_parameters.resize(self.plainproducts_parameters.sizeHint())
            self.plainproducts_parameters.show()
            
    def change_PP_parameters_panels(self, product):
        panels_with_plainproduct = [i for i in self.pb.panellist if self.crd.products[i] == product]
        for i in range(0, len(panels_with_plainproduct)):
            self.PP_parameters_panels[panels_with_plainproduct[i]] = min([4, i+1])
        return panels_with_plainproduct
            
    def change_parameter_value(self, action, j, i=None):
        product = self.crd.products[self.pb.panel]
        if action == 'stateChanged':
            input_state = ft.from_list_or_nolist(self.parameter_valuesw[j], i).checkState()
            ft.to_list_or_nolist(self.PP_parameter_values[product], j, input_state == 2, i)
        else:
            input_value = ft.from_list_or_nolist(self.parameter_valuesw[j], i).text()
            number = ft.to_number(input_value)
            if not number is None:
                ft.to_list_or_nolist(self.PP_parameter_values[product], j, number, i)
        
        if action == 'returnPressed':
            self.plainproducts_parameters.close()
        if action in ('stateChanged', 'returnPressed'):
            panels_with_plainproduct = self.change_PP_parameters_panels(product)
            self.pb.set_newdata(panels_with_plainproduct,plain_products_parameters_changed=True)
                
                  
    def startstop_download_oldercurrentdata(self):
        if not gv.data_sources[self.crd.selected_radar] in self.cds.source_classes:
            self.set_textbar('Downloading not implemented for radars from '+gv.data_sources[self.crd.selected_radar], 'red', 1)
            return
        
        #Setting self.dod[self.crd.selected_radar].download_times from within this function (different thread) is thread-safe, see this article:
        #http://effbot.org/pyfaq/what-kinds-of-global-value-mutation-are-thread-safe.htm
        action=self.download_startstopw.text() #action can be 'Start' or 'Stop'
        if action == 'Stop':            
            self.dod[self.crd.selected_radar].terminate()
            self.cd[self.crd.selected_radar].cd_message = None
                
            self.reset_download_widgets(self.crd.selected_radar)
        else:
            input_download_timerange=self.download_timerangew.text()
            if ft.to_number(input_download_timerange) is None or '.' in input_download_timerange or int(input_download_timerange)<0:
                #input_download_timerange must be a positive integer.
                self.set_textbar('Incorrect time range','red',1)
                return
                
            download_timerange_s=int(input_download_timerange)*60
            self.dod[self.crd.selected_radar].download_timerange_s=download_timerange_s
            self.radars_download_older_data.append(self.crd.selected_radar)
            self.pb.set_radarmarkers_data()
            self.pb.update() #For plotting the markers
            
            self.download_startstopw.setText('Stop')
            #Start the thread self.dod[self.crd.selected_radar]. If it is already running, then it is enough to update only the download timerange.
            self.dod[self.crd.selected_radar].start()
                                
    def set_download_widgets(self,timerange,startstop):
        self.download_timerangew.setText(str(timerange))
        self.download_startstopw.setText(startstop)
                
    def reset_download_widgets(self, radar):
        self.set_download_widgets('Download','Start')
        #Also reset the marker colors
        if radar in self.radars_download_older_data:
            self.radars_download_older_data.pop(self.radars_download_older_data.index(radar))
        self.pb.set_radarmarkers_data()
        self.pb.update() #For plotting the markers

    def start_automatic_download(self,radar):
        if not gv.data_sources[radar] in self.cds.source_classes:
            self.set_textbar('Downloading not implemented for radars from '+gv.data_sources[radar], 'red', 1)
            return
        
        if not radar in self.radars_automatic_download:
            self.radars_automatic_download.append(radar)
        self.pb.set_radarmarkers_data()
        self.pb.update() #For plotting the markers
        self.ad[radar].start()
        
    def stop_automatic_download(self,radar):
        self.radars_automatic_download.pop(self.radars_automatic_download.index(radar))
        self.pb.set_radarmarkers_data()
        self.pb.update()
        self.ad[radar].terminate()
        self.ad[radar].stop_timer()
        self.cd[radar].cd_message = None
        

    def cases_menu(self):
        cases_menu = QMenu(self)
        # cases_menu.setStyleSheet("QMenu::item{padding-left:5px; padding-right:5px;}\
        #                           QMenu::item::selected{background-color:rgb(100,255,255);}\
        #                           QMenu::item:default{padding-left:5px; padding-right:5px; color:#ff00ff;}")
        position = self.casesw.geometry()
        point = position.bottomLeft()

        if len(list(self.cases_lists)) == 0:
            action = cases_menu.addAction('No case lists have been created yet')
            action.setEnabled(False)
        else:
            action = cases_menu.addAction('Settings')
            action.triggered.connect(self.show_cases_settings_window)
            
            if self.current_case_list_name is None:
                self.current_case_list_name = list(self.cases_lists)[0]
            self.current_case_list = self.cases_lists[self.current_case_list_name]
            other_case_lists = {i:j for i,j in self.cases_lists.items() if not i == self.current_case_list_name}
            
            action = cases_menu.addAction('Unselect current case')
            action.triggered.connect(self.unselect_current_case)
            
            action = cases_menu.addAction('Modify case list or click URLs')
            action.triggered.connect(self.modify_case_list)
            
            delete_menu = cases_menu.addMenu('Delete case list')
            for list_name in self.cases_lists:
                action = delete_menu.addAction(list_name)
                action.triggered.connect(lambda state, list_name=list_name: self.delete_case_list(list_name))
            
            othercases_menu = cases_menu.addMenu('Select case from other list')
            # othercases_menu.setStyleSheet("QMenu::item{padding-left:5px; padding-right:15px;}\
            #                                    QMenu::item::selected{background-color:rgb(100,255,255);}")
            max_length = 100
            if len(other_case_lists) == 0:
                action = othercases_menu.addAction('No other lists available yet')
                action.setEnabled(False)
            else:
                for list_name in other_case_lists:
                    case_list = other_case_lists[list_name]    
                    othercase_menu = othercases_menu.addMenu(list_name+f'  ({len(case_list)})')
                    if len(case_list) == 0:
                        action = othercase_menu.addAction('This list is currently empty')
                        action.setEnabled(False)
            
                    for i, case_dict in enumerate(case_list):
                        if i % max_length == 0:
                            menu = othercase_menu.addMenu(case_dict['datetime']+' -') if len(case_list) > max_length else othercase_menu
                        action = menu.addAction(self.get_descriptor_for_case(case_dict))
                        action.triggered.connect(
                            lambda state, list_name=list_name, case_dict=case_dict: self.switch_to_case(list_name, case_dict))
       
            cases_menu.addSeparator()
            action = cases_menu.addAction(self.current_case_list_name+f' ({len(self.current_case_list)} cases)')
            action.setEnabled(False)
            
            i_current_case = self.get_case_index()
            action_before = None
            for i, case_dict in enumerate(self.current_case_list):
                if i % max_length == 0:
                    if (i_current_case != None and i <= i_current_case < i+max_length) or len(self.current_case_list) <= max_length:
                        menu = cases_menu
                    else:
                        dt1 = case_dict['datetime']
                        dt2 = self.current_case_list[min(i+max_length, len(self.current_case_list))-1]['datetime']
                        menu = QMenu(dt1+' - '+dt2, cases_menu)
                        # Use insertMenu instead of addAction, in order to always put these submenus above any listed actions in the main menu
                        cases_menu.insertMenu(action_before, menu)
                action = menu.addAction(self.get_descriptor_for_case(case_dict))
                if menu == cases_menu and action_before is None:
                    action_before = action
                if i == i_current_case:
                    menu.setDefaultAction(action)
                action.triggered.connect(
                            lambda state, list_name=self.current_case_list_name, case_dict=case_dict: self.switch_to_case(list_name, case_dict))
                
        cases_menu.popup(self.mapToGlobal(point))
        
    def get_descriptor_for_case(self, case_dict, short_date=True):
        description = case_dict['datetime'][2 if short_date else 0:]+' '+case_dict['radar']+' '+case_dict['label']
        if len(case_dict['pos_markers_latlons']) > 1:
            description += f" ({len(case_dict['pos_markers_latlons'])})"
        return description
    
    def show_cases_settings_window(self):
        self.cases_settings=QWidget()
        self.cases_settings.setWindowTitle('Case settings')
        cases_settings_layout=QFormLayout()
        
        self.cases_use_case_zoomw = QCheckBox(tristate=False)
        self.cases_use_case_zoomw.setCheckState(2 if self.cases_use_case_zoom else 0)
        cases_settings_layout.addRow(QLabel('Use zoom-level saved with case'),self.cases_use_case_zoomw)
        self.cases_use_case_zoomw.stateChanged.connect(self.change_cases_use_case_zoom)
        
        self.cases_offset_minutesw=QLineEdit(str(int(self.cases_offset_minutes)))
        cases_settings_layout.addRow(QLabel('Time offset relative to case (minutes, - sign for before case)'),self.cases_offset_minutesw)
        self.cases_offset_minutesw.editingFinished.connect(self.change_cases_offset_minutes)
        
        self.cases_animation_windoww=QLineEdit(', '.join(list(map(str, self.cases_animation_window))))
        cases_settings_layout.addRow(QLabel('Animation window: start_offset, end_offset (like time offset)'),self.cases_animation_windoww)
        self.cases_animation_windoww.editingFinished.connect(self.change_cases_animation_window)
        
        self.cases_looping_speedw=QLineEdit(str(self.cases_looping_speed))
        cases_settings_layout.addRow(QLabel('Looping speed (cases per second)'),self.cases_looping_speedw)
        self.cases_looping_speedw.editingFinished.connect(self.change_cases_looping_speed)
        
        self.cases_loop_subsetw = QCheckBox(tristate=False)
        self.cases_loop_subsetw.setCheckState(2 if self.cases_loop_subset else 0)
        cases_settings_layout.addRow(QLabel('Show only a subset of cases when looping'),self.cases_loop_subsetw)
        self.cases_loop_subsetw.stateChanged.connect(self.change_cases_loop_subset)
        
        self.cases_loop_subset_ncasesw=QLineEdit(str(self.cases_loop_subset_ncases))
        cases_settings_layout.addRow(QLabel('Size of subset of cases to loop over'),self.cases_loop_subset_ncasesw)
        self.cases_loop_subset_ncasesw.editingFinished.connect(self.change_cases_loop_subset_ncases)
        
        self.cases_settings.setLayout(cases_settings_layout)
        self.cases_settings.resize(self.cases_settings.sizeHint())
        self.cases_settings.show()
    def change_cases_use_case_zoom(self):
        self.cases_use_case_zoom = self.cases_use_case_zoomw.checkState() == 2
    def change_cases_offset_minutes(self):
        input_offset=self.cases_offset_minutesw.text()
        number=ft.to_number(input_offset)
        if not number is None:
            self.cases_offset_minutes=int(number)
        else: self.cases_offset_minutesw.setText(str(self.cases_offset_minutes))
    def change_cases_animation_window(self):
        input_window = self.cases_animation_windoww.text()
        try:
            start_offset, end_offset = map(int, input_window.split(','))
            if start_offset < end_offset:
                self.cases_animation_window = [start_offset, end_offset]
            else: raise Exception
        except Exception:
            self.cases_animation_windoww.setText(', '.join(list(map(str, self.cases_animation_window))))
    def change_cases_looping_speed(self):
        input_speed=self.cases_looping_speedw.text()
        number=ft.to_number(input_speed)
        if not number is None:
            self.cases_looping_speed=float(number)
        else: self.cases_looping_speedw.setText(str(self.cases_looping_speed))
    def change_cases_loop_subset(self):
        self.cases_loop_subset = self.cases_loop_subsetw.checkState() == 2
        # Updating loop_start_case_index here allows for repeatedly switching between using/not using a subset, while varying the subset 
        # of cases to loop over
        self.ani.loop_start_case_index = self.get_case_index()
    def change_cases_loop_subset_ncases(self):
        input_ncases=self.cases_loop_subset_ncasesw.text()
        number=ft.to_number(input_ncases)
        if not number is None:
            self.cases_loop_subset_ncases=int(number)
        else: self.cases_loop_subset_ncasesw.setText(str(self.cases_loop_subset_ncases))
    
    def unselect_current_case(self):
        self.current_case = None
        self.set_textbar()
    
    def delete_case_list(self, list_name):
        msg_box = QMessageBox()
        result = msg_box.question(self, 'Case list deletion', "Are you sure that you want to delete '"+list_name+"'?")
        if result == QMessageBox.Yes:
            del self.cases_lists[list_name]
            with open(cases_lists_filename,'wb') as f:
                pickle.dump(self.cases_lists,f)
            
            if list_name == self.current_case_list_name:
                self.current_case_list_name = None
                self.current_case = None
                if hasattr(self, 'modify_case_listw'):
                    self.modify_case_listw.close()
        
    def switch_to_case(self, list_name, case_dict, time_offset='default', set_data=True):
        # Convert to integer in the latter case, because when called from within nlr_animate.py it is given as a string
        time_offset = self.cases_offset_minutes if time_offset == 'default' else int(time_offset)
            
        if list_name != self.current_case_list_name and hasattr(self, 'modify_case_listw'):
            self.modify_case_listw.close()
            
        self.previous_case_list_name = self.current_case_list_name
        self.previous_case = self.current_case
            
        self.current_case_list_name = list_name
        # the case_list is not directly given as input, because that copies the object, and causes self.cases_lists 
        # to not get updated when self.current_case_list is updated by rearranging or deleting cases.
        self.current_case_list = self.cases_lists[list_name]
        self.current_case = case_dict
                
        date, time = case_dict['datetime'][:8], case_dict['datetime'][-4:]
        date, time = ft.next_date_and_time(date, time, time_offset)
        self.datew.setText(date); self.timew.setText(time)
        
        radar = case_dict['radar']
        if 'scannumbers_forduplicates' in case_dict and time_offset == 0: # Was not saved in the past so
            self.dsg.scannumbers_forduplicates = case_dict['scannumbers_forduplicates'].copy()
            # Also set scannumbers_forduplicates_radars[radar] in addition to scannumbers_forduplicates itself, because scannumbers_forduplicates 
            # will be reset when self.dsg.update_parameters is called
            self.dsg.scannumbers_forduplicates_radars[radar] = self.dsg.scannumbers_forduplicates.copy()
        
        if self.show_vwp and 'vvp_range_limits' in case_dict: 
            self.vvp_range_limits = case_dict['vvp_range_limits']
            self.vvp_height_limits = case_dict['vvp_height_limits']
            self.vvp_vmin_mps = case_dict['vvp_vmin_mps']
            self.vwp.display_manual_sfcobs = case_dict['display_manual_sfcobs']
            self.vwp_manual_sfcobs = case_dict['vwp_manual_sfcobs']
        
        # self.crd.selected_radar is used in self.update_stormmotion and self.pb.change_map_center
        self.crd.selected_radar = radar
        sm = case_dict.get('stormmotion', np.array([0, 0]))
        if type(sm) is dict and 'radar' in sm:
            # In the past only a single SM was supported. This updates SMs using old format to new format
            sm = {date+time:sm}
        self.update_stormmotion(sm)
        if self.pb.map_transforms['aeqd'].radar != radar:
            self.pb.change_map_center()
            
        scale = self.current_case['scale']
        center_shift = self.pb.panel_centers[0] - self.current_case['panel_center']
        trans = self.current_case['translate'][:2]+center_shift
        if not self.cases_use_case_zoom:
            # In this case the original zoom level should be maintained (not the one saved with the case). This is achieved by first 
            # changing view to that saved with the case, and then zooming back to the original zoom level using the zoom technique from 
            # https://github.com/vispy/vispy/blob/main/vispy/visuals/transforms/linear.py
            zoom = self.pb.panels_sttransforms[0].scale[0]/scale[0]
            scale = scale*zoom # Don't multiply in-place, since that would change self.gui.current_case['scale']
            trans = self.pb.panel_centers[0] - (self.pb.panel_centers[0] - trans[:2]) * zoom
        self.pb.set_panels_sttransforms_manually(scale, trans, panzoom_action=False, draw=set_data)

        self.switch_to_case_running = True
        if self.view_nearest_radar and self.use_storm_following_view:
            # The stored radar might not be the nearest radar, especially when combining storm-following view with a time offset.
            # This call of self.crd.process_datetimeinput chooses the nearest radar.
            self.crd.process_datetimeinput(set_data=set_data)
        else:
            self.crd.change_radar(radar, set_data=set_data)
        self.switch_to_case_running = False
                    
        self.pos_markers_latlons = case_dict['pos_markers_latlons'].copy()
        self.update_pos_marker_widgets()
        self.pb.set_sm_pos_markers()
        
        self.set_textbar()        
                
    def get_case_index(self, case_list='current', case_dict='current'):
        case_list = self.current_case_list if case_list == 'current' else case_list
        case_dict = self.current_case if case_dict == 'current' else case_dict
        if case_list is None or case_dict is None:
            return None
        cases_dts = [j['datetime'] for j in case_list]
        case = str(case_dict)
        while True:
            try:
                index = cases_dts.index(case_dict['datetime'])
            except Exception:
                return None
            if str(case_list[index]) == case:
                return index
            else:
                cases_dts[index] = '0'
    
    def get_next_case(self, direction=1):
        current_index = self.get_case_index()
        new_index = np.mod(current_index+direction, len(self.current_case_list))
        if 'cases' in self.ani.continue_type and self.cases_loop_subset:
            case2 = self.ani.loop_start_case_index
            case1 = max(0, case2+1-self.cases_loop_subset_ncases)
            if new_index < case1 or new_index > case2:
                new_index = case1 if direction == 1 else case2
        return self.current_case_list[new_index]
        
    def move_to_next_case(self, call_ID=None, direction=1, time_offset='default', set_data=True):
        if not self.current_case_list is None and not self.current_case is None:
            self.move_to_next_case_running = True
            next_case_dict = self.get_next_case(direction)
            self.switch_to_case(self.current_case_list_name, next_case_dict, time_offset, set_data)
            self.move_to_next_case_running = False
            
        if not call_ID is None:
            self.move_to_next_case_call_ID = call_ID
            
    def back_to_previous_case(self):
        if self.previous_case:
            self.switch_to_case(self.previous_case_list_name, self.previous_case)

    def hover(self, url):
        if url:
            QToolTip.showText(QCursor.pos(), url)
        else:
            QToolTip.hideText()
            
    def get_case_text(self, case_dict):
        descr = self.get_descriptor_for_case(case_dict, short_date=False)
        url_full = ""; url_short = ""
        for url in case_dict['urls']:
            url_first_part = url.replace('https://','').replace('http://','').replace('www.','')[:15]
            url_full += " <A href='"+url+"'>"+url_first_part+"</a>"
            url_short += " "+url_first_part
        return descr, url_full, url_short
    
    def get_case_key(self, case_dict):
        return str(case_dict)
    
    def add_item_to_case_list_widget(self, case_dict):
        case_item = ListWidgetItem()
        self.list_cases.addItem(case_item)
        descr, url_full, url_short = self.get_case_text(case_dict)
        key = self.get_case_key(case_dict)
        self.list_cases_labels[key] = QLabel(descr+url_full)
        self.list_cases_labels[key].setOpenExternalLinks(True)
        # Line below is deactivated, since accessing links by keyboard apparently doesn't go together with QListWidget's item selection feature
        # self.list_cases_labels[key].setTextInteractionFlags(Qt.LinksAccessibleByMouse | Qt.LinksAccessibleByKeyboard)
        self.list_cases_labels[key].linkHovered.connect(self.hover)
        self.list_cases.setItemWidget(case_item, self.list_cases_labels[key])
        
    def modify_item_in_case_list_widget(self, old_case_dict, new_case_dict):
        descr, url_full, url_short = self.get_case_text(old_case_dict)
        old_key = self.get_case_key(old_case_dict)
        descr, url_full, url_short = self.get_case_text(new_case_dict)
        new_key = self.get_case_key(new_case_dict)
        self.list_cases_labels[new_key] = self.list_cases_labels[old_key]
        if new_key != old_key:
            del self.list_cases_labels[old_key]
        self.list_cases_labels[new_key].setText(descr+url_short if self.list_cases.dragEnabled() else descr+url_full)
        
    def modify_case_list(self):
        self.modify_case_listw=QWidget()
        self.modify_case_listw.setWindowTitle('Modify case list')
        layout=QVBoxLayout()
        layout.addWidget(QLabel('Here you can rearrange or delete cases, copy or move cases to another list, or change case labels or add or click URLs.'))
        layout.addWidget(QLabel('Rearranging can be done either by sorting them by datetime, or by dragging and dropping them manually in the list'))
        layout.addWidget(QLabel('(after enabling rearrange mode). You can select multiple cases at once by using Ctrl/Shift.'))

        self.list_cases = QListWidget()
        self.list_cases.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list_cases.setDragDropMode(self.list_cases.InternalMove)
        self.list_cases.setDragEnabled(False)
        self.list_cases.setContentsMargins(0, 0, 0, 0)
        self.list_cases.model().rowsMoved.connect(self.adjust_to_case_swap)
        self.list_cases_labels = {}
        for case in self.current_case_list:
            self.add_item_to_case_list_widget(case)
           
        i_current_case = self.get_case_index()            
        if i_current_case != None:
            self.list_cases.setCurrentRow(i_current_case)
        layout.addWidget(self.list_cases)
        
        hbox = QHBoxLayout()
        button_sort = QPushButton('Sort by datetime', autoDefault=True); hbox.addWidget(button_sort)
        self.button_rearrange = QPushButton('Rearrange mode on', autoDefault=True); hbox.addWidget(self.button_rearrange)
        button_delete = QPushButton('Delete from list', autoDefault=True); hbox.addWidget(button_delete)
        self.button_copy = QPushButton('Copy to other list', autoDefault=True); hbox.addWidget(self.button_copy)
        self.button_move = QPushButton('Move to other list', autoDefault=True); hbox.addWidget(self.button_move)
        button_edit = QPushButton('Edit label/URLs', autoDefault=True); hbox.addWidget(button_edit)
        button_switch = QPushButton('Show case', autoDefault=True); hbox.addWidget(button_switch)

        button_sort.clicked.connect(self.sort_cases)
        self.button_rearrange.clicked.connect(self.enable_or_disable_rearrange_mode)
        button_delete.clicked.connect(self.delete_case_from_list)
        self.button_copy.clicked.connect(lambda: self.open_other_case_list_menu('copy'))
        self.button_move.clicked.connect(lambda: self.open_other_case_list_menu('move'))
        button_edit.clicked.connect(self.edit_case_label)
        button_switch.clicked.connect(self.switch_to_selected_case)
        layout.addLayout(hbox)
        
        self.modify_case_listw.setLayout(layout)
        self.modify_case_listw.resize(self.modify_case_listw.sizeHint())
        self.modify_case_listw.show()

    def sort_cases(self):
        keys = [''.join(self.get_case_text(case)) for case in self.current_case_list] 
        sort_indices = np.argsort(keys)
        self.list_cases.setSortingEnabled(True)
        self.list_cases.sortItems()
        save_list = self.current_case_list.copy()
        for i in range(len(keys)):
            self.current_case_list[i] = save_list[sort_indices[i]]
        with open(cases_lists_filename,'wb') as f:
            pickle.dump(self.cases_lists,f)
            
    def adjust_to_case_swap(self, parent, start, end, destination, row):
        save = self.current_case_list[start]
        if row < start:
            self.current_case_list[row+1:start+1] = self.current_case_list[row:start]
            self.current_case_list[row] = save
        else:
            self.current_case_list[start:row-1] = self.current_case_list[start+1:row]
            self.current_case_list[row-1] = save
        with open(cases_lists_filename,'wb') as f:
            pickle.dump(self.cases_lists,f)
        
    def enable_or_disable_rearrange_mode(self):
        self.list_cases.setDragEnabled(not self.list_cases.dragEnabled())
        for key in self.list_cases_labels:
            descr, url_full, url_short = self.get_case_text(eval(key))
            self.list_cases_labels[key].setText(descr+(url_short if self.list_cases.dragEnabled() else url_full))
        self.button_rearrange.setText('URL mode on' if self.list_cases.dragEnabled() else 'Rearrange mode on')
        
    def delete_case_from_list(self):
        selected_items = self.list_cases.selectedItems()
        if len(selected_items) == 0:
            return

        for case_item in selected_items:
            index = self.list_cases.row(case_item)
            self.list_cases.takeItem(index)
            case_dict = self.current_case_list[index]
            key = self.get_case_key(case_dict)
            del self.list_cases_labels[key]
            self.current_case_list.pop(index)
            
        index = self.get_case_index()
        if index is None:
            self.current_case = self.current_case_list[0] if self.current_case_list else None
            
        with open(cases_lists_filename,'wb') as f:
            pickle.dump(self.cases_lists,f)

    def open_other_case_list_menu(self, case_action='copy'): # or 'move'
        menu = QMenu(self)
        position = self.button_copy.geometry()
        point = position.bottomLeft()
        
        other_case_lists = {i:j for i,j in self.cases_lists.items() if not i == self.current_case_list_name}
        for list_name in other_case_lists:
            action = menu.addAction(list_name+f'  ({len(self.cases_lists[list_name])})')
            function = self.copy_to_other_list if case_action == 'copy' else self.move_to_other_list
            action.triggered.connect(lambda state, list_name=list_name: function(list_name))
        menu.addSeparator()
        action = menu.addAction('Create a new list')
        action.triggered.connect(lambda: self.define_name_new_list(case_action+'_to_other_list'))
                
        menu.popup(self.modify_case_listw.mapToGlobal(point))
    def copy_to_other_list(self, list_name):
        selected_items = self.list_cases.selectedItems()
        for item in selected_items:        
            index = self.list_cases.row(item)
            self.add_new_case_to_list(self.current_case_list[index], list_name)
        with open(cases_lists_filename,'wb') as f:
            pickle.dump(self.cases_lists,f)
    def move_to_other_list(self, list_name):
        self.copy_to_other_list(list_name)
        self.delete_case_from_list()
            
    def edit_case_label(self):
        selected_items = self.list_cases.selectedItems()
        if len(selected_items) == 0:
            return
        index = self.list_cases.row(selected_items[0])
        self.set_label_for_case(self.current_case_list_name, self.current_case_list[index])
        
    def switch_to_selected_case(self):
        selected_items = self.list_cases.selectedItems()
        if len(selected_items) == 0:
            return
        index = self.list_cases.row(selected_items[0])
        self.switch_to_case(self.current_case_list_name, self.current_case_list[index])
        
        
    def show_radar_menu(self, radars, pos):
        menu = QMenu(self)
        for radar in radars:
            action = menu.addAction(radar)
            action.triggered.connect(lambda state, radar=radar: self.crd.change_selected_radar(radar))
        menu.popup(self.mapToGlobal(pos))
        
    def showrightclickMenu(self, pos):
        # FIX (23 juli, op Eriks verzoek): een rechtsklik moet het achtergebleven puntje van de
        # laatste HCLASS/MESH-pop-up-klik (self.pb.visuals['click_marker'], zie nlr_plotting.py)
        # verbergen ZONDER tegelijk het rechtsklikmenu te tonen. Een eerdere poging deed dit in
        # nlr_plotting.py's on_mouse_release, maar deze functie hier wordt OOK rechtstreeks
        # aangeroepen door Qt's eigen customContextMenuRequested-signaal (zie __init__,
        # self.customContextMenuRequested.connect(self.showrightclickMenu)) - op Windows (Eriks
        # platform, zie de docstring/opmerking hieronder) gebeurt dat zelfs NA de expliciete
        # aanroep vanuit on_mouse_release, dus die eerdere fix werd domweg overschreven. Door de
        # check hier, helemaal bovenaan, te zetten - vóór alle andere logica - werkt het nu
        # ongeacht welk van de twee paden deze functie als eerste/laatste aanroept.
        if self.pb.visuals['click_marker'].visible:
            self.pb.visuals['click_marker'].visible = False
            self.pb.update()
            return
        self.rightmouseclick_Qpos = pos
                
        """The order in which events are handled differs among different PC's (different for Linux and Windows at least). For Windows, showing the right-click menu 
        occurs after the mouse_release event in nlr_plotting has been handled, whereas for Linux it occurs immediately after the mouse_press event has been handled.
        Because this latter order causes problems, the menu is not immediately drawn in this case, but the command to draw it is given in the function mouse_info
        from nlr_plotting, and occurs after the mouse_release event has been handled.
        """
        if ft.point_inside_rectangle(self.pb.last_mousepress_pos,self.pb.wpos['main'])[0] and not self.pb.mouse_moved_after_press:
            if self.pb.mouse_hold_right: self.need_rightclickmenu=True; return #The function gets called from self.pb.on_mouse_release
            else: self.need_rightclickmenu=False
            
            if self.pb.radar_mouse_selected:
                # Check whether multiple radars are located very close to the selected one. In that case it is hard or impossible to select
                # each radar by mouse, so in this case the right-click menu lists all these close-proximity radars instead of the actions below.
                selected_radar_xy = self.pb.radarcoords_xy[gv.radars_all.index(self.pb.radar_mouse_selected)]
                distances_to_radars = np.linalg.norm(self.pb.radarcoords_xy-selected_radar_xy, axis=1)
                max_distance = self.pb.get_max_dist_mouse_to_marker(f=0.5)
                radars = [j for i,j in enumerate(gv.radars_all) if distances_to_radars[i] < max_distance]
                if len(radars) > 1:
                    self.show_radar_menu(radars, pos)
                    return
            
            menu=QMenu(self)
                            
            selected_radar = self.pb.radar_mouse_selected if self.pb.radar_mouse_selected else self.crd.selected_radar
            if self.ad[selected_radar].isRunning():
                action = menu.addAction('Stop automatic download for '+selected_radar)
                action.triggered.connect(lambda: self.stop_automatic_download(selected_radar))
            else:
                action = menu.addAction('Start automatic download for '+selected_radar)
                action.triggered.connect(lambda: self.start_automatic_download(selected_radar))

            menu.addSeparator()
            
            action = menu.addAction('Set SM marker')
            if not self.pb.firstplot_performed:
                action.setEnabled(False)
            action.triggered.connect(lambda: self.set_sm_marker_properties())
            
            action = menu.addAction('Remove SM marker')
            if not self.sm_marker_present: 
                action.setEnabled(False)
            action.triggered.connect(lambda: self.remove_marker('sm'))
            
            
            submenu = menu.addMenu('Set SM vector')
            action = submenu.addAction('Calculate from marker')
            if not self.sm_marker_present or\
            self.pb.data_attr['scantime'].get(self.pb.panel, self.sm_marker_scantime) == self.sm_marker_scantime:
                action.setEnabled(False)
            action.triggered.connect(lambda: self.set_stormmotion_from_marker('single'))
            
            action = submenu.addAction('Calculate from marker and set new SM marker')
            if not self.sm_marker_present or\
            self.pb.data_attr['scantime'].get(self.pb.panel, self.sm_marker_scantime) == self.sm_marker_scantime:
                action.setEnabled(False)
            action.triggered.connect(lambda: (self.set_stormmotion_from_marker('single'), self.set_sm_marker_properties()))
                        
            action = submenu.addAction('Set manually')
            action.triggered.connect(lambda: self.set_stormmotion_manually('single'))
                            
            
            submenu = menu.addMenu('Set another SM vector')
            action = submenu.addAction('Calculate from marker')
            if not self.sm_marker_present or\
            self.pb.data_attr['scantime'].get(self.pb.panel, self.sm_marker_scantime) == self.sm_marker_scantime:
                action.setEnabled(False)
            action.triggered.connect(lambda: self.set_stormmotion_from_marker('multiple'))
            
            action = submenu.addAction('Calculate from marker and set new SM marker')
            if not self.sm_marker_present or\
            self.pb.data_attr['scantime'].get(self.pb.panel, self.sm_marker_scantime) == self.sm_marker_scantime:
                action.setEnabled(False)
            action.triggered.connect(lambda: (self.set_stormmotion_from_marker('multiple'), self.set_sm_marker_properties()))
            
            action = submenu.addAction('Set manually')
            action.triggered.connect(lambda: self.set_stormmotion_manually('multiple'))       
            
            
            if self.stormmotion[1] != 0. or len(self.stormmotion_save) == 0.:
                action = menu.addAction('Reset SM vector')
                action.triggered.connect(lambda: self.change_stormmotion(np.array([0, 0])))
                if self.stormmotion[1] == 0.:
                    action.setEnabled(False)
            else:
                action = menu.addAction('Restore SM vector')
                action.triggered.connect(lambda: self.change_stormmotion(self.stormmotion_save, False))
            
            menu.addSeparator()
            
            action = menu.addAction('Copy mouse coordinates to clipboard')
            action.triggered.connect(self.copy_mouse_coordinates_to_clipboard)
            
            action = menu.addAction('Set position marker: Paste from clipboard')
            action.triggered.connect(lambda: self.set_pos_markers_properties('Clipboard'))
            
            action = menu.addAction('Set position marker: Mouse position')
            action.triggered.connect(lambda: self.set_pos_markers_properties('Mouse'))
                                                                                        
            action = menu.addAction('Set position marker: Coordinate input')
            action.triggered.connect(self.set_marker_coordinates)
            
            marker_selected = not self.pb.marker_mouse_selected_index is None
            if len(self.pos_markers_latlons) or len(self.pos_markers_latlons_save) == 0:
                action = menu.addAction('Remove this marker' if marker_selected else 'Remove all markers')
                action.triggered.connect(lambda: self.remove_marker('pos' if marker_selected else 'all'))
                if len(self.pos_markers_latlons) == 0:
                    action.setEnabled(False)
            else:
                action = menu.addAction('Restore position markers')
                action.triggered.connect(self.restore_pos_markers)
                                   
            if self.crd.radar in gv.radars_with_adjustable_startazimuth:
                menu.addSeparator()
                action = menu.addAction('Adjust start azimuth of scans (certain radars)')
                action.triggered.connect(self.set_data_selected_startazimuth)
            
            menu.addSeparator()

            if not self.current_case_list_name is None:
                action = menu.addAction(f'Add case to current list ({self.current_case_list_name})')
                action.triggered.connect(lambda state, list_name=self.current_case_list_name: self.set_label_for_case(list_name))
                if not self.pb.firstplot_performed: 
                    action.setEnabled(False)
                submenu = menu.addMenu('Add case to other list')
            else:
                submenu = menu.addMenu('Add case to list')            
            if not self.pb.firstplot_performed: 
                submenu.setEnabled(False)
                
            for list_name in self.cases_lists:
                if not list_name == self.current_case_list_name:
                    action = submenu.addAction(list_name+f'  ({len(self.cases_lists[list_name])})')
                    action.triggered.connect(lambda state, list_name=list_name: self.set_label_for_case(list_name))                    
            submenu.addSeparator()
            action = submenu.addAction('Create new list')
            action.triggered.connect(lambda: self.define_name_new_list('set_label_for_case'))
            
            action = menu.addAction('Update current case')
            action.triggered.connect(lambda: self.add_case_to_list(self.current_case_list_name, self.current_case, False))
            if not self.current_case_shown():
                action.setEnabled(False)
                
            action = menu.addAction('Edit current case label')
            action.triggered.connect(lambda: self.set_label_for_case(self.current_case_list_name, self.current_case))
            if not self.current_case_shown():
                action.setEnabled(False)
                
            if self.current_case_shown() and 'extra_datetimes' in self.current_case and self.crd.date+self.crd.time in self.current_case['extra_datetimes']:
                action = menu.addAction('Edit label for this extra time')
            else:
                action = menu.addAction('Add extra time to case for animating')
            action.triggered.connect(self.set_label_for_extra_datetime)
            if not self.current_case_shown():
                action.setEnabled(False)
                
            if self.current_case_shown() and 'extra_datetimes' in self.current_case and self.crd.date+self.crd.time in self.current_case['extra_datetimes']:
                action = menu.addAction('Remove this extra time from case')
                action.triggered.connect(lambda: self.add_or_remove_extra_datetimes_case('remove single'))
            else:
                action = menu.addAction('Remove extra times from case')
                action.triggered.connect(lambda: self.add_or_remove_extra_datetimes_case('remove all'))
            if not self.current_case_shown():
                action.setEnabled(False)
            
            menu.popup(self.mapToGlobal(pos))
        elif self.show_vwp and ft.point_inside_rectangle(self.pb.last_mousepress_pos,self.pb.wpos['vwp'])[0] and not self.pb.mouse_moved_after_press:
            self.gui_vwp.showrightclickMenu(pos)

    def current_case_shown(self, mode='loose'):
        if self.current_case is None:
            return False
        
        datetime = self.crd.date+self.crd.time
        case_dts = [self.current_case['datetime']]+list(self.current_case.get('extra_datetimes', {}))
        max_datetimediff_s = max([6*3600, 60*np.abs(self.cases_animation_window).max()])
        if mode == 'strict':
            return datetime == self.current_case['datetime']
        elif mode == 'loose':
            return any(abs(ft.datetimediff_s(datetime, j)) <= max_datetimediff_s for j in case_dts)
                    
        
    def copy_mouse_coordinates_to_clipboard(self):
        x, y = self.pb.screencoord_to_xy(self.pb.last_mousepress_pos)
        lat, lon = ft.aeqd(gv.radarcoords[self.crd.radar], np.array([x, y]), inverse=True)
        pyperclip.copy(f'{lat}, {lon}')
            
    def set_marker_coordinates(self):
        self.marker_coordinates_widget = QWidget()
        self.marker_coordinates_widget.setWindowTitle('Set marker coordinates')
        layout = QVBoxLayout()
        formlayout = QFormLayout()
        
        self.pos_markers_latlonsw = {}
        self.pos_markers_labelsw = {}
        n = len(self.pos_markers_latlons)
        for j in range(max([10, n+1])):
            self.pos_markers_latlonsw[j] = QLineEdit()
            if j < n:
                self.pos_markers_latlonsw[j].setText(', '.join(format(i, '.6f') for i in self.pos_markers_latlons[j]))
            self.pos_markers_labelsw[j] = QLineEdit()
            self.pos_markers_labelsw[j].setPlaceholderText('optioneel label')
            if j < len(self.pos_markers_labels):
                self.pos_markers_labelsw[j].setText(self.pos_markers_labels[j])
            hbox = QHBoxLayout()
            hbox.addWidget(self.pos_markers_latlonsw[j], 5); hbox.addWidget(self.pos_markers_labelsw[j], 3)
            formlayout.addRow(QLabel(f'{j+1}'), hbox)
            self.pos_markers_latlonsw[j].editingFinished.connect(lambda j=j: self.set_pos_markers_properties('Coordinates', j))
            self.pos_markers_labelsw[j].editingFinished.connect(lambda j=j: self.set_pos_marker_label(j))
        
        layout.addWidget(QLabel('Input can be either in decimal or DMS format, and a - sign or the suffix N/S/E/W'))
        layout.addWidget(QLabel('is allowed. Supported formats are at least those of Google Maps, ESWD, Wikipedia,'))
        layout.addWidget(QLabel('SPC storm reports and Tornado Archive.'))
        layout.addWidget(QLabel("Input should be specified in the format 'latitude, longitude'."))
        layout.addWidget(QLabel("Het optionele label wordt getoond in de 3D volume-viewer (Ctrl+Shift+4), bij de"))
        layout.addWidget(QLabel("verticale lijn die op de positie van deze marker wordt getekend."))
        layout.addLayout(formlayout)
        self.marker_coordinates_widget.setLayout(layout)
        self.marker_coordinates_widget.resize(self.marker_coordinates_widget.sizeHint())
        self.marker_coordinates_widget.show()
    
    def set_pos_marker_label(self, index):
        # Los van set_pos_markers_properties, omdat een label ook zonder geldige coordinaten-invoer op deze regel
        # ingesteld moet kunnen worden (bv. eerst coordinaten intypen en pas daarna, in een aparte stap, het label).
        if index >= len(self.pos_markers_latlons):
            return # Geen zin om een label te zetten zonder dat er al een geldige marker op deze regel bestaat.
        while len(self.pos_markers_labels) <= index:
            self.pos_markers_labels.append('')
        self.pos_markers_labels[index] = self.pos_markers_labelsw[index].text().strip()
                
    def set_pos_markers_properties(self, input_type, index=None):
        n = len(self.pos_markers_positions)
        list_index = min([index, n]) if not index is None else n
        
        if input_type in ('Coordinates', 'Clipboard'):
            input_marker_latlon = self.pos_markers_latlonsw[index].text() if input_type == 'Coordinates' else pyperclip.paste()
            try:
                lat, lon = ft.determine_latlon_from_inputstring(input_marker_latlon)
                x, y = ft.aeqd(gv.radarcoords[self.crd.radar], np.array([lat, lon]))
            except Exception:
                if list_index < n and input_marker_latlon == '':
                    self.remove_marker('pos', list_index)
                return
        elif input_type == 'Mouse':
            x, y = self.pb.screencoord_to_xy(self.pb.last_mousepress_pos)
            lat, lon = ft.aeqd(gv.radarcoords[self.crd.radar], np.array([x, y]), inverse=True)
        
        if n <= list_index:
            self.pos_markers_positions.append([])
            self.pos_markers_latlons.append([])
            self.pos_markers_labels.append('')
        self.pos_markers_positions[list_index] = np.array([x, y])
        self.pos_markers_latlons[list_index] = np.array([lat, lon])
        self.update_pos_marker_widgets()
            
        self.pb.set_sm_pos_markers()
        self.pb.update()
        
    def update_pos_marker_widgets(self):
        try:
            for j in range(len(self.pos_markers_latlonsw)):
                if j < len(self.pos_markers_latlons):
                    self.pos_markers_latlonsw[j].setText(', '.join(format(i, '.6f') for i in self.pos_markers_latlons[j]))
                else:
                    self.pos_markers_latlonsw[j].clear()
                if j < len(self.pos_markers_labels):
                    self.pos_markers_labelsw[j].setText(self.pos_markers_labels[j])
                else:
                    self.pos_markers_labelsw[j].clear()
        except Exception:
            # Should only occur when the widgets don't exist
            pass
            
    def set_sm_marker_properties(self):
        self.sm_marker_position = self.pb.screencoord_to_xy(self.pb.last_mousepress_pos)
        self.sm_marker_latlon = ft.aeqd(gv.radarcoords[self.crd.radar],self.sm_marker_position,inverse=True)
        self.sm_marker_datetime = self.crd.date+self.crd.time
        self.sm_marker_scantime = self.pb.data_attr['scantime'][self.pb.panel]
        self.sm_marker_scandatetime = self.pb.data_attr['scandatetime'][self.pb.panel]
        
        self.sm_marker_present=True
        self.pb.set_sm_pos_markers()
        self.pb.update()
        
    def remove_marker(self, mode, index=None):
        self.pos_markers_latlons_save = self.pos_markers_latlons.copy()
        
        if mode in ('sm', 'all'):
            self.sm_marker_present = False    
        if mode == 'pos':
            if index is None:
                index = self.pb.marker_mouse_selected_index
            del self.pos_markers_positions[index]
            del self.pos_markers_latlons[index]
            if index < len(self.pos_markers_labels):
                del self.pos_markers_labels[index]
        elif mode == 'all':
            self.pos_markers_positions, self.pos_markers_latlons, self.pos_markers_labels = [], [], []        
            
        self.update_pos_marker_widgets()
        self.pb.set_sm_pos_markers()
        self.pb.update()
        
    def restore_pos_markers(self):
        self.pos_markers_latlons = self.pos_markers_latlons_save.copy()
        self.update_pos_marker_widgets()
        self.pb.set_sm_pos_markers()


    def set_stormmotion_from_marker(self, mode='single'):
        mouse_position=self.pb.screencoord_to_xy(self.pb.last_mousepress_pos)
        timediff_seconds=ft.datetimediff_s(self.sm_marker_scandatetime, self.pb.data_attr['scandatetime'][self.pb.panel])
        position_diff=mouse_position-self.sm_marker_position
        sm_direction=ft.calculate_azimuth(position_diff)
        sm_speed_mps=np.linalg.norm(position_diff)*1000./timediff_seconds
        if timediff_seconds<0:
            sm_direction=np.mod(sm_direction+180,360); sm_speed_mps*=-1
        self.change_stormmotion(np.array([sm_direction, sm_speed_mps]), convert_units=False, from_marker=True, mode=mode)
                
    def set_stormmotion_manually(self, mode='single'):
        self.set_stormmotion=QWidget()
        self.set_stormmotion.setWindowTitle('Storm motion')
        layout=QFormLayout()
        self.sm_direction=QLineEdit(format(self.stormmotion[0], '.1f')); self.sm_speed=QLineEdit(format(self.stormmotion[1]*self.pb.scale_factors['s'], '.1f'))

        layout.addRow(QLabel('A storm speed of 0 kts removes the storm motion from the title,'))
        layout.addRow(QLabel('except when displaying the storm-relative velocity.'))
        layout.addRow(QLabel('From (degrees)'),self.sm_direction)
        layout.addRow(QLabel('Speed ('+self.pb.productunits['s']+')'),self.sm_speed)

        self.sm_direction.editingFinished.connect(lambda: self.change_stormmotion(mode=mode))
        self.sm_speed.editingFinished.connect(lambda: self.change_stormmotion(mode=mode))  
        
        self.set_stormmotion.setLayout(layout)
        self.set_stormmotion.resize(self.set_stormmotion.sizeHint())
        self.set_stormmotion.show()
        
    def select_nearest_sm_datetime(self, ref_datetime):
        datetime = int(ref_datetime)
        sm_datetimes = np.sort(np.array(list(self.stormmotion_save), dtype='uint64'))
        return str(sm_datetimes[0] if datetime <= sm_datetimes[0] else sm_datetimes[sm_datetimes <= datetime][-1]) 
        
    def update_stormmotion_change_datetime_or_radar(self, datetime=None, radar=None, set_stormmotion=True):
        radar = radar if radar else self.crd.selected_radar
        datetime = datetime if datetime else self.crd.selected_date+self.crd.selected_time
        key = self.select_nearest_sm_datetime(datetime)
        sm_dict = self.stormmotion_save[key]
        
        if hasattr(self, 'previous_sm_request') and self.previous_sm_request[0] == (str(sm_dict), radar):
            sm = self.previous_sm_request[1]
        else:
            """The storm motion vector will be slightly different for an AEQD projection centered on a different radar.
            The method used here is slightly less accurate than reprojecting the full position difference vector,
            and calculating the new SM from this. But it works well enough, especially together with some measures in
            self.ani.update_datetimes_and_perform_firstplot that ensure that each animation iteration starts at the original
            panel view.
            """
            sm = sm_dict['sm']*np.array([np.pi/180, 1])
            # Always use the radar for which the storm motion was set (i.e. self.stormmotion_save['radar']) as reference.
            # Not doing that leads to slightly different SM when reprojecting back and forth between radars.
            old_radar = sm_dict['radar']
            if old_radar != radar:
                pos1, pos2 = np.zeros(2), np.array([np.sin(sm[0]), np.cos(sm[0])])
                pos1_latlon, pos2_latlon = ft.aeqd(gv.radarcoords[old_radar], np.array([pos1, pos2]), inverse=True)
                pos1, pos2 = ft.aeqd(gv.radarcoords[radar], np.array([pos1_latlon, pos2_latlon]))
                position_diff = pos1-pos2
                sm_direction=ft.calculate_azimuth(position_diff)
                sm_speed_mps=np.linalg.norm(position_diff)*sm[1]
                sm = np.array([sm_direction, sm_speed_mps])
            else:
                sm = sm_dict['sm']                
            self.previous_sm_request = [(str(sm_dict), radar), sm]
            
        if set_stormmotion:
            self.stormmotion = sm
        else:
            return sm
        
    def update_stormmotion(self, stormmotion, mode='single'):
        # Updates only storm motion, without performing additional (e.g. plotting) actions
        # stormmotion can be either a 2-element array, or a dictionary of the form {'sm':array, 'radar':radar}
        if type(stormmotion) is dict:
            self.stormmotion_save = stormmotion
            self.update_stormmotion_change_datetime_or_radar()
        else:
            self.stormmotion = stormmotion
            if stormmotion[1] != 0.:
                if mode == 'single':
                    self.stormmotion_save = {}
                self.stormmotion_save[self.crd.date+self.crd.time] = {'sm':stormmotion, 'radar':self.crd.radar}
        
    def change_stormmotion(self, stormmotion=None, convert_units=True, from_marker=False, mode='single'):
        if type(stormmotion) is dict:
            self.update_stormmotion(stormmotion, mode)
        else: 
            if stormmotion is None:
                input_sm_direction, input_sm_speed = self.sm_direction.text(), self.sm_speed.text()
            else:
                # stormmotion should be a 2-element array
                input_sm_direction, input_sm_speed = stormmotion.astype(str)
            
            number1, number2 = ft.to_number(input_sm_direction), ft.to_number(input_sm_speed)
            if not number1 is None and not number2 is None:
                sm_speed_mps = number2
                if convert_units:
                    sm_speed_mps /= self.pb.scale_factors['s']
                self.stormmotion = np.array([number1 % 360, sm_speed_mps])
                if self.stormmotion[1] == 0.:
                    self.stormmotion[0] = 0
                else:
                    # Save also the current radar with the storm motion, since the function update_stormmotion_change_datetime_or_radar
                    # requires information about the radar for which the saved storm motion is valid.
                    datetime = self.crd.date+self.crd.time
                    if from_marker and int(datetime) > int(self.sm_marker_datetime):
                        datetime = self.sm_marker_datetime
                    if mode == 'single':
                        self.stormmotion_save = {}
                    self.stormmotion_save[datetime] = {'sm':self.stormmotion, 'radar':self.crd.radar}                        
            else:
                self.sm_direction.setText(format(self.stormmotion[0], '.1f'))
                self.sm_speed.setText(format(self.stormmotion[1]*self.pb.scale_factors['s'], '.1f'))

        if self.pb.firstplot_performed:
            redraw_panellist = [j for j in self.pb.panellist if self.crd.products[j]=='s' or self.crd.products[j] in gv.plain_products_correct_for_SM]
            self.pb.set_newdata(redraw_panellist)
            if self.show_vwp:
                self.vwp.set_newdata()
                self.pb.set_draw_action('plotting_vwp')
                self.pb.update()
            
            
    def define_name_new_list(self, dest_function = 'set_label_for_case'):
        self.define_new_name=QWidget()
        self.define_new_name.setWindowTitle('Create new list')
        layout=QFormLayout()
        self.list_new_name=QLineEdit()

        layout.addRow(QLabel('Name of new list'),self.list_new_name)

        self.list_new_name.returnPressed.connect(lambda: self.create_new_list(dest_function))
        
        self.define_new_name.setLayout(layout)
        self.define_new_name.resize(self.define_new_name.sizeHint())
        self.define_new_name.show()
                                                                       
    def create_new_list(self, dest_function):
        list_name = self.list_new_name.text()
        if not list_name in self.cases_lists:
            self.set_textbar('')
            self.cases_lists[list_name] = []
            with open(cases_lists_filename,'wb') as f:
                pickle.dump(self.cases_lists,f)
            self.define_new_name.close()
            if dest_function == 'set_label_for_case':
                self.set_label_for_case(list_name)
            else:
                function = self.copy_to_other_list if dest_function == 'copy' else self.move_to_other_list
                function(list_name)
        else:
            self.set_textbar('This name already exists. Either choose a different name, or first delete the existing list.','red',2)
    
    def set_label_for_case(self, list_name, case_dict = None):
        # Set case_dict if you want to update an existing case
        self.set_case_label=QWidget()
        self.set_case_label.setWindowTitle(list_name)
        
        layout=QVBoxLayout()
        form_layout=QFormLayout()
        self.case_label=QLineEdit()
        if not case_dict is None:
            self.case_label.setText(case_dict['label'])
        form_layout.addRow(QLabel('Case label (optional, can be left empty)'),self.case_label)
        self.case_URLs = {}
        for i in range(5):
            self.case_URLs[i] = QLineEdit()
            if not case_dict is None and len(case_dict['urls']) > i:
                self.case_URLs[i].setText(case_dict['urls'][i])
            form_layout.addRow(QLabel(f'URL {i+1} (optional, can be left empty)'),self.case_URLs[i])
        self.case_button = QPushButton('Add case' if case_dict is None else 'Update label', autoDefault=True)
        self.case_button.clicked.connect(lambda: self.add_case_to_list(list_name, case_dict))
        layout.addLayout(form_layout)    
        layout.addWidget(self.case_button)
            
        self.set_case_label.setLayout(layout)
        self.set_case_label.resize(self.set_case_label.sizeHint())
        self.set_case_label.show()
            
    def add_new_case_to_list(self, case_dict, list_name=None):
        case_list = self.current_case_list if list_name is None else self.cases_lists[list_name]
            
        keys = [''.join(self.get_case_text(case)) for case in case_list]
        cases_sorted_before = (keys == np.sort(keys)).all()
        if cases_sorted_before:
            case_list.insert(bisect.bisect(keys, ''.join(self.get_case_text(case_dict))), case_dict)
        else:
            case_list += [case_dict]
                    
    def add_case_to_list(self, list_name, case_dict = None, update_label = True):
        # Set case_dict if you want to update an existing case
        # Set update_label to False if you want to keep using the current label for the case, and only want to update the case specs themselves
        if update_label:
            label = self.case_label.text()
            urls = []
            for i in self.case_URLs:
                if not self.case_URLs[i].text() == '':
                    urls.append(self.case_URLs[i].text())
        else:
            label, urls = case_dict['label'], case_dict['urls']
        
        if case_dict is None or not update_label:
            info = {'datetime': self.crd.date+self.crd.time,
                    'scannumbers_forduplicates': self.dsg.scannumbers_forduplicates,
                    'radar': self.crd.radar,
                    'translate': self.pb.panels_sttransforms[0].translate,
                    'scale': self.pb.panels_sttransforms[0].scale,
                    'panel_center': self.pb.panel_centers[0],
                    'pos_markers_latlons': self.pos_markers_latlons.copy()
                    }
            if not self.pb.data_empty[0] and not self.pb.data_isold[0]:
                info['scantime'] = self.pb.data_attr['scantime'][0]
                info['scandatetime'] = self.pb.data_attr['scandatetime'][0]
            if self.stormmotion[1] != 0.:
                info['stormmotion'] = self.stormmotion_save
            if self.show_vwp:
                info['vvp_range_limits'] = self.vvp_range_limits.copy()
                info['vvp_height_limits'] = self.vvp_height_limits.copy()
                info['vvp_vmin_mps'] = self.vvp_vmin_mps
                info['display_manual_sfcobs'] = self.vwp.display_manual_sfcobs
                info['vwp_manual_sfcobs'] = self.vwp_manual_sfcobs
            if case_dict and 'extra_datetimes' in case_dict:                
                info['extra_datetimes'] = case_dict['extra_datetimes']
        else: 
            info = case_dict.copy()
        info['label'] = label
        info['urls'] = urls
        
        self.current_case_list_name = list_name
        self.current_case_list = self.cases_lists[list_name]
        self.current_case = info
        self.set_textbar()
        
        already_present = self.get_case_index(self.current_case_list, info) != None
        if not already_present or case_dict:
            if case_dict:
                index = self.get_case_index(self.current_case_list, case_dict)
                self.current_case_list[index] = info
                
                if hasattr(self, 'modify_case_listw') and self.modify_case_listw.isVisible():
                    self.modify_item_in_case_list_widget(case_dict, info)
            else:
                self.add_new_case_to_list(info, list_name)
                if hasattr(self, 'modify_case_listw') and self.modify_case_listw.isVisible():
                    self.add_item_to_case_list_widget(info)
                
            if hasattr(self, 'set_case_label'):
                # This is always done when this widget is open, even when updating only the case and not the label. Reason is that a change
                # in case_dict means that line self.case_button.clicked.connect(lambda: self.add_case_to_list(list_name, case_dict)) in
                # self.set_label_for_case now refers to an old version of case_dict, which doesn't exist anymore. And this would lead to errors
                # in this function. 
                self.set_case_label.close()
            
            with open(cases_lists_filename,'wb') as f:
                pickle.dump(self.cases_lists,f)
        else:
            self.set_textbar('The exact same case already exists', 'red', 1)
            
    def set_label_for_extra_datetime(self):
        # Set case_dict if you want to update an existing case
        self.set_extra_datetime_label=QWidget()
        self.set_extra_datetime_label.setWindowTitle('Optional label')
        
        layout=QVBoxLayout()
        self.extra_datetime_label=QLineEdit()
        if 'extra_datetimes' in self.current_case and self.crd.date+self.crd.time in self.current_case['extra_datetimes']:
            self.extra_datetime_label.setText(self.current_case['extra_datetimes'][self.crd.date+self.crd.time]['label'])
        self.extra_datetime_label.returnPressed.connect(lambda: self.add_or_remove_extra_datetimes_case('add'))
        layout.addWidget(QLabel('Label for extra time (optional, can be left empty). Press Enter to confirm.'))
        layout.addWidget(self.extra_datetime_label)
            
        self.set_extra_datetime_label.setLayout(layout)
        self.set_extra_datetime_label.resize(self.set_extra_datetime_label.sizeHint())
        self.set_extra_datetime_label.show()
            
    def add_or_remove_extra_datetimes_case(self, mode):
        if mode == 'add':
            if not 'extra_datetimes' in self.current_case:
                self.current_case['extra_datetimes'] = {}
            label = self.extra_datetime_label.text()
            self.set_extra_datetime_label.close()
            info = {'label': label}
            if not self.pb.data_empty[0] and not self.pb.data_isold[0]:
                info['scantime'] = self.pb.data_attr['scantime'][0]
            self.current_case['extra_datetimes'][self.crd.date+self.crd.time] = info
        elif mode == 'remove single':
            del self.current_case['extra_datetimes'][self.crd.date+self.crd.time]
        elif mode == 'remove all':
            self.current_case['extra_datetimes'] = {}
            
        if len(self.current_case['extra_datetimes']) == 0:
            del self.current_case['extra_datetimes']
        
        with open(cases_lists_filename,'wb') as f:
            pickle.dump(self.cases_lists,f)
            
        self.set_textbar()

    def set_data_selected_startazimuth(self):
        self.set_startazimuth=QWidget()
        self.set_startazimuth.setWindowTitle('Start azimuth of scans')
        layout=QFormLayout()
        self.data_selected_startazimuthw=QLineEdit(format(self.data_selected_startazimuth, '.0f'))

        layout.addRow(QLabel('Change the start azimuth of the scans. It might be desired to change this'))
        layout.addRow(QLabel('azimuth since there is a discontinuity in the data where the scan starts,'))
        layout.addRow(QLabel('which is undesired when interesting echoes are located there.'))
        layout.addRow(QLabel('Start azimuth (degrees)'),self.data_selected_startazimuthw)

        self.data_selected_startazimuthw.editingFinished.connect(self.change_data_selected_startazimuth)
        
        self.set_startazimuth.setLayout(layout)
        self.set_startazimuth.resize(self.set_startazimuth.sizeHint())
        self.set_startazimuth.show()
        
    def change_data_selected_startazimuth(self):
        number = ft.to_number(self.data_selected_startazimuthw.text())
        if not number is None:
            self.data_selected_startazimuth = number
        self.data_selected_startazimuthw.setText(format(self.data_selected_startazimuth, '.0f'))
        self.pb.set_newdata(self.pb.panellist)
                        

    def change_plot_mode(self, new_mode):
        self.crd.plot_mode = new_mode
        
    def import_choices(self):
        try:
            try:
                with open(opa(os.path.join(gv.userdir+'/Generated_files/','saved_choices.pkl')),'rb') as f:
                    choices = pickle.load(f)
            except Exception: 
                with open(opa(os.path.join(gv.programdir+'/Input_files/','saved_choices_default.pkl')),'rb') as f:
                    choices = pickle.load(f)
        except Exception:
            choices = {}
        return choices
        
    def save_choice(self,choice_ID):
        if self.pb.firstplot_performed:
            choices = self.import_choices()
            dist_to_radar = {j:np.linalg.norm(self.pb.corners[j].mean(axis=0)) for j in self.pb.panellist}
            selected_heights = self.dsg.get_panel_center_heights(dist_to_radar, self.dsg.selected_scanangles, 
                                                                 self.dsg.get_scanangles_allproducts(self.dsg.scanangles_all_m), self.dsg.scanpair_present)
            selected_heights = {i:j for i,j in selected_heights.items() if self.crd.products[i] in gv.products_with_tilts}
            selected_scanangles = {i:j for i,j in self.dsg.selected_scanangles.items() if self.crd.products[i] in gv.products_with_tilts}
            choices[choice_ID] = {'panels':self.pb.panels, 'products':self.crd.products, 'selected_scanangles':selected_scanangles, 
                                  'selected_heights':selected_heights, 'range_nyquistvelocity_scanpairs_indices':self.dsg.range_nyquistvelocity_scanpairs_indices}
            with open(gv.userdir+'/Generated_files/saved_choices.pkl', 'wb') as f:
                pickle.dump(choices, f, protocol=2)

    def set_choice(self,choice_ID): 
        choices = self.import_choices()
        if not choice_ID in choices: 
            return
        
        choice = choices[choice_ID]
        # For backward compatibility:
        for attr in ('selected_scanangles', 'selected_heights'):
            choice[attr] = {i:j for i,j in choice[attr].items() if not j is None}
        
        panels_new = choice['panels']
        new_panellist = [self.pb.plotnumber_to_panelnumber[panels_new][j] for j in range(panels_new)]
        for j in new_panellist:
            # BUGFIX (27 juli 2026): zelfde vangnet als bij de opstart-productlijst - een opgeslagen
            # F-toets-preset kan een inmiddels verwijderde productcode bevatten (bv. 'hd'/HDR).
            self.crd.products[j] = choice['products'][j] if choice['products'][j] in gv.products_all else 'z'
            if self.crd.scan_selection_mode in ('scan', 'scanangle') and j in choice['selected_scanangles']:
                self.dsg.selected_scanangles[j] = choice['selected_scanangles'][j]
            if j in choice['range_nyquistvelocity_scanpairs_indices']:
                #Using self.dsg.range_nyquistvelocity_scanpairs_indices makes it possible to determine which scan is desired when there is 
                #a scan pair in which one has a large range but low Nyquist velocity, and the other a smaller range but larger Nyquist velocity.
                self.dsg.range_nyquistvelocity_scanpairs_indices[j] = choice['range_nyquistvelocity_scanpairs_indices'][j]
        if self.crd.scan_selection_mode == 'height':
            self.dsg.manually_set_panel_center_heights(choice['selected_heights'])
        
        for j in new_panellist:
            if self.crd.products[j] in gv.plain_products_with_parameters:
                self.change_PP_parameters_panels(self.crd.products[j])
        
        self.setting_saved_choice=True #Is used in the function self.pb.change_panels to let the program know that it is not desired to call
        #the functions self.crd.row_mode or self.crd.column_mode, as this could change the products and scans.
        #Is also used in the function self.dsg.check_need_scans_change, to let the program know that it is desired to choose a scan
        #with an appropriate Nyquist velocity, irregardless of what the current scan is.        
        
        if panels_new != self.pb.panels:
            if not self.pb.firstplot_performed:
                self.crd.process_datetimeinput()
            self.pb.panellist = new_panellist
            self.pb.change_panels(panels_new)
        else:
            self.crd.process_datetimeinput()
        self.setting_saved_choice=False

    def view_choices(self):
        choices = self.import_choices()
        
        choices_text = {'angle':{}, 'height':{}}
        for i in range(1, 13):
            choices_text['angle'][i] = choices_text['height'][i] = ''
            if i in choices:
                panels = choices[i]['panels']
                panellist = [self.pb.plotnumber_to_panelnumber[panels][j] for j in range(panels)]
                for k in choices_text:
                    # Errors can occur when selected_heights and selected_scanangles are not compatible (i.e. stored for different number
                    # of panels or different products). This needs to be improved later, for now a try-except block is used.
                    try:
                        for j in panellist:
                            product, scanangle = choices[i]['products'][j], choices[i]['selected_scanangles'].get(j, None)
                            height = choices[i].get('selected_heights', {}).get(j, scanangle)
                            specs = product
                            if not product in gv.plain_products:
                                specs += ft.format_nums(scanangle) if k == 'angle' else ft.format_nums(height)
                            choices_text[k][i] += ' '*bool(j and specs) + specs
                        choices_text[k][i] = self.compress_choice_text_and_split_rows(choices_text[k][i])
                    except Exception as e:
                        print(e, k, choices[i])
                        choices_text[k][i] = {0:'', 1:''}

        self.choices_widget = QWidget()
        self.choices_widget.setWindowTitle('Saved panel configurations')  

        layout = QGridLayout()
        layout.addWidget(QLabel('Choice'), 0, 0)
        layout.addWidget(QLabel('Elevation angle'), 0, 1)
        layout.addWidget(QLabel('Height'), 0, 4)
        self.choice_edits = {'angle':{}, 'height':{}}
        for i in choices:
            for j in self.choice_edits:
                self.choice_edits[j][i] = {}
                for k in (0, 1):
                    self.choice_edits[j][i][k] = QLineEdit(choices_text[j][i][k])
                    self.choice_edits[j][i][k].editingFinished.connect(lambda j=j, i=i: self.process_choice_edit(j, i))
            layout.addWidget(QLabel(str(i)), i, 0)
            layout.addWidget(self.choice_edits['angle'][i][0], i, 1)
            layout.addWidget(self.choice_edits['angle'][i][1], i, 2)
            layout.addWidget(QLabel(' '), i, 3)
            layout.addWidget(self.choice_edits['height'][i][0], i, 4)
            layout.addWidget(self.choice_edits['height'][i][1], i, 5)
        
        self.choices_widget.setLayout(layout)
        self.choices_widget.resize(self.choices_widget.sizeHint())
        self.choices_widget.show()
        
    def compress_choice_text_and_split_rows(self, text):
        def get_product(item):
            return ''.join([k for k in item if k.isalpha()])
        def get_value(item):
            return ''.join([k for k in item if k.isdigit() or k == '.'])
        def get_product_upper(product):
            return 'u'*(len(product) > 1)+product[-1].upper()
        def is_product_upper(product):
            return product and product == get_product_upper(product)
        items = text.split(' ')
        panels = len(items)
        cmpr_text = {0:'', 1:''}
        for i,j in enumerate(items):
            row = self.pb.rows_panels[panels][i]
            ncols = self.pb.cols_panels[panels][panels-1]+1
            product, value = get_product(j), get_value(j)
            value_before = items[i-1][1:]
            if i and value and product in items[i-1]:
                specs = value
            elif i and value and value == value_before:
                specs = product
            else:
                specs = product+value
            value_row0 = get_value(items[i % ncols])
            if row and specs != product and value and value == value_row0:
                last_item_row = cmpr_text[row].split(' ')[-1]
                if last_item_row and get_product(last_item_row) == get_product_upper(product):
                    n_present = cmpr_text[row][-1].isdigit()
                    n = int(cmpr_text[row][-1])+1 if n_present else 2
                    cmpr_text[row] = cmpr_text[row][:(-1 if n_present else None)]+str(n)
                    specs = ''
                else:
                    specs = get_product_upper(product)
            cmpr_text[row] += ' '*bool(cmpr_text[row] and specs) + specs
        all_upper = all(is_product_upper(get_product(j)) for j in cmpr_text[1].split(' ')) if cmpr_text[1] else False
        if all_upper:
            cmpr_text[1] = cmpr_text[1][:-1]
        return cmpr_text
    def combine_rows_and_decompress_choice_text(self, cmpr_text):
        items = {}
        for i in (0, 1):
            items[i] = cmpr_text[i].split(' ')
            if items[i][0]:
                for j,k in enumerate(items[i]):
                    if k[0].isalpha():
                        product = k[0]
                    product = product.lower() if k[0] != product else product
                    k_notproduct = k.replace(product, '')
                    if k_notproduct.replace('.', '').isdigit():
                        value = k_notproduct
                    elif product.isupper() or product in gv.plain_products:
                        value = ''
                    items[i][j] = product+value
        text = ' '.join(items[0])
        if items[1][0]:
            for i,j in enumerate(items[1]):
                if j[0].isupper():
                    product = j[0].lower()
                    n = int(j[1:]) if j[1:] else 1+len(items[0])-len(items[1])
                    specs = ' '.join([product+k[1:] for k in items[0][i:i+n]])
                else:
                    specs = j
                text += ' '*bool(text and specs) + specs
        return text
        
    def process_choice_edit(self, j, i): #j is value type ('angle'/'height'), i is choice
        choice_text = self.choice_edits[j][i][0].text().strip(), self.choice_edits[j][i][1].text().strip()
        choice_text = self.combine_rows_and_decompress_choice_text(choice_text)
        print(choice_text)
        items = choice_text.split(' ')
        panels = len(items)
        error = True
        if panels in self.pb.rows_panels:
            panellist = [self.pb.plotnumber_to_panelnumber[panels][j] for j in range(panels)]
            products, values = {}, {}
            for idx,k in enumerate(items):
                p = panellist[idx]
                if k[0] in gv.products_all and (k in gv.plain_products or k[1:].replace('.', '').isdigit()):
                    products[p] = k[0]
                    if k[1:]:
                        values[p] = float(k[1:])
            error = len(products) < panels
        for k in (0, 1):
            self.choice_edits[j][i][k].setStyleSheet('QLineEdit {color:'+('red' if error else 'black')+'}')
        if error:
            return
        
        choices = self.import_choices()
        key = 'selected_scanangles' if j == 'angle' else 'selected_heights'
        choices[i].update({'panels':panels, 'products':products, key:values})
        with open(opa(os.path.join(gv.userdir+'/Generated_files','saved_choices.pkl')),'wb') as f:
            pickle.dump(choices,f,protocol=2)
            

        
    def show_archiveddays(self):
        dates=self.dsg.get_dates_with_archived_data(self.crd.selected_radar,self.crd.selected_dataset)
        
        input_date=self.datew.text()
        if ft.correct_datetimeinput(input_date,'0000') and not input_date=='c':
            radars=self.dsg.get_radars_with_archived_data_for_date(input_date)
        else:
            radars=[]
        
        text='Radars with archived data for '+(input_date[1:] if 'c' in input_date and input_date!='c' else input_date)+': \n'
        for j in range(0,len(radars)):
            text+=radars[j]
            if j!=len(radars)-1:
                text+=', '
        if len(radars)==0:
            text+='-'
        text+='\n\n'
        
        text+='Dates with archived data for '+self.crd.selected_radar
        if self.crd.selected_radar in gv.radars_with_datasets:
            text+=' '+self.crd.selected_dataset+': \n'
        else: text+=': \n'
        for j in range(0,len(dates)):
            text+=str(dates[j])
            if j!=len(dates)-1:
                text+=', '
        if len(dates)==0:
            text+='- \n'

        msgbox_archiveddata=QMessageBox()
        msgbox_archiveddata.about(self, 'Dates with archived data', text)
                
    def show_scans_properties(self):
        if self.pb.firstplot_performed:
            properties_text='Scan angle, slant range, radial resolution, Nyquist velocity, low and high dual PRF Nyquist velocity. \n' +\
            'The low and high Nyquist velocity determine the correction terms that are applied to aliased velocities during dual-PRF dealiasing. '+\
            'If these are equal, then the possible correction terms are the same for both even and odd radials.\n'+\
            '--------------------------------------------------------------------------------------------------------------------\n'
            # The dashed line above is added to force QMessageBox to have a certain width, since without this it can be undesirably narrow. 
            for j in self.dsg.scanangles_all['z']:
                properties_text += 'Scan '+str(j)+': '+"%.2f" % self.dsg.scanangles_all_m['z'][j]+" \xb0, "+"%.1f" % self.dsg.radial_range_all['z'][j]+' km, '+"%.4f" % self.dsg.radial_res_all['z'][j]+' km, '
                if not self.dsg.nyquist_velocities_all_mps[j] is None:
                    properties_text+="%.1f" % (self.dsg.nyquist_velocities_all_mps[j]*self.pb.scale_factors['v'])+' '+self.pb.productunits['v'] +\
                    ', '+(("%.1f" % (self.dsg.low_nyquist_velocities_all_mps[j]*self.pb.scale_factors['v'])+' '+self.pb.productunits['v'] +\
                    ', '+"%.1f" % (self.dsg.high_nyquist_velocities_all_mps[j]*self.pb.scale_factors['v'])+' '+self.pb.productunits['v']) \
                    if not self.dsg.low_nyquist_velocities_all_mps[j] is None else '--, --')
                else: 
                    #If it was impossible to obtain the Nyquist velocities, then they are set to '--'
                    properties_text+='--, --, --'
                    
                if j!=list(self.dsg.scanangles_all['z'])[-1]:
                    properties_text+='\n'
                    
            msgbox_scanproperties=QMessageBox()
            msgbox_scanproperties.about(self, 'Scan properties', properties_text)

        
    def show_fullscreen(self):
        self.changing_fullscreen=True #Is set to False in the function self.pb.on_resize.
        if not self.fullscreen:
            self.showFullScreen(); self.fullscreen=True
        else:
            self.showMaximized(); self.fullscreen=False
        
    def change_continue_savefig(self, animation=False):
        if (self.creating_animation and not animation) or (self.continue_savefig and not self.creating_animation and animation):
            # When creating an animated gif this function should not respond to a "regular" continue savefig event, and vice versa.
            return
        
        self.continue_savefig=not self.continue_savefig
        if animation:
            self.creating_animation = not self.creating_animation
            if self.creating_animation:
                qm = QMessageBox
                choice = qm.No
                if hasattr(self, 'ani_widget') and self.ani_widget.isVisible():
                    choice = qm.question(self, 'NLradar', 'Do you want to retain any existing animation frames?', qm.Yes | qm.No)
                if choice == qm.No:
                    self.starting_animation = True
                    if os.path.exists(gv.animation_frames_directory):
                        try:
                            shutil.rmtree(gv.animation_frames_directory)
                        except Exception as e:
                            self.set_textbar(str(e), 'red', minimum_display_time=5)
                            return
                os.makedirs(gv.animation_frames_directory, exist_ok=True)
            else:
                self.save_animation()
        
        self.set_textbar() #Change the color into blue when self.continue_savefig=True, else the previous color
        if self.continue_savefig and not animation:
            save_continue_type=self.ani.continue_type; save_direction=self.ani.direction
            self.ani.continue_type='None' #Stop an animation or continuation to the left or right, because this would continue when the 
            #widget for selecting a directory is opened, which is not desired.
            self.savefig()
            if not save_continue_type is None: #Restart the animation or continuation to the left/right.
                self.ani.change_continue_type(save_continue_type,save_direction)
                
                
    def save_animation(self):
        self.ani_widget = QWidget()
        self.ani_widget.setWindowTitle('Save animation')
        
        self.ani_directoryw = QPushButton(os.path.dirname(self.animation_filename))
        self.ani_filenamew = QLineEdit(os.path.basename(self.animation_filename))
        self.ani_delay_framew = QRadioButton('frame')
        self.ani_delay_minutew = QRadioButton('minute')
        self.ani_delay_framew.setChecked(True) if self.ani_delay_ref == 'frame' else self.ani_delay_minutew.setChecked(True)
        group = QButtonGroup(); group.addButton(self.ani_delay_framew); group.addButton(self.ani_delay_minutew)
        hbox1 = QHBoxLayout(); hbox1.addWidget(self.ani_delay_framew); hbox1.addWidget(self.ani_delay_minutew)
        self.ani_delayw = QLineEdit(str(self.ani_delay))
        self.ani_delay_endw = QLineEdit(str(self.ani_delay_end))
        self.ani_start_datetimew = QLineEdit()
        self.ani_end_datetimew = QLineEdit()
        hbox2 = QHBoxLayout(); hbox2.addWidget(self.ani_start_datetimew); hbox2.addWidget(self.ani_end_datetimew)
        self.ani_sort_filesw = QCheckBox()
        self.ani_sort_filesw.setTristate(False)
        self.ani_sort_filesw.setCheckState(2 if self.ani_sort_files else 0)
        self.ani_group_datasetsw = QCheckBox()
        self.ani_group_datasetsw.setTristate(False)
        self.ani_group_datasetsw.setCheckState(2 if self.ani_group_datasets else 0)
        self.ani_qualityw = QLineEdit(str(self.ani_quality))
        self.ani_create_gifw = QPushButton('GIF')
        self.ani_create_mp4w = QPushButton('MP4')
        hbox3 = QHBoxLayout(); hbox3.addWidget(self.ani_create_gifw); hbox3.addWidget(self.ani_create_mp4w)
        
        self.ani_directoryw.clicked.connect(self.select_ani_directory)
        self.ani_delay_minutew.released.connect(lambda: self.check_delay_ref_sort_files_consistency('delay_ref'))
        self.ani_sort_filesw.stateChanged.connect(lambda: self.check_delay_ref_sort_files_consistency('sort_files'))
        self.ani_create_gifw.clicked.connect(lambda: self.create_ani('gif'))
        self.ani_create_mp4w.clicked.connect(lambda: self.create_ani('mp4'))
        
        layout = QFormLayout()
        layout.addRow(QLabel('Directory'), self.ani_directoryw)
        layout.addRow(QLabel('Filename (no extension needed)'), self.ani_filenamew)
        layout.addRow(QLabel('Set delay per'), hbox1)
        layout.addRow(QLabel('Delay between frames or minutes (cs)'), self.ani_delayw)
        layout.addRow(QLabel('Delay for final frame (cs)'), self.ani_delay_endw)
        layout.addRow(QLabel('All frames in the directory '+gv.animation_frames_directory))
        layout.addRow(QLabel('will be added to the animation. If you want to exclude some frames you can delete them'))
        layout.addRow(QLabel('from this directory, or if this suffices you can below specify a start and/or end (date)time'))
        layout.addRow(QLabel('between which to include frames.'))
        layout.addRow(QLabel('Start/end, format (YYYYMMDD)HHMM(SS)'), hbox2)
        layout.addRow(QLabel('Below you can choose to sort frames chronologically, and to group frames by dataset (i.e. by'))
        layout.addRow(QLabel('radar/dataset/subdataset). Frames for different cases are always grouped by case.'))
        layout.addRow(QLabel('When setting delay per minute, frames will always be sorted chronologically.'))
        layout.addRow(QLabel('Sort frames chronologically'), self.ani_sort_filesw)
        layout.addRow(QLabel('Group frames by dataset'), self.ani_group_datasetsw)
        layout.addRow(QLabel('MP4 video quality (0-10)'), self.ani_qualityw)
        layout.addRow(QLabel('Create animation'), hbox3)
        layout.addRow(QLabel('As long as you keep this window open, you can recreate the animation with different settings,'))
        layout.addRow(QLabel('or press CTRL+SHIFT+S again and start adding more frames to the animation.'))
         
        self.ani_widget.setLayout(layout)
        self.ani_widget.resize(self.ani_widget.sizeHint())
        self.ani_widget.show()
        
    def select_ani_directory(self):
        text = 'Select a directory'
        selected_dir = str(QFileDialog.getExistingDirectory(None,text,os.path.dirname(self.animation_filename)))
        if selected_dir:
            self.animation_filename = opa(selected_dir+'/'+os.path.basename(self.animation_filename))
            self.ani_directoryw.setText(selected_dir)
            
    def check_delay_ref_sort_files_consistency(self, change):
        if change == 'delay_ref':
            self.ani_sort_filesw.setCheckState(2)
        elif change == 'sort_files' and not self.ani_sort_filesw.isChecked():
            self.ani_delay_framew.setChecked(True)
        
    def set_ani_parameters(self):
        filename = self.ani_filenamew.text()
        if any(filename.endswith(j) for j in ('.gif', '.mp4')):
            filename = filename[:-4]
        self.animation_filename = opa(os.path.dirname(self.animation_filename)+'/'+filename)
        
        self.ani_delay_ref = 'frame' if self.ani_delay_framew.isChecked() else 'minute'
        
        number = ft.to_number(self.ani_delayw.text())
        if not number is None:
            self.ani_delay = ft.rifdot0(float(number))
        self.ani_delayw.setText(str(self.ani_delay))
        
        number = ft.to_number(self.ani_delay_endw.text())
        if not number is None:
            self.ani_delay_end = int(number)
        self.ani_delay_endw.setText(str(self.ani_delay_end))
        
        self.ani_sort_files = self.ani_sort_filesw.checkState() == 2
        self.ani_group_datasets = self.ani_group_datasetsw.checkState() == 2
        
        number = ft.to_number(self.ani_qualityw.text())
        if not number is None:
            self.ani_quality = ft.rifdot0(float(number))
        self.ani_qualityw.setText(str(self.ani_quality))
        
    def create_ani(self, ext='gif'):
        self.set_ani_parameters()
        
        try:
            frames_all = os.listdir(gv.animation_frames_directory)
            frame_numbers = [int(j[5:-4]) for j in frames_all]
            sort_output = sorted(zip(frame_numbers, frames_all))
            frames_all = [gv.animation_frames_directory+'/'+j[1] for j in sort_output]
            frame_numbers = np.array([j[0] for j in sort_output])
            
            frames_all, frames_datetimes, frames_datasets = np.array(frames_all), np.array(self.ani_frames_datetimes), np.array(self.ani_frames_datasets)
            if not len(frames_all) == len(frames_datetimes):
                # Some frames might have been removed manually from the directory, in which case the corresponding datasets
                # and datetimes should also be removed
                frames_datetimes = frames_datetimes[frame_numbers-1]
                frames_datasets = frames_datasets[frame_numbers-1]
                
            s1, s2 = self.ani_start_datetimew.text(), self.ani_end_datetimew.text()
            if s1 or s2:
                dts_frames = frames_datetimes[:,0]
                select = np.s_[:]
                if len(s1) in (0, 12, 14) and len(s2) in (0, 12, 14):
                    dt_start = None if not s1 or not s1.isdigit() else s1
                    dt_end = None if not s2 or not s2.isdigit() else s2
                    select = np.array([(not dt_start or ft.datetimediff_s(dt_start, j) >= 0) and (not dt_end or ft.datetimediff_s(j, dt_end) >= 0)
                                       for j in dts_frames])
                elif len(s1) in (0, 4, 6) and len(s2) in (0, 4, 6):
                    t_start = None if not s1 or not s1.isdigit() else s1
                    t_end = None if not s2 or not s2.isdigit() else s2
                    select = np.array([(not t_start or ft.timediff_s(t_start, j[-6:]) >= 0) and (not t_end or ft.timediff_s(j[-6:], t_end) >= 0)
                                       for j in dts_frames])
                frames_all, frames_datetimes, frames_datasets = frames_all[select], frames_datetimes[select], frames_datasets[select]
            
            datasets, indices = np.unique(frames_datasets, return_index=True)
            first_digits = [''.join([c for i,c in enumerate(j) if j[:i+1].isdigit()]) for j in datasets]
            viewing_cases = all(len(j) > 0 for j in first_digits) and len(np.unique(first_digits)) > 1
            if not self.ani_group_datasets and not viewing_cases:
                # Set all datasets equal, in order to prevent grouping different datasets separately.
                # Different cases are always grouped separately.
                frames_datasets = np.full(len(frames_datasets), '')
                datasets, indices = [''], [0]
            if self.ani_group_datasets and not viewing_cases:
                # Retain the original order for the unique datasets unless the datasets contain case indices
                datasets = frames_datasets[np.sort(indices)]
            
            frames, deltas = [], []
            for dataset in datasets:
                select = frames_datasets == dataset
                if self.ani_sort_files:
                    datetimes = frames_datetimes[select]
                    datetimes = sorted([list(dt_i)+[i] for i,dt_i in enumerate(datetimes)])
                    indices = [j[-1] for j in datetimes]
                    new_frames = list(frames_all[select][indices])
            
                    for i,dt_i in enumerate(datetimes[:-1]):
                        n = min(len(dt_i), len(datetimes[i+1])) - 1 # -1, since the last column of datetimes contains an index, see above
                        for j in range(n):
                            timediff = ft.datetimediff_s(dt_i[j], datetimes[i+1][j])
                            if timediff > 1:
                                frames.append(new_frames[i])
                                deltas.append(timediff/60*self.ani_delay)
                                break
                    frames.append(new_frames[-1])
                    deltas.append('ani_delay_end')
                else:
                    frames += list(frames_all[select])
                    deltas += [self.ani_delay]*(np.count_nonzero(select)-1)+['ani_delay_end']
            
            deltas = [(self.ani_delay if self.ani_delay_ref == 'frame' else j) if j != 'ani_delay_end' else self.ani_delay_end for j in deltas]
        
            animation_filename = self.animation_filename+f'.{ext}'
            if ext == 'gif':
                filenames_string = ' '.join(frames)+f' -o "{animation_filename}"'
                subprocess.run(f'gifsicle/gifsicle -d{self.ani_delay:.0f} --loop {filenames_string}')
                deltas_string = ' '.join([f'-d{j:.0f} "#{i}"' for i,j in enumerate(deltas)])
                subprocess.run(f'gifsicle/gifsicle -b "{animation_filename}" {deltas_string} --optimize')
            elif ext == 'mp4':         
                container = av.open(animation_filename, mode='w')
                shape = Image.open(frames[0]).size
                crf = 1+5*(10-self.ani_quality)
                stream = container.add_stream('libx264', width=shape[0], height=shape[1], pix_fmt='yuv420p', options={"crf":str(crf)})
        
                # Use a time resolution of ms, unless is needs to be higher to prevent possible errors.
                # These errors can occur when time_res/100*min(deltas) < 1, in which case the following error can occur:
                # "Application provided invalid, non monotonically increasing dts to muxer in stream".
                time_res = max(1000, int(np.ceil(100/min(deltas))))
                # delta-values correspond to a time resolution of cs, hence division by 100
                csum = time_res/100*np.cumsum(np.concatenate(([0], deltas)))
                stream.codec_context.time_base = Fraction(1, time_res)
                # ffmpeg time is "complicated". read more at https://github.com/PyAV-Org/PyAV/blob/main/docs/api/time.rst
                
                for i,frame in enumerate(frames+[frames[-1]]):
                    img = Image.open(frame)
                    frame = av.VideoFrame.from_image(img)
                    frame.pts = csum[i]
                    for packet in stream.encode(frame):
                        container.mux(packet)
                
                for packet in stream.encode(): # Flush stream
                    container.mux(packet)
                container.close()
                
            self.set_textbar('Animation created', 'green', 1)
        except Exception as e:
            try:
                for i in range(len(deltas)):
                    print(i, deltas[i], csum[i])
            except Exception:
                pass
            self.set_textbar(str(e), 'red', 3)
    
    def change_savefig_include_menubar(self, state):
        self.savefig_include_menubar = state == 2

    def show_volume_3d_viewer(self):
        """Sneltoets CTRL+SHIFT+4: opent een apart, interactief vispy-venster met een echte GPU volume-
        rendering (TurntableCamera: slepen=draaien/kantelen, scrollen=in-/uitzoomen) van het product in het
        huidige paneel, gereconstrueerd met get_volume_grid over een zelf met CTRL+SHIFT+slepen getekende
        rechthoek. Onthoudt het gebruikte product/gebied (self._volume3d_rect_params) zodat instellingen-
        wijzigingen (Settings -> Miscellaneous) het venster live kunnen verversen zonder opnieuw op
        CTRL+SHIFT+4 te moeten drukken -- zie _live_refresh_volume_3d en _render_volume_3d_scene.
        """
        if self.pb.volume3d_rect_a is None or self.pb.volume3d_rect_b is None:
            self.set_textbar('Teken eerst een gebied: CTRL+SHIFT+links-klik-en-slepen op de kaart.', 'red', 2)
            return

        j = self.pb.panel
        # BUGFIX (15 augustus 2026): gv.i_p zet elk afgeleid product terug naar zijn onderliggende scan-
        # product (nodig omdat scanangles_all alleen voor de echte, direct ingelezen producten gevuld is --
        # zie get_volume_grid). Voor PolRGB ('g') betekende dit dat de 3D-viewer altijd stilletjes 'z' liet
        # zien, ongeacht welk product er in het paneel stond -- get_volume_grid_polrgb heeft deze omweg
        # niet nodig (haalt zelf Z/CC/ZDR op), dus 'g' blijft hier voortaan gewoon 'g'. Andere afgeleide
        # producten (HCLASS/MESH/etc.) vallen nog wel terug op 'z' zoals voorheen -- dat is een apart,
        # groter punt (die hebben geen eigen get_volume_grid_X-tegenhanger) en dus bewust ongewijzigd
        # gelaten, niet aangepakt binnen deze PolRGB-wijziging.
        product = self.crd.products[j]
        if product != 'g':
            product = gv.i_p[product]
        # Scan-datum/tijd vastgelegd op DIT moment (7 juli 2026, op Eriks verzoek: "datum en tijd van de scan
        # mis ik in de 3D-weergave") -- net als product/x_range/y_range hierboven, zodat een latere live-
        # refresh (bv. door een Settings-wijziging) niet per ongeluk het datum/tijd van een ANDER, inmiddels
        # geselecteerd paneel/moment gaat tonen.
        scandatetime = self.pb.data_attr['scandatetime'].get(j)
        x1, y1 = self.pb.volume3d_rect_a
        x2, y2 = self.pb.volume3d_rect_b
        # De met CTRL+SHIFT+slepen getekende rechthoek (zie nlr_plotting.py: volume3d_rect_a/b) als exact
        # het gewenste gebied, met een kleine marge zodat de rand niet precies op de aangeklikte hoek valt.
        margin_km = 1.
        x_range = (min(x1, x2)-margin_km, max(x1, x2)+margin_km)
        y_range = (min(y1, y2)-margin_km, max(y1, y2)+margin_km)
        self._volume3d_rect_params = (product, x_range, y_range, scandatetime)
        # force_new_window=True: een expliciete CTRL+SHIFT+4-druk maakt altijd een vers venster met de
        # HUIDIGE rechthoek (die kan intussen veranderd zijn) -- alleen instellingen-wijzigingen (zie
        # _live_refresh_volume_3d) hertekenen het BESTAANDE venster in place.
        self._render_volume_3d_scene(force_new_window=True)

    def _live_refresh_volume_3d(self):
        """Wordt aangeroepen door elke change_volume3d_*-instellingsfunctie (Settings -> Miscellaneous): als
        er al een 3D-venster open staat, wordt dat venster in place opnieuw getekend met de nieuwe
        instelling(en) en hetzelfde product/gebied als de vorige keer, met behoud van de camerastand -- op
        Eriks verzoek (5 juli 2026: 'kan dat niet live'), net zoals show_cross_section al live doet bij
        wijzigingen aan de cross-section-instellingen. Doet niets als er nog geen venster open staat (dan
        gelden de nieuwe instellingen simpelweg de volgende keer dat CTRL+SHIFT+4 wordt gebruikt).

        Gebruikt een korte debounce (zie _volume3d_refresh_timer): de daadwerkelijke hertekening (relatief
        traag, met name door de tientallen tekst-labels die vispy telkens opnieuw moet aanmaken) wordt pas
        400ms na de LAATSTE wijziging uitgevoerd, zodat het snel na elkaar aanpassen van meerdere velden
        (Settings -> Miscellaneous) niet elk apart een volledige, dure heropbouw triggert -- op Eriks
        feedback (5 juli 2026: 'het gaat uiterst traag') dat de live-refresh zonder dit merkbaar hapert."""
        if getattr(self, '_volume_3d_canvas', None) is None or getattr(self, '_volume3d_rect_params', None) is None:
            return
        if not hasattr(self, '_volume3d_refresh_timer'):
            self._volume3d_refresh_timer = QTimer()
            self._volume3d_refresh_timer.setSingleShot(True)
            self._volume3d_refresh_timer.timeout.connect(self._live_refresh_volume_3d_now)
        self._volume3d_refresh_timer.start(400)

    def _live_refresh_volume_3d_now(self):
        """Doet de daadwerkelijke hertekening -- zie _live_refresh_volume_3d hierboven voor de debounce die
        hiernaartoe leidt."""
        try:
            self._render_volume_3d_scene(force_new_window=False)
        except Exception as e:
            print(f"[3D viewer] Live update mislukt: {e}"); print(traceback.format_exc())

    def _render_volume_3d_scene(self, force_new_window):
        """Doet het eigenlijke werk voor show_volume_3d_viewer/_live_refresh_volume_3d: herberekent het grid
        via get_volume_grid met de huidige instellingen en het onthouden product/gebied
        (self._volume3d_rect_params), en (her)bouwt daarmee de volume/kader/streepjes/labels -- ofwel in een
        gloednieuw venster (force_new_window=True, of als er nog geen venster open staat), ofwel BINNEN het
        al openstaande venster, met behoud van de camerastand (force_new_window=False).

        DIT IS HET MINST GEVALIDEERDE ONDERDEEL VAN HET HELE PROJECT: er is hier geen GPU/vispy beschikbaar
        om dit te testen (zie eerdere gesprek met Claude, juli 2026), dus dit is gebaseerd op vispy's
        documentatie en het al bewezen werkende reconstructie-principe.
        """
        # AXIS_LABEL_FONT_SIZE (23 juli, op Eriks verzoek): grootte van ALLE tekst die rechtstreeks in de
        # 3D-scene zelf hangt (aslabels/cijfers, "Noord 40 km"-randlabels, hoogte-liniaal-cijfers,
        # positiemarker-labels) - was overal los op 2400 gezet (5 plekken verderop), nu 1 instelbare
        # constante zodat verder finetunen niet weer 5 plekken hoeft. Deze tekst hangt in de 3D-
        # wereldcoordinaten van de scene zelf (parent=view.scene), niet in schermpixels zoals de meeste
        # andere tekst in NLradar - vandaar de heel andere orde van grootte t.o.v. een normale font_size
        # van bv. 10-14. LET OP: net als de rest van deze functie kon dit niet visueel getest worden (zie
        # docstring hierboven) - de exacte, prettige waarde hangt af van de camera-afstand/het kader-formaat
        # dat Erik gebruikt, dus dit is een eerste redelijke schatting (gehalveerd t.o.v. de oude 2400),
        # niet een gegarandeerd perfecte waarde.
        AXIS_LABEL_FONT_SIZE = 1200

        product, x_range, y_range, scandatetime = self._volume3d_rect_params
        # PolRGB (product 'g') heeft geen kleurentabel (RGBA-passthrough, R=Z/G=CC/B=ZDR -- zie
        # DataSource_General._calculate_polrgb) en kan daarom niet als scalair rooster+colormap worden
        # gereconstrueerd zoals elk ander product. get_volume_grid_polrgb haalt Z/CC/ZDR apart op en
        # combineert ze per voxel tot echte RGBA-kleuren -- zie hieronder (is_polrgb) voor hoe dat verder
        # door deze functie heen verwerkt wordt.
        is_polrgb = product == 'g'

        self.set_textbar(f"3D-grid reconstructie voor product '{product}' wordt berekend...", 'orange', 1)
        try:
            if is_polrgb:
                result = self.dsg.get_volume_grid_polrgb(
                    x_range, y_range, grid_res_km=self.volume3d_grid_res_km, z_max_km=self.volume3d_z_max_km,
                    z_res_km=self.volume3d_z_res_km, smoothing_sigma_cells=self.volume3d_smoothing_sigma)
            else:
                result = self.dsg.get_volume_grid(product, x_range, y_range, grid_res_km=self.volume3d_grid_res_km,
                                                   z_max_km=self.volume3d_z_max_km, z_res_km=self.volume3d_z_res_km,
                                                   smoothing_sigma_cells=self.volume3d_smoothing_sigma)
        except Exception as e:
            self.set_textbar(f"get_volume_grid gaf een fout: {e}", 'red', 3)
            print('_render_volume_3d_scene error:'); print(traceback.format_exc())
            return
        if result is None:
            self.set_textbar(f"Geen (niet-birdbath) scans beschikbaar voor product '{product}'.", 'red', 2)
            return

        try:
            from vispy import scene
            from vispy import color
        except ImportError as e:
            self.set_textbar(f"vispy.scene niet beschikbaar: {e}", 'red', 3)
            return

        if is_polrgb:
            rgba_full, x_axis, y_axis, z_axis = result['rgba'], result['x'], result['y'], result['z']
            grid = None
        else:
            grid, x_axis, y_axis, z_axis = result['grid'], result['x'], result['y'], result['z']
        # TERUGGEZET 7 juli 2026: de wijzigingen hierboven (data_values_colors i.p.v. cmaps_maxrange_masked
        # gebruiken voor vmin/vmax) raakten NIET alleen de nieuwe legenda, maar ook de daadwerkelijke
        # kleuren van de 3D-stormvorm zelf -- die werkte al goed en had hier niet aangeraakt moeten worden.
        # vmin/vmax voor de VOLUMEDATA ZELF dus weer terug naar de oorspronkelijke, bevestigd werkende bron
        # van 5 juli 2026. De legenda hieronder gebruikt voortaan zijn EIGEN, apart benoemde variabelen
        # (_legend_vmin/_legend_vmax), die niets meer met de volumedata's eigen clim te maken hebben -- zodat
        # verder puzzelen aan de legenda de 3D-vorm zelf nooit meer kan beinvloeden.
        # gv.cmaps_maxrange_masked['g'] is slechts een RGBA-uint8-placeholder (0-255), niet een echt
        # dBZ/%/dB-bereik -- voor PolRGB is er sowieso geen enkele scalaire clim zinvol (zie is_polrgb
        # hieronder), dus dan gewoon niet opzoeken.
        vmin, vmax = (0., 1.) if is_polrgb else gv.cmaps_maxrange_masked[product]



        # OMSLAG 5 juli 2026, 2e keer: Erik liet een voorbeeld van Brams eigen 3D-weergave zien (een zachte,
        # gloeiende, kleurrijke wolk waar je gaten/structuur doorheen kunt zien, bijv. een duidelijk
        # "oog"/gat in het midden). Dat is het klassieke uiterlijk van MIP (Maximum Intensity Projection)
        # volume-rendering: voor elke kijkstraal wordt alleen de HOOGSTE waarde langs die straal getoond, in
        # plaats van doorzichtigheid op te stapelen (dat laatste, method='translucent', was de vorige
        # poging, en blokkeerde alles ondanks lagere alpha -- zie Eriks feedback "ALLE data ontneemt het
        # zicht"). Met MIP kun je vanzelf "erdoorheen kijken": data blokkeert nooit iets anders, er wordt
        # alleen de sterkste waarde per straal getoond. De eerder geprobeerde puntenwolk (Markers) wordt
        # hiermee vervangen -- dit sluit dichter aan bij hoe Bram het doet.
        vertical_exaggeration = self.volume3d_vertical_exaggeration # Instelbaar via Settings -> Miscellaneous.

        # OMSLAG 5 juli 2026, 3e keer: Erik meldde een felle, gekleurde rand rondom de HELE vorm ("loopt er
        # helemaal omheen"), bevestigd als renderartefact (geen echte data). Vermoedelijke oorzaak: de
        # eerdere sentinelwaarde-truc (een kunstmatige "net iets te lage" waarde + een aangepaste colormap
        # met een doorzichtige zone) creeerde een harde kleurovergang die de textuur-interpolatie van vispy
        # bij elke rand van de vorm liet oplichten. Nieuwe, eenvoudigere aanpak: ECHTE NaN-waarden in de
        # data, met de GEWONE kleurentabel (self.pb.cm1[product]) direct, zonder aanpassingen. Bij MIP zou
        # dit vanzelf moeten werken: een vergelijking met NaN is altijd onwaar, dus een NaN-cel kan nooit als
        # "hoogste waarde langs de straal" gekozen worden -- in tegenstelling tot method='translucent'
        # (eerder geprobeerd, toen bleek dat NaN daar NIET als doorzichtig werd behandeld), kan MIP hier
        # heel anders mee omgaan. Nog niet eerder getest sinds we destijds meteen naar de sentinel-truc
        # overstapten; als dit een verkeerd-gekleurd blok oplevert i.p.v. de rand op te lossen, weten we dat
        # NaN ook bij MIP niet vanzelf wordt overgeslagen en moet de sentinel-truc met een andere overgang
        # terugkomen.
        # self.pb.cm1['g'] is (net als de colortable-CSV hierboven) slechts een placeholder -- PolRGB
        # gebruikt sowieso geen scalaire kleurentabel (zie is_polrgb). volume_data wordt voor PolRGB niet
        # gebruikt voor de daadwerkelijke kleuren (dat gebeurt rechtstreeks via rgba_full, zie de
        # puntenwolk-opbouw verderop), alleen als kleurloze plaatshouder voor de (bij PolRGB altijd
        # verborgen) Volume/MIP-visual -- zie hieronder waarom een echte MIP/translucent PolRGB-weergave
        # niet mogelijk is.
        if is_polrgb:
            volume_cmap = 'grays'
            volume_data = rgba_full[..., 0].astype('float32') # Alleen Z-kanaal (genormaliseerd), puur als
            #ongebruikte plaatshouder-textuur -- zie hierboven.
        else:
            volume_cmap = self.pb.cm1[product] # Dezelfde kleurentabel als de normale 2D-panelen, ongewijzigd.
            volume_data = grid.astype('float32') # Blijft NaN waar geen data is -- geen sentinelwaarde meer.
        if not is_polrgb and self.volume3d_min_value is not None:
            # Minimumwaarde-filter (8 juli 2026, op Eriks verzoek: "de 3D weergave heeft duidelijk last van
            # de lage dBZ waarden") -- waarden onder de grens worden hier al op NaN gezet, VOORDAT de Volume-
            # en puntenwolk-visuals worden opgebouwd. Daardoor werkt dit automatisch voor alle drie de
            # weergavemodi (MIP, translucent, puntenwolk) tegelijk, met dezelfde "NaN kan nooit de hoogste
            # waarde langs een straal zijn"-redenering als hierboven al gebruikt wordt voor de buitenrand.
            #
            # BIJGESTELD (8 juli 2026, Erik: "bij V ga je van -60 naar +60, dus moet je vanaf 0 filteren, naar
            # boven EN naar beneden"): voor een symmetrisch/tweezijdig product zoals V (waar negatief=naar de
            # radar toe en positief=van de radar af BEIDE fysiek betekenisvol zijn) is "alles onder de grens
            # weg" verkeerd -- dat zou alle inkomende (negatieve) waarden wegfilteren! In plaats daarvan moet
            # daar gefilterd worden op de ABSOLUTE waarde (dicht bij nul = zwak, ver van nul in beide
            # richtingen = sterk). Voor een eenzijdig product zoals Z (-20 tot 80, overwegend positief) blijft
            # de simpele "onder de grens" aanpak wel correct. Onderscheid gemaakt via een simpele heuristiek
            # op het kleurbereik zelf (vmin/vmax), i.p.v. hardgecodeerde productletters: als het bereik
            # ongeveer symmetrisch rond nul ligt, behandel het als tweezijdig.
            if vmin < 0 < vmax and abs(vmin+vmax) < 0.5*(vmax-vmin):
                volume_data[np.abs(volume_data) < self.volume3d_min_value] = np.nan
            else:
                volume_data[volume_data < self.volume3d_min_value] = np.nan
        elif is_polrgb and self.volume3d_min_value is not None:
            # PolRGB-equivalent van het minimumwaarde-filter hierboven: Z is bij PolRGB altijd eenzijdig
            # (net als bij een gewoon Z-paneel), dus geen symmetrie-heuristiek nodig. Werkt rechtstreeks op
            # de ruwe (ongevulde) Z-reconstructie -- niet op het al genormaliseerde R-kanaal, dat immers
            # gamma-gecorrigeerd is (Z_GAMMA) en dus geen lineaire dBZ-schaal meer heeft. Cellen onder de
            # grens worden onzichtbaar gemaakt (alpha=0) i.p.v. verwijderd, zodat de puntenwolk hieronder ze
            # vanzelf overslaat (zie de alpha>0-selectie verderop) net als bij "geen data".
            rgba_full[..., 3][result['z_raw'] < self.volume3d_min_value] = 0.

        if is_polrgb and self.volume3d_polrgb_cc_max is not None:
            # CC-zichtbaarheidsfilter (15 augustus 2026, op Eriks verzoek: "dat groen van de regen wil ik
            # kwijt" -- gewone regen heeft een hoge CC ONGEACHT Z, dus het Z-filter hierboven helpt daar niet
            # tegen). Volledig LOS van/onafhankelijk van volume3d_min_value: dit filtert op de ruwe CC-
            # reconstructie (result['cc_raw'], %), niet op Z. Voxels met CC BOVEN de grens worden onzichtbaar
            # (alpha=0) -- hagel heeft per definitie een lagere CC dan gewone regen, dus het hagelgebied blijft
            # hierdoor onaangeroerd. NaN in cc_raw (geen CC-data op die plek) telt hier niet mee als "boven de
            # grens" (een NaN-vergelijking is altijd onwaar), dus zulke voxels blijven ongemoeid door dit
            # filter -- alleen aantoonbaar hoge CC wordt weggefilterd.
            rgba_full[..., 3][result['cc_raw'] > self.volume3d_polrgb_cc_max] = 0.

        if self.volume3d_circular_area:
            # Cirkelvormig (eigenlijk: ellipsvormig) gebied (8 juli 2026, op Eriks verzoek: "zou dat ook een
            # cirkel kunnen zijn"): maskeert data BUITEN de ellips die precies in de getekende rechthoek past
            # (middelpunt = midden van x_range/y_range, halve-assen = halve breedte/hoogte) -- het kader/de
            # assen/tick-streepjes verderop blijven gewoon de VOLLE rechthoekige omvang tonen (x_axis/y_axis
            # zelf worden hier niet aangepast), alleen de gekleurde data zelf wordt rond weggesneden. Dezelfde
            # "NaN kan nooit de hoogste waarde langs een straal zijn"-redenering als bij het minimumwaarde-
            # filter hierboven zorgt dat dit automatisch voor alle drie de weergavemodi werkt.
            _cx, _cy = (x_range[0]+x_range[1])/2., (y_range[0]+y_range[1])/2.
            _a, _b = (x_range[1]-x_range[0])/2., (y_range[1]-y_range[0])/2.
            _xx, _yy = np.meshgrid(x_axis, y_axis) # Vorm (n_y, n_x), zelfde als een enkele z-laag van volume_data.
            _outside_ellipse = ((_xx-_cx)/_a)**2 + ((_yy-_cy)/_b)**2 > 1.
            if is_polrgb:
                rgba_full[:, _outside_ellipse, 3] = 0.
            else:
                volume_data[:, _outside_ellipse] = np.nan

        # Geografische context berekenen (afstand + windrichting vanaf de radar) -- dit gaat straks in de
        # venstertitel/statusbalk i.p.v. in de 3D-scene zelf, want scene.visuals.Text bleek op Eriks systeem
        # niet te renderen (waarschijnlijk ontbrekende freetype-py; zie gesprek 5 juli 2026) en verder
        # lapwerk daaraan bracht geen vooruitgang. Titelbalk/statusbalk/console zijn Qt-tekst, dus die werken
        # sowieso, ongeacht wat er met vispy's eigen tekst-rendering aan de hand is.
        box_center_x = (x_axis[0]+x_axis[-1])/2.
        box_center_y = (y_axis[0]+y_axis[-1])/2.
        distance_from_radar_km = float(np.hypot(box_center_x, box_center_y))
        bearing_deg = (90.-np.degrees(np.arctan2(box_center_y, box_center_x))) % 360. # Zelfde conventie als
        #get_volume_grid/get_cross_section: positieve y=noord (azimuth 0), positieve x=oost (azimuth 90).
        compass_16 = ['N', 'NNO', 'NO', 'ONO', 'O', 'OZO', 'ZO', 'ZZO',
                      'Z', 'ZZW', 'ZW', 'WZW', 'W', 'WNW', 'NW', 'NNW']
        bearing_label = compass_16[int((bearing_deg+11.25)//22.5) % 16]
        width_km, height_km = x_axis[-1]-x_axis[0], y_axis[-1]-y_axis[0]

        # Datum/tijd van de scan leesbaar opmaken (7 juli 2026) -- defensief geschreven omdat niet met
        # zekerheid bekend is of scandatetime hier altijd een cijfer-string YYYYMMDDHHMM(SS) is; bij twijfel
        # gewoon de ruwe waarde tonen in plaats van een foutmelding te riskeren.
        try:
            _sdt = str(scandatetime)
            if len(_sdt) >= 12 and _sdt.isdigit():
                scandatetime_str = f"{_sdt[:4]}-{_sdt[4:6]}-{_sdt[6:8]} {_sdt[8:10]}:{_sdt[10:12]}"
                if len(_sdt) >= 14:
                    scandatetime_str += f":{_sdt[12:14]}"
            else:
                scandatetime_str = _sdt
        except Exception:
            scandatetime_str = str(scandatetime)

        # Ingekort (7 juli 2026): de toetsen-uitleg (T/M/R) stond eerst ook hier, maar deze Windows-titelbalk
        # heeft een beperkte breedte en knipte de tekst af zodra er meer sneltoetsen bijkwamen. Die uitleg
        # staat nu in plaats daarvan in het vaste titel-paneel BINNENIN het 3D-venster zelf (zie
        # title_view.scene hierboven bij canvas/view), waar geen afkap-risico bestaat.
        # Ingekort (8 juli 2026, op Eriks verzoek): de kompas-/liniaal-uitleg is verwijderd omdat die al
        # rechtstreeks in de 3D-scene zelf staat (elke as heeft al zijn eigen kleur+naam, en de hoogte-liniaal
        # heeft al een "hoogte X km"-label) -- dubbelop. [MIP] is hier het startpunt (elke render begint
        # altijd met method='mip', zie de Volume-aanmaak verderop), en wordt LIVE bijgewerkt door de M/P-
        # toetsen (canvas.title wordt daar direct aangepast, zie toggle_3d_render_method/toggle_3d_pointcloud).
        # PolRGB start meteen in [RGB-MIP] (de nieuwe additieve 3-kanaals MIP-weergave, zie hierboven) --
        # elk ander product start zoals voorheen in [MIP].
        _start_mode = 'RGB-MIP' if is_polrgb else 'MIP'
        window_title = (f"NLradar 3D - '{product}' {scandatetime_str} | kader {width_km:.0f}x{height_km:.0f} km, "
                         f"{distance_from_radar_km:.0f} km {bearing_label} van radar | "
                         f"{vertical_exaggeration:.0f}x overdreven | [{_start_mode}] | (H voor toetsen)")

        # reuse=True: het al openstaande venster wordt in place hertekend (Settings-wijziging) i.p.v. een
        # nieuw venster te openen -- op Eriks verzoek (5 juli 2026: 'kan dat niet live'). Camera/venster/
        # canvas blijven dan ongewijzigd; alleen de daadwerkelijke 3D-inhoud (volume/kader/streepjes/labels)
        # wordt vervangen.
        reuse = (not force_new_window) and getattr(self, '_volume_3d_canvas', None) is not None
        if reuse:
            canvas = self._volume_3d_canvas
            view = self._volume_3d_view
            legend_view = self._volume_3d_legend_view
            title_view = self._volume_3d_title_view
            try:
                canvas.title = window_title
            except Exception:
                pass
            # BUGFIX (10 juli 2026, gevonden tijdens testen van de nieuwe J-video-export): deze lus verwijderde
            # voorheen OOK view.camera zelf als kind van view.scene (vispy plaatst de TurntableCamera daar
            # intern bij 'view.camera = "turntable"'), waardoor de camera na een live-refresh een "wees" werd.
            # Dat viel niet eerder op omdat er tot nu toe niemand na een live-refresh nog programmatisch aan
            # camera.azimuth/elevation zat -- de nieuwe video-export (self._run_volume_3d_export) doet dat
            # wel, en liep daardoor op de 2e tijdstap vast met "RuntimeError: ... is not a child of ...".
            # Nu wordt de camera expliciet overgeslagen zodat 'ie gewoon in de scenegraph blijft zitten.
            for child in list(view.scene.children): # Oude volume/kader/streepjes/labels verwijderen voordat
                if child is not view.camera:          #de nieuwe versie erin komt (camera zelf blijft staan).
                    child.parent = None
            for child in list(legend_view.scene.children): # Zelfde voor de kleurschaal-legenda.
                if child is not legend_view.camera:
                    child.parent = None
            for child in list(title_view.scene.children): # Zelfde voor de titel.
                if child is not title_view.camera:
                    child.parent = None
        else:
            # keys='interactive' deliberately omitted (crashte eerder met een RuntimeError in vispy's eigen
            # _set_keys, zie 5 juli 2026) -- regelt alleen toetsenbord-shortcuts, niet de muisbediening.
            # config=dict(alpha_size=0): dwingt af dat de OpenGL-framebuffer GEEN alpha-kanaal heeft.
            # Vermoede echte oorzaak van "ik kijk bij lage dBZ nog steeds op mn bureaublad" (5 juli 2026):
            # donkere pixels (lage dBZ, dicht bij zwart) kregen kennelijk een laag alpha-kanaal mee in de
            # framebuffer, waardoor Windows' vensterbeheer ze als doorzichtig behandelde -- de eerdere
            # Qt-widget-attributen pakten dit niet aan omdat het probleem op het niveau van de GL-context
            # zelf zit, niet de Qt-widget eromheen.
            try:
                canvas = scene.SceneCanvas(size=(900, 700), show=False, bgcolor=(0, 0, 0, 1),
                                            title=window_title, config=dict(alpha_size=0))
            except Exception as e:
                print(f"[3D viewer] config=alpha_size=0 niet ondersteund ({e}), val terug zonder die instelling.")
                canvas = scene.SceneCanvas(size=(900, 700), show=False, bgcolor=(0, 0, 0, 1), title=window_title)
            try:
                canvas.native.setAttribute(Qt.WA_TranslucentBackground, False)
                canvas.native.setAutoFillBackground(True)
                canvas.native.setAttribute(Qt.WA_OpaquePaintEvent, True)
            except Exception as e:
                print(f"[3D viewer] Kon transparantie-attribuut niet zetten ({e}).")
            canvas.show()
            # Grid met 2 rijen (7 juli 2026, uitgebreid): bovenaan een SMAL, VAST 2D-paneel over de volle
            # breedte voor een witte titel (scan-datum/tijd) die -- net als de legenda -- altijd blijft staan,
            # ongeacht hoe je de 3D-weergave zelf draait/kantelt/zoomt. Daaronder, net als voorheen: links de
            # hoofd-3D-weergave, rechts een smal, vast 2D-paneel voor de kleurschaal-legenda.
            grid = canvas.central_widget.add_grid()
            title_view = grid.add_view(row=0, col=0, col_span=2)
            # Teruggezet naar 1 regel (8 juli 2026, op Eriks verzoek: "geen tekst meer in de 3D-weergave zelf,
            # behalve de titel met datum/tijd -- de rest in een apart schermpje"). De toetsen-uitleg staat nu
            # in een echt, los Qt-popupvenster (zie show_volume_3d_help(), opgeroepen met de H-toets), niet
            # meer als tekst binnen de vispy-scene zelf.
            title_view.height_max = 36
            title_view.camera = 'panzoom'
            title_view.camera.interactive = False
            title_view.camera.rect = (0, 0, 600, 36) # Willekeurige, vaste lokale eenheden -- PanZoomCamera
            #rekt dit vanzelf uit naar de werkelijke, actuele pixelbreedte van de rij, ongeacht vensterformaat.
            self._volume_3d_title_view = title_view

            view = grid.add_view(row=1, col=0)
            legend_view = grid.add_view(row=1, col=1)
            legend_view.width_max = 130 # Vaste, smalle breedte in pixels voor de legenda-kolom.
            legend_view.camera = 'panzoom'
            legend_view.camera.interactive = False # Voorkomt dat een muisklik in dit paneel per ongeluk
            #gaat slepen/zoomen; de legenda-inhoud wordt hieronder zelf op maat gepositioneerd.
            self._volume_3d_legend_view = legend_view
            # Set up via the string shorthand ('turntable') rather than instantiating
            # scene.cameras.TurntableCamera directly -- on the test on 5 July 2026, dragging with the
            # directly-instantiated camera moved/panned the whole shape across the screen instead of
            # orbiting around it. The string form is the officially documented/most-used way to attach a
            # camera in vispy's own examples, and going through ViewBox.camera's setter this way is more
            # likely to correctly register the camera's mouse-event handlers than assigning an
            # already-constructed instance.
            view.camera = 'turntable'
            view.camera.fov = 45
            # Start recht van bovenaf (net als de normale platte kaart) i.p.v. schuin -- bij een schuine
            # startpositie oogden noord/oost meteen "verkeerd"/gespiegeld t.o.v. de platte kaart waar Erik
            # aan gewend is (zie feedback 5 juli 2026), ook al klopt de onderliggende richtingsformule.
            # Vanuit deze bovenaanzicht kan hij zelf kantelen (slepen) om de hoogte-structuur te zien.
            view.camera.azimuth = 0
            # 65 i.p.v. bijna-verticale 85: dicht bij recht-van-bovenaf (nog steeds "bovenaanzicht"-gevoel),
            # maar niet zo dicht bij de 90 graden pool dat verticaal slepen (kantelen) nauwelijks effect
            # lijkt te hebben -- zie Eriks feedback (5 juli 2026: "verticaal doet ie niet"), vergelijkbaar
            # met hoe een kompas raar aanvoelt vlak bij de Noordpool.
            view.camera.elevation = 65
            # Sluiten van het venster (kruisje) moet de referentie opruimen, anders denkt een latere
            # live-refresh nog dat dit (gesloten) venster bruikbaar is.
            def _on_3d_canvas_close(ev):
                self._volume_3d_canvas = None
                self._volume3d_following = False  # Voorkomt dat de A-meeloop-keten doortikt tegen een gesloten canvas.
            canvas.events.close.connect(_on_3d_canvas_close)
            self._volume_3d_canvas = canvas
            self._volume_3d_view = view

        horizontal_extent = max(x_axis[-1]-x_axis[0], y_axis[-1]-y_axis[0])

        # method='mip' (zie toelichting hierboven). Kort geexperimenteerd met een instelbare
        # mip/additive-keuze (5 juli 2026), maar 'additive' bleek te crashen zonder Python-foutmelding
        # (waarschijnlijk GPU-driver-niveau) -- die optie is weer volledig verwijderd, 'mip' is nu weer
        # gewoon hardcoded zoals voorheen. interpolation='linear' als CONSTRUCTOR-argument gaf eerder een
        # TypeError op de oude vispy-versie (0.14.x, geen 'interpolation' kwarg op VolumeVisual). Vanaf
        # vispy 0.16.2 (upgrade juli 2026, samen met Python 3.8 -> 3.11) is 'interpolation' wel een geldig
        # constructor-argument -- zie vispy.visuals.volume.VolumeVisual signature. Nu ingeschakeld om de
        # rand-overshoot hieronder te verhelpen.
        volume = scene.visuals.Volume(volume_data, clim=(vmin, vmax), cmap=volume_cmap, gamma=self.volume3d_gamma,
                                       method='mip', interpolation='linear', parent=view.scene)
        # self._volume_3d_visual wordt bij ELKE (her)tekening vervangen door het nieuwste volume-object, om
        # dezelfde reden als bij self._volume_3d_text_visuals hierboven: de M-toets-koppeling (zie verderop,
        # translucent/MIP omschakelen) moet ook na een live-refresh het JUISTE, actuele volume-object pakken.
        self._volume_3d_visual = volume
        volume.visible = not is_polrgb # Zie de toelichting bij pointcloud.visible hierboven: bij PolRGB is
        #de Volume/MIP-visual altijd verborgen (kleurloze plaatshouder-textuur, geen echte data).
        # VOORHEEN BEKENDE BEPERKING (opgelost door vispy-upgrade naar 0.16.2, juli 2026):
        # rondom ELKE rand van de vorm was een dun, fel gekleurd randje te zien (met name opvallend bij V).
        # Grondig gediagnosticeerd en bevestigd: dit was een interpolatie-"overshoot" op harde randen tussen
        # data en leegte -- exact hetzelfde mechanisme als de gedocumenteerde bicubic-overshoot bij de
        # F2-cross-section (zie cross_section_interpolation_mode hierboven). GEEN data-fout, GEEN
        # kleurbereik-fout (destijds al expliciet uitgesloten via diagnostiek). Met interpolation='linear'
        # i.p.v. de vaste 'nearest'/bicubic-only rendering van de oude vispy-versie zou deze rand nu weg
        # moeten zijn -- test dit bij de volgende V-weergave in de 3D-viewer.
        #
        # De regel hieronder (`volume.interpolation = 'nearest'`) is de voor de hand liggende fix (net als
        # bij de cross-section), maar bleek GEEN effect te hebben op de 3D volume-rendering: Erik zag exact
        # dezelfde rand met en zonder deze regel. Conclusie: scene.visuals.Volume ondersteunt het wijzigen
        # van de textuur-interpolatie op deze vispy-versie niet echt (de instelling wordt geaccepteerd
        # zonder foutmelding, maar heeft geen effect op de daadwerkelijke rendering) -- dit is dus een
        # beperking van vispy's Volume-visual hier, niet van onze data of aanpak.
        #
        # BESLISSING (Erik, 5 juli 2026): MIP-rendering behouden, deze rand accepteren als bekende
        # beperking. Enige alternatief (een puntenwolk i.p.v. Volume/MIP) heeft dit probleem niet, maar
        # oogt minder zacht/gloeiend -- bewust niet gekozen. Niet opnieuw aan gaan sleutelen zonder nieuwe
        # concrete aanwijzingen.
        try:
            volume.interpolation = 'nearest' # Zonder effect (zie hierboven), maar onschadelijk om te laten staan.
        except Exception:
            pass
        grid_res_km = x_axis[1]-x_axis[0] if len(x_axis) > 1 else 1.
        z_res_km = z_axis[1]-z_axis[0] if len(z_axis) > 1 else 1.
        volume.transform = scene.STTransform(
            translate=(x_axis[0], y_axis[0], z_axis[0]),
            scale=(grid_res_km, grid_res_km, z_res_km*vertical_exaggeration))

        # PolRGB "MIP-achtige" weergave (15 augustus 2026, op Eriks verzoek na de puntenwolk-versie: "gruwelijk
        # jammer dat we geen wolk hebben"): 3 LOSSE Volume-visuals (1 per kanaal: R=Z, G=CC, B=ZDR), elk met
        # een ZUIVERE zwart->kanaalkleur-colormap en method='mip', over elkaar heen getekend met ADDITIEVE
        # GL-blending (set_gl_state('additive', depth_test=False) i.p.v. de standaard 'translucent' die de
        # gewone Volume-visual hierboven gebruikt) -- een bekende vispy-techniek voor multikanaals volume-
        # rendering (vergelijkbaar met hoe microscopie-composietbeelden vaak worden opgebouwd). Dit is NIET
        # hetzelfde als de eerder geprobeerde en verwijderde method='additive' RAY-ACCUMULATIEMODUS (die
        # crashte, zie de toelichting hierboven bij "method='mip'") -- dit is de GL-BLENDMODUS tussen 3 losse,
        # elk voor zich heel gewoon MIP-renderende visuals, een andere laag van vispy's rendering-pijplijn.
        #
        # BELANGRIJKE, BEWUSTE BEPERKING (net als bij de eerdere additieve 3-MIP-optie besproken): elk kanaal
        # kiest voor elke kijkstraal ONAFHANKELIJK zijn eigen hoogste waarde -- de plek waar Z het hoogst is,
        # hoeft niet dezelfde te zijn als waar CC of ZDR het hoogst is. De resulterende kleur per pixel is dus
        # een BENADERING, geen geometrisch exact samengestelde PolRGB-kleur (die garantie geeft alleen de
        # puntenwolk hierboven, nog steeds beschikbaar via de P-toets). Bij scherpe overgangen (bijv. de rand
        # van een hagelkern) kan dit er daardoor net "verkeerd gemengd" uit laten zien -- geen bug, inherent
        # aan deze benadering.
        #
        # NOG NIET GETEST (zelfde reden als de rest van deze functie, zie docstring bovenaan): additieve
        # GL-blending over 3 overlappende Volume-visuals is in vispy's eigen voorbeelden een bekend werkende
        # aanpak, maar niet eerder in NLradar zelf gebruikt/gezien renderen.
        self._volume_3d_polrgb_visuals = []
        if is_polrgb:
            # FIX (16 september 2026, na Eriks melding "veel blauw en wit, herken het plaatje niet t.o.v.
            # 2D"): tot nu toe werd rgba_full[...,3] (alpha/zichtbaarheid) ALLEEN gebruikt om een voxel
            # volledig uit te sluiten (alpha<=0 -> NaN) -- de GELEIDELIJKE vervaging die alpha in 2D en in
            # de puntenwolk hieronder wel geeft (of dat nu de oude 2-punts ALPHA_GAMMA-fade is, of de
            # nieuwe ESSL 11-punts-curve) had in de RGB-MIP GEEN effect: een voxel met alpha=0.1 rendert
            # dan met exact dezelfde volle R/G/B-helderheid als alpha=1.0, zodra hij niet volledig is
            # uitgesloten. Gecombineerd met de al bekende, bewuste "elk kanaal kiest onafhankelijk zijn
            # eigen hoogste punt langs de kijkstraal"-beperking (zie hierboven) stapelt dit zwakke,
            # verspreide signalen in alle 3 kanalen overal even zwaar op tot lichte/witte tinten die in de
            # gefilterde 2D-weergave nooit zo zouden verschijnen. Fix: elk kanaal vooraf vermenigvuldigen
            # met zijn eigen alpha, zodat een zwakke-echo-voxel navenant minder bijdraagt aan de MIP i.p.v.
            # alles-of-niets -- consistent met hoe alpha al werkt in 2D en in de puntenwolk (face_color).
            _no_data = rgba_full[..., 3] <= 0. # Zelfde "geen data/uitgefilterd"-criterium (alpha<=0) als de
            #puntenwolk-selectie hieronder (alpha>0) -- consistente resultaten tussen beide weergavemodi.
            _alpha_weight = rgba_full[..., 3]
            _r_nan = np.where(_no_data, np.nan, rgba_full[..., 0]*_alpha_weight).astype('float32')
            _g_nan = np.where(_no_data, np.nan, rgba_full[..., 1]*_alpha_weight).astype('float32')
            _b_nan = np.where(_no_data, np.nan, rgba_full[..., 2]*_alpha_weight).astype('float32')
            _channel_cmaps = [color.Colormap([(0, 0, 0, 1), (1, 0, 0, 1)]),
                               color.Colormap([(0, 0, 0, 1), (0, 1, 0, 1)]),
                               color.Colormap([(0, 0, 0, 1), (0, 0, 1, 1)])]
            for _chan_data, _chan_cmap in zip((_r_nan, _g_nan, _b_nan), _channel_cmaps):
                _chan_volume = scene.visuals.Volume(_chan_data, clim=(0., 1.), cmap=_chan_cmap, method='mip',
                                                      interpolation='linear', parent=view.scene)
                # additive i.p.v. de standaard translucent GL-blendmodus: laat de 3 kanalen optellen i.p.v.
                # elkaar (gedeeltelijk) af te dekken. depth_test=False om dezelfde reden als bij de kader-
                # kubus hierboven (anders kan het ene kanaal het andere op dieptevolgorde wegdrukken i.p.v.
                # ermee op te tellen).
                _chan_volume.set_gl_state('additive', cull_face=False, depth_test=False)
                _chan_volume.transform = scene.STTransform(
                    translate=(x_axis[0], y_axis[0], z_axis[0]),
                    scale=(grid_res_km, grid_res_km, z_res_km*vertical_exaggeration))
                self._volume_3d_polrgb_visuals.append(_chan_volume)

        # Puntenwolk-weergave (8 juli 2026, op Eriks verzoek: "een dichte pixelweergave waar ik doorheen kan
        # kijken", i.p.v. MIP/translucent die last hebben van opstapelende (on)doorzichtigheid bij een dikke
        # cel). In plaats van een aaneengesloten oppervlak/textuur (Volume) worden hier alleen LOSSE punten
        # getekend, met echt LEGE ruimte ertussen -- dat is precies wat translucent niet kan bieden. Dit is
        # bewust een COMPLEET APARTE visual (scene.visuals.Markers), niet een aanpassing van de bestaande,
        # goedwerkende Volume/MIP/translucent-opzet hierboven, om geen enkel risico te lopen voor wat al
        # goed werkt. Zie ook de eerdere notitie hierboven (Erik, 5 juli 2026) die dit destijds om een ANDERE
        # reden (esthetiek, niet doorkijkbaarheid) bewust niet koos.
        _pc_stride = max(1, int(self.volume3d_pointcloud_stride))
        if is_polrgb:
            # PolRGB heeft ECHTE per-voxel RGBA-kleuren (rgba_full), geen scalaire waarde+colormap -- dus
            # geen volume_cmap.map(...)-stap zoals hieronder voor een gewoon product: de kleuren liggen al
            # klaar en gaan rechtstreeks in face_color. Selectie op alpha>0 i.p.v. np.isfinite (rgba_full
            # bevat geen NaN's, "geen data" is hier alpha=0 -- zie get_volume_grid_polrgb/hierboven).
            _pc_sub_rgba = rgba_full[::_pc_stride, ::_pc_stride, ::_pc_stride, :]
            _pc_iz, _pc_iy, _pc_ix = np.where(_pc_sub_rgba[..., 3] > 0.)
        else:
            _pc_sub = volume_data[::_pc_stride, ::_pc_stride, ::_pc_stride]
            _pc_iz, _pc_iy, _pc_ix = np.where(np.isfinite(_pc_sub))
        if len(_pc_iz) > 0:
            # Terug naar volledige-resolutie-indices, om exact dezelfde wereld-coordinaten te berekenen als
            # de transform van de Volume-visual hierboven gebruikt (zodat de puntenwolk perfect uitlijnt met
            # hetzelfde kader/dezelfde tick-streepjes).
            _pc_x = x_axis[0] + (_pc_ix*_pc_stride)*grid_res_km
            _pc_y = y_axis[0] + (_pc_iy*_pc_stride)*grid_res_km
            _pc_z = z_axis[0] + (_pc_iz*_pc_stride)*z_res_km*vertical_exaggeration
            if is_polrgb:
                _pc_colors = _pc_sub_rgba[_pc_iz, _pc_iy, _pc_ix, :]
            else:
                _pc_values = _pc_sub[_pc_iz, _pc_iy, _pc_ix]
                _pc_t = np.clip((_pc_values-vmin)/(vmax-vmin), 0., 1.)**self.volume3d_gamma
                _pc_colors = volume_cmap.map(_pc_t)
            pointcloud = scene.visuals.Markers(parent=view.scene)
            pointcloud.set_data(pos=np.column_stack([_pc_x, _pc_y, _pc_z]), face_color=_pc_colors,
                                 size=self.volume3d_pointcloud_point_size, edge_width=0)
        else:
            pointcloud = scene.visuals.Markers(parent=view.scene) # Leeg, voor het geval er (na subsampling) geen enkel geldig punt overblijft.
        # Standaard verborgen (alleen zichtbaar in de 'pointcloud'-weergavemodus, P-toets) -- ook bij PolRGB:
        # de nieuwe additieve RGB-MIP-weergave hierboven is daar voortaan de standaard (P-toets wisselt ernaar
        # terug naar de puntenwolk, met echte per-voxel-kleuren, zie toggle_3d_pointcloud verderop).
        pointcloud.visible = False
        self._volume_3d_pointcloud_visual = pointcloud

        # The scene-coordinate bounding box (voor het draadframe-kader en het kompas hieronder) -- z is
        # exaggerated (zie vertical_exaggeration), x/y niet.
        z_scene_min = z_axis[0]
        z_scene_max = z_axis[0]+(z_axis[-1]-z_axis[0])*vertical_exaggeration
        box_corners_x = (x_axis[0], x_axis[-1])
        box_corners_y = (y_axis[0], y_axis[-1])
        box_corners_z = (z_scene_min, z_scene_max)

        # Wireframe reference box around the reconstructed area (all 12 edges), so it's clear where the
        # visible storm shape sits within/relative to the selected area -- on 5 July 2026 there was no
        # spatial reference at all, making it impossible to tell scale, orientation, or whether the whole
        # selected area or only part of it was being shown.
        edges = []
        for (x0, x1) in [box_corners_x]:
            for y in box_corners_y:
                for z in box_corners_z:
                    edges.append([(x0, y, z), (x1, y, z)])
        for (y0, y1) in [box_corners_y]:
            for x in box_corners_x:
                for z in box_corners_z:
                    edges.append([(x, y0, z), (x, y1, z)])
        for (z0, z1) in [box_corners_z]:
            for x in box_corners_x:
                for y in box_corners_y:
                    edges.append([(x, y, z0), (x, y, z1)])
        edge_points = np.array(edges).reshape(-1, 3)
        # Lijst om alle kader-/tick-lijn-visuals te verzamelen (8 juli 2026, op Eriks verzoek: "de hele kubus-
        # omlijning met tics en al aan/uit te zetten") -- apart van self._volume_3d_text_visuals (die alleen
        # de TEKST-labels bevat), zodat de B-toets (zie verderop) alle lijn-elementen samen kan tonen/verbergen.
        self._volume_3d_wireframe_visuals = []
        # TEST (7 juli 2026): dekking verhoogd van 0.3 naar 0.8 -- de kader-kubus (volledige wireframe) was
        # nauwelijks nog zichtbaar t.o.v. het origineel (zie screenshotvergelijking), vermoedelijk omdat deze
        # nieuwere vispy-versie dunne, halfdoorzichtige lijnen anders combineert met de ondoorzichtige
        # 3D-volumedata erachter. Test of een hogere dekking de kubus terugbrengt.
        _box_line = scene.visuals.Line(pos=edge_points, connect='segments', color=(1, 1, 1, 0.8), width=1,
                            parent=view.scene)
        # FIX (7 juli 2026): dieptetest uitgeschakeld voor deze lijn. Zonder dit werd de kader-kubus door de
        # ondoorzichtige 3D-volumedata verborgen zodra een deel van een rand "achter" de data lag vanuit het
        # huidige camerastandpunt -- in het origineel (oudere vispy-versie) bleef de kubus altijd overal
        # zichtbaar, dwars door de gekleurde vorm heen. depth_test=False herstelt dat gedrag.
        _box_line.set_gl_state(depth_test=False)
        self._volume_3d_wireframe_visuals.append(_box_line)

        # Assenkader ALS KUBUS OM DE BUI HEEN (i.p.v. kruisende lijnen door het midden, zie feedback 5 juli
        # 2026: "kan dat assenstelsel niet als kubus om de bui heen"): streepjes elke 10 km op de 4
        # bodemranden van het kader zelf (rood=Noord-rand, groen=Oost-rand, blauw=Zuid-rand, geel=West-rand,
        # zelfde kleurbetekenis als voorheen, nu alleen verplaatst naar de rand i.p.v. het midden), plus een
        # apart hoogte-liniaal (wit, streepjes elke 2 km WERKELIJKE hoogte) op de verticale rand bij de
        # zuidwesthoek van het kader.
        tick_interval_km = self.volume3d_tick_interval_km # Instelbaar via Settings -> Miscellaneous.
        tick_len = 0.02*horizontal_extent
        text_visuals = [] # Alle tekst-objecten hierin verzamelen, zodat ze straks met 1 toets (T) samen
        #aan/uit gezet kunnen worden -- zie de key_press-koppeling verderop.
        edge_specs = [ # (naam, punt1, punt2, kleur) -- de 4 bodemranden van het kader.
            ('Zuid', (box_corners_x[0], box_corners_y[0]), (box_corners_x[1], box_corners_y[0]), (0, 0.4, 1)),
            ('Noord', (box_corners_x[0], box_corners_y[1]), (box_corners_x[1], box_corners_y[1]), (1, 0, 0)),
            ('West', (box_corners_x[0], box_corners_y[0]), (box_corners_x[0], box_corners_y[1]), (1, 1, 0)),
            ('Oost', (box_corners_x[1], box_corners_y[0]), (box_corners_x[1], box_corners_y[1]), (0, 1, 0)),
        ]
        for name, (x0, y0), (x1, y1), rgb in edge_specs:
            edge_len_km = float(np.hypot(x1-x0, y1-y0))
            unit = np.array([x1-x0, y1-y0])/edge_len_km if edge_len_km > 0 else np.array([0., 0.])
            perp = np.array([-unit[1], unit[0]])
            n_ticks = int(edge_len_km//tick_interval_km)
            tick_segments = []
            for k in range(n_ticks+1):
                pt = np.array([x0, y0])+unit*min(k*tick_interval_km, edge_len_km)
                tick_segments.append([(*(pt-perp*tick_len/2), z_scene_min), (*(pt+perp*tick_len/2), z_scene_min)])
                if k > 0 and k % 2 == 0: # Alleen bij elk 2e streepje een cijfer (i.p.v. elk streepje) --
                    #minder drukte, en k=0 is toch de hoek zelf (0 km).
                    tick_label_pos = pt+perp*tick_len*1.5
                    text_visuals.append(scene.visuals.Text(f"{int(k*tick_interval_km)}",
                                         pos=(*tick_label_pos, z_scene_min), color=rgb, font_size=AXIS_LABEL_FONT_SIZE,
                                         parent=view.scene))
            _tick_line = scene.visuals.Line(pos=np.array(tick_segments).reshape(-1, 3), connect='segments',
                                color=rgb, width=2, parent=view.scene)
            _tick_line.set_gl_state(depth_test=False) # Zelfde reden als bij de kader-kubus hierboven.
            self._volume_3d_wireframe_visuals.append(_tick_line)
            # Label in het MIDDEN van de rand (i.p.v. aan het eind) -- bij het eind kwamen "Noord" en "Oost"
            # toevallig in dezelfde hoek samen (ze delen de NO-hoek van het kader), waardoor de tekst over
            # elkaar heen viel (zie screenshot 5 juli 2026). Vanuit het midden van elke rand, met een klein
            # stukje naar buiten (weg van het midden van het kader), overlapt niets meer.
            edge_mid = np.array([x0, y0])+unit*edge_len_km/2.
            outward = edge_mid-np.array([box_center_x, box_center_y])
            outward_len = np.linalg.norm(outward)
            outward_unit = outward/outward_len if outward_len > 0 else perp
            label_pos = edge_mid+outward_unit*0.05*horizontal_extent
            text_visuals.append(scene.visuals.Text(f"{name} {edge_len_km:.0f} km", pos=(*label_pos, z_scene_min),
                                 color=rgb, font_size=AXIS_LABEL_FONT_SIZE, bold=True, parent=view.scene))

        # Hoogte-liniaal: streepjes elke 2 km WERKELIJKE hoogte (dus vertaald door de vertical_exaggeration
        # heen naar scherm-coordinaten), op de verticale rand bij de zuidwesthoek (x0, y0) van het kader.
        height_tick_interval_km = self.volume3d_height_tick_interval_km # Instelbaar via Settings -> Miscellaneous.
        n_height_ticks = int((z_axis[-1]-z_axis[0])//height_tick_interval_km)
        height_tick_segments = []
        corner_x, corner_y = box_corners_x[0], box_corners_y[0]
        for k in range(n_height_ticks+1):
            real_height_km = min(k*height_tick_interval_km, z_axis[-1]-z_axis[0])
            z_scene = z_axis[0]+real_height_km*vertical_exaggeration
            height_tick_segments.append([(corner_x-tick_len/2, corner_y, z_scene), (corner_x+tick_len/2, corner_y, z_scene)])
            if k > 0 and k % 2 == 0: # Alleen bij elk 2e streepje een cijfer -- zie toelichting bij de
                #bodemranden hierboven.
                text_visuals.append(scene.visuals.Text(f"{real_height_km:.0f}",
                                     pos=(corner_x-tick_len*2, corner_y, z_scene), color='white',
                                     font_size=AXIS_LABEL_FONT_SIZE, parent=view.scene))
        _height_ticks_line = scene.visuals.Line(pos=np.array(height_tick_segments).reshape(-1, 3), connect='segments',
                            color=(1, 1, 1, 0.9), width=2, parent=view.scene)
        _height_ticks_line.set_gl_state(depth_test=False) # Zelfde reden als bij de kader-kubus hierboven.
        self._volume_3d_wireframe_visuals.append(_height_ticks_line)
        # De verticale rand zelf ook tekenen, als duidelijke "liniaal-staaf" waarlangs de streepjes lopen.
        _height_bar_line = scene.visuals.Line(pos=np.array([[corner_x, corner_y, z_scene_min], [corner_x, corner_y, z_scene_max]]),
                            color=(1, 1, 1, 0.9), width=2, parent=view.scene)
        _height_bar_line.set_gl_state(depth_test=False) # Zelfde reden als bij de kader-kubus hierboven.
        self._volume_3d_wireframe_visuals.append(_height_bar_line)
        text_visuals.append(scene.visuals.Text(f"hoogte {z_axis[-1]-z_axis[0]:.1f} km",
                             pos=(corner_x, corner_y, z_scene_max), color='white', font_size=AXIS_LABEL_FONT_SIZE,
                             bold=True, parent=view.scene))

        # Positiemarkers (7 juli 2026): elke marker die je via het rechtermuisknop-menu op de kaart hebt gezet
        # ("Set position marker: Coordinate input", zie set_marker_coordinates()) en die binnen het huidige
        # 3D-kader valt, wordt hier als een verticale lijn getekend, van de bodem tot de bovenkant van het
        # kader -- zodat je in de 3D-weergave meteen ziet waar die positie zich bevindt t.o.v. de storm. Cyaan
        # gekozen om duidelijk te onderscheiden van de rood/groen/blauw/gele kompasranden. Het optionele label
        # (in te stellen in datzelfde scherm) wordt, indien ingevuld, bovenaan de lijn getoond.
        for _marker_i, _marker_pos in enumerate(self.pos_markers_positions):
            _mx, _my = float(_marker_pos[0]), float(_marker_pos[1])
            if x_axis[0] <= _mx <= x_axis[-1] and y_axis[0] <= _my <= y_axis[-1]:
                _marker_line = scene.visuals.Line(pos=np.array([[_mx, _my, z_scene_min], [_mx, _my, z_scene_max]]),
                                    color=(0, 1, 1, 0.9), width=2, parent=view.scene)
                _marker_line.set_gl_state(depth_test=False) # Zelfde reden als bij de kader-kubus hierboven.
                _marker_label = self.pos_markers_labels[_marker_i] if _marker_i < len(self.pos_markers_labels) else ''
                if _marker_label:
                    text_visuals.append(scene.visuals.Text(_marker_label, pos=(_mx, _my, z_scene_max),
                                         color='cyan', font_size=AXIS_LABEL_FONT_SIZE, bold=True, parent=view.scene))

        # Kleurschaal (7 juli 2026, definitieve aanpak): rechtstreeks het CSV-kleurtabel-bestand zelf inlezen
        # dat voor dit product actief staat (self.colortables_dirs_filenames[product]) i.p.v. via cm1/cm2 --
        # die laatste twee bleken allebei net niet de exacte, zichtbare kleuren/waarden te geven die Erik
        # verwachtte (2 mislukte pogingen, 7 juli 2026). Rechtstreeks uit het bronbestand lezen is
        # onafhankelijk van die eerdere verwarring, en werkt bovendien automatisch correct voor ELKE
        # kleurtabel die je ooit via Settings -> Color tables instelt (dus ook bij bv. de ESSL-tabellen).
        def _parse_colortable_csv(filepath):
            values, colors, step = [], [], None
            with open(filepath) as f:
                for line in f:
                    line = line.strip()
                    if line.lower().startswith('step:'):
                        try: step = float(line.split(':')[1].strip())
                        except Exception: pass
                        continue
                    if not line or ':' in line:
                        continue # Lege regels en andere header-regels (bv. "Units: m/s") overslaan.
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) < 4:
                        continue
                    try:
                        v = float(parts[0])
                        r, g, b = float(parts[1])/255., float(parts[2])/255., float(parts[3])/255.
                    except ValueError:
                        continue
                    values.append(v); colors.append((r, g, b, 1.0))
            return np.array(values), np.array(colors), step

        # Titel bovenaan (7 juli 2026, op Eriks verzoek): scan-datum/tijd wit, gecentreerd, in het vaste
        # titel-paneel dat hierboven bij canvas/view is aangemaakt -- blijft daardoor altijd op zijn plek
        # staan, ongeacht hoe de 3D-weergave zelf gedraaid/gekanteld/gezoomd wordt.
        _title_name = gv.productnames_cmapstab.get(product, product.upper()) if hasattr(gv, 'productnames_cmapstab') else product.upper()
        # De weergavemethode (MIP/translucent) wordt in de titel getoond en LIVE bijgewerkt wanneer je op de
        # M-toets drukt (zie toggle_3d_render_method verderop, die self._volume_3d_method_text.text aanpast) --
        # zonder dit was nergens te zien in welke van de twee modi je momenteel zat (Erik, 7 juli 2026).
        self._volume_3d_method_text = scene.visuals.Text(
            f"{_title_name} {scandatetime_str}",
            pos=(300, 15), color='white', font_size=14, bold=True, parent=title_view.scene)

        if is_polrgb:
            # PolRGB heeft geen kleurentabel (self.colortables_dirs_filenames['g'] is slechts een ongebruikte
            # placeholder, zie hierboven), dus geen zinvolle ENKELE kleurschaal-legenda zoals bij een gewoon
            # product. In plaats daarvan: 3 losse balkjes (R=Z, G=CC, B=ZDR), zelfde principe als de 2D-
            # PolRGB-legenda (PlottingPanels.draw_polrgb_legend/polrgb_channel_colors in nlr_plotting.py) --
            # zwart bij het kanaal-minimum, volle kanaalkleur bij het kanaal-maximum, Z met dezelfde
            # Z_GAMMA-correctie als de echte data (zie get_volume_grid_polrgb) zodat de balk hetzelfde
            # niet-lineaire verloop toont als de puntenwolk zelf.
            _params = self.polrgb_params
            _fallback = {'Z_MIN':-10.0,'Z_MAX':60.0,'CC_MIN':70.0,'CC_MAX':100.0,'ZDR_MIN':0.0,'ZDR_MAX':3.0,'Z_GAMMA':2.0}
            def _pp(key):
                return _params[key] if key in _params else _fallback[key]
            _bar_specs = [((1.,0.,0.), _pp('Z_MIN'), _pp('Z_MAX'), 'Z', 'dBZ', max(_pp('Z_GAMMA'), 0.1)),
                          ((0.,1.,0.), _pp('CC_MIN'), _pp('CC_MAX'), u'\u03c1HV', '%', 1.0),
                          ((0.,0.,1.), _pp('ZDR_MIN'), _pp('ZDR_MAX'), 'ZDR', 'dB', 1.0)]
            _n_rows = 120 # Per balk (i.p.v. 256 voor 1 enkele balk hierboven) -- 3 balkjes moeten samen in
            #hetzelfde smalle, vaste legend_view-paneel passen.
            _bar_width = 22
            _bar_gap = 55 # Verticale ruimte tussen 2 balkjes, voor de min/max-tekst + kanaallabel ertussen.
            _y = 0
            for _color, _vmin, _vmax, _label, _unit, _gamma in _bar_specs:
                _t = np.linspace(0., 1., _n_rows) ** _gamma
                _img = np.ones((_n_rows, 1, 4), dtype='float32')
                for _ch in range(3):
                    _img[:, 0, _ch] = _t*_color[_ch]
                _img_visual = scene.visuals.Image(_img, parent=legend_view.scene)
                _img_visual.transform = scene.STTransform(translate=(0, _y), scale=(_bar_width, 1))
                scene.visuals.Text(f"{_label} ({_unit})", pos=(_bar_width/2, _y-16), color='white',
                                    font_size=10, bold=True, parent=legend_view.scene)
                scene.visuals.Text(str(ft.rifdot0(ft.rndec(_vmax, 2))), pos=(_bar_width+6, _y),
                                    color='white', font_size=9, anchor_x='left', parent=legend_view.scene)
                scene.visuals.Text(str(ft.rifdot0(ft.rndec(_vmin, 2))), pos=(_bar_width+6, _y+_n_rows),
                                    color='white', font_size=9, anchor_x='left', parent=legend_view.scene)
                _y += _n_rows+_bar_gap
            legend_view.camera.rect = (-70, -30, _bar_width+140, _y+30)
        else:
            _ct_path = self.colortables_dirs_filenames[product]
            _ct_values, _ct_colors, _ct_step = _parse_colortable_csv(_ct_path)
            _legend_vmax, _legend_vmin = float(_ct_values[-1]), float(_ct_values[0])

            _n_cbar_rows = 256
            _cbar_bar_width = 22 # In dezelfde lokale eenheden als _n_cbar_rows (nu net als bij een afbeelding:
            #1 "pixel" breed voor schaling); bepaalt puur de zichtbare dikte van de balk in de legenda.
            # Rij 0 = bovenkant van de afbeelding = hoogste waarde (vmax); laatste rij = laagste waarde (vmin) --
            # dat komt overeen met hoe de bestaande 2D-kleurenbalken (zie de bijgevoegde screenshots) het tonen:
            # hoge waarden boven, lage waarden onder.
            # OMGEDRAAID (7 juli 2026): bleek dat rij 0 in dit legend_view/PanZoomCamera-paneel juist ONDERAAN
            # het scherm terechtkomt (y neemt hier kennelijk omhoog toe, net als bij een gewone Cartesische as,
            # NIET zoals bij een standaard afbeelding waar rij 0 boven zou moeten staan) -- vandaar dat -20/-60
            # eerst boven i.p.v. onder stond. Nu vmin/vmax verwisseld t.o.v. de vorige versie.
            _row_values = np.linspace(_legend_vmin, _legend_vmax, _n_cbar_rows)
            _cbar_img_data = np.ones((_n_cbar_rows, 1, 4), dtype='float32')
            for _ch in range(3):
                _cbar_img_data[:, 0, _ch] = np.interp(_row_values, _ct_values, _ct_colors[:, _ch])
            _cbar_image = scene.visuals.Image(_cbar_img_data, parent=legend_view.scene)
            _cbar_image.transform = scene.STTransform(scale=(_cbar_bar_width, 1))

            _cbar_unit = self.pb.productunits.get(product, '')
            _cbar_name = gv.productnames_cmapstab.get(product, product.upper()) if hasattr(gv, 'productnames_cmapstab') else product.upper()
            scene.visuals.Text(f"{_cbar_unit}" if _cbar_unit else _cbar_name, pos=(_cbar_bar_width/2, -20),
                                color='white', font_size=10, bold=True, parent=legend_view.scene)
            scene.visuals.Text(f"{_cbar_name}", pos=(_cbar_bar_width/2, _n_cbar_rows+20),
                                color='white', font_size=10, bold=True, parent=legend_view.scene)
            # Tick-stappen rechtstreeks uit de "Step:"-regel in het bestand zelf (bv. 10 voor Z, 15 voor V) --
            # zelfde regelmatige tick-afstand als in de bijgevoegde referentie-schermafbeeldingen. Als het
            # bestand geen Step-regel bevat, val terug op 5 gelijk verdeelde tick-punten.
            if _ct_step:
                _first_tick = np.ceil(_legend_vmin/_ct_step)*_ct_step
                _tick_values = np.arange(_first_tick, _legend_vmax+1e-6, _ct_step)
            else:
                _tick_values = _legend_vmin+np.array([0., 0.25, 0.5, 0.75, 1.])*(_legend_vmax-_legend_vmin)
            for _tick_value in _tick_values:
                _tick_text = str(ft.rifdot0(ft.rndec(_tick_value, 3)))
                _tick_frac = 0. if _legend_vmax == _legend_vmin else (_tick_value-_legend_vmin)/(_legend_vmax-_legend_vmin)
                _tick_row = _tick_frac*_n_cbar_rows
                scene.visuals.Text(_tick_text, pos=(_cbar_bar_width+18, _tick_row), color='white',
                                    font_size=9, anchor_x='left', parent=legend_view.scene)
            # Cameragebied ruim rond de balk (marge links/rechts voor tick-tekst, boven/onder voor de
            # eenheid/product-labels) -- vast (interactive=False hierboven), dus dit hoeft maar 1x goed te staan.
            # Marge flink verruimd (7 juli 2026): "Reflectivity" werd links afgeknipt (het woord is te breed voor
            # het gecentreerde label t.o.v. de smalle 22-brede balk) -- ruimere marge geeft de gecentreerde
            # tekst voldoende plek, ten koste van een ietsje kleinere balk/tekst binnen hetzelfde vaste
            # paneel (legend_view.width_max), wat geen probleem is.
            legend_view.camera.rect = (-70, -40, _cbar_bar_width+140, _n_cbar_rows+80)


        # self._volume_3d_text_visuals wordt bij ELKE (her)tekening vervangen door de nieuwste lijst, en de
        # toets-koppeling verwijst naar self._volume_3d_text_visuals (niet naar een lokale closure-variabele)
        # -- zo blijft de T-toets ook na een live-refresh de JUISTE (nieuwste) tekst-objecten aan/uit zetten,
        # zonder de key_press-handler telkens opnieuw (en dus meerdere keren tegelijk) te moeten koppelen.
        self._volume_3d_text_visuals = text_visuals

        # BUGFIX (10 juli 2026): wireframe/tekst-visuals worden hierboven bij ELKE (her)tekening helemaal
        # opnieuw aangemaakt (dus standaard weer zichtbaar), ongeacht een eerder met B/T ingestelde
        # combinatie. self._volume3d_wireframe_visible/_volume3d_text_visible (default True bij een
        # gloednieuw venster) worden hier toegepast zodat die instelling standhoudt over herbouwen heen --
        # essentieel voor de video-export (self._run_volume_3d_export), die _render_volume_3d_scene meerdere
        # keren achter elkaar aanroept.
        if not hasattr(self, '_volume3d_wireframe_visible'):
            self._volume3d_wireframe_visible = True
        if not hasattr(self, '_volume3d_text_visible'):
            self._volume3d_text_visible = True
        for wv in self._volume_3d_wireframe_visuals:
            wv.visible = self._volume3d_wireframe_visible
        for tv in self._volume_3d_text_visuals:
            tv.visible = self._volume3d_text_visible

        # Diagnose (15 juli 2026, op Eriks vraag "welke elevatiehoek is duplicate, dat kan ik zelf nooit
        # vinden"): welke scan-indices van het huidige radar/dataset binnen 1 volume vaker dan 1x worden
        # afgetast (zie de toelichting bij step_3d_time hierboven -- dat verklaart het "om en om"-patroon
        # bij tijdnavigatie), en of de nu in dit paneel geselecteerde scan daar toevallig bij hoort. Puur
        # informatief (console + korte statusbalk-melding), verandert niets aan de rendering zelf.
        try:
            _dup_scan_idx = sorted(i for i, jj in self.dsg.scannumbers_all['z'].items() if len(jj) > 1)
            _current_scan = self.crd.scans[j]
            if _dup_scan_idx:
                _dup_angles = {}
                for i in _dup_scan_idx:
                    try:
                        _dup_angles[i] = round(float(self.dsg.scanangles_all['z'][i]), 2)
                    except Exception:
                        _dup_angles[i] = '?'
                _current_flag = " <- DIT IS DE NU GESELECTEERDE SCAN" if _current_scan in _dup_scan_idx else ""
                print(f"[3D viewer] Elevatiehoeken met duplicaten binnen 1 volume (scan-index: hoek in graden): "
                      f"{_dup_angles}. Huidige scan-index in dit paneel: {_current_scan}{_current_flag}")
                if _current_scan in _dup_scan_idx:
                    self.set_textbar(f"Let op: huidige elevatiehoek ({_dup_angles.get(_current_scan, '?')} graden) "
                                      f"heeft duplicaten binnen 1 volume -- tijdnavigatie kan daardoor soms een "
                                      f"deel-stap i.p.v. een vol nieuw volume geven. Zie console voor alternatieven.",
                                      'orange', 4)
            else:
                print("[3D viewer] Geen elevatiehoeken met duplicaten gevonden voor dit radar/dataset.")
        except Exception as _e_dup:
            print(f"[3D viewer] Kon duplicate-scans niet bepalen: {_e_dup}")

        if not reuse:
            # Sneltoets T (binnen dit 3D-venster): alle tekstlabels in 1 keer aan/uit -- op verzoek van Erik,
            # 5 juli 2026. Alleen bij het aanmaken van een NIEUW venster koppelen, niet bij elke live-
            # refresh, anders zou dezelfde toets na een paar instellingswijzigingen meerdere keren per druk
            # toggelen.
            def toggle_3d_labels(event):
                if event.key is not None and event.key.name == 'T':
                    # BUGFIX (10 juli 2026, zie ook toggle_3d_wireframe hierboven): zelfde reden -- de vlag
                    # self._volume3d_text_visible wordt apart onthouden en na elke herbouw toegepast, zodat
                    # T-alleen (tekst uit, kubus aan) ook standhoudt tijdens de video-export.
                    new_visible = not self._volume_3d_text_visuals[0].visible if self._volume_3d_text_visuals else True
                    self._volume3d_text_visible = new_visible
                    for tv in self._volume_3d_text_visuals:
                        tv.visible = new_visible
                    canvas.update()
            canvas.events.key_press.connect(toggle_3d_labels)

            # Sneltoets M (binnen dit 3D-venster): wissel tussen 'mip' (huidige, scherpe standaard) en
            # 'translucent' (semi-transparant, laat eventuele interne structuur -- bv. een zwakke-echo-koker
            # (BWER) -- er vaag doorheen zien, ten koste van scherpte). Op verzoek van Erik, 7 juli 2026, om
            # zonder risico voor de standaardweergave (MIP blijft de default) te kunnen experimenteren.
            # Dezelfde reden als bij toggle_3d_labels hierboven om ALLEEN bij een nieuw venster te koppelen
            # (verwijst naar self._volume_3d_visual, niet naar de lokale closure-variabele 'volume').
            def toggle_3d_render_method(event):
                if event.key is not None and event.key.name == 'M':
                    # PolRGB (product 'g') heeft geen aparte translucent-variant: de additieve RGB-MIP-
                    # weergave hierboven (3 losse kanaal-volumes, additief samengesteld) is zelf al de
                    # PolRGB-tegenhanger van MIP. Een losse translucent-versie is (nog) niet gebouwd -- P
                    # blijft het enige alternatief (puntenwolk, met echte per-voxel-kleuren i.p.v. de
                    # per-kanaal-benadering van de additieve MIP-weergave).
                    if getattr(self, '_volume3d_rect_params', None) and self._volume3d_rect_params[0] == 'g':
                        self.set_textbar("PolRGB heeft geen aparte translucent-modus (wel een puntenwolk-"
                                          "alternatief met echte kleuren, zie P).", 'orange', 3)
                        return
                    current = self._volume_3d_visual.method
                    self._volume_3d_visual.method = 'translucent' if current == 'mip' else 'mip'
                    new_method = self._volume_3d_visual.method
                    self.set_textbar(f"3D-weergavemethode: {new_method}", 'green', 3)
                    # Modus-indicatie live bijwerken in de Windows-titelbalk (8 juli 2026, BIJGESTELD op Eriks
                    # verzoek: geen tekst meer in de 3D-scene zelf behalve de titel -- dus [MIP]/[TRANSLUCENT]
                    # staat nu in canvas.title i.p.v. in een aparte tekst-visual binnen de scene).
                    old_title = canvas.title
                    canvas.title = old_title[:old_title.index('[')] + f"[{new_method.upper()}]" + old_title[old_title.index(']')+1:]
                    canvas.update()
            canvas.events.key_press.connect(toggle_3d_render_method)

            # Sneltoets P (8 juli 2026): wisselt naar/van de puntenwolk-weergave (zie hierboven bij
            # canvas/view voor de opbouw). Losse, aparte toets t.o.v. M (i.p.v. M laten rouleren door 3
            # standen) zodat MIP/translucent zelf volledig ongewijzigd blijven werken zoals altijd.
            def toggle_3d_pointcloud(event):
                if event.key is not None and event.key.name == 'P':
                    is_g = bool(getattr(self, '_volume3d_rect_params', None)) and self._volume3d_rect_params[0] == 'g'
                    pc = self._volume_3d_pointcloud_visual
                    pc.visible = not pc.visible
                    if is_g:
                        # Bij PolRGB wisselt P tussen de additieve RGB-MIP-weergave (3 kanaal-volumes, zie
                        # hierboven) en de puntenwolk (echte per-voxel-kleuren) -- geen enkel scalair
                        # Volume-object om aan/uit te zetten zoals bij een gewoon product.
                        for _v in self._volume_3d_polrgb_visuals:
                            _v.visible = not pc.visible
                        new_mode = 'POINTCLOUD' if pc.visible else 'RGB-MIP'
                    else:
                        self._volume_3d_visual.visible = not pc.visible
                        new_mode = 'POINTCLOUD' if pc.visible else self._volume_3d_visual.method.upper()
                    self.set_textbar(f"3D-weergavemethode: {new_mode.lower()}", 'green', 3)
                    old_title = canvas.title
                    canvas.title = old_title[:old_title.index('[')] + f"[{new_mode}]" + old_title[old_title.index(']')+1:]
                    canvas.update()
            canvas.events.key_press.connect(toggle_3d_pointcloud)

            # Sneltoets H (8 juli 2026, BIJGESTELD op Eriks verzoek: "geen tekst meer in de 3D-weergave zelf,
            # behalve de titel -- de rest in een apart schermpje"): opent een ECHT, los Qt-venster met de
            # toetsen-uitleg, i.p.v. tekst binnen de vispy-scene zelf. Sluit je het 3D-venster, dan sluit dit
            # hulpschermpje (als kind-venster) automatisch mee.
            def show_3d_help(event):
                if event.key is not None and event.key.name == 'H':
                    if getattr(self, '_volume_3d_help_dialog', None) is not None:
                        try:
                            self._volume_3d_help_dialog.close()
                        except Exception:
                            pass
                        self._volume_3d_help_dialog = None
                        return
                    dialog = QWidget()
                    dialog.setWindowTitle('NLradar 3D - toetsenbediening')
                    layout = QVBoxLayout()
                    for line in (
                        'T       labels (windrichtingen, tick-cijfers, hoogte) aan/uit',
                        'M       wisselen tussen MIP en translucent',
                        'R       camera resetten naar de startpositie',
                        'P       puntenwolk-weergave aan/uit',
                        'H       dit schermpje tonen/verbergen',
                        'B       kader-kubus + tick-streepjes + hoogte-liniaal + labels aan/uit',
                        'S       huidige 3D-weergave opslaan als PNG-schermafbeelding',
                        'J       video exporteren (tijd + camerarotatie, instelbaar)',
                        'Rechts  1 scan vooruit in de tijd (3D wordt herbouwd)',
                        'Links   1 scan terug in de tijd (3D wordt herbouwd)',
                        'A       automatisch met de tijd mee laten lopen aan/uit',
                        '',
                        'Cyaan verticale lijn = positiemarker (rechtermuisknop op de kaart)',
                    ):
                        layout.addWidget(QLabel(line))
                    dialog.setLayout(layout)
                    dialog.resize(dialog.sizeHint())
                    dialog.show()
                    self._volume_3d_help_dialog = dialog
            canvas.events.key_press.connect(show_3d_help)

            # Sneltoets B (8 juli 2026, op Eriks verzoek: "de hele kubus-omlijning met tics en al aan/uit
            # zetten"): toont/verbergt in een keer alle kader-/tick-lijn-elementen (self._volume_3d_
            # wireframe_visuals, hierboven verzameld) -- los van de T-toets, die alleen de TEKST-labels
            # (windrichtingen, tick-cijfers, hoogte) raakt, niet de lijnen zelf.
            def toggle_3d_wireframe(event):
                if event.key is not None and event.key.name == 'B':
                    # Bijgesteld (8 juli 2026, Erik: "de hoogte en windrichtingen + getallen moeten ook
                    # tegelijk aan/uit"): B verbergt/toont nu zowel de lijn-elementen (kader-kubus, tick-
                    # streepjes, hoogte-liniaal) ALS de bijbehorende tekst-labels (windrichtingen, tick-
                    # cijfers, hoogte) in een keer -- i.p.v. dat je daarvoor apart ook nog T nodig had.
                    #
                    # BUGFIX (10 juli 2026, gevonden tijdens testen van de J-video-export): self._volume_3d_
                    # wireframe_visuals wordt bij ELKE (her)tekening HELEMAAL OPNIEUW aangemaakt (zie
                    # hierboven), dus zonder onthouden vlag verloor een eerder met B/T ingestelde combinatie
                    # (bv. "kubus uit, tics aan") zich na elke tijdstap tijdens de video-export -- de kubus
                    # kwam dan steeds terug in de opgeslagen beeldjes. self._volume3d_wireframe_visible wordt
                    # daarom nu apart onthouden en na elke herbouw toegepast (zie verderop bij text_visuals).
                    new_visible = not self._volume_3d_wireframe_visuals[0].visible if self._volume_3d_wireframe_visuals else True
                    self._volume3d_wireframe_visible = new_visible
                    self._volume3d_text_visible = new_visible
                    for wv in self._volume_3d_wireframe_visuals:
                        wv.visible = new_visible
                    for tv in self._volume_3d_text_visuals:
                        tv.visible = new_visible
                    canvas.update()
            canvas.events.key_press.connect(toggle_3d_wireframe)

            # Sneltoets S (8 juli 2026, op Eriks verzoek): slaat de HUIDIGE 3D-weergave (inclusief legenda,
            # titel, positiemarkers -- alles wat je op dat moment ziet) op als PNG-bestand. Gebruikt vispy's
            # eigen canvas.render() (pakt precies de inhoud van dit ene vispy-canvas, niet een schermafbeelding
            # van je hele beeldscherm/bureaublad), en PIL.Image om weg te schrijven -- beide al elders in dit
            # bestand gebruikt voor de gewone (2D) "sla op als afbeelding"-functie, dus geen nieuwe dependency.
            def save_3d_screenshot(event):
                if event.key is not None and event.key.name == 'S':
                    # BIJGESTELD (8 juli 2026, op Eriks verzoek: "krijg ik dan eerst nog een keuze waar ik het
                    # wil opslaan?") -- toont nu een echt "Opslaan als"-dialoogvenster (hetzelfde patroon als
                    # elders in dit bestand, bv. bij het kiezen van een kleurtabel-bestand), i.p.v. altijd
                    # automatisch naar een vaste map te schrijven.
                    try:
                        _dt_str = str(scandatetime).replace(':', '').replace(' ', '_').replace('-', '')
                        _default_dir = os.path.join(gv.userdir, 'Output_files', '3D_screenshots')
                        os.makedirs(_default_dir, exist_ok=True)
                        _default_name = f"3D_{product}_{_dt_str}_{pytime.strftime('%H%M%S')}.png"
                        _default_path = os.path.join(_default_dir, _default_name)
                        filepath = str(QFileDialog.getSaveFileName(None, 'Sla de 3D-weergave op als:',
                                       _default_path, filter='*.png', options=QFileDialog.DontUseNativeDialog)[0])
                        if not filepath:
                            return # Geannuleerd
                        if not filepath.lower().endswith('.png'):
                            filepath += '.png'
                        img_arr = canvas.render()
                        Image.fromarray(img_arr).save(filepath)
                        self.set_textbar(f"3D-schermafbeelding opgeslagen: {filepath}", 'green', 4)
                    except Exception as _e:
                        self.set_textbar(f"Opslaan 3D-schermafbeelding mislukt: {_e}", 'red', 4)
            canvas.events.key_press.connect(save_3d_screenshot)



            # Sneltoets R (binnen dit 3D-venster, 7 juli 2026, BIJGESTELD): zet de camera terug naar de
            # startpositie. Eerdere versie stelde alleen elevatie/azimut/fov bij op de BESTAANDE camera, maar
            # dat bleek onvoldoende -- Erik liep na verloop van tijd tegen een "kan niet meer voor-/achterover
            # kantelen"-probleem aan dat R zo niet oploste, terwijl het volledig sluiten en opnieuw openen van
            # het venster (waarbij een HELE NIEUWE camera wordt aangemaakt) het wel oploste. Dat wijst erop
            # dat TurntableCamera intern meer status bijhoudt dan alleen die drie zichtbare eigenschappen, die
            # tijdens langdurig slepen kennelijk kan "vastlopen". Nu maakt R daarom, net als bij het openen
            # van een vers venster, een VOLLEDIG NIEUWE camera aan (i.p.v. de bestaande bij te stellen) --
            # inclusief het opnieuw toepassen van de centrering, die anders verloren zou gaan.
            def reset_3d_camera(event):
                if event.key is not None and event.key.name == 'R':
                    view.camera = 'turntable'
                    view.camera.fov = 45
                    view.camera.azimuth = 0
                    view.camera.elevation = 65
                    view.camera.center = (box_center_x, box_center_y, (box_corners_z[0]+box_corners_z[1])/2.)
                    view.camera.set_range()
                    self.set_textbar("3D-camera volledig opnieuw aangemaakt en teruggezet naar startpositie", 'green', 3)
                    canvas.update()
            canvas.events.key_press.connect(reset_3d_camera)

            # Sneltoets J (10 juli 2026, op Eriks verzoek; eerst V, toen E overwogen, maar beide bleken al
            # globale product-sneltoetsen -- Z/A/M/H/R/E/L/V/UV/S/W/P/K/C/D/X/Q/T/Y/I/G zijn allemaal al
            # bezet via gv.products_all in nlr_globalvars.py. J is geverifieerd volledig vrij, zowel globaal
            # als binnen dit 3D-venster): opent een instellingendialoog voor het exporteren van een VIDEO
            # (i.p.v. een losse PNG zoals de S-toets), waarin zowel de tijd doorloopt (elke volgende
            # radarscan) als de camera een instelbare hoek verder draait. Zie
            # export_volume_3d_video/_run_volume_3d_export hieronder voor de daadwerkelijke implementatie.
            # Alleen bij een NIEUW venster koppelen, zelfde reden als bij T/M/P/H/B/S/R hierboven.
            def export_3d_video_key(event):
                if event.key is not None and event.key.name == 'J':
                    self.export_volume_3d_video()
            canvas.events.key_press.connect(export_3d_video_key)

            # Sneltoetsen Rechts/Links en A (15 juli 2026, op Eriks verzoek "kan ik de 3D met de tijd mee
            # laten lopen?"): Rechts/Links stapt de 3D-scene 1 scan vooruit/achteruit in de tijd, A schakelt
            # een doorlopende "automatisch meelopen"-modus aan/uit. Hergebruikt BEWUST dezelfde aanpak als
            # de al bestaande (nog niet door Erik bevestigde, zie LET OP bij _run_volume_3d_export hieronder)
            # J-video-export: self.crd.process_keyboardinput(1,0,0,'0',None,False) (=pijltje-rechts) om een
            # stap te zetten, en _wait_for_new_scandatetime om te wachten tot die nieuwe scan daadwerkelijk
            # is ingeladen voordat de 3D-scene wordt herbouwd -- dus hetzelfde risico als daar staat
            # beschreven (ongeteste aanname over hoe/wanneer nieuwe scandata precies beschikbaar komt).
            # Rechts/A zijn door Erik BEVESTIGD werkend; het achteruit-gedeelte (Links, -1) is een aanname
            # op basis daarvan, symmetrisch aan Rechts, maar nooit apart getest.
            def step_3d_time(direction):
                if getattr(self, '_volume3d_stepping', False):
                    return False  # Voorkomt overlappende stappen als er nog eentje bezig is.
                self._volume3d_stepping = True
                try:
                    j2 = self.pb.panel
                    old_scandatetime = self._volume3d_rect_params[3]
                    # BUGFIX TERUGGEDRAAID (15 juli 2026): hier stond kort 12*direction i.p.v. direction, in
                    # de veronderstelling dat leftright_step=12 (SHIFT+RECHTS/LINKS) altijd een VOLLEDIGE
                    # volumestap zou geven ongeacht "duplicate" scans binnen 1 volume. Dat bleek niet te
                    # kloppen: in nlr_changedata.py (perform_leftrightstep) betekent |lr_step|==12 een VASTE
                    # sprong van 60 minuten (letterlijk zo in commentaar: "Use a time step of 60 minutes"),
                    # los van de daadwerkelijke volumetijd van de radar (meestal enkele minuten) -- dus geen
                    # "volgende volledige scan", maar een vaste uur-sprong. Terug naar de oorspronkelijke 1/-1.
                    #
                    # De ECHTE oorzaak van Eriks "om en om slaat hij een compleet beeldje over"-waarneming zit
                    # in hetzelfde bestand: consider_duplicates (perform_leftrightstep, alleen relevant bij
                    # |lr_step|==1) is True als het momenteel zichtbare paneel een scan/elevatiehoek toont die
                    # BINNEN 1 volume vaker wordt afgetast (self.visible_productscans_with_duplicates, bv. een
                    # laagste elevatiehoek die om extra tijdresolutie vaker wordt gescand dan de rest van het
                    # volume -- vergelijkbaar met NEXRAD SAILS/MESO-SAILS). In dat geval stapt RECHTS soms
                    # alleen naar de eerstvolgende DUPLICATE van diezelfde elevatiehoek (tijd verschuift wel,
                    # maar de OVERIGE elevatiehoeken -- en dus het 3D-volume als geheel -- blijven van het
                    # oude volume) i.p.v. naar een echt nieuw, volledig volume met alle elevaties vernieuwd.
                    # Dat verklaart het "om en om"-patroon exact. Geen simpele parameterfix hiervoor gevonden
                    # die niet ook (zoals hierboven) iets anders breekt.
                    # POGING 2 (15 juli 2026, na Eriks "wat schiet ik hiermee op?"): i.p.v. alleen te wachten
                    # tot de LOSSE scan-tijd verandert (die verandert ook al bij een duplicate-deelstap, zie
                    # toelichting hierboven), nu blijven doorstappen tot het VOLUME-tijdstip zelf verandert.
                    # Onderbouwing, rechtstreeks uit nlr_changedata.py (perform_leftrightstep): het volume-
                    # tijdstip (self.date/self.time daar, hier self.crd.date/self.crd.time) wordt ALLEEN
                    # bijgewerkt als use_same_volumetime False is -- dus als er daadwerkelijk een nieuwe,
                    # volledige volumeronde is bereikt, niet bij een deelstap binnen dezelfde ronde. Dat is dus
                    # een betrouwbaarder signaal dan de losse scan-tijd. NOG NIET DOOR ERIK BEVESTIGD: dit gaat
                    # ervan uit dat self.crd.date/self.crd.time synchroon met process_keyboardinput worden
                    # bijgewerkt (net als bij de rest van dit 3D-onderdeel kon ik dit niet zelf natesten).
                    old_volume_dt = self.crd.date + self.crd.time
                    _max_attempts = 20  # Veiligheidsgrens tegen een oneindige lus als er iets onverwachts gebeurt.
                    for _attempt in range(_max_attempts):
                        self.crd.process_keyboardinput(direction, 0, 0, '0', None, False)
                        new_scandatetime = self._wait_for_new_scandatetime(j2, old_scandatetime)
                        if new_scandatetime is None:
                            richting = 'nieuwere' if direction > 0 else 'oudere'
                            self.set_textbar(f"3D: geen {richting} scan beschikbaar (einde bereikt).", 'orange', 2)
                            self._volume3d_following = False
                            return False
                        new_volume_dt = self.crd.date + self.crd.time
                        if new_volume_dt != old_volume_dt:
                            break  # Nu wel een echt nieuw, volledig volume bereikt.
                        old_scandatetime = new_scandatetime  # Nog dezelfde ronde (duplicate-substap) -- meteen door.
                    else:
                        self.set_textbar("3D: geen volledig nieuw volume gevonden na herhaald doorstappen "
                                          "(mogelijk blijvend dezelfde elevatiehoek-duplicaten) -- toont het "
                                          "laatst bereikte resultaat.", 'orange', 3)
                    product2, x_range2, y_range2, _ = self._volume3d_rect_params
                    self._volume3d_rect_params = (product2, x_range2, y_range2, new_scandatetime)
                    self._render_volume_3d_scene(force_new_window=False)
                    return True
                finally:
                    self._volume3d_stepping = False

            def volume3d_follow_tick():
                if not getattr(self, '_volume3d_following', False):
                    return
                ok = step_3d_time(1)
                if ok and getattr(self, '_volume3d_following', False):
                    # Klein pauzetje tussen stappen (via singleShot i.p.v. een blocking sleep), zodat het
                    # 3D-venster ondertussen gewoon interactief/draaibaar blijft.
                    QTimer.singleShot(50, volume3d_follow_tick)
                else:
                    self._volume3d_following = False
            self._volume3d_follow_tick_func = volume3d_follow_tick  # Referentie vasthouden voor de singleShot-keten.

            def handle_3d_time_keys(event):
                if event.key is None:
                    return
                name = event.key.name
                if name == 'Right' and not getattr(self, '_volume3d_following', False):
                    step_3d_time(1)
                elif name == 'Left' and not getattr(self, '_volume3d_following', False):
                    step_3d_time(-1)
                elif name == 'A':
                    self._volume3d_following = not getattr(self, '_volume3d_following', False)
                    if self._volume3d_following:
                        self.set_textbar("3D: automatisch meelopen met de tijd gestart ('A' om te stoppen).", 'green', 2)
                        self._volume3d_follow_tick_func()
                    else:
                        self.set_textbar('3D: automatisch meelopen met de tijd gestopt.', 'orange', 2)
            canvas.events.key_press.connect(handle_3d_time_keys)

            # Center the camera on the ACTUAL object instead of the default (0, 0, 0) -- (0, 0, 0) is the
            # radar's own location, which can easily be tens of km away from the selected storm. Alleen bij
            # een NIEUW venster: bij een live-refresh blijft de camera precies staan waar Erik 'm liet, dat
            # is nou net het punt van "live".
            view.camera.center = (box_center_x, box_center_y, (box_corners_z[0]+box_corners_z[1])/2.)
            view.camera.set_range()

        status_message = (f"3D-venster: '{product}' {scandatetime_str}, {width_km:.0f}x{height_km:.0f} km, "
                           f"{distance_from_radar_km:.0f} km {bearing_label} van de radar. "
                           f"Kompaslijnen: rood=Noord, groen=Oost, blauw=Zuid, geel=West, wit=omhoog. "
                           f"Hoogte-as is {vertical_exaggeration:.0f}x overdreven.")
        self.set_textbar(status_message, 'green', 4)
        print(f"[3D viewer] {status_message}")
        print(f"[3D viewer] Werkelijke hoogte van het gebied: {z_axis[-1]-z_axis[0]:.1f} km "
              f"(op het scherm {vertical_exaggeration:.0f}x uitgerekt).")

    def export_volume_3d_video(self):
        """Sneltoets J (binnen het 3D-venster, 10 juli 2026, op Eriks verzoek): toont een instellingen-
        dialoog en start daarna een video-export waarin zowel de TIJD doorloopt (elke volgende radarscan,
        alsof herhaaldelijk op pijltje-rechts gedrukt wordt -- zie regel ~799 hierboven voor die sneltoets
        zelf) als de camera een instelbare hoek verder draait over de hele video. Erik koos hiervoor
        (i.p.v. alleen tijd, of alleen rotatie) na een korte afweging: puur tijd is het duidelijkst te lezen
        qua storm-ontwikkeling, dus dat blijft de hoofdas, met een lichte/optionele rotatie erbovenop voor
        wat extra dimensionaliteit zonder verwarrend te worden (0 graden = camera blijft stilstaan).

        Gebruikt hetzelfde encodeermechanisme (PyAV/libx264) als de bestaande 2D-animatie-export
        (self.create_ani, ext='mp4') -- dus GEEN nieuwe afhankelijkheid t.o.v. ffmpeg."""
        if getattr(self, '_volume_3d_canvas', None) is None or getattr(self, '_volume3d_rect_params', None) is None:
            self.set_textbar('Geen 3D-venster open om een video van te exporteren.', 'red', 2)
            return
        if getattr(self, '_volume3d_exporting', False):
            self.set_textbar('Er loopt al een video-export.', 'orange', 2)
            return

        dialog = QWidget()
        dialog.setWindowTitle('3D-video exporteren')
        layout = QFormLayout()
        n_steps_w = QLineEdit('20')
        rotation_deg_w = QLineEdit('90')
        fps_w = QLineEdit('10')
        layout.addRow(QLabel('Aantal tijdstappen vooruit (elk = 1x pijltje-rechts):'), n_steps_w)
        layout.addRow(QLabel('Totale camerarotatie over de hele video (graden, 0 = camera stil):'), rotation_deg_w)
        layout.addRow(QLabel('Beeldjes per seconde (fps):'), fps_w)
        start_button = QPushButton('Start export', autoDefault=True)
        layout.addRow(start_button)
        dialog.setLayout(layout)
        dialog.resize(dialog.sizeHint())

        def start_export():
            try:
                n_steps = int(round(ft.to_number(n_steps_w.text())))
                rotation_deg = float(ft.to_number(rotation_deg_w.text()))
                fps = float(ft.to_number(fps_w.text()))
                if n_steps < 1 or fps <= 0:
                    raise ValueError('Ongeldige waarde(n)')
            except Exception:
                self.set_textbar('Ongeldige invoer voor tijdstappen/rotatie/fps.', 'red', 3)
                return
            dialog.close()
            product = self._volume3d_rect_params[0]
            _dt_str = str(self._volume3d_rect_params[3]).replace(':', '').replace(' ', '_').replace('-', '')
            default_dir = os.path.join(gv.userdir, 'Output_files', '3D_videos')
            os.makedirs(default_dir, exist_ok=True)
            default_name = f"3D_{product}_{_dt_str}_{pytime.strftime('%H%M%S')}.mp4"
            default_path = os.path.join(default_dir, default_name)
            filepath = str(QFileDialog.getSaveFileName(None, 'Sla de 3D-video op als:', default_path,
                           filter='*.mp4', options=QFileDialog.DontUseNativeDialog)[0])
            if not filepath:
                return  # Geannuleerd
            if not filepath.lower().endswith('.mp4'):
                filepath += '.mp4'
            self._run_volume_3d_export(n_steps, rotation_deg, fps, filepath)

        start_button.clicked.connect(start_export)
        dialog.show()
        self._volume3d_export_dialog = dialog  # Referentie vasthouden zodat Qt het venster niet meteen opruimt.

    def _run_volume_3d_export(self, n_steps, rotation_deg, fps, filepath):
        """Doet het eigenlijke werk voor export_volume_3d_video: stapt n_steps keer de tijd vooruit, draait
        de camera geleidelijk mee, en vangt na elke stap 1 beeldje met canvas.render() (hetzelfde mechanisme
        als save_3d_screenshot hierboven, S-toets). Aan het eind worden alle beeldjes samengevoegd tot 1
        MP4-bestand via _encode_frames_to_mp4.

        LET OP -- NOG NIET DOOR ERIK BEVESTIGD: dit gaat ervan uit dat self.crd.process_keyboardinput(1,0,0,
        '0',None,False) (dezelfde aanroep als de pijltje-rechts-sneltoets) de nieuwe scandata al volledig
        geladen heeft zodra de aanroep terugkeert, OF in elk geval kort daarna via Qt's event-loop (vandaar
        de wachtlus in _wait_for_new_scandatetime). Als het inladen van nieuwe scans in werkelijkheid via
        een aparte achtergrond-thread met eigen callback gebeurt (zoals bv. bij het downloaden van
        radardata), kan deze aanpak tekortschieten en moet die wachtlus vervangen worden door een echt
        signaal uit die laad-pijplijn. Graag testen en terugkoppelen of dit al dan niet werkt zoals bedoeld."""
        self._volume3d_exporting = True
        self.set_textbar('3D-video wordt geexporteerd, even geduld...', 'orange', 0)
        canvas = self._volume_3d_canvas
        view = self._volume_3d_view
        j = self.pb.panel
        azimuth_start = view.camera.azimuth
        azimuth_step = rotation_deg/(n_steps-1) if n_steps > 1 else 0.
        frames = []
        try:
            for i in range(n_steps):
                canvas.update()
                QApplication.processEvents()  # Zorgt dat vispy de scene ook echt (opnieuw) tekent voor render().
                img_arr = canvas.render()
                frames.append(img_arr.copy())
                if i == n_steps-1:
                    break
                old_scandatetime = self._volume3d_rect_params[3]
                # Zelfde fix als bij step_3d_time hierboven: blijven doorstappen tot het VOLUME-tijdstip
                # (self.crd.date+self.crd.time) daadwerkelijk verandert, i.p.v. te stoppen zodra alleen de
                # losse scan-tijd verandert (die kan ook al bij een duplicate-deelstap veranderen).
                old_volume_dt = self.crd.date + self.crd.time
                new_scandatetime = None
                for _attempt in range(20):
                    self.crd.process_keyboardinput(1, 0, 0, '0', None, False)  # Zelfde als pijltje-rechts.
                    new_scandatetime = self._wait_for_new_scandatetime(j, old_scandatetime)
                    if new_scandatetime is None:
                        break
                    if self.crd.date + self.crd.time != old_volume_dt:
                        break
                    old_scandatetime = new_scandatetime
                if new_scandatetime is None:
                    self.set_textbar(f"Einde van beschikbare data bereikt na {i+1} tijdstappen, video wordt "
                                      f"afgemaakt met wat er is.", 'orange', 3)
                    break
                product, x_range, y_range, _ = self._volume3d_rect_params
                self._volume3d_rect_params = (product, x_range, y_range, new_scandatetime)
                view.camera.azimuth = azimuth_start + azimuth_step*(i+1)
                self._render_volume_3d_scene(force_new_window=False)
            self._encode_frames_to_mp4(frames, fps, filepath)
            self.set_textbar(f"3D-video opgeslagen: {filepath} ({len(frames)} beeldjes)", 'green', 4)
        except Exception as e:
            self.set_textbar(f"3D-video-export mislukt: {e}", 'red', 3)
            print('_run_volume_3d_export error:'); print(traceback.format_exc())
        finally:
            self._volume3d_exporting = False

    def _wait_for_new_scandatetime(self, j, old_scandatetime, timeout_s=5.):
        """Wacht (met een korte polling-lus, zie ook LET OP hierboven bij _run_volume_3d_export) tot
        self.pb.data_attr['scandatetime'] voor paneel j verandert t.o.v. old_scandatetime. Geeft de nieuwe
        scandatetime terug, of None als er binnen timeout_s niets veranderde (bv. einde van de beschikbare
        data bereikt)."""
        start = pytime.time()
        while pytime.time()-start < timeout_s:
            QApplication.processEvents()
            current = self.pb.data_attr['scandatetime'].get(j)
            if current is not None and current != old_scandatetime:
                return current
            pytime.sleep(0.02)
        return None

    def _encode_frames_to_mp4(self, frames, fps, filepath):
        """Zet een lijst RGBA-numpy-arrays (zoals geleverd door vispy's canvas.render(), zie ook
        save_3d_screenshot hierboven) om naar 1 MP4-bestand. Gebruikt PyAV op dezelfde manier als
        self.create_ani(ext='mp4') hierboven -- dus geen nieuwe afhankelijkheid t.o.v. ffmpeg/imageio-ffmpeg
        nodig, want dat is al aanwezig en beproefd in dit bestand."""
        if not frames:
            raise ValueError('Geen beeldjes om te exporteren.')
        height, width = frames[0].shape[:2]
        # H264 vereist een even breedte/hoogte (zelfde reden als bij de bestaande 2D savefig, zie hieronder).
        width_even, height_even = int(np.ceil(width/2)*2), int(np.ceil(height/2)*2)
        fps_int = max(1, int(round(fps)))
        container = av.open(filepath, mode='w')
        stream = container.add_stream('libx264', rate=fps_int, width=width_even, height=height_even,
                                       pix_fmt='yuv420p', options={'crf': '18'})
        stream.codec_context.time_base = Fraction(1, fps_int)
        try:
            for i, arr in enumerate(frames):
                img = Image.fromarray(arr[:, :, :3])  # canvas.render() geeft RGBA; alpha niet nodig voor video.
                if (width_even, height_even) != (width, height):
                    padded = Image.new('RGB', (width_even, height_even))
                    padded.paste(img, (0, 0))
                    img = padded
                frame = av.VideoFrame.from_image(img)
                frame.pts = i
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():  # Flush stream, zelfde patroon als in create_ani.
                container.mux(packet)
        finally:
            container.close()

    def savefig(self, select_filename=True):
        #select_filename should be False when self.continue_savefig=True, after initially the directory and filename format have been chosen.
        
        if self.creating_animation:
            case_id = f'{self.get_case_index():04d}_' if self.current_case_shown() else ''
            dataspecs = case_id+''.join([self.dsg.get_dataspecs_string_panel(j) for j in self.pb.panellist])
            # Also check whether the file that contains an already saved frame still exists, since frames might have been manually
            # removed from gv.animation_frames_directory
            if hasattr(self, 'ani_frames_specs') and dataspecs in self.ani_frames_specs and os.path.exists(self.ani_frames_specs[dataspecs]):
                return
        
        img_arr1 = gloo.read_pixels(alpha=False)
        width, height = self.plotwidget.width(), self.plotwidget.height()
        if self.savefig_include_menubar:
            x, y = self.plotwidget.pos().x(), self.plotwidget.pos().y()
            full_width, full_height = self.width(), self.height()
            
            img = self.grab(QRect(QPoint(0, 0), QSize(full_width, y)))
            im = img.toImage()
            channels_count = 4
            s = im.bits().asstring(img.width() * img.height() * channels_count)
            img_arr2 = np.frombuffer(s, dtype=np.uint8).reshape((img.height(), img.width(), channels_count)).copy() # copy because otherwise read-only
            img_arr2[:,:,:3] = img_arr2[:,:,:3][:,:,::-1]
            
            # Make image width and height integer multiples of 2, since H264 video format (MP4) requires that
            im_width, im_height = int(np.ceil(full_width/2)*2), int(np.ceil(full_height/2)*2)
            img_arr = np.full((im_height, im_width, 3), img_arr2[0,0,0], dtype='uint8')
            img_arr[y:y+height,x:x+width] = img_arr1
            img_arr[:y,:] = img_arr2[:,:,:3]
        else:
            # Make image width and height integer multiples of 2, since H264 video format (MP4) requires that
            im_width, im_height = int(np.ceil(width/2)*2), int(np.ceil(height/2)*2)
            img_arr = np.full((im_height, im_width, 3), img_arr1[0,0,0], dtype='uint8')
            img_arr[:height, :width] = img_arr1
            
        im = Image.fromarray(img_arr)
                    
        if self.creating_animation:
            radar_dataset = self.crd.radar+('_'+self.crd.dataset if self.crd.radar in gv.radars_with_datasets else '')
            s = self.radardata_product_versions[radar_dataset]
            radar_dataset_subdataset = radar_dataset+(' '+s if s else '')
            datetimes = [self.pb.data_attr['scandatetime'][j] for j in self.pb.panellist]
            if self.starting_animation:
                self.ani_frame_number = 0
                self.ani_frames_specs = {}
                self.ani_frames_datetimes, self.ani_frames_datasets = [], []
                self.starting_animation = False

            self.ani_frame_number += 1
            self.ani_frames_datetimes += [datetimes]
            self.ani_frames_datasets += [case_id if case_id else radar_dataset_subdataset]
                
            filepath = opa(gv.animation_frames_directory+f'/frame{self.ani_frame_number}.gif')
            self.ani_frames_specs[dataspecs] = filepath
        else:
            if select_filename:
                try:
                    self.get_user_selected_filepath()
                except Exception as e:
                    print(e, 'get_user_filepath_savefig')
                    return
            
            savefig_filename = self.expand_filename()
            filepath = opa(self.savefig_dirname+'/'+savefig_filename)
        
        try:
            save_format=filepath[-3:]
            if save_format=='jpg':
                im=im.convert('RGB')
                im.save(filepath,quality=95,subsampling=0)
            elif save_format=='png':
                im=im.convert('RGB') #If this is not done, then the parts of the image that are transparent get a wrong color. 
                im.save(filepath,optimize=True,dpi=(72,72))
            elif save_format=='gif': 
                #I should make saving as GIF using dithering optional, by adding the possibility to set dithering on/off in the settings.
                #Converting to RGB before applying dithering is necessary, because otherwise it doesn't work.
                im=im.convert('RGB').convert('P', palette=Image.WEB, dither=Image.FLOYDSTEINBERG,colors=255)
                im.save(filepath)
        except Exception as e: print(e,'savefig'); pass
    
    def get_user_selected_filepath(self):
        text='Select a folder and filename (with extension, otherwise jpg) or file extension (without dot).'
        selected_dir_plus_filename_or_extension=str(QFileDialog.getSaveFileName(None,text,self.savefig_filename,options=QFileDialog.DontUseNativeDialog)[0])

        if selected_dir_plus_filename_or_extension:
            self.savefig_dirname=os.path.dirname(selected_dir_plus_filename_or_extension)
            basename=os.path.basename(selected_dir_plus_filename_or_extension)
            self.file_extension=basename[-3:]
            if not self.file_extension in ('gif','png','jpg'):
                self.file_extension='jpg'
                
            if basename in ('gif','png','jpg'):
                self.use_own_filename=False
            else:
                self.file_name=basename.replace('#', '*')
                if '.' in self.file_name:
                    self.file_name=self.file_name[:self.file_name.find('.')]
                self.use_own_filename=True
                
                self.file_number = None
                if '*' in self.file_name:
                    files_pattern = sorted([os.path.basename(j).split('.')[0] for j in glob.glob(self.savefig_dirname+'/'+self.file_name+'.'+self.file_extension)])
                    files_pattern = [j for j in files_pattern if j[len(self.file_name)-1:].isdigit()]
                    self.file_number = int(files_pattern[-1][len(self.file_name)-1:])+1 if len(files_pattern) > 0 else 1
                    self.file_name=self.file_name.replace('*', '')
                elif self.continue_savefig:
                    self.file_number = 1
                
            self.savefig_filename=selected_dir_plus_filename_or_extension
        else:
            self.continue_savefig=False
            raise Exception
                    
    def expand_filename(self):
        if self.use_own_filename:
            savefig_filename=self.file_name
            if not self.file_number is None:
                savefig_filename+=str(self.file_number)+'.'+self.file_extension
                self.file_number+=1
            else: 
                savefig_filename+='.'+self.file_extension
        else:
            def product_and_scan_string(panel,product,scan):
                string='u'*self.crd.using_unfilteredproduct[panel]
                string+=product+('('+self.crd.polarization[panel]+')')*self.crd.using_verticalpolarization[panel]
                if not product in gv.plain_products: 
                    string+=str(scan)
                return string
            
            duplicate_number = self.check_duplicates()

            savefig_filename = gv.radars_ascii_names[self.crd.radar].replace(' ','')+f'_{self.crd.dataset}'*(self.crd.radar in gv.radars_with_datasets)+'_'
            savefig_filename += self.crd.date[2:]+self.crd.time+'_'
            products_scans = ' '.join([product_and_scan_string(j,self.crd.products[j],self.crd.scans[j]) for j in self.pb.panellist])
            savefig_filename += ''.join(list(self.compress_choice_text_and_split_rows(products_scans).values())).replace(' ', '')
            savefig_filename += '_vwp'*self.show_vwp+f'_{duplicate_number}'*(duplicate_number > 0)+'.'+self.file_extension
        return savefig_filename
    
    def check_duplicates(self):
        duplicate_number = 0
        for j in self.pb.panellist:
            key = self.crd.products[j] if self.crd.products[j] in gv.plain_products else self.crd.scans[j]
            if len(self.dsg.scannumbers_all['z'][key]) > 1:
                duplicate_number = max([duplicate_number, self.dsg.scannumbers_forduplicates[key]+1])
        return duplicate_number
    
    
    def change_show_vwp(self):
        self.show_vwp = not self.show_vwp
        self.pb.on_resize()
        if self.show_vwp:
            self.vwp.set_newdata()
            self.pb.set_draw_action('plotting_vwp')
        self.pb.update()
        
    
    
    def extra(self):
        self.extra=QTabWidget()
        self.extra.setWindowTitle('NLradar extra')
        self.extrafiles=QWidget()
        self.extra.addTab(self.extrafiles,'Files')
        self.extra_tabfiles()
        self.extra.resize(self.extra.sizeHint())
        self.extra.show()        
        
    def extra_tabfiles(self):
        layout=QFormLayout()
        self.movefilesw=QPushButton('Change directory structure, by moving files from one directory structure to another one', autoDefault=True)
        self.remove_volumeattributesw=QPushButton('Remove saved volume attributes for particular radars and datasets', autoDefault=True)
        self.movefilesw.clicked.connect(self.movefiles_select)
        self.remove_volumeattributesw.clicked.connect(self.remove_volume_attributes_select)
        layout.addWidget(self.movefilesw)
        layout.addWidget(self.remove_volumeattributesw)
        self.extrafiles.setLayout(layout)
        
        
    def remove_volume_attributes_select(self):
        self.removeattributes=QWidget()
        
        hbox_datetimes=QHBoxLayout()
        self.removeattributes_startdatew=QLineEdit(self.crd.selected_date)
        self.removeattributes_startdatew.setToolTip('Start date (YYYYMMDD)')
        self.removeattributes_starttimew=QLineEdit('0000')
        self.removeattributes_starttimew.setToolTip('Start time (HHMM)')
        self.removeattributes_enddatew=QLineEdit(self.crd.selected_date)
        self.removeattributes_enddatew.setToolTip('End date (YYYYMMDD)')
        self.removeattributes_endtimew=QLineEdit('2359')
        self.removeattributes_endtimew.setToolTip('End time (HHMM)')
        
        hbox_datetimes.addWidget(self.removeattributes_startdatew); hbox_datetimes.addWidget(self.removeattributes_starttimew)
        hbox_datetimes.addWidget(self.removeattributes_enddatew); hbox_datetimes.addWidget(self.removeattributes_endtimew)
                
        vbox_sources=QVBoxLayout()
        vbox_sources.addWidget(QLabel('Data sources from which to include radars. No selection implies selecting current radar.'))
        self.removeattributes_sourcesw = {}
        hbox = QHBoxLayout()
        n = 5
        for i,j in enumerate(gv.data_sources_all):
            self.removeattributes_sourcesw[j] = QCheckBox(j)
            self.removeattributes_sourcesw[j].setTristate(False)
            hbox.addWidget(self.removeattributes_sourcesw[j])
            if (i+1) % n == 0:
                vbox_sources.addLayout(hbox)
                hbox = QHBoxLayout()
        if len(gv.data_sources_all) % n != 0:
            vbox_sources.addLayout(hbox)
                     
        layout=QVBoxLayout()
        layout.addWidget(QLabel('Remove volume attributes for a particular radar and dataset for the selected date and time range. These volume attributes include e.g. the scanangles for all'))
        layout.addWidget(QLabel('scans in the volume, and multiple other attributes. They are saved to a file after determining them once, to increase the speed of reading data.'))
        layout.addWidget(QLabel('It could however occur that the wrong attributes are saved for particular dates and times, e.g. because data from different datasets is mixed up.'))
        layout.addWidget(QLabel('If this occurs, then you can here remove the wrong attributes, without having to remove the complete file (which is NLradar/Generated_files/attribute_IDs.pkl).'))
        layout.addLayout(hbox_datetimes)
        
        layout.addLayout(vbox_sources) 
        
        self.removeattributes_removew=QPushButton('Remove', autoDefault=True)
        self.removeattributes_infow=QLineEdit(); self.removeattributes_infow.setReadOnly(True)
        layout.addWidget(self.removeattributes_removew)
        layout.addWidget(self.removeattributes_infow)
        
        self.removeattributes_startdatew.editingFinished.connect(self.update_removeattributes_enddate)
        self.removeattributes_removew.clicked.connect(self.remove_attributes)
        
        self.removeattributes.setLayout(layout)
        self.removeattributes.resize(self.removeattributes.sizeHint())
        self.removeattributes.show()
                        
    def update_removeattributes_enddate(self):
        #Doing this speeds up the selection part when moving files for only one day.
        self.removeattributes_enddatew.setText(self.removeattributes_startdatew.text())
        
    def remove_attributes(self):
        self.update_removeattributes_infow('','black')
        
        startdate=self.removeattributes_startdatew.text(); starttime=self.removeattributes_starttimew.text()
        enddate=self.removeattributes_enddatew.text(); endtime=self.removeattributes_endtimew.text()
        if not ft.correct_datetimeinput(startdate,starttime) or not ft.correct_datetimeinput(enddate,endtime): 
            self.update_removeattributes_infow('Incorrect dates and/or times','red')
            return
        startdatetime=startdate+starttime; enddatetime=enddate+endtime
       
        selected_sources = [i for i,j in self.removeattributes_sourcesw.items() if j.checkState() == 2]
        if selected_sources:
            selected_radars = [j for j in gv.radars_all if gv.data_sources[j] in selected_sources]
        else:
            selected_radars = [self.crd.selected_radar]

        for i in selected_radars:
            for j in ('Z', 'V'):
                key = i+('_'+j)*(i in gv.radars_with_datasets)
                for sub in self.dsg.attributes_IDs[key].copy():
                    for date in self.dsg.attributes_IDs[key][sub].copy():
                        if int(startdate) <= int(date) <= int(enddate):
                            for time in self.dsg.attributes_IDs[key][sub][date].copy():
                                if int(startdatetime) <= int(date+time) <= int(enddatetime):
                                    del self.dsg.attributes_IDs[key][sub][date][time]
                        if len(self.dsg.attributes_IDs[key][sub][date]) == 0:
                            del self.dsg.attributes_IDs[key][sub][date]
        
        #Also update the file that contains attributes_IDs
        with open(self.dsg.attributes_IDs_filename,'wb') as f:
            pickle.dump(self.dsg.attributes_IDs,f)
        
        self.time_last_removal_volumeattributes = pytime.time()
        self.update_removeattributes_infow('Done','green')
                                
    def update_removeattributes_infow(self,text,color):
        self.removeattributes_infow.setStyleSheet('QLineEdit {color:'+color+'}')
        self.removeattributes_infow.setText(text); self.removeattributes_infow.repaint()
                    
       
    def movefiles_select(self):
        self.movefiles=QWidget()
               
        dirs_layout=QFormLayout()                                     
        self.movefiles_oldstructurew=QLineEdit(self.movefiles_parameters['oldstructure'])
        self.movefiles_oldstructurew.setToolTip('Specify the old directory structure for the data that you want to move')
        self.movefiles_newstructurew=QLineEdit(self.movefiles_parameters['newstructure'])
        self.movefiles_newstructurew.setToolTip('Specify the new directory structure for the data that you want to move')
        dirs_layout.addRow(QLabel('From'),self.movefiles_oldstructurew)
        dirs_layout.addRow(QLabel('To'),self.movefiles_newstructurew)   
                
        hbox_datetimes=QHBoxLayout()
        self.movefiles_startdatew=QLineEdit(self.movefiles_parameters['startdate'])
        self.movefiles_startdatew.setToolTip('Start date (YYYYMMDD)/Emtpy for all files')
        self.movefiles_starttimew=QLineEdit(self.movefiles_parameters['starttime'])
        self.movefiles_starttimew.setToolTip('Start time (HHMM)/Emtpy for all files')
        self.movefiles_enddatew=QLineEdit(self.movefiles_parameters['enddate'])
        self.movefiles_enddatew.setToolTip('End date (YYYYMMDD)/Emtpy for all files')
        self.movefiles_endtimew=QLineEdit(self.movefiles_parameters['endtime'])
        self.movefiles_endtimew.setToolTip('End time (HHMM)/Emtpy for all files')
        
        hbox_datetimes.addWidget(self.movefiles_startdatew); hbox_datetimes.addWidget(self.movefiles_starttimew)
        hbox_datetimes.addWidget(self.movefiles_enddatew); hbox_datetimes.addWidget(self.movefiles_endtimew)
                        
        vbox_sources=QVBoxLayout()
        vbox_sources.addWidget(QLabel('Data sources from which to include radars. No selection implies selecting current radar.'))
        self.movefiles_sourcesw = {}
        hbox = QHBoxLayout()
        n = 5
        for i,j in enumerate(gv.data_sources_all):
            self.movefiles_sourcesw[j] = QCheckBox(j)
            self.movefiles_sourcesw[j].setTristate(False)
            hbox.addWidget(self.movefiles_sourcesw[j])
            if (i+1) % n == 0:
                vbox_sources.addLayout(hbox)
                hbox = QHBoxLayout()
        if len(gv.data_sources_all) % n != 0:
            vbox_sources.addLayout(hbox)
                    
        self.movefiles_removefilesinpreviousdirsw=QCheckBox('Remove files from Current folder'); self.movefiles_removefilesinpreviousdirsw.setTristate(False)
        self.movefiles_movew=QPushButton('Move files', autoDefault=True)
        self.movefiles_stopw=QPushButton('Stop', autoDefault=True)
        self.movefiles_undow=QPushButton('Undo moving files', autoDefault=True); self.movefiles_undow.setEnabled(False)
        self.movefiles_infow=QLineEdit(); self.movefiles_infow.setReadOnly(True)
                
        layout=QVBoxLayout()
        layout.addWidget(QLabel('Change the directory structure for a series of files. First select the old (current) base directory and directory structure, and then select the new ones.'))
        layout.addWidget(QLabel('Then select the date and time range for which you want to move data (leave empty if you want to move all files), and the radars for which you want to do this.'))
        layout.addLayout(dirs_layout)
        layout.addLayout(hbox_datetimes)
        
        layout.addLayout(vbox_sources)
                
        layout.addWidget(self.movefiles_removefilesinpreviousdirsw)
        layout.addWidget(self.movefiles_movew)
        layout.addWidget(self.movefiles_stopw)
        layout.addWidget(self.movefiles_undow)
        layout.addWidget(self.movefiles_infow)
        self.movefiles.setLayout(layout)     
        
        self.movefiles_startdatew.editingFinished.connect(self.update_movefiles_enddate)
        for j in (self.movefiles_startdatew,self.movefiles_starttimew,self.movefiles_enddatew,self.movefiles_endtimew,self.movefiles_oldstructurew,self.movefiles_newstructurew):
            j.editingFinished.connect(self.change_movefiles_parameters)
        self.movefiles_movew.clicked.connect(self.start_movefiles)
        self.movefiles_stopw.clicked.connect(self.stop_movefiles)
        self.movefiles_undow.clicked.connect(self.undo_movefiles)
        
        #Reset self.move_filenames and self.move_datetimes, which is required to let undoing the movement of files work correctly.
        self.move_filenames={}; self.move_datetimes={}
        
        self.movefiles.resize(self.movefiles.sizeHint())
        self.movefiles.show()
        
    def change_movefiles_parameters(self):
        for j in self.movefiles_parameters:
            exec('self.movefiles_parameters[j]=self.movefiles_'+j+'w.text()')        
        
    def update_movefiles_enddate(self):
        #Doing this speeds up the selection part when moving files for only one day.
        self.movefiles_enddatew.setText(self.movefiles_startdatew.text())
    """
    Before starting to undo the movement of files, it is required to first press the stop button, that stops the previous action (movement
    of files).
    """
    def start_movefiles(self):
        self.movefiles_movew.setEnabled(False); self.movefiles_undow.setEnabled(False)
        self.movefiles_stopw.setEnabled(True)
        self.move_files()
    def stop_movefiles(self,set_stop_movingfiles=True):
        self.movefiles_movew.setEnabled(True); self.movefiles_undow.setEnabled(True)
        self.movefiles_stopw.setEnabled(False)
        self.stop_movingfiles=True
    def undo_movefiles(self):
        self.movefiles_movew.setEnabled(False); self.movefiles_undow.setEnabled(False)
        self.movefiles_stopw.setEnabled(True)
        self.move_files(undo=True)
    def update_movefiles_infow(self,text,color):
        self.movefiles_infow.setStyleSheet('QLineEdit {color:'+color+'}')
        self.movefiles_infow.setText(text); self.movefiles_infow.repaint()
        if color=='red': #Enable/disable clicking at particular widgets
            self.stop_movefiles()
    def move_files(self,undo=False):
        self.update_movefiles_infow('','black')
        self.stop_movingfiles=False
                
        old_dir=self.movefiles_oldstructurew.text()
        old_dir=old_dir.replace('\\','/') #Replace backslashes by forward slashes, as this the rest of the code is built for dealing with
        #forward slashes.

        if not bg.check_correctness_dir_string(old_dir):
            self.update_movefiles_infow('Old directory structure is incorrect','red')     
        new_dir=self.movefiles_newstructurew.text()
        new_dir=new_dir.replace('\\','/') #Replace backslashes by forward slashes, as this the rest of the code is built for dealing with
        #forward slashes.
        if not bg.check_correctness_dir_string(new_dir):
            self.update_movefiles_infow('New directory structure is incorrect','red')     
        
        startdate=self.movefiles_startdatew.text(); starttime=self.movefiles_starttimew.text()
        enddate=self.movefiles_enddatew.text(); endtime=self.movefiles_endtimew.text()
        if not all([startdate=='',starttime=='',enddate=='',endtime=='']) and (
        not ft.correct_datetimeinput(startdate,starttime) or not ft.correct_datetimeinput(enddate,endtime)): 
            self.update_movefiles_infow('Incorrect dates and/or times','red')
            return
        startdatetime=startdate+starttime; enddatetime=enddate+endtime
        if startdatetime=='':
            #In this case all dates and times are included.
            startdatetime=None; enddatetime=None
        
        selected_sources = [i for i,j in self.movefiles_sourcesw.items() if j.checkState() == 2]
        if selected_sources:
            selected_radars = [j for j in gv.radars_all if gv.data_sources[j] in selected_sources]
        else:
            selected_radars = [self.crd.selected_radar]
              
        if not undo:
            removefiles_initialfolder=True if self.movefiles_removefilesinpreviousdirsw.checkState()==2 else False
        else:
            #Always remove files from the Archived folder when undoing the movement of files.
            removefiles_initialfolder=True
            
        old_directories=[]; new_directories=[]
        for i in selected_radars:
            if self.stop_movingfiles: return
                        
            if not undo:
                self.completely_selected_directories,self.move_filenames[i],self.move_datetimes[i]=self.dsg.get_filenames_and_datetimes_in_datetime_range(i,dir_string=old_dir,startdatetime=startdatetime,enddatetime=enddatetime,return_completely_selected_directories=True)
            else:
                if not i in self.move_filenames: 
                    #In this case there are no files to set back.
                    continue  
              
            old_directories, new_directories=self._move_files(self.completely_selected_directories,self.move_filenames[i],self.move_datetimes[i],old_directories,new_directories,removefiles_initialfolder,undo,i,old_dir=old_dir,new_dir=new_dir)
            if self.stop_movingfiles: return

        if self.movefiles_infow.text()=='':
            self.update_movefiles_infow('No files found','red')
             
        #Remove empty directories
        directories=np.unique(old_directories if not undo else new_directories)
        for j in directories:
            if os.path.exists(j) and bg.check_dir_empty(j):
                os.removedirs(j)
        self.movefiles_movew.setEnabled(True); self.movefiles_undow.setEnabled(True)
        self.movefiles_stopw.setEnabled(False)
        
    def _move_files(self,completely_selected_directories,filenames,datetimes,old_directories,new_directories,removefiles_initialfolder,undo,radar,dataset=None,old_dir=None,new_dir=None):
        #old_dir and new_dir are only used when move_type=='movefiles'.
        moved_directories=[]
        for k in range(0,len(filenames)):
            if self.stop_movingfiles: 
                return None,None
            
            date=str(datetimes[k])[:8]; time=str(datetimes[k])[-4:]
            
            old_directory=self.dsg.get_directory(date,time,radar,dir_string=old_dir)
            new_directory=self.dsg.get_directory(date,time,radar,dir_string=new_dir)

            if not old_directory in old_directories:
                old_directories.append(old_directory)
            if not new_directory in new_directories:
                new_directories.append(new_directory)
            
            if removefiles_initialfolder and old_directory in completely_selected_directories and (
            os.path.dirname(old_directory)==os.path.dirname(new_directory)):
                """In this case the complete directory is moved/renamed, because all files in the directory need to be replaced.
                This is much faster than renaming files individually.
                It is only done when only the name of the last subfolder is changed, because otherwise there is not much of an improvement
                in speed compared to moving files individually.
                """
                if old_directory in moved_directories:
                    continue
                else:
                    moving_text=radar+(' '+dataset if not dataset is None else '')+', '+date+time
                    moving_text+=', '+(old_directory+'->'+new_directory if not undo else new_directory+'->'+old_directory)
                    try:
                        if not undo: os.rename(old_directory,new_directory)
                        else: os.rename(new_directory,old_directory)
                        moved_directories.append(old_directory)
                        self.update_movefiles_infow(moving_text,'green')
                    except Exception as e:
                        self.update_movefiles_infow(str(e),'red')
            else:
                if undo and not os.path.exists(old_directory):
                    os.makedirs(old_directory)
                if not undo and not os.path.exists(new_directory):
                    os.makedirs(new_directory)

                if not undo:
                    old_path=opa(os.path.join(old_directory,filenames[k]))
                    new_path=opa(os.path.join(new_directory,filenames[k]))
                else:
                    #Reverse directories when undoing.
                    old_path=opa(os.path.join(new_directory,filenames[k]))
                    new_path=opa(os.path.join(old_directory,filenames[k]))
                                              
                moving_text=radar+(' '+dataset if not dataset is None else '')+', '+date+time
                moving_text+=', '+(old_directory+'->'+new_directory if not undo else new_directory+'->'+old_directory)
                try:
                    if removefiles_initialfolder:
                        if not os.path.exists(new_path):
                            shutil.move(old_path,new_path)
                        elif os.path.exists(old_path):
                            os.remove(old_path)
                    elif not os.path.exists(new_path):
                        shutil.copyfile(old_path,new_path)
                    self.update_movefiles_infow(moving_text,'green')
                except Exception as e:
                    if os.path.exists(new_path):
                        #In this case the error occurred while attempting to delete the old path, and this is not regarded as a failed movement.
                        self.update_movefiles_infow(moving_text,'green')
                    else:
                        self.update_movefiles_infow(str(e),'red')
                    
            QApplication.processEvents()
        return old_directories, new_directories


            
    def settings(self):
        self.settings=QTabWidget()
        self.settings.setWindowTitle('NLradar settings')
        self.settingsmain=QWidget(); self.settingsmap=QWidget(); self.settingsdownload=QWidget(); self.settingsdatastorage=QWidget(); self.settingscolortables=QWidget(); self.settingspolrgb=QWidget(); self.settingsalgorithms = QWidget(); self.settingsmiscellaneous=QWidget()
        self.settings.addTab(self.settingsmain,'Main')
        self.settings.addTab(self.settingsmap,'Map')
        self.settings.addTab(self.settingsdownload,'Download')
        self.settings.addTab(self.settingsdatastorage,'Data storage')
        self.settings.addTab(self.settingscolortables,'Color tables')
        self.settings.addTab(self.settingspolrgb,'PolRGB')
        self.settings.addTab(self.settingsalgorithms,'Algorithms')
        self.settings.addTab(self.settingsmiscellaneous,'Miscellaneous')
        self.settings_tabmain(); self.settings_tabmap(); self.settings_tabdownload(); self.settings_tabdatastorage(); self.settings_tabcolortables(); self.settings_tabpolrgb(); self.settings_tabalgorithms(); self.settings_tabmiscellaneous()
        self.settings.resize(self.settings.sizeHint())
        self.settings.show()
        
    def settings_tabmain(self):
        layout=QFormLayout()
        
        self.dimensions_mainw={}; self.fontsizes_mainw={}
        hboxes={'dimensions':{},'fontsizes':{}}
        for j in self.dimensions_main:
            self.dimensions_mainw[j]=QLineEdit(format(self.pb.scale_physicalsize(self.dimensions_main[j]), '.2f'))
            self.dimensions_mainw[j].editingFinished.connect(lambda j=j: self.change_wsizes(j))
            hboxes['dimensions'][j]=QHBoxLayout()
            hboxes['dimensions'][j].addWidget(self.dimensions_mainw[j]); hboxes['dimensions'][j].addStretch(30)       
        for j in self.fontsizes_main:
            self.fontsizes_mainw[j]=QLineEdit(format(self.pb.scale_pointsize(self.fontsizes_main[j]), '.1f'))
            self.fontsizes_mainw[j].editingFinished.connect(lambda j=j: self.change_fontsize(j))
            hboxes['fontsizes'][j]=QHBoxLayout()
            hboxes['fontsizes'][j].addWidget(self.fontsizes_mainw[j]); hboxes['fontsizes'][j].addStretch(30)       
            
        b_size=5
        hbox_bgcolor=QHBoxLayout(); hbox_panelbdscolor=QHBoxLayout()
        hboxes_colors=[hbox_bgcolor,hbox_panelbdscolor]
        for hbox in hboxes_colors:
            hbox.addStretch(0)      
            
        self.bgcolorw=QLineEdit()
        self.bgcolor_select=QPushButton('Select', autoDefault=True)
        hbox_bgcolor.addWidget(self.bgcolorw,b_size+5); hbox_bgcolor.addWidget(self.bgcolor_select,b_size+4)
        self.bgcolorw.setText(str(int(self.bgcolor[0]))+','+str(int(self.bgcolor[1]))+','+str(int(self.bgcolor[2])))
        
        self.panelbdscolorw=QLineEdit()
        self.panelbdscolor_select=QPushButton('Select', autoDefault=True)
        hbox_panelbdscolor.addWidget(self.panelbdscolorw,b_size+5); hbox_panelbdscolor.addWidget(self.panelbdscolor_select,b_size+4)
        self.panelbdscolorw.setText(str(int(self.panelbdscolor[0]))+','+str(int(self.panelbdscolor[1]))+','+str(int(self.panelbdscolor[2])))
        
        for hbox in hboxes_colors:
            hbox.addStretch(40)
        
        widgets=[[QLabel('Width color bar areas (cm)'),hboxes['dimensions']['width']],
                      [QLabel('Height title areas (cm)'),hboxes['dimensions']['height']],
                      [QLabel('Titles font size'),hboxes['fontsizes']['titles']],
                      [QLabel('Color bar labels font size'),hboxes['fontsizes']['cbars_labels']],
                      [QLabel('Color bar ticks font size'),hboxes['fontsizes']['cbars_ticks']],
                      [QLabel(''),QLabel('')],
                      [QLabel('Background color (RGB)'),hbox_bgcolor],
                      [QLabel('Color of panel borders (RGB)'),hbox_panelbdscolor]]
        for j in range(0,len(widgets)):
            layout.addRow(widgets[j][0],widgets[j][1])
            
        self.bgcolorw.editingFinished.connect(lambda: self.change_bgcolor('QLineEdit'))
        self.bgcolor_select.clicked.connect(lambda: self.change_bgcolor('QPushButton'))
        self.panelbdscolorw.editingFinished.connect(lambda: self.change_panelbdscolor('QLineEdit'))
        self.panelbdscolor_select.clicked.connect(lambda: self.change_panelbdscolor('QPushButton'))
                        
        self.settingsmain.setLayout(layout)
        
    def change_wsizes(self,dimension):
        input_size=self.dimensions_mainw[dimension].text()
        number=ft.to_number(input_size)
        if not number is None:
            self.dimensions_main[dimension] = number/self.pb.scale_physicalsize(1)
            self.pb.wdims[0 if dimension=='width' else 1] = self.dimensions_main[dimension]
            self.pb.on_resize()
            self.pb.update()
            
    def change_fontsize(self,visual):
        input_size=self.fontsizes_mainw[visual].text()
        number=ft.to_number(input_size)
        if not number is None:
            self.fontsizes_main[visual]=number/self.pb.scale_pointsize(1)
            self.pb.visuals[visual].font_size=number
            if visual in ('cbars_ticks', 'cbars_labels'): #cbars should be updated since the position of some of the cbar ticks and labels
                #is dependent on the font size.
                self.pb.set_cbars(resize=True) #set resize=True as otherwise no update takes place
            elif visual == 'titles': #Also here the position is font size dependent
                self.pb.set_titles()
            self.pb.update()
            
    def change_color(self,color_qlineedit,color_object,source,alpha=False):
        if source=='QLineEdit':
            inputcolor=color_qlineedit.text()
        else:
            colorwidget=QColorDialog()
            inputcolor=colorwidget.getColor(initial=QColor(*color_object[:4 if alpha else 3].astype(int)),
                                            options=QColorDialog.ShowAlphaChannel if alpha else QColorDialog.ColorDialogOptions())
        if (source=='QLineEdit' and ft.rgb(inputcolor,alpha=alpha)!=False) or (source=='QPushButton' and inputcolor.isValid()):
            if source=='QLineEdit':
                color_object=np.array(list(ft.rgb(inputcolor,alpha=alpha)))
            else:
                color_object=np.array(list(inputcolor.getRgb()))
                if not alpha: color_object=color_object[:3]
        return color_object                      
    
    def change_bgcolor(self,source):
        self.bgcolor = self.change_color(self.bgcolorw,self.bgcolor,source)
        self.bgcolorw.setText(str(int(self.bgcolor[0]))+','+str(int(self.bgcolor[1]))+','+str(int(self.bgcolor[2])))
        self.pb.visuals['background'].color=self.bgcolor/255.
        self.pb.update()    
        
    def change_panelbdscolor(self,source):
        self.panelbdscolor = self.change_color(self.panelbdscolorw,self.panelbdscolor,source)
        self.panelbdscolorw.setText(str(int(self.panelbdscolor[0]))+','+str(int(self.panelbdscolor[1]))+','+str(int(self.panelbdscolor[2])))
        self.pb.visuals['panel_borders'].set_data(color=self.panelbdscolor/255.)
        self.pb.update()        

    def settings_tabmap(self):
        map_layout=QFormLayout()
        hbox_bgmapcolor=QHBoxLayout(); hbox_mapvisibility=QHBoxLayout(); hbox_mapcolorfilter=QHBoxLayout()
        hbox_radardata_colorfilter=QHBoxLayout()
        hbox_maptiles_update_time = QHBoxLayout(); hbox_boxtitles=QHBoxLayout()
        hbox_basemap_source = QHBoxLayout(); hbox_basemap_source_maptiler_style = QHBoxLayout()
        hbox_basemap_source_maptiler_provider = QHBoxLayout()
        b_size=5
        hboxes_background=[hbox_bgmapcolor,hbox_mapvisibility,hbox_mapcolorfilter,hbox_radardata_colorfilter,hbox_maptiles_update_time,hbox_basemap_source,hbox_basemap_source_maptiler_style,hbox_basemap_source_maptiler_provider]
        for hbox in hboxes_background:
            hbox.addStretch(0)      
        
        self.bgmapcolorw=QLineEdit()
        self.bgmapcolor_select=QPushButton('Select', autoDefault=True)
        hbox_bgmapcolor.addWidget(self.bgmapcolorw,b_size+5); hbox_bgmapcolor.addWidget(self.bgmapcolor_select,b_size+4)
        self.bgmapcolorw.setText(str(int(self.bgmapcolor[0]))+','+str(int(self.bgmapcolor[1]))+','+str(int(self.bgmapcolor[2])))

        self.mapvis_false=QRadioButton('False'); self.mapvis_true=QRadioButton('True')
        self.mapvis_group=QButtonGroup(); self.mapvis_group.addButton(self.mapvis_false); self.mapvis_group.addButton(self.mapvis_true)
        hbox_mapvisibility.addWidget(self.mapvis_false,b_size); hbox_mapvisibility.addWidget(self.mapvis_true,b_size)
        self.mapvis_true.setChecked(True) if self.mapvisibility else self.mapvis_false.setChecked(True)
        
        self.mapcolorfilterw=QLineEdit(ft.list_to_string(self.mapcolorfilter))
        hbox_mapcolorfilter.addWidget(self.mapcolorfilterw)

        self.radardata_opacity_slider=QSlider(Qt.Horizontal)
        self.radardata_opacity_slider.setMinimum(0); self.radardata_opacity_slider.setMaximum(100)
        self.radardata_opacity_slider.setValue(int(round(self.radardata_colorfilter[3]*100)))
        self.radardata_opacity_slider.setToolTip("Opacity of the radar data itself. Lower this to let the basemap "
                                                  "(streets/place names) show through the radar echoes, uniformly "
                                                  "regardless of echo intensity.")
        self.radardata_opacity_label=QLabel(str(self.radardata_opacity_slider.value())+'%')
        self.radardata_opacity_label.setMinimumWidth(int(round(self.pb.scale_pixelsize(35))))
        hbox_radardata_colorfilter.addWidget(self.radardata_opacity_slider,b_size+8)
        hbox_radardata_colorfilter.addWidget(self.radardata_opacity_label,b_size)
        
        self.maptiles_update_timew = QLineEdit(str(self.maptiles_update_time))
        self.maptiles_update_timew.setToolTip('Map tiles are updated when the last occurrence of panning/zooming was this number of seconds ago.')
        hbox_maptiles_update_time.addWidget(self.maptiles_update_timew)

        self.basemap_source_localw=QRadioButton('Local (bundled tiles, works offline)')
        self.basemap_source_maptilerw=QRadioButton('MapTiler (live, scrollable map, requires API key + internet)')
        self.basemap_source_group=QButtonGroup(); self.basemap_source_group.addButton(self.basemap_source_localw); self.basemap_source_group.addButton(self.basemap_source_maptilerw)
        hbox_basemap_source.addWidget(self.basemap_source_localw,b_size+10); hbox_basemap_source.addWidget(self.basemap_source_maptilerw,b_size+10)
        self.basemap_source_maptilerw.setChecked(True) if self.basemap_source == 'MapTiler' else self.basemap_source_localw.setChecked(True)

        self.basemap_source_maptiler_stylew=QLineEdit(self.basemap_source_maptiler_style)
        self.basemap_source_maptiler_stylew.setToolTip("MapTiler map style/ID, e.g. 'dataviz-v4-dark'. Find this in your MapTiler dashboard under the map's 'Use vector style' URL: .../maps/<this part>/style.json")
        hbox_basemap_source_maptiler_style.addWidget(self.basemap_source_maptiler_stylew)

        self.basemap_source_maptiler_esriw=QRadioButton('Esri Dark Gray')
        self.basemap_source_maptiler_esristreetw=QRadioButton('Esri Street')
        self.basemap_source_maptiler_esrilightw=QRadioButton('Esri Light Gray')
        self.basemap_source_maptiler_esriimageryw=QRadioButton('Esri Satellite')
        self.basemap_source_maptiler_esritopow=QRadioButton('Esri Topo')
        self.basemap_source_maptiler_stadiaw=QRadioButton('Stadia Maps (requires API key)')
        self.basemap_source_maptiler_esriw.setToolTip('All Esri maps: free, no account or API key needed')
        self.basemap_source_maptiler_provider_group=QButtonGroup()
        self.basemap_source_maptiler_provider_group.addButton(self.basemap_source_maptiler_esriw)
        self.basemap_source_maptiler_provider_group.addButton(self.basemap_source_maptiler_esristreetw)
        self.basemap_source_maptiler_provider_group.addButton(self.basemap_source_maptiler_esrilightw)
        self.basemap_source_maptiler_provider_group.addButton(self.basemap_source_maptiler_esriimageryw)
        self.basemap_source_maptiler_provider_group.addButton(self.basemap_source_maptiler_esritopow)
        self.basemap_source_maptiler_provider_group.addButton(self.basemap_source_maptiler_stadiaw)
        hbox_basemap_source_maptiler_provider.addWidget(self.basemap_source_maptiler_esriw,b_size+10)
        hbox_basemap_source_maptiler_provider.addWidget(self.basemap_source_maptiler_esristreetw,b_size+10)
        hbox_basemap_source_maptiler_provider.addWidget(self.basemap_source_maptiler_esrilightw,b_size+10)
        hbox_basemap_source_maptiler_provider.addWidget(self.basemap_source_maptiler_esriimageryw,b_size+10)
        hbox_basemap_source_maptiler_provider.addWidget(self.basemap_source_maptiler_esritopow,b_size+10)
        hbox_basemap_source_maptiler_provider.addWidget(self.basemap_source_maptiler_stadiaw,b_size+10)
        if gv.frozen:
            # Gedeelde versie: geen kaarten met API-key
            self.basemap_source_maptilerw.setText('Live map (Esri: free, no account or API key, requires internet)')
            self.basemap_source_maptiler_stadiaw.setVisible(False)
        if self.basemap_source_maptiler_provider == 'stadia' and not gv.frozen:
            self.basemap_source_maptiler_stadiaw.setChecked(True)
        elif self.basemap_source_maptiler_provider == 'esri_street':
            self.basemap_source_maptiler_esristreetw.setChecked(True)
        elif self.basemap_source_maptiler_provider == 'esri_light':
            self.basemap_source_maptiler_esrilightw.setChecked(True)
        elif self.basemap_source_maptiler_provider == 'esri_imagery':
            self.basemap_source_maptiler_esriimageryw.setChecked(True)
        elif self.basemap_source_maptiler_provider == 'esri_topo':
            self.basemap_source_maptiler_esritopow.setChecked(True)
        else:
            self.basemap_source_maptiler_esriw.setChecked(True)
        
        for hbox in hboxes_background:
            hbox.addStretch(40)
        hbox_mapcolorfilter.addStretch(10)
        
        hbox_radars=QHBoxLayout()
        self.radars_selectcolorsw=QPushButton('Select size and colors radar markers', autoDefault=True)
        hbox_radars.addWidget(self.radars_selectcolorsw,b_size+5); hbox_radars.addStretch(50)
        
        hbox_boxtitles.addStretch(1); hbox_boxtitles.addWidget(QLabel('       Visibility'),b_size+3)
        hbox_boxtitles.addWidget(QLabel('                Colors (RGBA)'),b_size+3)
        hbox_boxtitles.addStretch(40)
        
        hboxes_lines={}; self.lines_widgets={}; self.lines_groups={}
        for j in self.lines_names:
            hboxes_lines[j]=QHBoxLayout()
            self.lines_widgets[j]=[QRadioButton('False'),QRadioButton('True'),QLineEdit(),QPushButton('Select', autoDefault=True)]
            self.lines_groups[j]=QButtonGroup()
            for i in range(0,2):
                self.lines_groups[j].addButton(self.lines_widgets[j][i])
                hboxes_lines[j].addWidget(self.lines_widgets[j][i])
                self.lines_widgets[j][1].setChecked(True) if j in self.lines_show else self.lines_widgets[j][0].setChecked(True)
            if j in self.lines_names:
                self.lines_widgets[j][2].setText(ft.list_to_string(self.lines_colors[j].astype(int)))

            hboxes_lines[j].addStretch(1)
            hboxes_lines[j].addWidget(self.lines_widgets[j][2],b_size+8)
            hboxes_lines[j].addWidget(self.lines_widgets[j][3],b_size+4)
            hboxes_lines[j].addStretch(30)
            
        self.lines_widthw=QLineEdit(str(self.lines_width))
        hbox_linewidth=QHBoxLayout(); hbox_linewidth.addWidget(self.lines_widthw); hbox_linewidth.addStretch(30)
        
        self.lines_antialiasw=QCheckBox(); self.lines_antialiasw.setTristate(False)
        self.lines_antialiasw.setCheckState(2 if self.lines_antialias else 0)
        hbox_antialias=QHBoxLayout(); hbox_antialias.addWidget(self.lines_antialiasw); hbox_antialias.addStretch(30)
            
        hbox_heightrings_derivedproducts=QHBoxLayout()
        self.show_heightrings_derivedproductsw={}
        for j in gv.plain_products:
            self.show_heightrings_derivedproductsw[j]=QCheckBox(j.upper())
            self.show_heightrings_derivedproductsw[j].setCheckState(2 if self.show_heightrings_derivedproducts[j] else 0)
            hbox_heightrings_derivedproducts.addWidget(self.show_heightrings_derivedproductsw[j])
        hbox_heightrings_derivedproducts.addStretch(30)
            
        self.showgridheightrings_panzoomw=QCheckBox(); self.showgridheightrings_panzoomw.setTristate(False)
        self.showgridheightrings_panzoomw.setCheckState(2 if self.showgridheightrings_panzoom else 0)
        self.showgridheightrings_panzoom_timew=QLineEdit(); self.showgridheightrings_panzoom_timew.setText(str(ft.rifdot0(self.showgridheightrings_panzoom_time)))
        
        self.gridheightrings_fontcolorw={}; self.gridheightrings_fontcolor_select={}
        for j in ('bottom','top'):
            self.gridheightrings_fontcolorw[j]=QLineEdit()
            self.gridheightrings_fontcolorw[j].setText(ft.list_to_string(self.gridheightrings_fontcolor[j].astype(int)))
            self.gridheightrings_fontcolor_select[j]=QPushButton('Select', autoDefault=True)
        self.gridheightrings_fontsizew=QLineEdit(format(self.pb.scale_pointsize(self.gridheightrings_fontsize), '.1f'))
        self.grid_showtextw=QCheckBox(); self.grid_showtextw.setTristate(False)
        self.grid_showtextw.setCheckState(2 if self.grid_showtext else 0)
        
        hbox_fontcolor=QHBoxLayout(); hbox_fontsize=QHBoxLayout(); hbox_showtextgrid=QHBoxLayout(); 
        
        hbox_showgridheightrings_panzoom=QHBoxLayout()
        hbox_showgridheightrings_panzoom_time=QHBoxLayout()
        hboxes=[hbox_fontcolor,hbox_fontsize,hbox_showtextgrid,hbox_showgridheightrings_panzoom,hbox_showgridheightrings_panzoom_time]
        for hbox in hboxes:
            hbox.addStretch(0)      
        hbox_fontcolor.addWidget(self.gridheightrings_fontcolorw['bottom']); hbox_fontcolor.addWidget(self.gridheightrings_fontcolor_select['bottom'])
        hbox_fontcolor.addWidget(self.gridheightrings_fontcolorw['top']); hbox_fontcolor.addWidget(self.gridheightrings_fontcolor_select['top'])
        hbox_fontsize.addWidget(self.gridheightrings_fontsizew)
        hbox_showtextgrid.addWidget(self.grid_showtextw)
        hbox_showgridheightrings_panzoom.addWidget(self.showgridheightrings_panzoomw)
        hbox_showgridheightrings_panzoom_time.addWidget(self.showgridheightrings_panzoom_timew)
        for hbox in hboxes:
            hbox.addStretch(10) if hbox==hbox_fontcolor else hbox.addStretch(30)
        
        map_widgets=[[QLabel('<b>Map</b>'),QLabel('')],
                     [QLabel('Background color (RGB)'),hbox_bgmapcolor],
                     [QLabel('Visibility'),hbox_mapvisibility],
                     [QLabel('Color filter (RGBA)'),hbox_mapcolorfilter],  
                     [QLabel('Radar data opacity'),hbox_radardata_colorfilter],
                     [QLabel('Map tiles update time (s)'), hbox_maptiles_update_time],

                     [QLabel('Radars'),hbox_radars],
                     [QLabel(''),QLabel('')],
                     [QLabel('<b>Lines</b>'),hbox_boxtitles],
                     [QLabel('Countries'),hboxes_lines['countries']],
                     [QLabel('Provinces'),hboxes_lines['provinces']],
                     [QLabel('Rivers'),hboxes_lines['rivers']],
                     [QLabel('Grid'),hboxes_lines['grid']],
                     [QLabel('Height rings'),hboxes_lines['heightrings']],
                     [QLabel('Line width'),hbox_linewidth],
                     [QLabel('Apply antialiasing to lines'),hbox_antialias],
                     [QLabel('Show height rings derived products'),hbox_heightrings_derivedproducts],
                     [QLabel('Show grid/height rings pan/zoom'),hbox_showgridheightrings_panzoom],
                     [QLabel('If not, update them after x seconds'),hbox_showgridheightrings_panzoom_time],
                     [QLabel('<b>Text grid and height rings</b>'),QLabel('')],
                     [QLabel('Font color bottom, top (RGB)'),hbox_fontcolor],
                     [QLabel('Font size'),hbox_fontsize],
                     [QLabel('Show grid coordinates'),hbox_showtextgrid]]
        for j in range(0,len(map_widgets)):
            map_layout.addRow(map_widgets[j][0],map_widgets[j][1])
        self.settingsmap.setLayout(map_layout)
        
        
        self.bgmapcolorw.editingFinished.connect(lambda: self.change_bgmapcolor('QLineEdit'))
        self.bgmapcolor_select.clicked.connect(lambda: self.change_bgmapcolor('QPushButton'))
        self.mapvis_false.toggled.connect(lambda: self.pb.change_mapvisibility(False))
        self.mapvis_true.toggled.connect(lambda: self.pb.change_mapvisibility(True))
        self.mapcolorfilterw.editingFinished.connect(self.pb.change_mapcolorfilter)
        self.radardata_opacity_slider.valueChanged.connect(self.pb.change_radardata_opacity)
        self.maptiles_update_timew.editingFinished.connect(self.change_maptiles_update_time)
        self.basemap_source_localw.toggled.connect(lambda checked: self.change_basemap_source('Local') if checked else None)
        self.basemap_source_maptilerw.toggled.connect(lambda checked: self.change_basemap_source('MapTiler') if checked else None)
        self.basemap_source_maptiler_stylew.editingFinished.connect(self.change_basemap_source_maptiler_style)
        self.basemap_source_maptiler_esriw.toggled.connect(lambda checked: self.change_basemap_source_maptiler_provider('esri') if checked else None)
        self.basemap_source_maptiler_esristreetw.toggled.connect(lambda checked: self.change_basemap_source_maptiler_provider('esri_street') if checked else None)
        self.basemap_source_maptiler_esrilightw.toggled.connect(lambda checked: self.change_basemap_source_maptiler_provider('esri_light') if checked else None)
        self.basemap_source_maptiler_esriimageryw.toggled.connect(lambda checked: self.change_basemap_source_maptiler_provider('esri_imagery') if checked else None)
        self.basemap_source_maptiler_esritopow.toggled.connect(lambda checked: self.change_basemap_source_maptiler_provider('esri_topo') if checked else None)
        self.basemap_source_maptiler_stadiaw.toggled.connect(lambda checked: self.change_basemap_source_maptiler_provider('stadia') if checked else None)
        self.radars_selectcolorsw.clicked.connect(self.select_properties_radar_markers)
        for line in self.lines_names:
            self.lines_widgets[line][0].toggled.connect(lambda state, line=line: self.change_line_state(line,False))
            self.lines_widgets[line][1].toggled.connect(lambda state, line=line: self.change_line_state(line,True))
            #No state for QLineEdit
            self.lines_widgets[line][2].editingFinished.connect(lambda line=line: self.change_line_state(line,'QLineEdit'))    
            self.lines_widgets[line][3].clicked.connect(lambda state, line=line: self.change_line_state(line,'QPushButton'))
        self.lines_widthw.editingFinished.connect(self.change_lines_width)
        self.lines_antialiasw.stateChanged.connect(self.change_lines_antialias)
        self.showgridheightrings_panzoomw.stateChanged.connect(self.change_showgridheightrings_panzoom)
        self.showgridheightrings_panzoom_timew.editingFinished.connect(self.change_showgridheightrings_panzoom_time)
        for j in ('bottom','top'):
            self.gridheightrings_fontcolorw[j].editingFinished.connect(lambda j=j: self.change_gridheightrings_fontcolor(j,'QLineEdit'))
            self.gridheightrings_fontcolor_select[j].clicked.connect(lambda state, j=j: self.change_gridheightrings_fontcolor(j,'QPushButton'))
        self.gridheightrings_fontsizew.editingFinished.connect(self.change_gridheightrings_fontsize)
        self.grid_showtextw.stateChanged.connect(self.change_grid_showtext)
        for j in gv.plain_products:
            self.show_heightrings_derivedproductsw[j].stateChanged.connect(lambda state, j=j: self.change_show_heightrings_derivedproducts(j))
        
        
    def select_properties_radar_markers(self):
        self.radar_markers_properties_widget=QWidget()
        self.radar_markers_properties_widget.setWindowTitle('Select colors and size radar markers')
        layout=QFormLayout()
        
        b_size=5
        
        self.radar_markersizew=QLineEdit(str(ft.round_float(self.pb.scale_pixelsize(self.radar_markersize))))
        layout.addRow(QLabel('Size'),self.radar_markersizew)
        types=('Default','Selected','Automatic download','Automatic download + selected')
        hboxes_types={}; self.radar_colorsw={}; self.radar_colors_selectw={}
        for j in types:
            hboxes_types[j]=QHBoxLayout()
            self.radar_colorsw[j]=QLineEdit()
            self.radar_colors_selectw[j]=QPushButton('Select', autoDefault=True)
            hboxes_types[j].addWidget(self.radar_colorsw[j],b_size+5); hboxes_types[j].addWidget(self.radar_colors_selectw[j],b_size+4)
            self.radar_colorsw[j].setText(ft.list_to_string(self.radar_colors[j]))
            layout.addRow(QLabel(j),hboxes_types[j])
            
        self.radar_markersizew.editingFinished.connect(self.change_radar_markersize)
        for j in types:
            self.radar_colorsw[j].editingFinished.connect(lambda j=j: self.change_radar_colors(j,'QLineEdit'))
            self.radar_colors_selectw[j].clicked.connect(lambda state,j=j: self.change_radar_colors(j,'QPushButton'))
            
        self.radar_markers_properties_widget.setLayout(layout)
        self.radar_markers_properties_widget.resize(self.radar_markers_properties_widget.sizeHint())
        self.radar_markers_properties_widget.show()
        
    def change_bgmapcolor(self,source):
        self.bgmapcolor = self.change_color(self.bgmapcolorw,self.bgmapcolor,source)
        self.bgmapcolorw.setText(str(int(self.bgmapcolor[0]))+','+str(int(self.bgmapcolor[1]))+','+str(int(self.bgmapcolor[2])))
        self.pb.visuals['background_map'].color=self.bgmapcolor/255.
        self.pb.update()
        
    def change_maptiles_update_time(self):
        inputtime=self.maptiles_update_timew.text()
        number=ft.to_number(inputtime)
        if not number is None and number>0: 
            self.maptiles_update_time = number

    def change_basemap_source(self, source):
        if source == self.basemap_source:
            return
        self.basemap_source = source
        # Debounce: reopening the Settings dialog can, in practice, fire this twice in quick succession for
        # what is really one user action (an artifact of how the radio buttons get re-initialized each time
        # Settings is opened). Rather than starting a full tile fetch immediately on every call, wait a brief
        # moment -- if another call comes in before that moment passes, only the LAST one actually proceeds.
        if hasattr(self, '_basemap_source_debounce_timer') and self._basemap_source_debounce_timer.isActive():
            self._basemap_source_debounce_timer.stop()
        self._basemap_source_debounce_timer = QTimer()
        self._basemap_source_debounce_timer.setSingleShot(True)
        self._basemap_source_debounce_timer.timeout.connect(lambda: self._apply_basemap_source_change(source))
        self._basemap_source_debounce_timer.start(150)

    def _apply_basemap_source_change(self, source):
        if source == 'MapTiler' and not gv.frozen and not self.api_keys.get('MapTiler', {}).get('maps', ''):
            self.set_textbar("MapTiler basemap selected, but no API key is set. Add one in Settings -> Download -> API keys.", 'red', 1)
        # Force a fresh tile fetch under the new source, rather than continuing to show whatever was cached
        # from the previous source.
        self.pb.get_active_mt().starting = True
        self.pb.update_map_tiles(separate_thread=False, draw_map=True)
        # set_maplineproperties now also depends on self.basemap_source (to hide NLradar's own country/
        # province/river lines when MapTiler -- which already renders its own borders -- is active), but
        # isn't otherwise re-evaluated by update_map_tiles, so refresh it explicitly here.
        self.pb.set_maplineproperties(self.pb.panellist)
        self.pb.update()

    def change_basemap_source_maptiler_style(self):
        input_style = self.basemap_source_maptiler_stylew.text().strip()
        if input_style:
            self.basemap_source_maptiler_style = input_style
            if self.basemap_source == 'MapTiler':
                self.pb.get_active_mt().starting = True
                self.pb.update_map_tiles(separate_thread=False, draw_map=True)
        else:
            self.basemap_source_maptiler_stylew.setText(self.basemap_source_maptiler_style)

    def change_basemap_source_maptiler_provider(self, provider):
        if provider == self.basemap_source_maptiler_provider:
            return
        self.basemap_source_maptiler_provider = provider
        if provider == 'stadia' and not self.api_keys.get('MapTiler', {}).get('maps', ''):
            self.set_textbar("Stadia Maps provider selected, but no API key is set. Add one in Settings -> Download -> API keys.", 'red', 1)
        if self.basemap_source == 'MapTiler':
            # Force a fresh tile fetch under the new provider, rather than continuing to show whatever was
            # cached from the previous one.
            self.pb.get_active_mt().starting = True
            self.pb.update_map_tiles(separate_thread=False, draw_map=True)
        
    def change_radar_markersize(self):
        inputsize=self.radar_markersizew.text()
        number=ft.to_number(inputsize)
        if not number is None and number>0: 
            self.radar_markersize=number/self.pb.scale_pixelsize(1)
        else: self.radar_markersizew.setText(str(ft.round_float(self.pb.scale_pixelsize(self.radar_markersize))))
        self.pb.set_radarmarkers_data()
        self.pb.update()
        
    def change_radar_colors(self,radartype,source):
        self.radar_colors[radartype]=self.change_color(self.radar_colorsw[radartype],self.radar_colors[radartype],source)
        self.radar_colorsw[radartype].setText(ft.list_to_string(self.radar_colors[radartype]))
        self.pb.set_radarmarkers_data()
        self.pb.update()
        
    def change_line_state(self,linetype,source):
        #Using isChecked() is required, because a signal is emitted when a radiobutton gets toggled, as well as when it gets untoggled
        if source==False and self.lines_widgets[linetype][0].isChecked(): 
            self.lines_show.remove(linetype)
            if linetype in self.ghtext_show: self.ghtext_show.remove(linetype)
        elif source==True and self.lines_widgets[linetype][1].isChecked(): 
            self.lines_show.append(linetype)
            if linetype in self.ghtext_names and not (linetype=='grid' and not self.grid_showtext): self.ghtext_show.append(linetype)            
        elif source in ('QLineEdit','QPushButton'):
            self.lines_colors[linetype]=self.change_color(self.lines_widgets[linetype][2],self.lines_colors[linetype],source,alpha=True)
            if linetype in self.lines_names:
                self.lines_widgets[linetype][2].setText(ft.list_to_string(self.lines_colors[linetype].astype(int)))
                self.pb.update_combined_lineproperties(range(10),changing_colors=True)
        
        if self.pb.firstplot_performed and linetype=='grid' and source==True: self.pb.set_grid()
        if self.pb.firstplot_performed and linetype=='heightrings' and source==True: self.pb.set_heightrings()
        if linetype in ('grid', 'heightrings'):
            self.pb.set_ghlineproperties(self.pb.panellist)
        else:
            self.pb.set_maplineproperties(self.pb.panellist)
        if linetype in self.ghtext_names:
            self.pb.set_ghtextproperties(self.pb.panellist)
        self.pb.update()
        
    def change_lines_width(self):
        input_value=self.lines_widthw.text()
        number=ft.to_number(input_value)
        if not number is None and number>0:
            self.lines_width=number
            self.pb.set_maplineproperties(self.pb.panellist)
            self.pb.update()
        self.lines_widthw.setText(str(self.lines_width))
        
    def change_lines_antialias(self):
        self.lines_antialias = True if self.lines_antialiasw.checkState()==2 else False
        #Drawing only needed for first panel, as the line visuals in the other panels are views of the first
        self.pb.visuals['map_lines'][0].antialias=self.lines_antialias
        self.pb.visuals['gh_lines'][0].antialias=self.lines_antialias
        self.pb.update()
            
    def change_show_heightrings_derivedproducts(self,product):
        self.show_heightrings_derivedproducts[product]=True if self.show_heightrings_derivedproductsw[product].checkState()==2 else False
        panellist_product=[j for j in self.pb.panellist if self.crd.products[j]==product]
        if len(panellist_product)>0:
            self.pb.set_heightrings(panellist_product)
            self.pb.set_ghlineproperties(panellist_product)
            self.pb.set_ghtextproperties(panellist_product)
            self.pb.update()
            
    def change_showgridheightrings_panzoom(self):
        self.showgridheightrings_panzoom=True if self.showgridheightrings_panzoomw.checkState()==2 else False
    def change_showgridheightrings_panzoom_time(self):
        input_showgridheightrings_panzoom_time=self.showgridheightrings_panzoom_timew.text()
        number=ft.to_number(input_showgridheightrings_panzoom_time)
        if not number is None and number>0.:
            self.showgridheightrings_panzoom_time=number
        else: self.showgridheightrings_panzoom_timew.setText(str(ft.rifdot0(self.showgridheightrings_panzoom_time)))
            
    def change_gridheightrings_fontcolor(self,bottom_top,source):
        self.gridheightrings_fontcolor[bottom_top]=self.change_color(self.gridheightrings_fontcolorw[bottom_top],self.gridheightrings_fontcolor[bottom_top],source)
        self.gridheightrings_fontcolorw[bottom_top].setText(ft.list_to_string(self.gridheightrings_fontcolor[bottom_top].astype(int)))
        if bottom_top == 'bottom': 
            self.pb.text_hor_bottom_colorfilter.filter = np.append(self.gridheightrings_fontcolor['bottom']/255., 1.)
        else:
            self.pb.text_hor_top_colorfilter.filter = np.append(self.gridheightrings_fontcolor['top']/255., 1.)
        self.pb.update()
    def change_gridheightrings_fontsize(self):
        inputsize=self.gridheightrings_fontsizew.text()
        number=ft.to_number(inputsize)
        if not number is None and number>0: 
            self.gridheightrings_fontsize=number/self.pb.scale_pointsize(1)
            for j in range(10):
                # Since 'text_hor2' and 'text_vert2' are visualviews, they don't need to have their font_size set too.
                self.pb.visuals['text_hor1'][j].font_size = number
                if j in self.pb.visuals['text_vert1']:
                    self.pb.visuals['text_vert1'][j].font_size = number
        else: self.gridheightrings_fontsizew.setText(str(ft.round_float(self.pb.scale_pointsize(self.gridheightrings_fontsize))))
        if 'grid' in self.lines_show:
            self.pb.set_grid() #Update the grid since the positions are dependent on the font size
            self.pb.set_ghtextproperties(self.pb.panellist)
            self.pb.update()
    def change_grid_showtext(self):
        self.grid_showtext=True if self.grid_showtextw.checkState()==2 else False
        if 'grid' in self.lines_show:
            self.ghtext_show.append('grid') if self.grid_showtext else self.ghtext_show.remove('grid') 
            self.pb.set_ghtextproperties(self.pb.panellist)
            self.pb.update()

        
        
    def settings_tabdownload(self):
        layout = QVBoxLayout()
        self.networktimeoutw=QLineEdit(); self.networktimeoutw.setText(str(ft.rifdot0(self.networktimeout)))
        self.minimum_downloadspeedw=QLineEdit(); self.minimum_downloadspeedw.setText(str(ft.rifdot0(self.minimum_downloadspeed)))
        self.networktimeoutw.editingFinished.connect(self.change_networktimeout)
        self.minimum_downloadspeedw.editingFinished.connect(self.change_minimum_downloadspeed)
        hbox = {}
        for widget in ('networktimeoutw', 'minimum_downloadspeedw'):
            hbox[widget] = QHBoxLayout()
            hbox[widget].addStretch(1); hbox[widget].addWidget(eval('self.'+widget)); hbox[widget].addStretch(50)
        formlayout=QFormLayout()
        formlayout.addRow(QLabel('Download timeout limit (seconds)'),hbox['networktimeoutw'])
        formlayout.addRow(QLabel('Minimum download speed (MB/minute)'),hbox['minimum_downloadspeedw'])
        layout.addLayout(formlayout)
            
        layout.addWidget(QLabel(''))
        self.api_keysw = copy.deepcopy(self.api_keys) # Is only done to get the same keys, values will be updated below
        labels = {'KNMI': "Set the API keys for the <b>KNMI</b> Data Platform. You can request these <A href='https://developer.dataplatform.knmi.nl/apis/'>here</a>.",
                  'DMI': "Set the API key for the <b>DMI</b> open data service. Follow the <A href='https://opendatadocs.dmi.govcloud.dk/en/Authentication'>following</a> guide to obtain a key for the radar data service.",
                  'Météo-France': "Set the API key for de <b>Météo-France</b> open data service. Subscribe to the radar data API <A href='https://portail-api.meteofrance.fr/web/en/api/DonneesPubliquesRadar'>here</a>, then click on 'configure the API' and generate a token.",
                  'MapTiler': "Set the API key for <b>MapTiler</b>, used for the optional live, scrollable basemap (Settings -> Map -> Basemap source). Get a free key <A href='https://www.maptiler.com/cloud/'>here</a>.",
                  'MeteoGate': "Set the API key for the EUMETNET <b>MeteoGate</b> Open Radar Data service, used for the Belgian KMI/skeyes/VMM radars (Wideumont, Jabbeke, Zaventem, Helchteren). Register and create a free key at the <A href='https://devportal.meteogate.eu/'>MeteoGate Developer Portal</a> (click 'Get API Key')."}
        labels_keys = {'KNMI': {'opendata': 'Open Data', 'sfcobs': 'Current 10 Minute Data KNMI Stations'},
                       'DMI': {'radardata': 'Radar Data'},
                       'Météo-France': {'radardata': 'Radar Data'},
                       'MapTiler': {'maps': 'Maps'},
                       'MeteoGate': {'radardata': 'Radar Data'}}
        for datasource in self.api_keys:
            label = QLabel(labels.get(datasource, f"Set the API key(s) for <b>{datasource}</b>."))
            label.setOpenExternalLinks(True)
            layout.addWidget(label)
            formlayout=QFormLayout()
            for key in self.api_keys[datasource]:
                self.api_keysw[datasource][key] = QLineEdit(self.api_keys[datasource][key])
                self.api_keysw[datasource][key].editingFinished.connect(lambda datasource=datasource, key=key: self.change_api_keys(datasource, key))
                hbox = QHBoxLayout()
                hbox.addWidget(self.api_keysw[datasource][key], 5); hbox.addStretch(2)
                formlayout.addRow(QLabel('API key '+labels_keys.get(datasource, {}).get(key, key)),hbox)
            layout.addLayout(formlayout)
            
        layout.addStretch(50)
        self.settingsdownload.setLayout(layout)
        
                
    def change_networktimeout(self):
        input_networktimeout=self.networktimeoutw.text()
        number=ft.to_number(input_networktimeout)
        if not number is None: 
            self.networktimeout=number if number>=0. else 0.
            self.networktimeoutw.setText(str(ft.rifdot0(self.networktimeout)))
        else: self.networktimeoutw.setText(str(ft.rifdot0(self.networktimeout)))
    def change_minimum_downloadspeed(self):
        input_minimum_downloadspeedw=self.minimum_downloadspeedw.text()
        number=ft.to_number(input_minimum_downloadspeedw)
        if not number is None: self.minimum_downloadspeed=number
        else: self.minimum_downloadspeedw.setText(str(ft.rifdot0(self.minimum_downloadspeed)))
    def change_api_keys(self, datasource, key):
        self.api_keys[datasource][key] = self.api_keysw[datasource][key].text().strip()
        

    def settings_tabdatastorage(self):
        datastorage_layout=QVBoxLayout()
        
        datastorage_text=["The program reads files from your disk, and here you can specify the directories in which your data is located. You can use both '/' and '\\' as the path separator.",
                          "If the directory names contain dates and times, then you can use variables like ${date} to specify this.",
                          "The following variables are allowed: ${date}, ${date+}, ${timeX}, ${timeX+}, ${datetimeX} and ${datetimeX+}. Here, dates and times have the formats YYYYMMDD and HHMM (and datetime the format",
                          "YYYYMMDDHHMM). Further, X should be a number, and represents the number of minutes to which a time will be floored to get the time in the name of the directory. When e.g. X=5 and",
                          "time=1442, then ${timeX} becomes 1440. When X=60, then ${timeX}=1400 etc. A condition for X is that dividing 1440 by X must give an integer.",
                          "Further, a variable with a plus indicates that the next date/time/datetime is used, where e.g.${timeX+} becomes ${timeX}+X minutes etc. Variables with a plus are not allowed to appear before the",
                          "corresponding variable without the plus.",
                          "Finally, time variables are not allowed to appear before the first date variable. Date and time variables are also not allowed to appear in more than 2 substrings of the directory, where a substring is",
                          "a part separated by '/' operators. And when a datetime variable is included, they may appear in only one substring.",
                          "You can also include ${radar} and ${radarID} variables. ${radar} represents the radar name without spaces, and ${radarID} is the identifier that is used in filenames to indicate which radar is used.",
                          "These identifiers can be found for the radars by mouse hovering over the labels in the left column of the window for selecting directories.",
                          "",
                          "More than one directory structure can be specified per radar and dataset, and different structure should be separated by a ';'. If the first part of the directory structures is repeated in all structures, then",
                          "you can put it in front, and separate it by a ';;' from the rest. This part is then put in front of all directory structures that are specified next.",
                          "",
                          "If the specified directory structures do not satisfy the requirements, then this is indicated by a red color.",
                          "You can see examples of the resulting complete directory path for the current date and time input by mouse hovering over the directory structures, if the input structure is correct.",
                          "You should preferentially not put files for multiple radars in the same directories. There should preferentially also be no empty directories."]
        for j in range(0,len(datastorage_text)):
            datastorage_layout.addWidget(QLabel(datastorage_text[j],font=self.help_font))  
        
        datastorage_widgets_layout=QFormLayout()
        
        self.radar_basedirw = QPushButton(self.radar_basedir, autoDefault=True)
        self.radar_basedir_defaultdirw=QPushButton('Default', autoDefault=True)
        hbox_basedir = QHBoxLayout(); 
        hbox_basedir.addWidget(self.radar_basedirw, 6); hbox_basedir.addWidget(self.radar_basedir_defaultdirw, 1)
        datastorage_widgets_layout.addRow(QLabel('Radar base directory'), hbox_basedir)
        self.radar_basedirw.clicked.connect(self.set_radar_basedir)
        self.radar_basedir_defaultdirw.clicked.connect(self.usedefault_basedir)
        
        self.dirselectw=QPushButton('Set', autoDefault=True)
        datastorage_widgets_layout.addRow(QLabel('Directory structures'),self.dirselectw)
        self.dirselectw.clicked.connect(self.selectdirs)

        self.derivedproducts_dirselectw=QPushButton(self.derivedproducts_dir, autoDefault=True)
        self.derivedproducts_defaultdirw=QPushButton('Default', autoDefault=True)
        hbox_derivedproducts=QHBoxLayout(); 
        hbox_derivedproducts.addWidget(self.derivedproducts_dirselectw,6); hbox_derivedproducts.addWidget(self.derivedproducts_defaultdirw,1)
        datastorage_widgets_layout.addRow(QLabel('Derived products'),hbox_derivedproducts)
        self.derivedproducts_dirselectw.clicked.connect(self.set_dir_derivedproducts)
        self.derivedproducts_defaultdirw.clicked.connect(self.usedefaultdir_derivedproducts)
        
        datastorage_layout.addLayout(datastorage_widgets_layout)
        self.settingsdatastorage.setLayout(datastorage_layout)
        
    def set_radar_basedir(self):
        _input=str(QFileDialog.getExistingDirectory(None, 'Select the folder:',self.radar_basedir))
        if _input:
            self.radar_basedir = _input
            self.radar_basedirw.setText(self.radar_basedir)
    def usedefault_basedir(self):
        self.radar_basedir = gv.default_basedir
        self.radar_basedirw.setText(self.radar_basedir)
                    
    def set_dir_derivedproducts(self):
        derivedproducts_dir_input=str(QFileDialog.getExistingDirectory(None, 'Select the folder:',self.derivedproducts_dir))
        if derivedproducts_dir_input!='':
            self.derivedproducts_dir=derivedproducts_dir_input    
            self.derivedproducts_dirselectw.setText(self.derivedproducts_dir)
    def usedefaultdir_derivedproducts(self):
        self.derivedproducts_dir=gv.derivedproducts_dir_Default
        self.derivedproducts_dirselectw.setText(self.derivedproducts_dir)
        
        
    def selectdirs(self):
        self.dirselect=QTabWidget()
        self.dirselect.setWindowTitle('Select directory structures')
        self.dirselecttabs={}
        self.radardirs_widgets, self.default_dirs_widgets = {}, {}
        self.additionaldirs_rds_widgets, self.additionaldirs_widgets = {}, {}
        for j in gv.data_sources_all:
            if not gv.radars.get(j): continue # kan gebeuren als alle radars van deze bron in radars_disabled staan
            self.dirselecttabs[j]=QWidget()
            self.dirselect.addTab(self.dirselecttabs[j],j)
            self.dirs_dstabs(j)
        self.dirselect.resize(self.dirselect.sizeHint())
        self.dirselect.show()
        
    def generate_dir_example_date_and_time(self):
        input_date=self.datew.text(); input_time=self.timew.text()
        if ft.correct_datetimeinput(input_date,input_time) and not input_time=='c':
            example_date=input_date
            example_time=input_time
        else:
            example_date=''.join(ft.get_ymdhm(pytime.time())[:3]); example_time=''.join(ft.get_ymdhm(pytime.time())[-2:])
        return example_date, example_time
        
    def get_radars_datasets_source(self, datasource):
        radars = gv.radars[datasource]
        rds = [] # All available radar_datasets for datasource
        for j in radars:
            rds += [f'{j}_Z', f'{j}_V'] if j in gv.radars_with_datasets else [j]
        return radars, rds

    def dirs_dstabs(self,datasource):
        layout=QVBoxLayout()
        grid_layout=QGridLayout()

        radars, rds = self.get_radars_datasets_source(datasource)
        datasets = np.unique(np.concatenate([['Z','V'] if j in gv.radars_with_datasets else [''] for j in radars]))
        
        layout.addWidget(QLabel('General directory structure used for all radars'+', per dataset'*(len(datasets) > 1)))

        radar = self.crd.selected_radar if self.crd.selected_radar in radars else radars[0]
        for i,dataset in enumerate(datasets):
            key = datasource+f'_{dataset}'*len(dataset)
            dir_string = self.radarsources_dirs[key]
            self.radardirs_widgets[key] = QLineEdit(dir_string)
            
            tool_tip = self.get_example_dir(dir_string, radars)
            self.radardirs_widgets[key].setToolTip(tool_tip)
            
            self.default_dirs_widgets[key] = QPushButton('Default', autoDefault=True)
            
            label = QLabel('All radars '+dataset)
            label.setToolTip(gv.radar_ids[radar])
            grid_layout.addWidget(label,i,0,1,1) #row, column, rowspan, colspan
            grid_layout.addWidget(self.radardirs_widgets[key],i,1,1,10)
            grid_layout.addWidget(self.default_dirs_widgets[key],i,11,1,1)
            self.default_dirs_widgets[key].clicked.connect(lambda state, key=key: self.use_default_dir(key))
            self.radardirs_widgets[key].editingFinished.connect(lambda source=datasource, key=key: self.change_radardata_dir(source, key))
        layout.addLayout(grid_layout)
        # The extra spaces are added to force the QTabWidget to have a certain width
        layout.addWidget(QLabel('Optional additional directory structures for individual radars'+' '*125))
            
        additional = [[i,j] for i,j in self.radardirs_additional.items() if self.dsg.split_radar_dataset(i)[0] in radars]
        
        grid_layout = QGridLayout()
        n = min(len(rds), 10)
        for i in range(n):
            key = datasource+f'_{i}'
            self.additionaldirs_rds_widgets[key] = QComboBox()
            self.additionaldirs_rds_widgets[key].addItems(['']+rds)
            self.additionaldirs_widgets[key] = QLineEdit()
            if i < len(additional):
                rd, additional_dir_string = additional[i]
                rd_radar = self.dsg.split_radar_dataset(rd)[0]
                
                self.additionaldirs_rds_widgets[key].setCurrentText(rd)
                self.additionaldirs_rds_widgets[key].setToolTip(gv.radar_ids[rd_radar])
                
                self.additionaldirs_widgets[key].setText(additional_dir_string)
                tool_tip = self.get_example_dir(self.radardata_dirs[rd], radars)
                self.additionaldirs_widgets[key].setToolTip(tool_tip)
                
            grid_layout.addWidget(self.additionaldirs_rds_widgets[key],i,0,1,1) #row, column, rowspan, colspan
            grid_layout.addWidget(self.additionaldirs_widgets[key],i,1,1,10)
            grid_layout.addWidget(QLabel(),i,11,1,1)
            
            self.additionaldirs_rds_widgets[key].currentTextChanged.connect(lambda text, source=datasource: self.change_individual_dirstrings(source))
            self.additionaldirs_widgets[key].editingFinished.connect(lambda source=datasource: self.change_individual_dirstrings(source))
        layout.addLayout(grid_layout)
        layout.addStretch(13-n-len(datasets))
              
        self.dirselecttabs[datasource].setLayout(layout)
        
    def get_example_dir(self, dir_string, radars):
        radar = self.crd.selected_radar if self.crd.selected_radar in radars else radars[0]
        example_date, example_time=self.generate_dir_example_date_and_time()
             
        example_dir_list = bg.dirstring_to_dirlist(dir_string)
        return ', '.join([bg.convert_dir_string_to_real_dir(d, self.radar_basedir, radar, example_date, example_time)
                              for j,d in enumerate(example_dir_list)])

    def change_radardata_dir(self, source, key): # key has format source_dataset
        # Replace backslashes by forward slashes, as the rest of the code is built for dealing with forward slashes.
        input_str=self.radardirs_widgets[key].text().replace('\\','/')
        input_dir_strings=bg.dirstring_to_dirlist(input_str)
            
        input_correct = all(bg.check_correctness_dir_string(j) for j in input_dir_strings)
        if input_correct:
            self.radardirs_widgets[key].setStyleSheet('QLineEdit {color:black}')
            self.radarsources_dirs[key] = input_str
            
            radars, rds = self.get_radars_datasets_source(source)
            self.update_radardirs_source(radars, rds)
            
            tool_tip = self.get_example_dir(input_str, radars)
            self.radardirs_widgets[key].setToolTip(tool_tip)
        else:
            self.radardirs_widgets[key].setStyleSheet('QLineEdit {color:red}')
            self.radardirs_widgets[key].setToolTip('')
            
    def update_radardirs_source(self, radars, rds):
        for rd in rds:
            radar, dataset = self.dsg.split_radar_dataset(rd)
            source = gv.data_sources[radar]
            key = source+f'_{dataset}'*len(dataset)
            self.radardata_dirs[rd] = self.radarsources_dirs[key]
            if rd in self.radardirs_additional:
                general_dirstring = self.radardata_dirs[rd].strip()
                self.radardata_dirs[rd] += '; '*(len(general_dirstring) and general_dirstring[-1] != ';')+self.radardirs_additional[rd]
            
            dir_strings = bg.dirstring_to_dirlist(self.radardata_dirs[rd])
            if self.radardata_dirs_indices[rd] >= len(dir_strings):
                self.radardata_dirs_indices[rd] = len(dir_strings)-1
                
    def use_default_dir(self, key):
        self.radarsources_dirs[key] = gv.radarsources_dirs_Default[key]
        self.radardirs_widgets[key].setText(self.radarsources_dirs[key])
        
    def change_individual_dirstrings(self, source):
        radars, rds = self.get_radars_datasets_source(source)        
        for rd in rds:
            if rd in self.radardirs_additional:
                del self.radardirs_additional[rd]

        for key in self.additionaldirs_rds_widgets:
            rd = self.additionaldirs_rds_widgets[key].currentText()
            rd_radar = self.dsg.split_radar_dataset(rd)[0]
            
            input_str = self.additionaldirs_widgets[key].text().replace('\\','/')
            input_dir_strings = bg.dirstring_to_dirlist(input_str)
                
            input_correct = all(bg.check_correctness_dir_string(j) for j in input_dir_strings)
            if rd and input_correct:
                self.additionaldirs_widgets[key].setStyleSheet('QLineEdit {color:black}')
                if input_str:
                    self.radardirs_additional[rd] = input_str
                
                self.update_radardirs_source(radars, rds)
                
                tool_tip = self.get_example_dir(self.radardata_dirs[rd], radars)
                self.additionaldirs_widgets[key].setToolTip(tool_tip)
                self.additionaldirs_rds_widgets[key].setToolTip(gv.radar_ids[rd_radar])
            else:
                self.additionaldirs_widgets[key].setStyleSheet('QLineEdit {color:'+('red' if not input_correct else 'black')+'}')
                self.additionaldirs_widgets[key].setToolTip('')
                self.additionaldirs_rds_widgets[key].setToolTip('')


    def settings_tabcolortables(self):
        colortables_layout=QVBoxLayout()
        colortables_form=QFormLayout()
        self.colortablesw={j:QPushButton(os.path.basename(self.colortables_dirs_filenames[j]), autoDefault=True) for j in gv.products_all}
        
        self.cmaps_minvaluesw={}; self.cmaps_maxvaluesw={}
        for j in gv.products_all:
            self.cmaps_minvaluesw[j]=QLineEdit(); self.cmaps_maxvaluesw[j]=QLineEdit()
            if not self.cmaps_minvalues[j]=='':
                #The scaling is necessary for velocities, to convert values in self.cmaps_minvalues (which have units of m/s) to the chosen unit.
                self.cmaps_minvaluesw[j].setText(str(ft.rifdot0(ft.r1dec(self.cmaps_minvalues[j]*self.pb.scale_factors[j]))))
            if not self.cmaps_maxvalues[j]=='':
                self.cmaps_maxvaluesw[j].setText(str(ft.rifdot0(ft.r1dec(self.cmaps_maxvalues[j]*self.pb.scale_factors[j]))))
            
        for j in gv.products_all:
            self.cmaps_minvaluesw[j].setToolTip('Minimum product value to display in the color map. Leave empty for use of color table minimum.')
            self.cmaps_maxvaluesw[j].setToolTip('Maximum product value to display in the color map. Leave empty for use of color table maximum.')
            hbox=QHBoxLayout()
            hbox.addWidget(self.colortablesw[j],20); hbox.addWidget(self.cmaps_minvaluesw[j],2); hbox.addWidget(self.cmaps_maxvaluesw[j],2)
            colortables_form.addRow(QLabel(gv.productnames_cmapstab[j]),hbox)
            self.colortablesw[j].clicked.connect(lambda state, j=j: self.change_colortables(j))
            self.cmaps_minvaluesw[j].editingFinished.connect(lambda j=j: self.change_cmaps_minvalues(j))
            self.cmaps_maxvaluesw[j].editingFinished.connect(lambda j=j: self.change_cmaps_maxvalues(j))
        
        self.set_default_colortables=QPushButton('Default', autoDefault=True)
        self.set_NWS_colortables=QPushButton('NWS', autoDefault=True)
        colortables_type_hbox=QHBoxLayout()
        colortables_type_hbox.addWidget(self.set_default_colortables); colortables_type_hbox.addWidget(self.set_NWS_colortables)
        colortables_layout.addLayout(colortables_form); colortables_layout.addLayout(colortables_type_hbox)
        colortables_layout.addStretch(100)
        self.settingscolortables.setLayout(colortables_layout)
        
        self.set_default_colortables.clicked.connect(lambda: self.change_colortables('Default'))
        self.set_NWS_colortables.clicked.connect(lambda: self.change_colortables('NWS'))
                
    def change_colortables(self,product):
        if product in ('Default','NWS'):
            new_colortables_dirs_filenames=gv.colortables_dirs_filenames_Default if product=='Default' else gv.colortables_dirs_filenames_NWS
            for j in new_colortables_dirs_filenames.keys():
                self.colortables_dirs_filenames[j]=new_colortables_dirs_filenames[j]
                self.colortablesw[j].setText(os.path.basename(self.colortables_dirs_filenames[j]))
        else:    
            colortables_dirs_filenames_input=str(QFileDialog.getOpenFileName(None, 'Select the color table:',os.path.dirname(self.colortables_dirs_filenames[product]),filter='*.csv')[0])
            if colortables_dirs_filenames_input!='': 
                self.colortables_dirs_filenames[product]=colortables_dirs_filenames_input
                self.colortablesw[product].setText(os.path.basename(self.colortables_dirs_filenames[product]))
        
        if self.pb.firstplot_performed:
            self.pb.set_newdata(self.pb.panellist)
        else:
            self.pb.set_cbars()
            self.pb.update()
                
    def change_cmaps_minvalues(self,product):
        input_text=self.cmaps_minvaluesw[product].text()
        input_value=ft.to_number(input_text)
        if input_text=='' or not input_value is None:
            self.cmaps_minvalues[product]='' if input_text=='' else input_value/self.pb.scale_factors[product]
            if self.pb.firstplot_performed:
                self.pb.set_newdata([j for j in self.pb.panellist if self.crd.products[j]==product])
        else:
            self.cmaps_minvaluesw[product].setText(str(self.cmaps_minvalues[product]))
                    
    def change_cmaps_maxvalues(self,product):
        input_text=self.cmaps_maxvaluesw[product].text()
        input_value=ft.to_number(input_text)
        if input_text=='' or not input_value is None:
            self.cmaps_maxvalues[product]='' if input_text=='' else input_value/self.pb.scale_factors[product]
            if self.pb.firstplot_performed:
                self.pb.set_newdata([j for j in self.pb.panellist if self.crd.products[j]==product])
        else:
            self.cmaps_maxvaluesw[product].setText(str(self.cmaps_maxvalues[product]))
        
        
    def settings_tabpolrgb(self):
        layout = QFormLayout()

        # ESSL-modus (bespoke checkbox, geen onderdeel van polrgb_field_specs hieronder: die velden zijn
        # allemaal getallen via change_polrgb_param, dit is een aan/uit-schakelaar met eigen handler -- zelfde
        # opzet als attenuation_correction_enabledw hierboven). AAN: vaste ESSL-poster-tabel (Van 't Veen/
        # Groenemeijer/Pucik, ECSS 2025 Utrecht) i.p.v. de doorlopende Z_MIN/Z_MAX/etc.-parameters hieronder,
        # die dan genegeerd worden (blijven wel zichtbaar/bewaard, voor als je teruggaat naar UIT).
        self.polrgb_essl_modew = QCheckBox('ESSL-tabel gebruiken i.p.v. onderstaande parameters')
        self.polrgb_essl_modew.setChecked(bool(self.polrgb_params.get('ESSL_MODE', False)))
        self.polrgb_essl_modew.setToolTip("Gebruikt de vaste RGB/alpha-opzoektabel uit de ESSL-poster "
            "(Van 't Veen, Groenemeijer & Pucik, ECSS 2025) i.p.v. de doorlopende passthrough hieronder: "
            "R=Z 30-60dBZ, G=CC 100-70% (omgekeerd), B=ZDR 0-4dB, alpha volgens de 11-punts Z-curve uit de "
            "poster. De parameters hieronder worden dan genegeerd (niet gewist).")
        self.polrgb_essl_modew.toggled.connect(self.change_polrgb_essl_mode)
        layout.addRow(QLabel(''), self.polrgb_essl_modew)

        polrgb_field_specs = [
            ('Z_MIN', 'Z scale minimum (dBZ)'),
            ('Z_MAX', 'Z scale maximum (dBZ)'),
            ('CC_MIN', 'CC scale minimum (%)'),
            ('CC_MAX', 'CC scale maximum (%)'),
            ('ZDR_MIN', 'ZDR scale minimum (dB)'),
            ('ZDR_MAX', 'ZDR scale maximum (dB)'),
            ('Z_FADE_LO', 'Visibility fade-in start (Z, dBZ)'),
            ('Z_FADE_HI', 'Visibility fade-in end (Z, dBZ)'),
            ('ALPHA_GAMMA', 'Visibility fade curve (gamma, <1 boosts weak echo)'),
            ('Z_GAMMA', 'Z color curve (gamma, >1 = lage dBZ donker/hoge dBZ snel rood, 1=lineair)'),
            ('CC_FALLBACK', 'CC fallback when missing but Z is valid (%)'),
            ('ZDR_FALLBACK', 'ZDR fallback when missing but Z is valid (dB)'),
        ]
        self.polrgb_paramsw = {}
        for key, label in polrgb_field_specs:
            widget = QLineEdit(str(ft.rifdot0(self.polrgb_params[key])))
            self.polrgb_paramsw[key] = widget
            hbox = QHBoxLayout(); hbox.addWidget(widget); hbox.addStretch(30)
            layout.addRow(QLabel(label), hbox)
            widget.editingFinished.connect(lambda key=key: self.change_polrgb_param(key))

        # ESSL-tabel zelf instelbaar (16 september 2026, op Eriks verzoek: Bram blijkt deze getallen per
        # publicatie/sessie te varieren -- zie posterbijlage vs. conferentie-abstract vs. live ESSL-viewer,
        # alle 3 met andere Z/CC/ZDR-grenzen). De 6 grenswaarden zijn gewone getallen -> hergebruikt de
        # generieke polrgb_field_specs/change_polrgb_param-mechaniek hierboven (dus ook automatisch
        # meegenomen in reset_polrgb_params via polrgb_paramsw). Werken ALLEEN als ESSL-modus aanstaat --
        # blijven verder gewoon bewaard/zichtbaar als dat niet zo is, net als de passthrough-parameters
        # hierboven wanneer ESSL wel aanstaat.
        layout.addRow(QLabel(''), QLabel('<b>ESSL-tabel</b> (alleen actief als de checkbox hierboven aanstaat)'))
        essl_range_specs = [
            ('ESSL_Z_MIN', 'ESSL: Z scale minimum (dBZ)'),
            ('ESSL_Z_MAX', 'ESSL: Z scale maximum (dBZ)'),
            ('ESSL_CC_MIN', 'ESSL: CC scale minimum (%)'),
            ('ESSL_CC_MAX', 'ESSL: CC scale maximum (%)'),
            ('ESSL_ZDR_MIN', 'ESSL: ZDR scale minimum (dB)'),
            ('ESSL_ZDR_MAX', 'ESSL: ZDR scale maximum (dB)'),
        ]
        for key, label in essl_range_specs:
            widget = QLineEdit(str(ft.rifdot0(self.polrgb_params[key])))
            self.polrgb_paramsw[key] = widget
            hbox = QHBoxLayout(); hbox.addWidget(widget); hbox.addStretch(30)
            layout.addRow(QLabel(label), hbox)
            widget.editingFinished.connect(lambda key=key: self.change_polrgb_param(key))

        # ESSL: 11-punts alpha(Z)-curve, bespoke tabel (2 kolommen Z/alpha x 11 rijen) i.p.v. de generieke
        # 1-veld-per-key-mechaniek hierboven: dit zijn 2 LIJSTEN (ESSL_ALPHA_Z/ESSL_ALPHA_V) i.p.v. losse
        # getallen, dus een eigen widget-structuur en handler (change_polrgb_essl_alpha) nodig. Z moet
        # strikt oplopend blijven (np.interp-eis, zie DataSource_General._essl_polrgb_channels) -- de
        # handler bewaart pas als alle 11 rijen geldige getallen zijn EN Z strikt oplopend is, anders wordt
        # het hele stel teruggezet naar de laatst geldige stand (voorkomt een half-geldige, verwarrende
        # tussentoestand in een 11-rijen-tabel).
        layout.addRow(QLabel(''), QLabel("ESSL: zichtbaarheid (alpha) vs. Z -- 11 punten, strikt oplopende Z"))
        essl_alpha_grid = QGridLayout()
        essl_alpha_grid.addWidget(QLabel('<b>Z (dBZ)</b>'), 0, 0)
        essl_alpha_grid.addWidget(QLabel('<b>Alpha</b>'), 0, 1)
        self.polrgb_essl_alpha_widgets = []
        essl_alpha_z = self.polrgb_params['ESSL_ALPHA_Z']
        essl_alpha_v = self.polrgb_params['ESSL_ALPHA_V']
        for i in range(len(essl_alpha_z)):
            z_widget = QLineEdit(str(ft.rifdot0(essl_alpha_z[i])))
            v_widget = QLineEdit(str(ft.rifdot0(essl_alpha_v[i])))
            essl_alpha_grid.addWidget(z_widget, i+1, 0)
            essl_alpha_grid.addWidget(v_widget, i+1, 1)
            self.polrgb_essl_alpha_widgets.append((z_widget, v_widget))
            z_widget.editingFinished.connect(self.change_polrgb_essl_alpha)
            v_widget.editingFinished.connect(self.change_polrgb_essl_alpha)
        essl_alpha_container = QWidget(); essl_alpha_container.setLayout(essl_alpha_grid)
        layout.addRow(QLabel(''), essl_alpha_container)

        self.polrgb_reset_defaultsw = QPushButton('Reset to defaults', autoDefault=True)
        layout.addRow(QLabel(''), self.polrgb_reset_defaultsw)
        self.polrgb_reset_defaultsw.clicked.connect(self.reset_polrgb_params)

        # Bespoke veld buiten polrgb_field_specs (15 augustus 2026): moet, anders dan de velden hierboven,
        # LEEG kunnen zijn (= geen grens) i.p.v. altijd een getal te vereisen -- change_polrgb_param
        # hierboven behandelt lege tekst als ongeldige invoer, niet als "uitgeschakeld". Zelfde None/leeg-
        # patroon als het analoge 3D-only veld (volume3d_polrgb_cc_max, Settings -> Miscellaneous), maar hier
        # voor de gewone 2D-weergave.
        self.polrgb_cc_hide_abovew = QLineEdit()
        self.polrgb_cc_hide_abovew.setText('' if self.polrgb_cc_hide_above is None else str(ft.rifdot0(self.polrgb_cc_hide_above)))
        self.polrgb_cc_hide_abovew.setToolTip("Pixels with CC (correlation coefficient, %) ABOVE this "
            "threshold are not drawn at all (fully transparent), regardless of their Z/ZDR -- useful to hide "
            "widespread ordinary rain (which has a high CC regardless of Z) without affecting hail (which "
            "has a lower CC by definition, so stays visible). Independent of CC scale minimum/maximum above, "
            "which only affect how brightly the green channel colors, not whether a pixel is drawn at all. "
            "Leave empty for no limit (original behaviour).")
        hbox_cc_hide = QHBoxLayout(); hbox_cc_hide.addWidget(self.polrgb_cc_hide_abovew); hbox_cc_hide.addStretch(30)
        layout.addRow(QLabel('Hide CC above (%, empty=no limit)'), hbox_cc_hide)
        self.polrgb_cc_hide_abovew.editingFinished.connect(self.change_polrgb_cc_hide_above)

        self.settingspolrgb.setLayout(layout)

    def change_polrgb_essl_mode(self, checked):
        self.polrgb_params['ESSL_MODE'] = bool(checked)
        self.pb.cmap_lastmodification_time['g'] = pytime.time()
        # Zelfde cache-reset als in change_polrgb_param hierboven -- nodig omdat dit veld, net als de andere
        # polrgb_params-sleutels, de daadwerkelijke PolRGB-berekening beinvloedt (niet alleen de kleurtabel).
        keys_to_delete = [k for k in self.dsg.stored_data if "'g'" in k or '"g"' in k]
        for k in keys_to_delete:
            del self.dsg.stored_data[k]
        if self.pb.firstplot_performed:
            self.pb.set_newdata([j for j in self.pb.panellist if self.crd.products[j] == 'g'])

    def change_polrgb_essl_alpha(self):
        # Bespoke handler voor de 11-rijen ESSL-alphatabel (zie settings_tabpolrgb): valideert ALLE 11 rijen
        # tegelijk (niet per los veld, zoals change_polrgb_param) omdat de eis "Z strikt oplopend" een
        # eigenschap van de hele lijst is, niet van 1 los getal -- np.interp (DataSource_General.
        # _essl_polrgb_channels) gedraagt zich onvoorspelbaar/stilletjes fout bij een niet-oplopende Z-reeks,
        # dus liever hier hard weigeren dan straks een stille rekenfout.
        z_values, v_values = [], []
        for z_widget, v_widget in self.polrgb_essl_alpha_widgets:
            z_num, v_num = ft.to_number(z_widget.text()), ft.to_number(v_widget.text())
            if z_num is None or v_num is None:
                self._reset_polrgb_essl_alpha_widgets()
                return
            z_values.append(float(z_num)); v_values.append(float(v_num))
        if any(z_values[i] >= z_values[i+1] for i in range(len(z_values)-1)):
            print('change_polrgb_essl_alpha: Z-waarden moeten strikt oplopend zijn, wijziging genegeerd.')
            self._reset_polrgb_essl_alpha_widgets()
            return
        self.polrgb_params['ESSL_ALPHA_Z'] = z_values
        self.polrgb_params['ESSL_ALPHA_V'] = v_values
        self.pb.cmap_lastmodification_time['g'] = pytime.time()
        # Zelfde cache-reset als in change_polrgb_param hierboven.
        keys_to_delete = [k for k in self.dsg.stored_data if "'g'" in k or '"g"' in k]
        for k in keys_to_delete:
            del self.dsg.stored_data[k]
        if self.pb.firstplot_performed:
            self.pb.set_newdata([j for j in self.pb.panellist if self.crd.products[j] == 'g'])

    def _reset_polrgb_essl_alpha_widgets(self):
        """Zet de 11-rijen ESSL-alphatabel terug naar de laatst opgeslagen (geldige) waarden in
        self.polrgb_params -- gebruikt door change_polrgb_essl_alpha bij ongeldige invoer en door
        reset_polrgb_params."""
        essl_alpha_z = self.polrgb_params['ESSL_ALPHA_Z']
        essl_alpha_v = self.polrgb_params['ESSL_ALPHA_V']
        for i, (z_widget, v_widget) in enumerate(self.polrgb_essl_alpha_widgets):
            z_widget.setText(str(ft.rifdot0(essl_alpha_z[i])))
            v_widget.setText(str(ft.rifdot0(essl_alpha_v[i])))

    def change_polrgb_param(self, key):
        input_text = self.polrgb_paramsw[key].text()
        try:
            number = float(input_text)
        except (ValueError, TypeError):
            number = None
        if number is not None:
            self.polrgb_params[key] = number
            self.pb.cmap_lastmodification_time['g'] = pytime.time()
            # Wis ook de stored_data-cache voor 'g': cmap_lastmodification_time alleen is niet genoeg,
            # want de PolRGB-berekening zelf (niet alleen de kleurtabel) hangt af van polrgb_params.
            # Zonder deze stap gebruikt check_presence_data_in_memory de gecachte RGBA-pixels en
            # roept _calculate_polrgb nooit opnieuw aan, waardoor Z_GAMMA e.d. geen effect hebben.
            keys_to_delete = [k for k in self.dsg.stored_data if "'g'" in k or '"g"' in k]
            for k in keys_to_delete:
                del self.dsg.stored_data[k]
            if self.pb.firstplot_performed:
                self.pb.set_newdata([j for j in self.pb.panellist if self.crd.products[j] == 'g'])
        else:
            self.polrgb_paramsw[key].setText(str(ft.rifdot0(self.polrgb_params[key])))

    def change_polrgb_cc_hide_above(self):
        # Bespoke handler i.p.v. change_polrgb_param hierboven: dit veld MOET leeg kunnen zijn (= geen
        # grens), terwijl change_polrgb_param lege tekst als ongeldige invoer behandelt -- zelfde None/leeg-
        # patroon als change_volume3d_polrgb_cc_max (Settings -> Miscellaneous, de analoge 3D-only versie).
        input_text = self.polrgb_cc_hide_abovew.text()
        number = ft.to_number(input_text)
        if input_text=='' or not number is None:
            self.polrgb_cc_hide_above = None if input_text=='' else float(number)
            self.pb.cmap_lastmodification_time['g'] = pytime.time()
            # Zelfde cache-reset als in change_polrgb_param hierboven -- nodig omdat dit veld, net als
            # polrgb_params, de daadwerkelijke PolRGB-berekening beinvloedt (niet alleen de kleurtabel).
            keys_to_delete = [k for k in self.dsg.stored_data if "'g'" in k or '"g"' in k]
            for k in keys_to_delete:
                del self.dsg.stored_data[k]
            if self.pb.firstplot_performed:
                self.pb.set_newdata([j for j in self.pb.panellist if self.crd.products[j] == 'g'])
        else:
            self.polrgb_cc_hide_abovew.setText('' if self.polrgb_cc_hide_above is None else str(ft.rifdot0(self.polrgb_cc_hide_above)))

    def reset_polrgb_params(self):
        self.polrgb_params = dict(polrgb_params_default)
        for key, widget in self.polrgb_paramsw.items():
            widget.setText(str(ft.rifdot0(self.polrgb_params[key])))
        self.polrgb_essl_modew.setChecked(bool(self.polrgb_params.get('ESSL_MODE', False)))
        self._reset_polrgb_essl_alpha_widgets()
        self.polrgb_cc_hide_above = None
        self.polrgb_cc_hide_abovew.setText('')
        self.pb.cmap_lastmodification_time['g'] = pytime.time()
        # Zelfde cache-reset als in change_polrgb_param hierboven.
        keys_to_delete = [k for k in self.dsg.stored_data if "'g'" in k or '"g"' in k]
        for k in keys_to_delete:
            del self.dsg.stored_data[k]
        if self.pb.firstplot_performed:
            self.pb.set_newdata([j for j in self.pb.panellist if self.crd.products[j] == 'g'])


    def settings_tabalgorithms(self):
        layout = QFormLayout()
        
        self.cartesian_product_resw= QLineEdit(str(self.cartesian_product_res))
        hbox= QHBoxLayout(); hbox.addWidget(self.cartesian_product_resw); hbox.addStretch(30)
        layout.addRow(QLabel('Product resolution of Cartesian derived products (km)'), hbox)
        self.cartesian_product_maxrangew= QLineEdit(str(self.cartesian_product_maxrange))
        hbox= QHBoxLayout(); hbox.addWidget(self.cartesian_product_maxrangew); hbox.addStretch(30)
        layout.addRow(QLabel('Maximum range of Cartesian derived products (km)'), hbox)
        
        self.settingsalgorithms.setLayout(layout)
        self.cartesian_product_resw.editingFinished.connect(lambda: self.change_cartesian_product_attr('res'))
        self.cartesian_product_maxrangew.editingFinished.connect(lambda: self.change_cartesian_product_attr('maxrange'))
                    
    def change_cartesian_product_attr(self, attr):
        number = ft.to_number(getattr(self, f'cartesian_product_{attr}w').text())
        if not number is None and number > 0:
            setattr(self, f'cartesian_product_{attr}', float(number))
        getattr(self, f'cartesian_product_{attr}w').setText(str(getattr(self, f'cartesian_product_{attr}')))
        if any([self.crd.products[j] in gv.plain_products_correct_for_SM for j in self.pb.panellist]) and self.stormmotion[1] != 0.:
            self.pb.set_newdata(self.pb.panellist)


    def settings_tabmiscellaneous(self):
        miscellaneous_layout=QFormLayout()
        hbox_max_radardata_in_memory_GBs=QHBoxLayout(); hbox_sleeptime_after_plotting=QHBoxLayout(); hbox_use_scissor=QHBoxLayout()
        hboxes=[hbox_max_radardata_in_memory_GBs,hbox_sleeptime_after_plotting,hbox_use_scissor]
        for hbox in hboxes:
            hbox.addStretch(1)
        
        self.max_radardata_in_memory_GBsw=QLineEdit(); self.max_radardata_in_memory_GBsw.setText(str(ft.rifdot0(self.max_radardata_in_memory_GBs)))
        hbox_max_radardata_in_memory_GBs.addWidget(self.max_radardata_in_memory_GBsw)
        self.sleeptime_after_plottingw=QLineEdit(); self.sleeptime_after_plottingw.setText(str(ft.rifdot0(self.sleeptime_after_plotting)))
        hbox_sleeptime_after_plotting.addWidget(self.sleeptime_after_plottingw)
        self.use_scissorw=QCheckBox(); self.use_scissorw.setTristate(False)
        self.use_scissorw.setCheckState(2 if self.use_scissor else 0)
        hbox_use_scissor.addWidget(self.use_scissorw)
        hbox_cross_section_resolution_factor=QHBoxLayout(); hbox_cross_section_resolution_factor.addStretch(1)
        self.cross_section_resolution_factorw=QLineEdit(); self.cross_section_resolution_factorw.setText(str(ft.rifdot0(self.cross_section_resolution_factor)))
        self.cross_section_resolution_factorw.setToolTip("Detail level of the vertical cross-section (F2/'Show "
            "cross-section >>'), as a multiplier on its base 150x80 raster -- e.g. 2 gives 300x160, 3 gives "
            "450x240. Higher looks sharper but, since the amount of underlying radar data along the line is "
            "fixed, pushing this too high can make the cross-section look sparse/speckled again instead of "
            "sharper. Takes effect immediately on any cross-section(s) currently open.")
        hbox_cross_section_resolution_factor.addWidget(self.cross_section_resolution_factorw)
        hbox_cross_section_resolution_factor.addStretch(50)
        hbox_cross_section_interpolation_mode=QHBoxLayout(); hbox_cross_section_interpolation_mode.addStretch(1)
        self.cross_section_interpolation_modew=QComboBox()
        self.cross_section_interpolation_modew.addItems(['nearest','bilinear','bicubic'])
        self.cross_section_interpolation_modew.setCurrentText(self.cross_section_interpolation_mode)
        self.cross_section_interpolation_modew.setToolTip("How the cross-section raster's bins are blended "
            "when stretched to the on-screen image. 'nearest' shows hard-edged blocks (no smoothing); "
            "'bilinear' smooths gently; 'bicubic' smooths the most, but can show a faint light/dark fringe "
            "right at hard edges (e.g. the top of an echo). Takes effect immediately.")
        hbox_cross_section_interpolation_mode.addWidget(self.cross_section_interpolation_modew)
        hbox_cross_section_interpolation_mode.addStretch(50)
        hbox_cross_section_n_samples=QHBoxLayout(); hbox_cross_section_n_samples.addStretch(1)
        self.cross_section_n_samplesw=QLineEdit(); self.cross_section_n_samplesw.setText(str(ft.rifdot0(self.cross_section_n_samples)))
        self.cross_section_n_samplesw.setToolTip("Number of sample points taken along the A/B line, per "
            "elevation scan, when computing a cross-section. Higher gives a more densely-filled raster "
            "(mainly useful if 'Cross-section detail level' above is increased), at the cost of a bit more "
            "computation each time the cross-section is (re)shown. Takes effect immediately.")
        hbox_cross_section_n_samples.addWidget(self.cross_section_n_samplesw)
        hbox_cross_section_n_samples.addStretch(50)
        hbox_cross_section_height_headroom=QHBoxLayout(); hbox_cross_section_height_headroom.addStretch(1)
        self.cross_section_height_headroom_percentw=QLineEdit(); self.cross_section_height_headroom_percentw.setText(str(ft.rifdot0(self.cross_section_height_headroom_percent)))
        self.cross_section_height_headroom_percentw.setToolTip("Extra empty space above the highest echo top, "
            "as a percentage, when setting the cross-section's vertical scale (there's always at least 8 km "
            "shown regardless). Higher shows more empty sky above the echo (more 'zoomed out' vertically); "
            "lower zooms in more tightly on the echo itself. Takes effect immediately.")
        hbox_cross_section_height_headroom.addWidget(self.cross_section_height_headroom_percentw)
        hbox_cross_section_height_headroom.addStretch(50)

        hbox_volume3d_grid_res_km=QHBoxLayout(); hbox_volume3d_grid_res_km.addStretch(1)
        self.volume3d_grid_res_kmw=QLineEdit(); self.volume3d_grid_res_kmw.setText(str(ft.rifdot0(self.volume3d_grid_res_km)))
        self.volume3d_grid_res_kmw.setToolTip("Horizontal grid resolution (km) for the 3D volume viewer "
            "(CTRL+SHIFT+4). Finer (lower) shows more detail but is "
            "slower to compute/render; coarser (higher) is quicker but blockier. Takes effect the next time "
            "the 3D viewer is opened, not on any already-open window.")
        hbox_volume3d_grid_res_km.addWidget(self.volume3d_grid_res_kmw)
        hbox_volume3d_grid_res_km.addStretch(50)
        hbox_volume3d_z_max_km=QHBoxLayout(); hbox_volume3d_z_max_km.addStretch(1)
        self.volume3d_z_max_kmw=QLineEdit(); self.volume3d_z_max_kmw.setText(str(ft.rifdot0(self.volume3d_z_max_km)))
        self.volume3d_z_max_kmw.setToolTip("Maximum height (km) included in the 3D volume reconstruction. "
            "Updates the 3D viewer live if one is currently open, otherwise takes effect the next time it's opened.")
        hbox_volume3d_z_max_km.addWidget(self.volume3d_z_max_kmw)
        hbox_volume3d_z_max_km.addStretch(50)
        hbox_volume3d_z_res_km=QHBoxLayout(); hbox_volume3d_z_res_km.addStretch(1)
        self.volume3d_z_res_kmw=QLineEdit(); self.volume3d_z_res_kmw.setText(str(ft.rifdot0(self.volume3d_z_res_km)))
        self.volume3d_z_res_kmw.setToolTip("Vertical grid resolution (km) for the 3D volume viewer. Finer "
            "(lower) is smoother but slower; coarser (higher) is quicker but blockier vertically. Updates "
            "the 3D viewer live if one is currently open, otherwise takes effect the next time it's opened.")
        hbox_volume3d_z_res_km.addWidget(self.volume3d_z_res_kmw)
        hbox_volume3d_z_res_km.addStretch(50)
        hbox_volume3d_vertical_exaggeration=QHBoxLayout(); hbox_volume3d_vertical_exaggeration.addStretch(1)
        self.volume3d_vertical_exaggerationw=QLineEdit(); self.volume3d_vertical_exaggerationw.setText(str(ft.rifdot0(self.volume3d_vertical_exaggeration)))
        self.volume3d_vertical_exaggerationw.setToolTip("How many times taller the height axis is stretched "
            "on screen in the 3D volume viewer, since a storm's real height (a few km) would otherwise look "
            "almost flat next to its horizontal extent (tens of km). 1 = no exaggeration (true to scale). "
            "Updates the 3D viewer live if one is currently open, otherwise takes effect the next time it's opened.")
        hbox_volume3d_vertical_exaggeration.addWidget(self.volume3d_vertical_exaggerationw)
        hbox_volume3d_vertical_exaggeration.addStretch(50)
        hbox_volume3d_smoothing_sigma=QHBoxLayout(); hbox_volume3d_smoothing_sigma.addStretch(1)
        self.volume3d_smoothing_sigmaw=QLineEdit(); self.volume3d_smoothing_sigmaw.setText(str(ft.rifdot0(self.volume3d_smoothing_sigma)))
        self.volume3d_smoothing_sigmaw.setToolTip("Strength of the horizontal smoothing between neighbouring "
            "grid columns in the 3D volume viewer, that turns the raw, blocky ('Minecraft') reconstruction "
            "into a smoother cloud shape. 0 disables smoothing entirely (raw/blocky). Updates the 3D viewer "
            "live if one is currently open, otherwise takes effect the next time it's opened.")
        hbox_volume3d_smoothing_sigma.addWidget(self.volume3d_smoothing_sigmaw)
        hbox_volume3d_smoothing_sigma.addStretch(50)
        hbox_volume3d_tick_interval_km=QHBoxLayout(); hbox_volume3d_tick_interval_km.addStretch(1)
        self.volume3d_tick_interval_kmw=QLineEdit(); self.volume3d_tick_interval_kmw.setText(str(ft.rifdot0(self.volume3d_tick_interval_km)))
        self.volume3d_tick_interval_kmw.setToolTip("Spacing (km) between the ruler tick marks along the 4 "
            "base edges of the 3D viewer's reference box. Updates the 3D viewer live if one is currently open, otherwise takes effect the next time it's opened.")
        hbox_volume3d_tick_interval_km.addWidget(self.volume3d_tick_interval_kmw)
        hbox_volume3d_tick_interval_km.addStretch(50)
        hbox_volume3d_height_tick_interval_km=QHBoxLayout(); hbox_volume3d_height_tick_interval_km.addStretch(1)
        self.volume3d_height_tick_interval_kmw=QLineEdit(); self.volume3d_height_tick_interval_kmw.setText(str(ft.rifdot0(self.volume3d_height_tick_interval_km)))
        self.volume3d_height_tick_interval_kmw.setToolTip("Spacing (km, real height, before the exaggeration "
            "above) between the ruler tick marks on the 3D viewer's vertical height ruler. Updates the 3D "
            "viewer live if one is currently open, otherwise takes effect the next time it's opened.")
        hbox_volume3d_height_tick_interval_km.addWidget(self.volume3d_height_tick_interval_kmw)
        hbox_volume3d_height_tick_interval_km.addStretch(50)
        hbox_volume3d_gamma=QHBoxLayout(); hbox_volume3d_gamma.addStretch(1)
        self.volume3d_gammaw=QLineEdit(); self.volume3d_gammaw.setText(str(ft.rifdot0(self.volume3d_gamma)))
        self.volume3d_gammaw.setToolTip("Gamma correction on the color intensity of the 3D volume data. "
            "1.0 = no change (original behaviour). Lower than 1 makes mid-range values relatively brighter/"
            "more contrasty; higher than 1 dims everything except the highest values. Purely a visual "
            "intensity translation -- the underlying data/values themselves are unaffected. Updates the 3D "
            "viewer live if one is currently open, otherwise takes effect the next time it's opened.")
        hbox_volume3d_gamma.addWidget(self.volume3d_gammaw)
        hbox_volume3d_gamma.addStretch(50)
        hbox_volume3d_pointcloud_stride=QHBoxLayout(); hbox_volume3d_pointcloud_stride.addStretch(1)
        self.volume3d_pointcloud_stridew=QLineEdit(); self.volume3d_pointcloud_stridew.setText(str(ft.rifdot0(self.volume3d_pointcloud_stride)))
        self.volume3d_pointcloud_stridew.setToolTip("Density of the point cloud 3D render mode (P key in the "
            "3D viewer). 1 = every grid point (dense, slow), higher = every Nth point in each direction "
            "(sparser, faster, more see-through). Updates the 3D viewer live if one is currently open, "
            "otherwise takes effect the next time it's opened.")
        hbox_volume3d_pointcloud_stride.addWidget(self.volume3d_pointcloud_stridew)
        hbox_volume3d_pointcloud_stride.addStretch(50)
        hbox_volume3d_pointcloud_point_size=QHBoxLayout(); hbox_volume3d_pointcloud_point_size.addStretch(1)
        self.volume3d_pointcloud_point_sizew=QLineEdit(); self.volume3d_pointcloud_point_sizew.setText(str(ft.rifdot0(self.volume3d_pointcloud_point_size)))
        self.volume3d_pointcloud_point_sizew.setToolTip("Size (screen pixels) of each point in the point cloud "
            "3D render mode (P key in the 3D viewer). Larger points overlap more, looking more like a solid "
            "surface again; smaller points give a sparser, more see-through look but can be hard to make out "
            "as a shape. Updates the 3D viewer live if one is currently open, otherwise takes effect the next "
            "time it's opened.")
        hbox_volume3d_pointcloud_point_size.addWidget(self.volume3d_pointcloud_point_sizew)
        hbox_volume3d_pointcloud_point_size.addStretch(50)
        hbox_volume3d_min_value=QHBoxLayout(); hbox_volume3d_min_value.addStretch(1)
        self.volume3d_min_valuew=QLineEdit()
        self.volume3d_min_valuew.setText('' if self.volume3d_min_value is None else str(ft.rifdot0(self.volume3d_min_value)))
        self.volume3d_min_valuew.setToolTip("Values below this threshold are not drawn at all in the 3D "
            "viewer (in any of the 3 render modes: MIP, translucent, or point cloud), instead of just being "
            "colored differently -- useful to get rid of clutter from widespread weak echo. For a symmetric, "
            "two-sided product like V (e.g. -60 to +60 m/s), this instead filters by absolute value around "
            "zero, keeping both strong inbound and strong outbound values while removing weak values near "
            "zero on both sides. Leave empty for no limit (original behaviour). Same unit as whichever "
            "product you're viewing (e.g. dBZ for Z, m/s for V). Updates the 3D viewer live if one is "
            "currently open, otherwise takes effect the next time it's opened.")
        hbox_volume3d_min_value.addWidget(self.volume3d_min_valuew)
        hbox_volume3d_min_value.addStretch(50)
        hbox_volume3d_polrgb_cc_max=QHBoxLayout(); hbox_volume3d_polrgb_cc_max.addStretch(1)
        self.volume3d_polrgb_cc_maxw=QLineEdit()
        self.volume3d_polrgb_cc_maxw.setText('' if self.volume3d_polrgb_cc_max is None else str(ft.rifdot0(self.volume3d_polrgb_cc_max)))
        self.volume3d_polrgb_cc_maxw.setToolTip("PolRGB only: voxels with CC (correlation coefficient, %) "
            "ABOVE this threshold are not drawn at all in the 3D viewer (RGB-MIP or point cloud), regardless "
            "of their Z/ZDR -- useful to hide widespread ordinary rain (which has a high CC regardless of Z) "
            "without affecting hail (which has a lower CC by definition, so stays visible). Independent of "
            "'minimum value to draw' above, which filters on Z instead. Leave empty for no limit (original "
            "behaviour). Has no effect for any product other than PolRGB. Updates the 3D viewer live if one "
            "is currently open, otherwise takes effect the next time it's opened.")
        hbox_volume3d_polrgb_cc_max.addWidget(self.volume3d_polrgb_cc_maxw)
        hbox_volume3d_polrgb_cc_max.addStretch(50)
        hbox_volume3d_circular_area=QHBoxLayout(); hbox_volume3d_circular_area.addStretch(1)
        self.volume3d_circular_areaw=QCheckBox(); self.volume3d_circular_areaw.setTristate(False)
        self.volume3d_circular_areaw.setCheckState(2 if self.volume3d_circular_area else 0)
        self.volume3d_circular_areaw.setToolTip("When checked, the area drawn with CTRL+SHIFT+drag is "
            "treated as an ELLIPSE inscribed within that rectangle (center = rectangle center, semi-axes = "
            "half width/height), instead of the full rectangle. Data outside that ellipse is masked out, in "
            "both the 2D preview of the drawn area and the 3D data itself -- the 3D box/axes/ticks always "
            "keep showing the full rectangular extent, only the colored data itself gets cut round. Affects "
            "the NEXT time you draw an area / open the 3D viewer, not an already-drawn rectangle.")
        hbox_volume3d_circular_area.addWidget(self.volume3d_circular_areaw)
        hbox_volume3d_circular_area.addStretch(50)

        for hbox in hboxes:
            hbox.addStretch(50)

        miscellaneous_widgets=[[QLabel('Maximum amount of radar data kept in memory (GB)'),hbox_max_radardata_in_memory_GBs],
                               [QLabel('Sleep time after plotting'),hbox_sleeptime_after_plotting],
                               [QLabel('Enable partial drawing of screen'),hbox_use_scissor],
                               [QLabel('Cross-section detail level (raster size multiplier)'),hbox_cross_section_resolution_factor],
                               [QLabel('Cross-section smoothing (interpolation)'),hbox_cross_section_interpolation_mode],
                               [QLabel('Cross-section line samples per scan'),hbox_cross_section_n_samples],
                               [QLabel('Cross-section vertical headroom (%)'),hbox_cross_section_height_headroom],
                               [QLabel('3D viewer: horizontal grid resolution (km)'),hbox_volume3d_grid_res_km],
                               [QLabel('3D viewer: maximum height (km)'),hbox_volume3d_z_max_km],
                               [QLabel('3D viewer: vertical grid resolution (km)'),hbox_volume3d_z_res_km],
                               [QLabel('3D viewer: vertical exaggeration factor'),hbox_volume3d_vertical_exaggeration],
                               [QLabel('3D viewer: horizontal smoothing strength'),hbox_volume3d_smoothing_sigma],
                               [QLabel('3D viewer: ruler tick spacing, horizontal (km)'),hbox_volume3d_tick_interval_km],
                               [QLabel('3D viewer: ruler tick spacing, height (km)'),hbox_volume3d_height_tick_interval_km],
                               [QLabel('3D viewer: color gamma (1.0 = normal)'),hbox_volume3d_gamma],
                               [QLabel('3D viewer: point cloud density (1=dense, higher=sparser)'),hbox_volume3d_pointcloud_stride],
                               [QLabel('3D viewer: point cloud point size (px)'),hbox_volume3d_pointcloud_point_size],
                               [QLabel('3D viewer: minimum value to draw (empty=no limit)'),hbox_volume3d_min_value],
                               [QLabel('3D viewer PolRGB: hide CC above (%, empty=no limit)'),hbox_volume3d_polrgb_cc_max],
                               [QLabel('3D viewer: circular (ellipse) area instead of rectangle'),hbox_volume3d_circular_area]]
        for j in range(0,len(miscellaneous_widgets)):
            miscellaneous_layout.addRow(miscellaneous_widgets[j][0],miscellaneous_widgets[j][1])

        self.cross_section_reset_defaultsw=QPushButton('Reset cross-section settings to defaults', autoDefault=True)
        miscellaneous_layout.addRow(QLabel(''), self.cross_section_reset_defaultsw)
        self.volume3d_reset_defaultsw=QPushButton('Reset 3D viewer settings to defaults', autoDefault=True)
        miscellaneous_layout.addRow(QLabel(''), self.volume3d_reset_defaultsw)

        self.settingsmiscellaneous.setLayout(miscellaneous_layout)
        self.max_radardata_in_memory_GBsw.editingFinished.connect(self.change_max_radardata_in_memory_GBs)
        self.sleeptime_after_plottingw.editingFinished.connect(self.change_sleeptime_after_plotting)
        self.use_scissorw.stateChanged.connect(self.change_use_scissor)
        self.cross_section_resolution_factorw.editingFinished.connect(self.change_cross_section_resolution_factor)
        self.cross_section_interpolation_modew.currentTextChanged.connect(self.change_cross_section_interpolation_mode)
        self.cross_section_n_samplesw.editingFinished.connect(self.change_cross_section_n_samples)
        self.cross_section_height_headroom_percentw.editingFinished.connect(self.change_cross_section_height_headroom_percent)
        self.cross_section_reset_defaultsw.clicked.connect(self.reset_cross_section_settings)
        self.volume3d_grid_res_kmw.editingFinished.connect(self.change_volume3d_grid_res_km)
        self.volume3d_z_max_kmw.editingFinished.connect(self.change_volume3d_z_max_km)
        self.volume3d_z_res_kmw.editingFinished.connect(self.change_volume3d_z_res_km)
        self.volume3d_vertical_exaggerationw.editingFinished.connect(self.change_volume3d_vertical_exaggeration)
        self.volume3d_smoothing_sigmaw.editingFinished.connect(self.change_volume3d_smoothing_sigma)
        self.volume3d_tick_interval_kmw.editingFinished.connect(self.change_volume3d_tick_interval_km)
        self.volume3d_height_tick_interval_kmw.editingFinished.connect(self.change_volume3d_height_tick_interval_km)
        self.volume3d_gammaw.editingFinished.connect(self.change_volume3d_gamma)
        self.volume3d_pointcloud_stridew.editingFinished.connect(self.change_volume3d_pointcloud_stride)
        self.volume3d_pointcloud_point_sizew.editingFinished.connect(self.change_volume3d_pointcloud_point_size)
        self.volume3d_min_valuew.editingFinished.connect(self.change_volume3d_min_value)
        self.volume3d_polrgb_cc_maxw.editingFinished.connect(self.change_volume3d_polrgb_cc_max)
        self.volume3d_circular_areaw.stateChanged.connect(self.change_volume3d_circular_area)
        self.volume3d_reset_defaultsw.clicked.connect(self.reset_volume3d_settings)

    def reset_cross_section_settings(self):
        """Resets the 4 tunable cross-section settings (Settings -> Miscellaneous) to their shipped defaults
        (see cross_section_settings_default), updates the widgets to reflect that, and immediately refreshes
        any cross-section(s) currently open -- the same 'reset, update widgets, refresh live view' pattern as
        reset_polrgb_params uses for the PolRGB tab."""
        d = cross_section_settings_default
        self.cross_section_resolution_factor = d['cross_section_resolution_factor']
        self.cross_section_interpolation_mode = d['cross_section_interpolation_mode']
        self.cross_section_n_samples = d['cross_section_n_samples']
        self.cross_section_height_headroom_percent = d['cross_section_height_headroom_percent']

        self.cross_section_resolution_factorw.setText(str(ft.rifdot0(self.cross_section_resolution_factor)))
        self.cross_section_interpolation_modew.setCurrentText(self.cross_section_interpolation_mode)
        self.cross_section_n_samplesw.setText(str(ft.rifdot0(self.cross_section_n_samples)))
        self.cross_section_height_headroom_percentw.setText(str(ft.rifdot0(self.cross_section_height_headroom_percent)))

        for panel in list(self.pb.cross_section_active_panels):
            self.pb.show_cross_section(panel)

    def reset_volume3d_settings(self):
        """Resets the 7 tunable 3D-viewer settings (Settings -> Miscellaneous) to their shipped defaults
        (see volume3d_settings_default), updates the widgets to reflect that, and live-refreshes the 3D
        viewer if one is currently open (see _live_refresh_volume_3d) -- same 'reset, update widgets,
        refresh live view' pattern as reset_cross_section_settings."""
        d = volume3d_settings_default
        self.volume3d_grid_res_km = d['volume3d_grid_res_km']
        self.volume3d_z_max_km = d['volume3d_z_max_km']
        self.volume3d_z_res_km = d['volume3d_z_res_km']
        self.volume3d_vertical_exaggeration = d['volume3d_vertical_exaggeration']
        self.volume3d_smoothing_sigma = d['volume3d_smoothing_sigma']
        self.volume3d_tick_interval_km = d['volume3d_tick_interval_km']
        self.volume3d_height_tick_interval_km = d['volume3d_height_tick_interval_km']
        self.volume3d_gamma = d['volume3d_gamma']
        self.volume3d_pointcloud_stride = d['volume3d_pointcloud_stride']
        self.volume3d_pointcloud_point_size = d['volume3d_pointcloud_point_size']
        self.volume3d_min_value = d['volume3d_min_value']
        self.volume3d_circular_area = d['volume3d_circular_area']
        self.volume3d_polrgb_cc_max = d['volume3d_polrgb_cc_max']

        self.volume3d_grid_res_kmw.setText(str(ft.rifdot0(self.volume3d_grid_res_km)))
        self.volume3d_z_max_kmw.setText(str(ft.rifdot0(self.volume3d_z_max_km)))
        self.volume3d_z_res_kmw.setText(str(ft.rifdot0(self.volume3d_z_res_km)))
        self.volume3d_vertical_exaggerationw.setText(str(ft.rifdot0(self.volume3d_vertical_exaggeration)))
        self.volume3d_smoothing_sigmaw.setText(str(ft.rifdot0(self.volume3d_smoothing_sigma)))
        self.volume3d_tick_interval_kmw.setText(str(ft.rifdot0(self.volume3d_tick_interval_km)))
        self.volume3d_height_tick_interval_kmw.setText(str(ft.rifdot0(self.volume3d_height_tick_interval_km)))
        self.volume3d_gammaw.setText(str(ft.rifdot0(self.volume3d_gamma)))
        self.volume3d_pointcloud_stridew.setText(str(ft.rifdot0(self.volume3d_pointcloud_stride)))
        self.volume3d_pointcloud_point_sizew.setText(str(ft.rifdot0(self.volume3d_pointcloud_point_size)))
        self.volume3d_min_valuew.setText('' if self.volume3d_min_value is None else str(ft.rifdot0(self.volume3d_min_value)))
        self.volume3d_polrgb_cc_maxw.setText('' if self.volume3d_polrgb_cc_max is None else str(ft.rifdot0(self.volume3d_polrgb_cc_max)))
        self.volume3d_circular_areaw.setCheckState(2 if self.volume3d_circular_area else 0)
        self._live_refresh_volume_3d()

    def change_volume3d_grid_res_km(self):
        number = ft.to_number(self.volume3d_grid_res_kmw.text())
        if not number is None and number>0:
            self.volume3d_grid_res_km = float(number)
            self._live_refresh_volume_3d()
        else: self.volume3d_grid_res_kmw.setText(str(ft.rifdot0(self.volume3d_grid_res_km)))

    def change_volume3d_z_max_km(self):
        number = ft.to_number(self.volume3d_z_max_kmw.text())
        if not number is None and number>0:
            self.volume3d_z_max_km = float(number)
            self._live_refresh_volume_3d()
        else: self.volume3d_z_max_kmw.setText(str(ft.rifdot0(self.volume3d_z_max_km)))

    def change_volume3d_z_res_km(self):
        number = ft.to_number(self.volume3d_z_res_kmw.text())
        if not number is None and number>0:
            self.volume3d_z_res_km = float(number)
            self._live_refresh_volume_3d()
        else: self.volume3d_z_res_kmw.setText(str(ft.rifdot0(self.volume3d_z_res_km)))

    def change_volume3d_vertical_exaggeration(self):
        number = ft.to_number(self.volume3d_vertical_exaggerationw.text())
        if not number is None and number>0:
            self.volume3d_vertical_exaggeration = float(number)
            self._live_refresh_volume_3d()
        else: self.volume3d_vertical_exaggerationw.setText(str(ft.rifdot0(self.volume3d_vertical_exaggeration)))

    def change_volume3d_smoothing_sigma(self):
        number = ft.to_number(self.volume3d_smoothing_sigmaw.text())
        if not number is None and number>=0:
            self.volume3d_smoothing_sigma = float(number)
            self._live_refresh_volume_3d()
        else: self.volume3d_smoothing_sigmaw.setText(str(ft.rifdot0(self.volume3d_smoothing_sigma)))

    def change_volume3d_tick_interval_km(self):
        number = ft.to_number(self.volume3d_tick_interval_kmw.text())
        if not number is None and number>0:
            self.volume3d_tick_interval_km = float(number)
            self._live_refresh_volume_3d()
        else: self.volume3d_tick_interval_kmw.setText(str(ft.rifdot0(self.volume3d_tick_interval_km)))

    def change_volume3d_height_tick_interval_km(self):
        number = ft.to_number(self.volume3d_height_tick_interval_kmw.text())
        if not number is None and number>0:
            self.volume3d_height_tick_interval_km = float(number)
            self._live_refresh_volume_3d()
        else: self.volume3d_height_tick_interval_kmw.setText(str(ft.rifdot0(self.volume3d_height_tick_interval_km)))

    def change_volume3d_gamma(self):
        number = ft.to_number(self.volume3d_gammaw.text())
        if not number is None and 0.1<=number<=5.0:
            self.volume3d_gamma = float(number)
            self._live_refresh_volume_3d()
        else: self.volume3d_gammaw.setText(str(ft.rifdot0(self.volume3d_gamma)))

    def change_volume3d_pointcloud_stride(self):
        number = ft.to_number(self.volume3d_pointcloud_stridew.text())
        if not number is None and number>=1:
            self.volume3d_pointcloud_stride = int(round(number))
            self._live_refresh_volume_3d()
        else: self.volume3d_pointcloud_stridew.setText(str(ft.rifdot0(self.volume3d_pointcloud_stride)))

    def change_volume3d_pointcloud_point_size(self):
        number = ft.to_number(self.volume3d_pointcloud_point_sizew.text())
        if not number is None and number>0:
            self.volume3d_pointcloud_point_size = float(number)
            self._live_refresh_volume_3d()
        else: self.volume3d_pointcloud_point_sizew.setText(str(ft.rifdot0(self.volume3d_pointcloud_point_size)))

    def change_volume3d_min_value(self):
        input_text = self.volume3d_min_valuew.text()
        number = ft.to_number(input_text)
        if input_text=='' or not number is None:
            self.volume3d_min_value = None if input_text=='' else float(number)
            self._live_refresh_volume_3d()
        else: self.volume3d_min_valuew.setText('' if self.volume3d_min_value is None else str(ft.rifdot0(self.volume3d_min_value)))

    def change_volume3d_polrgb_cc_max(self):
        input_text = self.volume3d_polrgb_cc_maxw.text()
        number = ft.to_number(input_text)
        if input_text=='' or not number is None:
            self.volume3d_polrgb_cc_max = None if input_text=='' else float(number)
            self._live_refresh_volume_3d()
        else: self.volume3d_polrgb_cc_maxw.setText('' if self.volume3d_polrgb_cc_max is None else str(ft.rifdot0(self.volume3d_polrgb_cc_max)))

    def change_volume3d_circular_area(self):
        self.volume3d_circular_area = True if self.volume3d_circular_areaw.checkState()==2 else False
        # Bestaande, al getekende rechthoek meteen bijwerken naar de nieuwe voorkeur (i.p.v. pas bij de
        # volgende keer tekenen), zodat de 2D-voorvertoning altijd meteen klopt met deze instelling.
        if getattr(self.pb, 'volume3d_rect_a', None) is not None:
            self.pb.draw_volume3d_rect()

    def change_cross_section_n_samples(self):
        input_text=self.cross_section_n_samplesw.text()
        number=ft.to_number(input_text)
        if not number is None and number>=100:
            self.cross_section_n_samples=int(round(number))
            for panel in list(self.pb.cross_section_active_panels):
                self.pb.show_cross_section(panel)
        else: self.cross_section_n_samplesw.setText(str(ft.rifdot0(self.cross_section_n_samples)))

    def change_cross_section_height_headroom_percent(self):
        input_text=self.cross_section_height_headroom_percentw.text()
        number=ft.to_number(input_text)
        if not number is None and number>=0:
            self.cross_section_height_headroom_percent=float(number)
            for panel in list(self.pb.cross_section_active_panels):
                self.pb.show_cross_section(panel)
        else: self.cross_section_height_headroom_percentw.setText(str(ft.rifdot0(self.cross_section_height_headroom_percent)))
        
    def change_cross_section_resolution_factor(self):
        input_text=self.cross_section_resolution_factorw.text()
        number=ft.to_number(input_text)
        if not number is None and number>0:
            self.cross_section_resolution_factor=float(number)
            # Live refresh: redraw any cross-section(s) currently open with the new raster size right away,
            # using their existing A/B line -- mirrors how change_polrgb_param immediately re-renders any
            # panel showing product 'g', rather than requiring the person to close and reopen the view.
            for panel in list(self.pb.cross_section_active_panels):
                self.pb.show_cross_section(panel)
        else: self.cross_section_resolution_factorw.setText(str(ft.rifdot0(self.cross_section_resolution_factor)))

    def change_cross_section_interpolation_mode(self):
        self.cross_section_interpolation_mode=self.cross_section_interpolation_modew.currentText()
        # Live and safe: nlr_plotting.py now pre-builds one ImageVisual per panel for EACH interpolation mode
        # at startup (see __init__ and cross_section_visual), rather than creating/mutating one at runtime --
        # so switching mode here just means re-running show_cross_section for any panel with an open cross-
        # section, which picks the already-fully-wired visual matching the new mode and hides the other 2.
        for panel in list(self.pb.cross_section_active_panels):
            self.pb.show_cross_section(panel)

    def change_max_radardata_in_memory_GBs(self):
        input_max_radardata_in_memory_GBs=self.max_radardata_in_memory_GBsw.text()
        number=ft.to_number(input_max_radardata_in_memory_GBs)
        if not number is None and number>=0:
            self.max_radardata_in_memory_GBs=float(number)
        else: self.max_radardata_in_memory_GBsw.setText(str(ft.rifdot0(self.max_radardata_in_memory_GBs)))
    def change_sleeptime_after_plotting(self):
        input_sleeptime_after_plotting=self.sleeptime_after_plottingw.text()
        number=ft.to_number(input_sleeptime_after_plotting)
        if not number is None and number>=0:
            self.sleeptime_after_plotting=number
        else: self.sleeptime_after_plottingw.setText(str(ft.rifdot0(self.sleeptime_after_plotting)))
    def change_use_scissor(self):
        self.use_scissor=True if self.use_scissorw.checkState()==2 else False
                
                
    def helpwidget(self):
        self.help=QTabWidget()
        self.help.setWindowTitle('NLradar help')
        self.helpgeneral=QWidget(); self.helpkeyboard=QWidget(); self.helpradardata=QWidget()
        self.helpproducts=QWidget(); self.helpcolortables=QWidget(); self.helpsettings=QWidget(); self.helpextra=QWidget()
        self.help.addTab(self.helpgeneral,'General')
        self.help.addTab(self.helpkeyboard,'Keyboard')
        self.help.addTab(self.helpradardata,'Radar data')
        self.help.addTab(self.helpproducts,'Products')
        self.help.addTab(self.helpcolortables,'Color tables')
        self.help.addTab(self.helpsettings,'Settings')
        self.help.addTab(self.helpextra,'Extra remarks')
                
        self.help_tabgeneral(); self.help_tabkeyboard(); self.help_tabradardata(); self.help_tabproducts(); self.help_tabcolortables(); self.help_tabsettings(); self.help_tabextra()
        self.help.resize(self.help.sizeHint())
        self.help.show()
        
    def help_tabgeneral(self):
        general_layout=QVBoxLayout()
        general_text=["It is recommended to read through this help before starting to use the program regularly.",
                      '',
                      "Current data can be obtained automatically by right-clicking at a radar, and selecting 'start automatic download for ...'. It can be stopped in a similar way.",
                      "Older data for the current day (and for the selected radar) can be obtained by using the download widgets in the menu. Both processes can run for multiple radars simultaneously.",
                      "IMPORTANT: For downloading KNMI data you will have to set API keys at <b>Settings/Download</b>.",
                      "For more information about obtaining current and storing archived data, see the tab <b>Radar data</b>.",
                      "",
                      "Switching to a different radar takes place by clicking on a radar marker. If you hold the CTRL key when clicking, the new radar is only selected, without plotting.",
                      "Most of the navigation takes place by means of keyboard shortcuts. It is therefore important to read the help section <b>Keyboard</b>.",
                      '',
                      'Zooming can not only be done with the scroll wheel of the mouse, but also by holding the right mouse button and moving the mouse. This way of zooming is more suitable when you need to make',
                      'small steps.',
                      '',
                      'The menu that pops up when right-clicking at the map gives you the option to place a marker at the mouse position. This can be done to mark a location, but the marker can also be used to determine a storm motion vector',
                      '(which is used for the storm-relative velocity). For the latter option place the marker at the starting point of the storm, and then go backward/forward in time (possibly changing other parameters',
                      'like the scan or radar), and then right-click again at the current position of the storm. You can then choose to calculate the storm motion vector. The storm motion vector can also be set manually via',
                      'the same right-click menu, or can be reset.',
                      'This menu is also used to provide radar-specific options. Currently there is one such option, which is the option to change the start azimuth of the scans, currently only available for Cabauw.',
                      'It also provides the option to add the current radar view to a list of cases, which can be assessed by clicking at the Cases widget in the menu bar.',
                      '',
                      "Color tables can be changed, as described at <b>Color tables</b>.",
                      '',
                      "Incomplete files cannot be opened by the program, and if you try so, an error is raised in the error log (the attached black window). In the case of unfinished downloads they are usually automatically",
                      "removed, but if this does not occur then you can manually remove them from the folder in which they are located. This folder is a subfolder with the name 'Download' of your base directory (selected at",
                      "<b>Settings/Radar data</b>). Unfinished downloads are automatically removed when closing the program.",
                      '',
                      "The color of a radar marker indicates whether the radar is selected, whether automatic download takes place and whether download of older data occurs. These colors can be selected at ",
                      "<b>Settings/Map</b>. A darker color indicates that download of older data occurs.",
                      "Multiple other properties of the map can be changed too in the <b>Settings</b>.",
                      '',
                      "Radar data gets stored in memory after it has been read from your hard disk, and the maximum amount of memory that the program can use for storing data can be selected at",
                      "<b>Settings/Miscellaneous</b> (default is 2 GB). Reading data from memory is clearly faster than reading them from the hard disk.",
                      "The program also saves a number of 'volume attributes' to a file after determining them once, to increase the speed. These volume attributes include e.g. the scanangles for all scans in the volume, and",
                      "they are used internally by the program. It could occur that wrong attributes get saved, resulting in incorrect plots. If this is the case, then you can remove these attributes under the menu item <b>Extra</b>.",
                      '',
                      "Basemap source: Bing."]
        for j in range(0,len(general_text)):
            qlabel=QLabel(general_text[j],font=self.help_font); qlabel.setOpenExternalLinks(True)
            general_layout.addWidget(qlabel)
        general_layout.addStretch(1)
        self.helpgeneral.setLayout(general_layout)
               
    def help_tabkeyboard(self):
        global plottimes_max
        keyboard_layout=QFormLayout()
        keyboard_text=[['ENTER','Plot for current input'],
                  ['SHIFT+click (twice)','Draw an A/B measuring line on the current panel'],
                  ['F2','Show/hide a vertical Velocity cross-section along the current A/B line'],
                  ['LEFT/RIGHT, SHIFT+LEFT/RIGHT','Go to previous/next radar volume, go one hour backward/forward in time.'],
                  ['END/SHIFT+END', "Set the date and time equal to 'c' (most current data), with/without plotting the data."],
                  ['SPACE',"Start/stop animation, or stop continuing backward/forward in time. Animation ends at input date and time. If both 'c', then end time gets updated when new data available."],        
                  ['</>','Go continuously backward/forward in time.'],
                  ['(SHIFT+)BACKSPACE','Go back to the previous combination of radar and dataset (and also date and time).'],
                  ['CTRL+SPACE, (CTRL+)SHIFT+SPACE','Loop through case list, animate case by looping through the animation window specified in Cases/Settings (while looping through case list too)'],
                  ['CTRL+(ENTER/LEFT/RIGHT, BACKSPACE)','Switch to current/previous/next case in currently selected case list, switch back to previously shown case (can be from other list)'],
                  ['(SHIFT+)HOME, CTRL+HOME','Reset view to radar-centered and (not) reset zoom, or reset zoom without resetting view'],
                  ['F','Enable/disable storm-following view, which moves view with currently set storm motion'],
                  ['ALT+1,2,3,4,6,8,0','Show 1,2,3,4,6,8 or 10 panels.'],
                  ['CTRL+1-10','Switch to the radar that is nth nearest to the panel centers.'],
                  ['N/ALT+N','Automatically select nearest radar after a change of view/Select which radar wavelength bands to include for automatic selection'],
                  ['SHIFT+D','Switch between the Z and V datasets, if available for the radar.'],
                  ['CTRL+D','Switch between directory structures if you specified multiple at <b>Settings/Data storage</b>.*'],
                  ['CTRL+P','Switch between different versions of products when available.*'],
                  ['SHIFT+S/E','Influences which scan is chosen when switching to new radar. Keep using same scan/choose scan with nearest elevation angle.'],
                  ['SHIFT+H','Similar to above, but choose scan with nearest height at panel center. This mode is however also applied when moving view with storm.*'],
                  ['SHIFT+N/A/R/C','Go to normal/all/row/column mode*'],
                  ['Z/R/V/S/W/D/P/K/C/X/E/A/M/L/Q/T/Y','Change the product of the selected panel, see the <b>Products</b> tab for more information.'],
                  ['SHIFT+Q','Change a parameter for some derived products (E/A/M), but only if that product is shown in the selected panel.*'],
                  ['SHIFT+U','View the unfiltered/filtered variant of the product. The unfiltered variant might not be available.'],
                  ['SHIFT+P','Change the polarization of the product. Only for dual-polarization radars.'],
                  ['SHIFT+V/ALT+V','Apply dealiasing to the velocity field/Select which dealiasing procedures are applied'],
                  ['SHIFT+I','Apply bilinear interpolation, only available for particular products'],
                  ['SHIFT+Z','Hide/show radar data.'],
                  ['1-9, 0 and SHIFT+1-5','Change scan for the selected panel to 1-15.*'],
                  ['DOWN/UP, SHIFT+DOWN/UP','Go one scan down/up for all panels, go one scan down/up for the selected panel.'],
                  ['SHIFT+F1-F12','Save current panel configuration as panel choice 1-12.'],
                  ['F1-F12','Show panel choice 1-12.'],
                  ['ALT+F1','Show all saved panel configurations, with possibility to edit them.'],
                  ['ALT+A','Show radars for which archived data is available for the selected date, and dates for which archived data is available for the selected radar.'],
                  ['ALT+P','Show the scan angle, range, radial resolution and Nyquist velocity for all scans in the current volume.'],
                  ['SHIFT+F','View at full screen.'],
                  ['CTRL+(S/ALT+S/SHIFT+S)','Save figure/continuously save figures (until pressing CTRL+ALT+S again)/add images to animation, which can be saved after pressing CTRL+SHIFT+S again.'],
                  ['*',"See the tab <b>Extra remarks</b> for more info"]]
        for j in range(0,len(keyboard_text)):
            keyboard_layout.addRow(QLabel(keyboard_text[j][0],font=self.help_font),QLabel(keyboard_text[j][1],font=self.help_font))  
        self.helpkeyboard.setLayout(keyboard_layout)
        
    def help_tabradardata(self):
        radardata_layout=QVBoxLayout()
        radardata_text=['<b>Current data</b>:',
                  "Most users will need to download the radar data, which can be done with the methods described at <b>General</b>. The data is stored in the directories that are selected at <b>Settings/Data storage</b>.",
                  "",
                  "<b>Archived data</b>:",
                  "The following description assumes that you need to download the archived data, and do not have it already at your hard disk. If the latter is the case, then it's enough to specify the location of the data",
                  "at <b>Settings/Data storage</b>. Also mentioned here are the file formats that are supported.",
                  'KNMI:',
                  "Data must be manually downloaded from the KNMI website, which can be done via these links; <A href='https://dataplatform.knmi.nl/catalog/datasets/index.html?x-dataset=radar_tar_volume_debilt&x-dataset-version=1.0'>De Bilt</a>, <A href='https://dataplatform.knmi.nl/catalog/datasets/index.html?x-dataset=radar_tar_volume_denhelder&x-dataset-version=1.0'>Den Helder</a>, <A href='https://dataplatform.knmi.nl/catalog/datasets/index.html?x-dataset=radar_tar_vol_full_herwijnen&x-dataset-version=1.0'>Herwijnen</a>.",
                  "The downloaded data must subsequently be put in the right directory, that has the structure given at <b>Settings/Data storage</b>. As an example:",
                  "If your directory structure is NLradar/Radar_data/KNMI/RAD${radarID}_OPER_O___TARVOL__L2__${date}T000000_${date+}T000000_0001, then you put the",
                  "downloaded .tar files in the directory NLradar/Radar_data/KNMI, and subsequently unpack the .tar file in the folder that is shown when right-clicking",
                  "at the file (this folder has the format RAD${radarID}_OPER_O___TARVOL__L2__${date}T000000_${date+}T000000_0001).",
                  "When done correctly you should then be able to plot the data. The files need to be of HDF5 format, and must have a name like RAD_NL60_VOL_NA_200806220010.h5.",
                  "KMI:",
                  "There is no open source for KMI data yet, but if you do have data, then it can be read if it has one of the formats described here.",
                  "The program can read both the hdf and binary .vol files that the KMI provides, at least if they have names like: 2003061015000300z.vol/2012060714002400Z.vol/2014081000000400dBZ.vol (same for",
                  "both datasets), 2014081000000400dBZ.vol.h5 (same for both datasets), 20170615100000.rad.bewid.pvol.dbzh.scanv.hdf/20170615100000.rad.bewid.pvol.dbzh.scanz.hdf (different for both datasets).",
                  "skeyes:",
                  "There is also no open source for skeyes data yet, but the program can read the data for Zaventem if it is of hdf/hdf5 format, at least if the files have names like",
                  "20170511134500.rad.bezav.pvol.dbzh.scan_abc.hdf (in fact delivered by the KMI), and EBBR140808152208.RAW0555.h5. hdf and hdf5 format files should not be put in the same folders,",
                  "because they need to be treated in different ways!",
                  "DWD:",
                  "There is also no open source for archived DWD data yet, but if you do have data, then it can be read if it is of BUFR format, with filenames such as",
                  "sweep_vol_v_0-20190809150057_10605--buf.",
                  "TU Delft:",
                  "Data can be manually downloaded from the <A href='https://opendap.tudelft.nl/thredds/catalog/IDRA/catalog.html'>TU Datacenter</a>. You need to pick the 'near_range', 'standard_range' or 'far_range' NetCDF files,",
                  "and download them into the right directory, with structure given at <b>Settings/Data storage</b>. The program assumes that at most one such NetCDF file is available per date."]
        
        for j in range(0,len(radardata_text)):
            qlabel=QLabel(radardata_text[j],font=self.help_font); qlabel.setOpenExternalLinks(True)
            radardata_layout.addWidget(qlabel)
        radardata_layout.addStretch(1)
        self.helpradardata.setLayout(radardata_layout)
        
    def help_tabproducts(self):
        products_layout=QFormLayout()
        products_text=[['Z/V/S/W/D/C/X/P/K','Reflectivity, radial velocity, storm-relative radial velocity, spectrum width, differential reflectivity, correlation coefficient, linear depolarisation ratio,',
                        "differential phase and specific differential phase."],
                       ['Q/T/Y', 'Signal quality index, clutter correction, clutter phase alignment. These products are currently only available for the new KNMI radars (Herwijnen and Den Helder). '],
                       ['R',"Rain intensity, calculated using the <A href='http://glossary.ametsoc.org/wiki/Marshall-palmer_relation'>Marshall-Palmer relation</a>."],
                       ['E','Echo tops, where the height of the highest reflectivity bin that satisfies Z>=dBZ_threshold (press SHIFT+Q) is shown.',
                        'Heights are calculated assuming the 4/3 Earth radius model (Doviak and Zrnic, 1993).'],
                       ['A','Pseudo-CAPPI (Constant Altitude Plan Position Indicator) at the height given by PCAPPI height (SHIFT+Q). The PCAPPI reflectivities',
                        'are calculated by linear interpolation between the (logarithmic) reflectivities at the scan below and above the PCAPPI height. If the PCAPPI height lies below',
                        'the lowest scan or above the highest scan, then the reflectivity for the lowest scan resp. highest scan is shown.'],
                       ['M',"Maximum reflectivity for the complete volume scan, where only reflectivity bins at a height above the Zmax minimum height (SHIFT+Q) are used",
                       'for determining the maximum reflectivity. '],
                       ['L',"Vertically integrated liquid, calculated by vertical integration of <A href='http://glossary.ametsoc.org/wiki/Vertically_integrated_liquid'>this</a> relation. The reflectivity below the lowest scan",
                        'is assumed to be constant, which might not be a good assumption. One can choose (SHIFT+Q) whether to cap reflectivity at 56 dBZ in the integration, which is commonly done to reduce',
                        'contribution from hail.']]
        for j in range(0,len(products_text)):
            qlabel2=QLabel(products_text[j][1],font=self.help_font); qlabel2.setOpenExternalLinks(True)
            products_layout.addRow(QLabel(products_text[j][0],font=self.help_font),qlabel2)
            if len(products_text[j])>2:
                for i in range(2,len(products_text[j])):
                    qlabel2=QLabel(products_text[j][i],font=self.help_font); qlabel2.setOpenExternalLinks(True)
                    products_layout.addRow(QLabel('',font=self.help_font),qlabel2)
        self.helpproducts.setLayout(products_layout)        
        
    def help_tabcolortables(self):
        colortables_layout=QVBoxLayout()
        colortables_text=["The color tables can be changed manually by selecting another file under <b>Settings/Color tables</b>, or by changing the currently selected file. The files can be found in the folder",
                      "<i>NLradar/Input_files</i>.",
                      "It is recommended to edit/create them in Notepad or a comparable application, because other applications might add undesired characters. Be sure to save them as CSV, and preferentially",
                      "under a name different from the defaultLast 5 color table's name.",
                      "The colors must be given as RGB combinations with values ranging from 0 to 255. You can view the colors corresponding to RGB combinations at <A href='https://www.colorschemer.com/rgb-color-codes'>Colorschemer</a>.",
                      'When you want to have discontinuous steps in them, you must specify 2 RGB combinations per product value. This is for example the case in the default color table for the reflectivity.',
                      "By adding the line 'Step: x' in the file you can choose to use a fixed step between subsequent ticks.",
                      "If you want to exclude some product values or values arising from the choice of a fixed step size for a tick label, then you need to add a line 'Exclude for ticks: x1,x2...' (with xi product values) to the",
                      "file.",
                      "When you want to include ticks at a position different from those regarded before, then you need to add a line 'Include for ticks: x1,x2...'.",
                      'In the case of the velocity and spectrum width you can choose between the units m/s, kts, mph and km/h. The default units are kts.',
                      "Finally, you can choose to shorten the first and/or last color segment of the colorbar. This could be useful when the colorbar should support quite extreme values,",
                      "but you don't want to have for example half of the colorbar showing this extreme range. To prevent this you can add a line below or above the first or last color definition.",
                      "That line should contain a single value, and the length of the corresponding color segment will then be based on this value instead of on the actual value (which will still",
                      "be used for the ticks). This option is used in the default color table for VIL."]
        for j in range(0,len(colortables_text)):
            qlabel=QLabel(colortables_text[j],font=self.help_font); qlabel.setOpenExternalLinks(True)
            colortables_layout.addWidget(qlabel)
        colortables_layout.addStretch(1)
        self.helpcolortables.setLayout(colortables_layout)

    def help_tabsettings(self):
        settings_layout=QFormLayout()
        settings_text=[['Map:',''],
                       ['Map tiles update time', 'Map tiles are updated when the last occurrence of panning/zooming was this number of seconds ago. You might need to increase this time in case',
                        'of a slow PC, for fast panning/zooming.'],
                       ['Apply antialiasing to lines','Antialiasing causes lines to look smoother, but it comes at the cost of increased GPU usage, and therefore could decrease the speed.'],
                       ['Height rings',"In the case of products with different scans, the numbers near this rings give the approximate height (above radar antenna level) at which the center of the radar beam is scanning",
                       "(calculated assuming the 4/3 earth radius model). This is also the case for two 'plain products' (products without different scans), which are the PCAPPI and the rain intensity.",
                       "In the case of other plain products, the rings are located at distances where particular scans are at there maximum range. The number on the inner side of the ring",
                       "now gives the height of the beam center for the scan that reaches its maximum range, and the number on the outer side of the ring gives the height of the beam center",
                       "for the highest scan that scans at a greater range."],               
                       ['Show grid/height rings pan/zoom','If false, then the grid and height rings are not visible during panning/zooming, which increases the plotting speed.'],
                       ['If not, update them after x seconds','Time after which they will again be visible, measured from the last pan/zoom action.'],
                       ['Download:',''],
                       ['Network timeout','The program stops trying to reach the KNMI server after this time. It could be necessary to increase this value when you have a slow internet connection.'],
                       ['Minimum download speed','The program stops a download when the speed is lower than this minimum speed. It could be necessary to decrease this value when you have a slow internet',
                        'connection.'],
                       ['Color tables:','A black color table is shown when the selected color table has an incorrect format.',
                        'By choosing a minimum and/or maximum product value to display, you can change the range of product values that is shown without the need to update the color table.',
                        "By clicking the buttons 'Default' and 'NWS' you can choose to use a set of default color tables (my own choices) or a set of tables used (at least for most",
                        'products) in some NWS offices.'],
                       ['Miscellaneous:','- The sleep time after plotting determines the amount of time during which no new plot command can be given after a particular plot is finished. Setting this to zero',
                        "can lead to plots not appearing immediately after they have been created, but only after the last plot in a plotting series has been created. This does not occur at all",
                        "PC's however, and you should experiment a little to find a value that works at your PC. It should be the smallest time that does not lead to the problem mentioned.",
                        'Furthermore, after the calculation of a derived product this sleep time is automatically increased with a factor of 2.'],
                       ['','- Partly drawing of the screen should be enabled if it does not lead to problems, because it increases the plotting speed. With some drivers it could lead to',
                        'the problem of a flickering background (white and black) however, and in this case it should be disabled.']]
        for j in range(0,len(settings_text)):
            settings_layout.addRow(QLabel(settings_text[j][0],font=self.help_font),QLabel(settings_text[j][1],font=self.help_font))  
            if len(settings_text[j])>2:
                for i in range(2,len(settings_text[j])):
                    settings_layout.addRow(QLabel('',font=self.help_font),QLabel(settings_text[j][i],font=self.help_font))
        self.helpsettings.setLayout(settings_layout)
                                
    def help_tabextra(self):
        extra_layout=QFormLayout()
        extra_text=[['CTRL+D','A switch of directory structure will only take place when there is data available for the currently selected date.',
                     'Regarding product versions: They are available for DWD hdf5 (reflectivity+velocity) and NWS NEXRAD L2 (reflectivity) files.'],
                    ['SHIFT+H','The program determines the beam elevation for the current scan at the center of the panel. After changing from radar, or from time',
                     'when moving view with storm, it finds the scan for which the new beam elevation at the center of the panel is closest to the value it was before.'],
                    ['SHIFT+N/A/R/C','- Normal mode; any change of product, scan, polarization, filtering or dealiasing only affects the currently selected panel.',
                     '- All-panel mode; when choosing another scan, the scan for each panel is increased by the same number as the scan for the selected panel.',
                     '- Row mode; when changing a product, the panels in the same row share it too. When changing a scan, the panel in the same column shares it too.',
                     '- Column mode; when changing a product, the panel in the same column shares it too. When changing a scan, the panels in the same row share it too.',
                     '- When the number of panels is equal to 2 or 3, then column mode is used when row mode is selected, because it is likely not desired to have all panels showing the same product,',
                     'as would be the case when using row mode.',
                     '- When using row mode when changing the number of panels, then all panels in the same row get assigned the product in the leftmost panel, and all panels in the same column',
                     'get assigned the scan of the upper panel in the column. When using column mode the same happens, except that the roles of products and scans are reversed.',
                     '- When changing the polarization, filtering or dealiasing settings, then these modes have the same effect as described above for products and scans, except that using the',
                     'all-panel mode implies that the polarization, filtering or dealiasing setting for all panels is set equal to that for the selected panel.'],
                    ['1-9, 0 and SHIFT+1-5','In the case of the new radars of KNMI, the last scan is a vertical scan, which is plotted horizontal however. This means that the distance to the radar is in fact the height',
                     'above radar level.'],
                    ['SHIFT+Q','Derived products are saved to a file after their calculation. Per product for which a setting can be changed, data will be written to the file for up to 4 different values of',
                     'the setting. If you use a fifth one, then one of the previous 4 will be removed, which is the one that has been displayed the least amount of times.'],
                    ['SHIFT+V','It is here attempted to remove the typical dual PRF aliasing errors. Most C-band radars scan with two PRFs, which usually differ for even and odd radials,',
                     'in order to extend the maximum velocity that can be detected. This comes however at the cost of introducing these aliasing errors, where the measured velocity deviates by',
                     'an integer multiple of a certain velocity from the true velocity. This certain velocity is (usually) twice the low/high Nyquist velocity for that particular radial, and',
                     'this low/high Nyquist velocity can be requested by pressing ALT+P. Even and odd radials are alternately scanned with the low/high Nyquist velocity, but whether even or odd',
                     'radials are scanned with the low Nyquist velocity differs per radar (or even per scan). If you want to know which Nyquist velocity is used for a particular radial, then you should',
                     'look at the correction terms that have been applied by the dealiasing algorithm.',
                     'In the case of some radars, the radar switches from PRF halfway a radial. If this is the case, then the possible correction terms are the same for all radials, and usually based on',
                     'the low Nyquist velocity. If this is the case, then the low and high Nyquist velocities that are shown by pressing ALT+P are equal.',
                     'To increase the complexity further: The possible correction terms can for some radars take on all integer multiples of once the low Nyquist velocity, instead of twice that velocity.',
                     'This increases the number of possible correction terms substantially, and makes it more difficult to correctly dealias the velocities. The result is then often an oversmoothed velocity field.'],
                    ['CTRL+S/CTRL+ALT+S',"You can choose to use a fixed name format, which has the form radar_dataset_datetime_productsandscans. You can also choose for using your own filename, to which increasing numbers",
                     "will be appended when you pressed CTRL+ALT+S. If you want to continue from numbering in already existing files, then you can append a '#' to the filename (before a possible extension).",
                     "This also works when pressing CTRL+S.",
                     "If you want to use the fixed name format, then you should only give the file extension as input (gif, png or jpg, without dot). If you want to use your own file format, then you should",
                     "give the filename as input, including the file extension (with dot). If you don't include the file extension, then the image is saved as jpg."]]
        
        for j in range(0,len(extra_text)):
            qlabel2=QLabel(extra_text[j][1],font=self.help_font); qlabel2.setOpenExternalLinks(True)
            extra_layout.addRow(QLabel(extra_text[j][0],font=self.help_font),qlabel2)
            if len(extra_text[j])>2:
                for i in range(2,len(extra_text[j])):
                    qlabel2=QLabel(extra_text[j][i],font=self.help_font); qlabel2.setOpenExternalLinks(True)
                    extra_layout.addRow(QLabel('',font=self.help_font),qlabel2)
        self.helpextra.setLayout(extra_layout)
                                
                                

    def closeEvent(self, event=None):
        self.derivedproducts_filename_version = self.dp.filename_version
        
        settings={}
        for j in range(0,len(variables_names_raw)): 
            settings[variables_names_raw[j]]=eval(variables_names_withclassreference[j])                
        with open(settings_filename,'wb') as f:
            pickle.dump(settings,f)
            
            
        #Dump info over the volume attributes into pickle files, which will be loaded during the next startup of the program in nlr_datasourcegeneral.py
        with open(self.dsg.attributes_descriptions_filename,'wb') as f:
            pickle.dump(self.dsg.attributes_descriptions,f)
        with open(self.dsg.attributes_IDs_filename,'wb') as f:
            pickle.dump(self.dsg.attributes_IDs,f)
        with open(self.dsg.attributes_variable_filename,'wb') as f:
            pickle.dump(self.dsg.attributes_variable,f)

        # Only tell Qt to quit once every settings file has been fully written to disk. The previous order (quit()
        # called first) meant the application shutdown raced against these writes -- if the process got torn
        # down before a write completed, the result would be an incomplete/stale settings file on the next
        # startup. This is a plausible explanation for a one-off, non-reproducible crash on a later startup
        # (e.g. a missing or stale dict entry) that can't be reproduced afterwards, since a subsequent normal
        # close would simply overwrite it correctly again.
        QCoreApplication.instance().quit()
             




def main():
    app = QApplication(sys.argv)
    gui = GUI()
    
    print(app.exec_())

    # sys.exit(app.exec_())
    # gui.closeEvent()
    print('finish')
        
if __name__ == '__main__':
    main()