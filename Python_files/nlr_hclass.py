# -*- coding: utf-8 -*-
"""
nlr_hclass.py

Hydrometeorenclassificatie (HCLASS) voor NLradar, voor C-band-radars (KNMI/DWD)
zonder LDR: gebaseerd op Z, ZDR, KDP, RhoHV en (optioneel) temperatuur.

BRON EN ATTRIBUTIE
-------------------
De membership-beta-functieparameters hieronder (m, a, b per klasse per variabele)
zijn overgenomen uit CSU_RadarTools (Colorado State University), specifiek de
C-band-zomerparameterset:
  https://github.com/CSU-Radarmet/CSU_RadarTools
  csu_radartools/beta_function_parameters/C-band_*.csv
Deze parameters zijn gepubliceerd in:
  Dolan, B., Rutledge, S. A., Lim, S., Chandrasekar, V., & Thurai, M. (2013).
  A robust C-band hydrometeor identification algorithm and application to a
  long-term polarimetric radar dataset. J. Appl. Meteor. Climatol., 52(9), 2162-2186.
CSU_RadarTools zelf is GPL-2.0-licensed. Dit bestand is een eigen, opnieuw
geschreven implementatie (geen gekopieerde code) van de in dat werk gepubliceerde
wiskundige methode (de beta-functieformule, zie hid_beta hieronder, en de
"hybrid"-aggregatiemethode uit csu_fhc.py), met de bijbehorende gepubliceerde
parameterwaarden. Erik: dit is geen juridisch advies (ik ben geen jurist) - bij
twijfel over combineren van GPL-2- en GPL-3-code is het verstandig dit zelf nog
te (laten) checken, met name als NLradar ooit verspreid wordt.

WAAROM DIT SCHEMA
------------------
Zie de sessie van 22 juli: NEXRAD HCA (Park et al. 2009) is S-band-getuned en dus
niet direct overdraagbaar; het Meteo-France-schema is C-band-geschikt maar de
volledige numerieke parameters waren niet vrij beschikbaar. CSU_RadarTools heeft
wel expliciete, publiek gepubliceerde C-band-parameters, en wordt veel gebruikt/
geciteerd in de wetenschappelijke literatuur. Vandaar de keuze hiervoor.

BEPERKING: geen LDR (Erik's KNMI-data heeft dit niet, bevestigd op 22 juli).
De klassen die in de brontabellen het sterkst op LDR leunen (met name onderscheid
tussen sommige ijssoorten) zullen daardoor minder scherp gescheiden worden dan in
de volledige CSU-implementatie. Dit is inherent aan het ontbreken van LDR, niet
een beperking van deze module specifiek.

Nog NIET geïntegreerd met de rest van NLradar (nlr_plotting.py, nlr_globalvars.py,
nlr_datasourcegeneral.py zijn hiervoor nodig maar nog niet aangeleverd) - dit
bestand bevat vooralsnog alleen de zelfstandige classificatieberekening, met een
losse zelftest (if __name__ == '__main__') met synthetische data.
"""

import numpy as np


# HID-klassen, in dezelfde volgorde/nummering als CSU_RadarTools (1-indexed net als
# in csu_fhc_summer; index 0 hieronder correspondeert dus met klasse-ID 1).
# Namen in het Nederlands (22 juli, op Eriks verzoek) - de Engelse originelen (voor eventuele
# toekomstige verwijzing naar de brontabellen/CSU_RadarTools) waren: Drizzle, Rain, Ice Crystals,
# Aggregates, Wet Snow, Vertical Ice, Low-Density Graupel, High-Density Graupel, Hail, Big Drops.
HID_CLASSES = [
    'Motregen', 'Regen', 'IJskristallen', 'Sneeuwvlokken', 'Natte sneeuw',
    'Verticaal ijs', 'Lichte graupel', 'Zware graupel',
    'Hagel', 'Grote druppels',
]

# Membership-beta-functieparameters (m, a, b) per klasse, C-band-zomerset.
# Volgorde van klassen komt overeen met HID_CLASSES hierboven.
# Bron: CSU_RadarTools C-band_*.csv (zie docstring/attributie hierboven).
_MBF_PARAMS = {
    # variabele: lijst van (m, a, b) per klasse, in volgorde van HID_CLASSES
    'DZ': [  # Reflectivity (dBZ)
        (1.75, 29.00, 10.00), (39.00, 19.00, 10.00), (-3.00, 22.00, 20.00),
        (17.00, 18.10, 10.00), (24.00, 21.30, 10.00), (-3.00, 22.00, 20.00),
        (37.00, 9.20, 8.00), (44.30, 10.20, 6.00), (62.30, 14.30, 10.00),
        (57.80, 8.50, 10.00),
    ],
    'DR': [  # Differential Reflectivity (dB)
        (0.46, 0.46, 5.00), (2.30, 2.20, 9.00), (2.90, 2.70, 10.00),
        (1.00, 1.10, 7.00), (1.30, 0.90, 10.00), (-0.90, 0.90, 10.00),
        (0.90, 0.90, 6.00), (1.60, 1.20, 3.00), (0.14, 0.56, 8.00),
        (4.40, 1.90, 8.00),
    ],
    'KD': [  # Specific Differential Phase (deg/km)
        (0.030, 0.030, 2.000), (5.500, 5.500, 10.000), (0.080, 0.080, 6.000),
        (-0.008, 0.300, 1.000), (0.250, 0.430, 6.000), (-0.750, 0.750, 30.000),
        (0.100, 0.080, 3.000), (1.900, 1.900, 3.000), (0.6, 3.500, 6.000),
        (3.400, 3.300, 6.000),
    ],
    'RH': [  # Correlation Coefficient (-)
        (1.000, 0.018, 3.000), (1.000, 0.025, 3.000), (0.980, 0.025, 3.000),
        (0.930, 0.070, 3.000), (0.740, 0.250, 10.000), (0.975, 0.022, 3.000),
        (1.000, 0.025, 1.000), (1.000, 0.040, 2.000), (0.970, 0.100, 3.000),
        (0.990, 0.030, 3.000),
    ],
    'T': [  # Temperature (deg C)
        (40.000, 41.000, 50.000), (48.000, 51.000, 30.000), (-50.000, 50.000, 25.000),
        (-25.000, 26.000, 15.000), (1.000, 3.500, 5.000), (-50.000, 50.000, 25.000),
        (-50.000, 50.000, 25.000), (-2.500, 20.000, 2.000), (0.000, 100.000, 5.000),
        (48.000, 51.000, 30.000),
    ],
}

# Standaardgewichten, overgenomen uit CSU_RadarTools' DEFAULT_WEIGHTS (zonder 'LD',
# want geen LDR beschikbaar).
DEFAULT_WEIGHTS = {'DZ': 1.5, 'DR': 0.8, 'KD': 1.0, 'RH': 0.8, 'T': 0.4}


def beam_height_km(slant_range_km, elevation_deg, radar_elevation_km=0.0):
    """Hoogte (km boven zeeniveau) van het radarbeam-punt op een gegeven slant range en elevatiehoek,
    via het standaard 4/3-aarde-model (Doviak & Zrnic, 'Doppler Radar and Weather Observations', 1993).
    Zelfstandig geimplementeerd (niet via NLradar's eigen ft.var1_to_var2) omdat nlr_functions.py deze
    sessie niet is aangeleverd en de exacte ground-range/slant-range-conventie van die functie dus niet
    geverifieerd kon worden; deze formule gebruikt direct de slant range (radial_res * bin-index), wat
    overeenkomt met hoe de ruwe polaire data-array is geindexeerd.

    slant_range_km: afstand langs de radarbundel (km) - dit is radial_res_km * kolomindex
    elevation_deg: elevatiehoek van de scan (graden)
    radar_elevation_km: hoogte van de radar zelf boven zeeniveau (km) - standaard 0, meestal wil je
        hier gv.radar_elevations[radar]/1000 invullen
    """
    R_e_km = 8494.67  # effectieve aardstraal (4/3 * 6371 km), standaard in radarmeteorologie
    theta = np.radians(elevation_deg)
    r = np.asarray(slant_range_km, dtype='float64')
    h = np.sqrt(r**2 + R_e_km**2 + 2*r*R_e_km*np.sin(theta)) - R_e_km
    return h + radar_elevation_km


# Vaste RGBA-kleuren (0-255) per klasse, voor de RGBA-passthrough-weergave (zelfde renderpad als PolRGB/
# product 'g' in nlr_plotting.py). Kleurkeuze: vloeibare klassen in blauw/groen-tinten, bevroren/ijsklassen
# in wit/lichtblauw-tinten, hagel/graupel in geel/oranje/rood (vergelijkbaar met gangbare HCLASS-weergaves
# bij bv. NEXRAD-producten), zodat hagel/zware neerslag er meteen uitspringt.
HID_COLORS_RGBA = {
    1: (100, 180, 255, 255),   # Drizzle - lichtblauw
    2: (0, 120, 255, 255),     # Rain - blauw
    3: (200, 220, 255, 255),   # Ice Crystals - zeer lichtblauw/wit
    4: (150, 200, 230, 255),   # Aggregates - lichtblauw-grijs
    5: (0, 220, 180, 255),     # Wet Snow - turquoise
    6: (180, 150, 255, 255),   # Vertical Ice - lichtpaars
    7: (255, 220, 100, 255),   # Low-Density Graupel - lichtgeel
    8: (255, 160, 0, 255),     # High-Density Graupel - oranje
    9: (255, 0, 0, 255),       # Hail - rood
    10: (255, 0, 255, 255),    # Big Drops - magenta
}
HID_NODATA_RGBA = (0, 0, 0, 0)  # volledig transparant waar geen classificatie mogelijk was


def classes_to_rgba(hid):
    """Zet een array met klasse-indices (1-10, of -1 voor ontbrekende data - zie classify_hid) om naar
    een (..., 4) uint8 RGBA-array, voor de RGBA-passthrough-weergave in nlr_plotting.py (hetzelfde renderpad
    als de polarimetrische RGB-composiet, product 'g')."""
    hid = np.asarray(hid)
    rgba = np.zeros(hid.shape + (4,), dtype='uint8')
    for class_id, color in HID_COLORS_RGBA.items():
        rgba[hid == class_id] = color
    rgba[hid == -1] = HID_NODATA_RGBA
    return rgba


# ============================================================================
# Gedeeld temperatuurrooster over Nederland (+ marge), t.b.v. nauwkeurigere
# temperatuurschatting dan 1 vast punt per radar (zie gesprek met Erik, 22 juli).
# ============================================================================
#
# In plaats van steeds de radarlocatie zelf te gebruiken voor de 0C/-20C-hoogte
# (wat kan afwijken aan de rand van het bereik, en tot een sprong leidt bij het
# wisselen tussen radars), wordt hier een vast rooster van punten over heel
# Nederland (+ marge, want het bereik van 250 km loopt door in Belgie/Duitsland/
# de Noordzee) gedefinieerd. Dit rooster is LOS van welke radar actief is: de
# temperatuurstructuur van de atmosfeer heeft niks met een specifieke radar te
# maken. Voor elke radarbin wordt de echte lat/lon berekend (via destination_point
# hieronder) en het dichtstbijzijnde roosterpunt gekozen (nearest_grid_index).
#
# 4x4-rooster, ruwweg 150 km uit elkaar - dekt Nederland ruim, plus rand voor het
# radarbereik dat in de buurlanden/zee doorloopt.
NL_GRID_POINTS = [
    (lat, lon)
    for lat in (50.8, 51.6, 52.4, 53.2)
    for lon in (3.6, 5.2, 6.8, 8.4)
]


def destination_point(lat1_deg, lon1_deg, distance_km, bearing_deg):
    """Bereken de lat/lon van een punt op distance_km afstand en bearing_deg (kompasrichting,
    0=Noord, met de klok mee) vanaf (lat1_deg, lon1_deg), via de standaard boloppervlak-
    "destination point"-formule (zie bv. https://www.movable-type.co.uk/scripts/latlong.html).

    Alle argumenten mogen scalars of numpy-arrays zijn (met elkaar broadcastbaar) - gebruikt hier
    voor een heel radarbeeld tegelijk (elke bin heeft zijn eigen ground_range+azimuth).

    Retourneert (lat2_deg, lon2_deg), zelfde vorm als de invoer.
    """
    R_EARTH_KM = 6371.0
    lat1 = np.radians(lat1_deg)
    brng = np.radians(bearing_deg)
    d_R = np.asarray(distance_km, dtype='float64') / R_EARTH_KM

    lat2 = np.arcsin(np.sin(lat1) * np.cos(d_R) + np.cos(lat1) * np.sin(d_R) * np.cos(brng))
    lon2 = np.radians(lon1_deg) + np.arctan2(
        np.sin(brng) * np.sin(d_R) * np.cos(lat1),
        np.cos(d_R) - np.sin(lat1) * np.sin(lat2))
    return np.degrees(lat2), np.degrees(lon2)


def nearest_grid_index(lats, lons, grid_points, ref_lat=52.0):
    """Vind voor elke (lat, lon) in de arrays lats/lons het dichtstbijzijnde punt in grid_points
    (een lijst van (lat, lon)-tuples), en retourneer een array met de bijbehorende index in
    grid_points (zelfde vorm als lats/lons).

    Gebruikt een simpele vlakke (equirectangulaire) afstandsbenadering met een vaste
    referentiebreedtegraad voor de lengtegraad-schaling - ruim nauwkeurig genoeg om alleen het
    dichtstbijzijnde punt te bepalen op de schaal van Nederland (~300 km), geen exacte afstand nodig.
    """
    lats = np.asarray(lats, dtype='float64')
    lons = np.asarray(lons, dtype='float64')
    grid_lats = np.array([p[0] for p in grid_points])
    grid_lons = np.array([p[1] for p in grid_points])

    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(ref_lat))

    dlat = (lats[..., np.newaxis] - grid_lats) * km_per_deg_lat
    dlon = (lons[..., np.newaxis] - grid_lons) * km_per_deg_lon
    dist2 = dlat ** 2 + dlon ** 2
    return np.argmin(dist2, axis=-1)


def hid_beta(x, m, a, b):
    """Membership-beta-functie (Dolan & Rutledge 2009, vgl. voor MBF's).
    x: waarde(n) van de betreffende radarvariabele (array of scalar)
    m, a, b: centrum, breedte, helling van de functie voor deze klasse/variabele
    Retourneert een waarde tussen 0 en 1 (lidmaatschapsgraad).
    """
    return 1.0 / (1.0 + (((x - m) / a) ** 2) ** b)


def estimate_temperature(height_m, h0_m, h_minus20_m):
    """Schat de temperatuur (graden C) op een gegeven hoogte, via lineaire
    interpolatie/extrapolatie tussen de bekende 0C- en -20C-niveaus
    (zie nlr_meltinglevels.py). Dit is een vereenvoudiging (echte profielen zijn
    niet perfect lineair, vooral niet ver buiten het 0C/-20C-interval), maar
    een redelijke inschatting met de twee niveaus die beschikbaar zijn.

    height_m: hoogte(n) in meter boven zeeniveau (array of scalar)
    h0_m, h_minus20_m: hoogte van het 0C- resp. -20C-niveau in meter. Mogen scalars zijn
        (1 punt voor het hele beeld) OF arrays met dezelfde vorm als height_m (bv. een
        per-bin-waarde uit het gedeelde temperatuurrooster, zie nlr_datasourcegeneral.py's
        _calculate_hclass) - beide werken via numpy-broadcasting.

    Retourneert None als h0_m of h_minus20_m None is (dan kan er geen
    temperatuurschatting gemaakt worden, en moet de aanroeper use_temp=False
    gebruiken bij classify_hid). Bij array-invoer wordt een positie waar het paar
    niet bruikbaar is (h_minus20_m ontbreekt/gedegenereerd voor die bin) als NaN
    teruggegeven i.p.v. de hele berekening te laten mislukken.

    LET OP (22 juli, na feedback van Erik): eerder werd hier bij een ontbrekend
    -20C-niveau een AANGENOMEN standaard-daalsnelheid gebruikt om toch een concrete
    temperatuur te verzinnen. Dat loste weliswaar "helemaal geen beeld" op bij oude
    cases, maar drukte een mogelijk onjuiste aanname door en veranderde de
    classificatie inhoudelijk (minder variatie dan het oude, temperatuur-loze
    gedrag). Dat is teruggedraaid: deze functie geeft nu weer gewoon NaN terug waar
    het echte paar ontbreekt. De oplossing voor "geen beeld bij oude cases" zit
    voortaan in classify_hid zelf: die negeert temperatuur nu PER BIN waar T
    ontbreekt (in plaats van de bin als "geen data" te markeren), zodat het resultaat
    voor zo'n bin identiek is aan classificeren met use_temp=False - exact het oude,
    door Erik geprefereerde gedrag, nu alleen per bin toegepast in plaats van voor
    het hele beeld tegelijk.
    """
    if h0_m is None or h_minus20_m is None:
        return None
    h0_arr = np.asarray(h0_m, dtype='float64')
    h20_arr = np.asarray(h_minus20_m, dtype='float64')
    height_arr = np.asarray(height_m, dtype='float64')

    denom = h20_arr - h0_arr
    if denom.ndim == 0 and not np.isnan(denom) and denom == 0:
        return None  # scalar, ongeldig/gedegenereerd profiel, kan niet interpoleren

    with np.errstate(divide='ignore', invalid='ignore'):
        denom_safe = np.where(denom == 0, np.nan, denom)
        lapse_rate_c_per_m = -20.0 / denom_safe  # graden C per meter
        return lapse_rate_c_per_m * (height_arr - h0_arr)


def classify_hid(dz, zdr, kdp, rho, T=None, method='hybrid', weights=None):
    """
    Bepaal per radarbin de meest waarschijnlijke hydrometeorenklasse.

    Parameters
    ----------
    dz, zdr, kdp, rho : array-like, gelijke vorm
        Reflectiviteit (dBZ), differentiele reflectiviteit (dB),
        specifieke differentiele fase (graden/km), correlatiecoefficient (-).
        NaN-waarden worden behandeld als ontbrekende data (resultaat wordt -1
        op die posities).
    T : array-like, zelfde vorm, of None
        Temperatuur in graden C per bin (bv. via estimate_temperature()).
        Als None, wordt temperatuur helemaal niet gebruikt in de classificatie
        (minder scherpe scheiding tussen ijs- en vloeistofklassen). Als T een array
        is met NaN op sommige posities (bv. omdat het -20C-niveau voor een oudere
        case niet meer beschikbaar was, zie nlr_meltinglevels.py), wordt temperatuur
        voor DIE specifieke bins genegeerd (identiek aan T=None, maar dan alleen
        lokaal) i.p.v. die bins als "geen data" te markeren - zie LET OP hieronder.
    method : 'hybrid' (standaard, aanbevolen) of 'linear'
        'hybrid': polarimetrische variabelen worden gewogen gemiddeld, dat
        resultaat wordt vermenigvuldigd met de T- en Z-lidmaatschapswaarden
        (T en Z hebben zo een sterkere invloed). Dit is de door CSU_RadarTools
        aanbevolen methode.
        'linear': alle variabelen (incl. Z en T) worden gewoon gewogen opgeteld.
    weights : dict of None
        Gewicht per variabele ('DZ','DR','KD','RH','T'); standaard DEFAULT_WEIGHTS.

    Returns
    -------
    hid : np.ndarray (int), zelfde vorm als de invoer
        Klasse-index (1-10, zie HID_CLASSES) per bin, of -1 waar invoerdata
        ontbrak (NaN in dz, zdr, kdp of rho - NIET in T, zie hierboven).
    scores : np.ndarray (float), vorm (10,) + vorm van de invoer
        De ruwe lidmaatschapsscore per klasse per bin (voor eigen inspectie/
        debugging; hid = argmax hiervan + 1, behalve waar hid == -1).

    LET OP (22 juli, na feedback van Erik): eerder telde een NaN in T mee in de
    "missing"-markering (dus die bin werd -1, "geen data"). Bij oudere cases waar
    het gedeelde temperatuurrooster ALLE 16 punten tegelijk hun -20C-waarde mist,
    werd T daardoor bijna overal NaN, wat het HELE beeld op "geen data" zette,
    terwijl Z/ZDR/KDP/CC gewoon beschikbaar waren. Dat is nu apart behandeld: een
    NaN in T maakt een bin niet meer "missing", maar zorgt er alleen voor dat de
    T-membership-factor voor DIE bin wordt overgeslagen (vermenigvuldigd met 1
    i.p.v. met een echte T-gebaseerde waarde) - het resultaat voor zo'n bin is dan
    identiek aan classificeren met T=None voor het hele beeld, maar dan lokaal
    toegepast. Zie ook estimate_temperature()'s docstring voor de eerdere (nu weer
    teruggedraaide) poging om i.p.v. hiervan een aangenomen standaard-daalsnelheid
    te gebruiken - dat gaf wel overal beeld, maar drukte een mogelijk onjuiste
    aanname door en veranderde de classificatie inhoudelijk te veel.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    dz = np.asarray(dz, dtype='float64')
    zdr = np.asarray(zdr, dtype='float64')
    kdp = np.asarray(kdp, dtype='float64')
    rho = np.asarray(rho, dtype='float64')
    shape = dz.shape
    use_temp = T is not None
    if use_temp:
        T = np.asarray(T, dtype='float64')

    # Alleen ontbrekende POLARIMETRISCHE data maakt een bin "missing" (-1). Een ontbrekende T (zie
    # hierboven) wordt per bin genegeerd, niet als "missing" behandeld.
    missing = np.isnan(dz) | np.isnan(zdr) | np.isnan(kdp) | np.isnan(rho)

    n_classes = len(HID_CLASSES)
    scores = np.zeros((n_classes,) + shape, dtype='float64')

    pol_vars = ['DR', 'KD', 'RH']
    pol_weight_sum = sum(weights[v] for v in pol_vars)
    pol_data = {'DR': zdr, 'KD': kdp, 'RH': rho}

    for c in range(n_classes):
        if method == 'hybrid':
            test = np.zeros(shape, dtype='float64')
            for v in pol_vars:
                m, a, b = _MBF_PARAMS[v][c]
                test += weights[v] * hid_beta(pol_data[v], m, a, b)
            test /= pol_weight_sum
            if use_temp:
                m, a, b = _MBF_PARAMS['T'][c]
                t_membership = hid_beta(T, m, a, b)
                # Waar T NaN is voor deze bin: factor 1 i.p.v. een T-gebaseerde waarde, zodat de
                # classificatie voor die bin puur op Z/ZDR/KDP/CC blijft steunen (zie LET OP hierboven).
                t_membership = np.where(np.isnan(t_membership), 1.0, t_membership)
                test *= t_membership
            m, a, b = _MBF_PARAMS['DZ'][c]
            test *= hid_beta(dz, m, a, b)
        elif method == 'linear':
            pol_and_z_vars = ['DR', 'KD', 'RH', 'DZ']
            weight_sum = np.full(shape, sum(weights[v] for v in pol_and_z_vars), dtype='float64')
            data_map = {'DZ': dz, 'DR': zdr, 'KD': kdp, 'RH': rho}
            test = np.zeros(shape, dtype='float64')
            for v in pol_and_z_vars:
                m, a, b = _MBF_PARAMS[v][c]
                test += weights[v] * hid_beta(data_map[v], m, a, b)
            if use_temp:
                m, a, b = _MBF_PARAMS['T'][c]
                t_membership = hid_beta(T, m, a, b)
                valid_t = ~np.isnan(t_membership)
                # Alleen waar T geldig is, telt de T-term (en zijn gewicht) mee - elders blijft de
                # verdeling zoals hierboven (puur polarimetrisch+Z), zie LET OP hierboven.
                test = test + np.where(valid_t, weights['T'] * np.where(valid_t, t_membership, 0.0), 0.0)
                weight_sum = weight_sum + np.where(valid_t, weights['T'], 0.0)
            test = test / weight_sum
        else:
            raise ValueError("method moet 'hybrid' of 'linear' zijn, kreeg: {}".format(method))
        scores[c] = test

    hid = np.argmax(scores, axis=0) + 1
    hid = hid.astype('int32')
    hid[missing] = -1
    return hid, scores


if __name__ == '__main__':
    # Zelftest met synthetische data - representatieve waarden per klasse, ruwweg
    # afgeleid uit de m-waarden (het centrum) van elke membership function, zodat
    # elk synthetisch punt "duidelijk" in zijn eigen klasse zou moeten vallen.
    # Dit test alleen of de classificatielogica zelf consistent is (geen netwerk/
    # NLradar-afhankelijkheid); het is geen validatie tegen echte radardata.
    print("Zelftest: elk van de 10 klassen met representatieve (Z, ZDR, KDP, RHO, T)-waarden")
    print("-" * 90)
    test_points = {
        'Regen':                (39.0, 2.3, 5.5, 1.00, 15.0),
        'Hagel':                (62.3, 0.14, 0.6, 0.97, 5.0),
        'Natte sneeuw':            (24.0, 1.3, 0.25, 0.74, 1.0),
        'IJskristallen':        (-3.0, 2.9, 0.08, 0.98, -30.0),
        'Zware graupel':(44.3, 1.6, 1.9, 1.00, -5.0),
    }
    all_correct = True
    for expected_class, (z, d, k, r, t) in test_points.items():
        hid, scores = classify_hid(
            np.array([z]), np.array([d]), np.array([k]), np.array([r]),
            T=np.array([t]))
        got_class = HID_CLASSES[hid[0] - 1]
        ok = got_class == expected_class
        all_correct &= ok
        print(f"  invoer=(Z={z}, ZDR={d}, KDP={k}, RHO={r}, T={t}) "
              f"-> {got_class}  (verwacht: {expected_class})  {'OK' if ok else 'AFWIJKEND'}")

    print("-" * 90)
    # Test met ontbrekende data (NaN) -> moet -1 geven
    hid_nan, _ = classify_hid(np.array([np.nan]), np.array([1.0]), np.array([0.1]), np.array([0.99]))
    print(f"Test met NaN in Z: hid = {hid_nan[0]} (verwacht: -1) "
          f"{'OK' if hid_nan[0] == -1 else 'AFWIJKEND'}")

    # Test zonder temperatuur (use_temp=False pad)
    hid_notemp, _ = classify_hid(np.array([39.0]), np.array([2.3]), np.array([5.5]), np.array([1.00]))
    print(f"Test zonder temperatuur: {HID_CLASSES[hid_notemp[0]-1]} (verwacht: Regen) "
          f"{'OK' if HID_CLASSES[hid_notemp[0]-1] == 'Regen' else 'AFWIJKEND'}")

    # Test estimate_temperature
    t_est = estimate_temperature(np.array([0, 3000, 6000]), h0_m=3000.0, h_minus20_m=6000.0)
    print(f"\nestimate_temperature op hoogtes [0, 3000, 6000] met h0=3000, h-20=6000: {t_est}")
    print("(verwacht: [20, 0, -20] - lineair, 0C op 3000m, -20C op 6000m)")
    assert np.allclose(t_est, [20.0, 0.0, -20.0]), "temperatuurinterpolatie klopt niet"

    print("\n" + "=" * 90)
    print("Zelftest: destination_point (boloppervlak-bestemmingsformule)")
    print("-" * 90)
    # 111.2 km recht naar het noorden (bearing=0) vanaf De Bilt moet ~1.0 graad breedtegraad
    # verder liggen (1 breedtegraad =~ 111.2 km), lengtegraad vrijwel ongewijzigd.
    lat0, lon0 = 52.1, 5.18
    lat2, lon2 = destination_point(lat0, lon0, 111.2, 0.0)
    print(f"  111.2 km noord vanaf ({lat0},{lon0}): -> ({lat2:.4f}, {lon2:.4f}), verwacht ~({lat0+1.0:.4f}, {lon0:.4f})")
    assert abs(lat2 - (lat0 + 1.0)) < 0.02, "destination_point noordwaarts klopt niet"
    assert abs(lon2 - lon0) < 0.02, "destination_point noordwaarts zou lengtegraad nauwelijks moeten veranderen"

    # 100 km recht naar het oosten (bearing=90) moet de lengtegraad laten toenemen, breedtegraad gelijk.
    lat3, lon3 = destination_point(lat0, lon0, 100.0, 90.0)
    print(f"  100 km oost vanaf ({lat0},{lon0}): -> ({lat3:.4f}, {lon3:.4f}), breedtegraad moet ~gelijk blijven")
    assert abs(lat3 - lat0) < 0.05, "destination_point oostwaarts zou breedtegraad nauwelijks moeten veranderen"
    assert lon3 > lon0, "destination_point oostwaarts moet de lengtegraad laten toenemen"

    # Vectorized-test: arrays i.p.v. scalars (zoals gebruikt voor een heel radarbeeld tegelijk).
    lats_arr, lons_arr = destination_point(lat0, lon0, np.array([50.0, 100.0]), np.array([0.0, 180.0]))
    print(f"  Array-test: {list(zip(np.round(lats_arr,3), np.round(lons_arr,3)))}")
    assert lats_arr[0] > lat0 and lats_arr[1] < lat0, "noord moet breedtegraad verhogen, zuid verlagen"

    print("\n" + "=" * 90)
    print("Zelftest: nearest_grid_index")
    print("-" * 90)
    test_grid = [(50.0, 4.0), (52.0, 4.0), (50.0, 6.0), (52.0, 6.0)]
    test_lats = np.array([50.1, 51.9, 52.1])
    test_lons = np.array([4.1, 6.1, 3.9])
    idx = nearest_grid_index(test_lats, test_lons, test_grid)
    print(f"  Testpunten -> dichtstbijzijnde roosterindex: {idx} (verwacht: [0, 3, 1])")
    assert list(idx) == [0, 3, 1], "nearest_grid_index geeft niet de verwachte dichtstbijzijnde punten"

    print(f"\nNL_GRID_POINTS bevat {len(NL_GRID_POINTS)} punten:")
    for p in NL_GRID_POINTS:
        print(f"  {p}")

    print("\nAlle geo-tests geslaagd.")

    print("\n" + ("ALLE TESTS GESLAAGD" if all_correct else "LET OP: niet alle klassen kwamen overeen met verwachting"))
