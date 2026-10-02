# Copyright (C) 2016-2024 Bram van 't Veen, bramvtveen94@hotmail.com
# Distributed under the GNU General Public License version 3, see <https://www.gnu.org/licenses/>.

import sys
import os
opa=os.path.abspath
import numpy as np
import datetime as dtime
from unidecode import unidecode

import nlr_functions as ft



python_version=str(sys.version_info[0])+'.'+str(sys.version_info[1])+'.'+str(sys.version_info[2])
    
# Installeerbare (PyInstaller-)versie: het programma staat in een alleen-lezen map (Program Files), dus alles wat
# NLradar wegschrijft (instellingen, caches, radardata, uitvoer) gaat dan naar %LOCALAPPDATA%\NLradar (userdir).
# Bij het gewoon draaien vanuit de broncode (python nlr.py) is userdir gelijk aan programdir: niets verandert.
frozen = getattr(sys, 'frozen', False)
if frozen:
    programdir=opa(os.path.dirname(sys.executable))
    userdir=opa(os.path.join(os.environ.get('LOCALAPPDATA') or os.path.expanduser('~'), 'NLradar'))
    # Brams code leest een aantal bestanden relatief t.o.v. de werkmap Python_files (Tables, _data, gifsicle)
    os.chdir(opa(programdir+'/Python_files'))
else:
    programdir=opa(os.path.dirname(os.path.dirname(__file__)))
    userdir=programdir
for _d in ('Generated_files', 'Radar_data', 'Output_files'):
    os.makedirs(opa(userdir+'/'+_d), exist_ok=True)
sys.path.append(opa(programdir+'/Python_files/vispy'))

unknown_products_logfile = opa(userdir+'/unknown_products_log.txt')
def log_product_check(message):
    """Appends a timestamped line to unknown_products_log.txt, both for detected unrecognized products/datasets and for
    confirmations that a check has run without finding anything unrecognized."""
    try:
        with open(unknown_products_logfile, 'a', encoding='utf-8') as f:
            f.write(dtime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')+' - '+message+'\n')
    except Exception:
        pass


radars_all, data_sources_all = [], []
radars = {}
data_sources, data_readsources = {}, {}
radar_ids, radarcoords, radar_elevations, radar_towerheights, radar_bands = {}, {}, {}, {}, {}
with open(programdir+'/Input_files/radars_eu.txt', 'r', encoding='utf-8') as f:
    data = ft.list_data(f.read(), '\t')
with open(programdir+'/Input_files/radars_us.txt', 'r', encoding='utf-8') as f:
    data += ft.list_data(f.read(), '\t')

for i,row in enumerate(data):
    if len(row) == 1:
        if ':' in row[0]:
            data_source, data_readsource = row[0].split(':')
        else:
            data_source = data_readsource = row[0]
        data_sources_all.append(data_source)
    else:
        radar = row[0]
        radars_all.append(radar)
        radars[data_source] = radars.get(data_source, []) + [radar]
        data_sources[radar] = data_source
        data_readsources[radar] = data_readsource
        radar_ids[radar] = row[1]*(row[1] != '.')
        radarcoords[radar] = [float(row[2]), float(row[3])]
        radar_elevations[radar] = int(row[5])
        radar_towerheights[radar] = int(row[4])-int(row[5])
        radar_bands[radar] = row[6]

# Radars die (tijdelijk) geen data meer leveren, bv. omdat het station is verplaatst. Ze blijven wel gewoon in
# radars_eu.txt/radars_us.txt staan, en worden hier alleen uit radars_all/radars[source] gefilterd, zodat ze
# niet meer op de kaart verschijnen en niet meer via mouseover/N-toets/CTRL+pijltjes bereikbaar zijn. Een radar
# weer aanzetten is dus een kwestie van de naam hieronder weghalen.
radars_disabled = {'Emden', 'De Bilt', 'Cabauw', 'Zaventem'}
radars_all = [radar for radar in radars_all if radar not in radars_disabled]
radars = {source: [radar for radar in radar_list if radar not in radars_disabled] for source, radar_list in radars.items()}

character_map = {'\u00F6':'oe','\u00FC':'ue'}
def replace_chars(string):
    for i,j in character_map.items():
        string = string.replace(i, j)
    return string
# Umlauts are not replaced by the desired characters, as ö/ü becomes o/u instead of oe/ue. So they are treated separately.
radars_ascii_names = {radar:unidecode(replace_chars(radar)) for radar in radars_all}



radars_with_datasets=radars['KMI']+radars['DWD']+radars['IMGW']+radars['DMI']+radars['UKMO']+radars['Austro Control']+radars['DHMZ']
#Radars for which the data is distributed over two datasets; one with large maximum range but small Nyquist velocity, and the other vice versa.
radars_with_double_volume=('Den Helder','Herwijnen') 
#Radars for which the volume can devided into two parts, with slightly different scans. This is the case for the new radars of the KNMI.
radars_with_onefileperdate=('Cabauw',)
radars_with_adjustable_startazimuth=('Cabauw',)
        
radarsources_dirs_Default = {}
default_basedir = userdir+'/Radar_data'
for source in radars:
    if not radars[source]: continue # kan gebeuren als alle radars van deze bron in radars_disabled staan
    datasets = np.unique(np.concatenate([['Z','V'] if j in radars_with_datasets else [''] for j in radars[source]]))
    for j in datasets:
        key = source+f'_{j}'*len(j)
        radarsources_dirs_Default[key] = '${basedir}/'+source.replace(' ','_')+'/'
        if source == 'KNMI':
            radarsources_dirs_Default[key] += 'RAD${radarID}_OPER_O___TARVOL__L2__${date}T000000_${date+}T000000_0001'
        elif source == 'DWD':
            radarsources_dirs_Default[key] += '${date}/${radar}_'+j+'/${time60}-${time60+}'
        else:
            radarsources_dirs_Default[key] += '${date}/${radar}'+f'_{j}'*len(j)
derivedproducts_dir_Default=default_basedir+'/Derived_products'

intervals_autodownload={'KNMI':300,'KMI':300,'skeyes':300,'VMM':300,'DWD':300,'TU Delft':300,'IMGW':300,'SHMU':300,'DHMZ':300,'DMI':300,'CHMI':300,'Microstep-MIS':300,'NWS':300,'ARRC':300,'Météo-France':300,'FMI':300,'ESTEA':300,'Meteo Romania':300, 'Geosphere Austria':300, 'SHMI':300}
default = list(range(0, 300, 60))
timeoffsets_autodownload={'KNMI':[75,120,180,240],'KMI':[75,120,180,240],'skeyes':[75,120,180,240],'VMM':[75,120,180,240],'DWD':[45,105,165,225,285],'TU Delft': [45,105,165,225,285],'IMGW':default,'SHMU':default,'DHMZ':default,'DMI':[135,180,240,300],'CHMI':default,'Microstep-MIS':default,'NWS':list(range(0, 300, 30)),'ARRC':[0],'Météo-France':default,'FMI':default,'ESTEA':default,'Meteo Romania':default, 'Geosphere Austria':default, 'SHMI':default}
multifilevolume_autodownload={'KNMI':False,'KMI':True,'skeyes':True,'VMM':True,'DWD':True,'TU Delft':False,'IMGW':True,'DMI':False,'CHMI':False,'NWS':False,'ARRC':False,'Météo-France':True,'FMI':False}
fileperscan_autodownload={'KNMI':False,'KMI':False,'skeyes':False,'VMM':False,'DWD':True,'TU Delft':False,'IMGW':False,'DMI':False,'CHMI':False,'NWS':False,'ARRC':False,'Météo-France':True,'FMI':False}
api_keys = {'KNMI': ['opendata', 'sfcobs'],
            'DMI': ['radardata'],
            'Météo-France': ['radardata'],
            'MapTiler': ['maps'],
            'MeteoGate': ['radardata']}
max_download_errors_tooslow = 2 #Attempts to download are aborted when the download is too slow for more than max_download_errors_tooslow times.
max_download_errors_nottooslow = 1 #Attempts to download are aborted when an exception different from the TooSlowException is encountered more
#than max_download_errors_nottooslow times.

volume_timestep_radars={j:5 for j in radars_all} # Typical timestep between radar volumes in minutes
for j in radars['IMGW']+radars['DMI']:
    volume_timestep_radars[j]=10
# volume_timestep_radars['Cabauw'] = 1

volume_attributes_all = ('scannumbers_all','scanangles_all','radial_bins_all','radial_res_all','scans_doublevolume',
                        'nyquist_velocities_all_mps','low_nyquist_velocities_all_mps','high_nyquist_velocities_all_mps',
                        'radial_range_all','scanangles_all_m')
volume_attributes_save = [j for j in volume_attributes_all if not j in ('radial_range_all','scanangles_all_m')]
#Attributes that can be different for each import product, and are assigned values by using 'exec' in nlr_importdata.py
volume_attributes_p = ('scannumbers_all','scanangles_all','radial_bins_all','radial_res_all')
                    
# 'uh' (POH) gebruikt een TWEE-karakter code, net als 'uv' - want alle 26 losse letters a-z zijn al
# bezet (23 als product hieronder, plus F/N als losse globale sneltoetsen elders in nlr.py), dus er is
# geen vrije losse letter meer over. 'u' is de enige letter die zelf nog GEEN losse sneltoets heeft
# (alleen als eerste karakter van de bestaande 'uv'-sequentie), dus een nieuwe 'u'+X-sequentie
# introduceert geen dubbelzinnigheid met een reeds bestaande losse letter-sneltoets.
# 'eb' (Echo base, 24 juli 2026, op Eriks verzoek - spiegelbeeld van 'e'/echotops) gebruikt bewust GEEN
# eigen losse letter (allemaal al bezet, zie hierboven bij 'uh') en ook geen tweede 'u'+X-sequentie zoals
# 'uh'/'uv' - want 'e' EN 'b' zijn beide al zelfstandige, veelgebruikte 1-letter-sneltoetsen (ETH resp.
# POSH), dus een 'E,B'-tweestapsreeks zou daarmee kunnen interfereren. In plaats daarvan is 'eb' bewust
# UITGESLOTEN van de automatische per-letter-sneltoetsenlus in nlr.py (products_alt_only_shortcuts) en
# alleen bereikbaar via ALT+E (aparte, losse QShortcut in nlr.py, net als ALT+M/ALT+V).
# 'vd' (VILD, VIL Density = VIL/ETH, 24 juli 2026) is een 'dependent' product (zie plain_products_functions
# in nlr_derived_plain.py) net als 'r' - hangt af van twee AL berekende bases (VIL en ETH bij vaste preset-1
# parameters). Zelfde reden als bij 'eb' hierboven: 'v' EN 'l' EN 'd' zijn alle drie al zelfstandige
# 1-letter-sneltoetsen, dus VILD is bewust uitgesloten van de letter-sneltoetsenlus, alleen ALT+L.
products_all=('z','a','m','h','r','e','eb','l','v','uv','s','w','p','k','c','d','x','q','t','y','i','g','j','o','b','uh','vd','si','zc')
# 'si' (SHI, 27 juli 2026): net als 'eb'/'vd' UITGESLOTEN van de automatische per-letter-sneltoetsenlus
# (zie hieronder in nlr.py). Reden: de lus zou 'S,I' proberen te binden, maar 'S' is al een LOSSE,
# op zichzelf staande 1-letter-sneltoets (SRV) - exact dezelfde soort Qt-shortcut-ambiguiteit die
# eerder al 'eb' (E+B, beide al losse letters) dwong tot een eigen ALT+X-toets i.p.v. de automatische
# lus (zie de toelichting daar). SHI krijgt daarom een eigen ALT+S-QShortcut in nlr.py.
products_alt_only_shortcuts = ('eb','vd','si','zc') # Producten die NIET via de normale letter(reeks)-sneltoetsenlus lopen, alleen via hun eigen ALT+X-QShortcut (zie hierboven).
# 'zc' (ZDR-kolomdiepte, 27 juli 2026): zelfde soort uitzondering - 'Z' is al een losse 1-letter-
# sneltoets (Reflectivity), dus 'Z,C' zou dezelfde Qt-shortcut-ambiguiteit geven. Eigen ALT+Z-QShortcut.
#i_p = import_products
i_p={'z':'z','r':'z','v':'v','uv':'v','s':'v','w':'w','d':'d','p':'p','k':'k','c':'c','x':'x','q':'q','t':'t','y':'y','i':'i','e':'z','eb':'z','a':'z','m':'z','h':'z','l':'z','g':'z','j':'z','o':'z','b':'z','uh':'z','vd':'z','si':'z','zc':'z'}
#Unfiltered products have the letter 'u' prepended
productnames_KNMI={'z':'Z','v':'V','w':'W','d':'ZDR','p':'PhiDP','k':'KDP','c':'RhoHV','q':'SQI','t':'CCOR','y':'CPA','uz':'uZ','ud':'uZDR','up':'uPhiDP'}
#There are 3 different name formats for KMI data, and each name format corresponds to a different file extension
productnames_KMI={'hdf':{'z':'dbzh','v':'vrad','w':'wrad'},'h5':{'z':'dBZ','v':'V','w':'W','d':'ZDR','p':'PhiDP','k':'KDP','c':'RhoHV','uz':'dBuZ','up':'uPhiDP'},'vol':{'z':['z','Z','dBZ'],'v':['v','V'],'w':['w','W'],'d':'ZDR','p':'PhiDP','k':'KDP','c':'RhoHV','uz':'dBuZ','up':'uPhiDP'}}
productnames_DWD={'hd5':{
    'z':'dbzh',
    'v':['vradh','tv'],
    'c':['rhohv', 'urhohv'],
    'p':'uphidp',
    'd':['zdr', 'uzdr', 'attcorrzdrcorr'],
    # Note: DWD does not publish a native KDP/KDPCorr file (confirmed by checking their open data file
    # listings), so 'k' is intentionally absent here. It is instead derived from the 'p' (PHIDP) files,
    # via aliasing in nlr_datasourcespecific.py (Source_DWD.get_file_availability_info) and the actual
    # KDP retrieval computation in nlr_importdata.py (DWD_odimh5.compute_kdp_from_phidp).
},
'buf.bz2':{'z':'z','v':'v'},
'buf':{'z':'z','v':'v'}}
productnames_TUDelft={'z':'equivalent_reflectivity_factor','v':'radial_velocity','w':'spectrum_width','d':'differential_reflectivity','p':'differential_phase','x':'linear_depolarisation_ratio'}
productnames_Leonardo={'z':['z', 'dBZ'],'v':['v', 'V'],'w':'W','d':'ZDR','p':'PhiDP','k':'KDP','c':'RhoHV','uz':'dBuZ','up':'uPhiDP','uk':'uKDP'}
productnames_DMI={'z':'DBZH','v':'VRAD','w':'WRAD','d':'ZDR','c':'RHOHV','p':'PHIDP','x':'LDR'}
productnames_AustroControl={'z':'PARA01','v':'PARA09','p':'PARA03','c':'PARA04','d':'PARA11','k':'PARA12'}
productnames_CHMI={'z':'PAG','uz':'PAJ','v':'PAH','w':'PAI','d':'PAK','c':'PAL','p':'PAM'}
productnames_SHMU={'z':'PAG','uz':'PAJ','v':'PAH','w':'PAI','d':'PAK','c':'PAL','p':'PAQ','k':'PAR'}
productnames_MeteoRomania={'z':'dBZ','v':'V','d':'ZDR','c':'RhoHV','k':'KDP'}
productnames_UKMO={'z':'REF','v':'VEL','w':'SW','d':'ZDR','c':'RHO','p':'PHI','q':'SQI','i':'CI','y':'CPA'}
productnames_NEXRAD={'z':'REF','v':'VEL','w':'SW','d':'ZDR','c':'RHO','p':'PHI'}
productnames_ARRC={'z':'DBZ','v':'VEL','w':'WIDTH','d':'ZDR','c':'RHOHV','p':'PHIDP'}


productnames={'z':'Z','a':'PCAPPI','m':'Zmax','h':'CMH','r':'RI','v':'V','uv':'uV','s':'SRV','w':'W','p':'PhiDP','k':'KDP','c':'CC','x':'LDR','d':'ZDR','e':'ETH','eb':'EB','l':'VIL','q':'SQI','t':'CCOR','y':'CPA','i':'CI','g':'PolRGB','j':'HCLASS','o':'MESH','b':'POSH','uh':'POH','vd':'VILD','si':'SHI','zc':'ZDRCOL'}
productnames_cmaps={'z':'Z','a':'PCAP','m':'Zmax','h':'CMH','r':'RI','v':'V','uv':'V','s':'SRV','w':'W','p':'PhiDP','k':'KDP','c':'CC','x':'LDR','d':'ZDR','e':'ETH','eb':'EB','l':'VIL','q':'SQI','t':'CCOR','y':'CPA','i':'CI','g':'PolRGB','j':'HCLASS','o':'MESH','b':'POSH','uh':'POH','vd':'VILD','si':'SHI','zc':'ZDRCOL'}
# 'zc' (ZDR-kolomdiepte, 27 juli 2026): diepte (km) van de ZDR-kolom boven het 0C-niveau (Kumjian &
# Ryzhkov 2008 / Snyder et al. 2015) - zie calculate_ZDRcol in nlr_derived_plain.py.
productnames_cmapstab={'z':'Reflectivity','a':'Pseudo CAPPI','m':'Maximum reflectivity','h':'Center of mass height','r':'Rain intensity','v':'Velocity','uv':'Unfiltered velocity','s':'Storm-relative velocity','w':'Spectrum width','p':'Differential phase','k':'Specific differential phase','c':'Correlation coefficient','x':'Linear depolarisation ratio','d':'Differential reflectivity','e':'Echo top height','eb':'Echo base height','l':'Vertically integrated liquid','q':'Signal quality index','t':'Clutter correction','y':'Clutter phase alignment','i':'Clutter indicator','g':'Polarimetric RGB composite (Z/CC/ZDR)','j':'Hydrometeor classification (C-band, Dolan et al. 2013, geen LDR)','o':'Maximum Estimated Size of Hail (MESH, Witt et al. 1998)','b':'Probability of Severe Hail (POSH, Witt et al. 1998)','uh':'Probability of Hail (POH, Waldvogel 1979 / Holleman 2001-herijking)','vd':'VIL Density (VILD = VIL/ETH, Amburn & Wolf 1997)','si':'Severe Hail Index (SHI, Witt et al. 1998) - tussenproduct van MESH/POSH','zc':'ZDR-kolomdiepte boven 0C-niveau (Kumjian & Ryzhkov 2008)'}
productunits_default={'z':'dBZ','a':'dBZ','m':'dBZ','h':'km','r':'mm/h','v':'kts','uv':'kts','s':'kts','w':'kts','p':u'\u00b0','k':u'\u00b0'+'/km','c':'%','x':'','d':'dB','e':'km','eb':'km','l':'kg/m\u00B2','q':'','t':'dB','y':'','i':'dB','g':'','j':'','o':'mm','b':'%','uh':'%','vd':'g/m\u00B3','si':'J/m/s','zc':'km'}
scale_factors_velocities={'m/s':1,'mph':2.23694,'kts':1.94384449,'km/h':3.6}
products_possibly_exclude_lowest_values=('z','r','a','m','e','eb','l','o','b','uh','vd','si') #Products for which it is possible to exclude the lowest part of the product range, 
#by means of the choice of minimum product values for the colormap.

products_with_interpolation=('z','a','m','r','l','o','b','uh','si','zc')
products_with_interpolation_and_binfilling = ('z','a','m') #If interpolation is applied to these products, then empty radar bins get filled by the average product value in
#the 4 neighbouring bins, if at least there are enough non-empty neighbouring bins, and if the average at least exceeds a particular product value.

velocity_dealiasing_settings = ('dual-PRF', 'dual-PRF + Unet VDA', 'dual-PRF + Unet VDA + extra')

# MESH-kalibratie-opties (24 juli, op Eriks verzoek na het zien van iRadar's MESH-kalibratie-dropdown),
# analoog aan velocity_dealiasing_settings hierboven. Alle drie herleiden SHI (nlr_mesh.severe_hail_index)
# naar een hageldiameter (mm), maar met verschillende empirische fits - zie nlr_mesh.py voor de volledige
# formules/bronvermelding. Eerste optie is de standaard/default (zelfde patroon als
# velocity_dealiasing_settings[-1] hierboven, hier is de eerste ipv de laatste de standaard).
mesh_calibration_settings = ('Witt 1998', 'Murillo & Homeyer P75', 'Murillo & Homeyer P95', 'Witt 1998 (S-band, ongecorrigeerd)')

"""The data gets stored as 8- or 16-bits unsigned integers (uint), and the number of bits is given in products_data_nbits. 
products_maxrange gives the maximum range of data values that is supported. If a data value falls outside the supported range, 
then the lower or upper limit of this range is shown instead.
Further, to let interpolation work properly, it is necessary to fill masked data elements with a value that differs by a not too small amount
from the minimum value that is supported by the color map. If this is not done, then at the boundary of regions with data, half of the 
first neighbouring bin without data gets filled with data. It is realized by enlarging the color map range by a factor of 
interpolation_fac/(1-interpolation_fac) in the negative direction, where the lower limit of this new range becomes the masking 
value.
To assure that it is possible to use this value as the masking value (the corresponding integer value cannot be smaller than zero), I use 
cmaps_maxrange_masked in converting float values to uint. cmaps_maxrange_masked follows from cmaps_maxrange in the same way as the enlarged color map
range follows from the input color map range (as given in the color table). The lower limit of cmaps_maxrange_masked is low enough to ensure that 
the integer value corresponding to the masking value is >=0.
For products for which interpolation is not allowed, it is enough to assure that it is possible to let the integer masking value be 1 less
than the minimum data value that represent non-masked data (without letting it become negative). This is assured by taking the lower limit
of cmaps_maxrange_masked to be 1/(2**n_bits-2)*products_maxrange less than the lower limit of cmaps_maxrange.
"""
#'a' should have the same number of bits as 'r', 'because 'a' is used in the calculation of 'r'!
#'g' (polarimetric RGB) and 'j' (HCLASS) do not go through the normal float->uint colormap pipeline (they
#produce their own (az,range,4) uint8 RGBA array directly), but they still need placeholder entries here
#since several places iterate generically over products_all.
products_data_nbits={'z':8,'a':8,'m':8,'h':8,'e':8,'eb':8,'l':16,'r':16,'v':16,'uv':16,'s':16,'w':8,'c':16,'d':8,'p':16,'k':8,'x':16,'q':16,'t':16,'y':16,'i':16,'g':8,'j':8,'o':16,'b':8,'uh':8,'vd':8,'si':16,'zc':16}
#The values in products_maxrange are valid for the default scale factors given in scale_factors_Default.
#If values in products_maxrange are changed for any of the plain products, then it is necessary to change the corresponding product versions in nlr_derivedproducts.py,
#because the scale factors used in converting floats to uints have changed. 
# 'eb' (Echo base) krijgt hetzelfde bereik als 'e'/'h' (hoogte-producten, 0-25km) - MESH is 24 juli op
# Eriks verzoek opgerekt van 0-80mm naar 0-120mm (zie colortable_MESH_default.csv), products_maxrange
# volgt hier mee zodat waarden boven het oude plafond niet alsnog worden afgekapt.
# 'si' (SHI, 27 juli 2026): bereik 0-500 J/m/s als veilige bovengrens - ter vergelijking komt de
# "severe hail"-drempel (Witt's 29mm-MESH-equivalent, ~130) en M&H's 40/64mm-drempels (~113/148, zie
# nlr_mesh.py) ruim binnen dit bereik; 16 bits (zoals MESH) voor voldoende resolutie i.v.m. de
# gevoelige sqrt/log-afgeleiden (MESH/POSH) die op deze waarde worden toegepast.
# 'si' (SHI, 27 juli 2026): plafond verhoogd van 500 naar 1500 J/m/s (products_maxrange) nadat Erik
# op een extreme kern exact 500,0 zag - een verdacht rond getal dat op afkapping bij het oude plafond
# wees, niet een toevallig exacte waarde. cmaps_maxrange (kleurenschaal) evenredig meegerekt van 400
# naar 600 (zie colortable_SHI_default.csv, dat op dezelfde manier is herschaald).
# 'si' (SHI, 27 juli 2026, vijfde en laatste bijstelling): cmaps_maxrange TERUGGEZET van 700 naar 400.
# Elke poging om boven de 400 nog extra, onderscheidende kleuren toe te voegen (eerst roze/magenta,
# daarna grijs/zwart) botste op een reeds gebruikte kleurfamilie ergens anders in de tabel (paars resp.
# zwart komen al voor rond de 267-367). Simpelste, evenwichtige oplossing: geen extra kleuren erboven -
# alles vanaf 400 toont gewoon de bovenste (witte) kleur, zoals bij MESH/POSH/POH ook gebruikelijk is.
# De exacte waarde blijft altijd gewoon afleesbaar via de klik-pop-up, ongeacht deze visuele grens.
# 'zc' (ZDR-kolomdiepte, 27 juli 2026): bereik 0-15km (opslag) resp. 0-8km (kleurenschaal) - de
# literatuur noemt kolommen "meer dan 3km" boven het 0C-niveau als sterk signaal; 8km dekt ruim
# de zeldzame extremen, 15km is een veilige opslagmarge.
products_maxrange={'z':[-35.,90.],'a':[-35.,90.],'m':[-35.,90.],'h':[0.,25.],'e':[0.,25.],'eb':[0.,25.],'l':[0.,500.],'r':[-3.,3.],'v':[-1000.,1000.],'uv':[-1000.,1000.],'s':[-1000.,1000.],'w':[0.,25.],'c':[0.,100.],'d':[-10.,20.],'p':[0.,360.],'k':[-10.,20.],'x':[-50.,0.],'q':[0,1],'t':[0,10],'y':[0,1],'i':[0,10],'g':[0,255],'j':[0,255],'o':[0.,150.],'b':[0.,100.],'uh':[0.,100.],'vd':[0.,20.],'si':[0.,1500.],'zc':[0.,15.]}
cmaps_maxrange={'z':[-35.,90.],'a':[-35.,90.],'m':[-35.,90.],'h':[0.,25.],'e':[0.,25.],'eb':[0.,25.],'l':[0.,200.],'r':[-3.,3.],'v':[-100.,100.],'uv':[-100.,100.],'s':[-100.,100.],'w':[0.,25.],'c':[0.,100.],'d':[-10.,20.],'p':[0.,360.],'k':[-10.,20.],'x':[-50.,0.],'q':[0,1],'t':[0,10],'y':[0,1],'i':[0,10],'g':[0,255],'j':[0,255],'o':[0.,120.],'b':[0.,100.],'uh':[0.,100.],'vd':[0.,10.],'si':[0.,400.],'zc':[0.,8.]}
#cmaps_maxrange gives for each product the range of values that is supported for the color map. At maximum 256 colors can be displayed, so there is a 
#trade-off between range and resolution. This range should at least not be larger than products_maxrange.
products_maxrange={j:np.array(products_maxrange[j]) for j in products_maxrange}
cmaps_maxrange={j:np.array(cmaps_maxrange[j]) for j in cmaps_maxrange}

#The values of interpolation_fac are chosen for each product in such a way that interpolation produces reasonably looking results.
interpolation_fac={'z':0.3,'m':0.3,'a':0.3,'r':0.7,'l':0.013,'o':0.013,'b':0.013,'uh':0.013,'si':0.013,'zc':0.013}
cmaps_maxrange_masked={}
for j in products_all:
    c_lim=cmaps_maxrange[j]
    c_range=c_lim[1]-c_lim[0]
    n_bits=products_data_nbits[j]
    if j in products_with_interpolation:
        if productunits_default[j] == 'dBZ':
            # Use dBZ spacing of exactly 0.5
            cmaps_maxrange_masked[j] = np.array([-37.5, 90.])
        else:
            cmaps_maxrange_masked[j]=c_lim-np.array([interpolation_fac[j]*c_range,0])
    else:
        #Make sure that the integer mask value that is used to indicate masked data can be at least 1 lower than the lowest integer data
        #value (the integer data values cannot be smaller than zero).
        #(2**n_bits-2 vs 2**n_bits-1 is used, because the range of integer values that represents non-masked data is 2**n_bits-2).
        cmaps_maxrange_masked[j]=c_lim-np.array([1/(2**n_bits-2)*c_range,0])
products_maxrange_masked={j:np.array([cmaps_maxrange_masked[j][0],products_maxrange[j][1]]) for j in products_all}

products_with_tilts=('z','v','uv','s','w','p','d','k','c','x','q','t','y','i','g','j')
products_with_tilts_derived = ('s',)
# These products are not stored in memory, since they can be cheaply calculated from their import product:
products_with_tilts_derived_nosave = ('s',)
plain_products=('e','eb','r','a','m','h','l','o','b','uh','vd','si','zc')
plain_products_affected_by_double_volume=('a','r')
plain_products_with_parameters=('e','a','m','h','l')
# 'eb' bewust NIET in plain_products_with_parameters: gebruikt een VASTE drempel (18.5dBZ, zelfde als
# ETH-preset 1), niet instelbaar via SHIFT+Q - dat zou de plain_products_parameter_description/
# PP_parameter_values-UI-plumbing vergen die nog niet is toegevoegd. Kan later alsnog, net als bij 'e'.
# 'si' (SHI, 27 juli 2026): zelfde storm-motion-correctie/elevatie-gedrag als MESH/POSH/POH, want SHI
# is precies dezelfde onderliggende SHI-integraal (zie calculate_SHI in nlr_derived_plain.py) - alleen
# zonder de laatste MESH/POSH-omzetstap.
plain_products_correct_for_SM = ['m','h','e','eb','l','o','b','uh','si']
# 'zc' (ZDR-kolomdiepte) BEWUST NIET hier toegevoegd: geen storm-motion-correctie ondersteund (zie
# Polar.calculate_ZDRmax_3D in derived/polar.py) - wordt altijd in de gewone polaire projectie
# berekend, ook als storm-motion-correctie voor andere producten aanstaat.
plain_products_parameter_description={'e':'Minimum reflectivity echo tops (dBZ)','a':'PCAPPI height (km)','m':'Minimum height Zmax (km)','h':['Cap Z values at 56 dBZ in calculation (yes/no)', 'Mask values where VIL < threshold (kg/m\u00B2)'],'l':['Cap Z values at 56 dBZ in calculation (yes/no)', 'Minimum height/lower boundary of integration (km)']}
plain_products_show_max_elevations=('e','eb','l','m','h','o','b','uh','vd','si','zc')
plain_products_show_true_elevations=('a','r')

CAPPI_height_R=1.5

# VILD (VIL/ETH): waarde voor VILD's EIGEN, interne ETH-basis. TIJDELIJK terug op 0.0dBZ (25 juli 2026,
# voor een directe A/B-vergelijking met de 18.5dBZ-versie op dezelfde case) - zie hierboven/de
# gespreksgeschiedenis voor de volledige afweging tussen randartefacten (bij 18.5) vs. ruisgevoeligheid
# (bij 0.0). Nog geen definitieve keuze - dit is een testversie.
VILD_ETH_THRESHOLD = 18.5

colortables_dirs_filenames_Default={'z':programdir+'/Input_files/Color_tables/colortable_Z.csv','a':programdir+'/Input_files/Color_tables/colortable_Z.csv','m':programdir+'/Input_files/Color_tables/colortable_Z.csv','uv':programdir+'/Input_files/Color_tables/colortable_V.csv','h':programdir+'/Input_files/Color_tables/Default/colortable_ET_default.csv','r':programdir+'/Input_files/Color_tables/Default/colortable_RI_default.csv','v':programdir+'/Input_files/Color_tables/colortable_V.csv','s':programdir+'/Input_files/Color_tables/colortable_V.csv','w':programdir+'/Input_files/Color_tables/Default/colortable_W_default.csv','k':programdir+'/Input_files/Color_tables/Default/colortable_KDP_default.csv','c':programdir+'/Input_files/Color_tables/Default/colortable_CC_default.csv','x':programdir+'/Input_files/Color_tables/Default/colortable_LDR_default.csv','q':programdir+'/Input_files/Color_tables/Default/colortable_SQI_default.csv','t':programdir+'/Input_files/Color_tables/Default/colortable_CCOR_default.csv','y':programdir+'/Input_files/Color_tables/Default/colortable_SQI_default.csv','i':programdir+'/Input_files/Color_tables/Default/colortable_CI_default.csv','p':programdir+'/Input_files/Color_tables/Default/colortable_PhiDP_default.csv','d':programdir+'/Input_files/Color_tables/Default/colortable_ZDR_default.csv','e':programdir+'/Input_files/Color_tables/Default/colortable_ET_default.csv',
# 'eb' (Echo base, 24 juli 2026): EIGEN nieuwe kleurtabel op Eriks verzoek (rechtstreeks uit zijn
# ECHOBASE.PNG-screenshot geëxtraheerd, NIET de bestaande ETH-tabel hergebruikt).
'eb':programdir+'/Input_files/Color_tables/Default/colortable_ECHOBASE_default.csv',
'l':programdir+'/Input_files/Color_tables/Default/colortable_VIL_default.csv',
# 'vd' (VILD, 24 juli 2026): EIGEN nieuwe kleurtabel op Eriks verzoek (rechtstreeks uit zijn VILD.PNG-
# screenshot geëxtraheerd), NIET de bestaande VIL-tabel hergebruikt.
'vd':programdir+'/Input_files/Color_tables/Default/colortable_VILD_default.csv',
'g':programdir+'/Input_files/Color_tables/colortable_Z.csv','j':programdir+'/Input_files/Color_tables/Default/colortable_HCLASS_default.csv','o':programdir+'/Input_files/Color_tables/Default/colortable_MESH_default.csv','b':programdir+'/Input_files/Color_tables/Default/colortable_POSH_default.csv',
# 'uh' (POH) heeft NOG GEEN eigen kleurtabel-bestand - wijst voorlopig naar dezelfde CSV als POSH ('b'),
# want beide zijn 0-100% kansen op dezelfde schaal. Ik heb colortable_POSH_default.csv zelf niet gezien
# deze sessie (geen bestandstoegang), dus ik kon geen nieuwe, apart geverifieerde CSV in het juiste
# formaat aanmaken zonder te gokken naar de syntax. Zodra Erik een eigen colortable_POH_default.csv wil
# (bv. andere kleuren), is dit de ENIGE regel die aangepast hoeft te worden.
'uh':programdir+'/Input_files/Color_tables/Default/colortable_POSH_default.csv',
# 'si' (SHI, 27 juli 2026): EIGEN nieuwe kleurtabel (colortable_SHI_default.csv), met stops rond de
# uit nlr_mesh.py afgeleide "severe hail"-omslagpunten (SHI~110-150 komt overeen met de 29mm/40mm/64mm-
# MESH-drempels) - zie de toelichting in dat CSV-bestand zelf.
'si':programdir+'/Input_files/Color_tables/Default/colortable_SHI_default.csv',
'zc':programdir+'/Input_files/Color_tables/Default/colortable_ZDRCOL_default.csv'}
colortables_dirs_filenames_NWS={'z':programdir+'/Input_files/Color_tables/NWS/colortable_Z.csv','a':programdir+'/Input_files/Color_tables/NWS/colortable_Z.csv','m':programdir+'/Input_files/Color_tables/NWS/colortable_Z.csv','uv':programdir+'/Input_files/Color_tables/NWS/colortable_V.csv','v':programdir+'/Input_files/Color_tables/NWS/colortable_V.csv','s':programdir+'/Input_files/Color_tables/NWS/colortable_V.csv','w':programdir+'/Input_files/Color_tables/NWS/colortable_W.csv','c':programdir+'/Input_files/Color_tables/NWS/colortable_CC.csv','p':programdir+'/Input_files/Color_tables/NWS/colortable_PhiDP.csv','l':programdir+'/Input_files/Color_tables/NWS/colortable_VIL.csv'}
colortables_dirs_filenames_Default={i:opa(j) for i,j in colortables_dirs_filenames_Default.items()}
colortables_dirs_filenames_NWS={i:opa(j) for i,j in colortables_dirs_filenames_NWS.items()}

# Gedeelde (installeerbare) versie: Eriks eigen kleurtabelkeuzes als standaard (overgenomen uit zijn
# instellingen, 23 sep 2026), omdat stored_settings.pkl niet wordt meegeleverd. Gedeelde versie: ook bij starten
# vanuit de broncode (test_NLradar.bat), zodat testen en installer hetzelfde laten zien.
if True:
    _ct = programdir+'/Input_files/Color_tables/'
    colortables_dirs_filenames_Default.update({
        'z': _ct+'colortable_Z_essl.csv', 'v': _ct+'colortable_V_essl.csv',
        'a': _ct+'colortable_Z_essl.csv', 'm': _ct+'colortable_Z_essl.csv',
        's': _ct+'colortable_V_essl.csv', 'uv': _ct+'colortable_V_essl.csv',
        'w': _ct+'NWS/colortable_W.csv', 'c': _ct+'NWS/colortable_CC.csv', 'p': _ct+'NWS/colortable_PhiDP.csv'})
animation_frames_directory = opa(userdir+'/Generated_files/ani_frames')

vwp_sm_names = {'MW': '0-6 km MW', 'LM': 'Bunkers LM', 'RM': 'Bunkers RM', 'SM':'Observed SM', 'DTM':'Deviant TM'}
vwp_sm_colors = {'MW': 'orange', 'LM': [0,1,0,1], 'RM': 'cyan', 'SM':'magenta', 'DTM':'brown'}