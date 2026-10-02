# Copyright (C) 2016-2024 Bram van 't Veen, bramvtveen94@hotmail.com
# Distributed under the GNU General Public License version 3, see <https://www.gnu.org/licenses/>.

import copy
import numpy as np
import time as pytime
import h5py
import os
import shutil
import traceback
from collections import OrderedDict
# import matplotlib.pyplot as plt

import nlr_background as bg
import nlr_functions as ft
import nlr_globalvars as gv
import nlr_attenuation as att
from derived.polar import Polar
from derived.cartesian import Cartesian



class DerivedPlain():
    def __init__(self, dsg_class, parent = None):
        self.dsg = dsg_class
        self.gui = self.dsg.gui
        self.crd = self.dsg.crd
        self.pb = self.gui.pb
        
        self.import_data_specs = None
        
        self.filename_version = 2
        # self.gui.derivedproducts_filename_version is set equal to self.filename_version in self.gui when closing the application
        if not self.gui.derivedproducts_filename_version == self.filename_version and os.path.exists(self.gui.derivedproducts_dir):
            shutil.rmtree(self.gui.derivedproducts_dir)
        
        self.file_content_version = 38
        self.file_content_version_sources = {'Météo-France':1} # Update when only content for particular data source changes
        self.product_version = {'e':16,'eb':1,'r':11,'a':5,'m':5,'h':8,'l':17,'o':3,'b':2,'uh':2,'vd':3,'si':3,'zc':1} #Product version, must be updated when the method for calculating the product has changed.
        # o:2->3, b:1->2, uh:1->2, si:2->3 (28 juli 2026): ZPHI-verzwakkingscorrectie toegevoegd aan
        # de berekening van alle 4 (nlr_attenuation.py, via get_hail_corrected_data_all hieronder).
        # Zonder deze ophoging zouden oude, op schijf gecachete resultaten (berekend VOOR deze
        # wijziging) als nog geldig worden beschouwd volgens check_if_product_at_disk - exact
        # dezelfde valkuil als bij MESH's eerdere versie-ophoging (zie de toelichting daar).
        # SHI ('si') opgehoogd van 1 naar 2 (27 juli): products_maxrange/cmaps_maxrange voor 'si' zijn
        # verhoogd (500->1500 resp. 400->600 J/m/s, zie nlr_globalvars.py) nadat Erik op een extreme kern
        # exact 500,0 zag - afkapping bij het oude plafond, geen toevallige waarde. Oude, op schijf
        # gecachete SHI-resultaten (berekend met de oude, te lage schaal) moeten daarom ongeldig worden,
        # net als bij MESH's eerdere versie-ophoging (zie hierboven).
        # MESH ('o') opgehoogd van 1 naar 2 (22 juli): sinds versie 1 zijn er meerdere
        # inhoudelijke wijzigingen geweest aan de berekening (de Edot-formulecorrectie, het
        # temperatuurpunt, de fix voor lege Open-Meteo-antwoorden, enz.) zonder dat dit
        # versienummer ooit werd opgehoogd. Daardoor bleven oude, op schijf gecachete MESH-
        # resultaten (geschreven met de OUDE, soms foutieve of zelfs volledig lege logica)
        # voor altijd geldig lijken volgens check_if_product_at_disk (regel ~111: "if version
        # != self.product_version[product]"), en werd calculate_MESH voor die tijdstippen
        # nooit meer aangeroepen - exact het "soms goed, meestal leeg"-patroon dat Erik zag.
        # Deze ophoging maakt AL die oude schijf-cache in één keer ongeldig.
        
        self.product_parameters = {'e':'min_dBZ_value_echotops','a':'PCAPPI_height','m':'Zmax_minheight','h':'cap_dBZ, VIL_threshold','l':'cap_dBZ, VIL_minheight'}
        self.product_attributes = {'e':['scans_ranges','elevations_minside','elevations_plusside'],'eb':['scans_ranges','elevations_minside','elevations_plusside'],'m':['scans_ranges','elevations_minside','elevations_plusside'],'h':['scans_ranges','elevations_minside','elevations_plusside'],'l':['scans_ranges','elevations_minside','elevations_plusside'],'o':['scans_ranges','elevations_minside','elevations_plusside'],'b':['scans_ranges','elevations_minside','elevations_plusside'],'uh':['scans_ranges','elevations_minside','elevations_plusside'],'si':['scans_ranges','elevations_minside','elevations_plusside'],'zc':['scans_ranges','elevations_minside','elevations_plusside']}
        self.product_proj_attributes = {'pol': ['product_radial_bins','product_radial_res','product_azimuthal_bins','product_azimuthal_res'],
                                        'car': ['product_xy_bins', 'product_res']}
        # Defines the maximum number of datasets per product within a file (different datasets have a different product parameter)
        # Allow 2 more for the cartesian products, because here also the resolution can be varied, instead of only the product parameter
        self.product_datasets_max = {'pol':8, 'car':12}
        # 4->8 / 6->12 (28 juli 2026): de nieuwe ZPHI-verzwakkingscorrectie-schakelaar (ALT+C,
        # attenuation_correction_enabled) verdubbelt het aantal mogelijke cache-varianten voor
        # MESH/POSH/POH/SHI ('_attoff'-achtervoegsel in get_dataset_name, zie daar) bovenop de
        # AL bestaande 4 MESH-kalibratievarianten. Het oude maximum van 4 was daar precies op
        # afgestemd (zie de toelichting bij get_dataset_name hierboven) - zonder deze verhoging
        # zou de "verwijder minst-gebruikte"-cache-opruiming hieronder (write_file) voortdurend
        # slots overschrijven, mogelijk zelfs het slot dat net is bekeken (Erik meldde: paneel/
        # pop-upwaarde bleef na het wisselen op de oude waarde staan, ook na expliciete
        # cache-invalidatie en geforceerde herberekening).
        
        # 'eb' (Echo base, 24 juli 2026): calculate_echobase hieronder, zelfde 'independent'-categorie als
        # ETH/VIL/MESH/POSH/POH (geen afhankelijkheid van een ander product, in tegenstelling tot 'r').
        # 'si' (SHI, 27 juli 2026): zelfde 'independent'-categorie als MESH/POSH/POH (geen afhankelijkheid
        # van een ander plain product).
        self.plain_products_functions = {'independent':{'e':self.calculate_echotops,'eb':self.calculate_echobase,'a':self.calculate_PCAPPI,'m':self.calculate_Zmax,'l':self.calculate_VIL,'o':self.calculate_MESH,'b':self.calculate_POSH,'uh':self.calculate_POH,'si':self.calculate_SHI,'zc':self.calculate_ZDRcol},'dependent':{'r':self.calculate_R,'vd':self.calculate_VILD}}
        # Only base products that are used to decide which function should be called need to be specified below
        self.products_requiring_heightsorted_data = ['e','eb','a','l','o','b','uh','si','zc']

        self.meta_PP = {j: {} for j in gv.plain_products}
        
        self.mapping_classes = {'pol': Polar(self), 'car': Cartesian(self)}
        # Cache voor de ZPHI-verzwakkingscorrectie (nlr_attenuation.py), alleen gebruikt door
        # MESH/POSH/POH/SHI (zie get_hail_corrected_data_all hieronder). Sleutel:
        # get_import_data_specs() (radar/dataset/tijd). Dit is een LRU-cache over MEERDERE
        # tijdstippen (21 september 2026, Eriks verzoek) - voorkomt niet alleen de 4x herhaling
        # binnen hetzelfde volume (1x per hagelproduct), maar houdt ook eerder bezochte
        # tijdstippen vast, zodat heen-en-weer bladeren ze niet steeds opnieuw hoeft te
        # berekenen. Puur een in-memory dict (OrderedDict) op de DerivedPlain-instantie -
        # bestaat niet meer zodra NLradar wordt afgesloten (geen eigen opslag naar schijf),
        # dus "leegmaken bij afsluiten" gebeurt vanzelf.
        self.hail_attenuation_products = ('o', 'b', 'uh', 'si')
        self._hail_corrected_data_all_cache = OrderedDict()
        # Maximum aantal tijdstippen dat tegelijk in de cache blijft (oudste eruit bij
        # overschrijding, LRU). Een volume van ~14-16 scans kost ruwweg 10-20 MB (afhankelijk
        # van radar/resolutie); 20 tijdstippen is dus típisch enkele honderden MB - ruim
        # genoeg om een sessie bladeren te dekken, zonder het geheugen open te laten lopen
        # bij lang doorbladeren. Desgewenst hoger te zetten als het geheugen het toelaat.
        self._hail_corrected_data_all_cache_maxsize = 20
        self.mapping_parameters = ['Zmax_3D','Zavg_3D','heights_3D','hdiffs','ZDRmax_3D']
        # 'ZDRmax_3D' (27 juli 2026): FIX - was vergeten toe te voegen toen calculate_ZDRcol/
        # Polar.calculate_ZDRmax_3D werden gebouwd. Zonder dit werd ZDRmax_3D wel op de Polar-instantie
        # berekend, maar nooit teruggekopieerd naar DerivedPlain, vandaar Eriks AttributeError. De
        # bestaande hasattr-guard hieronder (bij het kopieren) zorgt dat dit voor andere producten
        # (die geen ZDRmax_3D hebben) gewoon overgeslagen blijft worden - geen effect op MESH/POSH/etc.
        
    
    
    def get_dir_and_filename(self, proj):
        radar_dataset = self.dsg.get_radar_dataset(no_special_char=True)
        subdataset = self.dsg.get_subdataset(product=self.i_p)

        directory = self.gui.derivedproducts_dir+('/regular/' if proj == 'pol' else '/SM_correction/')+\
                    radar_dataset+'/'+subdataset+'/'+self.crd.date
        filename = directory+'/'+self.crd.date+self.crd.time+'.h5'
        return directory, filename

    def check_if_product_at_disk(self, p_param, proj, check_filtered_for_unfiltered=False):
        # check_filtered_for_unfiltered should be set to True when doing a 2nd attempt at importing data, now for the filtered version of a product, 
        # after it became clear that the unfiltered version is not available.
        product, param = self.get_product_and_param(p_param)        
        # .get(product, 0) i.p.v. [product] (22 juli, MESH-toevoeging): scannumbers_forduplicates
        # krijgt een entry voor een product pas via update_scannumbers_forduplicates(), die alleen
        # draait als bepaalde lengtes niet overeenkomen (zie DataSource_General.get_data) - bij
        # sommige navigatiepaden (bv. terug naar een eerdere case) kan dat voor een NIEUW
        # geregistreerd product zoals 'o' (MESH) even niet het geval zijn. 0 (het eerste/enige scan-
        # duplicaat) is dezelfde standaardwaarde die update_scannumbers_forduplicates zelf ook voor
        # een product zonder eerdere waarde zou kiezen (zie self.scannumbers_forduplicates.get(i, 0)
        # daar), dus dit is geen gok maar dezelfde conventie.
        double_volume_index = self.dsg.scannumbers_forduplicates.get(product, 0)
        
        _, filename = self.get_dir_and_filename(proj)
        if not os.path.exists(filename):
            return False
        
        product_at_disk = False
        try:
            with h5py.File(filename,'r+') as f:
                if self.check_need_update(f):
                    return False
                
                # When requesting an unfiltered product while this unfiltered version is unavailable (which becomes clear when importing the volumetric
                # data; self.using_unfilteredproduct), then this function is called again for the same p_param but for the filtered version. In this 
                # case the attribute 'u'+gv.i_p[product]+'_unavailable' is also updated, because it might be that the filtered version is indeed available, 
                # in which case this attribute does not get updated in write_file.
                if not check_filtered_for_unfiltered:
                    unfiltered_unavailable = f.attrs.get('u'+gv.i_p[product]+'_unavailable', False)
                else:
                    unfiltered_unavailable = self.productunfiltered and not self.using_unfilteredproduct
                    if unfiltered_unavailable:
                        f.attrs['u'+gv.i_p[product]+'_unavailable'] = True
                
                group_name = 'u' if self.productunfiltered and not unfiltered_unavailable else ''
                if len(self.dsg.scannumbers_all['z'][product])==2:
                    group_name += gv.productnames[product]+'_'+str(double_volume_index+1)
                else: 
                    group_name += gv.productnames[product]
                if not group_name in f:
                    return False
                
                group = f[group_name]
                version = group.attrs['version']
                if version != self.product_version[product]:
                    f.__delitem__(group_name)
                    return False
                
                if proj == 'car':
                    subgroup_present = False
                    for subgroup in group.values():
                        if (subgroup.attrs['stormmotion'] == self.gui.stormmotion).all():
                            subgroup_present = True; break
                    if not subgroup_present:
                        return False
                else:
                    subgroup = group
                
                dataset_name = self.get_dataset_name(product, param, proj)
                    
                datasets = list(subgroup)
                if dataset_name in datasets:
                    dataset = subgroup[dataset_name]
                    self.product_arrays[p_param] = np.asarray(dataset, dtype='uint'+str(gv.products_data_nbits[product]))
                    self.p_param_attributes[p_param] = {}
                    for attr in self.product_attributes.get(product, [])+self.product_proj_attributes[proj]+['proj']:
                        o = dataset if attr in self.product_proj_attributes[proj] and proj == 'car' else group
                        self.p_param_attributes[p_param][attr] = o.attrs[attr]
                        
                    dataset.attrs['n_displayed'] += 1
                    dataset.attrs['last_view_time'] = pytime.time()
                    self.producttimes[p_param] = group.attrs['producttime']
                    self.using_unfilteredproduct = self.productunfiltered and not unfiltered_unavailable
                    product_at_disk = True
        except Exception as e:
            print(e, product)
            pass
        
        return product_at_disk
    
    def check_need_update(self, f):
        version, version_source = f.attrs['version'], f.attrs.get('version_source', 0)
        total_volume_files_size = f.attrs['total_volume_files_size']
        return version != self.file_content_version or version_source != self.file_content_version_sources.get(self.dsg.data_source(), 0) or\
               total_volume_files_size != self.dsg.total_files_size      

    def get_dataset_name(self, product, param, proj):
        dataset_name = 'data' if not product in self.gui.PP_parameter_values else f'data_pval{param}'
        # BUGFIX (24 juli, opgelost bekend risico uit de vorige sessie): MESH ('o') heeft een instelbare
        # kalibratie (self.gui.mesh_calibration_setting, zie select_mesh_calibration_settings in nlr.py),
        # maar de schijf-cache-sleutel hield daar tot nu toe geen rekening mee - alleen self.product_version
        # bepaalde geldigheid, en dat blijft gelijk ongeacht de gekozen kalibratie. Gevolg: na het wisselen
        # van kalibratie kon een AL eerder op schijf gecacht MESH-resultaat (berekend met de vorige
        # kalibratie) stilzwijgend worden hergebruikt voor een tijdstip dat al eerder bekeken was - zonder
        # foutmelding, gewoon een verkeerd getal. Fix: de kalibratienaam wordt nu onderdeel van de
        # dataset-sleutel (net zoals hierboven al gebeurt voor param-waarden en, bij 'car', voor de
        # storm-motion-afhankelijke resolutie/bereik), zodat elke kalibratie zijn EIGEN, gescheiden
        # cache-slot krijgt i.p.v. elkaars resultaten te overschrijven/hergebruiken. Geen effect op andere
        # producten (alleen 'o' heeft mesh_calibration_setting). Blijft ruim onder het maximum aantal
        # datasets per groep (product_datasets_max: 4 voor polair/6 voor cartesisch, tegenover hooguit 3
        # kalibraties), dus de bestaande "verwijder oudste"-cache-opruiming hierboven wordt hier niet
        # geraakt.
        if product == 'o':
            calibration = getattr(self.gui, 'mesh_calibration_setting', gv.mesh_calibration_settings[0])
            dataset_name += '_calib' + calibration.replace(' ', '').replace('&', 'And')
        if product in ('o', 'b', 'uh', 'si', 'zc'):
            # BUGFIX (25 juli, zelfde soort valkuil als de MESH-kalibratiefix hierboven): MESH/POSH/POH
            # (en sinds 27 juli ook SHI en de ZDR-kolomdiepte 'zc', die beide hetzelfde 0C-niveau delen)
            # gebruiken alledrie het gedeelde temperatuurrooster (ensure_melting_level_grid_current).
            # Bij de automatische bronkeuze is dat geen probleem (die is een vaste functie van de
            # scandatum/-tijd, en de h5-bestandsnaam zelf is al per datum+tijd - zie get_dir_and_filename
            # hierboven). Maar bij de handmatige terugvaloptie (self.gui.melting_levels_manual_override,
            # zie nlr.py's select_melting_levels_override/ALT+W) kan de gebruiker voor PRECIES DEZELFDE
            # scan verschillende stations/datums/uren uittesten - zonder deze sleuteluitbreiding zou een
            # eerder (bijv. blanco, foutieve) resultaat stilzwijgend hergebruikt worden nadat een geldige
            # keuze is gemaakt, exact het "1 specifiek beeld blijft blanco/vastzitten"-patroon dat Erik
            # meldde. Alleen relevant zolang de override aanstaat; bij automatische keuze blijft de
            # sleutel ongewijzigd (geen extra cache-groei voor het normale pad).
            if getattr(self.gui, 'melting_levels_manual_override', False):
                dataset_name += '_mlman' + str(getattr(self.gui, 'melting_levels_manual_station', '')).replace(' ', '')
                dataset_name += str(getattr(self.gui, 'melting_levels_manual_date', '')).replace('-', '')
                dataset_name += 'h' + str(getattr(self.gui, 'melting_levels_manual_hour', ''))
        if product in ('o', 'b', 'uh', 'si'):
            # ZPHI-verzwakkingscorrectie aan/uit (28 juli 2026, ALT+C) - zelfde soort valkuil als
            # hierboven: zonder deze sleuteluitbreiding zou uit/aan-zetten een resultaat van de andere
            # stand kunnen hergebruiken. NIET van toepassing op 'zc' (die gebruikt geen gecorrigeerde
            # data, zie get_hail_corrected_data_all/nlr_attenuation.py - scope is uitdrukkelijk beperkt
            # tot o/b/uh/si, HCLASS 'j' zit hier niet bij want die gaat sowieso niet via deze schijf-cache).
            if not getattr(self.gui, 'attenuation_correction_enabled', True):
                dataset_name += '_attoff'
        if proj == 'car':
            dataset_name += f'_res{self.gui.cartesian_product_res}_maxrange{self.gui.cartesian_product_maxrange}'
        return dataset_name
    
    def write_file(self, p_param, proj):
        product, param = self.get_product_and_param(p_param)
        # Zelfde defensieve .get(product, 0) als in check_if_product_at_disk hierboven (22 juli,
        # MESH-toevoeging) - zie de toelichting daar.
        double_volume_index = self.dsg.scannumbers_forduplicates.get(product, 0)

        directory, filename = self.get_dir_and_filename(proj)
        os.makedirs(directory, exist_ok=True)
        try:
            with h5py.File(filename, 'r') as f:
                action = 'w' if self.check_need_update(f) else 'a'
        except Exception: 
            action = 'w'
        
        with h5py.File(filename, action) as f:
            f.attrs['version'] = self.file_content_version
            f.attrs['version_source'] = self.file_content_version_sources.get(self.dsg.data_source(), 0)
            f.attrs['total_volume_files_size'] = self.dsg.total_files_size
            if self.productunfiltered and not self.using_unfilteredproduct:
                # Indicate that the unfiltered version of the import product is unavailable, such that no time will be wasted on trying again 
                # when it is requested a next time
                f.attrs['u'+gv.i_p[product]+'_unavailable'] = True
                
            group_name = 'u' if self.using_unfilteredproduct else ''
            if len(self.dsg.scannumbers_all['z'][product])==2:
                #This is the case for the products in plain_products_affected_by_double_volume for the new radars of the KNMI.
                #Append in this case the volume number to the group_name.
                group_name += gv.productnames[product]+'_'+str(double_volume_index+1)
            else: 
                group_name += gv.productnames[product]
            group = f.create_group(group_name) if not group_name in f else f[group_name]
            group.attrs['version']=self.product_version[product]
            group.attrs['producttime']=self.producttimes[p_param]
            
            if proj == 'car':
                # In this case different subgroups are created for different cases (different storm motions), since storm motion influences the product
                subgroup_present = False
                for subgroup in group.values():
                    if (subgroup.attrs['stormmotion'] == self.gui.stormmotion).all():
                        subgroup_present = True; break
                        
                if not subgroup_present:
                    subgroup = group.create_group(f'case{len(group)+1}')
                    subgroup.attrs['stormmotion'] = self.gui.stormmotion
            else:
                subgroup = group
            
            dataset_name = self.get_dataset_name(product, param, proj)
                            
            datasets = list(subgroup)
            if len(datasets) == self.product_datasets_max[proj]:
                #Parameter values for which the plain product is currently being plotted, for which the dataset should not be overwritten.
                params_in_use = [self.gui.PP_parameter_values[product][self.gui.PP_parameters_panels[j]] for j in self.pb.panellist if self.crd.products[j] == self.product]
                n_displayed = [subgroup[j].attrs['n_displayed'] for j in datasets if not subgroup[j].attrs[self.product_parameters[product]] in params_in_use]
                min_displayed = np.min(n_displayed)
                # Multiple datasets might have been displayed by the minimum number of times. In that case delete the oldest of these
                last_view_times = np.array([[j, subgroup[dset].attrs['last_view_time']] for (j, dset) in enumerate(datasets) if n_displayed[j] == min_displayed], dtype='uint64')
                dataset_remove = datasets[last_view_times[np.argmin(last_view_times[:, 1])][0]]
                dataset = subgroup[dataset_name] = subgroup[dataset_remove]
                del subgroup[dataset_remove]
            else:
                if dataset_name in subgroup:
                    del subgroup[dataset_name]
                dataset = subgroup.create_dataset(dataset_name, self.product_arrays[p_param].shape, maxshape=(None, None) if proj == 'car' else None,
                                                  dtype='uint'+str(gv.products_data_nbits[product]), compression='gzip', track_times=False)
                
            #The -product_range/int_range corrects for the fact that the integer value 1 and not 0 corresponds to a product value of
            #gv.products_maxrange[product][0].
            #Masked elements get an integer value of 0.
            int_range = 2**gv.products_data_nbits[product]-2 #A value of zero is used for masked elements, so therefore -2 vs -1.
            product_range = gv.products_maxrange[product][1]-gv.products_maxrange[product][0]
            group.attrs['calibration_formula']=str(product_range/int_range)+'*PV+'+str(gv.products_maxrange[product][0]-product_range/int_range)
            
            self.product_arrays[p_param] = self.dsg.convert_dtype_float_to_uint(self.product_arrays[p_param], product)
            if not dataset.shape == self.product_arrays[p_param].shape:
                dataset.resize(self.product_arrays[p_param].shape)
            dataset[...]=self.product_arrays[p_param]
                
            dataset.attrs['n_displayed'] = 1
            dataset.attrs['last_view_time'] = pytime.time()
            if not param is None:
                dataset.attrs[self.product_parameters[product]] = param
            for (attr, value) in self.p_param_attributes[p_param].items():
                o = dataset if attr in self.product_proj_attributes[proj] and proj == 'car' else group
                o.attrs[attr] = value
                

                
    def calculate_plain_products(self, panellist):
        """This function returns an array of uint-type product values!!!
        """
        t = pytime.time()
        self.double_volume_index = self.dsg.scannumbers_forduplicates['a']
        
        panels_filtered = [j for j in panellist if not self.crd.productunfiltered[j]]
        panels_unfiltered = [j for j in panellist if self.crd.productunfiltered[j]]
        self._calculate_plain_products(panels_filtered)
        self._calculate_plain_products(panels_unfiltered, productunfiltered=True)
        print(pytime.time()-t, 't_derived_tot')
        
    def get_product_and_param(self, p_param):
        product = p_param[:p_param.index('_')] if '_' in p_param else p_param
        param = p_param[p_param.index('_')+1:] if '_' in p_param else None                            
        return product, param
    
    def check_p_params_at_disk(self, p_params_bases, p_params_projs, check_filtered_for_unfiltered=False):
        for p_param in p_params_bases.copy():
            proj = p_params_projs[p_param]
            if proj == 'car' and not self.gui.current_case_shown(mode='loose'):
                # Cartesian products are usually not saved, because they vary as a function of storm motion.
                # The exception is when a case is currently shown, in which case it is likely that the stored product will be of use again
                continue
            product, param = self.get_product_and_param(p_param)
            
            if self.check_if_product_at_disk(p_param, proj, check_filtered_for_unfiltered):
                del p_params_bases[p_param]
            else:
                for base_p_param in p_params_bases[p_param].copy():
                    base_product, param = self.get_product_and_param(base_p_param)
                    if not p_param == base_p_param and self.check_if_product_at_disk(base_p_param, proj, check_filtered_for_unfiltered):
                        p_params_bases[p_param].remove(base_p_param)
                        # Convert uint to float for calculations
                        self.product_arrays[base_p_param] = self.dsg.convert_dtype_float_to_uint(self.product_arrays[base_p_param], base_product, inverse=True)
        return p_params_bases
    
    def _calculate_plain_products(self, panellist, productunfiltered=False):
        self.productunfiltered = productunfiltered
        
        p_params_panels = {}
        p_params_projs = {}
        p_params_bases = {}
        for panel in panellist:
            product = self.crd.products[panel]
            p_param = product
            if product in self.gui.PP_parameter_values:
                key = self.gui.PP_parameters_panels[panel]
                param = self.gui.PP_parameter_values[product][key]
                p_param += f'_{param}'
            p_params_panels[panel] = p_param
            
            p_params_projs[p_param] = 'car' if self.gui.stormmotion[1] != 0. and product in gv.plain_products_correct_for_SM else 'pol'
            # 'vd' (VILD) is net als 'r' een 'dependent' product - hangt af van twee AL berekende bases
            # i.p.v. zelf de volledige kolom te doorlopen. VIL-basis blijft preset 1
            # (self.gui.PP_parameter_values['l'][1]) voor hergebruik-voordeel met een eventueel los VIL-
            # paneel. ETH-basis gebruikt gv.VILD_ETH_THRESHOLD (zie nlr_globalvars.py voor de huidige waarde
            # en de volledige afweging tussen randartefacten vs. ruisgevoeligheid bij een lagere drempel).
            p_params_bases[p_param] = {'r':[f'a_{gv.CAPPI_height_R}'],
                                        'vd':[f'l_{self.gui.PP_parameter_values["l"][1]}',
                                              f'e_{gv.VILD_ETH_THRESHOLD}']}.get(product, [p_param])
                
            for base_p_param in p_params_bases[p_param]:
                # Make sure that the projection of the base_p_params is also defined
                p_params_projs[base_p_param] = p_params_projs[p_param]
        # When p_params are already present at the disk, they will below be removed from p_params_bases. But p_params_bases_all will 
        # keep referring to the original list.
        p_params_bases_all = copy.deepcopy(p_params_bases)


        self.i_p = gv.i_p['z']

        self.product_arrays = {} # Will contain data arrays per p_param
        self.producttimes = {} # Will contain time ranges per product, not also per param because these times are independent of those
        self.p_param_attributes = {} # Will contain attributes per p_param that are needed for saving to a file and for setting some
        # product-specific parameters
        
        self.check_p_params_at_disk(p_params_bases, p_params_projs)


        if any([len(p_params_bases[p_param]) > 0 for p_param in p_params_bases]):
            self.import_data_plain_products()
            self.get_info_per_product()
            
            if productunfiltered and not self.using_unfilteredproduct:
                # Filtered products will be used instead, check whether these are maybe already available at the disk
                p_params_bases = self.check_p_params_at_disk(p_params_bases, p_params_projs, check_filtered_for_unfiltered=True)
            
            
        self.base_p_params = []
        p_params_functions = {}
        for p_param in p_params_bases:
            for base_p_param in p_params_bases[p_param]:
                base_product, param = self.get_product_and_param(base_p_param)
                self.base_p_params += [base_p_param]
                # It's possible that a product can best be calculated in the same function that is used for another product.
                # In that case put that other product in p_params_functions.
                p_param_function = {'h':'l_[True, 0]'}.get(base_product, base_p_param)
                p_params_functions[p_param_function] = p_params_functions.get(p_param_function, [])+[base_p_param]
        
                # Make sure that the projection of p_param_function is also defined
                p_params_projs[p_param_function] = p_params_projs[p_param]
        
            
        product_slices = {}
        for p_param in p_params_functions:
            product, param = self.get_product_and_param(p_param)
            self.product = product
            
            for attr in ('bottom_scan_removed', 'scans'):
                self.__dict__[attr] = self.__dict__[attr+'_products'][self.product]
            
            self.proj = p_params_projs[p_param]
            for attr in ('i_p', 'scans', 'scans_all', 'data_all', 'scanangles_all', 'radial_bins_all', 'radial_res_all', 
                         'radial_range_all', 'scantimes_all', 'radius_offsets_all'):
                self.mapping_classes[self.proj].__dict__[attr] = self.__dict__[attr]

            # ZPHI-verzwakkingscorrectie (nlr_attenuation.py), UITSLUITEND voor MESH/POSH/POH/SHI
            # (self.hail_attenuation_products) - 28 juli 2026, op Eriks expliciete keuze. De correctie
            # vervangt hier alleen de data_all die op de mapping-class-instantie (Polar/Cartesian) is
            # gezet, NOOIT self.data_all zelf (dat blijft de gedeelde, ongecorrigeerde bron voor
            # VIL/echotop/Zmax/PCAPPI/etc. die verderop in dezelfde volume-weergave nog aan de beurt komen).
            #
            # BELANGRIJK (cache-veiligheid): calculate_Zmax_and_Zavg_3D hieronder heeft een eigen
            # memoisatie (self.mapping_classes[self.proj].data_specs), die NIET afhangt van de inhoud
            # van data_all, alleen van get_import_data_specs() (radar/dataset/tijd). Zonder de expliciete
            # data_specs=None hieronder zou een hagelproduct na een net berekend niet-hagelproduct (of
            # andersom) de VERKEERDE (gecorrigeerde of ongecorrigeerde) Zavg_3D_all kunnen hergebruiken
            # i.p.v. opnieuw te rekenen - exact het soort stille-cache-bug waar dit project al meermaals
            # door is geraakt (MESH-kalibratie/VILD/in-memory-cache, zie projectgeheugen). Daarom wordt
            # de cache hier zowel vóór als NA de aanroep geforceerd ongeldig gemaakt bij een hagelproduct,
            # zodat zowel dit product als het eerstvolgende product (hagel of niet) altijd vers rekent.
            if self.product in self.hail_attenuation_products:
                # LET OP: data_all wordt hier ALTIJD expliciet gezet (nooit stilzwijgend overgeslagen),
                # ook als de correctie uit staat - anders zou een eerder hagelproduct MET correctie
                # (dat data_all op de mapping-class-instantie overschreef) zijn gecorrigeerde data_all
                # laten hangen voor een volgend hagelproduct waarvoor de correctie net uit staat. Beide
                # takken zetten data_all dus altijd expliciet (gecorrigeerd of het originele self.data_all)
                # en invalideren de cache, zodat er nooit kruisbesmetting tussen producten optreedt.
                if getattr(self.gui, 'attenuation_correction_enabled', True):
                    self.mapping_classes[self.proj].data_all = self.get_hail_corrected_data_all()
                else:
                    self.mapping_classes[self.proj].data_all = self.data_all
                self._invalidate_hail_mapping_caches()

            self.mapping_classes[self.proj].get_product_dimensions()
            self.product_shape = self.mapping_classes[self.proj].get_product_shape()
            self.product_slice = self.mapping_classes[self.proj].get_product_slice()
            product_slices[p_param] = self.product_slice
            
            self.mapping_classes[self.proj].get_parameters_data_mapping()
            self.mapping_classes[self.proj].calculate_Zmax_and_Zavg_3D()
            if self.product in self.hail_attenuation_products:
                self._invalidate_hail_mapping_caches()
            if self.product == 'zc' and self.proj == 'pol':
                # ZDR-kolomdiepte (27 juli 2026): eigen, MAX-only ZDR-rooster, los van het Z-rooster
                # hierboven - zie Polar.calculate_ZDRmax_3D. Alleen 'pol' (product 'zc' staat niet in
                # gv.plain_products_correct_for_SM, dus proj is hier altijd 'pol', maar de check kost
                # niets en documenteert de aanname expliciet).
                self.mapping_classes[self.proj].calculate_ZDRmax_3D()
            self.mapping_classes[self.proj].sort_data_and_get_hdiffs_if_necessary()
            for parameter in self.mapping_parameters:
                if hasattr(self.mapping_classes[self.proj], parameter):
                    self.__dict__[parameter] = self.mapping_classes[self.proj].__dict__[parameter]
                
            self.plain_products_functions['independent'][self.product](param)
            self.product_arrays[p_param] = self.product_array # Gets defined in the plain product function
                                   
            volume_starttime, volume_endtime = self.volume_starttime, self.volume_endtime             
            # In the case of a double volume the volume start and end time should be adjusted
            if len(self.dsg.scannumbers_all['z'][product])==2:
                i = self.double_volume_index
                volume_starttime, volume_endtime = ft.get_start_and_end_volumetime_from_scantimes([self.scantimes_all[j][-i] for j in self.dsg.scans_doublevolume[i]])
            volumetime = volume_starttime+'-'+volume_endtime                
            self.producttimes[p_param] = volumetime if self.proj == 'pol' or all([i[0] == volumetime for i in self.scantimes_all.values()]) else\
                                         self.mapping_classes['car'].avg_scantime

            if self.product in gv.plain_products_show_max_elevations:
                self.get_maxelevations_plainproducts()
                
            self.p_param_attributes[p_param] = {}
            for attr in self.product_attributes.get(product, []):
                self.p_param_attributes[p_param][attr] = self.__dict__[attr]
            for attr in self.product_proj_attributes[self.proj]:
                self.p_param_attributes[p_param][attr] = self.mapping_classes[self.proj].__dict__[attr]
            self.p_param_attributes[p_param]['proj'] = self.proj
            
            for p_param2 in p_params_functions[p_param]:
                # Update volume times and attributes for p_params that were calculated within the same function call
                self.producttimes[p_param2] = self.producttimes[p_param]
                self.p_param_attributes[p_param2] = self.p_param_attributes[p_param]
            
            self.crd.increase_sleeptime_after_plotting = True
            

        for p_param in p_params_bases: #Calculate products that depend on other products
            base_p_params = p_params_bases_all[p_param] # Use all base_p_params, not just the ones that had to be recalculated
            first_base = base_p_params[0]
            if not p_param == first_base:
                product, param = self.get_product_and_param(p_param)
                bins = [self.p_param_attributes[base]['product_radial_bins' if p_params_projs[base] == 'pol' else\
                                                  'product_xy_bins'] for base in base_p_params]
                # The below way of determining self.product_slice only works when the base_p_params together have not more than 2 different dimensions
                start = (max(bins)-min(bins))//2
                end = max(bins)-start
                self.product_slice = np.s_[start:end, start:end]
                self.plain_products_functions['dependent'][product](param)
                self.product_arrays[p_param] = self.product_array # Gets defined in the plain product function 
                self.producttimes[p_param] = self.producttimes[first_base]
                
                self.p_param_attributes[p_param] = {}
                # BUGFIX (25 juli, na Eriks crash "KeyError: elevations_minside" bij VILD/ALT+L): kopieerde
                # voorheen ALLEEN van first_base (VIL). Voor 'r' (met 1 base) maakte dat niets uit, maar
                # VILD heeft 2 bases (VIL EN ETH) - als om wat voor reden dan ook niet alle benodigde
                # attributen (bv. elevations_minside/plusside, scans_ranges) in first_base's eigen
                # attributenset zaten, ontbraken ze voor VILD helemaal. Nu van ALLE bases samenvoegen
                # (eerste basis heeft voorrang bij overlap, latere bases vullen alleen ontbrekende sleutels
                # aan) - kan niet slechter uitpakken dan voorheen, en dekt dit scenario alsnog af.
                for base in base_p_params:
                    for attr in self.p_param_attributes[base]:
                        if attr not in self.p_param_attributes[p_param]:
                            self.p_param_attributes[p_param][attr] = self.p_param_attributes[base][attr]


        for panel in panellist:                
            p_param = p_params_panels[panel]
            product, param = self.get_product_and_param(p_param)
            proj = self.p_param_attributes[p_param]['proj']

            if p_param in p_params_bases and (proj == 'pol' or self.gui.current_case_shown(mode='loose')) and self.product_arrays[p_param].dtype == 'float32':
                # First condition means that it had to be calculated, and second is included because only polar products
                # and cartesian products for cases are saved (cartesian products are storm motion dependent, and therefore quite variable).
                # Third is included to prevent that a file is created more than once when a p_param is displayed in multiple panels
                self.write_file(p_param, proj)
            elif self.product_arrays[p_param].dtype == 'float32':
                # This is done here instead of in self.dsg, because it's possible that one p_param is shown in more than
                # one panel, in which case more than one panel would use the same data array. That is normally not a problem,
                # except when values within this same array get converted multiple times to integers. And doing the conversion
                # here prevents that from happening in self.dsg, and thereby fixes an observed bug
                self.product_arrays[p_param] = self.dsg.convert_dtype_float_to_uint(self.product_arrays[p_param], product)
            
            for attr in self.p_param_attributes[p_param]:
                self.meta_PP[product][attr.replace('product_', '')] = self.p_param_attributes[p_param][attr]

            self.dsg.data[panel] = self.product_arrays[p_param]
            self.dsg.scantimes[panel] = self.producttimes[p_param]
            self.crd.using_unfilteredproduct[panel] = self.using_unfilteredproduct
            self.crd.using_verticalpolarization[panel] = False
               
        
    def get_import_data_specs(self):
        radar_dataset = self.dsg.get_radar_dataset()
        subdataset = self.dsg.get_subdataset(product=self.i_p)
        return radar_dataset+subdataset+str(self.dsg.total_files_size)+self.crd.date+self.crd.time+str(self.productunfiltered)

    def import_data_plain_products(self):
        """The extension _all is used for scans, scanangles and data when importing data, where all data is imported
        that is required for one of the derived products, such that it has to be done only once per volume.
        Because not all products require the use of all scans however, self.scans etc. are used in other functions, and contain only the data
        used for a particular product. The exceptions are functions that calculate things for multiple products in general, like 
        self.assign_radarbins_to_productbins, self.calculate_Zmax_3D and self.calculate_Zavg_3D, self.get_range_3D and 
        self.get_heights_3D.
        
        Further, for the new radars of the KNMI it is the case that the volume can be divided into 2 parts, giving a time resolution of 2.5 minutes
        for some products. These 2 parts do not contain the same scans however, because when going through the scans from bottom to top, then the
        the scans belong alternately to one or the other part of the volume. This has as a disadvantage that it can give a flip-flop effect when 
        viewing derived products for which the volume is divided into 2 parts. The exception is a PCAPPI at a height just above the ground, because 
        the lowest scan (0.3 degree) is included in both parts of the volume.
        This 'double volume' is handled by adding the derived products to the keys in self.dsg.scannumbers_all['z'] etc. and
        self.dsg.scannumbers_forduplicates, such that it can be handled in the same way as duplicate scans are handled. 
        """
        if self.get_import_data_specs() != self.import_data_specs:
            self.scans_all=[i for i, j in self.dsg.scanangles_all[self.i_p].items() if j != 90.]                            
            self.scanangles_all = {j:self.dsg.scanangle(self.i_p, j, 0) for j in self.scans_all}
            self.radial_bins_all = {j:self.dsg.radial_bins_all[self.i_p][j] for j in self.scans_all}
            self.radial_res_all = {j:self.dsg.radial_res_all[self.i_p][j] for j in self.scans_all}
            self.radial_range_all = {j:self.dsg.radial_range_all[self.i_p][j] for j in self.scans_all}
            
            #self.data_all contains for each scan a list of data arrays, which will usually have length 1, except for a duplicate scan in the case of
            #a double volume.
            self.data_all, self.scantimes_all, self.volume_starttime, self.volume_endtime, meta = self.dsg.get_data_multiple_scans(self.i_p, self.scans_all, productunfiltered=self.productunfiltered)
            self.using_unfilteredproduct = meta['using_unfilteredproduct']
            self.radius_offsets_all = meta.get('radius_offsets', {j:0 for j in self.data_all})
            
            self.import_data_specs = self.get_import_data_specs()

    def _invalidate_hail_mapping_caches(self):
        """Invalideert ALLE caches op de mapping-class-instantie (Polar/Cartesian) die de
        hagelproduct-Zavg_3D beinvloeden - niet alleen data_specs (die calculate_Zmax_and_Zavg_3D
        zelf bewaakt), maar OOK de aparte, TWEEDE cache-laag in sort_data_and_get_hdiffs_if_necessary
        (sorted_data/sorted_data_specs in cartesian.py, product_data/product_data_specs in polar.py).

        BUG GEVONDEN 28 juli 2026 (na lang zoeken met Erik, zie sessie-overleg): die tweede cache-laag
        gebruikt DEZELFDE sleutel (self.get_data_specs(), radar/dataset/tijd) als data_specs, maar wordt
        NOOIT geraakt door data_specs=None hierboven - het is een VOLLEDIG aparte dict. Gevolg: ook al
        werd Zavg_3D_all door calculate_Zmax_and_Zavg_3D telkens correct vers herberekend (bevestigd via
        get_hail_corrected_data_all's scan-diffs in de console), sort_data_and_get_hdiffs_if_necessary gaf
        ALTIJD het allereerste (gecachete) Zavg_3D/Zmax_3D terug zodra die sleutel 1x gezet was - want die
        sleutel verandert niet door het aan/uit zetten van de verzwakkingscorrectie. Dit verklaarde
        waarom MESH's uiteindelijke mm-waarde bit-voor-bit identiek bleef (bevestigd met een print vlak
        voor de self.dsg.data[panel]-toekenning) ondanks een correct werkende Z-correctie eronder.
        """
        mc = self.mapping_classes[self.proj]
        mc.data_specs = None
        for attr in ('sorted_data', 'sorted_data_specs', 'product_data', 'product_data_specs'):
            if hasattr(mc, attr):
                mc.__dict__[attr] = {}

    def get_hail_corrected_data_all(self):
        """Bouwt een ZPHI-verzwakkingsgecorrigeerde versie van self.data_all (nlr_attenuation.py),
        UITSLUITEND voor gebruik door de hagelproducten MESH/POSH/POH/SHI (self.hail_attenuation_products),
        op Eriks expliciete keuze om dit NIET toe te passen op de gewone Z-weergave of andere Z-afgeleide
        producten (VIL/echotop/Zmax/PCAPPI) die dezelfde self.data_all delen - 28 juli 2026.

        Haalt PhiDP ('p') en RhoHV ('c') op voor dezelfde self.scans_all als waarmee self.data_all zelf
        is gevuld (zie import_data_plain_products hierboven), en corrigeert per scan/duplicate-paar.
        Gecachet per get_import_data_specs() (LRU, max self._hail_corrected_data_all_cache_maxsize
        tijdstippen) zodat dit niet 4x wordt herhaald binnen hetzelfde volume (1x per hagelproduct)
        EN niet opnieuw hoeft bij het terugbladeren naar een al eerder bezocht tijdstip (21 september
        2026, Eriks verzoek - voorheen werd bij elke stap, ook terug, alles opnieuw berekend). Bij een
        fout (bv. PhiDP niet beschikbaar voor deze radar) wordt het ongecorrigeerde self.data_all
        teruggegeven - HCLASS/MESH/etc. draaien dan gewoon door zonder correctie, in plaats van te
        crashen; dat resultaat wordt bewust NIET gecached, zodat een tijdelijke fout (bv. een nog niet
        volledig binnengekomen bestand) bij een latere poging voor hetzelfde tijdstip opnieuw geprobeerd
        wordt in plaats van permanent ongecorrigeerd te blijven.
        """
        cache_key = self.get_import_data_specs()
        cached = self._hail_corrected_data_all_cache
        if cache_key in cached:
            cached.move_to_end(cache_key)  # meest recent gebruikt -> naar achteren (LRU)
            return cached[cache_key]

        try:
            data_phidp, _, _, _, _ = self.dsg.get_data_multiple_scans(
                'p', self.scans_all, productunfiltered=self.productunfiltered)
            data_rhohv, _, _, _, _ = self.dsg.get_data_multiple_scans(
                'c', self.scans_all, productunfiltered=self.productunfiltered)

            corrected = {}
            for j in self.scans_all:
                corrected[j] = []
                range_res_km = self.radial_res_all[j]
                for i in range(len(self.data_all[j])):
                    z_arr = self.data_all[j][i]
                    try:
                        phidp_arr = data_phidp[j][i]
                        rhohv_arr = data_rhohv[j][i] / 100.  # zelfde 0-100 -> 0-1 conventie als elders
                        if phidp_arr.shape == z_arr.shape and rhohv_arr.shape == z_arr.shape:
                            z_corr = att.correct_scan_zphi(z_arr, phidp_arr, rhohv_arr, range_res_km)
                        else:
                            print(f"get_hail_corrected_data_all: vorm-mismatch scan {j} duplicate {i} - "
                                  f"Z={z_arr.shape} PhiDP={phidp_arr.shape} RhoHV={rhohv_arr.shape} - GEEN correctie toegepast")
                            z_corr = z_arr
                    except Exception as e_inner:
                        print(f"get_hail_corrected_data_all: FOUT bij scan {j} duplicate {i}: {e_inner}")
                        traceback.print_exception(type(e_inner), e_inner, e_inner.__traceback__)
                        z_corr = z_arr
                    corrected[j].append(z_corr)
        except Exception as e:
            print(e, 'get_hail_corrected_data_all')
            traceback.print_exception(type(e), e, e.__traceback__)
            # Bewust NIET cachen (zie docstring) - dit tijdstip krijgt bij een volgend
            # bezoek opnieuw een kans op een geslaagde correctie.
            return self.data_all

        cached[cache_key] = corrected
        if len(cached) > self._hail_corrected_data_all_cache_maxsize:
            cached.popitem(last=False)  # oudste (minst recent gebruikte) eruit
        return corrected

    def get_info_per_product(self):
        self.bottom_scan_removed_products = {}; self.scans_products = {}; self.scanangles_products = {}
        self.products_per_indices = {}
        
        for product in self.plain_products_functions['independent']:
            self.bottom_scan_removed_products[product] = False
            
            if len(self.scans_all) > 1 and\
            ft.rndec(self.scanangles_all[self.scans_all[1]]-self.scanangles_all[self.scans_all[0]],3)<=0.1: 
                if product in ('a','m'):
                    if len(self.dsg.scannumbers_all['z'][product])==2:
                        #This is the case for the plain products in plain_products_affected_by_double_volume when having a double volume.                                               
                        #Take the first or second series of scans, depending on the value of self.double_volume_index.
                        self.scans_products[product] = self.dsg.scans_doublevolume[self.double_volume_index]
                        self.scanangles_products[product] = {j:self.dsg.scanangles_all[self.i_p][j] for j in self.scans_products[product]}
                    else:
                        self.scans_products[product] = self.scans_all.copy()
                        self.scanangles_products[product] = self.scanangles_all.copy()
                else:
                    self.bottom_scan_removed_products[product] = True
                    #Because the bottom scan is usually rather noisy it is removed, except when the difference between the lowest two scans is greater than 
                    #0.1 degrees, too prevent throwing away too much data. The exception is 'm', where the bottom scan is used at ranges that are not 
                    #spanned by other scans.
                    self.scans_products[product] = self.scans_all[1:]
                    self.scanangles_products[product] = {j:self.scanangles_all[j] for j in self.scans_products[product]}
            else:
                self.scans_products[product] = self.scans_all.copy()
                self.scanangles_products[product] = self.scanangles_all.copy()
            
            s = self.get_indices_scans(product, self.scans_products[product])
            key = str(s)
            if not key in self.products_per_indices:
                self.products_per_indices[key] = []
            self.products_per_indices[key].append(product)
        
    
    def get_maxelevations_plainproducts(self):
        #Function expects the scan range to increase for decreasing scanangle.
        self.scans_ranges = [[self.scans[j], self.radial_bins_all[self.scans[j]]] for j in\
                             range(len(self.scans[:-1])) if self.radial_bins_all[self.scans[j]]-2 >\
                             self.radial_bins_all[self.scans[j+1]]]
        self.scans_ranges.append([self.scans[-1],self.radial_bins_all[self.scans[-1]]])
        scanangles_maxside = np.asarray([self.scanangles_all[j[0]] for j in self.scans_ranges])
        scanangles_minside = np.append(self.scanangles_all[self.scans[0]],np.asarray([self.scanangles_all[j[0]] for j in self.scans_ranges])[:-1])
        self.scans_ranges = np.asarray([j[1] for j in self.scans_ranges])
        if not self.bottom_scan_removed:
            scanangles_minside = scanangles_minside[1:]
            self.elevations_minside = ft.r1dec(ft.var1_to_var2(self.scans_ranges[1:],scanangles_minside)[1])
        else:
            scanangles_minside[0] = self.scanangles_all[self.scans_all[0]] #Use the removed scanangle as the first scanangle
            self.elevations_minside = ft.r1dec(ft.var1_to_var2(self.scans_ranges,scanangles_minside)[1])
        self.elevations_plusside = ft.r1dec(ft.var1_to_var2(self.scans_ranges,scanangles_maxside)[1])
        
    def get_true_elevations_plainproducts(self, panellist, heights_scan1, radii_scan1):
        """For the products in plain_products_show_true_elevations this function determines the 'true' elevations at which the product is displayed.
        For the PCAPPI,'a', these elevations are given by the elevations for the lowest scanangle, plus two times the PCAPPI height, where
        the ring for the first PCAPPI height is shown at the range at which the highest scan reaches this height, and the second at the range at which 
        the lowest scan reaches the PCAPPI height. Then finally, some rings are removed when they are to close together.
        
        Function is called from within self.pb.get_parameters_heightrings, because these rings must be updated after every pan or zoom action.
        """
        heights, radii = {j:heights_scan1 for j in panellist}, {j:radii_scan1 for j in panellist}
        
        x = ((self.pb.corners[0][-1,0]-self.pb.corners[0][0,0])*self.pb.ncolumns)
        dr = x/8
        for j in panellist:
            if self.crd.products[j] in gv.plain_products_show_true_elevations:
                if self.crd.products[j]=='a': CAPPI_height = self.gui.PP_parameter_values['a'][self.gui.PP_parameters_panels[j]]
                elif self.crd.products[j]=='r': CAPPI_height = gv.CAPPI_height_R
                #Range at which the highest scan reaches a height equal to CAPPI_height
                angles = np.sort([self.dsg.scanangle(gv.i_p['a'], j, self.double_volume_index) for j in self.dsg.scanangles_all[gv.i_p['a']]])
                CAPPI_height_r1 = ft.var1_to_var2(CAPPI_height, angles[angles != 90.][-1], 'h+theta->gr')
                #Range at which the lowest scan reaches a height equal to CAPPI_height
                CAPPI_height_r2 = ft.var1_to_var2(CAPPI_height, self.dsg.scanangle(gv.i_p['a'], 1, self.double_volume_index), 'h+theta->gr')
                heights_greater_than_CAPPI_height = heights[j]>CAPPI_height
                heights[j] = np.append([CAPPI_height, CAPPI_height],heights[j][heights_greater_than_CAPPI_height])
                radii[j] = np.append([CAPPI_height_r1, CAPPI_height_r2],radii[j][heights_greater_than_CAPPI_height])
                
                retain = np.ones(len(radii[j]),dtype='bool')
                if radii[j][1]-radii[j][0] < 0.2*dr:
                    retain[0] = 0
                if len(radii[j]) > 2 and radii[j][2]-radii[j][1] < 0.5*dr:
                    retain[2] = 0
                
                heights[j] = heights[j][retain]
                radii[j] = radii[j][retain]
        return heights, radii
         
               
    def get_subscan_index(self, scan):
        #For radars with a double volume, this index indicates which part of the volume is used now in the calculations. It is 0 when viewing the first
        #part, and 1 when viewing the second part. 
        #For radars without a double volume, the index is always 0.
        if self.crd.radar in gv.radars_with_double_volume:
            index = min([self.double_volume_index, len(self.dsg.scannumbers_all['z'][scan])-1])
        else:
            index = 0
        return index
    
    def index_all(self, scan, sub_scan = None):
        if sub_scan is None:
            sub_scan = self.get_subscan_index(scan)
        return sum([len(self.data_all[j]) for j in self.scans_all if j<scan])+sub_scan
    
    def get_indices_scans(self, product=None, scans=None):
        if not product:
            product = self.product
        if not scans:
            scans = self.scans
        indices = [self.index_all(j) for j in scans]
        return indices if scans != self.scans_all or product in self.products_requiring_heightsorted_data else np.s_[:]
    
    def get_highest_notempty_scan(self):        
        self.highest_notempty = np.zeros(self.product_shape, dtype = 'int8')
        empty = np.ones(self.product_shape, dtype = bool)
        Z_empty = -30.
        for i in range(len(self.scans)-1,-1,-1):
            update = empty & (self.Zavg_3D[i] != Z_empty)
            self.highest_notempty[update] = i
            empty[update] = 0
            # The lines below are needed because at the edge of the range of a group of scans it can happen
            # that due to translation out-of-range bins (that have a height of 0) are positioned below inside-range bins.
            # These lines ensure that any reflectivity value above a 0-height is not included, to prevent issues with negative hdiffs
            height_0 = self.heights_3D[i] == 0.
            empty[height_0] = 1
        
    
    def calculate_echotops(self, param): 
        min_dBZ_value_echotops = float(param)
        self.product_array = np.full(self.product_shape, -1000., dtype='float32')
        
        product_array_update_before = None
        for i in range(len(self.heights_3D)):
            Z_i = self.Zmax_3D[i]
            h_i = self.heights_3D[i]
            product_array_update = Z_i >= min_dBZ_value_echotops
            self.product_array[product_array_update] = h_i[product_array_update]
            if i > 0:
                difference = product_array_update_before & ~product_array_update & (Z_i != -30.)
                Z_imin1 = self.Zmax_3D[i-1]
                self.product_array[difference] += (h_i[difference]-self.product_array[difference])*\
                    (Z_imin1[difference]-min_dBZ_value_echotops)/(Z_imin1[difference]-Z_i[difference])
            product_array_update_before = product_array_update.copy()

        
    def calculate_echobase(self, param=None):
        """
        Echo base (24 juli 2026, op Eriks verzoek, sneltoets ALT+E - analoog aan hoe iRadar dit doet):
        de LAAGSTE hoogte waarboven de reflectiviteit een vaste drempel overschrijdt - het spiegelbeeld van
        calculate_echotops hierboven (die juist de HOOGSTE zo'n hoogte geeft). Zelfde 'plain product'-
        architectuur en interpolatiegedachte (lineair interpoleren tussen de scan waar de drempel nog niet
        en de scan waar hij net wel wordt overschreden), maar dan voor de EERSTE overschrijding i.p.v. de
        laatste - vandaar de `~found`-voorwaarde hieronder, die ervoor zorgt dat een kolom NIET meer wordt
        bijgewerkt zodra er al een eerste overschrijding is vastgesteld (calculate_echotops laat een kolom
        juist bij ELKE latere overschrijding weer overschrijven, wat daar precies goed is voor "hoogste").

        Gebruikt bewust een VASTE drempel (net als POH's 45dBZ hierboven), niet instelbaar via SHIFT+Q zoals
        bij ETH - dat vergt nog niet-toegevoegde UI-plumbing (plain_products_with_parameters/
        PP_parameter_values). Kan later alsnog instelbaar gemaakt worden, als Erik dat wil.

        Kleurtabel: eigen, nieuwe colortable_ECHOBASE_default.csv (rechtstreeks uit Eriks ECHOBASE.PNG-
        screenshot geëxtraheerd), NIET de bestaande ETH-tabel.
        """
        ECHOBASE_DBZ_THRESHOLD = 18.5  # Zelfde als ETH-preset 1 (gv.PP_parameter_values['e'][1])

        self.product_array = np.full(self.product_shape, -1000., dtype='float32')
        found = np.zeros(self.product_shape, dtype=bool)
        for i in range(len(self.heights_3D)):
            Z_i = self.Zmax_3D[i]
            h_i = self.heights_3D[i]
            update_now = (Z_i >= ECHOBASE_DBZ_THRESHOLD) & ~found
            self.product_array[update_now] = h_i[update_now]
            if i > 0:
                Z_im1 = self.Zmax_3D[i-1]
                h_im1 = self.heights_3D[i-1]
                interp = update_now & (Z_im1 != -30.) & (Z_im1 < ECHOBASE_DBZ_THRESHOLD)
                self.product_array[interp] = h_im1[interp] + (h_i[interp]-h_im1[interp]) * \
                    (ECHOBASE_DBZ_THRESHOLD-Z_im1[interp])/(Z_i[interp]-Z_im1[interp])
            found |= update_now


    def calculate_Zmax(self, param):
        Zmax_minheight = float(param)
        Zmax_3D = self.Zmax_3D.copy()
        Zmax_3D[:-1][self.heights_3D[:-1]<Zmax_minheight] = -30.

        self.product_array = np.max(Zmax_3D, axis = 0)            
        empty = self.product_array==-30.
        self.product_array[empty] = self.pb.mask_values['m']
    
        
    def calculate_PCAPPI(self, param):
        """Calculate a PCAPPI, by interpolating between the two scans that surround the PCAPPI level. If PCAPPI_height is smaller than the
        minimum height for which data is available, then data for the lowest scan is shown at those positions. If PCAPPI_height is larger than
        the maximum height for which data is available, then data for the highest scan for which data is available is shown.
        """
        PCAPPI_height = float(param)                
        self.get_highest_notempty_scan()
        I, J = np.indices(self.highest_notempty.shape)
        height_highest_notempty = self.heights_3D[self.highest_notempty, I, J]
        
        self.product_array = np.full(self.product_shape, -1000., dtype='float32')
        
        Z_bottom = self.product_array.copy(); Z_top = self.product_array.copy()
        hdiff_bottom = np.zeros(self.product_array.shape, dtype='float32'); hdiff_top = hdiff_bottom.copy()
        
        for i in range(len(self.heights_3D)):
            Z_bottom_update = (self.heights_3D[i]<PCAPPI_height) & (self.heights_3D[i] > 0.)
            Z_bottom[Z_bottom_update] = self.Zavg_3D[i,Z_bottom_update]
            hdiff_bottom[Z_bottom_update] = np.abs(PCAPPI_height-self.heights_3D[i,Z_bottom_update])
            
            Z_top_update = (Z_top==-1000.) & (self.heights_3D[i]>PCAPPI_height)
            Z_top[Z_top_update] = self.Zavg_3D[i,Z_top_update]
            hdiff_top[Z_top_update] = np.abs(PCAPPI_height-self.heights_3D[i,Z_top_update])
        Zb = Z_bottom==-1000.; Zt = Z_top==-1000.

        hdiff_sum = hdiff_bottom+hdiff_top; hdiff_sum_nonzero = hdiff_sum!=0.
        #Update the bins in self.product_array for which hdiff_sum_nonzero==True
        self.product_array[hdiff_sum_nonzero] = (Z_bottom[hdiff_sum_nonzero]*hdiff_top[hdiff_sum_nonzero]+Z_top[hdiff_sum_nonzero]*hdiff_bottom[hdiff_sum_nonzero])/hdiff_sum[hdiff_sum_nonzero]
        self.product_array[height_highest_notempty < PCAPPI_height] = -1000.
        
        #Update the bins in self.product_array for which hdiff_sum_nonzero==False, but for which this occurs because Z_bottom or Z_top is available 
        #'exactly' at the CAPPI height.
        Z_bottom_exactly_at_CAPPI_height = (hdiff_sum_nonzero==False) & (Zb==False)
        Z_top_exactly_at_CAPPI_height = (hdiff_sum_nonzero==False) & (Zt==False)
        self.product_array[Z_bottom_exactly_at_CAPPI_height] = Z_bottom[Z_bottom_exactly_at_CAPPI_height]
        self.product_array[Z_top_exactly_at_CAPPI_height] = Z_bottom[Z_top_exactly_at_CAPPI_height]
        
        self.product_array[Zt] = self.Zavg_3D[-1, Zt]
        self.product_array[Zb] = self.Zavg_3D[0, Zb]           
        self.product_array[np.abs(self.product_array+35.)<0.1] = self.pb.mask_values['a']
                
        
    def calculate_VIL(self, param = None):
        """
        An important remark is that the part of the VIL below the lowest scan is calculated by assuming that the reflectivity there is the same as at the
        lowest scan. This is in a lot of cases a bad assumption, because of vertical tilting of storms with height due to the fact that different scans 
        are performed at different times, among other things. Just neglecting this part of the VIL can result however in a substantial underestimation, 
        and the current assumption then seems better.
        """            
        cap_dBZ, VIL_minheight = eval(param)
        # Multiple height p_params might be calculated in the same VIL call, as long as only the VIL threshold varies.
        p_params_h = [p_param for p_param in self.base_p_params if p_param.startswith('h_')]
        calculate_h = len(p_params_h) > 0
        self.get_highest_notempty_scan()
        
        self.product_array = np.zeros(self.product_shape, dtype = 'float32')
        if calculate_h:
            # Since reflectivity is capped in a different way for VIL versus CMH (see below), a different VIL array is needed for CMH.
            product_array_l = self.product_array.copy()
            product_array_h = self.product_array.copy()
            
        s = self.Zavg_3D != -30.
        Z_linear = np.zeros(self.Zavg_3D.shape, dtype = 'float32')
        Z_linear[s] = 10**(0.1*self.Zavg_3D[s])
        Zlinear_56dBZ = 10**5.6
        Z_linear_cap = np.minimum(Z_linear, Zlinear_56dBZ)
            
        for i in range(len(self.scans)-1):
            Z1, Z2 = Z_linear[i], Z_linear[i+1]
            Z1_cap, Z2_cap = Z_linear_cap[i], Z_linear_cap[i+1]
            s1, s2 = s[i], s[i+1]
            
            ss = (s1 | s2) & (self.highest_notempty >= i+1)
            Z = 0.5*(Z1[ss]+Z2[ss])
            if cap_dBZ:
                # Cap the average linear reflectivity, in agreement with https://vlab.noaa.gov/web/wdtd/-/vertically-integrated-liquid-vil-
                Z[Z > Zlinear_56dBZ] = Zlinear_56dBZ
            
            hdiffs = self.hdiffs[i+1, ss]
            if VIL_minheight:
                change = VIL_minheight > self.heights_3D[i, ss]
                hdiffs[change] = 1e3*np.maximum(0., self.heights_3D[i+1, ss][change]-VIL_minheight)
            self.product_array[ss] += (3.44*10**-6)*hdiffs*Z**(4./7.)
            
            if calculate_h:
                # Note that there is a discrepancy between how capping reflectivity is handled for VIL and for CMH. For VIL the average 
                # reflectivity is capped, while for CMH the individual reflectivities are capped. This is because there is no possibility 
                # here to cap the average reflectivity because of multiplication with height.
                z1, z2 = (Z1_cap, Z2_cap) if cap_dBZ else (Z1, Z2)
                hZ47 = 0.5*(self.heights_3D[i, ss]*z1[ss]**(4./7.)+
                            self.heights_3D[i+1, ss]*z2[ss]**(4./7.))
                product_array_h[ss] += (3.44*10**-6)*hdiffs*hZ47
                if cap_dBZ: # Otherwise use the above definition of Z
                    Z = 0.5*(z1[ss]+z2[ss])
                product_array_l[ss] += (3.44*10**-6)*hdiffs*Z**(4./7.)
            
            if i == 0:
                hdiffs = 1e3*np.maximum(0., self.heights_3D[0, s1]-VIL_minheight) if VIL_minheight else self.hdiffs[0, s1]
                z1 = Z1_cap if cap_dBZ else Z1
                delta_l = (3.44*10**-6)*hdiffs*z1[s1]**(4./7.)
                self.product_array[s1] += delta_l
                if calculate_h:
                    hZ47 = 0.5*self.heights_3D[0, s1]*z1[s1]**(4./7.)
                    product_array_h[s1] += (3.44*10**-6)*hdiffs*hZ47
                    product_array_l[s1] += delta_l
                    
        select = self.product_array < 1e-5
        if calculate_h:
            product_array_h /= product_array_l
            for p_param in p_params_h:
                _, param_h = self.get_product_and_param(p_param)
                VIL_threshold = eval(param_h)[1]
                select_h = self.product_array < VIL_threshold
                self.product_arrays[p_param] = product_array_h.copy()
                self.product_arrays[p_param][select_h] = self.pb.mask_values['h']
        self.product_array[select] = self.pb.mask_values['l']
        
        
    def calculate_MESH(self, param=None):
        """
        MESH (Maximum Estimated Size of Hail, mm), Witt et al. (1998): "An enhanced hail
        detection algorithm for the WSR-88D", Wea. Forecasting, 13, 286-303. Zie
        nlr_mesh.py voor de volledige rekenkern (los gebouwd en getest met synthetische
        hagelprofielen: een marginaal geval geeft ~8mm, een zware kern ~35mm, een extreem
        geval ~65mm - fysisch plausibele waarden) en een uitgebreide toelichting/
        bronvermelding bij elke stap van de formule.

        Zelfde architectuur als calculate_VIL hierboven: itereert over opeenvolgende
        scan-paren en integreert (trapeziumregel-achtig, net als VIL's Z^(4/7)-integratie)
        over de hoogte - hier het hagel-kinetische-energiesignaal i.p.v. VIL's vloeibare-
        waterintegrand, gewogen naar hoogte t.o.v. het 0C/-20C-niveau. Die twee waarden
        komen, net als bij HCLASS (zie nlr_hclass.py/nlr_datasourcegeneral.py), uit het
        gedeelde temperatuurrooster (nlr_hclass.NL_GRID_POINTS) - per (x,y)-kolom in dit
        Cartesische grid wordt de echte lat/lon berekend en het dichtstbijzijnde
        roosterpunt gekozen.

        In tegenstelling tot HCLASS (dat zonder temperatuur nog steeds - minder scherp -
        classificeert, zie de bugfix van 22 juli in nlr_hclass.classify_hid) kan MESH niet
        zonder het 0C/-20C-niveau: zonder die twee hoogtes is er geen zinvolle
        gewichtsfunctie te maken, en wordt het hele paneel op 'geen data' gezet.

        LET OP - vereenvoudiging t.o.v. HCLASS (22 juli, na een AttributeError-fix): HCLASS
        gebruikt per BIN het dichtstbijzijnde roosterpunt (het paneel beslaat daar het volle
        360-graden bereik van de radar). Voor MESH wordt hier bewust maar EEN roosterpunt
        gebruikt voor het HELE paneel - het dichtstbijzijnde bij de radar zelf - in plaats
        van per kolom. Reden: het gebied waarover deze 'plain'-producten rekenen (het
        Cartesische grid rond de huidige radar, max ~250km) is sowieso al kleiner dan de
        ~150km-afstand tussen roosterpunten, dus het verschil zou toch nauwelijks meetellen.
        Bovendien wisselt de coordinatenrepresentatie hier tussen Cartesisch (self.x/self.y,
        AEQD-km) en Polair (alleen self.groundranges, 1D langs de straal) afhankelijk van
        self.proj (bepaald door of storm-motion-correctie aanstaat, zie
        gv.plain_products_correct_for_SM) - een per-kolom-versie zou dus sowieso apart
        Cartesisch/Polair-logica nodig hebben. Deze vereenvoudiging vermijdt dat.
        """
        import nlr_mesh as msh
        import nlr_hclass as hc

        self.get_highest_notempty_scan()
        self.product_array = np.zeros(self.product_shape, dtype='float32')

        self.dsg.ensure_melting_level_grid_current()
        grid_results = getattr(self.dsg, 'melting_level_grid_results', None)
        if grid_results is None:
            self.product_array[:] = self.pb.mask_values['o']
            return

        # EEN roosterpunt voor het hele paneel (zie LET OP hierboven waarom niet per kolom, zoals
        # bij HCLASS). Op Eriks verzoek (22 juli) NIET meer gekoppeld aan de radarlocatie zelf (dat
        # gaf bij radars aan de rand van het land een minder representatief punt), maar aan een vast,
        # centraal punt in Nederland. Eerst De Bilt geprobeerd (52.1N/5.18O), maar Erik vond dat niet
        # centraal genoeg; nu het midden tussen Ede (~52.05N/5.66O) en Apeldoorn (~52.21N/5.97O).
        CENTRAL_NL_LAT, CENTRAL_NL_LON = 52.13, 5.8
        grid_idx = hc.nearest_grid_index(np.array([CENTRAL_NL_LAT]), np.array([CENTRAL_NL_LON]), hc.NL_GRID_POINTS)[0]
        h0_val = grid_results[grid_idx]['h0_m']
        h_minus20_val = grid_results[grid_idx]['h_minus20_m']

        # ZICHTBAARHEID (23 juli, op Eriks verzoek): sla op welke temperatuurbron/welk station
        # en welke 0C/-20C-waarden hier zijn gebruikt, zodat nlr_plotting.py dit in een zwevend
        # tooltip kan tonen. Niet per-bin (zoals bij HCLASS), want MESH gebruikt toch maar 1
        # vast punt voor het hele paneel - dus 1 dict volstaat, geen array nodig. 'source' is
        # 'forecast_api'/'archive_api' (Open-Meteo) of 'wyoming_sounding' (het archief-fallback,
        # zie nlr_meltinglevels.py); 'model' is bij Wyoming 'station_06260' (De Bilt) of
        # 'station_10304' (Meppen), bij Open-Meteo de gebruikte modelnaam (of None).
        self.dsg.mesh_source_info = {
            'source': grid_results[grid_idx].get('source'),
            'model': grid_results[grid_idx].get('model'),
            'h0_m': h0_val,
            'h_minus20_m': h_minus20_val,
            'datetime_used': grid_results[grid_idx].get('datetime_used'),
        }

        if h0_val is None or h_minus20_val is None:
            self.product_array[:] = self.pb.mask_values['o']
            return
        h0_km = h0_val/1000.
        h_minus20_km = h_minus20_val/1000.

        # self.Zavg_3D gebruikt -30 als "geen data"-sentinelwaarde (zie calculate_VIL
        # hierboven: "s = self.Zavg_3D != -30."), niet NaN - expliciet omzetten zodat
        # msh.hail_kinetic_energy_flux dit niet per ongeluk als een geldige, zeer lage
        # dBZ-waarde meerekent (10^(-2.39+0.084*-30) is weliswaar al vrijwel 0, maar
        # expliciet is beter dan impliciet).
        no_data_3D = self.Zavg_3D == -30.
        # NIEUW (25 juli 2026): welke Edot-formule (stap 1 van de SHI-berekening) gebruikt wordt,
        # hangt sinds de vierde kalibratieoptie ('Witt 1998 (S-band, ongecorrigeerd)', zie
        # nlr_mesh.MESH_EDOT_FUNCTIONS) ook af van self.gui.mesh_calibration_setting - voorheen was
        # dit altijd de C-band-gecorrigeerde variant, ongeacht de kalibratiekeuze (die veranderde
        # alleen de latere SHI->mm-stap). Fallback op de gecorrigeerde (standaard) variant bij een
        # onbekende/ontbrekende instelling, zelfde principe als calibration_func verderop. Geldt
        # ALLEEN voor MESH - POSH gebruikt altijd de gecorrigeerde variant (zie de docstring bij
        # select_mesh_calibration_settings in nlr.py: deze instelling raakt POSH niet).
        edot_func = msh.MESH_EDOT_FUNCTIONS.get(
            getattr(self.gui, 'mesh_calibration_setting', None), msh.hail_kinetic_energy_flux)
        Edot_3D = edot_func(self.Zavg_3D) * msh.reflectivity_weight(self.Zavg_3D)
        Edot_3D[no_data_3D] = 0.

        for i in range(len(self.scans)-1):
            h1, h2 = self.heights_3D[i], self.heights_3D[i+1]
            e1, e2 = Edot_3D[i], Edot_3D[i+1]
            ss = self.highest_notempty >= i+1

            wt1 = msh.height_weight(h1[ss], h0_km, h_minus20_km)
            wt2 = msh.height_weight(h2[ss], h0_km, h_minus20_km)
            integrand_avg = 0.5*(e1[ss]*wt1 + e2[ss]*wt2)
            hdiffs = self.hdiffs[i+1, ss]
            self.product_array[ss] += msh.SHI_SCALE*hdiffs*integrand_avg

        shi = self.product_array
        empty = shi <= 0.
        # Kalibratie-keuze (24 juli, op Eriks verzoek na iRadar's MESH-kalibratie-dropdown): kies de
        # SHI->mm-functie op basis van self.gui.mesh_calibration_setting (ingesteld via
        # nlr.py's select_mesh_calibration_settings(), zie ook nlr_mesh.MESH_CALIBRATION_FUNCTIONS voor
        # de drie beschikbare formules). Fallback op Witt 1998 (de standaard) als de instelling om wat
        # voor reden dan ook ontbreekt of een onbekende naam bevat, i.p.v. een KeyError/crash te geven.
        calibration_func = msh.MESH_CALIBRATION_FUNCTIONS.get(
            getattr(self.gui, 'mesh_calibration_setting', None), msh.mesh_from_shi)
        self.product_array = calibration_func(shi)
        self.product_array[empty] = self.pb.mask_values['o']
        
        
    def calculate_POSH(self, param=None):
        """
        POSH (Probability of Severe Hail, %), Witt et al. (1998): "An enhanced hail
        detection algorithm for the WSR-88D", Wea. Forecasting, 13, 286-303 (hun
        vergelijking 5) - zie nlr_mesh.py voor de rekenkern. Rechtstreeks tegen het
        originele artikel geverifieerd (24 juli), inclusief de ondergrens op de
        waarschuwingsdrempel WT (die letterlijk zo in de brontekst staat, geen eigen
        aanname): "If WT < 20 J m^-1 s^-1, then WT is set to 20".

        EXACT dezelfde SHI-opbouw als calculate_MESH hierboven (zelfde temperatuurpunt,
        zelfde scan-paar-integratie) - alleen de laatste omzetstap verschilt: MESH gaat
        via mesh_from_shi (SHI -> mm), POSH via posh_from_shi (SHI + H0 -> %). Zie
        calculate_MESH's docstring voor de volledige toelichting bij elke stap
        hieronder; die is hier niet herhaald om duplicatie te beperken.
        """
        import nlr_mesh as msh
        import nlr_hclass as hc

        self.get_highest_notempty_scan()
        self.product_array = np.zeros(self.product_shape, dtype='float32')

        self.dsg.ensure_melting_level_grid_current()
        grid_results = getattr(self.dsg, 'melting_level_grid_results', None)
        if grid_results is None:
            self.product_array[:] = self.pb.mask_values['b']
            return

        # Zelfde vaste, centrale roosterpunt als calculate_MESH (zie de uitgebreide
        # toelichting daar voor waarom niet per kolom en niet de radarlocatie zelf).
        CENTRAL_NL_LAT, CENTRAL_NL_LON = 52.13, 5.8
        grid_idx = hc.nearest_grid_index(np.array([CENTRAL_NL_LAT]), np.array([CENTRAL_NL_LON]), hc.NL_GRID_POINTS)[0]
        h0_val = grid_results[grid_idx]['h0_m']
        h_minus20_val = grid_results[grid_idx]['h_minus20_m']

        # Zelfde zichtbaarheidsinfo als bij MESH, voor een eventuele POSH-tooltip.
        self.dsg.posh_source_info = {
            'source': grid_results[grid_idx].get('source'),
            'model': grid_results[grid_idx].get('model'),
            'h0_m': h0_val,
            'h_minus20_m': h_minus20_val,
            'datetime_used': grid_results[grid_idx].get('datetime_used'),
        }

        if h0_val is None or h_minus20_val is None:
            self.product_array[:] = self.pb.mask_values['b']
            return
        h0_km = h0_val/1000.
        h_minus20_km = h_minus20_val/1000.

        no_data_3D = self.Zavg_3D == -30.
        Edot_3D = msh.hail_kinetic_energy_flux(self.Zavg_3D) * msh.reflectivity_weight(self.Zavg_3D)
        Edot_3D[no_data_3D] = 0.

        for i in range(len(self.scans)-1):
            h1, h2 = self.heights_3D[i], self.heights_3D[i+1]
            e1, e2 = Edot_3D[i], Edot_3D[i+1]
            ss = self.highest_notempty >= i+1

            wt1 = msh.height_weight(h1[ss], h0_km, h_minus20_km)
            wt2 = msh.height_weight(h2[ss], h0_km, h_minus20_km)
            integrand_avg = 0.5*(e1[ss]*wt1 + e2[ss]*wt2)
            hdiffs = self.hdiffs[i+1, ss]
            self.product_array[ss] += msh.SHI_SCALE*hdiffs*integrand_avg

        shi = self.product_array
        empty = shi <= 0.
        self.product_array = msh.posh_from_shi(shi, h0_val)
        self.product_array[empty] = self.pb.mask_values['b']
        
        
    def calculate_POH(self, param=None):
        """
        POH (Probability of Hail, alle grootte - niet specifiek severe, %), Waldvogel et al. (1979) /
        Holleman (2001)-herijking (zie nlr_mesh.py voor de rekenkern en volledige bronvermelding,
        inclusief de verificatie tegen Lukach et al. 2017 die de exacte vergelijking bevestigt).

        In tegenstelling tot MESH/POSH hierboven (die de HELE verticale kolom integreren via SHI) is POH
        veel eenvoudiger: alleen de hoogte waarop het 45dBZ-echo voor het laatst wordt waargenomen
        (van onder naar boven kijkend - hetzelfde principe als calculate_echotops hierboven, maar dan
        met een VAST 45dBZ-drempel i.p.v. een instelbare parameter), gecombineerd met het 0C-niveau.

        Hergebruikt daarom bewust calculate_echotops' interpolatielogica (lineair interpoleren tussen de
        twee scans waartussen de 45dBZ-drempel wordt overschreden, voor een preciezere hoogte dan de
        ruwe scan-resolutie), i.p.v. calculate_MESH/POSH's trapezium-SHI-integratie - want die berekent
        iets fundamenteel anders (een energie-integraal, geen enkele hoogte).

        Gebruikt hetzelfde vaste, centrale temperatuurpunt (Ede-Apeldoorn) als MESH/POSH - zie
        calculate_MESH's docstring hierboven voor de volledige toelichting waarom niet per kolom.
        """
        import nlr_mesh as msh
        import nlr_hclass as hc

        POH_DBZ_THRESHOLD = 45.0  # zelfde drempel als Waldvogel et al. (1979)/Holleman (2001) gebruiken

        self.dsg.ensure_melting_level_grid_current()
        grid_results = getattr(self.dsg, 'melting_level_grid_results', None)
        if grid_results is None:
            self.product_array = np.full(self.product_shape, self.pb.mask_values['uh'], dtype='float32')
            return

        CENTRAL_NL_LAT, CENTRAL_NL_LON = 52.13, 5.8
        grid_idx = hc.nearest_grid_index(np.array([CENTRAL_NL_LAT]), np.array([CENTRAL_NL_LON]), hc.NL_GRID_POINTS)[0]
        h0_val = grid_results[grid_idx]['h0_m']

        # Zelfde zichtbaarheidsinfo als bij MESH/POSH, voor een eventuele POH-tooltip.
        self.dsg.poh_source_info = {
            'source': grid_results[grid_idx].get('source'),
            'model': grid_results[grid_idx].get('model'),
            'h0_m': h0_val,
            'datetime_used': grid_results[grid_idx].get('datetime_used'),
        }

        if h0_val is None:
            self.product_array = np.full(self.product_shape, self.pb.mask_values['uh'], dtype='float32')
            return
        h0_km = h0_val/1000.

        # H45dBZ-hoogte per kolom, EXACT dezelfde interpolatielogica als calculate_echotops hierboven
        # (maar met de vaste POH_DBZ_THRESHOLD i.p.v. de instelbare min_dBZ_value_echotops-parameter).
        # self.heights_3D staat AL in KM (bevestigd via calculate_echotops hierboven: geen /1000-conversie
        # daar, en 'e' (echotops) heeft als output-eenheid gewoon km) - dus GEEN extra /1000 hier nodig.
        h45_km = np.full(self.product_shape, -1000., dtype='float32')
        update_before = None
        for i in range(len(self.heights_3D)):
            Z_i = self.Zmax_3D[i]
            h_i = self.heights_3D[i]
            update_now = Z_i >= POH_DBZ_THRESHOLD
            h45_km[update_now] = h_i[update_now]
            if i > 0:
                difference = update_before & ~update_now & (Z_i != -30.)
                Z_imin1 = self.Zmax_3D[i-1]
                h45_km[difference] += (h_i[difference]-h45_km[difference]) * \
                    (Z_imin1[difference]-POH_DBZ_THRESHOLD)/(Z_imin1[difference]-Z_i[difference])
            update_before = update_now.copy()

        # BUGFIX (24 juli, na Eriks melding "geen beeld" met een bevestigd zware hagelkern - MESH gaf
        # daar 70.8mm): h45 (dus ook self.heights_3D) staat AL in km. Deze functie deelde eerder ALSNOG
        # door 1000 (in de veronderstelling dat self.heights_3D in meter stond, zoals bij calculate_MESH's
        # eigen invoer h0_val/h_minus20_val WEL het geval is - maar dat zijn andere variabelen, uit het
        # temperatuurrooster, niet uit self.heights_3D). Gevolg: het hoogteverschil met h0_km werd overal
        # enorm negatief, en POH werd overal geclipt naar 0% - onzichtbaar, ook op een zware hagelkern.
        no_echo = h45_km == -1000.
        self.product_array = 100.*msh.poh_from_height_diff(h45_km, h0_km)  # fractie 0-1 -> percentage
        self.product_array[no_echo] = self.pb.mask_values['uh']
        
        
    def calculate_SHI(self, param=None):
        """
        SHI (Severe Hail Index, J/m/s), Witt et al. (1998) - het tussenresultaat waaruit MESH
        (2.54*sqrt(SHI), zie calculate_MESH hierboven) en POSH (29*ln(SHI/WT)+50, zie
        calculate_POSH hierboven) beide worden afgeleid. Tot nu toe alleen intern berekend en
        meteen omgezet; dit product toont de ruwe SHI-waarde zelf, ongeacht die twee laatste
        omzetstappen.

        EXACT dezelfde opbouw (zelfde vaste centrale temperatuurpunt Ede-Apeldoorn, zelfde
        scan-paar-trapeziumintegratie) als calculate_MESH/calculate_POSH hierboven - zie die
        docstrings voor de volledige toelichting bij elke stap. Alleen de laatste omzetstap
        ontbreekt hier: self.product_array IS de SHI-waarde, geen verdere bewerking.

        LET OP: gebruikt, net als POSH, ALTIJD de C-band-gecorrigeerde Edot-formule (Brook et al.
        2024), ONGEACHT de MESH-kalibratiekeuze (self.gui.mesh_calibration_setting) - diezelfde
        keuze raakt immers ook POSH niet (zie select_mesh_calibration_settings' docstring in
        nlr.py). SHI is zo het ene ondubbelzinnige basissignaal waar zowel de standaard-MESH als
        POSH uit voortkomen; de vierde ("Witt 1998, S-band ongecorrigeerd") kalibratieoptie is
        uitdrukkelijk alleen een MESH-vergelijkingshulpmiddel, geen alternatieve SHI-definitie.
        """
        import nlr_mesh as msh
        import nlr_hclass as hc

        self.get_highest_notempty_scan()
        self.product_array = np.zeros(self.product_shape, dtype='float32')

        self.dsg.ensure_melting_level_grid_current()
        grid_results = getattr(self.dsg, 'melting_level_grid_results', None)
        if grid_results is None:
            self.product_array[:] = self.pb.mask_values['si']
            return

        # Zelfde vaste, centrale roosterpunt als calculate_MESH/POSH hierboven.
        CENTRAL_NL_LAT, CENTRAL_NL_LON = 52.13, 5.8
        grid_idx = hc.nearest_grid_index(np.array([CENTRAL_NL_LAT]), np.array([CENTRAL_NL_LON]), hc.NL_GRID_POINTS)[0]
        h0_val = grid_results[grid_idx]['h0_m']
        h_minus20_val = grid_results[grid_idx]['h_minus20_m']

        # Zelfde zichtbaarheidsinfo als bij MESH/POSH/POH, voor een eventuele SHI-tooltip.
        self.dsg.shi_source_info = {
            'source': grid_results[grid_idx].get('source'),
            'model': grid_results[grid_idx].get('model'),
            'h0_m': h0_val,
            'h_minus20_m': h_minus20_val,
            'datetime_used': grid_results[grid_idx].get('datetime_used'),
        }

        if h0_val is None or h_minus20_val is None:
            self.product_array[:] = self.pb.mask_values['si']
            return
        h0_km = h0_val/1000.
        h_minus20_km = h_minus20_val/1000.

        no_data_3D = self.Zavg_3D == -30.
        Edot_3D = msh.hail_kinetic_energy_flux(self.Zavg_3D) * msh.reflectivity_weight(self.Zavg_3D)
        Edot_3D[no_data_3D] = 0.

        for i in range(len(self.scans)-1):
            h1, h2 = self.heights_3D[i], self.heights_3D[i+1]
            e1, e2 = Edot_3D[i], Edot_3D[i+1]
            ss = self.highest_notempty >= i+1

            wt1 = msh.height_weight(h1[ss], h0_km, h_minus20_km)
            wt2 = msh.height_weight(h2[ss], h0_km, h_minus20_km)
            integrand_avg = 0.5*(e1[ss]*wt1 + e2[ss]*wt2)
            hdiffs = self.hdiffs[i+1, ss]
            self.product_array[ss] += msh.SHI_SCALE*hdiffs*integrand_avg

        # Geen verdere omzetstap - self.product_array is hier al de ruwe SHI-waarde zelf, in
        # tegenstelling tot calculate_MESH (mesh_from_shi) en calculate_POSH (posh_from_shi)
        # hierboven. Zelfde "leeg = geen hagelsignaal"-conventie als MESH/POSH (shi<=0 -> mask),
        # zodat SHI=0-gebieden net als daar visueel niet van "geen data" te onderscheiden zijn.
        empty = self.product_array <= 0.
        self.product_array[empty] = self.pb.mask_values['si']
        
        
    def calculate_ZDRcol(self, param=None):
        """
        ZDR-kolomdiepte (km boven het 0C-niveau), Kumjian & Ryzhkov (2008) / Snyder et al. (2015):
        een ZDR-kolom is een opwaartse uitloper van ZDR>=1dB boven het omgevings-0C-niveau,
        gekoppeld aan sterke updrafts in onweersbuien - bevestigd tegen de brontekst (27 juli 2026).

        Bepaalt de HOOGSTE hoogte waarop ZDR nog >=1dB is - zelfde interpolatielogica als
        calculate_echotops hierboven (lineair interpoleren tussen de scan waar de drempel nog wel
        wordt gehaald en de scan waar niet meer, voor een preciezere hoogte dan de ruwe
        scan-resolutie) - en trekt daar het 0C-niveau vanaf. Nooit een overschrijding gevonden ->
        mask_values (geen kolom); wel gevonden maar de top ligt op of onder het 0C-niveau -> 0
        (geclipt, geen "negatieve kolom").

        ZDR_COLUMN_THRESHOLD = 1.0 dB is de in de literatuur gangbare drempel (Kumjian & Ryzhkov
        2008 e.v.). Z_QC_THRESHOLD = 20 dBZ (27 juli 2026, op Eriks verzoek na ruis in de eerste
        versie - zie hieronder bij de constante zelf) is een EIGEN, praktische toevoeging, niet uit
        de brontekst zelf afgeleid.

        Gebruikt hetzelfde vaste, centrale temperatuurpunt (Ede-Apeldoorn) als MESH/POSH/POH/SHI -
        zie calculate_MESH's docstring voor de volledige toelichting waarom niet per kolom. Net als
        POH is alleen het 0C-niveau nodig, geen -20C.

        Gebruikt self.ZDRmax_3D (Polar.calculate_ZDRmax_3D, apart van/naast self.Zmax_3D/Zavg_3D) -
        zie de toelichting daar voor de architectuur (een eigen, MAX-only ZDR-rooster, dat wél
        dezelfde geometrische mapping-tabellen hergebruikt als de Z-berekening, maar ZELF de
        ZDR-data ophaalt, los van self.data_all dat aan Z vastzit).

        LET OP - bekende vereenvoudiging: geen storm-motion-correctie (zie
        Polar.calculate_ZDRmax_3D - 'zc' staat bewust niet in gv.plain_products_correct_for_SM).
        """
        import nlr_hclass as hc

        ZDR_COLUMN_THRESHOLD = 1.0  # dB, Kumjian & Ryzhkov (2008)
        # Z_QC_THRESHOLD (27 juli 2026, toegevoegd op Eriks verzoek nadat de eerste versie op de
        # Hilversum-case tientallen losse, geïsoleerde streepjes over het hele land liet zien - ook
        # boven zee en ver van elke bui): EIGEN, praktische toevoeging, NIET uit Kumjian & Ryzhkov
        # zelf - die definieren de kolom puur via ZDR, zonder Z-eis. Zonder deze eis telt elke losse
        # bin met toevallig hoge ZDR (ruis, clutter, insecten, tweede-trip) mee als "kolom", ook ver
        # van een storm. 20 dBZ is een gangbare, redelijke ondergrens om zulke bins te weren zonder
        # de rand van een echte kolom af te snijden.
        Z_QC_THRESHOLD = 20.0  # dBZ

        self.dsg.ensure_melting_level_grid_current()
        grid_results = getattr(self.dsg, 'melting_level_grid_results', None)
        if grid_results is None:
            self.product_array = np.full(self.product_shape, self.pb.mask_values['zc'], dtype='float32')
            return

        CENTRAL_NL_LAT, CENTRAL_NL_LON = 52.13, 5.8
        grid_idx = hc.nearest_grid_index(np.array([CENTRAL_NL_LAT]), np.array([CENTRAL_NL_LON]), hc.NL_GRID_POINTS)[0]
        h0_val = grid_results[grid_idx]['h0_m']

        # Zelfde zichtbaarheidsinfo als bij MESH/POSH/POH/SHI, voor een eventuele ZDR-kolom-tooltip
        # (nog niet aangesloten in nlr_plotting.py - 'zc' is een gewoon scalair product, dus de
        # normale muis-uitlezing werkt al automatisch; een eventuele temperatuurbron-pop-up zou
        # dezelfde toevoeging vergen als bij MESH/POSH/POH/SHI, maar is niet essentieel).
        self.dsg.zdrcol_source_info = {
            'source': grid_results[grid_idx].get('source'),
            'model': grid_results[grid_idx].get('model'),
            'h0_m': h0_val,
            'datetime_used': grid_results[grid_idx].get('datetime_used'),
        }

        if h0_val is None:
            self.product_array = np.full(self.product_shape, self.pb.mask_values['zc'], dtype='float32')
            return
        h0_km = h0_val/1000.

        # HOOGSTE hoogte waar (ZDR>=drempel EN Z>=Z_QC_THRESHOLD) - zelfde patroon als
        # calculate_echotops, met een aanvulling: de precieze interpolatie (voor een hoogte tussen
        # twee scans) wordt ALLEEN toegepast als de overgang echt door het ZDR-criterium komt (beide
        # niveaus voldeden aan de Z-eis, alleen ZDR zakte onder 1dB) - niet als de overgang komt
        # doordat Z zelf onder de kwaliteitsdrempel zakt terwijl ZDR nog hoog is. In dat laatste geval
        # blijft gewoon de laatst geldige (ruwe) hoogte staan, i.p.v. op de verkeerde variabele te
        # interpoleren.
        h_zdrtop_km = np.full(self.product_shape, -1000., dtype='float32')
        update_before = None
        for i in range(len(self.heights_3D)):
            ZDR_i = self.ZDRmax_3D[i]
            Z_i = self.Zmax_3D[i]
            h_i = self.heights_3D[i]
            update_now = (ZDR_i >= ZDR_COLUMN_THRESHOLD) & (Z_i >= Z_QC_THRESHOLD)
            h_zdrtop_km[update_now] = h_i[update_now]
            if i > 0:
                Z_imin1 = self.Zmax_3D[i-1]
                ZDR_imin1 = self.ZDRmax_3D[i-1]
                # Verfijnde interpolatie ALLEEN als beide niveaus aan de Z-kwaliteitseis voldeden en
                # ZDR zelf de 1dB-drempel kruiste (dus de overgang komt echt door ZDR, niet doordat Z
                # onder Z_QC_THRESHOLD zakte terwijl ZDR nog hoog bleef).
                both_had_quality = (Z_imin1 >= Z_QC_THRESHOLD) & (Z_i >= Z_QC_THRESHOLD)
                zdr_crossed = update_before & ~update_now & both_had_quality & (ZDR_i != -30.) & \
                    (ZDR_imin1 >= ZDR_COLUMN_THRESHOLD) & (ZDR_i < ZDR_COLUMN_THRESHOLD)
                h_zdrtop_km[zdr_crossed] += (h_i[zdr_crossed]-h_zdrtop_km[zdr_crossed]) * \
                    (ZDR_imin1[zdr_crossed]-ZDR_COLUMN_THRESHOLD)/(ZDR_imin1[zdr_crossed]-ZDR_i[zdr_crossed])
                # Overgangen die puur door de Z-eis komen: h_zdrtop_km behoudt vanzelf zijn laatst
                # geldige (ruwe) hoogte, want die is al gezet toen update_now nog True was en wordt
                # hier niet overschreven - geen extra actie nodig.
            update_before = update_now.copy()

        no_column = h_zdrtop_km == -1000.
        self.product_array = np.clip(h_zdrtop_km - h0_km, 0., None)
        self.product_array[no_column] = self.pb.mask_values['zc']
        
        
    def convert_dBZ_to_mmph(self, data, inverse = False):
        if not inverse:
            data = np.power(np.divide(np.power(10.,np.divide(data, 10.)),200.),5./8.)
        else:
            data = 10.*np.log10(200.*np.power(data, 8./5.))
        return data

    def calculate_R(self, param = None):
        self.product_array = np.log10(self.convert_dBZ_to_mmph(self.product_arrays[f'a_{gv.CAPPI_height_R}']))
        #Log10 is used because the colormap for 'r' is logarithmic.
        self.product_array[self.product_array<=1e-6]=self.pb.mask_values['r']


    def calculate_VILD(self, param=None):
        """
        VILD (VIL Density, g/m^3), 24 juli 2026 op Eriks verzoek = VIL (kg/m^2) / ETH (km) - Amburn & Wolf
        (1997): een dichtheidsmaat die minder gevoelig is voor celhoogte alleen dan VIL zelf (een kleine,
        zeer verticale cel kan een hoge VIL geven puur door de hoogte, niet per se door de hoeveelheid
        water). Vuistregel uit dat onderzoek: VILD >= 3.5 g/m^3 wordt vaak als hagelindicatie gebruikt (een
        Amerikaanse, S-band-gekalibreerde vuistregel, geen harde natuurwet - zie OVERZICHT_functies.pdf).

        'dependent' product net als calculate_R hierboven: hergebruikt de AL berekende VIL- en ETH-arrays
        (self.product_arrays) i.p.v. zelf opnieuw over de kolom te integreren. VIL-basis is preset 1
        (self.gui.PP_parameter_values['l'][1]), ETH-basis is gv.VILD_ETH_THRESHOLD (momenteel 18.5dBZ,
        zelfde als ETH-preset 1) - zie nlr_globalvars.py voor de afweging (een lagere drempel verkleint
        randartefacten rond cellen, maar maakt ETH/VILD gevoeliger voor ruis/clutter; Erik koos voor de
        betrouwbaardere, hogere drempel).
        """
        vil_key = f'l_{self.gui.PP_parameter_values["l"][1]}'
        eth_key = f'e_{gv.VILD_ETH_THRESHOLD}'
        vil_array = self.product_arrays[vil_key]
        eth_array = self.product_arrays[eth_key]

        self.product_array = np.full(vil_array.shape, self.pb.mask_values['vd'], dtype='float32')
        # Beide arrays hebben hun eigen mask-sentinelwaarde voor lege kolommen (zie calculate_VIL en
        # calculate_echotops hierboven) - die sentinels liggen bij ontwerp altijd <=0 (VIL: expliciet
        # <1e-5 gemaskeerd; ETH: -1000 voor "geen overschrijding gevonden"), dus een simpele >0-check op
        # BEIDE volstaat om alleen bins met echte data in beide arrays te delen.
        valid = (vil_array > 0.) & (eth_array > 0.)
        self.product_array[valid] = vil_array[valid]/eth_array[valid]