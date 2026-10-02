# Copyright (C) 2016-2024 Bram van 't Veen, bramvtveen94@hotmail.com
# Distributed under the GNU General Public License version 3, see <https://www.gnu.org/licenses/>.

import sys
import os
import datetime as dtime
opa = os.path.abspath
import numpy as np
import time as pytime
import pickle
import copy
import traceback

from derived.nlr_derived_plain import DerivedPlain
from derived import nlr_derived_tilts as dt
import nlr_datasourcespecific as dss
import nlr_importdata as ird
import nlr_background as bg
import nlr_functions as ft
import nlr_globalvars as gv
import nlr_hclass as hc
import nlr_meltinglevels as nml
import nlr_attenuation as att

# Load Unet_VDA in a separate thread, as it takes ~10 seconds to load
VDA = None
def init_VDA():
    global VDA
    from dealiasing.unet_vda.unet_vda import Unet_VDA
    vda = Unet_VDA()
    vda(np.zeros((360, 100)), 30, np.arange(0, 360, 1), 1)
    VDA = vda
    print('Unet VDA imported')
import threading
threading.Thread(target=init_VDA).start()



"""DataSource_General contains functions that are related to the import of data, and contains functions that can in general be used for
different data sources (by forwarding calls to functions in the classes in nlr_datasourcespecific.py).
"""

class DataSource_General():
    #Base class for importing radar data
    def __init__(self, gui_class, crd_class, parent = None):        
        self.gui = gui_class
        self.crd = crd_class
        self.dp = DerivedPlain(dsg_class = self)
        self.pb = self.gui.pb
                        
        self.Leonardo_vol_rainbow3 = ird.Leonardo_vol_rainbow3(gui_class = self.gui, dsg_class = self)
        self.Leonardo_vol_rainbow5 = ird.Leonardo_vol_rainbow5(gui_class = self.gui, dsg_class = self)
        self.KNMI_hdf5 = ird.KNMI_hdf5(gui_class = self.gui, dsg_class = self)
        self.ODIM_hdf5 = ird.ODIM_hdf5(gui_class = self.gui, dsg_class = self)
        self.skeyes_hdf5 = ird.skeyes_hdf5(gui_class = self.gui, dsg_class = self)
        self.DWD_odimh5 = ird.DWD_odimh5(gui_class = self.gui, dsg_class = self)
        self.DWD_bufr = ird.DWD_BUFR(gui_class = self.gui, dsg_class = self)
        self.TUDelft_nc = ird.TUDelft_nc(gui_class = self.gui, dsg_class = self)
        self.NEXRAD_L2 = ird.NEXRAD_L2(gui_class = self.gui, dsg_class = self)
        self.NEXRAD_L3 = ird.NEXRAD_L3(gui_class = self.gui, dsg_class = self)
        self.CFRadial = ird.CFRadial(gui_class = self.gui, dsg_class = self)
        self.DORADE = ird.DORADE(gui_class = self.gui, dsg_class = self)
        self.MeteoFrance_BUFR = ird.MeteoFrance_BUFR(gui_class = self.gui, dsg_class = self)
        self.MeteoFrance_NetCDF = ird.MeteoFrance_NetCDF(gui_class = self.gui, dsg_class = self)
        self.UKMO_polar = ird.UKMO_Polar(gui_class = self.gui, dsg_class = self)
        
        self.source_KNMI = dss.Source_KNMI(gui_class = self.gui, dsg_class = self)
        self.source_KMI = dss.Source_KMI(gui_class = self.gui, dsg_class = self)
        self.source_skeyes = dss.Source_skeyes(gui_class = self.gui, dsg_class = self)
        self.source_VMM = self.source_KMI # VMM data can be completely handled by the self.source_KMI class
        self.source_DWD = dss.Source_DWD(gui_class = self.gui, dsg_class = self)
        self.source_TUDelft = dss.Source_TUDelft(gui_class = self.gui, dsg_class = self)
        self.source_Leonardo = dss.Source_Leonardo(gui_class = self.gui, dsg_class = self)
        self.source_DMI = dss.Source_DMI(gui_class = self.gui, dsg_class = self)
        self.source_CHMI = dss.Source_CHMI(gui_class = self.gui, dsg_class = self)
        self.source_SHMU = dss.Source_SHMU(gui_class = self.gui, dsg_class = self)
        self.source_AustroControl = dss.Source_AustroControl(gui_class = self.gui, dsg_class = self)
        self.source_NWS = dss.Source_NWS(gui_class = self.gui, dsg_class = self)
        self.source_ARRC = dss.Source_ARRC(gui_class = self.gui, dsg_class = self)
        self.source_MeteoFrance = dss.Source_MeteoFrance(gui_class = self.gui, dsg_class = self)
        self.source_UKMO = dss.Source_UKMO(gui_class = self.gui, dsg_class = self)
        self.source_classes = {'KNMI':self.source_KNMI,'KMI':self.source_KMI,'skeyes':self.source_skeyes,'VMM':self.source_VMM,'DWD':self.source_DWD, 'TU Delft': self.source_TUDelft, 'IMGW': self.source_Leonardo, 'SHMU':self.source_SHMU, 'DMI': self.source_DMI, 'Austro Control':self.source_AustroControl, 'CHMI': self.source_CHMI, 'NWS': self.source_NWS, 'ARRC': self.source_ARRC, 'Météo-France': self.source_MeteoFrance, 'UKMO':self.source_UKMO}
                
        
        self.radial_res_all = {}; self.radial_bins_all = {}; self.radial_range_all = {}
        # Scanangles might vary among duplicate scans, in which case self.scanangles_all for this scan will contain a dictionary
        # with a scanangle per duplicate index. self.scanangles_all_m contains the mean scanangle over all duplicates, which in
        # absence of duplicates or in case of a constant scanangle will just be equal to the corresponding entry in self.scanangles_all.
        # When only an approximate scanangle is needed, self.scanangles_all_m should be used. self.scanangles_all should be used
        # when needing the actual scanangle for a certain duplicate. In this case the function self.scanangle should be called,
        # since it can also handle the absence of duplicates or a constant scanangle per duplicate.
        self.scannumbers_all = {}; self.scanangles_all = {}; self.scanangles_all_m = {}
        self.nyquist_velocities_all_mps = {}; self.low_nyquist_velocities_all_mps = {}; self.high_nyquist_velocities_all_mps = {}
        self.scannumbers_forduplicates = {} # Specifies which scan will be requested when two or more scans have the same properties
        # Per-panel cache of the raw (un-normalized) Z/CC/ZDR arrays most recently computed for the polarimetric
        # RGB composite (product 'g'), used by Plotting.update_data_readout to show the actual physical values
        # under the mouse cursor instead of just the displayed RGBA color. See DataSource_General._calculate_polrgb.
        self.polrgb_raw_data = {}
        # Zelfde soort per-paneel-cache als polrgb_raw_data hierboven, maar dan voor de hydrometeoren-
        # classificatie (product 'j'): (hid, data_z, data_zdr, data_kdp, data_cc, T). Zie
        # DataSource_General._calculate_hclass en nlr_hclass.py.
        self.hclass_raw_data = {}
        self.determined_volume_attributes_radars = {}
        self.scannumbers_forduplicates_radars = {}
        
        self.scans_doublevolume = []                                                                                                                                                                                    
        self.variable_attributes = []
        self.update_volume_attributes = False # Should be set to True when some volume attributes have been updated during a call
        # of get_data, like what happens with NEXRAD level II data
        
        self.savevalues_refscans_unsorted = {j:[] for j in gv.radars_all} #Is used in nlr_importdata.py
        
        self.data = {j:-1e6*np.ones((362,1)).astype('float32') for j in range(10)}
        self.scantimes = {}
        # The left edge of the first radial might not correspond to an azimuth of 0°. If that's the case, this offset value should 
        # be set for the corresponding panel. Only values between -1 and +1 are allowed, bigger offsets should be prevented by rolling
        # the data array when necessary.
        self.data_azimuth_offset = {j:0. for j in range(10)}
        # Similarly as for the azimuth, with radius offsets between -1 and +1 km supported
        self.data_radius_offset = {j:0. for j in range(10)}
        self.stored_data = {}
        # For data sources listed here, any change in product/scan availability/content should be reflected in a corresponding change in scannumbers_all.
        # For these sources scannumbers_all is used to determine whether a product/scan in memory needs to be updated, in contrast to self.total_files_size.
        # Has as advantage that product/scan is updated only when actually needed (and not when some other part of the radar volume is updated/expanded). 
        # Disadvantage is that it can be cumbersome to get product/scan availability/content reflected in scannumbers_all, especially when there's a
        # difference in availability for filtered/unfiltered products.
        self.stored_data_sources_with_scannumbers_as_content_marker = ('DWD', 'NWS')
        
        self.range_nyquistvelocity_scanpairs_indices = {j:0 for j in range(10)}
        self.scans_radars = {} #Gets updated in self.pb.set_newdata!
        self.selected_scanangles={} #selected_scanangles is used when the radar volume structure changes during the course of a day, as can be the case
        #for the radar in Zaventem. When selecting a particular scanangle, that is not available with the next volume structure, then it is chosen 
        #automatically again when the structure changes back to something that contains the selected scanangle.
        #Should not be updated when going to the left/right.
        self.selected_scanangles_before = {}
        self.panel_center_heights = {}
        self.time_last_panzoom = 0
        self.time_last_forcedscanchange = 0
        self.time_last_choosenearestheight = 0
        self.time_last_choosenearestscanangle = 0
        
        # These objects store results of listing filenames and determines corresponding datetimes, in order to reduce the need
        # to re-determine them.
        self.directories_lastupdate_times = {}
        self.filenames_directory = {}
        self.nfiles_directory = {}
        self.datetimes_directory = {}

        self.files_datetimesdict = {}
        self.files_datetime = None # Files for the current datetime
        # It could be the case that multiple versions of a product are available for a single radar volume (and that these can't be considered
        # as a filtered-unfiltered pair). The following variables contain information about this, and are set when needed.
        self.product_versions_datetimesdict = None
        self.product_versions_directory = None # Available product versions for the whole current directory. 
        self.product_versions_datetime = None # Available product versions for the current datetime.
        self.products_version_dependent = None # Products that are version-dependent. This information is used to determine whether
        # derived products should be calculated separately for each product version.
        
        # Should be updated when volume attributes for all data sources should be reset.
        self.attributes_version = 16
        # Should be updated when the structure of volume attributes has changed only for particular data sources.
        # Updating it resets only attributes for that particular data source.
        self.attributes_version_sources = {'Météo-France':2, 'NWS':3, 'DWD':2}
                    
        self.scanangles = {}
        
        self.attributes_descriptions_filename = os.path.join(opa(gv.userdir+'/Generated_files'),'attributes_descriptions.pkl')
        self.attributes_IDs_filename = os.path.join(opa(gv.userdir+'/Generated_files'),'attributes_IDs.pkl')
        self.attributes_variable_filename = os.path.join(opa(gv.userdir+'/Generated_files'),'attributes_variable.pkl')
        try:
            if self.gui.reset_volume_attributes: raise Exception

            with open(self.attributes_descriptions_filename,'rb') as f:
                self.attributes_descriptions = pickle.load(f)

            with open(self.attributes_IDs_filename,'rb') as f:
                self.attributes_IDs = pickle.load(f)
            
            with open(self.attributes_variable_filename,'rb') as f:
                self.attributes_variable = pickle.load(f)
        except Exception:
            self.attributes_descriptions, self.attributes_IDs, self.attributes_variable = {}, {}, {}
            
        self.gui.reset_volume_attributes = False
            
        if not 'version' in self.attributes_descriptions or self.attributes_descriptions['version'] != self.attributes_version:
            self.attributes_descriptions = {'version':self.attributes_version}
            self.attributes_IDs, self.attributes_variable = {}, {}
        ft.init_dict_entries_if_absent(self.attributes_descriptions, 'version_sources', dict)
            
        for j in gv.radars_all:
            key_extensions = ('_V','_Z') if j in gv.radars_with_datasets else ('',)
            source = gv.data_sources[j]
            reset_attrs_radar = self.attributes_descriptions['version_sources'].get(source, 0) != self.attributes_version_sources.get(source, 0)
            for i in key_extensions:
                if reset_attrs_radar or j+i not in self.attributes_descriptions: self.attributes_descriptions[j+i] = {}
                if reset_attrs_radar or j+i not in self.attributes_IDs: self.attributes_IDs[j+i] = {}
                if reset_attrs_radar or j+i not in self.attributes_variable: self.attributes_variable[j+i] = {}
        self.attributes_descriptions['version_sources'] = self.attributes_version_sources
                
        
    def get_scans_information(self, set_data, delta_time=0):
        # delta_time is used in self.check_need_scans_change
        """This function should deliver the attributes self.radial_res_all, self.radial_bins_all, scanangles_all, 
        nyquist_velocities_all_mps, low_nyquist_velocities_all_mps, high_nyquist_velocities_all_mps, scannumbers_all, scannumbers_forduplicates.
        
        In this application the scans are ordered from lower to higher scanangle, so a higher scannumber (starting at 1)
        corresponds to a higher scanangle. It is however possible that the scans in the imported volume are not ordered in 
        the same way (as is the case for the new KNMI volumes). Because of this possibility, scannumbers_all maps each scan
        (according to my way of ordening) to the corresponding scannumber(s) in the data file, which are put in a list.
        They are put in a list because it is possible that some scans are performed more than once during a complete volume scan.
        If a scan is performed more than once, then the list will include more than one element.
        scannumbers_forduplicates is then a list with for each scan the index of the scannumber that is taken from the list
        of scannumbers in the imported radar volume. Its elements are zero when the scans are performed only once during a 
        complete volume scan, and it therefore has the form [0/1,0,0,0,...] for the new KNMI radars. 
        
        It is possible that the attributes differ (slightly) per product, and for this reason they are determined for each product that is available
        for a particular data source (excluding products that are derived from other ones, i.e. only products in gv.i_p). The only product 
        that must always be available is 'z', whether or not it is really available for the source. This is because attribute['z'] is used in most of the
        functions that use info about particular volume attributes. If 'z' is not available, then the attributes for 'z' should be equal to those for products
        that are available.
        self.scannumbers_all etc. therefore have the form {'z':{1:...,2:...}} etc.
        
        To prevent that attributes are obtained more than once for a particular file, information about it is stored in a file. Each particular series
        of attributes gets assigned a unique ID, and this ID in stored in self.attributes_IDs[radar_dataset][dir_string]. 
        Here, dir_string is the directory string that is present in self.gui.radardata_dirs, for the
        index given in self.gui.radardata_dirs_indices. dir_string is used as key in addition to radar_dataset, because multiple directories can be
        selected for one radar_dataset, and if these directories contain different files for the same date and time, this would lead to problems.
        A disadvantage of the use of dir_string is that the attributes must be determined again when the data is moved to another directory.
        
        self.attributes_descriptions[radar_dataset] contains the series of attributes that corresponds to a particular ID. 
        When calling this function, it is first checked whether an attribute ID is available for a particular radar, date and time, and if so, the 
        corresponding attributes are obtained from self.attributes_descriptions[radar_dataset].
        If no attribute ID is available, then the attributes will be obtained, and the ID corresponding to it is stored in 
        self.attributes_IDs[radar_dataset][dir_string].
        For radars for which the data for one volume is stored in multiple files, it is however possible that the attributes get stored at a moment
        at which not all files were available yet, such that the saved attributes should be updated when more files come available. In order to
        handle this, the number of files that was present when saving the attributes is also stored, which is done in
        self.attributes_IDs[radar_dataset][dir_string]. The values stored in this dictionary are therefore lists of the form 
        [attributes_ID,saved_total_files_size].
        
        Sometimes attributes can be expected to vary from volume to volume, in which case it's not wise to add them to self.attributes_descriptions,
        since that would mean that also the other less variable attributes are stored again and again. In this case they are stored separately
        in self.attributes_variable, and the corresponding entries in self.attributes_descriptions are replaced by the string 'variable'.
        
        self.attributes_descriptions, self.attributes_IDs and self.attributes_variable are dumped into a pickle file when exiting the program,
        which occurs in the function closeEvent in nlr.py.
        """

        # (deep) copy of attributes is not needed, since any change of the current volume attributes will come either with a deep copy
        # of attributes stored in a file, or in case of newly determining attributes will start with initialization of empty dictionary.
        # Should only be updated when setting data
        self.attrs_before = {j:self.__dict__[j] for j in gv.volume_attributes_all}
        self.attrs_before['scannumbers_forduplicates'] = self.scannumbers_forduplicates

        # De vroegere "1-punts" melting-level-opvraag (radarlocatie zelf, voor de losse 0C/-20C-balk in
        # nlr.py) is hier verwijderd (22 juli, op Eriks verzoek): de balk is weg, en HCLASS gebruikt al
        # het nauwkeurigere gedeelde temperatuurrooster (zie ensure_melting_level_grid_current en
        # _calculate_hclass hieronder), dat per bin het dichtstbijzijnde roosterpunt gebruikt i.p.v. 1
        # vast punt voor het hele beeld.

        # Set update_scanpairs_indices=True here, instead of further down in the 2nd call of this function. Since there
        # the already updated volume attributes would be compared with the old scan choices. 
        self.scanpair_present_before = self.check_presence_large_range_large_nyquistvelocity_scanpair(update_scanpairs_indices=True) if\
                                       self.attrs_before['scannumbers_all'] else False
        # print('scanpbefore', self.scanpair_present_before)
                  
        attributes_available = self.restore_volume_attributes()            

        if not attributes_available:
            try:
                self.nyquist_velocities_all_mps = {}; self.low_nyquist_velocities_all_mps = {}; self.high_nyquist_velocities_all_mps = {}
                for attr in gv.volume_attributes_p: #Defined in nlr_globalvars.
                    self.__dict__[attr] = {}
                    for j in (gv.i_p[p] for p in gv.products_with_tilts):
                        self.__dict__[attr][j] = {}
                    
                self.scans_doublevolume = [] #scans_doublevolume is used for the new radars of the KNMI, where the volume can be divided into 2 parts.
                #It is also saved to self.attributes_descriptions[radar_dataset]. It gets defined in the functions get_scans_information in
                #nlr_importdata.py, if a double volume is present.
                self.variable_attributes = [] #variable_attributes is used for volume attributes that are expected to be different for (almost)
                # each volume. These attributes are then stored separately
                
                self.source_classes[self.data_source()].get_scans_information()
                # Check for 'z'. It can happen that for other products the length is not zero. But other parts of the code
                # use len(self.scannumbers_all['z']) for multiple operations, and a length of 0 leads to errors there. 
                # So don't continue when no z-scan is available.
                # Check for any subdict whose key starts with 'z'. This is done since for self.product_versions_in1file=True, 
                # keys will have the product version appended to them. See self.process_products_with_pvs_in_keys for more info.
                if all(len(j) == 0 for i,j in self.scannumbers_all.items() if i[0] == 'z'):
                    raise Exception

                # Make sure that all volume attribute dictionaries are sorted in order of increasing scans. In the past an issue has been noted
                # due to non-ascending scans in one of the import classes, and sorting volume attributes here guarantees that this won't occur again,
                # regardless of the implementation of get_scans_information in the import classes. The time this sorting takes is negligible.
                for attr in gv.volume_attributes_save:
                    if isinstance(self.__dict__[attr], dict):
                        if attr in gv.volume_attributes_p:
                            for p in self.__dict__[attr]:
                                # Sorting only integer keys is done because there might be string keys for derived products in self.scannumbers_all['z']
                                self.__dict__[attr][p] = dict(sorted((i,j) for i,j in self.__dict__[attr][p].items() if isinstance(i, (int, np.integer))))
                        else:
                            self.__dict__[attr] = dict(sorted(self.__dict__[attr].items()))
                
                self.store_volume_attributes()
                        
                self.determined_volume_attributes_radars[self.crd.radar] = {j:eval('self.'+j, {'self': self}) for j in gv.volume_attributes_save}
                
            except Exception as e:
                print('restore attributes')
                self.restore_previous_attributes()
                # This exception should be catched in the function self.pb.set_newdata
                raise Exception('get_scans_information', e)

        self.get_derived_volume_attributes()
                
        if set_data:            
            self.scanpair_present = self.check_presence_large_range_large_nyquistvelocity_scanpair(update_scanpairs_indices=False)
            
            if not self.pb.firstplot_performed or (self.crd.scans != self.pb.scans_before and not self.gui.setting_saved_choice):
                # Initialize self.selected_scanangles for all panels when not self.pb.firstplot_performed
                self.update_selected_scanangles(update_allpanels=not self.pb.firstplot_performed)
                if not self.pb.firstplot_performed:
                    # Not doing this leads to undesired behaviour when setting choice with select nearest height, due to updating
                    # self.time_last_forcedscanchange
                    self.selected_scanangles_before = self.selected_scanangles.copy()
            
            if self.selected_scanangles != self.selected_scanangles_before:
                #Is only updated in the case of purposeful scan changes, i.e. changes in scans caused by pressing UP/DOWN or a number key, or by
                #pressing F1-F12 for a saved panel choice, or when the scans change during a change in panels (see function
                #self.pb.change_panels).
                self.time_last_forcedscanchange = pytime.time()
                
            #Ensure that the selected scans are available for panellist. This might not be the case after a change of radar/dataset
            panellist = self.pb.panellist if self.pb.firstplot_performed else range(self.pb.max_panels)
            max_scan = len(self.scanangles_all['z'])
            if not 1 in self.scanangles_all['z']:
                print('set_max_scan', max_scan, self.scannumbers_all, self.scanangles_all)
                raise Exception('max scan should not be 0!!!!!!!!')
            for j in panellist:
                self.crd.scans[j] = min(self.crd.scans[j], max_scan)
                            
            if not self.attrs_before['scannumbers_all'] or\
            [len(j) for j in self.attrs_before['scannumbers_all']['z'].values()] != [len(j) for j in self.scannumbers_all['z'].values()]:
                #Update self.scannumbers_forduplicates and self.crd.scans if scannumbers_all has changed, because their current values might be invalid
                #for the new scannumbers_all.
                self.update_scannumbers_forduplicates()
            else:
                try:
                    if len(self.attrs_before['scannumbers_all']['z']) != len(self.scannumbers_forduplicates):
                        pass  # Print hier stilgelegd (22 juli, op Eriks verzoek): deze regel is een
                        # bestaande, onschuldige waarschuwing van Bram (geen crash, geen effect op de
                        # daadwerkelijke berekening - ook niet op MESH, die intussen gewoon zinnige
                        # waarden bleef geven). Sinds de komst van product 'o' (MESH) komt deze mismatch
                        # vaker voor (zie de eerdere KeyError-fix in nlr_derived_plain.py, zelfde
                        # onderliggende oorzaak: scannumbers_forduplicates krijgt niet altijd via de
                        # normale weg een entry voor 'o'), en vulde het scherm/logbestand te veel.
                except Exception:
                    pass
                                                 
            if self.crd.scan_selection_mode != 'scan' or self.gui.setting_saved_choice:
                self.check_need_scans_change(delta_time)
            else:
                for j in self.pb.panellist:
                    if self.scanpair_present and self.crd.scans[j] == 1:
                        self.crd.scans[j] = (1,2)[self.range_nyquistvelocity_scanpairs_indices[j]]
                    elif not self.scanpair_present and self.scanpair_present_before and self.crd.scans[j] in (1,2):
                        self.crd.scans[j] = 1
                                
            
                
    def get_derived_volume_attributes(self):            
        self.radial_range_all, self.scanangles_all_m = {}, {}
        for j in self.scannumbers_all:
            self.radial_range_all[j], self.scanangles_all_m[j] = {}, {}
            for i,a in self.scanangles_all[j].items():
                self.radial_range_all[j][i] = self.radial_bins_all[j][i]*self.radial_res_all[j][i]
                self.scanangles_all_m[j][i] = sum(self.scanangles_all[j][i].values())/len(a) if isinstance(a, dict) else a
        # Indicate for plain products whether they are affected by a double volume. This information is used in nlr_changedata.py and
        # nlr_derivedproducts.py
        for j in gv.plain_products:
            self.scannumbers_all['z'][j] = [0,1] if len(self.scans_doublevolume)>0 and\
                j in gv.plain_products_affected_by_double_volume else [0]   
                
    def scanangle(self, product, scan, duplicate):
        # Helper function for obtaining a scan's scanangle that can handle the presence of a different scanangle per duplicate
        i_p = gv.i_p[product]
        if not scan in self.scanangles_all[i_p] or not scan in self.scanangles_all_m[i_p]:
            print(i_p, self.scanangles_all, self.scanangles_all_m)
        return self.scanangles_all[i_p][scan].get(duplicate, self.scanangles_all_m[i_p][scan]) if\
               isinstance(self.scanangles_all[i_p][scan], dict) else self.scanangles_all[i_p][scan]
               
    def duplicate(self, product, scan):
        # For NEXRAD L2 combi_scan product version, products other than 'z' have a different number of duplicates (half of that for 'z').
        # And in order to not have to add fake/duplicate duplicates for these other products, the duplicate index gets halved here.
        # This function should therefore be called whenever a product-specific duplicate index is needed.
        duplicate = self.scannumbers_forduplicates[scan]
        pv = self.gui.radardata_product_versions[self.radar_dataset]
        if pv == 'combi_scan' and product != 'z': # For NEXRAD L2 combi_scan
            duplicate //= 2
        return duplicate


    def get_subdataset(self, pv=None, product=None): # pv is product version
        """Gives a string representation of the subdataset (combination of selected directory string and product version). 
        Contains only information that is considered important to distinguish different subdatasets (and thus contains e.g.
        no parts of directory string that are duplicate among all directory strings for this radar_dataset).
        """
        dir_string_list, current_dir_string, n_dirs = self.get_variables(self.crd.radar, self.crd.dataset)[1:]
            
        subdataset = ''
        if n_dirs > 1:
            subpaths = [j.split('/') for j in dir_string_list]
            min_n_subpaths = min(map(len, subpaths))
            subpaths_equal = np.array([len(set(j[i] for j in subpaths)) == 1 for i in range(min_n_subpaths)], dtype='bool')
            n = min_n_subpaths-1 if subpaths_equal.all() else np.where(~subpaths_equal)[0][0]
            _subpaths = [j[n:] for j in subpaths]
            # Remove subpaths that contain variables, unless that gives the same result for all dir_strings
            _subpaths_novar = ['-'.join(i for i in j if not '${' in i) for j in _subpaths]
            _subpaths_novar_unique = set(_subpaths_novar)
            index = dir_string_list.index(current_dir_string)
            if len(_subpaths_novar_unique) == n_dirs:
                subdataset = _subpaths_novar[index]
            else:
                subdataset = '-'.join(_subpaths[index])
            
        if pv is None:
            pv = self.gui.radardata_product_versions[self.radar_dataset]
        i_p = gv.i_p.get(product, None)
        if pv and self.product_versions_datetime and (i_p is None or i_p in self.products_version_dependent):
            _pv = (pv if pv in self.product_versions_datetime else self.product_versions_datetime[0])
            subdataset += '_'*bool(subdataset)+_pv
        return subdataset
    
    def process_products_with_pvs_in_keys(self, mode='store'): #'store' or 'restore'
        # If self.product_versions_in1file=True, then volume attributes for different product versions have been determined all at once,
        # with product keys for different versions distinguished by appending the product version to the key. This function processes 
        # these products_with_pvs_in_keys, depending on the mode. If mode='store', then remove any non-pv key for this product (that doesn't
        # contain the product version). If mode='restore', then restore the non-pv key based on the currently selected product version.
        if not self.product_versions_in1file:
            return
        
        products_with_pvs_in_keys = set([j[0] for j in self.scannumbers_all if any(j.endswith(i) for i in self.product_versions_datetime)])    
        pv = self.gui.radardata_product_versions[self.radar_dataset]
        pv = pv if pv in self.product_versions_datetime else self.product_versions_datetime[0]
        for p in products_with_pvs_in_keys:
            for j in gv.volume_attributes_p:
                if mode == 'store' and p in self.__dict__[j]:
                    del self.__dict__[j][p]
                else:
                    self.__dict__[j][p] = self.__dict__[j][p+' '+pv]

    def store_volume_attributes(self):
        for j in self.scannumbers_all['z'].copy():
            # Remove keys for plain products if they are present. They lead to errors with sorting keys, and are not needed
            # since they are added afterwards.
            if isinstance(j, str):
                del self.scannumbers_all['z'][j]
        
        self.process_products_with_pvs_in_keys('store')
        
        current_attrs = copy.deepcopy([eval('self.'+j, {'self': self}) for j in gv.volume_attributes_save])
        self.compress_volume_attributes(current_attrs)
        
        variable_attrs = []
        for i,j in enumerate(gv.volume_attributes_save):
            if j in self.variable_attributes:
                variable_attrs.append(current_attrs[i])
                current_attrs[i] = 'variable'
                        
        current_attrs_ID = None; ID = 0
        for ID in self.attributes_descriptions[self.radar_dataset]:
            attrs = self.attributes_descriptions[self.radar_dataset][ID]
            if attrs == current_attrs:
                #The series of attributes is the same as one of the series that is already stored, and the corresponding ID is used.
                current_attrs_ID = ID; break
        if not current_attrs_ID:
            #The series of attributes is new, and a new ID is used, that is 1 higher than the largest existing one.
            current_attrs_ID = ID+1
            self.attributes_descriptions[self.radar_dataset][current_attrs_ID] = current_attrs
                                
        # if self.product_versions_in1file=True, then volume attributes for all product versions have been determined at once, hence no need
        # to use a different subdataset for different product versions.
        subdataset = self.get_subdataset(pv='' if self.product_versions_in1file else None)
        #Insert the attributes ID in self.attributes_IDs
        ft.create_subdicts_if_absent(self.attributes_IDs[self.radar_dataset], [subdataset, self.crd.date])
        data_selected_startazimuth = self.gui.data_selected_startazimuth if self.crd.radar in gv.radars_with_adjustable_startazimuth else 0
        self.attributes_IDs[self.radar_dataset][subdataset][self.crd.date][self.crd.time] = [current_attrs_ID,self.total_files_size,data_selected_startazimuth]
        
        if variable_attrs:
            ft.create_subdicts_if_absent(self.attributes_variable[self.radar_dataset], [subdataset, self.crd.date])
            self.attributes_variable[self.radar_dataset][subdataset][self.crd.date][self.crd.time] = variable_attrs
            
        self.process_products_with_pvs_in_keys('restore')
                        
    def restore_volume_attributes(self):
        subdataset = self.get_subdataset(pv='' if self.product_versions_in1file else None)
        try:
            attrs_ID,saved_total_files_size,saved_data_selected_startazimuth = self.attributes_IDs[self.radar_dataset][subdataset][self.crd.date][self.crd.time]
            data_selected_startazimuth = self.gui.data_selected_startazimuth if self.crd.radar in gv.radars_with_adjustable_startazimuth else 0
            if saved_total_files_size!=self.total_files_size or saved_data_selected_startazimuth!=data_selected_startazimuth:
                return False

            attrs = copy.deepcopy(self.attributes_descriptions[self.radar_dataset][attrs_ID])
            i_variable_attrs = [i for i,attr in enumerate(attrs) if attr == 'variable']
            if i_variable_attrs:
                variable_attrs = copy.deepcopy(self.attributes_variable[self.radar_dataset][subdataset][self.crd.date][self.crd.time])
                for i,j in enumerate(i_variable_attrs):
                    attrs[j] = variable_attrs[i]
            self.decompress_volume_attributes(attrs)
                 
            scannumbers_all = attrs[gv.volume_attributes_save.index('scannumbers_all')]
            attributes_available = all(i in (j[0] for j in scannumbers_all) for i in gv.i_p.values())
            if attributes_available:
                for i,j in enumerate(gv.volume_attributes_save):
                    self.__dict__[j] = attrs[i]
                    
            self.process_products_with_pvs_in_keys('restore')
            
            return attributes_available
        except Exception:
            return False

    def _merge_repeated_values(self, dic):
        keys, vals = np.array(list(dic), dtype=object), np.array(list(map(str, dic.values())))
        _, unique_indices = np.unique(vals, return_index=True)
        unique = [i for i in keys if i in keys[unique_indices]] # Preserve original order in keys
        # Compare string-converted values (vals) instead of dic values, since when dic values have
        # different types it can be that the first comparison yields False, while the second yields True.
        # This has been observed at least once with values that are 'equal' except for their type (float32 vs float64).
        hits = {k:keys[vals[i] == vals] for i,k in enumerate(keys) if k in unique}
                    
        for i,j in hits.items():
            diff = [j[k]-j[k-1] if all(type(l) == int for l in (j[k], j[k-1])) else 0 for k in range(1, len(j))]
            key = str(j[0])
            for k in range(len(diff)):
                if diff[k] == 1:
                    key = (key[:key.index(str(j[k]))] if k and diff[k-1] == 1 else key+'-')+str(j[k+1])
                else:
                    key += ','+str(j[k+1])
            if key.isdigit():
                key = int(key)
            dic[key] = dic[i]
            if not key == i:
                for k in j:
                    del dic[k]
                    
    def _split_merged_values(self, dic):
        for i,j in dic.copy().items():
            if isinstance(i, str) and any(k in i for k in (',', '-')):
                split_keys = i.split(',')
                for k,l in enumerate(split_keys.copy()):
                    if '-' in l:
                        n1, n2 = l.split('-')
                        split_keys[k] = list(range(int(n1), int(n2)+1))
                    else:
                        split_keys[k] = [int(l) if l.isdigit() else l]
                for k in sum(split_keys, []):
                    dic[k] = j
                del dic[i]
                
    def compress_volume_attributes(self, attributes):
        for attr in attributes:
            if not isinstance(attr, dict):
                continue
            products = list(attr)
            if not products or not isinstance(products[0], str):
                # Skip the attributes that are not product-dependent, such as the Nyquist velocities
                if products:
                    self._merge_repeated_values(attr)
                continue
            
            for p1 in products:
                hits = [p for p in products[:products.index(p1)] if attr[p1] == attr[p]]
                if hits:
                    attr[p1] = hits[0]
            products = [p for p in products if isinstance(attr[p], dict)]
            
            for p1 in products:
                for i in attr[p1]:
                    hits = [p for p in products[:products.index(p1)] if i in attr[p] and attr[p1][i] == attr[p][i]]
                    if hits:
                        attr[p1][i] = hits[0]
                        
            for p1 in products:
                self._merge_repeated_values(attr[p1])
            self._merge_repeated_values(attr)
            
    def decompress_volume_attributes(self, attributes):
        for i, attr in enumerate(attributes):
            if not isinstance(attr, dict):
                continue
            self._split_merged_values(attr)
            products = list(attr)
            if not isinstance(products[0], str):
                attributes[i] = self._sort_attributes_by_key(attr)
                continue
                    
            for p1 in products:
                if isinstance(attr[p1], str):
                    p2 = attr[p1]
                    attr[p1] = attr[p2].copy()
                else:
                    self._split_merged_values(attr[p1])
                    for j in attr[p1]:
                        if attr[p1][j] in products:
                            p2 = attr[p1][j]
                            attr[p1][j] = attr[p2][j]
                attr[p1] = self._sort_attributes_by_key(attr[p1])
                
    def _sort_attributes_by_key(self, attr):
        sorted_keys = sorted(attr)
        if not sorted_keys == list(attr):
            values = [attr[i] for i in sorted_keys]
            attr = dict(zip(sorted_keys, values))
        return attr
        

    def generate_dataspecs_string(self,product,productunfiltered,polarization,apply_dealiasing,panel,proj=None):
        # proj only needs to be specified for plain products
        #Include total_files_size, because a change in the number of scans in a volume means that a particular scan can get
        #different volume attributes compared to what was previously the case.
        radar_dataset = self.get_radar_dataset()
        # Exclude product version (pv) for self.product_versions_in1file=True. This is aimed at the NEXRAD L2 combi_scan pv,
        # where including the pv would cause the program to re-import when requesting the individual z_scan and v_scan pvs.
        # This should not cause issues with distinguishing between pvs, given differences in scannumbers_all between pvs.
        subdataset = self.get_subdataset('' if self.product_versions_in1file else None, product)
        dataspecs_string = radar_dataset+subdataset
                
        data_selected_startazimuth = self.gui.data_selected_startazimuth if self.crd.radar in gv.radars_with_adjustable_startazimuth else 0
        # dataspecs_string += str(self.total_files_size)+'_'+self.crd.date+self.crd.time+'_'+product+'_'+str(data_selected_startazimuth)
        dataspecs_string += '_'+self.crd.date+self.crd.time+'_'+product+'_'+str(data_selected_startazimuth)
        
        # It is assumed that any change in product/scan content is reflected either in a change in scannumbers_all or in a change in self.total_files_size.
        # For data sources in self.stored_data_sources_with_scannumbers_as_content_marker it is assumed to be the former.
        scannumbers_all = self.scannumbers_all[gv.i_p[product]]
        scan = self.crd.scans[panel]
        duplicate = self.duplicate(product, scan)
        scannumbers = scannumbers_all if product in gv.plain_products else scannumbers_all[scan][duplicate]
        if product in gv.plain_products:
            if product in gv.plain_products_with_parameters:
                dataspecs_string+= '_'+str(self.gui.PP_parameter_values[product][self.gui.PP_parameters_panels[panel]])
            dataspecs_string+= '_'+str(productunfiltered)+'_'+polarization
            content_marker = scannumbers if self.data_source() in self.stored_data_sources_with_scannumbers_as_content_marker else self.total_files_size
            dataspecs_string+= '_'+proj
            if proj == 'car':
                dataspecs_string+= '_'+str(self.gui.stormmotion)+'_'+str(self.gui.cartesian_product_res)+'_'+str(self.gui.cartesian_product_maxrange)
        else: 
            if gv.i_p[product] == 'v':
                dataspecs_string += '_'+str(apply_dealiasing) + '_' + self.gui.dealiasing_setting + '_' + str(self.gui.dealiasing_dualprf_n_it)
            dataspecs_string+= '_'+str(productunfiltered)+'_'+polarization
            content_marker = str(scannumbers) if self.data_source() in self.stored_data_sources_with_scannumbers_as_content_marker else\
                             str(self.total_files_size)+str(scan)+str(duplicate)
            dataspecs_string+= '_'+content_marker
        return dataspecs_string
    
    def get_dataspecs_string_panel(self, j, product=None, return_params=False): #j is the panel
        product = self.crd.products[j] if product is None else product
        productunfiltered = self.crd.using_unfilteredproduct.get(j, False)
        polarization = {True:'V', False:'H'}[self.crd.using_verticalpolarization.get(j, False)]
        apply_dealiasing = self.crd.apply_dealiasing[j]
        proj = self.dp.meta_PP[product]['proj'] if product in gv.plain_products else None
        dataspecs_string = self.generate_dataspecs_string(product,productunfiltered,polarization,apply_dealiasing,j,proj)
        if return_params:
            return dataspecs_string, productunfiltered, polarization, apply_dealiasing, proj
        else:
            return dataspecs_string
        
    def store_data_in_memory(self, j): #j is the panel
        product = self.crd.products[j]
        if product in gv.products_with_tilts_derived_nosave:
            # In this case import product is saved instead of actual product, since the latter can be cheaply calculated from import product.
            product = gv.i_p[product]
        
        dataspecs_string, productunfiltered, polarization, apply_dealiasing, proj = self.get_dataspecs_string_panel(j, product, True)
        # if not data changed we can still use self.crd.using_verticalpolarization[j] etc due to dataspecs_string_requested below
        if self.data_changed[j]:  
            self.stored_data[dataspecs_string] = {'last_use_time':pytime.time(),'data':self.data[j].copy(),'data_azimuth_offset':self.data_azimuth_offset[j],'data_radius_offset':self.data_radius_offset[j],'scantime':self.scantimes[j],'using_unfilteredproduct':self.crd.using_unfilteredproduct.get(j, False),'using_verticalpolarization':self.crd.using_verticalpolarization.get(j, False)}
        else:
            self.stored_data[dataspecs_string] = {'last_use_time':pytime.time(),'data':np.zeros((1,1))}
            
        if product in gv.plain_products:
            self.stored_data[dataspecs_string]['meta_PP'] = self.dp.meta_PP[product].copy()
                
        if productunfiltered != self.crd.productunfiltered[j] or polarization != self.crd.polarization[j]:
            # In this case the new data array is both stored for the actual and the requested combination of productunfiltered and 
            # polarization, but for the requested combination only the key for the actual combination is given as value.
            # This key can then be used to obtain the desired dictionary with data
            dataspecs_string_requested = self.generate_dataspecs_string(product,self.crd.productunfiltered[j],self.crd.polarization[j],apply_dealiasing,j,proj)
            self.stored_data[dataspecs_string_requested] = dataspecs_string    
            
        stored_data_size = np.sum([float(sys.getsizeof(j['data'])) for j in self.stored_data.values() if not isinstance(j, str)])
        #Convert to float, since the number might exceed the maximum value for 32-bit ints.
        while stored_data_size>1e9*self.gui.max_radardata_in_memory_GBs:
            #Remove the dataset with the most outdated last_use_time
            index = np.argmax([pytime.time()-j['last_use_time'] for j in self.stored_data.values() if not isinstance(j, str)])
            most_outdated_last_use_time_key = list(self.stored_data)[index]
            # Also remove possible other keys that map onto the key that will be removed
            keys_remove = [most_outdated_last_use_time_key]+\
                [j for j in self.stored_data if type(self.stored_data[j]) == str and self.stored_data[j] == most_outdated_last_use_time_key]
            for key in keys_remove:
                del self.stored_data[key]
            stored_data_size = np.sum([float(sys.getsizeof(j['data'])) for j in self.stored_data.values() if not isinstance(j, str)])
        
    def check_presence_data_in_memory(self,product,productunfiltered,polarization,apply_dealiasing,panel):
        if self.gui.max_radardata_in_memory_GBs <= 0:
            return
        if product in ('g', 'j', 'o', 'b', 'uh', 'si', 'zc'):
            # UITBREIDING (27 juli 2026): de ZDR-kolomdiepte ('zc') hangt net als SHI/MESH/POSH/POH af
            # van het gedeelde temperatuurrooster (ensure_melting_level_grid_current) - zelfde reden
            # als 'si' hierboven, dus hier meteen goed toegevoegd i.p.v. achteraf als bugfix.
            #
            # UITBREIDING (27 juli 2026): SHI ('si') had PRECIES hetzelfde probleem als MESH/POSH/POH
            # hieronder beschreven - het was bij de toevoeging van SHI als los product abusievelijk NIET
            # aan deze uitsluiting toegevoegd. Gevolg: een eerder berekend SHI-resultaat bleef in
            # self.stored_data hangen, ongeacht latere wijzigingen aan de temperatuurbron. Een latere
            # kleurtabel-aanpassing (cmaps_maxrange) triggerde via de cmap_lastmodification_time-check
            # verderop in deze functie WEL een cache-miss en dus een herberekening - waarbij een
            # inmiddels gewijzigde temperatuurbron (bv. een nieuwere Open-Meteo-modelrun) een ander
            # resultaat kon geven, ook al was de rekenformule zelf niet aangepast. Exact het patroon dat
            # Erik meldde: zowel de SHI-waarde als het gekleurde oppervlak veranderden na een
            # kleurtabel-wijziging, terwijl geen van beide daar rechtstreeks door zou moeten veranderen.
            #
            # UITBREIDING (25 juli): POSH ('b') en POH ('uh') hebben PRECIES hetzelfde probleem als
            # MESH ('o') hierboven beschreven - ook zij hangen af van het gedeelde temperatuurrooster
            # (ensure_melting_level_grid_current), dus een eerder berekend resultaat voor een tijdstip
            # kan verouderen zodra de temperatuurbron nadien verandert (automatisch herstel, OF de
            # handmatige terugvaloptie/ALT+W). Zonder deze uitsluiting bleef dat oude resultaat voor
            # altijd uit self.stored_data komen, ongeacht latere wijzigingen aan de temperatuurbron -
            # exact wat Erik meldde voor POH (reageerde nergens op, ook niet op een foutieve
            # handmatige stationskeuze die tot 'geen data' had moeten leiden).
            #
            # Oorspronkelijke toelichting bij MESH ('o'), 22 juli: een tijdstip dat AL eerder is bekeken
            # (en toen "geen data" gaf, omdat de temperatuur toen nog niet lukte) bleef voor
            # altijd dat oude, foute resultaat tonen uit self.stored_data, ook nadat de
            # onderliggende temperatuurbron later wel werkte. Vandaar 'o' hier ook toegevoegd.
            # FIX (22 juli): PolRGB en HCLASS slaan hun eigen component-arrays op in
            # self.polrgb_raw_data/self.hclass_raw_data (zie _calculate_polrgb/_calculate_hclass), NIET in
            # self.stored_data hieronder. Als deze functie bij een cache-hit self.import_data[panel]=False
            # zet, wordt _calculate_hclass/_calculate_polrgb overgeslagen: het getoonde beeld (self.data[panel])
            # wordt dan wel correct uit de cache hersteld, maar polrgb_raw_data/hclass_raw_data NIET - die
            # blijven op hun oude waarde staan (van de laatste keer dat er wel herberekend werd, bv. een
            # ander, tussentijds bekeken tijdstip). Gevolg: het scherm toont de juiste (gecachete) kleur,
            # maar de cursor-tooltip leest een verouderde klasse/waarde. Erik reproduceerde dit exact: na
            # Herwijnen 16 juli (correct: Hail) -> 27 juni -> terug naar 16 juli liet de tooltip weer "Rain"
            # zien op een plek die nog steeds zichtbaar rood (Hail) was. Door hier vroegtijdig te stoppen
            # blijft self.import_data[panel] op zijn default (True), en lopen 'g'/'j' altijd via de normale
            # _calculate_hclass/_calculate_polrgb-weg, zodat hclass_raw_data/polrgb_raw_data nooit kunnen
            # verouderen t.o.v. wat er staat getekend.
            return
        derived_nosave = product in gv.products_with_tilts_derived_nosave
        if derived_nosave:
            # In this case import product is saved instead of actual product, since the latter can be cheaply calculated from import product.
            product = gv.i_p[product]
        
        proj = None
        if product in gv.plain_products:
            proj = 'car' if self.gui.stormmotion[1] != 0. and product in gv.plain_products_correct_for_SM else 'pol'
        try:
            dataspecs_string = self.generate_dataspecs_string(product,productunfiltered,polarization,apply_dealiasing,panel,proj)
        except Exception:
            # Can happen when requested scan or duplicate is unavailable
            return
        
        if dataspecs_string in self.stored_data:
            data_dict = self.stored_data[dataspecs_string]
            if isinstance(data_dict, str): # In this case data_dict actually is a key that maps onto another data_dict
                data_dict = self.stored_data[data_dict]
            last_use_time = data_dict['last_use_time']
            if last_use_time<self.pb.cmap_lastmodification_time[product] or last_use_time<self.gui.time_last_removal_volumeattributes:
                return False #In this case the color map has been modified in the mean time, implying that
                #self.pb.mask_values_int[product] could have been changed. If this is the case then the number of masked elements
                #will likely change, which requires an update of the data.
                
            # An empty array has been saved to memory when attempts to import data were unsuccessful. In this case don't update the data
            # array and attributes, but also don't re-import data, which requires that self.import_data[panel] is still set to False.
            if data_dict['data'].size > 1:
                # FIX (6 juli 2026): voorheen werd hier alleen een kopie gemaakt als derived_nosave True was.
                # In alle andere gevallen wees self.data[panel] naar HETZELFDE array-object als de cache
                # (data_dict['data']), zonder kopie. Verderop wordt op self.data[panel] echter in-place
                # geklemd/aangepast (zie convert_dtype_float_to_uint), wat dus de GECACHTE data zelf blijvend
                # kon veranderen -- een aannemelijke oorzaak van willekeurige, moeilijk te reproduceren
                # verkeerde velocity-waarden bij Herwijnen, vooral bij snel wisselen tussen panelen/producten
                # (waarbij dezelfde cache-entry vaker kort na elkaar hergebruikt wordt). Nu altijd een kopie.
                self.data[panel] = data_dict['data'].copy()
                self.data_azimuth_offset[panel] = data_dict['data_azimuth_offset']
                self.data_radius_offset[panel] = data_dict['data_radius_offset']
                self.scantimes[panel] = data_dict['scantime']
                self.crd.using_unfilteredproduct[panel] = data_dict['using_unfilteredproduct']
                self.crd.using_verticalpolarization[panel] = data_dict['using_verticalpolarization']
                if product in gv.plain_products:
                    self.dp.meta_PP[product] = data_dict['meta_PP'].copy()
                
                self.data_changed[panel] = True
            self.import_data[panel] = False
            data_dict['last_use_time'] = pytime.time()
            
            
    def convert_dtype_float_to_uint(self,data,product,inverse=False):
        """Convert the data to unsigned integers.
        For an explanation of the process of converting floating point data values to unsigned integers, see nlr_globalvars.py.
        """
        n_bits = gv.products_data_nbits[product]
        p_lim = gv.products_maxrange[product]
        pm_lim = gv.products_maxrange_masked[product]
            
        if not inverse:
            # Defensieve kopie (6 juli 2026): 'data' hier zou in principe altijd al een eigen kopie moeten zijn
            # (zie de fix hierboven bij het ophalen uit self.stored_data), maar deze functie klemt/wijzigt data
            # hieronder IN-PLACE. Als 'data' via een ander pad toch nog een gedeelde referentie blijkt te zijn
            # (bv. rechtstreeks vanuit de cache, of gedeeld tussen panelen), voorkomt deze kopie dat die
            # onbedoeld blijvend wordt aangepast. Extra vangnet, geen vervanging voor de fix hierboven.
            data = data.copy()
            data_notmasked = (data!= self.pb.mask_values[product])
            #These 2 lines are necessary, to assure that no errors arise when p_lim does not capture the whole range of 
            #product values.
            data[(data<p_lim[0]) & data_notmasked] = p_lim[0]
            data[data>p_lim[1]] = p_lim[1]
            new_data = np.full(data.shape, self.pb.mask_values_int[product], f'uint{n_bits}')
            new_data[data_notmasked] = ft.convert_float_to_uint(data[data_notmasked],n_bits,pm_lim)
        else:
            data_notmasked = data != self.pb.mask_values_int[product]
            new_data = np.full(data.shape, self.pb.mask_values[product], 'float32')
            new_data[data_notmasked] = ft.convert_uint_to_float(data[data_notmasked],n_bits,pm_lim)
        return new_data
    
    def apply_binfilling(self,j): #j is the panel
        """If reflectivity is shown, then fill empty radar bins if at least 2 neighbouring bins are non-empty, in order to reduce ugly interpolation effects.
        This is only performed for bins with a reflectivity >= 20 dBZ, to prevent enlarging of areas with low reflectivity.
        """
        initial_dtype = self.data[j].dtype
        if initial_dtype != 'float32':
            self.data[j] = self.data[j].astype('float32')
        for i in range(3):
            data_mask = self.data[j] == 0. if initial_dtype != 'float32' else self.data[j] == self.pb.mask_values[self.crd.products[j]]
            neighbours = ft.get_window_sum((data_mask == False).astype('float32'), [0,1,0])
            bins_to_fill = (data_mask) & (neighbours >= 2)
            
            if initial_dtype == 'float32':
                self.data[j][data_mask] = 0
            self.data[j][bins_to_fill] = ft.get_window_sum(self.data[j], [0,1,0])[bins_to_fill] / neighbours[bins_to_fill]
            if initial_dtype == 'float32':
                self.data[j][(data_mask) & (bins_to_fill == False)] = self.pb.mask_values[self.crd.products[j]]
                
            unfill = np.zeros(self.data[j].shape, dtype='bool')
            if initial_dtype == 'float32':
                unfill[bins_to_fill] = self.data[j][bins_to_fill] < 20
            else:
                unfill[bins_to_fill] = self.data[j][bins_to_fill] < ft.convert_float_to_uint(20, gv.products_data_nbits[self.crd.products[j]], gv.products_maxrange_masked[self.crd.products[j]])
            self.data[j][unfill] = 0 if initial_dtype != 'float32' else self.pb.mask_values[self.crd.products[j]]
        if initial_dtype != 'float32':
            self.data[j] = self.data[j].astype(initial_dtype)

    def get_data(self, panellist, delta_time, change_radar, change_dataset, set_data):
        # from cProfile import Profile
        # profiler = Profile()
        # profiler.enable() 
        """This function checks for each panel if the selected product is available, and if so, then it imports the data.
        When the product is in plain_products, then the import of data takes place in get_data_multiple_scans.
        
        Returns self.data_changed, which specifies for each panel whether the data has been changed.
        
        if set_data=False, then only volume attributes are determined and returned. The volume attributes that correspond to the radar volume that is 
        currently displayed are restored after retrieving the desired volume attributes.
        
        If set_data=False, then this function returns retrieved_attrs. If set_data=True, then it returns self.data_changed!
        """        
        self.changing_radar = change_radar; self.changing_dataset = change_dataset
        
        self.select_files_datetime() #This updates self.files_datetime
        # Calculate total_files_size at the start of get_data. It's important to not wait with this until it's actually needed (e.g. in 
        # self.generate_dataspecs_string), since it's possible that the total size of files increases during the actions in get_data (like
        # when downloading current data), implying that a later determined files size might be larger than that used in earlier actions. 
        # And since total_files_size is used to decide whether updates are needed, it could then happen that it is incorrectly decided that
        # no update is needed. By calculating it at the start here it can happen that total_files_size is smaller than what's used in later
        # actions, but this only has as disadvantage that an unnecessary update is performed. Which is better than e.g. not updating 
        # half-finished scans.
        # self.total_files_size should also be used elsewhere where information about total files size is needed.
        self.total_files_size = self.get_total_volume_files_size()
        
        self.radar_dataset = self.get_radar_dataset()
        
        if set_data:
            self.radar_dataset_before = self.crd.before_variables['radar']
            
            if self.crd.before_variables['radar'] in gv.radars_with_datasets:
                self.radar_dataset_before+= ' '+self.crd.before_variables['dataset'] 
        
        self.get_scans_information(set_data, delta_time)
        
        if not set_data:
            retrieved_attrs = {j:self.__dict__[j] for j in gv.volume_attributes_all}
            self.restore_previous_attributes()
            return retrieved_attrs, self.total_files_size
        
        #Update self.scannumbers_forduplicates_radars[self.crd.radar], which is used in the function self.update_scannumbers_forduplicates
        self.scannumbers_forduplicates_radars[self.crd.radar] = self.scannumbers_forduplicates.copy()
            
        """It is first checked whether data is present in the memory for the selected product, productunfiltered, polarization and apply_dealiasing. 
        If not, then the function self.source_classes[self.data_source()].get_data is called.
        """
        self.data_changed = {j:False for j in panellist}
        self.import_data = {j:True for j in panellist}
        # from cProfile import Profile
        # profiler = Profile()
        # profiler.enable() 

        for j in panellist:
            #By setting it here to zero, this variable needs only be updated in nlr_importdata.py when it deviates from zero
            self.data_azimuth_offset[j] = 0.
            self.data_radius_offset[j] = 0.
            
            self.check_presence_data_in_memory(self.crd.products[j],self.crd.productunfiltered[j],self.crd.polarization[j],self.crd.apply_dealiasing[j],j)
            
            i_p, scan = gv.i_p[self.crd.products[j]], self.crd.scans[j]
            if not scan in self.scannumbers_forduplicates:
                print(self.scannumbers_forduplicates, self.scannumbers_all)
            duplicate = self.duplicate(i_p, scan)
            if duplicate >= len(self.scannumbers_all[i_p].get(scan, [])):
                # In this case the requested duplicate scan is not available for this product
                self.import_data[j] = False

        self.update_volume_attributes = False
        
        panellist_import = [j for j in panellist if self.import_data[j]]
        # dont_store_in_memory is used for products that can for some reason not yet be delivered in there expected final form.
        # It is e.g. used for dealiased velocity to which Unet VDA could not yet be applied, because it hadn't finished loading yet.
        self.dont_store_in_memory = {j:False for j in panellist_import}
        if panellist_import:
            panellist_plain = [j for j in panellist_import if self.crd.products[j] in gv.plain_products]
            panellist_notplain = [j for j in panellist_import if not self.crd.products[j] in gv.plain_products]
            
            for j in panellist_import:
                self.crd.using_unfilteredproduct[j] = False
                self.crd.using_verticalpolarization[j] = False
            
            # Order panellist_import in such a way that panels that import the same scan are processed consecutively. 
            # This allows some of the import code in nlr_importdata.py to efficiently obtain data for multiple products
            # self.scannumbers_panels is used in nlr_importdata.py, at least for NEXRAD L2 data
            self.scannumbers_panels = {}
            for j in panellist_notplain:
                i_p, scan = gv.i_p[self.crd.products[j]], self.crd.scans[j]
                if scan in self.scannumbers_all[i_p]:
                    # Convert to string, as self.scannumbers_all[i_p][scan] might contain entities that can't be sorted (e.g. None)
                    self.scannumbers_panels[j] = str(self.scannumbers_all[i_p][scan])
            panellist_notplain = sorted(self.scannumbers_panels, key=self.scannumbers_panels.get)
            for j in panellist_notplain:
                ft.create_subdicts_if_absent(self.scanangles, [self.crd.radar, self.crd.date])
                try:
                    # Mono PRF dealiasing normally is performed below, except when a special treatment is required, e.g. when more than one 
                    # Nyquist velocity is used for the scan. In that case the function self.perform_mono_prf_dealiasing is called in nlr_importdata.py,
                    # after which self.mono_prf_dealiasing_performed is set to True in this function.
                    self.mono_prf_dealiasing_performed = False
                    
                    if j in self.scantimes:
                        before = self.data[j], self.scantimes[j], self.data_azimuth_offset[j], self.data_radius_offset[j]
                    if self.crd.products[j] not in ('g', 'j'):
                        self.source_classes[self.data_source()].get_data(j)
                    else:
                        # The skipped call above would normally also set these two flags (see e.g.
                        # nlr_datasourcespecific.py's get_data methods) -- they're read unconditionally later
                        # (e.g. in store_data_in_memory), so they must still be set here even though the rest
                        # of that call isn't needed for 'g'/'f' (their component products are fetched
                        # separately, inside _calculate_polrgb/_calculate_hclass). Neither PolRGB nor HCLASS
                        # has a concept of using an unfiltered product or vertical polarization of its own, so
                        # False is the correct default in both cases.
                        self.crd.using_unfilteredproduct[j] = False
                        self.crd.using_verticalpolarization[j] = False
                    
                    if self.crd.requesting_latest_data and not self.changing_radar and 'before' in locals():
                        # When plotting recent data, check whether data is available for the azimuth of the panel's center. If not,
                        # go back to previous data. This check is useful for real-time data streams that provide partial scans, e.g. for NWS
                        panel_center_xy = self.pb.screencoord_to_xy(self.pb.panel_centers[j])
                        azimuth = ft.azimuthal_angle(panel_center_xy, deg=True)
                        row = int(azimuth//1)
                        if j in self.pb.data_attr['scantime'] and self.scantimes[j] != self.pb.data_attr['scantime'][j] and\
                        np.all(self.data[j][row] == self.pb.mask_values[self.crd.products[j]]):
                            print('back to before', j)
                            self.data[j], self.scantimes[j], self.data_azimuth_offset[j], self.data_radius_offset[j] = before
                            continue
                    
                    v_nyquist = self.nyquist_velocities_all_mps.get(self.crd.scans[j], None)
                    if gv.i_p[self.crd.products[j]] == 'v' and self.crd.apply_dealiasing[j] and 'Unet VDA' in self.gui.dealiasing_setting and\
                    self.data[j].dtype == 'float32' and v_nyquist not in (None, 999.) and v_nyquist <= self.gui.dealiasing_max_nyquist_vel and\
                    not self.mono_prf_dealiasing_performed:
                        # A Nyquist velocity of 999. indicates that it could not be determined, while it is at least high enough to include
                        # the scan in operations that require a sufficiently high Nyquist velocity.
                        self.data[j] = self.perform_mono_prf_dealiasing(j, self.data[j])
                except Exception as e:
                    print(e, 'get_data, panel '+str(j))
                    traceback.print_exception(type(e), e, e.__traceback__)
                    continue

                self.data_changed[j] = True

            if panellist_plain:
                # To do: think of error handling, and data_changed
                self.dp.calculate_plain_products(panellist_plain)
                for j in panellist_plain:
                    self.data_changed[j] = True
           
        for j in panellist_import:    
            product = self.crd.products[j]
            if product in gv.products_possibly_exclude_lowest_values:
                # Hide product values that are below the minimum value that the user wants to view
                if self.data[j].dtype.name.startswith('float'):
                    min_value, mask_value = self.pb.data_values_colors[product][0], self.pb.mask_values[product]
                else:
                    min_value, mask_value = self.pb.data_values_colors_int[product][0], self.pb.mask_values_int[product]
                self.data[j][self.data[j] < min_value] = mask_value
            if product == 'g':
                # Polarimetric RGB composite: R=Z, G=CC, B=ZDR (see DataSource_General._calculate_polrgb)
                self.data[j] = self._calculate_polrgb(j)
            elif product == 'j':
                # Hydrometeorenclassificatie (HCLASS), C-band, Z/ZDR/KDP/CC + temperatuur
                # (zie DataSource_General._calculate_hclass en nlr_hclass.py)
                self.data[j] = self._calculate_hclass(j)
            elif self.data[j].dtype.name.startswith('float'):
                self.data[j] = self.convert_dtype_float_to_uint(self.data[j], product)
            
            # When self.data_changed[j]=False an empty array (created in self.store_data_in_memory) will be saved to memory, to indicate that no
            # data has been obtained. In the past nothing was saved to memory at all, but this had as disadvantage that new requests of the same
            # data would lead to renewed attempts to import, which were a waste of time.
            if self.gui.max_radardata_in_memory_GBs > 0 and not self.dont_store_in_memory[j] and product not in ('g', 'j', 'o'):
                # 'g' (PolRGB), 'j' (HCLASS) en 'o' (MESH) worden nooit meer uit dit geheugencache teruggelezen (zie de
                # uitleg bij check_presence_data_in_memory hierboven), dus opslaan zou hier alleen geheugen
                # en tijd verspillen.
                self.store_data_in_memory(j)
                
        for j in (i for i in panellist if self.data_changed[i]):
            product = self.crd.products[j]            
            if product in gv.products_with_tilts_derived:
                self.calculate_derived_with_tilts(j)
            
            if self.pb.use_interpolation and product in gv.products_with_interpolation_and_binfilling:
                self.apply_binfilling(j)
                    
        if self.update_volume_attributes:
            # Some volume attributes have apparently been updated, and they also have to be updated in the attribute dictionaries.
            self.store_volume_attributes()
            self.get_derived_volume_attributes()
                  
        self.selected_scanangles_before = self.selected_scanangles.copy()
        # profiler.disable()
        # import pstats
        # stats = pstats.Stats(profiler).sort_stats('cumtime')
        # stats.print_stats(20)  
        return self.data_changed, self.total_files_size
    
    # ESSL ECSS2025-poster RGB/alpha lookup tables (Van 't Veen, Groenemeijer & Pucik, "Detecting severe storms
    # using an RGB composite combining polarimetric radar parameters", 12th ECSS, Utrecht, nov 2025) -- the
    # POSTER'S OWN published values. Tunable via Settings -> PolRGB (self.gui.polrgb_params, keys ESSL_Z_MIN/
    # ESSL_Z_MAX/ESSL_CC_MIN/ESSL_CC_MAX/ESSL_ZDR_MIN/ESSL_ZDR_MAX/ESSL_ALPHA_Z/ESSL_ALPHA_V), same "live
    # GUI values, fallback dict only guards a missing key" pattern as Z_MIN/CC_MIN/etc. above -- see the p()
    # helper in _calculate_polrgb/get_volume_grid_polrgb below, which resolves these before calling
    # _essl_polrgb_channels. "Reset to defaults" (nlr.py, polrgb_params_default) resets these back to
    # exactly these poster values, per Eriks explicit requirement (16 september 2026) that the default when
    # choosing 'default' stays the ESSL poster's own numbers, not NLradar's passthrough numbers.
    ESSL_Z_RANGE_POSTER_DEFAULT = (30.0, 60.0)      # Red: dBZ 30->60 maps linearly to 0->1
    ESSL_CC_RANGE_POSTER_DEFAULT = (70.0, 100.0)    # Green: CC% 100->70 maps linearly to 0->1 (inverted)
    ESSL_ZDR_RANGE_POSTER_DEFAULT = (0.0, 4.0)      # Blue: ZDR dB 0->4 maps linearly to 0->1
    # Alpha (visibility) vs Z: 11-point piecewise-linear curve, straight from the poster's table. x must be
    # strictly increasing for np.interp; values outside [-10, 40] clip to the nearest end (0.05 resp. 1.0).
    ESSL_ALPHA_Z_POSTER_DEFAULT = [-10.0, 0.0, 10.0, 15.0, 20.0, 24.0, 28.0, 31.0, 34.0, 37.0, 40.0]
    ESSL_ALPHA_V_POSTER_DEFAULT = [0.05, 0.12, 0.22, 0.29, 0.39, 0.48, 0.58, 0.67, 0.77, 0.88, 1.00]

    @staticmethod
    def _essl_polrgb_channels(data_z_filled, data_cc_filled, data_zdr_filled, nodata_z,
                               z_min, z_max, cc_min, cc_max, zdr_min, zdr_max, alpha_z, alpha_v):
        """R/G/B/alpha volgens de (instelbare, standaard=poster-)ESSL-tabel, i.p.v. de doorlopende passthrough
        van _calculate_polrgb. De 6 grenswaarden en de 11-punts alpha-curve komen van de aanroeper (al
        opgelost via polrgb_params/fallback_defaults, zie hierboven) -- deze functie kent zelf geen vaste
        getallen meer, puur de formulevorm. Zie ESSL_MODE in _calculate_polrgb's docstring voor de achtergrond
        en waarom hier GEEN gebruik wordt gemaakt van de eigen (a*(a*x0+1-a))-compositieformule van het
        artikel -- de echte GPU-alphablending van NLradar vervangt die formule al, en beter (geen zwarte-
        schijf-op-de-kaart-effect)."""
        def norm(arr, vmin, vmax):
            return np.clip((arr-vmin)/(vmax-vmin), 0.0, 1.0)
        r = norm(data_z_filled, z_min, z_max)
        g = 1.0 - norm(data_cc_filled, cc_min, cc_max)
        b = norm(data_zdr_filled, zdr_min, zdr_max)
        alpha = np.interp(data_z_filled, alpha_z, alpha_v)
        alpha = np.where(nodata_z, 0.0, alpha)
        return r, g, b, alpha

    def _calculate_polrgb(self, j):
        """Calculate a polarimetric RGB composite for panel j, combining reflectivity (Z), correlation
        coefficient (CC/RhoHV) and differential reflectivity (ZDR) into one (azimuth, range, 4) uint8 RGBA
        image, in the style of e.g. ARPA Lombardia's polarimetric composites.

        R = Z   (dBZ)
        G = CC  (%)
        B = ZDR (dB)
        A = visibility, see below

        Visibility (alpha) depends only on Z and fades in smoothly with it, so that low-Z clutter near the
        radar doesn't show up as colored speckle, without introducing a hard, unrealistic-looking cutoff edge.
        The alpha channel is real (not premultiplied against a fixed background): the radar_polar/radar_cartesian
        ImageVisuals use the 'translucent' GL state, so the GPU blends each pixel against whatever is actually
        underneath (the map), letting it show through where there's no echo instead of a solid black disc.

        CC and ZDR require a higher signal-to-noise ratio than Z to give a reliable estimate, so it's common
        for a pixel to have a valid Z value while CC and/or ZDR are unavailable for that same pixel (e.g. in
        weak/distant precipitation). Rather than blanking such a pixel entirely -- which would tear visible
        holes in otherwise-continuous precipitation areas -- each channel falls back independently to a
        neutral 'typical light rain' value (CC high, ZDR small positive) when its own data is missing. Only
        when Z itself is unavailable does the pixel become (fade towards) fully transparent.

        ESSL mode (self.gui.polrgb_params['ESSL_MODE'], default False/off): switches R/G/B/alpha from the
        continuous passthrough above to the fixed lookup-table RGB from the ESSL ECSS2025 poster (Van 't Veen,
        Groenemeijer & Pucik) -- R: Z 30->60dBZ, G: CC 100->70% (inverted), B: ZDR 0->4dB, each linearly
        interpolated (and clipped) between just those 2 points. Alpha uses the poster's own 11-point piecewise-
        linear Z-alpha curve (-10..40dBZ) instead of the 2-point ALPHA_GAMMA fade above. Real GPU alpha blending
        (translucent GL state, see class docstring above) is kept in BOTH modes -- the poster's own compositing
        formula x1=a*(a*x0+1-a) bakes translucency into RGB against an assumed black background, which would
        reintroduce exactly the black-disc-hides-the-map problem the real alpha channel was built to avoid (15
        augustus 2026), so it's deliberately not reproduced here. The toggle lives in polrgb_params like every
        other PolRGB setting; the Settings -> PolRGB checkbox itself belongs in nlr.py (not covered by this file).

        All numeric parameters below are tunable via Settings -> PolRGB (self.gui.polrgb_params) and persist
        across sessions; see nlr.py's settings_tabpolrgb/change_polrgb_param for the GUI side. The values
        looked up here ARE the current GUI values -- this isn't a one-time default, every call re-reads
        self.gui.polrgb_params, so a change in Settings takes effect on the next redraw without needing a
        restart. The hardcoded fallback dict guards only against a missing/corrupted key, not against the
        normal case of the user having tuned these away from their original defaults.
        """
        params = getattr(self.gui, 'polrgb_params', {})
        fallback_defaults = {
            'Z_MIN':-10.0, 'Z_MAX':60.0, 'CC_MIN':70.0, 'CC_MAX':100.0, 'ZDR_MIN':0.0, 'ZDR_MAX':3.0,
            'Z_FADE_LO':-15.0, 'Z_FADE_HI':10.0, 'ALPHA_GAMMA':0.6, 'CC_FALLBACK':97.0, 'ZDR_FALLBACK':0.5,
            'Z_GAMMA':2.0, 'ESSL_MODE':False,
            'ESSL_Z_MIN':self.ESSL_Z_RANGE_POSTER_DEFAULT[0], 'ESSL_Z_MAX':self.ESSL_Z_RANGE_POSTER_DEFAULT[1],
            'ESSL_CC_MIN':self.ESSL_CC_RANGE_POSTER_DEFAULT[0], 'ESSL_CC_MAX':self.ESSL_CC_RANGE_POSTER_DEFAULT[1],
            'ESSL_ZDR_MIN':self.ESSL_ZDR_RANGE_POSTER_DEFAULT[0], 'ESSL_ZDR_MAX':self.ESSL_ZDR_RANGE_POSTER_DEFAULT[1],
            'ESSL_ALPHA_Z':self.ESSL_ALPHA_Z_POSTER_DEFAULT, 'ESSL_ALPHA_V':self.ESSL_ALPHA_V_POSTER_DEFAULT,
        }
        def p(key):
            return params[key] if key in params else fallback_defaults[key]

        Z_MIN, Z_MAX = p('Z_MIN'), p('Z_MAX')
        CC_MIN, CC_MAX = p('CC_MIN'), p('CC_MAX')
        ZDR_MIN, ZDR_MAX = p('ZDR_MIN'), p('ZDR_MAX')
        Z_FADE_LO, Z_FADE_HI = p('Z_FADE_LO'), p('Z_FADE_HI')
        ALPHA_GAMMA = p('ALPHA_GAMMA')
        CC_FALLBACK, ZDR_FALLBACK = p('CC_FALLBACK'), p('ZDR_FALLBACK')
        Z_GAMMA = max(p('Z_GAMMA'), 0.1)
        ESSL_MODE = bool(p('ESSL_MODE'))
        ESSL_Z_MIN, ESSL_Z_MAX = p('ESSL_Z_MIN'), p('ESSL_Z_MAX')
        ESSL_CC_MIN, ESSL_CC_MAX = p('ESSL_CC_MIN'), p('ESSL_CC_MAX')
        ESSL_ZDR_MIN, ESSL_ZDR_MAX = p('ESSL_ZDR_MIN'), p('ESSL_ZDR_MAX')
        ESSL_ALPHA_Z, ESSL_ALPHA_V = np.array(p('ESSL_ALPHA_Z')), np.array(p('ESSL_ALPHA_V'))

        def norm(arr, vmin, vmax):
            return np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)

        scan = self.crd.scans[j]
        source = self.source_classes[self.data_source()]

        def fetch(product):
            # get_data_multiple_scans returns either (data, scantimes, volume_starttime, volume_endtime) or
            # (data, scantimes, volume_starttime, volume_endtime, meta) depending on the source -- e.g.
            # Leonardo_vol_rainbow3/5 (IMGW/Poland) return 4 values, every other source returns 5. Unpack
            # defensively so this works for either, rather than assuming a fixed 5-value return (which would
            # raise a ValueError for the 4-value sources).
            returns = source.get_data_multiple_scans(
                product, [scan], productunfiltered=False, polarization='H', apply_dealiasing=False)
            data, scantimes = returns[0], returns[1]
            arrays = data[scan]
            duplicate = self.duplicate(product, scan)
            duplicate = duplicate if duplicate < len(arrays) else 0
            scantime = scantimes[scan][duplicate] if scan in scantimes and duplicate < len(scantimes.get(scan, [])) else None
            return arrays[duplicate].astype('float32'), scantime

        try:
            data_z, scantime_z = fetch('z')
            data_cc, _ = fetch('c')
            data_zdr, _ = fetch('d')
            # The panel title reads self.scantimes[j] directly, and for every other product that's set inside
            # the per-source get_data() methods in nlr_importdata.py (which 'g' deliberately bypasses, see the
            # 'g' != check in get_data above). Without this, the title's displayed time would stay frozen at
            # whatever it was the last time this panel showed a different product -- the underlying data DOES
            # refresh correctly (via the fetch() calls above), only the displayed time was stuck. Z's scantime
            # is used since all three channels come from the same scan and should share virtually the same time.
            if scantime_z is not None:
                self.scantimes[j] = scantime_z
        except Exception as e:
            print(e, '_calculate_polrgb, panel '+str(j))
            traceback.print_exception(type(e), e, e.__traceback__)
            self.dont_store_in_memory[j] = True
            shape = self.data[j].shape if j in self.data and self.data[j].ndim == 2 else (1, 1)
            # Fully transparent RGBA (not opaque black), so a failed fetch shows the map underneath rather than
            # a solid black panel.
            return np.zeros((*shape, 4), dtype='uint8')

        # get_data_multiple_scans already masks missing values to NaN (see e.g. the 'd'-product handling, which
        # combines the Zh and Zv masks before subtracting). Any remaining mismatch in shape between products
        # (which in principle shouldn't occur for products imported from the same scan) is handled defensively.
        shape = data_z.shape
        if data_cc.shape != shape or data_zdr.shape != shape:
            n_az = min(data_z.shape[0], data_cc.shape[0], data_zdr.shape[0])
            n_rng = min(data_z.shape[1], data_cc.shape[1], data_zdr.shape[1])
            data_z, data_cc, data_zdr = (a[:n_az, :n_rng] for a in (data_z, data_cc, data_zdr))

        nodata_z = np.isnan(data_z)
        # Each channel falls back independently -- a missing CC or ZDR doesn't blank out a pixel that has a
        # perfectly valid Z value, it just makes that one channel render as 'typical light rain' instead.
        data_z_filled = np.where(nodata_z, Z_FADE_LO, data_z)
        data_cc_filled = np.where(np.isnan(data_cc), CC_FALLBACK, data_cc)
        data_zdr_filled = np.where(np.isnan(data_zdr), ZDR_FALLBACK, data_zdr)

        # Cache the raw (un-normalized, NaN where missing) physical values for this panel, so the mouse-cursor
        # readout (see Plotting.update_data_readout) can show the actual Z/CC/ZDR values under the cursor
        # without needing to re-fetch from disk on every mouse move. Keyed by panel; overwritten on every
        # recalculation, which is fine since the readout always wants the data for whatever is currently shown.
        self.polrgb_raw_data[j] = (data_z, data_cc, data_zdr)

        if ESSL_MODE:
            r, g, b, alpha = self._essl_polrgb_channels(
                data_z_filled, data_cc_filled, data_zdr_filled, nodata_z,
                ESSL_Z_MIN, ESSL_Z_MAX, ESSL_CC_MIN, ESSL_CC_MAX, ESSL_ZDR_MIN, ESSL_ZDR_MAX,
                ESSL_ALPHA_Z, ESSL_ALPHA_V)
        else:
            r = norm(data_z_filled, Z_MIN, Z_MAX) ** Z_GAMMA  # Z_GAMMA>1: lage dBZ blijft donker, hoge dBZ snel rood (ESSL-stijl).
            g = norm(data_cc_filled, CC_MIN, CC_MAX)
            b = norm(data_zdr_filled, ZDR_MIN, ZDR_MAX)

            # Visibility depends only on Z: that's the channel with the best sensitivity, and the one that defines
            # where precipitation is considered present at all. Gamma-correct so weak echo near the bottom of the
            # fade range becomes more visible, while clutter right at Z_FADE_LO still renders as fully transparent.
            alpha = norm(data_z_filled, Z_FADE_LO, Z_FADE_HI) ** ALPHA_GAMMA
            alpha[nodata_z] = 0.0

        # CC-zichtbaarheidsfilter (15 augustus 2026, op Eriks verzoek: "dat groen van de regen wil ik kwijt"
        # -- gewone regen heeft een hoge CC ONGEACHT Z, dus CC_MIN/CC_MAX hierboven -- die alleen de
        # kleurintensiteit bepalen, niet OF een pixel getekend wordt -- helpen daar niet tegen). Werkt op de
        # RUWE data_cc (voor de CC_FALLBACK-opvulling), zodat een pixel zonder CC-data NOOIT wordt
        # weggefilterd (een vergelijking met NaN is altijd onwaar) -- alleen aantoonbaar hoge CC verdwijnt.
        # Hagel heeft per definitie een lagere CC dan gewone regen, dus blijft hierdoor onaangeroerd. Los
        # van/onafhankelijk van het analoge 3D-only filter (volume3d_polrgb_cc_max in nlr.py).
        cc_hide_above = getattr(self.gui, 'polrgb_cc_hide_above', None)
        if cc_hide_above is not None:
            alpha[data_cc > cc_hide_above] = 0.0

        # Return RGBA (not RGB premultiplied against a fixed background): the ImageVisual is already configured
        # with the 'translucent' GL state (blend = src_alpha, one_minus_src_alpha), so passing a real alpha
        # channel lets vispy blend each pixel against whatever is actually underneath (the map), rather than
        # against a hardcoded black background -- which is what made the whole radar circle opaque black where
        # there's no echo, hiding the map underneath it.
        rgba = np.stack([r, g, b, alpha], axis=-1)
        return (rgba * 255).astype('uint8')

    def ensure_melting_level_grid_current(self):
        """Haalt (zo nodig) het gedeelde temperatuurrooster op (nlr_hclass.NL_GRID_POINTS) voor het
        HUIDIGE tijdstip, en cachet het resultaat in self.melting_level_grid_results.

        LOS van de radar (in tegenstelling tot ensure_melting_levels_current/self.melting_level_h0_m,
        die bij de radarlocatie zelf horen en de altijd-zichtbare balk voeden) - dit rooster hoort bij
        het TIJDSTIP alleen, want de temperatuurstructuur van de atmosfeer heeft niks met een specifieke
        radar te maken (zie gesprek met Erik, 22 juli). Daardoor wordt bij het wisselen tussen radars
        (bv. Herwijnen/Den Helder) dit rooster niet opnieuw opgehaald als het tijdstip gelijk blijft.
        """
        # BUGFIX-VOORKOMEND (25 juli, zelfde soort valkuil als de eerdere VILD-schijf-cachebug):
        # de cache-sleutel was tot nu toe alleen (datum, tijd) - als je de handmatige override
        # aan/uit zet, of het station/datum/uur daarvan wijzigt, terwijl de scan zelf hetzelfde
        # blijft, zou ensure_melting_level_grid_current hierboven anders VROEGTIJDIG terugkeren
        # met het oude (automatische of eerder handmatig gekozen) resultaat. Vandaar de
        # override-instellingen mee in de sleutel, maar ALLEEN als de override uit staat een
        # simpele (datum, tijd) zoals voorheen - zo blijft de bestaande caching voor het
        # automatische pad ongewijzigd.
        manual_override_active = getattr(self.gui, 'melting_levels_manual_override', False)
        if manual_override_active:
            current_key = (self.crd.date, self.crd.time, True,
                           getattr(self.gui, 'melting_levels_manual_station', None),
                           getattr(self.gui, 'melting_levels_manual_date', None),
                           getattr(self.gui, 'melting_levels_manual_hour', None))
        else:
            current_key = (self.crd.date, self.crd.time)
        if getattr(self, 'melting_level_grid_key', None) == current_key:
            return

        try:
            scan_dt = dtime.datetime.strptime(self.crd.date+self.crd.time, '%Y%m%d%H%M')
        except Exception as e:
            gv.log_product_check(
                f"WAARSCHUWING - HCLASS - kon datum/tijd niet parsen voor temperatuurrooster "
                f"(date={self.crd.date}, time={self.crd.time}): {e}")
            self.melting_level_grid_results = None
            self.melting_level_grid_key = current_key
            return

        try:
            if getattr(self.gui, 'melting_levels_manual_override', False):
                # HANDMATIGE TERUGVALOPTIE (25 juli, op Eriks verzoek, zie nlr.py's
                # select_melting_levels_override/ALT+W): in plaats van de automatische,
                # dichtstbijzijnde-tijd-keuze uit een op het TIJDSTIP gebaseerd station/uur,
                # gebruikt de gebruiker hier zelf een gekozen station+datum+uur. Net als de
                # bestaande automatische Wyoming-fallback (die ook maar 1 vaste waarde voor
                # heel Nederland geeft, zie nlr_meltinglevels.py) wordt hetzelfde resultaat
                # voor ALLE 16 roosterpunten gebruikt - een handmatig gekozen sounding is
                # sowieso altijd de Wyoming-bron, die nooit een rooster kent.
                try:
                    year, month, day = (int(x) for x in self.gui.melting_levels_manual_date.split('-'))
                    manual_result = nml.get_melting_levels_wyoming_manual(
                        self.gui.melting_levels_manual_station, year, month, day,
                        self.gui.melting_levels_manual_hour)
                    self.melting_level_grid_results = [manual_result] * len(hc.NL_GRID_POINTS)
                except (ValueError, AttributeError, nml.MeltingLevelError) as e:
                    gv.log_product_check(
                        f"WAARSCHUWING - HCLASS - handmatige Wyoming-keuze mislukt "
                        f"(station={getattr(self.gui, 'melting_levels_manual_station', '?')}, "
                        f"datum={getattr(self.gui, 'melting_levels_manual_date', '?')}, "
                        f"uur={getattr(self.gui, 'melting_levels_manual_hour', '?')}): {e}")
                    self.melting_level_grid_results = None
            else:
                self.melting_level_grid_results = nml.get_melting_levels_grid(hc.NL_GRID_POINTS, scan_dt)
        except Exception as e:
            print(e, 'ensure_melting_level_grid_current')
            traceback.print_exception(type(e), e, e.__traceback__)
            gv.log_product_check(
                f"WAARSCHUWING - HCLASS - ophalen temperatuurrooster mislukt op {self.crd.date} {self.crd.time}: {e}")
            self.melting_level_grid_results = None
        self.melting_level_grid_key = current_key

    def _calculate_hclass(self, j):
        """Bereken de hydrometeorenclassificatie (HCLASS) voor paneel j: Z, ZDR, KDP, RhoHV plus een
        geschatte temperatuur per bin (lineaire interpolatie/extrapolatie tussen het 0C- en -20C-niveau,
        zie nlr_meltinglevels.py), via het C-band-schema van Dolan et al. 2013
        (CSU_RadarTools-parameters, zie nlr_hclass.py voor bron/attributie).

        Zelfde opzet als _calculate_polrgb hierboven: haalt zijn eigen componentproducten op via
        get_data_multiple_scans (de normale get_data(j)-weg wordt overgeslagen, zie de 'g'/'f'-check
        in self.get_data()), en retourneert direct een kant-en-klare (azimuth, range, 4) uint8
        RGBA-array. Hetzelfde RGBA-passthrough-renderpad als PolRGB (product 'g') wordt hiervoor
        hergebruikt in nlr_plotting.py (is_rgb_product geldt daar voor 'g' EN 'f').

        Temperatuur per bin wordt (sinds 22 juli) geschat via een GEDEELD rooster van punten over heel
        Nederland (nlr_hclass.NL_GRID_POINTS), i.p.v. 1 vast punt op de radarlocatie zelf: voor elke bin
        wordt de echte lat/lon berekend (hc.destination_point) en het dichtstbijzijnde roosterpunt
        gekozen (hc.nearest_grid_index). Dit voorkomt afwijkingen aan de rand van het radarbereik en een
        sprong in temperatuur bij het wisselen tussen radars. Als het rooster niet beschikbaar is (bv.
        netwerkfout), wordt zonder temperatuur geclassificeerd (minder scherpe scheiding tussen
        ijs-/vloeistofklassen, zie nlr_hclass.classify_hid), in plaats van de hele berekening te laten
        mislukken.
        """
        scan = self.crd.scans[j]
        source = self.source_classes[self.data_source()]

        def fetch(product):
            # Zelfde defensieve unpack als _calculate_polrgb.fetch hierboven (zie daar voor uitleg).
            returns = source.get_data_multiple_scans(
                product, [scan], productunfiltered=False, polarization='H', apply_dealiasing=False)
            data, scantimes = returns[0], returns[1]
            arrays = data[scan]
            duplicate = self.duplicate(product, scan)
            duplicate = duplicate if duplicate < len(arrays) else 0
            scantime = scantimes[scan][duplicate] if scan in scantimes and duplicate < len(scantimes.get(scan, [])) else None
            return arrays[duplicate].astype('float32'), scantime

        try:
            data_z, scantime_z = fetch('z')
            data_zdr, _ = fetch('d')
            data_kdp, _ = fetch('k')
            data_cc, _ = fetch('c')
            if scantime_z is not None:
                self.scantimes[j] = scantime_z
        except Exception as e:
            print(e, '_calculate_hclass, panel '+str(j))
            traceback.print_exception(type(e), e, e.__traceback__)
            self.dont_store_in_memory[j] = True
            shape = self.data[j].shape if j in self.data and self.data[j].ndim == 2 else (1, 1)
            return np.zeros((*shape, 4), dtype='uint8')

        # ZPHI-verzwakkingscorrectie (C-band, Testud 2000/Bringi 2001/Gou 2019, zie
        # nlr_attenuation.py - alleen toegepast op HCLASS/MESH/POSH/POH/SHI, op Eriks
        # expliciete keuze, NIET op de gewone Z-weergave of PolRGB) - 28 juli 2026.
        # Corrigeert data_z en data_zdr vóór classify_hid; data_kdp/data_cc blijven
        # ongewijzigd (geen ZPHI-formule daarvoor, zie toelichting in nlr_attenuation.py).
        # Defensief: als PhiDP niet beschikbaar is voor deze radar/scan (bv. een
        # radarformaat zonder polarimetrie), gaat HCLASS gewoon door met de ongecorrigeerde
        # data, net als bij een ontbrekend temperatuurrooster hieronder.
        if getattr(self.gui, 'attenuation_correction_enabled', True):
            try:
                data_phidp, _ = fetch('p')
                range_res_km = self.radial_res_all['z'][scan]
                if data_phidp.shape == data_z.shape:
                    data_z, data_zdr = att.correct_scan_zphi(
                        data_z, data_phidp, data_cc / 100., range_res_km, ZDR_dBZ_2d=data_zdr)
                else:
                    print(f"_calculate_hclass: vorm-mismatch scan {scan} - "
                          f"Z={data_z.shape} PhiDP={data_phidp.shape} - GEEN correctie toegepast")
            except Exception as e:
                print(e, '_calculate_hclass attenuation correction, panel '+str(j))
                traceback.print_exception(type(e), e, e.__traceback__)

        # Zelfde defensieve vorm-afstemming als in _calculate_polrgb (in principe zouden vormen altijd
        # moeten matchen omdat ze uit dezelfde scan komen, maar dit voorkomt een crash in het
        # onwaarschijnlijke geval dat dat een keer niet zo is).
        shape = data_z.shape
        others = (data_zdr, data_kdp, data_cc)
        if any(a.shape != shape for a in others):
            n_az = min(data_z.shape[0], *(a.shape[0] for a in others))
            n_rng = min(data_z.shape[1], *(a.shape[1] for a in others))
            data_z, data_zdr, data_kdp, data_cc = (
                a[:n_az, :n_rng] for a in (data_z, data_zdr, data_kdp, data_cc))

        # RhoHV wordt in NLradar intern als percentage (0-100) opgeslagen (zie get_data_multiple_scans:
        # "if product == 'c': data[j][-1] *= 100."), maar de CSU-parameters (nlr_hclass.py) gaan uit
        # van een fractie (0-1), dus hier terugschalen.
        data_cc_frac = data_cc / 100.0

        T = None
        h0_2d = None
        h_minus20_2d = None
        grid_idx_2d = None
        try:
            self.ensure_melting_level_grid_current()
            grid_results = getattr(self, 'melting_level_grid_results', None)
            if grid_results is not None:
                angle = self.scanangles_all_m['z'][scan]
                radial_res = self.radial_res_all['z'][scan]  # km per bin
                radar_elevation_km = gv.radar_elevations.get(self.crd.radar, 0)/1000.0
                n_az, n_rng = shape
                slant_ranges_km = radial_res*(np.arange(n_rng)+0.5)
                heights_km = hc.beam_height_km(slant_ranges_km, angle, radar_elevation_km)
                heights_m_2d = np.broadcast_to(heights_km*1000.0, shape)

                # Grondafstand wordt hier benaderd als gelijk aan de slant range - een geldige
                # vereenvoudiging voor de lage elevatiehoeken (doorgaans de onderste, ~0.3-0.5 graden
                # scan) waarop HCLASS meestal wordt bekeken: cos(0.5 graden) verschilt < 0,004% van 1,
                # dus het verschil met de exacte grondafstand is voor het kiezen van een roosterpunt
                # (nauwkeurigheid ~150 km) totaal verwaarloosbaar.
                ground_ranges_km = slant_ranges_km
                azimuthal_res = 360.0/n_az
                # LET OP: gebruikt azimuth-offset=0 (het midden van bin 0 wordt dus als 0 graden/Noord
                # aangenomen). Dit is dezelfde vereenvoudiging die al impliciet gold in de vorige
                # (1-punts-)versie en in _calculate_polrgb hierboven: self.dsg.data_azimuth_offset[j]
                # wordt voor 'g'/'j' nooit gezet (die slaan de normale get_data(j)-weg over, waar dat
                # normaliter gebeurt). Voor de keuze van het dichtstbijzijnde roosterpunt (nauwkeurigheid
                # ~150 km) is een eventuele kleine afwijkende offset niet van belang.
                azimuths_deg = azimuthal_res*(np.arange(n_az)+0.5)

                lat0, lon0 = gv.radarcoords[self.crd.radar]
                az_2d, rng_2d = np.meshgrid(azimuths_deg, ground_ranges_km, indexing='ij')
                lats_2d, lons_2d = hc.destination_point(lat0, lon0, rng_2d, az_2d)

                grid_idx_2d = hc.nearest_grid_index(lats_2d, lons_2d, hc.NL_GRID_POINTS)

                grid_h0 = np.array([r['h0_m'] if r['h0_m'] is not None else np.nan for r in grid_results])
                grid_h20 = np.array([r['h_minus20_m'] if r['h_minus20_m'] is not None else np.nan for r in grid_results])
                h0_2d = grid_h0[grid_idx_2d]
                h_minus20_2d = grid_h20[grid_idx_2d]

                T = hc.estimate_temperature(heights_m_2d, h0_2d, h_minus20_2d)
        except Exception as e:
            print(e, '_calculate_hclass temperature grid estimation, panel '+str(j))
            traceback.print_exception(type(e), e, e.__traceback__)
            T = None
            h0_2d = None
            h_minus20_2d = None
            grid_idx_2d = None

        hid, _scores = hc.classify_hid(data_z, data_zdr, data_kdp, data_cc_frac, T=T)

        # Cache voor de mouse-cursor-uitlezing (zie Plotting.update_data_readout), analoog aan
        # self.polrgb_raw_data hierboven. h0_2d/h_minus20_2d (22 juli, op Eriks verzoek i.p.v. de losse
        # 0C/-20C-balk in nlr.py, die is verwijderd) laten de tooltip de daadwerkelijke 0C-/-20C-hoogte
        # voor DEZE specifieke bin tonen (uit het gedeelde temperatuurrooster), i.p.v. een apart
        # venstertje met 1 vaste waarde voor het hele beeld. grid_idx_2d (teruggezet op Eriks verzoek,
        # naast h0_2d/h_minus20_2d, niet in plaats ervan) laat de tooltip OOK tonen welk roosterpunt is
        # gebruikt.
        self.hclass_raw_data[j] = (hid, data_z, data_zdr, data_kdp, data_cc, T, h0_2d, h_minus20_2d, grid_idx_2d)

        return hc.classes_to_rgba(hid)

    def get_cross_section(self, product, point_a, point_b, n_samples=2000):
        """Computes a vertical cross-section of the given product ('v' for Velocity, 'z' for Reflectivity) along
        the straight line from point_a to point_b (both given as AEQD x/y coordinates in km from the radar, the
        same convention used by the A/B line tool in nlr_plotting.py).

        For each of n_samples points evenly spaced along the line, and for every available elevation scan,
        looks up the product value at that ground position and computes the physical height there (via the
        existing var1_to_var2 'gr+theta->h' formula, the same one already used elsewhere for e.g. choosing the
        scan closest to a given beam height). Returns a flat list of (distance_along_line_km, height_km,
        value) tuples -- one per (sample point, scan) combination that actually has data there -- ready to be
        handed to a plotting routine for the actual cross-section visual.

        This deliberately fetches ALL elevation scans in a single get_data_multiple_scans call (rather than
        looping over scans with separate calls), mirroring how _calculate_polrgb fetches its 3 channels --
        except _calculate_polrgb only ever asks for a single scan ([scan]) at a time, while this function asks
        for every available elevation scan at once. That wider request was found to disturb the data source's
        shared self.scannumbers_forduplicates/self.scannumbers_all bookkeeping (probably because requesting the
        full scan range triggers an internal recomputation path that normal single-scan, single-panel requests
        never hit), causing a KeyError further down the line the next time an ordinary panel tried to fetch its
        own data (see nlr_datasourcegeneral.duplicate). To avoid that, both are snapshotted before this call and
        restored immediately after (in a finally, so they're restored even if the fetch raises) -- this function
        only reads from the fetched arrays, it has no legitimate reason to leave any lasting change to that
        shared bookkeeping behind for the rest of the application.

        Dealiasing (correcting Nyquist-velocity wrap-around) is only meaningful for Velocity, never for
        Reflectivity, so apply_dealiasing is only passed as True when product == 'v'.
        """
        scan_to_angle = self.scanangles_all.get(product, {})
        # 90-degree 'birdbath' scans (if present) don't usefully contribute to a horizontal cross-section and
        # would need special-cased geometry (a single point straight up rather than a ground-range relation),
        # so they're excluded here.
        scans = sorted(scan for scan, angle in scan_to_angle.items() if isinstance(scan, int) and angle < 89.9)
        if not scans:
            return []

        source = self.source_classes[self.data_source()]
        scannumbers_forduplicates_snapshot = copy.deepcopy(self.scannumbers_forduplicates)
        scannumbers_all_snapshot = copy.deepcopy(self.scannumbers_all)
        try:
            returns = source.get_data_multiple_scans(product, scans, productunfiltered=False, polarization='H',
                                                       apply_dealiasing=(product == 'v'))
        finally:
            self.scannumbers_forduplicates = scannumbers_forduplicates_snapshot
            self.scannumbers_all = scannumbers_all_snapshot
        data_per_scan, scantimes = returns[0], returns[1]

        radar_xy = np.array([0., 0.]) # By definition: point_a/point_b are already in radar-centered AEQD km.
        line_vec = point_b-point_a
        line_length_km = np.linalg.norm(line_vec)
        if line_length_km == 0:
            return []
        t_values = np.linspace(0, 1, n_samples)
        sample_points = point_a+t_values[:, None]*line_vec # shape (n_samples, 2)
        distances_along_line = t_values*line_length_km

        # Ground range (km from radar) and azimuth (degrees, 0=North, clockwise) for every sample point.
        ground_ranges = np.linalg.norm(sample_points-radar_xy, axis=1)
        # Computed directly here (rather than via nlr_functions.azimuthal_angle, which only handles a single
        # scalar (x,y) pair, not an array of points) using the same 0=North-clockwise convention.
        azimuths = (90.-np.degrees(np.arctan2(sample_points[:, 1], sample_points[:, 0]))) % 360.

        points = [] # list of (distance_km, height_km, value)
        mask_value = self.pb.mask_values.get(product, None)
        for scan in scans:
            angle = scan_to_angle[scan]
            duplicate = self.duplicate(product, scan)
            arrays = data_per_scan.get(scan, [])
            if not arrays:
                continue
            arr = arrays[duplicate if duplicate < len(arrays) else 0]
            n_az, n_range = arr.shape
            radial_res = self.radial_res_all.get(product, {}).get(scan, None)
            if radial_res is None or radial_res <= 0:
                continue

            heights_km = ft.var1_to_var2(ground_ranges, angle, 'gr+theta->h')
            # NOTE: assumes a zero azimuth offset for row 0 of each scan. get_data_multiple_scans doesn't
            # expose a per-scan azimuth offset (that's only determined per-panel, during the normal get_data(j)
            # path that this function deliberately bypasses -- same as _calculate_polrgb does for Z/CC/ZDR).
            # True for the great majority of scans tested so far; revisit if a cross-section ever looks
            # azimuthally misaligned for a particular radar/scan.
            az_offset = 0.
            row_indices = np.round(((azimuths-az_offset) % 360.)/360.*n_az).astype('int64') % n_az
            col_indices = np.round(ground_ranges/radial_res).astype('int64')
            in_range_mask = col_indices < n_range

            for i in np.nonzero(in_range_mask)[0]:
                value = arr[row_indices[i], col_indices[i]]
                if mask_value is not None and value == mask_value:
                    continue
                if np.isnan(value):
                    continue
                points.append((distances_along_line[i], heights_km[i], float(value)))

        return points

    def get_velocity_cross_section(self, point_a, point_b, n_samples=2000):
        """Backwards-compatible wrapper around get_cross_section for Velocity specifically."""
        return self.get_cross_section('v', point_a, point_b, n_samples=n_samples)

    def get_volume_grid(self, product, x_range, y_range, grid_res_km=0.5, z_max_km=15., z_res_km=0.25,
                         apply_dealiasing=None, smoothing_sigma_cells=1.2):
        """Reconstructs a regular 3D Cartesian grid (x, y in AEQD km from the radar, z = height in km) for
        `product`, meant as the data backbone for a future 3D/volumetric view (see conversation with Erik,
        July 2026, about the "scan slierten, geen wolk" problem with a naive per-scan-surface 3D view).

        This is the direct 2D-area generalization of get_cross_section/get_velocity_cross_section above: same
        scan-fetching pattern (single get_data_multiple_scans call for every available elevation, with the
        same snapshot/restore of scannumbers_forduplicates/scannumbers_all -- see the docstring of
        get_cross_section for why that's needed), same idea of looking up a value per grid point via
        row/col indices into each scan's raw (azimuth, range) array. The one real difference is *what* gets
        interpolated: get_cross_section bins scattered (distance, height, value) points onto a raster and
        interpolates vertically per distance-column (see the vertical-interpolation block in
        nlr_plotting.show_cross_section); this function instead has an explicit x/y column for every grid
        cell up front (no binning needed, since there's exactly one ground_range/azimuth per column) and
        interpolates vertically per (x, y) column onto a fixed z-axis, using the same "only interpolate
        BETWEEN the lowest and highest scan with real data in that column, never extrapolate beyond" rule.

        x_range, y_range: (min_km, max_km) tuples, AEQD-relative to the radar (same convention as
        ab_line_a/ab_line_b for the existing cross-section line).
        grid_res_km: horizontal grid spacing (x and y).
        z_max_km/z_res_km: vertical grid extent and spacing.

        Returns a dict with:
            'grid'       : ndarray, shape (n_z, n_y, n_x). np.nan where no bracketing measurement exists
                           (above the highest scan, below the lowest scan, or beyond a scan's range/az
                           coverage) -- deliberately never invented/extrapolated, same policy as the
                           cross-section.
            'x', 'y'     : 1D arrays of grid-cell-center coordinates (km).
            'z'          : 1D array of grid-cell-center heights (km).
            'scans_used' : list of scan numbers that contributed (diagnostic only).
        Returns None if no non-birdbath scans are available for `product`.
        """
        scan_to_angle = self.scanangles_all.get(product, {})
        scans = sorted(scan for scan, angle in scan_to_angle.items() if isinstance(scan, int) and angle < 89.9)
        if not scans:
            return None

        source = self.source_classes[self.data_source()]
        scannumbers_forduplicates_snapshot = copy.deepcopy(self.scannumbers_forduplicates)
        scannumbers_all_snapshot = copy.deepcopy(self.scannumbers_all)
        try:
            returns = source.get_data_multiple_scans(
                product, scans, productunfiltered=False, polarization='H',
                apply_dealiasing=(product == 'v') if apply_dealiasing is None else apply_dealiasing)
        finally:
            self.scannumbers_forduplicates = scannumbers_forduplicates_snapshot
            self.scannumbers_all = scannumbers_all_snapshot
        data_per_scan, scantimes = returns[0], returns[1]

        x_min, x_max = x_range
        y_min, y_max = y_range
        n_x = max(2, int(round((x_max-x_min)/grid_res_km)))
        n_y = max(2, int(round((y_max-y_min)/grid_res_km)))
        x_axis = x_min+(np.arange(n_x)+0.5)*grid_res_km
        y_axis = y_min+(np.arange(n_y)+0.5)*grid_res_km
        xx, yy = np.meshgrid(x_axis, y_axis) # shape (n_y, n_x)
        ground_ranges = np.hypot(xx, yy).ravel() # (n_cols,)
        # Same 0=North-clockwise azimuth convention as get_cross_section.
        azimuths = ((90.-np.degrees(np.arctan2(yy, xx))) % 360.).ravel()
        n_cols = ground_ranges.size

        mask_value = self.pb.mask_values.get(product, None)
        scan_heights, scan_values, used_scans = [], [], []
        for scan in scans:
            angle = scan_to_angle[scan]
            duplicate = self.duplicate(product, scan)
            arrays = data_per_scan.get(scan, [])
            if not arrays:
                continue
            arr = arrays[duplicate if duplicate < len(arrays) else 0]
            n_az, n_range = arr.shape
            radial_res = self.radial_res_all.get(product, {}).get(scan, None)
            if radial_res is None or radial_res <= 0:
                continue

            heights_km = ft.var1_to_var2(ground_ranges, angle, 'gr+theta->h')
            az_offset = 0. # Same simplifying assumption as get_cross_section -- see its docstring.
            # TERUGGEZET (5 juli 2026) naar de oorspronkelijke nearest-neighbor lookup. Er is kort
            # geexperimenteerd met bilineaire interpolatie hier, als (foutieve) poging om een felle rand
            # rondom de 3D-vorm bij V op te lossen -- die rand bleek uiteindelijk een losstaande, bekende
            # vispy Volume/MIP-renderbeperking te zijn (zie show_volume_3d_viewer in nlr.py), dus deze
            # wijziging loste niets op. Omdat de bilineaire versie ook niet apart gevalideerd was met echte
            # productiedata (in tegenstelling tot deze nearest-neighbor versie, die de hele sessie door met
            # echte buien is getest en goedgekeurd), is besloten 'm terug te zetten i.p.v. een niet-bewezen
            # wijziging te laten staan zonder aangetoonde meerwaarde.
            row_idx = np.round((azimuths-az_offset) % 360./360.*n_az).astype('int64') % n_az
            col_idx = np.round(ground_ranges/radial_res).astype('int64')
            in_range = col_idx < n_range
            col_idx_clipped = np.where(in_range, col_idx, 0)
            raw = arr[row_idx, col_idx_clipped]

            valid = in_range.copy()
            if mask_value is not None:
                valid &= (raw != mask_value)
            if np.issubdtype(raw.dtype, np.floating):
                valid &= ~np.isnan(raw)
            vals = np.full(n_cols, np.nan, dtype='float32')
            vals[valid] = raw[valid]

            scan_heights.append(heights_km)
            scan_values.append(vals)
            used_scans.append(scan)

        if not scan_values:
            return None

        scan_heights = np.asarray(scan_heights) # (n_scans, n_cols)
        scan_values = np.asarray(scan_values)    # (n_scans, n_cols)

        n_z = max(1, int(round(z_max_km/z_res_km)))
        z_axis = (np.arange(n_z)+0.5)*z_res_km
        grid = np.full((n_z, n_cols), np.nan, dtype='float32')

        # Per grid column: interpolate between the lowest and highest scan with real data there, exactly
        # like the per-distance-column interpolation in show_cross_section, just done here per (x,y) column
        # instead. NOTE: this loops in plain Python over every grid column (as show_cross_section also does
        # over its distance-columns) -- fine for a first version / a modest, user-selected area, but if this
        # is later pushed to a much finer/larger grid, this loop (not the data fetch) is the first place to
        # optimize, e.g. by exploiting that scan_heights is already monotonically increasing per column
        # (scans are sorted by ascending angle, and height increases with angle for fixed ground range) to
        # vectorize the interpolation across all columns at once instead of column-by-column.
        for col in range(n_cols):
            v_col = scan_values[:, col]
            valid = ~np.isnan(v_col)
            if valid.sum() < 2:
                continue
            h_valid = scan_heights[valid, col]
            v_valid = v_col[valid]
            order = np.argsort(h_valid)
            h_valid, v_valid = h_valid[order], v_valid[order]
            lo, hi = h_valid[0], h_valid[-1]
            in_bounds = (z_axis >= lo) & (z_axis <= hi)
            if in_bounds.any():
                grid[in_bounds, col] = np.interp(z_axis[in_bounds], h_valid, v_valid)

        grid = grid.reshape(n_z, n_y, n_x)
        # TERUGGEDRAAID (5 juli 2026): een uitschieter-filter hier bleek verkeerd -- bij velocity zijn
        # scherpe overgangen vaak ECHTE structuur (windschering, mesocycloon), geen ruis, en het filter
        # verwijderde daardoor legitieme data. _remove_volume_grid_outliers blijft hieronder gedefinieerd
        # maar wordt niet meer aangeroepen.
        #
        # OOK TERUGGEDRAAID (5 juli 2026): de rand-erosie hieronder loste de felle rand rondom de vorm bij
        # V NIET op (grondig uitgezocht en uiteindelijk bevestigd: dat is een interpolatie-"overshoot" van
        # vispy's Volume-visual zelf op harde randen -- zie de toelichting in show_volume_3d_viewer in
        # nlr.py -- geen data-kenmerk, dus ook niet op te lossen door data weg te snijden). Erosie kostte
        # daarmee alleen onnodig een laagje echte randdata, zonder baat. _erode_volume_grid_edges blijft
        # hieronder gedefinieerd maar wordt niet meer aangeroepen.
        # Horizontale gladstrijking tussen NAAST ELKAAR liggende (x,y)-kolommen -- dit was tot nu toe de
        # ontbrekende stap: elke kolom werd al verticaal netjes geinterpoleerd (tussen scans), maar kolommen
        # onderling niet, wat de scherpe, blokkerige "Minecraft"-rand gaf die Erik terecht aanwees (5 juli
        # 2026) i.p.v. een vloeiende wolkvorm. NaN-bewust: lege cellen tellen niet mee in het gemiddelde, en
        # blijven leeg als er te weinig echte buren zijn (geen verzonnen data ver buiten de bui).
        # smoothing_sigma_cells<=0 slaat deze stap over (instelbaar via Settings -> Miscellaneous in nlr.py),
        # voor als je liever de ruwe, ongeladde reconstructie wilt zien.
        if smoothing_sigma_cells > 0:
            grid = self._smooth_volume_grid_horizontally(grid, sigma_cells=smoothing_sigma_cells)
        return {'grid': grid, 'x': x_axis, 'y': y_axis, 'z': z_axis, 'scans_used': used_scans}

    def get_volume_grid_polrgb(self, x_range, y_range, grid_res_km=0.5, z_max_km=15., z_res_km=0.25,
                                smoothing_sigma_cells=1.2):
        """PolRGB-equivalent van get_volume_grid hierboven, voor de 3D-viewer (CTRL+SHIFT+4, product 'g').

        get_volume_grid reconstrueert een 3D-rooster voor EEN scalair product met een kleurentabel erop
        toegepast; PolRGB heeft geen kleurentabel (R=Z, G=CC, B=ZDR, RGBA-passthrough -- zie
        DataSource_General._calculate_polrgb voor de 2D-versie van dezelfde aanpak). Deze functie roept
        get_volume_grid daarom 3x apart aan (voor 'z', 'c', 'd', met IDENTIEKE grid-parameters, dus
        identieke x/y/z-assen), en combineert de 3 roosters daarna per voxel met EXACT dezelfde formule/
        parameters (self.gui.polrgb_params) als _calculate_polrgb. Smoothing gebeurt dus per fysieke
        grootheid (dBZ/%/dB), VOOR het combineren tot kleur -- niet achteraf op de al-gecombineerde RGB
        (dat zou verkeerde tussenkleuren geven bij het middelen van bijvoorbeeld rood en blauw).

        Returns een dict met:
            'rgba'  : ndarray, shape (n_z, n_y, n_x, 4), float32 in [0, 1]. Alpha=0 waar geen Z-data
                      bestaat (buiten het geinterpoleerde bereik, net als NaN bij get_volume_grid).
            'z_raw' : ndarray, shape (n_z, n_y, n_x), de ruwe (ongevulde) Z-reconstructie in dBZ, NaN waar
                      geen data -- voor eventuele extra filtering (volume3d_min_value/cirkelvormig gebied)
                      door de aanroeper, op dezelfde manier als bij een gewoon scalair product.
            'cc_raw': ndarray, shape (n_z, n_y, n_x), de ruwe (ongevulde) CC-reconstructie in %, NaN waar
                      geen data -- voor het aparte CC-zichtbaarheidsfilter (volume3d_polrgb_cc_max) door de
                      aanroeper, los van/onafhankelijk van het Z-filter hierboven.
            'x', 'y', 'z' : 1D-assen, identiek aan een gewone get_volume_grid-aanroep met dezelfde parameters.
            'scans_used'  : scans gebruikt voor Z (diagnostisch, zoals bij get_volume_grid).
        Returns None als er geen (niet-birdbath) scans beschikbaar zijn voor Z, CC of ZDR.
        """
        result_z = self.get_volume_grid('z', x_range, y_range, grid_res_km=grid_res_km, z_max_km=z_max_km,
                                          z_res_km=z_res_km, apply_dealiasing=False,
                                          smoothing_sigma_cells=smoothing_sigma_cells)
        if result_z is None:
            return None
        result_cc = self.get_volume_grid('c', x_range, y_range, grid_res_km=grid_res_km, z_max_km=z_max_km,
                                           z_res_km=z_res_km, apply_dealiasing=False,
                                           smoothing_sigma_cells=smoothing_sigma_cells)
        result_zdr = self.get_volume_grid('d', x_range, y_range, grid_res_km=grid_res_km, z_max_km=z_max_km,
                                            z_res_km=z_res_km, apply_dealiasing=False,
                                            smoothing_sigma_cells=smoothing_sigma_cells)
        if result_cc is None or result_zdr is None:
            return None

        grid_z, grid_cc, grid_zdr = result_z['grid'], result_cc['grid'], result_zdr['grid']
        # Assen komen rechtstreeks uit x_range/y_range/grid_res_km/z_max_km/z_res_km (zie get_volume_grid),
        # dus identiek voor alle 3 aanroepen -- shapes horen daardoor altijd overeen te komen. Defensief
        # bijgeknipt voor het geval een van de 3 producten toch een andere n_range/scanopbouw blijkt te
        # hebben, zodat dit nooit met een IndexError crasht.
        if grid_cc.shape != grid_z.shape or grid_zdr.shape != grid_z.shape:
            n_z = min(grid_z.shape[0], grid_cc.shape[0], grid_zdr.shape[0])
            n_y = min(grid_z.shape[1], grid_cc.shape[1], grid_zdr.shape[1])
            n_x = min(grid_z.shape[2], grid_cc.shape[2], grid_zdr.shape[2])
            grid_z, grid_cc, grid_zdr = (a[:n_z, :n_y, :n_x] for a in (grid_z, grid_cc, grid_zdr))

        # Zelfde parameters/fallbacks als _calculate_polrgb (nooit los opnieuw verzinnen -- 1 bron van
        # waarheid, via Settings -> PolRGB).
        params = getattr(self.gui, 'polrgb_params', {})
        fallback_defaults = {
            'Z_MIN':-10.0, 'Z_MAX':60.0, 'CC_MIN':70.0, 'CC_MAX':100.0, 'ZDR_MIN':0.0, 'ZDR_MAX':3.0,
            'Z_FADE_LO':-15.0, 'Z_FADE_HI':10.0, 'ALPHA_GAMMA':0.6, 'CC_FALLBACK':97.0, 'ZDR_FALLBACK':0.5,
            'Z_GAMMA':2.0, 'ESSL_MODE':False,
            'ESSL_Z_MIN':self.ESSL_Z_RANGE_POSTER_DEFAULT[0], 'ESSL_Z_MAX':self.ESSL_Z_RANGE_POSTER_DEFAULT[1],
            'ESSL_CC_MIN':self.ESSL_CC_RANGE_POSTER_DEFAULT[0], 'ESSL_CC_MAX':self.ESSL_CC_RANGE_POSTER_DEFAULT[1],
            'ESSL_ZDR_MIN':self.ESSL_ZDR_RANGE_POSTER_DEFAULT[0], 'ESSL_ZDR_MAX':self.ESSL_ZDR_RANGE_POSTER_DEFAULT[1],
            'ESSL_ALPHA_Z':self.ESSL_ALPHA_Z_POSTER_DEFAULT, 'ESSL_ALPHA_V':self.ESSL_ALPHA_V_POSTER_DEFAULT,
        }
        def p(key):
            return params[key] if key in params else fallback_defaults[key]

        Z_MIN, Z_MAX = p('Z_MIN'), p('Z_MAX')
        CC_MIN, CC_MAX = p('CC_MIN'), p('CC_MAX')
        ZDR_MIN, ZDR_MAX = p('ZDR_MIN'), p('ZDR_MAX')
        Z_FADE_LO, Z_FADE_HI = p('Z_FADE_LO'), p('Z_FADE_HI')
        ALPHA_GAMMA = p('ALPHA_GAMMA')
        CC_FALLBACK, ZDR_FALLBACK = p('CC_FALLBACK'), p('ZDR_FALLBACK')
        Z_GAMMA = max(p('Z_GAMMA'), 0.1)
        ESSL_MODE = bool(p('ESSL_MODE'))
        ESSL_Z_MIN, ESSL_Z_MAX = p('ESSL_Z_MIN'), p('ESSL_Z_MAX')
        ESSL_CC_MIN, ESSL_CC_MAX = p('ESSL_CC_MIN'), p('ESSL_CC_MAX')
        ESSL_ZDR_MIN, ESSL_ZDR_MAX = p('ESSL_ZDR_MIN'), p('ESSL_ZDR_MAX')
        ESSL_ALPHA_Z, ESSL_ALPHA_V = np.array(p('ESSL_ALPHA_Z')), np.array(p('ESSL_ALPHA_V'))

        def norm(arr, vmin, vmax):
            return np.clip((arr-vmin)/(vmax-vmin), 0.0, 1.0)

        nodata_z = np.isnan(grid_z)
        data_z_filled = np.where(nodata_z, Z_FADE_LO, grid_z)
        data_cc_filled = np.where(np.isnan(grid_cc), CC_FALLBACK, grid_cc)
        data_zdr_filled = np.where(np.isnan(grid_zdr), ZDR_FALLBACK, grid_zdr)

        if ESSL_MODE:
            r, g, b, alpha = self._essl_polrgb_channels(
                data_z_filled, data_cc_filled, data_zdr_filled, nodata_z,
                ESSL_Z_MIN, ESSL_Z_MAX, ESSL_CC_MIN, ESSL_CC_MAX, ESSL_ZDR_MIN, ESSL_ZDR_MAX,
                ESSL_ALPHA_Z, ESSL_ALPHA_V)
        else:
            r = norm(data_z_filled, Z_MIN, Z_MAX) ** Z_GAMMA
            g = norm(data_cc_filled, CC_MIN, CC_MAX)
            b = norm(data_zdr_filled, ZDR_MIN, ZDR_MAX)
            alpha = norm(data_z_filled, Z_FADE_LO, Z_FADE_HI) ** ALPHA_GAMMA
            alpha[nodata_z] = 0.0

        rgba = np.stack([r, g, b, alpha], axis=-1).astype('float32')
        return {'rgba': rgba, 'z_raw': grid_z, 'cc_raw': grid_cc, 'x': result_z['x'], 'y': result_z['y'],
                'z': result_z['z'], 'scans_used': result_z['scans_used']}

    def _erode_volume_grid_edges(self, grid, erosion_cells=1):
        """Verwijdert een dunne laag (erosion_cells cellen breed) rondom de RAND van de geldige data (waar
        geldig overgaat in NaN), in alle richtingen (x, y, en z) tegelijk. Zie de toelichting hierboven in
        get_volume_grid voor waarom -- kort gezegd: een aanhoudende felle rand in de 3D-weergave die niet
        oplosbaar bleek via de kleurenschaal/interpolatie, dus nu weggesneden aan de bron in plaats van
        weergegeven en dan proberen te verdoezelen."""
        try:
            from scipy.ndimage import binary_erosion
        except ImportError:
            print('_erode_volume_grid_edges: scipy niet beschikbaar, sla rand-erosie over.')
            return grid
        valid = ~np.isnan(grid)
        if not valid.any() or erosion_cells <= 0:
            return grid
        eroded_valid = binary_erosion(valid, iterations=erosion_cells)
        cleaned = grid.copy()
        cleaned[valid & ~eroded_valid] = np.nan
        return cleaned

    def _remove_volume_grid_outliers(self, grid, threshold_frac=0.3):
        """Verwijdert (zet op NaN) cellen die sterk afwijken van de mediaan van hun 6 directe buren (boven,
        onder, noord, zuid, oost, west) -- zie de uitgebreide toelichting in get_volume_grid hierboven.
        threshold_frac is het toegestane verschil met die buur-mediaan, als fractie van de totale
        waardespreiding in het grid (dus zelfde soort robuuste, dimensieloze drempel voor elk product, of
        het nou dBZ of m/s is). Verwijdert alleen, verzint nooit een vervangende waarde."""
        valid = ~np.isnan(grid)
        if not valid.any():
            return grid
        value_range = float(np.nanmax(grid)-np.nanmin(grid))
        if value_range <= 0:
            return grid
        threshold = threshold_frac*value_range

        shifts = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
        neighbour_values = []
        for dz, dy, dx in shifts:
            shifted = np.roll(np.roll(np.roll(grid, dz, axis=0), dy, axis=1), dx, axis=2).copy()
            # np.roll wraps rond -- de kant waar het vandaan "wrapt" hoort niet als buur mee te tellen.
            if dz == 1: shifted[0, :, :] = np.nan
            elif dz == -1: shifted[-1, :, :] = np.nan
            if dy == 1: shifted[:, 0, :] = np.nan
            elif dy == -1: shifted[:, -1, :] = np.nan
            if dx == 1: shifted[:, :, 0] = np.nan
            elif dx == -1: shifted[:, :, -1] = np.nan
            neighbour_values.append(shifted)
        neighbour_stack = np.array(neighbour_values)
        with np.errstate(invalid='ignore'):
            # All-NaN warnings zijn hier verwacht (cellen zonder enige geldige buur) en onschuldig -- die
            # cellen blijven vanzelf ongemoeid (is_outlier wordt daar False, zie hieronder).
            old_settings = np.seterr(invalid='ignore')
            local_median = np.nanmedian(neighbour_stack, axis=0)
            np.seterr(**old_settings)

        is_outlier = valid & ~np.isnan(local_median) & (np.abs(grid-local_median) > threshold)
        cleaned = grid.copy()
        cleaned[is_outlier] = np.nan
        return cleaned

    def _smooth_volume_grid_horizontally(self, grid, sigma_cells=1.2):
        """NaN-bewuste Gaussische gladstrijking over alleen de x/y-assen van een get_volume_grid-resultaat
        (niet over z, want die richting is al vloeiend geinterpoleerd tussen scans in get_volume_grid zelf).
        Gebruikt scipy.ndimage als die beschikbaar is; anders wordt het ongeladde (blokkeriger) resultaat
        teruggegeven in plaats van te crashen.
        """
        try:
            from scipy.ndimage import gaussian_filter
        except ImportError:
            print('_smooth_volume_grid_horizontally: scipy niet beschikbaar, sla gladstrijking over.')
            return grid
        valid = ~np.isnan(grid)
        if not valid.any():
            return grid
        filled = np.where(valid, grid, 0.).astype('float32')
        weight = valid.astype('float32')
        # sigma=0 voor de z-as (axis 0) betekent: geen smoothing in die richting, alleen over y (axis 1) en
        # x (axis 2).
        smoothed_sum = gaussian_filter(filled, sigma=(0, sigma_cells, sigma_cells), mode='nearest')
        smoothed_weight = gaussian_filter(weight, sigma=(0, sigma_cells, sigma_cells), mode='nearest')
        with np.errstate(invalid='ignore', divide='ignore'):
            smoothed = smoothed_sum/smoothed_weight
        # Cellen met nauwelijks echte buren (grotendeels lege omgeving) teruggezet naar NaN, in plaats van
        # een verdunde/half-verzonnen waarde te tonen net buiten de rand van de bui.
        smoothed[smoothed_weight < 0.2] = np.nan
        return smoothed.astype('float32')

    def perform_mono_prf_dealiasing(self, j, data, vn=None, azis=None, da=None): # j is the panel
        if VDA is None:
            self.dont_store_in_memory[j] = True
            return data
        data[data == self.pb.mask_values['v']] = np.nan
        vn = self.nyquist_velocities_all_mps[self.crd.scans[j]] if vn is None else vn
        data = VDA(data, vn, azis, da, extra_dealias='extra' in self.gui.dealiasing_setting)
        self.mono_prf_dealiasing_performed = True
        return data
    
    def calculate_derived_with_tilts(self, j): # j is the panel
        """Currently only calculates SRV.
        Also, SRV is calculated from uint velocity data (which is dtype in which velocity is available at this point), 
        since this is both well possible and clearly cheaper than first converting uint to float and then back after calculation.
        This is taken into account in the function dt.calculate_srv_array.
        When later calculating more products than just SRV, there might be the desire to store these products into memory. 
        This might then be done in this function too.
        """
        product, i_p = self.crd.products[j], gv.i_p[self.crd.products[j]]
        mask_value_ip, mask_value_p = self.pb.mask_values_int[i_p], self.pb.mask_values_int[product]
        data_mask = self.data[j] == mask_value_ip
        if product == 's':
            self.data[j] = dt.calculate_srv_array(self.data[j], self.gui.stormmotion, self.data_azimuth_offset[j])
        self.data[j][data_mask] = mask_value_p

    def get_data_multiple_scans(self,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        """Function that imports data for products that require data from more than one scan
        """
        return self.source_classes[self.data_source()].get_data_multiple_scans(product,scans,productunfiltered,polarization,apply_dealiasing,max_range)
    
    def restore_previous_attributes(self):
        for j in self.attrs_before:
            self.__dict__[j] = self.attrs_before[j]
        
    def restore_previous_attributes_radar(self):
        for j in self.determined_volume_attributes_radars[self.crd.radar].keys():
            self.__dict__[j] = self.determined_volume_attributes_radars[self.crd.radar][j]
        #All attributes are now the same as for the last succesfull trial for self.crd.radar
    
    
    def data_source(self, radar = None):
        """Gives the data source for a given input radar, or if not given, the data source for the current radar; self.crd.radar.
        """
        radar = radar if radar else self.crd.radar
        return gv.data_readsources[radar]
    
    def update_scannumbers_forduplicates(self):
        """When going back to the previous plot, the previous values for scannumbers_forduplicates are restored in the function 
        self.crd.back_to_previous_plot.
        """
        self.scannumbers_forduplicates = {i:j for i,j in self.scannumbers_forduplicates.items() if i in self.scannumbers_all['z']}
        for i,j in self.scannumbers_all['z'].items():
            if self.crd.going_back_to_previous_plot:
                # self.scannumbers_forduplicates has already been set in self.crd.back_to_previous_plot, but it's possible that the saved
                # values used here were obtained for a smaller number of scans than what's available now. The following line ensures that
                # these new scans are added when needed.
                self.scannumbers_forduplicates[i] = self.scannumbers_forduplicates.get(i, 0)
            elif self.crd.lrstep_beingperformed:
                if self.scannumbers_forduplicates.get(i, 0)+1 > len(j):
                    self.scannumbers_forduplicates[i] = len(j)-1
                elif not i in self.scannumbers_forduplicates or self.crd.desired_timestep_minutes() < self.crd.volume_timestep_m:
                    self.scannumbers_forduplicates[i] = 0 if self.crd.lrstep_beingperformed == 1 else len(j)-1
            elif self.crd.requesting_latest_data:
                self.scannumbers_forduplicates[i] = len(j)-1
            else:
                #Use the last values for scannumbers_forduplicates that were used for this radar, if they are valid.
                #self.scannumbers_forduplicates_radars[self.crd.radar] gets updated in self.pb.set_newdata each time that function is called.
                previous_val_radar = self.scannumbers_forduplicates_radars.get(self.crd.radar, {self.crd.radar:{}}).get(i, 100)
                self.scannumbers_forduplicates[i] = min(len(j)-1, previous_val_radar)  
                  
                    
    def update_selected_scanangles(self, update_allpanels=False):
        panellist = range(self.pb.max_panels) if update_allpanels else self.pb.panellist
        for j in panellist:
            #'z' is always available as key for the volume attributes
            product = gv.i_p[self.crd.products[j]] if self.scanangles_all_m[gv.i_p[self.crd.products[j]]] else 'z'
            max_scanangle = max(self.scanangles_all_m[product].values())
            lowest_scan = (1, 2)[self.range_nyquistvelocity_scanpairs_indices[j]] if self.scanpair_present else 1
            if self.crd.scans[j] == lowest_scan:
                self.selected_scanangles[j] = 0.
            else:
                self.selected_scanangles[j] = self.scanangles_all_m[product].get(self.crd.scans[j], max_scanangle)          
                    
    def get_scanangles_allproducts(self, scanangles_all):
        """This function combines the dictionaries with per product the scanangles for all scans, given in scanangles_all, into one dictionary, that contains
        all the different scanangles that are available for the scans. If all scanangles are available for all products, then the returned dictionary is equal
        to scanangles_all[product], where product is any of the availabe products.
        """
        scanangles_allproducts = {}
        products = [p for p in scanangles_all if len(scanangles_all[p])]
        for j in scanangles_all['z']: #scanangles_all['z'] should contain keys for all available scans
            if j==1:
                scanangles_allproducts[j] = min(scanangles_all[p][1] for p in products)
            else:
                # print(j, scanangles_all, scanangles_allproducts)
                diff_greaterthanzero = np.array([[p, scanangles_all[p][j]-scanangles_allproducts[j-1]] for p in products if j in scanangles_all[p] and
                                                 scanangles_all[p][j]-scanangles_allproducts[j-1] > 0])
                p = diff_greaterthanzero[diff_greaterthanzero[:,1].argmin()][0] if len(diff_greaterthanzero) else 'z'
                scanangles_allproducts[j] = scanangles_all[p][j]
        return scanangles_allproducts
        
    def get_panel_center_heights(self, dist_to_radar_panels, scanangles, scanangles_all, scanpair_present, panellist=None):
        panellist = self.pb.panellist if panellist is None else panellist
        heights = {}
        for j in panellist:
            lowest_scan = (1, 2)[self.range_nyquistvelocity_scanpairs_indices[j]] if scanpair_present else 1
            if scanangles[j] == scanangles_all[lowest_scan]:
                # Set height equal to 0 when the lowest scan is shown, such that this scan will keep being shown
                heights[j] = 0.
            else:
                heights[j] = ft.var1_to_var2(dist_to_radar_panels[j], scanangles[j], 'gr+theta->h')
        return heights
    def manually_set_panel_center_heights(self, heights):
        # This function is called in self.gui.set_choice, in order to use selected heights from a saved choice.
        self.panel_center_heights = {'time':pytime.time(), 'heights':heights}
                                
    def check_need_scans_change(self, delta_time):
        """This function checks whether it is desired to switch from scans. This could be the case when the radar volume changes, e.g. because 
        of a change of radar or dataset. Or it can be because it is desired to keep scan beam height at panel center as constant as possible, 
        which can require a change of scans when switching from radar or when translating view with storm motion.
        """
        move_view_with_storm = self.gui.use_storm_following_view and self.pb.firstplot_performed and not self.gui.setting_saved_choice and\
                               (self.crd.process_datetimeinput_running or self.crd.lrstep_beingperformed)
        choose_nearest_height = self.crd.scan_selection_mode == 'height' and (self.changing_radar or move_view_with_storm or 
                                                                              self.gui.setting_saved_choice)
        try:
            s, sb = self.scanangles_all_m, self.attrs_before['scanangles_all_m']
            scanangles_all_changed = any([any([abs(sb[p][i]-s[p][i]) > 0.1 for i in s[p]]) for p in s])
        except Exception:
            scanangles_all_changed = True
        choose_nearest_scanangle = not choose_nearest_height and (scanangles_all_changed or self.gui.setting_saved_choice)
        #If self.gui.setting_saved_choice, then it is always desired to choose the scan with the appropriate Nyquist velocity 
        #(that is not too low when viewing a velocity-related product).
        if (self.pb.firstplot_performed or self.gui.setting_saved_choice) and (choose_nearest_height or choose_nearest_scanangle) and\
        len(self.scanangles_all['z']) > 1:
            """When choose_nearest_height == True this function chooses the scans in such a way that the beam elevation at the center
            of the screen is closest to the value that it was before, for the previous radar or time. 
            If this is not the case, then this function simply chooses the scan whose scanangle is closest to the selected scanangle.
            """

            if choose_nearest_height: 
                self.pb.get_corners()
                if move_view_with_storm and delta_time:
                    # When using storm-centering the translation of the view only takes place after calling this function, so self.pb.corners
                    # doesn't yet take into account this new translation. This translation is approximately performed here (the real
                    # translation depends on scan time differences, which are only available once the new scans are retrieved).
                    if abs(delta_time) <= self.pb.max_delta_time_move_view_with_storm:
                        # In the lines below the check len(self.scannumbers_all['z'][self.crd.scans[j]]) > 1 is important, but an issue
                        # is that it is dependent on self.crd.scans[j], which might be become different after finishing this function.
                        # This is especially the case when the previous radar volume was incomplete. Therefore, an initial iteration of
                        # this function is performed with delta_time=0, which updates self.crd.scans.
                        self.check_need_scans_change(0)
                        for j in self.pb.panellist:
                            dt_before = self.gui.current_case['datetime'] if self.gui.switch_to_case_running else self.crd.before_variables['datetime']
                            # delta_time is valid for panel 0. In the case that panel 0 shows duplicates and other panels not, delta_t
                            # for these panels should be based on the volume time difference.
                            delta_t = delta_time if len(self.scannumbers_all['z'][self.crd.scans[j]]) > 1 else\
                                      ft.datetimediff_s(dt_before, self.crd.date+self.crd.time)
                            self.pb.corners[j] += self.pb.translation_dist_km(delta_t, dt_before)
        
            #scanangles_all contains all scanangles for which data is present in the volume. It is determined in this way, because it is possible that the number
            #of scans differs per product, and therefore also the number of scanangles. If this is the case, then self.scanangles_all_m[product] refers for these 
            #scans to the nearest other scan (in terms of scanangle) for which data is present. 
            #If self.scanangles_all_m[product] contains less scanangles than scanangles_all, then using self.scanangles_all_m[product] could lead to finding a
            #suboptimal 'nearest' scan. When viewing product this would not matter, but when viewing another product for which other scanangles are available,
            #this could lead to displaying a suboptimal scan, which is not desired.
            scanangles_all = self.get_scanangles_allproducts(self.scanangles_all_m)
            scanangles_all_values = np.array(list(scanangles_all.values()))
            scans_radars = self.scans_radars.get(self.radar_dataset_before, {})
            if scans_radars:
                try:
                    scanangles_all_before = self.get_scanangles_allproducts(scans_radars['scanangles_all'])
                except Exception as e:
                    print(e, self.radar_dataset_before, scans_radars['scanangles_all'])
                    raise Exception(1/0)
            
            scanangles_before = {}
            for j in self.pb.panellist:
                try:
                    if j in scans_radars['panellist']:
                        scanangles_before[j] = scanangles_all_before[scans_radars['scans'][j]]
                    else:
                        scanangles_before[j] = scanangles_all[self.crd.scans[j]]
                except Exception:
                    #self.scans_radars is not yet defined in this case, so use the current values of the parameters.
                    scanangles_before[j] = scanangles_all[self.crd.scans[j]]

            if choose_nearest_height:
                save_time = self.panel_center_heights.get('time', None)
                need_update = not save_time or any(j > save_time for j in (self.time_last_panzoom, self.time_last_forcedscanchange, 
                              self.time_last_choosenearestscanangle, self.crd.time_last_change_scan_selection_mode))
                new_panels = None if need_update else [j for j in self.pb.panellist if not j in self.panel_center_heights['heights']]
                if need_update or new_panels:
                    """These are only updated when zooming or panning has taken place in the mean time, or when the scans have been changed 
                    purposefully, or when the nearest scanangle is chosen in the mean time. This implies e.g. that when going to multiple radars, 
                    always the first radar in a sequence is used for determining the center heights, such that the center heights 
                    do not change in the next part of the sequence.
                    """
                    old_corners = self.scans_radars.get(self.radar_dataset_before, {}).get('corners', self.pb.corners)
                    old_distance_to_radar = {j:np.linalg.norm(old_corners[j].mean(axis=0)) for j in self.pb.panellist}
                    heights = self.get_panel_center_heights(old_distance_to_radar, scanangles_before, scanangles_all_before,
                                                            self.scanpair_present_before, panellist=new_panels)
                    if need_update:
                        self.panel_center_heights = {'time':pytime.time(), 'heights':heights}
                    else:
                        self.panel_center_heights['heights'].update(heights)
            
            for j in self.pb.panellist:
                
                if choose_nearest_height:
                    #Find the scanangle for which the beam elevation at the center of the screen is closest to the value it was before,
                    #for the previous radar.
                    new_distance_to_radar = np.linalg.norm(self.pb.corners[j].mean(axis=0))
                    # print(j, scanangles_all_values, new_distance_to_radar, self.panel_center_heights['heights'][j])
                    new_scanangle = ft.find_scanangle_closest_to_beamelevation(scanangles_all_values,new_distance_to_radar,self.panel_center_heights['heights'][j])
                else:
                    """Find the scanangle that is closest to the selected scanangle. If there are 2 scanangles that are equally close,
                    then the one is chosen that is closest to the scanangle for which currently data is displayed in panel j.
                    """
                    new_scanangle, index = ft.closest_value(scanangles_all_values,self.selected_scanangles[j], return_index=True)
                    if len(scanangles_all) > 1:
                        scanangles_all_without_newscanangle = np.delete(scanangles_all_values, index)
                        next_closest_scanangle = ft.closest_value(scanangles_all_without_newscanangle, self.selected_scanangles[j])
                        if abs(self.selected_scanangles[j]-new_scanangle) == abs(self.selected_scanangles[j]-next_closest_scanangle) and\
                        abs(scanangles_before[j]-next_closest_scanangle) < abs(scanangles_before[j]-new_scanangle):
                            new_scanangle = next_closest_scanangle
                                      
                        if self.crd.plot_mode in ('Row', 'Column') or self.gui.setting_saved_choice:
                            """It might be the case that multiple selected scanangles get mapped onto the same actual new scanangle, in which case 
                            multiple panels might show the same product-scan combination. That is not desired under these conditions, so in that
                            case new_scanangle gets changed below.
                            """
                            for i in self.pb.panellist[:self.pb.panellist.index(j)]:
                                scanangle_i = scanangles_all[self.crd.scans[i]]
                                if new_scanangle == scanangle_i and self.crd.products[i] == self.crd.products[j] and (self.gui.setting_saved_choice or 
                                (gv.i_p[self.crd.products[i]] != 'v' or self.crd.apply_dealiasing[i] == self.crd.apply_dealiasing[j]) and
                                self.crd.productunfiltered[i] == self.crd.productunfiltered[j]) and len(scanangles_all_values)-index > 1:
                                    new_scanangle = scanangles_all_values[index+1]
                                    index += 1
                                                     
                if self.scanpair_present and new_scanangle in [scanangles_all[i] for i in (1,2)]:
                    self.crd.scans[j] = (1,2)[self.range_nyquistvelocity_scanpairs_indices[j]]
                else:
                    self.crd.scans[j] = [i for i,k in scanangles_all.items() if k==new_scanangle][0]
                
            if choose_nearest_height:
                # Only in this case. In other cases the selected scanangle should not change.
                self.update_selected_scanangles()
                # Also update self.selected_scanangles_before, since this change should not lead to updating self.time_last_forcedscanchange
                self.selected_scanangles_before = self.selected_scanangles.copy()
                    
            if choose_nearest_height:
                self.time_last_choosenearestheight = pytime.time()
            else:
                self.time_last_choosenearestscanangle = pytime.time()
                    
    def check_presence_large_range_large_nyquistvelocity_scanpair(self,update_scanpairs_indices = False):
        """A range-nyquist velocity scan pair is a pair of scans of which the first has a large range but low Nyquist velocity, and the second has
        a lower range but larger Nyquist velocity, while the scanangles for both scans differ by at most 0.1 degrees. When such a pair is present,
        it might be desired to display e.g. reflectivity for the scan with the larger range, and velocity for the scan with the higher Nyquist
        velocity. In order to let this also be the case when switching from radar, self.range_nyquistvelocity_scanpairs_indices is used, which
        stores the index of the scan in the scan pair that is displayed. 
        For a correct handling of the display of 2 scans of the scan pair, it is important that going up/down updates only the scans in the scan pair,
        because otherwise you will start viewing data for scans with clearly different scanangles, which is likely not desired. This is handled in
        the function self.crd.process_keyboardinput.
        
        This function determines whether a scan pair is present, and returns True if this is the case, and False otherwise. If a pair is present, but
        the scans involved are not 1 and 2, then scanpair_present = False, as in this case handling such a scan pair is much more difficult.
        Further, if update_scanpairs_indices = True, then this function determines the index of the scan in the scan pair that
        is currently being displayed. This index is 0 for the first scan, and 1 for the second.
        """
        scanpair_present = False
        if len(self.scanangles_all['z'])>1:
            scanangle1 = self.scanangles_all_m['z'][1]
            scanangle2 = self.scanangles_all_m['z'][2]
            
            #In the case of Zaventem it is possible that the first scan misses for the velocity, in which case all volume attributes for this 
            #scan are set equal to that for the second. In this case the condition for the difference in Nyquist velocities between both scans
            #is not satisfied (because it is zero), but the scans should still be regarded as a scan pair. That is ensured by including this
            #bool v_scan1_equals_scan2.
            try:
                #Don't check for 'scannumbers_all', because it is possible that scannumbers_all['v'][1] != scannumbers_all['v'][2], while the rest
                #of the attributes is equal. This is the case when the number of duplicates differs between scan 1 and 2 (when e.g. scan 2 has duplicates,
                #and for scan 1 only one of these duplicates is shown). In this case v_scan1_equals_scan2 should be True.
                v_scan1_equals_scan2 = all([getattr(self,j)['v'][1]==getattr(self,j)['v'][2] for j in gv.volume_attributes_p if not j=='scannumbers_all'])
            except Exception:
                #Occurs when no velocity is available, or when there is only one scan.
                v_scan1_equals_scan2 = False
            
            if ft.r1dec(np.abs(scanangle1-scanangle2)) <= 0.1 and self.radial_range_all['z'][1]/self.radial_range_all['z'][2] > 1.25 and\
            (v_scan1_equals_scan2 or any(self.nyquist_velocities_all_mps[j] is None for j in (1,2)) or 
            np.abs(self.nyquist_velocities_all_mps[2]/self.nyquist_velocities_all_mps[1]) > 1.25):
                scanpair_present = True
                
                if update_scanpairs_indices:
                    for i in (1,2):
                        for j in self.pb.panellist:
                            if self.crd.scans[j]==i and not self.crd.products[j] in gv.plain_products:
                                self.range_nyquistvelocity_scanpairs_indices[j] = i-1
        return scanpair_present
        
    def get_dir_string(self,radar,dataset=None,dir_index = None, return_dir_string_list = False):
        radar_dataset = self.get_radar_dataset(radar, dataset)
        
        dir_string_list = bg.dirstring_to_dirlist(self.gui.radardata_dirs[radar_dataset])
        index = dir_index if not dir_index is None else self.gui.radardata_dirs_indices[radar_dataset]
        if not return_dir_string_list:
            return dir_string_list[index]
        else:
            return dir_string_list[index], dir_string_list 
     
    def get_directory(self,date,time,radar,dataset = None,dir_string = None, dir_index = None):
        #Either dir_string or dataset has to be specified. dir_index is an optional argument for self.get_dir_string, 
        #which only has an effect when dir_string = None
        """Returns the directory in which the data is/should be stored for a particular combination of date and time
        """
        if dir_string is None:
            dir_string = self.get_dir_string(radar,dataset,dir_index)        
        if date=='c':
            #This should only be the case when time is also 'c'.
            return bg.get_last_directory(dir_string,self.gui.radar_basedir,radar,self.get_filenames_directory)
        else:
            return opa(bg.convert_dir_string_to_real_dir(dir_string,self.gui.radar_basedir,radar,date,time))
        
    def get_radar_dataset(self, radar=None, dataset=None, no_special_char=False):
        radar = radar if radar else self.crd.radar
        if no_special_char:
            radar = gv.radars_ascii_names[radar]
        dataset = dataset if dataset else self.crd.dataset
        return radar+f'_{dataset}'*(radar in gv.radars_with_datasets)
    def split_radar_dataset(self, radar_dataset):
        index = radar_dataset.find('_')
        radar = radar_dataset if index == -1 else radar_dataset[:index]
        dataset = radar_dataset[len(radar)+1:]
        return radar, dataset
    
    def get_variables(self,radar,dataset):
        radar_dataset = self.get_radar_dataset(radar, dataset)
        
        dir_string_list = bg.dirstring_to_dirlist(self.gui.radardata_dirs[radar_dataset])
        current_dir_string = dir_string_list[self.gui.radardata_dirs_indices[radar_dataset]]
        n = len(dir_string_list)
        return radar_dataset, dir_string_list, current_dir_string, n
        
    def get_nearest_directory(self,radar,dataset,date,time):
        """Returns the absolute path to the nearest directory for which data is available, where nearest is relative to the input date and time.
        As the function self.get_next_directory below, this function takes into account that multiple directory strings can be present in
        self.gui.radardata_dirs[radar_dataset]. That is also done in the same way as in that function, so I refer to that function
        for more explanation.
        """
        radar_dataset, dir_string_list, current_dir_string, n = self.get_variables(radar,dataset)
        #This should only be the case when time is also 'c'.
        find_last_dir = date == 'c'
        if find_last_dir:
            date = ''.join(ft.get_ymdhm(pytime.time())[:3])
        
        directory = bg.get_last_directory(current_dir_string,self.gui.radar_basedir,radar,self.get_filenames_directory)\
                    if find_last_dir else\
                    bg.get_nearest_directory(current_dir_string,self.gui.radar_basedir,radar,date,time,self.get_filenames_directory,self.get_datetimes_from_files)
        if directory:
            dir_date,_ = bg.get_date_and_time_from_dir(directory,current_dir_string,self.gui.radar_basedir,radar)
            if n == 1 or dir_date == date:
                return directory
        
        try:
            directories = []
            for j in dir_string_list:
                if j == current_dir_string:
                    directories.append(directory)
                else:
                    directories += [bg.get_last_directory(j,self.gui.radar_basedir,radar,self.get_filenames_directory)
                                    if find_last_dir else
                                    bg.get_nearest_directory(j,self.gui.radar_basedir,radar,date,time,self.get_filenames_directory,self.get_datetimes_from_files)]
            dir_dates = [bg.get_date_and_time_from_dir(d, dir_string_list[i], self.gui.radar_basedir, radar)[0]
                         if d else '19000101' for i,d in enumerate(directories)]
                
            datediffs = np.array([np.abs(ft.datetimediff_s(j+'0000',date+'0000')) for j in dir_dates],dtype = 'int64')
            index = np.argmin(datediffs)
            # The current directory string might have changed, so update the corresponding index.
            self.gui.radardata_dirs_indices[radar_dataset] = index
            return directories[index]  
        except Exception as e:
            print(e,'self.dsg.get_nearest_directory')
            return directory
        
            
    def get_next_directory(self,radar,dataset,date,time,direction,desired_newdate = None,desired_newtime = None): 
        """Returns the absolute path of the next directory for which data is available. If desired_newdate and desired_newtime are given,
        then this function first finds the directory for which the date and time are closest to the desired ones. If it differs from the current
        directory, then it is returned. If not, then this function finds the nearest next directory for which data is available.
        
        It is possible that multiple directory strings
        are given in self.gui.radardata_dirs[radar_dataset], and this function takes them all into account. This implies that it determines
        for each directory string the next directory, i.e. the first directory for which the date is changed in the direction given by direction.
        The function then determines the nearest next directory from all next subdirectories that have been found, and returns this one.
        If the nearest next directory has been found for a directory string that differs from the current one, then 
        self.gui.radardata_dirs_indices[radar_dataset] is updated.
        """
        radar_dataset, dir_string_list, current_dir_string, n = self.get_variables(radar,dataset)
        if desired_newdate and desired_newtime:
            directory = self.get_nearest_directory(radar,dataset,desired_newdate,desired_newtime)
            if directory != self.crd.directory:
                return directory
            
        """If no directory is returned, then continue with finding the nearest next directory for which there is data.
        """
        #directory is equal to the current directory if there is no next directory for the current directory string.
        directory = bg.get_next_directory(direction,current_dir_string,self.gui.radar_basedir,radar,self.get_filenames_directory,current_dir=self.crd.directory)
        dir_date,dir_time = bg.get_date_and_time_from_dir(directory,current_dir_string,self.gui.radar_basedir,radar)
        #dir_time is None if there is no ${time} variable in current_dir_string.
              
        #An exception occurs for example when there is no ${date} variable in a directory string, in which case comparing dates causes errors.
        #This should normally not occur, as it is unusual that there is no date in a directory string (that would mean that all data is located in the
        #same folder). If it occurs, then simply directory is returned.
        if True:
            """If n>1, first check whether there is a directory for the next date for current_dir_string, and if so, return this one.
            In the case that a ${time} variable is present in current_dir_string, it is also checked whether there is a next directory for the same
            date but for a different time.
            """
            next_date = ft.next_date(date, direction)
            if n == 1 or (not dir_time and dir_date == next_date) or (dir_time not in (time, None) and dir_date in (date, next_date)):
                return directory
            else:
                directories = []
                for j in dir_string_list:
                    if j == current_dir_string:
                        directories.append(directory)
                    else:
                        dir_j = bg.get_nearest_directory(j,self.gui.radar_basedir,radar,date,time,self.get_filenames_directory,self.get_datetimes_from_files)
                        dir_date_j, dir_time_j = bg.get_date_and_time_from_dir(dir_j,j,self.gui.radar_basedir,radar)
                        #A multiplication by direction is performed to ensure that this comparison gives the desired result also when direction==-1
                        if direction*int(dir_date_j) > direction*int(date):
                            directories.append(dir_j)
                        else:
                            directories.append(bg.get_next_directory(direction,j,self.gui.radar_basedir,radar,self.get_filenames_directory,current_dir=dir_j))
                dir_dates = [bg.get_date_and_time_from_dir(directories[j],dir_string_list[j],self.gui.radar_basedir,radar)[0] for j in range(n)]
                
                orig_dir_string_list = dir_string_list.copy()
                # Remove entries for directory strings that have no next directory in the desired direction
                i_dirs_remove = [i for i,dir_date in enumerate(dir_dates) if direction*int(dir_date) <= direction*int(date)]
                for obj in (dir_string_list, directories, dir_dates):
                    for i in i_dirs_remove[::-1]:
                        obj.pop(i)
                        
                if not dir_string_list:
                    #Return the current directory, because there is no next one
                    return opa(bg.convert_dir_string_to_real_dir(current_dir_string,self.gui.radar_basedir,radar,date,time))
                else:
                    datediffs = np.array([np.abs(ft.datetimediff_s(date+'0000',j+'0000')) for j in dir_dates],dtype = 'int64')
                    #All elements in datediffs are positive
                    index = np.argmin(datediffs)
                    selected_date = dir_dates[index]
                    if selected_date == dir_date:
                        #If dir_date equals selected_date, then always return directory, because it is in this case
                        #not desired that the directory string changes.
                        return directory
                    else:
                        #The current directory string has changed, such that the corresponding index must be updated.
                        new_dir_string = dir_string_list[index]
                        self.gui.radardata_dirs_indices[radar_dataset] = orig_dir_string_list.index(new_dir_string)
                        return directories[index]
        else:
            print(e,'self.dsg.get_next_directory')
            return directory
                             
    
    def get_download_directory(self,radar,dataset=None):
        """Determine the directory in which files that get downloaded should be stored. This directory is determined from dir_string, and is 
        the part of dir_string before any variable (with '${') is encountered. 
        If multiple dir_strings are provided, then for downloading always the 1st (hence dir_index = 0) is used.
        """
        dir_string = self.get_dir_string(radar,dataset,dir_index = 0) 
        return bg.get_download_directory(dir_string, self.gui.radar_basedir)
        
    def get_newest_datetimes_currentdata(self,radar,dataset):
        """Returns the datetimes of the 2 newest (newest first, second-newest second) radar volumes that are available for the radar.
        It is used by the automatic download part of 
        nlr_currentdata.py to determine whether the user is viewing the most recent scans (i.e. the scans that were most recent before the
        download was finished), which is used to determine whether it is desired to plot data for the downloaded file.
        """
        dir_string = self.get_dir_string(radar,dataset)        
        lastdir = bg.get_last_directory(dir_string,self.gui.radar_basedir,radar,self.get_filenames_directory)
        lastdir_filenames = self.get_filenames_directory(radar,lastdir)
        lastdir_datetimes = self.get_datetimes_from_files(radar,lastdir_filenames,lastdir)
        n1 = len(lastdir_datetimes)
        if n1 > 1:
            return lastdir_datetimes[-2:][::-1]
        else:
            secondlastdir = bg.get_next_directory(-1,dir_string,self.gui.radar_basedir,radar,self.get_filenames_directory,current_dir = lastdir)
            secondlastdir_filenames = self.get_filenames_directory(radar,secondlastdir)
            secondlastdir_datetimes = self.get_datetimes_from_files(radar,secondlastdir_filenames,secondlastdir)
            returns = np.append(lastdir_datetimes, secondlastdir_datetimes[-(2-n1):][::-1])
            if len(returns) == 0:
                return None, None
            elif len(returns) == 1:
                return returns[0], None
            else:
                return returns

    def get_filenames_directory(self,radar,directory):
        """Returns the filenames in a list with directory entries (which could also include directories, and which get removed from the list here).
        """
        if directory is None: return [] #Calling os.listdir with argument None lists the current working directory, which is not desired.
        return self.source_classes[self.data_source(radar)].get_filenames_directory(radar,directory)
               
    def get_datetimes_from_files(self,radar,filenames,directory=None,dtype = str,return_unique_datetimes = True, mode='simple'):
        # directory is in most cases not needed to determine datetimes from filenames, but Meteo-France archived files are an exception, since
        # they contain only time in their names. Hence directory is added as argument.
        """If filenames = None this function returns the datetimes of the files that are present in the directory corresponding to the particular 
        radar and dataset. If not filenames = None, then this function returns the datetimes of the filenames that are given as input.
        Mode can be either 'simple' or 'dates ', and the latter should be used when it is desired to also return dates present in the current
        directory if they are determined (is the case for TU Delft).
        The returned object is either an array of datetimes, 1 for each filename, or it is a dictionary with 
        """
        if directory:
            dir_string = self.get_variables(radar, self.crd.selected_dataset)[2]
            # self.get_datetimes_from_files_dirdate is used in self.source_MeteoFrance, because there are filenames that don't contain a date
            self.get_datetimes_from_files_dirdate = bg.get_date_and_time_from_dir(directory, dir_string, self.gui.radar_basedir, radar)[0]

        function = self.source_classes[self.data_source(radar)].get_datetimes_from_files
        returns = ([], []) if mode == 'dates' else []
        if len(filenames):
            returns = function(filenames,dtype,return_unique_datetimes, mode)
        return [np.asarray(j, dtype) for j in returns] if type(returns) == tuple else np.asarray(returns, dtype)
    
    def update_directories_lastupdate_times(self, directory):
        # This function gets called from a signal in nlr_currentdata.py
        self.directories_lastupdate_times[directory] = pytime.time()
    
    def get_datetimes_directory(self,radar,directory,dtype = str,return_unique_datetimes = True):
        lastupdate_time = self.filenames_directory.get(directory, {}).get('time', 0)
        if pytime.time() - lastupdate_time < 60 and lastupdate_time > self.directories_lastupdate_times.get(directory, 0):
            filenames = self.filenames_directory[directory]['fnames']
        else:
            filenames = self.get_filenames_directory(radar,directory)
            self.filenames_directory[directory] = {'time':pytime.time(), 'fnames':filenames}
        # This function gets regularly called from within self.crd.switch_to_nearby_radar, so it's important to cache results.
        # Earlier this was done using the modification time (mtime) of the directory that would also remove the need for always determining filenames.
        # But this method appeared to be not trustworthy, as some filesystems don't update it when the number of files in the directory changes.
        if not directory in self.nfiles_directory or self.nfiles_directory[directory] != len(filenames):
            self.datetimes_directory[directory] = self.get_datetimes_from_files(radar,filenames,directory,dtype,return_unique_datetimes, mode='simple')
            self.nfiles_directory[directory] = len(filenames)
        return self.datetimes_directory[directory]
    
    def get_product_versions(self, radar, filenames, datetimes):
        self.product_versions_datetimesdict = self.product_versions_directory = self.products_version_dependent =\
        self.product_versions_in1file = None
        source_class = self.source_classes[self.data_source(radar)]
        if getattr(source_class, 'get_product_versions', None) and len(filenames):
            # self.product_versions_in1file indicates whether the different product versions are contained in 1 file. If this is the case,
            # then volume attributes are determined for all product versions at once, and attributes for different versions are distinguished
            # by appending the product version to the product key, e.g. 'z' -> 'z z_scan'.
            self.product_versions_datetimesdict, self.products_version_dependent, self.product_versions_in1file =\
                source_class.get_product_versions(filenames, datetimes)
            if self.product_versions_datetimesdict:
                self.product_versions_directory = np.unique(np.concatenate(list(self.product_versions_datetimesdict.values())))
                # Warn once per radar+version-name if a product-version name (e.g. a DWD scan-strategy variant
                # like 'pcp'/'vol') that hasn't been seen before for this radar shows up -- this is the kind
                # of change that could silently break code relying on a specific, known set of version names
                # (e.g. get_volume_grid's plain-vs-versioned key lookup for the 3D viewer) without ever
                # triggering a data-format error, since NLradar would just treat it as yet another valid
                # version rather than something worth flagging.
                if not hasattr(self, '_known_product_versions'):
                    self._known_product_versions = {}
                known = self._known_product_versions.setdefault(radar, set())
                new_versions = set(self.product_versions_directory) - known
                for version_name in new_versions:
                    gv.log_product_check(
                        f"WAARSCHUWING - PRODUCTVERSIE - nieuwe, niet eerder geziene productversie {version_name!r} "
                        f"aangetroffen voor radar {radar} (bijv. een scanstrategie-variant). Dit is geen fout, maar "
                        f"kan erop wijzen dat de databron iets heeft gewijzigd."
                    )
                known.update(self.product_versions_directory)
                if not new_versions and known:
                    gv.log_product_check(f"OK - PRODUCTVERSIE - check uitgevoerd voor radar {radar}, geen nieuwe productversies gevonden.")
    
    def get_files(self,radar,directory,return_datetimes = False):
        filenames = self.get_filenames_directory(radar,directory)
        self.filenames_directory[directory] = {'time':pytime.time(), 'fnames':filenames}
            
        if radar in gv.radars_with_onefileperdate:
            #Here there is one file per date, and therefore multiple radar volumes per file. datetimes contains datetimes of all the radar volumes
            #within all the files, while dates lists just one date for each file.
            datetimes, self.dates = self.get_datetimes_from_files(radar,filenames,directory,dtype = str,return_unique_datetimes = False, mode='dates')
            #self.dates is used in the function self.get_dates_with_archived_data
            datetimes_unique = np.unique(datetimes)
            self.files_datetimesdict = {self.dates[j]:[filenames[j]] for j in range(len(self.dates))}
        else:
            datetimes = self.get_datetimes_from_files(radar,filenames,directory,dtype = str,return_unique_datetimes = False)
            datetimes_unique = np.unique(datetimes)
            self.files_datetimesdict = {j:filenames[datetimes==j] for j in datetimes_unique}
            
        self.get_product_versions(radar, filenames, datetimes)
            
        if return_datetimes:
            return datetimes_unique
    
    def select_files_datetime(self):
        if self.crd.radar in gv.radars_with_onefileperdate:
            self.files_datetime = self.files_datetimesdict[self.crd.date]
        else:
            self.files_datetime = self.files_datetimesdict[self.crd.date+self.crd.time]
            
        self.product_versions_datetime = None
        if self.product_versions_datetimesdict:
            self.product_versions_datetime = self.product_versions_datetimesdict[self.crd.date+self.crd.time]
    
    def get_total_volume_files_size(self, datetime=None):
        files_datetime = self.files_datetimesdict[datetime] if datetime else self.files_datetime
        # self.crd.directory and files_datetime can momentarily be out of sync with each other -- e.g. when
        # the dataset (Z/V) was just switched elsewhere (such as the automatic switch in
        # determine_list_filedatetimes in nlr_changedata.py) without self.crd.directory having been updated to
        # match yet. In that case a filename meant for the other dataset's directory may not exist in
        # self.crd.directory. Skip such files here rather than letting a FileNotFoundError propagate -- this
        # mirrors the defensive 'continue on missing file' approach already used elsewhere in this codebase for
        # similar races, and the resulting total is simply based on whichever files are actually present.
        total = 0
        for j in files_datetime:
            try:
                total += os.path.getsize(self.crd.directory+'/'+j)
            except FileNotFoundError:
                continue
        return total
    
    def get_filenames_and_datetimes_in_datetime_range(self,radar,dataset = None,dir_string = None,startdatetime = None,enddatetime = None,return_abspaths = False,return_completely_selected_directories = False):
        #Either dir_string or dataset should be given as input
        
        """Returns filenames and datetimes of all files within a particular datetime range. If return_abspaths, then filenames contains absolute paths
        to the files, and if return_completely_selected_directories, then names of directories that are completely selected get returned. This 
        information is used in nlr.py to determine whether it is allowed to move a complete directory instead of moving files individually, as the
        former method is much faster.
        """
        if dir_string is None:
            dir_string = self.get_dir_string(radar,dataset)
            
        dirs_abspaths = bg.get_abspaths_directories_in_datetime_range(dir_string,self.gui.radar_basedir,radar,startdatetime,enddatetime)[0]
        
        completely_selected_directories = []
        requested_filenames = np.array([],dtype = 'int64'); requested_datetimes = np.array([],dtype = 'int64')
        for i in dirs_abspaths:
            try:
                filenames = self.get_filenames_directory(radar,i)
                if return_abspaths:
                    filenames = np.array([opa(i+'/'+j) for j in filenames])
                datetimes = self.get_datetimes_from_files(radar,filenames,i,dtype = 'int64',return_unique_datetimes = False)
            
                if not startdatetime is None and not enddatetime is None:
                    requested = (datetimes >= int(startdatetime)) & (datetimes <= int(enddatetime))
                elif not startdatetime is None:
                    requested = datetimes>= int(startdatetime)
                elif not enddatetime is None:
                    requested = datetimes<= int(enddatetime)
                else:
                    requested = np.ones(len(datetimes),dtype = 'bool')
                    
                requested_filenames = np.append(requested_filenames,filenames[requested])
                requested_datetimes = np.append(requested_datetimes,datetimes[requested])
            
                if return_completely_selected_directories and np.count_nonzero(requested)==len(datetimes):
                    completely_selected_directories.append(i)
            except Exception as e:
                print(i, e)
            
        if return_completely_selected_directories:
            return completely_selected_directories,requested_filenames,requested_datetimes
        else:
            return requested_filenames,requested_datetimes
        
    def get_dates_with_archived_data(self,radar,dataset):
        if radar in gv.radars_with_onefileperdate:
            return self.dates
        else:
            _,dir_string_list = self.get_dir_string(radar,dataset,return_dir_string_list = True)
            dates = np.array([])
            for j in dir_string_list:
                dates = np.append(dates,bg.get_dates_with_archived_data(j,self.gui.radar_basedir,radar))
            return np.unique(dates)
    
    def get_radars_with_archived_data_for_date(self,date):   
        if date[0]=='c': date = date[1:]
        startdatetime = date+'0000'
        enddatetime = ''.join(ft.next_date_and_time(date,'0000',1440))
        
        radars_with_data = []
        for i in gv.radars_all:
            datasets = (None,) if not i in gv.radars_with_datasets else ('Z','V')
            for j in datasets:
                radar_dataset = self.get_radar_dataset(i, j)
                _,dir_string_list = self.get_dir_string(i,j,return_dir_string_list = True)
                for k in dir_string_list:
                    if '${date' in k:
                        for l in range(24):
                            # Check for each hour of the day whether a directory is available. This is a crude way to consider both directory formats
                            # with 1 folder per date and formats with folders for different times (e.g. hours).
                            directory = bg.convert_dir_string_to_real_dir(k, self.gui.radar_basedir, i, date, format(l, '04d'))
                            if os.path.exists(directory):
                                radars_with_data.append(radar_dataset)
                                break
                            elif not ('${datetime' in k or '${time' in k):
                                break
                    else:
                        dirs_abspaths,dates_filtered = bg.get_abspaths_directories_in_datetime_range(k,self.gui.radar_basedir,i,startdatetime,enddatetime)
                        if i in gv.radars_with_onefileperdate:
                            for path in dirs_abspaths:
                                files = np.sort(self.get_filenames_directory(i, path))
                                _, dates = self.get_datetimes_from_files(i, files, path, mode='dates')
                                if date in dates:
                                    radars_with_data.append(radar_dataset)
                                    break
                        elif len(dirs_abspaths)>0 and dates_filtered:
                            #If not dates_filtered, then it was not possible to determine which directories contain data for the input date, because no 
                            #${date} variable is located in dir_string.
                            radars_with_data.append(radar_dataset)
                    if radar_dataset in radars_with_data:
                        break
    
        return radars_with_data