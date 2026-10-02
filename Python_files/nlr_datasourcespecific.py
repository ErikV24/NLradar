# Copyright (C) 2016-2024 Bram van 't Veen, bramvtveen94@hotmail.com
# Distributed under the GNU General Public License version 3, see <https://www.gnu.org/licenses/>.

import os
opa=os.path.abspath
import numpy as np
import re
import time as pytime
import copy
import h5py

import nlr_globalvars as gv
import nlr_background as bg
import nlr_functions as ft


"""This module contains the functions that are specific to a particular data source. Each data source has a particular (and of course usually different...) way to store the data,
and classes that import data for a specific data format are contained in nlr_importdata.py. This is because there might be some data sources that use the same data format.
Because also in this case there are very likely differences in the naming of the files and folders in which they are stored, these data sources can't be totally treated the same,
and that's why there are two layers for importing data, this being the first.
Each class should at least implement the functions get_scans_information, get_data, get_data_multiple_scans, get_filenames_directory and get_datetimes_from_files.
"""

class Source_KNMI():
    def __init__(self, gui_class, dsg_class, parent = None):  
        self.gui=gui_class
        self.dsg=dsg_class
        self.crd=self.dsg.crd
        self.dp=self.dsg.dp
        self.pb = self.gui.pb
        
        
        
    def filepath(self):
        return self.crd.directory+'/'+self.dsg.files_datetime[0]
    
    def get_scans_information(self):
        self.dsg.KNMI_hdf5.get_scans_information(self.filepath())

    def get_data(self, j): #j is the panel
        self.dsg.KNMI_hdf5.get_data(self.filepath(),j)

    def get_data_multiple_scans(self,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        return self.dsg.KNMI_hdf5.get_data_multiple_scans(self.filepath(),product,scans,productunfiltered,polarization,apply_dealiasing,max_range) 
                    
    def get_filenames_directory(self,radar,directory):
        try:
            entries=os.listdir(directory)
        except Exception: entries=[]
        filenames=np.sort(np.array([j for j in entries if j[-2:]=='h5' and j[:15]=='RAD_NL'+gv.radar_ids[radar]+'_VOL_NA']))
        return filenames
    
    def get_datetimes_from_files(self,filenames,dtype=str,return_unique_datetimes=True, mode='simple'):
        datetimes=np.array([j[-15:-3] for j in filenames],dtype=dtype)
        return datetimes





class Source_KMI():
    """This class handles data from the radars of the KMI in Jabbeke and Wideumont, but also from the radar of skeyes in Zaventem, in the case 
    that it is provided in the same format as one of the formats in which the KMI provides their data. This is the case for files with a .hdf extension.
    """
    def __init__(self, gui_class, dsg_class, parent = None):  
        self.gui=gui_class
        self.dsg=dsg_class
        self.crd=self.dsg.crd
        self.dp=self.dsg.dp
        self.pb = self.gui.pb
        self._jabbeke_combined_scan_files = {} # combined_scan_index -> filepath, see _build_jabbeke_combined_z_scans
                
    def _current_z_dir_index(self):
        """Returns the currently-active directory index for this radar's 'Z' dataset (0 or 1) -- for Jabbeke,
        this reflects whichever of the 2 directories (long/short range, toggled via CTRL+D /
        self.crd.change_dir_index in nlr_changedata.py) is currently selected; for every other radar in this
        class there's only ever 1 directory, so this harmlessly returns 0."""
        radar_dataset = self.dsg.get_radar_dataset(self.crd.radar, 'Z')
        return self.gui.radardata_dirs_indices.get(radar_dataset, 0)
                                                                                            
    def correct_filename(self, filename, product=None):
        file_extension = os.path.splitext(filename)[1][1:]
        # Try the product string itself as a direct key first (this matters for 'u'-prefixed/unfiltered
        # products, e.g. 'uz' -- see filepath below, which builds this as 'u'+product when
        # productunfiltered is True). gv.productnames_KMI has its own dedicated entries for these (e.g.
        # 'uz': 'dBuZ' for the '.h5' extension -- TH, the unfiltered counterpart of DBZH, see
        # Source_MeteoGate in nlr_currentdata.py for how that file gets downloaded and saved). Falling
        # through to gv.i_p[product] instead would map 'uz' to the SAME category as 'z' (since i_p only
        # tracks broad categories like 'z'/'v'/'w', not the filtered/unfiltered distinction), which would
        # incorrectly look up the FILTERED product's name (dBZ) for what should be the unfiltered one
        # (dBuZ) -- or, before this 'uz' entry was even added to gv.productnames_KMI's lookup path here,
        # would silently match every '.h5' file as "correct" (see the git history of this function for that
        # earlier, more broken state), which is what caused SHIFT+U on a KMI/VMM radar to sometimes load the
        # wrong file's data into a Z-shaped panel and show '(OLD)' in the title bar (self.data_isold in
        # nlr_plotting.py, set whenever self.dsg.get_data raises an exception while reading mismatched data).
        pname = gv.productnames_KMI[file_extension].get(product, None)
        if pname is None:
            i_p = gv.i_p.get(product, None)
            pname = gv.productnames_KMI[file_extension].get(i_p, None)
        
        correct_name = False
        if file_extension == 'hdf':
            i1 = filename.index('pvol')
            i2 = filename.index('scan')
            correct_name = filename[i1+5:i2-1] == pname
            if correct_name and self.crd.radar in gv.radars_with_datasets:
                #Prevent that a file for the wrong dataset is used, in the case that it was put in the wrong directory.
                #For other file extensions there is no way to check this based on the filename only.
                dataset_str = 'scanv' if self.crd.dataset == 'V' else 'scanz'
                correct_name = filename[-9:-4] == dataset_str
        elif file_extension == 'h5':
            correct_name = filename[16:-3] == pname+'.vol' if pname else True
            # Note on Jabbeke's 2 DBZH range variants (short/long, see Source_MeteoGate in
            # nlr_currentdata.py): these are told apart purely by which of the 2 configured directories for
            # 'Jabbeke_Z' (see self.gui.radardata_dirs in nlr.py) self.crd.directory currently points at --
            # self.dsg.files_datetime, which this function filters, only ever contains files from ONE of
            # those 2 directories at a time (whichever is currently selected via
            # self.gui.radardata_dirs_indices['Jabbeke_Z'], switched with the existing CTRL+D shortcut). So
            # no extra check is needed here: by construction, a Jabbeke '.h5' DBZH file passing the check
            # above is always the one matching whichever range is currently selected, because the other
            # range's files simply aren't in this list to begin with.
            
        return correct_name

    def filepath(self, product=None, productunfiltered=False, polarization='H', source_function=None):
        if product:
            product = 'u'+product if productunfiltered else product
        try:
            #First try to obtain a filename for the correct product
            filename = [i for i in self.dsg.files_datetime if self.correct_filename(i, product)][0] 
        except Exception:
            if productunfiltered and polarization == 'V':
                return self.filepath(product[-1], False, polarization)
            elif polarization == 'V':
                return self.filepath(product[-1], productunfiltered, 'H')
            elif productunfiltered:
                return self.filepath(product[-1], False, polarization)
            filename = ''
            if product and source_function == self.get_scans_information and len(self.dsg.files_datetime):
                #Try to find any file with the correct date and time, and if found, then determine the product contained in it
                return self.filepath(source_function=source_function)
            
        filepath = self.crd.directory+'/'+filename if filename else ''
        if product is None:
            # The default product. 'z' does not need to be present, but the only information required for determining whether it is
            # possible to obtain the Nyquist velocities is that the product is not equal to 'v'.
            product = 'z'
        return [filepath, product] if source_function == self.get_scans_information else [filepath, productunfiltered, polarization]
    
    

    def _read_raw_scan_elevations(self, filepath):
        """Reads elangle/nbins/rscale directly for every scan in an ODIM HDF5 file, without touching any
        shared self.dsg state. Used by _build_jabbeke_combined_z_scans below to inspect both of Jabbeke's Z
        files (long and short range) independently before deciding how to combine them."""
        result = {}
        with h5py.File(filepath, 'r') as hf:
            scans = [int(j[7:]) for j in hf if j.startswith('dataset')]
            for j in scans:
                try:
                    attrs = hf['dataset'+str(j)]['where'].attrs
                    result[j] = (ft.rndec(float(attrs['elangle']), 2), int(attrs['nbins']), float(attrs['rscale'])/1000.)
                except Exception:
                    continue
        return result

    def _find_z_file(self, dir_index, product='z'):
        """Locates the actual Z-family file for the given directory index (0=long range, 1=short range for
        Jabbeke; other radars in this class only ever have dir_index 0) and product ('z' for DBZH/filtered,
        'uz' for TH/unfiltered -- both exist side by side in the same directory, per timestamp), independent
        of self.crd.directory -- same approach as the Z/V-search loops and _fix_z_scan_structure elsewhere in
        this class."""
        directory = self.dsg.get_directory(self.crd.date, self.crd.time, self.crd.radar, 'Z', dir_index=dir_index)
        if not directory or not os.path.exists(directory):
            return None
        try:
            files = os.listdir(directory)
        except Exception:
            return None
        matches = [i for i in files if i[:12] == self.crd.date+self.crd.time and self.correct_filename(i, product)]
        return directory+'/'+matches[0] if matches else None

    def _build_jabbeke_combined_z_scans(self):
        """Combines Jabbeke's 2 separate Z files (long range: 299 km; short range: 150 km) into a single,
        continuously-scannable list of elevations, instead of requiring CTRL+D to switch between the 2
        ranges. This mirrors the existing precedent in ODIM_hdf5.get_scans_information for Zaventem, where
        scannumbers_all[product][scan] already gives a RAW scan number relative to a product-SPECIFIC file
        (there, a separate v/w file; here, either of Jabbeke's 2 Z files) -- get_data below is extended to
        look up, per combined scan index, which of the 2 files it actually needs to open.

        IMPORTANT, confirmed by directly inspecting actual downloaded files (not just the MeteoGate catalog,
        which doesn't show this): the 'long' range file does NOT only contain its 6 unique low elevations
        (0.3-3.8 degrees) -- it redundantly ALSO contains the same 5 higher elevations (4.8/6.5/9.0/13.0/25.0
        degrees, at the same 150 km range) that the 'short' file has. These aren't simple duplicates though --
        confirmed by comparing actual pixel data between the two copies of the same elevation, they're 2
        genuinely independent scans of that elevation a few minutes apart (once as part of the long-range
        sweep, once as part of the short-range sweep), so which one gets shown is a real, meaningful choice,
        not just an implementation detail. Naively combining both files' full scan lists produces 5 duplicate
        elevations (20 raw entries for only 15 actually distinct elevations), which is confirmed to break
        both the DOWN/UP scan-stepping (each duplicate needs 2 presses to get past) and the 1-9/0/SHIFT+1-5
        direct-scan-number shortcuts (which only ever reach the first 15 of the 20 raw entries, permanently
        hiding the last 5) -- fixed below by de-duplicating on elevation angle, keeping only one occurrence
        per shared elevation.

        Which of the 2 copies wins for those 5 shared elevations is controlled by CTRL+D (the same shortcut
        that used to switch the whole range before this combining feature existed): dir_index 0 (long,
        '(299 km)') prefers the long file's copy, dir_index 1 (short, '(150 km)') prefers the short file's.
        This keeps CTRL+D meaningful instead of becoming a dead shortcut now that both ranges are always
        combined into one list -- it no longer switches the whole range, just which copy wins on overlap.

        Populates self._jabbeke_combined_scan_files[combined_scan_index] = filepath, and the usual
        scanangles_all/scannumbers_all/radial_bins_all/radial_res_all['z'] (plus every other already-present
        Z-family product key, e.g. 'uz'/'d'/'p'/'k'/'c' -- but NOT 'v'/'w', which come from a wholly separate
        file and must keep their own, independently-read structure).
        """
        long_filepath = self._find_z_file(dir_index=0)
        short_filepath = self._find_z_file(dir_index=1)

        # CTRL+D still toggles self.gui.radardata_dirs_indices as before (see _current_z_dir_index) -- reused
        # here to decide which file's copy of a shared elevation is preferred, via which one is added to
        # 'combined' FIRST below (Python's sort() is stable, so among entries with an identical elangle, the
        # one added first keeps that relative order after sorting, and is therefore the one the de-duplication
        # step right after keeps).
        preferred_first, preferred_second = (
            (long_filepath, short_filepath) if self._current_z_dir_index() == 0 else (short_filepath, long_filepath)
        )

        combined = []  # (elangle, filepath, raw_scan, nbins, rscale)
        for filepath in (preferred_first, preferred_second):
            if not filepath:
                continue
            for raw_scan, (elangle, nbins, rscale) in self._read_raw_scan_elevations(filepath).items():
                combined.append((elangle, filepath, raw_scan, nbins, rscale))

        if not combined:
            return False  # Neither file could be read -- caller falls back to the normal, single-file path.

        combined.sort(key=lambda t: t[0])

        # De-duplicate: keep only the first entry for each distinct elevation angle (see docstring above for
        # why duplicates occur at all, and how CTRL+D controls which copy is kept for a shared elevation).
        deduplicated = []
        seen_elangles = set()
        for entry in combined:
            elangle = entry[0]
            if elangle in seen_elangles:
                continue
            seen_elangles.add(elangle)
            deduplicated.append(entry)
        combined = deduplicated

        # Both the filtered (DBZH/'z') and unfiltered (TH/'uz') files live side by side in the same directory
        # per timestamp -- precompute the unfiltered sibling for each of the (up to 2) directories involved,
        # so SHIFT+U keeps working for the combined scans too instead of always forcing the filtered file.
        uz_sibling = {}
        for dir_index in (0, 1):
            z_fp = self._find_z_file(dir_index=dir_index, product='z')
            if z_fp:
                uz_sibling[z_fp] = self._find_z_file(dir_index=dir_index, product='uz')

        scanangles_all = {}
        scannumbers_all = {}
        radial_bins_all = {}
        radial_res_all = {}
        scan_files = {}
        for i, (elangle, filepath, raw_scan, nbins, rscale) in enumerate(combined, start=1):
            scanangles_all[i] = elangle
            scannumbers_all[i] = [raw_scan]
            radial_bins_all[i] = nbins
            radial_res_all[i] = rscale
            scan_files[i] = {'z': filepath, 'uz': uz_sibling.get(filepath)}

        self.dsg.scanangles_all['z'] = scanangles_all
        self.dsg.scannumbers_all['z'] = scannumbers_all
        self.dsg.radial_bins_all['z'] = radial_bins_all
        self.dsg.radial_res_all['z'] = radial_res_all
        self._jabbeke_combined_scan_files = scan_files

        # Copy onto every other already-present Z-family product key (e.g. 'uz'/'d'/'p'/'k'/'c'), same as
        # ODIM_hdf5.get_scans_information's own copy-to-all-keys step would -- but explicitly excluding
        # 'v'/'w', which come from a separate file and must keep their own structure untouched.
        for p in list(self.dsg.scannumbers_all):
            if p in ('v', 'w'):
                continue
            for attr in gv.volume_attributes_p:
                self.dsg.__dict__[attr][p] = copy.deepcopy(self.dsg.__dict__[attr]['z'])

        return True

    def _fix_z_scan_structure(self):
        """Re-reads Z's own actual file and re-populates the 'z'-keyed scan structure (scanangles_all,
        scannumbers_all, radial_bins_all, radial_res_all) from it. Needed because the normal scan-info read
        always uses the V file first (product='v', to obtain the Nyquist velocity -- see get_scans_information),
        and ODIM_hdf5.get_scans_information always stores whatever it just read under the 'z' key regardless of
        which file that actually was. For Jabbeke and Wideumont, V's file has FEWER elevations than Z's (always
        missing Z's lowest one(s) -- confirmed via the MeteoGate API), so without this correction Z's displayed
        scan angles get silently replaced by V's shorter set. For Helchteren this just re-derives the same
        values (Z and V share the exact same 12 elevations there), a harmless no-op.

        For Jabbeke specifically, this now combines BOTH of its Z files (long+short range) into one
        continuous scan list instead of just re-reading whichever one CTRL+D currently has selected -- see
        _build_jabbeke_combined_z_scans above.

        Called from BOTH get_scans_information AND get_data below -- NOT just get_scans_information -- because
        get_scans_information can be skipped entirely by NLradar's own cross-session volume-attribute cache
        (restore_volume_attributes in nlr_datasourcegeneral.py) whenever a radar/date/time combination has
        already been seen before, e.g. after switching to a different radar and back. get_data has no such
        cache and always runs, so only calling this from there guarantees the correction can't silently stop
        taking effect after a radar switch (confirmed: that's exactly what was happening before this was moved
        here -- switching to another radar and back made Z revert to showing V's angles again).
        """
        saved_nyquist = copy.deepcopy(self.dsg.nyquist_velocities_all_mps)
        saved_nyquist_low = copy.deepcopy(self.dsg.low_nyquist_velocities_all_mps)
        saved_nyquist_high = copy.deepcopy(self.dsg.high_nyquist_velocities_all_mps)
        # ODIM_hdf5.get_scans_information also has a side effect where it copies whatever it just read (labeled
        # 'z' internally) onto EVERY product key already present in self.dsg.scannumbers_all -- including 'v',
        # if that key already exists from elsewhere in the broader read pipeline. Save/restore these too, so
        # this correction can't overwrite V's own, correct structure with Z's in the other direction.
        saved_v_attrs = {attr: copy.deepcopy(self.dsg.__dict__[attr].get('v')) for attr in gv.volume_attributes_p
                          if 'v' in self.dsg.__dict__[attr]}
        try:
            if self.crd.radar == 'Jabbeke':
                if self._build_jabbeke_combined_z_scans():
                    return
                # Neither Z file could be read -- fall through to the normal, single-file behavior below as a
                # last resort, same as for every other radar in this class.
            # NOT using self.filepath('z', ...) here: that function only ever searches self.dsg.files_datetime,
            # which reflects self.crd.directory -- and that can still be pointing at the V directory at this
            # point. Instead, explicitly resolve Z's own directory first (independent of self.crd.directory)
            # and search directly within that, the same way the Z/V-search loops elsewhere already do.
            z_directory = self.dsg.get_directory(self.crd.date, self.crd.time, self.crd.radar, 'Z', dir_index=self._current_z_dir_index())
            z_filepath = None
            if z_directory and os.path.exists(z_directory):
                z_files = os.listdir(z_directory)
                z_matches = [i for i in z_files if i[:12] == self.crd.date+self.crd.time and self.correct_filename(i, 'z')]
                if z_matches:
                    z_filepath = z_directory+'/'+z_matches[0]
            if z_filepath:
                self.dsg.ODIM_hdf5.get_scans_information(z_filepath, 'z')
        except Exception:
            pass # If this fails for any reason, we're no worse off than before this correction existed.
        self.dsg.nyquist_velocities_all_mps = saved_nyquist
        self.dsg.low_nyquist_velocities_all_mps = saved_nyquist_low
        self.dsg.high_nyquist_velocities_all_mps = saved_nyquist_high
        for attr, value in saved_v_attrs.items():
            self.dsg.__dict__[attr]['v'] = value

    def get_scans_information(self):
        #product 'v' is tried initially, because it enables the program to obtain the Nyquist velocities
        #In the case of Zaventem it is however possible that the file that contains the reflectivity has more scans than the file that contains
        #the velocity. In this case it is therefore necessary to take 'z' as product.        
        product = 'v' if self.crd.radar != 'Zaventem' else 'z'
        filepath_hdf, filepath_product = self.filepath(product, source_function=self.get_scans_information)
        # NOTE: the '.vol' check just below (and its twin a few lines down) must test the actual file
        # EXTENSION, not just look for '.vol' as a substring -- Source_Leonardo's own files end in the literal
        # extension '.vol' (confirmed via Source_Leonardo.get_filenames_directory: filenames=[j for j in entries
        # if j[-4:]=='.vol']), whereas KMI/MeteoGate '.h5' files are deliberately named e.g.
        # '...dBZ.vol.h5' (see Source_MeteoGate.get_urls_and_savenames_downloadfile in nlr_currentdata.py --
        # the '.vol' there exists purely so Source_KMI.correct_filename's h5-extension check, filename[16:-3]
        # == pname+'.vol', matches). A plain substring check ('.vol' in i) can't tell these apart, and
        # incorrectly treats normal KMI h5 files as Leonardo data whenever filepath_product happens to differ
        # from the requested product -- which didn't matter for Helchteren before VRADH became unavailable
        # (filepath_product always equaled the requested 'v' then), but now that the 'v' lookup for Helchteren
        # legitimately falls back to a 'z' file instead (a mismatch), this false positive routed everything to
        # source_Leonardo (wrong format), breaking display even though the correct '.h5' file was right there.
        if (not filepath_hdf or filepath_product != product and any(i[-4:] == '.vol' for i in self.dsg.files_datetime)) \
        and self.crd.radar in gv.radars_with_datasets:
            # self.crd.directory's content may not match the dataset (Z or V) this call actually needs. The
            # previous approach here temporarily overwrote the SHARED instance attributes self.crd.directory and
            # self.dsg.files_datetime to probe the other dataset's folder, then restored them afterwards. That is
            # unsafe: those same attributes are read/written by the background auto-download mechanism at any
            # time, completely independently of user action (confirmed: the resulting file-not-found symptom
            # occurred consistently, not tied to animation/looping) -- so a concurrent access during the brief
            # window where these were temporarily repointed could interleave with this function's own logic,
            # leaving filepath_hdf/filepath_product built from a mix of the two dataset's state. This version
            # never mutates shared state at all: it lists each candidate directory directly into a local variable
            # and matches the filename locally, so there is nothing left for a concurrent access to race with.
            #
            # Deliberately NOT skipping a candidate whose directory happens to equal self.crd.directory (unlike
            # the old version): the very fact that we're here means the normal self.filepath() lookup against
            # self.dsg.files_datetime (which may be stale, e.g. from before the background downloader added a
            # new file) just failed for the current directory too -- a fresh, independent directory listing can
            # still succeed even when self.crd.directory is already the theoretically-correct one, which matters
            # in particular when viewing the V dataset itself (self.crd.directory already ends in '_V').
            for candidate_dataset in ('Z', 'V'):
                candidate_directory = self.dsg.get_directory(self.crd.date, self.crd.time, self.crd.radar, candidate_dataset, dir_index=0)
                if not candidate_directory or not os.path.exists(candidate_directory):
                    continue
                try:
                    candidate_files = os.listdir(candidate_directory)
                except Exception:
                    continue
                matches = [i for i in candidate_files if i[:12] == self.crd.date+self.crd.time and self.correct_filename(i, product)]
                if matches:
                    filepath_hdf, filepath_product = candidate_directory+'/'+matches[0], product
                    break
        if not filepath_hdf or filepath_product != product and any(i[-4:] == '.vol' for i in self.dsg.files_datetime):
            return self.dsg.source_Leonardo.get_scans_information()
        else:
            #product is used to determine whether it is possible to obtain the Nyquist velocities (only when product=='v')
            self.dsg.ODIM_hdf5.get_scans_information(filepath_hdf, filepath_product)
            if filepath_product == 'v':
                self._fix_z_scan_structure()
        
    def get_data(self, j): #j is the panel
        filepath, self.crd.using_unfilteredproduct[j], polarization = self.filepath(gv.i_p[self.crd.products[j]], self.crd.productunfiltered[j], 
                                                                                    self.crd.polarization[j])
        if not filepath and self.crd.radar in gv.radars_with_datasets:
            # Same underlying issue as in get_scans_information above -- see that function's comment for the full
            # explanation of why this searches locally (avoiding a race with the background auto-download
            # mechanism) and deliberately does NOT skip a candidate directory just because it equals
            # self.crd.directory (needed so viewing the V dataset itself can still succeed via a fresh listing).
            product_for_search = gv.i_p[self.crd.products[j]]
            for candidate_dataset in ('Z', 'V'):
                candidate_directory = self.dsg.get_directory(self.crd.date, self.crd.time, self.crd.radar, candidate_dataset, dir_index=0)
                if not candidate_directory or not os.path.exists(candidate_directory):
                    continue
                try:
                    candidate_files = os.listdir(candidate_directory)
                except Exception:
                    continue
                matches = [i for i in candidate_files if i[:12] == self.crd.date+self.crd.time and self.correct_filename(i, product_for_search)]
                if matches:
                    filepath = candidate_directory+'/'+matches[0]
                    self.crd.using_unfilteredproduct[j] = self.crd.productunfiltered[j]
                    polarization = self.crd.polarization[j]
                    break
        if self.crd.radar in gv.radars_with_datasets and gv.i_p.get(self.crd.products[j]) == 'z':
            # See _fix_z_scan_structure's docstring for why this must ALSO be called here, not just from
            # get_scans_information: this guarantees the correction runs every time a Z panel is actually
            # displayed, regardless of whether get_scans_information itself got skipped this time due to
            # NLradar's own cross-session attribute cache (confirmed: switching to a different radar and back
            # was enough to make Z silently revert to V's angles again, because that skipped
            # get_scans_information -- and therefore the correction -- entirely on the return visit).
            self._fix_z_scan_structure()
            if self.crd.radar == 'Jabbeke':
                # Jabbeke's Z scans are combined from 2 separate files (long+short range, see
                # _build_jabbeke_combined_z_scans) -- self.dsg.scannumbers_all['z'][scan] is already the
                # correct RAW scan number for whichever of those 2 files this particular combined scan index
                # actually lives in, but 'filepath' above was resolved generically (just 'the Z directory
                # CTRL+D currently has selected'), which is only right for scans that happen to live in that
                # same file. Override it with the specific file this scan actually needs -- and specifically
                # its 'uz' (TH/unfiltered) sibling when SHIFT+U is active, otherwise this would always show
                # the filtered (DBZH) data regardless of that setting.
                combined_entry = self._jabbeke_combined_scan_files.get(self.crd.scans[j])
                if combined_entry:
                    key = 'uz' if self.crd.productunfiltered[j] else 'z'
                    combined_filepath = combined_entry.get(key) or combined_entry.get('z')
                    if combined_filepath:
                        filepath = combined_filepath
                        self.crd.using_unfilteredproduct[j] = key == 'uz' and combined_entry.get('uz') is not None
        if filepath:
            self.crd.using_verticalpolarization[j] = polarization == 'V'
            self.dsg.ODIM_hdf5.get_data(filepath, j)
        else:
            return self.dsg.source_Leonardo.get_data(j)        
        
    def get_data_multiple_scans(self,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):        
        filepath, productunfiltered, polarization = self.filepath(product, productunfiltered, polarization)
        if filepath:
            data, scantimes, volume_starttime, volume_endtime, _ =\
                self.dsg.ODIM_hdf5.get_data_multiple_scans(filepath,product,scans,productunfiltered,polarization,apply_dealiasing,max_range)
        else:
            return self.dsg.source_Leonardo.get_data_multiple_scans(product,scans,productunfiltered,polarization,apply_dealiasing,max_range)
        
        meta = {'using_unfilteredproduct':productunfiltered, 'using_verticalpolarization':polarization == 'V'}
        return data, scantimes, volume_starttime, volume_endtime, meta
                
    
    def get_filenames_directory(self,radar,directory):
        try:
            entries=os.listdir(directory)
        except Exception: entries=[]
        filenames=np.sort(np.array([j for j in entries if os.path.splitext(j)[1][1:] in ('hdf','h5','vol')]))
        return filenames
    
    def get_datetimes_from_files(self,filenames,dtype=str,return_unique_datetimes=True, mode='simple'):
        datetimes=np.array([j[:12] for j in filenames],dtype=dtype)
        return np.unique(datetimes) if return_unique_datetimes else datetimes   
    




class Source_skeyes():
    """Data from skeyes (radar in Zaventem) can be delivered both by skeyes and the KMI, and they use somewhat different formats. The KMI
    uses the same format as one of the formats that they use for their other radars, which implies that the class Source_KMI can handle the data when 
    it is provided by the KMI.
    """
    def __init__(self, gui_class, dsg_class, parent = None):  
        self.gui=gui_class
        self.dsg=dsg_class
        self.crd=self.dsg.crd
        self.dp=self.dsg.dp
        self.pb = self.gui.pb
        
        """For this radar the scans are stored in 3 different files per volume. It is therefore necessary to store for each product and scan
        the index of the file in which that scan is located. This information is stored in self.dsg.scannumbers_all, in the manner explained in the
        class self.dsg.skeyes_hdf5.
        """
        
    
                
    def get_scans_information(self):
        if any(j in self.dsg.files_datetime[0] for j in ('.vol', 'scan_abc.hdf')):
            return self.dsg.source_classes['KMI'].get_scans_information()
                
        filepaths=[opa(os.path.join(self.crd.directory,j)) for j in self.dsg.files_datetime]
        self.dsg.skeyes_hdf5.get_scans_information(filepaths)                
        
    
    def get_data(self, j): #j is the panel
        if any(i in self.dsg.files_datetime[0] for i in ('.vol', 'scan_abc.hdf')):
            return self.dsg.source_classes['KMI'].get_data(j)
        
        """Scans are stored in 3 different files per volume, and file_index contains the index of the file that contains data for the scan
        self.crd.scans[j].
        """        
        file_index=self.dsg.scannumbers_all[gv.i_p[self.crd.products[j]]][self.crd.scans[j]][self.dsg.scannumbers_forduplicates[self.crd.scans[j]]][0]
        filepath=opa(os.path.join(self.crd.directory,self.dsg.files_datetime[file_index]))
        
        self.dsg.skeyes_hdf5.get_data(filepath, j)

    
    def get_data_multiple_scans(self,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        if any(j in self.dsg.files_datetime[0] for j in ('.vol', 'scan_abc.hdf')):
            return self.dsg.source_classes['KMI'].get_data_multiple_scans(product,scans,productunfiltered,polarization,apply_dealiasing,max_range)
        
        filepaths=[opa(os.path.join(self.crd.directory,j)) for j in self.dsg.files_datetime]
        return self.dsg.skeyes_hdf5.get_data_multiple_scans(filepaths,product,scans,productunfiltered,polarization,apply_dealiasing,max_range)


    def get_filenames_directory(self,radar,directory):        
        try:
            entries=os.listdir(directory)
        except Exception: entries=[]
        if any(any(j in i for j in ('.vol', 'scan_abc.hdf')) for i in entries):
            return self.dsg.source_classes['KMI'].get_filenames_directory(radar, directory)
        
        filenames=np.sort(np.array([j for j in entries if os.path.splitext(j)[1][1:] == 'h5']))
        
        #The code below is specifically for the h5 files delivered by skeyes, where some issues with repeated content must be addressed.
        #Cases in which hdf and h5 files are put in the same folder are not permitted!            
        
        startdatetimes=[j[4:16] for j in filenames]
        unique_startdatetimes=np.unique(startdatetimes)
        if len(startdatetimes)==len(unique_startdatetimes): return filenames
        
        """If not, then it is necessary to filter the files, in order to prevent that the same content is shown more than once by NLradar. 
        The first thing to check, is whether some files are simply 'repeated', in which case subsequent files have the same startdatetime, but a different file ID (the number 
        after RAW). This case has been observed, and can be recognized by the fact that both files have the same size. The code below indentifies these files, and removes
        one of them from the list with filenames. 
        If the resulting list of filenames contains only unique startdatetimes, then this list is returned. If not, then another method is used to deal with another case that
        leads to repeated startdatetimes.
        """
        rm_indices=[]
        for j in range(1,len(filenames)):
            if startdatetimes[j-1]==startdatetimes[j]:
                size1=os.path.getsize(opa(directory+'/'+filenames[j-1]))
                size2=os.path.getsize(opa(directory+'/'+filenames[j]))
                if size1==size2:
                    rm_indices.append(j)
                    
        filenames_filtered=np.delete(filenames,rm_indices)
        startdatetimes_filtered=[j[4:16] for j in filenames_filtered]
        unique_startdatetimes_filtered=np.unique(startdatetimes_filtered)
        if len(startdatetimes_filtered)==len(unique_startdatetimes_filtered): return filenames_filtered  
                        
        """For data from Zaventem from before 2013, the data is provided both in the format from after 2012 (usually 3 files per volume),
        and per scan separately. The resulting files are all put in the same folder, whereas this program needs only the files that have
        the first format. The other files are therefore removed from the list. The files that should remain on the list can be recognized
        based on their start time, which is equal to that of the next or previous file. Of 2 subsequent files with the same start time, 
        the one that should be taken is the one with the largest size. 
        In the current cases that I have seen, the file that should be taken is the first of a pair of files. Because I am not sure about
        whether this is generally true, I check which file should be taken by calculating the file sizes. This is done only for the first
        pair of files (for computational reasons), thus it is assumed that the order of volume and single scan file at least doesn't change
        during the course of 1 day.
        
        It is important to realize that the case in which only a few files have the same startdatetime (and
        thus only a few volumes), is not treated correctly in the current situation.
        """
        filenames_filtered=[]
        desired_file=None
        for j in range(1,len(filenames)):
            if startdatetimes[j-1]==startdatetimes[j]:
                if desired_file is None:
                    size1=os.path.getsize(opa(directory+'/'+filenames[j-1]))
                    size2=os.path.getsize(opa(directory+'/'+filenames[j]))
                    desired_file=-1 if size1>size2 else 0
                    #Take the largest file
                filenames_filtered.append(filenames[j+desired_file])
        filenames_filtered=np.array(filenames_filtered)
        
        return filenames_filtered
    
    def get_datetimes_from_files(self,filenames,dtype=str,return_unique_datetimes=True, mode='simple'):
        if any(any(j in i for j in ('.vol', 'scan_abc.hdf')) for i in filenames):
            return self.dsg.source_classes['KMI'].get_datetimes_from_files(filenames, dtype, return_unique_datetimes, mode)
        
        """For the skeyes hdf5 format there are multiple files (usually 3) per volume, such that the 
        datetimes determined here (per volume) are not simply the datetimes for all files, but the datetimes
        of the first files of all volumes. 
        Further, because the filename does not contain information about which file belongs to which volume, this
        must be determined in another way. It is here done by determining the absolute time for each datetime
        given in the filename, and then grouping files into one volume when the the absolute times differ by
        less than 120 seconds from those of the previous file in the volume.
        There is no fixed timestep between volumes (at least not after 2014), making it impossible to determine
        which file belongs to which volume based on solely the datetime.
        
        If the current method does not work, then a possible other method would be to add a particular file to the
        'current' volume, if the time to the last file in the current volume is smaller than the time to the next file
        after the currently regarded file. It would be added to a new volume if this condition is not satisfied.
        """
        datetimes=[]
        datetimes_h5=[]; seconds_h5=[]
        for j in filenames:
            datetimes_h5.append('20'+j[4:14])
            seconds_h5.append(int(j[14:16]))
        datetimes=np.array(datetimes)
        datetimes_h5=np.array(datetimes_h5); seconds_h5=np.array(seconds_h5)
        
        abstimes_h5=ft.get_absolutetimes_from_datetimes(datetimes_h5)+seconds_h5
        
        n_h5=len(datetimes_h5)
        nfiles_lastindex=0
        if n_h5>0:
            indices=[0]
            last_abstime=abstimes_h5[0]
            for j in range(1,n_h5+1):
                #The iteration for j=n_h5 is only included to apply the check for nfiles_lastindex==1 also to the last volume.
                if j<n_h5 and abstimes_h5[j]<last_abstime+120:
                    nfiles_lastindex+=1
                    indices.append(indices[-1])
                else:
                    if nfiles_lastindex==1:
                        #If a volume consists of just one file according to the above division of files into volume, then this
                        #file/volume is added to the nearest volume, except when the time difference exceeds 7.5 minutes.
                        #Adding is done to prevent the presence of very small volumes (containing only a few scans).
                        diff1=np.abs(abstimes_h5[indices[-1]]-abstimes_h5[indices[-2]])
                        diff2=diff1 if j==n_h5 else np.abs(abstimes_h5[indices[-1]]-abstimes_h5[j])
                        if np.min([diff1,diff2])<450:
                            if j==n_h5 or diff1<diff2:
                                indices[-1]=indices[-2]
                            else:
                                indices[-1]=j  
                                
                    if j<n_h5:
                        nfiles_lastindex=1
                        indices.append(j)
                if j<n_h5:
                    last_abstime=abstimes_h5[j]
                    
            datetimes=np.sort(np.append(datetimes,datetimes_h5[indices]))
            
        datetimes=datetimes.astype(dtype)
            
        return np.unique(datetimes) if return_unique_datetimes else datetimes   
    
    
    
    

class Source_DWD():
    def __init__(self, gui_class, dsg_class, parent = None):  
        self.gui = gui_class
        self.dsg = dsg_class
        self.crd = self.dsg.crd
        self.dp = self.dsg.dp
        self.pb = self.gui.pb
        self.import_classes = {'buf.bz2': self.dsg.DWD_bufr, 'buf': self.dsg.DWD_bufr, 'hd5': self.dsg.DWD_odimh5}
        self._warned_unrecognized_products = set() # Tracks (radar, product_str) combinations already warned about,
        #to avoid repeatedly warning about the same unrecognized product identifier.
        
        
        
    def get_extension(self, filename=None):
        filename = self.dsg.files_datetime[0] if not filename else filename
        return ('buf.' if '.bz2' in filename else '')+filename[-3:]
        
    def get_file_availability_info(self):     
        if self.get_extension() == 'hd5':
            files_pvs, pvs = self.files_product_versions_datetimesdict[self.crd.date+self.crd.time], self.dsg.product_versions_datetime
            filenames_per_pv = {pv:[] for pv in pvs}
            files_exclude = []
            product_files = {p:[] for p in gv.products_all}
            for i,f in enumerate(self.dsg.files_datetime):
                product_str = f.split('_')[-2]

                # DWD-HD5: TV is only used by the explicit TV reflectivity route.
                # Do not register tv as normal velocity here, otherwise V/SRV can load TV.
                # TH (the H-pol counterpart, used for unfiltered Z via SHIFT+U) is excluded here for the same
                # reason: both are handled separately elsewhere, not via the normal per-product matching below,
                # so they are not unrecognized products and shouldn't trigger the warning just below.
                if product_str in ('tv', 'th'):
                    files_exclude.append(f)
                    continue

                matches = [k for k,v in gv.productnames_DWD['hd5'].items() if product_str in ([v] if type(v) is str else v)]
                if not matches:
                    # This product identifier is not in gv.productnames_DWD['hd5'] (and is not 'th'/'tv', handled
                    # above), meaning this file is silently ignored below. Could be an unsupported product, but
                    # could also mean the DWD started providing a new product. Warn once per radar+identifier.
                    warn_key = (self.crd.radar, product_str)
                    if not warn_key in self._warned_unrecognized_products:
                        self._warned_unrecognized_products.add(warn_key)
                        msg = (f"onbekende DWD-productidentifier '{product_str}' aangetroffen voor radar {self.crd.radar} "
                               "(bijbehorende bestanden worden genegeerd). Mogelijk biedt de DWD een nieuw product aan "
                               "dat nog niet wordt ondersteund.")
                        print("NLradar: "+msg)
                        gv.log_product_check("WAARSCHUWING - DWD - "+msg)
                    files_exclude.append(f)
                    continue

                # It's possible that multiple versions are available for a single product, e.g. rhohv and urhohv for CC. In that
                # case show the first product version in gv.productnames_DWD['hd5'] that is available. The check for product_files
                # below ensures that no file for another product version gets added to the list of available files.
                if f.replace(product_str, '') in product_files[matches[0]]:
                    files_exclude.append(f)
                    continue
                product = matches[0]
                product_files[product].append(f.replace(product_str, '')) # Exclude the product string but keep the file id
                for pv in pvs:
                    if files_pvs[i] == pv or not product in self.dsg.products_version_dependent:
                        filenames_per_pv[pv].append(f)
            
            gv.log_product_check(f"OK - DWD - check uitgevoerd voor radar {self.crd.radar}, geen onbekende producten gevonden.")
            
            desired_pv = self.gui.radardata_product_versions[self.dsg.radar_dataset]
            pv = desired_pv if desired_pv in filenames_per_pv else pvs[0]
            if pv == 'combi':
                # In this case data for the 2 different product versions should be combined in nlr_importdata.py
                filenames = [f for f in self.dsg.files_datetime if not f in files_exclude]
            else:
                filenames = filenames_per_pv[pv]
        else:
            filenames = self.dsg.files_datetime
                
        self.products_per_fileid, self.fileids_per_product, self.files_per_product_per_fileid = {}, {}, {}
        for f in filenames:
            product_str =  f.split('_')[-2 if self.get_extension() == 'hd5' else 2]

            # DWD-HD5: TV is handled separately in get_data() when Z + Shift+U + V-pol is requested.
            # It must not become part of the normal velocity file lists.
            if self.get_extension() == 'hd5' and product_str == 'tv':
                continue

            product = [k for k,v in gv.productnames_DWD[self.get_extension()].items() if product_str in ([v] if type(v) is str else v)][0]
            product_str = '_'+product_str+'_'
            idx = f.index(product_str)+len(product_str)
            fileid = int(f[idx:f.index('-20')]) if '-20' in f else float(f[idx+15:-11])
                        
            ft.init_dict_entries_if_absent(self.fileids_per_product, product, list)
            ft.init_dict_entries_if_absent(self.products_per_fileid, fileid, list)
            ft.create_subdicts_if_absent(self.files_per_product_per_fileid, [product, fileid], type_last_entry=list)
                
            self.products_per_fileid[fileid].append(product)
            self.fileids_per_product[product].append(fileid)
            self.files_per_product_per_fileid[product][fileid].append(f)

        if self.get_extension() == 'hd5' and 'p' in self.fileids_per_product:
            # DWD does not publish a native KDP file. KDP is instead derived from PHIDP, so here 'k' is
            # aliased onto exactly the same underlying files as 'p' -- the actual KDP retrieval (smoothing
            # + differentiation of PHIDP) happens in nlr_importdata.py (DWD_odimh5.read_data), which checks
            # for i_p == 'k' and computes it from the PHIDP data it reads out of these same files.
            self.fileids_per_product['k'] = list(self.fileids_per_product['p'])
            self.files_per_product_per_fileid['k'] = {fid: list(files) for fid, files in
                                                        self.files_per_product_per_fileid['p'].items()}
            for fid in self.fileids_per_product['k']:
                if 'k' not in self.products_per_fileid[fid]:
                    self.products_per_fileid[fid].append('k')

    def get_scans_information(self):
        self.get_file_availability_info()
    
        filepaths = {}; products = {}
        for fileid in self.products_per_fileid:
            #If present, then use for each fileid the file that contains the velocity, because otherwise the Nyquist velocity cannot be
            #determined.
            products[fileid] = 'v' if 'v' in self.products_per_fileid[fileid] else self.products_per_fileid[fileid][0]
            filepaths[fileid] = opa(self.crd.directory+'/'+self.files_per_product_per_fileid[products[fileid]][fileid][0])
            
        self.import_classes[self.get_extension()].get_scans_information(filepaths, products, self.fileids_per_product)
                        
        
    def get_data(self, j): #j is the panel
        self.get_file_availability_info()

        requested_product = self.crd.products[j]
        extension = self.get_extension()

        # DWD HD5 reflectiviteit:
        # - Z zonder Shift+U gebruikt normale DBZH.
        # - Shift+U + Z gebruikt TH bij H-pol en TV bij V-pol.
        # Beide worden als product 'z' ingelezen, zodat de gewone reflectivity-kleurtabel actief blijft.
        if extension == 'hd5' and self.crd.productunfiltered[j] and requested_product == 'z':
            i_p = 'z'
            fileid = self.dsg.scannumbers_all[i_p][self.crd.scans[j]][0]
            marker = '_tv_' if self.crd.polarization[j] == 'V' else '_th_'
            refl_files = []

            for f in self.dsg.files_datetime:
                if marker not in f:
                    continue
                try:
                    idx = f.index(marker) + len(marker)
                    fid = int(f[idx:f.index('-20')]) if '-20' in f else float(f[idx+15:-11])
                except Exception:
                    continue
                if fid == fileid:
                    refl_files.append(f)

            if refl_files:
                filepaths = [opa(os.path.join(self.crd.directory, f)) for f in refl_files]
                self.import_classes[extension].get_data(filepaths, j, 'z')
                self.crd.using_unfilteredproduct[j] = True
                self.crd.using_verticalpolarization[j] = self.crd.polarization[j] == 'V'
                return

            raise Exception('Product not available')

        # DWD HD5: Shift+U heeft geen effect op andere producten dan Z.
        # Voor oude/niet-HD5 formaten blijft de bestaande uV-logica behouden.
        if extension != 'hd5' and self.crd.productunfiltered[j] and requested_product == 'v':
            requested_product = 'uv'

        i_p = gv.i_p.get(requested_product, requested_product)

        # SRV ('s') gebruikt velocity ('v'). Bij DWD HD5 nooit naar uV/TV schakelen.
        file_product = requested_product

        if requested_product == 's':
            file_product = 'uv' if extension != 'hd5' and self.crd.productunfiltered[j] else 'v'

        if file_product in self.fileids_per_product:
            fileid = self.dsg.scannumbers_all[i_p][self.crd.scans[j]][0]

            if extension == 'hd5':
                files = self.files_per_product_per_fileid[file_product][fileid]

                # DWD-HD5: normal velocity/SRV must use VRADH only.
                # TV is loaded only by the explicit TV reflectivity route above.
                if file_product == 'v':
                    vradh_files = [i for i in files if '_vradh_' in i.lower()]
                    if vradh_files:
                        files = vradh_files

                filepaths = [opa(os.path.join(self.crd.directory, i)) for i in files]
                self.import_classes[extension].get_data(filepaths, j, file_product)
            else:
                filepath = opa(os.path.join(
                    self.crd.directory,
                    self.files_per_product_per_fileid[file_product][fileid][0]
                ))
                self.import_classes[extension].get_data(filepath, j)

            self.crd.using_unfilteredproduct[j] = requested_product in ('uz', 'uv') and extension != 'hd5'
            self.crd.using_verticalpolarization[j] = False

        else:
            raise Exception('Product not available')

    def get_data_multiple_scans(self,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        self.get_file_availability_info()
        extension = self.get_extension()
        if extension == 'hd5':
            filepaths = {i: [opa(os.path.join(self.crd.directory, k)) for k in j] for i,j in self.files_per_product_per_fileid[product].items()}
            return self.import_classes[extension].get_data_multiple_scans(filepaths,product,scans,productunfiltered,polarization,apply_dealiasing,max_range)
        else:
            filepaths = {i: opa(os.path.join(self.crd.directory, j[0])) for i,j in self.files_per_product_per_fileid[product].items()}
            return self.import_classes[extension].get_data_multiple_scans(filepaths,product,scans,productunfiltered,polarization,apply_dealiasing,max_range)


    def get_product_versions(self, filenames, datetimes):
        if self.get_extension(filenames[0]) == 'hd5':
            files_product_versions = np.array([j[:j.index('_sweeph5onem')] for j in filenames])
    
            self.files_product_versions_datetimesdict, product_versions_datetimesdict = {}, {}
            for j in np.unique(datetimes):
                self.files_product_versions_datetimesdict[j] = files_product_versions[datetimes == j]
                product_versions_datetimesdict[j] = list(np.unique(self.files_product_versions_datetimesdict[j]))
                if len(product_versions_datetimesdict[j]) > 1:
                    # Add a product version for a combination of the 2 individual product versions
                    product_versions_datetimesdict[j].append('combi')
                    
            products_version_dependent = ['z', 'v'] # i.e. all products available for DWD
            product_versions_in1file = False
            return product_versions_datetimesdict, products_version_dependent, product_versions_in1file
        else:
            return None, None, None
 
    def get_filenames_directory(self,radar,directory):
        try:
            entries=os.listdir(directory)
        except Exception: entries=[]
        return np.sort([j for j in entries if any(j.endswith(i) for i in self.import_classes)])
    
    def get_datetimes_from_files(self,filenames,dtype=str,return_unique_datetimes=True, mode='simple'):
        #Flooring to 5 minutes is performed, because the DWD puts data for each scan separately in files, with a corresponding range of datetimes, 
        #while all these files belong to the same radar volume, that starts at the datetime to which a datetime below gets floored.
        # Historical DWD BUFR files might have a different naming compared to files made available on DWD's open data server, 
        # in which case the datetime also needs to be determined in a different way. Also note that just indexing on '20'
        # doesn't work, because '20' can be a fileid
        datetimes=np.array([int(np.floor(int(j[j.index('-20')+1: j.index('-20')+13] if '-20' in j\
                                             else j[j.index('20'): j.index('20')+12])/5)*5) for j in filenames],dtype=dtype)
        return np.unique(datetimes) if return_unique_datetimes else datetimes   
    
    
    
    
class Source_TUDelft():
    def __init__(self, gui_class, dsg_class, parent = None):  
        self.gui = gui_class
        self.dsg = dsg_class
        self.crd = self.dsg.crd
        self.dp = self.dsg.dp
        self.pb = self.gui.pb
        
            
    def filepath(self):
        return self.crd.directory+'/'+self.dsg.files_datetime[0]   
          
    def get_scans_information(self):
        return self.dsg.TUDelft_nc.get_scans_information(self.filepath())
                
    def get_data(self, j): #j is the panel
        return self.dsg.TUDelft_nc.get_data(self.filepath(), j)
                
    def get_data_multiple_scans(self,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        return self.dsg.TUDelft_nc.get_data_multiple_scans(self.filepath(),product,scans,productunfiltered,polarization,apply_dealiasing,max_range)
                
    def get_filenames_directory(self, radar, directory):
        try:
            entries=os.listdir(directory)
        except Exception: entries=[]
        filenames=np.sort(np.array([j for j in entries if j[-3:]=='.nc']))
        return filenames
                
    def get_datetimes_from_files(self,filenames,dtype=str,return_unique_datetimes=True, mode='simple'):
        """This function actually returns both datetimes and dates instead of just datetimes. This is done because there is one file per date, and
        the dates are also used in nlr_datasourcegeneral.py.
        """
        dates = []
        for filename in filenames:
            dates += [ft.format_date(filename[5:15], 'YYYY-MM-DD->YYYYMMDD')]
        dt = gv.volume_timestep_radars[self.crd.radar]
        times = [ft.time_to_minutes(j, inverse = True) for j in range(0, 1440, dt)]
        try:
            datetimes = np.concatenate([[j+i for i in times] for j in dates]).astype(dtype)
        except Exception: # happens when no dates are available
            datetimes = []
        return (datetimes, dates) if mode == 'dates' else datetimes
    
    
    
    
    
class Source_Leonardo():
    def __init__(self, gui_class, dsg_class, parent = None):  
        self.gui = gui_class
        self.dsg = dsg_class
        self.crd = self.dsg.crd
        self.dp = self.dsg.dp
        self.pb = self.gui.pb
        
        self.vol_classes = {'rainbow3':self.dsg.Leonardo_vol_rainbow3,'rainbow5':self.dsg.Leonardo_vol_rainbow5}
        
                
    def filepath(self, product, productunfiltered=False, polarization='H', source_function=None):
        product = 'u'+product if productunfiltered else product
        pol_suffix = 'v' if polarization == 'V' else ''
        try:
            # First try to obtain a filename for the correct product
            if type(gv.productnames_Leonardo[product]) is str:
                filename = [i for i in self.dsg.files_datetime if i[16:-4] == gv.productnames_Leonardo[product]+pol_suffix][0]
            else: # type list
                filename = [i for i in self.dsg.files_datetime if any(i[16:-4] == j+pol_suffix for j in gv.productnames_Leonardo[product])][0]
        except Exception:
            if productunfiltered and polarization == 'V':
                return self.filepath(product[-1], False, polarization)
            elif polarization == 'V':
                return self.filepath(product[-1], productunfiltered, 'H')
            elif productunfiltered:
                return self.filepath(product[-1], False, polarization)
            filename = ''
            if source_function == self.get_scans_information and len(self.dsg.files_datetime):
                #Try to find any file with the correct date and time, and if found, then determine the product contained in it
                filename = self.dsg.files_datetime[0]
                product = 'z' #The default product. 'z' does not need to be present, but the only information required for determining whether it is possible to 
                #obtain the Nyquist velocities is that the product is not equal to 'v'.
        filepath = self.crd.directory+'/'+filename if filename else ''
        return [filepath, product] if source_function == self.get_scans_information else [filepath, productunfiltered, polarization]
    
    def get_vol_class(self,filepath):
        with open(filepath,'rb') as vol:
            line = vol.read(10).decode('utf-8')
            vol.seek(0)
            vol_type = 'rainbow5' if line[0] == '<' else 'rainbow3'
            return self.vol_classes[vol_type]
    
        
    def get_scans_information(self):
        filepath, product = self.filepath('v', source_function=self.get_scans_information)
        return self.get_vol_class(filepath).get_scans_information(filepath, product)
        
    def get_data(self, j): #j is the panel    
        filepath, self.crd.using_unfilteredproduct[j], polarization = self.filepath(gv.i_p[self.crd.products[j]], self.crd.productunfiltered[j], 
                                                                                    self.crd.polarization[j])
        # The following lines are currently disabled, since they cause issues when looping through cases
        # """For old Wideumont data it is the case that the V-dataset does not contain reflectivities, such that you
        # need to switch from dataset to view reflectivities. If self.crd.products[j] == 'v', then without the following procedure it would
        # be the case that you need to switch manually to 'z' to view reflectivity, whereas with this procedure you will automatically show
        # the reflectivity for the Z-dataset (at least when no velocity is available for that dataset, as is the case with old Wideumont data).
        # """
        # changing_radardataset = self.dsg.changing_radar or self.dsg.changing_dataset
        # if not filepath and changing_radardataset:
        #     product = 'z' if self.crd.products[j] == 'v' else 'v'
        #     filepath, self.crd.using_unfilteredproduct[j], polarization = self.filepath(gv.i_p[product], self.crd.productunfiltered[j], 
        #                                                                             self.crd.polarization[j])
        #     if filepath:
        #         self.crd.products[j] = product

        self.crd.using_verticalpolarization[j] = polarization == 'V'
        return self.get_vol_class(filepath).get_data(filepath, j)
        
    def get_data_multiple_scans(self,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        filepath, productunfiltered, polarization = self.filepath(product, productunfiltered, polarization)
        data, scantimes, volume_starttime, volume_endtime =\
            self.get_vol_class(filepath).get_data_multiple_scans(filepath,product,scans,productunfiltered,polarization,apply_dealiasing,max_range)
        meta = {'using_unfilteredproduct':productunfiltered, 'using_verticalpolarization':polarization == 'V'}
        return data, scantimes, volume_starttime, volume_endtime, meta
        
    def get_filenames_directory(self, radar, directory):
        try:
            entries=os.listdir(directory)
        except Exception: entries=[]
        filenames=np.sort(np.array([j for j in entries if j[-4:]=='.vol']))
        return filenames
        
        
    def get_datetimes_from_files(self,filenames,dtype=str,return_unique_datetimes=True, mode='simple'):
        datetimes=np.array([int(os.path.basename(j)[:12]) for j in filenames],dtype=dtype)
        return np.unique(datetimes) if return_unique_datetimes else datetimes   
    
    



    
class Source_DMI():
    def __init__(self, gui_class, dsg_class, parent = None):  
        self.gui=gui_class
        self.dsg=dsg_class
        self.crd=self.dsg.crd
        self.dp=self.dsg.dp
        self.pb = self.gui.pb
        
        
        
    def filepath(self):
        return self.crd.directory+'/'+self.dsg.files_datetime[0]
    
    def get_scans_information(self):
        self.dsg.ODIM_hdf5.get_scans_information(self.filepath(), 'v')

    def get_data(self, j): #j is the panel
        self.dsg.ODIM_hdf5.get_data(self.filepath(),j)

    def get_data_multiple_scans(self,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        return self.dsg.ODIM_hdf5.get_data_multiple_scans(self.filepath(),product,scans,productunfiltered,polarization,apply_dealiasing,max_range) 
                    
    def get_filenames_directory(self,radar,directory):
        try:
            entries=os.listdir(directory)
        except Exception: entries=[]
        filenames=np.sort([j for j in entries if j.split('.')[-1] in ('h5', 'hdf') and gv.radar_ids[radar] in j])
        return filenames
    
    def get_datetimes_from_files(self,filenames,dtype=str,return_unique_datetimes=True, mode='simple'):
        i = filenames[0].index(self.dsg.get_datetimes_from_files_dirdate)
        datetimes=np.array([j[i:i+12] for j in filenames],dtype=dtype)
        return datetimes
  
    
  
    
  
class Source_SHMU():
    def __init__(self, gui_class, dsg_class, parent = None):  
        self.gui = gui_class
        self.dsg = dsg_class
        self.crd = self.dsg.crd
        self.dp = self.dsg.dp
        self.pb = self.gui.pb
        
        self.classes = {'vol':self.dsg.source_Leonardo, 'hdf':self.dsg.source_CHMI}
        
        
         
    def extension(self):
        return 'vol' if '.vol' in self.dsg.files_datetime[0] else 'hdf'
    
    def filepath(self, product):
        return self.classes[self.extension()].filepath(product)
    
    def get_scans_information(self):
        self.classes[self.extension()].get_scans_information()

    def get_data(self, j): #j is the panel
        self.classes[self.extension()].get_data(j)

    def get_data_multiple_scans(self,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        return self.classes[self.extension()].get_data_multiple_scans(product,scans,productunfiltered,polarization,apply_dealiasing,max_range) 
                    
    def get_filenames_directory(self,radar,directory):
        try:
            entries=os.listdir(directory)
        except Exception: entries=[]
        filenames=np.sort([j for j in entries if j.split('.')[-1] in ('h5', 'hdf', 'vol')])
        return filenames
    
    def get_datetimes_from_files(self,filenames,dtype=str,return_unique_datetimes=True, mode='simple'):
        i = filenames[0].index(self.dsg.get_datetimes_from_files_dirdate)
        datetimes=np.array([j[i:i+12] for j in filenames],dtype=dtype)
        return datetimes
    
    
    
    
    
class Source_AustroControl():
    def __init__(self, gui_class, dsg_class, parent = None):  
        self.gui=gui_class
        self.dsg=dsg_class
        self.crd=self.dsg.crd
        self.dp=self.dsg.dp
        self.pb = self.gui.pb
        
        
        
    def filepath(self, product, source_function=None):
        if 'PPIVol' in self.dsg.files_datetime[0]:
            filename = self.dsg.files_datetime[0]
            product = 'v'
        else:        
            filename = ''
            try:
                filename = [j for j in self.dsg.files_datetime if gv.productnames_AustroControl[product] in j][0]
            except Exception:
                if source_function == self.get_scans_information and len(self.dsg.files_datetime):
                    #Try to find any file with the correct date and time, and if found, then determine the product contained in it
                    filename = self.dsg.files_datetime[0]
                    product = 'z' #The default product. 'z' does not need to be present, but the only information required for determining whether it is possible to 
                    # obtain the Nyquist velocities is that the product is not equal to 'v'.
        filepath = self.crd.directory+'/'+filename if filename else ''
        return (filepath, product) if source_function == self.get_scans_information else filepath
    
    def get_scans_information(self):
        self.dsg.ODIM_hdf5.get_scans_information(*self.filepath('v', self.get_scans_information))

    def get_data(self, j): #j is the panel
        self.dsg.ODIM_hdf5.get_data(self.filepath(self.crd.products[j]),j)

    def get_data_multiple_scans(self,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        return self.dsg.ODIM_hdf5.get_data_multiple_scans(self.filepath(product),product,scans,productunfiltered,polarization,apply_dealiasing,max_range) 
                    
    def get_filenames_directory(self,radar,directory):
        try:
            entries=os.listdir(directory)
        except Exception: entries=[]
        filenames = np.sort([j for j in entries if j[-3:]=='hdf'])
        return filenames
    
    def get_datetimes_from_files(self,filenames,dtype=str,return_unique_datetimes=True, mode='simple'):
        if 'PPIVol' in filenames[0]:
            return np.array([''.join(j.split('-')[-2:])[:12] for j in filenames], dtype=dtype)
        else:
            return np.array([j[-14:-4].replace('-', '')+j[14:18] for j in filenames], dtype=dtype)





class Source_CHMI():
    def __init__(self, gui_class, dsg_class, parent = None):  
        self.gui=gui_class
        self.dsg=dsg_class
        self.crd=self.dsg.crd
        self.dp=self.dsg.dp
        self.pb = self.gui.pb
        
        
        
    def filepath(self, product=None):
        if self.crd.radar in ('Milešovka', 'Holesov'):
            return self.crd.directory+'/'+self.dsg.files_datetime[0]
        else:
            source = gv.data_sources[self.crd.radar].replace(' ', '')
            p_names = eval(f'gv.productnames_{source}')
            p = product if product[0] != 'u'  or product in p_names else product[1]
            filenames = [f for f in self.dsg.files_datetime if p_names.get(p, '') in f]
            # It's possible that the productname for one product is contained in the productname for another product, leading
            # to multiple hits. Selecting the filename with the shortest length leads to the right file.
            i = np.argmin([len(f) for f in filenames]) if filenames else None
            return self.crd.directory+'/'+filenames[i] if filenames else None
    
    def get_scans_information(self):
        self.dsg.ODIM_hdf5.get_scans_information(self.filepath('v'), 'v')

    def get_data(self, j): #j is the panel
        product = 'uz' if self.crd.products[j] == 'z' and self.crd.productunfiltered[j] else self.crd.products[j]
        self.dsg.ODIM_hdf5.get_data(self.filepath(product), j)
        self.crd.using_unfilteredproduct[j] = product == 'uz'

    def get_data_multiple_scans(self,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        return self.dsg.ODIM_hdf5.get_data_multiple_scans(self.filepath(product),product,scans,productunfiltered,polarization,apply_dealiasing,max_range) 
                    
    def get_filenames_directory(self,radar,directory):
        try:
            entries=os.listdir(directory)
        except Exception: entries=[]
        filenames = np.sort(np.array([j for j in entries if '.h' in j[-5:]]))
        return filenames
    
    def get_datetimes_from_files(self,filenames,dtype=str,return_unique_datetimes=True, mode='simple'):
        i = re.sub('[T_]', '', filenames[0]).index(self.dsg.get_datetimes_from_files_dirdate)
        datetimes=np.array([re.sub('[T_]', '', j)[i:i+12] for j in filenames],dtype=dtype)
        return datetimes
        
    
    
    

class Source_MeteoFrance():
    def __init__(self, gui_class, dsg_class, parent = None):
        self.gui=gui_class
        self.dsg=dsg_class
        self.crd=self.dsg.crd
        self.dp=self.dsg.dp
        self.pb = self.gui.pb
        
        self.import_classes = {'buf':self.dsg.MeteoFrance_BUFR, 'nc':self.dsg.MeteoFrance_NetCDF}


        
    def get_file_availability_info(self):
        filenames = self.dsg.files_datetime

        if filenames[0].endswith('.nc'):
            self.file_per_filetype = {}
            for f in filenames:
                filetype = 'PAG' if 'PAG' in f else 'PAM'
                self.file_per_filetype[filetype] = self.crd.directory+'/'+f
        else: # BUFR files
            self.filetypes_per_fileid, self.fileids_per_filetype, self.file_per_filetype_per_fileid = {}, {}, {}
            for f in filenames:
                filetype = 'PAG' if 'PAG' in f else 'PAM'
                fileid = f[f.index(filetype)+3]
                            
                ft.init_dict_entries_if_absent(self.fileids_per_filetype, filetype, list)
                ft.init_dict_entries_if_absent(self.filetypes_per_fileid, fileid, list)
                ft.init_dict_entries_if_absent(self.file_per_filetype_per_fileid, filetype, dict)
                    
                self.filetypes_per_fileid[fileid].append(filetype)
                self.fileids_per_filetype[filetype].append(fileid)
                self.file_per_filetype_per_fileid[filetype][fileid] = self.crd.directory+'/'+f
            
    def get_scans_information(self):
        self.get_file_availability_info()
        
        if self.dsg.files_datetime[0].endswith('.nc'):
            self.dsg.MeteoFrance_NetCDF.get_scans_information(self.file_per_filetype)
        else:
            filepaths = {}
            for fileid in self.filetypes_per_fileid:
                filetype = self.filetypes_per_fileid[fileid][0]
                filepaths[fileid] = self.file_per_filetype_per_fileid[filetype][fileid]
                
            self.dsg.MeteoFrance_BUFR.get_scans_information(filepaths, self.filetypes_per_fileid)

    def get_data(self, j): #j is the panel
        self.get_file_availability_info()
    
        i_p, scan = gv.i_p[self.crd.products[j]], self.crd.scans[j]
        if self.dsg.files_datetime[0].endswith('.nc'):
            filetype = self.dsg.scannumbers_all[i_p][scan][self.dsg.scannumbers_forduplicates[self.crd.scans[j]]].split(',')[0]            
            self.dsg.MeteoFrance_NetCDF.get_data(self.file_per_filetype[filetype], j)    
        else:   
            filetype, fileid = self.dsg.scannumbers_all[i_p][scan][self.dsg.scannumbers_forduplicates[self.crd.scans[j]]].split(',')[:2]
            filepath = self.file_per_filetype_per_fileid[filetype][fileid]
            self.dsg.MeteoFrance_BUFR.get_data(filepath, j)

    def get_data_multiple_scans(self,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        self.get_file_availability_info()
        
        i_p = gv.i_p[product]
        if self.dsg.files_datetime[0].endswith('.nc'):
            filetype = self.dsg.scannumbers_all[i_p][scans[0]][0].split(',')[0]
            filepath = self.file_per_filetype[filetype]
            return self.dsg.MeteoFrance_NetCDF.get_data_multiple_scans(filepath,product,scans,productunfiltered,polarization,apply_dealiasing,max_range)    
        else: 
            filepaths = {}
            for j in scans:
                filetype, fileid = self.dsg.scannumbers_all[i_p][j][0].split(',')[:2]
                filepaths[j] = self.file_per_filetype_per_fileid[filetype][fileid]
            return self.dsg.MeteoFrance_BUFR.get_data_multiple_scans(filepaths,product,scans,productunfiltered,polarization,apply_dealiasing,max_range) 
                    
    def get_filenames_directory(self, radar, directory):
        try:
            entries = os.listdir(directory)
        except Exception: entries = []
        filenames = np.sort([j for j in entries if any(i in j for i in ('PAM', 'PAG'))])
        if len(filenames) and filenames[0].endswith('.nc'):
            # NC files contain data for a time period of 15 minutes. I find it desired however to split this period up in volumes of 
            # 5 minutes here in the code
            filenames = np.repeat(filenames, 3)
        return filenames
    
    def get_datetimes_from_files(self, filenames, dtype=str, return_unique_datetimes=True, mode='simple'):
        if not len(filenames):
            return []
        if filenames[0].endswith('.nc'):
            datetimes = [j.split('_')[-1][:12] for j in filenames]
            for j in range(len(datetimes)):
                # Programmatically set different datetimes for different repetitions of the same filename, in order to split a 15-minute spanning
                # file up in multiple radar volumes
                datetimes[j] = ft.next_datetime(datetimes[j], 5*(j%3))
        elif filenames[0].startswith('T_'):
            datetimes = [j[16:28] for j in filenames]
        else:
            # No date is given in the filename, it is therefore determined in self.dsg.get_datetimes_from_files
            datetimes = [self.dsg.get_datetimes_from_files_dirdate+j[12:16] for j in filenames]
        if not filenames[0].endswith('.nc'):
            # -5 minutes, since end instead of start datetimes are given in the filenames
            datetimes = np.array([ft.next_datetime(j, -5) for j in datetimes], dtype=dtype)
        return np.unique(datetimes) if return_unique_datetimes else datetimes





class Source_UKMO():
    def __init__(self, gui_class, dsg_class, parent = None):
        self.gui=gui_class
        self.dsg=dsg_class
        self.crd=self.dsg.crd
        self.dp=self.dsg.dp
        self.pb = self.gui.pb


    def get_scans_information(self):
        filepaths = [self.crd.directory+'/'+j for j in self.dsg.files_datetime]
        self.dsg.UKMO_polar.get_scans_information(filepaths)
        
    def get_data(self, j):
        i_p, scan = gv.i_p[self.crd.products[j]], self.crd.scans[j]
        file_idx = self.dsg.scannumbers_all[i_p][scan][self.dsg.scannumbers_forduplicates[scan]]
        filepath = self.crd.directory+'/'+self.dsg.files_datetime[file_idx]
        self.dsg.UKMO_polar.get_data(filepath, j)

    def get_data_multiple_scans(self,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        i_p = gv.i_p[product]
        filepaths = {}
        for j in scans:
            file_idx = self.dsg.scannumbers_all[i_p][j][0]
            filepaths[j] = opa(self.crd.directory+'/'+self.dsg.files_datetime[file_idx])
        return self.dsg.UKMO_polar.get_data_multiple_scans(filepaths,product,scans,productunfiltered,polarization,apply_dealiasing,max_range) 
        
    def get_filenames_directory(self, radar, directory):
        try:
            entries = os.listdir(directory)
        except Exception: entries = []
        filenames = np.sort([j for j in entries if j[-7:] == '.dat.gz'])
        return filenames
    
    def get_datetimes_from_files(self, filenames, dtype=str, return_unique_datetimes=True, mode='simple'): 
        i = filenames[0].index('_raw')
        # el_numbers = np.array([int(j[-8]) for j in filenames])
        # filenames_el0 = filenames[el_numbers == 0]
        # datetimes_el0 = np.array([j[i-12:i] for j in filenames_el0])
        # delta_T = np.median(np.diff(ft.get_absolutetimes_from_datetimes(datetimes_el0)))/60  
        
        # The Z dataset actually has a timestep of 5 minutes, but flooring times to 5 minutes doesn't
        # work for separating the 2 volumes within a 10-minute window. So it's decided to floor to 10 minutes,
        # meaning that each 'volume' will actually contain 2 volumes, with each scan contained twice (duplicates).
        delta_T = 10
        datetimes = np.array([ft.floor_datetime(j[i-12:i], delta_T) for j in filenames])
        return np.unique(datetimes) if return_unique_datetimes else datetimes
                



    
class Source_NWS():
    def __init__(self, gui_class, dsg_class, parent = None):  
        self.gui=gui_class
        self.dsg=dsg_class
        self.crd=self.dsg.crd
        self.dp=self.dsg.dp
        self.pb = self.gui.pb
        
        self.import_classes = {2:self.dsg.NEXRAD_L2, 3:self.dsg.NEXRAD_L3}
        self.fileid_pos = {'N':2, 'T':3}
        # For NEXRAD 'Z' contains long-range low-res reflectivity, 'R' short-range higher-res
        # For TDWR 'Z' contains super-res reflectivity, 'R' legacy-res
        # In case of list the labels are listed in order of decreasing preference
        self.product_labels_l3 = {'z':{'N':['R','Z'], 'T':['Z','R']}, 'v':'V'}
        
        
    
    def get_level(self, filename=None):
        if not filename:
            filename = self.dsg.files_datetime[0]
        return 3 if not filename[5].isdigit() else 2
    def get_l3_type(self, filename=None):
        if not filename:
            filename = self.dsg.files_datetime[0]
        return filename[12] # Either 'N' for NEXRAD or 'T' for TDWR
    def l3_scandescr(self, l3_type, p_label, fileid):
        descr = f'_{l3_type}{p_label}'
        pos = self.fileid_pos[l3_type]
        return descr[:pos]+f'{fileid}'+descr[pos:]
    def get_product_labels(self, l3_type, product):
        labels = self.product_labels_l3[product]
        if type(labels) == dict:
            labels = labels[l3_type]
        return labels #can be either a 1-character string or a list with > 1 element. Both are iterable, so no need to put string in list
    
    def import_class(self):
        level = self.get_level()
        return self.dsg.NEXRAD_L2 if level == 2 else self.dsg.NEXRAD_L3
        
    def filepaths(self, product=None, scan=None, duplicate=None):
        level = self.get_level()
        if level == 2:
            return self.crd.directory+'/'+self.dsg.files_datetime[0]
        else:
            files = self.dsg.files_datetime
            l3_type = self.get_l3_type()
            fileids = np.array([j[j.index(f'_{l3_type}')+self.fileid_pos[l3_type]:][0] for j in files])
            filedatetimes = np.array([j[-12:] for j in files])
            indices = np.argsort([fileids[i]+filedatetimes[i] for i in range(len(files))])
            files, fileids = files[indices], fileids[indices]
            _files = {}
            for i,file in enumerate(files):
                fileid = fileids[i]
                fileproduct = [p for p,labels in self.product_labels_l3.items() if any(self.l3_scandescr(l3_type, j, fileid) in file for j in 
                                                                                       self.get_product_labels(l3_type, p))][0]
                ft.init_dict_entries_if_absent(_files, fileid, dict)
                ft.init_dict_entries_if_absent(_files[fileid], fileproduct, list)
                _files[fileid][fileproduct].append(self.crd.directory+'/'+file)
            if scan:
                # see NEXRAD_L3.get_scans_information for how scannumbers_all is set up
                fileid = self.dsg.scannumbers_all[product][scan][0][0]
                fileindex = self.dsg.scannumbers_all[product][scan][duplicate][1]
                return _files[fileid][product][fileindex]
            elif product:
                return {fnum:fdict[product] for fnum,fdict in _files.items()}
            else:
                return _files
        
    def get_scans_information(self):
        self.import_class().get_scans_information(self.filepaths())

    def get_data(self, j): #j is the panel
        i_p, scan = gv.i_p[self.crd.products[j]], self.crd.scans[j]
        filepath = self.filepaths(i_p, scan, self.dsg.scannumbers_forduplicates[scan])
        self.import_class().get_data(filepath, j)

    def get_data_multiple_scans(self,product,scans,productunfiltered=False,polarization='H',apply_dealiasing=True,max_range=None):
        i_p = gv.i_p[product]
        return self.import_class().get_data_multiple_scans(self.filepaths(i_p),product,scans,productunfiltered,polarization,apply_dealiasing,max_range) 
                

    def get_product_versions(self, filenames, datetimes):
        fname = filenames[0]
        if self.get_level(fname) == 2:
            radar = fname[:4]
            if radar[0] == 'T': # No different product versions for TDWR
                return None, None, None
            i = fname.find('_V')
            file_V_version = -1 if i == -1 else int(fname[i+2:i+4])
            # In a small examination I found that volumes with reflectivity also available for V-scans, have a V version of 3 or 6,
            # hence the following choice. Real-time volumes (bzip2-compressed) from Iowa State don't have the V version in the filename however. 
            # But since currently files always have reflectivity for V-scans, it is then assumed that it's present.
            v_scan_present = file_V_version % 3 == 0 or not fname[-3:] == '.gz'
            product_versions = ['z_scan']+['v_scan', 'combi_scan']*v_scan_present
            product_versions_datetimesdict = {j:product_versions for j in datetimes} 
            products_version_dependent = ['z'] # Only 'z' has 2 versions available
            product_versions_in1file = True
            return product_versions_datetimesdict, products_version_dependent, product_versions_in1file
        else:
            return None, None, None
    
    def get_filenames_directory(self, radar, directory):
        try:
            entries = os.listdir(directory)
            level = self.get_level(entries[0])
            if level == 3:
                # There are many kinds of L3 files. Here only those that contain the desired products are kept
                entries = [j for j in entries if 'SDUS' in j]
                l3_type = self.get_l3_type(entries[0])
                _entries = []
                for p in self.product_labels_l3:
                    plabels = self.get_product_labels(l3_type, p)
                    for plabel in plabels:
                        hits = [j for j in entries if j[12] == l3_type and
                                ((j[13].isdigit() and j[14] == plabel) or (j[14].isdigit() and j[13] == plabel))]
                        if hits:
                            _entries += hits
                            break # Use only files for the first plabel for which files are available
                entries = _entries
        except Exception: 
            entries = []
        filenames = np.sort(entries)
        return filenames
    
    def get_datetimes_from_files(self, filenames, dtype=str, return_unique_datetimes=True, mode='simple'):
        try:
            level = self.get_level(filenames[0])
            if level == 2:
                # Files downloaded from https://mesonet-nexrad.agron.iastate.edu contain an extra '_' between the radar ID and datetime
                datetimes = np.array([j[4:12]+j[13:17] if not j[4] == '_' else j[5:13]+j[14:18] for j in filenames], dtype=dtype)
            else:
                datetimes = np.array([ft.floor_datetime(j[-12:], 6) for j in filenames], dtype=dtype)
        except Exception:
            level = 0
            datetimes = np.array([], dtype=dtype)
        return np.unique(datetimes) if level == 3 and return_unique_datetimes else datetimes   
    




class Source_ARRC():
    def __init__(self, gui_class, dsg_class, parent = None):  
        self.gui=gui_class
        self.dsg=dsg_class
        self.crd=self.dsg.crd
        self.dp=self.dsg.dp
        self.pb = self.gui.pb
        
    def import_class(self):
        return self.dsg.CFRadial if self.dsg.files_datetime[0].endswith('.nc') else self.dsg.DORADE
    
    def get_scannumbers(self, filenames):
        if filenames[0].endswith('.nc'):        
            return np.array([format(int(j[j.rindex('_s')+2:-3]), '02d') for j in filenames])
        else:
            scannumbers = [j[:j.rindex('.')+2] for j in filenames]
            return np.array([j[-4:] if j[-5] == '.' else '0'+j[-3:] for j in scannumbers])
        
    def get_filedatetimes(self, filenames):
        if filenames[0].endswith('.nc'):
            return np.array([j[6:14]+j[15:21] for j in map(os.path.basename, filenames)])
        else:
            return np.array([('20' if int(j[5:7]) < 50 else '19')+j[5:17] for j in map(os.path.basename, filenames)])

    def filepaths(self, scan=None, duplicate=None):
        files = self.dsg.files_datetime
        scannumbers = self.get_scannumbers(files)
        filedatetimes = self.get_filedatetimes(files)
        indices = np.argsort([scannumbers[i]+filedatetimes[i] for i in range(len(files))])
        files, scannumbers = files[indices], scannumbers[indices]
        _files = {}
        for i,file in enumerate(files):
            scannumber = scannumbers[i]
            ft.init_dict_entries_if_absent(_files, scannumber, list)
            _files[scannumber].append(self.crd.directory+'/'+file)
        if scan:
            scannumber = self.dsg.scannumbers_all['z'][scan][0]
            return _files[scannumber][duplicate]
        else:
            return _files
        
    def get_scans_information(self):
        self.import_class().get_scans_information(self.filepaths())

    def get_data(self, j): #j is the panel
        filepath = self.filepaths(self.crd.scans[j], self.dsg.scannumbers_forduplicates[self.crd.scans[j]])
        self.import_class().get_data(filepath, j)
        
    def get_filenames_directory(self, radar, directory):
        filenames = []
        for root,dirs,files in os.walk(directory, topdown=True):
            folder = root.replace('\\', '/')[len(directory):].lstrip('/')
            filenames += [folder+'/'*(len(folder) > 0)+file for file in files]
        filenames = np.array([j for j in filenames if os.path.basename(j).startswith('swp.') or j.endswith('.nc')])
        print(filenames)
        scannumbers = self.get_scannumbers(filenames)
        filedatetimes = self.get_filedatetimes(filenames)
        indices = np.argsort([filedatetimes[i]+num for i,num in enumerate(scannumbers)])
        # print(list(filenames[indices]))
        return filenames[indices]
    
    def get_datetimes_from_files(self, filenames, dtype=str, return_unique_datetimes=True, mode='simple'):
        # print(filenames)
        scannumbers = self.get_scannumbers(filenames)
        datetimes = []
        for i, num in enumerate(scannumbers):
            file = os.path.basename(filenames[i])
            datetime = file[3:15] if file.endswith('.nc') else\
                       ('20' if int(file[5:7]) < 50 else '19')+file[5:15]
            if i == 0 or int(scannumbers[i-1].replace('.', ''))-int(num.replace('.', '')) > 5 or\
            ft.datetimediff_s(datetime, datetimes[-1]) > 600:
                volume_datetime = datetime
            datetimes.append(volume_datetime)
        # for i in range(len(filenames)):
        #     print(i, filenames[i], datetimes[i])
        return np.array(datetimes, dtype=dtype)