# Copyright (C) 2016-2024 Bram van 't Veen, bramvtveen94@hotmail.com
# Distributed under the GNU General Public License version 3, see <https://www.gnu.org/licenses/>.

from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt, QTimer, QObject, pyqtSignal

import vispy
vispy.set_log_level(verbose='warning')
from vispy import app
from vispy import color
from vispy import visuals
from vispy import gloo
from vispy.visuals.transforms import STTransform, ChainTransform
from vispy.visuals.filters import Clipper, ColorFilter

from OpenGL import GL

import numpy as np
from numpy import genfromtxt
import os
import re
opa=os.path.abspath
import time as pytime
import copy
import traceback

import nlr_background as bg
import nlr_customvispy as cv
import nlr_functions as ft
import nlr_globalvars as gv
import nlr_hclass as hc
import nlr_maptiles as mt
import nlr_maptiles_maptiler as mtt
from VWP.nlr_plottingvwp import PlottingVWP

# --- MONKEYPATCH voor vispy 0.14.1 (upgrade juli 2026, alleen vispy zelf bijgewerkt, Python 3.8 blijft) ---
# Bug in vispy zelf: wanneer een gedeelde view van een visual wordt aangemaakt via .view() (zoals
# hieronder bij self.visuals['map'][j] = self.visuals['map'][0].view()), roept vispy's basisklasse
# Visual.__init__ intern DIRECT _prepare_transforms(view) aan tijdens het aanmaken van de view -- dus
# VOORDAT de subclass-specifieke _init_view(view)-hook de kans heeft gehad om per-view instellingen te
# zetten (bv. ImageVisual._method_used). Resultaat: AttributeError ('VisualView' object has no
# attribute '_method_used').
#
# Generieke fix: elke VisualView/CompoundVisualView houdt een verwijzing (self._visual) bij naar het
# origineel. Als een ontbrekend attribuut wordt opgevraagd op de view, pakken we het gewoon van het
# origineel i.p.v. te crashen. Zodra de echte subclass-specifieke _init_view() daarna alsnog draait,
# wordt de eigen, correcte waarde op de view gezet -- deze fallback voorkomt puur de premature crash
# tijdens view()-constructie en heeft geen enkel effect op de uiteindelijke rendering.
from vispy.visuals.visual import BaseVisualView as _BaseVisualView
_orig_base_getattr = getattr(_BaseVisualView, '__getattr__', None)
def _patched_base_getattr(self, name):
    visual = self.__dict__.get('_visual')
    if visual is not None and name != '_visual':
        try:
            return getattr(visual, name)
        except AttributeError:
            pass
    if _orig_base_getattr is not None:
        return _orig_base_getattr(self, name)
    raise AttributeError(name)
_BaseVisualView.__getattr__ = _patched_base_getattr
# --- EINDE MONKEYPATCH ---



"""This class handles plotting of the radar panels and associated color bars and titles. 
Plotting of other things such as the VWP hodographs happens in separate classes, although some necessary actions such as those required to make room for these other
things happen also in this class. The same holds for initiating the plotting calls for these other things.
"""



class Plotting(QObject,app.Canvas):
    setback_gridheightrings_signal=pyqtSignal()
    update_map_tiles_signal=pyqtSignal()
    def __init__(self, gui_class, empty_init=False):
        if empty_init:
            # Is used to initialise this class without getting issues below with references to other classes that at this point don't exist yet.
            return
        super(Plotting, self).__init__()
        self.start_dpi = gui_class.screen_DPI()
        self.start_screen_size = gui_class.screen_size()
        app.Canvas.__init__(self, dpi=self.start_dpi) #It is important to set DPI manually, since at Linux vispy
        #is not able to determine it correctly automatically!
        self.gui=gui_class
        
        self.setback_gridheightrings_signal.connect(self.setback_gridheightrings)
        self.update_map_tiles_signal.connect(self.update_map_tiles)
        
        self.gui=gui_class
        self.crd=self.gui.crd
        self.ani=self.gui.ani
        self.dsg=self.crd.dsg
        self.dp=self.dsg.dp
        self.vwp = PlottingVWP(gui_class=self.gui, pb_class = self)
        self.mt = mt.MapTiles(self)
        self.mt_maptiler = mtt.MapTilesMapTiler(self)
        
        #Startup settings
        self.wdims = np.array([self.gui.dimensions_main['width'], self.gui.dimensions_main['height']])
        #First value gives x dimension of the 'left' and 'right' widgets. Second value gives y dimension of the 'top' and 'bottom' widgets.
        #In cm.
        
        self.panels=1; self.panellist=(0,)
        self.max_panels=10
        self.nrows=1; self.ncolumns=1
        self.panel=0       
        
        self.data_empty={j:True for j in range(self.max_panels)} #True when self.dsg.data[j] contains no radar data for panel j
        self.data_isold={j:True for j in range(self.max_panels)} #True when self.dsg.data[j] contains data that is not for the current time
        # Tracks, per (visual type, panel), whether the image visual's GLSL color-transform was last built for RGB
        # passthrough data (product 'g') or for scalar+colormap data (every other product). vispy's ImageVisual only
        # rebuilds this function when its cmap *property* is reassigned, which we deliberately skip for 'g'. So when
        # switching a panel into or out of 'g', we need to detect that here and force the rebuild ourselves (see
        # set_newdata).
        self.visual_colortransform_is_rgb={'radar_polar':{j:False for j in range(self.max_panels)},
                                            'radar_cartesian':{j:False for j in range(self.max_panels)}}
        # Parameters that are valid for the data that is currently shown in the panels. Since a panel might not get updated always, 
        # it is necessary to store these parameters and to not use current values.
        # 'xy_bins', 'res' replace 'radial_bins','radial_res','azimuthal_bins','azimuthal_res' when a plain product is displayed in 
        # cartesian coordinates
        self.data_attr = {j:{} for j in ('product','scanangle','radial_bins','radial_res','azimuthal_bins','azimuthal_res',
                                          'xy_bins', 'res', 'proj', 'scantime', 'scandatetime')}
        self.data_attr_before = copy.deepcopy(self.data_attr)
        
        self.heightrings_scanangles = {j:-1 for j in range(self.max_panels)}
        self.products_before = self.crd.products.copy(); self.scans_before=self.crd.scans.copy()  
        self.productunits=gv.productunits_default.copy()
        self.scale_factors={j: 1. for j in gv.products_all}
        self.firstdraw_map=True #It is important to draw the map for all panels during the first draw, because otherwise the scale gets screwed up when changing the
        #number of panels.
        self.changing_panels=False
        self.datareadout_text=''
        self.use_interpolation=False
        self.firstplot_performed=False
        self.max_delta_time_move_view_with_storm=12*3600
        self.mouse_hold=False
        self.mouse_hold_left=False
        self.mouse_hold_right=False
        self.last_mouse_pos_px=None
        self.gridheightrings_removed=False
        self.radar_mouse_selected=None
        # A/B line tool state (see set_ab_line_point / draw_ab_line): ab_line_panel is the panel the line was
        # drawn in, ab_line_a/ab_line_b are its two endpoints in AEQD x/y coordinates (km from the radar), with
        # ab_line_b staying None until the second Shift+click completes the line.
        self.ab_line_panel=None
        self.ab_line_a=None
        self.ab_line_b=None
        self.volume3d_rect_panel=None
        self.volume3d_rect_a=None
        self.volume3d_rect_b=None
        self.volume3d_rect_dragging=False

        # Shared cross-section marker (see set_xsection_marker_frac/update_xsection_marker): a single position,
        # expressed as a fraction along the A/B line (0=A, 1=B), driving BOTH a point on the map's A-B line AND
        # a vertical line at the matching distance on every currently-open cross-section plot at once, so the
        # two stay in lockstep as the marker is dragged on either side. self.cross_section_layout holds the
        # screen-pixel image bounds of each panel's currently-open cross-section (set in show_cross_section,
        # cleared in hide_cross_section), needed both to draw the plot-side marker and to hit-test clicks/drags
        # inside the cross-section image. xsection_marker_dragging is None while idle, or a small dict
        # identifying which side ('plot' or 'map', plus the panel for 'plot') the drag started on -- see
        # get_xsection_marker_click_frac, on_mouse_press/on_mouse_move/on_mouse_release.
        self.xsection_marker_frac=None
        self.cross_section_layout={}
        self.xsection_marker_dragging=None
        self.cross_section_active_panels=set()
        self.marker_mouse_selected_index = None
        self.in_view_mask_specs=None
        self.timer_setback_gridheightrings_running=False
        self.timer_update_map_tiles_running = False
        self.postpone_plotting_gridheightrings=False
        self.panels_horizontal_ghtext=list(range(self.max_panels)); self.panels_vertical_ghtext=[0,5]
        self.lines_order=['provinces','countries','rivers','grid','heightrings'] #Order in which lines are drawn, from bottom to top
        self.radarcoords_xy = np.array(ft.aeqd(gv.radarcoords[self.crd.radar], np.array([gv.radarcoords[j] for j in gv.radars_all])))
        self.zoomfactor_vispy=0.007
                
        self.cm1={}; self.cm2={}; 
        self.data_values_colors={}; self.data_values_colors_int={}
        self.data_values_ticks={}; self.tick_map = {}
        self.mask_values={}; self.mask_values_int={}; self.clim_int={}
        self.cmap_lastmodification_time={}
        self.cmaps_minvalues_before=self.gui.cmaps_minvalues.copy(); self.cmaps_maxvalues_before=self.gui.cmaps_maxvalues.copy()
        self.cbars_products_before=[]
        
        self.starting=True
        self.set_draw_action('updating_cbars')
        self.update_map_tiles_ondraw = False
        self.start_scissor_test=True
        
        self.base_range = 250. #The base/default panel y range (distance panel center to top of panel), is used on start-up and when resetting panel view
        #The fractional width of the VWP plot relative to that of the whole canvas. Gets updated on resizing in self.calculate_vwp_relxdim
        self.vwp_relxdim = 0.255
        
        self.unitcircle_vertices=cv.generate_vertices_circle([0,0],1,0,360,100)[:-1]


        #Information about the positions of panels. plotnumber increases from 1 to the total number of panels when going from the upperleft corner to the lowerright corner (zigzagging). 
        #For panelnumber the presence of 8 panels is assumed, and the number corresponds to the position in a 2x4 grid. An exception is the presence of 2 panels, where the second panel (panel 1, it starts at 0)
        #is assigned a panelnumber of 4 (meaning 5th panel), although it should have panelnumber 2 according to the above rule. The reason for this exception is that it works nicer when changing the number of panels.
        self.rows_panels={1:{0:0},2:{0:0,1:0},3:{0:0,1:0,2:0},4:{0:0,1:0,2:1,3:1},6:{0:0,1:0,2:0,3:1,4:1,5:1},8:{0:0,1:0,2:0,3:0,4:1,5:1,6:1,7:1},10:{0:0,1:0,2:0,3:0,4:0,5:1,6:1,7:1,8:1,9:1}}
        self.cols_panels={1:{0:0},2:{0:0,1:1},3:{0:0,1:1,2:2},4:{0:0,1:1,2:0,3:1},6:{0:0,1:1,2:2,3:0,4:1,5:2},8:{0:0,1:1,2:2,3:3,4:0,5:1,6:2,7:3},10:{0:0,1:1,2:2,3:3,4:4,5:0,6:1,7:2,8:3,9:4}}
        self.plotnumber_to_panelnumber={1:{0:0},2:{0:0,1:5},3:{0:0,1:1,2:2},4:{0:0,1:1,2:5,3:6},6:{0:0,1:1,2:2,3:5,4:6,5:7},8:{0:0,1:1,2:2,3:3,4:5,5:6,6:7,7:8},10:{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9}}
        self.panelnumber_to_plotnumber={}
        for j in self.plotnumber_to_panelnumber: 
            self.panelnumber_to_plotnumber[j]={v: k for k, v in self.plotnumber_to_panelnumber[j].items()}
            
        self.determine_panellist_nrows_ncolumns()
        self.set_widget_sizes_and_bounds()
        
        self.set_cmaps(gv.products_all)
        

        #Visuals are drawn in the order in which they are defined here
        self.visuals_order=['background','background_map','map','radar_polar','radar_cartesian','map_lines','gh_lines','text_hor1','text_hor2','text_vert1','text_vert2']
        self.visuals_order+=['sm_pos_markers','radar_markers','panel_borders','titles']
        self.visuals_order+=['cbar'+str(j) for j in range(10)]+['cbars_ticks','cbars_reflines','cbars_labels']
        self.visuals_order+=['polrgb_legend_bar'+str(j)+ch for j in range(self.max_panels) for ch in ('r','g','b')]
        self.visuals_order+=['polrgb_legend_ticks','polrgb_legend_labels','polrgb_legend_reflines']
        self.visuals_order+=['hclass_legend_markers','hclass_legend_labels']
        self.visuals_order+=['ab_line','ab_line_labels','ab_line_marker']
        # NIEUW (23 juli, op Eriks verzoek): laat een puntje achter op de plek waar de laatste
        # HCLASS/MESH-pop-up (zie show_extra_info_popup) is opgeroepen, zodat je nog kunt zien
        # waar je precies had geklikt nadat de pop-up weer weg is. Rechtsklik verbergt het weer
        # (zie on_mouse_release).
        self.visuals_order+=['click_marker']
        self.visuals_order+=['volume3d_rect','volume3d_rect_labels']
        self.cross_section_interpolation_modes = ('nearest', 'bilinear', 'bicubic') #The 3 pre-built variants
        #-- see the loop further down that creates one ImageVisual per mode per panel, and cross_section_
        #visual_key/show_cross_section/hide_cross_section, which switch between them by toggling .visible
        #rather than creating/mutating a visual at runtime (found to render solid black, see
        #change_cross_section_interpolation_mode in nlr.py for the history).
        cross_section_visual_names = ['cross_section_'+m for m in self.cross_section_interpolation_modes]
        self.visuals_order+=['cross_section_background']+cross_section_visual_names+['cross_section_axislines','cross_section_ticks','cross_section_marker','cross_section_back_frame','cross_section_back','cross_section_title','cross_section_frame','cross_section_toggle_button_frame','cross_section_toggle_button']
        self.visuals_panels=['map','radar_polar','radar_cartesian','map_lines','gh_lines','text_hor1','text_hor2','text_vert1','text_vert2','sm_pos_markers','radar_markers']+cross_section_visual_names+['cross_section_axislines','cross_section_ticks','cross_section_marker','cross_section_back','cross_section_back_frame','cross_section_title','cross_section_background','cross_section_frame','cross_section_toggle_button','cross_section_toggle_button_frame']
        #Visuals that are created for each panel separately
        self.visuals_global=[j for j in self.visuals_order if not j in self.visuals_panels] #Visuals that are not created for each panel separately
        
        self.visuals={j:({} if j in self.visuals_panels else None) for j in self.visuals_order}
        
        #self.visuals_widgets lists for each widget the visuals that are located in it. One visual could be located in multiple widgets, but this should only
        #be done when these widgets are always updated simultaneously, or when the widgets number of objects that is displayed is small.
        self.widgets=['left','right','bottom','top','main']
        self.visuals_widgets={}
        self.visuals_widgets['left']=['cbar'+str(j) for j in range(5)]+['cbars_ticks','cbars_reflines','cbars_labels']
        self.visuals_widgets['right']=['cbar'+str(j) for j in range(5,10)]+self.visuals_widgets['left'][-3:]
        self.visuals_widgets['bottom']=self.visuals_widgets['top']=['titles']
        polrgb_legend_bar_names = ['polrgb_legend_bar'+str(j)+ch for j in range(self.max_panels) for ch in ('r','g','b')]
        self.visuals_widgets['main']=['background_map']+self.visuals_panels+polrgb_legend_bar_names+['polrgb_legend_ticks','polrgb_legend_labels','polrgb_legend_reflines']+['hclass_legend_markers','hclass_legend_labels']+['ab_line','ab_line_labels','ab_line_marker']+['click_marker']+['volume3d_rect','volume3d_rect_labels']+['panel_borders']
        
        self.font_sizes = {'text_hor1':'self.gui.gridheightrings_fontsize', 'text_vert1':'self.gui.gridheightrings_fontsize', 
                           'text_hor2':'self.gui.gridheightrings_fontsize', 'text_vert2':'self.gui.gridheightrings_fontsize',
                           'titles':"self.gui.fontsizes_main['titles']", 'cbars_ticks':"self.gui.fontsizes_main['cbars_ticks']", 
                           'cbars_labels':"self.gui.fontsizes_main['cbars_labels']",
                           'polrgb_legend_ticks':"self.gui.fontsizes_main['cbars_ticks']*1.5",
                           'polrgb_legend_labels':"self.gui.fontsizes_main['cbars_labels']*1.4",
                           'hclass_legend_labels':"self.gui.fontsizes_main['cbars_labels']*1.2",
                           'ab_line_labels':"self.gui.fontsizes_main['cbars_labels']*1.4",
                           'volume3d_rect_labels':"self.gui.fontsizes_main['cbars_labels']*1.4",
                           'cross_section_ticks':"self.gui.fontsizes_main['cbars_ticks']*1.5",
                           'cross_section_back':"self.gui.fontsizes_main['cbars_labels']",
                           'cross_section_toggle_button':"self.gui.fontsizes_main['cbars_labels']",
                           'cross_section_title':"self.gui.fontsizes_main['cbars_labels']*1.6"}
                
        
        # set self.map_data and self.map_bounds
        self.update_map_tiles(xy_bounds = [-300, 300, -300, 300], separate_thread=False)
        self.shapefiles_latlon_combined, self.shapefiles_connect_combined = bg.import_shapefiles()
                
        self.lines_pos_combined={}; self.lines_connect_combined={}; self.lines_colors_combined={}
        for i in range(self.max_panels):
            self.lines_pos_combined[i]={}; self.lines_connect_combined[i]={}; self.lines_colors_combined[i]={}
            for j in self.gui.lines_names:
                self.lines_pos_combined[i][j]=[]
                self.lines_connect_combined[i][j]=[] #The last number of a sub-array of connect should always be zero, to prevent different lines from getting connected.
                self.lines_colors_combined[i][j]=[]
        self.update_combined_lineproperties(range(self.max_panels),changing_radar=True,start=True)
        
        self.ghtext_hor_pos_combined={}; self.ghtext_hor_strings_combined={}; self.ghtext_vert_pos_combined={}; self.ghtext_vert_strings_combined={}
        self.heights_text={}; self.heights_text_pos={}
        for i in range(self.max_panels):
            self.ghtext_hor_pos_combined[i]={}; self.ghtext_hor_strings_combined[i]={}; self.ghtext_vert_pos_combined[i]={}; self.ghtext_vert_strings_combined[i]={}
            for j in self.gui.ghtext_names:
                self.ghtext_hor_pos_combined[i][j]={}
                self.ghtext_hor_strings_combined[i][j]={}
                self.ghtext_vert_pos_combined[i][j]={}
                self.ghtext_vert_strings_combined[i][j]={}      
                
                
            
        self.visuals['background']=visuals.RectangleVisual(center=[0,0],color=self.gui.bgcolor/255)
        self.visuals['background'].transform=STTransform()
        
        self.visuals['background_map']=visuals.RectangleVisual(center=[0,0],color=self.gui.bgmapcolor/255.)
        self.visuals['background_map'].transform=STTransform(scale=self.wsize['main'],translate=self.wcenter['main'])
        
        self.map_transforms = {}
        self.map_initial_bounds = None; self.map_initial_scale = None
        self.ref_radial_bins = {}; self.ref_azimuthal_bins = {}
        self.polar_transforms = {}
        self.polar_transforms_individual = {'scanangle':{}, 'scale':{}, 'polar':{}}
        self.cartesian_transforms = {}
        self.cartesian_transforms_individual = {'scale_translate':{}}
        
        startup_string=''.join([str(j) for j in range(10)]) #Initialize all TextVisuals with startup_string, to shorten the time that is needed to perform the
        #first plot.
        
        self.map_colorfilter = ColorFilter(self.gui.mapcolorfilter)
        self.radardata_colorfilter = ColorFilter(self.gui.radardata_colorfilter)
        self.text_hor_top_colorfilter = ColorFilter(np.append(self.gui.gridheightrings_fontcolor['top']/255., 1.))
        self.text_hor_bottom_colorfilter = ColorFilter(np.append(self.gui.gridheightrings_fontcolor['bottom']/255., 1.))
        
        (scale_x, scale_y), (t_x, t_y) = self.get_map_sttransform_parameters()
        self.map_transforms['st'] = STTransform(scale=(scale_x,scale_y), translate=(t_x, t_y))
        self.map_transforms['aeqd'] = cv.LatLon_to_Azimuthal_Equidistant_Transform(self.crd.radar)
        for j in range(self.max_panels):
            if j==0:
                self.visuals['map'][j]=visuals.ImageVisual(self.map_data,interpolation='bilinear',method='impostor')
            else:
                #Create a view of the map in the first panel, such that the map data is stored only once in memory.
                #Using views of visuals is possible when all panels show the same visual.
                self.visuals['map'][j] = self.visuals['map'][0].view()
            self.visuals['map'][j].transform=STTransform(scale=(1,-1))*self.map_transforms['aeqd']*self.map_transforms['st']
            self.visuals['map'][j].attach(self.map_colorfilter)
            self.visuals['map'][j].visible = self.gui.mapvisibility
            
            self.visuals['radar_polar'][j] = visuals.ImageVisual(method='auto', cmap=self.cm1[self.crd.products[j]], clim=self.clim_int[self.crd.products[j]])
            self.visuals['radar_cartesian'][j] = visuals.ImageVisual(method='auto', cmap=self.cm1[self.crd.products[j]], clim=self.clim_int[self.crd.products[j]])
            self.visuals['radar_polar'][j].attach(self.radardata_colorfilter)
            self.visuals['radar_cartesian'][j].attach(self.radardata_colorfilter)
            self.polar_transforms_individual['scanangle'][j]=cv.Slantrange_to_Groundrange_Transform()
            self.polar_transforms_individual['scale'][j]=STTransform()
            self.polar_transforms_individual['polar'][j]=cv.PolarTransform()
            self.cartesian_transforms_individual['scale_translate'][j] = STTransform()
            self.visuals['radar_polar'][j].transform = self.polar_transforms_individual['polar'][j] * self.polar_transforms_individual['scanangle'][j] * self.polar_transforms_individual['scale'][j]
            self.visuals['radar_cartesian'][j].transform = self.cartesian_transforms_individual['scale_translate'][j]
            
            if j==0:
                self.visuals['map_lines'][j]=visuals.LineVisual(pos=None,connect=None,color=None,method='gl',antialias=self.gui.lines_antialias)
            else:
                self.visuals['map_lines'][j] = self.visuals['map_lines'][0].view()
            self.visuals['map_lines'][j].transform = STTransform(scale=(1,-1))*self.map_transforms['aeqd']
            self.visuals['gh_lines'][j] = visuals.LineVisual(pos=None,connect=None,color=None,method='gl',antialias=self.gui.lines_antialias)
            
            self.visuals['text_hor1'][j] = visuals.TextVisual(text=startup_string,pos=[-1e6,-1e6],color=np.ones(4),font_size=eval(self.font_sizes['text_hor1']),face='OpenSans-Bold',anchor_x='center',anchor_y='center')
            self.visuals['text_hor2'][j] = self.visuals['text_hor1'][j].view()
            self.visuals['text_hor1'][j].attach(self.text_hor_bottom_colorfilter, view=self.visuals['text_hor1'][j])
            self.visuals['text_hor1'][j].attach(self.text_hor_top_colorfilter, view=self.visuals['text_hor2'][j])
            if j in self.panels_vertical_ghtext:
                #For now only created for panel 0 and 5, because others don't need vertically oriented text atm.
                self.visuals['text_vert1'][j] = visuals.TextVisual(text=startup_string,pos=[-1e6,-1e6],color=np.ones(4),font_size=eval(self.font_sizes['text_vert1']),face='OpenSans-Bold',rotation=90,anchor_x='center',anchor_y='center')
                self.visuals['text_vert2'][j] = self.visuals['text_vert1'][j].view()
                self.visuals['text_vert1'][j].attach(self.text_hor_bottom_colorfilter, view=self.visuals['text_vert1'][j])
                self.visuals['text_vert1'][j].attach(self.text_hor_top_colorfilter, view=self.visuals['text_vert2'][j])
            
            if j==0: 
                self.visuals['sm_pos_markers'][j]=visuals.MarkersVisual(pos=np.array([[0,0]]))
                self.visuals['radar_markers'][j]=visuals.MarkersVisual(pos=np.array([[0,0]]))
            else: 
                self.visuals['sm_pos_markers'][j] = self.visuals['sm_pos_markers'][0].view()
                self.visuals['radar_markers'][j] = self.visuals['radar_markers'][0].view()
            self.visuals['sm_pos_markers'][j].visible=False
            
        self.panel_borders_width = 1
        self.visuals['panel_borders']=visuals.LineVisual(color=self.gui.panelbdscolor/255.,method='gl',width=self.scale_pixelsize(self.panel_borders_width))
        self.visuals['titles']=visuals.TextVisual(text=startup_string,pos=[-1e6,-1e6],color='black',bold=True,font_size=eval(self.font_sizes['titles']),face='OpenSans',anchor_x='center',anchor_y='top')
        
        for j in range(self.max_panels):
            self.visuals['cbar'+str(j)]=visuals.ColorBarVisual(pos=[0,0],size=[1,1],cmap=self.cm2[self.crd.products[j]],orientation='right',clim=[-1,1],label_color=(0,0,0,0),border_width=self.scale_pixelsize(1))
            for i in self.visuals['cbar'+str(j)]._ticks:
                i.visible=False 
            #Set the visibility of the TextVisuals for the ticks and the label to False, to prevent that they are drawn. This is because I don't use them.
            self.visuals['cbar'+str(j)]._label.visible=False

        # Small per-channel legend bars for the polarimetric RGB composite (product 'g'), shown directly in the
        # corner of each panel that currently displays 'g' (see set_polrgb_legend). One simple black->color
        # ColorBarVisual per channel (Z=red, CC=green, ZDR=blue), independent of the general cbar system above,
        # since that system is built around a single colormap per product, not three simultaneous channels.
        # Structured as flat, individually-named visuals ('polrgb_legend_bar'+panel+channel), matching the
        # 'cbar0'..'cbar9' pattern above -- NOT as per-panel dict entries in visuals_panels, since that would
        # make them inherit the per-panel pan/zoom transform applied to actual radar-image content, which would
        # break their fixed on-screen positioning whenever the user pans or zooms.
        polrgb_channel_colors = {'r':(1,0,0,1), 'g':(0,1,0,1), 'b':(0,0,1,1)}
        self.polrgb_channel_colors = polrgb_channel_colors # re-used in set_polrgb_legend to rebuild the
        #colormap in reverse when a channel's vmin > vmax (an intentionally inverted mapping, e.g. CC).
        for j in range(self.max_panels):
            for ch, channel_color in polrgb_channel_colors.items():
                # Color order [low-value-color, high-value-color] = [black, full channel color], matching the
                # low-to-high ordering used by the real product colorbars (self.cm2, see set_cmaps) for a
                # vertical 'right'-orientation ColorBarVisual -- this renders full color at the top, black at
                # the bottom. This assumes vmin < vmax; set_polrgb_legend swaps to the reverse colormap
                # (built from self.polrgb_channel_colors) whenever a channel is configured the other way
                # around, so the bright end of the bar always lines up with whichever value actually produces
                # that channel's maximum intensity, not just with a fixed top/bottom position.
                cmap = color.Colormap([(0,0,0,1), channel_color])
                bar = visuals.ColorBarVisual(pos=[0,0], size=[1,1], cmap=cmap, orientation='right', clim=[-1,1],
                                              label_color=(0,0,0,0), border_width=self.scale_pixelsize(1),
                                              border_color=(1,1,1,0.8))
                for tick in bar._ticks:
                    tick.visible = False
                bar._label.visible = False
                bar.visible = False
                self.visuals['polrgb_legend_bar'+str(j)+ch] = bar
        self.visuals['polrgb_legend_ticks']=visuals.TextVisual(text=startup_string,pos=[-1e6,-1e6],color='white',bold=True,font_size=eval(self.font_sizes['cbars_ticks']),face='OpenSans',anchor_x='left',anchor_y='center')
        self.visuals['polrgb_legend_labels']=visuals.TextVisual(text=startup_string,pos=[-1e6,-1e6],color='white',bold=True,font_size=eval(self.font_sizes['cbars_labels']),face='OpenSans',anchor_x='left',anchor_y='bottom')
        self.visuals['polrgb_legend_reflines']=visuals.LineVisual(pos=None,color=(1,1,1,0.8),method='gl',connect='segments',width=self.scale_pixelsize(1))

        # Legenda voor de hydrometeorenclassificatie (HCLASS, product 'j'): een kolom van kleine gekleurde
        # vierkantjes (een per klasse, zie nlr_hclass.HID_CLASSES/HID_COLORS_RGBA) met daarnaast de klassenaam.
        # Net als bij de PolRGB-legenda hierboven is dit EEN gedeelde MarkersVisual/TextVisual voor alle panelen
        # samen (niet per paneel), zodat er niet 10*max_panels aparte visual-objecten nodig zijn -- de
        # posities/kleuren worden per redraw opnieuw opgebouwd in set_hclass_legend(), net als de PolRGB-tick/
        # labelteksten hierboven.
        self.visuals['hclass_legend_markers']=visuals.MarkersVisual(pos=np.array([[-1e6,-1e6]]))
        self.visuals['hclass_legend_labels']=visuals.TextVisual(text=startup_string,pos=[-1e6,-1e6],color='white',bold=True,font_size=eval(self.font_sizes['hclass_legend_labels']),face='OpenSans',anchor_x='left',anchor_y='center')

        # A/B line tool: a simple two-point distance-measuring line, drawn by holding Shift while clicking (see
        # set_ab_line_point). One LineVisual for the line itself, and one TextVisual for its three labels ('A',
        # 'B', and the distance in km, shown at the line's midpoint) -- following the same general pattern as
        # the PolRGB legend's ticks/reflines above. The cross-section itself is toggled via F2 (all supported
        # panels at once) or a small per-panel button in the panel's top-right corner (see
        # cross_section_toggle_button, added alongside the other per-panel cross-section visuals below) --
        # an earlier version showed a 'Show cross-section >>' text link at the line's midpoint instead, but
        # that stood out too much against the rest of the interface's understated style.
        self.visuals['ab_line']=visuals.LineVisual(pos=None,color=(1,1,1,0.9),method='gl',connect='strip',width=self.scale_pixelsize(1.5))
        self.visuals['ab_line_labels']=visuals.TextVisual(text=startup_string,pos=[-1e6,-1e6],color='white',bold=True,font_size=eval(self.font_sizes['cbars_labels']),face='OpenSans',anchor_x='center',anchor_y='center')
        # Sleep-rechthoek voor het selecteren van een gebied (bijv. voor de 3D-volumeweergave, CTRL+SHIFT+4
        # in nlr.py) -- CTRL+SHIFT+links-slepen op de kaart. Een aparte visual/modus i.p.v. de bestaande
        # ab_line te hergebruiken, omdat Erik expliciet een ECHTE sleep-selectie met live voorvertoning wilde
        # zien tijdens het slepen zelf, in plaats van pas achteraf (na 2 losse Shift+klikken) te zien welk
        # gebied ontstaat (zie gesprek met Claude, 5 juli 2026). 5 punten (4 hoeken + terug naar de eerste)
        # zodat connect='strip' een gesloten rechthoek tekent i.p.v. een open lijn van 4 punten.
        self.visuals['volume3d_rect']=visuals.LineVisual(pos=None,color=(1,1,0,0.9),method='gl',connect='strip',width=self.scale_pixelsize(2.))
        self.visuals['volume3d_rect_labels']=visuals.TextVisual(text=startup_string,pos=[-1e6,-1e6],color='yellow',bold=True,font_size=eval(self.font_sizes['volume3d_rect_labels']),face='OpenSans',anchor_x='center',anchor_y='center')
        # Shared cross-section marker, map side (see set_xsection_marker_frac/update_xsection_marker): a single
        # draggable point on the A-B line itself, kept in lockstep with the vertical marker line(s) on whichever
        # cross-section plot(s) are currently open (self.visuals['cross_section_marker'], one per panel, set up
        # further below alongside the other per-panel cross-section visuals). A bright, high-contrast yellow
        # (rather than reusing the plain white of the A/B line itself) so it stands out clearly against both the
        # dark map and whatever colorful radar data happens to be underneath it.
        self.visuals['ab_line_marker']=visuals.MarkersVisual(pos=np.array([[0,0]]))
        self.visuals['ab_line_marker'].visible=False

        # Puntje dat achterblijft op de plek van de laatste HCLASS/MESH-pop-up-klik (23 juli, op
        # Eriks verzoek - anders is na het wegklikken van de pop-up niet meer te zien waar je
        # precies had geklikt). Een ander, opvallend rood i.p.v. het geel van ab_line_marker, om
        # verwarring met de A/B-lijn te voorkomen. Zie show_extra_info_popup (aanzetten) en
        # on_mouse_release (rechtsklik verbergt 'm weer).
        self.visuals['click_marker']=visuals.MarkersVisual(pos=np.array([[0,0]]))
        self.visuals['click_marker'].visible=False

        self.visuals['cbars_ticks']=visuals.TextVisual(text=startup_string,pos=[-1e6,-1e6],color='black',font_size=eval(self.font_sizes['cbars_ticks']),face='OpenSans',anchor_x='center',anchor_y='center')
        self.visuals['cbars_reflines']=visuals.LineVisual(pos=None,color='black',method='gl',connect=None,width=self.scale_pixelsize(1))
        self.visuals['cbars_labels']=visuals.TextVisual(text=startup_string,pos=[-1e6,-1e6],color='black',font_size=eval(self.font_sizes['cbars_labels']),bold=True,face='OpenSans',anchor_x='center',anchor_y='top')
                
        self.panel_bounds={}
        self.panel_corners={}
        self.panel_centers={j:np.array([0,0]) for j in range(self.max_panels)} #In screen coordinates
        self.clippers={j:Clipper() for j in range(self.max_panels)}
        #STTransforms for handling positioning of widgets in panels, and panning and zooming.
        # Use (approximately) the base panel y range on start-up. It is approximately, because 0.9*self.gui.screen_size() is used instead of
        # self.wcenter['main'], because the latter isn't up-to-date yet at this point in the initialisation process
        self.panels_sttransforms={j:STTransform(scale=1/self.base_range*0.5*0.9*self.gui.screen_size()[1]*np.array([1, 1])) for j in range(self.max_panels)}
        
        self.set_panel_sttransforms_and_clippers()
        self.set_panel_borders()  
                
        for i in range(self.max_panels):
            self.panels_sttransforms[i].dynamic=True
            for j in self.visuals_panels:
                if not i in self.visuals[j]: continue
                self.visuals[j][i].set_gl_state(depth_test=False, blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'))
                self.visuals[j][i].attach(self.clippers[i], view = self.visuals[j][i]) 
                #Important!!!: view = self.visuals[j][i] is required when some of the objects in self.visuals are views.
                #If view is not specified, then the clipper is always applied to the visual to which the view belongs, meaning that
                #due to nonoverlapping clippers this visual is not shown at all.

                if type(self.visuals[j][i].transform)==STTransform:
                    #This is done to prevent that the current STTransform and self.panels_sttransforms[i] are combined into one, which would cause the transform
                    #of that visual not to update when self.panels_sttransforms[i] is updated.
                    self.visuals[j][i].transform=ChainTransform(self.visuals[j][i].transform)
                self.visuals[j][i].transform=self.panels_sttransforms[i]*self.visuals[j][i].transform
                
            d = 1.5
            self.visuals['text_hor1'][i].transform = STTransform(translate=(d,d))*ChainTransform(self.visuals['text_hor1'][i].transform)
            if i in self.panels_vertical_ghtext:
                self.visuals['text_vert1'][i].transform = STTransform(translate=(d,d))*ChainTransform(self.visuals['text_vert1'][i].transform)

        # Velocity/Reflectivity vertical cross-section view (see show_cross_section/hide_cross_section), one
        # set of visuals PER PANEL (not a single shared instance) -- multiple panels (e.g. one showing Z,
        # another V) can each have their own cross-section split-view active at the same time, driven by the
        # same shared A/B line. Created here, AFTER the clipper/transform-attaching loop above (which the
        # 'if not i in self.visuals[j]: continue' check causes to skip these names entirely, since they don't
        # exist in self.visuals[j] yet at that point) -- these visuals deliberately do NOT get the automatic
        # self.panels_sttransforms[i] pan-zoom chain or self.clippers[i] attachment that every other
        # self.visuals_panels entry receives, since the cross-section image/labels/frame are positioned with
        # their own explicit screen-pixel coordinates within the top half of the (now split) panel, independent
        # of whatever pan/zoom state the panel's normal view happens to be in -- see show_cross_section.
        for i in range(self.max_panels):
            # Three separate ImageVisuals are pre-built per panel, one per entry in
            # self.cross_section_interpolation_modes ('nearest'/'bilinear'/'bicubic'), rather than a single one
            # whose .interpolation gets changed later -- see change_cross_section_interpolation_mode in nlr.py
            # and cross_section_visual_key below for why: switching interpolation on an existing/runtime-
            # created ImageVisual was found to render solid black, most likely because it bypasses the
            # transforms.configure(canvas=self, viewport=vp) wiring that on_resize normally does for every
            # visual (see on_resize) -- building all 3 up front here means that wiring happens the normal way
            # for all of them, and switching mode later is then just a matter of toggling which one is
            # .visible, never creating or mutating a visual outside of __init__/on_resize.
            for mode in self.cross_section_interpolation_modes:
                key = 'cross_section_'+mode
                self.visuals[key][i]=visuals.ImageVisual(method='auto', cmap=self.cm1['v'], clim=self.clim_int['v'], interpolation=mode)
                self.visuals[key][i].visible=False
            self.visuals['cross_section_ticks'][i]=visuals.TextVisual(text=startup_string,pos=[-1e6,-1e6],color='white',bold=True,font_size=eval(self.font_sizes['cbars_ticks'])*0.8,face='OpenSans',anchor_x='center',anchor_y='center')
            # Axis line + small perpendicular tick marks along the bottom (distance) and left (height) edges of
            # the cross-section image -- a plain LineVisual with connect='segments' (each consecutive pair of
            # points drawn as one disconnected line), the same general approach as the existing cbars_reflines
            # visual for the normal colorbars' own tick reference lines.
            self.visuals['cross_section_axislines'][i]=visuals.LineVisual(pos=None,color='white',method='gl',connect=None,width=self.scale_pixelsize(1))
            self.visuals['cross_section_axislines'][i].visible=False
            # Plot-side half of the shared cross-section marker (see set_xsection_marker_frac/
            # update_xsection_marker and self.visuals['ab_line_marker'], its map-side counterpart): a simple
            # vertical line spanning the full height of THIS panel's cross-section image, at the x position
            # corresponding to the marker's current distance-along-A-B fraction. Positioned in the same
            # screen-pixel space as cross_section_axislines/cross_section_ticks (i.e. recomputed on every
            # show_cross_section call, see self.cross_section_layout), not the data-space image transform, so
            # it stays correctly aligned regardless of image resolution. Same yellow as ab_line_marker so the
            # two are visually associated at a glance.
            self.visuals['cross_section_marker'][i]=visuals.LineVisual(pos=None,color=(1,1,0,0.9),method='gl',connect='strip',width=self.scale_pixelsize(2))
            self.visuals['cross_section_marker'][i].visible=False
            self.visuals['cross_section_back'][i]=visuals.TextVisual(text='Back',pos=[-1e6,-1e6],color=(0.85,0.85,0.85,1.0),bold=False,font_size=eval(self.font_sizes['cbars_labels']),face='OpenSans',anchor_x='center',anchor_y='top')
            self.visuals['cross_section_back'][i].visible=False
            # Thin rectangular outline around the '< Back' label, same approach as ab_line_show_cs_frame --
            # gives it a subtle button-like appearance instead of plain, unbounded white text.
            self.visuals['cross_section_back_frame'][i]=visuals.LineVisual(color=(0.85,0.85,0.85,0.8),method='gl',width=self.scale_pixelsize(1))
            self.visuals['cross_section_back_frame'][i].visible=False
            self.visuals['cross_section_title'][i]=visuals.TextVisual(text=startup_string,pos=[-1e6,-1e6],color='white',bold=True,font_size=eval(self.font_sizes['cbars_labels']),face='OpenSans',anchor_x='center',anchor_y='top')
            self.visuals['cross_section_title'][i].visible=False
            # Solid dark background behind the cross-section image, covering the top half of the split panel.
            # The shared map visual (self.visuals['map'][i], a view() of visuals['map'][0]) still renders at
            # full panel size underneath this (it isn't hidden, since the bottom half still wants the map as
            # context), so without this opaque rectangle the map/streets would show through wherever the
            # cross-section raster itself is transparent (e.g. NaN bins with no nearby scan data). Drawn as a
            # plain colored rectangle rather than reusing e.g. visuals['background'], since this needs to be
            # sized/positioned per-show_cross_section-call for just the top half of THIS specific panel, not
            # the whole canvas.
            self.visuals['cross_section_background'][i]=visuals.RectangleVisual(center=[0,0],color=(0.04,0.04,0.06,1.0))
            self.visuals['cross_section_background'][i].transform=STTransform()
            self.visuals['cross_section_background'][i].visible=False
            # A simple white rectangle outline around each half of the split panel (cross-section on top,
            # normal radar image cropped to the bottom -- see show_cross_section's clipper-narrowing), drawn
            # the same way as the existing self.visuals['panel_borders'] (a LineVisual with 5 points per
            # rectangle, the 5th repeating the 1st, and connect[4::5]=False to keep the two rectangles visually
            # separate rather than connected by a stray line).
            self.visuals['cross_section_frame'][i]=visuals.LineVisual(color=(1,1,1,1.0),method='gl',width=self.scale_pixelsize(2))
            self.visuals['cross_section_frame'][i].visible=False
            # Small, unobtrusive per-panel button (top-right corner) to toggle the cross-section split-view for
            # this specific panel -- a less visually prominent alternative to F2 (which toggles every supported
            # panel at once), and a replacement for an earlier version that showed a bright 'Show cross-section
            # >>' text link at the A/B line's midpoint, which stood out too much against the rest of this
            # understated interface. Visible whenever this panel shows a cross-section-capable product AND a
            # completed A/B line exists -- see update_cross_section_toggle_buttons, called from draw_ab_line
            # and set_panel_sttransforms_and_clippers (so it also repositions correctly on resize/panel-count
            # changes, alongside the other panel-corner UI elements).
            self.visuals['cross_section_toggle_button'][i]=visuals.TextVisual(text='X-section',pos=[-1e6,-1e6],color=(0.85,0.85,0.85,1.0),bold=False,font_size=eval(self.font_sizes['cbars_labels']),face='OpenSans',anchor_x='center',anchor_y='top')
            self.visuals['cross_section_toggle_button'][i].visible=False
            self.visuals['cross_section_toggle_button_frame'][i]=visuals.LineVisual(color=(0.85,0.85,0.85,0.8),method='gl',width=self.scale_pixelsize(1))
            self.visuals['cross_section_toggle_button_frame'][i].visible=False
            for visual_name in cross_section_visual_names+['cross_section_axislines','cross_section_ticks','cross_section_marker','cross_section_back','cross_section_back_frame','cross_section_title','cross_section_background','cross_section_frame','cross_section_toggle_button','cross_section_toggle_button_frame']:
                self.visuals[visual_name][i].set_gl_state(depth_test=False, blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'))
                                
        for j in self.visuals_global:
            #These do not need a pan-zoom transform or clipper
            if isinstance(self.visuals[j], list):
                for visual in self.visuals[j]:
                    visual.set_gl_state(depth_test=False, blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'))
            else:
                self.visuals[j].set_gl_state(depth_test=False, blend=True, blend_func=('src_alpha', 'one_minus_src_alpha'))
            
        self.coordmaps={j:self.panels_sttransforms[j].map for j in range(self.max_panels)}
        self.coordimaps={j:self.panels_sttransforms[j].imap for j in range(self.max_panels)}

        self.set_cbars()  
        self.set_maplineproperties(self.panellist)
        self.set_ghlineproperties(self.panellist)
        self.set_radarmarkers_data()
        self.set_sm_pos_markers()
        
        
        
    def physical_size_cm(self):
        #Physical size of the vispy canvas
        return np.array(self.size)/(self.gui.screen_DPI())*2.54
    
    def scale_physicalsize(self,size):
        return size * self.gui.screen_physicalsize()[1]/self.gui.ref_screen_physicalsize[1]
        
    def scale_pointsize(self,size):
        return self.scale_physicalsize(size)
        
    def scale_pixelsize(self,size):
        return size*self.gui.screen_size()[1]/self.gui.ref_screen_size[1]

    def scale(self,r,array):
        array=np.asarray(array)
        array[array<0.5]=r*array[array<0.5]
        array[array>=0.5]=1-r*(1-array[array>=0.5])
        return array
    def calc_bounds_and_size(self,widget,xmin,xmax,ymin,ymax):
        print('size=',self.size)
        bounds=np.round(np.array([[xmin,xmax],[ymin,ymax]])*np.reshape(self.size,(2,1))).astype('int')
        pos=np.array([bounds[:,0],[bounds[0,0],bounds[1,1]],bounds[:,1],[bounds[0,1],bounds[1,0]]])
        center=np.mean(pos,axis=0); center[1]=self.size[1]-center[1]
        size=bounds[:,1]-bounds[:,0]
        
        self.wbounds[widget]=bounds.copy(); self.wpos[widget]=pos.copy(); self.wcenter[widget]=center.copy(); self.wsize[widget]=size.copy()
    def set_widget_sizes_and_bounds(self):
        """Sets the bounds, corner positions center positions and sizes of the 5 parts in which the canvas is divided, i.e. a left and right part for the 
        color bars, a lower and upper part for the titles, and the main part for the panels.
        
        When viewing the window in maximized state (but not full screen), then the relative bounds of the widgets are given by self.ref_bounds.
        When the window size is different, then the widths/heights of the left and right/bottom and top part of the canvas are kept constant (in pixels), 
        to keep enough space for the titles and cbars. This means that the relative bounds are different for different window sizes.
        
        The unit of the parameters is pixels.
        self.wbounds, self.wpos and self.wcenter are determined relative to an origin at the top left corner, with the y axis pointing downwards
        (as Vispy usually does).
        # self.wbounds_yu, self.wpos_yu and self.wcenter_yu are defined in a similar way, but with the origin at the bottom left corner, 
        # and the y axis pointing upwards.
        
        Bounds are given in the format [[xmin,xmax],[ymin,ymax]]
        Corner positions start with the top left corner, and are listed in counterclockwise order.
        """
        wdims = self.scale_physicalsize(self.wdims)
        panels_width = self.physical_size_cm()[0] - 2*wdims[0]
        if self.gui.show_vwp: 
            panels_width = self.physical_size_cm()[0]*(1-self.vwp_relxdim) - 2*wdims[0]
        rel_main_bounds = []
        rel_main_bounds += [[wdims[0], panels_width+wdims[0]]]
        rel_main_bounds += [[wdims[1], self.physical_size_cm()[1]-wdims[1]]]
        rel_main_bounds = np.array(rel_main_bounds) / np.reshape(self.physical_size_cm(),(2,1))
        b=rel_main_bounds
        
        if b[0,0]<b[0,1] and b[1,0]<b[1,1]:
            if not hasattr(self, 'wbounds'):
                # Do this only on initialisation, since otherwise it might happen while updating the map tiles (which runs in another thread),
                # that no specs for the main widget are available. This way at least they are, although they might be outdated.
                self.wbounds={}; self.wpos={}; self.wcenter={}; self.wsize={}
            
            self.calc_bounds_and_size('main',b[0,0],b[0,1],b[1,0],b[1,1])
            self.calc_bounds_and_size('left',0,b[0,0],0,1)
            if self.gui.show_vwp:
                self.calc_bounds_and_size('right',b[0,1],1-self.vwp_relxdim,0,1)
            else:
                self.calc_bounds_and_size('right',b[0,1],1,0,1)
            self.calc_bounds_and_size('top',b[0,0],b[0,1],0,b[1,0])   
            self.calc_bounds_and_size('bottom',b[0,0],b[0,1],b[1,1],1)
            if self.gui.show_vwp:
                self.calc_bounds_and_size('vwp',1-self.vwp_relxdim,1,0,1)     
            print(self.wsize)
        
    def determine_panellist_nrows_ncolumns(self):  
        self.panellist=tuple(self.plotnumber_to_panelnumber[self.panels][j] for j in range(0,self.panels))
        self.nrows=self.rows_panels[self.panels][self.panels-1]+1
        self.ncolumns=self.cols_panels[self.panels][self.panels-1]+1

    def get_row_col_panel(self,panel):
        p=self.panelnumber_to_plotnumber[self.panels][panel]
        return self.rows_panels[self.panels][p],self.cols_panels[self.panels][p]
        
    def get_panel_for_position(self,pos):
        rel_pos=np.array(pos)-self.wpos['main'][0] #Position relative to top left corner of the main part of the canvas
        size_panel=np.array(self.wsize['main'])/np.array([self.ncolumns,self.nrows])
        
        floor=np.floor(rel_pos/size_panel)
        if floor[0]+1>=self.ncolumns: floor[0]=self.ncolumns-1
        if floor[1]+1>=self.nrows: floor[1]=self.nrows-1
        plot=int(self.ncolumns*floor[1]+floor[0])
        
        return self.plotnumber_to_panelnumber[self.panels][plot]
    
    def set_panel_info(self):
        self.panel_centers_before=self.panel_centers.copy() #These values are needed in the function self.set_panel_sttransforms_and_clippers
        
        for j in self.panellist:
            size_panel=np.array(self.wsize['main'])/np.array([self.ncolumns,self.nrows])
            row, col=self.get_row_col_panel(j)
            topleft=self.wpos['main'][0]+size_panel*np.array([col,row]) #In screen coordinates
            bottomright=topleft+size_panel
            
            # b=np.array([topleft[0],self.size[1]-bottomright[1],size_panel[0],size_panel[1]])
            # (x, y, w, h), with y measured with upward pointing y axis, because that is necessary for the clipper! Screen coordinates are measured with 
            #the y axis pointing downwards.
            self.panel_bounds[j]=np.array([topleft[0],self.size[1]-bottomright[1],size_panel[0],size_panel[1]])
            #First corner is the top left one, and the other 3 are listed in counterclockwise order. These are given for y axis pointing downward!
            self.panel_corners[j]=topleft+np.array([[0,0],[0,size_panel[1]],size_panel,[size_panel[0],0]])
            self.panel_centers[j]=0.5*(topleft+bottomright)
    
    def set_panel_borders(self):
        #Call this function after self.set_panel_info
        panel_borders_pos=np.zeros((self.panels*5,2),dtype='float32')
        panel_borders_connect=np.ones(self.panels*5,dtype='bool')
        i=0
        a=self.scale_pixelsize(0.0)
        for j in self.panellist:
            c=self.panel_corners[j]
            #Shift the vertices of the lines that represent the edges of the panel a pixels inwards, such that the lines are
            #plotted at the edges of the panel (first/lasts pixel row/column encompassed by panel).
            panel_borders_pos[i:i+5]=np.array([c[0]+a*np.array([1,-1]),c[1]+a*np.array([1,1]),c[2]+a*np.array([-1,1]),c[3]+a*np.array([-1,-1]),c[0]+a*np.array([1,-1])])
            panel_borders_connect[i+4]=0
            i+=5
            
        self.visuals['panel_borders'].set_data(pos=panel_borders_pos,connect=panel_borders_connect)
            
    def set_panel_sttransforms_and_clippers(self):
        self.set_panel_info()

        panel_center_shift=np.array(self.panels_sttransforms[0].translate[:2])-self.panel_centers_before[0] #Always use the first panel for calculating the
        #shift of the panel center, that is required to ensure that the center of each panel shows the same geographical location as before. 
        #Using the first panel is necessary, because this is the only panel that always gets updated by this function.
        for j in self.panellist:
            self.clippers[j].bounds = tuple(self.panel_bounds[j]*self.gui.screen_pixel_ratio())
            #Shift the geographical location of the center of the panels
            self.panels_sttransforms[j].translate=self.panel_centers[j]+panel_center_shift
        # Re-position the per-panel cross-section toggle buttons (top-right corner) whenever panel
        # positions/sizes change. Guarded with a dict-membership check (rather than calling unconditionally)
        # since this function is also called once during __init__, before the per-panel cross-section visuals
        # further down in __init__ have been created yet.
        if 'cross_section_toggle_button' in self.visuals and self.visuals['cross_section_toggle_button']:
            self.update_cross_section_toggle_buttons()
      
        
    def calculate_vwp_relxdim(self):
        f = self.size[1]/self.size[0] / 0.5136825645035183 # Reference value
        self.vwp_relxdim = 0.255*f #The fractional width of the VWP plot relative to that of the whole canvas
            
    def on_resize(self, event=None):
        font_size = int(round(self.scale_pixelsize(13)))
        # Set pixel size instead of point size, since setting point size leads to weird results. I.e. using value of 8 leads to much
        # much bigger fonts than size of 7.
        if font_size != self.gui.f1.pixelSize():
            self.gui.f1.setPixelSize(font_size)
            QApplication.instance().setFont(self.gui.f1)
            for widget in QApplication.allWidgets():
                f = widget.font()
                f.setPixelSize(font_size)
                widget.setFont(f)
        
        self.dpi = self.gui.screen_DPI()
        
        # Set canvas viewport and reconfigure visual transforms to match.
        vp = (0, 0, self.physical_size[0], self.physical_size[1])
        self.context.set_viewport(*vp)
        
        for j in self.visuals_order:
            if j in self.visuals_global:
                self.visuals[j].transforms.configure(canvas=self, viewport=vp)
                if hasattr(self.visuals[j], 'font_size'):
                    self.visuals[j].font_size = self.scale_pointsize(eval(self.font_sizes[j]))
            else:
                for visual in self.visuals[j].values():
                    visual.transforms.configure(canvas=self, viewport=vp)
                    if hasattr(visual, 'font_size'):
                        visual.font_size = self.scale_pointsize(eval(self.font_sizes[j]))
        
        # First resize the 5 widgets in which the canvas is divided, and then resize the visuals that reside in them.
        self.calculate_vwp_relxdim()
        self.set_widget_sizes_and_bounds()
        self.set_panel_sttransforms_and_clippers()
        self.set_panel_borders()
        
        # Now update all visuals with line widths, font sizes etc in order to use the new canvas dimensions and/or resolution.
        self.set_cbars(resize=True, set_cmaps=False)
        self.set_maplineproperties(self.panellist)
        self.set_radarmarkers_data()
        if self.firstplot_performed:
            if 'grid' in self.gui.lines_show: self.set_grid()
            if 'heightrings' in self.gui.lines_show: self.set_heightrings()
            if any([j in self.gui.lines_show for j in ('grid','heightrings')]):
                self.set_ghlineproperties(self.panellist); self.set_ghtextproperties(self.panellist)
            self.set_titles()
            
        self.visuals['background_map'].transform.scale=self.wsize['main']
        self.visuals['background_map'].transform.translate=self.wcenter['main']
                    
        if self.gui.show_vwp:
            self.vwp.on_resize()

        for panel in list(self.cross_section_active_panels):
            # Panel pixel bounds (self.panel_corners, used throughout show_cross_section to lay out the split
            # top/bottom halves) have just been recalculated above via set_panel_sttransforms_and_clippers, so
            # redo the cross-section layout against the new size rather than leaving it positioned for the old
            # window size.
            self.show_cross_section(panel)
                    
        self.set_draw_action('resizing')
        self.update_map_tiles_ondraw = True

    def set_draw_action(self, new_action):
        # Change self.draw_action in such a way that it combines the actions of new_action with those of any possibly existing self.draw_action 
        # Assumes that from left to right each draw action includes at least the widgets contained in the previous action.
        priority_order = {'no vwp':['panning_zooming', 'plotting', 'changing_panels', 'updating_cbars', 'resizing'],
                          'vwp':['vwp_only', 'plotting_vwp']}
        if not hasattr(self, 'draw_action') or self.draw_action is None:
            self.draw_action = new_action
        elif ('vwp' in new_action) != ('vwp' in self.draw_action):
            self.draw_action = 'plotting_vwp'
        elif ('vwp' in new_action) == ('vwp' in self.draw_action):
            vwp_mode = 'vwp' if 'vwp' in new_action else 'no vwp'
            i1, i2 = [priority_order[vwp_mode].index(j) for j in (self.draw_action, new_action)]
            self.draw_action = priority_order[vwp_mode][i1 if i1 > i2 else i2]

    def on_draw(self, ev):    
        # from cProfile import Profile
        # profiler = Profile()
        # profiler.enable()
        if self.update_map_tiles_ondraw:
            if self.draw_action in ('panning_zooming', 'resizing'):
                self.set_timer_update_map_tiles()
            else:
                self.update_map_tiles(separate_thread=False, draw_map=True)
        self.update_map_tiles_ondraw = False
                
        # UITGESCHAKELD (6 juli 2026): deze selectieve draw_widgets-optimalisatie (bv. tijdens pannen/zoomen
        # alleen 'main' opnieuw tekenen, kleurenbalken/titel overslaan) was origineel alleen visueel stabiel
        # dankzij GL.glDrawBuffer(GL.GL_FRONT_AND_BACK) verderop in on_draw(), die ervoor zorgde dat een
        # eenmalige tekenbeurt op BEIDE OpenGL-buffers tegelijk belandde (waardoor overgeslagen widgets bij
        # de volgende buffer-wissel niet verdwenen). Nu die aanroep op dit systeem faalt (GL_INVALID_OPERATION,
        # stil afgeschermd verderop), zorgt het overslaan van widgets voor zichtbaar geflikker tijdens
        # pannen/zoomen (kleurenbalken/titel die aan/uit knipperen). Simpelste robuuste fix: altijd ALLES
        # opnieuw tekenen, ongeacht draw_action -- iets minder snel tijdens interactief pannen/zoomen, maar
        # zonder geflikker.
        self.draw_widgets=self.widgets.copy()
        if self.gui.show_vwp:
            self.draw_widgets += ['vwp']
        if self.draw_action == 'vwp_only':
            self.draw_widgets = ['vwp']
            
        if not self.gui.use_scissor: self.draw_widgets=self.widgets
                             
        
        if self.starting or self.draw_action in ('resizing',None) or self.draw_widgets!=self.draw_widgets_before:
            topleft_corner=np.min([self.wpos[j][0] for j in self.draw_widgets],axis=0)
            bottomright_corner=np.max([self.wpos[j][2] for j in self.draw_widgets],axis=0)
            self.visuals['background'].transform.translate=np.mean([topleft_corner, bottomright_corner], axis = 0)
            self.visuals['background'].transform.scale=bottomright_corner-topleft_corner
        self.visuals['background'].draw()
                 
        if self.starting or self.draw_action in ('resizing',None) or self.draw_widgets!=self.draw_widgets_before:
            # TERUGGEDRAAID (6 juli 2026): dit was tijdelijk onvoorwaardelijk gemaakt (bij elke wijziging
            # van draw_widgets) in een poging het titel-positieprobleem op te lossen, maar dat bleek een
            # NIEUW, EIGEN probleem te veroorzaken: tijdens pannen/zoomen wordt draw_widgets bewust
            # verkleind tot enkel ['main'] (voor snelheid, de kleurenbalken links/rechts hoeven dan niet
            # opnieuw getekend te worden) -- een onvoorwaardelijke witte clear veegt dan het HELE canvas
            # wit, inclusief die kleurenbalken, wat zichtbaar was als geflikker tijdens het slepen. Het
            # titel-positieprobleem zelf is inmiddels apart opgelost via een y-positie-correctie in
            # set_titles() (zie _title_font_px hieronder in die functie), dus deze clear kan gewoon weer
            # beperkt worden tot alleen de allereerste tekenbeurt.
            if self.starting: gloo.clear('white')
            try:
                GL.glDrawBuffer(GL.GL_FRONT_AND_BACK)
            except Exception:
                pass
                        
            
        #Only plot visuals in the widgets for the main radar window here, not those in other windows such as the vwp window.
        #These are plotted in their own classes.
        visuals_to_plot = [self.visuals_widgets[j] for j in self.draw_widgets if j in self.widgets]
        if len(visuals_to_plot) > 0:
            visuals_to_plot=np.concatenate(visuals_to_plot)
        for j in self.visuals_order:
            if not j in visuals_to_plot: continue
            
            if j in self.visuals_global:
                if isinstance(self.visuals[j], list):
                    for visual in self.visuals[j]:
                        if visual.visible:
                            visual.draw()
                elif self.visuals[j].visible:
                    self.visuals[j].draw()
            else:    
                for i in self.panellist:
                    if i in self.visuals[j] and self.visuals[j][i].visible:
                        self.visuals[j][i].draw()
                        
        if self.firstdraw_map and self.gui.mapvisibility:
            #It is important to draw the map for all panels during the first draw, because otherwise the scale gets screwed up when changing the
            #number of panels.
            for j in range(self.max_panels):
                if not j in self.panellist:
                    self.visuals['map'][j].draw()
            self.firstdraw_map = False
            
        for j in self.panellist:
            if self.data_attr['proj'].get(j, None) == 'pol' and j not in self.ref_radial_bins:
                # These reference values are needed in self.set_newdata when changing the scale of the polar image transform
                self.ref_azimuthal_bins[j], self.ref_radial_bins[j] = self.dsg.data[j].shape[:2]
            
        if 'vwp' in self.draw_widgets:
            #Plot the vwp visuals
            self.vwp.on_draw()
        
        if self.draw_action in ('plotting', 'plotting_vwp', 'changing_panels') and self.gui.continue_savefig:
            self.swap_buffers() #swapping the buffers causes the offscreen buffer to become visible, which must be the case before saving the figure.
            pytime.sleep(0.01)
            self.gui.savefig(select_filename=False)
          
        self.draw_action=None
        self.starting=False
        self.draw_widgets_before=self.draw_widgets.copy()
        # profiler.disable()
        # import pstats
        # stats = pstats.Stats(profiler).sort_stats('cumtime')
        # stats.print_stats(30)  
        
            
    def set_panels_sttransforms_manually(self, scale=None, translate=None, panzoom_action=True, draw=True):
        for j in range(self.max_panels):
            if not scale is None: self.panels_sttransforms[j].scale = scale
            if not translate is None: self.panels_sttransforms[j].translate = translate
        self.set_panel_sttransforms_and_clippers()
            
        if panzoom_action:
            self.dsg.time_last_panzoom=pytime.time()
        
        if draw:
            if any([j in self.gui.lines_show for j in ('grid','heightrings')]) and self.firstplot_performed: 
                self.update_gridheightrings()
                
            self.set_draw_action('panning_zooming')
            self.update_map_tiles(separate_thread = False, draw_map = True)
            self.update()         
        
    def reset_panel_view(self, reset_zoom=True, reset_center=True):
        pc = self.panel_centers[0]
        ps, pt = self.panels_sttransforms[0].scale[:2], self.panels_sttransforms[0].translate[:2]
        if reset_zoom:
            new_scale = 1/self.base_range*self.wcenter['main'][1]*np.array([1, 1])
            if reset_center:
                self.set_panels_sttransforms_manually(new_scale, pc)
            else:
                zoom = new_scale/ps
                new_translate = pc-(pc-pt)*zoom
                self.set_panels_sttransforms_manually(new_scale, new_translate)
        else:
            self.set_panels_sttransforms_manually(translate=pc)
            
    def translation_dist_km(self, delta_time, start_datetime):
        # Obtain dt_bounds with seconds accuracy, since for small delta_time these seconds can be important.
        # Without seconds accuracy, it can be the case that a small change in delta_time leads to a larger change in the
        # contributions from different storm motions (through delta_t values that are changed by +/-60 seconds), which
        # can lead to small jumps in the translation.
        start_datetime = start_datetime+'00'
        end_datetime = ft.next_datetime_s(start_datetime, delta_time)
        dt_bounds = sorted([start_datetime, end_datetime])
        
        sm_datetimes = sorted([j for j in self.gui.stormmotion_save if int(dt_bounds[0]) < int(j+'00') < int(dt_bounds[1])])
        keys = [self.gui.select_nearest_sm_datetime(dt_bounds[0][:12])] + sm_datetimes
        sign = np.sign(delta_time)
        delta_time = abs(delta_time)
        dist = 0.
        for i,key in enumerate(keys):
            if i == 0 and len(keys) > 1:
                delta_t = ft.datetimediff_s(dt_bounds[0], keys[1])
            elif len(keys) > i+1:
                delta_t = ft.datetimediff_s(key, keys[i+1])
            else:
                delta_t = delta_time
            delta_time -= delta_t
            
            SM = self.gui.update_stormmotion_change_datetime_or_radar(key, self.crd.radar, set_stormmotion=False)
            SM = -SM[1]*np.array([np.sin(np.deg2rad(SM[0])), np.cos(np.deg2rad(SM[0]))])
            dist += SM*sign*delta_t/1e3
            
        return dist          
        
    def move_view_with_storm(self, panellist):
        # print('Move_view_with_storm')
        dt = abs(ft.datetimediff_s(self.crd.before_variables['datetime'], self.crd.date+self.crd.time))
        max_dt = 60*(self.ani.duration+60) if self.ani.continue_type[:3] == 'ani' else self.max_delta_time_move_view_with_storm
        # Prevent movement in case of very big timesteps, that e.g. occur when switching to the next day with available data
        if dt > max_dt and not self.gui.switch_to_case_running:
            # It's necessary to also reset self.ref_vals in this case
            self.ref_vals = {}
            return False
        
        # Use stormmotion_save, as stormmotion changes when changing radar
        meta = str(self.dsg.time_last_panzoom)+str(self.gui.stormmotion_save)
        if meta != getattr(self, 'ref_meta', None) or self.gui.switch_to_case_running:
            self.ref_vals = {}
            self.ref_meta = meta
        ref_vals = self.ref_vals.get(self.crd.radar, {})
        # Use fixed reference values for moving view with storm, in order to prevent that during an animation the view slightly changes 
        # with each iteration. These reference values are only updated when the user changes the view or storm motion
        if not ref_vals:
            try:
                ref_vals['scandatetime'] = self.gui.current_case.get('scandatetime', self.gui.current_case['datetime']) if\
                                     self.gui.switch_to_case_running else self.data_attr_before['scandatetime'][0]
            except Exception:
                return False
            ref_vals['datetime'] = self.gui.current_case['datetime'] if self.gui.switch_to_case_running else self.crd.before_variables['datetime']
            ref_vals['translate'] = self.panels_sttransforms[0].translate[:2]
            ref_vals['panel_center'] = self.panel_centers[0].copy()
            self.ref_vals[self.crd.radar] = ref_vals

        scale = np.array(self.panels_sttransforms[0].scale[:2])
        for j in panellist:
            try:
                # Time differene is now calculated relative to reference scan datetime of first panel, in contrast to what happened in the past
                # (relative to reference datetime for panel itself). This leads to more desired behaviour, with scans in each panel shown as 
                # you would see when moving up/down in first panel.
                dt = ft.datetimediff_s(ref_vals['scandatetime'], self.data_attr['scandatetime'][j])
            except Exception: 
                continue
            # Moving needs to occur in opposite direction
            distance_move = -self.translation_dist_km(dt, ref_vals['datetime'])*np.array([1, -1])*scale
            center_shift = self.panel_centers[j] - ref_vals['panel_center']
            self.panels_sttransforms[j].translate = ref_vals['translate'] + center_shift + distance_move

        self.update_map_tiles_ondraw = True
        return True
        
    def on_mouse_wheel(self, ev):
        if not ft.point_inside_rectangle(ev.pos,self.wpos['main'])[0]: return
        
        zoomfactor=(1 + self.zoomfactor_vispy) ** (ev.delta[1] * 30)
        
        pos=np.array(ev.pos)
        selected_panel=self.get_panel_for_position(pos)
        rel_pos=pos-self.panel_centers[selected_panel]
        for j in range(self.max_panels): #Zooming the transforms for all panels is necessary, to ensure that all panels keep showing the same area
            self.panels_sttransforms[j].zoom((zoomfactor,zoomfactor),center=self.panel_centers[j]+rel_pos,mapped=True)

        self.dsg.time_last_panzoom=pytime.time()

        if self.ab_line_panel is not None:
            # The A/B line's screen position (self.visuals['ab_line'], plus the shared cross-section marker
            # riding on it -- see update_xsection_marker) is recomputed from its stored AEQD x/y coordinates
            # every time draw_ab_line() runs; it doesn't update on its own just because self.panels_sttransforms
            # changed above. Click-drag panning and right-click zoom already call draw_ab_line() after touching
            # the transforms (see on_mouse_move) -- mouse-wheel zoom was missing the same call, which is why the
            # line (and, now, the marker) drifted out of place specifically when scrolling to zoom.
            self.draw_ab_line()

        if any([j in self.gui.lines_show for j in ('grid','heightrings')]) and self.firstplot_performed: 
            if self.gui.showgridheightrings_panzoom: 
                #This is needed to ensure that updating the heightrings occurs after the zooming has been performed
                self.update_gridheightrings(ev)
            else:
                if not self.gridheightrings_removed:
                    self.remove_gridheightrings()
                self.set_timer_setback_gridheightrings()
                
        self.update_data_readout()
            
        self.set_draw_action('panning_zooming')
        self.update_map_tiles_ondraw = True
        self.update()
        
    def set_ab_line_point(self, screen_pos):
        """Handles a Shift+click for the A/B line tool (see on_mouse_press). The first click sets point A; the
        second sets point B and draws the finished line with its length label. A further click after that
        starts a fresh line from scratch, discarding the old one.

        Points are stored as AEQD x/y coordinates (km from the radar, the same coordinate system
        screencoord_to_xy already converts to) rather than raw screen pixels, so the line stays correctly
        anchored to the actual displayed location across panning/zooming -- exactly like the underlying radar
        data itself, instead of drifting along with mouse-pixel space.
        """
        panel = self.get_panel_for_position(screen_pos)
        xy = self.screencoord_to_xy(np.array(screen_pos), panel)

        if self.ab_line_panel is None or self.ab_line_b is not None:
            # Starting a fresh line (either the very first click ever, or a new line after a completed one).
            self.ab_line_panel = panel
            self.ab_line_a = xy
            self.ab_line_b = None
        elif panel != self.ab_line_panel:
            # Clicking in a different panel restarts the line there, rather than mixing coordinate systems
            # from two different panels into one line.
            self.ab_line_panel = panel
            self.ab_line_a = xy
            self.ab_line_b = None
        else:
            self.ab_line_b = xy

        self.draw_ab_line()

    def draw_ab_line(self):
        if self.ab_line_a is None:
            self.visuals['ab_line'].visible = False
            self.visuals['ab_line_labels'].text = []
            self.update_cross_section_toggle_buttons()
            self.update()
            return

        panel = self.ab_line_panel
        screen_a = self.xycoord_to_screen(panel, self.ab_line_a)
        labels_text, labels_pos = ['A'], [screen_a]

        if self.ab_line_b is not None:
            screen_b = self.xycoord_to_screen(panel, self.ab_line_b)
            distance_km = np.linalg.norm(self.ab_line_a-self.ab_line_b)
            mid_screen = 0.5*(screen_a+screen_b)
            self.visuals['ab_line'].set_data(pos=np.array([screen_a, screen_b]))
            self.visuals['ab_line'].visible = True
            labels_text += ['B', str(ft.rifdot0(ft.r1dec(distance_km)))+' km']
            labels_pos += [screen_b, mid_screen]
        else:
            # Only point A has been set so far -- show just that marker/label, no line yet.
            self.visuals['ab_line'].visible = False

        self.visuals['ab_line_labels'].text = labels_text
        self.visuals['ab_line_labels'].pos = np.array(labels_pos)
        # Whether each panel's small top-right cross-section toggle button should be visible/clickable depends
        # on whether a completed A/B line now exists (see update_cross_section_toggle_buttons), so refresh
        # those buttons' visibility every time the line itself changes.
        self.update_cross_section_toggle_buttons()
        # The map-side marker's screen position depends on the panel's current pan/zoom (see
        # update_xsection_marker), so keep it in sync whenever the line itself gets redrawn -- in particular
        # while panning/zooming the panel the line lives in (see on_mouse_move), not just after an explicit
        # drag of the marker itself.
        self.update_xsection_marker()
        self.update()

    def draw_volume3d_rect(self):
        """Tekent/werkt de sleep-rechthoek bij (zie volume3d_rect_a/b, gezet via CTRL+SHIFT+links-slepen in
        on_mouse_press/move/release). Zelfde opzet als draw_ab_line hierboven, maar dan voor een rechthoek
        (4 hoeken, gesloten via connect='strip' met het beginpunt herhaald aan het eind) i.p.v. een lijn.

        BIJGESTELD (8 juli 2026, op Eriks verzoek: "zou dat ook een cirkel kunnen zijn" + "zie ik dan wel
        vooraf welk gebied in de 3D cirkel komt?"): als self.gui.volume3d_circular_area aanstaat, wordt hier
        een ELLIPS getekend die precies in de rechthoek past (i.p.v. de rechthoek zelf), zodat de 2D-
        voorvertoning altijd exact overeenkomt met wat er straks in 3D als data-gebied wordt gebruikt (zie
        de bijbehorende maskering in nlr.py's _render_volume_3d_scene). Het label met de afmetingen blijft
        de afmetingen van de OMSCHRIJVENDE rechthoek tonen (dus de volledige breedte/hoogte), niet van de
        ellips zelf, voor consistentie met hoe het kader/de assen in 3D dat ook doen."""
        if self.volume3d_rect_a is None or self.volume3d_rect_b is None:
            self.visuals['volume3d_rect'].visible = False
            self.visuals['volume3d_rect_labels'].text = []
            self.update()
            return

        panel = self.volume3d_rect_panel
        x1, y1 = self.volume3d_rect_a
        x2, y2 = self.volume3d_rect_b
        if getattr(self.gui, 'volume3d_circular_area', False):
            cx, cy = (x1+x2)/2., (y1+y2)/2.
            a, b = abs(x2-x1)/2., abs(y2-y1)/2.
            theta = np.linspace(0, 2*np.pi, 72)
            xy_ellipse = [(cx+a*np.cos(t), cy+b*np.sin(t)) for t in theta]
            screen_corners = [self.xycoord_to_screen(panel, np.array(c)) for c in xy_ellipse]
            # De ellips-puntenreeks is al gesloten (begin- en eindhoek van linspace vallen op hetzelfde
            # punt), dus geen extra herhaling van het beginpunt nodig zoals bij de rechthoek hieronder.
        else:
            xy_corners = [(min(x1, x2), min(y1, y2)), (max(x1, x2), min(y1, y2)),
                          (max(x1, x2), max(y1, y2)), (min(x1, x2), max(y1, y2))]
            screen_corners = [self.xycoord_to_screen(panel, np.array(c)) for c in xy_corners]
            screen_corners.append(screen_corners[0]) # Sluit de rechthoek (terug naar de eerste hoek).
        self.visuals['volume3d_rect'].set_data(pos=np.array(screen_corners))
        self.visuals['volume3d_rect'].visible = True

        width_km, height_km = abs(x2-x1), abs(y2-y1)
        mid_screen = self.xycoord_to_screen(panel, np.array([(x1+x2)/2., (y1+y2)/2.]))
        self.visuals['volume3d_rect_labels'].text = [f"{ft.rifdot0(ft.r1dec(width_km))} x "
                                                       f"{ft.rifdot0(ft.r1dec(height_km))} km"]
        self.visuals['volume3d_rect_labels'].pos = np.array([mid_screen])
        self.update()

    def set_xsection_marker_frac(self, frac):
        """Sets the shared cross-section marker's position, expressed as a fraction along the A/B line
        (0.0 = point A, 1.0 = point B), and redraws both halves of it -- see update_xsection_marker. Passing
        None hides the marker (used when there's no completed A/B line, or no cross-section open anywhere to
        show it on)."""
        self.xsection_marker_frac = None if frac is None else float(np.clip(frac, 0., 1.))
        self.update_xsection_marker()

    def update_xsection_marker(self):
        """Redraws the shared cross-section marker at the current self.xsection_marker_frac: a point on the
        A-B line on the map (self.visuals['ab_line_marker']) AND, simultaneously, a vertical line at the
        matching distance on EVERY panel that currently has its cross-section split-view open
        (self.visuals['cross_section_marker'][panel], see self.cross_section_layout) -- so dragging either one
        moves both at once. Hides both sides if there's no marker position set, or no completed A/B line to
        place it on."""
        frac = self.xsection_marker_frac
        have_line = self.ab_line_panel is not None and self.ab_line_b is not None
        if frac is None or not have_line:
            self.visuals['ab_line_marker'].visible = False
            for panel in self.cross_section_layout:
                self.visuals['cross_section_marker'][panel].visible = False
            return

        # Map side: linear interpolation between A and B in AEQD x/y km (the same coordinate system the line's
        # own endpoints are stored in -- see set_ab_line_point), converted to screen pixels the same way
        # draw_ab_line positions the line itself.
        xy = self.ab_line_a+frac*(self.ab_line_b-self.ab_line_a)
        screen_xy = self.xycoord_to_screen(self.ab_line_panel, xy)
        self.visuals['ab_line_marker'].set_data(pos=np.array([screen_xy]), face_color=(1,1,0,1),
            edge_color=(0,0,0,1), edge_width=self.scale_pixelsize(1.5), size=self.scale_pixelsize(11))
        self.visuals['ab_line_marker'].visible = True

        # Plot side: a vertical line at the corresponding x position, in every panel with an open cross-section
        # at once (there can be more than one, e.g. a Z panel and a V panel side by side -- see
        # show_cross_sections_for_ab_line), so the same distance-along-the-line is highlighted everywhere
        # simultaneously.
        for panel, layout in self.cross_section_layout.items():
            x = layout['img_left']+frac*(layout['img_right']-layout['img_left'])
            self.visuals['cross_section_marker'][panel].set_data(
                pos=np.array([[x, layout['img_top']], [x, layout['img_bottom']]], dtype='float32'))
            self.visuals['cross_section_marker'][panel].visible = True

    def get_xsection_marker_click_frac(self, screen_pos):
        """Hit-tests a mouse-press position against both halves of the shared cross-section marker's
        interactive area -- the currently-open cross-section image(s) and the A-B line on the map -- and, if
        it's a hit, returns (frac, drag_info) where frac is the corresponding distance-along-A-B fraction and
        drag_info records which side the drag should track for subsequent mouse-move events (see
        on_mouse_press/on_mouse_move). Returns (None, None) if the position doesn't hit either, or if there's
        no completed A/B line to begin with.

        Cross-section images are checked first: clicking anywhere inside an open cross-section's image
        (not just exactly on the current marker line) jumps the marker straight to that distance, which is a
        much easier target to hit than the thin marker line itself, and matches how one would expect a
        'click to move the marker here' plot to behave. The map's A-B line, being only ever a thin line (not
        an area), instead uses a small perpendicular-distance threshold around the line itself.
        """
        if self.ab_line_panel is None or self.ab_line_b is None:
            return None, None
        pos = np.array(screen_pos, dtype='float64')

        for panel, layout in self.cross_section_layout.items():
            if (layout['img_left'] <= pos[0] <= layout['img_right'] and
            layout['img_top'] <= pos[1] <= layout['img_bottom']):
                frac = (pos[0]-layout['img_left'])/(layout['img_right']-layout['img_left'])
                return float(np.clip(frac, 0., 1.)), {'mode':'plot', 'panel':panel}

        screen_a = self.xycoord_to_screen(self.ab_line_panel, self.ab_line_a)
        screen_b = self.xycoord_to_screen(self.ab_line_panel, self.ab_line_b)
        ab = screen_b-screen_a
        ab_len_sq = float(np.dot(ab, ab))
        if ab_len_sq > 0:
            t = float(np.dot(pos-screen_a, ab)/ab_len_sq)
            t_clipped = np.clip(t, 0., 1.)
            closest_point = screen_a+t_clipped*ab
            hit_radius = self.scale_pixelsize(12) # Generous click target, doesn't need to exactly match the
            #thin rendered line width itself -- same general approach as the cross-section '< Back'/toggle
            #button hit-tests elsewhere in this class.
            if np.linalg.norm(pos-closest_point) < hit_radius:
                return t_clipped, {'mode':'map'}

        return None, None

    def clear_ab_line(self):
        """Clears the current A/B line (its line, A/B/distance labels, and the per-panel cross-section toggle
        buttons that depend on it), and also closes any cross-section split-views that happen to be open --
        since those are driven by this same line, leaving them open after the line itself disappears would be
        confusing (a split-view with no way to re-derive what line it was showing). Bound to Delete (see
        nlr.py), as a single, predictable way to back out of the whole A/B-line/cross-section workflow at once.
        """
        self.hide_all_cross_sections() # Also resets the shared marker's state, since it has no line left to sit on.
        self.xsection_marker_dragging = None
        self.ab_line_panel = None
        self.ab_line_a = None
        self.ab_line_b = None
        self.draw_ab_line()
        self.update()
        # Force an IMMEDIATE repaint here, rather than only requesting one via self.update() (which just
        # schedules a redraw for the next regular Qt paint cycle) -- on at least one tested system, that
        # scheduled redraw wasn't visibly applied until some unrelated event (e.g. a scroll) came through and
        # forced a repaint anyway, leaving the 'A' label looking stuck on screen for longer than expected.
        # self.native is the underlying Qt widget for this vispy canvas; repaint() processes synchronously
        # instead of merely scheduling, and processEvents() flushes the Qt event queue so this doesn't have to
        # wait for whatever else is already pending.
        try:
            self.native.repaint()
            QApplication.processEvents()
        except Exception:
            pass

    def clear_volume3d_rect(self):
        """Wist de 3D-selectierechthoek (zie volume3d_rect_a/b) zonder een nieuwe te hoeven tekenen. Bound
        to CTRL+SHIFT+Delete (zie nlr.py) -- gewone Delete is al voor de A/B-lijn/cross-section in gebruik."""
        self.volume3d_rect_panel = None
        self.volume3d_rect_a = None
        self.volume3d_rect_b = None
        self.volume3d_rect_dragging = False
        self.draw_volume3d_rect()
        self.update()

    def toggle_cross_section(self, panel):
        if panel in self.cross_section_active_panels:
            self.hide_cross_section(panel)
        else:
            self.show_cross_section(panel)

    def toggle_cross_sections_for_ab_line(self):
        """Toggles the cross-section split-view for every panel with cross-section support (see
        cross_section_product_for_panel) at once -- used by the F2 keyboard shortcut, which (unlike a mouse
        click on a specific panel's '< Back'/'Show cross-section >>' label) has no single panel to target.
        If ANY supported panel currently has its cross-section open, this hides all of them; otherwise it
        shows all of them."""
        if self.cross_section_active_panels:
            self.hide_all_cross_sections()
        else:
            self.show_cross_sections_for_ab_line()

    def show_cross_sections_for_ab_line(self):
        """Activates the cross-section split-view for EVERY currently-displayed panel whose product has cross-
        section support (currently 'z' Reflectivity and 'v' Velocity -- see cross_section_product_for_panel),
        all driven by the single, shared A/B line (see set_ab_line_point). This is what the 'Show cross-section
        >>' click actually calls; show_cross_section(panel) itself only handles one panel at a time, so that it
        can also be used for re-showing/refreshing a single panel's cross-section independently (e.g. from the
        per-panel animation hook in set_newdata, or a possible future per-panel toggle)."""
        if self.ab_line_panel is None or self.ab_line_b is None:
            return
        for panel in self.panellist:
            if self.cross_section_product_for_panel(panel) is not None:
                self.show_cross_section(panel)

    def cross_section_product_for_panel(self, panel):
        """Returns the cross-section data product ('z' or 'v') appropriate for the product currently shown in
        the given panel, or None if that panel's product has no cross-section support. Currently a direct 1:1
        mapping (a 'z' panel gets a Reflectivity cross-section, a 'v' panel gets a Velocity cross-section), but
        kept as a separate lookup (rather than inlining 'self.crd.products[panel] in (\"z\",\"v\")' everywhere)
        in case support for additional products is added later.
        """
        # self.crd.products is a plain list/sequence indexed by panel number (like self.crd.scans), not a
        # dict -- it has no .get() method. An earlier version of this code assumed it was a dict (it isn't,
        # see e.g. self.crd.products[j] used throughout the rest of this file with plain index access), which
        # raised an AttributeError on every click of 'Show cross-section >>'.
        try:
            product = self.crd.products[panel]
        except (IndexError, KeyError):
            return None
        return product if product in ('z', 'v') else None

    def update_cross_section_toggle_buttons(self):
        """Positions and shows/hides the small per-panel cross-section toggle button (top-right corner of each
        panel) -- see self.visuals['cross_section_toggle_button'/'cross_section_toggle_button_frame'], set up
        in __init__. A button is shown for panel j only when BOTH: (1) j's current product has cross-section
        support (see cross_section_product_for_panel), and (2) a completed A/B line exists to drive it. Called
        from draw_ab_line (whenever the line itself changes) and from set_panel_sttransforms_and_clippers
        (whenever panel positions/sizes change, e.g. window resize or changing the panel count), so the
        buttons always end up in the correct top-right corner regardless of what changed.
        """
        have_line = self.ab_line_panel is not None and self.ab_line_b is not None
        for panel in range(self.max_panels):
            if panel not in self.visuals['cross_section_toggle_button']:
                continue # Can happen before __init__'s per-panel setup loop has run for this panel index.
            show_button = have_line and panel in self.panellist and self.cross_section_product_for_panel(panel) is not None
            if not show_button:
                self.visuals['cross_section_toggle_button'][panel].visible = False
                self.visuals['cross_section_toggle_button_frame'][panel].visible = False
                continue
            panel_corners = self.panel_corners[panel]
            # panel_corners order is [topleft, bottomleft, bottomright, topright] (see set_panel_info's
            # comment: "First corner is the top left one, and the other 3 are listed in counterclockwise
            # order"), so index 3 is the top-right corner.
            topright = panel_corners[3]
            margin = self.scale_pixelsize(8)
            btn_size = np.array([self.scale_pixelsize(100), self.scale_pixelsize(26)])
            # Plain text, no border/frame, matching the rest of the understated NLradar interface.
            btn_pos = topright+np.array([-margin-0.5*btn_size[0], margin])
            # Zelfde correctie als bij set_titles()/cbars_labels (6 juli 2026): anchor_y='top' wordt in
            # vispy 0.14.1 niet zuiver als bovenkant-ankerpunt behandeld, waardoor deze tekst te hoog stond.
            btn_pos = btn_pos + np.array([0, 1.5*self.scale_pointsize(eval(self.font_sizes['cbars_labels']))])
            self.visuals['cross_section_toggle_button'][panel].pos = btn_pos
            self.visuals['cross_section_toggle_button'][panel].visible = True
            self.visuals['cross_section_toggle_button_frame'][panel].visible = False

    def cross_section_visual(self, panel):
        """Returns the pre-built ImageVisual for the given panel matching the CURRENT
        self.gui.cross_section_interpolation_mode (see the 3-variants-per-panel setup in __init__) -- the
        single point every other method goes through to get/show/hide 'the' cross-section image, so that
        switching interpolation mode (see change_cross_section_interpolation_mode in nlr.py) never needs to
        create or mutate a visual, just pick a different, already-fully-wired one. getattr guards against an
        older stored_settings.pkl predating this setting."""
        mode = getattr(self.gui, 'cross_section_interpolation_mode', 'bicubic')
        if mode not in self.cross_section_interpolation_modes:
            mode = 'bicubic' # Guard against a corrupted/unrecognized stored value.
        return self.visuals['cross_section_'+mode][panel]

    def hide_other_cross_section_interpolation_visuals(self, panel):
        """Hides the OTHER (currently-unused) interpolation-mode visuals for this panel, so that at most one
        of the 3 pre-built variants (see cross_section_visual) is ever visible at once -- called from
        show_cross_section (right before showing the currently-selected one) and from
        change_cross_section_interpolation_mode (right after switching mode), so a stale, previously-visible
        variant from a different mode never lingers on screen underneath/behind the newly-selected one."""
        current = self.cross_section_visual(panel)
        for mode in self.cross_section_interpolation_modes:
            visual = self.visuals['cross_section_'+mode][panel]
            if visual is not current:
                visual.visible = False

    def show_cross_section(self, panel):
        """Activates the vertical cross-section view (Reflectivity or Velocity, depending on this panel's
        current product -- see cross_section_product_for_panel) along the current, shared A/B line (see
        set_ab_line_point) for the given panel specifically, splitting that one panel into a top half (the
        cross-section image) and bottom half (the normal radar view, cropped rather than replaced -- see
        further down) until hide_cross_section(panel) is called for it. Does nothing if no completed A/B line
        exists yet, or if this panel's product has no cross-section support."""
        if self.ab_line_panel is None or self.ab_line_b is None:
            return
        product = self.cross_section_product_for_panel(panel)
        if product is None:
            return
        n_samples = max(100, int(getattr(self.gui, 'cross_section_n_samples', 2000)))
        points = self.dsg.get_cross_section(product, self.ab_line_a, self.ab_line_b, n_samples=n_samples)
        if len(points) == 0:
            self.gui.set_textbar("No data found along this line.", 'red', 1)
            return

        distances, heights, values = (np.array([p[i] for p in points]) for i in range(3))
        # A flat 15% headroom margin on top of the 99th-percentile height (rather than using that percentile
        # directly as the top of the scale) gives a tall storm's overshooting top some empty space above it in
        # the rendered image, instead of being squeezed into just the topmost row or two of the n_height_bins
        # raster -- which is what made a genuinely tall, gradually-built-up tower (NOT a stray outlier; see the
        # per-column diagnostics, which show a smoothly rising sequence over many consecutive columns rather
        # than an isolated spike) appear to cut off abruptly right where the data actually just ran out of
        # vertical room to be drawn in.
        headroom_factor = 1. + max(0., getattr(self.gui, 'cross_section_height_headroom_percent', 15.))/100.
        max_height_km = max(8., np.percentile(heights, 99)*headroom_factor) # At least 8 km of headroom, but extend
        #further if genuinely high-altitude echo is present (e.g. a tall convective cell), so the
        #cross-section isn't needlessly cropped right at the top of a strong storm.
        line_length_km = np.linalg.norm(self.ab_line_b-self.ab_line_a)

        # Re-apply the current colormap/clim every time the cross-section is (re)shown, rather than relying
        # solely on the clim passed once at ImageVisual construction time (see __init__). self.clim_int[product]
        # can change at runtime (e.g. it depends on the current scan's Nyquist velocity for V, which differs
        # between scans/radars/moments), so a stale clim from whenever the app started would make the
        # cross-section's colors disagree with the normal panel right next to it -- in the most visible case,
        # real values well within the normal panel's range could end up clipped to the extreme end of a too-
        # narrow stale clim, making most of the image render as the same saturated end-of-scale color.
        # Re-assigning the cmap property (not just clim) also matters here: per the PolRGB color-transform fix
        # elsewhere in this codebase, vispy's ImageVisual only rebuilds its internal GLSL color-transform
        # function when the cmap property itself is reassigned, not merely when set_data/clim change -- so
        # skipping this would risk the visual keeping whatever color-transform function happened to be built
        # for its previous use.
        self.cross_section_visual(panel).cmap = self.cm1[product]
        self.cross_section_visual(panel).clim = self.clim_int[product]

        # Bin the scattered (distance, height, value) points onto a regular raster, since ImageVisual needs
        # a 2D grid, not a scattered point cloud. Resolution can't be pushed arbitrarily high, because the
        # number of available (distance, height) samples is limited by the number of elevation scans times
        # n_samples along the line -- a too-fine raster just produces a sparse, speckled result with mostly
        # empty bins (this was the root cause of the original "stray horizontal dashes" bug: back when
        # n_samples was still 300, a 400x200=80000-bin raster fed by only ~2000-2300 points was at best ~3%
        # filled, so nearly the whole image was empty/transparent except for a handful of isolated filled
        # pixels, which -- stretched across the full main-widget area -- looked like scattered dashed lines
        # rather than a filled cross-section). n_samples in get_velocity_cross_section/get_cross_section was
        # since raised to 2000 (see nlr_datasourcegeneral.py), giving far more raw points per elevation scan
        # than back then, which is what makes doubling the raster here (150x80 -> 300x160) safe to do without
        # reintroducing that same sparse/speckled look -- if the raster is ever pushed higher still, re-check
        # the per-column fill percentage (see filled_counts below) rather than assuming it stays fine.
        # Base 150x80 raster, scaled by the live-adjustable cross_section_resolution_factor (Settings ->
        # Miscellaneous -- see settings_tabmiscellaneous/change_cross_section_resolution_factor in nlr.py, the
        # same "tunable without editing code" pattern used for polrgb_params). getattr with a 1.0 fallback
        # guards against an older stored_settings.pkl that predates this setting.
        resolution_factor = getattr(self.gui, 'cross_section_resolution_factor', 1.0)
        n_dist_bins = max(1, int(round(150*resolution_factor)))
        n_height_bins = max(1, int(round(80*resolution_factor)))
        dist_bin = np.clip((distances/line_length_km*n_dist_bins).astype('int64'), 0, n_dist_bins-1)
        height_bin = np.clip((heights/max_height_km*n_height_bins).astype('int64'), 0, n_height_bins-1)

        raster = np.full((n_height_bins, n_dist_bins), np.nan, dtype='float32')
        # Where multiple points land in the same bin (more likely at long range, where beams from different
        # elevations can be close together), just keep the last one written -- a simple, cheap resolution
        # strategy that's good enough for this first version of the feature.
        raster[height_bin, dist_bin] = values

        # Vertical interpolation between scans: each elevation scan contributes a roughly-horizontal band of
        # filled bins (mostly empty above/below it). Linearly interpolating, per distance column, between the
        # nearest filled bin above and below any given empty bin turns those gaps into a smooth transition
        # between adjacent scans' values -- much closer to how a "real" vertical cross-section reads in other
        # radar software, instead of leaving hard physical gaps between every scan's band. Only the space
        # *between* the lowest and highest filled bin in a column is interpolated; above the highest scan and
        # below the lowest scan there's no bracketing data to interpolate from, so those stay empty (no
        # extrapolation -- we don't want to invent values where there's genuinely no nearby measurement).
        height_axis = np.arange(n_height_bins)
        # Diagnostic only: per-column count of filled rows and the topmost filled row BEFORE interpolation,
        # to distinguish a genuine scan-coverage edge (count drops to 0-1 and stays there) from a binning
        # artifact (count fluctuates/drops only briefly). Printed as a compact table rather than per-column,
        # to keep the console output readable.
        filled_counts = np.array([np.count_nonzero(~np.isnan(raster[:, col])) for col in range(n_dist_bins)])
        top_filled_row = np.array([
            (np.nonzero(~np.isnan(raster[:, col]))[0].min() if filled_counts[col] > 0 else -1)
            for col in range(n_dist_bins)])

        # Outlier filtering: a column whose topmost filled bin is far HIGHER (much higher row index, since at
        # this point in the code -- before the later raster[::-1] flip -- a higher row index means higher
        # altitude) than its neighbouring columns' tops almost certainly reflects a single noisy/stray point
        # (e.g. a second-trip echo or a mis-binned sample) rather than a real, physically-continuous high-
        # altitude feature -- a genuine tall convective cell shows up as a gradual rise over many columns, not
        # an isolated few-column spike that jumps far above the surrounding columns and then drops straight
        # back down (see the diagnostic above: a real run of e.g. '5,5,5,5,5' surrounded by an isolated
        # '53,54,54,54,55' before returning to '9,5,5,5,5' is exactly this pattern, and produced a hard,
        # physically-impossible-looking vertical 'step' in the rendered cross-section).
        #
        # For each column, compare its top against the MEDIAN top of a window of neighbouring columns. This is
        # done iteratively, excluding columns already flagged as outliers from later neighbour-median
        # calculations -- otherwise a multi-column-wide cluster of outliers (as in the example above, 5 columns
        # wide) would count each other as 'normal' neighbours and never get flagged, since the local median
        # would itself be dragged up by the very outliers it's trying to detect. The window is wide enough to
        # comfortably extend past such a cluster to reach genuinely unaffected columns on either side.
        window = 10 # Columns on each side to use for the local median.
        outlier_threshold = 8*n_height_bins//80 # Row-index difference beyond which a column's top is treated
        #as a stray outlier rather than a genuine local rise -- expressed relative to n_height_bins (originally
        #tuned as 8 rows out of 80) so it keeps representing the same physical height jump (roughly 1-2 km) if
        #n_height_bins is ever changed, rather than becoming twice as strict/lenient as intended. Chosen to
        #comfortably exceed the normal column-to-column variation seen in the diagnostic above (typically 1-3
        #rows out of 80) while still catching genuine jumps of several km.
        valid_cols = np.nonzero(top_filled_row >= 0)[0]
        is_outlier_col = np.zeros(n_dist_bins, dtype=bool)
        for _pass in range(3): # A few passes so a flagged outlier no longer pollutes its neighbours' medians.
            changed = False
            for col in valid_cols:
                if is_outlier_col[col]:
                    continue
                neighbour_idx = valid_cols[(valid_cols >= col-window) & (valid_cols <= col+window) & (valid_cols != col)]
                neighbour_idx = neighbour_idx[~is_outlier_col[neighbour_idx]]
                if neighbour_idx.size < 3:
                    continue # Too few reliable neighbours (e.g. right at the start/end of the line) to judge.
                local_median_top = np.median(top_filled_row[neighbour_idx])
                if top_filled_row[col] > local_median_top+outlier_threshold:
                    is_outlier_col[col] = True
                    changed = True
            if not changed:
                break

        # Only ever removes data -- it can't invent any. For each flagged column, drop bins above (i.e. with a
        # higher row index than) a freshly-recomputed local median (now excluding ALL flagged outlier columns,
        # not just the ones found before this particular column in the loop above), so the interpolation below
        # follows the surrounding, locally-consistent trend instead of jumping out to the stray point.
        for col in np.nonzero(is_outlier_col)[0]:
            neighbour_idx = valid_cols[(valid_cols >= col-window) & (valid_cols <= col+window) & (valid_cols != col)]
            neighbour_idx = neighbour_idx[~is_outlier_col[neighbour_idx]]
            if neighbour_idx.size == 0:
                continue
            local_median_top = np.median(top_filled_row[neighbour_idx])
            cutoff_row = int(round(local_median_top))
            raster[cutoff_row+1:, col] = np.nan

        for col in range(n_dist_bins):
            column = raster[:, col]
            filled_rows = np.nonzero(~np.isnan(column))[0]
            if filled_rows.size < 2:
                continue # Nothing to interpolate between with 0 or 1 filled bins in this column.
            lo, hi = filled_rows.min(), filled_rows.max()
            raster[lo:hi+1, col] = np.interp(height_axis[lo:hi+1], filled_rows, column[filled_rows])

        # Gap-fill: a final, light touch-up for any remaining isolated empty bins not resolved by the vertical
        # interpolation above (e.g. right at the left/right edge of a scan's azimuthal coverage, or a column
        # that had too few filled bins to interpolate). A few rounds of "fill empty bin from a filled neighbour"
        # closes these small gaps, without smearing data across genuinely large empty regions (e.g. above the
        # highest scan's coverage, which intentionally stays empty after the interpolation step above too).
        for _ in range(3):
            nan_mask = np.isnan(raster)
            if not nan_mask.any():
                break
            filled = raster.copy()
            # Average over up/down/left/right neighbours that do have data; leave bins untouched if none of
            # their neighbours are filled either (so large gaps shrink gradually instead of being bridged in
            # one step, and truly empty regions -- e.g. far above the radar's highest scan -- stay empty).
            neighbour_sum = np.zeros_like(raster); neighbour_count = np.zeros_like(raster)
            for shift, axis in ((1,0), (-1,0), (1,1), (-1,1)):
                shifted = np.roll(raster, shift, axis=axis)
                valid = ~np.isnan(shifted)
                # np.roll wraps around; mask off the wrapped-around edge so it doesn't pull data from the
                # opposite side of the raster.
                if axis == 0:
                    if shift == 1: valid[0, :] = False
                    else: valid[-1, :] = False
                else:
                    if shift == 1: valid[:, 0] = False
                    else: valid[:, -1] = False
                neighbour_sum[valid] += shifted[valid]
                neighbour_count[valid] += 1
            fillable = nan_mask & (neighbour_count > 0)
            filled[fillable] = neighbour_sum[fillable]/neighbour_count[fillable]
            raster = filled

        raster = raster[::-1] # Row 0 should be the TOP (highest height) for image display, matching every
        #other image visual's top-to-bottom = high-to-low-y convention used throughout this codebase.

        raster_uint = self.dsg.convert_dtype_float_to_uint(np.where(np.isnan(raster), self.mask_values.get(product, -1e6), raster), product)
        self.cross_section_visual(panel).set_data(raster_uint)

        # Split-panel layout: the panel area is divided 50/50 into a top half (this cross-section image) and a
        # bottom half (self.visuals['cross_section_overview'], the normal radar image for this panel/scan --
        # see further down). A small gap between the two halves keeps the white frame lines (cross_section_frame,
        # set up below) from visually touching.
        panel_corners = self.panel_corners[panel]
        panel_topleft, panel_bottomright = panel_corners[0], panel_corners[2]
        panel_w, panel_h = panel_bottomright-panel_topleft
        half_gap = self.scale_pixelsize(4)
        top_half_topleft = panel_topleft
        top_half_bottomright = panel_topleft+np.array([panel_w, panel_h/2-half_gap/2])
        bottom_half_topleft = panel_topleft+np.array([0, panel_h/2+half_gap/2])
        bottom_half_bottomright = panel_bottomright

        topleft, bottomright = top_half_topleft, top_half_bottomright
        panel_w, panel_h = bottomright-topleft
        margin = self.scale_pixelsize(75) # Room for the axis line, tick marks, and tick labels around the
        #cross-section image itself -- enlarged from an earlier, narrower margin that let height label text
        #(e.g. '10 km') extend into the image area on the left side.

        # Solid dark background for the entire top half (see self.visuals['cross_section_background'] in
        # __init__), covering the shared map visual so streets/place names don't show through wherever the
        # cross-section raster itself is transparent (e.g. empty/NaN bins with no nearby scan data).
        self.visuals['cross_section_background'][panel].transform.translate = 0.5*(topleft+bottomright)
        self.visuals['cross_section_background'][panel].transform.scale = (abs(panel_w), abs(panel_h))
        self.visuals['cross_section_background'][panel].visible = True

        self.cross_section_visual(panel).transform = STTransform(
            scale=((abs(panel_w)-2*margin)/n_dist_bins, (abs(panel_h)-2*margin)/n_height_bins),
            translate=(topleft[0]+margin, topleft[1]+margin))
        # Hide the other 2 (currently-unselected) interpolation-mode variants for this panel BEFORE showing
        # the selected one, so switching mode never briefly leaves 2 overlapping images visible at once.
        self.hide_other_cross_section_interpolation_visuals(panel)
        self.cross_section_visual(panel).visible = True

        # Axis labels: ticks every 12.5% along both axes (9 ticks total per axis), plus a short axis-name label
        # past each end, for easier reading of intermediate values without having to interpolate by eye between
        # widely-spaced ticks. Also draws a simple axis line with small perpendicular tick marks along the
        # bottom (distance) and left (height) edges of the cross-section image -- without these, the numbers
        # were floating with no clear reference to which exact row/column they belonged to, and with the
        # previous, narrower margin the height labels' text could extend into the image area itself rather
        # than staying in the margin beside it.
        tick_fracs = np.linspace(0, 1, 9)
        ticks_text, ticks_pos = [], []
        axislines_pos = []
        tick_len = self.scale_pixelsize(5) # Length of each small perpendicular tick mark.
        img_left, img_right = topleft[0]+margin, topleft[0]+margin+(abs(panel_w)-2*margin)
        img_top, img_bottom = topleft[1]+margin, topleft[1]+margin+(abs(panel_h)-2*margin)

        # Remember this panel's image bounds in screen-pixel space, for the shared cross-section marker (see
        # set_xsection_marker_frac/update_xsection_marker): needed both to draw its vertical line at the right
        # x position and to hit-test clicks/drags inside the image (see get_xsection_marker_click_frac). Kept
        # up to date on every (re)show of the cross-section, e.g. after a panel resize.
        self.cross_section_layout[panel] = dict(img_left=img_left, img_right=img_right, img_top=img_top, img_bottom=img_bottom)

        # X axis (distance): a horizontal line just below the image, with a short tick mark hanging down from
        # it at every tick_frac position, and the distance labels below those. The unit (km) is given once, in
        # the axis-name label, rather than repeated after every individual number.
        axislines_pos += [[img_left, img_bottom], [img_right, img_bottom]] # The axis line itself.
        for frac in tick_fracs:
            d_km = frac*line_length_km
            x = img_left+frac*(img_right-img_left)
            axislines_pos += [[x, img_bottom], [x, img_bottom+tick_len]] # One tick mark.
            ticks_text.append(str(ft.rifdot0(ft.r1dec(d_km))))
            ticks_pos.append([x, img_bottom+tick_len+self.scale_pixelsize(14)])
        ticks_text.append('Distance along A-B (km)')
        ticks_pos.append([0.5*(img_left+img_right), img_bottom+self.scale_pixelsize(36)])

        # Y axis (height): a vertical line just left of the image, with a short tick mark sticking out to the
        # left at every tick_frac position, and the height labels further left of those (clear of the image).
        axislines_pos += [[img_left, img_top], [img_left, img_bottom]] # The axis line itself.
        height_label_x = img_left-tick_len-self.scale_pixelsize(16)
        for frac in tick_fracs:
            h_km = frac*max_height_km
            y = img_top+(1-frac)*(img_bottom-img_top)
            axislines_pos += [[img_left, y], [img_left-tick_len, y]] # One tick mark.
            ticks_text.append(str(ft.rifdot0(ft.r1dec(h_km))))
            ticks_pos.append([height_label_x, y])
        ticks_text.append('Height (km)')
        ticks_pos.append([height_label_x, img_top-self.scale_pixelsize(20)])

        self.visuals['cross_section_ticks'][panel].text = ticks_text
        self.visuals['cross_section_ticks'][panel].pos = np.array(ticks_pos)
        self.visuals['cross_section_axislines'][panel].set_data(pos=np.array(axislines_pos, dtype='float32'), connect='segments')
        self.visuals['cross_section_axislines'][panel].visible = True

        # Back button: plain text, no border/frame, matching the rest of the understated NLradar interface
        # (e.g. the height/distance axis labels use the same approach -- bare text, no box around it).
        back_size = np.array([self.scale_pixelsize(80), self.scale_pixelsize(26)])
        back_center = topleft+np.array([self.scale_pixelsize(8)+0.5*back_size[0], self.scale_pixelsize(8)])
        # Zelfde correctie als bij set_titles()/cbars_labels (6 juli 2026): anchor_y='top' compensatie.
        back_center = back_center + np.array([0, 1.5*self.scale_pointsize(eval(self.font_sizes['cbars_labels']))])
        self.visuals['cross_section_back'][panel].pos = back_center
        self.visuals['cross_section_back'][panel].visible = True
        self.visuals['cross_section_back_frame'][panel].visible = False

        # Title/timestamp above the cross-section image, e.g. "Velocity cross-section  17:03:47Z" -- mirroring
        # the HH:MM:SSZ-style time format used by the normal panel titles (see self.scantimes[j], a
        # 'HH:MM:SS-HH:MM:SS' start-end string; only the start time is shown here, matching how e.g. the cursor
        # readout in the main title bar abbreviates it elsewhere). Positioned just INSIDE the top edge of the
        # panel (anchor_y='top', a small positive offset down from topleft[1]) -- placing it above topleft[1]
        # (i.e. outside the panel) put it in the main title bar's own space, where it was invisible/clipped.
        scantime_str = self.dsg.scantimes.get(panel, '')
        scantime_short = scantime_str.split('-')[0]+'Z' if scantime_str else ''
        product_title = {'v': 'Velocity', 'z': 'Reflectivity'}.get(product, gv.productnames.get(product, product))
        title_text = product_title+' cross-section'+((' '*2+scantime_short) if scantime_short else '')
        self.visuals['cross_section_title'][panel].text = title_text
        # Zelfde correctie als bij set_titles()/cbars_labels (6 juli 2026): anchor_y='top' compensatie.
        self.visuals['cross_section_title'][panel].pos = np.array([0.5*(topleft[0]+bottomright[0]), topleft[1]+self.scale_pixelsize(4)+2.0*self.scale_pointsize(eval(self.font_sizes['cbars_labels']))])
        self.visuals['cross_section_title'][panel].visible = True

        # Bottom half: simply clip the existing, normal radar_polar/radar_cartesian/map visuals for this panel
        # down to the bottom half, rather than hiding them and building a separate overview image. This keeps
        # the panel's actual pan/zoom state (self.panels_sttransforms[panel]) completely untouched -- panning
        # and zooming the bottom half therefore works exactly like it always does on a normal, non-split panel,
        # since nothing about that machinery changed; only the visible CROP changes. Four earlier attempts at
        # rebuilding an independent overview (via composed vispy transforms, then via a from-scratch CPU-side
        # raster) tried to make the bottom half show the data scaled/positioned to fit -- each approach put it
        # at a visibly wrong location for reasons that resisted explanation even after algebraic and empirical
        # verification, or (the CPU-side raster) ended up showing a fixed, non-scrollable view unrelated to the
        # current pan/zoom, which doesn't match what's actually wanted: the same scrollable, correctly
        # positioned map/radar view the panel already had, just cropped to the smaller area.
        #
        # self.clippers[panel] is shared by every visual in self.visuals_panels for this panel (map,
        # radar_polar, radar_cartesian, grid/heightring lines, etc. -- see the .attach() calls in __init__), so
        # narrowing it affects all of them together, which is fine here since none of those other elements are
        # being used differently while the cross-section is shown.
        panel_bounds_bottom_half = np.array([
            bottom_half_topleft[0], self.size[1]-bottom_half_bottomright[1],
            bottom_half_bottomright[0]-bottom_half_topleft[0], bottom_half_bottomright[1]-bottom_half_topleft[1]])
        self.clippers[panel].bounds = tuple(panel_bounds_bottom_half*self.gui.screen_pixel_ratio())

        # White frame lines around both halves (LineVisual with 2 separate rectangles, 5 points each -- the same
        # pattern used by self.visuals['panel_borders'], see set_panel_borders).
        def rect_points(tl, br):
            return [np.array([tl[0],tl[1]]), np.array([tl[0],br[1]]), np.array([br[0],br[1]]), np.array([br[0],tl[1]]), np.array([tl[0],tl[1]])]
        frame_pos = np.array(rect_points(top_half_topleft, top_half_bottomright)+rect_points(bottom_half_topleft, bottom_half_bottomright), dtype='float32')
        frame_connect = np.ones(10, dtype='bool')
        frame_connect[4] = False
        frame_connect[9] = False
        self.visuals['cross_section_frame'][panel].set_data(pos=frame_pos, connect=frame_connect)
        self.visuals['cross_section_frame'][panel].visible = True

        self.cross_section_active_panels.add(panel)

        # Show the shared cross-section marker right away, defaulting to the midpoint of the line the very
        # first time it appears (rather than requiring an initial click just to make it visible at all) --
        # see set_xsection_marker_frac/update_xsection_marker. If a marker position was already set (e.g. this
        # is a second panel's cross-section opening while one is already shown elsewhere, or a resize re-running
        # show_cross_section), that existing position is kept instead of being reset to the middle again.
        if self.xsection_marker_frac is None:
            self.xsection_marker_frac = 0.5
        self.update_xsection_marker()

        self.update()

    def hide_cross_section(self, panel):
        if panel in self.cross_section_active_panels:
            for radar_image in ('radar_polar', 'radar_cartesian'):
                if panel in self.visuals[radar_image]:
                    self.visuals[radar_image][panel].visible = (self.data_attr['proj'].get(panel) ==
                        ('pol' if radar_image == 'radar_polar' else 'car'))
            # Restore the clipper to the panel's full bounds (see show_cross_section, which narrows it to just
            # the bottom half) -- otherwise the panel would stay cropped to that half even after leaving the
            # cross-section view.
            self.clippers[panel].bounds = tuple(self.panel_bounds[panel]*self.gui.screen_pixel_ratio())
            self.cross_section_active_panels.discard(panel)
        for mode in self.cross_section_interpolation_modes:
            self.visuals['cross_section_'+mode][panel].visible = False
        self.visuals['cross_section_ticks'][panel].text = []
        self.visuals['cross_section_axislines'][panel].visible = False
        self.visuals['cross_section_marker'][panel].visible = False
        self.visuals['cross_section_back'][panel].visible = False
        self.visuals['cross_section_back_frame'][panel].visible = False
        self.visuals['cross_section_title'][panel].visible = False
        self.visuals['cross_section_background'][panel].visible = False
        self.visuals['cross_section_frame'][panel].visible = False
        self.cross_section_layout.pop(panel, None)
        if not self.cross_section_layout:
            # No panel has a cross-section open anymore -- hide the map-side marker too and forget its
            # position, so a freshly (re)opened cross-section later starts back at the midpoint default
            # (see show_cross_section) rather than resuming some now-stale, possibly off-screen position.
            self.xsection_marker_frac = None
            self.xsection_marker_dragging = None
            self.visuals['ab_line_marker'].visible = False
        self.update()

    def hide_all_cross_sections(self):
        """Hides the cross-section split-view for every panel where it's currently active (e.g. for the '< Back'
        click, which should leave every split panel -- not just one -- back in its normal, unsplit state)."""
        for panel in list(self.cross_section_active_panels):
            self.hide_cross_section(panel)

    def on_mouse_press(self, ev):
        if not ft.point_inside_rectangle(ev.pos,self.wpos['main'])[0] and not (
        self.gui.show_vwp and ft.point_inside_rectangle(ev.pos,self.wpos['vwp'])[0]):
            return
        self.last_mousepress_pos=np.array(ev.pos)
        self.mouse_moved_after_press=False
        
        if self.gui.show_vwp and ft.point_inside_rectangle(ev.pos,self.wpos['vwp'])[0]:
            #No more steps needed in this case
            return

        for panel in list(self.cross_section_active_panels):
            if self.visuals['cross_section_back'][panel].visible:
                # back_pos is the text's own anchor point: anchor_x='center'/anchor_y='top', so it's the
                # top-center point of the button (see show_cross_section, which draws the box symmetrically
                # around this same point) -- the hit-test box below must use the same convention.
                back_pos = np.array(self.visuals['cross_section_back'][panel].pos).flatten()[:2]
                # Zelfde ruimere klikzone als bij de X-section-knop hierboven (6 juli 2026).
                back_size = np.array([self.scale_pixelsize(100), self.scale_pixelsize(50)])
                back_hit_center_offset = np.array([0, 0.5*back_size[1]-self.scale_pixelsize(15)])
                hit = np.all(np.abs(np.array(ev.pos)-back_pos-back_hit_center_offset) < 0.5*back_size)
                if hit:
                    self.hide_cross_section(panel)
                    return

        # Small per-panel cross-section toggle button, top-right corner of each panel that currently has a
        # cross-section-capable product (Z or V) shown and a completed A/B line to work from -- see
        # update_cross_section_toggle_buttons. Checked for every panel in panellist, not just ones with an
        # active cross-section, since this is also how the cross-section gets turned ON in the first place.
        for panel in self.panellist:
            if panel in self.visuals['cross_section_toggle_button'] and self.visuals['cross_section_toggle_button'][panel].visible:
                # btn_pos is the text's own anchor point: anchor_x='center'/anchor_y='top' (see
                # update_cross_section_toggle_buttons, which draws the box symmetrically around this same
                # point), so the hit-test box needs no x-offset, only a y-offset to its vertical center.
                btn_pos = np.array(self.visuals['cross_section_toggle_button'][panel].pos).flatten()[:2]
                # Klikzone ruimer gemaakt (6 juli 2026): na de anchor_y-positiecorrectie bleek de klik net
                # buiten de oorspronkelijk kleine hit-box te vallen (~10-20 px verschil). In plaats van het
                # exacte pixel-verschil te blijven najagen, is de klikzone zelf simpelweg vergroot, met wat
                # extra marge naar boven toe (waar de klikken structureel net buiten vielen).
                btn_size = np.array([self.scale_pixelsize(120), self.scale_pixelsize(50)])
                btn_hit_center_offset = np.array([0, 0.5*btn_size[1]-self.scale_pixelsize(15)])
                hit = np.all(np.abs(np.array(ev.pos)-btn_pos-btn_hit_center_offset) < 0.5*btn_size)
                if hit:
                    self.toggle_cross_section(panel)
                    return

        modifiers = QApplication.keyboardModifiers()
        if ev.button==1 and bool(modifiers & Qt.ShiftModifier) and bool(modifiers & Qt.ControlModifier):
            # Sleep-rechthoek (bijv. voor de 3D-volumeweergave, CTRL+SHIFT+4 in nlr.py): CTRL+SHIFT+links-
            # klik-en-slepen. Gecontroleerd VOOR de gewone Shift-only A/B-lijn hieronder, want een bitwise
            # check op alleen Qt.ShiftModifier (zoals die A/B-lijn-check gebruikt) is ook True wanneer Ctrl
            # ernaast wordt ingedrukt -- zonder deze volgorde zou CTRL+SHIFT+klik dus per ongeluk de A/B-lijn
            # activeren in plaats van de sleep-rechthoek.
            panel = self.get_panel_for_position(ev.pos)
            xy = self.screencoord_to_xy(np.array(ev.pos), panel)
            self.volume3d_rect_panel = panel
            self.volume3d_rect_a = xy
            self.volume3d_rect_b = xy # Zelfde punt als startwaarde, zodat er meteen (een nulgroot) een
            #rechthoek bestaat om live bij te werken tijdens het slepen, i.p.v. pas na de eerste move.
            self.volume3d_rect_dragging = True
            self.draw_volume3d_rect()
            return

        if ev.button==1 and bool(modifiers & Qt.ShiftModifier):
            # A/B line tool: first Shift+click sets point A, second sets point B (and finalizes the line).
            # A further Shift+click after that starts a fresh line, discarding the old one -- simpler and more
            # predictable than requiring an explicit 'clear' action for what's meant to be a quick, lightweight
            # measuring tool.
            # Using a bitwise check here (rather than strict equality with Qt.ShiftModifier) so this still
            # registers even if Qt reports Shift alongside some other incidental modifier flag.
            self.set_ab_line_point(ev.pos)
            return

        # Shared cross-section marker (see set_xsection_marker_frac/get_xsection_marker_click_frac): a plain,
        # non-Shift left click/drag either inside an open cross-section image or near the A-B line on the map
        # grabs (and, via on_mouse_move, drags) the marker instead of starting the usual pan. Checked only
        # without Shift held, so Shift+click always keeps its existing meaning (setting a new A/B line point)
        # even if it happens to land close to the marker or line.
        if ev.button==1 and not bool(modifiers & Qt.ShiftModifier):
            frac, drag_info = self.get_xsection_marker_click_frac(ev.pos)
            if frac is not None:
                self.xsection_marker_dragging = drag_info
                self.set_xsection_marker_frac(frac)
                self.update()
                return

        if ev.button==1: self.mouse_hold_left=True; self.gridheightrings_removed=False
        else: self.mouse_hold_right=True
        self.mouse_hold=True
        
        self.check_presence_near_radars(ev.pos)
        self.check_presence_near_pos_markers(ev.pos)
        
        self.panel=self.get_panel_for_position(ev.pos)
        
    def on_mouse_release(self, ev):     
        if self.volume3d_rect_dragging:
            # Rondt het slepen af: volume3d_rect_a/b blijven staan (net als een afgeronde A/B-lijn) totdat
            # een nieuwe CTRL+SHIFT-sleep begint, zodat nlr.py (test_volume_grid_export/show_volume_3d_viewer)
            # ze kan uitlezen. mouse_hold_left/right/mouse_hold werden voor deze press niet True gezet (de
            # vroege 'return' in on_mouse_press slaat dat over), dus er is verder niets te herstellen.
            self.volume3d_rect_dragging = False
            return

        if self.xsection_marker_dragging is not None:
            # Ends a marker drag started in on_mouse_press (see get_xsection_marker_click_frac/on_mouse_move).
            # mouse_hold_left/right/mouse_hold were never set to True for this press in the first place (the
            # early 'return' there skips that), so there's nothing else for this release to undo.
            self.xsection_marker_dragging = None
            return

        self.mouse_hold_left=False; self.mouse_hold_right=False; self.mouse_hold=False
        
        if self.firstplot_performed and any([j in self.gui.lines_show for j in ('grid','heightrings')]) and\
        self.mouse_moved_after_press:
            if self.gui.showgridheightrings_panzoom: 
                self.update_gridheightrings(ev)
            else:
                self.set_timer_setback_gridheightrings()
                
        self.update_data_readout()

        # NIEUW (23 juli, op Eriks verzoek, vervangt het eerdere zwevende tooltipje): toon een
        # echt pop-up-venster met de HCLASS/MESH-brongegevens bij een gewone, "schone" klik -
        # geen sleep (mouse_moved_after_press), geen Shift/Ctrl (die hebben al hun eigen
        # betekenis, zie on_mouse_press), en niet vlak bij een radar-markering (die heeft zijn
        # eigen click-gedrag hieronder). self.hclass_tooltip_text/self.mesh_tooltip_text zijn
        # net gevuld door de aanroep van update_data_readout() hierboven.
        modifiers = QApplication.keyboardModifiers()
        if (ev.button == 1 and not self.mouse_moved_after_press
                and modifiers == Qt.NoModifier
                and self.radar_mouse_selected in (None, self.crd.selected_radar)):
            self.show_extra_info_popup(ev.pos)

        #button=1 refers to the left mouse button, 2 to the right one
        if ev.button==1:
            self.mouse_hold_left=False
            if self.radar_mouse_selected not in (None, self.crd.selected_radar) and not self.mouse_moved_after_press:
                modifiers = QApplication.keyboardModifiers()
                if modifiers == Qt.ControlModifier:
                    self.crd.change_selected_radar(self.radar_mouse_selected)
                else:
                    self.crd.change_radar(self.radar_mouse_selected)
        elif ev.button==2:
            self.mouse_hold_right=False
            # AANGEPAST (23 juli): het verbergen van het click_marker-puntje gebeurt niet meer
            # hier, maar centraal in showrightclickMenu zelf (nlr.py) - die functie wordt namelijk
            # OOK rechtstreeks door Qt's eigen customContextMenuRequested-signaal aangeroepen, dus
            # een check alleen hier bleek niet voldoende (zie de uitgebreide toelichting daar).
            if self.gui.need_rightclickmenu: 
                self.gui.showrightclickMenu(self.gui.rightmouseclick_Qpos)
 
    def on_mouse_move(self, ev): 
        if hasattr(self, 'previous_mouse_position') and (ev.pos == self.previous_mouse_position).all():
            # With my voice dictation software it sometimes happens that mouse move signals are emitted while the mouse is in fact not moving
            return
        self.previous_mouse_position=np.array(ev.pos)
        self.mouse_moved_after_press=True

        if self.volume3d_rect_dragging:
            # Actief slepen van de 3D-selectierechthoek (gestart in on_mouse_press): alleen punt B bijwerken
            # en opnieuw tekenen, geen normale pan/zoom-afhandeling -- vroege return net als bij
            # xsection_marker_dragging hieronder.
            self.volume3d_rect_b = self.screencoord_to_xy(np.array(ev.pos), self.volume3d_rect_panel)
            self.draw_volume3d_rect()
            return

        if self.xsection_marker_dragging is not None:
            # Actively dragging the shared cross-section marker (started in on_mouse_press -- see
            # get_xsection_marker_click_frac), on whichever side ('plot' or 'map') the drag began on. Unlike
            # the initial press, this deliberately does NOT re-run the same hit-test / re-check whether the
            # cursor is still strictly inside the image or within the line's hit radius -- once grabbed, the
            # marker should keep following the cursor smoothly even if it briefly strays slightly outside
            # those bounds while dragging, exactly like e.g. a slider handle would.
            mode = self.xsection_marker_dragging['mode']
            if mode == 'plot':
                panel = self.xsection_marker_dragging['panel']
                layout = self.cross_section_layout.get(panel)
                if layout is not None: # Guards against the cross-section having been closed mid-drag.
                    frac = (ev.pos[0]-layout['img_left'])/(layout['img_right']-layout['img_left'])
                    self.set_xsection_marker_frac(frac)
            elif mode == 'map' and self.ab_line_panel is not None and self.ab_line_b is not None:
                screen_a = self.xycoord_to_screen(self.ab_line_panel, self.ab_line_a)
                screen_b = self.xycoord_to_screen(self.ab_line_panel, self.ab_line_b)
                ab = screen_b-screen_a
                ab_len_sq = float(np.dot(ab, ab))
                if ab_len_sq > 0:
                    t = float(np.dot(np.array(ev.pos, dtype='float64')-screen_a, ab)/ab_len_sq)
                    self.set_xsection_marker_frac(t)
            self.update()
            return

        if self.mouse_hold_left:
            for j in self.panellist:
                if j == 0:
                    self.panels_sttransforms[0].move(np.array(ev.pos)-ev.last_event.pos)
                else:
                    self.panels_sttransforms[j].translate = self.panels_sttransforms[0].translate[:2]+(self.panel_centers[j]-self.panel_centers[0])
            if self.ab_line_panel is not None:
                self.draw_ab_line()

        elif self.mouse_hold_right:
            p1c = np.array(ev.last_event.pos)[:2]
            p2c = np.array(ev.pos)[:2]
            zoomfactor = (1 + self.zoomfactor_vispy) ** ((p2c-p1c) * np.array([1, -1]))
            zoomfactor=zoomfactor[0]*zoomfactor[1]
            
            pos=ev.press_event.pos
            selected_panel=self.get_panel_for_position(pos)
            rel_pos=pos-self.panel_centers[selected_panel]
            for j in range(self.max_panels): #Zooming the transforms for all panels is necessary, to ensure that all panels keep showing the same area
                self.panels_sttransforms[j].zoom((zoomfactor,zoomfactor),center=self.panel_centers[j]+rel_pos,mapped=True)
            if self.ab_line_panel is not None:
                self.draw_ab_line()
                                
        if self.mouse_hold:
            #Used in the function self.dsg.check_need_scans_change.
            self.dsg.time_last_panzoom=pytime.time()
            
            if self.timer_setback_gridheightrings_running:
                #It is possible that a timer is running when moving the mouse just after using the mouse wheel (where the timer is set), which
                #is stopped here.
                self.timer_setback_gridheightrings.stop()
            if self.firstplot_performed and not self.gui.showgridheightrings_panzoom and (
            any([j in self.gui.lines_show for j in ('grid','heightrings')])) and not self.gridheightrings_removed:
                self.remove_gridheightrings()
                      
            self.set_draw_action('panning_zooming')
            self.update_map_tiles_ondraw = True
            self.update()
        else:
            if not ft.point_inside_rectangle(ev.pos,self.wpos['main'])[0]:
                self.gui.plotwidget.setCursor(Qt.ArrowCursor)
                return
            self.last_mouse_pos_px = ev.pos
            self.check_presence_near_radars(ev.pos)
            self.check_presence_near_pos_markers(ev.pos)
            self.update_data_readout()
            
    def show_extra_info_popup(self, pos):
        """Toont een klein, echt pop-up-venster met de HCLASS- of MESH-brongegevens (Z/ZDR/KDP/
        CC/0C/-20C/roosterpunt/temperatuurbron/tijd) - vervangt sinds 23 juli (op Eriks verzoek)
        het eerdere zwevende QToolTip-tooltipje, dat na meerdere pogingen (leaveEvent-hook, vaste
        zichtbaarheidsduur) niet naar tevredenheid bleef werken. Wordt aangeroepen vanuit
        on_mouse_release bij een gewone, niet-gesleepte klik. self.hclass_tooltip_text/
        self.mesh_tooltip_text worden gevuld door update_data_readout(), die daar vlak voor deze
        aanroep al is uitgevoerd - als geen van beide iets bevat (bv. geklikt buiten een geldige
        HCLASS/MESH-classificatie, of een ander product), gebeurt hier niets.

        pos: het scherm-coordinaat van de klik (ev.pos uit on_mouse_release), voor het
        achterblijvende puntje (self.visuals['click_marker'] - zie __init__ en on_mouse_release
        voor het weer verbergen via een rechtsklik).
        """
        tekst = self.hclass_tooltip_text if self.hclass_tooltip_text is not None else (
            self.mesh_tooltip_text if self.mesh_tooltip_text is not None else (
            self.posh_tooltip_text if self.posh_tooltip_text is not None else (
            self.poh_tooltip_text if self.poh_tooltip_text is not None else self.shi_tooltip_text)))
        if tekst is None:
            return
        titel = 'HCLASS' if self.hclass_tooltip_text is not None else (
            'MESH' if self.mesh_tooltip_text is not None else (
            'POSH' if self.posh_tooltip_text is not None else (
            'POH' if self.poh_tooltip_text is not None else 'SHI')))

        # Puntje op de kliklocatie neerzetten VOOR het (modale) pop-up-venster wordt geopend, zodat
        # het al zichtbaar is terwijl je de pop-up leest, en blijft staan nadat je 'm hebt weggeklikt.
        self.visuals['click_marker'].set_data(pos=np.array([pos]), face_color=(1,0,0,1),
            edge_color=(1,1,1,1), edge_width=self.scale_pixelsize(1.5), size=self.scale_pixelsize(11))
        self.visuals['click_marker'].visible = True
        self.update()

        msgbox = QMessageBox(QMessageBox.NoIcon, titel, tekst, QMessageBox.Ok, self.native)
        msgbox.exec_()

    def update_data_readout(self):
        pos = self.last_mouse_pos_px
        if pos is None:
            pos = self.wcenter['main']
        
        if not ft.point_inside_rectangle(pos,self.wpos['main'])[0]: return
        
        panel=self.get_panel_for_position(pos)
        xy_coord=self.screencoord_to_xy(pos)
        
        radius=np.linalg.norm(xy_coord)
        
        polrgb_text = None
        hclass_text = None
        # OMBOUW (23 juli, op Eriks verzoek): waren lokale variabelen (hclass_tooltip_text/
        # mesh_tooltip_text), alleen gebruikt voor het zwevende QToolTip-tooltipje hieronder.
        # Erik wilde af van dat tooltipje (bleef ondanks meerdere pogingen niet prettig
        # verdwijnen/verschijnen) - vervangen door een echt pop-up-venster bij een klik (zie
        # on_mouse_release). Als instance-attributen opgeslagen zodat on_mouse_release ze na
        # deze aanroep nog kan uitlezen.
        self.hclass_tooltip_text = None
        self.mesh_tooltip_text = None
        self.posh_tooltip_text = None
        self.poh_tooltip_text = None
        self.shi_tooltip_text = None
        if self.firstplot_performed:
            try:
                product=self.data_attr['product'][panel]
                if not product in gv.plain_products: 
                    scanangle=self.data_attr['scanangle'][panel]
                    
                if self.data_attr['proj'][panel] == 'pol':
                    radial_res=self.data_attr['radial_res'][panel]
                    azimuthal_res=self.data_attr['azimuthal_res'][panel]
                    azimuth=ft.azimuthal_angle(xy_coord, deg=True)
                    row=int(np.floor(np.mod(azimuth-self.dsg.data_azimuth_offset[panel], 360)/azimuthal_res))
                    # row_rawdata: index voor polrgb_raw_data/hclass_raw_data (nlr_datasourcegeneral.py),
                    # die WEL dezelfde azimutale binning hebben maar NIET de onderstaande +1-rij-padding,
                    # omdat die caches al gevuld worden vóórdat die padding verderop wordt toegepast op
                    # self.dsg.data[panel] zelf. Zonder deze aparte index zou de uitlezing structureel een
                    # rij verschoven zijn t.o.v. het getoonde beeld (gemeld door Erik, 22 juli: hagel op het
                    # scherm gaf "Rain" in de tooltip).
                    row_rawdata = row
                    if self.dsg.data[panel].shape[0] > self.data_attr['azimuthal_bins'][panel]:
                        row += 1 #+1 for the added radials for interpolation
                    
                    if product in gv.plain_products or scanangle == 90.: 
                        col=int(np.floor((radius-self.dsg.data_radius_offset[panel])/radial_res))
                        self.cursor_elevation='--' if product in gv.plain_products else radius
                    else:
                        sr, self.cursor_elevation = ft.var1_to_var2(radius, scanangle, 'gr+theta->sr+h')
                        col = int(np.floor((sr-self.dsg.data_radius_offset[panel])/radial_res))
                else:
                    res = self.data_attr['res'][panel]
                    xy_bins = self.data_attr['xy_bins'][panel]
                    row = int(0.5*xy_bins-np.ceil(xy_coord[1]/res))
                    col = int(0.5*xy_bins+np.floor(xy_coord[0]/res))
                    self.cursor_elevation = '--'
            except Exception:
                #Occurs e.g. when panel not in self.data_attr['scanangle'], which is the case when the panel is empty.
                self.cursor_elevation='--'
                
            try:
                if product == 'g':
                    data_z, data_cc, data_zdr = self.dsg.polrgb_raw_data[panel]
                    z_val, cc_val, zdr_val = data_z[row_rawdata, col], data_cc[row_rawdata, col], data_zdr[row_rawdata, col]
                    z_text = '--' if np.isnan(z_val) else '%.5s' % str(np.round(z_val, 1))
                    cc_text = '--' if np.isnan(cc_val) else '%.5s' % str(np.round(cc_val, 1))
                    zdr_text = '--' if np.isnan(zdr_val) else '%.5s' % str(np.round(zdr_val, 1))
                    polrgb_text = 'Z='+z_text+' dBZ, CC='+cc_text+' %, ZDR='+zdr_text+' dB'
                    raise Exception('Polarimetric RGB composite has no single scalar value to show in the normal way')
                if product == 'j':
                    hid, data_z, data_zdr, data_kdp, data_cc, data_t, data_h0, data_h20, data_gridix = self.dsg.hclass_raw_data[panel]
                    class_id = hid[row_rawdata, col]
                    class_name = hc.HID_CLASSES[class_id-1] if class_id in hc.HID_COLORS_RGBA else '--'
                    z_val, zdr_val, kdp_val, cc_val = data_z[row_rawdata,col], data_zdr[row_rawdata,col], data_kdp[row_rawdata,col], data_cc[row_rawdata,col]
                    z_text = '--' if np.isnan(z_val) else '%.5s' % str(np.round(z_val, 1))
                    zdr_text = '--' if np.isnan(zdr_val) else '%.5s' % str(np.round(zdr_val, 1))
                    kdp_text = '--' if np.isnan(kdp_val) else '%.5s' % str(np.round(kdp_val, 2))
                    cc_text = '--' if np.isnan(cc_val) else '%.5s' % str(np.round(cc_val, 1))
                    # 0C-/-20C-hoogte (uit het gedeelde temperatuurrooster) die voor DEZE bin is gebruikt
                    # voor de temperatuurschatting. Vervangt sinds 22 juli de losse 0C/-20C-balk in nlr.py
                    # (op Eriks verzoek verwijderd, ten gunste van meer ruimte voor de textbar) - hier
                    # staat de waarde nu direct bij de rest van de HCLASS-details. data_h0/data_h20 zijn
                    # None als het rooster niet beschikbaar was (bv. netwerkfout) toen dit paneel werd
                    # berekend, of NaN voor een bin waarvan het dichtstbijzijnde roosterpunt zelf mislukte.
                    if data_h0 is not None and data_h20 is not None:
                        h0_val, h20_val = data_h0[row_rawdata, col], data_h20[row_rawdata, col]
                        h0_text = '--' if np.isnan(h0_val) else '%.0f m' % h0_val
                        h20_text = '--' if np.isnan(h20_val) else '%.0f m' % h20_val
                    else:
                        h0_text = h20_text = '--'
                    # Roosterpunt (nlr_hclass.NL_GRID_POINTS) dat voor DEZE bin is gebruikt - teruggezet
                    # op Eriks verzoek (22 juli), naast de 0C/-20C-regel, niet in plaats ervan.
                    if data_gridix is not None:
                        grid_idx = int(data_gridix[row_rawdata, col])
                        grid_lat, grid_lon = hc.NL_GRID_POINTS[grid_idx]
                        grid_text = '%.1f' % grid_lat + u'\xb0N, ' + '%.1f' % grid_lon + u'\xb0O'
                        # ZICHTBAARHEID (23 juli, op Eriks verzoek): datum/tijd waarvoor deze
                        # temperatuurdata daadwerkelijk geldt - kan afwijken van de scantijd
                        # (bv. Wyoming-sounding van een ander uur). Zelfde gedeelde
                        # roosterresultaat als calculate_MESH gebruikt, hier per-bin opgehaald
                        # via dezelfde grid_idx als de HCLASS-classificatie zelf al gebruikte.
                        grid_results = getattr(self.dsg, 'melting_level_grid_results', None)
                        if grid_results is not None:
                            dt_used_str = grid_results[grid_idx].get('datetime_used')
                            tijd_text = dt_used_str.replace('T', ' ')+' UTC' if dt_used_str else '--'
                        else:
                            tijd_text = '--'
                    else:
                        grid_text = '--'
                        tijd_text = '--'
                    # Korte tekst voor de altijd-zichtbare (smalle) balk bovenin: alleen de klassenaam.
                    hclass_text = class_name
                    # Uitgebreide, meerregelige tekst voor het zwevende tooltip bij de muis (zie onderaan
                    # deze functie) - daar is wel ruimte voor alle brongegevens.
                    self.hclass_tooltip_text = (class_name+'\n'
                                            'Z='+z_text+' dBZ, ZDR='+zdr_text+' dB\n'
                                            'KDP='+kdp_text+u'\xb0/km, CC='+cc_text+' %\n'
                                            '0'+u'\xb0'+'C: '+h0_text+'   -20'+u'\xb0'+'C: '+h20_text+'\n'
                                            'roosterpunt: '+grid_text+'\n'
                                            'temperatuurdata van: '+tijd_text)
                    raise Exception('HCLASS composite has no single scalar value to show in the normal way')
                if product == 'o':
                    # ZICHTBAARHEID (23 juli, op Eriks verzoek): toon welke temperatuurbron/welk
                    # station en welke 0C/-20C-waarden voor dit MESH-paneel zijn gebruikt. In
                    # tegenstelling tot HCLASS/PolRGB hierboven wordt hier GEEN Exception gegooid -
                    # de normale numerieke mm-waarde (data_text hieronder) moet gewoon getoond
                    # blijven worden; dit voegt alleen een extra, zwevend tooltipje toe.
                    info = getattr(self.dsg, 'mesh_source_info', None)
                    if info is not None:
                        h0_val, h20_val = info.get('h0_m'), info.get('h_minus20_m')
                        h0_text = '--' if h0_val is None else '%.0f m' % h0_val
                        h20_text = '--' if h20_val is None else '%.0f m' % h20_val
                        source, model = info.get('source'), info.get('model')
                        if source == 'wyoming_sounding':
                            # model is 'station_06260' (De Bilt), 'station_10304' (Meppen),
                            # 'station_10410' (Essen) of 'station_10113' (Norderney)
                            station_naam = {'station_06260': 'De Bilt', 'station_10304': 'Meppen',
                                            'station_10410': 'Essen', 'station_10113': 'Norderney'}.get(
                                model, model)
                            bron_text = 'Wyoming-archief, station ' + str(station_naam)
                        elif source in ('forecast_api', 'archive_api'):
                            bron_text = 'Open-Meteo (' + (model or 'automatisch') + ')'
                        else:
                            bron_text = '--'
                        dt_used_str = info.get('datetime_used')
                        tijd_text = dt_used_str.replace('T', ' ')+' UTC' if dt_used_str else '--'
                        self.mesh_tooltip_text = ('MESH\n'
                                             'temperatuurbron: ' + bron_text + '\n'
                                             '0' + u'\xb0' + 'C: ' + h0_text + '   -20' + u'\xb0' + 'C: ' + h20_text + '\n'
                                             'temperatuurdata van: ' + tijd_text)
                if product == 'b':
                    # Zelfde zichtbaarheid als MESH hierboven, maar dan voor POSH - gebruikt
                    # self.dsg.posh_source_info (zie calculate_POSH in nlr_derived_plain.py),
                    # dat op dezelfde manier is opgebouwd als mesh_source_info.
                    info = getattr(self.dsg, 'posh_source_info', None)
                    if info is not None:
                        h0_val, h20_val = info.get('h0_m'), info.get('h_minus20_m')
                        h0_text = '--' if h0_val is None else '%.0f m' % h0_val
                        h20_text = '--' if h20_val is None else '%.0f m' % h20_val
                        source, model = info.get('source'), info.get('model')
                        if source == 'wyoming_sounding':
                            station_naam = {'station_06260': 'De Bilt', 'station_10304': 'Meppen',
                                            'station_10410': 'Essen', 'station_10113': 'Norderney'}.get(
                                model, model)
                            bron_text = 'Wyoming-archief, station ' + str(station_naam)
                        elif source in ('forecast_api', 'archive_api'):
                            bron_text = 'Open-Meteo (' + (model or 'automatisch') + ')'
                        else:
                            bron_text = '--'
                        dt_used_str = info.get('datetime_used')
                        tijd_text = dt_used_str.replace('T', ' ')+' UTC' if dt_used_str else '--'
                        self.posh_tooltip_text = ('POSH\n'
                                             'temperatuurbron: ' + bron_text + '\n'
                                             '0' + u'\xb0' + 'C: ' + h0_text + '   -20' + u'\xb0' + 'C: ' + h20_text + '\n'
                                             'temperatuurdata van: ' + tijd_text)
                if product == 'uh':
                    # Zelfde soort zichtbaarheid als MESH/POSH hierboven, maar dan voor POH - gebruikt
                    # self.dsg.poh_source_info (zie calculate_POH in nlr_derived_plain.py). POH gebruikt
                    # ALLEEN het 0C-niveau (geen -20C, in tegenstelling tot MESH/POSH - POH is een simpel
                    # lineair verband met het 45dBZ-echo t.o.v. het 0C-niveau, geen SHI-integraal), dus
                    # de -20C-regel wordt hier weggelaten.
                    info = getattr(self.dsg, 'poh_source_info', None)
                    if info is not None:
                        h0_val = info.get('h0_m')
                        h0_text = '--' if h0_val is None else '%.0f m' % h0_val
                        source, model = info.get('source'), info.get('model')
                        if source == 'wyoming_sounding':
                            station_naam = {'station_06260': 'De Bilt', 'station_10304': 'Meppen',
                                            'station_10410': 'Essen', 'station_10113': 'Norderney'}.get(
                                model, model)
                            bron_text = 'Wyoming-archief, station ' + str(station_naam)
                        elif source in ('forecast_api', 'archive_api'):
                            bron_text = 'Open-Meteo (' + (model or 'automatisch') + ')'
                        else:
                            bron_text = '--'
                        dt_used_str = info.get('datetime_used')
                        tijd_text = dt_used_str.replace('T', ' ')+' UTC' if dt_used_str else '--'
                        self.poh_tooltip_text = ('POH\n'
                                             'temperatuurbron: ' + bron_text + '\n'
                                             '0' + u'\xb0' + 'C: ' + h0_text + '\n'
                                             'temperatuurdata van: ' + tijd_text)
                if product == 'si':
                    # Zelfde soort zichtbaarheid als MESH/POSH/POH hierboven, maar dan voor SHI zelf -
                    # gebruikt self.dsg.shi_source_info (zie calculate_SHI in nlr_derived_plain.py),
                    # dat op dezelfde manier is opgebouwd als mesh_source_info/posh_source_info (met
                    # zowel het 0C- als -20C-niveau, want SHI gebruikt net als MESH/POSH de volledige
                    # kolomintegraal tussen die twee niveaus).
                    info = getattr(self.dsg, 'shi_source_info', None)
                    if info is not None:
                        h0_val, h20_val = info.get('h0_m'), info.get('h_minus20_m')
                        h0_text = '--' if h0_val is None else '%.0f m' % h0_val
                        h20_text = '--' if h20_val is None else '%.0f m' % h20_val
                        source, model = info.get('source'), info.get('model')
                        if source == 'wyoming_sounding':
                            station_naam = {'station_06260': 'De Bilt', 'station_10304': 'Meppen',
                                            'station_10410': 'Essen', 'station_10113': 'Norderney'}.get(
                                model, model)
                            bron_text = 'Wyoming-archief, station ' + str(station_naam)
                        elif source in ('forecast_api', 'archive_api'):
                            bron_text = 'Open-Meteo (' + (model or 'automatisch') + ')'
                        else:
                            bron_text = '--'
                        dt_used_str = info.get('datetime_used')
                        tijd_text = dt_used_str.replace('T', ' ')+' UTC' if dt_used_str else '--'
                        self.shi_tooltip_text = ('SHI\n'
                                             'temperatuurbron: ' + bron_text + '\n'
                                             '0' + u'\xb0' + 'C: ' + h0_text + '   -20' + u'\xb0' + 'C: ' + h20_text + '\n'
                                             'temperatuurdata van: ' + tijd_text)
                n_bits=gv.products_data_nbits[product]
                pm_lim=gv.products_maxrange_masked[product]
                cursor_datavalue_int=self.dsg.data[panel][row,col]
                if cursor_datavalue_int==self.mask_values_int[product]:
                    self.cursor_datavalue='--'
                else:
                    self.cursor_datavalue=ft.convert_uint_to_float(cursor_datavalue_int,n_bits,pm_lim)
                    self.cursor_datavalue*=self.scale_factors[product]
                    if product=='r':
                        self.cursor_datavalue=10**self.cursor_datavalue
                if product == 'o' and self.mesh_tooltip_text is not None:
                    # ZICHTBAARHEID (23 juli, op Eriks verzoek): de hageldikte op de kliklocatie
                    # zelf (self.cursor_datavalue, hierboven al berekend voor de balk bovenin)
                    # ook in de pop-up tonen, niet alleen de temperatuurbron. Moet HIER staan
                    # (na de berekening van cursor_datavalue hierboven), niet bij de rest van
                    # mesh_tooltip_text verderop, want die wordt AL opgebouwd voordat
                    # cursor_datavalue bekend is.
                    mesh_text = '--' if self.cursor_datavalue == '--' else '%.1f mm' % self.cursor_datavalue
                    self.mesh_tooltip_text = 'MESH: ' + mesh_text + '\n' + self.mesh_tooltip_text[len('MESH\n'):]
                if product == 'b' and self.posh_tooltip_text is not None:
                    # Zelfde constructie als bij MESH hierboven, maar dan het POSH-percentage.
                    posh_text = '--' if self.cursor_datavalue == '--' else '%.0f%%' % self.cursor_datavalue
                    self.posh_tooltip_text = 'POSH: ' + posh_text + '\n' + self.posh_tooltip_text[len('POSH\n'):]
                if product == 'uh' and self.poh_tooltip_text is not None:
                    # Zelfde constructie als bij MESH/POSH hierboven, maar dan het POH-percentage.
                    poh_text = '--' if self.cursor_datavalue == '--' else '%.0f%%' % self.cursor_datavalue
                    self.poh_tooltip_text = 'POH: ' + poh_text + '\n' + self.poh_tooltip_text[len('POH\n'):]
                if product == 'si' and self.shi_tooltip_text is not None:
                    # Zelfde constructie als bij MESH/POSH/POH hierboven, maar dan de ruwe SHI-waarde
                    # (J/m/s) zelf - geen sqrt/ln-omzetting, in tegenstelling tot MESH/POSH.
                    shi_text = '--' if self.cursor_datavalue == '--' else '%.1f J/m/s' % self.cursor_datavalue
                    self.shi_tooltip_text = 'SHI: ' + shi_text + '\n' + self.shi_tooltip_text[len('SHI\n'):]
            except Exception: self.cursor_datavalue='--'   
            try:
                if product in ('g', 'j'):
                    raise Exception('Polarimetric RGB / HCLASS composite has no single scalar value to show')
                min_value, max_value = [ft.convert_uint_to_float(j, n_bits, pm_lim)*self.scale_factors[product] for j in self.get_min_max_in_view(panel)]
            except Exception:
                min_value, max_value = '--', '--'
        else:
            self.cursor_elevation='--'; self.cursor_datavalue='--'
            min_value, max_value = '--', '--'
                                   
        self.cursor_latlon=ft.aeqd(gv.radarcoords[self.crd.radar],xy_coord,inverse=True)
        self.cursor_radius=radius if not self.gui.sm_marker_present else \
        ft.calculate_great_circle_distance_from_xy(gv.radarcoords[self.crd.radar], xy_coord, self.gui.sm_marker_position)
        
        lat_text='%6.6s' % str(np.around(self.cursor_latlon[0]*1000)/1000)
        lon_text='%6.6s' % str(np.around(self.cursor_latlon[1]*1000)/1000)
        x_text, y_text = '%5.5s' % xy_coord[0], '%5.5s' % xy_coord[1]
        r_text='%5.5s' % str(np.around(self.cursor_radius*1000)/1000)
        h_text='%5.5s' % str(np.around(self.cursor_elevation*1000)/1000) if self.cursor_elevation!='--' else '%5.5s' % self.cursor_elevation
        if polrgb_text is not None:
            data_text = polrgb_text
            product_unit = ''
        elif hclass_text is not None:
            data_text = hclass_text
            product_unit = ''
        else:
            factor=1000 if self.cursor_datavalue!='--' and np.abs(self.cursor_datavalue)<0.01 else 100
            data_text='%5.5s' % (str(np.around(self.cursor_datavalue*factor)/factor) if self.cursor_datavalue!='--' else self.cursor_datavalue)+'/'+\
                      '%5.5s' % (str(np.around(min_value*factor)/factor) if min_value!='--' else min_value)+'/'+\
                      '%5.5s' % (str(np.around(max_value*factor)/factor) if max_value!='--' else max_value)
            try: #An error occurs when product is not defined, as is the case when self.data_attr['product'][panel] does not exist.
                product_unit = self.productunits[product]
            except: product_unit = ''
        self.datareadout_text='('+lat_text+', '+lon_text+'), ('+x_text+', '+y_text+'), r='+r_text+' km, h='+h_text+' km, '+data_text+' '+product_unit
        self.gui.set_textbar()

        # VIERDE FIX (23 juli, op Eriks verzoek): het zwevende QToolTip-tooltipje (Z/ZDR/KDP/CC/
        # 0C/-20C/roosterpunt/tijd voor HCLASS, temperatuurbron/0C/-20C/tijd voor MESH) bleek na
        # meerdere pogingen (leaveEvent-hook, vaste duur van 2s dan 15s) nog steeds niet prettig
        # te werken - Erik wilde er helemaal vanaf. Vervangen door een echt pop-up-venster dat
        # verschijnt bij een KLIK op de kaart (zie on_mouse_release/show_extra_info_popup)
        # i.p.v. continu bij hoveren. self.hclass_tooltip_text/self.mesh_tooltip_text worden
        # hierboven al gevuld (of op None gezet) bij elke aanroep van deze functie, en blijven
        # als instance-attribuut staan totdat on_mouse_release ze na een klik uitleest.

    def screencoord_to_xy(self, pos, panel=None):
        if len(pos.shape)>1:
            xy_coord=[]
            for j in pos:
                panel = self.get_panel_for_position(j) if panel is None else panel
                xy_coord.append(self.coordimaps[panel](j)[:2])
            return np.array(xy_coord)*np.array([1,-1]) # y-coordinate was reversed.
        else:
            panel = self.get_panel_for_position(pos) if panel is None else panel
            return np.array(self.coordimaps[panel](pos)[:2])*np.array([1,-1]) # y-coordinate was reversed.
        
    def xycoord_to_screen(self, panel, pos):
        if len(pos.shape)>1:
            screen_coord=[]
            for j in pos:
                screen_coord.append(self.coordmaps[panel](j*np.array([1,-1]))[:2]) # reverse y-coordinate.
            return np.array(screen_coord)
        else:
            return np.array(self.coordmaps[panel](pos*np.array([1,-1]))[:2]) # reverse y-coordinate.
        
    def get_in_view_mask_specs(self, data, panel):
        specs = str(self.corners[panel])+str(data.shape)+str(self.data_attr['radial_res'][panel] if self.data_attr['proj'][panel] == 'pol' else
                                                      self.data_attr['res'][panel])
        if self.data_attr['proj'][panel] == 'pol':
            specs += str(self.data_attr['scanangle'][panel])+str(self.dsg.data_radius_offset[panel])+str(self.dsg.data_azimuth_offset[panel])
        return specs
    def get_min_max_in_view(self, panel):
        data = self.dsg.data[panel]
        if self.data_attr['proj'][panel] == 'pol' and data.shape[0] > self.data_attr['azimuthal_bins'][panel]:
            data = data[1:-1] # Exclude the 2 extra rows for interpolation
        
        self.get_corners()
        if self.get_in_view_mask_specs(data, panel) != self.in_view_mask_specs:
            if self.data_attr['proj'][panel] == 'pol':
                dr, theta = self.data_attr['radial_res'][panel], self.data_attr['scanangle'][panel]
                r = ft.var1_to_var2(self.dsg.data_radius_offset[panel] + dr*(0.5+np.arange(data.shape[1], dtype='float32')), 
                                    theta, 'sr+theta->gr')
                da = self.data_attr['azimuthal_res'][panel]
                azi = np.deg2rad(self.dsg.data_azimuth_offset[panel] + da*(0.5+np.arange(data.shape[0], dtype='float32')))
                x = r[np.newaxis,:]*np.sin(azi[:,np.newaxis])
                y = r[np.newaxis,:]*np.cos(azi[:,np.newaxis])
            else:
                xy_bins = self.data_attr['xy_bins'][panel]
                res = self.data_attr['res'][panel]
                coords = res*(0.5+np.arange(-0.5*xy_bins, 0.5*xy_bins, dtype='float32'))
                x, y = np.meshgrid(coords, coords[::-1], copy=False)
            
            corners = self.corners[panel]
            self.in_view_mask = (x >= corners[0][0]) & (x  <= corners[-1][0]) & (y >= corners[1][1]) & (y <= corners[0][1])
            self.in_view_mask_specs = self.get_in_view_mask_specs(data, panel)
            
        in_view = data[self.in_view_mask & (data != self.mask_values_int[self.crd.products[panel]])]
        return in_view.min(), in_view.max()
        
    def get_max_dist_mouse_to_marker(self, f=1):
        self.get_corners()
        return f*15*self.ydim*self.nrows/1e3
    
    def check_presence_near_radars(self, pos):
        xy_mouse = self.screencoord_to_xy(pos)
        distances_to_radars = np.linalg.norm(self.radarcoords_xy-xy_mouse, axis=1)
        min_distance_index = np.argmin(distances_to_radars)
        max_distance = self.get_max_dist_mouse_to_marker()
        if min(distances_to_radars) < max_distance:
            self.radar_mouse_selected = gv.radars_all[min_distance_index]
            self.gui.set_textbar()
            if self.radar_mouse_selected != self.crd.selected_radar:
                self.gui.plotwidget.setCursor(Qt.PointingHandCursor)
        else: 
            self.radar_mouse_selected = None
            self.gui.plotwidget.setCursor(Qt.ArrowCursor)
            
    def check_presence_near_pos_markers(self, pos):
        if len(self.gui.pos_markers_positions):
            xy_mouse = self.screencoord_to_xy(pos)
            xy_markers = np.array(self.gui.pos_markers_positions)
            distances_to_markers = np.linalg.norm(xy_markers-xy_mouse, axis=1)
            min_distance_index = np.argmin(distances_to_markers)
            max_distance = self.get_max_dist_mouse_to_marker()
            if distances_to_markers[min_distance_index] < max_distance:
                self.marker_mouse_selected_index = min_distance_index
            else:
                self.marker_mouse_selected_index = None
            
    def update_gridheightrings(self,event=None):
        if 'grid' in self.gui.lines_show: self.set_grid()
        if 'heightrings' in self.gui.lines_show: self.set_heightrings()
        self.set_ghlineproperties(self.panellist)
        self.set_ghtextproperties(self.panellist)
        self.set_draw_action('panning_zooming')
        self.update()
            
    def remove_gridheightrings(self):
        self.postpone_plotting_gridheightrings=True
        self.gridheightrings_removed=True
        self.set_ghlineproperties(self.panellist)
        for j in range(self.max_panels):
            if j in self.visuals['text_hor1']:
                self.visuals['text_hor1'][j].visible=False
                self.visuals['text_hor2'][j].visible=False
            if j in self.visuals['text_vert1']:
                self.visuals['text_vert1'][j].visible=False
                self.visuals['text_vert2'][j].visible=False   
            
    def setback_gridheightrings(self):
        self.timer_setback_gridheightrings_running=False
        self.postpone_plotting_gridheightrings=False
        self.gridheightrings_removed=False
        # self.set_ghlineproperties(self.panellist)
        for j in range(self.max_panels):
            if j in self.panels_horizontal_ghtext:
                self.visuals['text_hor1'][j].visible = self.visuals['text_hor2'][j].visible = True
            if j in self.panels_vertical_ghtext:
                self.visuals['text_vert1'][j].visible = self.visuals['text_vert2'][j].visible = True       
        self.update_gridheightrings()
        
        if self.gui.view_nearest_radar:
            self.crd.switch_to_nearby_radar(1)
        
    def set_timer_setback_gridheightrings(self):
        if self.timer_setback_gridheightrings_running:
            self.timer_setback_gridheightrings.stop()
        self.timer_setback_gridheightrings_running=True
        self.postpone_plotting_gridheightrings=True
        #Time is in ms for the timer, not s
        self.timer_setback_gridheightrings=QTimer()
        self.timer_setback_gridheightrings.setSingleShot(True)
        self.timer_setback_gridheightrings.timeout.connect(self.setback_gridheightrings_signal.emit)
        self.timer_setback_gridheightrings.start(int(self.gui.showgridheightrings_panzoom_time*1000.))
        

            
    def change_interpolation(self):
        self.use_interpolation = not self.use_interpolation
        panellist = [j for j in self.panellist if self.crd.products[j] in ('z','a','m')]
        self.set_newdata(panellist)
        
    def set_interpolation(self):
        for j in self.panellist:
            try:
                if self.use_interpolation and self.data_attr['product'][j] in gv.products_with_interpolation:                
                    self.visuals['radar_polar'][j].interpolation='bilinear'
                    self.visuals['radar_cartesian'][j].interpolation='bilinear'
                else: 
                    self.visuals['radar_polar'][j].interpolation='nearest'
                    self.visuals['radar_cartesian'][j].interpolation='nearest'
            except Exception: # Happens when j not in self.data_attr['product']
                continue
        self.update()
            
    def set_radarmarkers_data(self):
        # Selected radar should be drawn last, in order to always plot it on top.
        face_colors = [self.gui.radar_colors['Automatic download + selected' if self.crd.selected_radar in self.gui.radars_automatic_download else 'Selected']]
        radars = [self.crd.selected_radar]
        for j in self.gui.radars_automatic_download:
            if j != self.crd.selected_radar:
                face_colors.append(self.gui.radar_colors['Automatic download'])
                radars.append(j)
        for j in gv.radars_all:
            if not j in self.gui.radars_automatic_download+[self.crd.selected_radar]:
                face_colors.append(self.gui.radar_colors['Default'])
                radars.append(j)
        # Use slightly different marker sizes for different radar wavelength bands, the biggest for S-band
        scale_fac = {'S':1.15, 'C':1, 'X':1/1.15}
        coords_xy, sizes = [], []
        for i,j in enumerate(radars):
            if j in self.gui.radars_download_older_data:
                #Blend the color with black, to get a darker color that indicates that download of older data is being performed.
                face_colors[i] = ft.blend_rgba_colors_1D(np.append(face_colors[i],0), np.array([64,64,64,0]), 0.5)[:3]
            coords_xy.append(self.radarcoords_xy[gv.radars_all.index(j)])
            sizes.append(scale_fac[gv.radar_bands[j]]*self.scale_pixelsize(self.gui.radar_markersize))
        # Reverse array entries in order to put selected radar last            
        coords_xy, face_colors, sizes = np.array(coords_xy)[::-1], np.array(face_colors)[::-1], np.array(sizes)[::-1]
        
        self.visuals['radar_markers'][0].set_data(pos=coords_xy*np.array([1,-1]),symbol='disc',size=sizes,edge_width=1,face_color=face_colors/255.,edge_color='black')

            
    def set_newdata(self, panellist, delta_time=0, source_function=None, set_data=True, plain_products_parameters_changed=False, apply_storm_centering=False):
        # from cProfile import Profile
        # profiler = Profile()
        # profiler.enable()
        """
        This is the function that handles changes in radar, date, time and dataset. Other functions, from which this function is called, change the
        selected radar, date, time and dataset, but these changes are only actually realized when no exceptions are raised during the evaluation of this 
        function.       
        At first, changes in radar and dataset are handled, which require an update of the lists with scans information, and in the case of a change
        of radar changes in the map.
        Secondly the need to update the colormaps and colorbars is evaluated, and they are updated if needed. After that the new data is imported by
        calling self.dsg.get_data. In the case of exceptions, parameters are set back to their previous values.
        Finally the data in the image visual gets updated, as is its transform and colormap when required.  
        
        set_data=False if this function is only called to obtain the radar volume attributes, which are then returned as the second output argument, next
        to error_received.
        """
        self.crd.date=self.crd.selected_date; self.crd.time=self.crd.selected_time
                
        if self.firstplot_performed and set_data:
            #Save data about the previous plot, which is used in the function self.dsg.check_need_scans_change.
            self.get_corners()
            #self.crd.radar en self.crd.dataset have not yet been updated here, so I don't need to use self.crd.before_variables['radar'] etc.
            radar_dataset=self.crd.radar+(' '+self.crd.dataset if self.crd.radar in gv.radars_with_datasets else '')
            #self.crd.scans can already have been updated, so use self.scans_before here.
            self.dsg.scans_radars[radar_dataset]={'time':pytime.time(),'panellist':self.panellist,'scans':self.scans_before.copy(),
                                                  'scanangles_all':self.dsg.scanangles_all_m.copy(),'corners':self.corners.copy()}
        
        radar_changed = self.crd.radar != self.crd.selected_radar
        dataset_changed = self.crd.dataset != self.crd.selected_dataset
        # for set_data=False, self.crd.radar and self.crd.dataset are restored after checking the presence of data.
        save_radar, save_dataset = self.crd.radar, self.crd.dataset
        self.crd.radar, self.crd.dataset = self.crd.selected_radar, self.crd.selected_dataset
        if set_data:
            if self.map_transforms['aeqd'].radar != self.crd.selected_radar:
                # Use a different condition than radar_changed, since this function might already have been called in self.gui.switch_to_case
                self.change_map_center()
            if self.gui.stormmotion[1] != 0.:
                self.gui.update_stormmotion_change_datetime_or_radar()
                                        
            self.changed_colortables=self.set_cmaps([self.crd.products[j] for j in self.panellist])
            #Updates colormaps if the corresponding color tables have been changed.
            
            if radar_changed or dataset_changed or self.crd.changing_subdataset or delta_time:
                #Update the 'before' variables
                self.update_before_variables(radar_changed,self.crd.changing_subdataset,dataset_changed)            
        
        
        self.data_empty_before = self.data_empty.copy()
        self.data_attr_before = copy.deepcopy(self.data_attr)
        if set_data and self.crd.process_datetimeinput_running:
            #Reset self.dsg.data, to prevent that data for the old radar/dataset is shown when new data is unavailable for a particular panel.
            #Also reset it when changing the time through self.crd.process_datetimeinput, because otherwise the old data could differ greatly
            #in time from the curently selected date and time.
            self.dsg.data = {j:np.zeros((1,1), dtype='uint8') for j in range(self.max_panels)}
            self.data_empty={j:True for j in range(self.max_panels)}
            self.data_isold={j:True for j in range(self.max_panels)}
            self.data_attr = {j:{} for j in self.data_attr}
                        
        t=pytime.time()
        try:
            # print('getting data', self.crd.lrstep_beingperformed, 0 in self.data_attr_before['scandatetime'])
            returns=self.dsg.get_data(panellist, delta_time, radar_changed, dataset_changed, set_data)
            if set_data:
                data_changed, total_file_size = returns
            else:
                retrieved_attrs, total_file_size = returns
                
            if self.crd.lrstep_beingperformed and self.crd.change_radar_running and 0 in self.data_attr_before['scandatetime']:
                scandate = ft.get_scandate(self.crd.date, self.crd.time, self.dsg.scantimes[0])
                scandatetime = scandate+ft.scantime_to_flattime(self.dsg.scantimes[0])
                if np.sign(int(scandatetime)-int(self.data_attr_before['scandatetime'][0])) != np.sign(self.crd.lrstep_beingperformed):
                    self.data_attr = copy.deepcopy(self.data_attr_before)
                    self.data_empty = self.data_empty_before.copy()
                    # Also update self.scans_before, since otherwise a manual/forced scan change might be detected in
                    # self.dsg.get_scans_information, while in fact the change of scans could have taken place in 
                    # self.dsg.check_need_scans_change
                    self.scans_before = self.crd.scans.copy()
                    print('again')
                    # from_timer can be set to True, since it's not needed to check for sleep time again
                    return self.crd.process_keyboardinput(self.crd.lrstep_beingperformed, from_timer=True)
                # Given the return statement above, only one subcall of self.crd.process_keyboardinput will continue here, which is the last.
                # But for this last subcall radar_changed will have been set to False, since the change of radar took place in a previous subcall.
                # This is not desired, since a change of radar has in fact taken place. So set it to True here.
                radar_changed = True
                
        except Exception as e:
            # print('exxie', self.dsg.scannumbers_all, self.dsg.scanangles_all, self.crd.products, self.crd.scans)
            traceback.print_exception(type(e), e, e.__traceback__)
            description = str(e.args[0])
            if type(e) == MemoryError:
                self.gui.set_textbar('Not enough memory available to read file.', 'red', 1)
            elif description == 'get_scans_information':
                # Add datetime to self.crd.filedatetimes_errors and thereby exclude it in self.crd.get_filedatetimes.
                # Exception is when the total file size changes, in which case it won't be excluded. This total file size is therefore
                # saved with the datetime.
                ft.create_subdicts_if_absent(self.crd.filedatetimes_errors, [self.crd.directory, self.crd.date+self.crd.time])
                total_files_size = self.dsg.get_total_volume_files_size()
                self.crd.filedatetimes_errors[self.crd.directory][self.crd.date+self.crd.time] = total_files_size
                self.crd.determine_list_filedatetimes()
                self.gui.set_textbar('Could not read file(s). Try a different time.', 'red', 1)
            elif 'none' in description.lower():
                raise Exception('None!!!')
            else:                
                self.gui.set_textbar('Could not obtain requested product and/or scan. Try a another option.', 'red', 1)
            # For set_data=False return retrieved attrs, which is now an empty dictionary
            
            retrieved_attrs = {}
            data_changed = {j:False for j in panellist}
            # In the case of errors we usually continue below, for example in order to update the panels with an empty data array when
            # switching from radar (in which case it's undesired to keep showing the previous data, since it's for the previous radar).
            # But return when self.firstplot_performed = False, since otherwise errors occur below.
            if set_data and not self.firstplot_performed:
                return
        # print(pytime.time()-t, 't_total_import')

        
        if set_data:            
            for j in panellist:
                self.data_isold[j] = not data_changed[j]
                if not data_changed[j]:
                    continue
                
                self.data_empty[j]=False
                # Remove any previous dictionary entries for this panel, since params are different between plain and non-plain products. 
                # And the number of panels in self.data_attr['scanangle'] is for example used further down.
                ft.remove_keys_from_all_subdicts(self.data_attr, [j])
                self.data_attr['product'][j] = self.crd.products[j]
                self.data_attr['scantime'][j] = self.dsg.scantimes[j]
                scandate = ft.get_scandate(self.crd.date, self.crd.time, self.dsg.scantimes[j])
                self.data_attr['scandatetime'][j] = scandate+ft.scantime_to_flattime(self.dsg.scantimes[j])
                if not self.crd.products[j] in gv.plain_products:
                    p = gv.i_p[self.crd.products[j]]
                    self.data_attr['scanangle'][j]=self.dsg.scanangle(p, self.crd.scans[j], self.dsg.scannumbers_forduplicates[self.crd.scans[j]])
                    self.data_attr['radial_bins'][j]=self.dsg.radial_bins_all[p][self.crd.scans[j]]
                    self.data_attr['radial_res'][j]=self.dsg.radial_res_all[p][self.crd.scans[j]]
                    self.data_attr['azimuthal_bins'][j]=self.dsg.data[j].shape[0]
                    self.data_attr['azimuthal_res'][j]=360/self.data_attr['azimuthal_bins'][j]
                    self.data_attr['proj'][j] = 'pol'
                else:
                    for param in self.data_attr:
                        if param in self.dp.meta_PP[self.crd.products[j]]:
                            self.data_attr[param][j]=self.dp.meta_PP[self.crd.products[j]][param]


        if not set_data:
            self.crd.radar=save_radar; self.crd.dataset=save_dataset
        else:            
            move_view_with_storm = self.gui.use_storm_following_view and apply_storm_centering and self.gui.stormmotion[1] > 0.
            # print('move', move_view_with_storm, apply_storm_centering)
            if move_view_with_storm:
                panellist_move = [j for j in panellist if data_changed[j]]
                if panellist_move:
                    move_view_with_storm = self.move_view_with_storm(panellist_move)
                
            if source_function!=self.change_panels:
                # If source_function==self.change_panels, then it is always desired to update grid and height rings because of changed 
                # panel dimensions. In that case that is done in a simple manner in the function itself.
                
                gh_panellist = panellist_move if move_view_with_storm else panellist
                # if self.data_empty has changed, then the grid and heightrings should always be updated, because for at least 1 panel they change
                # from visible to invisible or vice versa.
                gh_changed = any([self.data_empty_before[j] != self.data_empty[j] for j in panellist]) 
                #When self.timer_setback_gridheightrins_running=True, then heightrings are currently not visible, and  will be plotted after the
                #timer is finished. In this case it is therefore not desired to update them here, as this will make them visible.
                if not self.firstplot_performed or (radar_changed and not self.postpone_plotting_gridheightrings) or move_view_with_storm:
                    if 'grid' in self.gui.lines_show:
                        self.set_grid(gh_panellist); gh_changed=True
                
                plain_products=[[j,self.data_attr['product'][j]] for j in panellist if self.data_attr['product'].get(j, None) in gv.plain_products]
                plain_products_before=[[j,self.data_attr_before['product'][j]] for j in panellist if self.data_attr_before['product'].get(j, None) in gv.plain_products]
                
                #Update height rings only if the scanangle differs by more than 0.05 degrees from the angle on which the height rings are currently
                #based, to prevent that there are too much changes (which leads to undesired flickering and slows down plotting a bit).
                scanangles_updated = self.update_heightrings_scanangles()
                if not self.firstplot_performed or (not self.postpone_plotting_gridheightrings and (radar_changed or scanangles_updated or (
                plain_products_parameters_changed or plain_products != plain_products_before))) or move_view_with_storm:
                    if 'heightrings' in self.gui.lines_show:
                        self.set_heightrings(gh_panellist); gh_changed=True
                
                if gh_changed: self.set_ghlineproperties(gh_panellist)
                if gh_changed: self.set_ghtextproperties(gh_panellist)
                
                
            for j in panellist:     
                if data_changed[j] and self.data_attr['proj'][j] == 'pol' and (self.use_interpolation or self.dsg.data_azimuth_offset[j]):
                    self.dsg.data[j] = np.concatenate(([self.dsg.data[j][-1]], self.dsg.data[j], [self.dsg.data[j][0]]))
                    #The last azimuth is prepended to the array, and the first azimuth is appended to the array to ensure that interpolation 
                    #works correctly between 360 and 0 degrees.
                
                radar_image = 'radar_polar' if not data_changed[j] or self.data_attr['proj'][j] == 'pol' else 'radar_cartesian'
                other_radar_image = 'radar_cartesian' if radar_image == 'radar_polar' else 'radar_polar'
                self.visuals[radar_image][j].set_data(self.dsg.data[j])
                self.visuals[radar_image][j].visible = True
                self.visuals[other_radar_image][j].visible = False
                
                if data_changed[j]:
                    is_rgb_product = self.data_attr['product'][j] in ('g', 'j') # 'g'=PolRGB, 'j'=HCLASS:
                    # beide leveren al kant-en-klare RGBA-data (zie _calculate_polrgb/_calculate_hclass in
                    # nlr_datasourcegeneral.py), dus geen scalaire kleurtransform maar passthrough.
                    # UITBREIDING (6 juli 2026): voorheen werd de GLSL-kleurtransform-rebuild alleen
                    # afgedwongen bij het wisselen tussen PolRGB en een gewoon product (zie hieronder,
                    # oorspronkelijk commentaar van Bram). Als hetzelfde soort "oude shader-status blijft
                    # hangen"-probleem zich ook kan voordoen bij andere wisselingen (bv. van radarstation,
                    # met hetzelfde product) -- wat zou passen bij waargenomen gedrag waarbij de juiste
                    # waarde/kleur-koppeling correct berekend is (3D-weergave klopt) maar de 2D-weergave
                    # soms een oude, niet-passende kleurtoewijzing blijft tonen na het wisselen -- dan is de
                    # veiligste aanpak om dit ALTIJD af te dwingen bij nieuwe data, i.p.v. alleen in dit ene
                    # specifieke geval. Kost vermoedelijk verwaarloosbaar veel extra tijd t.o.v. de eerdere,
                    # fragiele voorwaardelijke aanpak verderop.
                    self.visuals[radar_image][j]._need_colortransform_update = True
                    if is_rgb_product != self.visual_colortransform_is_rgb[radar_image][j]:
                        # Switching into or out of the polarimetric RGB composite: the GLSL color-transform function
                        # (scalar+colormap lookup vs. RGB passthrough) must be rebuilt, since vispy normally only does
                        # this when the cmap *property* is reassigned (see ImageVisual.cmap setter), which we don't
                        # touch for 'g'. Without this, the visual keeps using whichever color-transform was built for
                        # the previous product on this panel, applied to the new (differently-shaped) data.
                        self.visual_colortransform_is_rgb[radar_image][j] = is_rgb_product
                    if not is_rgb_product and (
                    self.visuals[radar_image][j].cmap != self.cm1[self.data_attr['product'][j]] or
                    list(map(int, self.visuals[radar_image][j].clim)) != self.clim_int[self.data_attr['product'][j]]):
                        # Changing the image attributes unnecessarily for every panel is relatively slow.
                        self.visuals[radar_image][j].cmap=self.cm1[self.data_attr['product'][j]]
                        self.visuals[radar_image][j].clim=self.clim_int[self.data_attr['product'][j]]
                    
                    if self.data_attr['proj'][j] == 'pol':
                        if self.data_attr['product'][j] in gv.plain_products:
                            radial_res=self.data_attr['radial_res'][j]
                            azimuthal_res=self.data_attr['azimuthal_res'][j]
                            scanangle=0.
                        else:
                            radial_res=self.data_attr['radial_res'][j]
                            azimuthal_res=self.data_attr['azimuthal_res'][j]
                            scanangle=self.data_attr['scanangle'][j]
                        azimuthal_bins, radial_bins = self.dsg.data[j].shape[:2]
                        rbr_product = radial_bins*radial_res
                        abr_product = azimuthal_bins*azimuthal_res
                        
                        if self.polar_transforms_individual['scanangle'][j].scanangle!=scanangle or\
                        self.polar_transforms_individual['scale'][j].scale[0]!=rbr_product or self.polar_transforms_individual['scale'][j].scale[1]!=abr_product:                            
                            self.polar_transforms_individual['scanangle'][j].scanangle=scanangle
                            # self.ref_radial_bins and self.ref_azimuthal_bins are set in self.on_draw
                            ref_radial_bins = self.ref_radial_bins.get(j, radial_bins)
                            ref_azimuthal_bins = self.ref_azimuthal_bins.get(j, azimuthal_bins)
                            self.polar_transforms_individual['scale'][j].scale=(rbr_product/ref_radial_bins, abr_product/ref_azimuthal_bins)
                            #Division by the reference values is necessary to get the correct scaling when the number of radial bins changes.
                        
                        #The -azimuthal_res is for the added radial for interpolation
                        delta_a = -azimuthal_res if azimuthal_bins > self.data_attr['azimuthal_bins'][j] else 0
                        translate = np.array([self.dsg.data_radius_offset[j], self.dsg.data_azimuth_offset[j]+delta_a])
                        if (self.polar_transforms_individual['scale'][j].translate[:2] != translate).any():
                            self.polar_transforms_individual['scale'][j].translate = translate
                    else:
                        res = self.data_attr['res'][j]
                        xy_bins = self.data_attr['xy_bins'][j]
                        self.cartesian_transforms_individual['scale_translate'][j].scale = (res, res)
                        self.cartesian_transforms_individual['scale_translate'][j].translate = (-xy_bins*res/2, -xy_bins*res/2)                    
                                                                    
                elif self.data_empty[j] and not j in self.ref_radial_bins:
                    #Ensure that the reference number of radial bins is correct when the first image that is viewed after starting 
                    #the program is 'empty' (or in fact a 362*1 invisible array).
                    print(self.dsg.data[j].shape, 'ref_shape')
                    self.ref_azimuthal_bins[j], self.ref_radial_bins[j] = self.dsg.data[j].shape[:2]

            # Keep every active cross-section split-view in sync with the normal animation/time-navigation:
            # a new scan/timestep arriving for a panel changes what its cross-section should show (new data
            # along the same A/B line), so show_cross_section needs to be re-run to recompute the raster for
            # the new moment. This only fires for panels whose data actually changed (data_changed[panel]), so
            # panels without an active cross-section, or timesteps where a given panel's data didn't change,
            # don't trigger an unnecessary recomputation.
            for panel in list(self.cross_section_active_panels):
                if panel in panellist and data_changed.get(panel, False):
                    self.show_cross_section(panel)

            self.products_before = self.crd.products.copy()
            self.scans_before = self.crd.scans.copy()
                    
            self.set_interpolation()
            self.set_titles()
            update_cbars_products = self.set_cbars(set_cmaps=False) #The colormaps have already been set.
            self.set_polrgb_legend()
            self.set_hclass_legend()
            
            plot_vwp = self.gui.show_vwp and (self.vwp.data_name != self.vwp.get_current_data_name() or 
                                              self.gui.switch_to_case_running)
            # Always plot VWP when switching to a case, because if the datetime doesn't change, then the storm motion might still do so
            if plot_vwp:
                self.vwp.set_newdata()
                
            # t = pytime.time()                
            # self.update_data_readout()
            # print(pytime.time()-t,'readout')
            
            if plot_vwp: self.set_draw_action('plotting_vwp')
            elif len(update_cbars_products)>0: self.set_draw_action('updating_cbars')
            elif not self.changing_panels: self.set_draw_action('plotting')
            self.update()
            
            if (not self.firstplot_performed or radar_changed or dataset_changed or self.crd.changing_subdataset or delta_time) and\
            any(data_changed.values()):
                self.update_current_variables()
                                                                    
            if any(data_changed.values()):
                self.firstplot_performed = True
                                
            if self.gui.current_case_shown():
                self.gui.set_textbar()  

            # profiler.disable()
            # import pstats
            # stats = pstats.Stats(profiler).sort_stats('cumtime')
            # stats.print_stats(50)        
        if not set_data:
            return retrieved_attrs
    
    
    def update_current_variables(self):
        radar_dataset = self.dsg.get_radar_dataset()
        self.crd.current_variables['date']=self.crd.date
        self.crd.current_variables['time']=self.crd.time
        self.crd.current_variables['datetime']=self.crd.date+self.crd.time
        self.crd.current_variables['dataset']=self.crd.dataset
        self.crd.current_variables['radar']=self.crd.radar
        self.crd.current_variables['radardir_index']=self.gui.radardata_dirs_indices[radar_dataset]
        self.crd.current_variables['product_version']=self.gui.radardata_product_versions[radar_dataset]        
        self.crd.current_variables['scannumbers_forduplicates']=self.dsg.scannumbers_forduplicates.copy()
    
    def update_before_variables(self,radar_changed,radardir_index_changed,dataset_changed):
        """It is important that self.crd.before_variables['radar'] etc are updated at the same moment, to ensure a consistent set of previous variables.
        """ 
        if len(self.crd.current_variables):
            if any([radar_changed,radardir_index_changed,dataset_changed]):
                self.crd.rd_before_variables={j:self.crd.current_variables[j] for j in ('radar','dataset','radardir_index','product_version')}
                self.crd.rd_before_variables['scannumbers_forduplicates']=self.crd.current_variables['scannumbers_forduplicates'].copy()
            self.crd.before_variables=copy.deepcopy(self.crd.current_variables)
            
            
    def change_map_center(self):
        old_radar = self.map_transforms['aeqd'].radar
        new_radar = self.crd.selected_radar
        center_screencoords=self.panel_centers[0]
        center_xycoord_before=self.screencoord_to_xy(center_screencoords)
        center_latlon=np.array(ft.aeqd(gv.radarcoords[old_radar],center_xycoord_before,inverse=True))
        center_xycoord_after=ft.aeqd(gv.radarcoords[new_radar],center_latlon) 
        for j in self.panellist:
            self.panels_sttransforms[j].move(np.array([1,-1])*(center_xycoord_before-center_xycoord_after)*np.array(self.panels_sttransforms[j].scale[:2]))
        #The view is changed in such a way that the center of the view is not displaced under the change of projection.
    
        (scale_x, scale_y), (t_x, t_y) = self.get_map_sttransform_parameters()
        self.map_transforms['st'].scale = (scale_x, scale_y)
        self.map_transforms['st'].translate = (t_x, t_y)
        self.map_transforms['aeqd'].radar = new_radar
        
        self.radarcoords_xy = np.array(ft.aeqd(gv.radarcoords[new_radar], np.array([gv.radarcoords[j] for j in gv.radars_all])))
        if self.gui.sm_marker_present or len(self.gui.pos_markers_positions): 
            self.set_sm_pos_markers()
            
        self.set_radarmarkers_data()
        
    def get_map_sttransform_parameters(self):
        if self.map_initial_bounds is None: 
            self.map_initial_bounds = copy.deepcopy(self.map_bounds)
            
            map_scalefac=(self.map_bounds['lat'][1]-self.map_bounds['lat'][0])/self.map_data.shape[0]
            self.map_initial_scale = map_scalefac*np.array([1,-1])
        
        mapbound_ratio_lat = (self.map_initial_bounds['lat'][1]-self.map_initial_bounds['lat'][0])/(self.map_bounds['lat'][1]-self.map_bounds['lat'][0])
        mapbound_ratio_lon = (self.map_initial_bounds['lon'][1]-self.map_initial_bounds['lon'][0])/(self.map_bounds['lon'][1]-self.map_bounds['lon'][0])
        scale_x, scale_y = self.map_initial_scale/np.array([mapbound_ratio_lon, mapbound_ratio_lat])
        t_x, t_y=self.map_bounds['lon'][0], self.map_bounds['lat'][1]
        return (scale_x, scale_y), (t_x, t_y)

        
    def get_latlon_bounds(self, xy_bounds=None):
        if xy_bounds is None:
            self.get_corners()
            x_min, x_max = min(self.corners[j][0,0] for j in self.panellist), max(self.corners[j][-1,0] for j in self.panellist)
            y_min, y_max = min(self.corners[j][1,1] for j in self.panellist), max(self.corners[j][0,1] for j in self.panellist)            
        else:
            x_min, x_max, y_min, y_max = xy_bounds
            
        x_range = np.linspace(x_min, x_max, 10)
        y_range = np.linspace(y_min, y_max, 10)
        x_grid, y_grid = np.meshgrid(x_range, y_range)
        xy = np.transpose([x_grid.flatten(), y_grid.flatten()])
        
        radar_latlon = gv.radarcoords[self.crd.radar]
        latlon = ft.aeqd(radar_latlon, xy, inverse=True)
        lat_min, lat_max = latlon[:,0].min(), latlon[:,0].max()
        lon_min, lon_max = latlon[:,1].min(), latlon[:,1].max()
        return lat_min, lat_max, lon_min, lon_max
        
    def get_active_mt(self):
        """Returns the currently active map-tile source object (local bundled tiles, or live MapTiler tiles),
        based on self.gui.basemap_source. Both expose the same interface (isRunning/quit/start/
        set_radar_and_mapbounds/run_outside_thread/finished_signal), so callers can use this method without
        needing to know which source is active."""
        return self.mt_maptiler if self.gui.basemap_source == 'MapTiler' else self.mt

    def update_map_tiles(self, xy_bounds = None, separate_thread = True, draw_map = False):
        if not self.gui.mapvisibility and not self.starting: return
        
        active_mt = self.get_active_mt()
        # Stop the *other* source's thread too, in case a switch happened while it was still running.
        for mt_obj in (self.mt, self.mt_maptiler):
            if mt_obj.isRunning():
                mt_obj.quit()
            
        lat_min, lat_max, lon_min, lon_max = self.get_latlon_bounds(xy_bounds)
        try:
            update_map = (self.map_panels_scale_before != self.panels_sttransforms[0].scale).any() or (
                         lat_min < self.map_bounds['lat'][0] or lat_max > self.map_bounds['lat'][1] or
                         lon_min < self.map_bounds['lon'][0] or lon_max > self.map_bounds['lon'][1])
        except Exception:
            update_map = True
            
        if update_map:
            active_mt.set_radar_and_mapbounds(self.crd.radar, lat_min, lat_max, lon_min, lon_max)
            if separate_thread: 
                active_mt.start()
            else:
                self.map_data, self.map_bounds = active_mt.run_outside_thread()
                if draw_map:
                    self.draw_map_tiles()
        
    def set_timer_update_map_tiles(self):
        if not self.gui.mapvisibility: return
        if self.timer_update_map_tiles_running:
            self.timer_update_map_tiles.stop()
        #Time is in ms for the timer, not s
        self.timer_update_map_tiles_running = True
        self.timer_update_map_tiles=QTimer()
        self.timer_update_map_tiles.setSingleShot(True)
        self.timer_update_map_tiles.timeout.connect(self.update_map_tiles)
        # MapTiler tiles require a network fetch + CPU reprojection, both far slower than reading the bundled
        # local tiles from disk. With the same short delay used for local tiles, a continuous pan/zoom gesture
        # keeps restarting this timer (see the .stop() above) before the previous fetch has even finished,
        # which shows up as stuttering during the gesture rather than a clean, single update once it ends.
        delay = self.gui.maptiles_update_time if self.gui.basemap_source != 'MapTiler' else max(self.gui.maptiles_update_time, 0.5)
        self.timer_update_map_tiles.start(int(delay*1000)) #s to ms
        
    def draw_map_tiles(self, tiles_map = None, tiles_bounds = None):
        if not tiles_map is None:
            self.map_data = tiles_map; self.map_bounds = tiles_bounds
            self.map_panels_scale_before = self.panels_sttransforms[0].scale
        self.visuals['map'][0].set_data(self.map_data)
        # ImageVisual.set_data() does NOT re-trigger a colortransform rebuild on its own -- that decision
        # (RGB-passthrough vs. scalar/colormap path, the latter of which averages R+G+B into a single
        # luminance value) is made once, the first time data is ever bound to this visual, and then stays
        # fixed regardless of what shape later set_data() calls supply. Force it to be re-evaluated here so a
        # bad first-startup binding (e.g. from a failed local-tile fetch returning oddly-shaped data) can't
        # permanently darken/desaturate every subsequent basemap, including a perfectly fine 3-channel
        # MapTiler image.
        self.visuals['map'][0]._need_colortransform_update = True
        self.visuals['map'][0].update()

        (scale_x, scale_y), (t_x, t_y) = self.get_map_sttransform_parameters()
        self.map_transforms['st'].scale = (scale_x, scale_y)
        self.map_transforms['st'].translate = (t_x, t_y)
        self.update()
        
        
    def set_titles(self):
        relwidth = self.wsize['main'][0] / self.gui.screen_size()[0]
        if any(j not in self.data_attr['scandatetime'] or j not in self.data_isold for j in self.panellist):
            print(list(self.data_attr['scandatetime']), self.data_isold)
        scandates = [self.data_attr['scandatetime'][j][:8] for j in self.panellist if not self.data_isold[j]]
        # During the switch to another day, it could happen that some scans are for the previous day, and others for the next. In that case show the
        # date that is most common in the panels.
        date = max(set(scandates), key=scandates.count) if scandates else self.crd.date
        jabbeke_range_km = None
        if self.crd.radar == 'Jabbeke':
            jabbeke_range_km = 150 if self.gui.radardata_dirs_indices.get('Jabbeke_Z', 0) == 1 else 300
        title_top, title_bottom, paneltitles=bg.get_titles(relwidth,self.gui.fontsizes_main['titles'],self.crd.radar,self.panels,self.panellist,self.panelnumber_to_plotnumber,self.plotnumber_to_panelnumber,self.data_empty,self.data_isold,self.data_attr['product'],date,self.data_attr['scantime'],self.data_attr['scanangle'],self.crd.using_unfilteredproduct,self.crd.using_verticalpolarization,self.crd.apply_dealiasing,self.productunits,self.gui.stormmotion,self.gui.PP_parameter_values,self.gui.PP_parameters_panels,self.gui.show_vwp,jabbeke_range_km)

        titles_text_top=[title_top]+[paneltitles[j] for j in paneltitles if j<5 or self.panels==2]
        titles_text_bottom=[title_bottom]+[paneltitles[j] for j in paneltitles if j>=5 and self.panels!=2]
        
        dy_bottom = dy_top = self.scale_pixelsize(2)
        dy_bottom += 0.5*self.scale_pixelsize(self.panel_borders_width) # For the bottom panel borders
        titles_top_ypos = self.wpos['top'][0,1]+dy_top; titles_bottom_ypos = self.wpos['bottom'][0,1]+dy_bottom
        # TIJDELIJKE CORRECTIE (6 juli 2026): in vispy 0.14.1 lijkt anchor_y='top' bij TextVisual niet meer
        # zuiver de bovenkant van de tekst als ankerpunt te nemen (zoals de rest van deze code aanneemt),
        # maar eerder de baseline/het midden -- waardoor de tekst ca. een fontgrootte omhoog verschuift
        # t.o.v. de bedoelde positie (zichtbaar als een stukje titel dat boven het canvas uitsteekt, en
        # 'KNMI' dat in de kaart terechtkomt i.p.v. in de witte balk eronder). Compenseer door de
        # berekende y-positie met ongeveer de fontgrootte naar beneden te verschuiven.
        _title_font_px = self.scale_pointsize(eval(self.font_sizes['titles']))
        titles_top_ypos += 1.5*_title_font_px
        titles_bottom_ypos += 1.5*_title_font_px
        xleft=self.wpos['top'][0,0]; xright=self.wpos['top'][-1,0]
        xdim_1p=(xright-xleft)/self.ncolumns
        
        textpos_title_top=np.array([[(xleft+xright)/2,titles_top_ypos]])
        textpos_title_bottom=np.array([[(xleft+xright)/2,titles_bottom_ypos]])

        if self.panels!=2:
            textpos_paneltitles_top=np.array([[xleft+(j+0.5)*xdim_1p,titles_top_ypos] for j in paneltitles if j<5])
            textpos_paneltitles_bottom=np.array([[xleft+(j-4.5)*xdim_1p,titles_bottom_ypos] for j in paneltitles if j>=5])
        else:
            textpos_paneltitles_top=np.array([[xleft+(j+0.5-(4 if j==5 else 0))*xdim_1p,titles_top_ypos] for j in paneltitles])    
            textpos_paneltitles_bottom=[]
            
        titles_textpos_top=textpos_title_top if len(textpos_paneltitles_top)==0 else np.concatenate((textpos_title_top,textpos_paneltitles_top),axis=0)
        titles_textpos_bottom=textpos_title_bottom if len(textpos_paneltitles_bottom)==0 else np.concatenate((textpos_title_bottom,textpos_paneltitles_bottom),axis=0)

        self.visuals['titles'].text=titles_text_top+titles_text_bottom
        if len(titles_textpos_bottom)==0:
            self.visuals['titles'].pos=titles_textpos_top
        else:         
            self.visuals['titles'].pos=np.concatenate([titles_textpos_top,titles_textpos_bottom])



    def change_panels(self,new_panels):
        if new_panels==self.panels: return
        self.panels=new_panels
        self.panel=0
        self.changing_panels=True
                
        self.determine_panellist_nrows_ncolumns()
        self.set_panel_sttransforms_and_clippers()
        self.set_panel_borders()
             
                                
        products_panels=[self.crd.products[j] for j in self.panellist]
        for product in gv.plain_products_with_parameters:
            if product in products_panels:
                self.gui.change_PP_parameters_panels(product)

                    
        if self.firstplot_performed and self.crd.plot_mode in ('Row','Column') and not self.gui.setting_saved_choice: 
            #If self.gui.setting_saved_choice=True, then the products and scans should not change here.
            #When plot_mode in ('Row','Column') the products and scans are changed in such a way that their values are in agreement with what
            #is desired in the row or column mode.
            products_panellist=[self.crd.products[j] for j in self.panellist]
            n_products=len(set(products_panellist))
            if n_products>1:
                #Only when displaying at least 2 different products, because when displaying only one product it is likely not desired to show the
                #same scan for all panels in the same row/column.
                for j in tuple(range(0,self.panels//2))+((5,) if self.panels>=5 else ()):
                    #The leftmost and upper panels are taken as reference panels, where the number of reference panels depends on the number of panels.
                    if self.crd.plot_mode=='Row': self.crd.row_mode(panel=j)
                    else: self.crd.column_mode(panel=j)
                self.dsg.update_selected_scanangles()
        elif self.firstplot_performed:
            notplain_products_panels=[j for j in products_panels if not j in gv.plain_products]
            if len(set(notplain_products_panels))==len(notplain_products_panels) and not self.gui.setting_saved_choice: 
                #When all not-plain products are unique, then it is assumed that it is desired to view those products for the same scan, 
                #which is set to be the scan of the first panel.
                for j in [i for i in self.panellist if self.crd.products[i] in gv.products_with_tilts]:                        
                    #It is possible that there are 2 scans with the same scanangle, which are in the case of changing panels both allowed in
                    #the same row. If no exception would be made for a change in panels, then you would always have either in both panels
                    #the first or in both panels the second of those 2 scans with the same scanangle, while it can be desired to show them
                    #both (because 1 has a larger range and the other a higher Nyquist velocity for the new KNMI radars).
                    scanangle_j=self.data_attr['scanangle'].get(j, 999)
                    scanangle_0=self.data_attr['scanangle'].get(0, scanangle_j)
                    if scanangle_j != scanangle_0:  
                        self.crd.scans[j]=self.crd.scans[0]
                self.dsg.update_selected_scanangles()                     

        if not self.firstplot_performed:
            self.crd.process_datetimeinput()
        else:
            #Panels which show data for a previous radar are updated here
            self.set_newdata(self.panellist,source_function=self.change_panels)
             
            if not self.postpone_plotting_gridheightrings:
                if 'grid' in self.gui.lines_show: self.set_grid()
                if 'heightrings' in self.gui.lines_show: self.set_heightrings() 
                if any([j in self.gui.lines_show for j in ('grid','heightrings')]):
                    self.set_ghlineproperties(self.panellist)
            if len(self.gui.ghtext_show)>0 and not self.postpone_plotting_gridheightrings: self.set_ghtextproperties(self.panellist)
        
        self.set_timer_update_map_tiles() #This is done here instead of in the function self.on_draw, because self.draw_action is not always set to
        #'changing_panels' here.
        
        self.set_draw_action('changing_panels')
        self.update()
        
        self.changing_panels = False
        
        

    def set_cmaps(self,products):
        tickslim_modified,ticks_steps,excluded_values_for_ticks,included_values_for_ticks,changed_colortables,self.productunits=bg.set_colortables(self.gui.colortables_dirs_filenames,products,self.productunits) 
        
        for product in products: 
            if product in ('v','s','w'):
                self.scale_factors[product]=gv.scale_factors_velocities[self.productunits[product]]
                
            if not product in changed_colortables and (self.gui.cmaps_minvalues[product]!=self.cmaps_minvalues_before[product] or 
            self.gui.cmaps_maxvalues[product]!=self.cmaps_maxvalues_before[product]):
                changed_colortables.append(product)

            if product in changed_colortables:
                product_cbar=product
                cbar_colors = np.flipud(genfromtxt(opa(os.path.join(gv.userdir+'/Generated_files','colortable_'+product_cbar+'_added.csv')), delimiter=','))
                scale = self.scale_factors[product]
                self.data_values_colors[product]=cbar_colors[:,0]/scale
                step=ticks_steps[product_cbar]
                if step!='-': step /= self.scale_factors[product]
                tickslim = tickslim_modified[product]
                for j in tickslim:
                    if not tickslim[j] is None:
                        tickslim[j] /= self.scale_factors[product]
                
                
                remove_indices=[]
                prepend_productvalue=[]; append_productvalue=[]
                prepend_colors=np.array([]); append_colors=np.array([])
                
                for j in range(0,len(self.data_values_colors[product])):
                    if self.gui.cmaps_minvalues[product]!='' and self.gui.cmaps_minvalues[product]>=self.data_values_colors[product][0] and (
                    self.gui.cmaps_minvalues[product]<self.data_values_colors[product][-1] and self.data_values_colors[product][j]<self.gui.cmaps_minvalues[product]):
                        if self.data_values_colors[product][j+1]>self.gui.cmaps_minvalues[product]:
                            prepend_productvalue=[self.gui.cmaps_minvalues[product]]
                            color1=cbar_colors[j][1:4]
                            color2=cbar_colors[j][4:] if not cbar_colors[j][4]==-1. else cbar_colors[j+1][1:4]
                            position=(self.gui.cmaps_minvalues[product]-self.data_values_colors[product][j])/(self.data_values_colors[product][j+1]-self.data_values_colors[product][j])
                            prepend_colors=np.append(ft.interpolate_2colors(color1,color2,position),color2)
                        remove_indices.append(j) 
                    if self.gui.cmaps_maxvalues[product]!='' and self.gui.cmaps_maxvalues[product]>self.data_values_colors[product][0] and (
                    self.gui.cmaps_maxvalues[product]<=self.data_values_colors[product][-1] and self.data_values_colors[product][j]>=self.gui.cmaps_maxvalues[product]):
                        if self.data_values_colors[product][j-1]<self.gui.cmaps_maxvalues[product]:
                            append_productvalue=[self.gui.cmaps_maxvalues[product]]
                            color1=cbar_colors[j-1][1:4]
                            color2=cbar_colors[j-1][4:] if not cbar_colors[j-1][4]==-1. else cbar_colors[j][1:4]
                            position=(self.gui.cmaps_maxvalues[product]-self.data_values_colors[product][j-1])/(self.data_values_colors[product][j]-self.data_values_colors[product][j-1])
                            append_colors=np.append(ft.interpolate_2colors(color1,color2,position),color2)
                        remove_indices.append(j) 
                        
                self.data_values_colors[product]=np.delete(self.data_values_colors[product],remove_indices)
                self.data_values_colors[product]=np.array(prepend_productvalue+list(self.data_values_colors[product])+append_productvalue)
                if product=='r': self.data_values_colors[product]=np.log10(self.data_values_colors[product])
                
                
                """For an explanation of the process of converting floating point data values to unsigned integers, see nlr_globalvars.py.
                """
                c_lim=gv.cmaps_maxrange[product]
                cm_lim=gv.cmaps_maxrange_masked[product]
                pm_lim=gv.products_maxrange_masked[product]                    
                n_bits=gv.products_data_nbits[product]

                self.data_values_colors[product][self.data_values_colors[product]<c_lim[0]]=c_lim[0]
                self.data_values_colors[product][self.data_values_colors[product]>c_lim[1]]=c_lim[1]
                
                self.mask_values[product]=pm_lim[0]    
                self.mask_values_int[product]=0
                self.clim_int[product]=list(ft.convert_float_to_uint(np.array(cm_lim),n_bits,pm_lim))
                self.data_values_colors_int[product]=ft.convert_float_to_uint(self.data_values_colors[product],n_bits,pm_lim)
                
                         
                cbar_colors=np.delete(cbar_colors,remove_indices,axis=0)
                if len(prepend_productvalue)==1:
                    cbar_colors=np.concatenate([[np.append([prepend_productvalue],prepend_colors)],cbar_colors],axis=0)
                if len(append_productvalue)==1:
                    cbar_colors=np.concatenate([cbar_colors,[np.append([append_productvalue],append_colors)]],axis=0)
                        
                color_list1=cbar_colors[:,1:4]
                color_list2=cbar_colors[:,4:]
                color_list=np.zeros(8)
                color_data_values=[self.mask_values[product],c_lim[0]]
                
                if not product in gv.products_possibly_exclude_lowest_values:
                    #Ensure that product values below the minimum value in self.data_values_color[product], but above the minimum allowed value (given by c_lim[0]),
                    #get the correct color, which is the color for the minimum product value in self.data_values_colors[product]
                    
                    #Not when product in gv.products_possibly_exclude_lowest_values, because here it is desired that values below the minimum value in 
                    #self.data_values_color[product] are filtered away.
                    color_list=np.append(color_list,np.append(color_list1[0]/255.,1))
                    color_data_values.append(c_lim[0])
                    
                    cmap2_starti=3 #Is used below in determining the array with control points
                else:
                    cmap2_starti=2
                
                for i in range(0,len(color_list2)):
                    color_list=np.append(color_list,np.append(color_list1[i]/255.,1))
                    color_data_values.append(self.data_values_colors[product][i])
                    if i<len(color_list2)-1 and color_list2[i][0]!=-1.:
                        color_list=np.append(color_list,np.append(color_list2[i]/255.,1.))
                        color_data_values.append(self.data_values_colors[product][i+1])
                     
                #Ensure that product values above the maximum value in self.data_values_color[product], but below the maximum allowed value (given by c_lim[1]),
                #get the correct color, which is the color for the maximum product value in self.data_values_colors[product]
                color_list=np.append(color_list,color_list[-4:])
                color_data_values.append(c_lim[1])
                                            
                color_list=np.reshape(color_list,(int(len(color_list)/4),4))
                color_data_values=np.array(color_data_values)
            
                
                cmap1_range=cm_lim[1]-cm_lim[0]
                controls1=(color_data_values-cm_lim[0])/cmap1_range
                if gv.productunits_default[product] == 'dBZ':
                    # Ensure that values exactly at color boundary get assigned the 'next' color (i.e. for the higher value)
                    controls1[1:-1] -= 1e-6
                
                self.tick_map[product] = {}
                if not tickslim['start'] is None and\
                (self.data_values_colors[product][0] < tickslim['start'] < self.data_values_colors[product][1]):
                    self.tick_map[product][tickslim['start']*scale] = self.data_values_colors[product][0]*scale
                    self.data_values_colors[product][0] = tickslim['start']
                if not tickslim['end'] is None and\
                (self.data_values_colors[product][-1] > tickslim['end'] > self.data_values_colors[product][-2]):
                    self.tick_map[product][tickslim['end']*scale] = self.data_values_colors[product][-1]*scale
                    self.data_values_colors[product][-1] = tickslim['end']
                d_lim=[self.data_values_colors[product][0],self.data_values_colors[product][-1]]
                cmap2_range=d_lim[1]-d_lim[0]
                controls2=(color_data_values[cmap2_starti:-1]-d_lim[0])/cmap2_range
                controls2[controls2 < 0.] = 0.
                controls2[controls2 > 1.] = 1.

                #self.cm1 contains the color map that is used for the image visual (with a blank space for missing data), while 
                #self.cm2 contains the color map that is used for the color bar (without a blank space).

                self.cm1[product]=color.Colormap(color_list, controls=controls1, interpolation='linear')
                self.cm2[product]=color.Colormap(color_list[cmap2_starti:-1], controls=controls2, interpolation='linear')
                
                if step=='-':
                    self.data_values_ticks[product]=self.data_values_colors[product].copy()
                else:
                    #The step assigned in the color table is used as the step between subsequent ticks.
                    start=np.floor(self.data_values_colors[product][0]/step)*step
                    stop=np.ceil(self.data_values_colors[product][-1]/step)*step
                    dvt=np.linspace(start,stop,int(np.round((stop-start)/step+1)))
                    #The addition/subtraction of 1e-6 is to prevent that ticks are removed at the boundaries of the cbar range
                    remove_indices=np.nonzero((dvt+1e-6<self.data_values_colors[product][0]) | (dvt-1e-6>self.data_values_colors[product][-1]))
                    self.data_values_ticks[product]=np.delete(dvt,remove_indices)
                
                # Obtain the values that correspond to the chosen unit.
                self.data_values_ticks[product]*=self.scale_factors[product]
                self.data_values_colors[product]*=self.scale_factors[product]
                
                #Data values that are included for ticks are added here.
                for j in included_values_for_ticks[product_cbar]:
                    if not j in self.data_values_ticks[product] and not (j < self.data_values_colors[product][0] or j > self.data_values_colors[product][-1]):
                        self.data_values_ticks[product]=np.append(self.data_values_ticks[product],j)
                self.data_values_ticks[product]=np.sort(self.data_values_ticks[product])
                
                #Data values that are excluded for ticks are removed here.
                self.data_values_ticks[product]=np.array([j for j in self.data_values_ticks[product] if j not in excluded_values_for_ticks[product_cbar]])
                # Make sure that no repeated values occur. These could otherwise for example occur when the color table extends far
                # beyond the supported product value range, in which case all values beyond the limit are changed to the limit value
                self.data_values_ticks[product] = np.unique(self.data_values_ticks[product])
                
                self.cmap_lastmodification_time[product]=pytime.time()

        self.cmaps_minvalues_before=self.gui.cmaps_minvalues.copy(); self.cmaps_maxvalues_before=self.gui.cmaps_maxvalues.copy()
        return changed_colortables

    def set_polrgb_legend(self):
        """Show a small Bram-style 3-segment legend (Z=red, CC=green, ZDR=blue), stacked vertically (Z on top,
        ZDR at the bottom) in the top-left corner of every panel that currently displays the polarimetric RGB
        composite (product 'g'), reflecting the current Settings -> PolRGB values. Hides the legend for every
        other panel. A reference line (with its value) is added partway through a segment only when that
        segment's value range is wide enough that an intermediate value is actually useful to show -- for a
        narrow range (e.g. ZDR's default 0-3 dB) the min/max ticks already convey the scale clearly enough."""
        params = self.gui.polrgb_params
        # FIX (16 september 2026, na Eriks screenshot: legende toonde nog -10/60/70/100/0/3 -- de
        # passthrough-standaardwaarden -- terwijl ESSL_MODE aanstond en de daadwerkelijke rendering dus de
        # ESSL-tabel gebruikte). Sinds dezelfde sessie is die ESSL-tabel zelf ook instelbaar geworden (keys
        # ESSL_Z_MIN/ESSL_Z_MAX/ESSL_CC_MIN/ESSL_CC_MAX/ESSL_ZDR_MIN/ESSL_ZDR_MAX/ESSL_ALPHA_Z/ESSL_ALPHA_V
        # in polrgb_params, default = de poster's eigen waarden -- zie _essl_polrgb_channels in
        # nlr_datasourcegeneral.py en settings_tabpolrgb/change_polrgb_essl_range hieronder in nlr.py). De
        # legende las tot nu toe altijd rechtstreeks params['Z_MIN'] etc., ongeacht ESSL_MODE -- die
        # parameters worden in ESSL-modus genegeerd door de berekening zelf, maar de legende wist daar niets
        # van. CC is in ESSL-modus omgekeerd (laag CC=hoog groen, i.p.v. de standaard hoog CC=hoog groen) --
        # weergegeven door vmin/vmax om te draaien, zelfde patroon als de bestaande vmin>vmax-omkeerlogica
        # hieronder al ondersteunt voor een handmatig omgekeerd geconfigureerd kanaal.
        essl_mode = bool(params.get('ESSL_MODE', False))
        if essl_mode:
            essl_z_min, essl_z_max = params['ESSL_Z_MIN'], params['ESSL_Z_MAX']
            essl_cc_min, essl_cc_max = params['ESSL_CC_MIN'], params['ESSL_CC_MAX']
            essl_zdr_min, essl_zdr_max = params['ESSL_ZDR_MIN'], params['ESSL_ZDR_MAX']
            bar_specs = [
                ('r', essl_z_min, essl_z_max, 'Z', 'dBZ'),
                ('g', essl_cc_max, essl_cc_min, u'\u03c1HV', '%'),
                ('b', essl_zdr_min, essl_zdr_max, 'ZDR', 'dB'),
            ]
        else:
            bar_specs = [
                ('r', params['Z_MIN'], params['Z_MAX'], 'Z', 'dBZ'),
                ('g', params['CC_MIN'], params['CC_MAX'], u'\u03c1HV', '%'),
                ('b', params['ZDR_MIN'], params['ZDR_MAX'], 'ZDR', 'dB'),
            ]
        range_threshold_for_midtick = 40 #A segment only gets an intermediate reference line/tick when its
        #value range (vmax-vmin) exceeds this. Chosen so that e.g. Z's default 80-point range (-10 to 70) gets
        #one, while CC's default 30-point range and ZDR's default 3-point range don't.

        # Cleared and rebuilt each call (cheap: at most a handful of panels x 3 bars x up to 3 ticks).
        ticks_text, ticks_pos, labels_text, labels_pos = [], [], [], []
        reflines_pos = []

        bar_width = self.scale_pixelsize(12) #Smaller dan voorheen, meer zoals ESSL.
        bar_height_frac = 0.62 #Fraction of the panel's (smaller of width/height) used for the TOTAL stacked
        #height (label rows + bars + gaps between blocks). Larger than before since each bar itself is also taller.
        label_row_height = self.scale_pixelsize(16) #Vertical space reserved for the channel+unit label text itself.
        label_above_gap = self.visuals['polrgb_legend_labels'].font_size*self.scale_pixelsize(1.0) #Gap between the
        #label text and the top of its bar, sized to roughly one font height so the label doesn't crowd the
        #tick value (e.g. '60') that sits right at the top of the bar.
        block_gap = self.scale_pixelsize(36) #Gap between one block's bar (bottom) and the next block's label (top).
        margin_x = self.scale_pixelsize(40) #Horizontal margin from the panel's left edge.
        margin_y = self.scale_pixelsize(40) #Vertical margin from the panel's top edge.
        label_gap = self.scale_pixelsize(6) #Horizontal gap between the bar and its tick-value text.
        # Z-balk krijgt meer ruimte dan CC en ZDR (vergelijkbaar met ESSL), de andere twee zijn gelijk.
        # Z_HEIGHT_FACTOR bepaalt hoe veel groter Z is t.o.v. CC/ZDR (2.0 = Z is twee keer zo hoog).
        Z_HEIGHT_FACTOR = 2.0

        for j in range(self.max_panels):
            is_polrgb_panel = j in self.panellist and self.data_attr['product'].get(j) == 'g' and not self.data_empty.get(j, True)
            for ch, *_ in bar_specs:
                self.visuals['polrgb_legend_bar'+str(j)+ch].visible = is_polrgb_panel
            if not is_polrgb_panel:
                continue

            topleft = self.panel_corners[j][0] #(x, y) of the panel's top-left corner, downward-y screen coords -
            #matching the coordinate convention used elsewhere for un-transformed visuals (e.g. set_individual_cbar).
            left_x = topleft[0]+margin_x
            panel_w, panel_h = self.panel_corners[j][2]-self.panel_corners[j][0]
            total_height = bar_height_frac*min(abs(panel_w), abs(panel_h))
            # Each of the 3 blocks = label_row_height + label_above_gap + segment_height (the bar itself), with
            # block_gap of clear space between consecutive blocks (2 such gaps for 3 blocks).
            fixed_height_per_block = label_row_height+label_above_gap
            # Z (index 0) krijgt Z_HEIGHT_FACTOR keer de hoogte van CC en ZDR (index 1 en 2).
            # Totale hoogte = 3*fixed_height + 2*block_gap + Z_segment + 2*small_segment
            # Z_segment = Z_HEIGHT_FACTOR * small_segment
            # => total_height - 3*fixed - 2*gap = (Z_HEIGHT_FACTOR + 2) * small_segment
            small_segment = (total_height - 3*fixed_height_per_block - 2*block_gap) / (Z_HEIGHT_FACTOR + 2)
            segment_heights = [Z_HEIGHT_FACTOR * small_segment, small_segment, small_segment]
            # Round to whole pixels before computing the center: a half-pixel offset here would make the GPU
            # rasterize the bar's left and right border edges with visibly different effective thickness, even
            # though the border geometry itself is perfectly symmetric (see vispy's border.py) -- this is what
            # caused the left/right border-thickness mismatch reported earlier.
            bar_width_px = round(bar_width)
            bar_center_x = round(left_x)+0.5*bar_width_px

            value_text = lambda v: str(ft.rifdot0(ft.r1dec(v)))
            current_y = topleft[1]+margin_y
            for i, (ch, vmin, vmax, label, unit) in enumerate(bar_specs):
                segment_height = segment_heights[i]
                block_top_y = current_y
                label_y = block_top_y+label_row_height
                seg_top_y = label_y+label_above_gap
                bar_center_y = seg_top_y+0.5*segment_height
                current_y = seg_top_y+segment_height+(block_gap if i < 2 else 0)
                bar = self.visuals['polrgb_legend_bar'+str(j)+ch]
                bar.pos = np.array([bar_center_x, bar_center_y])
                bar.size = np.array([segment_height, bar_width_px]) #(major_axis, minor_axis), matching set_individual_cbar's convention
                bar.border_width = self.scale_pixelsize(1)

                # The bar's own gradient always renders full color at the top / black at the bottom (see the
                # colormap built in __init__), which only lines up with the tick labels below when vmin <
                # vmax. When a channel is intentionally configured the other way around (vmin > vmax, e.g.
                # CC_MIN=100/CC_MAX=70 -- see the PolRGB CC discussion), the value that actually produces
                # this channel's full intensity (always vmax, regardless of whether vmax is numerically
                # larger or smaller than vmin) ends up at the BOTTOM label position once sorted below, so the
                # bar's gradient needs to be reversed too, or the bright end of the bar would visually
                # contradict which label is next to it.
                channel_color = self.polrgb_channel_colors[ch]
                n_steps = 64
                t = np.linspace(0.0, 1.0, n_steps)

                if ch == 'r':
                    # Z-kanaal: gamma-gecorrigeerd zodat de balk hetzelfde niet-lineaire verloop toont
                    # als het echte radarbeeld. Z_GAMMA>1: balk blijft lang donker, wordt pas laat rood.
                    # In ESSL-modus is het Z-kanaal een gewone lineaire mapping (geen gamma-curve, zie
                    # _essl_polrgb_channels) -- vandaar hier vast op 1.0 i.p.v. params['Z_GAMMA'].
                    z_gamma = 1.0 if essl_mode else max(params.get('Z_GAMMA', 2.0), 0.1)
                    intensity = t ** z_gamma
                    cr = np.array(channel_color[:3])  # (1,0,0)
                    if vmin > vmax:
                        gradient = [tuple(float(v)*cr[k] for k in range(3)) + (1.0,) for v in reversed(intensity)]
                    else:
                        gradient = [tuple(float(v)*cr[k] for k in range(3)) + (1.0,) for v in intensity]
                else:
                    # CC en ZDR: lineair, maar wél meelopen met de ingestelde grenzen (zwart=vmin, vol=vmax).
                    cr = np.array(channel_color[:3])
                    if vmin > vmax:
                        gradient = [tuple(float(v)*cr[k] for k in range(3)) + (1.0,) for v in reversed(t)]
                    else:
                        gradient = [tuple(float(v)*cr[k] for k in range(3)) + (1.0,) for v in t]
                bar.cmap = color.Colormap(gradient)

                #The larger of the 2 configured values goes at the top of the segment, the smaller at the
                #bottom -- this must be based on which value is actually larger, not simply on which one is
                #called vmin/vmax, since a channel's min/max can be intentionally configured with vmin >
                #vmax (e.g. to invert that channel's mapping). Using vmax/vmin directly here previously
                #assumed vmax > vmin always, which put the smaller number at the top and larger at the
                #bottom whenever a channel was configured the other way around.
                value_top, value_bottom = max(vmin, vmax), min(vmin, vmax)
                ticks_text.append(value_text(value_top)); ticks_pos.append([bar_center_x+0.5*bar_width_px+label_gap, seg_top_y])
                ticks_text.append(value_text(value_bottom)); ticks_pos.append([bar_center_x+0.5*bar_width_px+label_gap, seg_top_y+segment_height])
                if abs(vmax-vmin) > range_threshold_for_midtick:
                    mid_value = 0.5*(vmin+vmax)
                    mid_y = seg_top_y+0.5*segment_height
                    ticks_text.append(value_text(mid_value)); ticks_pos.append([bar_center_x+0.5*bar_width_px+label_gap, mid_y])
                    #Reference line drawn fully across the segment width, matching the existing cbars_reflines style.
                    reflines_pos.append([bar_center_x-0.5*bar_width_px, mid_y])
                    reflines_pos.append([bar_center_x+0.5*bar_width_px, mid_y])

                #Label (with unit) placed directly above the bar, left-aligned with the bar's left edge.
                labels_text.append(label+(' ('+unit+')' if unit else ''))
                # Correctie (6 juli 2026): net als anchor_y='top' elders in dit bestand, blijkt anchor_y='bottom'
                # in vispy 0.14.1 ook niet zuiver als bedoeld ankerpunt behandeld te worden -- de tekst kwam te
                # laag uit (overlappend met de tick/balk eronder). Deze correctie raakt ALLEEN de tekstpositie
                # (labels_pos), niet label_y zelf, want die laatste wordt ook gebruikt om de balk-layout
                # (seg_top_y/bar_center_y) te berekenen -- een eerdere versie paste per ongeluk label_y zelf aan,
                # waardoor de balkjes zelf ook mee omhoog schoven.
                label_text_y = label_y - 1.5*self.visuals['polrgb_legend_labels'].font_size*self.scale_pixelsize(1.0)
                labels_pos.append([bar_center_x-0.5*bar_width_px, label_text_y])


        if len(ticks_text) > 0:
            self.visuals['polrgb_legend_ticks'].text = ticks_text
            self.visuals['polrgb_legend_ticks'].pos = np.array(ticks_pos)
        else:
            self.visuals['polrgb_legend_ticks'].text = []
        if len(labels_text) > 0:
            self.visuals['polrgb_legend_labels'].text = labels_text
            self.visuals['polrgb_legend_labels'].pos = np.array(labels_pos)
        else:
            self.visuals['polrgb_legend_labels'].text = []
        if len(reflines_pos) > 0:
            self.visuals['polrgb_legend_reflines'].set_data(pos=np.array(reflines_pos), connect='segments', color=(1,1,1,0.8))
            self.visuals['polrgb_legend_reflines'].visible = True
        else:
            self.visuals['polrgb_legend_reflines'].visible = False

    def set_hclass_legend(self):
        """Toont voor elk paneel dat momenteel de hydrometeorenclassificatie (product 'j', zie
        DataSource_General._calculate_hclass en nlr_hclass.py) toont een kolom van gekleurde vierkantjes
        (een per klasse, zie nlr_hclass.HID_CLASSES/HID_COLORS_RGBA) met de klassenaam ernaast, in de
        linkerbovenhoek van het paneel. Verborgen voor elk ander paneel. Zelfde opzet als
        set_polrgb_legend hierboven (een gedeelde MarkersVisual/TextVisual voor alle panelen samen,
        per redraw opnieuw opgebouwd), maar dan voor een categorische in plaats van een continue schaal.
        """
        marker_pos, marker_colors = [], []
        labels_text, labels_pos = [], []

        margin_x = self.scale_pixelsize(10)
        margin_y = self.scale_pixelsize(20)
        swatch_size = self.scale_pixelsize(12)
        row_height = self.scale_pixelsize(16)
        label_gap = self.scale_pixelsize(6)

        # Volgorde van boven naar beneden: heftig -> licht, consistent met de kleurtabel
        # (colortable_HCLASS_default.csv, waar dezelfde volgorde van boven (heftig) naar beneden (licht)
        # wordt gebruikt voor het kleine kleurenbalkje naast het paneel). Afgesproken met Erik (22 juli) na
        # gesprek over hevigheid en de dubbelzinnige positie van Verticaal ijs (weinig neerslagmassa, maar
        # wel een actief stormsignaal - vandaar vlak boven IJskristallen i.p.v. helemaal onderaan).
        # Dit is GEEN wetenschappelijk vastgestelde hevigheidsschaal, puur een redelijke, met Erik afgestemde
        # inschatting - de interne klasse-nummers (hc.HID_CLASSES-index) blijven ongewijzigd en bepalen niets
        # aan deze volgorde.
        severity_order_top_to_bottom = [
            'Hagel', 'Grote druppels', 'Zware graupel', 'Lichte graupel', 'Natte sneeuw',
            'Regen', 'Sneeuwvlokken', 'Verticaal ijs', 'IJskristallen', 'Motregen',
        ]
        name_to_class_id = {name: i+1 for i, name in enumerate(hc.HID_CLASSES)}

        for j in range(self.max_panels):
            is_hclass_panel = j in self.panellist and self.data_attr['product'].get(j) == 'j' and not self.data_empty.get(j, True)
            if not is_hclass_panel:
                continue
            topleft = self.panel_corners[j][0] #(x, y) van de linkerbovenhoek van het paneel, zelfde conventie
            #als bij set_polrgb_legend/set_individual_cbar.
            swatch_x = topleft[0]+margin_x+0.5*swatch_size
            current_y = topleft[1]+margin_y+0.5*swatch_size
            for class_name in severity_order_top_to_bottom:
                class_id = name_to_class_id[class_name]
                color_rgba = np.array(hc.HID_COLORS_RGBA[class_id])/255.
                marker_pos.append([swatch_x, current_y])
                marker_colors.append(color_rgba)
                labels_text.append(class_name)
                labels_pos.append([swatch_x+0.5*swatch_size+label_gap, current_y])
                current_y += row_height

        if marker_pos:
            self.visuals['hclass_legend_markers'].set_data(
                pos=np.array(marker_pos), symbol='square', size=swatch_size,
                face_color=np.array(marker_colors), edge_color=(1,1,1,0.8),
                edge_width=self.scale_pixelsize(1))
            self.visuals['hclass_legend_markers'].visible = True
        else:
            self.visuals['hclass_legend_markers'].visible = False

        if labels_text:
            self.visuals['hclass_legend_labels'].text = labels_text
            self.visuals['hclass_legend_labels'].pos = np.array(labels_pos)
        else:
            self.visuals['hclass_legend_labels'].text = []

    def set_individual_cbar(self,product,cbar_posnumber):
        j=cbar_posnumber
        j_side=int(np.mod(j,4))
        p='left' if j<4 else 'right'
        s=-1 if p=='left' else 1
        n_cbars_side=len([j for j in self.cbars_pos.values() if (j < 4 if p=='left' else j>3)])
        dv=self.data_values_colors[product][::-1]
        dvt=self.data_values_ticks[product][::-1]
        i=0 if s==-1 else -1 #Index of relevant corner used below
        
        cbar_blocksize=np.array([0.28*self.wsize[p][0],self.wsize[p][1]/n_cbars_side])
        cbar_size=cbar_blocksize-np.array([0,self.wsize['top'][1]+self.wsize['bottom'][1]+1]) #The +1 appears to provide better results
        if cbar_size[1]<cbar_size[0]: cbar_size=np.array([1,1]) #The major axis is not allowed to be smaller than the minor axis
        if cbar_size[0] == 0: cbar_size[0] = 1 #Ensure that the width is at least 1 pixel
        dx = 1 if p == 'left' else 0 #An extra pixel of whitespace appears to be needed on the left side in order to get the same white margin between the main widget
        # and the colorbars on both sides
        cbar_pos=np.array([self.wpos[p][s,0]+s*self.scale_pixelsize(3+dx)+0.5*s*cbar_blocksize[0],(j_side+0.5)*cbar_blocksize[1]])
        
        cbar_corners=cbar_pos+0.5*cbar_size*np.array([[-1,-1],[-1,1],[1,1],[1,-1]])
        
        cbars_ticks_xpos=cbar_corners[i,0]+0.5*s*np.abs(self.wbounds[p][0,i]-cbar_corners[i,0])
        cbars_ticks_relpos=np.array([0,cbar_size[1]])*np.reshape((dvt-dv[0])/(dv[-1]-dv[0]),(len(dvt),1))
        a = -self.scale_pixelsize(2)*self.visuals['cbars_ticks'].font_size/7.5
        cbars_ticks_relpos[0, 1] -= a; cbars_ticks_relpos[-1, 1] += a
        
        ctr=cbars_ticks_relpos
        cbars_reflines_relpos=ctr[(np.abs(dvt-dv[0])>1e-3) & (np.abs(dvt-dv[-1])>1e-3)] #Don't put reference lines next to ticks that are located near the upper
        #or lower edge of the cbar, because that is not needed, and looks ugly.
        
        self.visuals['cbar'+str(j)].cmap=self.cm2[product]
        self.visuals['cbar'+str(j)].pos=cbar_pos
        self.visuals['cbar'+str(j)].size=cbar_size[::-1] #Size should have format (major_axis, minor_axis)
        self.visuals['cbar'+str(j)].border_width = self.scale_pixelsize(1) # Setting border_width is important for when screen properties changes
                
        dvt = [k if not k in self.tick_map[product] else self.tick_map[product][k] for k in dvt]
        self.cbars_ticks[j]=[str(ft.rifdot0(ft.rndec(k,3))) for k in (dvt if product!='r' else np.power(10,dvt))]
        self.cbars_ticks_pos[j]=np.array([cbars_ticks_xpos,cbar_corners[i,1]])+cbars_ticks_relpos
        
        self.cbars_reflines_pos[j]=np.zeros((4*len(cbars_reflines_relpos),2))
        self.cbars_reflines_pos[j][::4]=cbar_corners[0]+cbars_reflines_relpos
        self.cbars_reflines_pos[j][1::4]=cbar_corners[0]+cbars_reflines_relpos+0.25*np.array([cbar_size[0],0])
        self.cbars_reflines_pos[j][2::4]=cbar_corners[-1]+cbars_reflines_relpos
        self.cbars_reflines_pos[j][3::4]=cbar_corners[-1]+cbars_reflines_relpos-0.25*np.array([cbar_size[0],0])
        
        self.cbars_labels[j]=[str(self.productunits[product]),str(gv.productnames_cmaps[product])]
        widget_avg_xpos=np.mean(self.wbounds[p][0])
        ts_y=self.wsize['top'][1]
        # Zelfde correctie als bij set_titles() (6 juli 2026): anchor_y='top' wordt in vispy 0.14.1 niet
        # zuiver als bovenkant-ankerpunt behandeld, waardoor deze labels (net als de titel) een fontgrootte
        # te hoog uitkwamen en over de "-20"/andere tick-labels heen stonden. Zelfde compenserende
        # verschuiving toegepast.
        _cbarlabel_font_px = self.scale_pointsize(eval(self.font_sizes['cbars_labels']))
        # Extra horizontale correctie (6 juli 2026) voor alleen de onderste rij (productnaam, bv. 'PolRGB'):
        # stond aan de linkerkant iets te ver naar links en aan de rechterkant iets te ver naar rechts --
        # d.w.z. te ver naar de buitenrand. 's' is -1 voor 'left' en +1 voor 'right' (zie hierboven), dus
        # -s*bedrag verschuift in beide gevallen juist iets naar het midden toe.
        # Correctie (6 juli 2026, bijgesteld): deze horizontale verschuiving bleek specifiek nodig voor de
        # (langere) 'PolRGB'-tekst, maar duwde de korte 'Z'/'V'-productlabels van gewone producten juist uit
        # hun al-correcte gecentreerde positie. Daarom alleen toepassen voor product 'g' (PolRGB) en 'j'
        # (HCLASS) - beide hebben een langere naam dan de meeste gewone producten (22 juli).
        _bottom_label_x_shift = -s*self.scale_pixelsize(15) if product in ('g', 'j') else 0
        self.cbars_labels_pos[j]=np.array([[widget_avg_xpos,cbar_corners[0,1]-ts_y+self.scale_pixelsize(1)+1.5*_cbarlabel_font_px],
                                           [widget_avg_xpos+_bottom_label_x_shift,cbar_corners[1,1]+self.scale_pixelsize(4.5)+1.5*_cbarlabel_font_px]])
           
    def set_cbars_ticks_and_labels(self):        
        cbars_ticks=np.concatenate(list(self.cbars_ticks.values()))
        cbars_ticks_pos=np.concatenate(list(self.cbars_ticks_pos.values()))
        
        cbars_reflines_pos=np.concatenate(list(self.cbars_reflines_pos.values()))
        cbars_reflines_connect=np.ravel(np.array([[1,0] for j in range(0,int(len(cbars_reflines_pos)/2))])).astype('bool')

        cbars_labels=np.concatenate(list(self.cbars_labels.values()))
        cbars_labels_pos=np.concatenate(list(self.cbars_labels_pos.values()))

        self.visuals['cbars_ticks'].text=cbars_ticks; self.visuals['cbars_ticks'].pos=cbars_ticks_pos
        self.visuals['cbars_reflines'].set_data(pos=cbars_reflines_pos,connect=cbars_reflines_connect,width=self.scale_pixelsize(1))
        self.visuals['cbars_labels'].text=cbars_labels; self.visuals['cbars_labels'].pos=cbars_labels_pos

    def set_cbars(self,resize=False,set_cmaps=True):   
        self.cbars_products, self.cbars_nproducts, self.cbars_pos, update_cbars_pos, update_cbars_products=bg.determine_colortables(self.data_attr['product'],self.panels,[j for j in self.panellist if not self.data_empty[j]])

        if set_cmaps:
            self.changed_colortables = self.set_cmaps(self.cbars_products)
        for j in range(0,len(self.cbars_products)):
            if self.cbars_products[j] in self.changed_colortables and update_cbars_products.count(self.cbars_products[j])<self.cbars_products.count(self.cbars_products[j]):
                update_cbars_pos.append(self.cbars_pos[j])
                update_cbars_products.append(self.cbars_products[j])
        
        if resize or self.cbars_products!=self.cbars_products_before:
            self.cbars_ticks={}; self.cbars_reflines={}; self.cbars_labels={}
            self.cbars_ticks_pos={}; self.cbars_reflines_pos={}; self.cbars_labels_pos={}
            
            update_cbars_products=self.cbars_products; update_cbars_pos=[self.cbars_pos[j] for j in sorted(list(self.cbars_pos.keys()))] #Sorting is important
            
            for j in range(self.max_panels):
                self.visuals['cbar'+str(j)].visible=j in self.cbars_pos.values()
                              
        for j in range(0,len(update_cbars_products)):
            self.set_individual_cbar(update_cbars_products[j],update_cbars_pos[j])
        if len(update_cbars_products)>0:
            self.set_cbars_ticks_and_labels()
            
        self.cbars_products_before=self.cbars_products.copy()
        return update_cbars_products                            
                
    def get_corner_specs(self):
        return [str(self.panels_sttransforms[j].translate[:2])+str(self.panels_sttransforms[j].scale[:2]) for j in self.panellist]
    def get_corners(self):
        if not hasattr(self, 'corner_specs') or self.get_corner_specs() != self.corner_specs:
            # Without setting panel=j, the right edge of the panel is actually considered to be part of the next panel, causing the calculated
            # right panel corners to be the same as the left corners 
            self.corners = {j:self.screencoord_to_xy(self.panel_corners[j], panel=j) for j in self.panellist}
            self.xdim, self.ydim = self.corners[0][-1]-self.corners[0][1]
            self.corner_specs = self.get_corner_specs()
          
    def set_grid(self, panellist=None):
        panellist = self.panellist if panellist is None else panellist
        
        self.get_corners()
        physical_size_cm_main = self.physical_size_cm() - 2*self.scale_physicalsize(self.wdims)
        rel_xdim = self.size[0]/self.gui.screen_size()[0] * (1-self.vwp_relxdim*self.gui.show_vwp)
        self.gridlines_vertices, self.gridlines_connect, self.gridlines_text_hor_pos, self.gridlines_text_hor, self.gridlines_text_vert_pos, self.gridlines_text_vert =\
            bg.determine_gridpos(physical_size_cm_main,rel_xdim,self.corners,self.visuals['text_hor1'][0].font_size,self.panels,panellist,self.nrows,self.ncolumns,self.gui.show_vwp)
        self.update_combined_lineproperties(panellist,changing_grid=True)
        self.update_combined_ghtextproperties(panellist,changing_grid=True)
         
    def set_heightrings(self, panellist=None):
        panellist = self.panellist if panellist is None else panellist
        panellist = [j for j in panellist if not self.data_empty[j]]
        
        self.get_parameters_heightrings()
        
        for j in panellist:
            product = self.data_attr['product'][j]
            derivedproduct_noheightrings=product in self.gui.show_heightrings_derivedproducts and (
            self.gui.show_heightrings_derivedproducts[product]==0)
            if not derivedproduct_noheightrings and len(self.vertices_circles[j])>0:
                if product in gv.plain_products_show_true_elevations:
                    # Don't show text for heights for which the plain product has been calculated, since it is already shown in the title, 
                    # and it can lead to annoyingly closely spaced text labels for low-elevation PCAPPIs.
                    h_remove = {'a':self.gui.PP_parameter_values['a'][self.gui.PP_parameters_panels[j]], 'r':gv.CAPPI_height_R}
                    i_heights_remove = [i for i,h in enumerate(self.heights[j]) if h == h_remove[product]]
                    for o in ('heights', 'radii', 'textangles'):
                        self.__dict__[o][j] = np.delete(self.__dict__[o][j], i_heights_remove)
                
                self.heights_text[j] = ft.format_nums(self.heights[j], dec=2)
                if product not in gv.plain_products_show_max_elevations:
                    self.heights_text_pos[j]=[self.radii[j][i]*np.array([np.sin(self.textangles[j][i]),np.cos(self.textangles[j][i])]) for i in range(0,len(self.radii[j]))]
                else:
                    pos_offset=20*self.ydim*self.nrows/1e3
                    if product in gv.plain_products:
                        heights_text_pos1=[(self.radii[j][i]+pos_offset)*np.array([np.sin(self.textangles[j][i]),np.cos(self.textangles[j][i])]) for i in range(len(self.radii[j])-len(self.dp.meta_PP[product]['elevations_minside']),len(self.radii[j]))]
                    heights_text_pos2=[(self.radii[j][i]-pos_offset)*np.array([np.sin(self.textangles[j][i]),np.cos(self.textangles[j][i])]) for i in range(0,len(self.radii[j]))]
                    if product in gv.plain_products and len(heights_text_pos1)>0:
                        self.heights_text_pos[j]=np.concatenate(tuple([heights_text_pos1,heights_text_pos2]))
                    else: self.heights_text_pos[j]=heights_text_pos2
            else:
                self.vertices_circles[j]=[]; self.vertices_circles_connect[j]=[]
                self.heights_text[j]=[]; self.heights_text_pos[j]=[] 
                
        self.update_combined_lineproperties(panellist,changing_heightrings=True)
        self.update_combined_ghtextproperties(panellist,changing_heightrings=True)
            
    def update_heightrings_scanangles(self):
        heightrings_updated = False
        for j in self.data_attr['scanangle']:
            if not self.crd.lrstep_beingperformed or abs(self.heightrings_scanangles[j]-self.data_attr['scanangle'][j]) > 0.05:
                self.heightrings_scanangles[j] = self.data_attr['scanangle'][j]
                heightrings_updated = True
        return heightrings_updated
        
    def get_parameters_heightrings(self,start=0):
        if not self.dsg.scanangles_all['z']:
            return
        
        self.get_corners()
        
        panellist_nonempty=[j for j in self.panellist if not self.data_empty[j]]
        panellist_no_pp=[j for j in panellist_nonempty if not self.data_empty[j] and self.data_attr['product'][j] not in gv.plain_products]
        panellist_pp_max_elevations=[j for j in panellist_nonempty if not self.data_empty[j] and self.data_attr['product'][j] in gv.plain_products_show_max_elevations] 
        panellist_pp_true_elevations=[j for j in panellist_nonempty if not self.data_empty[j] and self.data_attr['product'][j] in gv.plain_products_show_true_elevations] 
        panellist_determine_heightrings = panellist_no_pp+panellist_pp_true_elevations
        
        self.heights={}; self.radii={}
        if len(panellist_determine_heightrings):
            rel_xdim = self.size[0]/self.gui.screen_size()[0] * (1-self.vwp_relxdim*self.gui.show_vwp)
            self.update_heightrings_scanangles()
            angle1 = self.dsg.scanangle('z', 1, 0)
            scanangles = np.array([self.heightrings_scanangles[j] if j in panellist_no_pp else angle1 for j in panellist_determine_heightrings])
            use_previous_hrange = self.crd.lrstep_beingperformed
            self.heights, self.radii = bg.determine_heightrings(rel_xdim,self.corners,self.ncolumns,panellist_determine_heightrings,scanangles,use_previous_hrange)
        if len(panellist_pp_true_elevations):
            panel = panellist_pp_true_elevations[0]
            heights_scan1, radii_scan1 = self.heights[panel], self.radii[panel]
            new_heights, new_radii = self.dp.get_true_elevations_plainproducts(panellist_pp_true_elevations, heights_scan1, radii_scan1)
            self.heights={**self.heights,**new_heights}; self.radii={**self.radii,**new_radii}
        if len(panellist_pp_max_elevations):
            for j in panellist_pp_max_elevations:
                product = self.data_attr['product'][j]
                # VANGNET (25 juli, na Eriks crash "KeyError: elevations_minside" bij VILD/ALT+L): de
                # onderliggende oorzaak (waarom self.dp.meta_PP[product] deze sleutels soms mist) is nog
                # niet gevonden - dit voorkomt in elk geval de crash (geen hoogtecirkel-tekst voor dat
                # paneel i.p.v. een afgesloten programma) en print een regel zodat de volgende keer
                # duidelijk is WELK product en WELKE sleutels ontbraken.
                meta = self.dp.meta_PP.get(product, {})
                missing = [k for k in ('elevations_minside','elevations_plusside','scans_ranges') if k not in meta]
                if missing:
                    print(f'WAARSCHUWING: meta_PP[{product!r}] mist {missing} - geen hoogtecirkel-tekst voor dit paneel')
                    self.heights[j] = np.array([]); self.radii[j] = np.array([])
                else:
                    self.heights[j]=np.concatenate((meta['elevations_minside'],meta['elevations_plusside']))
                    self.radii[j]=meta['scans_ranges']
        
        self.textangles=bg.determine_textangles(self.corners,self.panels,panellist_nonempty,self.data_attr['product'],self.radii)
                    
        self.vertices_circles={}; self.vertices_circles_connect={}
        for i in self.panellist:
            if i in panellist_nonempty and len(self.radii[i])>0:
                for j, r in enumerate(self.radii[i]):
                    vertices_add=r*self.unitcircle_vertices
                    if j==0:
                        self.vertices_circles[i]=vertices_add
                        self.vertices_circles_connect[i]=np.ones(len(vertices_add))
                    else:
                        self.vertices_circles[i]=np.concatenate((self.vertices_circles[i],vertices_add),axis=0)
                        self.vertices_circles_connect[i]=np.append(self.vertices_circles_connect[i],np.ones(len(vertices_add)))
                    self.vertices_circles_connect[i][-1]=0
                self.vertices_circles_connect[i]=self.vertices_circles_connect[i].astype('bool')
            else: self.vertices_circles[i]=[]; self.vertices_circles_connect[i]=[]
                                    
              
    def update_combined_ghtextproperties(self,panellist,changing_heightrings=False,changing_grid=False):
        for i in panellist:
            # if i in getattr(self, 'dont_draw', []):
            #     continue
            if changing_grid:
                self.ghtext_hor_pos_combined[i]['grid']=self.gridlines_text_hor_pos[i]
                self.ghtext_hor_strings_combined[i]['grid']=self.gridlines_text_hor[i]
                self.ghtext_vert_pos_combined[i]['grid']=self.gridlines_text_vert_pos[i]
                self.ghtext_vert_strings_combined[i]['grid']=self.gridlines_text_vert[i]
            if changing_heightrings:
                self.ghtext_hor_pos_combined[i]['heightrings']=self.heights_text_pos[i]
                self.ghtext_hor_strings_combined[i]['heightrings']=self.heights_text[i]
            
    def include_texttype(self,panel,text_type,hor_vert):
        pos_combined=self.ghtext_hor_pos_combined if hor_vert=='hor' else self.ghtext_vert_pos_combined
        return text_type in self.gui.ghtext_show and not len(pos_combined[panel][text_type])==0 and not (
        text_type in ('grid','heightrings') and (self.gridheightrings_removed or self.data_empty[panel]))
    def set_ghtextproperties(self,panellist):
        self.ghtext_order = ['grid','heightrings'] #From bottom to top
        for i in self.panels_horizontal_ghtext:
            if i in panellist:
                pos = []
                try: #The multiplication by -1 for the y coordinates is because the y axis is flipped in VisPy
                    pos = np.array([1,-1])*np.concatenate([self.ghtext_hor_pos_combined[i][j] for j in self.ghtext_order if self.include_texttype(i,j,'hor')])
                    text = np.concatenate([self.ghtext_hor_strings_combined[i][j] for j in self.ghtext_order if self.include_texttype(i,j,'hor')])
                    self.visuals['text_hor1'][i].text = text
                    self.visuals['text_hor1'][i].pos = pos
                except Exception:
                    pass
                self.visuals['text_hor1'][i].visible = self.visuals['text_hor2'][i].visible = len(pos) > 0
        for i in self.panels_vertical_ghtext:
            if i in panellist:
                pos = []
                if 'grid' in self.gui.ghtext_show:
                    try:
                        pos = np.array([1,-1])*np.concatenate([self.ghtext_vert_pos_combined[i][j] for j in self.ghtext_order if self.include_texttype(i,j,'vert')])
                        ghtext_vert_strings = np.concatenate([self.ghtext_vert_strings_combined[i][j] for j in self.ghtext_order if self.include_texttype(i,j,'vert')])
                        self.visuals['text_vert1'][i].text = ghtext_vert_strings
                        self.visuals['text_vert1'][i].pos = pos
                    except Exception:
                        pass
                self.visuals['text_vert1'][i].visible = self.visuals['text_vert2'][i].visible = len(pos) > 0
                
    def update_combined_lineproperties(self,panellist,changing_radar=False,changing_colors=False,changing_grid=False,changing_heightrings=False,start=False):
        for i in panellist:
            if changing_radar:
                for j in self.shapefiles_latlon_combined:
                    self.lines_pos_combined[i][j]=self.shapefiles_latlon_combined[j]
                    if start: 
                        self.lines_connect_combined[i][j]=self.shapefiles_connect_combined[j]
                        self.lines_colors_combined[i][j]=np.ones((len(self.lines_pos_combined[i][j]),4))*self.gui.lines_colors[j]/255.
            if changing_grid:
                self.lines_pos_combined[i]['grid']=self.gridlines_vertices[i]
                self.lines_connect_combined[i]['grid']=self.gridlines_connect[i]
                self.lines_colors_combined[i]['grid']=np.ones((len(self.gridlines_vertices[i]),4))*self.gui.lines_colors['grid']/255.
            if changing_heightrings:
                self.lines_pos_combined[i]['heightrings']=self.vertices_circles[i]
                self.lines_connect_combined[i]['heightrings']=self.vertices_circles_connect[i]
                self.lines_colors_combined[i]['heightrings']=np.ones((len(self.vertices_circles[i]),4))*self.gui.lines_colors['heightrings']/255.
                        
            if changing_colors:
                for j in self.gui.lines_names:
                    self.lines_colors_combined[i][j]=np.ones((len(self.lines_pos_combined[i][j]),4))*self.gui.lines_colors[j]/255.
                    
    
    def include_linetype(self,panel,line_type):
        if line_type in self.gui.lines_show and not len(self.lines_pos_combined[panel][line_type])==0 and not (
        line_type in ('grid','heightrings') and (self.gridheightrings_removed or self.data_empty[panel])):
            return True
        else:
            return False       
    def set_maplineproperties(self,panellist):   
        if self.gui.basemap_source == 'MapTiler':
            # The MapTiler basemap already renders its own country/province borders and place labels, so
            # showing NLradar's own map_lines layer on top would just duplicate them (potentially with a
            # slightly different, distracting style/position). Hide the whole layer rather than only some
            # of the configured line types, and leave self.gui.lines_show itself untouched so the previous
            # selection is restored automatically when switching back to the local tiles.
            for i in range(self.max_panels):
                self.visuals['map_lines'][i].visible=False
            return
        lines = [j for j in self.lines_order[:3] if j in self.gui.lines_show]
        if any([j in self.gui.lines_show for j in lines]):
            for i in range(self.max_panels):
                self.visuals['map_lines'][i].visible=True
                
            lines_pos=np.concatenate([self.lines_pos_combined[i][j] for j in lines if self.include_linetype(i,j)])
            lines_connect=np.concatenate([self.lines_connect_combined[i][j] for j in lines if self.include_linetype(i,j)])
            lines_colors=np.concatenate([self.lines_colors_combined[i][j] for j in lines if self.include_linetype(i,j)])
            
            self.visuals['map_lines'][0].set_data(pos=lines_pos,connect=lines_connect,color=lines_colors,width=self.scale_pixelsize(self.gui.lines_width))   
        else: 
            for i in range(self.max_panels):
                self.visuals['map_lines'][i].visible=False
                
    def set_ghlineproperties(self,panellist):
        lines = [j for j in self.lines_order[-2:] if j in self.gui.lines_show]
        for i in panellist:
            if len(lines)>0:
                self.visuals['gh_lines'][i].visible=True
                try:
                    lines_pos=np.array([1,-1])*np.concatenate([self.lines_pos_combined[i][j] for j in lines if self.include_linetype(i,j)])                 
                    lines_connect=np.concatenate([self.lines_connect_combined[i][j] for j in lines if self.include_linetype(i,j)])
                    lines_colors=np.concatenate([self.lines_colors_combined[i][j] for j in lines if self.include_linetype(i,j)])
                    # if self.visuals['gh_lines'][i].pos is None or lines_pos.shape != self.visuals['gh_lines'][i].pos.shape or (lines_pos != self.visuals['gh_lines'][i].pos).any():
                    self.visuals['gh_lines'][i].set_data(pos=lines_pos,connect=lines_connect,color=lines_colors,width=self.scale_pixelsize(self.gui.lines_width))   
                except Exception:
                    #Occurs when no line coordinates are available, such that np.concatenate raises an exception.
                    self.visuals['gh_lines'][i].visible=False
            else: 
                self.visuals['gh_lines'][i].visible=False
                            
    def set_sm_pos_markers(self):
        # Update cartesian positions because of a possible switch of radar
        if self.gui.sm_marker_present:
            self.gui.sm_marker_position = ft.aeqd(gv.radarcoords[self.crd.radar], self.gui.sm_marker_latlon)
        self.gui.pos_markers_positions = [ft.aeqd(gv.radarcoords[self.crd.radar], j) for j in self.gui.pos_markers_latlons]
        
        pos = self.gui.pos_markers_positions + ([self.gui.sm_marker_position] if self.gui.sm_marker_present else [])
        if len(pos):
            face_color = ['white']*len(self.gui.pos_markers_positions) + (['red'] if self.gui.sm_marker_present else [])
            #-1 because y-coordinate is flipped
            self.visuals['sm_pos_markers'][0].scaling = False
            self.visuals['sm_pos_markers'][0].set_data(pos=np.array(pos)*np.array([1,-1]),symbol='disc',size=self.scale_pixelsize(9),
                                                       edge_width=1,face_color=face_color,edge_color='black')
        for j in range(self.max_panels):
            self.visuals['sm_pos_markers'][j].visible = len(pos) > 0
                                   
    def change_mapvisibility(self,visibility):
        self.gui.mapvisibility=True if self.gui.mapvis_true.isChecked() else False
        for j in self.panellist:
            if not self.gui.mapvisibility:
                self.visuals['map'][j].visible=False
            else:                        
                self.visuals['map'][j].visible=True
        self.update_map_tiles()
        self.update()
                                   
    def change_mapcolorfilter(self):
        inputfilter=self.gui.mapcolorfilterw.text()
        color_filter= ft.rgb(inputfilter,alpha=True)
        if color_filter: #ft.rgb returns False if the input is not an RGB(A) series
            self.gui.mapcolorfilter = color_filter
            self.map_colorfilter.filter = self.gui.mapcolorfilter
            self.update()
        else:
            self.gui.mapcolorfilterw.setText(ft.list_to_string(self.gui.mapcolorfilter))

    def change_radardata_opacity(self, value):
        alpha = value/100.
        self.gui.radardata_colorfilter = (1.0, 1.0, 1.0, alpha)
        self.radardata_colorfilter.filter = self.gui.radardata_colorfilter
        self.gui.radardata_opacity_label.setText(str(value)+'%')
        self.update()
            
    def change_radarimage_visibility(self):
        for j in self.panellist:
            radar_image = 'radar_polar' if self.data_attr['proj'][j] == 'pol' else 'radar_cartesian'
            if self.visuals[radar_image][j].visible:
                self.visuals[radar_image][j].visible=False
            else:
                self.visuals[radar_image][j].visible=True      
        self.update()
