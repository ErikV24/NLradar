# -*- mode: python -*-
# PyInstaller-spec voor de deelbare NLradar-installer. Wordt aangeroepen door build_installer.py,
# met als werkmap build_installer\stage\Python_files (schone kopie van Python_files).
import os, sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules
sys.setrecursionlimit(10000)

here = os.getcwd()

hiddenimports = (
    ['vispy.app.backends._pyqt5', 'vispy.glsl',
     'netCDF4.utils', 'cftime', 'h5py.defs', 'h5py.utils', 'h5py.h5ac', 'h5py._proxy',
     'PyQt5.QtOpenGL', 'PyQt5.QtTest', 'OpenGL.platform.win32',
     # eigen submappen zonder __init__.py (namespace packages) expliciet meenemen
     'derived.cartesian', 'derived.polar', 'derived.nlr_derived_plain', 'derived.nlr_derived_tilts',
     'VWP.nlr_vwp', 'VWP.nlr_plottingvwp', 'VWP.vvp', 'VWP.vwp_functions', 'VWP.sfc_obs',
     'decoders.dorade', 'decoders.nexrad_l2', 'decoders.nexrad_l3', 'decoders.ukmo_polar',
     'dealiasing.nlr_dealiasing', 'dealiasing.unet_vda.unet_vda',
     'dealiasing.unet_vda.src.dealias', 'dealiasing.unet_vda.src.feature_extraction',
     'dealiasing.unet_vda.src.layers']
    + collect_submodules('numpy_bufr') + collect_submodules('nexradaws')
    + collect_submodules('vispy.app.backends')
)

datas = collect_data_files('vispy') + collect_data_files('pyproj') + collect_data_files('certifi')

a = Analysis([os.path.join(here, 'nlr.py')],
             pathex=[here],
             binaries=[],
             datas=datas,
             hiddenimports=hiddenimports,
             hookspath=[],
             runtime_hooks=[],
             excludes=['PyQt4', 'PySide2', 'PySide6', 'PyQt6', 'pyart', 'IPython', 'jupyter', 'notebook'],
             noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='NLradar',
          debug=False,
          strip=False,
          upx=False,
          console=True,
          icon=os.path.join(here, '..', '..', '..', 'NLradar.ico'))
coll = COLLECT(exe,
               a.binaries,
               a.datas,
               strip=False,
               upx=False,
               name='NLradar')
