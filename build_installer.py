# -*- coding: utf-8 -*-
"""
Bouwt de deelbare Windows-installer van NLradar.
Wordt aangeroepen door build_installer.bat (met de Python van NLradar_env38_new).

Stappen:
 1. PyInstaller installeren in de venv (als die er nog niet in zit)
 2. Schone kopie van Python_files maken in build_installer\\stage (zonder backups/testscripts/util,
    en met de Stadia-key uit nlr_maptiles_maptiler.py leeggemaakt -- jouw eigen bestanden blijven onaangeroerd)
 3. PyInstaller draaien -> build_installer\\dist\\NLradar\\NLradar.exe
 4. Databestanden erbij kopiëren (Input_files, Tables, _data, gifsicle, Unet-model, icoon)
 5. Inno Setup draaien (als geïnstalleerd) -> build_installer\\NLradar_setup.exe
"""
import os, sys, re, shutil, subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))          # ...\NLradar
PF = os.path.join(ROOT, 'Python_files')
WORK = os.path.join(ROOT, 'build_installer')
STAGE = os.path.join(WORK, 'stage')
DIST = os.path.join(WORK, 'dist', 'NLradar')
PYINSTALLER_VERSION = '6.22.3'   # laatste versie met Python 3.8-ondersteuning

def stap(tekst):
    print('\n=== ' + tekst + ' ===', flush=True)

def fout(tekst):
    print('\nFOUT: ' + tekst)
    sys.exit(1)

if sys.version_info[:2] != (3, 8):
    fout('Dit moet met de Python 3.8 van NLradar_env38_new gedraaid worden, niet met Python %d.%d.' % sys.version_info[:2])

# 1 ---------------------------------------------------------------------------------------------
stap('1/5 PyInstaller controleren')
try:
    import PyInstaller
    print('PyInstaller', PyInstaller.__version__, 'is aanwezig')
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyinstaller==' + PYINSTALLER_VERSION])

# 2 ---------------------------------------------------------------------------------------------
stap('2/5 Schone kopie van de code maken')
if os.path.exists(WORK):
    shutil.rmtree(WORK)
skip_dirs = {'vispy_OLD_BACKUP', '_short', 'util', '__pycache__', 'volume_grid_test_output',
             'Tables', '_data', 'gifsicle', 'models'}   # data-mappen worden in stap 4 los gekopieerd
def ignore(d, names):
    weg = []
    for n in names:
        p = os.path.join(d, n)
        if os.path.isdir(p):
            if n in skip_dirs: weg.append(n)
        elif not n.endswith('.py'):
            weg.append(n)            # .bak/.001/.160926/.pdf/.txt/.sqlite3 etc.
        elif d == PF and (n.startswith(('check_', 'fix', 'clear_')) or
                          n in ('jb.py', 'cache.py', 'show_archive.py', 'test.py', 'vtest.py')):
            weg.append(n)            # losse hulp-/diagnosescripts (bevatten deels eigen keys)
    return weg
shutil.copytree(PF, os.path.join(STAGE, 'Python_files'), ignore=ignore)

mt = os.path.join(STAGE, 'Python_files', 'nlr_maptiles_maptiler.py')
s = open(mt, encoding='utf-8').read()
s, n = re.subn(r"STADIA_API_KEY = '[^']*'", "STADIA_API_KEY = ''", s)
if n != 1:
    fout('Stadia-key niet gevonden in nlr_maptiles_maptiler.py (verwacht 1 keer, gevonden %d).' % n)
open(mt, 'w', encoding='utf-8').write(s)
print('Stadia-key leeggemaakt in de kopie')

# Controle: niets dat op een API-key lijkt in de code-kopie (UUID, lange hex-reeks, KNMI-JWT) en in Input_files
# Publieke reeksen uit Brams originele GitHub-code (geen persoonlijke key): kachelmannwetter-URL in nlr.py
PUBLIEK = {'18584bea4ec779beb796d3770ed37f8e'}
key_patroon = re.compile(r"eyJ[A-Za-z0-9_-]{20,}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}|[0-9a-fA-F]{32,}")
for basis in (STAGE, os.path.join(ROOT, 'Input_files')):
    for d, _, fs in os.walk(basis):
        for f in fs:
            if f.lower().endswith(('.png', '.jpg', '.shp', '.shx', '.dbf', '.prj')):
                continue
            tekst = open(os.path.join(d, f), 'rb').read().decode('latin-1')
            m = next((m for m in key_patroon.finditer(tekst) if m.group(0) not in PUBLIEK), None)
            if m:
                fout('Mogelijke API-key gevonden in %s: %s...' % (os.path.join(d, f), m.group(0)[:8]))
print('Geen API-keys gevonden in code-kopie en Input_files')

# 3 ---------------------------------------------------------------------------------------------
stap('3/5 PyInstaller draaien (duurt enkele minuten, tensorflow is groot)')
subprocess.check_call([sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean',
                       '--distpath', os.path.join(WORK, 'dist'),
                       '--workpath', os.path.join(WORK, 'pyi_work'),
                       os.path.join(ROOT, 'nlr_installer.spec')],
                      cwd=os.path.join(STAGE, 'Python_files'))
if not os.path.exists(os.path.join(DIST, 'NLradar.exe')):
    fout('NLradar.exe is niet gemaakt.')

# 4 ---------------------------------------------------------------------------------------------
stap('4/5 Databestanden kopiëren')
def kopieer(bron, doel):
    if not os.path.exists(bron):
        fout('Ontbreekt: ' + bron)
    if os.path.isdir(bron):
        shutil.copytree(bron, doel, ignore=shutil.ignore_patterns('__pycache__'))
    else:
        os.makedirs(os.path.dirname(doel), exist_ok=True)
        shutil.copy2(bron, doel)
    print('  ' + os.path.relpath(doel, DIST))

kopieer(os.path.join(ROOT, 'Input_files'), os.path.join(DIST, 'Input_files'))
kopieer(os.path.join(ROOT, 'NLradar.ico'), os.path.join(DIST, 'NLradar.ico'))
kopieer(os.path.join(ROOT, 'LICENSE'), os.path.join(DIST, 'LICENSE'))
kopieer(os.path.join(ROOT, 'ATTRIBUTION.txt'), os.path.join(DIST, 'ATTRIBUTION.txt'))
for sub in ('Tables', '_data', 'gifsicle'):
    kopieer(os.path.join(PF, sub), os.path.join(DIST, 'Python_files', sub))
kopieer(os.path.join(PF, 'dealiasing', 'unet_vda', 'models'),
        os.path.join(DIST, 'Python_files', 'dealiasing', 'unet_vda', 'models'))

# 5 ---------------------------------------------------------------------------------------------
stap('5/5 Installer maken met Inno Setup')
iscc = None
kandidaten = []
for v in ('7', '6'):
    kandidaten += [r'C:\Program Files\Inno Setup %s\ISCC.exe' % v, r'C:\Program Files (x86)\Inno Setup %s\ISCC.exe' % v,
                   os.path.join(os.environ.get('LOCALAPPDATA', ''), r'Programs\Inno Setup %s\ISCC.exe' % v)]
for p in kandidaten:
    if os.path.exists(p):
        iscc = p; break
if iscc is None:
    print('Inno Setup niet gevonden. Installeer het (gratis): https://jrsoftware.org/isdl.php')
    print('en draai build_installer.bat daarna opnieuw.')
    print('De programmamap zelf is wel al klaar en te testen: ' + os.path.join(DIST, 'NLradar.exe'))
    sys.exit(1)
subprocess.check_call([iscc, os.path.join(ROOT, 'NLradar_installer.iss')])
print('\nKLAAR: ' + os.path.join(WORK, 'NLradar_setup.exe'))
