@echo off
rem Bouwt NLradar_setup.exe. Zoekt de venv NLradar_env38_new naast deze map, anders op C:\ of D:\.
set "PY="
if exist "%~dp0..\NLradar_env38_new\Scripts\python.exe" set "PY=%~dp0..\NLradar_env38_new\Scripts\python.exe"
if not defined PY if exist "C:\NLradar_env38_new\Scripts\python.exe" set "PY=C:\NLradar_env38_new\Scripts\python.exe"
if not defined PY if exist "D:\NLradar_env38_new\Scripts\python.exe" set "PY=D:\NLradar_env38_new\Scripts\python.exe"
if not defined PY (
  echo Venv NLradar_env38_new niet gevonden.
  pause
  exit /b 1
)
echo Gebruikte Python: %PY%
"%PY%" "%~dp0build_installer.py"
echo.
pause
