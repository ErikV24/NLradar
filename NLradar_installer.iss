; Inno Setup-script voor de deelbare NLradar-installer. Wordt aangeroepen door build_installer.py.
; Installeert per gebruiker, zonder adminrechten, en zonder snelkoppeling in de Opstartmap.
; Instellingen/caches/radardata komen in %LOCALAPPDATA%\NLradar en blijven bij deïnstalleren staan.

#define MyAppName "NLradar 2026"
#define MyAppVersion "2026.09"
#define MyAppExeName "NLradar.exe"
#define SrcDir SourcePath + "build_installer\dist\NLradar"

[Setup]
AppId={{7C3E5B1A-4F2D-4B8E-9A61-2D0F5E8C4A17}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=NLradar (Bram van 't Veen / Erik)
DefaultDirName={autopf}\{#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir={#SourcePath}build_installer
OutputBaseFilename=NLradar_setup
SetupIconFile={#SourcePath}NLradar.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile={#SourcePath}LICENSE

[Languages]
Name: "dutch"; MessagesFile: "compiler:Languages\Dutch.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#SrcDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\NLradar.ico"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\NLradar.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent
