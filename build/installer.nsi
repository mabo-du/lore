; Lore NSIS Installer Script
; Usage: makensis /DPRODUCT_VERSION="0.1.2" /DINPUT_DIR="dist\Lore" build\installer.nsi

!ifndef PRODUCT_VERSION
  !define PRODUCT_VERSION "0.0.0"
!endif

!ifndef INPUT_DIR
  !define INPUT_DIR "dist\Lore"
!endif

Name "Lore ${PRODUCT_VERSION}"
OutFile "dist\Lore-Setup-${PRODUCT_VERSION}.exe"
InstallDir "$PROGRAMFILES64\Lore"
RequestExecutionLevel admin

Section "Install"
  SetOutPath "$INSTDIR"
  File /r "${INPUT_DIR}\*.*"

  CreateDirectory "$SMPROGRAMS\Lore"
  CreateShortCut "$SMPROGRAMS\Lore\Lore.lnk" "$INSTDIR\Lore.exe"
  CreateShortCut "$DESKTOP\Lore.lnk" "$INSTDIR\Lore.exe"

  WriteUninstaller "$INSTDIR\uninstall.exe"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Lore" \
    "DisplayName" "Lore"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Lore" \
    "UninstallString" "$INSTDIR\uninstall.exe"
  WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Lore" \
    "DisplayVersion" "${PRODUCT_VERSION}"
SectionEnd

Section "Uninstall"
  Delete "$SMPROGRAMS\Lore\Lore.lnk"
  RMDir "$SMPROGRAMS\Lore"
  Delete "$DESKTOP\Lore.lnk"
  RMDir /r "$INSTDIR"
  DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\Lore"
SectionEnd
