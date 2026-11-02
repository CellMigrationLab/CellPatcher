@ECHO ON
echo Running pre-uninstall
"%PREFIX%\python.exe" -m labconstrictor_tools unregister --name "CellPatcher" --prefix "%PREFIX%" >NUL 2>&1
IF ERRORLEVEL 1 IF EXIST "%USERPROFILE%\.labconstrictor\apps\CellPatcher.json" echo WARNING: could not remove CellPatcher from the LabConstrictor tools list; if Napari or Fiji still list it, delete the files %USERPROFILE%\.labconstrictor\apps\CellPatcher.json and CellPatcher.schema.json. 1>&2
"%PREFIX%\python.exe" -c "from menuinst.api import remove; import os; remove(os.path.join(r'%PREFIX%', 'CellPatcher', 'notebook_launcher.json'))"
SET "ARP_KEY=HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\CellPatcher"
reg delete "%ARP_KEY%" /f >NUL 2>&1
echo Pre-uninstall completed!
SetLocal EnableDelayedExpansion
