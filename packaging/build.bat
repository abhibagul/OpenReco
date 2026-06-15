@echo off
REM Build the standalone openreco.exe (Windows). Double-click this, or run it from a terminal.
REM Close any running openreco.exe first, or the build can't overwrite dist\openreco.exe.
cd /d "%~dp0\.."
echo Building openreco.exe ... (close any running openreco.exe first)
python -m PyInstaller packaging\openreco.spec --noconfirm
echo.
if exist "dist\openreco.exe" (
  echo Done -^> dist\openreco.exe
) else (
  echo Build did not produce dist\openreco.exe -- check the output above.
)
pause
