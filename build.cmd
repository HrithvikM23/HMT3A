@echo off
setlocal
cd /d "%~dp0"
py -3.11 scripts\build_exe.py %*
if not errorlevel 1 exit /b 0
python scripts\build_exe.py %*
exit /b %ERRORLEVEL%
