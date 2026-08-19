@echo off
setlocal
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_worker.ps1" %*
exit /b %ERRORLEVEL%
