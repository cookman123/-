@echo off
rem Daily update: TW stock prices, indices, news.
rem Called by Windows Task Scheduler. Can also be run by double-clicking.
rem Exit code 0 = all stocks aligned to the same last date (safe to pick).
rem Exit code 2 = NOT aligned; do not use that data for stock selection.
rem Paths are derived from this file's own location, so no hardcoded
rem non-ASCII path survives cmd.exe's ANSI codepage. Do not add any
rem non-ASCII character to this file.
setlocal
cd /d "%~dp0"
if not exist "data\daily" mkdir "data\daily"
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set TODAY=%%i
set LOG=data\daily\run_%TODAY%.log
echo ==== start %DATE% %TIME% ==== >> "%LOG%"
python -u -W ignore daily_update.py --quiet >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%
echo ==== exit=%RC% %DATE% %TIME% ==== >> "%LOG%"
endlocal & exit /b %RC%