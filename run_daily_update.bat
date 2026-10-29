@echo off
rem Daily update of ALL data sources (daily_all.py): prices, indices, ETFs,
rem oil, news, T86, fundamentals, revenue, PTT, GDELT, futures, corporate
rem actions, delisted, price-jump fixes, feature rebuild.
rem Called by Windows Task Scheduler. Can also be run by double-clicking.
rem Exit code 0 = every step succeeded.
rem Exit code 3 = at least one step failed; see data\daily\all_YYYYMMDD.txt.
rem Paths are derived from this file's own location, so no hardcoded
rem non-ASCII path survives cmd.exe's ANSI codepage. Do not add any
rem non-ASCII character to this file.
setlocal
cd /d "%~dp0"
if not exist "data\daily" mkdir "data\daily"
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set TODAY=%%i
set LOG=data\daily\run_%TODAY%.log
echo ==== start %DATE% %TIME% ==== >> "%LOG%"
python -u -W ignore daily_all.py >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%
echo ==== exit=%RC% %DATE% %TIME% ==== >> "%LOG%"
endlocal & exit /b %RC%