@echo off
rem Intraday quotes (intraday.py): waits for 09:00, polls TWSE MIS every 10 s
rem until 13:36, writes db\local.db (quotes, quote_ticks) and settles paper
rem orders that hit their open / stop / target. Exits early on holidays.
rem Called by Windows Task Scheduler on weekdays at 08:55.
rem Log: data\daily\intraday_YYYYMMDD.log
rem Do not add any non-ASCII character to this file (see run_daily_update.bat).
setlocal
cd /d "%~dp0"
if not exist "data\daily" mkdir "data\daily"
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set TODAY=%%i
set LOG=data\daily\intraday_%TODAY%.log
echo ==== start %DATE% %TIME% ==== >> "%LOG%"
python -u -W ignore intraday.py >> "%LOG%" 2>&1
set RC=%ERRORLEVEL%
echo ==== exit=%RC% %DATE% %TIME% ==== >> "%LOG%"
endlocal & exit /b %RC%
