@echo off
rem Start the local stack at logon: Ollama, preload the chat model
rem (gemma3:12b, kept in memory with keep_alive -1) and the Streamlit
rem frontend on http://127.0.0.1:8501.
rem Called by Windows Task Scheduler at logon. Can also be run by double-clicking.
rem Log: data\daily\startup.log
rem Do not add any non-ASCII character to this file (see run_daily_update.bat).
setlocal
cd /d "%~dp0"
if not exist "data\daily" mkdir "data\daily"
set LOG=data\daily\startup.log
set OLLAMA=http://localhost:11434
echo ==== start %DATE% %TIME% ==== >> "%LOG%"

rem Ollama normally starts from the Startup folder; start it if it is not up yet.
curl -s -o nul %OLLAMA%/api/tags || start "" "%LOCALAPPDATA%\Programs\Ollama\ollama app.exe"
set N=0
:wait
curl -s -o nul %OLLAMA%/api/tags && goto up
set /a N+=1
if %N% geq 60 (
    echo ollama not reachable after 120s >> "%LOG%"
    goto front
)
ping -n 3 127.0.0.1 > nul
goto wait

:up
rem num_ctx must match NUM_CTX in local_ask.py, otherwise the first question reloads the model.
curl -s %OLLAMA%/api/generate -d "{\"model\":\"gemma3:12b\",\"keep_alive\":-1,\"options\":{\"num_ctx\":8192}}" >> "%LOG%" 2>&1
echo. >> "%LOG%"
echo model loaded %TIME% >> "%LOG%"

:front
rem Skip if the frontend is already running.
curl -s -o nul http://127.0.0.1:8501 && goto done
rem pythonw has no console, so the frontend keeps running after this script exits.
start "" pythonw -m streamlit run frontend/app.py >> "%LOG%" 2>&1

:done
echo ==== done %DATE% %TIME% ==== >> "%LOG%"
endlocal
