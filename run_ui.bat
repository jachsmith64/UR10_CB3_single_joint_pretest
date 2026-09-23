@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

rem Keep this file pure ASCII - see the note in install_deps.bat.

set "MSG=%~dp0docs\messages"
set "VENV_PY=%~dp0.venv\Scripts\python.exe"

if not exist "%VENV_PY%" (
    echo.
    type "%MSG%\need_setup.txt"
    echo.
    pause
    exit /b 1
)

rem UTF-8 for Python's own output too, so Chinese never turns into mojibake.
set PYTHONIOENCODING=utf-8
set PYTHONPATH=%~dp0src

rem Load the operator's own settings if there are any, otherwise the defaults.
rem configs\local_*.json is git-ignored on purpose: it holds site values
rem (robot IP, camera serial) that must not travel with the tool.
set "CONFIG=%~dp0configs\experiment_default.json"
if exist "%~dp0configs\local_site.json" set "CONFIG=%~dp0configs\local_site.json"

echo.
type "%MSG%\ui_start.txt"
echo.
echo (config: %CONFIG%)
echo.

"%VENV_PY%" -m sj_pretest.ui --config "%CONFIG%"
set "EXITCODE=%errorlevel%"

if not "%EXITCODE%"=="0" (
    echo.
    echo [ERROR] The UI exited with code %EXITCODE%.
    echo If there is a traceback above, please send it to me.
    echo To check the flow without any hardware first, run run_dry_run_example.bat
    echo.
    pause
)
exit /b %EXITCODE%
