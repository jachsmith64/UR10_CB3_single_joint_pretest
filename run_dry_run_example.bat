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

set PYTHONIOENCODING=utf-8
set PYTHONPATH=%~dp0src

type "%MSG%\dry_run_start.txt"
echo.

rem The dry-run subcommand forces mode=dry_run internally: it cannot reach a
rem real robot even if the config file says otherwise. That is deliberate -
rem the command line has no per-step human confirmation gate.
"%VENV_PY%" -m sj_pretest.cli dry-run ^
    --config "%~dp0configs\experiment_default.json" ^
    --output-root "%~dp0outputs" ^
    --stamp dryrun_example ^
    --joints J1 ^
    --amplitudes 0.01,0.05,0.2 ^
    --repeats 2 ^
    --stride 8 ^
    --formal-joints J1 ^
    --formal-step 0.05 ^
    --group A ^
    --json "%~dp0outputs\dryrun_example_summary.json"

if errorlevel 1 (
    echo.
    echo [ERROR] The dry run failed - see the messages above.
    echo.
    pause
    exit /b 1
)

echo.
type "%MSG%\dry_run_done.txt"
echo.
pause
exit /b 0
