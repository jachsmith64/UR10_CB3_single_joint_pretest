@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

rem ============================================================
rem  NOTE FOR FUTURE EDITORS: keep this file pure ASCII.
rem  cmd.exe mis-parses its own batch file as soon as a multi-byte
rem  UTF-8 sequence appears -- it loses track of line boundaries and
rem  starts executing fragments of lines. So every Chinese message
rem  lives in docs\messages\*.txt and is printed with "type", which
rem  copies the UTF-8 bytes straight to the console (the console is
rem  put into UTF-8 mode by the chcp 65001 above).
rem ============================================================

set "MSG=%~dp0docs\messages"
set "VENV_PY=%~dp0.venv\Scripts\python.exe"

type "%MSG%\setup_start.txt"
echo.

if exist "%VENV_PY%" (
    echo [1/3] .venv already exists - skipping creation.
    goto install
)

echo [1/3] Creating virtual environment .venv ...

rem Prefer 3.12: this tool is tested on 3.12.
rem Each attempt is its own "if" block on purpose. Squeezing
rem "if not defined X cmd && set X" onto one line makes cmd bind the
rem && to the whole if-statement, which can run the set when the test
rem failed. Separate blocks avoid that entirely.
set "BASE_PY="
if not defined BASE_PY (
    py -3.12 -c "import sys" >nul 2>&1 && set "BASE_PY=py -3.12"
)
if not defined BASE_PY (
    py -3.11 -c "import sys" >nul 2>&1 && set "BASE_PY=py -3.11"
)
if not defined BASE_PY (
    py -3 -c "import sys" >nul 2>&1 && set "BASE_PY=py -3"
)
if not defined BASE_PY (
    python -c "import sys" >nul 2>&1 && set "BASE_PY=python"
)

if not defined BASE_PY (
    echo.
    type "%MSG%\need_python.txt"
    echo.
    pause
    exit /b 1
)

echo        using interpreter: %BASE_PY%
%BASE_PY% -m venv "%~dp0.venv"
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to create .venv - see the message above.
    echo.
    pause
    exit /b 1
)

:install
echo.
echo [2/3] Installing dependencies (first run takes a few minutes) ...
"%VENV_PY%" -m pip install --upgrade pip --quiet
"%VENV_PY%" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo.
    echo [ERROR] pip install failed.
    echo If it is a network problem, try a domestic mirror, e.g.
    echo    .venv\Scripts\python.exe -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    echo.
    pause
    exit /b 1
)

echo.
echo [3/3] Self-check ...
rem tkinter is part of the standard library but is a separate Windows
rem component; check it here so a missing one shows up now and not when
rem the operator is standing next to the robot.
"%VENV_PY%" -c "import numpy, cv2, scipy, tkinter; print('   core OK: numpy', numpy.__version__, '/ opencv', cv2.__version__, '/ scipy', scipy.__version__, '/ tkinter', tkinter.TkVersion)"
if errorlevel 1 (
    echo.
    echo [ERROR] Self-check failed - numpy / opencv / scipy / tkinter is broken.
    echo         numpy, opencv, scipy come from pip.
    echo         tkinter comes with the Python installer: re-run it and enable
    echo         "tcl/tk and IDLE".
    echo.
    pause
    exit /b 1
)
"%VENV_PY%" -c "import matplotlib, PIL" >nul 2>&1 && echo    optional OK: matplotlib / Pillow || echo    optional deps partly missing - harmless, only used by the reused offline plotting code
"%VENV_PY%" -c "import rtde_control" >nul 2>&1 && echo    hardware OK: ur_rtde present - the hardware mode is available || echo    hardware note: ur_rtde NOT installed - dry_run / replay only (see START_HERE.md)

echo.
type "%MSG%\setup_done.txt"
echo.
pause
exit /b 0
