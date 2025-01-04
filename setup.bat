@echo off
:menu
cls
echo ================================
echo   Python Environment Manager
echo ================================
echo 1. Setup/Activate Environment
echo 2. Run Program
echo 3. Exit
echo.

set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" goto setup
if "%choice%"=="2" goto run
if "%choice%"=="3" exit

:setup
echo Creating Python virtual environment...

REM Check if venv already exists
IF EXIST myenv (
    echo Virtual environment already exists.
) ELSE (
    python -m venv myenv
    echo Virtual environment created.
)

REM Activate venv
call myenv\Scripts\activate.bat

REM Check if requirements.txt exists and install if it does
IF EXIST requirements.txt (
    echo Installing requirements...
    pip install -r requirements.txt
) ELSE (
    echo No requirements.txt found. Skipping package installation.
)

echo Setup complete! Virtual environment is now active.
goto menu

:run
REM Ensure venv is activated
call myenv\Scripts\activate.bat

REM Replace 'main.py' with your actual Python file name
python main.py
pause
goto menu 