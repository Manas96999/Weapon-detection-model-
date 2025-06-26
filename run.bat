@echo off
REM Batch file to activate virtual environment and run the Flask app

REM --- Configuration ---
REM Set the path to your virtual environment's activation script
REM Common locations are 'venv\Scripts\activate.bat' or '.venv\Scripts\activate.bat'
SET VENV_PATH=YL\Scripts\activate.bat

REM Set the name of your Python script
SET PYTHON_SCRIPT=app.py
REM --- End Configuration ---

REM Check if the virtual environment path exists
IF NOT EXIST "%VENV_PATH%" (
    echo ERROR: Virtual environment not found at %VENV_PATH%
    echo Please update the VENV_PATH variable in this script.
    pause
    exit /b 1
)

REM Check if the Python script exists
IF NOT EXIST "%PYTHON_SCRIPT%" (
    echo ERROR: Python script %PYTHON_SCRIPT% not found.
    echo Please make sure the script is in the same directory as this batch file or update PYTHON_SCRIPT.
    pause
    exit /b 1
)

echo Activating virtual environment...
call "%VENV_PATH%"

echo Starting Flask application (%PYTHON_SCRIPT%)...
python "%PYTHON_SCRIPT%"

echo Application stopped.
pause
