@echo off
echo Starting Document Q&A Engine...
echo.

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo Python is not installed or not in PATH!
    echo Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

echo.
echo Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install dependencies!
    pause
    exit /b 1
)

echo.
echo Creating environment file...
if not exist .env (
    copy .env.example .env
    echo Please edit .env file with your configuration and run this script again.
    notepad .env
    pause
    exit /b 0
)

echo.
echo Starting API server...
start "Document Q&A API" python run_api.py

echo.
echo Waiting for API server to start...
timeout /t 5 /nobreak > nul

echo.
echo Starting web interface...
start "Document Q&A Web" python run_streamlit.py

echo.
echo Document Q&A Engine is starting up...
echo API Documentation: http://localhost:8000/docs
echo Web Interface: http://localhost:8501
echo.
echo Press any key to exit (this will close the servers)...
pause > nul

echo.
echo Shutting down servers...
taskkill /f /im python.exe
echo Servers stopped.
pause