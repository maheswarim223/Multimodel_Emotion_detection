@echo off
echo.
echo  ================================================================
echo   EmotiSense — Unified Emotion Detection Dashboard
echo  ================================================================
echo.

:: ─── Check Python ───
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Python not found. Please install Python 3.9+
    pause & exit /b 1
)

:: ─── Install backend deps ───
echo  [1/3] Installing backend dependencies...
pip install flask flask-cors pandas numpy scikit-learn nltk joblib librosa --quiet
echo  [OK] Backend dependencies ready.
echo.

:: ─── Start Flask backend in a new terminal window ───
echo  [2/3] Starting Flask backend on http://localhost:5050 ...
start "EmotiSense Backend" cmd /k "cd /d "%~dp0backend" && python app.py"
timeout /t 3 /nobreak >nul

:: ─── Start React frontend ───
echo  [3/3] Starting React frontend on http://localhost:3000 ...
start "EmotiSense Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo.
echo  ================================================================
echo   App is starting up!
echo   Frontend  ->  http://localhost:3000
echo   Backend   ->  http://localhost:5050
echo  ================================================================
echo.
echo  Both windows will open automatically.
echo  Close them to shut down the servers.
echo.
pause
