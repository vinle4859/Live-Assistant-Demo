@echo off
title LEMON ASSISTANT EDGE LAUNCHER
color 0b
cls

:: Step 0: Ensure we run in the batch file's own directory
pushd %~dp0

echo ========================================================
echo       LEMON ROBOT ASSISTANT DEPLOYMENT STARTUP
echo ========================================================
echo.

:: Step 1: Validate .env configuration file
if exist .env goto :env_ok
echo [-] ERROR: .env configuration file not found in current directory!
echo     Please copy .env.example to .env and configure your Google Cloud project credentials.
pause
exit /b 1
:env_ok

:: Step 2: Check Python installation
where python >nul 2>&1
if %errorlevel% equ 0 goto :python_ok
echo [-] ERROR: Python is not installed or not in System PATH!
echo     Please download and install Python 3.10+ from python.org.
pause
exit /b 1
:python_ok

:: Step 3: Setup Virtual Environment
if exist .venv goto :venv_ok
echo [+] Creating virtual environment .venv...
python -m venv .venv
if %errorlevel% equ 0 goto :venv_ok
echo [-] Failed to create virtual environment!
pause
exit /b 1
:venv_ok

echo [+] Activating virtual environment...
call .venv\Scripts\activate.bat

:: Step 3b: Smart dependency stamp check
set NEED_INSTALL=0
if not exist .venv\install.stamp (
    set NEED_INSTALL=1
    goto :dependency_check_done
)

:: Run comparison safely using a single line (no parentheses with pipe)
xcopy /d /y /l requirements.txt .venv\install.stamp 2>nul | findstr /b "1" >nul
if %errorlevel% equ 0 set NEED_INSTALL=1

:dependency_check_done
if %NEED_INSTALL% equ 0 goto :skip_install
echo [+] Installing/updating dependencies (this may take a moment)...
python -m pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 goto :install_fail
copy /y requirements.txt .venv\install.stamp >nul
echo [+] Dependency stamp updated.
goto :skip_install

:install_fail
echo [-] WARNING: Failed to install some dependencies. Startup will continue.

:skip_install

:: Step 4: Run environment check script
python tools/check_env.py
if %errorlevel% equ 0 goto :env_check_ok
echo [-] CRITICAL CONFIG ERROR: Environment check failed.
echo     Please fix the issues in your .env file listed above.
pause
exit /b 1
:env_check_ok

:: Step 5: Verify Audio Configuration
echo [+] Running audio diagnostics...
python tools/verify_audio.py
set AUDIO_STATUS=%errorlevel%
if %AUDIO_STATUS% equ 0 goto :audio_ok
echo.
echo [!] WARNING: Audio check reported errors (Code %AUDIO_STATUS%).
echo     Ensure microphone/speakers are connected.
echo.
set /p CONFIRM="Press ENTER to ignore and launch anyway, or Ctrl+C to abort..."
:audio_ok

:: Step 6: Cloudflared Setup
if exist cloudflared.exe goto :cloudflared_ok
echo [+] cloudflared.exe not found. Downloading latest version from GitHub...
powershell -Command "Invoke-WebRequest -Uri 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe' -OutFile 'cloudflared.exe'"
if exist cloudflared.exe goto :cloudflared_ok
echo [-] WARNING: Failed to download cloudflared.exe automatically.
echo     You will need to manually download it and run the tunnel or connect via LAN.
:cloudflared_ok

:: Step 7: Launch Cloudflare Tunnel and Scrape Ephemeral URL
set TUNNEL_SUCCESS=0
if not exist cloudflared.exe goto :skip_tunnel

echo [+] Initializing Cloudflare Quick Tunnel in background...
set LOG_FILE=%TEMP%\cloudflared_lemon.log

:: Clean up any lingering cloudflared processes before starting
taskkill /f /im cloudflared.exe >nul 2>&1
:: Wait 1 second for Windows to release file locks on the log file
timeout /t 1 /nobreak >nul 2>&1

if exist "%LOG_FILE%" del /f /q "%LOG_FILE%"

:: Start cloudflared background process redirecting output to log file
start "Lemon Cloudflared Tunnel" /min cmd /c "cloudflared.exe tunnel --url http://localhost:8000 > "%LOG_FILE%" 2>&1"

:: Wait and poll log file for public URL
echo [+] Waiting for Cloudflare tunnel URL...
powershell -NoProfile -NonInteractive -Command "$logFile = '%LOG_FILE%'; $url = $null; for ($i = 0; $i -lt 30; $i++) { if (Test-Path $logFile) { $content = Get-Content $logFile -Raw; if ($content -match 'https://[a-zA-Z0-9\-]+\.trycloudflare\.com') { $url = $Matches[0]; break; } } Start-Sleep -Milliseconds 500; }; if ($url) { Write-Output ('URL:' + $url) } else { Write-Output 'TIMEOUT' }" > "%TEMP%\tunnel_result.txt"

set /p TUNNEL_OUT=<"%TEMP%\tunnel_result.txt"
del /f /q "%TEMP%\tunnel_result.txt"

if not "%TUNNEL_OUT:~0,4%"=="URL:" goto :tunnel_failed
set TUNNEL_SUCCESS=1
set PUBLIC_URL=%TUNNEL_OUT:~4%
echo.
echo ========================================================
echo  TUNNEL STATUS: ONLINE
echo  PUBLIC OPERATOR DASHBOARD URL:
echo  %PUBLIC_URL%
echo ========================================================
echo.
goto :skip_tunnel

:tunnel_failed
echo [-] WARNING: Cloudflare tunnel startup timed out or failed.
echo     Check the log at: %LOG_FILE%
echo     You can still access the dashboard locally on LAN.
echo.

:skip_tunnel

:: Step 8: Launch Python Live Assistant
echo [+] Starting Lemon Voice Assistant in live Q^&A mode...
python main.py --mode live

:: Step 9: Cleanup on exit
echo [+] Shutting down Cloudflare tunnel and cleaning up...
taskkill /f /im cloudflared.exe >nul 2>&1
if exist "%TEMP%\cloudflared_lemon.log" del /f /q "%TEMP%\cloudflared_lemon.log"
echo [+] Shutdown complete.
pause
popd
