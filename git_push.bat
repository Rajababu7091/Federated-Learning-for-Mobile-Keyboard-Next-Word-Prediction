@echo off
setlocal

set GIT="C:\Program Files\Git\bin\git.exe"

echo ================================================================================
echo PUSHING TO GITHUB
echo ================================================================================
echo.

cd /d "%~dp0"

echo [1/7] Initializing repository...
%GIT% init
echo.

echo [2/7] Adding all files...
%GIT% add .
echo.

echo [3/7] Committing files...
%GIT% commit -m "Initial commit: Federated Learning Keyboard with Attention-GRU"
echo.

echo [4/7] Adding remote...
%GIT% remote add origin https://github.com/KOVVURIPCDURGAREDDY/Major-Project.git 2>nul
if errorlevel 1 (
    %GIT% remote set-url origin https://github.com/KOVVURIPCDURGAREDDY/Major-Project.git
)
echo.

echo [5/7] Setting main branch...
%GIT% branch -M main
echo.

echo [6/7] Configuring user (if needed)...
%GIT% config user.email "your-email@example.com"
%GIT% config user.name "KOVVURIPCDURGAREDDY"
echo.

echo [7/7] Pushing to GitHub...
echo You will be prompted for GitHub credentials.
echo.
%GIT% push -u origin main

if errorlevel 1 (
    echo.
    echo [ERROR] Push failed!
    echo.
    echo Try running: git push -u origin main
    echo.
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo SUCCESS!
echo ================================================================================
echo.
echo View at: https://github.com/KOVVURIPCDURGAREDDY/Major-Project
echo.
pause
