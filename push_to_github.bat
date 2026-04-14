@echo off
echo ================================================================================
echo PUSHING TO GITHUB
echo ================================================================================
echo.
echo Repository: https://github.com/KOVVURIPCDURGAREDDY/Major-Project.git
echo.

REM Check if git is installed
git --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Git is not installed!
    echo.
    echo Please install Git from: https://git-scm.com/download/win
    echo.
    echo After installing Git, run this script again.
    pause
    exit /b 1
)

echo [OK] Git is installed
echo.

REM Initialize repository if not already initialized
if not exist .git (
    echo [1/6] Initializing Git repository...
    git init
    echo.
) else (
    echo [1/6] Git repository already initialized
    echo.
)

REM Add all files
echo [2/6] Adding all files...
git add .
echo.

REM Commit
echo [3/6] Committing files...
git commit -m "Initial commit: Federated Learning Keyboard with Attention-GRU, FedProx+FedNova, DP-SGD"
echo.

REM Add remote (ignore error if already exists)
echo [4/6] Adding remote repository...
git remote add origin https://github.com/KOVVURIPCDURGAREDDY/Major-Project.git 2>nul
if errorlevel 1 (
    echo Remote already exists, updating URL...
    git remote set-url origin https://github.com/KOVVURIPCDURGAREDDY/Major-Project.git
)
echo.

REM Set main branch
echo [5/6] Setting main branch...
git branch -M main
echo.

REM Push to GitHub
echo [6/6] Pushing to GitHub...
echo.
echo You may be prompted for GitHub credentials.
echo If using 2FA, use a Personal Access Token instead of password.
echo Generate token at: https://github.com/settings/tokens
echo.
git push -u origin main

if errorlevel 1 (
    echo.
    echo [ERROR] Push failed!
    echo.
    echo Common issues:
    echo 1. Authentication failed - Use Personal Access Token
    echo 2. Repository doesn't exist - Create it on GitHub first
    echo 3. No internet connection
    echo.
    echo See GITHUB_PUSH_INSTRUCTIONS.txt for detailed help
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo SUCCESS! Project pushed to GitHub
echo ================================================================================
echo.
echo View your repository at:
echo https://github.com/KOVVURIPCDURGAREDDY/Major-Project
echo.
pause
