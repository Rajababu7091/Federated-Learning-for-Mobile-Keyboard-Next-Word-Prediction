@echo off
setlocal

set GIT="C:\Program Files\Git\bin\git.exe"

echo ================================================================================
echo PUSHING TO NEW GITHUB REPOSITORY
echo ================================================================================
echo.

cd /d "%~dp0"

echo [1/5] Removing old remote...
%GIT% remote remove origin 2>nul
echo.

echo [2/5] Adding new remote...
%GIT% remote add origin https://github.com/KOVVURIPCDURGAREDDY/Major-Project.git
echo.

echo [3/5] Setting main branch...
%GIT% branch -M main
echo.

echo [4/5] Configuring user...
%GIT% config user.email "durgareddy@example.com"
%GIT% config user.name "KOVVURIPCDURGAREDDY"
echo.

echo [5/5] Pushing to GitHub...
%GIT% push -u origin main

if errorlevel 1 (
    echo.
    echo [ERROR] Push failed!
    echo Check your GitHub credentials.
    echo.
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo SUCCESS! Project pushed to GitHub
echo ================================================================================
echo.
echo View at: https://github.com/KOVVURIPCDURGAREDDY/Major-Project
echo.
pause
