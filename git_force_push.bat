@echo off
setlocal

set GIT="C:\Program Files\Git\bin\git.exe"

echo ================================================================================
echo FORCE PUSHING TO GITHUB (This will overwrite remote repository)
echo ================================================================================
echo.

cd /d "%~dp0"

echo [1/3] Pulling remote changes with rebase...
%GIT% pull origin main --rebase --allow-unrelated-histories
echo.

echo [2/3] Pushing to GitHub...
%GIT% push -u origin main

if errorlevel 1 (
    echo.
    echo Pull failed, trying force push...
    echo.
    echo [3/3] Force pushing (this will overwrite remote)...
    %GIT% push -u origin main --force
)

if errorlevel 1 (
    echo.
    echo [ERROR] Push still failed!
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
