@echo off
rem Runs the Wi-Fi test on the borrower. Bypassing the script policy applies to this run only.
if "%~1"=="" (
    echo Usage: Run-Test.cmd ^<lender address^> [-Quick] [-Log] [-Steps network,probe,bench,chat]
    pause
    exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Test-Borrow.ps1" %*
