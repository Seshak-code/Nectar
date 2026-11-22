@echo off
rem Runs the Wi-Fi test on the borrower. Bypassing the script policy applies to this run only.
rem With Nectar installed and a GPU borrowed in the Nectar app, no address is needed.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Test-Borrow.ps1" %*
