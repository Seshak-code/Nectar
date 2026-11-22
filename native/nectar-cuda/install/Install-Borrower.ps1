<#
.SYNOPSIS
Installs Nectar on a PC that borrows GPUs, or removes it (-Uninstall). Run as administrator.

.DESCRIPTION
- C:\Program Files\Nectar: nectar.exe (the app) and nectar-svc.exe (the service).
- System32\nvcuda.dll: Nectar's own, which every CUDA program loads. While no GPU is borrowed
  it reports no device at once, like a PC without one. Nothing of NVIDIA's is installed:
  their hybrid runtime arrives from the lender on the first borrow, into
  %ProgramData%\Nectar\drivers\<version>\, matching the lender's driver. Left out on a PC
  whose System32\nvcuda.dll is NVIDIA's: its CUDA programs keep its own GPU, and it borrows
  through Vulkan only.
- Nectar's Vulkan driver (nectar_vk.dll, with nectar_icd.json) in C:\Program Files\Nectar,
  registered with the Vulkan loader (HKLM\SOFTWARE\Khronos\Vulkan\Drivers). Like nvcuda.dll it
  lists no device while nothing is borrowed; the PC's own GPUs stay listed beside it.
- The NectarBorrow service (LocalSystem, starts with Windows), which finds lenders, holds the
  lease and writes %ProgramData%\Nectar\borrow.cfg while a GPU is borrowed.
- %ProgramData%\Nectar: readable by users, writable only by SYSTEM and Administrators, so no
  user can point other users' programs at another server or hybrid runtime. The shared
  secret in it is readable by SYSTEM and Administrators only.
- A Start-menu shortcut to Nectar.

.EXAMPLE
.\Install-Borrower.ps1 -Secret "the secret entered in the lender's Nectar window"
.EXAMPLE
.\Install-Borrower.ps1 -Uninstall
#>
param(
    [string] $Secret,
    [switch] $Uninstall,
    # The folder with nectar.exe, nectar-svc.exe and nvcuda.dll (default: this script's)
    [string] $Source
)

$ErrorActionPreference = "Stop"
if (-not $Source) { $Source = $PSScriptRoot }

$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Run this in an administrator PowerShell"
}

$service = "NectarBorrow"
$programs = Join-Path $env:ProgramFiles "Nectar"
$state = Join-Path $env:ProgramData "Nectar"
$system = Join-Path $env:SystemRoot "System32"
$driver = Join-Path $system "nvcuda.dll"
$shortcut = Join-Path $env:ProgramData "Microsoft\Windows\Start Menu\Programs\Nectar.lnk"
$vulkanDrivers = "HKLM:\SOFTWARE\Khronos\Vulkan\Drivers"
$manifest = Join-Path $programs "nectar_icd.json"

# Whether System32\nvcuda.dll is NVIDIA's (it carries their version resource; Nectar's has none)
function Test-NvidiaDriver {
    if (-not (Test-Path $driver)) { return $false }
    $company = (Get-Item $driver).VersionInfo.CompanyName
    $company -and $company -match "NVIDIA"
}

function Remove-Service {
    if (Get-Service $service -ErrorAction SilentlyContinue) {
        # Stopping returns a borrowed GPU and removes borrow.cfg
        Stop-Service $service -ErrorAction SilentlyContinue
        & sc.exe delete $service | Out-Null
    }
}

if ($Uninstall) {
    Remove-Service
    Remove-ItemProperty $vulkanDrivers -Name $manifest -ErrorAction SilentlyContinue
    Get-Process nectar -ErrorAction SilentlyContinue | Stop-Process -Force
    if ((Test-Path $driver) -and -not (Test-NvidiaDriver)) { Remove-Item $driver -Force }
    foreach ($path in $programs, $state, $shortcut) {
        if (Test-Path $path) { Remove-Item $path -Recurse -Force }
    }
    Write-Host "Nectar removed." -ForegroundColor Green
    return
}

if (-not $Secret -or $Secret.Length -lt 8) {
    throw "Give the shared secret entered in the lender's Nectar window: -Secret <at least 8 characters>"
}
$vulkanOnly = Test-NvidiaDriver
foreach ($file in "nectar.exe", "nectar-svc.exe", "nvcuda.dll", "nectar_vk.dll", "nectar_icd.json") {
    if (-not (Test-Path (Join-Path $Source $file))) { throw "No $file in $Source" }
}

Remove-Service
Get-Process nectar -ErrorAction SilentlyContinue | Stop-Process -Force

New-Item -ItemType Directory -Force $programs | Out-Null
foreach ($file in "nectar.exe", "nectar-svc.exe", "nvcuda.dll", "nectar_vk.dll", "nectar_icd.json") {
    Copy-Item (Join-Path $Source $file) $programs -Force
}
if ($vulkanOnly) {
    Write-Host "Installed  $programs (this PC has NVIDIA's CUDA driver, so it borrows through Vulkan only)"
} else {
    Copy-Item (Join-Path $Source "nvcuda.dll") $driver -Force
    Write-Host "Installed  $programs and $driver"
}

# The Vulkan loader reads drivers from here; 0 means enabled. New-Item -Force would empty an
# existing key, unregistering other drivers, so it only creates a missing one.
if (-not (Test-Path $vulkanDrivers)) { New-Item $vulkanDrivers -Force | Out-Null }
New-ItemProperty $vulkanDrivers -Name $manifest -PropertyType DWord -Value 0 -Force | Out-Null
Write-Host "Vulkan     $manifest registered"

# Users read; only SYSTEM and Administrators write. The secret: SYSTEM and Administrators only.
New-Item -ItemType Directory -Force $state | Out-Null
& icacls.exe $state /inheritance:r /grant:r "*S-1-5-18:(OI)(CI)F" "*S-1-5-32-544:(OI)(CI)F" "*S-1-5-32-545:(OI)(CI)RX" | Out-Null
$secretFile = Join-Path $state "secret"
[IO.File]::WriteAllText($secretFile, $Secret)
& icacls.exe $secretFile /inheritance:r /grant:r "*S-1-5-18:F" "*S-1-5-32-544:F" | Out-Null
Write-Host "Secret     stored in $secretFile (SYSTEM and Administrators only)"

New-Service -Name $service -DisplayName "Nectar borrowing" `
    -Description "Finds PCs that lend a GPU and keeps the borrowed one connected" `
    -BinaryPathName "`"$programs\nectar-svc.exe`" run" -StartupType Automatic | Out-Null
Start-Service $service
Write-Host "Service    $service running"

$shell = New-Object -ComObject WScript.Shell
$link = $shell.CreateShortcut($shortcut)
$link.TargetPath = Join-Path $programs "nectar.exe"
$link.Description = "Lend or borrow a GPU"
$link.Save()

Write-Host ""
Write-Host "Done. Open Nectar from the Start menu, pick the lender and press Borrow." -ForegroundColor Green
Write-Host "$(if ($vulkanOnly) { 'Vulkan' } else { 'CUDA and Vulkan' }) programs then use the borrowed GPU on their own."
