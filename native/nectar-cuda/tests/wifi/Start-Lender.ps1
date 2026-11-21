<#
.SYNOPSIS
Gets this PC ready to lend its GPU, then opens Nectar, where "Lend this GPU" starts lending.

.DESCRIPTION
Lending answers on two ports: TCP 43212 for CUDA calls and UDP 43211 for discovery. Windows
Firewall must let the laptop in on both. This script checks for the rules and, when one is
missing, prints the command that adds it, for an administrator PowerShell; it never changes the
firewall itself. Each rule admits only the borrower (or, without -Borrower, the local network),
only on Private networks, and only for nectar-cuda-server.

Every connection must prove the shared secret entered in Nectar's Lend screen, and the GPU is
lent to one borrower at a time. What follows the hello is not encrypted.

.EXAMPLE
.\Start-Lender.ps1 -Borrower 192.168.1.183
#>
param(
    # The laptop's address, to narrow the firewall rules this script suggests
    [string] $Borrower
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path "$PSScriptRoot\..\..\..\..").Path
$folder = Join-Path $repo "build\nectar-cuda"
$server = Join-Path $folder "nectar-cuda-server.exe"
$app = Join-Path $folder "nectar.exe"

foreach ($file in $server, $app) {
    if (-not (Test-Path $file)) { throw "No $file yet: run Build-All.ps1 first" }
}

foreach ($gpu in Get-CimInstance Win32_VideoController | Where-Object Name -match "NVIDIA") {
    Write-Host "Lending      $($gpu.Name), driver $($gpu.DriverVersion)"
}
$public = $false
foreach ($address in Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notmatch '^(127\.|169\.254\.)' }) {
    $network = Get-NetConnectionProfile -InterfaceIndex $address.InterfaceIndex -ErrorAction SilentlyContinue
    $category = "no network profile"
    if ($network) { $category = "$($network.NetworkCategory) network" }
    if ($network -and $network.NetworkCategory -eq "Public") { $public = $true }
    Write-Host ("  {0,-16} {1} ({2})" -f $address.IPAddress, $address.InterfaceAlias, $category)
}
if ($public) {
    Write-Warning "A network above is Public. The rules only apply on Private networks; if the laptop connects through a Public one, set it to Private in Settings > Network & internet."
}

# Block rules win over allow rules. Windows adds them when its "allow access" prompt is cancelled.
$blocked = @(Get-NetFirewallApplicationFilter -Program $server -ErrorAction SilentlyContinue |
    Get-NetFirewallRule -ErrorAction SilentlyContinue | Where-Object { $_.Action -eq "Block" -and $_.Enabled -eq "True" })
foreach ($rule in $blocked) {
    Write-Warning "Firewall rule '$($rule.DisplayName)' blocks nectar-cuda-server. Remove it in an administrator PowerShell: Remove-NetFirewallRule -Name '$($rule.Name)'"
}

$remote = if ($Borrower) { $Borrower } else { "LocalSubnet" }
$missing = @()
foreach ($rule in @(
        @{ Name = "Nectar CUDA server"; Protocol = "TCP"; Port = 43212 },
        @{ Name = "Nectar discovery"; Protocol = "UDP"; Port = 43211 })) {
    $existing = Get-NetFirewallRule -DisplayName $rule.Name -ErrorAction SilentlyContinue
    if ($existing) {
        $addresses = ($existing | Get-NetFirewallAddressFilter).RemoteAddress -join ", "
        Write-Host ("Firewall     '{0}' lets in {1} on {2} {3}" -f $rule.Name, $addresses, $rule.Protocol, $rule.Port)
        # Rules made for the server's old name (nectar-cuda-server.exe) admit a file that no longer runs
        $program = ($existing | Get-NetFirewallApplicationFilter).Program | Select-Object -First 1
        if ($program -and $program -ne $server) {
            $missing += "Get-NetFirewallRule -DisplayName '$($rule.Name)' | Set-NetFirewallApplicationFilter -Program '$server'"
        }
    } else {
        $missing += "New-NetFirewallRule -DisplayName '$($rule.Name)' -Direction Inbound -Action Allow -Protocol $($rule.Protocol) -LocalPort $($rule.Port) -Program '$server' -Profile Private -RemoteAddress $remote"
    }
}
if ($missing) {
    Write-Host ""
    Write-Host "Windows Firewall would keep the laptop out. In an administrator PowerShell, run:" -ForegroundColor Yellow
    Write-Host ""
    $missing | ForEach-Object { Write-Host "  $_" }
    Write-Host ""
    if (-not $Borrower) {
        Write-Host "LocalSubnet lets in every device on this network (each must still know the secret)." -ForegroundColor Yellow
        Write-Host "With -Borrower <laptop address>, the commands let in only the laptop." -ForegroundColor Yellow
    }
    Write-Host "Then run this script again."
    return
}

Write-Host ""
Write-Host "Opening Nectar: choose 'Lend this GPU', enter the shared secret and press Start lending." -ForegroundColor Green
Start-Process $app

