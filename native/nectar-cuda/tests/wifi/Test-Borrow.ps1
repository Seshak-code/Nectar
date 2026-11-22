<#
.SYNOPSIS
Borrows the lender's GPU over the network, checks it end to end, and keeps every log.

.DESCRIPTION
Runs on the borrower (the laptop), from the folder New-BorrowerBundle.ps1 made. Each step's
output goes to results\<computer>-<time>\, which is zipped at the end for reading on the lender.
While a program runs, its output shows indented (Nectar's own lines in cyan), and every 15 s
a line gives the time and what has crossed the network, and while the chat answers, how many
tokens it has generated.

  network  The Wi-Fi link, ping times, and whether the lender's port answers
  link     Nectar's link check, once for each profile (gaming, AI), with its advice. Needs Nectar
           installed and a GPU borrowed.
  probe    cuda-probe (vk-probe for Vulkan): the lender's GPU, time per call, transfer speed,
           and a kernel or compute shader
  frames   Vulkan only: vk-probe opens a window and shows frames from the lender's GPU in it
  direct3d Vulkan only: a Direct3D 11 program with shaders and 2,000 draws a frame, through DXVK
  bench    llama-bench with the bundled model, compared with the lender's own speed
  chat     llama-server answers one question; the answer must match the lender's word for word

The steps run for each API in -Api: CUDA (llama.cpp\, results in the results folder itself)
and Vulkan (llama.cpp-vulkan\, results in its vulkan\ subfolder).

With Nectar installed (Install-Borrower.ps1) and a GPU borrowed in the Nectar app, the programs
run as they are and find the borrowed GPU through System32\nvcuda.dll and Nectar's registered
Vulkan driver; the lender's address comes from the Nectar service. Vulkan programs also see
this PC's own GPU, so llama.cpp is told to use the one named "(Nectar)". Without Nectar
installed, give the lender's address: CUDA programs then run through nectar-cuda-run, Vulkan
programs get Nectar's driver from VK_DRIVER_FILES, and both work only with a lender that does
not lend (no --lend).

With -Reference it runs on the lender instead, on the real driver without Nectar, and records
the expected results in expected\. New-BorrowerBundle.ps1 does this.

Written for Windows PowerShell 5.1, which every Windows 10 and 11 machine has.

.EXAMPLE
Run-Test.cmd
.EXAMPLE
Run-Test.cmd -Quick -Steps probe,bench
#>
param(
    [Parameter(Position = 0)] [string] $Lender,
    [int] $Port = 43212,
    [string[]] $Steps = @("network", "link", "probe", "frames", "direct3d", "bench", "chat"),
    [string[]] $Api = @("cuda", "vulkan"),
    # A short benchmark instead of the full one
    [switch] $Quick,
    # Logs every CUDA call Nectar makes (large)
    [switch] $Log,
    [switch] $Reference
)

$ErrorActionPreference = "Stop"
# Run-Test.cmd passes "probe,bench" as one string
$Steps = @($Steps | ForEach-Object { $_ -split "," } | Where-Object { $_ })
$Api = @($Api | ForEach-Object { $_ -split "," } | Where-Object { $_ })

$bundle = $PSScriptRoot
$nectar = Join-Path $bundle "nectar"
$cudaBin = Join-Path $bundle "llama.cpp\bin"
$vulkanBin = Join-Path $bundle "llama.cpp-vulkan\bin"
$expectedRoot = Join-Path $bundle "expected"
# The API being tested, its llama.cpp, where its results go and what they're compared with;
# set for each API below
$current = "cuda"
$bin = $cudaBin
$expected = $expectedRoot
# llama.cpp's -dev for the Vulkan device to use
$vulkanDevice = $null
$model = Get-ChildItem (Join-Path $bundle "models") -Filter *.gguf -ErrorAction SilentlyContinue |
    Select-Object -First 1
$serverPort = 8093

$systemDriver = Join-Path $env:SystemRoot "System32\nvcuda.dll"
# Nectar's nvcuda.dll in System32 (Install-Borrower.ps1); NVIDIA's carries their version resource
$installed = (Test-Path $systemDriver) -and ((Get-Item $systemDriver).VersionInfo.CompanyName -notmatch "NVIDIA")

# Asks the Nectar service (svc/src/pipe.rs): key=value lines, repeated keys as lists
function Ask-Nectar([string] $command) {
    $pipe = New-Object IO.Pipes.NamedPipeClientStream(".", "nectar", [IO.Pipes.PipeDirection]::InOut)
    try { $pipe.Connect(3000) } catch { throw "The Nectar service is not running: install with Install-Borrower.ps1" }
    $writer = New-Object IO.StreamWriter($pipe); $writer.AutoFlush = $true
    $reader = New-Object IO.StreamReader($pipe)
    $writer.WriteLine($command)
    $status = @{ found = @(); check_line = @(); advice = @() }
    while ($true) {
        $line = $reader.ReadLine()
        if (-not $line) { break }
        $key, $value = $line -split "=", 2
        if ($status[$key] -is [array]) { $status[$key] += $value } else { $status[$key] = $value }
    }
    $pipe.Dispose()
    $status
}

function Get-BorrowStatus { Ask-Nectar "status" }

$vulkanManifest = Join-Path $env:ProgramFiles "Nectar\nectar_icd.json"
$vulkanRegistered = $null -ne (Get-ItemProperty "HKLM:\SOFTWARE\Khronos\Vulkan\Drivers" -Name $vulkanManifest -ErrorAction SilentlyContinue)

if ($Reference) {
    $out = $expected
    $Steps = @($Steps | Where-Object { $_ -ne "network" })
    # The real driver only
    Remove-Item Env:VK_DRIVER_FILES, Env:VK_ICD_FILENAMES, Env:NECTAR_VK_SERVER -ErrorAction SilentlyContinue
} elseif ($installed) {
    if ($Api -contains "vulkan" -and -not $vulkanRegistered) {
        throw "Nectar's Vulkan driver isn't registered: install again with this bundle's Install-Borrower.ps1, or run with -Api cuda"
    }
    $status = Get-BorrowStatus
    if ($status.state -ne "borrowed") {
        throw "No GPU is borrowed ($($status.message)): open Nectar, pick the lender and press Borrow"
    }
    # found=name|address|...: the address of the lender borrowed from
    $entry = $status.found | Where-Object { $_.StartsWith("$($status.lender)|") } | Select-Object -First 1
    if (-not $Lender -and $entry) {
        $address = ($entry -split "\|")[1]
        $Lender = $address.Substring(0, $address.LastIndexOf(":"))
        $Port = [int]$address.Substring($address.LastIndexOf(":") + 1)
    }
    $out = Join-Path $bundle ("results\{0}-{1:yyyyMMdd-HHmmss}" -f $env:COMPUTERNAME, (Get-Date))
    if ($Log) { $env:NECTAR_CUDA_LOG = "1" }
} else {
    if (-not $Lender) { throw "Install Nectar with Install-Borrower.ps1, or name a lender that does not lend: Run-Test.cmd <address>" }
    $out = Join-Path $bundle ("results\{0}-{1:yyyyMMdd-HHmmss}" -f $env:COMPUTERNAME, (Get-Date))
    $env:NECTAR_CUDA_SERVER = "${Lender}:$Port"
    # Nectar's Vulkan driver alone, instead of this PC's own
    $env:VK_DRIVER_FILES = Join-Path $nectar "nectar_icd.json"
    $env:NECTAR_VK_SERVER = "${Lender}:$Port"
    if ($Log) { $env:NECTAR_CUDA_LOG = "1" }
}
if ($Log -and -not $Reference) { $env:NECTAR_VK_LOG = "1" }
New-Item -ItemType Directory -Force $out | Out-Null

# Windows sleeps after its idle time even while a test runs, which drops the connection.
# This asks it not to for as long as this script runs, as video players do; no setting changes.
Add-Type -Namespace Nectar -Name Power -MemberDefinition @'
[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);
'@
# ES_CONTINUOUS | ES_SYSTEM_REQUIRED
$null = [Nectar.Power]::SetThreadExecutionState([uint32]"0x80000001")

$summary = New-Object System.Collections.Generic.List[string]

function Note([string] $line, [string] $color = "Gray") {
    Write-Host $line -ForegroundColor $color
    $summary.Add($line)
}

function Save([string] $name, $lines) {
    $lines | Out-File -FilePath (Join-Path $out $name) -Encoding utf8
}

# Quotes arguments the way the C runtime splits a command line
function Join-Arguments([string[]] $arguments) {
    ($arguments | ForEach-Object {
        if ($_ -ne "" -and $_ -notmatch '[\s"]') { $_ }
        else { '"' + (($_ -replace '(\\*)"', '$1$1\"') -replace '(\\+)$', '$1$1') + '"' }
    }) -join " "
}

# Starts a program with its output in <name>.out and <name>.err. Without Nectar installed, a
# CUDA program runs through nectar-cuda-run so that it borrows the lender's GPU; a Vulkan
# program finds Nectar's driver through VK_DRIVER_FILES.
function Start-Program([string] $name, [string] $exe, [string[]] $arguments, [switch] $Direct) {
    if (-not $Direct -and -not $Reference -and -not $installed -and $current -eq "cuda") {
        $arguments = @("--dll", (Join-Path $nectar "nvcuda.dll"), $exe) + $arguments
        $exe = Join-Path $nectar "nectar-cuda-run.exe"
    }
    $options = @{
        FilePath = $exe; PassThru = $true; NoNewWindow = $true
        RedirectStandardOutput = (Join-Path $out "$name.out"); RedirectStandardError = (Join-Path $out "$name.err")
    }
    # Start-Process refuses an empty argument list
    if ($arguments) { $options.ArgumentList = Join-Arguments $arguments }
    $process = Start-Process @options
    # Windows PowerShell only keeps the exit code if the handle was opened while it ran
    $null = $process.Handle
    $process
}

function Stop-Programs($process) {
    if ($process -and -not $process.HasExited) { Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue }
    Get-Process llama-server, llama-bench -ErrorAction SilentlyContinue |
        Where-Object { $_.Path -and ($_.Path.StartsWith($cudaBin, [StringComparison]::OrdinalIgnoreCase) -or
            $_.Path.StartsWith($vulkanBin, [StringComparison]::OrdinalIgnoreCase)) } |
        Stop-Process -Force -ErrorAction SilentlyContinue
}

# Where each output file was read up to, and the part of a line not finished yet
$tails = @{}

# Prints what a program wrote to <name>.err (and with -Stdout, <name>.out) since last time,
# indented, as it runs. The files keep everything. The model's metadata listing, a hundred
# lines that say nothing about borrowing, is left out here.
function Show-Output([string] $name, [switch] $Stdout) {
    $files = @("$name.err")
    if ($Stdout) { $files += "$name.out" }
    foreach ($file in $files) {
        $path = Join-Path $out $file
        if (-not (Test-Path $path)) { continue }
        if (-not $tails.ContainsKey($path)) { $tails[$path] = @{ Position = 0L; Partial = "" } }
        $tail = $tails[$path]
        try {
            $stream = New-Object IO.FileStream($path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
            try {
                if ($stream.Length -le $tail.Position) { continue }
                $null = $stream.Seek($tail.Position, [IO.SeekOrigin]::Begin)
                $bytes = New-Object byte[] ($stream.Length - $tail.Position)
                $read = $stream.Read($bytes, 0, $bytes.Length)
                $tail.Position += $read
                $text = $tail.Partial + [Text.Encoding]::UTF8.GetString($bytes, 0, $read)
            } finally { $stream.Dispose() }
        } catch { continue }

        $lines = @($text -split "`n")
        $tail.Partial = $lines[-1]
        # 0..-1 would count down in PowerShell
        if ($lines.Count -lt 2) { continue }
        foreach ($line in $lines[0..($lines.Count - 2)]) {
            $line = $line.TrimEnd("`r")
            if ($line -and $line -notmatch 'llama_model_loader: - (kv|type)') {
                $color = "DarkGray"
                if ($line -like "nectar-cuda:*" -or $line -like "nectar-vk*") { $color = "Cyan" }
                Write-Host "    | $line" -ForegroundColor $color
            }
        }
    }
}

# Bytes this machine's network adapters have sent and received
function Get-Traffic {
    $sent = 0L; $received = 0L
    foreach ($adapter in Get-NetAdapter -Physical -ErrorAction SilentlyContinue | Where-Object Status -eq "Up") {
        $statistics = Get-NetAdapterStatistics -Name $adapter.Name -ErrorAction SilentlyContinue
        if ($statistics) { $sent += $statistics.SentBytes; $received += $statistics.ReceivedBytes }
    }
    @{ Sent = $sent; Received = $received; At = [DateTime]::UtcNow }
}

# One line saying a step is still going: time, and what has crossed the network since the
# step began and lately, so a model upload over Wi-Fi shows as moving
function Write-Heartbeat([string] $what, $clock, $start, $last, [string] $extra = "") {
    $now = Get-Traffic
    $seconds = [math]::Max(($now.At - $last.At).TotalSeconds, 0.001)
    $line = "  {0}: {1:N0} s, sent {2:N0} MB, received {3:N0} MB, lately {4:N1} MB/s up, {5:N1} MB/s down" -f `
        $what, $clock.Elapsed.TotalSeconds, (($now.Sent - $start.Sent) / 1e6), (($now.Received - $start.Received) / 1e6),
        (($now.Sent - $last.Sent) / 1e6 / $seconds), (($now.Received - $last.Received) / 1e6 / $seconds)
    if ($extra) { $line += "; $extra" }
    Write-Host $line -ForegroundColor White
    $now
}

# Waits for a program, showing its output as it comes and every 15 s how it is going.
# Returns its exit code, or $null if it ran out of time.
function Wait-Program($process, [string] $name, [string] $what, [int] $seconds, [switch] $Stdout) {
    $clock = [Diagnostics.Stopwatch]::StartNew()
    $start = Get-Traffic
    $last = $start
    $next = 15
    while (-not $process.WaitForExit(500)) {
        Show-Output $name -Stdout:$Stdout
        if ($clock.Elapsed.TotalSeconds -ge $seconds) {
            Stop-Programs $process
            return $null
        }
        if ($clock.Elapsed.TotalSeconds -ge $next) {
            $last = Write-Heartbeat $what $clock $start $last
            $next += 15
        }
    }
    $process.WaitForExit()
    Show-Output $name -Stdout:$Stdout
    $process.ExitCode
}

function Describe-Exit($code) {
    if ($null -eq $code) { return "timed out" }
    # Hex shows NTSTATUS codes such as 0xC0000409 the way Windows writes them
    if ($code -lt 0 -or $code -gt 0xFFFF) { return ("exit 0x{0:X8}" -f [int]$code) }
    "exit $code"
}

function Find-Device([string] $path) {
    Get-Content $path -ErrorAction SilentlyContinue |
        ForEach-Object { if ($_ -match 'Device 0: (.+?), compute capability') { $Matches[1] } } |
        Select-Object -First 1
}

# Why a llama.cpp run did not use the lender's GPU, or nothing. llama.cpp falls back to the
# CPU, or to a GPU of the laptop's own, without failing, so its exit code says nothing.
# Calls Nectar does not support yet are noted; programs often carry on without them.
# llama-server does not log its device, so -NoDevice skips that check.
function Find-BorrowProblem([string] $name, [switch] $NoDevice) {
    $path = Join-Path $out "$name.err"
    $log = Get-Content $path -ErrorAction SilentlyContinue
    $unsupported = '^nectar-cuda: .* not supported|^nectar-vk.*can.t be carried'
    foreach ($line in $log | Where-Object { $_ -match $unsupported } | Select-Object -First 5) {
        Note "  warning: $line" "Yellow"
    }
    $nectarProblem = $log | Where-Object { $_ -like "nectar-cuda: cannot*" -or $_ -match '^nectar-vk.*(cannot reach|lost the lender)' } |
        Select-Object -First 1
    if ($nectarProblem) { return $nectarProblem }
    # The Vulkan device was chosen by name before the run (-dev)
    if ($NoDevice -or $current -eq "vulkan") { return $null }

    $device = Find-Device $path
    if (-not $device) { return "llama.cpp found no CUDA device" }
    if (-not $Reference) {
        $lender = Find-Device (Join-Path $expected "bench.err")
        if ($lender -and $device -ne $lender) { return "llama.cpp ran on $device, not the lender's $lender" }
    }
    $null
}

function Read-Json([string] $path) {
    if (-not (Test-Path $path)) { return $null }
    try { [IO.File]::ReadAllText($path, [Text.Encoding]::UTF8) | ConvertFrom-Json } catch { $null }
}

# The Vulkan device llama.cpp should use, as its -dev name (VulkanN): the one named "(Nectar)",
# or on the lender itself the first. Null if there is none. The list goes to devices.txt.
function Select-VulkanDevice {
    $ErrorActionPreference = "Continue"
    $list = @(& (Join-Path $bin "llama-bench.exe") --list-devices 2>&1 | ForEach-Object { "$_" })
    $ErrorActionPreference = "Stop"
    Save "devices.txt" $list
    $devices = @($list | Where-Object { $_ -match '^\s*(Vulkan\d+): (.+)$' } |
        ForEach-Object { [pscustomobject]@{ Id = $Matches[1]; Name = $Matches[2] } })
    foreach ($device in $devices) { Note "  $($device.Id): $($device.Name)" }
    if ($Reference) { $chosen = $devices | Select-Object -First 1 }
    else { $chosen = $devices | Where-Object { $_.Name -like "*(Nectar)*" } | Select-Object -First 1 }
    if ($chosen) { $chosen.Id } else { $null }
}

# llama.cpp's arguments choosing the device, for Vulkan
function Device-Arguments {
    if ($current -eq "vulkan" -and $vulkanDevice) { @("-dev", $vulkanDevice) } else { @() }
}

function Step-System {
    $os = Get-CimInstance Win32_OperatingSystem
    $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
    $lines = @(
        "Computer    $env:COMPUTERNAME",
        "Windows     $($os.Caption) $($os.Version)",
        "CPU         $($cpu.Name.Trim()), $($cpu.NumberOfCores) cores",
        ("Memory      {0:N1} GiB" -f ($os.TotalVisibleMemorySize / 1MB)),
        "PowerShell  $($PSVersionTable.PSVersion)"
    )
    foreach ($gpu in Get-CimInstance Win32_VideoController) {
        $lines += "GPU         $($gpu.Name), driver $($gpu.DriverVersion)"
    }
    $driver = Get-Item (Join-Path $env:SystemRoot "System32\nvcuda.dll") -ErrorAction SilentlyContinue
    if ($driver) { $lines += "Own CUDA    nvcuda.dll $($driver.VersionInfo.FileVersion) in System32" }
    else { $lines += "Own CUDA    none (no NVIDIA driver)" }
    $manifest = Join-Path $bundle "manifest.txt"
    if (Test-Path $manifest) { $lines += ""; $lines += Get-Content $manifest }
    Save "system.txt" $lines
    Note ("System      {0}; {1}" -f $env:COMPUTERNAME, $cpu.Name.Trim())
}

function Step-Network {
    $lines = @("Lender $Lender, port $Port", "")

    $wifi = netsh wlan show interfaces 2>$null | Where-Object {
        $_ -match '^\s*(State|Radio type|Band|Channel|Receive rate \(Mbps\)|Transmit rate \(Mbps\)|Signal)\s*:'
    }
    if ($wifi) { $lines += "Wi-Fi (netsh wlan show interfaces):"; $lines += $wifi; $lines += "" }
    foreach ($adapter in Get-NetAdapter -Physical -ErrorAction SilentlyContinue | Where-Object Status -eq "Up") {
        $lines += "Adapter: $($adapter.Name), $($adapter.InterfaceDescription), $($adapter.LinkSpeed)"
    }

    # This PC's own GPUs and the APIs their drivers offer, for planning other vendors
    $detect = Join-Path $nectar "nectar-detect.exe"
    if (Test-Path $detect) {
        & $detect 2>&1 | Out-File (Join-Path $out "detect.txt") -Encoding utf8
    }

    $count = 10
    $pings = @(Test-Connection -ComputerName $Lender -Count $count -ErrorAction SilentlyContinue)
    # Windows PowerShell names the time ResponseTime, PowerShell 7 Latency
    $times = @($pings | ForEach-Object {
        if ($_.PSObject.Properties["ResponseTime"]) { $_.ResponseTime } else { $_.Latency }
    })
    if ($times.Count -gt 0) {
        $stats = $times | Measure-Object -Minimum -Maximum -Average
        $ping = "ping {0}/{1:N1}/{2} ms (min/avg/max), {3} of {4} lost" -f `
            $stats.Minimum, $stats.Average, $stats.Maximum, ($count - $times.Count), $count
    } else {
        $ping = "ping: no replies (the lender's firewall may block ping; that alone is fine)"
    }
    $lines += ""; $lines += $ping

    $client = New-Object Net.Sockets.TcpClient
    $clock = [Diagnostics.Stopwatch]::StartNew()
    try { $open = $client.ConnectAsync($Lender, $Port).Wait(3000) } catch { $open = $false }
    $connectMs = $clock.Elapsed.TotalMilliseconds
    $client.Close()
    if ($open) { $tcp = "port $Port open, connected in {0:N1} ms" -f $connectMs }
    else { $tcp = "port $Port did not answer within 3 s" }
    $lines += $tcp
    Save "network.txt" $lines

    $link = ($wifi | Where-Object { $_ -match 'Radio type|Receive rate|Signal' } |
        ForEach-Object { ($_ -split ':', 2)[1].Trim() }) -join ", "
    if ($link) { Note "Wi-Fi       $link" }
    Note "Network     $ping"
    if ($open) {
        Note "Lender      $tcp" "Green"
    } else {
        Note "Lender      FAIL: $tcp. Is the desktop lending (Nectar, Lend this GPU), and does its firewall rule allow this laptop?" "Red"
    }
    $open
}

# Nectar's link check under each profile, leaving the user's choice as it was
function Step-Link {
    $chosen = (Get-BorrowStatus).profile
    $lines = @()
    foreach ($profile in "gaming", "ai") {
        Ask-Nectar "profile $profile" | Out-Null
        Start-Sleep 2
        $answer = Ask-Nectar "check"
        if ($answer.error) { Note "Link        $($answer.error)" "Red"; break }
        do {
            Start-Sleep 1
            $status = Get-BorrowStatus
        } while ($status.check -eq "running")
        Note "Link check for $profile" "Cyan"
        foreach ($line in @($status.check_line) + @($status.advice)) {
            Note "  $line"
            $lines += "${profile}: $line"
        }
    }
    if ($status.wifi_held -eq "1") { Note "  Wi-Fi is in low-latency mode while the GPU is borrowed" }
    $lines += "wifi_held=$($status.wifi_held)"
    Save "link.txt" $lines
    Ask-Nectar "profile $chosen" | Out-Null
}

# A window of frames from the lender's GPU, checked on screen
function Step-Frames {
    $arguments = @("--mib", "16", "--calls", "10", "--present", "300", "--size", "1280x720")
    if (-not $Reference) { $arguments = @("--device", "(Nectar)") + $arguments }
    Write-Host "frames: vk-probe --present"
    $process = Start-Program "frames" (Join-Path $nectar "vk-probe.exe") $arguments -Direct
    $code = Wait-Program $process "frames" "vk-probe" 300 -Stdout
    $output = Get-Content (Join-Path $out "frames.out") -ErrorAction SilentlyContinue
    foreach ($line in $output | Where-Object { $_ -match '^(present|on screen)' }) { Note "  $line" }
    if ($code -eq 0 -and ($output | Where-Object { $_ -like "PASS*" })) { Note "Frames      PASS" "Green" }
    else {
        $failure = $output | Where-Object { $_ -like "FAIL*" } | Select-Object -First 1
        Note "Frames      FAIL ($(Describe-Exit $code)) $failure; see frames.out and frames.err" "Red"
    }
}

# A Direct3D 11 program through DXVK, which turns it into Vulkan for Nectar's driver
function Step-Direct3D {
    $probe = Join-Path $bundle "direct3d\d3d11_probe.exe"
    if (-not (Test-Path $probe)) { Note "Direct3D    no d3d11_probe in this bundle" "Yellow"; return }
    if (-not $Reference) { $env:DXVK_FILTER_DEVICE_NAME = "(Nectar)"; $env:NECTAR_VK_STATS = "1" }
    $env:DXVK_LOG_PATH = $out
    Write-Host "direct3d: d3d11_probe through DXVK"
    $process = Start-Program "direct3d" $probe @("--frames", "300", "--draws", "2000", "--size", "1280x720") -Direct
    $code = Wait-Program $process "direct3d" "d3d11_probe" 600 -Stdout
    Remove-Item Env:DXVK_FILTER_DEVICE_NAME, Env:DXVK_LOG_PATH, Env:NECTAR_VK_STATS -ErrorAction SilentlyContinue
    $output = Get-Content (Join-Path $out "direct3d.out") -ErrorAction SilentlyContinue
    foreach ($line in $output | Where-Object { $_ -match '^(adapter|frames|on screen)' }) { Note "  $line" }
    if ($code -eq 0 -and ($output | Where-Object { $_ -like "PASS*" })) { Note "Direct3D    PASS" "Green" }
    else {
        $failure = $output | Where-Object { $_ -like "FAIL*" } | Select-Object -First 1
        Note "Direct3D    FAIL ($(Describe-Exit $code)) $failure; see direct3d.out and d3d11_probe_d3d11.log" "Red"
    }
}

function Step-Probe {
    if ($current -eq "vulkan") {
        $arguments = @()
        if (-not $Reference) { $arguments = @("--device", "(Nectar)") }
        Write-Host "probe: vk-probe"
        $process = Start-Program "probe" (Join-Path $nectar "vk-probe.exe") $arguments -Direct
    } else {
        if ($Reference -or $installed) { $dll = $systemDriver }
        else { $dll = Join-Path $nectar "nvcuda.dll" }
        Write-Host "probe: cuda-probe on $dll"
        $process = Start-Program "probe" (Join-Path $nectar "cuda-probe.exe") @($dll) -Direct
    }
    $code = Wait-Program $process "probe" "cuda-probe" 300 -Stdout
    $output = Get-Content (Join-Path $out "probe.out") -ErrorAction SilentlyContinue
    $passed = $code -eq 0 -and ($output | Where-Object { $_ -like "PASS*" })
    foreach ($line in $output | Where-Object { $_ -match '^(device \d|latency|upload|download|kernel|compute|again)' }) {
        Note "  $line"
    }
    if ($passed) { Note "Probe       PASS" "Green" }
    else {
        $failure = $output | Where-Object { $_ -like "FAIL*" } | Select-Object -First 1
        Note "Probe       FAIL ($(Describe-Exit $code)) $failure; see probe.out and probe.err" "Red"
    }
}

function Step-Bench {
    if ($Quick) { $size = @("-p", "64", "-n", "16", "-r", "1") }
    else { $size = @("-p", "512", "-n", "128", "-r", "3") }
    Write-Host "bench: llama-bench (loading the model sends it all to the lender, which takes a while over Wi-Fi)"
    $clock = [Diagnostics.Stopwatch]::StartNew()
    $process = Start-Program "bench" (Join-Path $bin "llama-bench.exe") (@("-m", $model.FullName, "-ngl", "99", "-o", "json") + $size + (Device-Arguments))
    $code = Wait-Program $process "bench" "llama-bench" 1800
    $seconds = $clock.Elapsed.TotalSeconds
    Copy-Item (Join-Path $out "bench.out") (Join-Path $out "bench.json") -Force -ErrorAction SilentlyContinue

    $rows = Read-Json (Join-Path $out "bench.json")
    if ($code -ne 0 -or -not $rows) {
        Note "Bench       FAIL ($(Describe-Exit $code)) after $([math]::Round($seconds)) s; see bench.err" "Red"
        return
    }
    $problem = Find-BorrowProblem "bench"
    if ($problem) {
        Note "Bench       FAIL: $problem; see bench.err" "Red"
        return
    }
    $references = @{}
    if (-not $Reference) {
        # Not @(...): Windows PowerShell would nest the array ConvertFrom-Json returns
        $referenceRows = Read-Json (Join-Path $expected "bench.json")
        foreach ($row in $referenceRows) {
            if ($row) { $references[(Test-Name $row)] = $row.avg_ts }
        }
    }
    foreach ($row in $rows) {
        $name = Test-Name $row
        $line = "  {0,-6} {1,8:N1} t/s +/- {2:N1}" -f $name, $row.avg_ts, $row.stddev_ts
        if ($references.ContainsKey($name)) {
            $line += "   lender natively {0:N1} t/s, so {1:N0}%" -f $references[$name], (100 * $row.avg_ts / $references[$name])
        }
        Note $line
    }
    Note ("Bench       PASS in {0:N0} s, model load included" -f $seconds) "Green"
}

function Test-Name($row) {
    if ($row.n_gen -eq 0) { "pp$($row.n_prompt)" } else { "tg$($row.n_gen)" }
}

function Step-Chat {
    Write-Host "chat: llama-server answers one question"
    $clock = [Diagnostics.Stopwatch]::StartNew()
    $process = Start-Program "chat-server" (Join-Path $bin "llama-server.exe") (@(
        "-m", $model.FullName, "-ngl", "99", "--host", "127.0.0.1", "--port", "$serverPort") + (Device-Arguments))
    $ready = $false
    $next = 15
    $start = Get-Traffic
    $last = $start
    while (-not $ready -and -not $process.HasExited -and $clock.Elapsed.TotalSeconds -lt 900) {
        try {
            $health = Invoke-WebRequest "http://127.0.0.1:$serverPort/health" -UseBasicParsing -TimeoutSec 2
            $ready = $health.StatusCode -eq 200
        } catch { Start-Sleep -Milliseconds 500 }
        Show-Output "chat-server"
        if ($clock.Elapsed.TotalSeconds -ge $next) {
            $last = Write-Heartbeat "llama-server loading" $clock $start $last
            $next += 15
        }
    }
    Show-Output "chat-server"
    if (-not $ready) {
        $code = $null
        if ($process.HasExited) { $code = $process.ExitCode }
        Stop-Programs $process
        Note "Chat        FAIL: llama-server never got ready ($(Describe-Exit $code)); see chat-server.err" "Red"
        return
    }
    $loadSeconds = $clock.Elapsed.TotalSeconds

    $request = [IO.File]::ReadAllBytes((Join-Path $bundle "chat-request.json"))
    $wantedTokens = (Read-Json (Join-Path $expected "chat-reply.json")).timings.predicted_n
    if ($wantedTokens) { Write-Host "  asking; the lender's answer is $wantedTokens tokens" } else { Write-Host "  asking" }
    # Sent in the background, so the server's output and the tokens it has generated so far
    # show while it answers
    Add-Type -AssemblyName System.Net.Http
    $client = New-Object Net.Http.HttpClient
    # Generous, so a slow link gives a slow answer to compare rather than none
    $client.Timeout = [TimeSpan]::FromSeconds(3600)
    try {
        $content = New-Object Net.Http.ByteArrayContent -ArgumentList (, $request)
        $content.Headers.ContentType = [Net.Http.Headers.MediaTypeHeaderValue]::Parse("application/json; charset=utf-8")
        $task = $client.PostAsync("http://127.0.0.1:$serverPort/v1/chat/completions", $content)
        $clock = [Diagnostics.Stopwatch]::StartNew()
        $start = Get-Traffic
        $last = $start
        $next = 15
        while (-not $task.Wait(500)) {
            Show-Output "chat-server"
            if ($clock.Elapsed.TotalSeconds -ge $next) {
                # llama-server's slots say how far each answer has got
                $progress = ""
                try {
                    $slots = (Invoke-WebRequest "http://127.0.0.1:$serverPort/slots" -UseBasicParsing -TimeoutSec 2).Content
                    $decoded = [regex]::Matches($slots, '"n_decoded"\s*:\s*(\d+)') |
                        ForEach-Object { [int]$_.Groups[1].Value } | Measure-Object -Maximum
                    if ($decoded.Count) { $progress = "$($decoded.Maximum) of $wantedTokens tokens" }
                } catch {}
                $last = Write-Heartbeat "answering" $clock $start $last $progress
                $next += 15
            }
        }
        Show-Output "chat-server"
        $response = $task.Result
        $text = $response.Content.ReadAsStringAsync().Result
        if (-not $response.IsSuccessStatusCode) { throw "HTTP $([int]$response.StatusCode): $text" }
        [IO.File]::WriteAllText((Join-Path $out "chat-reply.json"), $text, (New-Object Text.UTF8Encoding $false))
    } catch {
        Stop-Programs $process
        $reason = $_.Exception
        while ($reason.InnerException) { $reason = $reason.InnerException }
        Note "Chat        FAIL: the request failed: $($reason.Message)" "Red"
        return
    } finally {
        $client.Dispose()
    }
    Stop-Programs $process
    # A run on the CPU or another GPU would also give a different answer below
    $problem = Find-BorrowProblem "chat-server" -NoDevice
    if ($problem) {
        Note "Chat        FAIL: $problem; see chat-server.err" "Red"
        return
    }

    $reply = Read-Json (Join-Path $out "chat-reply.json")
    $message = $reply.choices[0].message
    Save "chat-answer.txt" @("[thinking]", $message.reasoning_content, "", "[answer]", $message.content)
    $timings = $reply.timings
    Note ("  load {0:N0} s; prompt {1} tokens at {2:N1} t/s; generated {3} tokens at {4:N1} t/s" -f `
        $loadSeconds, $timings.prompt_n, $timings.prompt_per_second, $timings.predicted_n, $timings.predicted_per_second)

    if ($Reference) {
        Note "Chat        recorded" "Green"
        return
    }
    $wanted = Read-Json (Join-Path $expected "chat-reply.json")
    if (-not $wanted) {
        Note "Chat        answered, but there is no expected answer to compare with" "Yellow"
        return
    }
    $expectedMessage = $wanted.choices[0].message
    $same = ($message.content -ceq $expectedMessage.content) -and
        ($message.reasoning_content -ceq $expectedMessage.reasoning_content)
    $speed = ""
    if ($wanted.timings.predicted_per_second) {
        $speed = " Generation at {0:N0}% of the lender's native {1:N1} t/s." -f `
            (100 * $timings.predicted_per_second / $wanted.timings.predicted_per_second), $wanted.timings.predicted_per_second
    }
    if ($same) { Note "Chat        PASS: the answer matches the lender's word for word.$speed" "Green" }
    else { Note "Chat        FAIL: the answer differs from the lender's; compare chat-answer.txt with expected\chat-answer.txt.$speed" "Red" }
}

# ---- run the steps

if ($Reference) { Note "Reference run on $env:COMPUTERNAME, natively without Nectar" "Cyan" }
elseif ($installed) { Note "Borrowed from $($status.lender) (${Lender}:$Port) through Nectar's System32\nvcuda.dll; logs in $out" "Cyan" }
else { Note "Borrowing from ${Lender}:$Port; logs in $out" "Cyan" }
if (-not $model -and ($Steps -contains "bench" -or $Steps -contains "chat")) {
    throw "No model in $(Join-Path $bundle 'models')"
}

Step-System
$reachable = $true
if ($Steps -contains "network") { $reachable = Step-Network }
if ($reachable -and $installed -and -not $Reference -and $Steps -contains "link") { Step-Link }
$top = $out
if (-not $reachable) {
    Note "Stopped: the lender cannot be reached" "Red"
} else {
    try {
        foreach ($current in $Api) {
            if ($current -eq "vulkan") {
                $bin = $vulkanBin
                $out = Join-Path $top "vulkan"
                $expected = Join-Path $expectedRoot "vulkan"
                New-Item -ItemType Directory -Force $out | Out-Null
                Note ""
                Note "Vulkan (llama.cpp-vulkan; logs in $out)" "Cyan"
                $vulkanDevice = Select-VulkanDevice
                if (-not $vulkanDevice) {
                    Note "Vulkan      FAIL: llama.cpp sees no Vulkan device named (Nectar); see vulkan\devices.txt" "Red"
                    if ($Steps -contains "probe") { Step-Probe }
                    continue
                }
                Note "  using $vulkanDevice"
            } elseif ($current -eq "cuda") {
                $bin = $cudaBin
                $out = $top
                $expected = $expectedRoot
                Note ""
                Note "CUDA (llama.cpp)" "Cyan"
            } else {
                throw "Unknown API ${current}: cuda or vulkan"
            }
            if ($Steps -contains "probe") { Step-Probe }
            if ($current -eq "vulkan" -and $Steps -contains "frames") { Step-Frames }
            if ($current -eq "vulkan" -and $Steps -contains "direct3d") { Step-Direct3D }
            if ($Steps -contains "bench") { Step-Bench }
            if ($Steps -contains "chat") { Step-Chat }
        }
    } finally {
        $out = $top
        Stop-Programs $null
    }
}

Save "summary.txt" $summary
if (-not $Reference) {
    $zip = "$out.zip"
    Compress-Archive -Path (Join-Path $out "*") -DestinationPath $zip -Force
    Write-Host ""
    Write-Host "Results: $zip" -ForegroundColor Cyan
    Write-Host "Copy it to the desktop and unzip it into build\laptop-bundle\results\ in the Nectar repo."
}
