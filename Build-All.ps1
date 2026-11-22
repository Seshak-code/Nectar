param (
    # Defaults to build in the repo (set below: Windows PowerShell has no $PSScriptRoot here)
    [Parameter()]
    [string]
    $Output,

    [Parameter()]
    [switch]
    $BuildDebug
)

if ($env:OS -ne "Windows_NT")
{
    throw "This builds the Windows files; elsewhere run cargo build --release in native/nectar-cuda"
}

if (-not $Output)
{
    $Output = Join-Path $PSScriptRoot "build"
}

# The native builds go to $Output\native\..., which in the repo root are the source folders
if ([IO.Path]::GetFullPath($Output).TrimEnd('\', '/') -eq $PSScriptRoot)
{
    throw "Build into a folder of its own, such as build (the default), not the repo root"
}

# Nectar needs Rust. Its nvcuda.dll gets a folder of its own, so a CUDA program
# started from $Output never loads it by accident.
$Cargo = Get-Command cargo -ErrorAction SilentlyContinue
if (-not $Cargo)
{
    $Cargo = Get-Command (Join-Path $HOME ".cargo\bin\cargo.exe") -ErrorAction SilentlyContinue
}

if (-not $Cargo)
{
    throw "Rust was not found: install it from https://rustup.rs"
}

$CargoProfile = if ($BuildDebug) { "debug" } else { "release" }
$CargoTarget = [IO.Path]::GetFullPath((Join-Path $Output "native\nectar-cuda"))

$CargoArgs = @("build", "--manifest-path", (Join-Path $PSScriptRoot "native\nectar-cuda\Cargo.toml"), "--target-dir", $CargoTarget)
if (-not $BuildDebug)
{
    $CargoArgs += "--release"
}

# From the workspace's folder, so cargo reads its .cargo\config.toml (static C runtime)
Push-Location (Join-Path $PSScriptRoot "native\nectar-cuda")
& $Cargo @CargoArgs
Pop-Location
if ($LASTEXITCODE -ne 0)
{
    throw "cargo build failed"
}

$CudaOutput = Join-Path $Output "nectar-cuda"
New-Item -ItemType Directory -Force $CudaOutput | Out-Null
# The lend/borrow app and service (nectar.exe, nectar-svc.exe) and nectar-detect beside the
# server, and the Vulkan driver with its manifest and vk-probe
foreach ($File in "nvcuda.dll", "nectar-server.exe", "nectar-cuda-run.exe", "cuda-probe.exe", "nectar.exe", "nectar-svc.exe", "nectar-detect.exe", "nectar_vk.dll", "vk-probe.exe", "nectar-netsim.exe")
{
    Copy-Item (Join-Path $CargoTarget "$CargoProfile\$File") $CudaOutput
}
Copy-Item (Join-Path $PSScriptRoot "native\nectar-cuda\install\nectar_icd.json") $CudaOutput

# The tracer stands in for the driver under the same name, so it gets its own folder
New-Item -ItemType Directory -Force (Join-Path $CudaOutput "trace") | Out-Null
Copy-Item (Join-Path $CargoTarget "$CargoProfile\nvcudatrace.dll") (Join-Path $CudaOutput "trace\nvcuda.dll")
