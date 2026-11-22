<#
.SYNOPSIS
Puts together the folder to copy to the laptop for the Wi-Fi test.

.DESCRIPTION
Run on the lender after Build-All.ps1. The folder (build\laptop-bundle\Nectar-Borrower) holds:

  nectar\       Nectar: the app and service, nvcuda.dll, the Vulkan driver (nectar_vk.dll and
                nectar_icd.json), Install-Borrower.ps1, plus nectar-cuda-run, cuda-probe and
                vk-probe for testing. Nothing of NVIDIA's: the lender sends its hybrid runtime
                when the laptop borrows.
  llama.cpp\    llama.cpp's programs built for CUDA, with NVIDIA's cuBLAS and CUDA runtime
  llama.cpp-vulkan\
                llama.cpp's programs built for Vulkan (ggml-org's release build)
  direct3d\     d3d11_probe, a Direct3D 11 program, with DXVK's d3d11.dll and dxgi.dll beside it
  models\       the model
  expected\     this PC's own results, run natively, for the laptop to compare against
  Run-Test.cmd  what to run on the laptop

The NVIDIA and Microsoft runtime files in llama.cpp\ are for your own machines; don't publish
the folder.

Copies are skipped when the file is already there unchanged, so running it again is quick.
#>
param(
    [string] $Output,
    [string] $Llama = "$env:LOCALAPPDATA\Programs\llama.cpp",
    # llama.cpp's Vulkan build: the folder holding bin\llama-bench.exe
    [string] $LlamaVulkan = "$env:USERPROFILE\tools\llama.cpp-b12030-vulkan",
    # DXVK's release: the folder holding x64\d3d11.dll
    [string] $Dxvk = "$env:USERPROFILE\tools\dxvk-3.1.1\dxvk-3.1.1",
    [string] $Model,
    # Leave the model out (the laptop then needs its own copy in models\)
    [switch] $NoModel,
    # Skip recording this PC's results in expected\
    [switch] $NoReference
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path "$PSScriptRoot\..\..\..\..").Path
if (-not $Output) { $Output = Join-Path $repo "build\laptop-bundle\Nectar-Borrower" }
if (-not $Model) { $Model = Join-Path $Llama "models\gemma-4-12b-it-qat-q4_0.gguf" }
$built = Join-Path $repo "build\nectar-cuda"

$nectarFiles = "nvcuda.dll", "nectar.exe", "nectar-svc.exe", "nectar-cuda-run.exe", "cuda-probe.exe", "nectar-detect.exe",
    "nectar_vk.dll", "nectar_icd.json", "vk-probe.exe"
foreach ($file in $nectarFiles) {
    if (-not (Test-Path (Join-Path $built $file))) { throw "No $file in ${built}; run Build-All.ps1 first" }
}
# A build older than its sources would test old code on the laptop
$sources = Join-Path $repo "native\nectar-cuda"
$newest = Get-ChildItem $sources -Recurse -File -Include *.rs, Cargo.toml, Cargo.lock |
    Where-Object { $_.FullName -notmatch '\\(target|codegen|tests)\\' } |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
$builtAt = (Get-ChildItem $built -File | Measure-Object LastWriteTime -Maximum).Maximum
if ($newest.LastWriteTime -gt $builtAt) {
    throw "$($newest.FullName) changed after the build in $built; run Build-All.ps1 first"
}
if (-not (Test-Path "$Llama\bin\llama-server.exe")) { throw "No llama.cpp in $Llama" }
if (-not (Test-Path "$LlamaVulkan\bin\llama-server.exe")) { throw "No llama.cpp Vulkan build in $LlamaVulkan" }
if (-not $NoModel -and -not (Test-Path $Model)) { throw "No model at $Model" }
if (-not (Test-Path "$Dxvk\x64\d3d11.dll")) { throw "No DXVK in $Dxvk" }

# Copies changed files only; robocopy's exit codes below 8 mean success
function Copy-Folder([string] $from, [string] $to, [string[]] $files = @("*"), [switch] $Mirror) {
    $options = @("/NJH", "/NJS", "/NDL", "/NFL", "/NP", "/R:1", "/W:1")
    if ($Mirror) { $options += "/MIR" }
    & robocopy $from $to @files @options | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "Copying $from to $to failed (robocopy $LASTEXITCODE)" }
}

Write-Host "Bundle       $Output"
$nectar = Join-Path $Output "nectar"
$bin = Join-Path $Output "llama.cpp\bin"
$vulkanBin = Join-Path $Output "llama.cpp-vulkan\bin"
New-Item -ItemType Directory -Force $nectar, $bin, $vulkanBin, (Join-Path $Output "models") | Out-Null

Copy-Folder $built $nectar $nectarFiles
Copy-Item (Join-Path $repo "native\nectar-cuda\install\Install-Borrower.ps1") $nectar -Force
# Left by bundles from before the lender sent it
Remove-Item (Join-Path $nectar "nvcudart_hybrid64.dll") -ErrorAction SilentlyContinue
Write-Host "Nectar       $($nectarFiles -join ', '), Install-Borrower.ps1"

Copy-Folder "$Llama\bin" $bin -Mirror
Write-Host "llama.cpp    $Llama\bin"
Copy-Folder "$LlamaVulkan\bin" $vulkanBin -Mirror
Copy-Item (Join-Path $LlamaVulkan "LICENSE*") (Join-Path $Output "llama.cpp-vulkan") -ErrorAction SilentlyContinue
Write-Host "llama.cpp    $LlamaVulkan\bin (Vulkan)"

# The Direct3D 11 probe, built with DXVK beside it
$direct3d = Join-Path $Output "direct3d"
New-Item -ItemType Directory -Force $direct3d | Out-Null
$vswhereFirst = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vcvars = & $vswhereFirst -latest -products * -find "VC\Auxiliary\Build\vcvars64.bat" | Select-Object -First 1
$probeSource = Join-Path $repo "native\nectar-cuda\tests\d3d11_probe.cpp"
& cmd /c "call `"$vcvars`" >nul 2>nul && cd /d `"$direct3d`" && cl /nologo /EHsc /O2 /MT `"$probeSource`" /Fe:d3d11_probe.exe /link d3d11.lib dxgi.lib d3dcompiler.lib user32.lib gdi32.lib" | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Building d3d11_probe failed" }
Remove-Item (Join-Path $direct3d "d3d11_probe.obj") -ErrorAction SilentlyContinue
Copy-Item "$Dxvk\x64\d3d11.dll", "$Dxvk\x64\dxgi.dll" $direct3d -Force
Write-Host "Direct3D     d3d11_probe with $Dxvk"

# Microsoft's C++ runtime, which llama.cpp uses, next to its programs. Nectar's own binaries
# carry theirs (.cargo\config.toml).
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vs = & $vswhere -latest -products * -property installationPath
$redist = Get-ChildItem "$vs\VC\Redist\MSVC\*\x64" -Directory |
    Where-Object { $_.Parent.Name -match '^\d+(\.\d+)+$' } |
    Sort-Object { [version]$_.Parent.Name } -Descending | Select-Object -First 1
if (-not $redist) { throw "No Visual C++ redistributable files in $vs" }
$runtime = @(Get-ChildItem "$($redist.FullName)\Microsoft.VC*.CRT", "$($redist.FullName)\Microsoft.VC*.OpenMP" -Directory)
foreach ($folder in $runtime) {
    $files = @("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll", "vcomp140.dll") |
        Where-Object { Test-Path (Join-Path $folder.FullName $_) }
    if ($files) {
        Copy-Folder $folder.FullName $bin $files
        Copy-Folder $folder.FullName $vulkanBin $files
        # Left in nectar\ by bundles from before Nectar carried its own
        foreach ($file in $files) { Remove-Item (Join-Path $nectar $file) -ErrorAction SilentlyContinue }
    }
}
Write-Host "C++ runtime  $($redist.Parent.Name)"

if (-not $NoModel) {
    $modelFile = Get-Item $Model
    Write-Host ("Model        {0} ({1:N1} GB), copying if changed..." -f $modelFile.Name, ($modelFile.Length / 1GB))
    Copy-Folder $modelFile.DirectoryName (Join-Path $Output "models") @($modelFile.Name)
}

foreach ($file in "Test-Borrow.ps1", "Run-Test.cmd", "chat-request.json", "README.md") {
    Copy-Item (Join-Path $PSScriptRoot $file) $Output -Force
}

# What the bundle was made from, so results can be traced back
$protocol = Get-Content (Join-Path $repo "native\nectar-cuda\protocol\src\lib.rs") |
    Select-String "pub const VERSION: u32 = (\d+)" | ForEach-Object { $_.Matches[0].Groups[1].Value }
$generation = Get-Content (Join-Path $repo "native\nectar-cuda\protocol\src\functions.rs") |
    Select-String "pub const GENERATION: u32 = (0x[0-9a-f]+)" | ForEach-Object { $_.Matches[0].Groups[1].Value }
$ErrorActionPreference = "Continue"
$llamaVersion = (& "$bin\llama-bench.exe" --version 2>&1 | ForEach-Object { "$_" } | Where-Object { $_ -like "version:*" }) -join ""
$llamaVulkanVersion = (& "$vulkanBin\llama-bench.exe" --version 2>&1 | ForEach-Object { "$_" } | Where-Object { $_ -like "version:*" }) -join ""
$ErrorActionPreference = "Stop"
$gpu = Get-CimInstance Win32_VideoController | Where-Object Name -match "NVIDIA" | Select-Object -First 1
$manifest = @(
    "Bundle made $(Get-Date -Format 'yyyy-MM-dd HH:mm') on $env:COMPUTERNAME",
    "Lender GPU  $($gpu.Name), driver $($gpu.DriverVersion)",
    "Protocol    version $protocol, generation $generation",
    "llama.cpp   $llamaVersion (CUDA)",
    "llama.cpp   $llamaVulkanVersion (Vulkan)",
    "DXVK        $(Split-Path $Dxvk -Leaf)",
    "Hybrid      sent by the lender when the laptop borrows",
    "",
    "SHA-256 of Nectar's files:"
)
foreach ($file in Get-ChildItem $nectar -File) {
    $manifest += "  {0}  {1}" -f (Get-FileHash $file.FullName -Algorithm SHA256).Hash.ToLower(), $file.Name
}
$manifest | Out-File (Join-Path $Output "manifest.txt") -Encoding utf8

if (-not $NoReference) {
    Write-Host ""
    Write-Host "Recording this PC's own results in expected\ (natively, without Nectar)..."
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Output "Test-Borrow.ps1") -Reference
    if ($LASTEXITCODE -ne 0) { throw "The reference run failed" }
}

$size = (Get-ChildItem $Output -Recurse -File | Where-Object { $_.FullName -notlike "*\results\*" } |
    Measure-Object Length -Sum).Sum
Write-Host ""
Write-Host ("Done: {0} ({1:N1} GB)" -f $Output, ($size / 1GB)) -ForegroundColor Green
Write-Host "Copy that folder to the laptop and follow README.md in it."
