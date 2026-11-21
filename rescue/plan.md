# Nectar: lend a GPU, borrow it as if it were installed

## NOW: Phase 3, Nectar's Vulkan driver for compute and games (proposed 2026-11-01)

### Context
- **What the user asked for:** Direct3D games through DXVK, built on Steam's Proton and FEX compatibility lists. Their reasons:
  - It lets the open hardware design hold drivers too.
  - Their Mac (M4 or newer, macOS only) makes a good stress test, reached over SSH from the desktop.
  - The Linux games machine is decided later.
- **What it rests on:** DXVK (Direct3D 9–11) and vkd3d-proton (Direct3D 12) only turn DirectX into Vulkan. A borrowed GPU needs **a Vulkan driver that reaches the lender**: Phase 3, planned here so that it can do graphics from the start.
- **Why Proton matters:** on Linux, Proton already runs every DirectX game on Vulkan, and Vulkan drivers there are user-mode files that need no signing. So the Linux borrower gets DirectX games, including Direct3D 12, without touching the game. FEX (inside Proton's ARM64 builds) carries that to ARM Linux.
- **What doesn't change:** Windows keeps needing DXVK placed beside each game. Xbox-app games stay out, since they need a WDDM driver.
- **Untouched:** the CUDA path, which keeps working as it is.

### Milestones
Approval now covers **3.0–3.2**. Milestones 3.3–3.4 are designed here, and get a go/no-go once 3.2's Wi‑Fi numbers show what each wait for the lender costs.

**3.0 Setup** (each install confirmed when it comes up):
- **Vulkan's registry:** `vk.xml` from Khronos' Vulkan-Headers (Apache-2.0/MIT), into `C:\Users\Sriram\tools\vulkan-headers-<version>`.
- **The Mac:**
  - The user turns on Remote Login. The desktop gets an ed25519 key, whose public half the user adds to the Mac.
  - **Tools on the Mac:** Xcode Command Line Tools, rustup, and LunarG's Vulkan SDK for macOS (loader, MoltenVK, KosmicKrisp, vulkaninfo, glslc). Then cmake and llama.cpp built with Vulkan.
- **Rename:** `nectar-cuda-server` becomes `nectar-server`. It lends several APIs now, so the app, Build-All, the scripts and the README all follow. The workspace folder keeps its name for now.

**3.1 The Vulkan protocol, the borrower's driver, and the lender's side:**
- **Codegen** (`codegen/`, new `vk` module) reads `vk.xml` with `roxmltree` (MIT/Apache-2.0; codegen only). It writes:
  - The types, with `pNext` chains serialized through `structextends`.
  - The command ids.
  - Encoders and decoders, for the borrower's driver and for the lender.
  - A coverage report, as the CUDA codegen does.
- **The wire format and its design ideas** come from Mesa's Venus (MIT; the venus-protocol license is *to verify*); its code isn't used:
  - **Handles:** the client assigns ids for non-dispatchable handles, so creates and destroys don't wait.
  - **Command buffers:** `vkCmd*` calls are recorded locally into a stream and sent at `vkQueueSubmit`.
  - **Answers kept locally:** properties, features and formats are fetched once and kept, like CUDA's kept answers.
  - **The transport:** it reuses the batching, deferred errors and upload cache already in the CUDA protocol (`protocol`, `client/src/calls.rs`).
- **Mapped memory:**
  - **Uploads:** memory the app maps is shadowed on the borrower. At submit or flush, changed 4 KiB pages are found by hash and sent.
  - **Read-back:** on fence waits and wait-idle, the lender sends only the pages of host-cached memory the GPU changed.
  - **Later:** the OS's write tracking (`GetWriteWatch`, or a write-protect fault elsewhere) replaces hashing if hashing costs too much.
- **Borrower's driver:** new crate `vk`, built as `nectar_vk.dll`, `libnectar_vk.so` or `libnectar_vk.dylib`.
  - **What it is:** a Vulkan loader driver (ICD, interface version 5+) whose dispatchable objects carry the loader's header.
  - **Its device:** "<GPU> (Nectar)". The detection already treats that as remote.
  - **While nothing is borrowed,** or `borrow.cfg` has no `vulkan`, it reports **0 devices**.
  - **Registered by the installers:**
    - Windows: `HKLM\SOFTWARE\Khronos\Vulkan\Drivers`.
    - Linux: `/usr/share/vulkan/icd.d/nectar_icd.json`.
    - macOS: `/usr/local/share/vulkan/icd.d`.
- **Lender:** `nectar-server` gets a Vulkan personality behind `HELLO_VULKAN = 2`. This is the deferred split: lease, cache, discovery and control stay shared, and each personality is a module (`server/src/cuda.rs`, `server/src/vulkan.rs`).
  - It loads the system Vulkan loader with `libloading`, and replays recorded streams into real command buffers.
  - **Discovery:** it offers `api=vulkan <version> <origin> <driver>` from `nectar-detect`.
  - **The service:** `BORROWABLE` gains `vulkan` where the driver is installed.
- **Done when** `vulkaninfo` on each borrower lists "NVIDIA GeForce RTX 5070 (Nectar)":
  - Windows over loopback.
  - Linux in WSL, against the desktop and against lavapipe as a software lender.
  - **The Mac over Wi‑Fi:** an ARM64 macOS borrower on an NVIDIA lender, the stress test.

**3.2 Compute: llama.cpp's Vulkan backend, both directions, across vendors and OSes:**
- **The pairs:**
  - The Mac borrows the RTX 5070.
  - The laptop borrows the RTX 5070.
  - The desktop borrows the laptop's **Intel Arc**, which needs the laptop to lend.
  - The Windows laptop borrows the **Mac's Apple GPU** through KosmicKrisp. This needs macOS 26 on the Mac; otherwise MoltenVK.
- **Measured:** tokens per second against native, waits per token, and bytes per token. The output must match native byte for byte with greedy sampling.
- These numbers decide 3.3: each wait for the lender costs about 6 ms over Wi‑Fi.

**3.3 Frames on screen (after go/no-go):**
- **The borrower's driver** implements `VK_KHR_surface`, the platform surfaces (Win32, Xlib/XCB/Wayland, Metal) and `VK_KHR_swapchain` locally. The lender renders into its own images.
- **On present:**
  - **The lender encodes** the frame with **Vulkan Video encode** (H.264/H.265/AV1). That's vendor-neutral; support on driver 621 is *to verify*, with NVENC as the fallback.
  - **The borrower decodes** it with the OS's decoder (Media Foundation, VA-API, VideoToolbox) and draws it into the program's window.
  - **First version:** uncompressed frames at a small size, to get it correct.
- **Fences and acquires** are answered locally where it's safe, so a frame costs about one wait for the lender.
- **Acceptance:** vkcube in a window on the Windows laptop and on the Mac, with fps and input-to-photon latency measured.

**3.4 Games (after go/no-go):**
- **Windows borrower:** DXVK's `d3d11.dll` and `dxgi.dll` beside the game. DXVK is under the zlib license; downloading it gets confirmed. First a Steam Deck Verified Direct3D 11 game with no anti-cheat, then Direct3D 12 through vkd3d-proton (LGPL-2.1), then a native Vulkan game.
- **Linux borrower (once there's a machine):** Steam + Proton with Nectar's driver as the system Vulkan driver, and no change to the game. OpenGL games go through Mesa's Zink (OpenGL on Vulkan).
- **ARM Linux:** Proton ARM64 with FEX, once an ARM Linux device exists. The M4 Mac can't run Asahi.
- **Mac:** compute and vkcube only. Games through Wine on macOS are *to verify*: CrossOver mostly uses Apple's D3DMetal rather than Vulkan.

### Game compatibility data (feeds Phase 4's catalog)
- **One record per Steam app id:**
  - **Valve's rating:** Steam Deck / SteamOS / Steam Frame, entered by hand from store pages, since no documented API was found.
  - **ProtonDB report summary:** from its ODbL export. Downloading it gets confirmed. Credited, and anything we publish from it shares alike.
  - **Anti-cheat status:** from AreWeAntiCheatYet. Its MIT license is *to verify* first.
  - **Renderer:** Direct3D 9/10/11 → DXVK, Direct3D 12 → vkd3d-proton, Vulkan, OpenGL → Zink.
  - **Where it ran through Nectar,** and how fast.
- **Use in 3.4:** pick test games, and leave out every anti-cheat game. Later, `nectar games` says whether a game should work on a borrowed GPU.

### Open hardware: the lender holds drivers (M4 spec addition)
- **What the management MCU (Zephyr) holds:** on its flash or SD card, the **signed driver catalog** and a **package store**:
  - Nectar's borrower packages for each platform: Windows x64, Linux x64/arm64, macOS.
  - The user-mode driver pieces the catalog pins, such as an Arm64 hybrid runtime.
- **What it does with them:** it serves them to borrowers even while the GPU host is off or booting, beside its discovery beacon.
- **Checks:** borrowers verify the catalog signature and each file's SHA-256, as in Phase 4.
- **What it doesn't do:** the MCU never runs GPU drivers. The host's kernel drivers stay in the host's OS image, which the catalog only describes.
- **Licensing:** the appliance fetches vendor files from the vendor's own site for its owner, who accepts the vendor's terms. It never ships with them.
- This becomes part of `hardware/protocols.md`: the MCU's file service, and the control request a borrower uses to fetch a package by key.

### Phase 3 verification
- **Unit tests:**
  - Generated encode and decode round trips for every struct with a `pNext` chain.
  - The page-diff logic.
  - The driver's loader handshake.
  - Zero devices while nothing is borrowed.
- **3.1:** `vulkaninfo --summary` on each borrower, as above, plus Khronos' conformance test suite (dEQP-VK `api.*` and `memory.*` subsets) over loopback on Windows and WSL. Failures are listed, not hidden.
- **3.2:** the llama.cpp numbers above, with results kept like the Wi‑Fi CUDA runs.
- **Regression:** the CUDA probe and llama.cpp still pass on Windows and Linux, and `nectar-detect` still reports the right things.
- **Mac:** everything is run over SSH from this desktop, with logs kept in `build\mac-results\`.

### Progress (2026-11-01)
- **3.0 done:**
  - `vk.xml` and `video.xml` from Vulkan-Headers `vulkan-sdk-1.4.363.0` are in `tools\vulkan-headers-1.4.363.0`.
  - The desktop's Mac key is `~/.ssh/nectar_mac`; the user adds it on the Mac and sends the address.
  - **Rename:** `nectar-cuda-server` became `nectar-server`. `Start-Lender.ps1` prints the command that points the existing "Nectar CUDA server" firewall rule at the new file.
- **3.1 done on Windows and Linux; the Mac waits for SSH:**
  - **New crates:**
    - `vkcodegen`: reads `vk.xml` with `roxmltree`; it has no libclang dependency.
    - `vksys`: 19k lines of types. Layouts were checked against C sizes.
    - `vkwire`: generated tables plus a hand-written interpreter, with 5 round-trip tests.
    - `vk`: the ICD.
    - `vkprobe`: `vk-probe`, with a hand-assembled SPIR-V compute shader.
  - **Commands:** 670 declared. 445 are sent without waiting, 220 wait for an answer, and 5 are waits the lender runs off the ordered stream. A command is sent without waiting only when `VK_SUCCESS` is its only success code. 17 can't be carried, among them descriptor update templates, which DXVK needs.
  - **Lender:** the Vulkan personality is `server/src/vulkan.rs`. CUDA is now optional on a lender, and discovery offers `api=vulkan`.
  - **Tested:**
    - **`vulkaninfo` through Nectar:** "NVIDIA GeForce RTX 5070 (Nectar)". The device section matches native except the video, window-system and two not-carried extensions, plus live memory-budget numbers.
    - **`vk-probe`, 64 MiB:** PASS on Windows through Nectar to the RTX 5070, 41 µs per round trip on loopback.
    - **`vk-probe` on lavapipe:** PASS Linux→Linux (WSL), and PASS Windows→Linux, which crosses OS and vendor through a software driver.
    - **Tests:** 56 pass on Windows and 53 on Linux, and clippy is clean.
    - **CUDA regression:** passes (llama-bench tg16 69 t/s on loopback).
  - **Fixed along the way:**
    - The loader unloaded the driver under its threads, so the driver now pins itself.
    - Commands of extensions that aren't carried are no longer offered, since the loader answers debug callbacks itself.
    - Memory is refreshed only when a fence has signalled.
- **3.2 on loopback done (2026-11-01):**
  - **Setup:** llama.cpp b12030 `win-vulkan-x64` (SHA-256 checked against GitHub's digest) in `tools\llama.cpp-b12030-vulkan`.
  - **Results** (Gemma 4 12B Q4_0, RTX 5070):
    - llama-bench pp512 2743 t/s against 2924 natively (94%), and tg128 57.9 against 68.5 (85%).
    - llama-server chat: byte-identical 506 tokens, 51.6 t/s against 65.0 (79%).
  - **What it took:**
    - **Device memory the CPU can map** (the 5070's 214 MB BAR window) is offered as GPU-only. ggml put buffers there, and the lender read it over PCIe after every wait.
    - **Program writes:** the shadow is page-aligned (an alignment assert in ggml) and write-watched on Windows. Only readable pages are asked about.
    - **GPU writes:** after a wait the shadow's pages go stale instead of being fetched. A touch fetches the stale pages around it, and the lender (`FETCH`) sends the changed ones. ggml's 566 MB staging buffer had cost about 70 ms of hashing per wait.
    - **Leak fix:** the lender frees memory a borrower left behind before destroying its device. A crashed borrower's 6 GB had stayed allocated.
  - **Per token:** 12 fence waits and about 2 fetches (round trips), 21 polls, 27 submits, 4,150 commands, 322 KB up, 63 KB down.
  - **Tests:** 58 pass, and clippy is clean. `vk-probe` passes Windows→Windows and Linux→Linux, and `cuda-probe` still passes.
- **Laptop bundle for Vulkan ready (2026-11-01):**
  - **Installer:** `Install-Borrower.ps1` installs `nectar_vk.dll` and `nectar_icd.json` and registers them with the Vulkan loader under `HKLM\SOFTWARE\Khronos\Vulkan\Drivers`. It creates the key only if missing; `New-Item -Force` would have emptied it.
  - **Service:** svc borrows `vulkan` too (`BORROWABLE`).
  - **Bundle:** `llama.cpp-vulkan\`, plus `vk-probe` and Nectar's Vulkan driver in `nectar\`.
  - **Test script:** `Test-Borrow.ps1 -Api cuda,vulkan` runs the steps for each API. For Vulkan, llama.cpp is pointed at the device named "(Nectar)" with `-dev`.
  - **Dry run on loopback, all PASS:**
    - CUDA: tg 92%, chat identical.
    - Vulkan: pp 92%, tg 84%, chat identical at 78%.
  - **Not tested here:** the installed path (service plus registered driver), since the desktop refuses the installer.
- **3.2 still to do:**
  - **Wi‑Fi:** the user's laptop pass with the bundle.
  - **The Mac:** waits for the Mac's address.
  - **The Intel Arc lender:** the laptop lends.
  - **Linux and macOS borrowers** still hash pages; their write and touch tracking comes later (mprotect or userfaultfd).
  - **If Wi‑Fi shows the cost of the 12 waits per token:** answer waits for copy-only submits on the borrower, with the lender holding later writes to that memory until the copy completes.

### Not in this phase
- ARM64 Linux builds: they wait for the user's apt install.
- The Linux games borrower: decided later.
- Phase 4's signed catalog service.
- HIP, Windows on Arm native, Android.

## DONE: Phases 1–2 of "Nectar for any GPU, on any device" (approved 2026-10-31)

### Context
- **The user's request:** make Nectar general purpose.
  - AMD and Intel GPUs, and maybe Apple M-series.
  - NVIDIA, AMD and Intel GPUs working on ARM devices: macOS, Linux and Android.
  - Detect community drivers where official ones don't exist.
  - A driver-detector service for updates and for checking which libraries to load.
- **Where Nectar is today:** one vertical slice.
  - **API:** the CUDA driver API, so NVIDIA only.
  - **Lender:** Windows x64.
  - **Borrower:** Windows x64.
  - **What's already vendor-neutral:** the transport, discovery, lease, secret, upload cache, service state machine and app logic.
  - **What's tied to Windows:** the code that touches the OS. It sits in a few files: `lending/src/crypto.rs` (BCrypt), the server's keep-alive, the client's `DllMain`, the system-folder check and the hybrid search, the `redirect` crate, svc's pipe and service code, and the Win32 app.
  - **What's portable:** the server loads the driver with `libloading`, which works on every OS. The protocol is little-endian with fixed widths; the CUDA bindings use only `long long`.

### The model: three independent axes
Nectar remotes **APIs, not hardware**. Generalizing means filling in this table one cell at a time.

| Axis | Today | To add |
| --- | --- | --- |
| **API ("personality")** that borrower apps call | CUDA | Vulkan (the cross-vendor one), HIP (AMD apps), OpenCL / Level Zero (Intel and general compute) |
| **Lender platform:** OS × CPU arch × GPU driver | Windows x64 + NVIDIA's official driver | Linux x64/arm64 with official or community drivers; macOS with Vulkan-on-Metal |
| **Borrower platform:** OS × CPU arch | Windows x64 | Linux x64/arm64, Windows ARM64, macOS, Android |

**Rule:** a borrower app gets a lender whose GPU offers the **same API**. Cross-API translation happens only through existing open translators running *on the lender*. Example: ZLUDA, as the lender's "CUDA driver" on an AMD GPU. Nectar never translates APIs itself.

### What's possible where (verified 2026-10-31)

**"A GPU working on an ARM device" means two different things:**
- **Borrowed:** the GPU sits in an x64 PC, and the ARM device uses it over the network. This needs only software, and it is exactly what Nectar does.
- **Physically attached:** the GPU is plugged into an ARM board, which then lends it. This works only on Linux.
  - Apple Silicon Macs don't support eGPUs at all.
  - Android has no dGPU support.

  So on a Mac or a phone, a borrowed GPU is the only way to get an NVIDIA, AMD or Intel GPU.

**Borrower platforms:**

| Borrower | CUDA apps | Vulkan apps | Notes |
| --- | --- | --- | --- |
| Windows x64 | ✅ done | Vulkan personality | |
| Windows ARM64 | x64 apps under Prism emulation: the x64 build *(to verify)*. Native ARM64 apps: CUDA 13.4 + driver 616 add Windows on Arm (N1X/RTX Spark) | Vulkan personality | Native ARM64 needs an ARM64 hybrid runtime from NVIDIA's ARM64 driver package (see the catalog). The `redirect` detour under emulation is *to verify* |
| Linux x64 / arm64 | `libcuda.so.1` shim. NVIDIA ships aarch64 CUDA (SBSA) | Vulkan ICD | No hybrid runtime and no LoadLibrary redirect on Linux |
| macOS (Apple Silicon) | none: CUDA apps don't exist on macOS | Vulkan ICD via the Vulkan loader | Metal apps can't borrow (Metal assumes Apple's GPU). Not planned |
| Android | none | Only in apps that load custom drivers (the libadrenotools pattern, used by emulators). System-wide needs root | Later |

**Lender platforms (GPU + driver → APIs lendable):**

| Lender | Official driver | Community driver (Mesa) |
| --- | --- | --- |
| NVIDIA, Windows | CUDA ✅, Vulkan, OpenCL | none |
| NVIDIA, Linux x64/arm64 | CUDA, Vulkan, OpenCL (arm64 SBSA official). On Pi 5/Rockchip: works with a community kernel patch and 4K pages | NVK: Vulkan 1.4 conformant on RTX 50 since Mesa 25.2. No CUDA |
| AMD, Windows | HIP SDK 7.2 (RDNA3/4), Vulkan, OpenCL | none |
| AMD, Linux | ROCm/HIP, AMDVLK | RADV (Vulkan), rusticl (OpenCL). On Pi 5: Vulkan works with a small cache-coherency patch; no ROCm |
| Intel Arc, Windows | Vulkan, Level Zero, OpenCL | none |
| Intel Arc, Linux | compute-runtime (Level Zero, OpenCL) | ANV (Vulkan, Intel-maintained in Mesa). On Pi 5: partial, with glitches |
| Apple M1/M2, macOS | Metal only | Vulkan via KosmicKrisp (Mesa, Vulkan 1.4 conformant, Metal 4 on macOS 26+) or MoltenVK |
| Apple M1/M2, Asahi Linux | none | Honeykrisp (Vulkan conformant), rusticl. M3 is officially supported, but its GPU isn't usable yet; M4 isn't supported |
| AMD GPU through ZLUDA (lender) | | CUDA apps run on AMD through ZLUDA v6+ (ROCm 7, full llama.cpp support; a hobby project now). NVIDIA's EULA restricts translation layers; that is the user's call |

**So Vulkan is the general-purpose API.** Every vendor, official or community, and every OS, macOS included through Metal, has a conformant Vulkan driver. llama.cpp's Vulkan backend is the obvious first workload, and Vulkan is also the path to games later (DXVK and vkd3d on the borrower).

### Community driver detection: standard, not guesswork
- **Vulkan:** every driver reports `VkPhysicalDeviceDriverProperties`: `driverID`, driver name and info, and **conformance version**. `VkDriverId` (from the public `vk.xml`) names each implementation:
  - NVIDIA proprietary, NVK.
  - AMD proprietary, AMDVLK, RADV.
  - Intel proprietary, ANV.
  - Honeykrisp, Turnip, PanVK.
  - MoltenVK, KosmicKrisp.
  - Software: llvmpipe/lavapipe, Dozen (Vulkan on D3D12).

  A small table maps each one to official, community or software.
- **Other APIs:**
  - CUDA: `cuDriverGetVersion`.
  - HIP: `hipDriverGetVersion`.
  - Level Zero: `zeDriverGetProperties`.
  - OpenCL: platform vendor and version. That covers rusticl, PoCL and the vendors' own.
- **Kernel driver:**
  - Linux: `/sys/class/drm/card*/device/{vendor,device,driver}`, which gives nvidia, nouveau, amdgpu, i915, xe, asahi, msm or panfrost.
  - Windows: SetupAPI over the display class.
- **How libraries are loaded:** all of them through `libloading`, already a workspace dependency. No SDK is needed at build time, and a missing library means "API not offered".

### Driver catalog (the "driver detector service")
- **Local first:** detection needs no network. Each machine knows what it has.
- **Catalog:** a **signed, static catalog**, not a live server. It is one file of entries, published as a release asset of the user's fork. Each entry maps (vendor, device family, OS, arch) to:
  - **Drivers:** the known-good official and community drivers (kernel and user mode), with minimum versions, the APIs each provides, notes ("Pi 5: needs the 4K-page kernel and patch X") and the **official source URL**.
  - **Nectar's own needs per platform:** for example, where the matching `nvcudart_hybrid` comes from for Windows ARM64 or for a Windows borrower of a Linux lender. That is an NVIDIA package URL, the path inside it and a SHA-256.
- **Safety, because this is a supply-chain surface:**
  - Catalog signature: ed25519, with the public key built into Nectar.
  - Downloads: only from an allowlist of vendor and distro domains, and every file is checked against its pinned SHA-256 before it is loaded.
  - User-mode files: these go only into the `%ProgramData%\Nectar\drivers\…` sandbox (or its Linux and macOS equivalents), and only after the user agrees.
  - **Kernel drivers are never installed by Nectar.** It says what to install and where from.
- **`nectar doctor`:** combines the local inventory with the catalog. For example: "RTX 5070 on Linux: proprietary 580 present (CUDA, Vulkan). NVK available with Mesa ≥ 25.2 (Vulkan only)."
- **Checking for updates:** daily, metadata only.
- **Hosting:** where the catalog is published, and when, is the user's decision. That is the outward-facing step.

### Phases

**Approve now: Phases 1 and 2. Nothing else is built without a later approval.**

1. **Vendor-neutral foundation** (protocol VERSION 7):
   - **New crate `detect`** with a `nectar-detect` CLI. It produces an inventory: for each GPU, the vendor and PCI ids, name, VRAM, kernel driver, and its APIs, each with version, implementation, origin (official, community or software) and conformance. It is used by the lender's discovery answer, by the app's Lend screen and by `nectar doctor` later.
   - **`LenderInfo`** gains `vendor`, `os`, `arch` and repeated `api=` lines. They replace `driver`, which becomes the CUDA api line's version.
   - **Hellos name the personality.** `HELLO_CUDA` and `HELLO_CONTROL` stay, and Vulkan and HIP get values later. The borrower's control hello carries its own OS and arch, so the lender sends the hybrid only to a matching platform (Windows x64).
   - **`borrow.cfg`** gains `apis=`. `nvcuda` treats "no `cuda` in apis" as no device.
   - **Server:**
     - `--cuda-driver <path>` loads a chosen CUDA driver library. That's the ZLUDA-on-AMD experiment, for anyone with an AMD GPU.
     - The CUDA handling becomes one personality behind the shared core: listener, admission, lease, cache and control.
2. **Linux port, x64 and arm64** (lender, borrower and service):
   - **Crypto:**
     - **Change:** replace BCrypt with RustCrypto's `sha2`, `hmac` and `getrandom` (MIT/Apache), on every OS. The existing RFC 4231 tests stay.
     - **Why:** well-reviewed crates rather than hand-written SHA-256.
     - **New crates:** these three, plus `libc` for the Unix socket options and peer credentials.
   - **Server:**
     - Platform code goes into `platform/{windows,unix}.rs`: keep-alive, folders (`$XDG_CACHE_HOME/nectar`) and the hybrid (none on Linux).
     - On Linux it loads `libcuda.so.1`.
   - **Client:** `libcuda.so.1` as a cdylib.
     - Windows-only exports (D3D, WGL interop) are cfg-gated, and Linux-only ones are added from codegen run against the Linux headers.
     - It has no hybrid and no redirect: it's installed in a directory listed in `/etc/ld.so.conf.d` ahead of the system's, or used through `LD_LIBRARY_PATH` while testing.
     - **The dark-API export tables:** the same on Linux. ZLUDA supports Linux; *to verify with static cudart*.
   - **Service (`svc`):**
     - **Shared:** the engine and the borrow state machine already are. Only the pipe and the service glue split by platform.
     - **On Linux:**
       - It runs as a systemd unit, `nectar-borrow.service`.
       - The socket is `/run/nectar.sock`, open to the `nectar` group.
       - `/etc/nectar/secret` is root-only, and `/var/lib/nectar/borrow.cfg` is world-readable.
   - **CLI `nectar`** for Linux, and for Windows too: `status`, `list`, `borrow <name>`, `return`, `lend`. A GUI and tray for Linux and macOS come later.
   - **`Install-Borrower.sh`:** the Linux installer, which the user runs with sudo.

**Later phases, each needing its own approval:**

3. **Vulkan personality, compute first.** A borrower Vulkan ICD, registered for good and showing 0 devices when nothing is borrowed (as `nvcuda` does), with a lender-side server.
   - **Codegen** from `vk.xml` (Apache-2.0/MIT), the same way as the CUDA codegen from `cuda.h`.
   - **Design ideas from Mesa Venus / virglrenderer** (MIT; the venus-protocol license is *to verify*):
     - Command buffers are recorded locally and sent at submit.
     - Handle ids are assigned by the client, so creates don't wait.
     - Mapped memory is shadowed on the borrower. Dirty ranges are uploaded at submit, and read-back ranges are downloaded at fence wait. The upload cache is reused.
   - **Acceptance:** llama.cpp's Vulkan backend borrowed over Wi-Fi in both directions. That gives two vendors with no new hardware:
     - Laptop → desktop's RTX 5070.
     - Desktop → laptop's **Intel Arc**.
   - **Driver coverage:** lavapipe in WSL covers a community or software driver.
4. **Signed driver catalog + `nectar doctor`** (as above).
5. **AMD:** a HIP personality (it replaces `amdhip64`, a runtime-level API like cudart), plus ZLUDA on the lender. Needs an AMD GPU.
6. **Apple:** a macOS lender (the Vulkan server on KosmicKrisp or MoltenVK) and a macOS borrower (the Vulkan ICD). Needs a Mac.
7. **Windows ARM64 native:** an ARM64X `nvcuda.dll`, so x64 and ARM64 apps both load it, with the ARM64 hybrid from the catalog. Needs a Snapdragon or N1X laptop.
8. **Android:** a Vulkan driver package for apps that load custom drivers. Needs a phone.
9. **Graphics and presentation (games):** frames come back as video (NVENC, AMF or QSV on the lender). This gets its own design study. Anti-cheat games stay out.

**Hardware that unlocks the later phases (the user's choice):**
- A Raspberry Pi 5: an arm64 Linux borrower, and a lender with an AMD card through the patch.
- Any M-series Mac: an M1 or M2 also allows Asahi.
- An Android phone with an Adreno GPU.
- A Windows on ARM laptop.

### Phases 1–2: how they're tested, without new hardware
- **Unit tests:** `cargo test --workspace` on Windows and in WSL (Ubuntu 24.04, already installed and running). WSL has NVIDIA's `/usr/lib/wsl/lib/libcuda.so.1` plus Mesa's ICDs, including lavapipe.
- **`nectar-detect`:**
  - **Desktop:** RTX 5070, with CUDA 13.5, Vulkan and OpenCL, all official.
  - **WSL:** CUDA through WSL, plus lavapipe listed as software.
  - **Laptop:** Intel Arc, with Vulkan, Level Zero and OpenCL. This runs in the user's next laptop pass.
- **Linux borrower:** `cuda-probe` built for Linux in WSL, using Nectar's `libcuda.so.1`, borrows from the Windows lender on the desktop. The address is given by hand, since UDP broadcast doesn't cross into WSL.
- **Linux lender:** the server runs in WSL on NVIDIA's WSL `libcuda`. The Windows borrower on the desktop runs `cuda-probe` and `llama-bench` through it. That proves a cross-OS lender.
- **ARM64 borrower without ARM hardware:** cross-built for `aarch64-unknown-linux-gnu`. It runs under `qemu-aarch64` in WSL: the unit tests, then `cuda-probe` borrowing the desktop's GPU over TCP.
- **Installs this needs, each confirmed with the user when it comes up:**
  - In WSL: `build-essential`, rustup + stable, `gcc-aarch64-linux-gnu`, `qemu-user`.
  - No changes to Windows itself.
- **Docs:**
  - `docs/general-purpose.md`: this design, with the matrices and sources.
  - The README crate table, plus a "platforms" section.

### Progress (2026-11-01)
- **Phase 1 done:**
  - `detect` + `nectar-detect`. It reports the desktop's CUDA, Vulkan (conformance 1.4.3.3) and OpenCL, plus Nectar's own driver as remote. CUDA is loaded from System32, as programs get it.
  - Discovery format 2 (`NECTAR-LEND 2`), with `vendor`, `os`, `arch` and `api=` lines.
  - `borrow.cfg` `apis=`.
  - `--cuda-driver`. A server refuses Nectar's own driver.
  - The hybrid is fetched only by a Windows borrower of the same OS and CPU.
- **Phase 2 done for x64:**
  - **Crypto:** RustCrypto's hmac, sha2 and getrandom.
  - **Lender:** the server runs on `libcuda.so.1` (Unix keep-alive via libc).
  - **Borrower:** the client is `libcuda.so.1` (soname set; 773 exports; the thread id comes from pthread_self, as in ZLUDA).
  - **Service:** svc runs under systemd with `/run/nectar.sock` (`nectar` group, 0660).
  - **App:** `app` is a window on Windows and the `nectar` command elsewhere.
  - **Installer:** `install/install-borrower.sh` + `nectar-borrow.service`.
  - **Windows-only crates:** `redirect`, `trace` and `run` are gated, so the whole workspace builds on Linux.
- **Tested:**
  - **Unit tests:** 50 pass on Windows and 47 on Linux, and clippy is clean on both.
  - **In WSL:** Linux client ↔ Linux server PASS. The whole Linux flow passed: `nectar lend`, `nectar-svc console`, `nectar borrow`, then the probe from `borrow.cfg`, then return gives NO_DEVICE.
  - **Across OSes:** Windows client → Linux server: probe PASS, and llama-bench ran (tg16 16.7 t/s through the WSL NAT relay).
  - **Windows regression:** the loopback borrow passed, with the hybrid fetched.
  - **Laptop bundle:** rebuilt, because discovery changed, and now carries `nectar-detect`. The test saves `detect.txt`.
- **Deviations:**
  - The TCP protocol stays VERSION 6, since nothing on it changed; discovery went to format 2.
  - The CUDA "personality split" in the server waits for the Vulkan personality, since one personality doesn't need it.
  - On Linux, the secret lives in `/var/lib/nectar` with mode 0600 (not `/etc/nectar`).
  - The CLI is Unix-only; Windows keeps the window.
- **Not done:**
  - arm64 cross-build and qemu run. They need the user's `sudo apt-get install gcc-aarch64-linux-gnu qemu-user`.
  - `install-borrower.sh` was not run. It refuses in WSL, correctly, since NVIDIA's `libcuda` is there.
- **The user's proposed test, an Xbox-app game:** it needs Direct3D 12 through a WDDM driver, so it's the acceptance test for the graphics track. It's recorded in `docs/general-purpose.md`, and the nearer CUDA test is an unmodified installed program.

### Open questions recorded, not blocking
- Can a Windows borrower use a hybrid runtime whose driver version differs from a Linux lender's? The hybrid is keyed by CUDA version. If it can't, the catalog supplies the matching Windows package.
- Under Prism, can x64 apps on Windows ARM64 load an x64 `nvcuda.dll` from System32, and does retour's detour work there? Needs a device.

## DONE: the lend/borrow app for Nectar CUDA (approved scope, 2026-10-31)

### Context
- **Where things stand:** Nectar CUDA works over Wi‑Fi. Generation runs at 31% of native, the chat answer is byte-identical, and the lender caches the model. But every use is a CLI launch through `nectar-cuda-run`.
- **The user's vision:**
  - The laptop app finds a desktop GPU that's ready to lend.
  - One click installs a pseudo driver and connects.
  - Applications then find the GPU natively.
  - If the connection drops, the driver disconnects and the app shows "no host available" until the desktop is ready again.
- **Decisions:**
  - Apps see the GPU. Task Manager presence is a design study, since a WDDM driver needs test-signing (ruled out) or Microsoft signing.
  - Rust with plain Win32 (`windows-sys`, plus Windows' own BCrypt for HMAC; no new crates).
  - Borrowing is CUDA only. `borrow.cfg` is defined for `nvcuda.dll`, and NectarVlk will read its own config later.
  - One borrower at a time.
  - Nothing of NVIDIA's goes in System32.
  - A shared secret guards lending.
  - **Later:** other operating systems, and the open hardware lender with a microcontroller.

### How it works
- **Desktop (lender):**
  - `nectar.exe` → **Lend** (the secret is entered in the window) starts `nectar-cuda-server --lend` as a child process.
  - The server answers discovery and holds an exclusive lease for one borrower.
  - The window shows the GPU, its state (ready / lent to <laptop> / grace), the borrower's calls/s and MB/s, and the server log.
- **Laptop (borrower):**
  - **`nectar.exe`:**
    - Lists the lenders it finds, with GPU, VRAM, state and round-trip time.
    - **Borrow** / **Return**, a status line and a tray icon.
    - A live readout of round-trip time and traffic (from the heartbeat).
    - Talks to the service over `\\.\pipe\nectar`.
  - **`nectar-svc` (Windows service, LocalSystem):** does the privileged parts.
    - **On borrow:**
      - Authenticates the control connection with the secret, and receives a lease token.
      - Fetches the lender's `nvcudart_hybrid64.dll` into `%ProgramData%\Nectar\drivers\<lender driver version>\` if it isn't there yet.
      - Writes `%ProgramData%\Nectar\borrow.cfg` (`server`, `token`, `driver`).
      - Heartbeats every 1 s.
    - **Heartbeats missed:** amber "reconnecting" while it tries again. `borrow.cfg` stays during the grace period (20 s), so running and new apps keep working if the link returns. After that it removes `borrow.cfg` and goes grey: "no host available".
    - **When the lender answers discovery again,** it re-borrows on its own.
    - **On return:** releases the lease and removes the file.
  - **The pseudo driver:** only Nectar's own `nvcuda.dll` goes in `System32`.
    - NVIDIA's hybrid runtime stays in Nectar's sandbox, matched exactly to the lender's driver version.
    - The installer still refuses where NVIDIA's own `nvcuda.dll` is in System32, since it would replace it.
    - **Not borrowed:** `cuInit` answers `CUDA_ERROR_NO_DEVICE` at once, without the network.
    - **Borrowed:** every CUDA app uses the lender with no launcher.
    - **Lender lost mid-run:** calls return `DEVICE_UNAVAILABLE`, as today.

### Steps
1. **Lending protocol** (new crate `native/nectar-cuda/lending`; server changes in `server/src/main.rs`; protocol VERSION 6).
   - **Secret:** HMAC-SHA256 via Windows BCrypt (`Win32_Security_Cryptography`), tested against RFC 4231 vectors.
     - **Discovery:** the borrower broadcasts on UDP 43211 every 2 s with `NECTAR-LEND 1`, a nonce, a timestamp and an HMAC. A lender answers only a valid probe, with key=value lines (name, gpu, vram, driver, port, state, borrower) plus its own HMAC over the borrower's nonce, so neither side trusts an impostor. Replies answer an outbound packet, so the laptop needs no firewall rule; the lender's rule adds UDP 43211.
     - **Every hello** (control and CUDA): the lender sends a random challenge.
       - A control hello answers it with an HMAC under the secret.
       - A CUDA hello answers it with an HMAC under the lease token.
       - Without a valid answer the connection is refused.
     - **Deferred, noted in the plan:** the CUDA stream after the hello stays unencrypted and unauthenticated.
   - **Lease:**
     - **Taking it:** a control hello with the secret takes the lease and gets a random token.
     - **While held:** other borrowers are refused with the holder's name.
     - **Release:** only when heartbeats are missed **and** the holder has no established control or CUDA connection, for 30 s (longer than the borrower's 20 s grace, so it can reclaim). It is also released on Return.
   - **Heartbeat:** 1 s each way. The lender's beat carries calls/s, MB/s each way and the RTT it saw.
   - **Hybrid:** the control connection can fetch the lender's `nvcudart_hybrid64.dll` along with its version.
   - **Unit tests with a fake clock:**
     - Discovery encode/decode and HMAC (including wrong secret and replayed or stale timestamp).
     - Lease rules.
     - Grace with live connections.
     - Grace with dead connections.
     - Reconnect during grace.
     - A second borrower refused.
2. **nvcuda.dll as a system driver** (`client/src/lib.rs`).
   - **Which lender:** `NECTAR_CUDA_SERVER` (dev/testing), otherwise `%ProgramData%\Nectar\borrow.cfg` (server + token + driver version), otherwise no device at once.
   - **Hybrid search:** `%ProgramData%\Nectar\drivers\<version>\` first; the rest of today's order stays.
   - **Refused (lent to someone else):** `NO_DEVICE`, with a log line naming the holder.
   - Skip the `LoadLibrary` redirect when the DLL is itself `System32\nvcuda.dll`.
3. **`nectar-svc`** (new crate `native/nectar-cuda/svc`; raw service API from `windows-sys` `Win32_System_Services`).
   - **Commands** `install`, `uninstall` and `run`, plus a console mode for testing on the desktop.
   - **Pipe protocol** in JSON lines: `lenders`, `borrow <id>`, `return`, `status` and events.
   - **State machine:** idle → borrowing → borrowed ⇄ grace (amber) → lost (grey) → re-borrow when the lender is back. Tests use a fake clock.
   - **On start:** removes a leftover `borrow.cfg` (crash safety).
   - **Secret** stored in `%ProgramData%\Nectar\secret`, readable only by SYSTEM and Administrators.
4. **`nectar.exe`** (new crate `native/nectar-cuda/app`, Win32 window + ListView + tray icon).
   - **Borrow mode:**
     - Lenders list, Borrow/Return and a manual address field.
     - Status: green "Using RTX 5070 on DESKTOP", amber "Reconnecting…" during grace, grey "No host available".
     - Live RTT and traffic. Per-token rates aren't visible to the service, which only sees connections; per app they stay in the app's own output.
   - **Lend mode:** enter the secret, start/stop lending, and show the borrower and its traffic.
   - Writes nothing privileged itself.
5. **Installers.** I write them; the user runs them as administrator, and nothing is installed by me.
   - **`Install-Borrower.ps1 -Secret <s>` (laptop):**
     - Copies the app and service to `C:\Program Files\Nectar`.
     - Puts only Nectar's `nvcuda.dll` in System32.
     - Stores the secret, registers and starts the service, and adds a Start-menu shortcut.
     - `-Uninstall` reverses all of it, including the sandbox.
   - **Lend mode / `Start-Lender.ps1`:** the printed firewall rule also covers UDP 43211.
   - A bundle script produces the laptop package (no NVIDIA files in it any more).
6. **Laptop pass (user),** then the extra tests:
   - A 2 s Wi‑Fi drop stays green or amber through grace and recovers.
   - First-prompt timing: about 80 round trips, plus model load of about 95 s over the wire or 11 s from cache.
7. **Desktop Ethernet (hardware, parallel):** a USB or PCIe GbE adapter halves the 1 MiB logits wait.
8. **Device Manager entry (experimental, after the laptop pass).**
   - While borrowed, the service creates a software device "NVIDIA GeForce RTX 5070 (Nectar)" with `SwDeviceCreate` and removes it on return or loss.
   - Without a signed INF it may show under "Other devices" with a warning. Report what it looks like and keep it only if it's clean.
9. **Task Manager: design study** (`docs/driver/wddm-render-driver.md`, no code).
   - What the GPU list in Task Manager needs: a WDDM adapter with engines and memory segments (DXGK); IddCx display-only adapters don't show as GPUs.
   - A render-only WDDM miniport design: utilization and VRAM reported from the lender, work still remoted at the API level (WiCi measured kernel-level forwarding as 51× slower).
   - Signing path: EV certificate, attestation, HLK.
   - Effort estimate and a go/no-go.

### Progress (2026-10-31)
- **Done, tested on the desktop (both roles, loopback):** steps 1–5 and 9.
  - **Crates:** `lending`, `svc` and `app` added. The server has `--lend`, `nvcuda.dll` reads `borrow.cfg`, and VERSION is 6.
  - **Static C runtime:** every binary links it (`.cargo/config.toml`, Build-All runs cargo from the workspace folder).
  - **Laptop tooling:** `install/Install-Borrower.ps1`. `Start-Lender.ps1` checks the firewall (TCP 43212 + UDP 43211) and opens the app. The bundle has no NVIDIA or VC files in `nectar\`. `Test-Borrow` runs programs directly when Nectar is installed.
  - **Passed:**
    - Discovery with HMAC both ways.
    - Borrow, with the token and the hybrid arriving in the sandbox.
    - `cuda-probe` and llama.cpp using `borrow.cfg` only.
    - A wrong token refused with the holder's name.
    - A 5 s outage: amber, then recovered.
    - A 40 s outage: grey, `borrow.cfg` removed, programs see no device, then re-borrowed.
    - Return releases the lease.
    - The app driven by control ID: Lend start, borrow, "Lent to…", return.
  - 43 unit tests; clippy clean.
  - The pipe protocol is key=value lines rather than JSON (no serde).
- **Not tested here:** the installer (it refuses on the desktop, which has NVIDIA's driver) and the service under SCM (only console mode). These need the laptop pass (step 6).
- **Next:** the laptop pass (user). Then the Device Manager experiment (step 8) and Ethernet on the desktop (step 7).

### Verification
- **Unit tests:** cargo tests for discovery messages and HMAC, lease and grace rules (fake clock), `borrow.cfg` parsing, and the service state machine.
- **Desktop, both roles on one machine** (no System32 install, since it has NVIDIA's driver; CUDA only):
  - **Setup:** Lend mode in one window, the service in console mode with a test `ProgramData` folder.
  - **Discovery:** the app lists the desktop, and a wrong secret lists nothing.
  - **Borrow:** writes `borrow.cfg` and fills the sandbox. Then `cuda-probe` and llama.cpp through `nectar-cuda-run` find the GPU, using the token.
  - **Drops:** stopping the server for 5 s stays amber and recovers. Stopping it for 40 s goes grey, removes the file, then re-borrows when it's back.
  - **Exclusivity:** a second borrower is refused, and Return releases the lease.
- **Laptop, run by the user:**
  - `Install-Borrower.ps1 -Secret` as admin, then Borrow in the app.
  - `cuda-probe` and llama.cpp run with no launcher.
  - The 2 s drop test, and a long drop that goes "no host available" with apps reporting no device, then recovering.
  - First-prompt timing, then `-Uninstall`.

---

## Background

### Context (original)

The baseline works: the 2023 Juice runtime renders vkcube and the Quake II remaster on the desktop's RTX 5070, and vkcube also ran over Wi‑Fi. Right now every launch is a CLI command.

The goal is a **lending** model:
- **Desktop:** the Nectar app *lends* its RTX 5070.
- **Laptop:** the Nectar app *borrows* it. Windows then treats it like an installed GPU until it is returned.
- **Later:** the same lender side must run on a dedicated OCuLink appliance, with an open hardware spec (off-the-shelf boards like LattePanda-class modules, plus a custom PCB with OCuLink, Wi‑Fi 6 and/or GbE, and an RTOS).

### What "Windows sees it as an installed GPU" takes

Each level is more of an "installed GPU", and much more work:

| Level | What Windows/apps see | Needs | Milestone |
| --- | --- | --- | --- |
| 1. System Vulkan driver | Every Vulkan app lists "Juice GPU [RTX 5070]" while borrowed | Admin installer + service writing HKLM registry. No signing | M1 |
| 2. Device Manager presence | "RTX 5070 (Nectar)" under Display adapters, plus an optional virtual monitor | User-mode IddCx driver; test-signing during development, attestation signing to ship | M2a |
| 3. Real WDDM render GPU | DirectX/DXGI, Task Manager and every app see a real adapter | Kernel display miniport + D3D user-mode driver that forwards over the network. Multi-month or longer research; EV cert + Microsoft attestation | M2b (design only) |

Until level 3 exists, DirectX, OpenGL and CUDA apps run on the borrowed GPU through one-click launch: Juice's proxy DLLs via `launch.exe`. This path already works.

### Code facts the design relies on

- **One session = one borrowed GPU.** It serves many apps (`cmd/agent/app/session.go`: one `Renderer_Win` per connecting process) and lives until `DELETE /v1/session/{id}`.
- **How an app is wired to a session:** env vars `VK_DRIVER_FILES` pointing at `JuiceVlk.json`, plus `JUICE_CFG_OVERRIDE` holding `{id, servers, requirements}`, with the app started via `launch.exe` (`cmd/juicify/app/juicify.go:321-334`).
- **The agent is our Go code**, so endpoints and discovery can be added there. Only `Renderer_Win`/`Renderer_Linux` and the client DLLs are closed, and they're x86_64 only.
- **No MinGW needed:** juicify's only cgo is `version_windows.h` (`LoadLibrary`/`GetProcAddress`), which can become pure Go.

---

## M1: Nectar app with Lend/Borrow (build this first)

### Step 0: Prerequisites and feasibility checks
1. Install Go (`winget install GoLang.Go`) and Inno Setup (`winget install JRSoftware.InnoSetup`). I'll confirm with you before installing.
2. Answer four questions by hand on the desktop. Get a session with `curl -X POST /v1/request/session`, then run vkcube directly with env vars and no `launch.exe`.
   - **A:** Does the Vulkan driver work without `launch.exe`?
   - **B:** Can two apps share one session?
   - **C:** Does `JuiceVlk.dll` fail harmlessly when there's no session or the lender can't be reached? It must show zero devices and must not hang or crash.
   - **D:** Does the DLL read the session `id` from `juice.cfg` next to it, or only from the `JUICE_CFG_OVERRIDE` env var?

   The answers decide how level 1 is wired (see Borrow mode). If A or C fails, level 1 is dropped and Vulkan also uses one-click launch.

   **Results (2026-10-31, 2023.08.10 runtime, trimmed agent):**
   - **A passed:** vkcube run directly, with only the env vars, selects "JUICE GPU".
   - **B passed:** two vkcube processes run at once, each with its own `Renderer_Win`.
   - **C failed:** on any server error, `JuiceVlk.dll` logs `F]` and aborts the app (exit 0xC0000409). It does not return a Vulkan error the loader could skip.
   - **D:** the DLL has no `id` key at all. Every process requests its own session from `servers`, read from `JUICE_CFG_OVERRIDE`, from the file named by `JUICE_CFG_PATH`, or from `juice.cfg` beside the DLL. It never sends DELETE.
   - **So:** a borrow cannot be a held Juice session. It is a lease the lender's agent enforces, plus a service-written config that names the lender. Level 1 needs either a guard ICD in front of `JuiceVlk.dll` (C) or is dropped. That decision is pending with the user.
   - **Agent compatibility fixes this needed:**
     - The session JSON keeps the `address` key.
     - A connect with no body is accepted, and its connection id becomes the session id.
     - The session ends when that connection's `Renderer_Win` exits.
   - **Config-key tests (`tests/Test-JuiceConfig.ps1`, 2026-10-31):**
     - Each app opens 4 connections to one `Renderer_Win`, which exits about 15 s after the app closes; the session then closes.
     - `id`, `activePaths`, `excludedPaths` (5 path forms) and `persistent` changed nothing for the Vulkan driver. The path keys may only affect `launch.exe` injection, which is untested.
     - An unreachable server aborts the app after about 2 s of connect retries; the guard's 250 ms check prevents this.
     - Losing `Renderer_Win` or the agent mid-run freezes the app instead of aborting it. The client logs `Server disconnected`. Nectar should offer to end apps that have `JuiceVlk.dll` loaded when the lender is lost.

### Step 1: Shared packages
- **`pkg/juiceclient/`**: moved out of juicify.
  - `Configuration` struct; `RequestSession`/`WaitForSession`/`CancelSession`.
  - Pure-Go `GetJuiceVersion`.
  - `Launch(exe, args)`: runs via `launch.exe` with the working directory set to the game's own folder (the Quake lesson), inside a `pkg/winjob` job object.
  - juicify switches to this package and drops cgo.
- **`pkg/lending/`**: the extension point for future hardware.
  - **`LenderInfo` record:** name, id, version, GPUs (name, VRAM, PCI bus), **link** (`pcie` | `oculink` | `thunderbolt`), **network** (`ethernet` | `wifi6` | `wifi6e` | `wifi7`, link speed, RSSI), **backend** (`juice`; open backends can be added later), and state (`available` | `lent` | `busy`).
  - **Discovery:** a UDP broadcast beacon on port 43211, sent every 1 s as compact JSON, plus a listener. It was chosen over mDNS so a small RTOS management MCU with lwIP could also send or answer it.
- **Agent changes (`cmd/agent/app/`):**
  - New `GET /v1/lender` endpoint returning `LenderInfo`.
  - New `GET /v1/sessions` endpoint returning the borrowers.
  - Beacon sending, behind a `--discoverable` flag.
  - Add `TCP_NODELAY` to accepted connections.

### Step 2: Background service `nectar-svc` (`golang.org/x/sys/windows/svc`, runs as LocalSystem)
- **Lend mode:**
  - Supervises `agent.exe --discoverable` from the bundled server runtime and restarts it if it crashes.
  - Adds a Windows Firewall rule for the agent port on Private networks only.
  - Can accept borrowers automatically or ask in the tray.
  - "Stop lending" cancels every session.
- **Borrow mode:**
  - Listens for beacons and keeps a list of lenders.
  - **Borrow:** take a lease on the lender (new agent endpoint), then heartbeat every 2 s with RTT and jitter. After 3 missed heartbeats, reconnect with backoff. While leased, the agent accepts `/v1/request/session` only from the lease holder, and each app gets its own session (check D).
  - **Return:** release the lease, which cancels that borrower's sessions.
  - **Make the GPU visible (level 1).** The service writes and removes these itself, so no admin prompt is needed at borrow time:
    - `HKLM\SOFTWARE\Khronos\Vulkan\Drivers\<...>\JuiceVlk.json` while a GPU is borrowed, and removed when it's returned.
    - The lender's address, in the `juice.cfg` `servers` beside the installed `JuiceVlk.dll` (or in a file a guard ICD names through `JUICE_CFG_PATH`). It is read when an app loads the driver, so apps started before the borrow, including games Steam starts, pick it up with no env var.
  - **Crash safety:** at service start, it removes leftover registry and config entries and releases the last lease recorded in `%ProgramData%\Nectar\state.json`.
  - **Optional** "always borrow from <lender>" setting for a machine that is always on.
- **IPC:** a named pipe `\\.\pipe\nectar` (`github.com/Microsoft/go-winio`) carrying JSON commands and events for the UI.

### Step 3: UI `nectar.exe`
- **UI library:** `github.com/tailscale/walk`, a pure-Go Win32 library with a tray icon and windows. Built with `-H=windowsgui`, with the icon and manifest from `go-winres`.
- **Tray icon:** grey (idle), green (lending or borrowed), amber (reconnecting).
- **Window, Lend tab:** shows the GPU and VRAM, a Lend/Stop toggle, the current borrowers, and an auto-accept checkbox.
- **Window, Borrow tab:**
  - Discovered lenders, each with GPU, link, network and RTT, plus a manual-IP field.
  - Borrow/Return buttons.
  - Live latency.
  - A "My apps" list with one-click launch.
  - Checkboxes for "Run on borrowed GPU" in the Explorer right-click menu and auto-borrow at login.
- **`nectar.exe run <exe> [args]`:** used by the Explorer menu and Steam launch options (`"C:\Program Files\Nectar\nectar.exe" run %command%`). It gets the session from the service and launches through `Launch`.

### Step 4: Installer `installer/nectar.iss` (Inno Setup)
- Installs to `C:\Program Files\Nectar\`: `nectar.exe`, `nectar-svc.exe`, `NectarVlk.dll`/`.json`, and `runtime\client` + `runtime\server` (the Juice binaries).
- **Juice licensing (`runtime\client\TERMS.md`):**
  - Personal, non-commercial use only.
  - No distributing any Juice component to third parties.
  - No disassembling, decompiling or reverse engineering, except where the law overrides that.
  - So a published `NectarSetup.exe` must not bundle the Juice binaries. Instead, it downloads the official Juice release at install time, and the user accepts Juice's terms there. Bundling is fine only for the user's own machines.
- Registers and starts the service and adds a Start-menu entry.
- First launch asks "Lend or Borrow?" (it can be changed later).
- The uninstaller stops the service and removes every registry, firewall and env entry.
- **Build scripts:** `Build-All.ps1` gains `nectar` and `nectar-svc`, and a new `Package.ps1` produces `dist\NectarSetup.exe`.

---

## M2: Nectar CUDA, open CUDA remoting for AI (design from WiCi, arXiv 2608.24204)

**Why:** AI on the RTX 5070 is blocked. Juice's closed `juicejuda.dll`/`cuda_re.cpp` reverse-engineers internal `CUmodule` layouts and fails on the driver's `CUDA version=13050`. WiCi (Shen et al., Aug 2026) shows a better approach for exactly our setup (consumer GPU, Wi‑Fi 6, nearby client). It intercepts at the **CUDA driver API** and **parses fatbins** for kernel names and parameters instead of relying on driver internals, so it doesn't break on new CUDA versions. WiCi has no published code, so we implement the design ourselves, in the open. Graphics stays on Juice; WiCi lists graphics as future work.

- **Where:** `native/nectar-cuda/`, a Rust workspace (the user chose Rust): `protocol`, `server` (lender), `client` (the borrower's `nvcuda.dll`) and `probe` (`cuda-probe`). `Build-All.ps1` builds it into `build\nectar-cuda\`.
- **Tools installed:**
  - Rust 1.99.1 (rustup, MSVC).
  - The CUDA 13.4.1 parts unpacked to `C:\Users\Sriram\tools\cuda-13.4.1`, from NVIDIA's official zips, with checksums verified: cudart (`cuda.h`), crt, nvcc, libnvvm, cccl and nvml_dev. The full installer was avoided because it bundles a display driver. The lender's driver is 621.01 (CUDA 13.5).
- **Clean-room rule:**
  - Built only from public headers and documentation (`cuda.h`, `nvml.h`, `fatbinary_section.h`, the Vulkan spec) and from open-source projects, used under their licenses: ZLUDA (Rust, Apache-2.0/MIT) for the `nvcuda.dll` replacement and `cuGetExportTable`, Mesa Venus for Vulkan serialization, and DXVK for D3D → Vulkan.
  - No disassembling or porting code from NVIDIA or Juice binaries; both licenses forbid it, and it would taint the open-source code.
  - Observing public API calls is allowed. A pass-through tracing `nvcuda.dll` records the call sequences that WiCi-style replay and batching need.
  - Kernel parameter layouts come from `cuFuncGetParamInfo` (CUDA 12.4+) on the lender, cached per function, instead of reading driver internals.
- **Staying current as NVIDIA changes CUDA:**
  - Implement only the driver API (`cuda.h`), which NVIDIA keeps backward compatible through versioned symbols (`_v2`, and `cuGetProcAddress` with a version). Never depend on driver internals; that dependency is what broke Juice at CUDA 13050.
  - Generate the exports, the `cuGetProcAddress` symbol and version map, and the marshalling from `cuda.h`. A small override file gives pointer direction and size where the header can't.
  - A function that isn't supported yet returns `CUDA_ERROR_NOT_SUPPORTED` and logs its name, instead of crashing.
  - The lender resolves every call by name and version through the real driver's `cuGetProcAddress`, so driver updates on the lender need no Nectar change.
  - Keep `cuGetExportTable` (cudart's undocumented tables), the one fragile area, in its own module, with ZLUDA's open-source implementation as the reference.
  - CI, with a self-hosted runner on the desktop: when a new CUDA toolkit or driver appears, regenerate from the new header, build, and run CUDA samples, PyTorch and llama.cpp both locally and borrowed. Report new functions and failures.
  - Expect small regenerate-and-test updates a few times a year. New CUDA features with new semantics still need hand-written support.
  - `client/`: a replacement `nvcuda.dll` on the borrower that exports the `cu*` driver API. The export table and stubs are generated from `cuda.h`, so coverage tracks the header, not hand-written code.
  - `server/`: a lender daemon calling the real driver, one thread per borrower on the native context. It plugs into `LenderInfo.backend` as `nectar-cuda`, beside `juice`.
  - `protocol/`: a TCP framing header (fragment info + `normal`/`batched` payload type). Large transfers such as >500 MB tensors are split into fragments.
- **Latency techniques, in WiCi's order of impact:**
  1. **Trace replay:** repeated call sequences, such as per-token decode, are recorded on both sides. A replay sends only the first call and its arguments, and falls back on mismatch. WiCi measured +52% token rate.
  2. **Pattern batching:** several calls per round trip.
  3. **Speculative local success:** for calls whose outcome is predictable.
  4. **Content-hash weight cache on the lender:** model tensors keyed by hash on disk, so reloading a model only sends what's missing. WiCi measured −23 to −36% load time.
- **Launch integration:** `Launch()` starts CUDA apps through `nectar-cuda-run`, which loads the client `nvcuda.dll` into the app before it runs. The DLL search path does not work, because cudart loads from System32 only. The Borrow tab gets an "AI app" type (llama.cpp, Python/PyTorch).
- **Acceptance:**
  - llama.cpp CUDA build, Qwen3‑8B GGUF, borrowed over Wi‑Fi from the RTX 5070. Target ≥ 30% of native token rate, which is WiCi's measured 0.35 range.
  - Then the PyTorch matmul one-liner that fails today.
- **Order:** Step 0 is a tiny spike: forward `cuInit`, `cuDeviceGet*` and `cuMemAlloc` so `deviceQuery` works remotely. Then fatbin parsing + `cuLaunchKernel` → llama.cpp → batching/replay/cache. This milestone is independent of M1's UI, so it can run in its own session.
- **Step 0 done (2026-10-31):**
  - 19 hand-written functions: init, version, error names, devices, primary context, memory info, alloc/free, and HtoD/DtoH copies.
  - Function ids are FNV-1a hashes of the versioned symbol. Handles pass through unchanged.
  - The server frees whatever a borrower leaves behind when it disconnects.
  - `cuda-probe` over loopback shows the RTX 5070 and CUDA 13.5, a 256 MiB round trip matches, 30 µs per call, about 1.6 GB/s up and 1.0 GB/s down. The real driver directly: 10.9 and 4.2 GB/s.
  - With no lender, `cuInit` returns `CUDA_ERROR_NO_DEVICE` after the 2 s connect timeout. A lender lost mid-run gives `CUDA_ERROR_DEVICE_UNAVAILABLE`, so the app exits instead of freezing.
  - Not yet:
    - Borrowers share one server process (isolation).
    - No authentication. The server listens on 0.0.0.0 by default; use `--listen 127.0.0.1:…` until a PSK exists.
    - One connection per process, with calls serialized.
- **Step 1 done (2026-10-31):**
  - **Generator:** `codegen` runs bindgen on `cudaTypedefs.h` (with `__CUDA_API_VERSION_INTERNAL`, which also includes `cuda.h`) through libclang 18.1.1 from PyPI, unpacked to `C:\Users\Sriram\tools\libclang`. It generates sys types, ids, client exports, server dispatch and the tracer, and writes `codegen/report.md`.
  - **Coverage of the 773 driver exports:** 417 generated, 11 written by hand, 253 unsupported with a reason, 92 driver internals not in `cuda.h`.
  - **`cuGetProcAddress`:** the real driver returns internal addresses rather than its exports. So 681 of the 682 `PFN_<symbol>_v<version>` variants in `cudaTypedefs.h` are matched to exports by signature, and resolution is local (`protocol::proc_address`). The one unmatched is `cuDeviceGetNvSciSyncAttributes`, which is Linux-only.
  - **Launcher:** cudart loads `nvcuda.dll` from System32 only, so copying it next to the app does nothing. `nectar-cuda-run` starts the app suspended and loads Nectar's DLL first (CreateRemoteThread + LoadLibraryW, no hooks). This replaces the earlier assumption under "Launch integration".
  - **Trace of `tests/runtime_probe.cu` (static cudart 13.4):**
    - 660 calls to 27 functions, 509 of them `cuGetProcAddress` lookups.
    - Kernels go through `cuLibraryLoadData` → `cuLibraryGetKernel` → `cuLaunchKernel`.
    - 9 `cuGetExportTable` UUIDs: dd9e98a4…, f8cff951…, 6bd5fb6c…, a094798c…, 42d85a81…, 21318c60…, c693336e…, 263e8860…, d4082055….
  - **Through Nectar:** cudart stops at the first export table with error 35 (insufficient driver). Driver-API programs (`cuda-probe`) work.
- **Step 2 done (2026-10-31):**
  - **Export tables** (`client/src/dark_api.rs` and `integrity.rs`): the ten tables ZLUDA documents, ported from ZLUDA 08b19b2 under MIT, credited in `LICENSE-ZLUDA`. The integrity check is HMAC-MD2 with a fixed key, and ZLUDA's 5 test vectors pass.
  - **Unknown tables:** `f8cff951…` gets `CUDA_ERROR_INVALID_VALUE`, as ZLUDA does, and the runtime tolerates that. `21318c60…` it does not.
  - **GPU code and launches:**
    - Modules and libraries are sent as bytes, sized from the fatbin wrapper or fatbin header, ELF section ends, or the NUL ending PTX.
    - `cuLaunchKernel` (and `_ptsz`) copies arguments using sizes from the lender. The lender queries `cuKernelGetParamInfo` for handles that came from `cuLibraryGetKernel` and `cuFuncGetParamInfo` otherwise, and the client caches the sizes.
    - `cuDeviceGetLuid` written by hand.
  - **Verified:** `cuda-probe` through Nectar runs a PTX vector add as a module and as a library, 1M elements correct, about 1 ms per launch plus sync over loopback.
  - **Hybrid runtime:** `nvcudart_hybrid64.dll` ships only with NVIDIA's display driver, not the toolkit. Nectar looks in `NECTAR_CUDA_HYBRID`, next to its DLL, System32, then the DriverStore. ZLUDA redistributes NVIDIA's 13.0.94 copy; Nectar should not.
  - **Runtime programs** (`runtime_probe.cu`) with the driver's 13.5.40 hybrid get through init, `cuDriverGetVersion`, device queries and the LUID, then stop at `21318c60…` with cudaErrorInvalidValue. No public source documents it.
- **Step 3 done (2026-10-31): the CUDA runtime works through Nectar.**
  - **Black-box table tracer** (`trace/src/tables.rs`):
    - Naked-asm trampolines, 512 slots, 16 bytes apart. Each saves rcx/rdx/r8/r9 and xmm0–3, copies 16 stack arguments, and records the args, 32-byte snapshots of the memory they point to before and after, and the result.
    - Unit tests cover a 6-argument call through a thunk.
    - `NECTAR_CUDA_TRACE_TABLES=unknown|all|none`. `CUDART_INTERFACE` and `INTEGRITY_CHECK` are never wrapped, because their addresses are hashed.
  - **Findings on the real driver 13.5 with `runtime_probe.cu`:**
    - `21318c60…` has 100 entries (size 800, entry 91 NULL) and `f8cff951…` has 3. The runtime asks for both but calls no entry.
    - The calls observed match ZLUDA's descriptions: context checks [2]/[3] return 0; the tools buffers have 1024 and 14 entries; the hybrid token is 0x1201aea0.
    - Context-storage `get` with no current context returns 201 (INVALID_CONTEXT); Nectar now does the same.
    - The real context-storage table has 29 entries and the context-checks table 17, more than ZLUDA's.
  - **Nectar:** stub tables with the real layouts (naked-asm stubs that log their first call and return NOT_SUPPORTED). At process exit, WSANOTINITIALISED now gives `CUDA_ERROR_DEINITIALIZED`, quietly.
  - **Verified:** `runtime_probe.cu` (static cudart 13.4) through Nectar computes correctly and exits cleanly in 0.20 s, against 0.15 s natively. With no lender it fails cleanly (error 35). `cuda-probe` still passes.
- **Step 4, PyTorch done (2026-10-31):**
  - **Setup:** PyTorch 2.7.1+cu118 in the Store Python 3.13. It has no sm_120 kernels, but the driver JIT-compiles its PTX, so it runs natively. `tests/torch_probe.py` runs a cuBLAS matmul and an elementwise kernel and checks both against the CPU.
  - **Through Nectar:** PASS in 0.23 s over loopback, against 0.28 s natively, about 760 calls. Without a lender, `is_available()` is False.
  - **What it took:**
    - **Async host↔device copies** (`cuMemcpyHtoDAsync_v2` / `DtoHAsync_v2` and `_ptsz`): copied on the lender's stream, then the stream is synchronized.
    - **Host memory** (`cuMemHostAlloc` / `AllocHost` / `FreeHost` / `GetFlags` / `HostRegister` / `Unregister`): page-aligned memory local to the borrower; registering is a no-op.
    - **cuBLAS profiling hooks**, which the tracer observed: `TOOLS_TLS[2](record, domain, id, data)` writes a tools object into `*record` and returns 0; `21318c60[4](object, state, 0x100, 0x100)` sets `*state = 1` and returns 0. The real `TOOLS_TLS` has 3 entries; ZLUDA's empty entries would crash NVIDIA's cuBLAS, and ZLUDA avoids that only by shipping its own cuBLAS.
  - **Python:** the Store `python.exe` starts the interpreter through a broker, so the launcher's job never sees it. Python instead preloads Nectar's DLL with `ctypes.WinDLL` before `import torch`, via `NECTAR_CUDA_PRELOAD` in the probe; a `.pth` hook comes later. The launcher now follows ordinary child processes with a job object.
  - **Safety:** the lender only runs `cuMemcpy` / `cuMemcpyAsync` (and the stream variants) when both addresses are GPU memory (`cuPointerGetAttribute`). Otherwise the unified addressing on the lender would treat a borrower's host address as the server's own memory. `cuda-probe` checks GPU-to-GPU still works and a host address is refused.
- **Step 4, llama.cpp done (2026-10-31):**
  - **Setup:** llama.cpp build 8642541, built here against CUDA 13.4.1 with cuBLAS 13.7 and installed to `%LOCALAPPDATA%\Programs\llama.cpp` (bin on the user PATH), with Gemma 4 12B (QAT Q4_0, 6.5 GB) in its `models` folder.
  - **Why it first bypassed Nectar:** cuBLAS and cuBLASLt load `System32\nvcuda.dll` by full path (`GetSystemDirectoryW` + `LoadLibraryExW`). That loads the real driver beside the injected one, and every call went to the real driver.
  - **Fix:** a new `redirect` crate. When Nectar's DLL (or the tracer) loads, it detours kernelbase's `LoadLibraryExW` with the retour crate (BSD-2-Clause; its dependencies libudis86-sys, region, mmap-fixed-fixed and slice-pool2 are MIT) and answers every non-data load of a file named `nvcuda.dll` with itself. It pins itself so `FreeLibrary` can't unload the detour; without that, `cuda-probe` crashed at exit. The tracer loads the real driver past its own redirect.
  - **What else it needed:**
    - `cuLaunchKernelEx` (and `_ptsz`), whose attributes are sent as they are. Protocol VERSION 3, because the launch request now carries an attribute list.
    - `f8cff951[2]`, which the tracer showed is a timing mark (phase, value, QPC reading, QPC frequency; writes nothing). Implemented as a no-op returning 0.
    - CUDA graphs and VMM pools worked unchanged.
  - **Results over loopback:**
    - `llama-bench`: pp512 2615 t/s against 3299 natively (79%); tg128 51.4 t/s against 71.5 (72%).
    - `llama-server` chat with greedy sampling: a byte-identical 416-token answer, generating at 44.4 t/s against 68.8 (65%).
    - With no lender, CUDA init fails cleanly and llama.cpp runs on the CPU.
  - `cuda-probe`, `runtime_probe.cu` and `torch_probe.py` still pass.
- **Wi‑Fi test kit ready (2026-10-31), `native/nectar-cuda/tests/wifi/`:**
  - **`New-BorrowerBundle.ps1`** builds `build\laptop-bundle\Nectar-Borrower` (7.1 GB). It contains:
    - Nectar's borrower files.
    - The driver's hybrid runtime, so the laptop needs no NVIDIA driver.
    - The MSVC 14.51 runtime.
    - llama.cpp and the model.
    - The desktop's native results in `expected\`.

    It refuses a build older than the sources.
  - **`Start-Lender.ps1`** won't start the server until a firewall rule exists, scoped to the borrower and to Private networks. It prints the admin command and never adds the rule itself. It logs connections with timestamps.
  - **`Test-Borrow.ps1` / `Run-Test.cmd`** run on the laptop: network (Wi‑Fi link, ping, port), cuda-probe, llama-bench, and a llama-server chat compared word for word. Results are zipped for reading on the desktop.
  - **Dry run on the desktop over loopback:** pp512 77%, tg128 71%, chat identical at 67%.
  - **Build-All fix:** `Build-All.ps1` now defaults to `build\` and refuses the repo root. With the old default (the current folder), the native builds went into the source folders.
- **First Wi‑Fi run (2026-10-31, laptop NRUSIMHA, Wi‑Fi 7; desktop on Wi‑Fi 6E RZ608 at 1.2 Gbps, no Ethernet):**
  - Probe passed with 5.8 ms per call, 68 MB/s up, 43 MB/s down.
  - llama-bench pp512 reached 14.9 t/s (61 waits per token, ~1,175 per prompt pass).
  - The bench died when the laptop slept, and the chat timed out.
- **Fewer waits, lender cache, LZ4 (2026-10-31):**
  - **No waiting:** calls that only queue work are sent without waiting (`client/src/calls.rs`); errors go to the next synchronize. A stream synchronize waits only if work may be running.
  - **Kept answers:** current context, devices, driver version, occupancy, and device and kernel attributes (fetched all at once via `nectarAttributes`).
  - **Batching:** requests are batched (flush on wait, 16 KiB, or 1 ms), and the server flushes replies when its input is drained.
  - **Upload cache:** uploads of 1 MiB or more go by XXH3‑128 content hash (`nectarUpload`, `nectarCacheIndex`), kept by the lender in `%LOCALAPPDATA%\Nectar\cuda-cache` (20 GB, LRU trimmed at start, hash verified before storing).
  - **LZ4, then removed:** it was added, then taken out on the user's call. Q4_0 weights come to 99.6% and real logits to 100.4% (91.8% shuffled), and graphics would design its own compression in its own transport if it needs any.
  - **When on:** the cache is on by default for non-loopback lenders. Protocol VERSION 5 (4 had compression). Crate: xxhash-rust 0.8 (BSL-1.0).
  - **Test kit:** keeps the laptop awake (SetThreadExecutionState); chat timeout 3600 s.
  - **Dry run** (loopback with 5.8 ms per wait, cache and LZ4 forced on, before LZ4 was removed): pp512 86%, tg128 89%, chat byte-identical at 86%. The added delay understates a real link.
  - **Not used:** a pasted analysis derived from decompiling Juice's binaries, per the clean-room rule.
- **Second Wi‑Fi run (2026-10-31 20:31, protocol 5), all PASS:**
  - **Link:** 5.97 ms per call, 70 MB/s up, 41 MB/s down.
  - **llama-bench:** pp512 1391 t/s (40%), tg128 24.0 t/s (31%).
  - **Chat:** byte-identical, 20.8 t/s (28%). Meets the M2 target of ≥ 30% for tg.
  - **Model load:** the first upload took ~95 s inside a 193 s bench, and the chat load came from the lender's cache in 11 s (338 files, 6.6 GB).
  - **Per-token cost:** 42 ms against 13 ms natively, matching 1 RTT + 1 MiB of logits at 41 MB/s. The lender's Wi‑Fi upstream (RZ608, 1.2 Gbps) is now the limit.
  - **Test kit:** shows program output live and a 15 s heartbeat with traffic and chat tokens. The server logs the test's port check plainly.
- **Step 4, next:**
  - Faster lender link: a USB or PCIe Ethernet adapter, or a better Wi‑Fi card, on the desktop. Logits time scales directly with its upstream.
  - Remaining first-pass waits: library loads, kernel lookups, param layouts (~80 round trips on the first prompt).
  - **Logits:** real llama.cpp logits (25 tokens, 1 MiB each) are 100.4% of their size under LZ4, and 91.8% after byte-plane shuffling (1.09×, only the exponent byte shrinks). That is below the 1.3× bar, so no shuffle codec. Per-token cost over Wi‑Fi is about 1 round trip plus 1 MiB of logits.
  - **Trace replay (WiCi +52%):** measured against a forward-and-wait baseline. Nectar already sends without waiting, with about 1 wait per token, and that wait is real: the host needs the logits to pick the next token. Replay would only save the per-token request bytes (a few KB), so it comes after the second Wi‑Fi run shows where the time goes. If it's needed, mine tracer logs offline for the per-token cycle first.
  - **Fair share between borrowers:** WiCi §5.4 saw a second client halve the first's token rate. Before a second borrower is supported, add per-borrower fair share on the server's reply path and GPU queue.
  - **Cache, decided:**
    - The key is the hash, with length checked on read and the hash recomputed on store; that's enough against accidental collisions.
    - Hashing can't overlap sending, because the hash decides whether the bytes are sent at all.
    - Uploads are raw bytes, which don't depend on the driver version, so the cache isn't split by version.
    - The 1 MiB threshold already covers nearly all of Gemma's 6.5 GB (337 files).
  - A newer PyTorch (CUDA 12.8+, with real sm_120 kernels; a large install, ask first), plus bigger workloads such as a small model forward pass.
  - Support host addresses in `cuMemcpy` properly: the client tracks device allocations and turns host-side copies into HtoD/DtoH.
  - The `.pth` preload for Python.
  - When `cuDriverGetVersion` can't reach the lender, report a local version so runtime programs say "no device" instead of error 35.
  - Get `nvcudart_hybrid64.dll` onto a borrower that has no NVIDIA driver, as a **driver-version sandbox** (the user's idea, refined):
    - Don't install NVIDIA's driver on the borrower. It refuses without NVIDIA hardware, and forcing it onto a virtual device would load its kernel driver against missing hardware.
    - Keep NVIDIA's user-mode pieces per lender driver version in `%ProgramData%\Nectar\drivers\<version>\`. They are never installed system-wide, are loaded only into programs Nectar launches, and match the lender's version exactly (the hello carries it). They don't clash with the laptop's own GPU drivers.
    - **Filling it:** for the user's own machines, the lender can send its copy. For a public release, the borrower downloads NVIDIA's official driver package of that version and extracts only those files, so NVIDIA remains the distributor (same pattern as the Juice binaries).
    - Nectar's hybrid search checks the sandbox first.
  - **Launcher must follow child processes:** the Store `python.exe`, venv launchers and `py.exe` start the real interpreter as a child. Done first with a job object plus new-process notifications (children are loaded within milliseconds of starting). Later, hook process creation in the target to close the race for children that load CUDA immediately.
  - Then test borrowing over Wi‑Fi.
- **Latency idea taken from GPUDirect RDMA, without its hardware:** receive network data into pinned host buffers (`cuMemHostAlloc`) and overlap the network with the GPU copy. NCCL's chunked, pipelined transfers are a reference for large copies.

## M3: Driver track (after M1, can run as its own session)

**Lesson from WiCi:** at kernel-driver level, `nvidia-smi` alone took about 800 network round trips (51× slower). A level‑3 forwarding WDDM driver would be extremely chatty over Wi‑Fi, so actual GPU work keeps going through user-mode API remoting (Juice for graphics, M2 for CUDA). The driver track is mainly about how the GPU appears to Windows.

- **M3a, Device Manager presence (level 2):** `driver/NectarIdd/`, a UMDF2 IddCx driver based on Microsoft's IddSampleDriver.
  - When a GPU is borrowed, the service creates a software device named after the lent GPU (e.g. "NVIDIA GeForce RTX 5070 (Nectar)") with `SwDeviceCreate`, and removes it on return.
  - **Build:** VS 2022 + WDK.
  - **Dev install:** `bcdedit /set testsigning on`, which requires Secure Boot off.
  - **Shipping:** EV certificate + Partner Center attestation.
  - **Docs state plainly:** this gives presence and a virtual monitor; DirectX apps still render through one-click launch.
- **M2b, real WDDM render GPU (level 3):** `docs/driver/wddm-render-driver.md`, a design and feasibility study, no code.
  - Architecture: a render-only DXGK miniport, a D3D12 user-mode driver that serialises over the network, and how VRAM is mapped remotely.
  - Constraints: the 2 s TDR timeout, and signing and HLK testing.
  - References: Windows GPU-PV (the open Linux dxgkrnl driver documents a paravirtual GPU protocol) and Mesa Venus/virglrenderer.
  - Effort estimate and a go/no-go recommendation.

## M3: Open hardware lender spec (docs, can run in parallel)

`hardware/`, licensed CERN-OHL-S-2.0, with docs under CC-BY-SA-4.0. Board specifics are researched and verified during writing, not assumed.

- **`requirements.md`:**
  - PCIe 3.0/4.0 x4 over OCuLink (SFF-8611/8612).
  - Power: GPU via PSU 8-pin/12V-2x6, ATX `PS_ON`/`PWR_OK` sequencing.
  - Networking: Wi‑Fi 6/6E/7 via M.2 E-key (AX210/BE200-class) and/or 1–2.5 GbE (I226-class).
  - Thermals.
  - **Hard constraint:** the GPU host must be a CPU with a PCIe root complex running Linux (x86_64 for the Juice backend), because GPU vendor drivers and Juice don't run on an RTOS.
- **`reference-designs.md`:**
  - Tier 1: off-the-shelf mini-PC + OCuLink dock.
  - Tier 2: LattePanda-class x86 module with an OCuLink carrier (verify which models expose PCIe x4).
  - Tier 3: custom open PCB, split into a compute module socket (x86 or ARM64 SoC, with ARM64 needing an open backend), the OCuLink port, Wi‑Fi/GbE, and a **management MCU running Zephyr RTOS** for power sequencing, fans, temperature, LEDs, Wi‑Fi provisioning over BLE, and the discovery beacon while the host boots.
- **`protocols.md`:** the `LenderInfo` beacon format from `pkg/lending`, and the MCU↔host UART protocol.
- **`appliance-os.md`:** headless Linux plus `agent` (built for linux/amd64) plus `Renderer_Linux` from the JuiceServer-linux bundle, run as a systemd service.
  - Juice's terms forbid shipping its binaries preinstalled or selling the appliance with them. The owner installs Juice for personal use, or the appliance runs an open backend such as M2's `nectar-cuda`.

---

## Files (M1)
- **New:**
  - `pkg/juiceclient/`, `pkg/lending/` (+ tests)
  - `cmd/nectar/`, `cmd/nectar-svc/`, `cmd/nectar/winres/`
  - `installer/nectar.iss`, `Package.ps1`
- **Modified:**
  - `cmd/juicify/app/juicify.go` and `juicify_windows.go`: use `pkg/juiceclient`, drop cgo.
  - `cmd/agent/app/agent.go` and `endpoints.go`: lender info, sessions list, beacon.
  - `pkg/server/server.go`: NoDelay.
  - `Build-All.ps1`, `go.mod`, `docs/baseline-setup.md`.
- **Untouched:** the closed Juice binaries.

## Verification
1. **Automated:** `go build ./...`, `go vet ./...`, `go test ./pkg/...`.
   - Beacon encode/decode and listener tests.
   - Connector state machine against an `httptest` fake agent: borrow → heartbeat → lender lost → reconnect → return, plus stale-session cleanup.
2. **juicify regression:** the rebuilt juicify runs vkcube as before.
3. **Desktop loopback** (one machine in both modes):
   - Install `NectarSetup.exe`.
   - Lend: the tray shows "RTX 5070 · lending".
   - Borrow from `127.0.0.1`:
     - `vulkaninfo --summary` (run plainly) lists the Juice GPU.
     - vkcube run *directly*, with no juicify, renders through Juice.
     - The Quake II remaster launched via the app list or right-click renders.
   - Return: `vulkaninfo` no longer lists it, and the HKLM key and env are gone.
   - Kill `nectar-svc`, then restart it: the cleanup runs.
   - Stop the agent: the borrower goes amber, then reconnects.
4. **Laptop:** install, check that the desktop appears automatically in the Borrow tab, borrow over Wi‑Fi, and repeat step 3.
5. **M2a / M3:** reviewed as separate deliverables. M2a is checked in a test-signed VM or on the laptop by confirming the Device Manager entry appears on borrow and disappears on return.
