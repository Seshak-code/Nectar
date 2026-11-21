# Nectar for any GPU, on any device

Status: Phases 1 and 2 are built (2026-11-01). The rest is a plan; each phase gets its own approval. Facts checked on 2026-10-31; claims marked *(to verify)* need hardware.

## The model

Nectar remotes **APIs, not hardware**. A borrower's program calls an API, and Nectar carries out each call on the lender's GPU. Making Nectar general purpose means filling in three independent axes, one cell at a time:

| Axis | Done | To add |
| --- | --- | --- |
| **API** ("personality") borrower programs call | CUDA | Vulkan (cross-vendor), HIP (AMD programs), OpenCL / Level Zero (Intel, general compute) |
| **Lender platform:** OS × CPU × GPU driver | Windows x64 and Linux x64, NVIDIA's driver | Linux arm64; community drivers; macOS through Vulkan-on-Metal |
| **Borrower platform:** OS × CPU | Windows x64, Linux x64 | Linux arm64, Windows on Arm, macOS, Android |

**Rule:** a program gets a lender that offers the API it calls. Nectar never translates between APIs itself. Where an open translator exists, it runs on the lender; for example, ZLUDA as the lender's CUDA driver on an AMD GPU (`nectar-cuda-server --cuda-driver <ZLUDA's nvcuda.dll>`).

## "A GPU working on an ARM device"

There are two meanings:
- **Borrowed:** the GPU stays in an x64 PC and the ARM device uses it over the network. This is software only, and exactly what Nectar does. On a Mac or a phone it's the only way: Apple Silicon Macs support no eGPU, and Android has no dGPU support.
- **Attached:** the GPU is plugged into an ARM board, which lends it. This is Linux only.

**Borrowers:**

| Borrower | CUDA programs | Vulkan programs |
| --- | --- | --- |
| Windows x64 | ✅ | Vulkan personality |
| Windows on Arm | x64 programs under emulation: the x64 build *(to verify)*. Native Arm64 CUDA arrived with CUDA 13.4 and driver 616 (NVIDIA's N1X / RTX Spark), and needs an Arm64 hybrid runtime | Vulkan personality |
| Linux x64 / arm64 | ✅ x64 (`libcuda.so.1`). NVIDIA ships aarch64 CUDA (SBSA) | Vulkan ICD |
| macOS | None: CUDA programs don't exist on macOS | Vulkan ICD through the Vulkan loader. Metal programs can't borrow: Metal assumes Apple's GPU |
| Android | None | Only programs that load their own driver (the libadrenotools pattern emulators use). System-wide needs root |

**Lenders:**

| GPU, OS | Official driver | Community driver |
| --- | --- | --- |
| NVIDIA, Windows | CUDA ✅, Vulkan, OpenCL | none |
| NVIDIA, Linux x64/arm64 | CUDA ✅ (x64, WSL), Vulkan, OpenCL; arm64 servers officially. Raspberry Pi 5 and Rockchip: with a community kernel patch and 4K pages | NVK: Vulkan 1.4 conformant on RTX 50 since Mesa 25.2. No CUDA |
| AMD, Windows | HIP SDK 7.2 (RDNA3/4), Vulkan, OpenCL | none |
| AMD, Linux | ROCm/HIP, AMDVLK | RADV (Vulkan), rusticl (OpenCL). Pi 5: Vulkan with a small patch, no ROCm |
| AMD through ZLUDA | | CUDA programs on AMD (ZLUDA v6+, ROCm 7, llama.cpp). Now a hobby project; NVIDIA's EULA restricts translation layers, so it's the user's call |
| Intel Arc, Windows | Vulkan, Level Zero, OpenCL | none |
| Intel Arc, Linux | compute-runtime (Level Zero, OpenCL), ANV (Vulkan, Intel's own in Mesa) | Pi 5: partial |
| Apple M-series, macOS | Metal only | KosmicKrisp (Mesa, Vulkan 1.4 conformant on Metal 4, macOS 26+) or MoltenVK |
| Apple M1/M2, Asahi Linux | none | Honeykrisp (conformant Vulkan), rusticl. M3 is supported but its GPU isn't usable yet; M4 isn't supported |

**So Vulkan is the general-purpose API.** Every vendor and every OS has a conformant Vulkan driver, official or community, macOS included through Metal. It's also the road to games (DXVK and vkd3d turn DirectX into Vulkan).

## Detecting drivers, community ones included

`nectar-detect` (crate `detect`) loads each API's library at run time, with no SDK. A missing library means the API isn't offered.

- **Vulkan:** `VkPhysicalDeviceDriverProperties` gives a registered driver id (`VkDriverId`), the driver's name and version, and the conformance version it passed. Each id maps to one of four kinds:
  - **Official:** NVIDIA, AMD (and AMDVLK), Intel (and ANV), Qualcomm, Arm…
  - **Community:** NVK, RADV, Honeykrisp, Turnip, PanVK.
  - **Layered:** MoltenVK, KosmicKrisp, Dozen.
  - **Software:** llvmpipe, SwiftShader.
- **Other APIs:**
  - **CUDA:** `cuDriverGetVersion`. Nectar's own driver exports `nectarIdentify` and is reported as remote, never initialized. ZLUDA marks its devices' names.
  - **OpenCL:** each platform is one driver: NVIDIA's, AMD's or Intel's own, rusticl, PoCL.
  - **HIP and Level Zero:** read the same way. Untried, since no AMD or Intel GPU was at hand.
- **Kernel drivers:** SetupAPI's display class on Windows, `/sys/class/drm` on Linux.

On the desktop it reports the RTX 5070 with CUDA 13.5, Vulkan 1.4.357 (NVIDIA proprietary 621.01, conformance 1.4.3.3) and OpenCL 3.0. In WSL it reports CUDA 13.5 through WSL's driver, and lavapipe as software Vulkan.

## The driver catalog (the "driver detector service")

Phase 4, not built. The design:

- **Local first.** Detection needs no network.
- **A signed, static catalog** rather than a live server: one file published with the user's releases. Each entry maps (vendor, device family, OS, CPU) to:
  - **Drivers:** the known-good drivers, official and community, kernel and user mode. Each comes with its minimum version, the APIs it gives, notes (such as "Pi 5: needs the 4K-page kernel"), and its **official download page**.
  - **Nectar's own needs per platform:** for example, where an Arm64 or matching hybrid runtime comes from: NVIDIA's package URL, the path inside it and its SHA-256.
- **Safety, because this is a supply-chain surface:**
  - An ed25519 signature on the catalog, checked against a key built into Nectar.
  - Downloads only from vendor and distribution domains, each file checked against its pinned SHA-256 before it's loaded.
  - User-mode files only into Nectar's sandbox, and only after the user agrees.
  - **Kernel drivers are never installed by Nectar:** it says what to install and where from.
- **`nectar doctor`** will combine the local inventory with the catalog, for example: "RTX 5070 on Linux: NVIDIA 580 present (CUDA, Vulkan). NVK available with Mesa ≥ 25.2 (Vulkan only)."
- **Updates:** metadata is checked daily.
- **Hosting:** where and when the catalog is published is the user's decision.

## Phases

1. ✅ **Vendor-neutral foundation:**
   - `detect` and `nectar-detect`.
   - Discovery format 2: vendor, OS, CPU and `api=` lines.
   - `borrow.cfg` `apis=`.
   - `--cuda-driver`.
   - The hybrid runtime is fetched only by a Windows borrower of the same CPU.
2. ✅ **Linux x64 port:**
   - Portable crypto.
   - The server on `libcuda.so.1`.
   - The client as `libcuda.so.1` (soname set).
   - The service under systemd with a Unix socket.
   - The `nectar` command.
   - `install/install-borrower.sh`.
   - **Not done yet:** arm64 builds wait for the cross-compiler and qemu in WSL.
3. **Vulkan personality, compute first:** the driver and the lender's side work (3.1); llama.cpp is next (3.2).
   - **The borrower side:** a Vulkan ICD (crate `vk`) that lists no device when nothing is borrowed.
   - **Code** generated from `vk.xml` (`vkcodegen`):
     - Types (`vksys`).
     - Tables of every struct and command, which one interpreter (`vkwire`) encodes and decodes.
     - The lender's call thunks.
   - **Design ideas from Mesa's Venus** (MIT; the venus-protocol license is *to verify*):
     - Calls that need no answer (every `vkCmd*`, submits, most creates) are batched and sent without waiting.
     - The borrower chooses handle ids.
     - Blocking waits run beside the ordered stream on the lender.
     - Mapped memory is shadowed: changed pages go up at submit and come back after a wait.
   - **Tested:**
     - `vulkaninfo` shows "NVIDIA GeForce RTX 5070 (Nectar)", matching the real device.
     - `vk-probe` (a compute shader over 64 MiB of mapped memory) passes Windows→Windows on the RTX 5070, Linux→Linux on lavapipe, and Windows→Linux.
   - **Acceptance:** llama.cpp's Vulkan backend over Wi‑Fi, both ways: the laptop borrows the RTX 5070, and the desktop borrows the laptop's **Intel Arc**. That's two vendors with no new hardware.
4. **The signed driver catalog** and `nectar doctor`.
5. **AMD:** a HIP personality, plus ZLUDA on the lender. Needs an AMD GPU.
6. **Apple:** a macOS lender (Vulkan over KosmicKrisp/MoltenVK) and a macOS borrower (the Vulkan ICD). Needs a Mac.
7. **Windows on Arm, native:** an ARM64X `nvcuda.dll` for x64 and Arm64 programs alike. Needs a Snapdragon or N1X laptop.
8. **Android:** a Vulkan driver package for programs that load their own driver. Needs a phone.
9. **Graphics and games:** frames come back as video (NVENC, AMF or QSV on the lender). This gets its own design study; games with anti-cheat stay out.

**Hardware that unlocks later phases:**
- A Raspberry Pi 5: an arm64 borrower, and a lender with an AMD card.
- Any M-series Mac (M1/M2 also allow Asahi).
- An Android phone with an Adreno GPU.
- A Windows on Arm laptop.

## Acceptance tests by layer

"Invisible to the application" means different things per layer, so each gets its own test:

| What it proves | Test |
| --- | --- |
| The CUDA driver API is right, and programs find the GPU with no launcher | A CUDA program installed normally, unmodified, on an installed borrower: llama.cpp, PyTorch, or Blender's Cycles on CUDA |
| Cross-platform | A Windows program on a Linux lender ✅ (llama.cpp, from WSL), and the other way round |
| Vulkan personality | `vulkaninfo` lists the borrowed GPU; llama.cpp's Vulkan backend; then a Vulkan game without anti-cheat |
| Windows treats it as an OS-level GPU (DirectX, Task Manager, the Xbox app) | A game from the Xbox PC app |

**Why the Xbox app test comes last.** Xbox PC games are Direct3D 12 programs, installed as protected packages. Direct3D reaches a GPU only through a WDDM kernel driver, which is the forwarding render driver that `docs/driver/wddm-render-driver.md` (option A) rules out over Wi‑Fi. A user-mode API driver such as Nectar's can't be found by them. Many Xbox titles also carry anti-cheat. So it's the acceptance test for the graphics track only:
- **Nearer route:** the Vulkan personality plus vkd3d-proton (Direct3D 12 on Vulkan) inside the game's process. Protected packages make that injection hard *(to verify per title)*.
- **Full route:** a WDDM driver.

## Sources

- NVK on Blackwell: [Phoronix](https://www.phoronix.com/news/Vulkan-1.4-NVK-Blackwell), [Collabora](https://www.collabora.com/news-and-blog/news-and-events/mesa-25.2-brings-new-hardware-support-for-nouveau-users.html)
- Asahi Linux M3/M4: [IT-Connect](https://www.it-connect.tech/asahi-linux-officially-supports-m3-macs-but-the-gpu-is-still-missing/), [GamingOnLinux](https://www.gamingonlinux.com/2026/09/asahi-linux-now-officially-supports-macs-with-the-m3-series-soc/page=1/)
- KosmicKrisp: [LunarG](https://www.lunarg.com/lunarg-achieves-vulkan-1-3-conformance-with-kosmickrisp-on-apple-silicon/), [Mesa docs](https://docs.mesa3d.org/drivers/kosmickrisp.html)
- ZLUDA: [Phoronix, v6](https://phoronix.com/news/ZLUDA-6-Released), [Phoronix, llama.cpp](https://phoronix.com/news/ZLUDA-Q4-2025-Report)
- HIP SDK on Windows: [AMD system requirements](https://rocm.docs.amd.com/projects/install-on-windows/en/docs-7.2/reference/system-requirements.html)
- GPUs on Raspberry Pi: [Jeff Geerling](https://www.jeffgeerling.com/blog/2025/nvidia-graphics-cards-work-on-pi-5-and-rockchip/), [Sentinel Core](https://www.crowdsupply.com/sanctuary-systems/sentinel-core/updates/the-current-state-of-gpu-drivers-on-arm-linux-and-what-it-means-for-sentinel-core)
- Windows on Arm CUDA: [HWBusters](https://hwbusters.com/news/nvidias-first-windows-on-arm-geforce-driver-just-confirmed-the-n1x-specs/)
- Android custom drivers: [libadrenotools](https://github.com/bylaws/libadrenotools)
- Venus: [Collabora](https://collabora.com/news-and-blog/blog/2022/10/19/a-look-at-vulkan-extensions-in-venus); `VkDriverId`: [Vulkan reference](https://docs.vulkan.org/refpages/latest/refpages/source/VkDriverId.html)
