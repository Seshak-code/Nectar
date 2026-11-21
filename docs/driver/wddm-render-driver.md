# A borrowed GPU in Task Manager: design study

Status: a study, no code. It covers what it takes for a GPU borrowed with Nectar to appear in Task Manager's Performance tab as itself, while the work keeps going over Nectar's CUDA remoting. Claims marked *(to verify)* need a test on real hardware before anything is built on them.

## Where Nectar stands

- **What already works:** to CUDA programs, a borrowed GPU already looks installed. `Install-Borrower.ps1` puts Nectar's `nvcuda.dll` in System32, `nectar-svc` holds the lease, and CUDA programs find the lender's RTX 5070 with no launcher. That flow is tested on the desktop; the laptop pass is pending.
- **Measured on the laptop over Wi‑Fi** (through `nectar-cuda-run`):
  - Generation: 24 tokens/s (31% of native).
  - Prompt processing: 1391 tokens/s (40%).
  - Chat: byte-identical answers.
- **What Windows itself doesn't know:** that there is a GPU. Device Manager and Task Manager list only the laptop's own Intel Arc.

## What Task Manager needs

- **Where the list comes from:** Task Manager's GPU list comes from the graphics kernel (dxgkrnl). Each entry is an adapter driven by a **WDDM kernel-mode driver**, which reports its engines (3D, Compute, Copy, …), its memory segments (dedicated and shared), and per-process usage. Task Manager reads them through the D3DKMT statistics interfaces. A process can't add an entry from user mode: no API takes "a GPU over there" from an application.
- **Kinds of adapters it can list:**
  - **Full display and render (WDDM):** what NVIDIA's own driver is.
  - **Render-only WDDM:** a render adapter with no display outputs, like the GPU in a hybrid laptop as Windows sees it.
  - **Compute-only, MCDM:** Microsoft Compute Driver Model, the WDDM subset for devices that only compute. NPUs use it, and Windows 11's Task Manager lists them. *(to verify: whether a third-party MCDM device shows with its engines and memory just as an NPU does)*
- **Kinds it won't list:**
  - **Indirect display (IddCx):** a user-mode driver for a virtual monitor. It renders on another adapter and has no engines of its own, so it adds a Display adapters entry in Device Manager but *(to verify)* no GPU in Task Manager.
  - **Software devices (SwDeviceCreate):** a Device Manager entry only, without a driver. That's the experimental Step 8 of the plan.

## Options

### A. A forwarding render driver

A WDDM render-only kernel driver plus a D3D12 user-mode driver that sends graphics and compute over the network, so DirectX itself runs on the borrowed GPU.

- **Why not:**
  - **It's chatty:** WiCi measured `nvidia-smi` forwarded at the kernel driver as 51× slower, with about 800 round trips. A Direct3D runtime talking to a kernel driver is chattier still.
  - **Timeouts:** Windows' 2 s timeout detection and recovery (TDR) resets adapters whose work stalls, and a Wi‑Fi hiccup would trip it.
  - **Size:** it is a GPU driver stack of the size vendors build with teams.
- **Verdict:** no.

### B. A compute-only presence driver (recommended, if anything)

An **MCDM** kernel driver for a virtual compute device.

- **What Task Manager shows:** an entry such as "NVIDIA GeForce RTX 5070 (Nectar)", with a Compute engine and dedicated memory. Its utilization and memory figures are what the lender reports through the heartbeat; the lender reads them from NVML.
- **What it does:** no work goes through it. CUDA keeps using Nectar's `nvcuda.dll` and the network protocol, as now.
- **It doesn't break DirectX apps:** compute-only devices appear through DXCore, not DXGI, so ordinary DirectX programs and games don't enumerate it or pick it to render. *(to verify on Windows 11 with a test device)*
- **Lifetime:** `nectar-svc` would create the device's node on borrow and remove it on return or loss, as Step 8 does for the software device. The driver has no hardware to talk to: it reports figures the service writes through a private IOCTL.
- **Effort:** a minimal MCDM driver is mostly DXGK callbacks that report capabilities, a node, segments and statistics. For someone with WDK experience, expect weeks to a first version and longer to make it robust. Driver bugs crash the whole PC (a bugcheck), so development needs a kernel debugger.

### C. Nothing in the kernel

Stay at what works now: CUDA programs see the GPU, and the Nectar app and tray show its state, round trip and traffic.

Task Manager's GPU list stays as it is.

## What signing takes

- **Installing a kernel driver:** Windows 11 with Secure Boot loads only kernel drivers signed by Microsoft.
- **Shipping:**
  - An EV code-signing certificate, a few hundred dollars a year, issued to a person or company after identity checks.
  - A Microsoft Partner Center hardware dashboard account.
  - Attestation signing: Microsoft signs the driver after an automated check. That is enough for Windows 11 retail PCs, without the Hardware Lab Kit.
  - Full WHQL certification, which needs the HLK, is optional.
- **Development:** a test-signed driver on a machine with test-signing on. The desktop and the laptop stay as they are, so development would happen in a **Hyper-V virtual machine** with test-signing on inside it only, debugged from the host over KDNET. An MCDM device needs no physical hardware, so a VM is enough to bring the driver up. The lender's figures can be simulated.

## Recommendation

- **Not now.** The CUDA path is what makes the GPU usable, and it works. Task Manager presence adds visibility, not capability, at the cost of a kernel driver, a certificate and Microsoft signing.
- **When it's worth it:** for a public release. Then option B: an MCDM presence driver, developed in a test-signed VM, shipped attestation-signed, with figures from the lender's NVML through the heartbeat.
- **Before committing, three checks, each a day or two in a test-signed VM:**
  1. A third-party MCDM device shows in Task Manager's Performance tab with its name, engine and memory.
  2. DirectX programs and games don't pick it.
  3. A user-mode service can create and remove its device node with `SwDeviceCreate` and an INF match, without a reboot.

  If any of these fails, option C stands.
