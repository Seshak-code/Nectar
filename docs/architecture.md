# One app, two profiles

Decided 2026-11-05. Nectar is one app for lending a GPU over the air. The user chooses what it's for, **Gaming & rendering** or **AI & compute**, and that choice tunes the link. Each kind of work has its own driver on the borrower; the app, the service and the lease are shared, so one **Borrow** press gives a computer both.

| | AI & compute | Gaming & rendering |
| --- | --- | --- |
| **Programs** | PyTorch, llama.cpp, other CUDA programs; Vulkan builds on lenders without CUDA | Vulkan games; Direct3D games through DXVK and vkd3d-proton |
| **Borrower's driver** | `nvcuda.dll` (`libcuda.so.1` on Linux), `nectar_vk.dll` | `nectar_vk.dll` |
| **What limits it** | Each token's logits crossing the link | Uploads per frame, and the delay from a frame's submit to its picture |
| **What comes back** | The GPU's results, as memory | Frames, through Nectar's portable codec |
| **Socket buffers** | 4 MiB each way | 512 KiB up, 1 MiB down |
| **Wi‑Fi queue** | Best effort | Video |
| **Status** | Works over Wi‑Fi | Frames in a window on loopback, Windows and Linux; Wi‑Fi and real games next |

`nectar_vk.dll` carries Vulkan compute too, such as llama.cpp's Vulkan backend. That's the route to AI on lenders without CUDA (Intel, Apple), and it's why the profile tunes the link and doesn't choose a driver: both are there whichever is chosen.

## GPUs of any vendor

- **Vulkan is the common API.** Every vendor has a Vulkan driver, official or community, on every OS, and nothing in the Vulkan path is NVIDIA's. Frames are coded by Nectar's own code, on the lender's CPU, so they need no vendor's encoder either.
- **The app says when a lender doesn't fit the choice** (`Profile::note`).
- **A borrower with NVIDIA's own driver** can borrow through Vulkan: the installers leave Nectar's CUDA driver out there, and its CUDA programs keep its own GPU.
- **Tested so far:** lending the RTX 5070 and Mesa's lavapipe, across Windows and Linux. Intel Arc needs the laptop to lend, and the Mac waits for its address. `install-borrower-macos.sh` and the macOS paths were written without a Mac to build on.

## Configuring over Wi‑Fi

- **Low-latency mode** (`lending/src/wifi.rs`): a Wi‑Fi card normally leaves its channel now and then to look for access points, and naps between packets. Either stalls a connection for tens of milliseconds. On Windows, while the lease's connection lasts, each computer on Wi‑Fi holds its media streaming mode on and background scans off, by Wi‑Fi service calls any program may make. They end with the connection. It never runs on loopback, and `NECTAR_WIFI_TUNE=0` turns it off. **It hasn't been switched on yet: I kept it off this desktop, so the laptop run is its first.**
- **Socket buffers and queues** (`lending/src/profile.rs`): as the table above says. The video queue comes from qWAVE's audio and video class on Windows (DSCP 40, no administrator needed) and DSCP 34 elsewhere, both of which Wi‑Fi's default mapping puts in AC_VI. Whether a given router honours it is what the link check can show.
- **The link check** (`lending/src/link.rs`): 100 round trips, then 2 s each way, as the connection is tuned, with what the system says it had to send twice. Each end says what it's connected by. On Windows without location permission, details may be missing from older systems, so it uses the realtime connection quality query first, which doesn't need that permission.

## What both share

- **The app:** `nectar`, with one list of lenders and one Borrow and Return.
- **The service:** `nectar-svc` holds the lease and writes `borrow.cfg`, whose `apis=` line tells each DLL whether it has a GPU.
- **The lender:** `nectar-server`, with a personality per API, and the lender's unmodified vendor driver.
- **Trust:** discovery, the shared secret, and the token each connection proves.

## AI & compute

**Today:** unmodified CUDA programs through `nvcuda.dll`. Over Wi‑Fi, 31% of native in llama-bench and 28% in chat, with an identical answer.

**Where the time goes:** a token takes about 42 ms against 13 ms natively. About 6 ms is the round trip and about 26 ms is 1 MiB of logits at 41 MB/s, so the desktop's Wi‑Fi upstream is the main limit.

**Order of work:**
1. **A faster lender link.** Gigabit Ethernet on the desktop should bring generation from about 24 to about 35 t/s. That's an estimate, to be measured.
2. **Measure the link:** TCP retransmissions, jitter, and why a 1.2 Gbps link carries 41 MB/s.
3. **Transport changes,** only if step 2 shows they'd help.

**What a DLL can't do:** `nvcuda.dll` sees kernels and memory copies, and no "generate" call. Sampling happens in the program. So it can't move the generation loop to the lender while the program stays unmodified.

**Open: generation on the lender.** Running the loop there takes an engine on the lender, such as llama-server or vLLM, reached through its own API. It should run near native speed, but such a program talks to a server and doesn't see a GPU. If it's built, it's a second mode of this profile, started from the same app.

**Research, only with an engine like that:** these all change the program's kernels, so a driver can't apply them. Their figures are hypotheses until measured, and on a 12 GB RTX 5070, not the 4090 or 5090 they were written for.
- QuIP# 2-bit weights with dequantization in registers. Whether TMA and `wgmma` are worth using on consumer Blackwell needs testing; sources disagree.
- Speculative decoding with resident-expert routing.
- CUDA conditional graphs for expert routing.

## Gaming & rendering

**Pipeline:**
1. The game runs on the borrower. DXVK or vkd3d-proton turns Direct3D into Vulkan, beside the game.
2. `nectar_vk.dll` records commands locally. Nothing crosses the network while a frame is drawn.
3. The commands, and the mapped memory the game changed, go up with the submit. Memory the GPU can't write isn't fetched back first.
4. The lender renders into ordinary images. On present, the frame goes to the card's video encoder as H.264 (`vkwire/src/video_encode.rs`) where the lender has one and the borrower a decoder. Otherwise it's coded by compute shaders on the same GPU (`vkwire/src/frame_shader.rs`), or by `vkwire/src/frame.rs` on the CPU where that can't be done.
5. The frame comes back, and the borrower decodes it and draws it in the game's window. Two are on their way at a time, and the quality follows how long frames take to code and to arrive.

**Why a codec of Nectar's own first:** it needs nothing installed and no vendor's hardware, so it works on any lender and borrower, and it can be tested here. A hardware encoder should beat it where one exists; that's the next step, with UDP and error correction for frames, once real frames over the air show how much it matters.

**No proxy `dxgi.dll` is needed.** DXVK already provides `dxgi.dll` and builds its adapter from the Vulkan device, whose properties come from the lender's GPU. `DXVK_FILTER_DEVICE_NAME` selects "(Nectar)". The matching vkd3d-proton setting is still to verify.

**The limit that was: uploads.** At the 70 MB/s we measured up, 60 fps allows about 1.2 MB per frame, and through DXVK a frame of 2,000 draws holds about 1.5 MB of commands. The stream going up is now compressed against the frames before it, to about 115 KB (`protocol/src/pack.rs`). With semaphore waits answered on the borrower and pipelines compiled off the stream, that test program goes from 26 to 97 fps on a simulated link at the measured rates. Real games write textures and pages that may compress less: Persona 4 Golden's town wrote 3 MB a frame and hasn't been through the simulator. For demanding games, floats that all change don't compress at all, and textures need a different approach: `docs/texture-streaming.md`. The target is 120 fps, with compression allowed about 0.833 ms a frame.

**Done for DXVK:** descriptor update templates go up as plain updates; allocations are counted on the borrower instead of waiting; what a device can do is kept; memory the GPU can't have written isn't fetched; submits flush at once. Extensions passing host addresses or about the swapchain aren't offered. Persona 4 Golden, unmodified, runs at 60 fps in its 3D town on the simulated measured link, walking about: median 61 fps against 61 natively, worst 10% of samples 59, and lowest 50 with its uploads already kept or 29 from an empty cache. It writes about 3.3 MB a frame, and about 350 KB crosses. **That was at 1024x768 with a 60 Hz limit, as the desktop had fallen back to that.** At 2560x1440 it's a median of 77 fps against 121 natively, about 64%, worst 10% 63, lowest 57. It hasn't crossed real radios.

The Witcher 3's Direct3D 12 remaster, through vkd3d-proton beside it, plays on the same link. At 2560x1440 with ray tracing off, walking, it's a median of 36.7 fps, worst 10% 30.9, against about 47 on Direct3D 12 itself, about 78%. Native there is limited by the GPU, at 96% busy; through Nectar the lender's GPU is about 60% busy. What held it back turned out to be mostly CPU time on the game's render thread rather than the link. Sampling its threads found, in order: about 27,000 separate descriptor copy calls a frame; every mapping marked stale about 30 times a frame under one lock; and asks about copies still running. Compressing on four threads halved its time, with no effect on the frame rate that could be told from the spread of runs. The frame coder ran at a low quality at this size, as it held every frame to 12 ms and came down faster than it went up; it now goes by the time between the program's frames. With that out of the way the next limit was 2 frames on their way at a time, each taking about 50 ms there, which is 40 fps; the count now follows the program, up to 4. One run with 3 gave a median of 44.7 fps at a quality of about 50; the count following by itself hasn't run in a game yet.

**Task Manager** stays the end goal for how the GPU appears. It needs a kernel-mode driver on the borrower (`docs/driver/wddm-render-driver.md`), separate from everything above, and shows the GPU without changing how work reaches it.

## Transport

Over Wi‑Fi both directions are TCP today, frames included. UDP with error correction for frames is planned once frames have been measured over the air.

| Traffic | Transport | Why |
| --- | --- | --- |
| API calls, both tracks | TCP | Calls must arrive complete and in order, so lost packets need resending whatever carries them |
| Frames | TCP today; UDP with Reed-Solomon planned | A late frame is useless, so repair beats resending |
| Control and heartbeats | TCP | As today |
| Discovery | UDP broadcast | As today |

**To test:** whether the video queue reaches AC_VI, and helps, on this router and these cards.

## Parked

| Idea | Why |
| --- | --- |
| GPUDirect RDMA | NVIDIA's guide covers Linux, Tesla and Quadro GPUs, and changes to the other device's kernel driver. No sign that it works with a Wi‑Fi card |
| BaM | The prototype used a custom Linux driver, SSDs unbound from the NVMe driver, and an A100. That breaks the stock-driver rule, and it can't run on a Windows lender. Worth a look on a Linux appliance |
| MCS floor, retry limit, steering traffic by MLO link | No Windows control found. MLO also needs Wi‑Fi 7 at both ends; the desktop has Wi‑Fi 6E |
| Activation streaming | With every layer on the lender, only prompts and token IDs need to cross |

## Reading

- WiCi: Shen, Li, Xu, Meng. [WiCi: Wireless GPU Computing Infrastructure](https://arxiv.org/abs/2608.24204).
- BaM: Qureshi et al. [GPU-Initiated On-Demand High-Throughput Storage Access in the BaM System Architecture](https://arxiv.org/abs/2203.04910).
- Fiddler: Kamahori et al. [Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models](https://arxiv.org/abs/2402.07033). One machine's CPU and GPU, not a network.
- QMoE: Frantar, Alistarh. [QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models](https://arxiv.org/abs/2310.16795).
- Skliar et al. [Mixture of Cache-Conditional Experts for Efficient Mobile Device Inference](https://arxiv.org/abs/2412.00099).
- Zhu et al. [Pre-Attention Expert Prediction and Prefetching for Mixture-of-Experts Large Language Models](https://arxiv.org/abs/2511.10676).
