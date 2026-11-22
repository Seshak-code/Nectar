# What to build next: research, November 2026

Status: research and a proposed order, no code. It covers four questions: sending frames over Wi‑Fi, compression Nectar doesn't have yet, making the CPU talk less to the GPU, and making a borrowed GPU look installed, including what that means for anti-cheat and for other systems. What Nectar already does is left out. Claims marked *(to verify)* come from one source, or are estimates of mine that haven't been measured.

## Where Nectar stands

Measured on the simulated link at 2560x1440, before this research:

- **The Witcher 3:** about 37–45 fps walking against about 47 natively. About 10 MB of writes a frame go up as about 1 MB. Frames come down at 270–480 KB each, at a quality of 20–50.
- **Persona 4 Golden:** 73–77 fps against 121.
- **Frames down:** Nectar's own tile codec, with unchanged tiles skipped, over TCP.
- **Looking installed:** Vulkan programs see the GPU as a registered driver. Direct3D games need DXVK or vkd3d-proton placed beside them.

## 1. Frames over Wi‑Fi

### Hardware video encoding through Vulkan Video

- **What's new:** encoding through Vulkan Video now exists on every desktop vendor. An Igalia status page last updated in May 2026 lists H.264, H.265 and AV1 encode for NVIDIA's driver on Windows and Linux, for AMD's on Windows and for RADV. Intel's ANV has H.264 and H.265, with AV1 merged for Mesa 26.3 on Arc Alchemist. Sunshine added Vulkan encoding in May 2026.
- **Why it matters here:** Nectar's coder has no motion compensation. It sends every changed tile whole, so a camera turn resends the whole picture. A video codec predicts each frame from the last, which is where most of its saving comes from. I'd expect 5–10 times fewer bytes at the same quality *(to verify by measuring)*.
- **Why Vulkan Video and not NVENC:** it's one API on any lender, which keeps the "any vendor" rule. Nectar's lender already speaks Vulkan.
- **The borrower's side:** it needs a decoder. Vulkan Video decode where the borrower's own GPU has it, otherwise the system's (Media Foundation, VideoToolbox, MediaCodec, VA-API).
- **Settings for low latency:** no B-frames, intra-refresh in place of periodic key frames, and one slice per packet group so a loss damages a strip and not a frame. None of the sources measured encode latency, so that's *to verify* on the RTX 5070.
- **The codec of Nectar's own stays** as the fallback where the lender can't encode.

### UDP with error correction, sized per frame

- **Tooth (NSDI 2025):** sets the amount of error correction for each frame by its size, where WebRTC's default sets one rate for all. In field tests on a cloud gaming service it cut stalls by 40–85% and raised bitrate by 11–29%. Those are the authors' own deployment figures.
- **Tencent's loss study (NSDI 2026):** reports 99.87% of frames delivered within 100 ms in production, by finding where each loss comes from and treating each kind differently.
- **For Nectar:** frames are the one kind of traffic where repair beats resending. `docs/architecture.md` already plans Reed-Solomon over UDP for them. Tooth says the amount per frame matters more than the code.

### Congestion and the radio itself

- **Pudica (NSDI 2024), SQP, ACE (SIGCOMM 2025):** congestion control for cloud gaming aims at no queue at all, and at sending a frame's bytes spread over the frame time and not as a burst. Nectar only changes quality today. Pacing a frame's packets is cheap to add once frames are on UDP.
- **BLADE and Law (NSDI 2026):** both find that long delays on Wi‑Fi come from short spells where no packet gets delivered, caused by how stations contend for the channel. BLADE reports 5 times lower tail latency and 90% fewer stalls, but it changes the access point. Nectar can't do that. What it can take from them: judge a link by its worst spells and not its average, and keep airtime low, which smaller frames do.

## 2. Compression not built yet

What exists: an LZ over a 256 MiB ring on several threads, large uploads kept by hash, and wavelet coding for large mips.

| Idea | What it's for | Expected gain | Source |
| --- | --- | --- | --- |
| **A dictionary of records** | Half of The Witcher 3's 9 MB a frame is 16-byte foliage records that come back in another order. The LZ finds each one but pays 4–5 bytes to say where. An index into a table of records seen would cost 2–2.5 bytes | About 250–300 KB off the 1 MB a frame *(my estimate from the capture)* | Nectar's own capture |
| **Float coding against the last frame** | Floats that change every frame, such as bones and particles, where the LZ finds nothing. XOR with the same record last frame, then pack the bits that are left | Unknown until measured in a game | ALP and G-ALP (DaMoN 2025), Falcon (arXiv 2511.04140, 10.8 GB/s on a GPU) |
| **Decompressing on the lender's GPU** | `VK_EXT_memory_decompression` takes GDeflate, in pieces of up to 64 KiB. It would take decoding off the lender's stream thread, which is one of the stages each frame waits on | A few ms a frame on the lender *(to verify, as is which drivers have it)* | Vulkan specification |
| **Sending only what a copy reads** | Nectar sends whole 4 KiB pages the game touched. 26% of the capture was zeros | Up to a quarter of the bytes before compression, less after | Nectar's own capture |

I found nothing usable on neural texture compression for this: no runtime a driver could apply to textures it didn't author.

## 3. The CPU talking less to the GPU

**What the research says:**

- **Trinity (OSDI 2022):** an Android emulator that keeps a "projection space" in the guest. Calls about contexts and handles are answered there, and only drawing reaches the host. It reports 93–97% of native speed.
- **Gleam (arXiv 2607.23115, July 2026):** CUDA remoting over a LAN, close to Nectar's AI track. It sends calls from a second thread, fetches answers in batches ahead of need, and caches model weights. It reports 1.4–24 times better remoting efficiency than its baselines.
- **Both confirm the direction Nectar took.** Handles chosen locally, calls sent without waiting, kept answers and the weight cache are the same ideas.

**What's left, from this month's own measurements:**

- **Descriptor heaps.** vkd3d-proton merged `VK_EXT_descriptor_heap` support in May 2026. With it, a game's descriptors become plain memory writes, and the 27,000 copy calls a frame that Nectar now batches would not exist at all. Nectar hides the extension today because its structures carry host addresses. Supporting it means carrying heap memory as mapped memory, which Nectar has, and translating descriptor bytes on the lender, which is new. This is the largest CPU item left for Direct3D 12 games.
- **A faster encoder on the game's threads.** After batching, samples still catch the render thread inside Nectar's table-driven encoder. Encoders written per command for the hundred or so that games call most would be several times faster *(to verify)*.
- **Decoding in parallel on the lender.** One thread decodes the whole stream. Command buffers recorded on different game threads could be replayed on different lender threads.

## 4. Looking installed, and anti-cheat

### What "installed" means, by API

- **Vulkan:** done. The driver is registered with the system's loader, and nothing is placed beside a game.
- **Direct3D 9 to 12:** Windows only gives these to a WDDM adapter, which is a kernel driver plus user-mode drivers that the Direct3D runtime loads into the game. That is also the only way into Task Manager.

### New since the October study: open WDDM drivers that forward rendering

`docs/driver/wddm-render-driver.md` ruled out a forwarding render driver as vendor-sized work. There is now open prior art:

- **The viogpu3d line** (Red Hat's virtio-win, continued this year as Yttrium, Triton and vmlab-win3d): a WDDM 2 kernel miniport under BSD-3-Clause and MPL-2.0, with Mesa's `d3d10umd` as the user-mode driver for Direct3D 9, 10 and 11, Zink for OpenGL, and Mesa's Venus as the Vulkan driver. It forwards a Windows guest's rendering to the host's Vulkan driver. Yttrium also publishes Windows on ARM builds.
- **What it proves:** the kernel side and the Direct3D 9–11 side can be built from open parts whose licences allow reuse.
- **What it doesn't:**
  - **Direct3D 12 has no real user-mode driver in any of them.** vmlab-win3d replaces the system's `d3d12.dll` with a bridge to vkd3d-proton, which is the same trick as placing it beside the game, moved to System32. A true Direct3D 12 driver for a forwarding adapter would be new work.
  - **They cross shared memory inside one PC,** not Wi‑Fi. Windows resets an adapter whose work stalls for 2 seconds.
  - **All are alpha and test-signed,** with Secure Boot off.

### Signing

- **To load on a normal Windows 11 PC:** an EV certificate, a Partner Center account, and attestation signing by Microsoft. That is automated and takes hours.
- **Limits:** attestation-signed drivers can't be offered through Windows Update. That needs the full compatibility programme and its test kit. One report says submissions will need a software bill of materials from March 2027 *(to verify)*.
- **Test-signing is not a route to users.** It needs Secure Boot off, which the stricter anti-cheats refuse outright.

### Anti-cheat: what's true and what I'd build

- **What anti-cheat looks at that Nectar touches:**
  - **Files and code in the game's process.** DXVK or vkd3d-proton beside a game is a changed game folder. A signed driver that the Direct3D runtime loads is the sanctioned route, the same one NVIDIA's own driver uses.
  - **The state of the system.** Riot's Vanguard On-Demand mode wants Windows 11 25H2 with Secure Boot, TPM 2.0, VBS, HVCI and IOMMU on. A Microsoft-signed driver is compatible with all of that. A test-signed one is compatible with none.
  - **Where the game is running.** Virtual machines and remote sessions are widely refused. Cloud services are let in game by game, by arrangement with the publisher.
- **What a signed driver doesn't settle:** with a borrowed GPU, every draw call and every finished frame exists in the clear on another computer. That is what a "second PC" cheat wants, so anti-cheat vendors have a real reason to refuse a remote GPU. Whether a game allows one is the vendor's and the publisher's decision, not something a driver can earn alone.
- **The honest route:** a signed driver that says what it is ("… (Nectar)"), nothing injected, and then asking the vendors, as cloud services do. Easy Anti-Cheat and BattlEye both have per-game switches for Proton, so they have a precedent for allowing an unusual graphics path.
- **What I won't build:** anything that hides Nectar from an anti-cheat, reports itself as the real NVIDIA hardware to pass a check, or loads kernel code Microsoft hasn't signed. That is evasion of a security control. It would also get users banned.
- **On Linux:** anti-cheat under Proton runs in user mode, and Nectar is already an ordinary system Vulkan driver there, with nothing beside the game. Whether a given title accepts it is untested.
- **Windows may change under this:** Microsoft is moving security products out of the kernel and is reported to be working with studios on anti-cheat outside it. I found no dates or details for games.

## 5. Other systems

| System | Vulkan programs | Direct3D games | Looks installed | Notes |
| --- | --- | --- | --- | --- |
| Windows x64 | Yes, today | DXVK or vkd3d-proton beside the game | Vulkan only | A WDDM adapter is the missing piece |
| Windows on ARM | Driver not built yet | As above, under emulation | No | Yttrium's ARM builds show the kernel side ports |
| Linux x64 and ARM, SteamOS | Yes, today | Yes, through Proton, with no change to the game | **Yes** | No kernel work at all. The best first target for "invisible" |
| macOS, Apple silicon | Through the Vulkan loader only | No | No | Apple allows no third-party GPU drivers, and Metal games can't borrow. A Mac can lend through KosmicKrisp, which is Vulkan 1.4 conformant on macOS 26 |
| Android | Only inside apps built with libadrenotools | No | No | System-wide needs root. Android 9 or later, 64-bit ARM |
| iOS | No | No | No | No driver loading. KosmicKrisp for iOS is reported as coming, which would only help lending |

Hardware video (section 1) is what makes the thin systems practical: every one of them has a video decoder, and none can run Nectar's compute-shader path.

## 6. The order I'd build in

1. **Hardware video for frames, through Vulkan Video.** The largest visible gain: picture quality at 2560x1440 and bytes coming down. It can be proven with unit tests and `vk-probe --present` before any game runs.
   - **Built 2026-11-12:** H.264 from the lender's encoder, decoded by Media Foundation on a Windows borrower. The README has how it works and the numbers.
   - **What it showed:** a 2560x1440 frame of a moving picture is 3 to 15 KB against the tile coder's 270 KB to 1.5 MB. Decoding on the CPU takes 5 to 7 ms alone and 11 ms beside a running lender, which is now the slow step, so decoding on the borrower's GPU is the next thing for video.
   - **Not yet run in a game.**
2. **A dictionary of records, and float coding, for what goes up.** Contained in `protocol/src/pack.rs`, and testable without a game.
   - **Waiting on a capture.** The capture these estimates came from is gone, and a compressor built without the bytes it's for is a guess in the hottest path there is. `NECTAR_VK_CAPTURE=<file>` on the borrower now keeps a GiB of what's packed, as it was, starting `NECTAR_VK_CAPTURE_AFTER` seconds in, so the next game run can record it.
   - **What working it through showed:** an index into a table of records only beats a tightly coded offset by about half a byte a record, since naming one of a million records takes 20 bits and a distance of 9 MB takes 24. Most of today's cost is the offset's coding, which spends 4 bytes on those 24 bits. That may be the cheaper thing to fix, and the capture will say.
3. **Frames over UDP with error correction sized per frame, and pacing.** This matters most on real radios, which Nectar hasn't crossed yet.
4. **Descriptor heaps.** It removes the largest remaining CPU cost for Direct3D 12 games.
5. **A WDDM adapter.** First the compute-only presence driver the October study recommends, then a study of the viogpu3d line for rendering. This one needs decisions that are yours: buying an EV certificate, opening a Partner Center account, and a virtual machine with test-signing on for development.

Linux is where "invisible, on any game Proton runs" is already true, so it's the cheapest place to show the end goal working.

## Sources

- [Vulkan Video ecosystem status](https://blogs.igalia.com/vjaquez/vulkan-video-status/) (Igalia)
- [Intel ANV enables Vulkan Video AV1 encoding](https://phoronix.com/news/Intel-DG2-Vulkan-Video-AV1) (Phoronix)
- [Sunshine adds Vulkan encoding](https://gamingonlinux.com/2026/05/sunshine-game-streaming-tool-adds-vulkan-encoding-plus-xdg-pipewire-and-kwin-direct-screencast-capture) (GamingOnLinux)
- [Tooth: fine-grained FEC in cloud gaming streaming](https://www.usenix.org/conference/nsdi25/presentation/an) (NSDI 2025)
- [From Source to Solution: tackling packet losses in large-scale cloud gaming](https://www.usenix.org/conference/nsdi26/presentation/wang-jing) (NSDI 2026)
- [BLADE: adaptive Wi‑Fi contention control](https://www.usenix.org/conference/nsdi26/presentation/guo-fengqian) (NSDI 2026)
- [Law: towards consistent low latency in 802.11 home networks](https://www.usenix.org/system/files/conference/nsdi26/nsdi26spring_shen-yibin_prepub.pdf) (NSDI 2026)
- [Pudica: near-zero queuing delay for cloud gaming](https://usenix.org/conference/nsdi24/presentation/wang-shibo) (NSDI 2024)
- [SQP: congestion control for low-latency interactive video](https://arxiv.org/pdf/2207.11857)
- [Falcon: GPU lossless float compression](https://arxiv.org/abs/2511.04140v1)
- [VK_EXT_memory_decompression](https://docs.vulkan.org/features/latest/features/proposals/VK_EXT_memory_decompression.html)
- [Trinity: high-performance mobile emulation through graphics projection](https://www.usenix.org/conference/osdi22/presentation/gao) (OSDI 2022)
- [Gleam: adaptive network-efficient CUDA API remoting](https://arxiv.org/abs/2607.23115)
- [VK_EXT_descriptor_buffer proposal](https://docs.vulkan.org/features/latest/features/proposals/VK_EXT_descriptor_buffer.html) and [VK_EXT_descriptor_heap proposal](https://docs.vulkan.org/features/latest/features/proposals/VK_EXT_descriptor_heap.html)
- [VKD3D-Proton merges descriptor heap support](https://phoronix.com/news/vkd3d-proton-descriptor-heaps) (Phoronix)
- [yttrium-virtio-gpu](https://github.com/milesbuckton/yttrium-virtio-gpu) and [vmlab-win3d](https://github.com/VMLabDev/vmlab-win3d)
- [Triton v0.3](https://www.linuxcompatible.org/story/triton-v03-lands-critical-fix-for-directx-gpu-acceleration-in-qemu-kvm-windows-vms)
- [Attestation signing](https://learn.microsoft.com/windows-hardware/drivers/dashboard/code-signing-attestation) and [driver code signing requirements](https://learn.microsoft.com/en-ca/windows-hardware/drivers/dashboard/code-signing-reqs) (Microsoft)
- [GPU paravirtualization](https://learn.microsoft.com/windows-hardware/drivers/display/gpu-paravirtualization) (Microsoft)
- [Windows 11 anti-cheat compared](https://windowsforum.com/news/windows-11-anti-cheat-compared-vanguard-on-demand-eac-errors-and-battleye-requirements.449026/) (secondary source, for Vanguard On-Demand's requirements)
- [Proton and anti-cheat](https://partner.steamgames.com/doc/steamhardware/proton) (Steamworks)
- [Microsoft removes antivirus from the Windows kernel](https://pureinfotech.com/microsoft-removes-antivirus-windows-kernel/) (secondary source)
- [KosmicKrisp achieves Vulkan 1.4 conformance](https://www.khronos.org/news/permalink/kosmickrisp-achieves-vulkan-1.4-conformance-on-apple-silicon) (Khronos)
- [libadrenotools](https://github.gg22.workers.dev/bylaws/libadrenotools) and [custom Adreno drivers](https://ppsspp.org/docs/reference/custom-drivers) (PPSSPP)
