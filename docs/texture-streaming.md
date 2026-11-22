# Textures over the link

What it would take to carry a game's texture streaming over Wi‑Fi, and whether cutting colour bits and raising saturation afterwards can get 100x. Researched and partly tested on 2026-11-06. Keeping textures by hash, sending mips smallest first and coding large mips by wavelet on slow links are built; the rest of what's here isn't.

## The target

At 120 fps a frame is 8.33 ms, and compression gets about 10% of that, 0.833 ms. The link measured between the laptop and desktop carries about 68 MB/s up, which is about 566 KB a frame.

The one streaming rate found is Epic's Matrix Awakens demo, at about 10 MB a frame at 30 fps, about 300 MB/s. It's one demo, and no measured figures for The Witcher 3 or other games were found. If that rate is typical, new textures need about 4.4x to fit the link, before commands and anything else. That is the number to beat for what's seen for the first time. 100x isn't needed for it.

## Where 100x does and doesn't come from

A game's textures are already compressed when they reach the driver: BC1 at 4 bits a pixel, BC7 at 8. The size to compare with is that, not raw colour at 24 or 32.

| Method | What it gets | Source |
| --- | --- | --- |
| A lossless packer on ordinary BC7 | About 1.06x, on one test | Oodle's author |
| Re-encoding BC7 with some loss, so a packer works (RDO), lambda 40 | About 2.06x, RMSE 4.2 | Oodle's author |
| Across BC1–7 game data at lambda 40 | About 4x | Oodle's author |
| Basis Universal ETC1S | 0.3–3 bits a pixel at low to medium quality; 0.3–1.25 is the usual range | Basis Universal |
| Sending only a hash, for what the lender has | Limited only by the hash's size | Nectar's own |

So from BC7, re-encoding to ETC1S's range is about 6x to 27x, and from BC1 about 3x to 13x, at the cost of quality. Against raw 32-bit colour the same files are 25x to 100x, which is probably where a figure like 100x comes from, but a game doesn't send raw colour.

The only way found to reach 100x and far beyond, with no loss at all, is not to send the texture twice. A game reuses its textures across areas and sessions, and the lender can keep them. Nectar already measured this kind of saving: the second load of a model over Vulkan sent 84 MB instead of 6,973 MB, about 83x.

## Colour bits and saturation: tested

Tried on 5 crops of 256x256 from real rendered frames of Persona 4 Golden: colour in YCoCg with chroma averaged, luma and chroma cut to fewer bits, a lossless packer (zlib, slower than Nectar's) over that, then chroma multiplied by a gain on the way back. Raw is 24 bits a pixel. dE is the colour difference a person can start to see at about 1 to 2.

| Bits kept | Chroma resolution | Gain | Bits a pixel | Against raw | PSNR | Mean dE | Worst 1% dE | Saturation kept |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 luma, all chroma | half | 1.00 | 3.78 | 6.3x | 42.5 | 0.51 | 8.9 | 99% |
| 6 luma, 6 chroma | half | 1.00 | 1.97 | 12x | 40.9 | 1.16 | 9.0 | 98% |
| 6 luma, 4 chroma | half | 1.00 | 1.72 | 14x | 36.7 | 2.15 | 12.6 | 90% |
| 6 luma, 4 chroma | half | 1.25 | 1.72 | 14x | 31.5 | 3.00 | 20.1 | 102% |
| 5 luma, 4 chroma | quarter | 1.00 | 1.14 | 21x | 32.9 | 2.94 | 22.7 | 89% |
| 5 luma, 4 chroma | quarter | 1.25 | 1.14 | 21x | 30.2 | 3.56 | 26.0 | 101% |
| 4 luma, 3 chroma | quarter | 1.00 | 0.79 | 30x | 30.2 | 3.82 | 27.0 | 88% |
| 4 luma, 3 chroma | quarter | 1.25 | 0.79 | 30x | 27.5 | 4.33 | 29.3 | 100% |

**What this shows:**
- **The idea is half right.** Fewer chroma bits do lose saturation, down to 88–90%, and a gain does bring the average back to 100%.
- **It restores the average, not the information.** Every gain above 1 made the colours further from the original, by PSNR and by dE. A saturation gain can't say which of the values that were merged each pixel had.
- **The bit cut itself tops out around 30x against raw, with visible errors.** Dropped bits also bring banding, and simply scaling back up leaves it there.

**What this doesn't show:** how it looks. A person may prefer the restored saturation even though the numbers are worse, and that's worth a look side by side. These are frames rather than textures, from a game with simple art, with a packer chosen for convenience.

No published work was found on using saturation to recover lost colour bits. The established methods in the same family are chroma subsampling, which Nectar's frame codec already uses, and colour spaces such as YCoCg and ITP that hide banding better.

## What a driver can safely make lossy

A driver sees a texture's format, not what it's for. Normal maps, masks, look-up tables and data textures can't take colour tricks, and a saturation gain would corrupt them.

The format is one honest signal: `*_SRGB` formats hold colour. Lossy methods should be limited to those, and only when the user allows it, as it changes what the game drew. It could be a choice in the app's Gaming profile. Everything else stays exact.

## Measured on Nectar's upload path

Direct3D 11 through DXVK, 2,000 draws at 1920x1080, on the simulated measured link, with 16 frames of uploads repeating:

| Written per frame | Crosses the link | Frame rate |
| --- | ---: | ---: |
| Commands only, 1.5 MB | 132 KB | 99 fps |
| 4 MiB, nearly the same each frame | 149 KB | 40 fps |
| 16 MiB, nearly the same each frame | 201 KB | 25 fps |
| 4 MiB of floats that all change | 4.7 MB | 11 fps |
| 4 MiB that won't compress | 4.7 MB | 11 fps |
| 16 MiB that won't compress | 17 MB | 3.7 fps |
| 4 MiB that won't compress, second time, by hash | 2.1 MB | 24 fps |
| 16 MiB that won't compress, second time, by hash | 5.8 MB | 7.4 fps |

**Three limits show up here:**
1. **Floats that all change don't compress at all with an LZ.** This is what skinned vertices and matrices look like, and the probe's are only an imitation. Only the low bytes change, which breaks every 4-byte match. Coding each buffer as its difference from the same place in the last frame should help, and is untested.
2. **Copying is its own cost.** The frames with nearly nothing to send still fall to 40 and 25 fps. Most of the time goes to finding what was written, and to copying and storing it on the lender. Compression doesn't touch that.
3. **Kept uploads mostly missed.** They were hashed in fixed 4 MiB blocks from wherever a write began, and DXVK's staging memory moves each frame, so the same texture hashed differently. The next section fixes that.

## Textures by hash: built

**The first idea didn't work as designed.** The plan was to hash exactly what each `vkCmdCopyBufferToImage` reads. The regions came out right, down to each mip, but with DXVK the bytes had usually gone by then: the game's thread writes a texture, DXVK records the copy later on another thread, and a submit in between sent whatever was written, as a shader might read it. The driver can't tell which it is, because DXVK makes one huge buffer with every usage.

**So there are two ways in** (`vk/src/memory.rs`, `vk/src/uploads.rs`):
- **A known copy:** the exact bytes it reads are hashed, in blocks of 128 KiB. A program that records its copies before submitting gets this, as does DXVK, as long as the wait below works out.
- **No known copy:** bytes are cut where their content says, into pieces of 32–512 KiB, about 96 KiB on average. The same bytes are cut the same way wherever they sit, as long as they move by a multiple of 8 bytes, and whatever was written beside them. Only the pieces at each end of a texture miss. It reads 8-byte words, not bytes.

**No more wait.** Each flush that used a kept upload used to wait a round trip, about 9 ms on the measured link, in case the lender couldn't find it. Now the lender guarantees it: what it can't store it holds in memory for the connection, and tells the borrower to stop sending more to keep. If kept bytes do go missing, say the folder is deleted mid-game, the connection ends, as it does for a missing shader.

**The disk is protected.** What a game rewrites each frame never comes again, so it isn't kept:
- Bytes seen before are kept.
- The first bytes going to a place, a mip of an image or a part of a buffer, are kept, when the copy is known.
- With no known copy, a run of 2 MiB or more is kept at first sight, up to 2 GiB and then 1 MB/s. A frame's constants are shorter.
- Anything else goes as it is.

The cache's limit now holds while running: at the limit it keeps no more, and on starting it trims to 3/4 of the limit to leave room. Small pieces share one file, which may take half the limit.

**Fixed along the way.** The log line `an upload's bytes do not match its hash` was probably another thread writing as the bytes were hashed. What's kept is now hashed from the copy that's sent, and the line didn't appear in these runs.

**Measured**, Direct3D 11 through DXVK at 1920x1080 with 2,000 draws, on the simulated measured link. Every run reads the texture back at the end and compares it byte for byte with what was uploaded. All matched:

| Upload each frame | First time | Kept | Directly on the GPU |
| --- | ---: | ---: | ---: |
| 5.6 MB BC7 chain, a new texture each frame | 10.5 fps | 40 fps | 810 fps |
| 22 MB BC7 chain, a new texture each frame | 4.3 fps | 26 fps | 456 fps |
| 22 MB BC7 chain, the same texture rewritten | | 24 fps | |
| 358 MB BC7 chain, a new texture each frame | 5.3 s a frame | 0.46 s a frame | |

The 358 MB chain took 360 MB of uploads per frame down to 978 KB on the link, about 370x. At that rate 900 MB would be about 2.5 MB, about 37 ms on the link. The link has stopped being the limit for repeats: of the 0.46 s, about 250 ms is the lender reading 387 MB from its cache and copying it, and about 50 ms is hashing on the borrower.

The earlier measurement for comparison: 16 MiB of noise, second time, sent 5.8 MB at 7.4 fps.

It helps AI work too. Vulkan prompt processing on the simulated link in `llama-bench` went from about 1,580 t/s without cutting to about 2,120 t/s with it, once its repeated writes were kept, and a second model load still saves about 7 GB.

**Limits:**
- **First sight is unchanged by this.** 358 MB takes 5.3 s to cross the first time, if the game waits for it. The next section stops it having to, and lossy coding is what would shorten the crossing.
- **These are imitation textures.** They're random bytes, from a probe that repeats them, not a real game's. Hit rates in a real game aren't measured.
- **Rewritten data can take several laps.** The same image getting new bytes crosses once as it is, once to keep, then goes by hash.
- **A game that moves bytes by less than 8** gets no matches when nothing says what they are. DXVK's staging was seen at multiples of 16 KiB or more.
- **Hashing is on the submitting thread.** At about 50 ms for 358 MB, it goes over 0.833 ms from about 6 MB a frame.

## The worst case: 900 MB in a frame, and lossy coding by mip

The user's worst case is 900 MB of textures in one frame. Researched and partly tested on 2026-11-06. Nothing here is built.

**What 900 MB a frame means on this desktop:**

| Limit | Per second | Per frame at 120 fps | 900 MB takes |
| --- | ---: | ---: | ---: |
| Simulated measured link, up | 68 MB/s | 566 KB | 13.2 s |
| This desktop's PCIe 4.0 x16, to the GPU (about) | 31.5 GB/s | 262 MB | 29 ms, or 3.4 frames |
| The worst case itself, if every frame | 108 GB/s | 900 MB | One frame |

So 900 MB can't reach this GPU within one 120 fps frame even with no network, and at that rate the RTX 5070's 12 GB would be replaced every 0.11 s. It can only be a burst. Fitting it on the link in one frame takes about 1,590x, or about 0.005 bits a texel from BC7, which no codec found comes near. It has to be spread over time, and the mip chain is how.

**A mip chain is already split by frequency.** Each level holds the detail the level below can't, and a driver sees which level each `vkCmdCopyBufferToImage` region writes. In a 900 MB chain the top mip is 675 MB, the next 169 MB, then 42, 10.5, and 3.5 MB for all the rest. A wavelet splits a picture into the same bands, so one stream can carry the chain coarse to fine.

**Worked out for the measured link, with mips sent smallest first.** This is arithmetic, not a test, and assumes the trial's ratios hold on real textures. Low mips may well compress less than this:

| Mips delivered | BC bytes | Ratio assumed | Crosses | Time at 68 MB/s |
| --- | ---: | ---: | ---: | ---: |
| 5th and below | 3.5 MB | 10x | 350 KB | Within the first frame |
| 4th | 10.5 MB | 10x | 1.1 MB | 15 ms |
| 3rd | 42 MB | 10x | 4.2 MB | 62 ms |
| 2nd | 169 MB | 10x | 17 MB | 250 ms |
| Top | 675 MB | 20x | 34 MB | 500 ms |
| All | 900 MB | About 16x | About 56 MB | About 0.8 s, about 100 frames |

Until each level arrives the lender would clamp the image to the levels it has, which the game doesn't see, and the picture starts blurred and sharpens within a second. Anything seen before costs only its hash and stays sharp, so a burst like this should only happen once for each texture.

### Trial

Scalar Rust on one thread of the Ryzen 7 5800X, no new dependencies: 5 crops of real Persona 4 Golden frames (one 1024x1024, four 512x512), with a simple BC encoder of the trial's own. PSNR is against what the GPU would show natively, the game's BC data decoded. MB/s counts BC bytes. The code is in the session's scratchpad, not the repository.

- **Route A** changes BC blocks directly, borrowing a neighbour's or a recent block's indices or endpoints where that costs little, then splits headers from indices and uses Nectar's packer. The lender only unpacks.
- **Route B** decodes the top mip and codes it as a 5-level 5/3 wavelet, YCoCg with chroma at half resolution, with bit-packing and no real entropy coder. The lender decodes it, makes the mips below and encodes BC again.

**BC7, mode 6 only:**

| Scheme | Ratio | Top mip | Next | Next | Below |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lossless, as the link does now | 1.5x | Exact | Exact | Exact | Exact |
| A, same lambda for all mips, 32 | 2.1x | 38.8 | 39.2 | 39.5 | 41.0 |
| A, same lambda for all mips, 128 | 3.1x | 31.8 | 32.0 | 32.6 | 33.7 |
| A, by mip, 512/128/32 | 4.3x | 26.1 | 32.0 | 39.5 | 41.0 |
| B, same step for all bands, 8 | 5.1x | 35.4 | 36.1 | 37.2 | 37.9 |
| B, by band, step 4, x4, 2, 1 | 6.0x | 35.2 | 37.5 | 39.2 | 40.1 |
| B, same step for all bands, 16 | 9.4x | 30.7 | 31.8 | 33.1 | 34.8 |
| B, by band, step 8, x4, 2, 1 | 10.6x | 31.5 | 34.0 | 36.0 | 37.7 |
| B, same step for all bands, 32 | 21.4x | 25.9 | 26.8 | 28.0 | 29.9 |
| B, by band, step 8, x16, 4, 2 | 19.7x | 28.3 | 30.8 | 33.5 | 36.5 |
| Top mip left out, blurred | 4x | 29.0 | Exact | Exact | Exact |
| Top two mips left out, blurred | 16x | 23.6 | 25.5 | Exact | Exact |

**BC1:**

| Scheme | Ratio | Top mip | Next | Next | Below |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lossless, as the link does now | 1.3x | Exact | Exact | Exact | Exact |
| A, same lambda for all mips, 128 | 1.9x | 35.8 | 36.2 | 36.9 | 38.2 |
| A, by mip, 512/128/32 | 2.2x | 29.3 | 36.2 | 44.8 | 46.2 |
| B, same step for all bands, 16 | 3.9x | 30.2 | 29.4 | 29.9 | 30.3 |
| B, by band, step 8, x4, 2, 1 | 4.7x | 30.4 | 30.7 | 31.6 | 32.5 |
| B, by band, step 16, x4, 2, 1 | 9.7x | 27.2 | 28.2 | 29.0 | 29.9 |
| Top mip left out, blurred | 4x | 27.8 | Exact | Exact | Exact |

**BC5, on imitation normal maps** (each frame's brightness taken as a height):

| Scheme | Ratio | Top mip | Next | Next | Below |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lossless, as the link does now | 1.3x | Exact | Exact | Exact | Exact |
| A, by mip, 128/32/8 | 2.6x | 31.7 | 39.0 | 45.7 | 45.0 |
| B, same step for all bands, 32 | 14.8x | 27.2 | 28.5 | 30.2 | 33.1 |
| B, by band, step 8, x4, 2, 1 | 8.6x | 30.9 | 33.5 | 36.1 | 39.2 |
| B, by band, step 16, x4, 2, 1 | 19.6x | 27.8 | 30.1 | 32.8 | 36.1 |

**One thread's speed, MB/s of BC bytes:**

| | Encoding | Decoding | Encoding BC again |
| --- | ---: | ---: | ---: |
| Lossless, as the link does now | 288–345 | 900–1,176 | None |
| Route A | 7–17 | 460–640 | None |
| Route B | 15–44 | 17–58 | 21–43 |

**What this shows:**
1. **Coding by frequency works.** At about the same size, quantising the highest bands harder took the top mip from no gain to about 2 dB better, and the mips below about 1–5 dB better. It spends least on the level seen only up close.
2. **Route B reaches about 10–20x, and Route A about 2–4x.** Route A's results agree with Oodle's published 2–4x, and it needs nothing on the lender beyond an unpack.
3. **The ratios cost quality on the top mip.** At about 10–20x the top mip is near 27–31 dB here. Leaving the top mip out and upsampling the next gets 28–29 dB, at 4x with the rest exact. PSNR says nothing about how sharp each looks, so this needs a look by eye, but the blur is the bar any codec has to beat.
4. **None of this meets 0.833 ms on one CPU thread.** For 900 MB of BC7, Route B would take about 27 s on one thread, which is why the transforms would have to run as compute shaders.

**What this doesn't show:** real textures. These are frames from a game with simple art, and only 5 crops. The BC encoder is simple. BC7 here uses only mode 6, while real BC7 mixes 8 modes. The normal maps are imitations, and PSNR is a poor measure for them, since what matters is the angle. The mips were made from the top mip by the same filter as the lender's, which favours Route B, as a game's own mips may differ. The wavelet has no real entropy coder, which would shrink it further. Nothing was judged by eye, and no GPU version was run. BC6H and ASTC weren't tried.

### By format

| Format | Bits a texel | What the research found | For Nectar |
| --- | ---: | --- | --- |
| BC1 | 4 | RDO plus a packer, about 2–4x across BC formats | Route A is simple. Route B only gains at high ratios, as it starts from 4 bits and BC1's own block errors cost bits |
| BC5 | 8 | Two BC4 channels. Precision matters on smooth, glossy surfaces. Chroma tricks don't apply | Route B on X and Y, with an error measure for angle. A driver can't tell a normal map from other data, so this needs care |
| BC7 | 8 | 8 modes, and often under 10% smaller without RDO. bc7prep gets 5–15%, RDO 2–4x. UASTC is 5–8 bits with a fast path to BC7 | Route B gains most here. Route A will be harder with real mixes of modes |
| BC6H | 8 | 14 modes with an irregular layout. UASTC HDR 6x6 is about 3.56 bits and transcodes to BC6H, losing a fraction of a dB. Real-time GPU encoders exist | Hardest. HDR needs a wavelet with more precision, and PyroWave's author saw PSNR flatten from FP16 precision |
| ASTC | 0.89–8 | Very flexible, and the most complex | The RTX 5070 reports no ASTC, so a game can't send it to this lender. It only matters on other lenders |

### Fast decoding

| Method | Speed reported | Ratio or rate | Notes |
| --- | --- | --- | --- |
| PyroWave, a wavelet in Vulkan compute | Under 0.1 ms to decode 1080p 4:2:0, 0.25 ms to encode 4K, on an RX 9070 XT | About 1.5 bits a pixel at the quality its author found hard to tell apart | The author's own figures. No entropy coder, and rate control picks bits to drop per 32x32 block. It's for video, not textures |
| HTJ2K, on a GPU | 8K HDR at over 120 fps on a high-end GPU, about 4 Gpixels/s | Designed to stay close to JPEG 2000's | From its developers. Its bands match mips |
| JPEG XS, on a GPU | UHD 4:2:2 at 190 fps on a 2017 GPU | Visually lossless to about 3 bits a pixel | Old figures |
| Selkie / Mermaid / Kraken, on a CPU | About 5–6, 3 and 1.9 GB/s | 2.4–2.9x, 3–3.7x and 3.1–4x on RDO BC data | Vendor figures. Proprietary. A lossless layer over RDO |
| LZ4 | About 4.2–4.5 GB/s, on a build its reporter calls slow | 2.4–2.8x on RDO BC data | Open. Nectar's packer is the same kind |
| GDeflate, on a GPU | 24 GB in a little over 3 s on an RTX 4090, by NVIDIA's estimate | Deflate's, about 2.7–3.4x on RDO BC data | Open format, through `VK_EXT_memory_decompression`, which the RTX 5070 reports. Encoding on the borrower wasn't measured |
| JPEG in CUDA | 0.23–0.46 ms a frame at 1080p, RTX 4090 | 0.64 bits a pixel at 37.4 dB, against BC1's 4 bits at 38.4, on one scene | Decodes only blocks in view, and mips cut those from 81k to 12k. Its authors doubt the gain justifies the effort. It changes how a game samples, so a driver can't use it |
| RTX Neural Texture Compression | No timings on its page | About 5 bits a texel against BCn's 24, for a bundle of 3 textures, 40–50 dB | Its encoding time isn't stated, so live use is unproven. Its license needs checking |

No figure was found for how fast BC can be encoded on a GPU: Spark gives its numbers only in charts. That's the unknown Route B rests on, as the lender has to encode BC again.

### Where this leaves the order of work

1. **Keep textures by hash.** Built, above. It's the only way to carry a repeat in about a frame, and it's lossless.
2. **Send mips smallest first.** Built, above, with the lender filling waiting mips rather than clamping the image, which would have meant finding and rewriting every descriptor that uses it.
3. **A wavelet by band.** Built, above, on the CPU, as a step before the real bytes rather than in place of them, so nothing's lost and no permission is needed. Compute shaders are what would make it quicker.
4. **Route A** as the plain fallback: about 2–4x, and nothing to run on the lender but an unpack. As it encodes at about 7–17 MB/s a thread here, it can only cover a small part of a burst.

## Mips smallest first: built

`vk/src/later.rs`. A first-seen texture's copy goes ahead with something standing in for its large mips, and the game carries on. The real mips follow in the background, smallest first, and are copied in when each has all arrived. The picture starts coarse and sharpens, and nothing is lost in the end.

| 1 upload in a run of frames, first time | Longest frame, ordinary | Longest frame, mips later | Until it's all sharp |
| --- | ---: | ---: | ---: |
| 22 MB BC7 chain | 329 ms | 42 ms | 1.05 s |
| 87 MB BC7 chain | 1,077 ms | 102 ms | 4.95 s |
| 358 MB BC7 chain | 4,779 ms | 406 ms | 14.7 s |

With no uploads, the longest frame is 19 ms at 107 fps. While 358 MB crossed the game held about 80 fps. Sent the ordinary way it stops for 4.8 s, then runs on, so spreading the bytes out takes about 3 times as long to finish, in exchange for not stopping. These ran on the simulated measured link, each ending with a byte-for-byte read back, which all matched, including one taken straight after the upload, which has to wait for the real mips. A repeat of the 358 MB chain stops the game for 335 ms, nearly all of it hashing and fetching from the lender's cache.

**What it took:**
- **Knowing what the bytes are in time.** Nothing can be held back once it's gone. DXVK turns a Direct3D call into commands only after the call, so its commands can't read what was written after they were recorded. So for DXVK, recognised by the engine name it gives Vulkan, a submit sends only what was written before its command buffers finished recording, and what a command buffer not yet submitted copies waits for that one. Other programs may record first and write afterwards, so theirs all goes as before.
- **Long writes.** A long write can still be under way as a command buffer finishes, which makes it look needed. A submit about to send 4 MiB or more that nothing explains waits for a copy to be recorded: until the run has stopped growing for 20 ms, 500 ms at the most. This can only delay a submit. Memory where it comes to nothing three times stops being waited for.
- **A stand-in.** The lender fills a waiting mip by repeating the blocks of the mip below, or if that isn't there yet, as DXVK sends the largest first, of a sample of the mip's own blocks of at most 64 KiB, which goes straight away. Repeating blocks needs no decoder, so it works the same for every format with no knowledge of BC1, BC5, BC6H or BC7.
- **Staying out of the game's way.** Bytes go a block at a time, smaller mips overtaking larger. After each batch an echo from the lender shows how long they took. Over 8 ms more than the quickest seen means they're queuing along the way, so the batch halves. Otherwise it grows.
- **Putting it in place.** The mip's bytes wait in memory of their own on the lender, then are copied in on the same queue, leaving the mip in the layout the game's last barrier left it in. A lock keeps that in order with the game's submits.

**When a mip can wait:** a plain 2D texture with one layer, one sample and optimal tiling, only ever sampled, with the whole mip copied tightly packed, for the first time. A rewrite goes the ordinary way, since it's what the game wants to see. A chain with under 1 MiB to send goes the ordinary way, as does anything once 2 GiB is being held. Up to 256 KiB a submit goes at once, so small mips are real from the start.

**So the game can't tell:** a submit that copies from such a texture or blits it, clears it, writes to a waiting mip or moves it to another queue family first waits until the real mips are in place. Destroying a texture drops what's on its way to it. `vkCmdWaitEvents` isn't followed, so once it's seen nothing more waits. `NECTAR_VK_MIPS_LATER=0` turns all of this off.

**Limits:**
- **It's what the game drew, changed for a while.** A shader sampling a waiting mip gets the stand-in, for up to several seconds at the largest sizes. If a game reads a texture as data, such as a look-up table, rather than as a picture, that could affect more than how it looks. This is untested.
- **What it looks like hasn't been judged.** The test textures are random bytes. Repeating blocks should keep colours about right with a tiled pattern at close range, but nobody has looked at it.
- **Holding back writes trusts DXVK's behaviour.** It's a rule of how DXVK works today, by reading its behaviour here, not a promise of its. Deferred contexts and other versions haven't been tried. The read-backs check textures, and buffers a shader reads are only checked by what shows on screen in one small test. An early version of this left pages unsent, and zeros reached a read-back in 4 of 10 runs; that's fixed and tested, with 0 of 44 runs since, but it shows what a mistake here costs. `NECTAR_VK_HOLD_WRITES=0` turns it off, and mips then only wait when the bytes are still there.
- **vkd3d-proton gets none of the holding back,** as Direct3D 12 games may record first and write later. Copies it records before submitting can still wait.
- **The first frame still costs.** 406 ms at 358 MB is hashing and copying the bytes on the borrower, and making the stand-in on the lender. None of that was measured stage by stage.
- **Real games and radios are untested.**

## Coding large mips by wavelet: built

`vkwire/src/wavelet.rs`, `vkwire/src/bc.rs`, on the CPU at both ends. On a slow link, a waiting mip crosses twice: first coded, about 6 to 8 times smaller on real pictures, then as its real bytes. The lender decodes the coded version into blocks off the ordered stream and copies it in, replacing the stand-in. So this changes what's on screen for a while, and nothing in the end, and needs no permission to lose anything.

**On real pictures**, BC7 chains made as a mosaic of Persona 4 Golden frames, one upload in a run of frames. Each ended with a byte-for-byte read back, which matched:

| Link | Size | Coded version in place | Exact, with coding | Exact, without coding |
| --- | ---: | ---: | ---: | ---: |
| Measured, 68 MB/s | 22 MB | Not coded | 0.82 s | 0.82 s |
| Measured, 68 MB/s | 358 MB | Not coded | 9.92 s | 9.92 s |
| 20 dB weaker | 22 MB | 0.82–0.97 s | 3.0–4.1 s | Not run |
| 20 dB weaker | 87 MB | 3.14 s | 16.1 s | 14.0 s |
| 20 dB weaker | 358 MB | 13.1 s | 78.8 s | Not run |
| Busy | 87 MB | 2.64 s | 17.7 s | 18.8 s |

Coded mips at the measured link, forced on to compare: 87 MB in place at 1.73 s and exact at 4.01 s, against exact at 2.41 s not coded, and about 12% off the frame rate.

**The driver decides.** On a good link it's not worth it: the real bytes get there about as soon, and coding costs the game CPU. So after a first window of real bytes, what's getting through decides: under 20 MB/s, mips are coded. That first estimate read 23–31 MB/s on the measured link and 9–20 MB/s on the slower ones, which is closer than is comfortable. `NECTAR_VK_WAVELET=1` always codes, and `0` never.

**How it looks.** Screenshots of the top mip at 1:1 were looked at, by the agent that built this, not yet by the user or on a game. The coded stage is slightly soft, with faint ringing around sharp edges and text, and hard to tell from the real one at a glance. The stand-in before it is a coarse mosaic of flat colours. That mosaic replaced a first version that repeated each block's detail, which made a strong pattern.

**Speed, one thread of the Ryzen 7 5800X, ms a megapixel of BC7:**

| | Encoding | Decoding |
| --- | ---: | ---: |
| BC7 decode, BC7 encode (part of the below) | 8.8 | 16.2 |
| Transform and bits | 8.9 | 9.7 |
| All | 17.7 | 25.9 |

The transform was first written plainly at about 50 ms, and on the measured link it was then slower than sending the real bytes. Working along whole rows, and shifting instead of dividing, made it about 5 times quicker with the same size and quality. Half the logical CPUs are used at each end.

**Size against quality, on 5 crops of real frames, against the BC decoded:**

| | Ratio | PSNR |
| --- | ---: | ---: |
| BC7 sRGB, 3 bits | 8.1x | 31.6 |
| BC7 sRGB, 4 bits | 14.9x | 28.2 |
| BC1 sRGB, 3 bits | 3.7x | 29.4 |
| BC1 sRGB, 4 bits | 7.2x | 26.4 |

It uses 3 bits. BC7 here means mode 6 as the trial's encoder made it, so on a real mix of modes these may differ.

**What it is:** the top mip in tiles of 1024 texels, each with a 5-level 5/3 wavelet, finer bands losing more bits, bit-packed with no entropy coder. sRGB goes as YCoCg with chroma at half size. Anything else might be data, such as a normal map, so its channels go apart at full size, which is less than half as effective.

**The BC decoders were checked against the RTX 5070** with `vk-probe --bc-check 1`, on 16,384 random blocks of each format. BC7, in all 8 modes, matches exactly. BC1 to BC5 differ by up to 7 of 255, as GPUs may in how they work out colours between endpoints.

**Limits:**
- **It's on the CPU, not compute shaders.** That was the earlier recommendation, and it should still be the next step if this needs to go faster. For now half the CPU goes to it during a burst, on the game's own machine at one end: the frame rate falls while it runs, and a laptop has less to give. This was all measured with both ends sharing this desktop's CPU, which overstates the cost and has never been measured on a laptop.
- **BC6H isn't coded,** nor is BC1 with one-bit alpha, nor anything else `Kind::of` doesn't know. They keep the stand-in until their real bytes arrive.
- **The test pictures are frames,** with sharp text and flat art, not a game's textures. No real game has run through this.
- **Each mip is coded apart,** so nothing uses the mip below, and the same bands aren't shown as they arrive. Both would help, and neither is built.

## Recommendation, in order

1. **Keep textures by what they are.** Built, as above. A repeat costs a hash, wherever it sits in staging memory, with no loss, and it's kept across sessions on the lender. About 370x on a large chain.
2. **Delta against the last frame** for buffers that are rewritten each frame. Untested. If it works, this helps the demanding games most.
3. **Send a first-seen texture's small mips first.** Built, as above.
4. **Keeping only the lossy version,** for `*_SRGB` textures seen for the first time, in the 6x to 27x range, if the user allows it, so the real bytes never need to cross. That would save the link, where what's built saves only the waiting. Compute decoding at this rate is well established: GDeflate is reported at 28–33 GB/s on one benchmark, and Spark's author claims real-time BC7 encoding, though no independent numbers were found.
5. **Saturation restoring** only if a side-by-side look says it's worth it. It costs almost nothing to try on frames, in the decoder.

For bursts of first-seen textures, the section on the worst case replaces 4 with a wavelet by band.

The direction is the reverse of frames: for textures the borrower's CPU compresses and the lender's GPU decompresses. A laptop's CPU is the limit there.

## Sources

- [Oodle Texture](https://www.radgametools.com/oodletexture.htm), and its author's [sample run](https://cbloomrants.blogspot.com/2020/06/oodle-texture-sample-run.html) and [compressor comparison](https://cbloomrants.blogspot.com/2020/07/performance-of-various-compressors-on.html?m=0). Vendor figures, on one data set.
- [bc7enc_rdo](https://github.com/richgel999/bc7enc_rdo): 10–50% size reductions, open source.
- [Basis Universal](https://skia.googlesource.com/external/github.com/BinomialLLC/basis_universal/+/refs/tags/v2_0): ETC1S and UASTC rates.
- [Spark](https://ludicon.com/spark): real-time GPU texture encoders. Vendor claims.
- [NVIDIA on GDeflate](https://developer.nvidia.com/blog/accelerating-load-times-for-directx-games-and-apps-with-gdeflate-for-directstorage), and [Tom's Hardware on its throughput](https://www.tomshardware.com/news/nvidia-driver-boosts-gpu-decompression-by-17-percent-on-pcie-50-ssds).
- [Matrix Awakens' streaming rate](https://multiplayer.it/notizie/unreal-engine-5-la-velocita-dellssd-conta-meno-del-previsto-nello-streaming-dati.html), one report of Epic's figure.
- [Banding across colour spaces](https://www.cl.cam.ac.uk/~gd355/publications/hvei19_paper.pdf), University of Cambridge.
- [PyroWave README](https://raw.githubusercontent.com/Themaister/pyrowave/master/README.md) and [its author's write-up](https://themaister.net/blog/2025/06/16/i-designed-my-own-ridiculously-fast-game-streaming-video-codec-pyrowave/). The author's own figures.
- [Variable-Rate Texture Compression](https://arxiv.org/html/2510.08166), JPEG decoded in CUDA as it's sampled.
- [RTX Neural Texture Compression](https://github.com/NVIDIA-RTX/RTXNTC).
- [Oodle's compressors on RDO data](http://cbloomrants.blogspot.com/2020/07/performance-of-various-compressors-on.html), [its rate allocation](http://cbloomrants.blogspot.com/2021/02/rate-allocation-in-oodle-texture.html) and [bc7prep](http://cbloomrants.blogspot.com/2020/06/oodle-texture-bc7prep-data-flow.html).
- [Basis Universal's README](https://sources.debian.org/src/basis-universal/2.0.2-3/README.md), for UASTC and its HDR rates.
- [GPU decoding of HTJ2K](https://kakadusoftware.com/wp-content/uploads/ICIP2019_GPU.pdf), from HTJ2K's developers, and [its call for proposals](https://jpeg.org/downloads/htj2k/wg1n76037-HTJ2K_final_cfp.pdf).
- [JPEG XS](https://en.wikipedia.org/wiki/JPEG_XS), and [decoding it on a GPU](https://www.aminer.org/pub/5bbacb1e17c44aecc4eaa86c/decoding-jpeg-xs-on-a-gpu).
- [BC6 and BC7 layouts](https://themaister.net/blog/2020/08/30/compressed-gpu-texture-formats-a-review-and-compute-shader-decoders-part-2/), and [normal maps in BC5](https://www.ludicon.com/castano/blog/?p=1602).
- [Real-time BC6H on the GPU](https://gdcvault.com/play/1022953/Real-time-BC6H-Compression-on).
- [VK_EXT_memory_decompression](https://docs.vulkan.org/features/latest/features/proposals/VK_EXT_memory_decompression.html).
