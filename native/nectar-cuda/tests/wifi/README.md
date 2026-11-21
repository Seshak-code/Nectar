# Borrowing the desktop's GPU from the laptop

The desktop lends its GPU from the Nectar app. The laptop installs Nectar once, then borrows the GPU from its own Nectar app. While it is borrowed, CUDA and Vulkan programs on the laptop find the GPU on their own, the way they would find one inside the laptop. Vulkan programs list it as "NVIDIA GeForce RTX 5070 (Nectar)", next to the laptop's own GPU. The test here then checks both APIs end to end, and keeps every log.

## On the desktop (the lender)

All commands below run in the Nectar repo.

1. Build: `.\Build-All.ps1`
2. Make the bundle for the laptop:
   ```powershell
   .\native\nectar-cuda\tests\wifi\New-BorrowerBundle.ps1
   ```
   This writes `build\laptop-bundle\Nectar-Borrower`, about 7 GB with the model. It also records the desktop's own results in `expected\`, so keep the GPU otherwise idle while it runs. It refuses a build older than Nectar's sources.
3. Get ready to lend:
   ```powershell
   .\native\nectar-cuda\tests\wifi\Start-Lender.ps1 -Borrower <laptop address>
   ```
   - **The firewall:** lending needs Windows Firewall to let the laptop in on TCP 43212 (CUDA) and UDP 43211 (finding lenders). The first time, the script prints the two commands that add those rules, for an administrator PowerShell. The rules admit only the laptop and only on Private networks; without `-Borrower` they admit the whole local network.
   - **Lending:** once the rules exist, the script opens Nectar. Choose **Lend this GPU**, enter a shared secret of at least 8 characters, and press **Start lending**. The window shows who borrows the GPU and the server's log, which is also saved to `%LOCALAPPDATA%\Nectar\lender-logs\`.
   - **Who can connect:** every connection must prove the secret, and the GPU goes to one laptop at a time. The calls themselves are not encrypted, so lend on your own network only.

## On the laptop (the borrower)

1. Copy the `Nectar-Borrower` folder to the laptop, with a USB drive or a network share.
2. Install Nectar once, in an administrator PowerShell, in that folder:
   ```powershell
   .\nectar\Install-Borrower.ps1 -Secret "<the same secret>"
   ```
   It refuses on a PC that has NVIDIA's own driver. `-Uninstall` removes everything it added. It installs:
   - **The app and service:** Nectar and its service in `C:\Program Files\Nectar`.
   - **The pseudo driver:** Nectar's own `nvcuda.dll` in System32. NVIDIA's files are not installed: their hybrid runtime arrives from the desktop when you borrow, in `%ProgramData%\Nectar\drivers\`.
   - **The Vulkan driver:** `nectar_vk.dll` in `C:\Program Files\Nectar`, registered with the Vulkan loader. It lists no device while nothing is borrowed. The laptop's own GPU stays as it is.
   - If Nectar was installed from an earlier bundle, install again from this one: the earlier one had no Vulkan driver.
3. Open **Nectar** from the Start menu. The desktop appears in the list within a couple of seconds. If it doesn't, type its address and press **Add address**. Pick it and press **Borrow**.
   - **Green:** "Using the GPU on DESKTOP", with the round trip and traffic.
   - **Amber:** "Reconnecting...": the link dropped. Programs carry on if it comes back within 20 s.
   - **Grey:** "No host available": the desktop has been gone longer than that, and programs see no GPU. Nectar borrows it again on its own once the desktop is back.
4. Run the test in the bundle folder, with no address:
   ```
   Run-Test.cmd
   ```
   - **What it runs:** the probe, benchmark and chat for CUDA, then the same for Vulkan with llama.cpp's Vulkan build. For Vulkan, llama.cpp is told to use the device named "(Nectar)", not the laptop's own.
   - **Options:**
     - `-Quick` for a short benchmark.
     - `-Steps probe,bench` for only those steps (from `network`, `probe`, `bench`, `chat`).
     - `-Api vulkan` (or `cuda`) for one API only.
     - `-Log` to log every call Nectar makes, which is large.
   - **Results:** they go to `results\<laptop>-<time>\` (Vulkan's in its `vulkan\` folder) and a `.zip` beside it. On the desktop, unzip it into `build\laptop-bundle\results\`.

While each program runs, its output shows in the window, indented, with Nectar's own lines in cyan. Every 15 s a white line gives the time so far and how much the laptop has sent and received. While the chat answers, that line also says how many tokens it has generated. The test keeps the laptop awake while it runs; keep it plugged in.

With CUDA, the first load sends all 6.5 GB of the model to the desktop, which keeps it (`%LOCALAPPDATA%\Nectar\cuda-cache`, up to 20 GB). After that, a load sends only a hash per tensor. Vulkan doesn't use that cache yet, so each Vulkan load sends the whole model: about 95 s per load over this Wi‑Fi, twice in the Vulkan steps (bench and chat).

## What the laptop needs

- 64-bit Windows 10 or 11 and a CPU with AVX2 (llama.cpp runs some work on the CPU).
- No NVIDIA GPU or driver, and no Visual C++ runtime: Nectar's files carry their own.

## What's in the bundle

| Folder or file | What it is |
| --- | --- |
| `nectar\` | Nectar: `nectar.exe` (the app), `nectar-svc.exe` (the service), `nvcuda.dll`, the Vulkan driver (`nectar_vk.dll`, `nectar_icd.json`), `Install-Borrower.ps1`, and `cuda-probe.exe`, `vk-probe.exe` and `nectar-cuda-run.exe` for testing |
| `llama.cpp\bin\` | llama.cpp for CUDA, with NVIDIA's cuBLAS and CUDA runtime and Microsoft's C++ runtime |
| `llama.cpp-vulkan\bin\` | llama.cpp for Vulkan (ggml-org's release b12030), with Microsoft's C++ runtime |
| `models\` | The model |
| `expected\` | The desktop's own results: probe, benchmark, and the chat answer (Vulkan's in `expected\vulkan\`) |
| `manifest.txt` | What the bundle was made from: driver, protocol, llama.cpp, file hashes |

The NVIDIA and Microsoft files in `llama.cpp\` are for your own machines only. Don't publish the bundle.

## If something goes wrong

- **The desktop isn't listed:**
  - Check that Nectar on the desktop says "Lending: ready for a borrower".
  - Check that both PCs have the same secret, and that the desktop's network is Private.
  - Run `Start-Lender.ps1` again to check the firewall rules.
  - Then try **Add address**.
- **"Refused: lent to ..."**: another laptop has the GPU. It frees within 30 s of that laptop pressing Return or going away.
- **The test says no GPU is borrowed:** press Borrow in Nectar first. The service's log is `%ProgramData%\Nectar\svc.log`.
- **Antivirus:** Nectar's files are unsigned, and an `nvcuda.dll` that isn't NVIDIA's can look odd to antivirus products. If one quarantines a file, restore it and allow `C:\Program Files\Nectar` and `System32\nvcuda.dll`.
- **The chat answer differs:** compare `chat-answer.txt` with `expected\chat-answer.txt` (for Vulkan, the ones in `vulkan\`). The answers can only match if the desktop's driver hasn't changed since the bundle was made.
- **Vulkan finds no "(Nectar)" device:** check that a GPU is borrowed, and that `vulkan\devices.txt` lists the laptop's own GPU. If Nectar's line is missing there, install again from this bundle.

