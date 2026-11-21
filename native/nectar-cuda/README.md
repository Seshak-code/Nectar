# Nectar CUDA

CUDA on a borrowed GPU: an `nvcuda.dll` for the borrower that carries out each CUDA driver API call on the lender's GPU.

Built only from public headers (`cuda.h`, `cudaTypedefs.h`) and open-source references. No NVIDIA or Juice binary is disassembled.

| Crate | What it is |
| --- | --- |
| `sys` | CUDA types and constants, generated from `cuda.h` |
| `protocol` | The wire format, function ids, and the `cuGetProcAddress` table. Uses [xxhash-rust](https://crates.io/crates/xxhash-rust) (BSL-1.0) |
| `detect` | `nectar-detect`: this computer's GPUs, their kernel drivers, and the APIs their drivers offer (CUDA, Vulkan, OpenCL, HIP, Level Zero), official or community |
| `lending` | Lending: discovery and the shared secret (HMAC-SHA256 with RustCrypto's [hmac](https://crates.io/crates/hmac) and [sha2](https://crates.io/crates/sha2), [getrandom](https://crates.io/crates/getrandom); MIT/Apache-2.0), the lease with its grace, `borrow.cfg`, the hybrid runtime's sandbox |
| `redirect` | Windows: answers every load of `nvcuda.dll` in a process with Nectar's DLL (or the tracer) |
| `client` | The borrower's CUDA driver: `nvcuda.dll` on Windows, `libcuda.so.1` on Linux |
| `server` | `nectar-cuda-server`, which runs calls on the lender's GPU; with `--lend`, it lends it |
| `svc` | `nectar-svc`, the borrower's service (Windows service or systemd): finds lenders, holds the lease, writes `borrow.cfg` |
| `app` | `nectar`: lend this computer's GPU, or borrow one. A window on Windows (plain Win32), a command elsewhere |
| `run` | Windows: `nectar-cuda-run`, which starts a program with Nectar's `nvcuda.dll` loaded in place of the real one |
| `trace` | Windows: a pass-through `nvcuda.dll` that records every call a program makes (built as `nvcudatrace.dll`) |
| `probe` | `cuda-probe`, which checks any CUDA driver library end to end |
| `codegen` | The generator; not part of the normal build |

Every Windows binary links the C runtime statically (`.cargo\config.toml`), so a borrower needs no Visual C++ runtime.

## Platforms

| | Lender | Borrower |
| --- | --- | --- |
| Windows x64 | ✅ `nectar` → Lend | ✅ `install\Install-Borrower.ps1`, `nvcuda.dll` in System32 |
| Linux x64 | ✅ `nectar lend` (tested in WSL on NVIDIA's WSL driver) | ✅ `install/install-borrower.sh`, `libcuda.so.1` first in the library path (tested in WSL) |
| Linux arm64 | Builds expected; not yet built | Builds expected; not yet built or run |
| Windows on Arm, macOS, Android | Not yet | Not yet |

Lenders and borrowers mix: a Windows program runs on a Linux lender, and the other way round. CUDA and Vulkan are lent so far; the README at the repository's root has the overview, and `docs/general-purpose.md` in the repository has the plan for other vendors, APIs and platforms.

On Linux, build with `cargo build --release` in this folder; the borrower's library is `libnvcuda.so`, which `install-borrower.sh` installs as `/opt/nectar/lib/libcuda.so.1`. To try it without installing, start a lender with `nectar lend` (or `nectar-cuda-server` on loopback) and run programs with `LD_LIBRARY_PATH` pointing at a folder holding `libnvcuda.so` as `libcuda.so.1`, plus `NECTAR_CUDA_SERVER`.

## Lending and borrowing

- **The lender:** `nectar.exe` → **Lend this GPU** runs `nectar-cuda-server --lend` with a shared secret.
- **The borrower:** `install\Install-Borrower.ps1 -Secret …` installs the app, the `NectarBorrow` service and Nectar's `nvcuda.dll` in System32, once.
  - The app lists the lenders the service finds. **Borrow** takes the lease, and CUDA programs then find the GPU on their own, with no launcher.
  - While nothing is borrowed, `cuInit` answers "no device" at once, like a PC without a GPU.
  - `tests\wifi\README.md` has the steps.
- **Discovery:** the borrower broadcasts a probe on UDP 43211. Lenders that know the secret answer it; each side proves the secret with an HMAC over the other's nonce. The answer (format 2) names the lender's GPU vendor, OS and CPU, and each API it lends (`api=cuda 13.5 official NVIDIA`); `borrow.cfg` records the APIs borrowed (`apis=cuda`).
- **Connections:** each starts with a challenge from the lender. The service's control connection proves the secret and takes the lease, which comes with a random token. Programs' CUDA connections prove the token, which the service writes to `%ProgramData%\Nectar\borrow.cfg`. Only the hello is authenticated: the calls after it are neither encrypted nor signed.
- **The lease:** the GPU goes to one borrower at a time.
  - **On the lender:** the lease ends 30 s after the holder was last heard from with no connection open.
  - **On the borrower:**
    - The service goes amber ("Reconnecting...") when heartbeats stop, and keeps `borrow.cfg`.
    - After 20 s it removes `borrow.cfg` and goes grey ("No host available"), and programs see no device.
    - It borrows again on its own when the lender is back.
- **NVIDIA's hybrid runtime:** the lender sends its copy on the first borrow, into `%ProgramData%\Nectar\drivers\<CUDA version>-<hash>\`, matching its driver. Nothing of NVIDIA's is installed system-wide.

## Running a program on a borrowed GPU without installing

For development, against a server started without `--lend`, which takes any connection, so keep it on loopback or a trusted network:

```powershell
# On the lender
nectar-cuda-server --listen 127.0.0.1:43212

# On the borrower
$env:NECTAR_CUDA_SERVER = "127.0.0.1:43212"
nectar-cuda-run program.exe
```

The CUDA runtime only loads `nvcuda.dll` from System32, so a copy next to the program is never used. `nectar-cuda-run` loads Nectar's DLL into the program before the program starts. Processes the program starts get it too, through a job object, a few milliseconds after they start.

The runtime asks for `nvcuda.dll` by name, and Windows hands it the DLL already loaded under that name. NVIDIA's cuBLAS asks for `C:\Windows\System32\nvcuda.dll` by full path, which would load the real driver beside Nectar's, and llama.cpp then runs entirely on the local GPU. So when Nectar's DLL loads, it detours `LoadLibraryExW` in `kernelbase.dll` (with the [retour](https://crates.io/crates/retour) crate), which every `LoadLibrary` variant goes through, and answers any load of a file named `nvcuda.dll` with itself. Loads as a data file or resource are left alone. The detour lives in Nectar's DLL, so the DLL pins itself: `FreeLibrary` no longer unloads it.

### Python and PyTorch

The Store's `python.exe` runs the interpreter through a Windows broker that `nectar-cuda-run` cannot follow. So in Python, load Nectar's DLL before importing torch; the CUDA runtime then gets it the same way:

```python
import ctypes
ctypes.WinDLL(r"C:\path\to\nectar-cuda\nvcuda.dll")  # before import torch
import torch
```

`tests/torch_probe.py` does this when `NECTAR_CUDA_PRELOAD` names the DLL. A `.pth` file in site-packages could do it at interpreter startup, which would also cover venvs and worker processes.

### Over Wi‑Fi

`tests\wifi\` holds the test of borrowing from another machine: a script that bundles everything the borrower needs (Nectar, llama.cpp, the model and the lender's own results), a script that starts the lender, and the test the borrower runs. Its README has the steps.

The first run over Wi‑Fi (Wi‑Fi 7 laptop, Wi‑Fi 6E desktop, both through the router) worked but crawled: 5.8 ms per round trip, and llama.cpp waited for 61 of them per generated token and over a thousand per prompt pass, so prompt processing ran at 14.9 tokens/s. Since then (`client/src/calls.rs`):
- Calls that only queue work on the GPU are sent without waiting, and errors they meet are reported by the next synchronize. A stream synchronize waits only when work may still be running.
- Answers that cannot change are kept, such as the current context and device and kernel attributes, the latter fetched all at once.
- Uploads of 1 MiB or more go by content hash (XXH3‑128), and the lender keeps them on disk, so loading the same model again sends 16 bytes per tensor instead of 6.5 GB.
- Nothing is compressed. Under LZ4, Q4_0 weights come to 99.6% of their size and real llama.cpp logits to 100.4%, or 91.8% with the bytes of each float split into planes first, so compressing would cost time for nothing.

llama.cpp now waits about once per generated token. Its chat answer stays byte-identical to native with all of this on.

| Gemma 4 12B, RTX 5070 | Native | Loopback | Loopback + 5.8 ms per wait |
| --- | --- | --- | --- |
| `llama-bench` pp512 | 3448 t/s | 3056 t/s | 3018 t/s |
| `llama-bench` tg128 | 71.5 t/s | 67.7 t/s | 68.8 t/s |
| `llama-server` chat, generation | 68.4 t/s | 62.0 t/s | 57.6 t/s |

The added delay only slows the borrower's waits; on a real link the lender also gets each request later, and each token's 1 MiB of logits crosses the link. Over the real Wi‑Fi above (5.97 ms per round trip, 70 MB/s up, 41 MB/s down), all steps passed:

| Gemma 4 12B over Wi‑Fi | Native on the lender | Borrowed by the laptop |
| --- | --- | --- |
| `llama-bench` pp512 | 3517 t/s | 1391 t/s (40%) |
| `llama-bench` tg128 | 76.5 t/s | 24.0 t/s (31%) |
| `llama-server` chat, generation | 73.7 t/s | 20.8 t/s (28%), byte-identical answer |
| Model load | – | 6.5 GB uploaded once (about 95 s); the next load, from the lender's cache, 11 s |

A generated token takes about 42 ms against 13 ms natively. The difference is one round trip (6 ms) plus the token's 1 MiB of logits at 41 MB/s (26 ms), so the lender's Wi‑Fi upstream is now the limit, not the number of calls.

### Tracing

To see what a program calls, run it through the tracer. It redirects loads of `nvcuda.dll` the same way, and loads the real driver past its own redirect. When the program exits, the report is written to `%TEMP%\nectar-cuda-trace-<pid>.txt`:

```powershell
nectar-cuda-run --dll trace\nvcuda.dll program.exe
```

## Regenerating after a new CUDA release

1. Get the new CUDA headers (the `cuda_cudart` component has `cuda.h` and `cudaTypedefs.h`) and libclang.
2. Run the generator:

   ```powershell
   $env:LIBCLANG_PATH = "<folder with libclang.dll>"
   cargo run --release -p nectar-cuda-codegen -- <CUDA include folder> --refresh-exports
   ```

   `--refresh-exports` rereads the export list from the installed driver.
3. Read the diff of `codegen/report.md`. It shows what changed: functions that are new, now generated, or still unsupported, and why.
4. Build, then run `cuda-probe` and `tests/runtime_probe.cu` both directly and through Nectar.

Client and server must come from the same generation; the connection hello checks this.

## How calls are carried

The generator classifies every parameter:
- **Plain value:** sent as raw bytes.
- **Pointer to one value or plain-data struct:** copied to the lender and back, and may be NULL.
- **Input string:** sent as text.

Some functions are marked unsupported, with the reason in `codegen/report.md`, until they are written by hand: callbacks, host memory, arrays, pointers to pointers, and structs containing host pointers. Unsupported functions return `CUDA_ERROR_NOT_SUPPORTED` and log their name once.

`cuGetProcAddress` is answered locally from `cudaTypedefs.h`, using the same rule as the driver: the newest variant that is not newer than the version asked for. The real driver returns internal addresses, so its answers cannot be mapped back to exports.
