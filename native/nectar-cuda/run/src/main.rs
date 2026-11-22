//! nectar-cuda-run: starts a program with Nectar's nvcuda.dll in place of the real driver
//! (windows.rs). Windows only: on Linux a program uses Nectar's libcuda.so.1 when it comes
//! first in the library path.

#[cfg(windows)]
mod windows;

fn main() {
    #[cfg(windows)]
    windows::main();
    #[cfg(not(windows))]
    {
        eprintln!(
            "nectar-cuda-run is for Windows. On Linux, put Nectar's libcuda.so.1 first in the \
             library path: LD_LIBRARY_PATH=<its folder> <program>"
        );
        std::process::exit(2);
    }
}
