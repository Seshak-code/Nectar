//! Makes every load of nvcuda.dll in this process give Nectar's DLL.
//!
//! The CUDA runtime loads nvcuda.dll by name, so Windows hands it the copy nectar-cuda-run
//! already loaded. NVIDIA's cuBLAS loads it by full path from System32 instead, and Windows
//! then loads the real driver beside Nectar's; llama.cpp ends up entirely on the real one.
//! So when Nectar's DLL is loaded, it detours LoadLibraryExW in kernelbase.dll, which
//! LoadLibraryA, LoadLibraryW and LoadLibraryExA all go through, and answers any load whose
//! file name is nvcuda.dll with itself. Loads as a data file or resource are left alone.

use std::sync::OnceLock;

use retour::GenericDetour;
use windows_sys::Win32::Foundation::{HANDLE, HMODULE, MAX_PATH};
use windows_sys::Win32::System::LibraryLoader::{
    GetModuleFileNameW, GetModuleHandleW, GetProcAddress, LOAD_LIBRARY_AS_DATAFILE,
    LOAD_LIBRARY_AS_DATAFILE_EXCLUSIVE, LOAD_LIBRARY_AS_IMAGE_RESOURCE,
};

use crate::{log, logging};

type LoadLibraryExW = unsafe extern "system" fn(*const u16, HANDLE, u32) -> HMODULE;

struct Redirect {
    detour: GenericDetour<LoadLibraryExW>,
    /// This DLL's full path, NUL-terminated
    own_path: Vec<u16>,
}

static REDIRECT: OnceLock<Redirect> = OnceLock::new();

/// Installs the detour; `module` is this DLL
pub(crate) fn install(module: HMODULE) {
    if let Err(error) = unsafe { try_install(module) } {
        log(&format!(
            "cannot redirect loads of nvcuda.dll ({error}); programs that load it by full path will use the real driver"
        ));
    }
}

unsafe fn try_install(module: HMODULE) -> Result<(), String> {
    let mut own_path = vec![0u16; MAX_PATH as usize * 4];
    let length = unsafe { GetModuleFileNameW(module, own_path.as_mut_ptr(), own_path.len() as u32) };
    if length == 0 || length as usize >= own_path.len() {
        return Err("this DLL's path is unknown".into());
    }
    own_path.truncate(length as usize + 1);

    let kernelbase: Vec<u16> = "kernelbase.dll\0".encode_utf16().collect();
    let target = unsafe {
        GetProcAddress(GetModuleHandleW(kernelbase.as_ptr()), c"LoadLibraryExW".as_ptr().cast())
    }
    .ok_or("kernelbase.dll has no LoadLibraryExW")?;
    let target: LoadLibraryExW = unsafe { std::mem::transmute(target) };

    let detour = unsafe { GenericDetour::new(target, load_library as LoadLibraryExW) }
        .map_err(|error| error.to_string())?;
    let redirect = REDIRECT.get_or_init(|| Redirect { detour, own_path });
    unsafe { redirect.detour.enable() }.map_err(|error| error.to_string())
}

unsafe extern "system" fn load_library(name: *const u16, file: HANDLE, flags: u32) -> HMODULE {
    let redirect = REDIRECT.get().expect("enabled only once installed");
    let as_data =
        LOAD_LIBRARY_AS_DATAFILE | LOAD_LIBRARY_AS_DATAFILE_EXCLUSIVE | LOAD_LIBRARY_AS_IMAGE_RESOURCE;
    if flags & as_data != 0 || name.is_null() || !is_nvcuda(unsafe { wide(name) }) {
        return unsafe { redirect.detour.call(name, file, flags) };
    }

    if logging() {
        let name = String::from_utf16_lossy(unsafe { wide(name) });
        log(&format!("load of {name} answered with Nectar's nvcuda.dll"));
    }
    // Loading this DLL by its own path takes a reference, as the caller expects
    unsafe { redirect.detour.call(redirect.own_path.as_ptr(), file, 0) }
}

/// A NUL-terminated UTF-16 string, without the NUL
unsafe fn wide<'a>(text: *const u16) -> &'a [u16] {
    let mut length = 0;
    while unsafe { *text.add(length) } != 0 {
        length += 1;
    }
    unsafe { std::slice::from_raw_parts(text, length) }
}

/// Whether a path names nvcuda.dll (or nvcuda, to which Windows adds .dll)
fn is_nvcuda(path: &[u16]) -> bool {
    let start = path
        .iter()
        .rposition(|&unit| unit == u16::from(b'\\') || unit == u16::from(b'/'))
        .map_or(0, |separator| separator + 1);
    let file = String::from_utf16_lossy(&path[start..]).to_ascii_lowercase();
    file == "nvcuda.dll" || file == "nvcuda"
}

#[cfg(test)]
mod tests {
    use super::is_nvcuda;

    fn wide(text: &str) -> Vec<u16> {
        text.encode_utf16().collect()
    }

    #[test]
    fn names() {
        assert!(is_nvcuda(&wide("nvcuda.dll")));
        assert!(is_nvcuda(&wide("NVCUDA.DLL")));
        assert!(is_nvcuda(&wide("nvcuda")));
        assert!(is_nvcuda(&wide(r"C:\WINDOWS\system32\nvcuda.dll")));
        assert!(is_nvcuda(&wide("C:/Windows/System32/NvCuda.dll")));
        assert!(!is_nvcuda(&wide("nvcuda64.dll")));
        assert!(!is_nvcuda(&wide("nvcudart_hybrid64.dll")));
        assert!(!is_nvcuda(&wide(r"C:\nvcuda.dll\other.dll")));
    }
}
