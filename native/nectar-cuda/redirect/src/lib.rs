//! Makes every load of nvcuda.dll in a process give the DLL that installed the redirect:
//! Nectar's driver, or the tracer.
//!
//! The CUDA runtime loads nvcuda.dll by name, so Windows hands it the copy nectar-cuda-run
//! already loaded. NVIDIA's cuBLAS loads it by full path from System32 instead, and Windows
//! then loads the real driver beside the injected one; llama.cpp ends up entirely on the
//! real one. So the redirect detours LoadLibraryExW in kernelbase.dll, which LoadLibraryA,
//! LoadLibraryW and LoadLibraryExA all go through, and answers any load whose file name is
//! nvcuda.dll with the installing DLL. Loads as a data file or resource are left alone.
//!
//! Installing it pins the DLL in the process: FreeLibrary no longer unloads it.

use std::path::Path;
use std::sync::OnceLock;
use std::{io, os::windows::ffi::OsStrExt};

use retour::GenericDetour;
use windows_sys::Win32::Foundation::{HANDLE, HMODULE, MAX_PATH};
use windows_sys::Win32::System::LibraryLoader::{
    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_PIN, GetModuleFileNameW,
    GetModuleHandleExW, GetModuleHandleW, GetProcAddress, LOAD_LIBRARY_AS_DATAFILE,
    LOAD_LIBRARY_AS_DATAFILE_EXCLUSIVE, LOAD_LIBRARY_AS_IMAGE_RESOURCE, LoadLibraryExW,
};

type LoadLibraryExW = unsafe extern "system" fn(*const u16, HANDLE, u32) -> HMODULE;

struct Redirect {
    detour: GenericDetour<LoadLibraryExW>,
    /// The installing DLL's full path, NUL-terminated
    own_path: Vec<u16>,
    log: bool,
}

static REDIRECT: OnceLock<Redirect> = OnceLock::new();

/// Redirects loads of nvcuda.dll to `module`, logging each one to stderr when `log` is set.
///
/// # Safety
///
/// `module` must be the calling DLL's handle. Call once, from its DllMain.
pub unsafe fn install(module: HMODULE, log: bool) -> Result<(), String> {
    let mut own_path = vec![0u16; MAX_PATH as usize * 4];
    let length = unsafe { GetModuleFileNameW(module, own_path.as_mut_ptr(), own_path.len() as u32) };
    if length == 0 || length as usize >= own_path.len() {
        return Err("the DLL's path is unknown".into());
    }
    own_path.truncate(length as usize + 1);

    let kernelbase: Vec<u16> = "kernelbase.dll\0".encode_utf16().collect();
    let target = unsafe {
        GetProcAddress(GetModuleHandleW(kernelbase.as_ptr()), c"LoadLibraryExW".as_ptr().cast())
    }
    .ok_or("kernelbase.dll has no LoadLibraryExW")?;
    let target: LoadLibraryExW = unsafe { std::mem::transmute(target) };

    // The detour lives in this DLL, so it must never be unloaded once kernelbase jumps here
    let mut pinned = std::ptr::null_mut();
    let flags = GET_MODULE_HANDLE_EX_FLAG_PIN | GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS;
    if unsafe { GetModuleHandleExW(flags, module as *const u16, &mut pinned) } == 0 {
        return Err(format!("cannot pin the DLL: {}", io::Error::last_os_error()));
    }

    let detour = unsafe { GenericDetour::new(target, load_library as LoadLibraryExW) }
        .map_err(|error| error.to_string())?;
    let redirect = REDIRECT.get_or_init(|| Redirect {
        detour,
        own_path,
        log,
    });
    unsafe { redirect.detour.enable() }.map_err(|error| error.to_string())
}

/// Loads a library without the redirect, as the tracer loads the real driver
pub fn load_real(path: &Path) -> io::Result<HMODULE> {
    let path: Vec<u16> = path.as_os_str().encode_wide().chain([0]).collect();
    let module = match REDIRECT.get() {
        Some(redirect) => unsafe { redirect.detour.call(path.as_ptr(), std::ptr::null_mut(), 0) },
        None => unsafe { LoadLibraryExW(path.as_ptr(), std::ptr::null_mut(), 0) },
    };
    if module.is_null() {
        return Err(io::Error::last_os_error());
    }
    Ok(module)
}

unsafe extern "system" fn load_library(name: *const u16, file: HANDLE, flags: u32) -> HMODULE {
    let redirect = REDIRECT.get().expect("enabled only once installed");
    let as_data =
        LOAD_LIBRARY_AS_DATAFILE | LOAD_LIBRARY_AS_DATAFILE_EXCLUSIVE | LOAD_LIBRARY_AS_IMAGE_RESOURCE;
    if flags & as_data != 0 || name.is_null() || !is_nvcuda(unsafe { wide(name) }) {
        return unsafe { redirect.detour.call(name, file, flags) };
    }

    if redirect.log {
        let name = String::from_utf16_lossy(unsafe { wide(name) });
        let own = String::from_utf16_lossy(&redirect.own_path[..redirect.own_path.len() - 1]);
        eprintln!("nectar-cuda: load of {name} answered with {own}");
    }
    // Loading the DLL by its own path takes a reference, as the caller expects
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
