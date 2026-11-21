//! OpenCL, through the ICD loader (OpenCL.dll, libOpenCL.so.1, or macOS's framework). Each
//! platform is one driver: NVIDIA's, AMD's or Intel's own, or a community one such as Mesa's
//! rusticl, or PoCL on the CPU.

use std::ffi::c_void;
use std::ptr;

use crate::{Api, ApiKind, Origin, load, text};

#[cfg(windows)]
const LIBRARIES: &[&str] = &["OpenCL.dll"];
#[cfg(target_os = "macos")]
const LIBRARIES: &[&str] = &["/System/Library/Frameworks/OpenCL.framework/OpenCL"];
#[cfg(all(unix, not(target_os = "macos")))]
const LIBRARIES: &[&str] = &["libOpenCL.so.1", "libOpenCL.so"];

const PLATFORM_VERSION: u32 = 0x0901;
const PLATFORM_NAME: u32 = 0x0902;
const DEVICE_TYPE: u32 = 0x1000;
const DEVICE_VENDOR_ID: u32 = 0x1001;
const DEVICE_GLOBAL_MEM_SIZE: u32 = 0x101f;
const DEVICE_NAME: u32 = 0x102b;
const DRIVER_VERSION: u32 = 0x102d;
const DEVICE_TYPE_CPU: u64 = 1 << 1;
const DEVICE_TYPE_ALL: u64 = 0xffff_ffff;

type Id = *mut c_void;
type GetPlatforms = unsafe extern "system" fn(u32, *mut Id, *mut u32) -> i32;
type GetInfo = unsafe extern "system" fn(Id, u32, usize, *mut c_void, *mut usize) -> i32;
type GetDevices = unsafe extern "system" fn(Id, u64, u32, *mut Id, *mut u32) -> i32;

/// A platform's driver: its name and who makes it
fn platform(name: &str) -> (&'static str, Origin) {
    let lower = name.to_ascii_lowercase();
    if lower.contains("rusticl") {
        ("rusticl", Origin::Community)
    } else if lower.contains("clover") {
        ("Clover", Origin::Community)
    } else if lower.contains("portable computing language") || lower.contains("pocl") {
        ("PoCL", Origin::Community)
    } else if lower.contains("nvidia") {
        ("NVIDIA", Origin::Official)
    } else if lower.contains("amd") {
        ("AMD", Origin::Official)
    } else if lower.contains("intel") {
        ("Intel", Origin::Official)
    } else if lower.contains("apple") {
        ("Apple", Origin::Official)
    } else if lower.contains("opencl on d3d12") || lower.contains("clon12") {
        ("OpenCLOn12", Origin::Layered)
    } else {
        ("unknown platform", Origin::Unknown)
    }
}

unsafe fn string(get: GetInfo, id: Id, what: u32) -> String {
    let mut buffer = [0u8; 512];
    unsafe { get(id, what, buffer.len(), buffer.as_mut_ptr().cast(), ptr::null_mut()) };
    text(&buffer)
}

unsafe fn number<T: Default>(get: GetInfo, id: Id, what: u32) -> T {
    let mut value = T::default();
    unsafe {
        get(id, what, size_of::<T>(), (&mut value as *mut T).cast(), ptr::null_mut())
    };
    value
}

pub fn detect() -> Vec<Api> {
    let Some(library) = load(LIBRARIES) else {
        return Vec::new();
    };
    unsafe {
        let (Ok(platforms), Ok(platform_info), Ok(devices), Ok(device_info)) = (
            library.get::<GetPlatforms>(b"clGetPlatformIDs\0"),
            library.get::<GetInfo>(b"clGetPlatformInfo\0"),
            library.get::<GetDevices>(b"clGetDeviceIDs\0"),
            library.get::<GetInfo>(b"clGetDeviceInfo\0"),
        ) else {
            return Vec::new();
        };
        let (platforms, platform_info, devices, device_info) =
            (*platforms, *platform_info, *devices, *device_info);
        let mut count = 0;
        if platforms(0, ptr::null_mut(), &mut count) != 0 || count == 0 {
            return Vec::new();
        }
        let mut ids = vec![ptr::null_mut(); count as usize];
        platforms(count, ids.as_mut_ptr(), ptr::null_mut());

        let mut apis = Vec::new();
        for id in ids {
            let name = string(platform_info, id, PLATFORM_NAME);
            let version = string(platform_info, id, PLATFORM_VERSION);
            // "OpenCL 3.0 CUDA 13.5.40" → "3.0"
            let version = version.split_whitespace().nth(1).unwrap_or_default().to_string();
            let (driver, origin) = platform(&name);

            let mut count = 0;
            if devices(id, DEVICE_TYPE_ALL, 0, ptr::null_mut(), &mut count) != 0 {
                continue;
            }
            let mut device_ids = vec![ptr::null_mut(); count as usize];
            devices(id, DEVICE_TYPE_ALL, count, device_ids.as_mut_ptr(), ptr::null_mut());
            for device in device_ids {
                let cpu = number::<u64>(device_info, device, DEVICE_TYPE) & DEVICE_TYPE_CPU != 0;
                apis.push(Api {
                    kind: ApiKind::OpenCl,
                    version: version.clone(),
                    device: string(device_info, device, DEVICE_NAME),
                    vendor_id: number::<u32>(device_info, device, DEVICE_VENDOR_ID),
                    device_id: 0,
                    memory_mib: number::<u64>(device_info, device, DEVICE_GLOBAL_MEM_SIZE) >> 20,
                    driver: if driver == "unknown platform" { name.clone() } else { driver.to_string() },
                    driver_version: string(device_info, device, DRIVER_VERSION),
                    origin: if cpu { Origin::Software } else { origin },
                    conformance: String::new(),
                });
            }
        }
        apis
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn platforms() {
        assert_eq!(platform("NVIDIA CUDA"), ("NVIDIA", Origin::Official));
        assert_eq!(platform("rusticl"), ("rusticl", Origin::Community));
        assert_eq!(platform("Intel(R) OpenCL Graphics"), ("Intel", Origin::Official));
        assert_eq!(platform("AMD Accelerated Parallel Processing").1, Origin::Official);
        assert_eq!(platform("Portable Computing Language").1, Origin::Community);
    }
}
