//! CUDA, through the driver library (nvcuda.dll, libcuda.so.1). Nectar's own nvcuda.dll
//! exports `nectarIdentify`; it is reported as remote without being initialized, which would
//! connect to the lender.

use std::ffi::{CStr, c_char, c_int};

use crate::{Api, ApiKind, Origin, load, text_ptr};

/// The driver programs get: on Windows the CUDA runtime loads it from System32 only, so a
/// copy beside this program (such as Nectar's own in its build folder) doesn't count
fn libraries() -> Vec<String> {
    if cfg!(windows) {
        let system = std::env::var("SystemRoot").unwrap_or_else(|_| r"C:\Windows".to_string());
        vec![format!(r"{system}\System32\nvcuda.dll")]
    } else {
        vec!["libcuda.so.1".to_string(), "libcuda.so".to_string()]
    }
}

type Init = unsafe extern "system" fn(u32) -> c_int;
type GetInt = unsafe extern "system" fn(*mut c_int) -> c_int;
type DeviceGet = unsafe extern "system" fn(*mut c_int, c_int) -> c_int;
type GetName = unsafe extern "system" fn(*mut c_char, c_int, c_int) -> c_int;
type TotalMem = unsafe extern "system" fn(*mut usize, c_int) -> c_int;
type Identify = unsafe extern "C" fn() -> *const c_char;

pub fn detect() -> Vec<Api> {
    let Some(library) = load(LIBRARIES) else {
        return Vec::new();
    };
    unsafe {
        if let Ok(identify) = library.get::<Identify>(b"nectarIdentify\0") {
            let name = CStr::from_ptr(identify()).to_string_lossy().into_owned();
            return vec![Api {
                kind: ApiKind::Cuda,
                version: String::new(),
                device: "a borrowed GPU, when one is".to_string(),
                vendor_id: 0x10de,
                device_id: 0,
                memory_mib: 0,
                driver: name,
                driver_version: String::new(),
                origin: Origin::Remote,
                conformance: String::new(),
            }];
        }
        let (Ok(init), Ok(version), Ok(count), Ok(get), Ok(name), Ok(memory)) = (
            library.get::<Init>(b"cuInit\0"),
            library.get::<GetInt>(b"cuDriverGetVersion\0"),
            library.get::<GetInt>(b"cuDeviceGetCount\0"),
            library.get::<DeviceGet>(b"cuDeviceGet\0"),
            library.get::<GetName>(b"cuDeviceGetName\0"),
            library.get::<TotalMem>(b"cuDeviceTotalMem_v2\0"),
        ) else {
            return Vec::new();
        };
        if init(0) != 0 {
            return Vec::new();
        }
        let mut driver = 0;
        let mut devices = 0;
        version(&mut driver);
        count(&mut devices);
        let version = format!("{}.{}", driver / 1000, driver % 1000 / 10);
        (0..devices)
            .filter_map(|ordinal| {
                let mut device = 0;
                if get(&mut device, ordinal) != 0 {
                    return None;
                }
                let mut buffer = [0 as c_char; 256];
                name(buffer.as_mut_ptr(), buffer.len() as c_int, device);
                let mut bytes = 0usize;
                memory(&mut bytes, device);
                let device = text_ptr(&buffer);
                // ZLUDA runs CUDA on AMD GPUs and marks the device's name
                let zluda = device.contains("ZLUDA");
                Some(Api {
                    kind: ApiKind::Cuda,
                    version: version.clone(),
                    device,
                    vendor_id: if zluda { 0x1002 } else { 0x10de },
                    device_id: 0,
                    memory_mib: (bytes >> 20) as u64,
                    driver: if zluda { "ZLUDA" } else { "NVIDIA" }.to_string(),
                    driver_version: String::new(),
                    origin: if zluda { Origin::Layered } else { Origin::Official },
                    conformance: String::new(),
                })
            })
            .collect()
    }
}
