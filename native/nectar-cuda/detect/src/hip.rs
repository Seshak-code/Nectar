//! HIP, AMD's CUDA-like API, through its runtime library (amdhip64_*.dll from the HIP SDK on
//! Windows, libamdhip64.so from ROCm on Linux). Not yet tried on AMD hardware.

use std::ffi::{c_char, c_int};

use crate::{Api, ApiKind, Origin, load, text_ptr};

#[cfg(windows)]
const LIBRARIES: &[&str] = &["amdhip64_7.dll", "amdhip64_6.dll", "amdhip64.dll"];
#[cfg(not(windows))]
const LIBRARIES: &[&str] = &["libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so"];

type Init = unsafe extern "C" fn(u32) -> c_int;
type GetInt = unsafe extern "C" fn(*mut c_int) -> c_int;
type GetName = unsafe extern "C" fn(*mut c_char, c_int, c_int) -> c_int;
type TotalMem = unsafe extern "C" fn(*mut usize, c_int) -> c_int;

pub fn detect() -> Vec<Api> {
    let Some(library) = load(LIBRARIES) else {
        return Vec::new();
    };
    unsafe {
        let (Ok(init), Ok(version), Ok(count), Ok(name), Ok(memory)) = (
            library.get::<Init>(b"hipInit\0"),
            library.get::<GetInt>(b"hipDriverGetVersion\0"),
            library.get::<GetInt>(b"hipGetDeviceCount\0"),
            library.get::<GetName>(b"hipDeviceGetName\0"),
            library.get::<TotalMem>(b"hipDeviceTotalMem\0"),
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
        // HIP versions are major * 10000000 + minor * 100000 + patch
        let version = format!("{}.{}", driver / 10_000_000, driver / 100_000 % 100);
        (0..devices)
            .map(|device| {
                let mut buffer = [0 as c_char; 256];
                name(buffer.as_mut_ptr(), buffer.len() as c_int, device);
                let mut bytes = 0usize;
                memory(&mut bytes, device);
                Api {
                    kind: ApiKind::Hip,
                    version: version.clone(),
                    device: text_ptr(&buffer),
                    vendor_id: 0x1002,
                    device_id: 0,
                    memory_mib: (bytes >> 20) as u64,
                    driver: "AMD".to_string(),
                    driver_version: String::new(),
                    origin: Origin::Official,
                    conformance: String::new(),
                }
            })
            .collect()
    }
}
