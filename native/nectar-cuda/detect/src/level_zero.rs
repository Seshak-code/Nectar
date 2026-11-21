//! Level Zero, Intel's low-level compute API, through its loader (ze_loader.dll,
//! libze_loader.so.1). Only the API version and the number of devices are read; OpenCL names
//! the same devices.

use std::ffi::c_void;
use std::ptr;

use crate::{Api, ApiKind, Origin, load};

#[cfg(windows)]
const LIBRARIES: &[&str] = &["ze_loader.dll"];
#[cfg(not(windows))]
const LIBRARIES: &[&str] = &["libze_loader.so.1", "libze_loader.so"];

/// ZE_INIT_FLAG_GPU_ONLY
const GPU_ONLY: u32 = 1;

type Handle = *mut c_void;
type Init = unsafe extern "system" fn(u32) -> i32;
type DriverGet = unsafe extern "system" fn(*mut u32, *mut Handle) -> i32;
type ApiVersion = unsafe extern "system" fn(Handle, *mut u32) -> i32;
type DeviceGet = unsafe extern "system" fn(Handle, *mut u32, *mut Handle) -> i32;

pub fn detect() -> Vec<Api> {
    let Some(library) = load(LIBRARIES) else {
        return Vec::new();
    };
    unsafe {
        let (Ok(init), Ok(drivers), Ok(api_version), Ok(devices)) = (
            library.get::<Init>(b"zeInit\0"),
            library.get::<DriverGet>(b"zeDriverGet\0"),
            library.get::<ApiVersion>(b"zeDriverGetApiVersion\0"),
            library.get::<DeviceGet>(b"zeDeviceGet\0"),
        ) else {
            return Vec::new();
        };
        if init(GPU_ONLY) != 0 {
            return Vec::new();
        }
        let mut count = 0;
        drivers(&mut count, ptr::null_mut());
        let mut handles = vec![ptr::null_mut(); count as usize];
        drivers(&mut count, handles.as_mut_ptr());
        handles
            .into_iter()
            .filter_map(|driver| {
                let mut version = 0;
                api_version(driver, &mut version);
                let mut count = 0;
                devices(driver, &mut count, ptr::null_mut());
                (count > 0).then(|| Api {
                    kind: ApiKind::LevelZero,
                    version: format!("{}.{}", version >> 16, version & 0xffff),
                    device: format!("{count} device(s)"),
                    vendor_id: 0x8086,
                    device_id: 0,
                    memory_mib: 0,
                    driver: "Intel".to_string(),
                    driver_version: String::new(),
                    origin: Origin::Official,
                    conformance: String::new(),
                })
            })
            .collect()
    }
}
