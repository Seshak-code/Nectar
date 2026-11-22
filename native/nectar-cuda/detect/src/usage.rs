//! How busy this computer's GPU is right now, in percent, as a lender reports it to its borrower.
//! Asked of whatever can say, without any SDK:
//!
//! - NVML, on NVIDIA's drivers on Windows and Linux: GPU 0, the one that's lent
//! - Windows' GPU counters, the numbers Task Manager shows, for any vendor: the busiest engine
//! - sysfs on Linux, where the kernel driver has `gpu_busy_percent` (amdgpu)
//!
//! Other GPUs, and software ones such as lavapipe, have no answer. A thread samples once a second,
//! from the first time anyone asks.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Once, OnceLock};
use std::thread;
use std::time::Duration;

use libloading::Library;

const UNKNOWN: u32 = u32::MAX;
static LATEST: AtomicU32 = AtomicU32::new(UNKNOWN);

/// The GPU's load in percent, as last sampled. None if nothing can say, or not yet.
pub fn gpu_percent() -> Option<u32> {
    static SAMPLING: Once = Once::new();
    SAMPLING.call_once(|| {
        thread::spawn(|| {
            let mut source = source();
            loop {
                LATEST.store(source().unwrap_or(UNKNOWN), Ordering::Relaxed);
                thread::sleep(Duration::from_secs(1));
            }
        });
    });
    Some(LATEST.load(Ordering::Relaxed)).filter(|&percent| percent != UNKNOWN)
}

type Source = Box<dyn FnMut() -> Option<u32> + Send>;

fn source() -> Source {
    if let Some(nvml) = nvml() {
        return nvml;
    }
    platform()
}

/// NVIDIA's management library
fn nvml() -> Option<Source> {
    type Init = unsafe extern "C" fn() -> i32;
    type Handle = unsafe extern "C" fn(u32, *mut usize) -> i32;
    type Rates = unsafe extern "C" fn(usize, *mut [u32; 2]) -> i32;

    static NVML: OnceLock<Option<Library>> = OnceLock::new();
    let library = NVML
        .get_or_init(|| {
            let name = if cfg!(windows) { "nvml.dll" } else { "libnvidia-ml.so.1" };
            unsafe { Library::new(name) }.ok()
        })
        .as_ref()?;
    unsafe {
        let init = *library.get::<Init>(b"nvmlInit_v2\0").ok()?;
        let handle = *library.get::<Handle>(b"nvmlDeviceGetHandleByIndex_v2\0").ok()?;
        let rates = *library.get::<Rates>(b"nvmlDeviceGetUtilizationRates\0").ok()?;
        if init() != 0 {
            return None;
        }
        let mut device = 0usize;
        if handle(0, &mut device) != 0 {
            return None;
        }
        Some(Box::new(move || {
            let mut utilization = [0u32; 2];
            (rates(device, &mut utilization) == 0).then_some(utilization[0].min(100))
        }))
    }
}

/// From Windows' counters, named like
/// `pid_1234_luid_0x0_0xA_phys_0_eng_3_engtype_Compute`, one per process and engine: the sum per
/// engine, and of those the busiest.
#[cfg_attr(not(any(windows, test)), allow(dead_code))]
fn busiest_engine(counters: &[(String, f64)]) -> Option<u32> {
    let mut engines: HashMap<&str, f64> = HashMap::new();
    for (name, value) in counters {
        let engine = name.find("luid_").map_or(name.as_str(), |at| &name[at..]);
        *engines.entry(engine).or_default() += value;
    }
    engines.values().copied().max_by(f64::total_cmp).map(|busiest| busiest.round().clamp(0.0, 100.0) as u32)
}

#[cfg(windows)]
fn platform() -> Source {
    use windows_sys::Win32::System::Performance::{
        PDH_FMT_COUNTERVALUE_ITEM_W, PDH_FMT_DOUBLE, PDH_MORE_DATA, PdhAddEnglishCounterW, PdhCollectQueryData,
        PdhGetFormattedCounterArrayW, PdhOpenQueryW,
    };

    let path: Vec<u16> = r"\GPU Engine(*)\Utilization Percentage".encode_utf16().chain(Some(0)).collect();
    let (mut query, mut counter) = (0isize, 0isize);
    let opened = unsafe {
        PdhOpenQueryW(std::ptr::null(), 0, (&mut query as *mut isize).cast()) == 0
            && PdhAddEnglishCounterW(query as _, path.as_ptr(), 0, (&mut counter as *mut isize).cast()) == 0
    };
    if !opened {
        return Box::new(|| None);
    }
    Box::new(move || unsafe {
        if PdhCollectQueryData(query as _) != 0 {
            return None;
        }
        let (mut size, mut count) = (0u32, 0u32);
        if PdhGetFormattedCounterArrayW(counter as _, PDH_FMT_DOUBLE, &mut size, &mut count, std::ptr::null_mut()) != PDH_MORE_DATA {
            return None;
        }
        // u64s, for the items' alignment
        let mut buffer = vec![0u64; (size as usize).div_ceil(8)];
        let items = buffer.as_mut_ptr().cast::<PDH_FMT_COUNTERVALUE_ITEM_W>();
        if PdhGetFormattedCounterArrayW(counter as _, PDH_FMT_DOUBLE, &mut size, &mut count, items) != 0 {
            return None;
        }
        let counters: Vec<(String, f64)> = std::slice::from_raw_parts(items, count as usize)
            .iter()
            .filter(|item| item.FmtValue.CStatus == 0)
            .map(|item| {
                let name = windows_sys::core::PCWSTR::from(item.szName);
                let length = (0..).take_while(|&i| *name.add(i) != 0).count();
                (String::from_utf16_lossy(std::slice::from_raw_parts(name, length)), item.FmtValue.Anonymous.doubleValue)
            })
            .collect();
        busiest_engine(&counters)
    })
}

#[cfg(target_os = "linux")]
fn platform() -> Source {
    let file = std::fs::read_dir("/sys/class/drm").ok().and_then(|cards| {
        cards
            .filter_map(Result::ok)
            .map(|card| card.path().join("device/gpu_busy_percent"))
            .find(|path| path.exists())
    });
    Box::new(move || std::fs::read_to_string(file.as_ref()?).ok()?.trim().parse::<u32>().ok().map(|percent| percent.min(100)))
}

#[cfg(not(any(windows, target_os = "linux")))]
fn platform() -> Source {
    Box::new(|| None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engines_add_up_by_process_and_the_busiest_counts() {
        let counter = |name: &str, value: f64| (name.to_string(), value);
        let counters = [
            counter("pid_10_luid_0x0_0xA_phys_0_eng_0_engtype_3D", 30.0),
            counter("pid_20_luid_0x0_0xA_phys_0_eng_0_engtype_3D", 25.5),
            counter("pid_10_luid_0x0_0xA_phys_0_eng_3_engtype_Copy", 40.0),
            counter("pid_30_luid_0x0_0xB_phys_0_eng_0_engtype_3D", 2.0),
        ];
        assert_eq!(busiest_engine(&counters), Some(56));
        assert_eq!(busiest_engine(&[]), None);
        assert_eq!(busiest_engine(&[counter("pid_1_luid_0x0_0xA_eng_0", 170.0)]), Some(100));
    }

    #[test]
    fn asking_is_safe_anywhere() {
        assert!(gpu_percent().is_none_or(|percent| percent <= 100));
    }
}
