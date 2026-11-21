//! The GPUs the operating system knows, and their kernel drivers: the display class through
//! SetupAPI on Windows, /sys/class/drm on Linux. Elsewhere none are listed, and the APIs alone
//! describe the GPUs.

use crate::Adapter;

/// The vendor and device ids in a PCI hardware id or modalias, such as
/// `PCI\VEN_10DE&DEV_2F04&SUBSYS…` or `pci:v000010DEd00002F04…`
pub fn pci_ids(text: &str) -> Option<(u32, u32)> {
    let upper = text.to_ascii_uppercase();
    let hex_after = |key: &str, digits: usize| {
        let start = upper.find(key)? + key.len();
        u32::from_str_radix(upper.get(start..start + digits)?, 16).ok()
    };
    if upper.starts_with("PCI\\") {
        Some((hex_after("VEN_", 4)?, hex_after("DEV_", 4)?))
    } else if upper.starts_with("PCI:V") {
        Some((hex_after("PCI:V", 8)?, hex_after("D", 8)?))
    } else {
        None
    }
}

#[cfg(windows)]
pub fn detect() -> Vec<Adapter> {
    use std::ptr;
    use windows_sys::Win32::Devices::DeviceAndDriverInstallation::*;
    use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
    use windows_sys::Win32::System::Registry::{KEY_READ, RegCloseKey, RegQueryValueExW};

    fn wide_text(buffer: &[u16]) -> String {
        let end = buffer.iter().position(|&c| c == 0).unwrap_or(buffer.len());
        String::from_utf16_lossy(&buffer[..end])
    }

    let mut adapters = Vec::new();
    unsafe {
        let set = SetupDiGetClassDevsW(&GUID_DEVCLASS_DISPLAY, ptr::null(), ptr::null_mut(), DIGCF_PRESENT);
        if set == INVALID_HANDLE_VALUE as HDEVINFO {
            return adapters;
        }
        let property = |info: &SP_DEVINFO_DATA, which: u32| {
            let mut buffer = [0u16; 1024];
            let ok = SetupDiGetDeviceRegistryPropertyW(
                set,
                info,
                which,
                ptr::null_mut(),
                buffer.as_mut_ptr().cast(),
                (buffer.len() * 2) as u32,
                ptr::null_mut(),
            );
            // A multi-string's first entry, or a plain string
            if ok != 0 { wide_text(&buffer) } else { String::new() }
        };
        let mut index = 0;
        loop {
            let mut info: SP_DEVINFO_DATA = std::mem::zeroed();
            info.cbSize = size_of::<SP_DEVINFO_DATA>() as u32;
            if SetupDiEnumDeviceInfo(set, index, &mut info) == 0 {
                break;
            }
            index += 1;
            let (vendor_id, device_id) = pci_ids(&property(&info, SPDRP_HARDWAREID)).unwrap_or((0, 0));
            let mut driver_version = String::new();
            let key = SetupDiOpenDevRegKey(set, &info, DICS_FLAG_GLOBAL, 0, DIREG_DRV, KEY_READ);
            if key != INVALID_HANDLE_VALUE as _ {
                let mut buffer = [0u16; 128];
                let mut size = (buffer.len() * 2) as u32;
                if RegQueryValueExW(
                    key,
                    windows_sys::core::w!("DriverVersion"),
                    ptr::null(),
                    ptr::null_mut(),
                    buffer.as_mut_ptr().cast(),
                    &mut size,
                ) == 0
                {
                    driver_version = wide_text(&buffer);
                }
                RegCloseKey(key);
            }
            adapters.push(Adapter {
                name: property(&info, SPDRP_DEVICEDESC),
                vendor_id,
                device_id,
                kernel_driver: property(&info, SPDRP_SERVICE),
                driver_version,
            });
        }
        SetupDiDestroyDeviceInfoList(set);
    }
    adapters
}

#[cfg(target_os = "linux")]
pub fn detect() -> Vec<Adapter> {
    use std::fs;
    use std::path::Path;

    let read = |path: &Path| fs::read_to_string(path).map(|text| text.trim().to_string()).unwrap_or_default();
    let hex = |text: String| u32::from_str_radix(text.trim_start_matches("0x"), 16).unwrap_or(0);
    let Ok(entries) = fs::read_dir("/sys/class/drm") else {
        return Vec::new();
    };
    let mut cards: Vec<_> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        // card0, card1…, not their connectors (card0-HDMI-A-1) or render nodes
        .filter(|name| name.strip_prefix("card").is_some_and(|n| n.bytes().all(|b| b.is_ascii_digit())))
        .collect();
    cards.sort();
    cards
        .into_iter()
        .filter_map(|card| {
            let device = Path::new("/sys/class/drm").join(card).join("device");
            let driver = fs::read_link(device.join("driver")).ok()?;
            let kernel_driver = driver.file_name()?.to_string_lossy().into_owned();
            let (mut vendor_id, mut device_id) = (hex(read(&device.join("vendor"))), hex(read(&device.join("device"))));
            if vendor_id == 0 {
                // Not PCI, such as Apple's (asahi), Qualcomm's (msm) or Arm's (panfrost) GPUs
                (vendor_id, device_id) = pci_ids(&read(&device.join("modalias"))).unwrap_or((0, 0));
            }
            let driver_version = read(&Path::new("/sys/module").join(&kernel_driver).join("version"));
            Some(Adapter {
                name: String::new(),
                vendor_id,
                device_id,
                kernel_driver,
                driver_version,
            })
        })
        .collect()
}

#[cfg(not(any(windows, target_os = "linux")))]
pub fn detect() -> Vec<Adapter> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_pci_ids() {
        assert_eq!(pci_ids(r"PCI\VEN_10DE&DEV_2F04&SUBSYS_00001234&REV_A1"), Some((0x10de, 0x2f04)));
        assert_eq!(pci_ids("pci:v00001002d0000744Csv00001DA2sd0000D475bc03sc00i00"), Some((0x1002, 0x744c)));
        assert_eq!(pci_ids(r"ROOT\BasicDisplay"), None);
    }
}
