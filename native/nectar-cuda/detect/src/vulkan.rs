//! Vulkan, through the loader (vulkan-1.dll, libvulkan.so.1, libvulkan.1.dylib). Each device's
//! driver is named by `VkPhysicalDeviceDriverProperties` (Vulkan 1.2): a registered driver id,
//! the driver's own name and version, and the conformance version it passed.
//!
//! The structures are declared here from the Vulkan specification, only as far as they're read.

use std::ffi::{c_char, c_void};
use std::ptr;

use crate::{Api, ApiKind, Origin, load, text};

#[cfg(windows)]
const LIBRARIES: &[&str] = &["vulkan-1.dll"];
#[cfg(target_os = "macos")]
const LIBRARIES: &[&str] = &["libvulkan.1.dylib", "libMoltenVK.dylib"];
#[cfg(target_os = "android")]
const LIBRARIES: &[&str] = &["libvulkan.so"];
#[cfg(all(unix, not(target_os = "macos"), not(target_os = "android")))]
const LIBRARIES: &[&str] = &["libvulkan.so.1", "libvulkan.so"];

const STRUCTURE_TYPE_APPLICATION_INFO: u32 = 0;
const STRUCTURE_TYPE_INSTANCE_CREATE_INFO: u32 = 1;
const STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2: u32 = 1000059001;
const STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES: u32 = 1000196000;
const STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES: u32 = 1000071004;
const API_VERSION_1_3: u32 = (1 << 22) | (3 << 12);
const MEMORY_HEAP_DEVICE_LOCAL: u32 = 1;
/// VkPhysicalDeviceType
const DEVICE_TYPE_CPU: u32 = 4;
/// Vulkan's vendor id for Mesa's own software drivers (VK_VENDOR_ID_MESA)
const VENDOR_MESA: u32 = 0x10005;

#[repr(C)]
struct ApplicationInfo {
    s_type: u32,
    next: *const c_void,
    application_name: *const c_char,
    application_version: u32,
    engine_name: *const c_char,
    engine_version: u32,
    api_version: u32,
}

#[repr(C)]
struct InstanceCreateInfo {
    s_type: u32,
    next: *const c_void,
    flags: u32,
    application: *const ApplicationInfo,
    layer_count: u32,
    layers: *const *const c_char,
    extension_count: u32,
    extensions: *const *const c_char,
}

/// VkPhysicalDeviceProperties2 with VkPhysicalDeviceProperties as bytes: only its first
/// fields are read (apiVersion, driverVersion, vendorID, deviceID, deviceType, deviceName),
/// and the whole structure is 824 bytes
#[repr(C)]
struct Properties2 {
    s_type: u32,
    next: *mut c_void,
    properties: [u8; 1024],
}

#[repr(C)]
struct DriverProperties {
    s_type: u32,
    next: *mut c_void,
    driver_id: u32,
    driver_name: [u8; 256],
    driver_info: [u8; 256],
    conformance: [u8; 4],
}

#[repr(C)]
struct IdProperties {
    s_type: u32,
    next: *mut c_void,
    device_uuid: [u8; 16],
    driver_uuid: [u8; 16],
    device_luid: [u8; 8],
    device_node_mask: u32,
    device_luid_valid: u32,
}

/// VkPhysicalDeviceMemoryProperties: 32 memory types of 8 bytes, then the heap count at 260
/// and 16 heaps of 16 bytes (size, flags) from 264
#[repr(C, align(8))]
struct MemoryProperties([u8; 520]);

type Instance = *mut c_void;
type PhysicalDevice = *mut c_void;
type CreateInstance =
    unsafe extern "system" fn(*const InstanceCreateInfo, *const c_void, *mut Instance) -> i32;
type DestroyInstance = unsafe extern "system" fn(Instance, *const c_void);
type EnumerateDevices = unsafe extern "system" fn(Instance, *mut u32, *mut PhysicalDevice) -> i32;
type GetProperties2 = unsafe extern "system" fn(PhysicalDevice, *mut Properties2);
type GetMemory = unsafe extern "system" fn(PhysicalDevice, *mut MemoryProperties);

/// A registered Vulkan driver (VkDriverId): its name, and who makes it
pub fn driver(id: u32) -> (&'static str, Origin) {
    use Origin::*;
    match id {
        1 => ("AMD proprietary", Official),
        2 => ("AMDVLK", Official),
        3 => ("RADV", Community),
        4 => ("NVIDIA proprietary", Official),
        5 => ("Intel proprietary", Official),
        // ANV, Intel's own driver in Mesa
        6 => ("ANV", Official),
        7 => ("Imagination proprietary", Official),
        8 => ("Qualcomm proprietary", Official),
        9 => ("Arm proprietary", Official),
        10 => ("SwiftShader", Software),
        11 => ("GGP proprietary", Official),
        12 => ("Broadcom proprietary", Official),
        13 => ("llvmpipe", Software),
        14 => ("MoltenVK", Layered),
        15 => ("CoreAVI proprietary", Official),
        16 => ("Juice", Remote),
        17 => ("VeriSilicon proprietary", Official),
        18 => ("Turnip", Community),
        // V3DV, the Raspberry Pi's own driver in Mesa
        19 => ("V3DV", Official),
        20 => ("PanVK", Community),
        21 => ("Samsung proprietary", Official),
        22 => ("Venus", Remote),
        23 => ("Dozen", Layered),
        24 => ("NVK", Community),
        25 => ("PowerVR (Mesa)", Official),
        26 => ("Honeykrisp", Community),
        27 => ("Vulkan SC emulation", Layered),
        28 => ("KosmicKrisp", Layered),
        29 => ("gfxstream", Remote),
        _ => ("unregistered driver", Unknown),
    }
}

/// A Vulkan version number as text
fn version(value: u32) -> String {
    format!("{}.{}.{}", (value >> 22) & 0x7f, (value >> 12) & 0x3ff, value & 0xfff)
}

/// A driver version when the driver gives no text: NVIDIA and Intel on Windows pack it their
/// own way, everyone else as a Vulkan version
fn driver_version(vendor_id: u32, value: u32) -> String {
    match vendor_id {
        0x10de => format!(
            "{}.{}.{}.{}",
            value >> 22,
            (value >> 14) & 0xff,
            (value >> 6) & 0xff,
            value & 0x3f
        ),
        0x8086 if cfg!(windows) => format!("{}.{}", value >> 14, value & 0x3fff),
        _ => version(value),
    }
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

pub fn detect() -> Vec<Api> {
    let Some(library) = load(LIBRARIES) else {
        return Vec::new();
    };
    unsafe {
        let (Ok(create), Ok(destroy), Ok(enumerate), Ok(properties), Ok(memory)) = (
            library.get::<CreateInstance>(b"vkCreateInstance\0"),
            library.get::<DestroyInstance>(b"vkDestroyInstance\0"),
            library.get::<EnumerateDevices>(b"vkEnumeratePhysicalDevices\0"),
            library.get::<GetProperties2>(b"vkGetPhysicalDeviceProperties2\0"),
            library.get::<GetMemory>(b"vkGetPhysicalDeviceMemoryProperties\0"),
        ) else {
            return Vec::new();
        };
        let application = ApplicationInfo {
            s_type: STRUCTURE_TYPE_APPLICATION_INFO,
            next: ptr::null(),
            application_name: c"nectar-detect".as_ptr(),
            application_version: 1,
            engine_name: ptr::null(),
            engine_version: 0,
            api_version: API_VERSION_1_3,
        };
        let info = InstanceCreateInfo {
            s_type: STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            next: ptr::null(),
            flags: 0,
            application: &application,
            layer_count: 0,
            layers: ptr::null(),
            extension_count: 0,
            extensions: ptr::null(),
        };
        let mut instance = ptr::null_mut();
        if create(&info, ptr::null(), &mut instance) != 0 {
            return Vec::new();
        }
        let mut count = 0;
        enumerate(instance, &mut count, ptr::null_mut());
        let mut devices = vec![ptr::null_mut(); count as usize];
        enumerate(instance, &mut count, devices.as_mut_ptr());
        devices.truncate(count as usize);

        let apis = devices
            .into_iter()
            .map(|device| {
                let mut ids: IdProperties = std::mem::zeroed();
                ids.s_type = STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
                let mut driver_properties: DriverProperties = std::mem::zeroed();
                driver_properties.s_type = STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
                driver_properties.next = (&mut ids as *mut IdProperties).cast();
                let mut all: Properties2 = std::mem::zeroed();
                all.s_type = STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
                all.next = (&mut driver_properties as *mut DriverProperties).cast();
                properties(device, &mut all);
                let mut heaps = MemoryProperties([0; 520]);
                memory(device, &mut heaps);
                describe(&all.properties, &driver_properties, &heaps.0)
            })
            .collect();
        destroy(instance, ptr::null());
        apis
    }
}

/// One device, from what the driver filled in
fn describe(properties: &[u8], driver_properties: &DriverProperties, heaps: &[u8]) -> Api {
    let api_version = u32_at(properties, 0);
    let vendor_id = u32_at(properties, 8);
    let device_id = u32_at(properties, 12);
    let device_type = u32_at(properties, 16);
    let device = text(&properties[20..276]);

    let heap_count = (u32_at(heaps, 260) as usize).min(16);
    let local: u64 = (0..heap_count)
        .map(|heap| 264 + heap * 16)
        .filter(|&at| u32_at(heaps, at + 8) & MEMORY_HEAP_DEVICE_LOCAL != 0)
        .map(|at| u64::from_le_bytes(heaps[at..at + 8].try_into().unwrap()))
        .sum();

    let (mut name, mut origin) = driver(driver_properties.driver_id);
    let reported = text(&driver_properties.driver_name);
    if driver_properties.driver_id == 0 {
        // A Vulkan 1.0/1.1 driver without VK_KHR_driver_properties
        name = "unknown driver";
        if device_type == DEVICE_TYPE_CPU || vendor_id == VENDOR_MESA {
            origin = Origin::Software;
        }
    }
    // Nectar's own Vulkan driver, once there is one, names its devices this way
    if device.ends_with("(Nectar)") {
        origin = Origin::Remote;
    }
    let info = text(&driver_properties.driver_info);
    let c = driver_properties.conformance;
    Api {
        kind: ApiKind::Vulkan,
        version: version(api_version),
        device,
        vendor_id,
        device_id,
        memory_mib: local >> 20,
        driver: if reported.is_empty() || name.to_lowercase().contains(&reported.to_lowercase()) {
            name.to_string()
        } else {
            format!("{name} ({reported})")
        },
        driver_version: if info.is_empty() {
            driver_version(vendor_id, u32_at(properties, 4))
        } else {
            info
        },
        origin,
        conformance: if c == [0; 4] {
            String::new()
        } else {
            format!("{}.{}.{}.{}", c[0], c[1], c[2], c[3])
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn community_drivers_are_recognized() {
        assert_eq!(driver(24), ("NVK", Origin::Community));
        assert_eq!(driver(3), ("RADV", Origin::Community));
        assert_eq!(driver(26), ("Honeykrisp", Origin::Community));
        assert_eq!(driver(4).1, Origin::Official);
        assert_eq!(driver(13).1, Origin::Software);
        assert_eq!(driver(28).1, Origin::Layered);
        assert_eq!(driver(999).1, Origin::Unknown);
    }

    #[test]
    fn versions() {
        assert_eq!(version((1 << 22) | (4 << 12) | 312), "1.4.312");
        // NVIDIA's packing: 10 bits, 8, 8, 6
        assert_eq!(driver_version(0x10de, (591 << 22) | (1 << 14)), "591.1.0.0");
    }

    #[test]
    fn describes_a_device() {
        let mut properties = [0u8; 1024];
        properties[0..4].copy_from_slice(&((1u32 << 22) | (4 << 12) | 312).to_le_bytes());
        properties[8..12].copy_from_slice(&0x10deu32.to_le_bytes());
        properties[12..16].copy_from_slice(&0x2f04u32.to_le_bytes());
        properties[16..20].copy_from_slice(&2u32.to_le_bytes());
        properties[20..20 + 23].copy_from_slice(b"NVIDIA GeForce RTX 5070");
        let mut driver_properties: DriverProperties = unsafe { std::mem::zeroed() };
        driver_properties.driver_id = 24;
        driver_properties.driver_name[..3].copy_from_slice(b"NVK");
        driver_properties.driver_info[..11].copy_from_slice(b"Mesa 25.2.8");
        driver_properties.conformance = [1, 4, 0, 0];
        let mut heaps = [0u8; 520];
        heaps[260..264].copy_from_slice(&2u32.to_le_bytes());
        heaps[264..272].copy_from_slice(&(12u64 << 30).to_le_bytes());
        heaps[272..276].copy_from_slice(&1u32.to_le_bytes());
        heaps[280..288].copy_from_slice(&(32u64 << 30).to_le_bytes());

        let api = describe(&properties, &driver_properties, &heaps);
        assert_eq!(api.device, "NVIDIA GeForce RTX 5070");
        assert_eq!(api.version, "1.4.312");
        assert_eq!((api.vendor_id, api.device_id), (0x10de, 0x2f04));
        assert_eq!(api.memory_mib, 12 << 10);
        assert_eq!(api.driver, "NVK");
        assert_eq!(api.driver_version, "Mesa 25.2.8");
        assert_eq!(api.origin, Origin::Community);
        assert_eq!(api.conformance, "1.4.0.0");
    }
}
