//! Finds a computer's GPUs and what their drivers offer, without any SDK: each API's library is
//! loaded at run time, and a missing library means that API isn't offered.
//!
//! - `adapters`: the GPUs the OS knows and their kernel drivers (SetupAPI on Windows, sysfs on
//!   Linux)
//! - `cuda`, `vulkan`, `opencl`, `hip`, `level_zero`: one module per API
//!
//! Vulkan tells drivers apart by a registered id (`VkDriverId`), which is how community drivers
//! such as NVK, RADV or Honeykrisp are recognized. A borrowed GPU (Nectar's own nvcuda) is
//! reported as remote and never initialized.

pub mod adapters;
pub mod cuda;
pub mod hip;
pub mod level_zero;
pub mod opencl;
pub mod vulkan;

use std::ffi::c_char;
use std::fmt;

/// Who made the driver behind an API
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    /// The GPU's vendor
    Official,
    /// An open-source driver the vendor doesn't make, such as NVK, RADV or Honeykrisp
    Community,
    /// A translation onto another API, such as MoltenVK (onto Metal) or Dozen (onto D3D12)
    Layered,
    /// Runs on the CPU
    Software,
    /// A GPU on another computer: Nectar, or another remoting driver
    Remote,
    Unknown,
}

impl Origin {
    pub fn name(self) -> &'static str {
        match self {
            Origin::Official => "official",
            Origin::Community => "community",
            Origin::Layered => "layered",
            Origin::Software => "software",
            Origin::Remote => "remote",
            Origin::Unknown => "unknown",
        }
    }

    pub fn parse(name: &str) -> Self {
        match name {
            "official" => Origin::Official,
            "community" => Origin::Community,
            "layered" => Origin::Layered,
            "software" => Origin::Software,
            "remote" => Origin::Remote,
            _ => Origin::Unknown,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ApiKind {
    Cuda,
    Vulkan,
    OpenCl,
    Hip,
    LevelZero,
}

impl ApiKind {
    pub const ALL: [ApiKind; 5] = [
        ApiKind::Cuda,
        ApiKind::Vulkan,
        ApiKind::OpenCl,
        ApiKind::Hip,
        ApiKind::LevelZero,
    ];

    /// The short name used in discovery and borrow.cfg
    pub fn key(self) -> &'static str {
        match self {
            ApiKind::Cuda => "cuda",
            ApiKind::Vulkan => "vulkan",
            ApiKind::OpenCl => "opencl",
            ApiKind::Hip => "hip",
            ApiKind::LevelZero => "levelzero",
        }
    }

    pub fn parse(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.key() == key)
    }
}

impl fmt::Display for ApiKind {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(match self {
            ApiKind::Cuda => "CUDA",
            ApiKind::Vulkan => "Vulkan",
            ApiKind::OpenCl => "OpenCL",
            ApiKind::Hip => "HIP",
            ApiKind::LevelZero => "Level Zero",
        })
    }
}

/// One device as one API sees it
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Api {
    pub kind: ApiKind,
    /// The API version the driver supports, such as 13.5 for CUDA or 1.4.312 for Vulkan
    pub version: String,
    /// The device's name as the API reports it
    pub device: String,
    /// PCI vendor id, 0 when the API doesn't say
    pub vendor_id: u32,
    /// PCI device id, 0 when the API doesn't say
    pub device_id: u32,
    pub memory_mib: u64,
    /// Which driver implements the API, such as "NVIDIA proprietary" or "NVK"
    pub driver: String,
    pub driver_version: String,
    pub origin: Origin,
    /// The Khronos conformance version the driver claims (Vulkan), empty otherwise
    pub conformance: String,
}

/// A GPU the operating system knows, with its kernel driver
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Adapter {
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    /// The kernel driver: a Windows service name such as nvlddmkm, or a Linux module such as
    /// amdgpu, nouveau or xe
    pub kernel_driver: String,
    pub driver_version: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Inventory {
    pub os: &'static str,
    pub arch: &'static str,
    pub adapters: Vec<Adapter>,
    pub apis: Vec<Api>,
}

/// Everything this computer offers. Loads each API's library and initializes it, so it can
/// take a second.
pub fn inventory() -> Inventory {
    let mut apis = Vec::new();
    apis.extend(cuda::detect());
    apis.extend(vulkan::detect());
    apis.extend(opencl::detect());
    apis.extend(hip::detect());
    apis.extend(level_zero::detect());
    Inventory {
        os: std::env::consts::OS,
        arch: std::env::consts::ARCH,
        adapters: adapters::detect(),
        apis,
    }
}

impl Inventory {
    /// The APIs of local GPUs, in the order a lender offers them: real GPUs first, then
    /// translations, then software; never remote ones
    pub fn lendable(&self) -> Vec<&Api> {
        let rank = |origin: Origin| match origin {
            Origin::Official => 0,
            Origin::Community => 1,
            Origin::Unknown => 2,
            Origin::Layered => 3,
            Origin::Software => 4,
            Origin::Remote => 5,
        };
        let mut apis: Vec<&Api> = self
            .apis
            .iter()
            .filter(|api| api.origin != Origin::Remote)
            .collect();
        apis.sort_by_key(|api| (rank(api.origin), api.kind));
        apis
    }

    /// The adapter an API's device is, if the OS lists it
    pub fn adapter_of(&self, api: &Api) -> Option<&Adapter> {
        if api.vendor_id == 0 || api.origin == Origin::Remote {
            return None;
        }
        let same_vendor = || self.adapters.iter().filter(|a| a.vendor_id == api.vendor_id);
        same_vendor()
            .find(|a| api.device_id != 0 && a.device_id == api.device_id)
            .or_else(|| (same_vendor().count() == 1).then(|| same_vendor().next()).flatten())
    }

    /// A readable report, grouped by GPU
    pub fn report(&self) -> String {
        let mut out = format!("This computer: {} {}\n", self.os, self.arch);
        let mut shown = vec![false; self.apis.len()];
        for adapter in &self.adapters {
            out += &format!(
                "\n{} ({} {:04x}:{:04x})\n  Kernel driver: {} {}\n",
                if adapter.name.is_empty() { "GPU" } else { &adapter.name },
                vendor_name(adapter.vendor_id),
                adapter.vendor_id,
                adapter.device_id,
                if adapter.kernel_driver.is_empty() { "unknown" } else { &adapter.kernel_driver },
                adapter.driver_version,
            );
            let mut any = false;
            for (index, api) in self.apis.iter().enumerate() {
                if !shown[index] && self.adapter_of(api) == Some(adapter) {
                    shown[index] = true;
                    any = true;
                    out += &api_line(api);
                }
            }
            if !any {
                out += "  No API found for it: no user-mode driver, or one Nectar doesn't know\n";
            }
        }
        let rest: Vec<&Api> = self
            .apis
            .iter()
            .zip(&shown)
            .filter(|(_, shown)| !**shown)
            .map(|(api, _)| api)
            .collect();
        if !rest.is_empty() {
            out += "\nNot matched to a GPU above (remote, software, or not listed by the OS)\n";
            for api in rest {
                out += &api_line(api);
            }
        }
        if self.apis.is_empty() {
            out += "\nNo GPU API found\n";
        }
        out
    }
}

fn api_line(api: &Api) -> String {
    let mut line = format!("  {:<11} {:<10} {}", api.kind.to_string(), api.version, api.device);
    if api.memory_mib > 0 {
        line += &format!(", {} MiB", api.memory_mib);
    }
    line += &format!("\n              {}", api.driver);
    if !api.driver_version.is_empty() {
        line += &format!(" {}", api.driver_version);
    }
    line += &format!(", {}", api.origin.name());
    if !api.conformance.is_empty() {
        line += &format!(", conformance {}", api.conformance);
    }
    line + "\n"
}

/// A PCI vendor's name
pub fn vendor_name(id: u32) -> &'static str {
    match id {
        0x10de => "NVIDIA",
        0x1002 | 0x1022 => "AMD",
        0x8086 => "Intel",
        0x106b => "Apple",
        0x5143 => "Qualcomm",
        0x13b5 => "Arm",
        0x14e4 => "Broadcom",
        0x1010 => "Imagination",
        0x1414 => "Microsoft",
        0x10005 => "Mesa",
        _ => "unknown vendor",
    }
}

/// A NUL-terminated C string in a buffer, as text
pub(crate) fn text(buffer: &[u8]) -> String {
    let end = buffer.iter().position(|&byte| byte == 0).unwrap_or(buffer.len());
    String::from_utf8_lossy(&buffer[..end]).trim().to_string()
}

pub(crate) fn text_ptr(buffer: &[c_char]) -> String {
    let bytes: Vec<u8> = buffer.iter().map(|&c| c as u8).collect();
    text(&bytes)
}

/// The first of `names` that loads
pub(crate) fn load(names: &[&str]) -> Option<libloading::Library> {
    names
        .iter()
        .find_map(|name| unsafe { libloading::Library::new(name) }.ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn api(kind: ApiKind, vendor_id: u32, device_id: u32, origin: Origin) -> Api {
        Api {
            kind,
            version: String::new(),
            device: String::new(),
            vendor_id,
            device_id,
            memory_mib: 0,
            driver: String::new(),
            driver_version: String::new(),
            origin,
            conformance: String::new(),
        }
    }

    #[test]
    fn apis_match_adapters_by_vendor_and_device() {
        let nvidia = Adapter { vendor_id: 0x10de, device_id: 0x2f04, ..Default::default() };
        let intel = Adapter { vendor_id: 0x8086, device_id: 0x64a0, ..Default::default() };
        let inventory = Inventory {
            os: "windows",
            arch: "x86_64",
            adapters: vec![nvidia.clone(), intel.clone()],
            apis: vec![
                api(ApiKind::Cuda, 0x10de, 0, Origin::Official),
                api(ApiKind::Vulkan, 0x8086, 0x64a0, Origin::Official),
                api(ApiKind::Vulkan, 0x10005, 0, Origin::Software),
                api(ApiKind::Cuda, 0x10de, 0, Origin::Remote),
            ],
        };
        // One NVIDIA adapter, so CUDA (which gives no device id) is it
        assert_eq!(inventory.adapter_of(&inventory.apis[0]), Some(&nvidia));
        assert_eq!(inventory.adapter_of(&inventory.apis[1]), Some(&intel));
        assert_eq!(inventory.adapter_of(&inventory.apis[2]), None);

        let lendable: Vec<(ApiKind, Origin)> =
            inventory.lendable().iter().map(|api| (api.kind, api.origin)).collect();
        assert_eq!(
            lendable,
            [
                (ApiKind::Cuda, Origin::Official),
                (ApiKind::Vulkan, Origin::Official),
                (ApiKind::Vulkan, Origin::Software)
            ]
        );
    }

    #[test]
    fn names_round_trip() {
        for kind in ApiKind::ALL {
            assert_eq!(ApiKind::parse(kind.key()), Some(kind));
        }
        for origin in [Origin::Official, Origin::Community, Origin::Layered, Origin::Software, Origin::Remote] {
            assert_eq!(Origin::parse(origin.name()), origin);
        }
    }
}
