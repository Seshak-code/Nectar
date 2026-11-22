//! The computer's network link: whether a connection crosses Wi-Fi, how good that Wi-Fi is, and
//! holding it in low-latency mode while a GPU is lent or borrowed.
//!
//! A Wi-Fi card normally leaves its channel now and then to look for better access points, and
//! naps between packets to save power. Either stalls a connection for tens of milliseconds, which
//! a frame or a token notices. On Windows, the Wi-Fi service's background scan can be turned off
//! and its media streaming mode on, by any program, for as long as it asks: both go back when the
//! holder lets go or the Wi-Fi disconnects. Elsewhere these need other means, and nothing is held.

use std::net::IpAddr;

/// What a connection's local address crosses
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Kind {
    Loopback,
    Wifi,
    Wired,
    #[default]
    Unknown,
}

impl Kind {
    pub fn key(self) -> &'static str {
        match self {
            Kind::Loopback => "loopback",
            Kind::Wifi => "wifi",
            Kind::Wired => "wired",
            Kind::Unknown => "unknown",
        }
    }

    pub fn parse(key: &str) -> Kind {
        match key {
            "loopback" => Kind::Loopback,
            "wifi" => Kind::Wifi,
            "wired" => Kind::Wired,
            _ => Kind::Unknown,
        }
    }
}

/// A connected Wi-Fi interface
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Link {
    pub interface: String,
    /// Such as "Wi-Fi 7"; empty when unknown
    pub standard: String,
    pub receive_mbps: u32,
    pub transmit_mbps: u32,
    /// 0 to 100
    pub signal: u32,
}

/// What `local`, a connection's local address, crosses
pub fn kind_of(local: IpAddr) -> Kind {
    if local.is_loopback() { Kind::Loopback } else { platform::kind_of(local) }
}

/// The connected Wi-Fi interfaces. Only reads.
pub fn links() -> Vec<Link> {
    platform::links()
}

/// Low-latency mode, until dropped
pub use platform::Hold;

/// Holds Wi-Fi in low-latency mode for a connection between `local` and `peer`, when it crosses
/// Wi-Fi. NECTAR_WIFI_TUNE=0 leaves it alone.
pub fn hold(local: IpAddr, peer: IpAddr) -> Option<Hold> {
    if peer.is_loopback() || std::env::var("NECTAR_WIFI_TUNE").is_ok_and(|value| value == "0") {
        return None;
    }
    (kind_of(local) == Kind::Wifi).then(platform::hold).flatten()
}

#[cfg(windows)]
mod platform {
    use std::ffi::c_void;
    use std::net::IpAddr;
    use std::sync::OnceLock;

    use libloading::os::windows::{LOAD_LIBRARY_SEARCH_SYSTEM32, Library};
    use windows_sys::Win32::Foundation::HANDLE;
    use windows_sys::Win32::NetworkManagement::IpHelper::{
        GAA_FLAG_SKIP_ANYCAST, GAA_FLAG_SKIP_DNS_SERVER, GAA_FLAG_SKIP_FRIENDLY_NAME, GAA_FLAG_SKIP_MULTICAST,
        GetAdaptersAddresses, IP_ADAPTER_ADDRESSES_LH,
    };
    use windows_sys::Win32::NetworkManagement::WiFi::{
        WLAN_CONNECTION_ATTRIBUTES, WLAN_INTERFACE_INFO_LIST, wlan_interface_state_connected,
        wlan_intf_opcode_background_scan_enabled, wlan_intf_opcode_current_connection,
        wlan_intf_opcode_media_streaming_mode,
    };
    use windows_sys::core::GUID;

    use super::{Kind, Link};

    const AF_UNSPEC: u32 = 0;
    const AF_INET: u16 = 2;
    const AF_INET6: u16 = 23;
    const IF_TYPE_ETHERNET: u32 = 6;
    const IF_TYPE_IEEE80211: u32 = 71;

    pub fn kind_of(local: IpAddr) -> Kind {
        let flags = GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST | GAA_FLAG_SKIP_DNS_SERVER | GAA_FLAG_SKIP_FRIENDLY_NAME;
        let mut size = 32_768u32;
        // u64s, for the structure's alignment
        let mut buffer = vec![0u64; size as usize / 8];
        let first = buffer.as_mut_ptr().cast::<IP_ADAPTER_ADDRESSES_LH>();
        if unsafe { GetAdaptersAddresses(AF_UNSPEC, flags, std::ptr::null(), first, &mut size) } != 0 {
            return Kind::Unknown;
        }
        let mut adapter = first as *const IP_ADAPTER_ADDRESSES_LH;
        while let Some(current) = unsafe { adapter.as_ref() } {
            let mut address = current.FirstUnicastAddress as *const _;
            while let Some(unicast) = unsafe { (address as *const windows_sys::Win32::NetworkManagement::IpHelper::IP_ADAPTER_UNICAST_ADDRESS_LH).as_ref() } {
                if unsafe { matches(unicast.Address.lpSockaddr.cast(), local) } {
                    return match current.IfType {
                        IF_TYPE_IEEE80211 => Kind::Wifi,
                        IF_TYPE_ETHERNET => Kind::Wired,
                        _ => Kind::Unknown,
                    };
                }
                address = unicast.Next;
            }
            adapter = current.Next;
        }
        Kind::Unknown
    }

    /// Whether a sockaddr holds `ip`
    unsafe fn matches(address: *const u8, ip: IpAddr) -> bool {
        if address.is_null() {
            return false;
        }
        unsafe {
            let family = (address as *const u16).read_unaligned();
            match ip {
                IpAddr::V4(ip) if family == AF_INET => std::slice::from_raw_parts(address.add(4), 4) == ip.octets(),
                IpAddr::V6(ip) if family == AF_INET6 => std::slice::from_raw_parts(address.add(8), 16) == ip.octets(),
                _ => false,
            }
        }
    }

    type Open = unsafe extern "system" fn(u32, *const c_void, *mut u32, *mut HANDLE) -> u32;
    type Close = unsafe extern "system" fn(HANDLE, *const c_void) -> u32;
    type Enumerate = unsafe extern "system" fn(HANDLE, *const c_void, *mut *mut WLAN_INTERFACE_INFO_LIST) -> u32;
    type Query = unsafe extern "system" fn(HANDLE, *const GUID, i32, *const c_void, *mut u32, *mut *mut c_void, *mut i32) -> u32;
    type Set = unsafe extern "system" fn(HANDLE, *const GUID, i32, u32, *const c_void, *const c_void) -> u32;
    type Free = unsafe extern "system" fn(*const c_void);

    struct Api {
        // Stays loaded: unloading it while another thread opens a client can hang
        _library: Library,
        open: Open,
        close: Close,
        enumerate: Enumerate,
        query: Query,
        set: Set,
        free: Free,
    }

    /// The Wi-Fi service's API, loaded here since a PC without Wi-Fi may not have its DLL
    fn api() -> Option<&'static Api> {
        static API: OnceLock<Option<Api>> = OnceLock::new();
        API.get_or_init(|| unsafe {
            let library = Library::load_with_flags("wlanapi.dll", LOAD_LIBRARY_SEARCH_SYSTEM32).ok()?;
            Some(Api {
                open: *library.get::<Open>(b"WlanOpenHandle\0").ok()?,
                close: *library.get::<Close>(b"WlanCloseHandle\0").ok()?,
                enumerate: *library.get::<Enumerate>(b"WlanEnumInterfaces\0").ok()?,
                query: *library.get::<Query>(b"WlanQueryInterface\0").ok()?,
                set: *library.get::<Set>(b"WlanSetInterface\0").ok()?,
                free: *library.get::<Free>(b"WlanFreeMemory\0").ok()?,
                _library: library,
            })
        })
        .as_ref()
    }

    /// A client of the Wi-Fi service
    struct Client {
        api: &'static Api,
        handle: HANDLE,
    }

    /// wlan_intf_opcode_realtime_connection_quality, which unlike the current connection needs no
    /// location permission (Windows 11 24H2 and later): the start of
    /// WLAN_REALTIME_CONNECTION_QUALITY
    const REALTIME_QUALITY: i32 = 19;
    #[repr(C)]
    struct RealtimeQuality {
        phy: i32,
        link_quality: u32,
        receive_kbps: u32,
        transmit_kbps: u32,
    }

    impl Client {
        fn open() -> Option<Client> {
            let api = api()?;
            let (mut version, mut handle) = (0u32, std::ptr::null_mut());
            (unsafe { (api.open)(2, std::ptr::null(), &mut version, &mut handle) } == 0).then_some(Client { api, handle })
        }

        /// Asks about an interface; `read` gets the answer's start
        fn query<T>(&self, interface: &GUID, opcode: i32, read: impl FnOnce(&T) -> Link) -> Option<Link> {
            let (mut size, mut data, mut kind) = (0u32, std::ptr::null_mut(), 0i32);
            let queried = unsafe {
                (self.api.query)(self.handle, interface, opcode, std::ptr::null(), &mut size, &mut data, &mut kind)
            };
            if queried != 0 || data.is_null() {
                return None;
            }
            let link = (size as usize >= size_of::<T>()).then(|| read(unsafe { &*data.cast::<T>() }));
            unsafe { (self.api.free)(data) };
            link
        }

        /// The connected interfaces: their ids and names
        fn connected(&self) -> Vec<(GUID, String)> {
            let mut list = std::ptr::null_mut();
            let mut found = Vec::new();
            unsafe {
                if (self.api.enumerate)(self.handle, std::ptr::null(), &mut list) != 0 || list.is_null() {
                    return found;
                }
                let items = std::slice::from_raw_parts((*list).InterfaceInfo.as_ptr(), (*list).dwNumberOfItems as usize);
                for item in items.iter().filter(|item| item.isState == wlan_interface_state_connected) {
                    let name = &item.strInterfaceDescription;
                    let end = name.iter().position(|&unit| unit == 0).unwrap_or(name.len());
                    found.push((item.InterfaceGuid, String::from_utf16_lossy(&name[..end])));
                }
                (self.api.free)(list.cast());
            }
            found
        }

        fn set_flag(&self, interface: &GUID, opcode: i32, value: bool) -> bool {
            let value = value as i32;
            unsafe { (self.api.set)(self.handle, interface, opcode, 4, (&value as *const i32).cast(), std::ptr::null()) == 0 }
        }
    }

    impl Drop for Client {
        fn drop(&mut self) {
            unsafe { (self.api.close)(self.handle, std::ptr::null()) };
        }
    }

    fn standard(phy: i32) -> &'static str {
        match phy {
            11 => "Wi-Fi 7",
            10 => "Wi-Fi 6",
            8 => "Wi-Fi 5",
            7 => "Wi-Fi 4",
            _ => "",
        }
    }

    pub fn links() -> Vec<Link> {
        let Some(client) = Client::open() else { return Vec::new() };
        let mut links = Vec::new();
        for (interface, name) in client.connected() {
            let quality = client.query(&interface, REALTIME_QUALITY, |quality: &RealtimeQuality| Link {
                standard: standard(quality.phy).to_string(),
                receive_mbps: quality.receive_kbps / 1000,
                transmit_mbps: quality.transmit_kbps / 1000,
                signal: quality.link_quality,
                ..Link::default()
            });
            let details = quality.or_else(|| {
                client.query(&interface, wlan_intf_opcode_current_connection, |connection: &WLAN_CONNECTION_ATTRIBUTES| {
                    let attributes = &connection.wlanAssociationAttributes;
                    Link {
                        standard: standard(attributes.dot11PhyType).to_string(),
                        receive_mbps: attributes.ulRxRate / 1000,
                        transmit_mbps: attributes.ulTxRate / 1000,
                        signal: attributes.wlanSignalQuality,
                        ..Link::default()
                    }
                })
            });
            // Without location permission Windows may tell neither
            links.push(Link { interface: name, ..details.unwrap_or_default() });
        }
        links
    }

    /// Low-latency mode on the interfaces it was set on, until dropped
    pub struct Hold {
        client: Client,
        interfaces: Vec<GUID>,
    }

    // The handle is only used to let go, from whichever thread drops this
    unsafe impl Send for Hold {}

    pub fn hold() -> Option<Hold> {
        let client = Client::open()?;
        let mut interfaces = Vec::new();
        for (interface, _) in client.connected() {
            let streaming = client.set_flag(&interface, wlan_intf_opcode_media_streaming_mode, true);
            let quiet = client.set_flag(&interface, wlan_intf_opcode_background_scan_enabled, false);
            if streaming || quiet {
                interfaces.push(interface);
            }
        }
        (!interfaces.is_empty()).then_some(Hold { client, interfaces })
    }

    impl Drop for Hold {
        fn drop(&mut self) {
            for interface in &self.interfaces {
                self.client.set_flag(interface, wlan_intf_opcode_media_streaming_mode, false);
                self.client.set_flag(interface, wlan_intf_opcode_background_scan_enabled, true);
            }
        }
    }
}

#[cfg(unix)]
mod platform {
    use std::net::IpAddr;

    use super::{Kind, Link};

    /// The interface holding `local`
    fn interface_of(local: IpAddr) -> Option<String> {
        let mut first: *mut libc::ifaddrs = std::ptr::null_mut();
        if unsafe { libc::getifaddrs(&mut first) } != 0 {
            return None;
        }
        let mut found = None;
        let mut current = first;
        while let Some(entry) = unsafe { current.as_ref() } {
            if unsafe { holds(entry.ifa_addr, local) } {
                found = Some(unsafe { std::ffi::CStr::from_ptr(entry.ifa_name) }.to_string_lossy().into_owned());
                break;
            }
            current = entry.ifa_next;
        }
        unsafe { libc::freeifaddrs(first) };
        found
    }

    unsafe fn holds(address: *const libc::sockaddr, ip: IpAddr) -> bool {
        let Some(family) = (unsafe { address.as_ref() }).map(|address| address.sa_family as i32) else { return false };
        unsafe {
            match ip {
                IpAddr::V4(ip) if family == libc::AF_INET => {
                    (*address.cast::<libc::sockaddr_in>()).sin_addr.s_addr.to_ne_bytes() == ip.octets()
                }
                IpAddr::V6(ip) if family == libc::AF_INET6 => (*address.cast::<libc::sockaddr_in6>()).sin6_addr.s6_addr == ip.octets(),
                _ => false,
            }
        }
    }

    pub fn kind_of(local: IpAddr) -> Kind {
        let Some(interface) = interface_of(local) else { return Kind::Unknown };
        if cfg!(target_os = "linux") {
            let folder = std::path::Path::new("/sys/class/net").join(&interface);
            if folder.join("wireless").exists() {
                Kind::Wifi
            } else if folder.join("device").exists() {
                Kind::Wired
            } else {
                Kind::Unknown
            }
        } else {
            Kind::Unknown
        }
    }

    /// The signal from /proc/net/wireless, on Linux; rates need the kernel's nl80211
    pub fn links() -> Vec<Link> {
        let Ok(text) = std::fs::read_to_string("/proc/net/wireless") else { return Vec::new() };
        text.lines().skip(2).filter_map(parse_wireless).collect()
    }

    /// `wlan0: 0000   54.  -56.  -256 ...`, whose link quality is out of 70
    fn parse_wireless(line: &str) -> Option<Link> {
        let (interface, rest) = line.split_once(':')?;
        let quality: f32 = rest.split_whitespace().nth(1)?.trim_end_matches('.').parse().ok()?;
        Some(Link {
            interface: interface.trim().to_string(),
            signal: (quality * 100.0 / 70.0).min(100.0) as u32,
            ..Link::default()
        })
    }

    pub struct Hold;

    pub fn hold() -> Option<Hold> {
        None
    }

    #[cfg(test)]
    #[test]
    fn proc_net_wireless_parses() {
        let link = parse_wireless("wlan0: 0000   54.  -56.  -256        0      0      0      0     12        0").unwrap();
        assert_eq!(link.interface, "wlan0");
        assert_eq!(link.signal, 77);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loopback_is_never_held() {
        let loopback: IpAddr = "127.0.0.1".parse().unwrap();
        assert_eq!(kind_of(loopback), Kind::Loopback);
        assert!(hold(loopback, loopback).is_none());
    }

    #[test]
    fn kinds_have_names() {
        for kind in [Kind::Loopback, Kind::Wifi, Kind::Wired, Kind::Unknown] {
            assert_eq!(Kind::parse(kind.key()), kind);
        }
    }

    #[test]
    fn reading_the_links_is_safe_anywhere() {
        for link in links() {
            assert!(link.signal <= 100 && !link.interface.is_empty());
        }
    }
}
