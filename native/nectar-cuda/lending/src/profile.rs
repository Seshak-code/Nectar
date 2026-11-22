//! What the user borrows a GPU for: gaming and rendering, or AI and compute. One app, one lease,
//! one Borrow button; the choice decides how the link is tuned, and both drivers and the lender
//! follow it.
//!
//! | | Gaming & rendering | AI & compute |
//! | --- | --- | --- |
//! | Socket buffers | 512 KiB up, 1 MiB down: a frame's commands don't wait behind older data | 4 MiB each way: results and tensors cross in bulk |
//! | Wi-Fi queue | Video (AC_VI), which contends for the air sooner | Best effort (AC_BE), whose transmissions have no time limit but the frame's |
//!
//! Whichever is chosen, a Wi-Fi link is held in low-latency mode while the GPU is borrowed
//! (wifi.rs). A socket is tuned once, as it connects, so a new choice reaches programs started
//! after it.

use std::net::TcpStream;
use std::{env, fs, io};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Profile {
    Gaming,
    /// The default: what works through Nectar with any lender that has CUDA
    #[default]
    Ai,
}

impl Profile {
    pub const ALL: [Profile; 2] = [Profile::Gaming, Profile::Ai];

    /// How files, commands and the wire name it
    pub fn key(self) -> &'static str {
        match self {
            Profile::Gaming => "gaming",
            Profile::Ai => "ai",
        }
    }

    pub fn parse(key: &str) -> Option<Profile> {
        Profile::ALL.into_iter().find(|profile| profile.key() == key.trim())
    }

    /// How the app names it
    pub fn label(self) -> &'static str {
        match self {
            Profile::Gaming => "Gaming & rendering",
            Profile::Ai => "AI & compute",
        }
    }

    pub fn tuning(self) -> Tuning {
        match self {
            Profile::Gaming => Tuning {
                send_buffer: 512 << 10,
                receive_buffer: 1 << 20,
                video_queue: true,
            },
            Profile::Ai => Tuning {
                send_buffer: 4 << 20,
                receive_buffer: 4 << 20,
                video_queue: false,
            },
        }
    }

    /// What this choice means on a lender offering `apis`, when the user should know
    pub fn note(self, apis: &[String]) -> Option<String> {
        let has = |api: &str| apis.iter().any(|offered| offered == api);
        match self {
            Profile::Gaming if !has("vulkan") => {
                Some("This lender offers no Vulkan, so games can't use its GPU.".to_string())
            }
            Profile::Ai if !has("cuda") && has("vulkan") => Some(
                "This GPU has no CUDA: use AI programs built for Vulkan, such as llama.cpp's Vulkan build."
                    .to_string(),
            ),
            _ => None,
        }
    }

    fn file() -> std::path::PathBuf {
        crate::home().join("profile")
    }

    /// The user's choice, which nectar-svc keeps
    pub fn saved() -> Profile {
        fs::read_to_string(Self::file()).ok().and_then(|text| Profile::parse(&text)).unwrap_or_default()
    }

    pub fn save(self) -> io::Result<()> {
        fs::create_dir_all(crate::home())?;
        fs::write(Self::file(), self.key())
    }

    /// What this program's connections follow: NECTAR_PROFILE for testing, otherwise borrow.cfg
    pub fn current() -> Profile {
        env::var("NECTAR_PROFILE")
            .ok()
            .and_then(|key| Profile::parse(&key))
            .or_else(|| crate::BorrowConfig::read().map(|config| config.profile))
            .unwrap_or_default()
    }
}

/// A profile, as a connection's socket gets it
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tuning {
    pub send_buffer: usize,
    pub receive_buffer: usize,
    /// Mark the traffic for Wi-Fi's video queue
    pub video_queue: bool,
}

/// Tunes a connected socket for `profile`. Loopback is left as it is: there's no air to tune for.
pub fn tune(stream: &TcpStream, profile: Profile) {
    if stream.peer_addr().is_ok_and(|peer| peer.ip().is_loopback()) {
        return;
    }
    apply(stream, profile.tuning());
}

fn apply(stream: &TcpStream, tuning: Tuning) {
    platform::buffers(stream, tuning.send_buffer, tuning.receive_buffer);
    if tuning.video_queue {
        platform::video_queue(stream);
    }
}

#[cfg(windows)]
mod platform {
    use std::ffi::c_void;
    use std::net::TcpStream;
    use std::os::windows::io::AsRawSocket;
    use std::sync::OnceLock;

    use libloading::os::windows::{LOAD_LIBRARY_SEARCH_SYSTEM32, Library};
    use windows_sys::Win32::Networking::WinSock::{SO_RCVBUF, SO_SNDBUF, SOL_SOCKET, setsockopt};

    pub fn buffers(stream: &TcpStream, send: usize, receive: usize) {
        let socket = stream.as_raw_socket() as usize;
        for (option, size) in [(SO_SNDBUF, send), (SO_RCVBUF, receive)] {
            let size = size as i32;
            unsafe { setsockopt(socket, SOL_SOCKET, option, (&size as *const i32).cast(), 4) };
        }
    }

    type CreateHandle = unsafe extern "system" fn(*const [u16; 2], *mut *mut c_void) -> i32;
    type AddSocket = unsafe extern "system" fn(*mut c_void, usize, *const c_void, i32, u32, *mut u32) -> i32;

    struct Qos {
        _library: Library,
        handle: usize,
        add: AddSocket,
    }

    /// Windows' qWAVE, loaded here since not every edition has it
    fn qos() -> Option<&'static Qos> {
        static QOS: OnceLock<Option<Qos>> = OnceLock::new();
        QOS.get_or_init(|| unsafe {
            let library = Library::load_with_flags("qwave.dll", LOAD_LIBRARY_SEARCH_SYSTEM32).ok()?;
            let create = *library.get::<CreateHandle>(b"QOSCreateHandle\0").ok()?;
            let add = *library.get::<AddSocket>(b"QOSAddSocketToFlow\0").ok()?;
            let mut handle = std::ptr::null_mut();
            (create(&[1, 0], &mut handle) != 0).then_some(Qos { _library: library, handle: handle as usize, add })
        })
        .as_ref()
    }

    /// qWAVE's audio and video class marks packets with DSCP 40, which Wi-Fi puts in its video
    /// queue, and needs no administrator
    pub fn video_queue(stream: &TcpStream) {
        const QOS_TRAFFIC_TYPE_AUDIO_VIDEO: i32 = 3;
        const QOS_NON_ADAPTIVE_FLOW: u32 = 2;
        if let Some(qos) = qos() {
            let mut flow = 0u32;
            let socket = stream.as_raw_socket() as usize;
            unsafe {
                (qos.add)(
                    qos.handle as *mut c_void,
                    socket,
                    std::ptr::null(),
                    QOS_TRAFFIC_TYPE_AUDIO_VIDEO,
                    QOS_NON_ADAPTIVE_FLOW,
                    &mut flow,
                )
            };
        }
    }
}

#[cfg(unix)]
mod platform {
    use std::net::TcpStream;
    use std::os::fd::AsRawFd;

    fn set(stream: &TcpStream, level: i32, option: i32, value: i32) {
        unsafe {
            libc::setsockopt(
                stream.as_raw_fd(),
                level,
                option,
                (&value as *const i32).cast(),
                size_of::<i32>() as libc::socklen_t,
            )
        };
    }

    pub fn buffers(stream: &TcpStream, send: usize, receive: usize) {
        set(stream, libc::SOL_SOCKET, libc::SO_SNDBUF, send as i32);
        set(stream, libc::SOL_SOCKET, libc::SO_RCVBUF, receive as i32);
    }

    /// DSCP 34 (AF41), which Wi-Fi puts in its video queue
    pub fn video_queue(stream: &TcpStream) {
        set(stream, libc::IPPROTO_IP, libc::IP_TOS, 34 << 2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles_have_names() {
        for profile in Profile::ALL {
            assert_eq!(Profile::parse(profile.key()), Some(profile));
        }
        assert_eq!(Profile::parse(" gaming\n"), Some(Profile::Gaming));
        assert_eq!(Profile::parse("cooking"), None);
        assert_eq!(Profile::default(), Profile::Ai);
    }

    #[test]
    fn gaming_queues_less_than_ai() {
        let (gaming, ai) = (Profile::Gaming.tuning(), Profile::Ai.tuning());
        assert!(gaming.send_buffer < ai.send_buffer && gaming.video_queue && !ai.video_queue);
    }

    #[test]
    fn notes_say_when_a_lender_does_not_fit() {
        let apis = |list: &[&str]| list.iter().map(|api| api.to_string()).collect::<Vec<_>>();
        assert_eq!(Profile::Gaming.note(&apis(&["cuda", "vulkan"])), None);
        assert_eq!(Profile::Ai.note(&apis(&["cuda", "vulkan"])), None);
        assert!(Profile::Gaming.note(&apis(&["cuda"])).unwrap().contains("no Vulkan"));
        assert!(Profile::Ai.note(&apis(&["vulkan"])).unwrap().contains("no CUDA"));
    }

    #[test]
    fn a_tuned_socket_still_carries_data() {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        // apply, since tune leaves loopback alone
        apply(&client, Profile::Gaming.tuning());
        apply(&server, Profile::Ai.tuning());
        client.write_all(b"ok").unwrap();
        let mut answer = [0u8; 2];
        server.read_exact(&mut answer).unwrap();
        assert_eq!(&answer, b"ok");
    }
}
