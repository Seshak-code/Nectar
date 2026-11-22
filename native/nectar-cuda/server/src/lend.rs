//! Lending (--lend, nectar-lending): discovery, the shared secret, the lease, and the
//! borrower's control connection.
//!
//! Every connection starts with a challenge. The borrower's control connection proves the
//! shared secret and takes the lease; its CUDA connections prove the lease's token. While
//! the lease is held, everyone else is refused with the holder's name. Without --lend the
//! server sends no challenge and takes any connection, as tests on loopback do.

use std::io::{self, BufRead, Read, Write};
use std::net::UdpSocket;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use std::{fs, thread};

use nectar_cuda_protocol::{self as protocol, Writer};
use nectar_lending::crypto::{self, hmac};
use nectar_lending::discovery::{self, LenderInfo, State};
use nectar_lending::lease::{GRACE, Lease};
use nectar_lending::Profile;

pub(crate) struct Lending {
    secret: Vec<u8>,
    lease: Mutex<Lease>,
    /// What the borrower chose the GPU for; its connections are tuned by it
    profile: Mutex<Profile>,
    /// This lender, as discovery describes it (its state is filled in when answering)
    info: LenderInfo,
    /// The driver's hybrid runtime: its key and bytes
    hybrid: Option<(String, Vec<u8>)>,
    /// Totals over the lease's CUDA connections, for the borrower's heartbeat
    pub(crate) calls: AtomicU64,
    pub(crate) received: AtomicU64,
    pub(crate) sent: AtomicU64,
}

impl Lending {
    pub(crate) fn new(secret: Vec<u8>, info: LenderInfo, hybrid: Option<(String, Vec<u8>)>) -> Self {
        Self {
            secret,
            lease: Mutex::new(Lease::default()),
            profile: Mutex::new(Profile::default()),
            info,
            hybrid,
            calls: AtomicU64::new(0),
            received: AtomicU64::new(0),
            sent: AtomicU64::new(0),
        }
    }

    fn lease(&self) -> MutexGuard<'_, Lease> {
        self.lease
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub(crate) fn profile(&self) -> Profile {
        *self.profile.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Answers borrowers' probes on UDP port 43211 at `ip`, the address the server listens
    /// on, from a thread of its own
    pub(crate) fn announce(self: &Arc<Self>, ip: std::net::IpAddr) -> io::Result<()> {
        let socket = UdpSocket::bind((ip, discovery::PORT))?;
        let lending = Arc::clone(self);
        thread::spawn(move || {
            let mut buffer = [0u8; 2048];
            loop {
                let Ok((length, from)) = socket.recv_from(&mut buffer) else {
                    continue;
                };
                let Ok(text) = std::str::from_utf8(&buffer[..length]) else {
                    continue;
                };
                let Some(nonce) = discovery::check_probe(&lending.secret, text, unix_now()) else {
                    continue;
                };
                let mut info = lending.info.clone();
                if let Some(holder) = lending.lease().holder(Instant::now()) {
                    info.state = State::Lent;
                    info.borrower = holder.to_string();
                }
                let answer = discovery::answer(&lending.secret, &nonce, &info);
                let _ = socket.send_to(answer.as_bytes(), from);
            }
        });
        Ok(())
    }

    /// After the borrower's hello: the challenge, its proof and the verdict. Returns the
    /// lease's token the connection holds, or None when it was refused.
    pub(crate) fn admit(
        &self,
        kind: i32,
        reader: &mut impl Read,
        writer: &mut impl Write,
        version: i32,
        peer: &str,
    ) -> io::Result<Option<String>> {
        let challenge = crypto::random::<32>();
        protocol::write_lender_hello(writer, version, &challenge)?;
        let (mac, name) = protocol::read_proof(reader)?;
        let now = Instant::now();

        let verdict = if kind == protocol::HELLO_CONTROL {
            if !crypto::same(&mac, &hmac(&self.secret, &challenge)) {
                Err("the shared secret is wrong".to_string())
            } else {
                match self.lease().take(&name, now, || crypto::hex(&crypto::random::<32>())) {
                    Ok(token) => {
                        eprintln!("{peer} {name} borrows this GPU");
                        Ok(token)
                    }
                    Err(holder) => Err(format!("lent to {holder}")),
                }
            }
        } else {
            let mut lease = self.lease();
            match lease.token(now) {
                Some(token) if crypto::same(&mac, &hmac(token.as_bytes(), &challenge)) => {
                    lease.open(&token, now);
                    Ok(token)
                }
                Some(_) => Err(format!("lent to {}", lease.holder(now).unwrap_or("another borrower"))),
                None => Err("not borrowed: borrow it in the Nectar app first".to_string()),
            }
        };

        match verdict {
            Ok(token) => {
                let text = if kind == protocol::HELLO_CONTROL { token.as_str() } else { "" };
                protocol::write_verdict(writer, true, text)?;
                Ok(Some(token))
            }
            Err(reason) => {
                eprintln!("{peer} {name} refused: {reason}");
                protocol::write_verdict(writer, false, &reason)?;
                Ok(None)
            }
        }
    }

    /// A connection that held the lease has ended
    pub(crate) fn leave(&self, token: &str) {
        self.lease().close(token, Instant::now());
    }

    /// The borrower's control connection: heartbeats, the hybrid runtime, and return. It
    /// ends when the borrower returns the GPU or has been silent for the lease's grace.
    pub(crate) fn control(
        &self,
        token: &str,
        reader: &mut impl BufRead,
        writer: &mut impl Write,
        peer: &str,
    ) -> io::Result<()> {
        loop {
            let (request, body) = match protocol::read_message(reader) {
                Ok(message) => message,
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
                    ) =>
                {
                    eprintln!("{peer} silent for {} s; its lease waits out its grace", GRACE.as_secs());
                    return Ok(());
                }
                Err(error) => return Err(error),
            };

            let mut reply = Writer::new();
            match request {
                protocol::BEAT => {
                    if !self.lease().beat(token, Instant::now()) {
                        protocol::write_message(writer, nectar_cuda_sys::CUDA_ERROR_NOT_PERMITTED as u32, &[])?;
                        writer.flush()?;
                        return Ok(());
                    }
                    reply
                        .u64(self.calls.load(Ordering::Relaxed))
                        .u64(self.received.load(Ordering::Relaxed))
                        .u64(self.sent.load(Ordering::Relaxed))
                        .u32(nectar_detect::usage::gpu_percent().unwrap_or(u32::MAX));
                }
                protocol::HYBRID => match &self.hybrid {
                    Some((key, bytes)) => {
                        reply.bytes(key.as_bytes()).bytes(bytes);
                    }
                    None => {
                        protocol::write_message(writer, nectar_cuda_sys::CUDA_ERROR_NOT_FOUND as u32, &[])?;
                        writer.flush()?;
                        continue;
                    }
                },
                protocol::PROFILE => {
                    if let Some(profile) = Profile::parse(&String::from_utf8_lossy(&body)) {
                        eprintln!("{peer} chose {}", profile.label());
                        *self.profile.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = profile;
                    }
                }
                protocol::RETURN => {
                    self.lease().give_back(token);
                    eprintln!("{peer} returned this GPU");
                    protocol::write_message(writer, 0, &[])?;
                    writer.flush()?;
                    return Ok(());
                }
                _ => {
                    protocol::write_message(writer, nectar_cuda_sys::CUDA_ERROR_NOT_SUPPORTED as u32, &[])?;
                    writer.flush()?;
                    continue;
                }
            }
            protocol::write_message(writer, 0, &[reply.as_slice()])?;
            writer.flush()?;
        }
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs())
}

/// The newest nvcudart_hybrid64.dll in the NVIDIA driver's DriverStore folders, and its bytes
pub(crate) fn find_hybrid(cuda_version: i32) -> Option<(String, Vec<u8>)> {
    let root = std::env::var_os("SystemRoot").map(std::path::PathBuf::from)?;
    let repository = root.join(r"System32\DriverStore\FileRepository");
    let newest = fs::read_dir(repository)
        .ok()?
        .filter_map(Result::ok)
        .filter(|entry| entry.file_name().to_string_lossy().to_ascii_lowercase().starts_with("nv"))
        .map(|entry| entry.path().join(nectar_lending::HYBRID))
        .filter_map(|path| Some((fs::metadata(&path).ok()?.modified().ok()?, path)))
        .max_by_key(|(modified, _)| *modified)?;
    let bytes = fs::read(&newest.1).ok()?;
    Some((nectar_lending::hybrid_key(cuda_version, &bytes), bytes))
}
