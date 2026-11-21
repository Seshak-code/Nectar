//! Lending a GPU: what the lender, nectar-svc on the borrower and Nectar's nvcuda.dll share.
//!
//! - `discovery`: finding lenders on the local network, both sides proving a shared secret
//! - `lease`: one borrower at a time, with a grace period for short drops
//! - `BorrowConfig`: what nectar-svc writes while a GPU is borrowed, and nvcuda.dll reads
//! - The sandbox for NVIDIA's hybrid runtime (Windows only): `%ProgramData%\Nectar\drivers\<key>\`,
//!   filled from the lender, matching its driver exactly, and never installed system-wide
//!
//! Nectar's folder is `%ProgramData%\Nectar` on Windows and `/var/lib/nectar` elsewhere, or
//! NECTAR_HOME for tests.

pub mod crypto;
pub mod discovery;
pub mod lease;

use std::net::{TcpStream, ToSocketAddrs};
use std::path::PathBuf;
use std::time::Duration;
use std::{env, fs, io};

/// Opens a connection to a lender for `hello` (nectar_cuda_protocol::HELLO_VULKAN and so on),
/// proving the lease's token when the lender asks for it
pub fn connect(server: &str, token: &str, hello: i32, timeout: Duration) -> io::Result<TcpStream> {
    use nectar_cuda_protocol as protocol;

    let mut last_error = io::Error::new(io::ErrorKind::NotFound, format!("{server} has no address"));
    for address in server.to_socket_addrs()? {
        match TcpStream::connect_timeout(&address, timeout) {
            Ok(mut stream) => {
                stream.set_nodelay(true)?;
                protocol::write_hello(&mut stream, hello)?;
                let (_, challenge) = protocol::read_lender_hello(&mut stream)?;
                // A lender that lends asks for the lease's token
                if !challenge.is_empty() {
                    let mac = crypto::hmac(token.as_bytes(), &challenge);
                    protocol::write_proof(&mut stream, &mac, &computer_name())?;
                    protocol::read_verdict(&mut stream)?;
                }
                return Ok(stream);
            }
            Err(error) => last_error = error,
        }
    }
    Err(last_error)
}

/// Nectar's folder for machine-wide state
pub fn home() -> PathBuf {
    if let Some(home) = env::var_os("NECTAR_HOME") {
        return PathBuf::from(home);
    }
    if cfg!(windows) {
        env::var_os("ProgramData")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(r"C:\ProgramData"))
            .join("Nectar")
    } else {
        PathBuf::from("/var/lib/nectar")
    }
}

/// Where nectar-svc takes commands from the app: the named pipe `\\.\pipe\nectar` on
/// Windows; elsewhere a Unix socket, /run/nectar.sock, or nectar.sock in NECTAR_HOME for tests
pub fn service_address() -> PathBuf {
    if cfg!(windows) {
        PathBuf::from(r"\\.\pipe\nectar")
    } else if let Some(home) = env::var_os("NECTAR_HOME") {
        PathBuf::from(home).join("nectar.sock")
    } else {
        PathBuf::from("/run/nectar.sock")
    }
}

/// Where the hybrid runtime for the lender driver `key` is kept
pub fn driver_folder(key: &str) -> PathBuf {
    home().join("drivers").join(key)
}

pub const HYBRID: &str = "nvcudart_hybrid64.dll";

/// Names a lender's hybrid runtime: the CUDA version its driver supports and the start of
/// the file's content hash, since driver releases can share a CUDA version
pub fn hybrid_key(cuda_version: i32, bytes: &[u8]) -> String {
    let hash = nectar_cuda_protocol::content_hash(bytes);
    format!("{cuda_version}-{}", crypto::hex(&hash[..6]))
}

/// This computer's name, as borrowers and lenders show it
pub fn computer_name() -> String {
    let name = if cfg!(windows) {
        env::var("COMPUTERNAME").ok()
    } else {
        ["/proc/sys/kernel/hostname", "/etc/hostname"]
            .iter()
            .find_map(|path| fs::read_to_string(path).ok())
            .or_else(|| env::var("HOSTNAME").ok())
    };
    name.map(|name| name.trim().to_string())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

/// A borrowed GPU, as nectar-svc records it in `borrow.cfg` and nvcuda.dll reads it
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BorrowConfig {
    /// The lender's CUDA server, host:port
    pub server: String,
    /// The lease's token, which the CUDA connections prove
    pub token: String,
    /// The key of the hybrid runtime in the sandbox (driver_folder)
    pub driver: String,
    /// The lender's name, for messages
    pub lender: String,
    /// The APIs borrowed, such as cuda; each of Nectar's drivers checks for its own
    pub apis: Vec<String>,
}

impl BorrowConfig {
    pub fn path() -> PathBuf {
        home().join("borrow.cfg")
    }

    /// The borrowed GPU, or None when nothing is borrowed
    pub fn read() -> Option<Self> {
        Self::parse(&fs::read_to_string(Self::path()).ok()?)
    }

    pub fn parse(text: &str) -> Option<Self> {
        let value = |key: &str| {
            text.lines()
                .filter_map(|line| line.split_once('='))
                .find(|(k, _)| k.trim() == key)
                .map(|(_, value)| value.trim().to_string())
        };
        Some(Self {
            server: value("server").filter(|server| !server.is_empty())?,
            token: value("token").unwrap_or_default(),
            driver: value("driver").unwrap_or_default(),
            lender: value("lender").unwrap_or_default(),
            // Files from before the apis line borrowed CUDA
            apis: value("apis")
                .unwrap_or_else(|| "cuda".to_string())
                .split(',')
                .map(|api| api.trim().to_string())
                .filter(|api| !api.is_empty())
                .collect(),
        })
    }

    pub fn text(&self) -> String {
        format!(
            "# Written by nectar-svc while a GPU is borrowed\nserver={}\ntoken={}\ndriver={}\nlender={}\napis={}\n",
            self.server,
            self.token,
            self.driver,
            self.lender,
            self.apis.join(",")
        )
    }

    /// Whether `api` is borrowed
    pub fn has(&self, api: &str) -> bool {
        self.apis.iter().any(|borrowed| borrowed == api)
    }

    /// Writes the file whole: a program starting meanwhile reads the old one or the new one
    pub fn write(&self) -> io::Result<()> {
        let path = Self::path();
        fs::create_dir_all(home())?;
        let part = path.with_extension("part");
        fs::write(&part, self.text())?;
        fs::rename(&part, &path)
    }

    pub fn remove() -> io::Result<()> {
        match fs::remove_file(Self::path()) {
            Err(error) if error.kind() != io::ErrorKind::NotFound => Err(error),
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn borrow_config_round_trips() {
        let config = BorrowConfig {
            server: "192.168.1.89:43212".into(),
            token: "ab12".into(),
            driver: "13050-0011aabbccdd".into(),
            lender: "DESKTOP".into(),
            apis: vec!["cuda".into(), "vulkan".into()],
        };
        assert_eq!(BorrowConfig::parse(&config.text()), Some(config.clone()));
        assert!(config.has("vulkan") && !config.has("hip"));
        assert!(BorrowConfig::parse("server=a:1\n").unwrap().has("cuda"));
        assert!(!BorrowConfig::parse("server=a:1\napis=\n").unwrap().has("cuda"));
        assert_eq!(BorrowConfig::parse("token=x\n"), None);
        assert_eq!(BorrowConfig::parse("server=\n"), None);
    }

    #[test]
    fn hybrid_keys_follow_the_bytes() {
        assert_eq!(hybrid_key(13050, b"a"), hybrid_key(13050, b"a"));
        assert_ne!(hybrid_key(13050, b"a"), hybrid_key(13050, b"b"));
        assert!(hybrid_key(13050, b"a").starts_with("13050-"));
    }
}
