//! Lend mode: runs nectar-server --lend from nectar.exe's folder with the shared secret,
//! and keeps its output for the window. The secret is remembered for this Windows user,
//! encrypted with DPAPI.

use std::io::{BufRead, BufReader, Read, Write};
use std::os::windows::process::CommandExt;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::{env, fs, ptr, thread};

use windows_sys::Win32::Foundation::LocalFree;
use windows_sys::Win32::Security::Cryptography::{
    CRYPT_INTEGER_BLOB, CryptProtectData, CryptUnprotectData,
};

const CREATE_NO_WINDOW: u32 = 0x0800_0000;

pub struct Lender {
    child: Child,
    /// Lines the server has written that the window has not shown yet
    pub output: Arc<Mutex<Vec<String>>>,
}

impl Lender {
    pub fn start(secret: &str) -> std::io::Result<Self> {
        let server = env::current_exe()?
            .parent()
            .map(|folder| folder.join("nectar-server.exe"))
            .unwrap_or_default();
        // NECTAR_LEND_LISTEN moves it, such as onto loopback for tests
        let listen = env::var("NECTAR_LEND_LISTEN")
            .unwrap_or_else(|_| format!("0.0.0.0:{}", nectar_cuda_protocol::DEFAULT_PORT));
        let mut child = Command::new(&server)
            .args(["--lend", "--listen", &listen])
            // Not on the command line, where other users could read it
            .env("NECTAR_LEND_SECRET", secret)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .creation_flags(CREATE_NO_WINDOW)
            .spawn()
            .map_err(|error| {
                std::io::Error::new(error.kind(), format!("{}: {error}", server.display()))
            })?;

        let output = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::new(Mutex::new(open_log()));
        if let Some(stderr) = child.stderr.take() {
            collect(stderr, &output, &log);
        }
        if let Some(stdout) = child.stdout.take() {
            collect(stdout, &output, &log);
        }
        Ok(Self { child, output })
    }

    /// Whether the server is still running
    pub fn running(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    pub fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for Lender {
    fn drop(&mut self) {
        self.stop();
    }
}

/// The local time as YYYYMMDD-HHMMSS, or HH:MM:SS.mmm for log lines
fn local_time(for_file: bool) -> String {
    let mut now: windows_sys::Win32::Foundation::SYSTEMTIME = unsafe { std::mem::zeroed() };
    unsafe { windows_sys::Win32::System::SystemInformation::GetLocalTime(&mut now) };
    if for_file {
        format!(
            "{:04}{:02}{:02}-{:02}{:02}{:02}",
            now.wYear, now.wMonth, now.wDay, now.wHour, now.wMinute, now.wSecond
        )
    } else {
        format!(
            "{:02}:{:02}:{:02}.{:03}",
            now.wHour, now.wMinute, now.wSecond, now.wMilliseconds
        )
    }
}

/// %LOCALAPPDATA%\Nectar\lender-logs\server-<time>.log, which keeps what the window shows
fn open_log() -> Option<fs::File> {
    let folder = PathBuf::from(env::var_os("LOCALAPPDATA")?).join(r"Nectar\lender-logs");
    fs::create_dir_all(&folder).ok()?;
    fs::File::create(folder.join(format!("server-{}.log", local_time(true)))).ok()
}

fn collect(
    stream: impl Read + Send + 'static,
    output: &Arc<Mutex<Vec<String>>>,
    log: &Arc<Mutex<Option<fs::File>>>,
) {
    let output = Arc::clone(output);
    let log = Arc::clone(log);
    thread::spawn(move || {
        for line in BufReader::new(stream).lines() {
            let Ok(line) = line else { break };
            let line = format!("{} {line}", local_time(false));
            if let Some(file) = log.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).as_mut() {
                let _ = writeln!(file, "{line}");
            }
            output
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(line);
        }
    });
}

fn secret_path() -> Option<PathBuf> {
    env::var_os("LOCALAPPDATA").map(|folder| PathBuf::from(folder).join(r"Nectar\lend-secret"))
}

/// The secret saved last time, if any
pub fn saved_secret() -> Option<String> {
    let sealed = fs::read(secret_path()?).ok()?;
    let open = dpapi(&sealed, false)?;
    String::from_utf8(open).ok()
}

pub fn save_secret(secret: &str) {
    if let (Some(path), Some(sealed)) = (secret_path(), dpapi(secret.as_bytes(), true)) {
        if let Some(folder) = path.parent() {
            let _ = fs::create_dir_all(folder);
        }
        let _ = fs::write(path, sealed);
    }
}

/// Encrypts (seal) or decrypts bytes for this Windows user
fn dpapi(data: &[u8], seal: bool) -> Option<Vec<u8>> {
    let input = CRYPT_INTEGER_BLOB {
        cbData: data.len() as u32,
        pbData: data.as_ptr().cast_mut(),
    };
    let mut output = CRYPT_INTEGER_BLOB {
        cbData: 0,
        pbData: ptr::null_mut(),
    };
    let ok = unsafe {
        if seal {
            CryptProtectData(&input, ptr::null(), ptr::null(), ptr::null(), ptr::null(), 0, &mut output)
        } else {
            CryptUnprotectData(&input, ptr::null_mut(), ptr::null(), ptr::null(), ptr::null(), 0, &mut output)
        }
    };
    if ok == 0 {
        return None;
    }
    let bytes = unsafe { std::slice::from_raw_parts(output.pbData, output.cbData as usize) }.to_vec();
    unsafe { LocalFree(output.pbData.cast()) };
    Some(bytes)
}
