//! nvcuda.dll for a borrowed GPU: exports the CUDA driver API and carries out each call
//! on the lender through nectar-server.
//!
//! The lender is NECTAR_CUDA_SERVER (host:port), or 127.0.0.1 on the default port.
//!
//! Unlike Juice's driver, a missing or lost lender never ends the application. cuInit
//! reports CUDA_ERROR_NO_DEVICE when the lender cannot be reached, and every call after a
//! lost connection reports CUDA_ERROR_DEVICE_UNAVAILABLE.
//!
//! Handles (contexts, device pointers) are the lender's values, passed through unchanged:
//! the application only hands them back.
//!
//! Calls that only queue work on the GPU do not wait for the lender, and answers that
//! cannot change are kept (calls.rs).
//!
//! Most exports are generated from cuda.h (generated.rs, see codegen/report.md). This file
//! holds the connection and the functions that need more than copying values: those that
//! move host memory, return strings, or hand out function pointers.
//!
//! # Safety
//!
//! Every export has the contract cuda.h documents for it: pointer arguments must be valid
//! for the reads and writes that function makes.

#![allow(non_snake_case, clippy::missing_safety_doc)]

mod calls;
mod code;
mod dark_api;
mod generated;
mod host;
mod integrity;
mod launch;
mod vmm;

// The generated export table finds hand-written functions at the crate root
pub use code::{cuLibraryLoadData, cuModuleLoadData, cuModuleLoadDataEx};
pub use host::{
    cuMemAllocHost_v2, cuMemFreeHost, cuMemHostAlloc, cuMemHostGetFlags, cuMemHostRegister_v2,
    cuMemHostUnregister, cuMemcpyDtoHAsync_v2, cuMemcpyDtoHAsync_v2_ptsz, cuMemcpyHtoDAsync_v2,
    cuMemcpyHtoDAsync_v2_ptsz,
};
pub use launch::{cuLaunchKernel, cuLaunchKernel_ptsz, cuLaunchKernelEx, cuLaunchKernelEx_ptsz};
pub use vmm::{cuDeviceGetPCIBusId, cuMemCreate, cuMemGetAllocationGranularity, cuMemSetAccess};

use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::{CStr, CString, c_char, c_int, c_uint, c_void};
use std::io::{self, BufReader, BufWriter, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::{Condvar, Mutex, MutexGuard, Once, OnceLock};
use std::time::{Duration, Instant};
use std::{env, ptr, slice, thread};

use calls::Kind;
use nectar_cuda_protocol::{self as protocol, Reader, Writer, id};
use nectar_cuda_sys::*;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);

/// Tells tools such as nectar-detect that this is Nectar's driver, so they don't initialize it
/// (which would connect to the lender) to learn about a local GPU
#[unsafe(no_mangle)]
pub extern "C" fn nectarIdentify() -> *const c_char {
    concat!("Nectar ", env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

/// After a failed connection attempt, calls fail at once for this long
const RETRY_AFTER: Duration = Duration::from_secs(5);

/// Requests that are not waited for are sent at the latest this long after the first of
/// them, or once FLUSH_SIZE has gathered, if no call that waits sends them sooner
const FLUSH_AFTER: Duration = Duration::from_millis(1);
const FLUSH_SIZE: usize = 16 << 10;

/// Once this many replies are unread, the older half is read, so the lender never blocks
/// writing replies the borrower is not reading. Each is 8 bytes, far less than the socket
/// buffers hold.
const MAX_PENDING: usize = 4096;

struct Connection {
    reader: BufReader<TcpStream>,
    writer: BufWriter<TcpStream>,
    /// Requests sent without waiting whose replies are unread, oldest first
    pending: VecDeque<Pending>,
    /// How many of them are copies to host memory
    copying: usize,
    /// When the oldest request still in `writer` was written
    unflushed_since: Option<Instant>,
    /// Whether work sent without waiting may still be running on the GPU
    running: bool,
    /// The first error a request sent without waiting met, for the next synchronize
    deferred: Option<CUresult>,
    /// Kept answers (Kind::Keep), by function and key
    kept: HashMap<(u32, Vec<u8>), Vec<u8>>,
    /// Devices, kernels and functions whose attributes were all fetched into `kept`
    fetched: HashSet<(u32, Vec<u8>)>,
    /// The uploads the lender keeps by content hash (protocol::UPLOAD)
    uploads: Uploads,
}

enum Uploads {
    /// Every upload is sent as it is
    Off,
    /// The lender has not been asked for its index yet
    Unknown,
    /// The hashes the lender keeps
    Kept(HashSet<protocol::Hash>),
}

/// What the reply to a request sent without waiting carries
enum Pending {
    /// Only its CUresult
    Result(u32),
    /// Whether the lender kept an upload under this hash
    Upload(protocol::Hash),
    /// Bytes for host memory, as for cuMemcpyDtoHAsync
    Copy {
        function: u32,
        destination: usize,
        size: usize,
    },
}

/// The outcome of a call that reached the lender, and how
enum Carried {
    Waited(CUresult, Vec<u8>),
    Sent(CUresult),
    Kept(Vec<u8>),
}

impl Connection {
    fn new(stream: TcpStream, keep_uploads: bool) -> io::Result<Self> {
        Ok(Self {
            reader: BufReader::with_capacity(1 << 16, stream.try_clone()?),
            writer: BufWriter::with_capacity(1 << 16, stream),
            pending: VecDeque::new(),
            copying: 0,
            unflushed_since: None,
            running: false,
            deferred: None,
            kept: HashMap::new(),
            fetched: HashSet::new(),
            uploads: if keep_uploads {
                Uploads::Unknown
            } else {
                Uploads::Off
            },
        })
    }

    /// Copies bytes to GPU memory, as `function` with `head` before the bytes would. Uploads
    /// of protocol::CACHE_MIN or more go by content hash, and only the hash crosses the
    /// network once the lender keeps them.
    fn upload(
        &mut self,
        function: u32,
        head: &[u8],
        destination: CUdeviceptr,
        stream: u64,
        data: &[u8],
    ) -> io::Result<Carried> {
        if data.len() < protocol::CACHE_MIN || !self.lender_keeps_uploads()? {
            let size = (data.len() as u64).to_le_bytes();
            return self.carry(function, &[head, &size, data]);
        }

        let hash = protocol::content_hash(data);
        let mut request = Writer::new();
        request.u32(function).u64(destination).u64(stream).value(&hash);
        let kept = matches!(&self.uploads, Uploads::Kept(kept) if kept.contains(&hash));
        if kept {
            request.u8(0).u64(data.len() as u64);
            self.send(protocol::UPLOAD, &[request.as_slice()], Pending::Upload(hash))?;
        } else {
            request.u8(1).u64(data.len() as u64);
            self.send(
                protocol::UPLOAD,
                &[request.as_slice(), data],
                Pending::Upload(hash),
            )?;
        }
        Ok(Carried::Sent(CUDA_SUCCESS))
    }

    /// For an attribute of a device, kernel or function not kept yet, keeps all of that
    /// one's attributes in one round trip (protocol::ATTRIBUTES). `key` is the attribute,
    /// then the rest of the request. Returns whether the lender answered.
    fn fetch_attributes(&mut self, function: u32, key: &[u8]) -> io::Result<bool> {
        if !matches!(
            function,
            id::cuDeviceGetAttribute | id::cuKernelGetAttribute | id::cuFuncGetAttribute
        ) || key.len() < 4
        {
            return Ok(false);
        }
        let rest = &key[4..];
        // Attributes the driver refuses are asked for one by one, not fetched again
        if !self.fetched.insert((function, rest.to_vec())) {
            return Ok(false);
        }
        let (result, results) =
            self.exchange(protocol::ATTRIBUTES, &[&function.to_le_bytes(), rest])?;
        if result != CUDA_SUCCESS {
            return Ok(false);
        }

        let mut reader = Reader::new(&results);
        for _ in 0..reader.u32()? {
            let attribute = reader.i32()?;
            let value = reader.i32()?;
            // As the lender would answer the one call: the optional value written
            let mut answer = vec![1];
            answer.extend_from_slice(&value.to_le_bytes());
            let key = [&attribute.to_le_bytes()[..], rest].concat();
            self.kept.insert((function, key), answer);
        }
        Ok(true)
    }

    /// Asks the lender, the first time, which uploads it keeps
    fn lender_keeps_uploads(&mut self) -> io::Result<bool> {
        if let Uploads::Unknown = self.uploads {
            let (result, results) = self.exchange(protocol::CACHE_INDEX, &[])?;
            self.uploads = Uploads::Off;
            if result == CUDA_SUCCESS {
                let mut reader = Reader::new(&results);
                let count = reader.u64()?;
                let mut kept = HashSet::new();
                for _ in 0..count {
                    kept.insert(reader.take(16)?.try_into().unwrap());
                }
                self.uploads = Uploads::Kept(kept);
            }
        }
        Ok(matches!(self.uploads, Uploads::Kept(_)))
    }

    fn flush(&mut self) -> io::Result<()> {
        self.unflushed_since = None;
        self.writer.flush()
    }

    /// Sends a request without waiting for its reply
    fn send(&mut self, function: u32, parts: &[&[u8]], pending: Pending) -> io::Result<()> {
        // A copy's reply may be large, so it is read before anything else is sent: replies
        // the borrower leaves unread stay small
        if self.copying > 0 {
            self.catch_up()?;
        }

        protocol::write_message(&mut self.writer, function, parts)?;
        if matches!(pending, Pending::Copy { .. }) {
            self.copying += 1;
        }
        self.pending.push_back(pending);

        match self.unflushed_since {
            None => {
                self.unflushed_since = Some(Instant::now());
                FLUSH.notify_one();
            }
            Some(since)
                if since.elapsed() >= FLUSH_AFTER || self.writer.buffer().len() >= FLUSH_SIZE =>
            {
                self.flush()?
            }
            Some(_) => {}
        }

        if self.pending.len() >= MAX_PENDING {
            self.flush()?;
            self.receive(MAX_PENDING / 2)?;
        }
        Ok(())
    }

    /// Sends a request and waits for its reply, which follows those of earlier requests
    fn exchange(&mut self, function: u32, parts: &[&[u8]]) -> io::Result<(CUresult, Vec<u8>)> {
        protocol::write_message(&mut self.writer, function, parts)?;
        self.catch_up()?;
        let (result, results) = protocol::read_message(&mut self.reader)?;
        Ok((result as CUresult, results))
    }

    /// Waits for the replies to every request sent without waiting
    fn catch_up(&mut self) -> io::Result<()> {
        self.flush()?;
        if let Some(latency) = test_latency() {
            thread::sleep(latency);
        }
        self.receive(self.pending.len())
    }

    /// Reads the replies to the oldest `count` requests sent without waiting
    fn receive(&mut self, count: usize) -> io::Result<()> {
        for _ in 0..count {
            let Some(pending) = self.pending.pop_front() else {
                break;
            };
            let (result, results) = protocol::read_message(&mut self.reader)?;
            let result = result as CUresult;
            match pending {
                Pending::Result(function) => {
                    if result != CUDA_SUCCESS {
                        self.failed(function, result);
                    }
                }
                Pending::Upload(hash) => {
                    if result != CUDA_SUCCESS {
                        self.failed(protocol::UPLOAD, result);
                    } else if results.first() == Some(&1)
                        && let Uploads::Kept(kept) = &mut self.uploads
                    {
                        kept.insert(hash);
                    }
                }
                Pending::Copy {
                    function,
                    destination,
                    size,
                } => {
                    self.copying -= 1;
                    if result != CUDA_SUCCESS {
                        self.failed(function, result);
                        continue;
                    }
                    let data = Reader::new(&results).bytes()?;
                    if data.len() != size {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "short copy"));
                    }
                    unsafe {
                        ptr::copy_nonoverlapping(data.as_ptr(), destination as *mut u8, size)
                    };
                }
            }
        }
        Ok(())
    }

    /// Keeps the first error of a request sent without waiting, for the next synchronize
    fn failed(&mut self, function: u32, result: CUresult) {
        if self.deferred.is_none() {
            log(&format!(
                "{} failed with {result}; the next synchronize reports it",
                protocol::call_name(function)
            ));
            self.deferred = Some(result);
        }
    }

    /// Carries out a call as calls.rs says
    fn carry(&mut self, function: u32, parts: &[&[u8]]) -> io::Result<Carried> {
        match calls::kind(function) {
            Kind::Send { runs } => {
                self.send(function, parts, Pending::Result(function))?;
                self.running |= runs;
                Ok(Carried::Sent(CUDA_SUCCESS))
            }

            Kind::Synchronize { always } if !always && !self.running && self.copying == 0 => {
                self.send(function, parts, Pending::Result(function))?;
                Ok(Carried::Sent(self.deferred.take().unwrap_or(CUDA_SUCCESS)))
            }

            Kind::Synchronize { .. } => {
                let (result, results) = self.exchange(function, parts)?;
                self.running = false;
                let deferred = self.deferred.take();
                Ok(match deferred {
                    Some(deferred) if result == CUDA_SUCCESS => Carried::Waited(deferred, results),
                    _ => Carried::Waited(result, results),
                })
            }

            Kind::Keep { size } => {
                let key = match parts {
                    [request] => calls::key(request, size).map(|key| (function, key.to_vec())),
                    _ => None,
                };
                if let Some(kept) = key.as_ref().and_then(|key| self.kept.get(key)) {
                    return Ok(Carried::Kept(kept.clone()));
                }
                if let Some((_, key)) = &key
                    && self.fetch_attributes(function, key)?
                    && let Some(kept) = self.kept.get(&(function, key.clone()))
                {
                    return Ok(Carried::Waited(CUDA_SUCCESS, kept.clone()));
                }
                let (result, results) = self.exchange(function, parts)?;
                if result == CUDA_SUCCESS
                    && let Some(key) = key
                {
                    self.kept.insert(key, results.clone());
                }
                Ok(Carried::Waited(result, results))
            }

            Kind::Forget => {
                self.kept.clear();
                self.fetched.clear();
                let (result, results) = self.exchange(function, parts)?;
                Ok(Carried::Waited(result, results))
            }

            Kind::Wait => {
                let (result, results) = self.exchange(function, parts)?;
                Ok(Carried::Waited(result, results))
            }
        }
    }
}

/// Wakes the thread that sends requests no call has sent yet (flush_in_background)
static FLUSH: Condvar = Condvar::new();

fn flush_in_background() {
    let mut state = state();
    loop {
        let mut wait = None;
        if let State::Connected(connection) = &mut *state
            && let Some(since) = connection.unflushed_since
        {
            let age = since.elapsed();
            if age >= FLUSH_AFTER {
                // A broken connection shows up again on the next call, which reports it
                let _ = connection.flush();
            } else {
                wait = Some(FLUSH_AFTER - age);
            }
        }
        state = match wait {
            Some(wait) => {
                FLUSH
                    .wait_timeout(state, wait)
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .0
            }
            None => FLUSH
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner()),
        };
    }
}

/// NECTAR_CUDA_TEST_LATENCY_MS adds that long to every wait for the lender, to try out a
/// slower network on loopback
fn test_latency() -> Option<Duration> {
    static LATENCY: OnceLock<Option<Duration>> = OnceLock::new();
    *LATENCY.get_or_init(|| {
        env::var("NECTAR_CUDA_TEST_LATENCY_MS")
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .filter(|milliseconds| *milliseconds > 0.0)
            .map(|milliseconds| Duration::from_secs_f64(milliseconds / 1000.0))
    })
}

enum State {
    Disconnected { failed_at: Option<Instant> },
    Connected(Box<Connection>),
    Lost,
}

static STATE: Mutex<State> = Mutex::new(State::Disconnected { failed_at: None });

fn state() -> MutexGuard<'static, State> {
    STATE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

pub(crate) fn log(message: &str) {
    eprintln!("nectar-cuda: {message}");
}

/// NECTAR_CUDA_LOG=1 logs every call and its result
pub(crate) fn logging() -> bool {
    static LOGGING: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *LOGGING.get_or_init(|| env::var_os("NECTAR_CUDA_LOG").is_some())
}

#[cfg(windows)]
#[unsafe(no_mangle)]
extern "system" fn DllMain(
    module: windows_sys::Win32::Foundation::HMODULE,
    reason: u32,
    _reserved: *mut c_void,
) -> windows_sys::core::BOOL {
    // Installed as System32\nvcuda.dll it is the one every load finds: nothing to redirect
    if reason == windows_sys::Win32::System::SystemServices::DLL_PROCESS_ATTACH
        && !in_system_folder(module)
        && let Err(error) = unsafe { nectar_cuda_redirect::install(module, logging()) }
    {
        log(&format!(
            "cannot redirect loads of nvcuda.dll ({error}); programs that load it by full path will use the real driver"
        ));
    }
    1
}

/// Whether this library was loaded from System32, where Nectar's installer puts it
#[cfg(windows)]
fn in_system_folder(module: windows_sys::Win32::Foundation::HMODULE) -> bool {
    use windows_sys::Win32::System::LibraryLoader::GetModuleFileNameW;
    use windows_sys::Win32::System::SystemInformation::GetSystemDirectoryW;

    let mut own = [0u16; 1024];
    let mut system = [0u16; 512];
    let own_length = unsafe { GetModuleFileNameW(module, own.as_mut_ptr(), own.len() as u32) } as usize;
    let system_length =
        unsafe { GetSystemDirectoryW(system.as_mut_ptr(), system.len() as u32) } as usize;
    if own_length == 0 || system_length == 0 || system_length >= system.len() {
        return false;
    }
    let own = String::from_utf16_lossy(&own[..own_length]);
    let system = String::from_utf16_lossy(&system[..system_length]);
    std::path::Path::new(&own)
        .parent()
        .is_some_and(|folder| folder.to_string_lossy().eq_ignore_ascii_case(system.trim_end_matches('\\')))
}

/// An on/off setting: 1 or 0, unset for the default
fn setting(name: &str) -> Option<bool> {
    match env::var(name).ok()?.as_str() {
        "1" => Some(true),
        "0" => Some(false),
        _ => None,
    }
}

/// The lender and the lease token to prove to it: NECTAR_CUDA_SERVER (and NECTAR_CUDA_TOKEN)
/// for testing, otherwise the GPU nectar-svc has borrowed (borrow.cfg). None when nothing is
/// borrowed: then there is no GPU, as on a PC without one.
fn lender() -> Option<(String, String)> {
    if let Ok(server) = env::var("NECTAR_CUDA_SERVER") {
        return Some((server, env::var("NECTAR_CUDA_TOKEN").unwrap_or_default()));
    }
    nectar_lending::BorrowConfig::read()
        .filter(|config| config.has("cuda"))
        .map(|config| (config.server, config.token))
}

fn connect() -> io::Result<Connection> {
    let Some((server, token)) = lender() else {
        return Err(io::Error::new(io::ErrorKind::NotFound, NOT_BORROWED));
    };

    let mut last_error =
        io::Error::new(io::ErrorKind::NotFound, format!("{server} has no address"));
    for address in server.to_socket_addrs()? {
        match TcpStream::connect_timeout(&address, CONNECT_TIMEOUT) {
            Ok(stream) => {
                stream.set_nodelay(true)?;
                nectar_lending::profile::tune(&stream, nectar_lending::Profile::current());
                // Over loopback, keeping uploads would only cost time
                let remote = !address.ip().is_loopback();
                let keep_uploads = setting("NECTAR_CUDA_KEEP_UPLOADS").unwrap_or(remote);
                let mut connection = Connection::new(stream, keep_uploads)?;
                protocol::write_hello(&mut connection.writer, protocol::HELLO_CUDA)?;
                let (_, challenge) = protocol::read_lender_hello(&mut connection.reader)?;
                // A lender that lends asks for the lease's token
                if !challenge.is_empty() {
                    let mac = nectar_lending::crypto::hmac(token.as_bytes(), &challenge);
                    protocol::write_proof(
                        &mut connection.writer,
                        &mac,
                        &nectar_lending::computer_name(),
                    )?;
                    protocol::read_verdict(&mut connection.reader)?;
                }
                return Ok(connection);
            }
            Err(error) => last_error = error,
        }
    }

    Err(last_error)
}

const NOT_BORROWED: &str = "no GPU is borrowed";

/// Connects on first use, and again after RETRY_AFTER if that failed
fn ensure_connected(state: &mut State) -> bool {
    match state {
        State::Connected(_) => true,
        State::Lost => false,
        State::Disconnected { failed_at } => {
            if failed_at.is_some_and(|at| at.elapsed() < RETRY_AFTER) {
                return false;
            }
            match connect() {
                Ok(connection) => {
                    *state = State::Connected(Box::new(connection));
                    static FLUSHER: Once = Once::new();
                    FLUSHER.call_once(|| {
                        thread::spawn(flush_in_background);
                    });
                    true
                }
                Err(error) => {
                    // Every CUDA program would say so while nothing is borrowed
                    if error.kind() == io::ErrorKind::PermissionDenied {
                        log(&format!("the lender refused: {error}"));
                    } else if error.to_string() != NOT_BORROWED {
                        log(&format!("cannot reach the lender: {error}"));
                    } else if logging() {
                        log(NOT_BORROWED);
                    }
                    *state = State::Disconnected {
                        failed_at: Some(Instant::now()),
                    };
                    false
                }
            }
        }
    }
}

/// Sends a request made of `parts` and returns the results, or the CUresult on failure.
/// Whether it waits for the lender is up to calls.rs.
pub(crate) fn call(function: u32, parts: &[&[u8]]) -> Result<Vec<u8>, CUresult> {
    with_connection(function, |connection| connection.carry(function, parts))
}

/// Sends a copy from GPU memory to host memory without waiting. The bytes are written to
/// `destination` when the reply is read, at the latest by the next call that waits, which
/// includes the synchronize the program needs before it reads them.
pub(crate) fn copy_to_host(
    function: u32,
    request: &[u8],
    destination: *mut c_void,
    size: usize,
) -> CUresult {
    let pending = Pending::Copy {
        function,
        destination: destination as usize,
        size,
    };
    let sent = with_connection(function, |connection| {
        connection.send(function, &[request], pending)?;
        Ok(Carried::Sent(CUDA_SUCCESS))
    });
    match sent {
        Ok(_) => CUDA_SUCCESS,
        Err(result) => result,
    }
}

/// Copies bytes to GPU memory, as `function` would with `head`, then the bytes' length and
/// the bytes; large uploads go by content hash (Connection::upload)
pub(crate) fn upload(
    function: u32,
    head: &[u8],
    destination: CUdeviceptr,
    stream: u64,
    data: &[u8],
) -> Result<Vec<u8>, CUresult> {
    with_connection(function, |connection| {
        connection.upload(function, head, destination, stream, data)
    })
}

fn with_connection(
    function: u32,
    carry: impl FnOnce(&mut Connection) -> io::Result<Carried>,
) -> Result<Vec<u8>, CUresult> {
    let mut state = state();
    if !ensure_connected(&mut state) {
        return Err(match *state {
            State::Lost => CUDA_ERROR_DEVICE_UNAVAILABLE,
            _ => CUDA_ERROR_NOT_INITIALIZED,
        });
    }
    let State::Connected(connection) = &mut *state else {
        unreachable!()
    };

    let carried = carry(connection);

    if logging() {
        let name = protocol::call_name(function);
        match &carried {
            Ok(Carried::Waited(result, _)) => log(&format!("{name} = {result}")),
            Ok(Carried::Sent(result)) => log(&format!("{name} = {result} (sent)")),
            Ok(Carried::Kept(_)) => log(&format!("{name} = 0 (kept)")),
            // Calls during process exit, after Windows sockets shut down (see below)
            Err(error) if error.raw_os_error() == Some(10093) => {}
            Err(error) => log(&format!("{name}: {error}")),
        }
    }

    match carried {
        Ok(Carried::Waited(CUDA_SUCCESS, results) | Carried::Kept(results)) => Ok(results),
        Ok(Carried::Sent(CUDA_SUCCESS)) => Ok(Vec::new()),
        Ok(Carried::Waited(result, _) | Carried::Sent(result)) => Err(result),
        // The process is ending and Windows sockets are already shut down (WSANOTINITIALISED):
        // the CUDA runtime unloads its code this late. The real driver says the same then.
        Err(error) if error.raw_os_error() == Some(10093) => {
            *state = State::Lost;
            Err(CUDA_ERROR_DEINITIALIZED)
        }
        Err(error) => {
            log(&format!("lost the lender: {error}"));
            *state = State::Lost;
            Err(CUDA_ERROR_DEVICE_UNAVAILABLE)
        }
    }
}

/// Decodes the results of a successful call into the caller's out parameters
pub(crate) fn finish(
    reply: Result<Vec<u8>, CUresult>,
    decode: impl FnOnce(&mut Reader) -> io::Result<()>,
) -> CUresult {
    match reply {
        Ok(results) => match decode(&mut Reader::new(&results)) {
            Ok(()) => CUDA_SUCCESS,
            Err(_) => CUDA_ERROR_UNKNOWN,
        },
        Err(result) => result,
    }
}

/// What every function that cannot be carried yet returns. Each is logged once, so the
/// log lists exactly what an application needed.
pub(crate) fn unsupported(function: u32) -> CUresult {
    static REPORTED: Mutex<Option<HashSet<u32>>> = Mutex::new(None);

    let mut reported = REPORTED
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if reported.get_or_insert_with(HashSet::new).insert(function) {
        log(&format!(
            "{} is not supported yet",
            protocol::function_name(function).unwrap_or("an unknown function")
        ));
    }
    CUDA_ERROR_NOT_SUPPORTED
}

fn args(build: impl FnOnce(&mut Writer)) -> Writer {
    let mut writer = Writer::new();
    build(&mut writer);
    writer
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuInit(flags: c_uint) -> CUresult {
    {
        let mut state = state();
        if !ensure_connected(&mut state) {
            return match *state {
                State::Lost => CUDA_ERROR_DEVICE_UNAVAILABLE,
                _ => CUDA_ERROR_NO_DEVICE,
            };
        }
    }

    finish(
        call(
            id::cuInit,
            &[args(|w| {
                w.u32(flags);
            })
            .as_slice()],
        ),
        |_| Ok(()),
    )
}

/// Hands out this library's export for a symbol, picked from cudaTypedefs.h the way the
/// driver picks it. Needs no lender, so the runtime gets as far as cuInit even without one.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGetProcAddress_v2(
    symbol: *const c_char,
    pfn: *mut *mut c_void,
    cudaVersion: c_int,
    flags: cuuint64_t,
    symbolStatus: *mut CUdriverProcAddressQueryResult,
) -> CUresult {
    if symbol.is_null() || pfn.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let name = unsafe { CStr::from_ptr(symbol) }.to_string_lossy();

    let found = protocol::proc_address(&name, cudaVersion, protocol::per_thread_stream(flags))
        .and_then(|export| {
            generated::exports()
                .get(export)
                .copied()
                .ok_or(protocol::SYMBOL_NOT_FOUND)
        });

    unsafe {
        *pfn = found.map_or(ptr::null_mut(), |address| address.cast_mut());
        if !symbolStatus.is_null() {
            *symbolStatus = found.map_or_else(|status| status, |_| protocol::SYMBOL_FOUND);
        }
    }

    if found.is_ok() {
        CUDA_SUCCESS
    } else {
        CUDA_ERROR_NOT_FOUND
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGetProcAddress(
    symbol: *const c_char,
    pfn: *mut *mut c_void,
    cudaVersion: c_int,
    flags: cuuint64_t,
) -> CUresult {
    unsafe { cuGetProcAddress_v2(symbol, pfn, cudaVersion, flags, ptr::null_mut()) }
}

/// The CUDA runtime asks for undocumented tables of functions here (dark_api.rs). Tables
/// this library lacks are logged once each; like ZLUDA, it answers CUDA_ERROR_INVALID_VALUE,
/// and the runtime carries on without them.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGetExportTable(
    table: *mut *const c_void,
    table_id: *const CUuuid,
) -> CUresult {
    static REPORTED: Mutex<Option<HashSet<[u8; 16]>>> = Mutex::new(None);

    if table.is_null() || table_id.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    let uuid: [u8; 16] = unsafe { (*table_id).bytes }.map(|byte| byte as u8);
    if let Some(found) = dark_api::export_table(&uuid) {
        if logging() {
            let hex: String = uuid.iter().map(|byte| format!("{byte:02x}")).collect();
            log(&format!("export table {hex} provided"));
        }
        unsafe { *table = found };
        return CUDA_SUCCESS;
    }

    let mut reported = REPORTED
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if reported.get_or_insert_with(HashSet::new).insert(uuid) {
        let hex: String = uuid.iter().map(|byte| format!("{byte:02x}")).collect();
        log(&format!("export table {hex} is not provided"));
    }

    unsafe { *table = ptr::null() };
    CUDA_ERROR_INVALID_VALUE
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuDeviceGetLuid(
    luid: *mut c_char,
    deviceNodeMask: *mut c_uint,
    dev: CUdevice,
) -> CUresult {
    finish(
        call(
            id::cuDeviceGetLuid,
            &[args(|w| {
                w.i32(dev);
            })
            .as_slice()],
        ),
        |r| {
            let bytes = r.take(8)?;
            let mask = r.u32()?;
            unsafe {
                if !luid.is_null() {
                    ptr::copy_nonoverlapping(bytes.as_ptr(), luid.cast(), 8);
                }
                if !deviceNodeMask.is_null() {
                    *deviceNodeMask = mask;
                }
            }
            Ok(())
        },
    )
}

/// Error texts must stay valid for the life of the process, so each is kept once
fn error_text(function: u32, error: CUresult, text: *mut *const c_char) -> CUresult {
    static TEXTS: Mutex<Option<HashMap<(u32, CUresult), &'static CString>>> = Mutex::new(None);

    if text.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    let mut texts = TEXTS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let texts = texts.get_or_insert_with(HashMap::new);
    if let Some(cached) = texts.get(&(function, error)) {
        unsafe { *text = cached.as_ptr() };
        return CUDA_SUCCESS;
    }

    let fetched = match call(
        function,
        &[args(|w| {
            w.i32(error);
        })
        .as_slice()],
    ) {
        Ok(results) => Reader::new(&results)
            .bytes()
            .ok()
            .and_then(|bytes| CString::new(bytes).ok()),
        // Without a lender, at least name the errors this library reports itself
        Err(_) => local_error_name(error).map(|name| CString::new(name).unwrap()),
    };

    match fetched {
        Some(fetched) => {
            let fetched: &'static CString = Box::leak(Box::new(fetched));
            texts.insert((function, error), fetched);
            unsafe { *text = fetched.as_ptr() };
            CUDA_SUCCESS
        }
        None => CUDA_ERROR_INVALID_VALUE,
    }
}

fn local_error_name(error: CUresult) -> Option<&'static str> {
    Some(match error {
        CUDA_SUCCESS => "CUDA_SUCCESS",
        CUDA_ERROR_INVALID_VALUE => "CUDA_ERROR_INVALID_VALUE",
        CUDA_ERROR_NOT_INITIALIZED => "CUDA_ERROR_NOT_INITIALIZED",
        CUDA_ERROR_DEVICE_UNAVAILABLE => "CUDA_ERROR_DEVICE_UNAVAILABLE",
        CUDA_ERROR_NO_DEVICE => "CUDA_ERROR_NO_DEVICE",
        CUDA_ERROR_NOT_FOUND => "CUDA_ERROR_NOT_FOUND",
        CUDA_ERROR_NOT_SUPPORTED => "CUDA_ERROR_NOT_SUPPORTED",
        CUDA_ERROR_UNKNOWN => "CUDA_ERROR_UNKNOWN",
        _ => return None,
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGetErrorName(
    error: CUresult,
    text: *mut *const c_char,
) -> CUresult {
    error_text(id::cuGetErrorName, error, text)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGetErrorString(
    error: CUresult,
    text: *mut *const c_char,
) -> CUresult {
    error_text(id::cuGetErrorString, error, text)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuDeviceGetName(
    name: *mut c_char,
    length: c_int,
    device: CUdevice,
) -> CUresult {
    if name.is_null() || length <= 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }

    finish(
        call(
            id::cuDeviceGetName,
            &[args(|w| {
                w.i32(length).i32(device);
            })
            .as_slice()],
        ),
        |r| {
            let text = r.bytes()?;
            let count = text.len().min(length as usize - 1);
            unsafe {
                ptr::copy_nonoverlapping(text.as_ptr(), name.cast(), count);
                *name.add(count) = 0;
            }
            Ok(())
        },
    )
}

/// Host to device: the data goes straight from the caller's buffer to the socket
fn memcpy_htod(
    function: u32,
    destination: CUdeviceptr,
    source: *const c_void,
    size: usize,
) -> CUresult {
    if source.is_null() && size > 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }

    let data = if size == 0 {
        &[][..]
    } else {
        unsafe { slice::from_raw_parts(source.cast::<u8>(), size) }
    };
    let head = args(|w| {
        w.u64(destination);
    });

    finish(upload(function, head.as_slice(), destination, 0, data), |_| Ok(()))
}

fn memcpy_dtoh(
    function: u32,
    destination: *mut c_void,
    source: CUdeviceptr,
    size: usize,
) -> CUresult {
    if destination.is_null() && size > 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }

    finish(
        call(
            function,
            &[args(|w| {
                w.u64(source).u64(size as u64);
            })
            .as_slice()],
        ),
        |r| {
            let data = r.bytes()?;
            if data.len() != size {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "short copy"));
            }
            unsafe { ptr::copy_nonoverlapping(data.as_ptr(), destination.cast(), size) };
            Ok(())
        },
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpyHtoD_v2(
    destination: CUdeviceptr,
    source: *const c_void,
    size: usize,
) -> CUresult {
    memcpy_htod(id::cuMemcpyHtoD_v2, destination, source, size)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpyHtoD_v2_ptds(
    destination: CUdeviceptr,
    source: *const c_void,
    size: usize,
) -> CUresult {
    memcpy_htod(id::cuMemcpyHtoD_v2_ptds, destination, source, size)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpyDtoH_v2(
    destination: *mut c_void,
    source: CUdeviceptr,
    size: usize,
) -> CUresult {
    memcpy_dtoh(id::cuMemcpyDtoH_v2, destination, source, size)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpyDtoH_v2_ptds(
    destination: *mut c_void,
    source: CUdeviceptr,
    size: usize,
) -> CUresult {
    memcpy_dtoh(id::cuMemcpyDtoH_v2_ptds, destination, source, size)
}
