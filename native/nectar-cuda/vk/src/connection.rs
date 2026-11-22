//! The connection to the lender. Calls that need no answer are batched and sent without
//! waiting; a call that needs one waits for the answer carrying its request id, while other
//! threads keep calling. A reader thread hands each answer to the thread waiting for it.
//!
//! The program's threads only append what's going to a queue. A thread of its own compresses it and
//! writes it to the socket, so neither compressing nor a full link holds up the program's frame until
//! the queue is full.
//!
//! Every message is protocol::write_message's frame: the command (or request) id, then the
//! request id (0 when no answer is wanted), then the call. An answer's frame names the request,
//! then holds the result (u64), the first error of a call sent without waiting since the last
//! answer (i32), and the outputs.

use std::collections::HashMap;
use std::io::{self, BufReader, BufWriter, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::time::Duration;
use std::{env, thread};

use nectar_cuda_protocol as protocol;
use nectar_vk_wire::messages::{PACKED, PUSHED, REACHED, STOP_KEEPING};
use protocol::pack::PackedWriter;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);
/// Queued past this, it's worth waking the sending thread without being asked
const WAKE_AT: usize = 1 << 16;
/// Queued past this, the program waits: the link can't carry what it's making
const QUEUE_LIMIT: usize = 16 << 20;

/// An answer as the reader thread hands it over, and how long it took to arrive
type Arrived = io::Result<(Vec<u8>, Duration)>;

pub struct Connection {
    queue: Mutex<Queue>,
    queue_changed: Condvar,
    waiting: Mutex<HashMap<u32, Sender<Arrived>>>,
    next_request: AtomicU32,
    lost: AtomicBool,
    /// Bytes sent and received, for the log
    pub sent: AtomicU64,
    pub received: AtomicU64,
    /// Whether large uploads go by content hash: to a lender over the network, unless
    /// NECTAR_VK_KEEP_UPLOADS says otherwise
    pub keep_uploads: bool,
    /// The lender can keep no more
    pub keep_stopped: AtomicBool,
    /// Bytes that have gone out compressed, and the nanoseconds that took
    packed_bytes: AtomicU64,
    packing_nanos: AtomicU64,
}

/// What's waiting to go
#[derive(Default)]
struct Queue {
    bytes: Vec<u8>,
    /// It should go now, rather than wait for more
    flush: bool,
    /// Went to the socket's buffer with nothing to say it should go on, so a later flush has work to do
    unflushed: bool,
}

/// An answer: the call's result, an earlier call's error, and the outputs
pub struct Answer {
    pub result: u64,
    pub deferred: i32,
    pub body: Vec<u8>,
    /// Where the outputs start in body
    pub at: usize,
    /// How long it took to arrive once it had started to
    pub crossing: Duration,
}

/// An answer on its way
pub struct Later(Receiver<Arrived>);

impl Later {
    /// The answer, if it's come
    pub fn arrived(&self) -> Option<io::Result<u64>> {
        match self.0.try_recv() {
            Ok(Ok((body, _))) => Some(protocol::Reader::new(&body).u64()),
            Ok(Err(error)) => Some(Err(error)),
            Err(std::sync::mpsc::TryRecvError::Empty) => None,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => Some(Err(io::Error::new(io::ErrorKind::NotConnected, "the lender is gone"))),
        }
    }

    pub fn wait(self) -> io::Result<Answer> {
        let (body, crossing) = self
            .0
            .recv()
            .map_err(|_| io::Error::new(io::ErrorKind::NotConnected, "the lender is gone"))??;
        let mut reader = protocol::Reader::new(&body);
        let result = reader.u64()?;
        let deferred = reader.i32()?;
        Ok(Answer { result, deferred, body, at: 12, crossing })
    }
}

/// The lender and the lease's token: NECTAR_VK_SERVER (and NECTAR_VK_TOKEN) for testing,
/// otherwise the GPU nectar-svc has borrowed, when Vulkan is borrowed
fn lender() -> Option<(String, String)> {
    if let Ok(server) = env::var("NECTAR_VK_SERVER") {
        return Some((server, env::var("NECTAR_VK_TOKEN").unwrap_or_default()));
    }
    nectar_lending::BorrowConfig::read()
        .filter(|config| config.has("vulkan"))
        .map(|config| (config.server, config.token))
}

/// The connection, made on first use; None while nothing is borrowed or the lender is gone
pub fn get() -> Option<Arc<Connection>> {
    // Every call comes through here, on all of a program's threads
    static CONNECTION: RwLock<Option<Arc<Connection>>> = RwLock::new(None);
    if let Some(connection) = CONNECTION.read().unwrap_or_else(|p| p.into_inner()).as_ref().filter(|c| !c.lost.load(Ordering::Relaxed)) {
        return Some(Arc::clone(connection));
    }
    let mut slot = CONNECTION.write().unwrap_or_else(|p| p.into_inner());
    if let Some(connection) = slot.as_ref().filter(|c| !c.lost.load(Ordering::Relaxed)) {
        return Some(Arc::clone(connection));
    }
    let (server, token) = lender()?;
    let stream = match nectar_lending::connect(&server, &token, protocol::HELLO_VULKAN, CONNECT_TIMEOUT) {
        Ok(stream) => stream,
        Err(error) => {
            crate::log(&format!("cannot reach the lender {server}: {error}"));
            return None;
        }
    };
    let remote = stream.peer_addr().is_ok_and(|peer| !peer.ip().is_loopback());
    let keep_uploads = env::var("NECTAR_VK_KEEP_UPLOADS").map_or(remote, |value| value == "1");
    // Copying is quicker than compressing where nothing crosses a link
    let packed = env::var("NECTAR_VK_PACK").map_or(remote, |value| value == "1");
    let reader = stream.try_clone().ok()?;
    // The threads below run for the rest of the process: the loader must not unload the driver
    // under them when the program destroys its instance
    pin();
    let mut writer = PackedWriter::new(BufWriter::with_capacity(1 << 16, stream));
    // NECTAR_VK_CAPTURE names a file for a GiB of what's packed, as it was, from NECTAR_VK_CAPTURE_AFTER
    // seconds in: a game's loading would fill it otherwise
    if let Some(file) = env::var_os("NECTAR_VK_CAPTURE").and_then(|path| std::fs::File::create(path).ok()) {
        let after = env::var("NECTAR_VK_CAPTURE_AFTER").ok().and_then(|seconds| seconds.parse().ok()).unwrap_or(0);
        writer.capture(file, 1 << 30, std::time::Instant::now() + std::time::Duration::from_secs(after));
    }
    if packed {
        protocol::write_message(&mut writer, PACKED, &[&0u32.to_le_bytes()]).ok()?;
        writer.start().ok()?;
    }
    let connection = Arc::new(Connection {
        queue: Mutex::new(Queue::default()),
        queue_changed: Condvar::new(),
        waiting: Mutex::new(HashMap::new()),
        next_request: AtomicU32::new(1),
        lost: AtomicBool::new(false),
        sent: AtomicU64::new(0),
        received: AtomicU64::new(0),
        keep_uploads,
        keep_stopped: AtomicBool::new(false),
        packed_bytes: AtomicU64::new(0),
        packing_nanos: AtomicU64::new(0),
    });
    let sending = Arc::clone(&connection);
    thread::spawn(move || sending.send_queue(writer));
    let receiving = Arc::clone(&connection);
    thread::spawn(move || receiving.receive(BufReader::with_capacity(1 << 16, reader)));
    let flushing = Arc::downgrade(&connection);
    thread::spawn(move || {
        // Calls sent without waiting go out within a millisecond or so; on Windows a sleep can take
        // 15 ms, so submits flush themselves
        while let Some(connection) = flushing.upgrade() {
            if connection.lost.load(Ordering::Relaxed) {
                break;
            }
            let _ = connection.flush();
            drop(connection);
            thread::sleep(Duration::from_millis(1));
        }
    });
    *slot = Some(Arc::clone(&connection));
    Some(connection)
}

impl Connection {
    fn receive(&self, mut reader: BufReader<TcpStream>) {
        loop {
            match protocol::read_message_timed(&mut reader) {
                Ok((request, body, crossing)) => {
                    self.received.fetch_add(body.len() as u64 + 8, Ordering::Relaxed);
                    if request == REACHED {
                        let mut news = protocol::Reader::new(&body);
                        if let (Ok(semaphore), Ok(value)) = (news.u64(), news.u64()) {
                            crate::timelines::reached(semaphore, value);
                        }
                        continue;
                    }
                    if request == PUSHED {
                        crate::memory::pushed(&body);
                        continue;
                    }
                    if request == STOP_KEEPING {
                        self.keep_stopped.store(true, Ordering::Relaxed);
                        continue;
                    }
                    let sender = self.waiting.lock().unwrap_or_else(|p| p.into_inner()).remove(&request);
                    if let Some(sender) = sender {
                        let _ = sender.send(Ok((body, crossing)));
                    }
                }
                Err(error) => {
                    crate::log(&format!("lost the lender: {error}"));
                    self.lost.store(true, Ordering::Relaxed);
                    self.queue_changed.notify_all();
                    crate::timelines::lost();
                    for (_, sender) in self.waiting.lock().unwrap_or_else(|p| p.into_inner()).drain() {
                        let _ = sender.send(Err(io::Error::new(error.kind(), error.to_string())));
                    }
                    return;
                }
            }
        }
    }

    /// The sending thread: compresses what's queued and writes it out
    fn send_queue(&self, mut writer: PackedWriter<BufWriter<TcpStream>>) {
        let mut sending = Vec::new();
        loop {
            let flush = {
                let mut queue = self.queue.lock().unwrap_or_else(|p| p.into_inner());
                while queue.bytes.is_empty() && !queue.flush {
                    if self.lost.load(Ordering::Relaxed) {
                        return;
                    }
                    queue = self.queue_changed.wait(queue).unwrap_or_else(|p| p.into_inner());
                }
                std::mem::swap(&mut queue.bytes, &mut sending);
                self.queue_changed.notify_all();
                let flush = std::mem::take(&mut queue.flush);
                queue.unflushed = !flush;
                flush
            };
            let written = writer.write_all(&sending).and_then(|()| if flush { writer.flush() } else { Ok(()) });
            sending.clear();
            self.packed_bytes.store(writer.wrote, Ordering::Relaxed);
            self.packing_nanos.store(writer.packing, Ordering::Relaxed);
            if let Err(error) = written {
                crate::log(&format!("lost the lender: {error}"));
                self.lost.store(true, Ordering::Relaxed);
                self.queue_changed.notify_all();
                return;
            }
        }
    }

    /// Adds a message to the queue
    fn enqueue(&self, head: u32, request: u32, body: &[u8], flush: bool) -> io::Result<()> {
        let mut queue = self.queue.lock().unwrap_or_else(|p| p.into_inner());
        while queue.bytes.len() > QUEUE_LIMIT {
            if self.lost.load(Ordering::Relaxed) {
                return Err(io::Error::new(io::ErrorKind::NotConnected, "the lender is gone"));
            }
            queue = self.queue_changed.wait(queue).unwrap_or_else(|p| p.into_inner());
        }
        protocol::write_message(&mut queue.bytes, head, &[&request.to_le_bytes(), body])?;
        queue.flush |= flush;
        if flush || queue.bytes.len() >= WAKE_AT {
            self.queue_changed.notify_all();
        }
        Ok(())
    }

    /// Bytes that have gone out compressed, or 0 if nothing is compressed, and the nanoseconds that
    /// compressing took
    pub fn sent_packed(&self) -> (u64, u64) {
        (self.packed_bytes.load(Ordering::Relaxed), self.packing_nanos.load(Ordering::Relaxed))
    }

    /// Sends what's waiting now, rather than within the flushing thread's next wake-up
    pub fn flush(&self) -> io::Result<()> {
        let mut queue = self.queue.lock().unwrap_or_else(|p| p.into_inner());
        if !queue.bytes.is_empty() || queue.unflushed {
            queue.flush = true;
            self.queue_changed.notify_all();
        }
        Ok(())
    }

    /// Sends without waiting for an answer
    pub fn send(&self, head: u32, body: &[u8]) -> io::Result<()> {
        if self.lost.load(Ordering::Relaxed) {
            return Err(io::Error::new(io::ErrorKind::NotConnected, "the lender is gone"));
        }
        self.enqueue(head, 0, body, false)?;
        self.sent.fetch_add(body.len() as u64 + 12, Ordering::Relaxed);
        Ok(())
    }

    /// Sends and waits for the answer
    pub fn ask(&self, head: u32, body: &[u8]) -> io::Result<Answer> {
        self.ask_later(head, body)?.wait()
    }

    /// Sends at once, and the answer can be waited for later
    pub fn ask_later(&self, head: u32, body: &[u8]) -> io::Result<Later> {
        if self.lost.load(Ordering::Relaxed) {
            return Err(io::Error::new(io::ErrorKind::NotConnected, "the lender is gone"));
        }
        let request = self.next_request.fetch_add(1, Ordering::Relaxed).max(1);
        let (sender, receiver) = channel();
        self.waiting.lock().unwrap_or_else(|p| p.into_inner()).insert(request, sender);
        if let Err(error) = self.enqueue(head, request, body, true) {
            self.waiting.lock().unwrap_or_else(|p| p.into_inner()).remove(&request);
            return Err(error);
        }
        self.sent.fetch_add(body.len() as u64 + 12, Ordering::Relaxed);
        Ok(Later(receiver))
    }
}

impl Answer {
    pub fn outputs(&self) -> protocol::Reader<'_> {
        protocol::Reader::new(&self.body[self.at..])
    }
}

/// Keeps this driver loaded until the process ends
#[cfg(windows)]
fn pin() {
    use windows_sys::Win32::System::LibraryLoader::{
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_PIN, GetModuleHandleExW,
    };
    let mut module = std::ptr::null_mut();
    let flags = GET_MODULE_HANDLE_EX_FLAG_PIN | GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS;
    unsafe { GetModuleHandleExW(flags, pin as *const u16, &mut module) };
}

/// Keeps this driver loaded until the process ends: opened again with RTLD_NODELETE
#[cfg(unix)]
fn pin() {
    let mut info: libc::Dl_info = unsafe { std::mem::zeroed() };
    if unsafe { libc::dladdr(pin as *const libc::c_void, &mut info) } != 0 && !info.dli_fname.is_null() {
        unsafe { libc::dlopen(info.dli_fname, libc::RTLD_NOW | libc::RTLD_NODELETE) };
    }
}
