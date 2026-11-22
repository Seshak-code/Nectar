//! The Vulkan personality: runs a borrower's Vulkan calls on this computer's Vulkan driver,
//! through the system's Vulkan loader.
//!
//! Calls run in the order they arrive, on this connection's thread. Waits for the GPU
//! (vkWaitForFences and the like) run on a thread of their own, so a wait can't hold up the
//! call that would end it, as when one of the program's threads waits on a fence another is
//! about to submit. Their answers name the request, so they can come back out of order.
//!
//! Pipelines sent without waiting compile on threads of their own too (Compiler), so that a slow
//! compile doesn't hold up the frames behind it. Only what uses a pipeline waits for it. What a
//! compile reads, which the program may destroy as soon as it thinks the pipeline exists, is
//! destroyed once no compile is running.
//!
//! Handles: the borrower chooses ids for what it creates; what this side hands out (physical
//! devices, queues) gets ids with the top bit set. Memory the borrower maps is mapped here as
//! a whole, and kept in step page by page (vkwire's messages).

use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::{CStr, CString, c_char, c_void};
use std::io::{self, BufReader, BufWriter, Write};
use std::net::TcpStream;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::hash::BuildHasherDefault;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use std::{ptr, thread};

use libloading::Library;
use nectar_cuda_protocol::pack::PackedReader;
use nectar_cuda_protocol::{self as protocol, Reader, Writer};
use nectar_vk_wire::bc::Kind;
use nectar_vk_wire::frame::Encoder;
use nectar_vk_wire::wavelet;
use nectar_vk_wire::frame_shader;
use nectar_vk_wire::messages::{FETCH, FRAME, GPU_USAGE, HOLD, KEEP, KEPT, CODED_PART, DECODE_MIP, DESCRIPTOR_COPIES, ECHO, FILL_MIP, FILL_MIP_FROM, MAP, PACKED, PAGE, PUSHED, REACHED, REFRESH, STOP_KEEPING, UNMAP, VIDEO_CLOSE, VIDEO_FRAME, VIDEO_OPEN, WATCH, WRITE};
use nectar_vk_wire::tables::{Level, Mode, Ret};
use nectar_vk_wire::descriptor_copies;
use nectar_vk_wire::video_encode;
use nectar_vk_wire::wire::{self, Arena, Handles};
use nectar_vk_wire::{COMMANDS, thunks};
use xxhash_rust::xxh3::xxh3_64;

use crate::cache::{Cache, Hash};

type GetProcAddr = unsafe extern "system" fn(u64, *const c_char) -> *const c_void;
type MapMemory = unsafe extern "system" fn(u64, u64, u64, u64, u32, *mut *mut c_void) -> i32;
type UnmapMemory = unsafe extern "system" fn(u64, u64);
type UpdateDescriptorSets = unsafe extern "system" fn(u64, u32, *const c_void, u32, *const nectar_vk_sys::VkCopyDescriptorSet);
type DeviceWaitIdle = unsafe extern "system" fn(u64) -> i32;
type Destroy = unsafe extern "system" fn(u64, *const c_void);
type FreeMemory = unsafe extern "system" fn(u64, u64, *const c_void);
type WaitForFences = unsafe extern "system" fn(u64, u32, *const u64, u32, u64) -> i32;
type WaitSemaphores = unsafe extern "system" fn(u64, *const nectar_vk_sys::VkSemaphoreWaitInfo, u64) -> i32;

const VK_ERROR_FEATURE_NOT_PRESENT: i32 = -8;
const VK_TIMEOUT: i32 = 2;
/// How often a watch looks up to see if it's still wanted, in nanoseconds
const WATCH_LOOK_UP: u64 = 100_000_000;
const VK_ERROR_MEMORY_MAP_FAILED: i32 = -5;
const VK_WHOLE_SIZE: u64 = u64::MAX;
/// How long a frame may wait for its copy, in nanoseconds. Only a GPU in trouble takes this long.
const FRAME_WAIT: u64 = 5_000_000_000;
/// What a compile may be reading when the program destroys it
const COMPILES_READ: &[&str] = &[
    "vkDestroyShaderModule",
    "vkDestroyPipelineLayout",
    "vkDestroyRenderPass",
    "vkDestroyPipelineCache",
    "vkDestroyPipeline",
    "vkDestroyDescriptorSetLayout",
    "vkDestroySampler",
    "vkDestroySamplerYcbcrConversion",
];
/// Calls that are about every pipeline made so far
const AFTER_COMPILES: &[&str] = &["vkGetPipelineCacheData", "vkMergePipelineCaches", "vkDestroyDevice"];
/// Destroys held back past this many wait for the compiles
const MOST_PARKED: usize = 4096;
/// Ids this side gives out
const LENDER_IDS: u64 = 1 << 63;

struct Loader {
    _library: Library,
    get_instance_proc_addr: GetProcAddr,
}

/// The system's Vulkan loader, loaded once
fn loader() -> Option<&'static Loader> {
    static LOADER: OnceLock<Option<Loader>> = OnceLock::new();
    LOADER
        .get_or_init(|| {
            let names: &[&str] = if cfg!(windows) {
                &["vulkan-1.dll"]
            } else if cfg!(target_os = "macos") {
                &["libvulkan.1.dylib", "libMoltenVK.dylib"]
            } else {
                &["libvulkan.so.1", "libvulkan.so"]
            };
            let library = names.iter().find_map(|name| unsafe { Library::new(name) }.ok())?;
            let get = unsafe { *library.get::<GetProcAddr>(b"vkGetInstanceProcAddr\0").ok()? };
            Some(Loader { _library: library, get_instance_proc_addr: get })
        })
        .as_ref()
}

fn command_index(id: u32) -> Option<usize> {
    static INDEX: OnceLock<HashMap<u32, usize>> = OnceLock::new();
    INDEX
        .get_or_init(|| COMMANDS.iter().enumerate().map(|(i, c)| (c.id, i)).collect())
        .get(&id)
        .copied()
}

/// Copies out of mapped memory. Memory for uploads is usually write-combined, which ordinary reads
/// cross at about 100 MB/s; on x86 streaming loads read it many times faster, and on other memory
/// they're ordinary loads.
fn read_mapped(source: *const u8, target: &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("sse4.1") {
        unsafe { read_streaming(source, target) };
        return;
    }
    unsafe { ptr::copy_nonoverlapping(source, target.as_mut_ptr(), target.len()) };
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
unsafe fn read_streaming(source: *const u8, target: &mut [u8]) {
    use std::arch::x86_64::{__m128i, _mm_mfence, _mm_storeu_si128, _mm_stream_load_si128};

    const LINE: usize = 16;
    let length = target.len();
    let mut at = 0;
    unsafe {
        _mm_mfence();
        // Up to the first aligned address, and after the last, one aligned load covers the bytes
        while at < length {
            let address = source.add(at);
            let aligned = (address as usize & !(LINE - 1)) as *const __m128i;
            let skip = address as usize & (LINE - 1);
            let take = (LINE - skip).min(length - at);
            if take == LINE {
                _mm_storeu_si128(target.as_mut_ptr().add(at).cast(), _mm_stream_load_si128(aligned as *mut __m128i));
            } else {
                let mut line = [0u8; LINE];
                _mm_storeu_si128(line.as_mut_ptr().cast(), _mm_stream_load_si128(aligned as *mut __m128i));
                target[at..at + take].copy_from_slice(&line[skip..skip + take]);
            }
            at += take;
        }
    }
}

/// Each page's hash as the borrower last had it. A watching thread shares these with the session, and
/// looks at the memory only while the mapping is alive.
struct PageHashes {
    alive: Mutex<bool>,
    pages: Vec<AtomicU64>,
}

struct Mapping {
    pointer: *mut u8,
    size: usize,
    hashes: Arc<PageHashes>,
}

impl Drop for Mapping {
    fn drop(&mut self) {
        *self.hashes.alive.lock().unwrap_or_else(|p| p.into_inner()) = false;
    }
}

impl Mapping {
    fn new(pointer: *mut u8, size: usize, hashes: Vec<u64>) -> Self {
        let hashes = PageHashes { alive: Mutex::new(true), pages: hashes.into_iter().map(AtomicU64::new).collect() };
        Self { pointer, size, hashes: Arc::new(hashes) }
    }

    /// Adds the runs of pages in `pages` that changed since the borrower last had them, and
    /// takes them as the borrower's
    fn changed(&self, id: u64, pages: std::ops::Range<usize>, runs: &mut Vec<(u64, usize, usize, *mut u8)>) {
        changed_pages(&self.hashes, self.pointer, self.size, id, pages, runs);
    }
}

fn changed_pages(hashes: &PageHashes, pointer: *mut u8, size: usize, id: u64, pages: std::ops::Range<usize>, runs: &mut Vec<(u64, usize, usize, *mut u8)>) {
    {
        let mut buffer = [0u8; PAGE];
        let mut run: Option<usize> = None;
        let end = pages.end.min(hashes.pages.len());
        for page in pages.start..=end {
            let changed = page < end && {
                let start = page * PAGE;
                let length = (start + PAGE).min(size) - start;
                read_mapped(unsafe { pointer.add(start) }, &mut buffer[..length]);
                let hash = xxh3_64(&buffer[..length]);
                hashes.pages[page].swap(hash, Ordering::Relaxed) != hash
            };
            match (changed, run) {
                (true, None) => run = Some(page),
                (false, Some(first)) => {
                    runs.push((id, first * PAGE, (page * PAGE).min(size), pointer));
                    run = None;
                }
                _ => {}
            }
        }
    }
}

#[derive(Default)]
struct Session {
    ids: Handled<u64>,
    reals: Handled<u64>,
    next_id: u64,
    /// The device each dispatchable handle belongs to
    owner: Handled<u64>,
    /// The device a new handle belongs to, while a call's outputs are recorded
    current_owner: u64,
    instances: Vec<u64>,
    devices: Vec<u64>,
    procs: HashMap<(u64, usize), usize, BuildHasherDefault<Quick>>,
    /// Used again from call to call
    arena: Arena,
    /// Used again from one DESCRIPTOR_COPIES to the next
    copying: Vec<nectar_vk_sys::VkCopyDescriptorSet>,
    /// The device and its vkUpdateDescriptorSets
    update_sets: Option<(u64, UpdateDescriptorSets)>,
    /// Device and size of each memory object, by its real handle
    memory: HashMap<u64, (u64, u64)>,
    mappings: HashMap<u64, Mapping>,
    /// How many times mapped memory has been read for the borrower, and held while reading it and sending it
    reading: Arc<Mutex<u64>>,
    /// The first error of a call sent without waiting, for the next answer
    deferred: i32,
    /// Uploads kept by content hash, shared with the CUDA personality
    cache: Option<Arc<Cache>>,
    /// The borrower has been told to stop sending KEEPs
    told_to_stop: bool,
    /// Bytes of mips made from the ones below, for the log
    mips_made: u64,
    /// Mips coded by wavelet.rs arriving, by the memory they're to end up in
    coded: HashMap<u64, Vec<u8>>,
    /// The thread that decodes them, one at a time
    decoder: Option<(Sender<DecodeJob>, JoinHandle<()>)>,
    /// What came in that form, and what it was decoded to, for the log
    mips_decoded: (u64, u64),
    /// Uploads that came by hash and the bytes that saved, and the time getting them took, for the log
    uploads_kept: (u64, u64, Duration),
    /// What codes each swapchain's frames, by the first memory they're copied to, or by the key of its video
    coders: HashMap<u64, Coder>,
    /// The devices created with what encoding video takes
    video: HashMap<u64, VideoDevice>,
    compiler: Arc<Compiler>,
    compile_threads: Vec<JoinHandle<()>>,
    /// Destroys held back until no compile is running
    parked: Vec<Parked>,
    /// The program used a pipeline that couldn't be compiled, or a shader that isn't here
    broken: bool,
    /// How long the ordered stream waited for compiles, for the log
    compile_waits: Duration,
    watches: Arc<Watches>,
    /// Shaders that came by hash, and the bytes that saved, for the log
    shaders_kept: (u64, u64),
    /// What couldn't be kept in the cache, for the rest of this connection, as the borrower relies on it
    held: HashMap<Hash, Vec<u8>>,
}

const VK_ERROR_FORMAT_NOT_SUPPORTED: i32 = -11;

/// A coded mip to decode into mapped memory, which the borrower leaves mapped until it's answered
struct DecodeJob {
    coded: Vec<u8>,
    kind: Kind,
    srgb: bool,
    across: usize,
    down: usize,
    to: usize,
    request: u32,
}

/// Decodes coded mips off the stream, so the game's calls don't wait for them
fn decode_mips(jobs: Receiver<DecodeJob>, out: Out) {
    let threads = thread::available_parallelism().map_or(1, |count| (count.get() / 2).max(1));
    for job in jobs {
        // The mapping stays until the answer goes, and the size was checked against it
        let done = unsafe { wavelet::decode_mip(job.kind, job.srgb, &job.coded, job.across, job.down, &wavelet::QUALITY, threads, job.to as *mut u8) };
        let result = if done { 0 } else { VK_ERROR_FORMAT_NOT_SUPPORTED };
        let _ = answer(&out, job.request, result as i64 as u64, 0, &[]);
        let _ = out.lock().unwrap_or_else(|p| p.into_inner()).flush();
    }
}

/// Part of a mapping a watching thread is to send when the work it's watching is done
struct Pushing {
    memory: u64,
    pages: std::ops::Range<usize>,
    hashes: Arc<PageHashes>,
    pointer: *mut u8,
    size: usize,
}

// The pointer is only followed while `hashes` says the mapping is alive
unsafe impl Send for Pushing {}

/// What's changed in what's to be pushed: memory, offset, bytes
fn pushed_runs(pushing: &[Pushing]) -> Vec<(u64, u64, Vec<u8>)> {
    let mut out = Vec::new();
    for part in pushing {
        let alive = part.hashes.alive.lock().unwrap_or_else(|p| p.into_inner());
        if !*alive {
            continue;
        }
        let mut runs = Vec::new();
        changed_pages(&part.hashes, part.pointer, part.size, part.memory, part.pages.clone(), &mut runs);
        for (memory, start, end, pointer) in runs {
            let mut bytes = vec![0u8; end - start];
            read_mapped(unsafe { pointer.add(start) }, &mut bytes);
            out.push((memory, start as u64, bytes));
        }
    }
    out
}

/// Threads waiting for timeline values the borrower wants to hear about (WATCH)
#[derive(Default)]
struct Watches {
    state: Mutex<Watching>,
    changed: Condvar,
}

#[derive(Default)]
struct Watching {
    /// How many are waiting on each semaphore, by real handle
    live: HashMap<u64, usize>,
    /// Semaphores about to be destroyed
    cancelled: HashSet<u64>,
    /// All of them are to stop
    closing: bool,
}

impl Watches {
    fn lock(&self) -> std::sync::MutexGuard<'_, Watching> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn wanted(&self, semaphore: u64) -> bool {
        let state = self.lock();
        !state.closing && !state.cancelled.contains(&semaphore)
    }

    fn finished(&self, semaphore: u64) {
        let mut state = self.lock();
        if let Some(count) = state.live.get_mut(&semaphore) {
            *count -= 1;
            if *count == 0 {
                state.live.remove(&semaphore);
            }
        }
        self.changed.notify_all();
    }

    /// Stops those on a semaphore before it's destroyed
    fn cancel(&self, semaphore: u64) {
        let mut state = self.lock();
        if !state.live.contains_key(&semaphore) {
            return;
        }
        state.cancelled.insert(semaphore);
        while state.live.contains_key(&semaphore) {
            state = self.changed.wait(state).unwrap_or_else(|p| p.into_inner());
        }
        state.cancelled.remove(&semaphore);
    }

    /// Stops them all
    fn cancel_all(&self) {
        let mut state = self.lock();
        state.closing = true;
        while !state.live.is_empty() {
            state = self.changed.wait(state).unwrap_or_else(|p| p.into_inner());
        }
        state.closing = false;
    }
}

/// A hasher for handles, which come from the program and from the driver and aren't chosen to collide. Every
/// handle in every command is looked up, and SipHash came to a large share of the stream's time.
#[derive(Default, Clone, Copy)]
struct Quick(u64);

impl std::hash::Hasher for Quick {
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.write_u64(u64::from_le_bytes(word));
        }
    }

    fn write_u64(&mut self, value: u64) {
        self.0 = (self.0.rotate_left(5) ^ value).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }

    fn write_usize(&mut self, value: usize) {
        self.write_u64(value as u64);
    }

    fn finish(&self) -> u64 {
        self.0 ^ (self.0 >> 29)
    }
}

type Handled<V> = HashMap<u64, V, BuildHasherDefault<Quick>>;

/// A call decoded and ready to run, which can move to another thread
struct Ready {
    index: usize,
    function: usize,
    _arena: Arena,
    call: wire::Call,
}

// What the pointers in a decoded call point to is in its arena, or is the driver's
unsafe impl Send for Ready {}

impl Ready {
    fn run(&self) -> u64 {
        unsafe { thunks::call(self.index, self.function as *const c_void, &self.call.slots) }
    }
}

struct Parked(Ready);

/// Pipelines being compiled
#[derive(Default)]
struct Compiling {
    jobs: VecDeque<(Ready, u64)>,
    running: usize,
    /// Ids of pipelines not yet compiled
    pending: HashSet<u64>,
    /// Compiled, not yet taken into the session's handles: id, real, device
    done: Vec<(u64, u64, u64)>,
    failed: HashSet<u64>,
    closing: bool,
    /// How many, and the time they took between them
    compiled: u64,
    took: Duration,
}

#[derive(Default)]
struct Compiler {
    state: Mutex<Compiling>,
    changed: Condvar,
}

impl Compiler {
    fn lock(&self) -> std::sync::MutexGuard<'_, Compiling> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn work(&self) {
        let mut state = self.lock();
        loop {
            let Some((ready, device)) = state.jobs.pop_front() else {
                if state.closing {
                    return;
                }
                state = self.changed.wait(state).unwrap_or_else(|p| p.into_inner());
                continue;
            };
            state.running += 1;
            drop(state);
            let started = Instant::now();
            let result = ready.run() as i64 as i32;
            let took = started.elapsed();
            let mut bound = Bound::default();
            let command = &COMMANDS[ready.index];
            if result >= 0 {
                let mut w = Writer::new();
                let _ = unsafe { wire::encode_outputs(command, &ready.call, true, &mut w, &mut bound) };
            } else {
                eprintln!("vulkan: {} failed with {result}", command.name);
            }
            state = self.lock();
            state.running -= 1;
            state.compiled += 1;
            state.took += took;
            for id in ready.call.created_ids() {
                state.pending.remove(&id);
                match bound.0.iter().find(|(bound, real)| *bound == id && *real != 0) {
                    Some(&(id, real)) => state.done.push((id, real, device)),
                    None => {
                        state.failed.insert(id);
                    }
                }
            }
            self.changed.notify_all();
            drop(ready);
        }
    }
}

/// The handles a call made
#[derive(Default)]
struct Bound(Vec<(u64, u64)>);

impl Handles for Bound {
    fn outgoing(&mut self, _: bool, value: u64) -> u64 {
        value
    }
    fn incoming(&mut self, _: bool, id: u64) -> u64 {
        id
    }
    fn create(&mut self, _: bool) -> (u64, u64) {
        unreachable!()
    }
    fn bind(&mut self, id: u64, real: u64) {
        self.0.push((id, real));
    }
}

/// One frame to wait for and code
struct FrameJob {
    arrived: Instant,
    request: u32,
    deferred: i32,
    wait: WaitForFences,
    device: u64,
    fence: u64,
    pixels: usize,
    width: u32,
    height: u32,
    quality: u32,
    /// `pixels` holds what the shader already coded
    coded_on_gpu: bool,
}

/// One frame to wait for and have the card's encoder make a picture of
struct VideoJob {
    arrived: Instant,
    request: u32,
    deferred: i32,
    wait: WaitForFences,
    device: u64,
    fence: u64,
    /// The buffer the frame's NV12 is in
    nv12: u64,
    quality: u32,
    key: bool,
}

enum Job {
    Frame(FrameJob),
    Video(VideoJob),
}

/// What a device created here encodes video with
struct VideoDevice {
    physical: u64,
    family: u32,
    /// Held while the family's one queue is in use, as each swapchain's encoder uses it from its own thread
    queue: Arc<Mutex<()>>,
}

/// What a device being created was asked for before the encoder's queue and extensions were added, and
/// the lists they were added to, which the call reads
struct VideoOffer {
    physical: u64,
    family: u32,
    queues: (u32, *const nectar_vk_sys::VkDeviceQueueCreateInfo),
    extensions: (u32, *const *const c_char),
    _queues: Vec<nectar_vk_sys::VkDeviceQueueCreateInfo>,
    _extensions: Vec<*const c_char>,
    _priority: Box<f32>,
}

impl VideoOffer {
    /// Puts the request back as the program made it
    unsafe fn take_back(&self, info: u64) {
        let info = info as usize as *mut nectar_vk_sys::VkDeviceCreateInfo;
        unsafe {
            ((*info).queueCreateInfoCount, (*info).pQueueCreateInfos) = self.queues;
            ((*info).enabledExtensionCount, (*info).ppEnabledExtensionNames) = self.extensions;
        }
    }
}

/// A thread coding one swapchain's frames in the order they came. Each is coded against the one
/// before, so the order is what the borrower must decode them in.
struct Coder {
    jobs: Sender<Job>,
    thread: JoinHandle<()>,
    /// The device its video encoder is on, if it has one
    video_device: u64,
}

/// The card's encoder, and what's held while its queue is in use
struct Video(video_encode::Encoder, Arc<Mutex<()>>);

impl Coder {
    fn new(out: Out, video: Option<Video>, video_device: u64) -> Self {
        let (jobs, received) = channel::<Job>();
        let thread = thread::spawn(move || {
            let mut encoder = Encoder::default();
            let mut video = video;
            for job in received {
                let job = match job {
                    Job::Frame(job) => job,
                    Job::Video(job) => {
                        let result = unsafe { (job.wait)(job.device, 1, &job.fence, 1, FRAME_WAIT) };
                        if result != 0 {
                            eprintln!("vulkan: a frame's colours weren't ready: {result}");
                            let _ = answer(&out, job.request, result as i64 as u64, job.deferred, &[]);
                            continue;
                        }
                        let Some(Video(encoder, queue)) = video.as_mut() else {
                            let _ = answer(&out, job.request, VK_ERROR_FEATURE_NOT_PRESENT as i64 as u64, job.deferred, &[]);
                            continue;
                        };
                        let coding = Instant::now();
                        let alone = queue.lock().unwrap_or_else(|p| p.into_inner());
                        let coded = unsafe { encoder.encode(job.nv12, job.quality, job.key) };
                        drop(alone);
                        let _ = match coded {
                            Ok((coded, key)) => {
                                let mut w = Writer::new();
                                w.u32(job.arrived.elapsed().as_micros() as u32).u32(coding.elapsed().as_micros() as u32).u8(key as u8).bytes(coded);
                                answer(&out, job.request, 0, job.deferred, w.as_slice())
                            }
                            Err(error) => {
                                eprintln!("vulkan: a frame wasn't encoded: {error}");
                                answer(&out, job.request, VK_ERROR_FEATURE_NOT_PRESENT as i64 as u64, job.deferred, &[])
                            }
                        };
                        continue;
                    }
                };
                let result = unsafe { (job.wait)(job.device, 1, &job.fence, 1, FRAME_WAIT) };
                if result != 0 {
                    eprintln!("vulkan: a frame's copy didn't finish: {result}");
                    let _ = answer(&out, job.request, result as i64 as u64, job.deferred, &[]);
                    continue;
                }
                let coding = Instant::now();
                let coded = if job.coded_on_gpu {
                    let length = frame_shader::output_size(job.width, job.height) as usize;
                    let output = unsafe { std::slice::from_raw_parts(job.pixels as *const u8, length) };
                    frame_shader::gather(output, job.width, job.height, job.quality)
                } else {
                    let length = job.width as usize * job.height as usize * 4;
                    let pixels = unsafe { std::slice::from_raw_parts(job.pixels as *const u8, length) };
                    encoder.encode(pixels, job.width, job.height, job.quality)
                };
                let mut w = Writer::new();
                w.u32(job.arrived.elapsed().as_micros() as u32).u32(coding.elapsed().as_micros() as u32).bytes(&coded);
                let _ = answer(&out, job.request, 0, job.deferred, w.as_slice());
            }
            if let Some(Video(mut encoder, queue)) = video {
                let _alone = queue.lock().unwrap_or_else(|p| p.into_inner());
                unsafe { encoder.destroy() };
            }
        });
        Self { jobs, thread, video_device }
    }

    /// Ends once the frames it has are done
    fn finish(self) {
        drop(self.jobs);
        let _ = self.thread.join();
    }
}

impl Handles for Session {
    fn outgoing(&mut self, dispatchable: bool, real: u64) -> u64 {
        if real == 0 {
            return 0;
        }
        if let Some(&id) = self.reals.get(&real) {
            return id;
        }
        self.next_id += 1;
        let id = LENDER_IDS | self.next_id;
        self.ids.insert(id, real);
        self.reals.insert(real, id);
        if dispatchable {
            self.owner.insert(real, self.current_owner);
        }
        id
    }

    fn incoming(&mut self, _: bool, id: u64) -> u64 {
        if id == 0 {
            return 0;
        }
        if let Some(&real) = self.ids.get(&id) {
            return real;
        }
        self.compiled(id)
    }

    fn create(&mut self, _: bool) -> (u64, u64) {
        unreachable!("the lender never chooses a borrower's handle")
    }

    fn bind(&mut self, id: u64, real: u64) {
        self.ids.insert(id, real);
        self.reals.insert(real, id);
        self.owner.insert(real, self.current_owner);
    }

    fn code_keep(&mut self, hash: &Hash, code: &[u8]) {
        self.keep_or_hold(hash, code);
    }

    fn code_in(&mut self, hash: &Hash, size: u64) -> Option<Vec<u8>> {
        let code = self.find_kept(hash, size);
        match &code {
            Some(_) => {
                self.shaders_kept.0 += 1;
                self.shaders_kept.1 += size;
            }
            // Nothing correct can follow
            None => self.broken = true,
        }
        code
    }
}

/// Outputs of a wait carry no handles
struct NoHandles;

impl Handles for NoHandles {
    fn outgoing(&mut self, _: bool, value: u64) -> u64 {
        value
    }
    fn incoming(&mut self, _: bool, id: u64) -> u64 {
        id
    }
    fn create(&mut self, _: bool) -> (u64, u64) {
        unreachable!()
    }
    fn bind(&mut self, _: u64, _: u64) {}
}

type Out = Arc<Mutex<BufWriter<TcpStream>>>;

fn answer(out: &Out, request: u32, result: u64, deferred: i32, outputs: &[u8]) -> io::Result<()> {
    let mut head = Writer::new();
    head.u64(result).i32(deferred);
    let mut writer = out.lock().unwrap_or_else(|p| p.into_inner());
    protocol::write_message(&mut *writer, request, &[head.as_slice(), outputs])?;
    writer.flush()
}

impl Session {
    /// A pipeline that wasn't in the handles: waits for its compile if it's still running
    fn compiled(&mut self, id: u64) -> u64 {
        let compiler = Arc::clone(&self.compiler);
        let mut state = compiler.lock();
        let started = Instant::now();
        while state.pending.contains(&id) {
            state = compiler.changed.wait(state).unwrap_or_else(|p| p.into_inner());
        }
        self.compile_waits += started.elapsed();
        let failed = state.failed.contains(&id);
        let done = std::mem::take(&mut state.done);
        drop(state);
        for (id, real, device) in done {
            self.ids.insert(id, real);
            self.reals.insert(real, id);
            self.owner.insert(real, device);
        }
        self.broken |= failed;
        self.ids.get(&id).copied().unwrap_or(0)
    }

    /// Hands a pipeline call to the compile threads
    fn compile_later(&mut self, ready: Ready, device: u64) {
        if self.compile_threads.is_empty() {
            let threads = thread::available_parallelism().map_or(4, |count| count.get() / 2).clamp(2, 8);
            for _ in 0..threads {
                let compiler = Arc::clone(&self.compiler);
                // Compilers can go deep
                if let Ok(thread) = thread::Builder::new().stack_size(8 << 20).spawn(move || compiler.work()) {
                    self.compile_threads.push(thread);
                }
            }
        }
        let mut state = self.compiler.lock();
        state.pending.extend(ready.call.created_ids());
        state.jobs.push_back((ready, device));
        self.compiler.changed.notify_all();
    }

    fn compiling(&self) -> bool {
        let state = self.compiler.lock();
        state.running > 0 || !state.jobs.is_empty()
    }

    fn run_parked(&mut self) {
        for Parked(ready) in std::mem::take(&mut self.parked) {
            ready.run();
        }
    }

    /// Waits until no compile is left, and runs the destroys held back for them
    fn after_compiles(&mut self) {
        let compiler = Arc::clone(&self.compiler);
        let started = Instant::now();
        let mut state = compiler.lock();
        while state.running > 0 || !state.jobs.is_empty() {
            state = compiler.changed.wait(state).unwrap_or_else(|p| p.into_inner());
        }
        drop(state);
        self.compile_waits += started.elapsed();
        self.run_parked();
    }

    fn proc_address(&mut self, loader: &Loader, index: usize, first: u64) -> *const c_void {
        let command = &COMMANDS[index];
        let (dispatch, device) = match command.level {
            Level::Global => (0, false),
            Level::Instance => (self.instances.last().copied().unwrap_or(0), false),
            Level::Device => (self.owner.get(&first).copied().filter(|&d| d != 0).unwrap_or(first), true),
        };
        if let Some(&address) = self.procs.get(&(dispatch, index)) {
            return address as *const c_void;
        }
        let instance = self.instances.last().copied().unwrap_or(0);
        let get_device: Option<GetProcAddr> = device.then(|| unsafe {
            std::mem::transmute::<*const c_void, Option<GetProcAddr>>((loader.get_instance_proc_addr)(
                instance,
                c"vkGetDeviceProcAddr".as_ptr(),
            ))
        })
        .flatten();
        // A command promoted to core may be known to an older driver only by its extension name
        let mut address = ptr::null();
        for suffix in ["", "KHR", "EXT"] {
            let Ok(name) = CString::new(format!("{}{suffix}", command.name)) else { continue };
            address = unsafe {
                match get_device {
                    Some(get) => get(dispatch, name.as_ptr()),
                    None => (loader.get_instance_proc_addr)(dispatch, name.as_ptr()),
                }
            };
            if !address.is_null() {
                break;
            }
        }
        self.procs.insert((dispatch, index), address as usize);
        address
    }

    fn device_function<T>(&mut self, loader: &Loader, device: u64, name: &str) -> Option<T> {
        let index = COMMANDS.iter().position(|c| c.name == name)?;
        let address = self.proc_address(loader, index, device);
        (!address.is_null()).then(|| unsafe { std::mem::transmute_copy::<*const c_void, T>(&address) })
    }

    /// Runs one command; answers it unless it was sent without waiting
    fn run(&mut self, loader: &Loader, index: usize, r: &mut Reader, request: u32, out: &Out) -> io::Result<()> {
        let command = &COMMANDS[index];
        if !self.parked.is_empty() && !self.compiling() {
            self.run_parked();
        }
        let mut arena = std::mem::take(&mut self.arena);
        let call = match unsafe { wire::decode_call(command, r, &mut arena, self) } {
            Ok(call) => call,
            Err(error) => {
                eprintln!("vulkan: {} can't be carried out: {error:?}", command.name);
                if self.broken {
                    return Err(io::Error::other("a shader the program uses isn't here"));
                }
                return self.fail(request, VK_ERROR_FEATURE_NOT_PRESENT, out);
            }
        };
        if self.broken {
            return Err(io::Error::other("the program uses a pipeline that couldn't be compiled"));
        }
        let first = call.slots.first().copied().unwrap_or(0);
        let function = self.proc_address(loader, index, first);
        if function.is_null() {
            return self.fail(request, VK_ERROR_FEATURE_NOT_PRESENT, out);
        }
        self.current_owner = match command.level {
            Level::Device => self.owner.get(&first).copied().filter(|&d| d != 0).unwrap_or(first),
            _ => 0,
        };

        if request == 0 {
            if matches!(command.name, "vkCreateGraphicsPipelines" | "vkCreateComputePipelines") {
                let device = self.current_owner;
                self.compile_later(Ready { index, function: function as usize, _arena: arena, call }, device);
                return Ok(());
            }
            if COMPILES_READ.contains(&command.name) && self.compiling() {
                if self.parked.len() >= MOST_PARKED {
                    self.after_compiles();
                }
                self.parked.push(Parked(Ready { index, function: function as usize, _arena: arena, call }));
                return Ok(());
            }
        }
        if AFTER_COMPILES.contains(&command.name) {
            self.after_compiles();
        }
        match command.name {
            "vkDestroySemaphore" => self.watches.cancel(call.slots[1]),
            "vkDestroyDevice" => {
                self.watches.cancel_all();
                // Its encoders go before it does
                let on_it: Vec<u64> = self.coders.iter().filter(|(_, coder)| coder.video_device == first && first != 0).map(|(&key, _)| key).collect();
                for key in on_it {
                    if let Some(coder) = self.coders.remove(&key) {
                        coder.finish();
                    }
                }
                self.video.remove(&first);
            }
            _ => {}
        }
        // Compile threads share a cache, however the program promised to use it
        if command.name == "vkCreatePipelineCache" {
            let info = call.slots[1] as usize as *mut nectar_vk_sys::VkPipelineCacheCreateInfo;
            unsafe { (*info).flags &= !nectar_vk_sys::VK_PIPELINE_CACHE_CREATE_EXTERNALLY_SYNCHRONIZED_BIT };
        }

        if command.mode == Mode::Wait && request != 0 {
            // Off the ordered stream: the program may be about to submit what this waits for
            struct Moved(Arena, wire::Call);
            unsafe impl Send for Moved {}
            let moved = Moved(arena, call);
            let out = Arc::clone(out);
            let function = function as usize;
            let deferred = std::mem::take(&mut self.deferred);
            thread::spawn(move || {
                // The whole value, so it moves as one (and is Send)
                let moved = moved;
                let Moved(_arena, call) = moved;
                let result = unsafe { thunks::call(index, function as *const c_void, &call.slots) };
                let succeeded = command.ret != Ret::Result || (result as i64 as i32) >= 0;
                let mut w = Writer::new();
                let _ = unsafe { wire::encode_outputs(command, &call, succeeded, &mut w, &mut NoHandles) };
                let _ = answer(&out, request, result, deferred, w.as_slice());
            });
            return Ok(());
        }

        let mut offer = if command.name == "vkCreateDevice" { self.offer_video(loader, &call.slots) } else { None };
        let mut result = unsafe { thunks::call(index, function, &call.slots) };
        if let Some(offered) = offer.as_ref() {
            unsafe { offered.take_back(call.slots[1]) };
            // As the program asked, if the driver won't have the encoder beside it
            if (result as i64 as i32) < 0 {
                eprintln!("vulkan: the device can't be created with the video encoder ({}), so without", result as i64 as i32);
                offer = None;
                result = unsafe { thunks::call(index, function, &call.slots) };
            }
        }
        let succeeded = command.ret != Ret::Result || (result as i64 as i32) >= 0;
        let mut w = Writer::new();
        if let Err(error) = unsafe { wire::encode_outputs(command, &call, succeeded, &mut w, self) } {
            eprintln!("vulkan: {}: outputs can't be carried: {error:?}", command.name);
        }
        if succeeded {
            self.after(command.name, &call.slots);
            if let Some(offered) = offer {
                let device = unsafe { (call.slots[3] as usize as *const u64).read_unaligned() };
                self.video.insert(device, VideoDevice { physical: offered.physical, family: offered.family, queue: Arc::new(Mutex::new(())) });
            }
        }
        arena.reuse(call);
        self.arena = arena;
        if request != 0 {
            let deferred = std::mem::take(&mut self.deferred);
            answer(out, request, result, deferred, w.as_slice())
        } else {
            if !succeeded && self.deferred == 0 {
                self.deferred = result as i64 as i32;
                eprintln!("vulkan: {} failed with {}", command.name, self.deferred);
            }
            Ok(())
        }
    }

    /// Adds to a device being created what encoding its frames as video takes: the encoder's queue, which
    /// the program doesn't know of, and the extensions. None if the card has no encoder, or the program
    /// asked for that queue itself, or NECTAR_VK_VIDEO=0.
    fn offer_video(&self, loader: &Loader, slots: &[u64]) -> Option<VideoOffer> {
        if std::env::var("NECTAR_VK_VIDEO").is_ok_and(|value| value == "0") {
            return None;
        }
        let instance = *self.instances.last()?;
        let physical = slots[0];
        let info = slots[1] as usize as *mut nectar_vk_sys::VkDeviceCreateInfo;
        let get = loader.get_instance_proc_addr;
        let find = move |name: &CStr| unsafe { std::mem::transmute::<*const c_void, nectar_vk_sys::PFN_vkVoidFunction>(get(instance, name.as_ptr())) };
        unsafe {
            let Some(family) = video_encode::encode_family(&find, physical as usize as nectar_vk_sys::VkPhysicalDevice) else {
                eprintln!("vulkan: the card has no H.264 encoder, so frames won't go as video");
                return None;
            };
            let queues = ((*info).queueCreateInfoCount, (*info).pQueueCreateInfos);
            let asked: &[nectar_vk_sys::VkDeviceQueueCreateInfo] = if queues.1.is_null() { &[] } else { std::slice::from_raw_parts(queues.1, queues.0 as usize) };
            if asked.iter().any(|queue| queue.queueFamilyIndex == family) {
                eprintln!("vulkan: the program asked for the encoder's queue family ({family}) itself, so frames won't go as video");
                return None;
            }
            let priority = Box::new(1.0f32);
            let mut all_queues = asked.to_vec();
            all_queues.push(nectar_vk_sys::VkDeviceQueueCreateInfo {
                sType: nectar_vk_sys::VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                queueFamilyIndex: family,
                queueCount: 1,
                pQueuePriorities: &*priority,
            });
            let extensions = ((*info).enabledExtensionCount, (*info).ppEnabledExtensionNames);
            let enabled: &[*const c_char] = if extensions.1.is_null() { &[] } else { std::slice::from_raw_parts(extensions.1, extensions.0 as usize) };
            let mut all_extensions = enabled.to_vec();
            for extension in video_encode::DEVICE_EXTENSIONS {
                if !enabled.iter().any(|&name| CStr::from_ptr(name) == extension) {
                    all_extensions.push(extension.as_ptr());
                }
            }
            ((*info).queueCreateInfoCount, (*info).pQueueCreateInfos) = (all_queues.len() as u32, all_queues.as_ptr());
            ((*info).enabledExtensionCount, (*info).ppEnabledExtensionNames) = (all_extensions.len() as u32, all_extensions.as_ptr());
            Some(VideoOffer { physical, family, queues, extensions, _queues: all_queues, _extensions: all_extensions, _priority: priority })
        }
    }

    /// VIDEO_OPEN: an encoder for a swapchain's frames, on a thread of its own as the other coder is
    fn video_open(&mut self, loader: &Loader, r: &mut Reader, request: u32, out: &Out) -> io::Result<()> {
        let (key, device, width, height) = (r.u64()?, r.u64()?, r.u32()?, r.u32()?);
        let device = self.ids.get(&device).copied().unwrap_or(0);
        let instance = self.instances.last().copied().unwrap_or(0);
        let Some((physical, family, queue_lock)) = self.video.get(&device).map(|video| (video.physical, video.family, Arc::clone(&video.queue))) else {
            eprintln!("vulkan: no video for frames of {width}x{height}: their device wasn't created with the encoder ({} were)", self.video.len());
            return self.fail(request, VK_ERROR_FEATURE_NOT_PRESENT, out);
        };
        let get = loader.get_instance_proc_addr;
        let find_instance = move |name: &CStr| unsafe { std::mem::transmute::<*const c_void, nectar_vk_sys::PFN_vkVoidFunction>(get(instance, name.as_ptr())) };
        let get_device = unsafe { std::mem::transmute::<*const c_void, Option<GetProcAddr>>(get(instance, c"vkGetDeviceProcAddr".as_ptr())) };
        let Some(get_device) = get_device else {
            return self.fail(request, VK_ERROR_FEATURE_NOT_PRESENT, out);
        };
        let find_device = move |name: &CStr| unsafe { std::mem::transmute::<*const c_void, nectar_vk_sys::PFN_vkVoidFunction>(get_device(device, name.as_ptr())) };
        let get_queue = unsafe { std::mem::transmute::<nectar_vk_sys::PFN_vkVoidFunction, Option<nectar_vk_sys::PFN_vkGetDeviceQueue>>(find_device(c"vkGetDeviceQueue")) };
        let Some(get_queue) = get_queue else {
            return self.fail(request, VK_ERROR_FEATURE_NOT_PRESENT, out);
        };
        let real_device = device as usize as nectar_vk_sys::VkDevice;
        let mut queue = ptr::null_mut();
        unsafe { get_queue(real_device, family, 0, &mut queue) };
        let made = {
            let _alone = queue_lock.lock().unwrap_or_else(|p| p.into_inner());
            unsafe { video_encode::Encoder::new(&find_instance, &find_device, physical as usize as nectar_vk_sys::VkPhysicalDevice, real_device, family, queue, width, height) }
        };
        match made {
            Ok(encoder) => {
                eprintln!("vulkan: frames of {width}x{height} go as H.264 from the card's encoder");
                if let Some(before) = self.coders.insert(key, Coder::new(Arc::clone(out), Some(Video(encoder, queue_lock)), device)) {
                    before.finish();
                }
                let deferred = std::mem::take(&mut self.deferred);
                answer(out, request, 0, deferred, &family.to_le_bytes())
            }
            Err(error) => {
                eprintln!("vulkan: no video for frames of {width}x{height}: {error}");
                self.fail(request, VK_ERROR_FEATURE_NOT_PRESENT, out)
            }
        }
    }

    /// VIDEO_FRAME: a frame for a swapchain's encoder, which waits for its colours itself
    fn video_frame(&mut self, loader: &Loader, r: &mut Reader, request: u32, out: &Out) -> io::Result<()> {
        let arrived = Instant::now();
        let (key, buffer, device, fence) = (r.u64()?, r.u64()?, r.u64()?, r.u64()?);
        let (quality, key_frame) = (r.u32()?, r.u8()? != 0);
        let device = self.ids.get(&device).copied().unwrap_or(0);
        let fence = self.ids.get(&fence).copied().unwrap_or(0);
        let nv12 = self.ids.get(&buffer).copied().unwrap_or(0);
        let Some(wait) = self.device_function::<WaitForFences>(loader, device, "vkWaitForFences") else {
            return self.fail(request, VK_ERROR_FEATURE_NOT_PRESENT, out);
        };
        if nv12 == 0 || !self.coders.contains_key(&key) {
            return self.fail(request, VK_ERROR_FEATURE_NOT_PRESENT, out);
        }
        let deferred = std::mem::take(&mut self.deferred);
        let job = Job::Video(VideoJob { arrived, request, deferred, wait, device, fence, nv12, quality, key: key_frame });
        if self.coders.get(&key).is_none_or(|coder| coder.jobs.send(job).is_err()) {
            return self.fail(request, VK_ERROR_FEATURE_NOT_PRESENT, out);
        }
        Ok(())
    }

    /// VIDEO_CLOSE: the encoder goes once its frames are done
    fn video_close(&mut self, r: &mut Reader, request: u32, out: &Out) -> io::Result<()> {
        let key = r.u64()?;
        if let Some(coder) = self.coders.remove(&key) {
            coder.finish();
        }
        let deferred = std::mem::take(&mut self.deferred);
        answer(out, request, 0, deferred, &[])
    }

    fn fail(&mut self, request: u32, result: i32, out: &Out) -> io::Result<()> {
        if request == 0 {
            if self.deferred == 0 {
                self.deferred = result;
            }
            return Ok(());
        }
        let deferred = std::mem::take(&mut self.deferred);
        answer(out, request, result as i64 as u64, deferred, &[])
    }

    /// What this side needs to remember about a call that succeeded
    fn after(&mut self, name: &str, slots: &[u64]) {
        let output = |slot: u64| unsafe { (slot as usize as *const u64).read_unaligned() };
        match name {
            "vkCreateInstance" => self.instances.push(output(slots[2])),
            "vkCreateDevice" => {
                let device = output(slots[3]);
                self.owner.insert(device, device);
                self.devices.push(device);
            }
            "vkAllocateMemory" => {
                let info = slots[1] as usize as *const nectar_vk_sys::VkMemoryAllocateInfo;
                let size = unsafe { (*info).allocationSize };
                self.memory.insert(output(slots[3]), (slots[0], size));
            }
            "vkFreeMemory" => {
                self.memory.remove(&slots[1]);
            }
            "vkDestroyDevice" => self.devices.retain(|&d| d != slots[0]),
            "vkDestroyInstance" => self.instances.retain(|&i| i != slots[0]),
            _ => {}
        }
    }

    /// MAP: the memory, mapped whole, and its contents; or if `lazy`, its size, taking the borrower
    /// to hold zeros, so that only what is touched and isn't zero crosses
    fn map(&mut self, loader: &Loader, r: &mut Reader, request: u32, out: &Out) -> io::Result<()> {
        let id = r.u64()?;
        let lazy = r.u8()? != 0;
        let real = self.ids.get(&id).copied().unwrap_or(0);
        let Some(&(device, size)) = self.memory.get(&real) else {
            return self.fail(request, VK_ERROR_MEMORY_MAP_FAILED, out);
        };
        if !self.mappings.contains_key(&id) {
            let Some(map) = self.device_function::<MapMemory>(loader, device, "vkMapMemory") else {
                return self.fail(request, VK_ERROR_FEATURE_NOT_PRESENT, out);
            };
            let mut pointer = ptr::null_mut();
            let result = unsafe { map(device, real, 0, VK_WHOLE_SIZE, 0, &mut pointer) };
            if result != 0 {
                return self.fail(request, result, out);
            }
            let size = size as usize;
            let hashes = if lazy {
                let mut hashes = vec![xxh3_64(&[0u8; PAGE]); size.div_ceil(PAGE)];
                if !size.is_multiple_of(PAGE)
                    && let Some(last) = hashes.last_mut()
                {
                    *last = xxh3_64(&vec![0u8; size % PAGE]);
                }
                hashes
            } else {
                let bytes = unsafe { std::slice::from_raw_parts(pointer as *const u8, size) };
                bytes.chunks(PAGE).map(xxh3_64).collect()
            };
            self.mappings.insert(id, Mapping::new(pointer.cast(), size, hashes));
        }
        let mapping = &self.mappings[&id];
        let mut w = Writer::new();
        if lazy {
            w.u64(mapping.size as u64);
        } else {
            w.bytes(unsafe { std::slice::from_raw_parts(mapping.pointer, mapping.size) });
        }
        let deferred = std::mem::take(&mut self.deferred);
        answer(out, request, 0, deferred, w.as_slice())
    }

    /// WRITE: bytes the program wrote
    fn write(&mut self, r: &mut Reader) -> io::Result<()> {
        let (id, offset, bytes) = (r.u64()?, r.u64()? as usize, r.bytes()?);
        self.store(id, offset, bytes);
        Ok(())
    }

    /// Keeps bytes, or holds them if they can't be kept. False if they couldn't.
    fn keep_or_hold(&mut self, hash: &Hash, bytes: &[u8]) -> bool {
        let kept = self.cache.as_ref().is_some_and(|cache| match cache.keep(hash, bytes) {
            Ok(()) => true,
            Err(error) => {
                eprintln!("cannot keep {} bytes: {error}", bytes.len());
                false
            }
        });
        if !kept {
            self.held.insert(*hash, bytes.to_vec());
        }
        kept
    }

    /// What's kept or held under a hash
    fn find_kept(&self, hash: &Hash, size: u64) -> Option<Vec<u8>> {
        self.held
            .get(hash)
            .filter(|bytes| bytes.len() as u64 == size)
            .cloned()
            .or_else(|| self.cache.as_ref().and_then(|cache| cache.find(hash, size).ok()))
    }

    /// KEEP: as WRITE, and keeps the bytes
    fn keep(&mut self, r: &mut Reader, out: &Out) -> io::Result<()> {
        let (id, offset) = (r.u64()?, r.u64()? as usize);
        let hash: Hash = r.take(16)?.try_into().map_err(|_| io::ErrorKind::InvalidData)?;
        let bytes = r.bytes()?;
        if !self.keep_or_hold(&hash, bytes) && !self.told_to_stop {
            self.told_to_stop = true;
            let mut writer = out.lock().unwrap_or_else(|p| p.into_inner());
            protocol::write_message(&mut *writer, STOP_KEEPING, &[])?;
            writer.flush()?;
        }
        self.store(id, offset, bytes);
        Ok(())
    }

    /// KEPT: as WRITE, of bytes kept earlier
    fn kept(&mut self, r: &mut Reader) -> io::Result<()> {
        let (id, offset) = (r.u64()?, r.u64()? as usize);
        let hash: Hash = r.take(16)?.try_into().map_err(|_| io::ErrorKind::InvalidData)?;
        let size = r.u64()?;
        let started = Instant::now();
        // Nothing correct can follow
        let bytes = self.find_kept(&hash, size).ok_or_else(|| io::Error::other("a kept upload isn't here"))?;
        self.store(id, offset, &bytes);
        self.uploads_kept.0 += 1;
        self.uploads_kept.1 += size;
        self.uploads_kept.2 += started.elapsed();
        Ok(())
    }

    /// FILL_MIP: makes a mip out of the one below, in mapped memory, for the copy about to read it
    fn fill_mip(&mut self, r: &mut Reader) -> io::Result<()> {
        let (from_id, from_offset, from_across, from_down) = (r.u64()?, r.u64()? as usize, r.u32()? as usize, r.u32()? as usize);
        let (to_id, to_offset, across, down) = (r.u64()?, r.u64()? as usize, r.u32()? as usize, r.u32()? as usize);
        let (block, flatten) = (r.u32()? as usize, r.u32()?);
        let wanted = from_across.checked_mul(from_down).and_then(|blocks| blocks.checked_mul(block));
        let size = across.checked_mul(down).and_then(|blocks| blocks.checked_mul(block));
        let (Some(wanted), Some(size)) = (wanted, size) else { return Err(io::ErrorKind::InvalidData.into()) };
        let Some(mapping) = self.mappings.get(&from_id).filter(|mapping| from_offset.checked_add(wanted).is_some_and(|end| end <= mapping.size)) else {
            return Err(io::ErrorKind::InvalidData.into());
        };
        // Copied first: this memory is often uncached, and it's read many times over
        let mut from = unsafe { std::slice::from_raw_parts(mapping.pointer.add(from_offset), wanted) }.to_vec();
        if let Some((kind, _)) = Kind::of(flatten as _).filter(|(kind, _)| kind.block_bytes() == block) {
            from.chunks_exact_mut(block).for_each(|sample| nectar_vk_wire::bc::flatten_block(kind, sample));
        }
        let mut made = vec![0u8; size];
        nectar_vk_wire::expand_mip(&from, from_across, from_down, &mut made, across, down, block, 2);
        self.store(to_id, to_offset, &made);
        self.mips_made += size as u64;
        Ok(())
    }

    /// FILL_MIP_FROM: makes a mip out of a sample of its blocks, for the copy about to read it
    fn fill_mip_from(&mut self, r: &mut Reader) -> io::Result<()> {
        let (to_id, to_offset, across, down) = (r.u64()?, r.u64()? as usize, r.u32()? as usize, r.u32()? as usize);
        let (block, scale) = (r.u32()? as usize, r.u32()? as usize);
        let samples = r.bytes()?;
        let Some(size) = across.checked_mul(down).and_then(|blocks| blocks.checked_mul(block)) else { return Err(io::ErrorKind::InvalidData.into()) };
        if scale == 0 {
            return Err(io::ErrorKind::InvalidData.into());
        }
        let mut made = vec![0u8; size];
        nectar_vk_wire::expand_mip(samples, across.div_ceil(scale), down.div_ceil(scale), &mut made, across, down, block, scale);
        self.store(to_id, to_offset, &made);
        self.mips_made += size as u64;
        Ok(())
    }

    /// CODED_PART: more of a coded mip
    fn coded_part(&mut self, r: &mut Reader) -> io::Result<()> {
        let id = r.u64()?;
        let part = r.bytes()?;
        self.coded.entry(id).or_default().extend_from_slice(part);
        Ok(())
    }

    /// DECODE_MIP: decodes a coded mip into mapped memory, off the stream
    fn decode_mip(&mut self, r: &mut Reader, request: u32, out: &Out) -> io::Result<()> {
        let (id, format, across, down) = (r.u64()?, r.u32()? as i32, r.u32()? as usize, r.u32()? as usize);
        let coded = self.coded.remove(&id).unwrap_or_default();
        let Some((kind, srgb)) = Kind::of(format as _) else {
            return self.fail(request, VK_ERROR_FORMAT_NOT_SUPPORTED, out);
        };
        let size = across.checked_mul(down).and_then(|blocks| blocks.checked_mul(kind.block_bytes()));
        let Some(to) = size.and_then(|size| self.mappings.get(&id).filter(|mapping| mapping.size >= size)).map(|mapping| mapping.pointer as usize) else {
            return self.fail(request, VK_ERROR_MEMORY_MAP_FAILED, out);
        };
        self.mips_decoded.0 += coded.len() as u64;
        self.mips_decoded.1 += size.unwrap_or(0) as u64;
        if self.decoder.is_none() {
            let (jobs, receiver) = channel();
            let answers = Arc::clone(out);
            self.decoder = Some((jobs, thread::spawn(move || decode_mips(receiver, answers))));
        }
        let job = DecodeJob { coded, kind, srgb, across, down, to, request };
        if let Some((jobs, _)) = &self.decoder
            && jobs.send(job).is_ok()
        {
            return Ok(());
        }
        self.fail(request, VK_ERROR_FEATURE_NOT_PRESENT, out)
    }

    /// HOLD: waits here for fences the borrower answered for, so nothing after them runs early
    fn hold(&mut self, loader: &Loader, r: &mut Reader) -> io::Result<()> {
        let device = self.ids.get(&r.u64()?).copied().unwrap_or(0);
        let count = r.u32()?;
        let mut fences = Vec::with_capacity(count as usize);
        for _ in 0..count {
            fences.push(self.ids.get(&r.u64()?).copied().unwrap_or(0));
        }
        if let Some(wait) = self.device_function::<WaitForFences>(loader, device, "vkWaitForFences") {
            let result = unsafe { wait(device, count, fences.as_ptr(), 1, u64::MAX) };
            if result != 0 && self.deferred == 0 {
                self.deferred = result;
                eprintln!("vulkan: a held wait failed with {result}");
            }
        }
        Ok(())
    }

    /// FRAME: a swapchain's frame, coded on the swapchain's own thread so the program's next commands
    /// don't wait for it. The borrower copies no new frame over this one until the answer comes.
    fn frame(&mut self, loader: &Loader, r: &mut Reader, request: u32, out: &Out) -> io::Result<()> {
        let arrived = Instant::now();
        let (key, id, device, fence) = (r.u64()?, r.u64()?, r.u64()?, r.u64()?);
        let (width, height, quality) = (r.u32()?, r.u32()?, r.u32()?);
        let coded_on_gpu = r.u8()? != 0;
        let length = if coded_on_gpu { frame_shader::output_size(width, height) as usize } else { width as usize * height as usize * 4 };
        let Some(pixels) = self.mappings.get(&id).filter(|mapping| mapping.size >= length).map(|mapping| mapping.pointer as usize) else {
            return self.fail(request, VK_ERROR_MEMORY_MAP_FAILED, out);
        };
        let device = self.ids.get(&device).copied().unwrap_or(0);
        let fence = self.ids.get(&fence).copied().unwrap_or(0);
        let Some(wait) = self.device_function::<WaitForFences>(loader, device, "vkWaitForFences") else {
            return self.fail(request, VK_ERROR_FEATURE_NOT_PRESENT, out);
        };
        let deferred = std::mem::take(&mut self.deferred);
        let coder = self.coders.entry(key).or_insert_with(|| Coder::new(Arc::clone(out), None, 0));
        let job = FrameJob { arrived, request, deferred, wait, device, fence, pixels, width, height, quality, coded_on_gpu };
        if coder.jobs.send(Job::Frame(job)).is_err() {
            return self.fail(request, VK_ERROR_FEATURE_NOT_PRESENT, out);
        }
        Ok(())
    }

    /// WATCH: says when a timeline semaphore reaches a value. On a thread of its own, as a wait is:
    /// what signals one may depend on what the program does after hearing about another.
    fn watch(&mut self, loader: &Loader, r: &mut Reader, out: &Out) -> io::Result<()> {
        let (queue, id, value) = (r.u64()?, r.u64()?, r.u64()?);
        let mut pushing = Vec::new();
        for _ in 0..r.u32()? {
            let (memory, start, end) = (r.u64()?, r.u64()? as usize, r.u64()? as usize);
            if let Some(mapping) = self.mappings.get(&memory) {
                let pages = start / PAGE..end.min(mapping.size).div_ceil(PAGE);
                pushing.push(Pushing { memory, pages, hashes: Arc::clone(&mapping.hashes), pointer: mapping.pointer, size: mapping.size });
            }
        }
        let asked_to_push = !pushing.is_empty();
        let reading = Arc::clone(&self.reading);
        let queue = self.ids.get(&queue).copied().unwrap_or(0);
        let device = self.owner.get(&queue).copied().unwrap_or(0);
        let semaphore = self.ids.get(&id).copied().unwrap_or(0);
        let Some(wait) = self.device_function::<WaitSemaphores>(loader, device, "vkWaitSemaphores") else {
            return Ok(());
        };
        let watches = Arc::clone(&self.watches);
        *watches.lock().live.entry(semaphore).or_insert(0) += 1;
        let out = Arc::clone(out);
        thread::spawn(move || {
            let info = nectar_vk_sys::VkSemaphoreWaitInfo {
                sType: nectar_vk_sys::VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
                pNext: ptr::null(),
                flags: 0,
                semaphoreCount: 1,
                pSemaphores: &semaphore,
                pValues: &value,
            };
            while watches.wanted(semaphore) {
                let result = unsafe { wait(device, &info, WATCH_LOOK_UP) };
                if result == VK_TIMEOUT {
                    continue;
                }
                if result != 0 {
                    // Whoever's waiting will hear of the trouble from the calls after
                    eprintln!("vulkan: a watched semaphore's wait failed: {result}");
                }
                let mut w = Writer::new();
                w.u64(id).u64(value);
                // Read and sent in one, so that what's on the link is in the order it was read
                let mut generation = reading.lock().unwrap_or_else(|p| p.into_inner());
                let mut push = Writer::new();
                if asked_to_push && result == 0 {
                    *generation += 1;
                    push.u64(id).u64(value).u64(*generation);
                    let runs = pushed_runs(&pushing);
                    push.u32(runs.len() as u32);
                    for (memory, offset, bytes) in &runs {
                        push.u64(*memory).u64(*offset).bytes(bytes);
                    }
                }
                let mut writer = out.lock().unwrap_or_else(|p| p.into_inner());
                if asked_to_push && result == 0 {
                    let _ = protocol::write_message(&mut *writer, PUSHED, &[push.as_slice()]);
                }
                let _ = protocol::write_message(&mut *writer, REACHED, &[w.as_slice()]).and_then(|()| writer.flush());
                drop(writer);
                drop(generation);
                break;
            }
            watches.finished(semaphore);
        });
        Ok(())
    }

    /// DESCRIPTOR_COPIES: many copies, in one driver call
    fn descriptor_copies(&mut self, loader: &Loader, r: &mut Reader) -> io::Result<()> {
        if !self.parked.is_empty() && !self.compiling() {
            self.run_parked();
        }
        let (device, count) = descriptor_copies::decode_head(r)?;
        let device = self.incoming(false, device);
        let function = match self.update_sets {
            Some((known, function)) if known == device => function,
            _ => {
                let Some(function) = self.device_function::<UpdateDescriptorSets>(loader, device, "vkUpdateDescriptorSets") else {
                    return Ok(());
                };
                self.update_sets = Some((device, function));
                function
            }
        };
        let mut copies = std::mem::take(&mut self.copying);
        copies.clear();
        let (mut before, mut source, mut target) = (descriptor_copies::DescriptorCopy::default(), 0, 0);
        for index in 0..count {
            let copy = descriptor_copies::decode_next(r, &before)?;
            // Sets seldom change from one to the next, and looking them up is most of the work
            if index == 0 || copy.source_set != before.source_set {
                source = self.incoming(false, copy.source_set);
            }
            if index == 0 || copy.target_set != before.target_set {
                target = self.incoming(false, copy.target_set);
            }
            copies.push(nectar_vk_sys::VkCopyDescriptorSet {
                sType: nectar_vk_sys::VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET,
                pNext: ptr::null(),
                srcSet: source,
                srcBinding: copy.source_binding,
                srcArrayElement: copy.source_element,
                dstSet: target,
                dstBinding: copy.target_binding,
                dstArrayElement: copy.target_element,
                descriptorCount: copy.count,
            });
            before = copy;
        }
        unsafe { function(device, 0, ptr::null(), copies.len() as u32, copies.as_ptr()) };
        self.copying = copies;
        Ok(())
    }

    /// protocol::CACHE_INDEX: the content hashes kept
    fn cache_index(&mut self, request: u32, out: &Out) -> io::Result<()> {
        let Some(cache) = &self.cache else {
            return self.fail(request, VK_ERROR_FEATURE_NOT_PRESENT, out);
        };
        let index = cache.index();
        let mut w = Writer::new();
        w.u64(index.len() as u64);
        for hash in &index {
            w.raw(hash);
        }
        let deferred = std::mem::take(&mut self.deferred);
        answer(out, request, 0, deferred, w.as_slice())
    }

    /// Puts bytes in a mapping
    fn store(&mut self, id: u64, offset: usize, bytes: &[u8]) {
        if let Some(mapping) = self.mappings.get_mut(&id) {
            let end = (offset + bytes.len()).min(mapping.size);
            let target = unsafe { std::slice::from_raw_parts_mut(mapping.pointer, mapping.size) };
            let incoming = &bytes[..end - offset];
            target[offset..end].copy_from_slice(incoming);
            // Hashed from what came in where whole pages did: this memory is often uncached, and
            // reading it back would take far longer than writing it
            for page in offset / PAGE..end.div_ceil(PAGE) {
                let start = page * PAGE;
                let page_end = (start + PAGE).min(mapping.size);
                let hash = if start >= offset && page_end <= end {
                    xxh3_64(&incoming[start - offset..page_end - offset])
                } else {
                    xxh3_64(&target[start..page_end])
                };
                mapping.hashes.pages[page].store(hash, Ordering::Relaxed);
            }
        }
    }

    /// REFRESH: the pages the GPU changed since the borrower last had them, in runs
    fn refresh(&mut self, request: u32, out: &Out) -> io::Result<()> {
        let reading = Arc::clone(&self.reading);
        let mut generation = reading.lock().unwrap_or_else(|p| p.into_inner());
        *generation += 1;
        let mut runs = Vec::new();
        for (&id, mapping) in &self.mappings {
            mapping.changed(id, 0..mapping.hashes.pages.len(), &mut runs);
        }
        self.send_runs(runs, *generation, request, out)
    }

    /// FETCH: as REFRESH, for ranges of one mapping
    fn fetch(&mut self, r: &mut Reader, request: u32, out: &Out) -> io::Result<()> {
        let (id, count) = (r.u64()?, r.u32()?);
        let reading = Arc::clone(&self.reading);
        let mut generation = reading.lock().unwrap_or_else(|p| p.into_inner());
        *generation += 1;
        let mut runs = Vec::new();
        for _ in 0..count {
            let (offset, length) = (r.u64()? as usize, r.u64()? as usize);
            if let Some(mapping) = self.mappings.get(&id) {
                let end = offset.saturating_add(length).min(mapping.size).div_ceil(PAGE);
                mapping.changed(id, offset / PAGE..end, &mut runs);
            }
        }
        self.send_runs(runs, *generation, request, out)
    }

    fn send_runs(&mut self, runs: Vec<(u64, usize, usize, *mut u8)>, generation: u64, request: u32, out: &Out) -> io::Result<()> {
        let mut w = Writer::new();
        w.u64(generation).u32(runs.len() as u32);
        for (id, start, end, pointer) in runs {
            let mut bytes = vec![0u8; end - start];
            read_mapped(unsafe { pointer.add(start) }, &mut bytes);
            w.u64(id).u64(start as u64).bytes(&bytes);
        }
        let deferred = std::mem::take(&mut self.deferred);
        answer(out, request, 0, deferred, w.as_slice())
    }

    fn unmap(&mut self, loader: &Loader, r: &mut Reader) -> io::Result<()> {
        let id = r.u64()?;
        if let Some(coder) = self.coders.remove(&id) {
            coder.finish();
        }
        if self.mappings.remove(&id).is_some() {
            let real = self.ids.get(&id).copied().unwrap_or(0);
            if let Some(&(device, _)) = self.memory.get(&real)
                && let Some(unmap) = self.device_function::<UnmapMemory>(loader, device, "vkUnmapMemory")
            {
                unsafe { unmap(device, real) };
            }
        }
        Ok(())
    }

    /// Frees what a borrower left behind: its memory (which destroying a device doesn't free), its
    /// devices, then its instances
    fn release(&mut self, loader: &Loader) {
        // It writes to mappings, so it has to finish before they go
        if let Some((jobs, thread)) = self.decoder.take() {
            drop(jobs);
            let _ = thread.join();
        }
        for (_, coder) in self.coders.drain() {
            coder.finish();
        }
        self.watches.cancel_all();
        self.after_compiles();
        {
            let mut state = self.compiler.lock();
            state.closing = true;
            self.compiler.changed.notify_all();
            if state.compiled > 0 {
                eprintln!(
                    "vulkan: {} pipeline calls compiled off the stream in {:.0} ms between them; the stream waited {:.0} ms for them",
                    state.compiled,
                    state.took.as_secs_f64() * 1000.0,
                    self.compile_waits.as_secs_f64() * 1000.0
                );
            }
        }
        for thread in self.compile_threads.drain(..) {
            let _ = thread.join();
        }
        if self.shaders_kept.0 > 0 {
            eprintln!(
                "vulkan: {} shaders came by hash, saving {:.1} MB on the link",
                self.shaders_kept.0,
                self.shaders_kept.1 as f64 / 1e6
            );
        }
        if self.mips_decoded.0 > 0 {
            eprintln!(
                "vulkan: {:.1} MB of mips came coded in {:.1} MB until the real ones came",
                self.mips_decoded.1 as f64 / 1e6,
                self.mips_decoded.0 as f64 / 1e6
            );
        }
        if self.mips_made > 0 {
            eprintln!("vulkan: {:.1} MB of mips were made from the ones below until the real ones came", self.mips_made as f64 / 1e6);
        }
        if self.uploads_kept.0 > 0 {
            eprintln!(
                "vulkan: {} uploads came by hash, saving {:.1} MB on the link, in {:.1} ms getting them",
                self.uploads_kept.0,
                self.uploads_kept.1 as f64 / 1e6,
                self.uploads_kept.2.as_secs_f64() * 1e3
            );
        }
        self.mappings.clear();
        for device in std::mem::take(&mut self.devices) {
            if let Some(wait) = self.device_function::<DeviceWaitIdle>(loader, device, "vkDeviceWaitIdle") {
                unsafe { wait(device) };
            }
            if let Some(free) = self.device_function::<FreeMemory>(loader, device, "vkFreeMemory") {
                let left: Vec<u64> = self.memory.iter().filter(|(_, (d, _))| *d == device).map(|(&m, _)| m).collect();
                for memory in left {
                    self.memory.remove(&memory);
                    unsafe { free(device, memory, ptr::null()) };
                }
            }
            if let Some(destroy) = self.device_function::<Destroy>(loader, device, "vkDestroyDevice") {
                unsafe { destroy(device, ptr::null()) };
            }
        }
        for instance in std::mem::take(&mut self.instances) {
            let destroy: Option<Destroy> = unsafe {
                std::mem::transmute::<*const c_void, Option<Destroy>>((loader.get_instance_proc_addr)(
                    instance,
                    c"vkDestroyInstance".as_ptr(),
                ))
            };
            if let Some(destroy) = destroy {
                unsafe { destroy(instance, ptr::null()) };
            }
        }
    }
}

/// Serves one borrower's Vulkan connection, after the hello
pub fn serve(reader: BufReader<TcpStream>, writer: BufWriter<TcpStream>, cache: Option<Arc<Cache>>, peer: &str) -> io::Result<()> {
    let Some(loader) = loader() else {
        return Err(io::Error::other("no Vulkan loader on this computer"));
    };
    eprintln!("{peer} uses Vulkan");
    let out: Out = Arc::new(Mutex::new(writer));
    let mut session = Session { cache, ..Session::default() };
    let mut reader = PackedReader::new(reader);
    let mut body = Vec::new();
    let outcome = loop {
        // A large one doesn't stay
        if body.capacity() > 1 << 20 {
            body = Vec::new();
        }
        let head = match protocol::read_message_into(&mut reader, &mut body) {
            Ok(head) => head,
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::UnexpectedEof | io::ErrorKind::ConnectionReset | io::ErrorKind::ConnectionAborted
                ) =>
            {
                break Ok(());
            }
            Err(error) => break Err(error),
        };
        let mut r = Reader::new(&body);
        let request = match r.u32() {
            Ok(request) => request,
            Err(error) => break Err(error),
        };
        let handled = match head {
            PACKED => {
                reader.start();
                Ok(())
            }
            MAP => session.map(loader, &mut r, request, &out),
            WRITE => session.write(&mut r),
            KEEP => session.keep(&mut r, &out),
            KEPT => session.kept(&mut r),
            FILL_MIP => session.fill_mip(&mut r),
            FILL_MIP_FROM => session.fill_mip_from(&mut r),
            CODED_PART => session.coded_part(&mut r),
            DECODE_MIP => session.decode_mip(&mut r, request, &out),
            ECHO => answer(&out, request, 0, 0, &[]),
            FRAME => session.frame(loader, &mut r, request, &out),
            VIDEO_OPEN => session.video_open(loader, &mut r, request, &out),
            VIDEO_FRAME => session.video_frame(loader, &mut r, request, &out),
            VIDEO_CLOSE => session.video_close(&mut r, request, &out),
            WATCH => session.watch(loader, &mut r, &out),
            DESCRIPTOR_COPIES => session.descriptor_copies(loader, &mut r),
            HOLD => session.hold(loader, &mut r),
            GPU_USAGE => {
                let percent = nectar_detect::usage::gpu_percent().unwrap_or(u32::MAX);
                answer(&out, request, 0, 0, &percent.to_le_bytes())
            }
            protocol::CACHE_INDEX => session.cache_index(request, &out),
            REFRESH => session.refresh(request, &out),
            FETCH => session.fetch(&mut r, request, &out),
            UNMAP => session.unmap(loader, &mut r),
            id => match command_index(id) {
                Some(index) => session.run(loader, index, &mut r, request, &out),
                None => session.fail(request, VK_ERROR_FEATURE_NOT_PRESENT, &out),
            },
        };
        if let Err(error) = handled {
            break Err(error);
        }
        // Answers to calls sent without waiting go out once nothing more is queued
        if reader.drained() && reader.inner().buffer().is_empty() {
            let _ = out.lock().unwrap_or_else(|p| p.into_inner()).flush();
        }
    };
    if reader.came > 0 {
        eprintln!(
            "vulkan: {:.1} MB came up compressed to {:.1} MB",
            reader.became as f64 / 1e6,
            reader.came as f64 / 1e6
        );
    }
    session.release(loader);
    outcome
}
