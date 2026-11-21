//! The Vulkan personality: runs a borrower's Vulkan calls on this computer's Vulkan driver,
//! through the system's Vulkan loader.
//!
//! Calls run in the order they arrive, on this connection's thread. Waits for the GPU
//! (vkWaitForFences and the like) run on a thread of their own, so a wait can't hold up the
//! call that would end it, as when one of the program's threads waits on a fence another is
//! about to submit. Their answers name the request, so they can come back out of order.
//!
//! Handles: the borrower chooses ids for what it creates; what this side hands out (physical
//! devices, queues) gets ids with the top bit set. Memory the borrower maps is mapped here as
//! a whole, and kept in step page by page (vkwire's messages).

use std::collections::HashMap;
use std::ffi::{CString, c_char, c_void};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::net::TcpStream;
use std::sync::{Arc, Mutex, OnceLock};
use std::{ptr, thread};

use libloading::Library;
use nectar_cuda_protocol::{self as protocol, Reader, Writer};
use nectar_vk_wire::messages::{MAP, PAGE, REFRESH, UNMAP, WRITE};
use nectar_vk_wire::tables::{Level, Mode, Ret};
use nectar_vk_wire::wire::{self, Arena, Handles};
use nectar_vk_wire::{COMMANDS, thunks};
use xxhash_rust::xxh3::xxh3_64;

type GetProcAddr = unsafe extern "system" fn(u64, *const c_char) -> *const c_void;
type MapMemory = unsafe extern "system" fn(u64, u64, u64, u64, u32, *mut *mut c_void) -> i32;
type UnmapMemory = unsafe extern "system" fn(u64, u64);
type DeviceWaitIdle = unsafe extern "system" fn(u64) -> i32;
type Destroy = unsafe extern "system" fn(u64, *const c_void);
type FreeMemory = unsafe extern "system" fn(u64, u64, *const c_void);

const VK_ERROR_DEVICE_LOST: i32 = -4;
const VK_ERROR_FEATURE_NOT_PRESENT: i32 = -8;
const VK_ERROR_MEMORY_MAP_FAILED: i32 = -5;
const VK_WHOLE_SIZE: u64 = u64::MAX;
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

struct Mapping {
    pointer: *mut u8,
    size: usize,
    /// Each page's hash as the borrower last had it
    hashes: Vec<u64>,
}

impl Mapping {
    /// Adds the runs of pages in `pages` that changed since the borrower last had them, and
    /// takes them as the borrower's
    fn changed(&mut self, id: u64, pages: std::ops::Range<usize>, runs: &mut Vec<(u64, usize, usize, *mut u8)>) {
        let bytes = unsafe { std::slice::from_raw_parts(self.pointer, self.size) };
        let mut run: Option<usize> = None;
        let end = pages.end.min(self.hashes.len());
        for page in pages.start..=end {
            let changed = page < end && {
                let start = page * PAGE;
                let hash = xxh3_64(&bytes[start..(start + PAGE).min(self.size)]);
                let changed = hash != self.hashes[page];
                self.hashes[page] = hash;
                changed
            };
            match (changed, run) {
                (true, None) => run = Some(page),
                (false, Some(first)) => {
                    runs.push((id, first * PAGE, (page * PAGE).min(self.size), self.pointer));
                    run = None;
                }
                _ => {}
            }
        }
    }
}

#[derive(Default)]
struct Session {
    ids: HashMap<u64, u64>,
    reals: HashMap<u64, u64>,
    next_id: u64,
    /// The device each dispatchable handle belongs to
    owner: HashMap<u64, u64>,
    /// The device a new handle belongs to, while a call's outputs are recorded
    current_owner: u64,
    instances: Vec<u64>,
    devices: Vec<u64>,
    procs: HashMap<(u64, usize), usize>,
    /// Device and size of each memory object, by its real handle
    memory: HashMap<u64, (u64, u64)>,
    mappings: HashMap<u64, Mapping>,
    /// The first error of a call sent without waiting, for the next answer
    deferred: i32,
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
        if id == 0 { 0 } else { self.ids.get(&id).copied().unwrap_or(0) }
    }

    fn create(&mut self, _: bool) -> (u64, u64) {
        unreachable!("the lender never chooses a borrower's handle")
    }

    fn bind(&mut self, id: u64, real: u64) {
        self.ids.insert(id, real);
        self.reals.insert(real, id);
        self.owner.insert(real, self.current_owner);
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
        let mut arena = Arena::default();
        let call = match unsafe { wire::decode_call(command, r, &mut arena, self) } {
            Ok(call) => call,
            Err(error) => {
                eprintln!("vulkan: {} can't be carried out: {error:?}", command.name);
                return self.fail(request, VK_ERROR_FEATURE_NOT_PRESENT, out);
            }
        };
        let first = call.slots.first().copied().unwrap_or(0);
        let function = self.proc_address(loader, index, first);
        if function.is_null() {
            return self.fail(request, VK_ERROR_FEATURE_NOT_PRESENT, out);
        }
        self.current_owner = match command.level {
            Level::Device => self.owner.get(&first).copied().filter(|&d| d != 0).unwrap_or(first),
            _ => 0,
        };

        if command.mode == Mode::Wait && request != 0 {
            // Off the ordered stream: the program may be about to submit what this waits for
            struct Moved(Arena, wire::Call);
            unsafe impl Send for Moved {}
            let moved = Moved(arena, call);
            let out = Arc::clone(out);
            let function = function as usize;
            let deferred = std::mem::take(&mut self.deferred);
            thread::spawn(move || {
                let Moved(_arena, call) = moved;
                let result = unsafe { thunks::call(index, function as *const c_void, &call.slots) };
                let succeeded = command.ret != Ret::Result || (result as i64 as i32) >= 0;
                let mut w = Writer::new();
                let _ = unsafe { wire::encode_outputs(command, &call, succeeded, &mut w, &mut NoHandles) };
                let _ = answer(&out, request, result, deferred, w.as_slice());
            });
            return Ok(());
        }

        let result = unsafe { thunks::call(index, function, &call.slots) };
        let succeeded = command.ret != Ret::Result || (result as i64 as i32) >= 0;
        let mut w = Writer::new();
        if let Err(error) = unsafe { wire::encode_outputs(command, &call, succeeded, &mut w, self) } {
            eprintln!("vulkan: {}: outputs can't be carried: {error:?}", command.name);
        }
        if succeeded {
            self.after(command.name, &call.slots);
        }
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
                let info = slots[1] as usize as *const nectar_vk_wire_sys::VkMemoryAllocateInfo;
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

    /// MAP: the memory, mapped whole, and its contents
    fn map(&mut self, loader: &Loader, r: &mut Reader, request: u32, out: &Out) -> io::Result<()> {
        let id = r.u64()?;
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
            let bytes = unsafe { std::slice::from_raw_parts(pointer as *const u8, size) };
            let hashes = bytes.chunks(PAGE).map(xxh3_64).collect();
            self.mappings.insert(id, Mapping { pointer: pointer.cast(), size, hashes });
        }
        let mapping = &self.mappings[&id];
        let bytes = unsafe { std::slice::from_raw_parts(mapping.pointer, mapping.size) };
        let mut w = Writer::new();
        w.bytes(bytes);
        let deferred = std::mem::take(&mut self.deferred);
        answer(out, request, 0, deferred, w.as_slice())
    }

    /// WRITE: bytes the program wrote
    fn write(&mut self, r: &mut Reader) -> io::Result<()> {
        let (id, offset, bytes) = (r.u64()?, r.u64()? as usize, r.bytes()?);
        if let Some(mapping) = self.mappings.get_mut(&id) {
            let end = (offset + bytes.len()).min(mapping.size);
            let target = unsafe { std::slice::from_raw_parts_mut(mapping.pointer, mapping.size) };
            target[offset..end].copy_from_slice(&bytes[..end - offset]);
            for page in offset / PAGE..end.div_ceil(PAGE) {
                let start = page * PAGE;
                mapping.hashes[page] = xxh3_64(&target[start..(start + PAGE).min(mapping.size)]);
            }
        }
        Ok(())
    }

    /// REFRESH: the pages the GPU changed since the borrower last had them, in runs
    fn refresh(&mut self, request: u32, out: &Out) -> io::Result<()> {
        let started = std::time::Instant::now();
        let mut w = Writer::new();
        let mut runs = Vec::new();
        for (&id, mapping) in &mut self.mappings {
            let bytes = unsafe { std::slice::from_raw_parts(mapping.pointer, mapping.size) };
            let pages = mapping.hashes.len();
            let mut run: Option<usize> = None;
            for page in 0..=pages {
                let changed = page < pages && {
                    let start = page * PAGE;
                    let hash = xxh3_64(&bytes[start..(start + PAGE).min(mapping.size)]);
                    let changed = hash != mapping.hashes[page];
                    mapping.hashes[page] = hash;
                    changed
                };
                match (changed, run) {
                    (true, None) => run = Some(page),
                    (false, Some(first)) => {
                        runs.push((id, first * PAGE, (page * PAGE).min(mapping.size), mapping.pointer));
                        run = None;
                    }
                    _ => {}
                }
            }
        }
        if started.elapsed().as_millis() >= 100 {
            let mapped: usize = self.mappings.values().map(|m| m.size).sum();
            eprintln!("slow refresh: {} ms over {} MiB mapped", started.elapsed().as_millis(), mapped >> 20);
        }
        w.u32(runs.len() as u32);
        for (id, start, end, pointer) in runs {
            let bytes = unsafe { std::slice::from_raw_parts(pointer.add(start), end - start) };
            w.u64(id).u64(start as u64).bytes(bytes);
        }
        let deferred = std::mem::take(&mut self.deferred);
        answer(out, request, 0, deferred, w.as_slice())
    }

    fn unmap(&mut self, loader: &Loader, r: &mut Reader) -> io::Result<()> {
        let id = r.u64()?;
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
        self.mappings.clear();
        for device in std::mem::take(&mut self.devices) {
            if let Some(wait) = self.device_function::<DeviceWaitIdle>(loader, device, "vkDeviceWaitIdle") {
                unsafe { wait(device) };
            }
            if let Some(free) = self.device_function::<FreeMemory>(loader, device, "vkFreeMemory") {
                let left: Vec<u64> = self.memory.iter().filter(|(_, &(d, _))| d == device).map(|(&m, _)| m).collect();
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
pub fn serve(mut reader: BufReader<TcpStream>, writer: BufWriter<TcpStream>, peer: &str) -> io::Result<()> {
    let Some(loader) = loader() else {
        return Err(io::Error::other("no Vulkan loader on this computer"));
    };
    eprintln!("{peer} uses Vulkan");
    let out: Out = Arc::new(Mutex::new(writer));
    let mut session = Session::default();
    let outcome = loop {
        let (head, body) = match protocol::read_message(&mut reader) {
            Ok(message) => message,
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
            MAP => session.map(loader, &mut r, request, &out),
            WRITE => session.write(&mut r),
            REFRESH => session.refresh(request, &out),
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
        if reader.buffer().is_empty() {
            let _ = out.lock().unwrap_or_else(|p| p.into_inner()).flush();
        }
    };
    session.release(loader);
    let _ = VK_ERROR_DEVICE_LOST;
    outcome
}
