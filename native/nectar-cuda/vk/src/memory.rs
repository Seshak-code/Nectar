//! Mapped memory. The program gets a shadow copy on the borrower; the lender maps the real
//! memory. Before work is submitted, pages the program changed go to the lender; pages the GPU
//! changed come back once a wait says the work is done.
//!
//! On Windows the shadow is allocated with write watching, so the system says which pages the
//! program wrote. After a wait nothing is fetched: the pages are marked stale (no access), and
//! the program's first touch of one fetches the stale pages around it, of which the lender sends
//! only those the GPU changed. A wait costs nothing, and a program that maps a large staging
//! buffer but uses a little of it moves only that little. Elsewhere, pages are compared by hash
//! both ways, and a wait fetches every page the GPU changed.
//!
//! Stale pages can't be read by the system on the program's behalf (WriteFile from mapped
//! memory, say) without a touch first: NECTAR_VK_EAGER=1 fetches after every wait instead.

use std::collections::HashMap;
use std::sync::Mutex;

use nectar_cuda_protocol::Writer;
use nectar_vk_wire::messages::{MAP, PAGE, REFRESH, UNMAP, WRITE};
use xxhash_rust::xxh3::xxh3_64;

use crate::connection::Connection;

struct Mapping {
    shadow: platform::Shadow,
    size: usize,
    /// Each page's hash as the lender last had it (unused while pages are watched)
    hashes: Vec<u64>,
    /// Pages marked stale, to fetch on first touch
    stale: Vec<bool>,
    /// Runs of pages made readable since pages were last marked stale
    fresh: Vec<(usize, usize)>,
    /// Pages to fetch on the next touch: more while the program reads straight through
    window: usize,
    /// The page after the last fetch, to tell reading straight through
    next: usize,
}

impl Mapping {
    fn bytes(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.shadow.pointer, self.size) }
    }

    fn pages(&self) -> usize {
        self.size.div_ceil(PAGE)
    }

    /// The pages the program wrote since the last reset, in address order, and resets them if
    /// `reset`. Only readable pages can have been written, so only they are asked about.
    fn written(&mut self, reset: bool) -> Vec<usize> {
        let mut written = Vec::new();
        for &(first, end) in &self.fresh {
            self.shadow.written(first..end, reset, &mut written);
        }
        written.sort_unstable();
        written
    }
}

/// Mappings by memory id
static MAPPINGS: Mutex<Option<HashMap<u64, Mapping>>> = Mutex::new(None);

/// Pages first fetched on a touch, and the most fetched at once
const FIRST_WINDOW: usize = (1 << 20) / PAGE;
const LARGEST_WINDOW: usize = (64 << 20) / PAGE;

fn hashes(bytes: &[u8]) -> Vec<u64> {
    bytes.chunks(PAGE).map(xxh3_64).collect()
}

/// Whether pages are watched and fetched on touch, rather than compared by hash
fn watched() -> bool {
    static WATCHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *WATCHED.get_or_init(|| platform::WATCHED && std::env::var_os("NECTAR_VK_EAGER").is_none())
}

/// The shadow of a memory object, mapping it on the lender the first time
pub fn map(connection: &Connection, memory: u64) -> Result<*mut u8, i32> {
    let mut mappings = MAPPINGS.lock().unwrap_or_else(|p| p.into_inner());
    let mappings = mappings.get_or_insert_with(HashMap::new);
    if let Some(mapping) = mappings.get_mut(&memory) {
        return Ok(mapping.shadow.pointer);
    }
    let mut w = Writer::new();
    w.u64(memory);
    let answer = connection.ask(MAP, w.as_slice()).map_err(|_| crate::DEVICE_LOST)?;
    if answer.result as i64 as i32 != 0 {
        return Err(answer.result as i64 as i32);
    }
    let contents = answer.outputs().bytes().map_err(|_| crate::DEVICE_LOST)?;
    let size = contents.len();
    let pages = size.div_ceil(PAGE);
    let mut mapping = Mapping {
        shadow: platform::Shadow::new(size).ok_or(VK_ERROR_OUT_OF_HOST_MEMORY)?,
        size,
        hashes: if watched() { Vec::new() } else { hashes(contents) },
        stale: vec![false; pages],
        fresh: vec![(0, pages)],
        window: FIRST_WINDOW,
        next: usize::MAX,
    };
    mapping.bytes().copy_from_slice(contents);
    if watched() {
        mapping.shadow.reset(0, size);
        platform::catch_touches();
    }
    let pointer = mapping.shadow.pointer;
    mappings.insert(memory, mapping);
    Ok(pointer)
}

const VK_ERROR_OUT_OF_HOST_MEMORY: i32 = -1;

/// Sends the pages the program changed, in runs; called before work is submitted and when the
/// program flushes or unmaps
pub fn flush(connection: &Connection) {
    let mut mappings = MAPPINGS.lock().unwrap_or_else(|p| p.into_inner());
    let Some(mappings) = mappings.as_mut() else { return };
    for (&memory, mapping) in mappings.iter_mut() {
        let written = if watched() { mapping.shadow.written(true) } else { compare(mapping) };
        let size = mapping.size;
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(mapping.shadow.pointer, size) };
        for (first, end) in runs(&written) {
            let (start, end) = (first * PAGE, (end * PAGE).min(size));
            let mut w = Writer::new();
            w.u64(memory).u64(start as u64).bytes(&bytes[start..end]);
            let _ = connection.send(WRITE, w.as_slice());
        }
    }
}

/// The pages whose hash changed since the lender last had them, taking the new hashes
fn compare(mapping: &mut Mapping) -> Vec<usize> {
    let size = mapping.size;
    let bytes: &[u8] = unsafe { std::slice::from_raw_parts(mapping.shadow.pointer, size) };
    let mut changed = Vec::new();
    for (page, old) in mapping.hashes.iter_mut().enumerate() {
        let start = page * PAGE;
        let hash = xxh3_64(&bytes[start..(start + PAGE).min(size)]);
        if hash != *old {
            *old = hash;
            changed.push(page);
        }
    }
    changed
}

/// Runs of consecutive pages, as (first, end)
fn runs(pages: &[usize]) -> Vec<(usize, usize)> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for &page in pages {
        match runs.last_mut() {
            Some((_, end)) if *end == page => *end += 1,
            _ => runs.push((page, page + 1)),
        }
    }
    runs
}

/// Takes in what the GPU wrote to mapped memory; called after a wait for the GPU
pub fn refresh(connection: &Connection) {
    let mut mappings = MAPPINGS.lock().unwrap_or_else(|p| p.into_inner());
    let Some(mappings) = mappings.as_mut().filter(|m| !m.is_empty()) else { return };
    if watched() {
        for mapping in mappings.values_mut() {
            mark_stale(mapping);
        }
        return;
    }
    let Ok(answer) = connection.ask(REFRESH, &[]) else { return };
    let mut r = answer.outputs();
    let Ok(count) = r.u32() else { return };
    for _ in 0..count {
        let (Ok(memory), Ok(offset), Ok(bytes)) = (r.u64(), r.u64(), r.bytes()) else { return };
        if let Some(mapping) = mappings.get_mut(&memory) {
            let start = offset as usize;
            let end = (start + bytes.len()).min(mapping.size);
            mapping.bytes()[start..end].copy_from_slice(&bytes[..end - start]);
            let size = mapping.size;
            for page in start / PAGE..end.div_ceil(PAGE) {
                let from = page * PAGE;
                mapping.hashes[page] = xxh3_64(&mapping.bytes()[from..(from + PAGE).min(size)]);
            }
        }
    }
}

/// Marks the pages made readable since last time stale, except those the program wrote and
/// hasn't sent yet: its writes are newer than anything the GPU could have done to them
fn mark_stale(mapping: &mut Mapping) {
    // In address order
    let written = mapping.shadow.written(false);
    let kept = |page: usize| written.binary_search(&page).is_ok();
    for (first, end) in std::mem::take(&mut mapping.fresh) {
        let mut page = first;
        while page < end {
            let start = page;
            while page < end && !kept(page) {
                page += 1;
            }
            if page > start {
                mapping.shadow.protect(start * PAGE, (page - start) * PAGE, false);
                mapping.stale[start..page].fill(true);
            }
            while page < end && kept(page) {
                page += 1;
            }
        }
    }
    mapping.fresh = runs(&written);
    mapping.window = FIRST_WINDOW;
    mapping.next = usize::MAX;
}

/// A touch at `address`: if it is a stale page, fetches the stale pages around it and makes
/// them readable. False when the address isn't in a stale page of a mapping.
#[cfg_attr(not(windows), allow(dead_code))]
fn touch(address: usize) -> bool {
    let mut guard = MAPPINGS.lock().unwrap_or_else(|p| p.into_inner());
    let Some(mappings) = guard.as_mut() else { return false };
    let Some((&memory, mapping)) = mappings.iter_mut().find(|(_, m)| {
        let base = m.shadow.pointer as usize;
        address >= base && address < base + m.size
    }) else {
        return false;
    };
    let page = (address - mapping.shadow.pointer as usize) / PAGE;
    if !mapping.stale[page] {
        // Another thread fetched it first
        return true;
    }
    // Reading straight on from the last fetch fetches twice as much; otherwise the block of
    // FIRST_WINDOW pages around the touch
    let first = if page == mapping.next {
        mapping.window = (mapping.window * 2).min(LARGEST_WINDOW);
        page
    } else {
        mapping.window = FIRST_WINDOW;
        page - page % FIRST_WINDOW
    };
    let end = (first + mapping.window).min(mapping.pages());
    let wanted = runs(&(first..end).filter(|&p| mapping.stale[p]).collect::<Vec<_>>());
    let Some(connection) = crate::connection::get() else { return false };
    let mut w = Writer::new();
    w.u64(memory).u32(wanted.len() as u32);
    for &(start, end) in &wanted {
        w.u64((start * PAGE) as u64).u64(((end - start) * PAGE) as u64);
    }
    let Ok(answer) = connection.ask(nectar_vk_wire::messages::FETCH, w.as_slice()) else { return false };
    for &(start, end) in &wanted {
        mapping.shadow.protect(start * PAGE, (end - start) * PAGE, true);
        mapping.stale[start..end].fill(false);
        mapping.fresh.push((start, end));
    }
    let mut r = answer.outputs();
    let count = r.u32().unwrap_or(0);
    for _ in 0..count {
        let (Ok(_), Ok(offset), Ok(bytes)) = (r.u64(), r.u64(), r.bytes()) else { break };
        let start = offset as usize;
        let end = (start + bytes.len()).min(mapping.size);
        mapping.bytes()[start..end].copy_from_slice(&bytes[..end - start]);
        mapping.shadow.reset(start, end - start);
    }
    mapping.next = end;
    if crate::logging() {
        let asked: usize = wanted.iter().map(|(s, e)| e - s).sum();
        crate::log(&format!("fetch: {asked} pages asked, {} bytes came", answer.body.len() - answer.at));
    }
    true
}

/// Ends the mapping of freed memory
pub fn forget(connection: &Connection, memory: u64) {
    let removed = MAPPINGS.lock().unwrap_or_else(|p| p.into_inner()).as_mut().and_then(|m| m.remove(&memory));
    if removed.is_some() {
        let mut w = Writer::new();
        w.u64(memory);
        let _ = connection.send(UNMAP, w.as_slice());
    }
}

#[cfg(windows)]
mod platform {
    use std::ffi::c_void;
    use std::sync::Once;

    use windows_sys::Win32::System::Diagnostics::Debug::{AddVectoredExceptionHandler, EXCEPTION_POINTERS};
    use windows_sys::Win32::System::Memory::{
        GetWriteWatch, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, MEM_WRITE_WATCH, PAGE_NOACCESS, PAGE_READWRITE,
        ResetWriteWatch, VirtualAlloc, VirtualFree, VirtualProtect, WRITE_WATCH_FLAG_RESET,
    };

    use super::PAGE;

    pub const WATCHED: bool = true;

    const EXCEPTION_ACCESS_VIOLATION: i32 = 0xC000_0005_u32 as i32;
    const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;
    const EXCEPTION_CONTINUE_SEARCH: i32 = 0;

    /// Page-aligned (programs may rely on minMemoryMapAlignment), with writes watched
    pub struct Shadow {
        pub pointer: *mut u8,
        size: usize,
    }

    // The shadow is only reached through MAPPINGS' lock, or by the program as its mapped memory
    unsafe impl Send for Shadow {}

    impl Shadow {
        pub fn new(size: usize) -> Option<Shadow> {
            let size = size.max(1).div_ceil(PAGE) * PAGE;
            let flags = MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH;
            let pointer = unsafe { VirtualAlloc(std::ptr::null(), size, flags, PAGE_READWRITE) };
            (!pointer.is_null()).then_some(Shadow { pointer: pointer.cast(), size })
        }

        /// Adds the pages in `pages` written since the last reset, and resets them if `reset`
        pub fn written(&mut self, pages: Range<usize>, reset: bool, written: &mut Vec<usize>) {
            let length = (pages.end * PAGE).min(self.size) - pages.start * PAGE;
            let mut addresses = vec![std::ptr::null_mut::<c_void>(); pages.len()];
            let (mut count, mut granularity) = (addresses.len(), 0u32);
            let flags = if reset { WRITE_WATCH_FLAG_RESET } else { 0 };
            let start = unsafe { self.pointer.add(pages.start * PAGE) };
            let failed = unsafe {
                GetWriteWatch(flags, start.cast(), length, addresses.as_mut_ptr(), &mut count, &mut granularity)
            };
            if failed == 0 {
                let base = self.pointer as usize;
                written.extend(addresses[..count].iter().map(|&a| (a as usize - base) / PAGE));
            }
        }

        /// Forgets writes to bytes `start..start + length`: they came from the lender
        pub fn reset(&mut self, start: usize, length: usize) {
            let first = start / PAGE * PAGE;
            let end = (start + length).div_ceil(PAGE) * PAGE;
            unsafe { ResetWriteWatch(self.pointer.add(first).cast(), end - first) };
        }

        pub fn protect(&mut self, start: usize, length: usize, readable: bool) {
            let protection = if readable { PAGE_READWRITE } else { PAGE_NOACCESS };
            let mut old = 0;
            unsafe { VirtualProtect(self.pointer.add(start).cast(), length, protection, &mut old) };
        }
    }

    impl Drop for Shadow {
        fn drop(&mut self) {
            unsafe { VirtualFree(self.pointer.cast(), 0, MEM_RELEASE) };
        }
    }

    /// Catches the program's touches of stale pages, first in line
    pub fn catch_touches() {
        static ONCE: Once = Once::new();
        ONCE.call_once(|| unsafe {
            AddVectoredExceptionHandler(1, Some(on_exception));
        });
    }

    unsafe extern "system" fn on_exception(pointers: *mut EXCEPTION_POINTERS) -> i32 {
        let record = unsafe { &*(*pointers).ExceptionRecord };
        if record.ExceptionCode != EXCEPTION_ACCESS_VIOLATION || record.NumberParameters < 2 {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        if super::touch(record.ExceptionInformation[1]) { EXCEPTION_CONTINUE_EXECUTION } else { EXCEPTION_CONTINUE_SEARCH }
    }
}

#[cfg(not(windows))]
mod platform {
    use std::alloc::{Layout, alloc_zeroed, dealloc};

    use super::PAGE;

    pub const WATCHED: bool = false;

    /// Page-aligned: programs may rely on minMemoryMapAlignment
    pub struct Shadow {
        pub pointer: *mut u8,
        layout: Layout,
    }

    // The shadow is only reached through MAPPINGS' lock, or by the program as its mapped memory
    unsafe impl Send for Shadow {}

    impl Shadow {
        pub fn new(size: usize) -> Option<Shadow> {
            let layout = Layout::from_size_align(size.max(1), PAGE).ok()?;
            let pointer = unsafe { alloc_zeroed(layout) };
            (!pointer.is_null()).then_some(Shadow { pointer, layout })
        }

        pub fn written(&mut self, _reset: bool) -> Vec<usize> {
            Vec::new()
        }

        pub fn reset(&mut self, _start: usize, _length: usize) {}

        pub fn protect(&mut self, _start: usize, _length: usize, _readable: bool) {}
    }

    impl Drop for Shadow {
        fn drop(&mut self) {
            unsafe { dealloc(self.pointer, self.layout) };
        }
    }

    pub fn catch_touches() {}
}
