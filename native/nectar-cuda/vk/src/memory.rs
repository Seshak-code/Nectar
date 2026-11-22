//! Mapped memory. The program gets a shadow copy on the borrower; the lender maps the real
//! memory. Before work is submitted, pages the program changed go to the lender; pages the GPU
//! changed come back when the program touches them after a wait.
//!
//! After a wait nothing is fetched: the shadow's pages are marked stale (no access), and the
//! program's first touch of one fetches the stale pages around it, of which the lender sends only
//! those the GPU changed. A wait costs nothing, and a program that maps a large staging buffer but
//! uses a little of it moves only that little.
//!
//! So only pages touched since the last wait can have been written. On Windows the shadow is
//! allocated with write watching, and the system names them. Elsewhere they're compared by hash:
//! write protection would be exact too, but the system would refuse to read a file into such a
//! page (read() fails with EFAULT).
//!
//! Touching a stale page the GPU hasn't been given a way to write only makes it readable: nothing can
//! have changed, so a game writing its constants into a mapped buffer each frame doesn't fetch them back
//! first (Writers says how that's known, down to the byte ranges, as DXVK makes one huge buffer of a
//! whole allocation, so one copy into it says nothing about the rest). Those pages still go stale, so
//! that each submit looks only at the pages in use, however large the mapping. Where the system watches
//! writes, memory nothing could have changed goes stale only every so often, and a touch makes a large
//! block of it readable, as a game's upload rings walk through it all the time and each fault costs.
//!
//! A copy into mapped memory writes once. Until the work it was submitted with is known to have finished
//! it doesn't count: the program can't be expecting it yet, and a game reads what's beside it. When that's
//! finished and the pages it wrote have been fetched, they can't change again, so touching them needs no
//! more round trips. A storage descriptor lets a shader write at any time, so what it names counts, if a
//! shader seen does write through a binding of that number. DXVK uses them for read-only buffers too.
//!
//! Large writes go by content hash: the lender keeps them, and the next load of the same textures or
//! model sends only the hashes, wherever in staging memory they sit. Hashes need no answer, so this
//! adds no wait. Where the copy that reads them is already known, they're hashed exactly as that copy
//! reads them (uploads.rs). A program often writes first and records the copy later on another thread,
//! as DXVK does, so that a submit in between must send bytes with nothing yet saying what they are.
//! Those are cut where their content says (next_cut), so the same bytes are cut the same way wherever
//! they sit, whatever was written beside them.
//!
//! What a game rewrites all the time would only fill the lender's disk, so it isn't kept (Keeper).
//!
//! DXVK writes a texture and only later records its copy, and a submit in between would send it with
//! nothing saying what it is. But DXVK turns each Direct3D call into commands after the call, so
//! commands can only read what was written before they were recorded. So for DXVK a submit sends what
//! was written before its command buffers finished recording, and leaves the rest. Other programs may
//! record first and write afterwards, as a shader's constants often are, so theirs all goes.
//!
//! A long write can still be under way as a command buffer finishes, and so look as if it's needed. A
//! submit about to send a very long run that nothing explains waits a moment for the copy that reads it
//! to be recorded (wait_for_copies). That can only delay it, and memory where no copy ever comes stops
//! being waited for.
//!
//! Stale pages can't be used by the system on the program's behalf (WriteFile from mapped memory,
//! read() into it) without a touch first: NECTAR_VK_EAGER=1 fetches after every wait instead, and
//! compares every page by hash both ways.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

const VK_WHOLE_SIZE: u64 = u64::MAX;

use nectar_cuda_protocol::{CACHE_INDEX, Hash, Reader, Writer, content_hash};
use nectar_vk_wire::messages::{FILL_MIP, FILL_MIP_FROM, KEEP, KEPT, MAP, PAGE, REFRESH, UNMAP, WRITE};
use nectar_vk_wire::wire::CodeWay;
use xxhash_rust::xxh3::xxh3_64;

use crate::connection::Connection;
use crate::later;
use crate::uploads::{Region, Submit};

struct Mapping {
    shadow: platform::Shadow,
    size: usize,
    /// Each page's hash as the lender last had it (unused while the system watches writes)
    hashes: Vec<u64>,
    /// Pages marked stale, to fetch on first touch
    stale: Vec<bool>,
    /// Runs of pages made readable since pages were last marked stale
    fresh: Vec<(usize, usize)>,
    /// Pages to fetch on the next touch: more while the program reads straight through
    window: usize,
    /// The page after the last fetch, to tell reading straight through
    next: usize,
    /// Read only on the lender (map_readback): never stale, and nothing to send
    readback: bool,
    /// Runs of pages a thread is asking the lender about now
    fetching: Vec<(usize, usize)>,
    /// How many times pages have been marked stale, or could have been
    marked: u64,
    /// When they last were
    marked_at: Instant,
    /// Pages written and not yet sent, with the count of command buffers finished when they were
    /// first seen written. Only used while writes can wait (holding_writes).
    unsent: BTreeMap<usize, u64>,
}

impl Mapping {
    fn bytes(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.shadow.pointer, self.size) }
    }

    fn pages(&self) -> usize {
        self.size.div_ceil(PAGE)
    }

    /// The pages the program wrote since they were last in step, in address order, taking them
    /// as in step if `reset`. Only readable pages can have been written, so only they are looked at.
    fn written(&mut self, reset: bool) -> Vec<usize> {
        // Memory that doesn't go stale keeps gathering these
        if self.fresh.len() > 64 {
            self.fresh.sort_unstable();
            let mut joined: Vec<(usize, usize)> = Vec::with_capacity(self.fresh.len());
            for &(first, end) in &self.fresh {
                match joined.last_mut() {
                    Some(last) if first <= last.1 => last.1 = last.1.max(end),
                    _ => joined.push((first, end)),
                }
            }
            self.fresh = joined;
        }
        let mut written = Vec::new();
        for (first, end) in self.fresh.clone() {
            if platform::WRITE_WATCH {
                self.shadow.written(first..end, reset, &mut written);
            } else {
                self.compare(first..end, reset, &mut written);
            }
        }
        written.sort_unstable();
        written
    }

    /// Adds the pages in `pages` whose hash changed, taking the new hashes if `reset`
    fn compare(&mut self, pages: std::ops::Range<usize>, reset: bool, changed: &mut Vec<usize>) {
        let size = self.size;
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(self.shadow.pointer, size) };
        for page in pages {
            let start = page * PAGE;
            let hash = xxh3_64(&bytes[start..(start + PAGE).min(size)]);
            if hash != self.hashes[page] {
                if reset {
                    self.hashes[page] = hash;
                }
                changed.push(page);
            }
        }
    }

    /// Takes bytes `start..start + length` as in step with the lender
    fn settle(&mut self, start: usize, length: usize) {
        if platform::WRITE_WATCH {
            self.shadow.reset(start, length);
        } else {
            let size = self.size;
            let bytes: &[u8] = unsafe { std::slice::from_raw_parts(self.shadow.pointer, size) };
            for page in start / PAGE..(start + length).div_ceil(PAGE) {
                let from = page * PAGE;
                self.hashes[page] = xxh3_64(&bytes[from..(from + PAGE).min(size)]);
            }
        }
    }
}

/// Which memory the GPU can write. A buffer's usage says little, since allocators such as DXVK's
/// make one huge buffer with every usage. What counts is what's seen being used to write, and where:
/// copies, storage descriptors and transform feedback, all of which go through the driver. A shader can
/// also write through a buffer's address with nothing to see, so if any shader could do that, every
/// buffer with an address counts.
#[derive(Default)]
struct Writers {
    /// Each buffer and its memory once bound, and whether it has an address
    buffers: HashMap<u64, Buffer>,
    /// Each texel view's buffer and what part of it
    views: HashMap<u64, (u64, u64, u64)>,
    /// Memory that all counts as written, as it holds an image or a buffer used in ways not watched
    memory: HashSet<u64>,
    /// In other memory, the byte ranges something can write at any time: end by start, not touching
    ranges: HashMap<u64, BTreeMap<u64, u64>>,
    /// The same, that only a shader can write through a storage descriptor, by binding number
    storage: HashMap<u32, HashMap<u64, BTreeMap<u64, u64>>>,
    /// The binding numbers shaders have been seen writing through
    bindings_written: HashSet<u32>,
    /// A shader has been seen writing somewhere that couldn't be followed to a binding, or that can't be
    /// looked inside: every binding counts
    every_binding: bool,
    /// The ranges commands write once, not yet fetched since they did
    once: Once,
    /// Memory of buffers with addresses
    addressed: HashSet<u64>,
    /// A shader that may write through addresses has been seen
    addresses_write: bool,
    /// A command that writes by address has been seen, so anything could have been written
    everything: bool,
    /// Memory nothing on the GPU writes: see `memory_allocated`
    never: HashSet<u64>,
    /// Memory only commands that write once write: see `memory_allocated`
    copies_only: HashSet<u64>,
}

#[derive(Clone, Copy, Default)]
struct Buffer {
    memory: u64,
    /// Where it is in that memory
    offset: u64,
    size: u64,
    /// Its usage allows something this file doesn't watch
    unwatched: bool,
    addressed: bool,
}

static WRITERS: Mutex<Option<Writers>> = Mutex::new(None);

fn writers<R>(f: impl FnOnce(&mut Writers) -> R) -> R {
    f(WRITERS.lock().unwrap_or_else(|p| p.into_inner()).get_or_insert_with(Writers::default))
}

/// Usage bits that write only in ways seen here, or not at all: transfer, uniform and storage
/// (texel) buffers, index, vertex, indirect, and transform feedback
const WATCHED_USAGE: u32 = 0x1 | 0x2 | 0x4 | 0x8 | 0x10 | 0x20 | 0x40 | 0x80 | 0x100 | 0x800 | 0x1000;
const USAGE_ADDRESS: u32 = 0x20000;

/// A buffer was created. Anything chained on its create info could change its usage, so counts as
/// written.
pub fn buffer_created(buffer: u64, size: u64, usage: u32, chained: bool) {
    let info = Buffer {
        memory: 0,
        offset: 0,
        size,
        unwatched: chained || usage & !(WATCHED_USAGE | USAGE_ADDRESS) != 0,
        addressed: usage & USAGE_ADDRESS != 0,
    };
    writers(|writers| writers.buffers.insert(buffer, info));
}

pub fn buffer_destroyed(buffer: u64) {
    writers(|writers| writers.buffers.remove(&buffer));
}

pub fn buffer_bound(buffer: u64, memory: u64, offset: u64) {
    writers(|writers| {
        // A buffer not seen being created counts as written
        let entry = writers.buffers.entry(buffer).or_insert(Buffer { unwatched: true, ..Buffer::default() });
        entry.memory = memory;
        entry.offset = offset;
        let info = *entry;
        if info.unwatched {
            writers.memory.insert(memory);
        }
        if info.addressed {
            writers.addressed.insert(memory);
        }
    });
}

/// An image's memory: what it holds is the GPU's to write
pub fn image_bound(memory: u64) {
    writers(|writers| {
        writers.memory.insert(memory);
        writers.copies_only.remove(&memory);
    });
}

/// What commands write once: each command buffer's, then where they've gone, in the order of the work
/// that does it
#[derive(Default)]
struct Once {
    /// What each command buffer writes: memory, start, end
    recording: HashMap<u64, Vec<(u64, u64, u64)>>,
    /// By memory, end, the last submit that writes it and the first, by start. Ranges don't overlap.
    ranges: HashMap<u64, BTreeMap<u64, (u64, u64, u64)>>,
    /// Submits waiting on a semaphore: when one has finished, so has work on other queues, of which it
    /// isn't known here how much
    waiters: HashSet<u64>,
    /// The last submit, and those not known to have finished with their queues
    submits: u64,
    running: BTreeMap<u64, u64>,
    /// Which submit each fence and timeline value says has finished
    fences: HashMap<u64, u64>,
    values: HashMap<u64, Vec<(u64, u64)>>,
}

impl Once {
    /// Every submit up to here has finished
    fn finished_through(&self) -> u64 {
        self.running.keys().next().map_or(self.submits, |first| first - 1)
    }

    fn add(&mut self, memory: u64, start: u64, end: u64, submit: u64) {
        let ranges = self.ranges.entry(memory).or_default();
        let (mut start, mut end, mut submit, mut first) = (start, end, submit, submit);
        if let Some((&before, &(before_end, before_submit, before_first))) = ranges.range(..=start).next_back()
            && before_end > start
        {
            start = before;
            end = end.max(before_end);
            submit = submit.max(before_submit);
            first = first.min(before_first);
        }
        let overlapping: Vec<u64> = ranges.range(start..end).map(|(&at, _)| at).collect();
        for at in overlapping {
            if let Some((other_end, other_submit, other_first)) = ranges.remove(&at) {
                end = end.max(other_end);
                submit = submit.max(other_submit);
                first = first.min(other_first);
            }
        }
        ranges.insert(start, (end, submit, first));
    }

    /// Whether work known to have finished wrote these bytes, and they haven't been fetched since
    fn touches(&self, memory: u64, start: u64, end: u64) -> bool {
        self.ranges.get(&memory).is_some_and(|ranges| {
            ranges.range(..end).rev().take_while(|(_, range)| range.0 > start).any(|(_, range)| !self.running.contains_key(&range.2))
        })
    }

    /// Whether any work known to have finished wrote this memory, and it hasn't been fetched since
    fn any_finished(&self, memory: u64) -> bool {
        self.ranges.get(&memory).is_some_and(|ranges| ranges.values().any(|range| !self.running.contains_key(&range.2)))
    }

    /// These bytes have been fetched, having been asked for when `through` was as far as work was known
    /// to have finished: what's written by that work won't change them again
    fn fetched(&mut self, memory: u64, start: u64, end: u64, through: u64) {
        let Some(ranges) = self.ranges.get_mut(&memory) else { return };
        let mut first = start;
        if let Some((&before, &(before_end, ..))) = ranges.range(..=start).next_back()
            && before_end > start
        {
            first = before;
        }
        let overlapping: Vec<(u64, u64, u64, u64)> = ranges.range(first..end).map(|(&at, &(to, submit, earliest))| (at, to, submit, earliest)).collect();
        for (at, to, submit, earliest) in overlapping {
            if submit > through {
                continue;
            }
            ranges.remove(&at);
            if at < start {
                ranges.insert(at, (start, submit, earliest));
            }
            if to > end {
                ranges.insert(end, (to, submit, earliest));
            }
        }
    }

    /// Everything on a queue up to this submit has finished
    fn finish(&mut self, submit: u64) {
        let Some(&queue) = self.running.get(&submit) else { return };
        if self.waiters.remove(&submit) {
            self.running.retain(|&at, _| at > submit);
            self.waiters.retain(|&at| at > submit);
            return;
        }
        let done: Vec<u64> = self.running.range(..=submit).filter(|(_, on)| **on == queue).map(|(&at, _)| at).collect();
        for at in done {
            self.running.remove(&at);
        }
    }
}

/// A command buffer began recording, or ran another's
pub fn recording_began(commands: u64) {
    writers(|writers| writers.once.recording.insert(commands, Vec::new()));
}

pub fn recording_freed(commands: u64) {
    writers(|writers| writers.once.recording.remove(&commands));
}

pub fn recording_includes(commands: u64, others: &[u64]) {
    writers(|writers| {
        let inner: Vec<_> = others.iter().filter_map(|other| writers.once.recording.get(other)).flatten().copied().collect();
        writers.once.recording.entry(commands).or_default().extend(inner);
    });
}

/// A command in `commands` writes this part of `buffer`, once, when it runs
pub fn writes_once(commands: u64, buffer: u64, offset: u64, size: u64) {
    writers(|writers| {
        if let Some(info) = writers.buffers.get(&buffer).copied() {
            let size = if size == VK_WHOLE_SIZE { info.size.saturating_sub(offset) } else { size };
            let start = info.offset + offset;
            writers.once.recording.entry(commands).or_default().push((info.memory, start, start + size));
        }
    });
}

/// Work went to a queue: what these command buffers write will be written, and these will say it's done.
/// Returns what the lender could send ahead of that news.
pub fn work_submitted(queue: u64, commands: &[u64], fence: u64, signals: &[(u64, u64)], waits: bool) -> Pushing {
    let pushing = writers(|writers| {
        let once = &mut writers.once;
        once.submits += 1;
        let submit = once.submits;
        once.running.insert(submit, queue);
        if waits {
            once.waiters.insert(submit);
        }
        let writes: Vec<(u64, u64, u64)> = commands.iter().filter_map(|commands| once.recording.get(commands)).flatten().copied().collect();
        let wanted = writes.clone();
        for (memory, start, end) in writes {
            once.add(memory, start, end, submit);
        }
        if fence != 0 {
            once.fences.insert(fence, submit);
        }
        for &(semaphore, value) in signals {
            once.values.entry(semaphore).or_default().push((value, submit));
        }
        Pushing { submit, ranges: Vec::new() }.with(wanted)
    });
    pushing.only_mapped()
}

/// What a submit writes in mapped memory, for the lender to send before it says the work is done, so that
/// reading it needs no trip. Read-backs wait behind everything going up, so a trip for each is what most
/// limits a game that makes them each frame.
///
/// NECTAR_VK_PUSH=0 leaves them to be asked for.
#[derive(Default)]
pub struct Pushing {
    submit: u64,
    /// Memory, start and end
    pub ranges: Vec<(u64, u64, u64)>,
}

/// The most that's pushed for one submit. More is likely not all read, and it can wait to be asked for.
const PUSH_MOST: u64 = 4 << 20;

impl Pushing {
    fn with(mut self, ranges: Vec<(u64, u64, u64)>) -> Self {
        self.ranges = ranges;
        self
    }

    fn only_mapped(mut self) -> Self {
        static PUSHING: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let on = *PUSHING.get_or_init(|| std::env::var("NECTAR_VK_PUSH").map_or(true, |value| value != "0"));
        let total: u64 = self.ranges.iter().map(|(_, start, end)| end - start).sum();
        if !on || !on_touch() || total > PUSH_MOST {
            self.ranges.clear();
            return self;
        }
        let mappings = MAPPINGS.lock().unwrap_or_else(|p| p.into_inner());
        self.ranges.retain(|(memory, ..)| mappings.as_ref().and_then(|mappings| mappings.get(memory)).is_some_and(|mapping| !mapping.readback));
        self
    }
}

/// Pages that came ahead of the news, until they're stale and can go in. By memory and page.
#[derive(Default)]
struct PushedPages {
    pages: HashMap<u64, BTreeMap<usize, PushedPage>>,
    /// What's on its way, by the semaphore and value it comes with
    awaited: HashMap<(u64, u64), Pushing>,
}

struct PushedPage {
    /// Where this came in the lender's reading of mapped memory
    generation: u64,
    bytes: Vec<u8>,
}

impl PushedPages {
    fn arrived(&mut self, memory: u64, generation: u64, offset: usize, bytes: &[u8]) {
        let pages = self.pages.entry(memory).or_default();
        for (index, page) in bytes.chunks(PAGE).enumerate() {
            pages.insert(offset / PAGE + index, PushedPage { generation, bytes: page.to_vec() });
        }
    }

    /// Takes those of a memory that `stale` says can go in
    fn take(&mut self, memory: u64, stale: impl Fn(usize) -> bool) -> Vec<(usize, Vec<u8>)> {
        let Some(pages) = self.pages.get_mut(&memory) else { return Vec::new() };
        let ready: Vec<usize> = pages.keys().copied().filter(|&page| stale(page)).collect();
        ready.into_iter().filter_map(|page| pages.remove(&page).map(|pushed| (page, pushed.bytes))).collect()
    }

    /// Takes what came from before the lender's reading number `generation`, of these pages. That reading
    /// only brings what's changed since, so these still have to go in, under what it brought.
    fn older(&mut self, memory: u64, first: usize, end: usize, generation: u64) -> Vec<(usize, Vec<u8>)> {
        let Some(pages) = self.pages.get_mut(&memory) else { return Vec::new() };
        let ready: Vec<usize> = pages.range(first..end).filter(|(_, pushed)| pushed.generation < generation).map(|(&page, _)| page).collect();
        ready.into_iter().filter_map(|page| pages.remove(&page).map(|pushed| (page, pushed.bytes))).collect()
    }

    /// The program wrote this page: what the GPU wrote before that is out of date
    fn overwritten(&mut self, memory: u64, written: &[usize]) {
        if let Some(pages) = self.pages.get_mut(&memory) {
            for page in written {
                pages.remove(page);
            }
        }
    }

    fn forget(&mut self, memory: u64) {
        self.pages.remove(&memory);
    }

    fn has(&self, memory: u64) -> bool {
        self.pages.get(&memory).is_some_and(|pages| !pages.is_empty())
    }
}

static PUSHED_PAGES: std::sync::LazyLock<Mutex<PushedPages>> = std::sync::LazyLock::new(Mutex::default);

/// These will come with this value's news
pub fn push_awaited(semaphore: u64, value: u64, pushing: &Pushing) {
    let copy = Pushing { submit: pushing.submit, ranges: pushing.ranges.clone() };
    PUSHED_PAGES.lock().unwrap_or_else(|p| p.into_inner()).awaited.insert((semaphore, value), copy);
}

/// PUSHED from the lender, on the receiving thread, which mustn't wait for a touch's fetch. What came goes
/// in when it next would have been fetched; from now on those ranges need no asking.
pub fn pushed(body: &[u8]) {
    let mut r = Reader::new(body);
    let (Ok(semaphore), Ok(value), Ok(generation), Ok(count)) = (r.u64(), r.u64(), r.u64(), r.u32()) else { return };
    let mut pushed = PUSHED_PAGES.lock().unwrap_or_else(|p| p.into_inner());
    for _ in 0..count {
        let (Ok(memory), Ok(offset), Ok(bytes)) = (r.u64(), r.u64(), r.bytes()) else { break };
        pushed.arrived(memory, generation, offset as usize, bytes);
    }
    let awaited = pushed.awaited.remove(&(semaphore, value));
    drop(pushed);
    if let Some(awaited) = awaited {
        writers(|writers| {
            for (memory, start, end) in awaited.ranges {
                writers.once.fetched(memory, start, end, awaited.submit);
            }
        });
    }
}

/// A fence the program waited for is signalled
pub fn fence_signalled(fence: u64) {
    writers(|writers| {
        if let Some(submit) = writers.once.fences.remove(&fence) {
            writers.once.finish(submit);
        }
    });
}

/// A timeline semaphore reached a value
pub fn value_reached(semaphore: u64, value: u64) {
    writers(|writers| {
        let once = &mut writers.once;
        let Some(values) = once.values.get_mut(&semaphore) else { return };
        let done: Vec<u64> = values.iter().filter(|(at, _)| *at <= value).map(|(_, submit)| *submit).collect();
        values.retain(|(at, _)| *at > value);
        for submit in done {
            once.finish(submit);
        }
    });
}

/// A queue, or with 0 every queue, has finished all it was given
pub fn queue_idle(queue: u64) {
    writers(|writers| {
        let once = &mut writers.once;
        once.running.retain(|_, on| queue != 0 && *on != queue);
        let running = &once.running;
        once.waiters.retain(|at| running.contains_key(at));
    });
}

/// Adds a range to those that don't touch, joining any it does
fn add_range(ranges: &mut BTreeMap<u64, u64>, mut start: u64, mut end: u64) {
    if let Some((&before, &before_end)) = ranges.range(..=start).next_back()
        && before_end >= start
    {
        start = before;
        end = end.max(before_end);
    }
    let touching: Vec<u64> = ranges.range(start..=end).map(|(&at, _)| at).collect();
    for at in touching {
        end = end.max(ranges.remove(&at).unwrap_or(end));
    }
    ranges.insert(start, end);
}

fn range_touches(ranges: &BTreeMap<u64, u64>, start: u64, end: u64) -> bool {
    ranges.range(..end).next_back().is_some_and(|(_, &last)| last > start)
}

/// A command or descriptor that lets the GPU write all of `buffer`
pub fn written(buffer: u64) {
    written_range(buffer, 0, VK_WHOLE_SIZE);
}

/// A command or descriptor that lets the GPU write this part of `buffer`
pub fn written_range(buffer: u64, offset: u64, size: u64) {
    writers(|writers| {
        if let Some(info) = writers.buffers.get(&buffer).copied() {
            let size = if size == VK_WHOLE_SIZE { info.size.saturating_sub(offset) } else { size };
            let start = info.offset + offset;
            add_range(writers.ranges.entry(info.memory).or_default(), start, start + size);
        }
    });
}

pub fn view_created(view: u64, buffer: u64, offset: u64, range: u64) {
    writers(|writers| writers.views.insert(view, (buffer, offset, range)));
}

/// A storage descriptor at this binding names this part of `buffer`
pub fn storage_named(binding: u32, buffer: u64, offset: u64, size: u64) {
    writers(|writers| {
        if let Some(info) = writers.buffers.get(&buffer).copied() {
            let size = if size == VK_WHOLE_SIZE { info.size.saturating_sub(offset) } else { size };
            let start = info.offset + offset;
            add_range(writers.storage.entry(binding).or_default().entry(info.memory).or_default(), start, start + size);
        }
    });
}

/// A storage texel view at this binding that a shader may write through names its part of a buffer
pub fn view_written(binding: u32, view: u64) {
    if let Some((buffer, offset, range)) = writers(|writers| writers.views.get(&view).copied()) {
        storage_named(binding, buffer, offset, range);
    }
}

/// What a shader writes through
pub fn storage_writes(writes: &StorageWrites) {
    writers(|writers| {
        writers.bindings_written.extend(writes.bindings.iter().copied());
        writers.every_binding |= writes.unknown;
    });
}

/// A shader that may write through addresses, or one that can't be looked inside
pub fn addresses_write() {
    writers(|writers| {
        writers.addresses_write = true;
        writers.every_binding = true;
    });
}

/// A command that writes by address
pub fn anything_written() {
    writers(|writers| writers.everything = true);
}

/// Whether SPIR-V may write through a buffer's address: stores, copies or atomics to a pointer into
/// the physical storage buffer class. Ids are unique in a module, so a word that names one of those
/// pointer types is one, whichever instruction it's in.
pub fn writes_through_addresses(code: &[u32]) -> bool {
    const OP_TYPE_POINTER: u32 = 32;
    const OP_STORE: u32 = 62;
    const OP_COPY_MEMORY: u32 = 63;
    const OP_COPY_MEMORY_SIZED: u32 = 64;
    const OP_ATOMIC_FIRST: u32 = 227;
    const OP_ATOMIC_LAST: u32 = 242;
    const OP_VENDOR_FIRST: u32 = 4000;
    const PHYSICAL_STORAGE_BUFFER: u32 = 5349;

    let instructions = || {
        let mut at = 5;
        std::iter::from_fn(move || {
            let word = *code.get(at)?;
            let length = ((word >> 16) as usize).max(1);
            let operands = code.get(at + 1..at + length)?;
            at += length;
            Some((word & 0xffff, operands))
        })
    };
    let mut types = HashSet::new();
    let mut pointers = HashSet::new();
    for (opcode, operands) in instructions() {
        if opcode == OP_TYPE_POINTER && operands.get(1) == Some(&PHYSICAL_STORAGE_BUFFER) {
            types.insert(operands[0]);
        } else if let [result_type, result, ..] = operands
            && types.contains(result_type)
        {
            pointers.insert(*result);
        }
    }
    if pointers.is_empty() {
        return false;
    }
    instructions().any(|(opcode, operands)| match opcode {
        OP_STORE | OP_COPY_MEMORY | OP_COPY_MEMORY_SIZED => operands.first().is_some_and(|target| pointers.contains(target)),
        // Atomics, and vendors' instructions, which aren't all read here
        OP_ATOMIC_FIRST..=OP_ATOMIC_LAST | OP_VENDOR_FIRST.. => operands.iter().any(|word| pointers.contains(word)),
        _ => false,
    })
}

/// A buffer's memory and where it is in that, once bound
pub fn buffer_place(buffer: u64) -> Option<(u64, u64)> {
    writers(|writers| writers.buffers.get(&buffer).filter(|info| info.memory != 0).map(|info| (info.memory, info.offset)))
}

/// What SPIR-V may write through descriptors
#[derive(Debug, Default, PartialEq)]
pub struct StorageWrites {
    /// Binding numbers it writes through
    pub bindings: Vec<u32>,
    /// It writes somewhere that can't be followed back to a binding
    pub unknown: bool,
}

/// Which bindings SPIR-V writes through: stores, copies or atomics to a pointer into the storage buffer
/// class (or the uniform class, where old modules keep them), and image writes. Each is followed back
/// through access chains and loads to the variable it came from, and that variable's binding.
pub fn storage_writes_of(code: &[u32]) -> StorageWrites {
    const OP_IMAGE_TEXEL_POINTER: u32 = 60;
    const OP_LOAD: u32 = 61;
    const OP_STORE: u32 = 62;
    const OP_COPY_MEMORY: u32 = 63;
    const OP_COPY_MEMORY_SIZED: u32 = 64;
    const OP_ACCESS_CHAIN: u32 = 65;
    const OP_IN_BOUNDS_ACCESS_CHAIN: u32 = 66;
    const OP_PTR_ACCESS_CHAIN: u32 = 67;
    const OP_TYPE_POINTER: u32 = 32;
    const OP_VARIABLE: u32 = 59;
    const OP_DECORATE: u32 = 71;
    const OP_COPY_OBJECT: u32 = 83;
    const OP_IMAGE_WRITE: u32 = 99;
    const OP_ATOMIC_LOAD: u32 = 227;
    const OP_ATOMIC_STORE: u32 = 228;
    const OP_ATOMIC_LAST: u32 = 242;
    const OP_VENDOR_FIRST: u32 = 4000;
    // An access chain of NVIDIA's, which DXVK uses for structured buffers
    const OP_RAW_ACCESS_CHAIN_NV: u32 = 5398;
    const UNIFORM: u32 = 2;
    const STORAGE_BUFFER: u32 = 12;
    const BINDING: u32 = 33;

    let instructions = || {
        let mut at = 5;
        std::iter::from_fn(move || {
            let word = *code.get(at)?;
            let length = ((word >> 16) as usize).max(1);
            let operands = code.get(at + 1..at + length)?;
            at += length;
            Some((word & 0xffff, operands))
        })
    };
    let mut bindings: HashMap<u32, u32> = HashMap::new();
    let mut storage_types = HashSet::new();
    let mut storage_pointers = HashSet::new();
    // What each id came from, as far back as a variable
    let mut roots: HashMap<u32, u32> = HashMap::new();
    for (opcode, operands) in instructions() {
        match (opcode, operands) {
            (OP_DECORATE, [target, BINDING, binding, ..]) => {
                bindings.insert(*target, *binding);
            }
            (OP_TYPE_POINTER, [id, class, ..]) if *class == STORAGE_BUFFER || *class == UNIFORM => {
                storage_types.insert(*id);
            }
            (OP_VARIABLE, [_, id, ..]) => {
                roots.insert(*id, *id);
            }
            (OP_ACCESS_CHAIN | OP_IN_BOUNDS_ACCESS_CHAIN | OP_PTR_ACCESS_CHAIN | OP_RAW_ACCESS_CHAIN_NV | OP_LOAD | OP_COPY_OBJECT | OP_IMAGE_TEXEL_POINTER, [_, id, base, ..]) => {
                if let Some(root) = roots.get(base).copied() {
                    roots.insert(*id, root);
                }
            }
            _ => {}
        }
        if let [result_type, result, ..] = operands
            && storage_types.contains(result_type)
        {
            storage_pointers.insert(*result);
        }
    }
    let mut writes = StorageWrites::default();
    let mut through = |target: u32| match roots.get(&target).and_then(|root| bindings.get(root)) {
        Some(binding) => writes.bindings.push(*binding),
        None => writes.unknown = true,
    };
    for (opcode, operands) in instructions() {
        match opcode {
            OP_IMAGE_WRITE => {
                if let Some(image) = operands.first() {
                    through(*image);
                }
            }
            OP_STORE | OP_COPY_MEMORY | OP_COPY_MEMORY_SIZED | OP_ATOMIC_STORE => {
                if let Some(target) = operands.first().filter(|target| storage_pointers.contains(target) || roots.get(target).is_some_and(|root| bindings.contains_key(root))) {
                    through(*target);
                }
            }
            // These read, or only say where
            OP_ATOMIC_LOAD | OP_RAW_ACCESS_CHAIN_NV => {}
            // Atomics, with what they change after a type and a result; and vendors' instructions,
            // which aren't all read here
            OP_ATOMIC_LOAD..=OP_ATOMIC_LAST | OP_VENDOR_FIRST.. => {
                for word in operands {
                    if storage_pointers.contains(word) || roots.get(word).is_some_and(|root| bindings.contains_key(root)) {
                        through(*word);
                    }
                }
            }
            _ => {}
        }
    }
    writes.bindings.sort_unstable();
    writes.bindings.dedup();
    writes
}

/// Whether a buffer's memory is mapped, or isn't known
pub fn buffer_is_mapped(buffer: u64) -> bool {
    let memory = writers(|writers| writers.buffers.get(&buffer).map(|entry| entry.memory));
    match memory {
        Some(memory) if memory != 0 => {
            MAPPINGS.lock().unwrap_or_else(|p| p.into_inner()).as_ref().is_some_and(|mappings| mappings.contains_key(&memory))
        }
        _ => true,
    }
}

impl Writers {
    /// Whether nothing could have changed in this memory since it was last looked at
    fn quiet(&self, memory: u64) -> bool {
        if self.never.contains(&memory) {
            return true;
        }
        if !self.copies_only.contains(&memory) && (self.everything || self.memory.contains(&memory) || (self.addresses_write && self.addressed.contains(&memory))) {
            return false;
        }
        self.ranges.get(&memory).is_none_or(BTreeMap::is_empty)
            && !self.once.any_finished(memory)
            && !self.storage.iter().any(|(binding, memories)| (self.every_binding || self.bindings_written.contains(binding)) && memories.contains_key(&memory))
    }
}

/// Of these runs of pages, those the GPU could have written. The rest can't have changed.
fn pages_gpu_can_write(memory: u64, runs: &[(usize, usize)]) -> Vec<(usize, usize)> {
    writers(|writers| {
        if writers.never.contains(&memory) {
            return Vec::new();
        }
        let watched = writers.copies_only.contains(&memory);
        if !watched && (writers.everything || writers.memory.contains(&memory) || (writers.addresses_write && writers.addressed.contains(&memory))) {
            return runs.to_vec();
        }
        let ranges = writers.ranges.get(&memory);
        let storage: Vec<&BTreeMap<u64, u64>> = writers
            .storage
            .iter()
            .filter(|(binding, _)| writers.every_binding || writers.bindings_written.contains(binding))
            .filter_map(|(_, memories)| memories.get(&memory))
            .collect();
        let pages: Vec<usize> = runs
            .iter()
            .flat_map(|&(first, end)| first..end)
            .filter(|&page| {
                let (start, end) = ((page * PAGE) as u64, ((page + 1) * PAGE) as u64);
                ranges.is_some_and(|ranges| range_touches(ranges, start, end))
                    || storage.iter().any(|ranges| range_touches(ranges, start, end))
                    || writers.once.touches(memory, start, end)
            })
            .collect();
        self::runs(&pages)
    })
}

fn memory_forgotten(memory: u64) {
    writers(|writers| {
        writers.memory.remove(&memory);
        writers.ranges.remove(&memory);
        for memories in writers.storage.values_mut() {
            memories.remove(&memory);
        }
        writers.once.ranges.remove(&memory);
        writers.addressed.remove(&memory);
        writers.never.remove(&memory);
        writers.copies_only.remove(&memory);
    });
    PUSHED_PAGES.lock().unwrap_or_else(|p| p.into_inner()).forget(memory);
}

/// Time spent keeping memory in step, for NECTAR_VK_STATS: nanoseconds sending what the program
/// wrote, and handling touches of stale pages, and how many touches
pub static FLUSH_NANOS: AtomicU64 = AtomicU64::new(0);
pub static TOUCH_NANOS: AtomicU64 = AtomicU64::new(0);
pub static TOUCHES: AtomicU64 = AtomicU64::new(0);
/// Touches that had to ask the lender
pub static ASKS: AtomicU64 = AtomicU64::new(0);

/// Mappings by memory id
static MAPPINGS: Mutex<Option<HashMap<u64, Mapping>>> = Mutex::new(None);
/// Pages that go readable at once where nothing could have changed: 16 MiB
const QUIET_WINDOW: usize = 4096;
/// How long what's readable stays so where nothing could have changed, which is what each submit has to
/// look at for writes
const QUIET_STALE_AFTER: std::time::Duration = std::time::Duration::from_millis(100);
/// Some pages that were being fetched have come
static FETCHED: std::sync::Condvar = std::sync::Condvar::new();

/// Pages first fetched on a touch, and the most fetched at once
const FIRST_WINDOW: usize = (1 << 20) / PAGE;
const LARGEST_WINDOW: usize = (64 << 20) / PAGE;

fn hashes(bytes: &[u8]) -> Vec<u64> {
    bytes.chunks(PAGE).map(xxh3_64).collect()
}

/// Whether pages go stale after a wait and are fetched on touch, rather than fetched every wait
fn on_touch() -> bool {
    static ON_TOUCH: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON_TOUCH.get_or_init(|| std::env::var_os("NECTAR_VK_EAGER").is_none())
}

/// Whether the system says which pages were written, rather than hashes
fn write_watched() -> bool {
    platform::WRITE_WATCH && on_touch()
}

/// The shadow of a memory object, mapping it on the lender the first time
pub fn map(connection: &Connection, memory: u64) -> Result<*mut u8, i32> {
    {
        let mappings = MAPPINGS.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(mapping) = mappings.as_ref().and_then(|mappings| mappings.get(&memory)) {
            return Ok(mapping.shadow.pointer);
        }
    }
    // Not holding the lock across the trip, as everything else that touches memory needs it. The program
    // can't be mapping the same memory on two threads at once.
    let mut w = Writer::new();
    w.u64(memory).u8(on_touch() as u8);
    let answer = connection.ask(MAP, w.as_slice()).map_err(|_| crate::DEVICE_LOST)?;
    if answer.result as i64 as i32 != 0 {
        return Err(answer.result as i64 as i32);
    }
    let mut mappings = MAPPINGS.lock().unwrap_or_else(|p| p.into_inner());
    let mappings = mappings.get_or_insert_with(HashMap::new);
    if on_touch() {
        let size = answer.outputs().u64().map_err(|_| crate::DEVICE_LOST)? as usize;
        let mapping = stale_mapping(size).ok_or(VK_ERROR_OUT_OF_HOST_MEMORY)?;
        platform::catch_touches();
        let pointer = mapping.shadow.pointer;
        mappings.insert(memory, mapping);
        return Ok(pointer);
    }
    let contents = answer.outputs().bytes().map_err(|_| crate::DEVICE_LOST)?;
    let size = contents.len();
    let pages = size.div_ceil(PAGE);
    let mut mapping = Mapping {
        shadow: platform::Shadow::new(size).ok_or(VK_ERROR_OUT_OF_HOST_MEMORY)?,
        size,
        hashes: if write_watched() { Vec::new() } else { hashes(contents) },
        stale: vec![false; pages],
        fresh: vec![(0, pages)],
        window: FIRST_WINDOW,
        next: usize::MAX,
        readback: false,
        fetching: Vec::new(),
        marked: 0,
        marked_at: Instant::now(),
        unsent: BTreeMap::new(),
    };
    mapping.bytes().copy_from_slice(contents);
    let pointer = mapping.shadow.pointer;
    mappings.insert(memory, mapping);
    Ok(pointer)
}

const VK_ERROR_OUT_OF_HOST_MEMORY: i32 = -1;

/// A new shadow of `size` bytes, all zeros and all stale: nothing crosses until it's touched, and
/// then only pages that aren't zero. Mapping a large staging buffer costs nothing.
fn stale_mapping(size: usize) -> Option<Mapping> {
    let pages = size.div_ceil(PAGE);
    let mut hashes = if write_watched() { Vec::new() } else { vec![xxh3_64(&[0u8; PAGE]); pages] };
    if !size.is_multiple_of(PAGE)
        && let Some(last) = hashes.last_mut()
    {
        *last = xxh3_64(&vec![0u8; size % PAGE]);
    }
    let mut mapping = Mapping {
        shadow: platform::Shadow::new(size)?,
        size,
        hashes,
        stale: vec![true; pages],
        fresh: Vec::new(),
        window: FIRST_WINDOW,
        next: usize::MAX,
        readback: false,
        fetching: Vec::new(),
        marked: 0,
        marked_at: Instant::now(),
        unsent: BTreeMap::new(),
    };
    mapping.shadow.reset(0, size);
    mapping.shadow.protect(0, size, platform::Access::None);
    Some(mapping)
}

/// Whether the program only reads what was written before it recorded the commands that read it, as
/// DXVK does. NECTAR_VK_HOLD_WRITES=0 says otherwise.
static HOLD_WRITES: AtomicBool = AtomicBool::new(false);
/// Command buffers finished recording
static FINISHED: AtomicU64 = AtomicU64::new(0);

/// The program is Direct3D 12 through vkd3d-proton. DXVK's DXGI may be in the same process, with an
/// instance of its own for looking at adapters, and that says nothing about how the writes behave.
static D3D12: AtomicBool = AtomicBool::new(false);
/// The flags of each memory type, as the program was told them
static MEMORY_TYPES: Mutex<Vec<nectar_vk_sys::VkMemoryPropertyFlags>> = Mutex::new(Vec::new());

/// The program named the engine it's built on
pub fn engine_named(name: &[u8]) {
    if name.starts_with(b"vkd3d") {
        D3D12.store(true, Ordering::Relaxed);
    }
    let holds = name == b"DXVK" && !D3D12.load(Ordering::Relaxed) && on_touch() && std::env::var("NECTAR_VK_HOLD_WRITES").map_or(true, |value| value != "0");
    HOLD_WRITES.store(holds, Ordering::Relaxed);
}

pub fn memory_types_told(flags: &[nectar_vk_sys::VkMemoryPropertyFlags]) {
    *MEMORY_TYPES.lock().unwrap_or_else(|p| p.into_inner()) = flags.to_vec();
}

/// Memory was allocated. vkd3d-proton makes Direct3D 12's upload heaps from what the CPU can map but
/// doesn't cache, and in Direct3D 12 nothing on the GPU writes those: not a copy, and not a shader. All
/// its shaders write through addresses and it builds acceleration structures, so nothing else here would
/// rule these out, and each page the game writes after a wait would go and ask.
///
/// A custom heap could be shader-writable and of this kind, but the CPU reading it back would be very
/// slow on any GPU, so a game doesn't.
///
/// Its read-back heaps, which it makes from what the CPU caches, are written only by copies: Direct3D 12
/// lets nothing else write them. The Witcher 3 agrees, with no byte in minutes of play coming back from
/// outside a copy's range. A shader's unordered access to a custom heap of this kind is the exception,
/// which a storage descriptor on it would still show, but not an address.
pub fn memory_allocated(memory: u64, memory_type: u32) {
    if !D3D12.load(Ordering::Relaxed) {
        return;
    }
    let flags = MEMORY_TYPES.lock().unwrap_or_else(|p| p.into_inner()).get(memory_type as usize).copied().unwrap_or(0);
    if flags & nectar_vk_sys::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT == 0 {
        return;
    }
    if flags & nectar_vk_sys::VK_MEMORY_PROPERTY_HOST_CACHED_BIT == 0 {
        writers(|writers| writers.never.insert(memory));
    } else {
        writers(|writers| writers.copies_only.insert(memory));
    }
}

pub fn holding_writes() -> bool {
    HOLD_WRITES.load(Ordering::Relaxed)
}

/// A command buffer finished recording: what's written by now may be read by it. Returns which this
/// was, to say what its submit must send.
pub fn command_buffer_ended() -> u64 {
    if !holding_writes() {
        return 0;
    }
    let mut mappings = MAPPINGS.lock().unwrap_or_else(|p| p.into_inner());
    let finished = FINISHED.fetch_add(1, Ordering::Relaxed) + 1;
    for mapping in mappings.iter_mut().flat_map(|mappings| mappings.values_mut()) {
        let written = mapping.written(true);
        finished_recording(&mut mapping.unsent, written, finished);
    }
    finished
}

/// What's waiting to go, as a command buffer finishes recording: all that's been written so far,
/// including what was noticed since the last one finished, may be read by it
fn finished_recording(unsent: &mut BTreeMap<usize, u64>, written: Vec<usize>, finished: u64) {
    for later in unsent.values_mut().filter(|when| **when == u64::MAX) {
        *later = finished;
    }
    for page in written {
        unsent.entry(page).or_insert(finished);
    }
}

/// Takes from `unsent` the pages a submit must send: those written before `due`, and any part of what
/// the submit copies, but not what only a command buffer yet to be submitted copies. In address order.
fn due_pages(unsent: &mut BTreeMap<usize, u64>, due: u64, regions: &[(usize, usize)], elsewhere: &[(usize, usize)]) -> Vec<usize> {
    let touches = |page: usize, region: &(usize, usize)| page * PAGE < region.0 + region.1 && (page + 1) * PAGE > region.0;
    let within = |page: usize, region: &(usize, usize)| region.0 <= page * PAGE && (page + 1) * PAGE <= region.0 + region.1;
    let pages: Vec<usize> = unsent
        .iter()
        .filter(|&(&page, &written)| {
            regions.iter().any(|region| touches(page, region)) || (written <= due && !elsewhere.iter().any(|region| within(page, region)))
        })
        .map(|(&page, _)| page)
        .collect();
    for page in &pages {
        unsent.remove(page);
    }
    pages
}

/// A run of writes nothing explains that's worth waiting a moment to find out about
const LONG_RUN: usize = (4 << 20) / PAGE;
/// How long a run must stop growing, with no copy recorded, to give up on one coming; and the most
/// waited in all
const COPY_DUE: std::time::Duration = std::time::Duration::from_millis(20);
const COPY_WAIT_MOST: std::time::Duration = std::time::Duration::from_millis(500);
/// Times no copy came for a long run in a memory object before it stops being waited for
const WAITS_IN_VAIN: u32 = 3;

/// Long runs of pages a submit would send with nothing to say what they are: for each, its first page
/// and length. A run counts if it holds a page the submit would send; the rest of it is what's growing.
fn long_runs(unsent: &BTreeMap<usize, u64>, due: u64, claims: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let claimed = |page: usize| claims.iter().any(|claim| claim.0 <= page * PAGE && (page + 1) * PAGE <= claim.0 + claim.1);
    let mut found = Vec::new();
    let (mut start, mut length, mut needed) = (0, 0, false);
    let mut finish = |start: usize, length: usize, needed: bool| {
        if needed && length >= LONG_RUN {
            found.push((start, length));
        }
    };
    for (&page, &written) in unsent {
        if claimed(page) {
            continue;
        }
        if length > 0 && page == start + length {
            length += 1;
            needed |= written <= due;
        } else {
            finish(start, length, needed);
            (start, length, needed) = (page, 1, written <= due);
        }
    }
    finish(start, length, needed);
    found
}

/// Before a submit: if it would send a long run that nothing explains, gives the copy that reads it a
/// moment to be recorded, calling `claims` for what's copied by then
pub fn wait_for_copies(submit: &Submit, claims: impl Fn() -> Vec<Region>) {
    static IN_VAIN: Mutex<Option<HashMap<u64, u32>>> = Mutex::new(None);
    let Some(due) = submit.due else { return };
    let started = Instant::now();
    let (mut last, mut changed) = (Vec::new(), started);
    loop {
        let elsewhere = claims();
        let spans = |memory: u64| -> Vec<(usize, usize)> {
            submit.regions.iter().chain(&elsewhere).filter(|region| region.memory == memory).map(|region| (region.offset, region.size)).collect()
        };
        let long: Vec<(u64, usize)> = {
            let mut mappings = MAPPINGS.lock().unwrap_or_else(|p| p.into_inner());
            let mut in_vain = IN_VAIN.lock().unwrap_or_else(|p| p.into_inner());
            let in_vain = in_vain.get_or_insert_with(HashMap::new);
            let mut long = Vec::new();
            for (&memory, mapping) in mappings.iter_mut().flat_map(|mappings| mappings.iter_mut()) {
                if in_vain.get(&memory).copied().unwrap_or(0) >= WAITS_IN_VAIN {
                    continue;
                }
                for page in mapping.written(true) {
                    mapping.unsent.entry(page).or_insert(u64::MAX);
                }
                let length: usize = long_runs(&mapping.unsent, due, &spans(memory)).iter().map(|run| run.1).sum();
                if length > 0 {
                    long.push((memory, length));
                }
            }
            long.sort_unstable();
            if !long.is_empty() && (started.elapsed() >= COPY_WAIT_MOST || (long == last && changed.elapsed() >= COPY_DUE)) {
                for (memory, _) in &long {
                    *in_vain.entry(*memory).or_insert(0) += 1;
                }
                return;
            }
            long
        };
        if long.is_empty() {
            return;
        }
        if long != last {
            last = long;
            changed = Instant::now();
        }
        std::thread::sleep(std::time::Duration::from_micros(500));
    }
}

/// Sends the pages the program changed, in runs; called before the GPU could see them, with what the
/// work about to go copies
pub fn flush(connection: &Arc<Connection>, submit: &Submit) {
    let started = Instant::now();
    send_written(connection, submit);
    FLUSH_NANOS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

fn send_written(connection: &Arc<Connection>, submit: &Submit) {
    let regions = &submit.regions;
    let mut mappings = MAPPINGS.lock().unwrap_or_else(|p| p.into_inner());
    let Some(mappings) = mappings.as_mut() else { return };
    let mut written_in: Vec<(u64, Vec<usize>)> = Vec::new();
    for (&memory, mapping) in mappings.iter_mut() {
        let mut written = if on_touch() { mapping.written(true) } else { compare(mapping) };
        if holding_writes() {
            for page in written {
                mapping.unsent.entry(page).or_insert(u64::MAX);
            }
            let spans = |list: &[Region]| list.iter().filter(|region| region.memory == memory).map(|region| (region.offset, region.size)).collect::<Vec<_>>();
            // Sent by something that wasn't a submit, everything goes
            let due = submit.due.unwrap_or(u64::MAX);
            written = due_pages(&mut mapping.unsent, due, &spans(regions), &spans(&submit.elsewhere));
        }
        if !written.is_empty() {
            PUSHED_PAGES.lock().unwrap_or_else(|p| p.into_inner()).overwritten(memory, &written);
            written_in.push((memory, written));
        }
    }
    let mut chosen = choose_later(connection, mappings, &written_in, submit);
    for (memory, written) in &written_in {
        let memory = *memory;
        let mapping = &mappings[&memory];
        let size = mapping.size;
        let bytes: &[u8] = unsafe { std::slice::from_raw_parts(mapping.shadow.pointer, size) };
        let mut mine: Vec<Region> = regions.iter().filter(|region| region.memory == memory && region.size >= REGION_MIN).copied().collect();
        let mut done_here = Vec::new();
        // Hashed already, it goes whole by its blocks even if part went earlier, so a repeat still matches
        mine.retain(|region| {
            let looked_at = (memory, region.offset);
            match chosen.hashes.remove(&looked_at) {
                Some(hashes) => {
                    send_region(connection, region, &bytes[region.offset..region.offset + region.size], &hashes);
                    done_here.push((region.offset, region.size));
                    false
                }
                None => true,
            }
        });
        // What's coming later needn't go now. Neither goes again what's done, unless a page holds
        // something else too.
        let later_here: Vec<(usize, usize)> = mine
            .iter()
            .filter(|region| chosen.waiting.contains(&(memory, region.offset)))
            .map(|region| (region.offset, region.size))
            .chain(done_here.iter().copied())
            .collect();
        let written: Vec<usize> = written
            .iter()
            .copied()
            .filter(|page| !later_here.iter().any(|span| span.0 <= page * PAGE && (page + 1) * PAGE <= span.0 + span.1))
            .collect();
        mine.sort_unstable_by_key(|region| region.offset);
        for (first, end) in runs(&written) {
            let (start, end) = (first * PAGE, (end * PAGE).min(size));
            for piece in pieces(start, end, &mine) {
                match piece {
                    Piece::Copied(region) if chosen.waiting.contains(&(memory, region.offset)) => {}
                    Piece::Copied(region) => send_region(connection, &region, &bytes[region.offset..region.offset + region.size], &[]),
                    Piece::Rest(from, to) => send_run(connection, memory, from, &bytes[from..to]),
                }
            }
        }
    }
    for (message, fill) in &chosen.fills {
        let _ = connection.send(*message, fill.as_slice());
    }
}

/// What a copy reads is kept in blocks of this much at most, which is also how much can hold up the
/// game's traffic when a mip is coming later
pub const KEPT_BLOCK: usize = 128 << 10;
/// A copy under this goes as it is: its hash would save too little
const REGION_MIN: usize = PAGE;
/// Hashes remembered as seen once, past which they're forgotten
const SEEN_MOST: usize = 1 << 20;
/// Writes with no known copy are cut into pieces at least this long and at most CUT_MOST, about 96 KiB
/// on average. 900 MB of them come to about 500 KB of hashes.
const CUT_LEAST: usize = 32 << 10;
const CUT_MOST: usize = 512 << 10;
/// One word in this many ends a piece
const CUT_BITS: u32 = 13;
/// Words a cut depends on, which is how many it takes to shift one out
const CUT_WINDOW: usize = 64 * 8;
/// A run with no known copy this long may be a load, and is worth keeping the first time it's seen;
/// a frame's constants written in a ring are shorter, and never come again
const LOAD_LEAST: usize = 2 << 20;
/// What may be kept the first time it's seen: this much at once, and more at this rate
const FIRST_SIGHT_MOST: f64 = 2.0 * (1u64 << 30) as f64;
const FIRST_SIGHT_PER_SECOND: f64 = 1e6;

/// Why bytes are being sent, which says if they're worth keeping
#[derive(Clone, Copy)]
enum Sending {
    /// A copy to this place reads them
    Copied((u64, u64)),
    /// Nothing says what they are, but they're in a run this long
    Unknown(usize),
}

/// How long the next piece of `bytes` is, by its content. Words are read at multiples of 8 from the
/// start of the memory, `offset` being where `bytes` starts: uploads sit at larger multiples than that,
/// so the same bytes give the same words wherever they are. Reading words rather than bytes is what
/// makes this quick enough for a frame.
fn next_cut(bytes: &[u8], offset: usize) -> usize {
    if bytes.len() <= CUT_LEAST {
        return bytes.len();
    }
    let end = bytes.len().min(CUT_MOST);
    let mut at = CUT_LEAST - CUT_WINDOW;
    at += (8 - (offset + at) % 8) % 8;
    let mut mixed = 0u64;
    while at + 8 <= end {
        let word = u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
        mixed = (mixed << 1).wrapping_add(word.wrapping_mul(0x9E37_79B1_85EB_CA87));
        at += 8;
        if at >= CUT_LEAST && mixed >> (64 - CUT_BITS) == 0 {
            return at;
        }
    }
    end
}

/// What the lender keeps, which it gave once, and what's decided what to send it to keep
struct Keeper {
    hashes: HashSet<Hash>,
    /// Places written to
    places: HashSet<(u64, u64)>,
    /// Hashes that went without being kept
    seen: HashSet<Hash>,
    /// Bytes that may be kept the first time they're seen, and when that was worked out
    first_sight: f64,
    worked_out: Instant,
}

impl Keeper {
    fn new(hashes: HashSet<Hash>) -> Self {
        Self { hashes, places: HashSet::new(), seen: HashSet::new(), first_sight: FIRST_SIGHT_MOST, worked_out: Instant::now() }
    }

    /// Whether new bytes are worth keeping. Bytes seen before are. The first bytes for a place are
    /// likely a texture or a tensor loaded once. Other bytes rewriting it are likely made each frame,
    /// and never come again. Where the place isn't known yet, a long run may be a load, for as long as
    /// the allowance for that lasts, which limits what a game writing long runs each frame can store.
    fn worth_keeping(&mut self, hash: Hash, size: usize, sending: Sending, now: Instant) -> bool {
        if self.places.len() >= SEEN_MOST {
            self.places.clear();
        }
        if self.seen.len() >= SEEN_MOST {
            self.seen.clear();
        }
        let again = !self.seen.insert(hash);
        let first = match sending {
            Sending::Copied(place) => self.places.insert(place),
            Sending::Unknown(run) if run >= LOAD_LEAST => {
                let since = now.saturating_duration_since(self.worked_out).as_secs_f64();
                self.worked_out = now;
                self.first_sight = (self.first_sight + since * FIRST_SIGHT_PER_SECOND).min(FIRST_SIGHT_MOST);
                let allowed = self.first_sight >= size as f64;
                if allowed && !again {
                    self.first_sight -= size as f64;
                }
                allowed
            }
            Sending::Unknown(_) => false,
        };
        first || again
    }
}

/// A run of written bytes, divided by what's copied out of them
#[derive(Debug, PartialEq)]
enum Piece {
    Copied(Region),
    Rest(usize, usize),
}

/// Divides bytes `start..end` by `regions`, in address order. Every byte is in exactly one piece.
/// A region reaching outside goes with the rest, as it may include pages the lender already has, or
/// that are stale; so does one overlapping a region before it.
fn pieces(start: usize, end: usize, regions: &[Region]) -> Vec<Piece> {
    let mut pieces = Vec::new();
    let mut at = start;
    for region in regions {
        if region.offset >= at && region.offset + region.size <= end {
            if region.offset > at {
                pieces.push(Piece::Rest(at, region.offset));
            }
            pieces.push(Piece::Copied(*region));
            at = region.offset + region.size;
        }
    }
    if at < end {
        pieces.push(Piece::Rest(at, end));
    }
    pieces
}

/// The hashes the lender keeps; None when it keeps none
fn keeper(connection: &Connection) -> &'static Mutex<Option<Keeper>> {
    static KEEPER: std::sync::OnceLock<Mutex<Option<Keeper>>> = std::sync::OnceLock::new();
    KEEPER.get_or_init(|| {
        let hashes = connection.ask(CACHE_INDEX, &[]).ok().filter(|answer| answer.result == 0).and_then(|answer| {
            let mut r = answer.outputs();
            let count = r.u64().ok()?;
            let mut hashes = HashSet::new();
            for _ in 0..count {
                hashes.insert(Hash::try_from(r.take(16).ok()?).ok()?);
            }
            Some(Keeper::new(hashes))
        });
        Mutex::new(hashes)
    })
}

/// Shader code under this goes whole: its hash would save too little
const KEPT_CODE_MIN: usize = 1 << 10;

/// How shader code goes: by hash, to a lender that keeps things and for code worth it
pub(crate) fn code_way(code: &[u8]) -> CodeWay {
    let Some(connection) = crate::connection::get().filter(|connection| connection.keep_uploads && code.len() >= KEPT_CODE_MIN) else {
        return CodeWay::Whole;
    };
    let mut keeper = keeper(&connection).lock().unwrap_or_else(|p| p.into_inner());
    let Some(keeper) = keeper.as_mut() else { return CodeWay::Whole };
    let hash = content_hash(code);
    if keeper.hashes.insert(hash) { CodeWay::Keep(hash) } else { CodeWay::Kept(hash) }
}

/// Held from choosing how a command's shader code goes until the command is on its way: code one
/// thread says is kept must not overtake another thread's sending it to keep
pub(crate) fn code_order(connection: &Connection, name: &str) -> Option<std::sync::MutexGuard<'static, ()>> {
    static ORDER: Mutex<()> = Mutex::new(());
    let carries_code = matches!(
        name,
        "vkCreateShaderModule" | "vkCreateGraphicsPipelines" | "vkCreateComputePipelines" | "vkCreateRayTracingPipelinesKHR" | "vkCreateRayTracingPipelinesNV"
    );
    (carries_code && connection.keep_uploads).then(|| ORDER.lock().unwrap_or_else(|p| p.into_inner()))
}

/// Sends bytes as they are
fn send_bytes(connection: &Connection, memory: u64, offset: usize, bytes: &[u8]) {
    let mut w = Writer::new();
    w.u64(memory).u64(offset as u64).bytes(bytes);
    let _ = connection.send(WRITE, w.as_slice());
}

/// Mips that are to come later, as decided for a submit
#[derive(Default)]
struct Chosen {
    /// What not to send now, by memory and offset
    waiting: HashSet<(u64, usize)>,
    /// Hashes of each block of what's been looked at, not to work them out twice
    hashes: HashMap<(u64, usize), Vec<Hash>>,
    /// What makes each waiting mip out of the one below or a sample of its own, smallest first, to go
    /// after what they read
    fills: Vec<(u32, Writer)>,
}

/// Decides which of the mips this submit copies can come later, takes a copy of their bytes for the
/// thread to send, and sends what makes do until then
fn choose_later(connection: &Arc<Connection>, mappings: &HashMap<u64, Mapping>, written_in: &[(u64, Vec<usize>)], submit: &Submit) -> Chosen {
    let mut chosen = Chosen::default();
    if !connection.keep_uploads || !later::enabled() || connection.keep_stopped.load(Ordering::Relaxed) {
        return chosen;
    }
    // Only what's here to read, and has mostly still to go: part may have gone already, if it was being
    // written when an earlier submit went
    let worth_waiting = |region: &Region| {
        let Some(mapping) = mappings.get(&region.memory) else { return false };
        let pages = region.offset / PAGE..(region.offset + region.size).div_ceil(PAGE);
        let total = pages.len();
        let written = written_in.iter().find(|(memory, _)| *memory == region.memory).map_or(&[][..], |(_, written)| &written[..]);
        let unsent = pages.clone().filter(|page| written.binary_search(page).is_ok()).count();
        unsent * 2 >= total && pages.into_iter().all(|page| !mapping.stale[page])
    };
    let mut images: BTreeMap<u64, BTreeMap<u32, Region>> = BTreeMap::new();
    for region in submit.regions.iter().filter(|region| region.mip.is_some()) {
        if worth_waiting(region) {
            images.entry(region.place.0).or_default().insert(region.mip.unwrap_or(0), *region);
        }
    }
    let mut keeper_guard = keeper(connection).lock().unwrap_or_else(|p| p.into_inner());
    let Some(keeper) = keeper_guard.as_mut() else { return chosen };
    let (mut spent, mut held_now, mut fills) = (0usize, 0usize, Vec::new());
    for (image, mips) in &images {
        let bytes_of = |region: &Region| {
            let mapping = &mappings[&region.memory];
            unsafe { std::slice::from_raw_parts(mapping.shadow.pointer.add(region.offset), region.size) }
        };
        // Smallest first
        let mut sizes = Vec::new();
        for (mip, region) in mips.iter().rev() {
            let hashes: Vec<Hash> = bytes_of(region).chunks(KEPT_BLOCK).map(content_hash).collect();
            let unkept = hashes.iter().zip(bytes_of(region).chunks(KEPT_BLOCK)).filter(|(hash, _)| !keeper.hashes.contains(*hash)).map(|(_, block)| block.len()).sum();
            sizes.push((*mip, unkept));
            chosen.hashes.insert((region.memory, region.offset), hashes);
        }
        let waiting = later::choose(&sizes, &mut spent, &mut |mip| {
            let may = later::may_wait(*image, mip, submit.queue, mips[&mip].memory, mips[&mip].size, held_now);
            if may {
                held_now += mips[&mip].size;
            }
            may
        });
        for mip in waiting {
            let region = &mips[&mip];
            let (block, width, height) = region.block;
            let blocks = |extent: (u32, u32)| (extent.0.div_ceil(width as u32), extent.1.div_ceil(height as u32));
            let to = blocks(region.extent);
            let mut w = Writer::new();
            // The one below is better, if it's here
            let message = if let Some(below) = mips.get(&(mip + 1)) {
                let from = blocks(below.extent);
                w.u64(below.memory).u64(below.offset as u64).u32(from.0).u32(from.1);
                w.u64(region.memory).u64(region.offset as u64).u32(to.0).u32(to.1).u32(block as u32);
                // Repeating a block's detail makes a pattern, and flat colours a small picture. Over a
                // stand-in that's been done already.
                w.u32(if chosen.waiting.contains(&(below.memory, below.offset)) { 0 } else { later::format_of(*image) as u32 });
                FILL_MIP
            } else {
                let scale = later::sample_scale(to.0 as usize, to.1 as usize, block);
                let (mut samples, _, _) = nectar_vk_wire::sample_mip(bytes_of(region), to.0 as usize, to.1 as usize, block, scale);
                // Where it's known how, flat colours look like a small picture rather than a pattern
                if let Some(kind) = later::kind_of(*image) {
                    samples.chunks_exact_mut(block).for_each(|sample| nectar_vk_wire::bc::flatten_block(kind, sample));
                }
                w.u64(region.memory).u64(region.offset as u64).u32(to.0).u32(to.1).u32(block as u32).u32(scale as u32).bytes(&samples);
                FILL_MIP_FROM
            };
            fills.push((message, w));
            chosen.waiting.insert((region.memory, region.offset));
            chosen.hashes.remove(&(region.memory, region.offset));
            later::wait(connection, *image, mip, submit.queue, region.memory, bytes_of(region).to_vec(), region.place);
        }
    }
    drop(keeper_guard);
    chosen.fills = fills;
    chosen
}

/// Sends a block of a mip that came later, as send_region would have
pub fn send_later_block(connection: &Connection, memory: u64, offset: usize, block: &[u8], place: (u64, u64), index: usize) {
    let mut keeper = keeper(connection).lock().unwrap_or_else(|p| p.into_inner());
    match keeper.as_mut() {
        Some(keeper) if block.len() >= REGION_MIN => {
            let place = (place.0, place.1 ^ ((index as u64) << 24));
            send_block(connection, keeper, memory, offset, block, Sending::Copied(place), None);
        }
        _ => send_bytes(connection, memory, offset, block),
    }
}

/// Sends a block by its hash, or to keep, or as it is if it isn't worth keeping
fn send_block(connection: &Connection, keeper: &mut Keeper, memory: u64, offset: usize, block: &[u8], sending: Sending, hash: Option<Hash>) {
    let hash = hash.unwrap_or_else(|| content_hash(block));
    if keeper.hashes.contains(&hash) {
        let mut w = Writer::new();
        w.u64(memory).u64(offset as u64).raw(&hash).u64(block.len() as u64);
        let _ = connection.send(KEPT, w.as_slice());
    } else if connection.keep_stopped.load(Ordering::Relaxed) || !keeper.worth_keeping(hash, block.len(), sending, Instant::now()) {
        send_bytes(connection, memory, offset, block);
    } else {
        // Hashed again from the copy that goes, as another thread may be writing these pages: the
        // lender must hold what its hash says, since later only the hash will come
        let copy = block.to_vec();
        let hash = content_hash(&copy);
        let mut w = Writer::new();
        w.u64(memory).u64(offset as u64).raw(&hash).bytes(&copy);
        let _ = connection.send(KEEP, w.as_slice());
        keeper.hashes.insert(hash);
    }
}

/// Sends the bytes a copy reads, by hash, which may have been worked out already
fn send_region(connection: &Connection, region: &Region, bytes: &[u8], hashes: &[Hash]) {
    let mut keeper = keeper(connection).lock().unwrap_or_else(|p| p.into_inner());
    let Some(keeper) = keeper.as_mut() else {
        send_bytes(connection, region.memory, region.offset, bytes);
        return;
    };
    let mut at = 0;
    for (index, block) in bytes.chunks(KEPT_BLOCK).enumerate() {
        if block.len() >= REGION_MIN {
            let place = (region.place.0, region.place.1 ^ ((index as u64) << 24));
            send_block(connection, keeper, region.memory, region.offset + at, block, Sending::Copied(place), hashes.get(index).copied());
        } else {
            send_bytes(connection, region.memory, region.offset + at, block);
        }
        at += block.len();
    }
}

/// Sends a run of written bytes starting at `start`, in pieces by hash where that's worth it
fn send_run(connection: &Connection, memory: u64, start: usize, bytes: &[u8]) {
    let mut at = 0;
    if connection.keep_uploads && bytes.len() > CUT_LEAST {
        let mut keeper = keeper(connection).lock().unwrap_or_else(|p| p.into_inner());
        if let Some(keeper) = keeper.as_mut() {
            while bytes.len() - at > CUT_LEAST {
                let cut = next_cut(&bytes[at..], start + at);
                send_block(connection, keeper, memory, start + at, &bytes[at..at + cut], Sending::Unknown(bytes.len()), None);
                at += cut;
            }
        }
    }
    if at < bytes.len() {
        send_bytes(connection, memory, start + at, &bytes[at..]);
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
    if on_touch() {
        // Where the system doesn't watch writes, what's fresh is compared on each submit, so it has to go
        // stale all the same to keep that small
        let looking: Vec<u64> = if platform::WRITE_WATCH {
            let pushed = PUSHED_PAGES.lock().unwrap_or_else(|p| p.into_inner());
            writers(|writers| {
                mappings
                    .iter()
                    .filter(|(memory, mapping)| !writers.quiet(**memory) || pushed.has(**memory) || mapping.marked_at.elapsed() >= QUIET_STALE_AFTER)
                    .map(|(memory, _)| *memory)
                    .collect()
            })
        } else {
            mappings.keys().copied().collect()
        };
        for mapping in mappings.values_mut() {
            mapping.marked += 1;
        }
        for memory in looking {
            if let Some(mapping) = mappings.get_mut(&memory) {
                mark_stale(mapping);
            }
        }
        return;
    }
    let Ok(answer) = connection.ask(REFRESH, &[]) else { return };
    let mut r = answer.outputs();
    let (Ok(_), Ok(count)) = (r.u64(), r.u32()) else { return };
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
/// hasn't sent yet: its writes are newer than anything the GPU could have done to them.
///
/// Another thread may be writing as this runs, so writes are stopped first: a page written just
/// after being asked about and then marked stale would never be sent. A thread that writes
/// meanwhile faults, waits for this to finish, and carries on through a touch.
fn mark_stale(mapping: &mut Mapping) {
    if mapping.readback {
        return;
    }
    mapping.marked_at = Instant::now();
    let fresh = std::mem::take(&mut mapping.fresh);
    for &(first, end) in &fresh {
        mapping.shadow.protect(first * PAGE, (end - first) * PAGE, platform::Access::Read);
    }
    mapping.fresh = fresh;
    // In address order
    let mut written = mapping.written(false);
    if !mapping.unsent.is_empty() {
        written.extend(mapping.unsent.keys());
        written.sort_unstable();
        written.dedup();
    }
    // Whole pages of the system's, which may be larger than PAGE (16 KiB on Apple's CPUs)
    let granule = (platform::page_size() / PAGE).max(1);
    let written: Vec<usize> = written
        .iter()
        .flat_map(|&page| {
            let first = page - page % granule;
            first..(first + granule).min(mapping.pages())
        })
        .collect::<std::collections::BTreeSet<usize>>()
        .into_iter()
        .collect();
    let kept = |page: usize| written.binary_search(&page).is_ok();
    for (first, end) in std::mem::take(&mut mapping.fresh) {
        let mut page = first;
        while page < end {
            let start = page;
            while page < end && !kept(page) {
                page += 1;
            }
            if page > start {
                mapping.shadow.protect(start * PAGE, (page - start) * PAGE, platform::Access::None);
                mapping.stale[start..page].fill(true);
            }
            let start = page;
            while page < end && kept(page) {
                page += 1;
            }
            if page > start {
                mapping.shadow.protect(start * PAGE, (page - start) * PAGE, platform::Access::ReadWrite);
            }
        }
    }
    mapping.fresh = runs(&written);
    mapping.window = FIRST_WINDOW;
    mapping.next = usize::MAX;
}

/// A touch at `address`: if it is a stale page, fetches the stale pages around it and makes
/// them readable. False when the address isn't in a stale page of a mapping.
fn touch(address: usize) -> bool {
    let started = Instant::now();
    let handled = touched(address);
    TOUCH_NANOS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
    TOUCHES.fetch_add(1, Ordering::Relaxed);
    handled
}

fn touched(address: usize) -> bool {
    let mut guard = MAPPINGS.lock().unwrap_or_else(|p| p.into_inner());
    loop {
        let Some(mappings) = guard.as_mut() else { return false };
        let Some((&memory, mapping)) = mappings.iter_mut().find(|(_, m)| {
            let base = m.shadow.pointer as usize;
            address >= base && address < base + m.size
        }) else {
            return false;
        };
        let page = (address - mapping.shadow.pointer as usize) / PAGE;
        // What's come ahead goes in now, which may be all this needed
        let ready = {
            let (stale, fetching) = (&mapping.stale, &mapping.fetching);
            PUSHED_PAGES.lock().unwrap_or_else(|p| p.into_inner()).take(memory, |page| stale.get(page).copied().unwrap_or(false) && !in_runs(fetching, page))
        };
        for (at, bytes) in ready {
            let start = at * PAGE;
            mapping.shadow.protect(start, PAGE, platform::Access::ReadWrite);
            let end = (start + bytes.len()).min(mapping.size);
            mapping.bytes()[start..end].copy_from_slice(&bytes[..end - start]);
            mapping.settle(start, end - start);
            mapping.stale[at] = false;
            mapping.fresh.push((at, at + 1));
        }
        if !mapping.stale[page] {
            // Another thread fetched it first
            return true;
        }
        // Another thread is fetching it now
        if in_runs(&mapping.fetching, page) {
            guard = FETCHED.wait(guard).unwrap_or_else(|p| p.into_inner());
            continue;
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
        // Where nothing could have changed, a large block goes at once, as it's what the program is using
        let (first, window) = if platform::WRITE_WATCH && writers(|writers| writers.quiet(memory)) {
            (page - page % QUIET_WINDOW, QUIET_WINDOW)
        } else {
            (first, mapping.window)
        };
        let end = (first + window).min(mapping.pages());
        let wanted = runs(&(first..end).filter(|&p| mapping.stale[p] && !in_runs(&mapping.fetching, p)).collect::<Vec<_>>());
        // Where nothing can have changed there's no need to ask, so ask only if some of it could have
        let asking = pages_gpu_can_write(memory, &wanted);
        let quiet: Vec<(usize, usize)> = {
            let asked: HashSet<usize> = asking.iter().flat_map(|&(start, end)| start..end).collect();
            self::runs(&wanted.iter().flat_map(|&(start, end)| start..end).filter(|page| !asked.contains(page)).collect::<Vec<_>>())
        };
        for &(start, end) in &quiet {
            mapping.shadow.protect(start * PAGE, (end - start) * PAGE, platform::Access::ReadWrite);
            mapping.stale[start..end].fill(false);
            mapping.fresh.push((start, end));
        }
        let wanted = asking;
        // Nothing here could have changed where the program touched. The rest stays stale until it's touched:
        // its copy may not have run, and the program knows that, as it hasn't looked for it yet.
        if !in_runs(&wanted, page) {
            mapping.next = end;
            return true;
        }
        let Some(connection) = crate::connection::get() else { return false };
        // The lock isn't held across the trip: everything else that touches memory or finishes a wait
        // needs it. Whoever wants these pages meanwhile waits for them.
        mapping.fetching.extend_from_slice(&wanted);
        let marked = mapping.marked;
        drop(guard);
        let answer = ask_for_pages(&connection, memory, &wanted);
        guard = MAPPINGS.lock().unwrap_or_else(|p| p.into_inner());
        let mapping = guard.as_mut().and_then(|mappings| mappings.get_mut(&memory));
        if let Some(mapping) = mapping {
            mapping.fetching.retain(|run| !wanted.contains(run));
            if let Some(answer) = &answer {
                put_fetched(mapping, memory, &wanted, answer, marked);
            }
            mapping.next = end;
        }
        FETCHED.notify_all();
        return answer.is_some();
    }
}

fn in_runs(runs: &[(usize, usize)], page: usize) -> bool {
    runs.iter().any(|&(start, end)| start <= page && page < end)
}

/// Asks the lender for what's changed in these pages
fn ask_for_pages(connection: &Connection, memory: u64, wanted: &[(usize, usize)]) -> Option<crate::connection::Answer> {
    let mut w = Writer::new();
    w.u64(memory).u32(wanted.len() as u32);
    for &(start, end) in wanted {
        w.u64((start * PAGE) as u64).u64(((end - start) * PAGE) as u64);
    }
    // What's finished by now is what this will bring back, and it won't change again
    let through = writers(|writers| writers.once.finished_through());
    ASKS.fetch_add(1, Ordering::Relaxed);
    let answer = connection.ask(nectar_vk_wire::messages::FETCH, w.as_slice()).ok()?;
    writers(|writers| {
        for &(start, end) in wanted {
            writers.once.fetched(memory, (start * PAGE) as u64, (end * PAGE) as u64, through);
        }
    });
    Some(answer)
}

/// Puts in what the lender answered about these pages. `marked` is how many times pages had been marked
/// stale when it was asked: if it's more now, some thread has since heard of work finishing that this
/// may have been read before, so they go in and stay stale, to be asked about again when touched.
fn put_fetched(mapping: &mut Mapping, memory: u64, wanted: &[(usize, usize)], answer: &crate::connection::Answer, marked: u64) {
    for &(start, end) in wanted {
        mapping.shadow.protect(start * PAGE, (end - start) * PAGE, platform::Access::ReadWrite);
    }
    let mut r = answer.outputs();
    let generation = r.u64().unwrap_or(0);
    for &(start, end) in wanted {
        let older = PUSHED_PAGES.lock().unwrap_or_else(|p| p.into_inner()).older(memory, start, end, generation);
        for (at, bytes) in older {
            let start = at * PAGE;
            let end = (start + bytes.len()).min(mapping.size);
            mapping.bytes()[start..end].copy_from_slice(&bytes[..end - start]);
            mapping.settle(start, end - start);
        }
    }
    let count = r.u32().unwrap_or(0);
    for _ in 0..count {
        let (Ok(_), Ok(offset), Ok(bytes)) = (r.u64(), r.u64(), r.bytes()) else { break };
        let start = offset as usize;
        let end = (start + bytes.len()).min(mapping.size);
        mapping.bytes()[start..end].copy_from_slice(&bytes[..end - start]);
        mapping.settle(start, end - start);
    }
    for &(start, end) in wanted {
        if mapping.marked == marked {
            mapping.stale[start..end].fill(false);
            mapping.fresh.push((start, end));
        } else {
            mapping.shadow.protect(start * PAGE, (end - start) * PAGE, platform::Access::None);
        }
    }
    if crate::logging() {
        let asked: usize = wanted.iter().map(|(s, e)| e - s).sum();
        crate::log(&format!("fetch: memory {memory}, {asked} pages asked, {} bytes came", answer.body.len() - answer.at));
    }
}

/// Maps memory only the lender reads (a swapchain's frames), so that nothing is kept in step
#[cfg_attr(not(any(windows, target_os = "linux")), allow(dead_code))]
pub fn map_readback(connection: &Connection, memory: u64) -> Result<(), i32> {
    map(connection, memory)?;
    let mut mappings = MAPPINGS.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(mapping) = mappings.as_mut().and_then(|mappings| mappings.get_mut(&memory)) {
        mapping.readback = true;
        mapping.shadow.protect(0, mapping.size, platform::Access::ReadWrite);
        mapping.stale.fill(false);
        mapping.fresh.clear();
    }
    Ok(())
}

/// Ends the mapping of freed memory
pub fn forget(connection: &Connection, memory: u64) {
    memory_forgotten(memory);
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
    use std::ops::Range;
    use std::sync::Once;

    use windows_sys::Win32::System::Diagnostics::Debug::{AddVectoredExceptionHandler, EXCEPTION_POINTERS};
    use windows_sys::Win32::System::Memory::{
        GetWriteWatch, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE, ResetWriteWatch,
        VirtualAlloc, VirtualFree, VirtualProtect,
    };

    use super::PAGE;

    pub const WRITE_WATCH: bool = true;

    /// What the program may do with a page
    #[derive(Clone, Copy)]
    pub enum Access {
        None,
        Read,
        ReadWrite,
    }

    pub fn page_size() -> usize {
        PAGE
    }

    const MEM_WRITE_WATCH: u32 = 0x0020_0000;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
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

        pub fn protect(&mut self, start: usize, length: usize, access: Access) {
            let protection = match access {
                Access::None => PAGE_NOACCESS,
                Access::Read => PAGE_READONLY,
                Access::ReadWrite => PAGE_READWRITE,
            };
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

#[cfg(unix)]
mod platform {
    use std::ops::Range;
    use std::sync::Once;
    use std::sync::atomic::{AtomicUsize, Ordering};

    pub const WRITE_WATCH: bool = false;

    /// What the program may do with a page
    #[derive(Clone, Copy)]
    pub enum Access {
        None,
        Read,
        ReadWrite,
    }

    pub fn page_size() -> usize {
        static SIZE: AtomicUsize = AtomicUsize::new(0);
        match SIZE.load(Ordering::Relaxed) {
            0 => {
                let size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) }.max(1) as usize;
                SIZE.store(size, Ordering::Relaxed);
                size
            }
            size => size,
        }
    }

    /// Whole pages of the system's, so they can be protected
    pub struct Shadow {
        pub pointer: *mut u8,
        size: usize,
    }

    // The shadow is only reached through MAPPINGS' lock, or by the program as its mapped memory
    unsafe impl Send for Shadow {}

    impl Shadow {
        pub fn new(size: usize) -> Option<Shadow> {
            let size = size.max(1).div_ceil(page_size()) * page_size();
            let pointer = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    size,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANON,
                    -1,
                    0,
                )
            };
            (pointer != libc::MAP_FAILED).then_some(Shadow { pointer: pointer.cast(), size })
        }

        pub fn written(&mut self, _pages: Range<usize>, _reset: bool, _written: &mut Vec<usize>) {}

        pub fn reset(&mut self, _start: usize, _length: usize) {}

        pub fn protect(&mut self, start: usize, length: usize, access: Access) {
            let first = start / page_size() * page_size();
            let end = (start + length).div_ceil(page_size()) * page_size();
            let protection = match access {
                Access::None => libc::PROT_NONE,
                Access::Read => libc::PROT_READ,
                Access::ReadWrite => libc::PROT_READ | libc::PROT_WRITE,
            };
            unsafe { libc::mprotect(self.pointer.add(first).cast(), end.min(self.size) - first, protection) };
        }
    }

    impl Drop for Shadow {
        fn drop(&mut self) {
            unsafe { libc::munmap(self.pointer.cast(), self.size) };
        }
    }

    /// What handled these signals before: they get whatever isn't a touch of a stale page
    static mut PREVIOUS_SEGV: Option<libc::sigaction> = None;
    static mut PREVIOUS_BUS: Option<libc::sigaction> = None;

    /// Catches the program's touches of stale pages. On the program's own stack, not a signal
    /// stack: a touch talks to the lender.
    pub fn catch_touches() {
        static ONCE: Once = Once::new();
        ONCE.call_once(|| unsafe {
            let mut action: libc::sigaction = std::mem::zeroed();
            action.sa_sigaction = on_fault as *const () as usize;
            action.sa_flags = libc::SA_SIGINFO | libc::SA_RESTART;
            libc::sigemptyset(&mut action.sa_mask);
            let mut previous: libc::sigaction = std::mem::zeroed();
            if libc::sigaction(libc::SIGSEGV, &action, &mut previous) == 0 {
                PREVIOUS_SEGV = Some(previous);
            }
            // What macOS raises for such a page
            if libc::sigaction(libc::SIGBUS, &action, &mut previous) == 0 {
                PREVIOUS_BUS = Some(previous);
            }
        });
    }

    unsafe extern "C" fn on_fault(signal: i32, info: *mut libc::siginfo_t, context: *mut libc::c_void) {
        unsafe {
            if super::touch((*info).si_addr() as usize) {
                return;
            }
            let previous = if signal == libc::SIGSEGV { PREVIOUS_SEGV } else { PREVIOUS_BUS };
            match previous {
                Some(previous) if previous.sa_sigaction == libc::SIG_IGN => {}
                Some(previous) if previous.sa_sigaction != libc::SIG_DFL => {
                    if previous.sa_flags & libc::SA_SIGINFO != 0 {
                        let handler: unsafe extern "C" fn(i32, *mut libc::siginfo_t, *mut libc::c_void) =
                            std::mem::transmute(previous.sa_sigaction);
                        handler(signal, info, context);
                    } else {
                        let handler: unsafe extern "C" fn(i32) = std::mem::transmute(previous.sa_sigaction);
                        handler(signal);
                    }
                }
                // Back to the default, so returning runs the same instruction and ends the
                // program as it would have
                _ => {
                    let mut action: libc::sigaction = std::mem::zeroed();
                    action.sa_sigaction = libc::SIG_DFL;
                    libc::sigaction(signal, &action, std::ptr::null_mut());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn region(offset: usize, size: usize) -> Region {
        Region { memory: 1, offset, size, place: (2, 3), mip: None, block: (1, 1, 1), extent: (0, 0) }
    }

    #[test]
    fn pieces_cover_every_byte_once() {
        let at = |piece: &Piece| match piece {
            Piece::Copied(region) => (region.offset, region.offset + region.size),
            Piece::Rest(from, to) => (*from, *to),
        };
        let cases: [(usize, usize, Vec<Region>); 6] = [
            (0, 8192, vec![]),
            (0, 8192, vec![region(0, 8192)]),
            (4096, 20480, vec![region(5000, 4096), region(9096, 5000), region(16000, 4096)]),
            // Reaching outside the run, before and after
            (4096, 8192, vec![region(0, 5000), region(8000, 4096)]),
            // Overlapping
            (0, 20480, vec![region(0, 8192), region(4096, 8192), region(12288, 4096)]),
            // Nothing in this run
            (40960, 45056, vec![region(0, 4096)]),
        ];
        for (start, end, regions) in cases {
            let pieces = pieces(start, end, &regions);
            let mut next = start;
            for piece in &pieces {
                let (from, to) = at(piece);
                assert_eq!(from, next, "{pieces:?}");
                assert!(to > from);
                next = to;
            }
            assert_eq!(next, end);
        }
        assert_eq!(
            pieces(4096, 20480, &[region(5000, 4096)]),
            vec![Piece::Rest(4096, 5000), Piece::Copied(region(5000, 4096)), Piece::Rest(9096, 20480)]
        );
        assert_eq!(pieces(0, 8192, &[region(0, 9000)]), vec![Piece::Rest(0, 8192)]);
    }

    #[test]
    fn long_runs_are_those_a_submit_would_send_unexplained() {
        let run = |first: usize, length: usize, written: u64| (first..first + length).map(move |page| (page, written));
        let long = LONG_RUN;
        // Written before the command buffers finished, or partly: it's growing
        let mut unsent: BTreeMap<usize, u64> = run(0, long, 3).collect();
        assert_eq!(long_runs(&unsent, 5, &[]), vec![(0, long)]);
        unsent.extend(run(long, 10, u64::MAX));
        assert_eq!(long_runs(&unsent, 5, &[]), vec![(0, long + 10)]);
        // The submit wouldn't send any of it
        assert_eq!(long_runs(&unsent, 2, &[]), Vec::new());
        // Explained by a copy
        assert_eq!(long_runs(&unsent, 5, &[(0, (long + 10) * PAGE)]), Vec::new());
        // Too short, once a copy explains most of it
        assert_eq!(long_runs(&unsent, 5, &[(0, long * PAGE)]), Vec::new());
        // Two apart
        let unsent: BTreeMap<usize, u64> = run(0, long, 1).chain(run(long + 5, long, 1)).collect();
        assert_eq!(long_runs(&unsent, 5, &[]), vec![(0, long), (long + 5, long)]);
    }

    #[test]
    fn ranges_join_where_they_touch() {
        let mut ranges = BTreeMap::new();
        add_range(&mut ranges, 100, 200);
        add_range(&mut ranges, 300, 400);
        assert_eq!(ranges, BTreeMap::from([(100, 200), (300, 400)]));
        // Inside, overlapping and touching
        add_range(&mut ranges, 120, 150);
        assert_eq!(ranges, BTreeMap::from([(100, 200), (300, 400)]));
        add_range(&mut ranges, 150, 250);
        assert_eq!(ranges, BTreeMap::from([(100, 250), (300, 400)]));
        add_range(&mut ranges, 250, 300);
        assert_eq!(ranges, BTreeMap::from([(100, 400)]));
        // Over several
        add_range(&mut ranges, 500, 600);
        add_range(&mut ranges, 700, 800);
        add_range(&mut ranges, 50, 750);
        assert_eq!(ranges, BTreeMap::from([(50, 800)]));
        assert!(range_touches(&ranges, 0, 51));
        assert!(range_touches(&ranges, 799, 900));
        assert!(!range_touches(&ranges, 0, 50));
        assert!(!range_touches(&ranges, 800, 900));
    }

    #[test]
    fn what_is_written_once_stops_counting_when_finished_and_fetched() {
        // Two submits on a queue, one on another
        let mut once = Once { submits: 3, running: BTreeMap::from([(1, 10), (2, 10), (3, 20)]), ..Once::default() };
        once.add(7, 100, 200, 1);
        once.add(7, 150, 300, 2);
        // Still running, it doesn't count: the program can't be expecting it
        assert!(!once.touches(7, 100, 300));
        assert!(!once.any_finished(7));
        // Overlapping, they're kept as one, with the first and the last
        assert_eq!(once.ranges[&7], BTreeMap::from([(100, (300, 2, 1))]));
        // Fetched before it finished, it still counts afterwards
        assert_eq!(once.finished_through(), 0);
        once.fetched(7, 0, 400, once.finished_through());
        assert!(!once.touches(7, 100, 200));
        // The first of two that overlap finishing is enough: what it wrote may be being read
        once.running.remove(&1);
        assert!(once.touches(7, 299, 400));
        assert!(!once.touches(7, 300, 400));
        assert!(!once.touches(8, 0, 400));
        assert!(once.any_finished(7));
        // A later submit's finish covers earlier ones on its queue
        once.finish(2);
        assert_eq!(once.finished_through(), 2);
        // A piece is fetched: the rest still counts
        once.fetched(7, 150, 200, 2);
        assert_eq!(once.ranges[&7], BTreeMap::from([(100, (150, 2, 1)), (200, (300, 2, 1))]));
        // Another queue's unfinished work holds up later ones, but not these
        once.add(7, 400, 500, 3);
        once.fetched(7, 0, 600, once.finished_through());
        assert_eq!(once.ranges[&7], BTreeMap::from([(400, (500, 3, 3))]));
        once.finish(3);
        assert_eq!(once.finished_through(), 3);
    }

    #[test]
    fn a_copy_stops_being_asked_about_once_finished_and_fetched() {
        let (buffer, memory, commands, queue, fence) = (66701u64, 66702u64, 66703u64, 66704u64, 66705u64);
        buffer_created(buffer, 1 << 20, 0x3, false);
        buffer_bound(buffer, memory, 0);
        recording_began(commands);
        writes_once(commands, buffer, PAGE as u64, 16);
        // Recorded, it writes nothing until it's submitted
        assert_eq!(pages_gpu_can_write(memory, &[(0, 8)]), Vec::new());
        work_submitted(queue, &[commands], fence, &[], false);
        // Running, it doesn't count
        assert_eq!(pages_gpu_can_write(memory, &[(0, 8)]), Vec::new());
        fence_signalled(fence);
        assert_eq!(pages_gpu_can_write(memory, &[(0, 8)]), vec![(1, 2)]);
        let through = writers(|writers| writers.once.running.keys().next().map_or(writers.once.submits, |first| first - 1));
        assert!(through >= 1);
        writers(|writers| writers.once.fetched(memory, PAGE as u64, 2 * PAGE as u64, through));
        assert_eq!(pages_gpu_can_write(memory, &[(0, 8)]), Vec::new());
        // Submitted again, it writes again
        work_submitted(queue, &[commands], fence, &[], false);
        assert_eq!(pages_gpu_can_write(memory, &[(0, 8)]), Vec::new());
        fence_signalled(fence);
        assert_eq!(pages_gpu_can_write(memory, &[(0, 8)]), vec![(1, 2)]);
    }

    #[test]
    fn work_that_waited_finishes_what_came_before_it_on_every_queue() {
        let mut once = Once { submits: 4, running: BTreeMap::from([(1, 10), (2, 20), (3, 30), (4, 10)]), ..Once::default() };
        once.add(7, 100, 200, 1);
        once.add(7, 300, 400, 2);
        once.add(7, 500, 600, 4);
        once.waiters.insert(3);
        // Another queue's work could be what it waited for, and asking when it isn't only costs a trip
        once.finish(3);
        assert!(once.touches(7, 100, 200));
        assert!(once.touches(7, 300, 400));
        assert!(!once.touches(7, 500, 600));
        assert_eq!(once.running, BTreeMap::from([(4, 10)]));
    }

    #[test]
    fn memory_is_quiet_until_something_could_have_changed_it() {
        let (buffer, memory, commands, queue, fence) = (66801u64, 66802u64, 66803u64, 66804u64, 66805u64);
        let quiet = || writers(|writers| writers.quiet(memory));
        buffer_created(buffer, 1 << 20, 0x3, false);
        buffer_bound(buffer, memory, 0);
        assert!(quiet());
        recording_began(commands);
        writes_once(commands, buffer, 0, 16);
        work_submitted(queue, &[commands], fence, &[], false);
        assert!(quiet(), "still running");
        fence_signalled(fence);
        assert!(!quiet());
        writers(|writers| writers.once.fetched(memory, 0, PAGE as u64, 1 << 40));
        assert!(quiet());
        written_range(buffer, 0, 16);
        assert!(!quiet());
        memory_forgotten(memory);
    }

    #[test]
    fn only_pages_the_gpu_can_write_are_asked_about() {
        let (buffer, memory) = (66601u64, 66602u64);
        buffer_created(buffer, 1 << 20, 0x3, false);
        buffer_bound(buffer, memory, PAGE as u64);
        // Nothing yet
        assert_eq!(pages_gpu_can_write(memory, &[(0, 100)]), Vec::new());
        // A copy into a few bytes of the second page of the buffer, which is the third of the memory
        written_range(buffer, PAGE as u64 + 10, 20);
        assert_eq!(pages_gpu_can_write(memory, &[(0, 100)]), vec![(2, 3)]);
        assert_eq!(pages_gpu_can_write(memory, &[(3, 100)]), Vec::new());
        // Across a page boundary
        written_range(buffer, 5 * PAGE as u64 - 1, 2);
        assert_eq!(pages_gpu_can_write(memory, &[(0, 100)]), vec![(2, 3), (5, 7)]);
        // All of it
        written(buffer);
        assert_eq!(pages_gpu_can_write(memory, &[(0, 300)]), vec![(1, 257)]);
    }

    #[test]
    fn pages_noticed_between_command_buffers_are_due_with_the_next() {
        // Noticed by a submit or while waiting for a copy, so after anything had finished recording
        let mut unsent = BTreeMap::from([(1, 3), (2, u64::MAX)]);
        assert_eq!(due_pages(&mut unsent.clone(), 3, &[], &[]), vec![1]);
        // What finishes next could read them, as well as what's been written since
        finished_recording(&mut unsent, vec![2, 7], 4);
        assert_eq!(unsent, BTreeMap::from([(1, 3), (2, 4), (7, 4)]));
        assert_eq!(due_pages(&mut unsent, 4, &[], &[]), vec![1, 2, 7]);
    }

    #[test]
    fn a_submit_sends_what_it_could_read() {
        let page = |number: usize| number * PAGE;
        let unsent = || BTreeMap::from([(0, 1), (1, 2), (2, 5), (3, 5), (4, u64::MAX), (5, u64::MAX)]);
        // Written before its command buffers finished, and no more
        assert_eq!(due_pages(&mut unsent(), 2, &[], &[]), vec![0, 1]);
        assert_eq!(due_pages(&mut unsent(), 5, &[], &[]), vec![0, 1, 2, 3]);
        // Nothing to do with a submit, everything goes
        assert_eq!(due_pages(&mut unsent(), u64::MAX, &[], &[]), vec![0, 1, 2, 3, 4, 5]);
        // What it copies goes even if it was written after, as does a page only partly copied
        assert_eq!(due_pages(&mut unsent(), 1, &[(page(4) + 100, 200)], &[]), vec![0, 4]);
        // What a later submit copies waits for it, unless a page holds something else too
        assert_eq!(due_pages(&mut unsent(), 5, &[], &[(page(1), page(2))]), vec![0, 3]);
        assert_eq!(due_pages(&mut unsent(), 5, &[], &[(page(1) + 1, page(2))]), vec![0, 1, 3]);
        // What goes stops waiting
        let mut left = unsent();
        due_pages(&mut left, 2, &[], &[]);
        assert_eq!(left.keys().copied().collect::<Vec<_>>(), vec![2, 3, 4, 5]);
    }

    #[test]
    fn what_is_rewritten_is_kept_only_when_it_comes_again() {
        let mut keeper = Keeper::new(HashSet::new());
        let (a, b, c) = ([1u8; 16], [2u8; 16], [3u8; 16]);
        let now = Instant::now();
        let copied = |to: u64| Sending::Copied((to, 0));
        // The first bytes for a place
        assert!(keeper.worth_keeping(a, 4096, copied(10), now));
        assert!(keeper.worth_keeping(a, 4096, copied(11), now));
        // New bytes rewriting it, seen for the first time
        assert!(!keeper.worth_keeping(b, 4096, copied(10), now));
        assert!(!keeper.worth_keeping(c, 4096, copied(10), now));
        // Those same bytes again
        assert!(keeper.worth_keeping(b, 4096, copied(10), now));
    }

    #[test]
    fn what_nothing_explains_is_kept_while_the_allowance_lasts() {
        let mut keeper = Keeper::new(HashSet::new());
        let now = Instant::now();
        let piece = |seed: u8| [seed; 16];
        let long = Sending::Unknown(LOAD_LEAST);
        // A short run, as a frame's constants are, only when it comes again
        assert!(!keeper.worth_keeping(piece(1), 4096, Sending::Unknown(LOAD_LEAST - 1), now));
        assert!(keeper.worth_keeping(piece(1), 4096, Sending::Unknown(LOAD_LEAST - 1), now));
        // A long one at first sight, until that's used up
        let most = FIRST_SIGHT_MOST as usize;
        assert!(keeper.worth_keeping(piece(2), most - 1000, long, now));
        assert!(keeper.worth_keeping(piece(3), 1000, long, now));
        assert!(!keeper.worth_keeping(piece(4), 1000, long, now));
        // Seen before it's still worth it, and costs nothing
        assert!(keeper.worth_keeping(piece(4), 1000, long, now));
        // More comes with time
        assert!(keeper.worth_keeping(piece(5), 1000, long, now + std::time::Duration::from_secs(1)));
    }

    #[test]
    fn the_same_bytes_are_cut_the_same_way_wherever_they_sit() {
        let mut noise = 0x2545_f491_4f6c_dd1du64;
        let mut bytes = vec![0u8; 16 << 20];
        for word in bytes.as_chunks_mut::<8>().0 {
            noise ^= noise << 13;
            noise ^= noise >> 7;
            noise ^= noise << 17;
            *word = noise.to_le_bytes();
        }
        let cuts = |bytes: &[u8], offset: usize| {
            let (mut at, mut ends) = (0, Vec::new());
            while at < bytes.len() {
                let cut = next_cut(&bytes[at..], offset + at);
                assert!(cut > 0 && cut <= CUT_MOST);
                at += cut;
                ends.push(offset + at);
            }
            ends
        };
        let whole = cuts(&bytes, 0);
        let last = whole.len() - 1;
        for (cut, _) in whole[..last].iter().zip(0..) {
            assert!(*cut >= CUT_LEAST);
        }
        // About 96 KiB on average
        let average = bytes.len() / whole.len();
        assert!((64 << 10..160 << 10).contains(&average), "{average}");

        // The tail of the same bytes, at another place, after other bytes: it settles on the same cuts
        for (taken, moved) in [(1 << 20, 4096usize), (3 << 20, 64 << 10), (5 << 20, 1088)] {
            let tail = cuts(&bytes[taken..], moved);
            let tail: HashSet<usize> = tail[..tail.len() - 1].iter().map(|end| end - moved + taken).collect();
            let whole: HashSet<usize> = whole[..last].iter().copied().filter(|end| *end > taken).collect();
            let shared = tail.intersection(&whole).count();
            assert!(shared * 100 >= whole.len() * 95, "{shared} of {}", whole.len());
        }

        // The same words all over, which never give a cut, stop at the longest
        assert_eq!(next_cut(&vec![7u8; 1 << 20], 0), CUT_MOST);
        // Under the least it's all one piece
        assert_eq!(next_cut(&bytes[..CUT_LEAST], 0), CUT_LEAST);
    }

    #[test]
    fn runs_join_neighbours() {
        assert_eq!(runs(&[1, 2, 3, 7, 9, 10]), [(1, 4), (7, 8), (9, 11)]);
        assert_eq!(runs(&[]), []);
    }

    fn mapping(pages: usize) -> Mapping {
        let size = pages * PAGE;
        let shadow = platform::Shadow::new(size).unwrap();
        let bytes = unsafe { std::slice::from_raw_parts(shadow.pointer, size) };
        Mapping {
            hashes: if platform::WRITE_WATCH { Vec::new() } else { hashes(bytes) },
            shadow,
            size,
            stale: vec![false; pages],
            fresh: vec![(0, pages)],
            window: FIRST_WINDOW,
            next: usize::MAX,
            readback: false,
            fetching: Vec::new(),
            marked: 0,
            marked_at: Instant::now(),
            unsent: BTreeMap::new(),
        }
    }

    #[test]
    fn written_pages_are_found_and_taken_as_in_step() {
        let mut m = mapping(8);
        m.settle(0, 8 * PAGE);
        assert_eq!(m.written(true), Vec::<usize>::new());
        m.bytes()[2 * PAGE + 5] = 1;
        m.bytes()[6 * PAGE] = 1;
        assert_eq!(m.written(false), [2, 6]);
        assert_eq!(m.written(true), [2, 6]);
        assert_eq!(m.written(true), Vec::<usize>::new());
    }

    fn gpu_can_write(memory: u64) -> bool {
        !pages_gpu_can_write(memory, &[(0, 16)]).is_empty()
    }

    #[test]
    fn pushed_pages_go_in_when_stale_and_in_order() {
        let mut pushed = PushedPages::default();
        let page = |value: u8| vec![value; PAGE];
        let memory = 1;
        pushed.arrived(memory, 5, PAGE, &[page(1), page(2)].concat());
        // Not stale yet: the program hasn't heard of this work, so it stays
        assert!(pushed.take(memory, |_| false).is_empty());
        // Newer news of the same page replaces it
        pushed.arrived(memory, 6, 2 * PAGE, &page(3));
        let taken = pushed.take(memory, |at| at == 2);
        assert_eq!(taken, vec![(2, page(3))]);
        assert_eq!(pushed.take(memory, |_| true), vec![(1, page(1))]);

        // A fetch read later than what came ahead, and earlier than what came after that. The first has
        // to go in all the same, as the fetch only brings what's changed since it.
        pushed.arrived(memory, 7, 0, &[page(4), page(5)].concat());
        pushed.arrived(memory, 9, 2 * PAGE, &page(6));
        assert_eq!(pushed.older(memory, 1, 3, 8), vec![(1, page(5))]);
        assert_eq!(pushed.take(memory, |_| true), vec![(0, page(4)), (2, page(6))]);

        // The program's own write is newer than the GPU's before it
        pushed.arrived(memory, 10, 0, &page(7));
        pushed.overwritten(memory, &[0]);
        assert!(pushed.take(memory, |_| true).is_empty());

        // A short last page of a memory stays short
        pushed.arrived(memory, 11, 3 * PAGE, &[8u8; 100]);
        assert_eq!(pushed.take(memory, |_| true)[0].1.len(), 100);

        pushed.arrived(memory, 12, 0, &page(9));
        pushed.forget(memory);
        assert!(pushed.take(memory, |_| true).is_empty());
    }

    #[test]
    fn direct3d_12s_upload_heaps_are_never_written() {
        const VISIBLE: u32 = nectar_vk_sys::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        const CACHED: u32 = nectar_vk_sys::VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
        let (upload, readback, before) = (9101, 9102, 9103);
        memory_types_told(&[VISIBLE, VISIBLE | CACHED]);
        // Until it's known to be Direct3D 12, nothing is ruled out
        if !D3D12.load(Ordering::Relaxed) {
            memory_allocated(before, 0);
            image_bound(before);
            assert!(gpu_can_write(before));
        }
        engine_named(b"vkd3d-proton");
        engine_named(b"DXVK");
        assert!(!holding_writes(), "DXVK's DXGI beside vkd3d-proton says nothing about the writes");
        memory_allocated(upload, 0);
        memory_allocated(readback, 1);
        for memory in [upload, readback] {
            image_bound(memory);
        }
        assert!(!gpu_can_write(upload));
        // Bound to an image it isn't a read-back heap
        assert!(gpu_can_write(readback));
        let copied = 9104;
        memory_allocated(copied, 1);
        writers(|writers| writers.memory.insert(copied));
        assert!(!gpu_can_write(copied), "only copies write a read-back heap, whatever else is seen");
        writers(|writers| writers.once.add(copied, 0, 16, 1));
        assert!(gpu_can_write(copied));
        memory_forgotten(copied);
        memory_forgotten(upload);
        image_bound(upload);
        assert!(gpu_can_write(upload));
        D3D12.store(false, Ordering::Relaxed);
    }

    #[test]
    fn only_what_the_gpu_can_write_counts() {
        // Ids well away from anything else in these tests. Tests share this state, so none of them
        // says that addresses write except the one that checks it, on its own ids, by asking
        // directly.
        let (upload, copied, storage, chained, image, view_memory) = (9001, 9002, 9003, 9004, 9005, 9006);
        const EVERY_USAGE_AS_DXVK_ASKS: u32 = 0x219ff;
        const UNIFORM_AND_COPIES: u32 = 0x10 | 0x1 | 0x2;

        // One huge buffer with every usage is written by nothing until something is seen
        buffer_created(1, 4096, EVERY_USAGE_AS_DXVK_ASKS, false);
        buffer_bound(1, upload, 0);
        assert!(!gpu_can_write(upload));

        buffer_created(2, 4096, UNIFORM_AND_COPIES, false);
        buffer_bound(2, copied, 0);
        assert!(!gpu_can_write(copied));
        written(2);
        assert!(gpu_can_write(copied));

        buffer_created(3, 4096, 0x20, false);
        buffer_bound(3, storage, 0);
        assert!(!gpu_can_write(storage));
        written(3);
        assert!(gpu_can_write(storage));

        // Chained create info, a usage not watched, and a buffer never seen created all count
        buffer_created(4, 4096, UNIFORM_AND_COPIES, true);
        buffer_bound(4, chained, 0);
        assert!(gpu_can_write(chained));
        memory_forgotten(chained);
        assert!(!gpu_can_write(chained));
        buffer_bound(5, chained, 0);
        assert!(gpu_can_write(chained));
        memory_forgotten(chained);
        buffer_created(6, 4096, 0x400_0000, false);
        buffer_bound(6, chained, 0);
        assert!(gpu_can_write(chained));

        image_bound(image);
        assert!(gpu_can_write(image));

        buffer_created(7, 4096, 0x8, false);
        buffer_bound(7, view_memory, 0);
        view_created(70, 7, 0, 4096);
        assert!(!gpu_can_write(view_memory));
        view_written(7, 70);
        // It's named, and counts as soon as a shader writes through that binding
        assert!(!gpu_can_write(view_memory));
        storage_writes(&StorageWrites { bindings: vec![6], unknown: false });
        assert!(!gpu_can_write(view_memory));
        storage_writes(&StorageWrites { bindings: vec![7], unknown: false });
        assert!(gpu_can_write(view_memory));
    }

    #[test]
    fn shaders_are_read_for_writes_to_storage() {
        const HEADER: [u32; 5] = [0x0723_0203, 0x10000, 0, 30, 0];
        let module = |body: &[u32]| [&HEADER[..], body].concat();
        // %2 a pointer type into storage buffers, %3 a variable of it, %4 an access chain into it
        let pointer_type = [(4 << 16) | 32, 2, 12, 1];
        let variable = [(4 << 16) | 59, 2, 3, 12];
        let chain = [(5 << 16) | 65, 2, 4, 3, 7];
        // Binding 5 on the variable
        let binding = [(4 << 16) | 71, 3, 33, 5];
        let with = |body: &[&[u32]]| module(&[&binding[..], &pointer_type, &variable, &chain, &body.concat()].concat());
        let none = StorageWrites::default();
        // Reading writes nothing
        assert_eq!(storage_writes_of(&with(&[&[(4 << 16) | 61, 1, 5, 4]])), none);
        // Storing through the chain writes its binding
        let five = StorageWrites { bindings: vec![5], unknown: false };
        assert_eq!(storage_writes_of(&with(&[&[(3 << 16) | 62, 4, 9]])), five);
        // A store to somewhere else, such as a local
        assert_eq!(storage_writes_of(&with(&[&[(3 << 16) | 62, 20, 9]])), none);
        // Atomics change it too, but an atomic load doesn't
        assert_eq!(storage_writes_of(&with(&[&[(7 << 16) | 234, 1, 8, 4, 1, 0, 9]])), five);
        assert_eq!(storage_writes_of(&with(&[&[(6 << 16) | 227, 1, 8, 4, 1, 0]])), none);
        // NVIDIA's raw access chain is followed like any other, and writes nothing itself
        let raw = [(8 << 16) | 5398, 2, 50, 3, 7, 7, 7, 2];
        assert_eq!(storage_writes_of(&with(&[&raw, &[(4 << 16) | 61, 1, 5, 50]])), none);
        assert_eq!(storage_writes_of(&with(&[&raw, &[(3 << 16) | 62, 50, 9]])), five);
        // An image is loaded from a variable, and written through what's loaded
        let image_variable = [(4 << 16) | 59, 30, 31, 0];
        let image_binding = [(4 << 16) | 71, 31, 33, 9];
        let load = [(4 << 16) | 61, 32, 33, 31];
        let write = [(4 << 16) | 99, 33, 2, 3];
        assert_eq!(
            storage_writes_of(&module(&[&image_binding[..], &image_variable, &load, &write].concat())),
            StorageWrites { bindings: vec![9], unknown: false }
        );
        // Writing through a pointer into storage that can't be followed counts as anywhere
        let lost = module(&[&pointer_type[..], &[(4 << 16) | 200, 2, 40, 1], &[(3 << 16) | 62, 40, 9]].concat());
        assert_eq!(storage_writes_of(&lost), StorageWrites { bindings: vec![], unknown: true });
        // An image write that can't be followed too
        assert_eq!(storage_writes_of(&module(&[(4 << 16) | 99, 1, 2, 3])), StorageWrites { bindings: vec![], unknown: true });
    }

    #[test]
    fn shaders_are_read_for_writes_through_addresses() {
        const HEADER: [u32; 5] = [0x0723_0203, 0x10000, 0, 30, 0];
        let module = |body: &[u32]| [&HEADER[..], body].concat();
        // %2 is a pointer type into the physical storage buffer class, %3 a load of a pointer of that
        // type, and %4 a storage buffer pointer that's not of the class
        let pointer_type = [(4 << 16) | 32, 2, 5349, 1];
        let other_type = [(4 << 16) | 32, 9, 12, 1];
        let reads = [&pointer_type[..], &[(4 << 16) | 61, 2, 3, 8], &[(4 << 16) | 61, 1, 5, 3]].concat();
        assert!(!writes_through_addresses(&module(&reads)));
        let stores = [&reads[..], &[(3 << 16) | 62, 3, 5]].concat();
        assert!(writes_through_addresses(&module(&stores)));
        let atomic = [&reads[..], &[(7 << 16) | 234, 1, 6, 3, 7, 7, 5]].concat();
        assert!(writes_through_addresses(&module(&atomic)));
        // A store to another class's pointer
        let elsewhere = [&other_type[..], &[(4 << 16) | 61, 9, 4, 8], &[(3 << 16) | 62, 4, 5]].concat();
        assert!(!writes_through_addresses(&module(&elsewhere)));
        assert!(!writes_through_addresses(&[]));
    }

    #[test]
    fn a_new_mapping_holds_nothing_to_send() {
        let mut m = stale_mapping(3 * PAGE + 100).unwrap();
        assert_eq!(m.pages(), 4);
        assert!(m.stale.iter().all(|&stale| stale));
        assert!(m.written(true).is_empty());
        // What a touch does, without the lender: nothing came, so the zeros stand
        m.shadow.protect(0, m.size, platform::Access::ReadWrite);
        m.fresh.push((0, 4));
        assert!(m.written(false).is_empty());
        assert_eq!(m.bytes()[3 * PAGE + 99], 0);
        m.bytes()[3 * PAGE + 99] = 7;
        assert_eq!(m.written(true), [3]);
    }

    #[test]
    fn pages_go_stale_except_those_not_yet_sent() {
        let mut m = mapping(8);
        m.settle(0, 8 * PAGE);
        m.bytes()[3 * PAGE] = 1;
        mark_stale(&mut m);
        // Whole system pages stay with a written one (the system's page is PAGE here on Windows,
        // and on Linux on x86)
        let granule = (platform::page_size() / PAGE).max(1);
        let first = 3 - 3 % granule;
        let kept: Vec<usize> = (0..8).filter(|&page| !m.stale[page]).collect();
        assert_eq!(kept, (first..(first + granule).min(8)).collect::<Vec<_>>());
        assert_eq!(m.fresh, [(first, (first + granule).min(8))]);
        // Its byte is still there, still to send, and it can still be written
        assert_eq!(m.bytes()[3 * PAGE], 1);
        m.bytes()[3 * PAGE] = 2;
        assert_eq!(m.written(true), [3]);
    }
}
