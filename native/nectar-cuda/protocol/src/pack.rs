//! Fast compression for a stream of calls: an LZ whose window spans blocks, so a frame's commands
//! are copied from the frame before's, which are nearly the same. Blocks are at most BLOCK long.
//!
//! A packed block is sequences of: a token (the high 4 bits the literal count, the low 4 bits the
//! match length less MIN_MATCH, 15 in either meaning more follows, in bytes of 255 and one under),
//! the literals, then, unless the block ended, the match's offset as a little-endian base-128 number,
//! 0 meaning the same offset as the last match in the block. Calls change a few values and keep the
//! rest, so the same offset comes up again and again, and it's tried first.
//!
//! The window is a ring, so nothing ever has to slide, which would stop the stream for as long as it took
//! to copy. A block that wouldn't fit before the ring's end goes at its start, and what's above it is
//! still there to copy from, until a later block is written over it. An offset counts back around the
//! ring: a match never runs past the ring's end, nor into the block being written.
//!
//! The window is long as games write a lot. The Witcher 3 writes about 9 MB a frame, with its foliage
//! coming back in other orders, and much of it repeats from further back than a second or so: 64 MiB
//! brought 385 MB of that to 100 MB, and 256 MiB to 64 MB.
//!
//! Blocks are packed several at a time, on several threads, as compressing was the longest wait in a
//! frame of The Witcher 3's. All of a batch's bytes go in the ring first, and it's only read until they're
//! packed. A block can copy from anything before it, in the batch or not, as the other end unpacks in
//! order, but not from a block after it, which the other end won't have. The threads share one table of
//! where things were seen, without locking: an entry that one overwrites is only a match another
//! misses, and what an entry points at is always compared before it's used.
//!
//! Bytes that don't compress go as they are, into both ends' windows all the same, as a later block in
//! the batch may have found them. Unpacking checks its bounds: what comes off the network can only read
//! this stream's own window.
//!
//! On a stream, each block is a u32 of its length times two, plus one if it's as it is, then its
//! bytes (Writer and Reader).

use std::io::{self, Read, Write};
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::pool::Pool;

/// The most one block holds
pub const BLOCK: usize = 1 << 16;
/// How far back a match can reach: the size of the ring, which each end only uses as it's reached
const WINDOW: usize = 256 << 20;
const MIN_MATCH: usize = 4;
/// A match that isn't at the last offset costs more than a few bytes are worth
const FAR_LEAST: usize = 8;
const TABLE_BITS: u32 = 20;
/// The most packed at once: this many blocks go out together when they're waiting
const BATCH: usize = 32;
/// Threads compressing, this one among them. They take from the game's cores, so not all of them.
/// NECTAR_PACK_THREADS says how many.
const MOST_THREADS: usize = 4;
static PACKING: Pool = Pool::new();

fn threads() -> usize {
    static THREADS: OnceLock<usize> = OnceLock::new();
    *THREADS.get_or_init(|| {
        std::env::var("NECTAR_PACK_THREADS").ok().and_then(|value| value.parse().ok()).unwrap_or_else(|| {
            std::thread::available_parallelism().map_or(1, |count| count.get() / 2).clamp(1, MOST_THREADS)
        })
    })
}
/// Bytes tried before a block that hasn't shrunk is given up on
const TRIAL: usize = 4 << 10;
/// How far apart positions inside a match are remembered
const SKIP: usize = 32;

fn bad() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, "a packed block doesn't unpack")
}

/// # Safety
/// `at + 4` must be within `bytes`.
#[inline(always)]
unsafe fn word(bytes: &[u8], at: usize) -> u32 {
    debug_assert!(at + 4 <= bytes.len());
    unsafe { (bytes.as_ptr().add(at) as *const u32).read_unaligned() }
}

/// # Safety
/// `at + 8` must be within `bytes`.
#[inline(always)]
unsafe fn long_word(bytes: &[u8], at: usize) -> u64 {
    debug_assert!(at + 8 <= bytes.len());
    unsafe { (bytes.as_ptr().add(at) as *const u64).read_unaligned() }
}

/// # Safety
/// `at + 8` must be within `bytes`.
#[inline(always)]
unsafe fn slot(bytes: &[u8], at: usize) -> usize {
    let value = unsafe { long_word(bytes, at) };
    (value.wrapping_mul(0x9E37_79B1_85EB_CA87) >> (64 - TABLE_BITS)) as usize
}

/// How far `a` and `b` agree, up to `limit` bytes
#[inline(always)]
fn agree(bytes: &[u8], a: usize, b: usize, limit: usize) -> usize {
    debug_assert!(a.max(b) + limit <= bytes.len());
    let mut length = 0;
    while length + 8 <= limit {
        // Within `limit` of both
        let difference = unsafe { long_word(bytes, a + length) ^ long_word(bytes, b + length) };
        if difference != 0 {
            return length + (difference.trailing_zeros() / 8) as usize;
        }
        length += 8;
    }
    while length < limit && bytes[a + length] == bytes[b + length] {
        length += 1;
    }
    length
}

/// Where a match at `at` with this offset reads from, around a ring of `window` bytes
#[inline(always)]
fn source_of(at: usize, offset: usize, window: usize) -> usize {
    if offset <= at { at - offset } else { at + window - offset }
}

fn long_count(out: &mut Vec<u8>, mut left: usize) {
    while left >= 255 {
        out.push(255);
        left -= 255;
    }
    out.push(left as u8);
}

#[inline(always)]
fn sequence(out: &mut Vec<u8>, literals: &[u8], matched: Option<(usize, u32)>) {
    let long = matched.map_or(0, |(length, _)| length - MIN_MATCH);
    out.push(((literals.len().min(15) as u8) << 4) | long.min(15) as u8);
    if literals.len() >= 15 {
        long_count(out, literals.len() - 15);
    }
    out.extend_from_slice(literals);
    if let Some((_, mut offset)) = matched {
        while offset >= 0x80 {
            out.push(offset as u8 | 0x80);
            offset >>= 7;
        }
        out.push(offset as u8);
        if long >= 15 {
            long_count(out, long - 15);
        }
    }
}

/// One direction's packing end
pub struct Packer {
    history: Vec<u8>,
    at: usize,
    /// The ring has gone round, so what's above `at` is old bytes rather than nothing
    round: bool,
    table: Vec<AtomicU32>,
    /// Each block of a batch, packed, and whether that came out smaller
    coded: Vec<Mutex<(Vec<u8>, bool)>>,
}

impl Default for Packer {
    fn default() -> Self {
        Self::sized(WINDOW)
    }
}

/// The ring and the table, as all the threads see them
struct Shared<'a> {
    history: &'a [u8],
    table: &'a [AtomicU32],
    round: bool,
    /// The end of the batch: what's above it, on a ring that's gone round, is the same at both ends
    frontier: usize,
}

impl Shared<'_> {
    /// How much a match at `scan` from `source` may be, or 0 if it can't be there at all
    #[inline(always)]
    fn room(&self, source: usize, scan: usize, end: usize) -> usize {
        if source < scan {
            end - scan
        } else if self.round && source >= self.frontier {
            (end - scan).min(self.history.len() - source)
        } else {
            0
        }
    }

    /// Packs the block at start..end into `out`, and returns false if it should go as it is
    fn pack(&self, start: usize, end: usize, out: &mut Vec<u8>) -> bool {
        out.clear();
        let (bytes, window) = (self.history, self.history.len());
        let last_start = end.saturating_sub(12);
        let (mut scan, mut anchor, mut misses) = (start, start, 0usize);
        let mut last_offset = 0;
        let mut tried = false;
        while scan < last_start {
            if !tried && scan - start >= TRIAL {
                tried = true;
                if out.len() * 8 > (scan - start) * 7 {
                    return false;
                }
            }
            // scan is below last_start, 12 before the end
            let wanted = unsafe { word(bytes, scan) };
            let mut found = None;
            if last_offset != 0 {
                let source = source_of(scan, last_offset, window);
                let room = self.room(source, scan, end);
                if room >= MIN_MATCH && unsafe { word(bytes, source) } == wanted {
                    found = Some((last_offset, 0, agree(bytes, source, scan, room)));
                }
            }
            if found.is_none() {
                let entry = &self.table[unsafe { slot(bytes, scan) }];
                let candidate = entry.load(Ordering::Relaxed) as usize;
                let room = self.room(candidate, scan, end);
                if room >= FAR_LEAST && unsafe { word(bytes, candidate) } == wanted {
                    let length = agree(bytes, candidate, scan, room);
                    if length >= FAR_LEAST {
                        let offset = if candidate < scan { scan - candidate } else { scan + window - candidate };
                        found = Some((offset, offset, length));
                    }
                }
                entry.store(scan as u32, Ordering::Relaxed);
            }
            let Some((offset, written, length)) = found else {
                misses += 1;
                scan += 1 + (misses >> 5);
                continue;
            };
            misses = 0;
            sequence(out, &bytes[anchor..scan], Some((length, written as u32)));
            // A few of those passed over, so a later block can find them: only so many, as copying a
            // long match should cost little
            let stop = (scan + length).min(last_start);
            let mut skipped = scan + 1;
            while skipped < stop {
                self.table[unsafe { slot(bytes, skipped) }].store(skipped as u32, Ordering::Relaxed);
                skipped += SKIP;
            }
            last_offset = offset;
            scan += length;
            anchor = scan;
        }
        sequence(out, &bytes[anchor..end], None);
        out.len() < end - start
    }
}

impl Packer {
    fn sized(window: usize) -> Self {
        let table = (0..1usize << TABLE_BITS).map(|_| AtomicU32::new(0)).collect();
        let coded = (0..BATCH).map(|_| Mutex::new((Vec::new(), false))).collect();
        Self { history: vec![0; window], at: 0, round: false, table, coded }
    }

    /// Packs these bytes into blocks, each given to `emit` in order with whether it's packed: if not, it goes
    /// as it is.
    pub fn pack_all(&mut self, bytes: &[u8], mut emit: impl FnMut(&[u8], bool) -> io::Result<()>) -> io::Result<()> {
        let window = self.history.len();
        let mut blocks = bytes.chunks(BLOCK).peekable();
        while blocks.peek().is_some() {
            if self.at + BLOCK > window {
                self.at = 0;
                self.round = true;
            }
            // As many as go in one piece, before the end of the ring
            let mut places = Vec::with_capacity(BATCH);
            while places.len() < BATCH && self.at + BLOCK <= window {
                let Some(block) = blocks.next() else { break };
                let start = self.at;
                self.at = start + block.len();
                self.history[start..self.at].copy_from_slice(block);
                places.push((start, self.at));
            }
            let shared = Shared { history: &self.history, table: &self.table, round: self.round, frontier: self.at };
            let (next, coded, places_ref) = (AtomicUsize::new(0), &self.coded, &places);
            let work = || {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(&(start, end)) = places_ref.get(index) else { return };
                    let mut slot = coded[index].lock().unwrap_or_else(|p| p.into_inner());
                    let slot = &mut *slot;
                    slot.1 = shared.pack(start, end, &mut slot.0);
                }
            };
            PACKING.run(threads().min(places.len()).saturating_sub(1), &work);
            for (index, &(start, end)) in places.iter().enumerate() {
                let slot = coded[index].lock().unwrap_or_else(|p| p.into_inner());
                if slot.1 {
                    emit(&slot.0, true)?;
                } else {
                    emit(&self.history[start..end], false)?;
                }
            }
        }
        Ok(())
    }
}

/// One direction's unpacking end
pub struct Unpacker {
    history: Vec<u8>,
    at: usize,
    round: bool,
}

impl Default for Unpacker {
    fn default() -> Self {
        Self::sized(WINDOW)
    }
}

impl Unpacker {
    fn sized(window: usize) -> Self {
        Self { history: vec![0; window], at: 0, round: false }
    }
}

fn read_count(coded: &[u8], from: &mut usize, mut value: usize) -> io::Result<usize> {
    loop {
        let byte = *coded.get(*from).ok_or_else(bad)?;
        *from += 1;
        value = value.checked_add(byte as usize).ok_or_else(bad)?;
        if byte != 255 {
            return Ok(value);
        }
    }
}

impl Unpacker {
    /// A block that went as it is
    pub fn raw(&mut self, block: &[u8]) -> io::Result<()> {
        if block.len() > BLOCK {
            return Err(bad());
        }
        if self.at + BLOCK > self.history.len() {
            self.at = 0;
            self.round = true;
        }
        self.history[self.at..self.at + block.len()].copy_from_slice(block);
        self.at += block.len();
        Ok(())
    }

    /// Unpacks a block
    pub fn unpack(&mut self, coded: &[u8]) -> io::Result<&[u8]> {
        let mut last_offset = 0;
        let window = self.history.len();
        if self.at + BLOCK > window {
            self.at = 0;
            self.round = true;
        }
        let start = self.at;
        let limit = start + BLOCK;
        let (mut from, mut to) = (0, start);
        loop {
            let token = *coded.get(from).ok_or_else(bad)?;
            from += 1;
            let mut literals = (token >> 4) as usize;
            if literals == 15 {
                literals = read_count(coded, &mut from, 15)?;
            }
            let source = coded.get(from..from.checked_add(literals).ok_or_else(bad)?).ok_or_else(bad)?;
            if to + literals > limit {
                return Err(bad());
            }
            self.history[to..to + literals].copy_from_slice(source);
            from += literals;
            to += literals;
            if from == coded.len() {
                break;
            }
            let (mut written, mut shift) = (0usize, 0);
            loop {
                let byte = *coded.get(from).ok_or_else(bad)?;
                from += 1;
                written |= ((byte & 0x7F) as usize) << shift;
                if byte < 0x80 {
                    break;
                }
                shift += 7;
                if shift > 28 {
                    return Err(bad());
                }
            }
            let offset = if written == 0 { last_offset } else { written };
            let mut length = (token & 15) as usize;
            if length == 15 {
                length = read_count(coded, &mut from, 15)?;
            }
            length += MIN_MATCH;
            if offset == 0 || offset >= window || to + length > limit || (offset > to && !self.round) {
                return Err(bad());
            }
            last_offset = offset;
            let source = source_of(to, offset, window);
            if source < to {
                if offset >= length {
                    self.history.copy_within(source..source + length, to);
                } else {
                    for index in to..to + length {
                        self.history[index] = self.history[index - offset];
                    }
                }
            } else {
                // From the last time round: not what's being written, and not past the end
                if source < to + length || source + length > window {
                    return Err(bad());
                }
                self.history.copy_within(source..source + length, to);
            }
            to += length;
        }
        self.at = to;
        Ok(&self.history[start..to])
    }
}

/// Packs what's written into blocks: when one is full, and on flush. Until `start` it passes
/// everything through as it is.
pub struct PackedWriter<W: Write> {
    inner: W,
    packer: Option<Box<Packer>>,
    pending: Vec<u8>,
    /// Bytes that went out in blocks
    pub wrote: u64,
    /// Time spent packing them, in nanoseconds
    pub packing: u64,
    /// Where what's packed is also kept as it was, how many more bytes of it, and from when: for working
    /// on the packer with what a real program writes. Each lot that's packed together goes as a u32 of
    /// its length, then its bytes.
    capture: Option<(std::fs::File, u64, std::time::Instant)>,
}

impl<W: Write> PackedWriter<W> {
    pub fn new(inner: W) -> Self {
        Self { inner, packer: None, pending: Vec::new(), wrote: 0, packing: 0, capture: None }
    }

    /// Keeps the first `most` bytes of what's packed from `from` on in `file`, as they were
    pub fn capture(&mut self, file: std::fs::File, most: u64, from: std::time::Instant) {
        self.capture = Some((file, most, from));
    }

    /// From here on, blocks. What's been written so far goes first, as it is.
    pub fn start(&mut self) -> io::Result<()> {
        self.inner.flush()?;
        self.packer = Some(Box::default());
        self.pending = Vec::with_capacity(BATCH * BLOCK);
        Ok(())
    }

    fn send_blocks(&mut self) -> io::Result<()> {
        let Some(packer) = &mut self.packer else { return Ok(()) };
        if self.pending.is_empty() {
            return Ok(());
        }
        if let Some((file, left, _)) = self.capture.as_mut().filter(|(_, _, from)| *from <= std::time::Instant::now()) {
            let kept = file.write_all(&(self.pending.len() as u32).to_le_bytes()).and_then(|()| file.write_all(&self.pending));
            *left = left.saturating_sub(self.pending.len() as u64 + 4);
            if kept.is_err() || *left == 0 {
                self.capture = None;
            }
        }
        let started = std::time::Instant::now();
        let (inner, wrote) = (&mut self.inner, &mut self.wrote);
        let mut writing = std::time::Duration::ZERO;
        packer.pack_all(&self.pending, |body, packed| {
            let began = std::time::Instant::now();
            let head = (body.len() as u32) << 1 | !packed as u32;
            inner.write_all(&head.to_le_bytes())?;
            inner.write_all(body)?;
            *wrote += body.len() as u64 + 4;
            writing += began.elapsed();
            Ok(())
        })?;
        self.packing += started.elapsed().saturating_sub(writing).as_nanos() as u64;
        self.pending.clear();
        Ok(())
    }
}

impl<W: Write> Write for PackedWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.packer.is_none() {
            return self.inner.write(bytes);
        }
        let take = bytes.len().min(BATCH * BLOCK - self.pending.len());
        self.pending.extend_from_slice(&bytes[..take]);
        if self.pending.len() == BATCH * BLOCK {
            self.send_blocks()?;
        }
        Ok(take)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.send_blocks()?;
        self.inner.flush()
    }
}

/// Unpacks a PackedWriter's stream. Until `start` it passes everything through as it is.
pub struct PackedReader<R: Read> {
    inner: R,
    unpacker: Option<Box<Unpacker>>,
    block: Vec<u8>,
    at: usize,
    coded: Vec<u8>,
    /// Bytes that came in blocks, and what they became
    pub came: u64,
    pub became: u64,
}

impl<R: Read> PackedReader<R> {
    pub fn new(inner: R) -> Self {
        Self { inner, unpacker: None, block: Vec::new(), at: 0, coded: Vec::new(), came: 0, became: 0 }
    }

    pub fn start(&mut self) {
        self.unpacker = Some(Box::default());
    }

    pub fn inner(&self) -> &R {
        &self.inner
    }

    /// Whether all it has read so far has been handed on
    pub fn drained(&self) -> bool {
        self.at == self.block.len()
    }
}

impl<R: Read> Read for PackedReader<R> {
    fn read(&mut self, target: &mut [u8]) -> io::Result<usize> {
        let Some(unpacker) = &mut self.unpacker else { return self.inner.read(target) };
        while self.at == self.block.len() {
            let mut head = [0u8; 4];
            self.inner.read_exact(&mut head)?;
            let head = u32::from_le_bytes(head);
            let length = (head >> 1) as usize;
            if length > 2 * BLOCK {
                return Err(bad());
            }
            self.coded.resize(length, 0);
            self.inner.read_exact(&mut self.coded)?;
            self.block.clear();
            if head & 1 == 1 {
                unpacker.raw(&self.coded)?;
                self.block.extend_from_slice(&self.coded);
            } else {
                self.block.extend_from_slice(unpacker.unpack(&self.coded)?);
            }
            self.at = 0;
            self.came += length as u64 + 4;
            self.became += self.block.len() as u64;
        }
        let take = target.len().min(self.block.len() - self.at);
        target[..take].copy_from_slice(&self.block[self.at..self.at + take]);
        self.at += take;
        Ok(take)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whats_packed_can_be_kept_as_it_was() {
        let path = std::env::temp_dir().join(format!("nectar-capture-test-{}", std::process::id()));
        let mut writer = PackedWriter::new(Vec::new());
        writer.start().unwrap();
        writer.capture(std::fs::File::create(&path).unwrap(), 24, std::time::Instant::now());
        for lot in [&b"the first lot"[..], b"the second", b"past the most that's kept"] {
            writer.write_all(lot).unwrap();
            writer.flush().unwrap();
        }
        let kept = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        let mut expected = Vec::new();
        for lot in [&b"the first lot"[..], b"the second"] {
            expected.extend_from_slice(&(lot.len() as u32).to_le_bytes());
            expected.extend_from_slice(lot);
        }
        assert_eq!(kept, expected);
    }

    #[test]
    fn a_stream_comes_back_the_same() {
        let mut writer = PackedWriter::new(Vec::new());
        writer.write_all(b"before").unwrap();
        writer.start().unwrap();
        let mut sent = Vec::new();
        for frame in 0..40 {
            let frame_bytes = commands(frame);
            writer.write_all(&frame_bytes).unwrap();
            sent.extend_from_slice(&frame_bytes);
            // More than a block at once, of what won't compress
            let bytes = noise(frame as u64, BLOCK * 2 + 17);
            writer.write_all(&bytes).unwrap();
            sent.extend_from_slice(&bytes);
            writer.flush().unwrap();
        }
        writer.flush().unwrap();
        let mut reader = PackedReader::new(&writer.inner[..]);
        let mut before = [0u8; 6];
        reader.read_exact(&mut before).unwrap();
        assert_eq!(&before, b"before");
        reader.start();
        let mut came = vec![0u8; sent.len()];
        reader.read_exact(&mut came).unwrap();
        assert!(came == sent);
        assert!(reader.drained());
        assert!(reader.read(&mut before).is_err(), "the stream has ended");
    }

    fn noise(seed: u64, length: usize) -> Vec<u8> {
        let mut state = seed | 1;
        (0..length)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                state as u8
            })
            .collect()
    }

    /// Goes through both ends, however many blocks it makes; returns the bytes it took
    fn through(packer: &mut Packer, unpacker: &mut Unpacker, bytes: &[u8]) -> usize {
        let mut came = Vec::new();
        let mut cost = 0;
        packer
            .pack_all(bytes, |body, packed| {
                cost += body.len();
                if packed {
                    came.extend_from_slice(unpacker.unpack(body).unwrap());
                } else {
                    unpacker.raw(body).unwrap();
                    came.extend_from_slice(body);
                }
                Ok(())
            })
            .unwrap();
        assert!(came == bytes);
        assert_eq!(packer.at, unpacker.at);
        cost
    }

    /// Something like a frame's commands: records with a value or two different each time
    fn commands(frame: u32) -> Vec<u8> {
        let mut bytes = Vec::new();
        for draw in 0..800u32 {
            bytes.extend_from_slice(&0x1234_5678u32.to_le_bytes());
            bytes.extend_from_slice(&draw.to_le_bytes());
            bytes.extend_from_slice(&[7u8; 24]);
            bytes.extend_from_slice(&(draw.wrapping_mul(2654435761) ^ frame).to_le_bytes());
            bytes.extend_from_slice(&0xDEAD_BEEF_u32.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn blocks_come_back_the_same() {
        let (mut packer, mut unpacker) = (Packer::sized(BLOCK * 8), Unpacker::sized(BLOCK * 8));
        for block in [vec![], vec![1u8], vec![9u8; 11], vec![0u8; BLOCK], b"abcdabcdabcdabcdabcdabcdabcd".repeat(100), commands(1)] {
            through(&mut packer, &mut unpacker, &block);
        }
    }

    #[test]
    fn the_next_frame_costs_little() {
        let (mut packer, mut unpacker) = (Packer::default(), Unpacker::default());
        let first = commands(1);
        let cost = through(&mut packer, &mut unpacker, &first);
        let next = through(&mut packer, &mut unpacker, &commands(2));
        assert!(next <= cost, "{cost} then {next}");
        assert!(next * 4 < first.len(), "{next} of {}", first.len());
    }

    #[test]
    fn noise_goes_as_it_is_and_both_ends_keep_it() {
        let (mut packer, mut unpacker) = (Packer::sized(BLOCK * 8), Unpacker::sized(BLOCK * 8));
        let block = noise(5, BLOCK);
        assert_eq!(through(&mut packer, &mut unpacker, &block), BLOCK);
        assert_eq!(packer.at, BLOCK);
        // Seen once, even as it was, it's found again
        assert!(through(&mut packer, &mut unpacker, &block) < 400);
        for frame in 0..3 {
            through(&mut packer, &mut unpacker, &commands(frame));
            let mixed: Vec<u8> = noise(frame as u64 + 9, 200).into_iter().chain(commands(frame)).take(BLOCK).collect();
            through(&mut packer, &mut unpacker, &mixed);
        }
    }

    #[test]
    fn a_batch_copies_only_from_what_comes_before_it() {
        // Few enough different blocks that each is found again, in the batch, before it and after it, on a
        // ring small enough that what's above is often the same as a block still to come. A copy from a
        // block further on in the batch would come out as what used to be there.
        const RING: usize = BLOCK * 40;
        let palette: Vec<Vec<u8>> = (0..6).map(|seed| noise(seed, BLOCK)).collect();
        let (mut packer, mut unpacker) = (Packer::sized(RING), Unpacker::sized(RING));
        let mut state = 0x1234_5678_9abc_def1u64;
        let mut pick = move |limit: usize| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as usize % limit
        };
        let mut rounds = 0;
        for _ in 0..120 {
            let mut bytes = Vec::new();
            for _ in 0..pick(BATCH * 2) + 1 {
                let block = &palette[pick(palette.len())];
                // Some shifted, so that matches don't all line up with blocks
                bytes.extend_from_slice(&block[pick(4096)..]);
            }
            through(&mut packer, &mut unpacker, &bytes);
            rounds += packer.round as usize;
        }
        assert!(rounds > 0, "the ring went round");
    }

    #[test]
    fn however_many_threads_it_comes_back_the_same_size_or_near_it() {
        // One stream, in batches and one block at a time
        let mut frames = Vec::new();
        for frame in 0..30 {
            frames.extend_from_slice(&commands(frame % 3).repeat(8));
        }
        let (mut packer, mut unpacker) = (Packer::sized(BLOCK * 64), Unpacker::sized(BLOCK * 64));
        let batched = through(&mut packer, &mut unpacker, &frames);
        let (mut packer, mut unpacker) = (Packer::sized(BLOCK * 64), Unpacker::sized(BLOCK * 64));
        let one_at_a_time: usize = frames.chunks(BLOCK).map(|block| through(&mut packer, &mut unpacker, block)).sum();
        assert!(batched < one_at_a_time * 2, "{batched} in batches and {one_at_a_time} one at a time");
    }

    #[test]
    fn the_window_goes_round() {
        // A ring small enough to go round many times, of what repeats from a few blocks back and so often
        // from across its end
        const RING: usize = BLOCK * 8;
        let (mut packer, mut unpacker) = (Packer::sized(RING), Unpacker::sized(RING));
        let blocks: Vec<Vec<u8>> = (0..5).map(|seed| noise(seed, BLOCK / 2)).collect();
        let (mut small, mut raw) = (0, 0);
        for frame in 0..400 {
            // Noise goes as it is unless it's after something that compresses, as a trial would see
            let mut block = commands(frame as u32).repeat(2);
            block.truncate(BLOCK / 2);
            block.extend_from_slice(&blocks[frame % blocks.len()]);
            let cost = through(&mut packer, &mut unpacker, &block);
            if cost < BLOCK / 4 {
                small += 1;
            }
            // Now and then one that goes as it is
            if frame % 17 == 0 {
                let bytes = noise(frame as u64 + 5000, BLOCK);
                if through(&mut packer, &mut unpacker, &bytes) == BLOCK {
                    raw += 1;
                }
            }
        }
        assert!(packer.round && unpacker.round);
        assert!(small > 380, "{small} of 400 found what they repeat");
        assert!(raw > 10);
    }

    #[test]
    fn a_match_can_reach_back_across_the_end_of_the_ring() {
        const RING: usize = BLOCK * 4;
        let (mut packer, mut unpacker) = (Packer::sized(RING), Unpacker::sized(RING));
        for frame in 0..3 {
            through(&mut packer, &mut unpacker, &vec![frame as u8 + 1; BLOCK]);
        }
        // At the top of the ring
        let mut top = vec![7u8; 8192];
        top.extend_from_slice(&noise(1, BLOCK - 8192));
        through(&mut packer, &mut unpacker, &top);
        assert_eq!(packer.at, RING);
        assert!(!packer.round);
        // Back at the bottom, and the only other copy of this is above where it's going
        let cost = through(&mut packer, &mut unpacker, &top);
        assert!(packer.round && unpacker.round);
        assert!(cost * 8 < BLOCK, "{cost} for what's still in the ring");
    }

    #[test]
    fn what_comes_off_the_network_cant_read_outside_the_window() {
        let cases: &[&[u8]] = &[
            // A match before anything was written
            &[0x04, 1, 2, 3, 4, 0x7F],
            // A match with no last offset to repeat
            &[0x04, 1, 2, 3, 4, 0],
            // Cut short
            &[0xF0],
            &[0x50, 1, 2],
            // Far too long
            &[0x1F, 1, 1, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            // An offset that never ends
            &[0x10, 1, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF],
        ];
        for case in cases {
            assert!(Unpacker::sized(BLOCK * 2).unpack(case).is_err(), "{case:?}");
        }
        // Once it's gone round: a match mustn't run past the end, nor into what's being written
        let mut unpacker = Unpacker::sized(BLOCK * 2);
        unpacker.round = true;
        unpacker.at = 0;
        // Offset 2 from 0 is the ring's last 2 bytes, and 8 bytes won't fit there
        assert!(unpacker.unpack(&[0x04, 2]).is_err());
        // Offset nearly all the way round is what's just ahead, which isn't written yet
        let around = BLOCK * 2 - 1;
        let coded = [0x04, (around & 0x7F) as u8 | 0x80, ((around >> 7) & 0x7F) as u8 | 0x80, (around >> 14) as u8];
        assert!(unpacker.unpack(&coded).is_err());
        // An offset of the whole ring is nothing
        let whole = BLOCK * 2;
        let coded = [0x04, (whole & 0x7F) as u8 | 0x80, ((whole >> 7) & 0x7F) as u8 | 0x80, (whole >> 14) as u8];
        assert!(unpacker.unpack(&coded).is_err());
    }

    /// Run with --ignored --nocapture to see how fast it is
    #[test]
    #[ignore]
    fn speed() {
        let frames: Vec<Vec<u8>> = (0..64).map(|frame| commands(frame).repeat(2)).collect();
        for (name, frames) in [("commands", frames), ("noise", (0..64).map(|seed| noise(seed, BLOCK)).collect())] {
            let (mut packer, mut unpacker) = (Packer::default(), Unpacker::default());
            let (mut coded, mut total, mut packing, mut unpacking) = (0, 0, 0.0, 0.0);
            let bytes: Vec<u8> = frames.concat();
            for _ in 0..10 {
                let mut blocks = Vec::new();
                let timer = std::time::Instant::now();
                packer
                    .pack_all(&bytes, |body, packed| {
                        blocks.push((body.to_vec(), packed));
                        Ok(())
                    })
                    .unwrap();
                packing += timer.elapsed().as_secs_f64();
                total += bytes.len();
                let timer = std::time::Instant::now();
                for (body, packed) in &blocks {
                    coded += body.len();
                    if *packed {
                        unpacker.unpack(body).unwrap();
                    } else {
                        unpacker.raw(body).unwrap();
                    }
                }
                unpacking += timer.elapsed().as_secs_f64();
            }
            println!(
                "{name}: {:.1}% of {:.0} MB; packs at {:.0} MB/s, unpacks at {:.0} MB/s",
                coded as f64 * 100.0 / total as f64,
                total as f64 / 1e6,
                total as f64 / 1e6 / packing,
                total as f64 / 1e6 / unpacking
            );
        }
    }
}
