//! A mip coded by how much it matters: a 5/3 wavelet splits a picture into bands of detail, the mips
//! below in all but name, and the finest bands are given the fewest bits. It stands in for a mip's real
//! bytes, about 10 to 20 times fewer of them, until those arrive (vk/src/later.rs).
//!
//! A mip is cut into tiles, each coded alone, so tiles can go on as many threads as there are and a large
//! mip takes no more memory per thread than a small one. In a tile, each plane's bands go coarsest
//! first as groups of 4x4 coefficients: a 0 bit for a group of zeros; otherwise a 1, 5 bits saying how
//! many bits each coefficient takes, then the coefficients. It's plain bit packing, quick to undo.
//!
//! sRGB is colour, so it goes as YCoCg with chroma at half the size, which people don't miss. Anything
//! else might be data, such as a normal map or a packed mask, whose channels have nothing to do with
//! each other, so they go apart, all at full size.
//!
//! All this runs while a game does, so it's written to be quick: each pass works along whole rows, which
//! the compiler can turn into vector instructions and the cache likes, and steps are powers of 2, as
//! dividing for every coefficient took more time than anything else.
//!
//! The same bands could be sent coarsest first and shown as they come. They aren't yet.

#![allow(clippy::chunks_exact_to_as_chunks)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

use crate::bc::{self, Kind};

/// A tile's side in blocks, and in texels where there are no blocks
const TILE_BLOCKS: usize = 256;
const TILE_TEXELS: usize = 1024;
/// The most times a tile's split in two each way
const LEVELS: usize = 5;
/// A band this small isn't split further
const SMALLEST: usize = 8;
/// Far more than 8-bit texels can make, and small enough that joining the bands of nonsense can't overflow
const COEFFICIENT_MOST: i32 = 1 << 18;

/// How hard each band is quantised, as how many bits to drop: the bands one split from the top lose
/// `bits + extra[0]`, the next `bits + extra[1]`, and so on. The coarsest band loses 2 fewer than `bits`.
#[derive(Clone, Copy)]
pub struct Quality {
    pub bits: u32,
    pub extra: [u32; LEVELS],
}

/// About 8 times smaller than BC7 on real frames, with the finest detail softened
pub const QUALITY: Quality = Quality { bits: 3, extra: [2, 1, 0, 0, 0] };

/// The bits a band loses. Chroma starts at half the size, so its finest bands count as the next ones.
fn bits_of(quality: &Quality, level: usize, chroma: bool) -> u32 {
    if level == 0 {
        quality.bits.saturating_sub(2)
    } else {
        quality.bits + quality.extra[(level - 1 + chroma as usize).min(LEVELS - 1)]
    }
}

/// One split of a row: the low half then the high half. `scratch` holds as much as the row.
fn split_row(row: &mut [i32], scratch: &mut [i32]) {
    let n = row.len();
    if n < 2 {
        return;
    }
    let (low, high) = (n.div_ceil(2), n / 2);
    let (evens, odds) = scratch[..n].split_at_mut(low);
    for (even, value) in evens.iter_mut().zip(row.iter().step_by(2)) {
        *even = *value;
    }
    for (odd, value) in odds.iter_mut().zip(row.iter().skip(1).step_by(2)) {
        *odd = *value;
    }
    // Each odd one less what's expected from the evens beside it; past the end, the end's mirrored
    let last_even = evens[low - 1];
    let pairs = high.min(low - 1);
    for ((odd, a), b) in odds[..pairs].iter_mut().zip(&evens[..pairs]).zip(&evens[1..=pairs]) {
        *odd -= (*a + *b) >> 1;
    }
    if high > pairs {
        odds[high - 1] -= last_even;
    }
    // Then each even plus a quarter of the odds beside it
    evens[0] += (odds[0] + odds[0] + 2) >> 2;
    let inner = high.saturating_sub(1).min(low - 1);
    for ((even, a), b) in evens[1..=inner].iter_mut().zip(&odds[..inner]).zip(&odds[1..=inner]) {
        *even += (*a + *b + 2) >> 2;
    }
    if low > inner + 1 {
        evens[low - 1] += (odds[high - 1] + odds[high - 1] + 2) >> 2;
    }
    row.copy_from_slice(&scratch[..n]);
}

fn join_row(row: &mut [i32], scratch: &mut [i32]) {
    let n = row.len();
    if n < 2 {
        return;
    }
    let (low, high) = (n.div_ceil(2), n / 2);
    let (evens, odds) = row.split_at_mut(low);
    evens[0] -= (odds[0] + odds[0] + 2) >> 2;
    let inner = high.saturating_sub(1).min(low - 1);
    for ((even, a), b) in evens[1..=inner].iter_mut().zip(&odds[..inner]).zip(&odds[1..=inner]) {
        *even -= (*a + *b + 2) >> 2;
    }
    if low > inner + 1 {
        evens[low - 1] -= (odds[high - 1] + odds[high - 1] + 2) >> 2;
    }
    let pairs = high.min(low - 1);
    for ((odd, a), b) in odds[..pairs].iter_mut().zip(&evens[..pairs]).zip(&evens[1..=pairs]) {
        *odd += (*a + *b) >> 1;
    }
    if high > pairs {
        odds[high - 1] += evens[low - 1];
    }
    for (at, value) in evens.iter().enumerate() {
        scratch[at * 2] = *value;
    }
    for (at, value) in odds.iter().enumerate() {
        scratch[at * 2 + 1] = *value;
    }
    row.copy_from_slice(&scratch[..n]);
}

/// The rows of `plane`, `width` apart, one split down its first `w` columns and `h` rows: the same
/// lifting as split_row, a whole row of values at a time. `rows` holds as much as the plane.
fn split_columns(plane: &mut [i32], width: usize, w: usize, h: usize, rows: &mut [i32]) {
    if h < 2 {
        return;
    }
    let (low, high) = (h.div_ceil(2), h / 2);
    let row = |y: usize| y * width..y * width + w;
    for i in 0..high {
        let (a, b) = (2 * i, if 2 * i + 2 < h { 2 * i + 2 } else { 2 * i });
        let (to, from) = (&mut rows[(low + i) * width..(low + i) * width + w], &plane[row(2 * i + 1)]);
        for (((out, value), x), y) in to.iter_mut().zip(from).zip(&plane[row(a)]).zip(&plane[row(b)]) {
            *out = *value - ((*x + *y) >> 1);
        }
    }
    for i in 0..low {
        let (a, b) = (low + if i == 0 { 0 } else { i - 1 }, low + i.min(high - 1));
        let (lows, highs) = rows.split_at_mut(low * width);
        for (((out, value), x), y) in lows[i * width..i * width + w].iter_mut().zip(&plane[row(2 * i)]).zip(&highs[(a - low) * width..(a - low) * width + w]).zip(&highs[(b - low) * width..(b - low) * width + w]) {
            *out = *value + ((*x + *y + 2) >> 2);
        }
    }
    for y in 0..h {
        plane[row(y)].copy_from_slice(&rows[row(y)]);
    }
}

fn join_columns(plane: &mut [i32], width: usize, w: usize, h: usize, rows: &mut [i32]) {
    if h < 2 {
        return;
    }
    let (low, high) = (h.div_ceil(2), h / 2);
    let row = |y: usize| y * width..y * width + w;
    for i in 0..low {
        let (a, b) = (low + if i == 0 { 0 } else { i - 1 }, low + i.min(high - 1));
        let to = &mut rows[2 * i * width..2 * i * width + w];
        for (((out, value), x), y) in to.iter_mut().zip(&plane[row(i)]).zip(&plane[row(a)]).zip(&plane[row(b)]) {
            *out = *value - ((*x + *y + 2) >> 2);
        }
    }
    for i in 0..high {
        let (a, b) = (2 * i, if 2 * i + 2 < h { 2 * i + 2 } else { 2 * i });
        // Evens are in rows already; odds go between them
        let (before, after) = rows.split_at_mut((2 * i + 1) * width);
        let first = &before[a * width..a * width + w];
        if b == a {
            for ((out, value), x) in after[..w].iter_mut().zip(&plane[row(low + i)]).zip(first) {
                *out = *value + *x;
            }
        } else {
            let (to, rest) = after.split_at_mut(width);
            for (((out, value), x), y) in to[..w].iter_mut().zip(&plane[row(low + i)]).zip(first).zip(&rest[..w]) {
                *out = *value + ((*x + *y) >> 1);
            }
        }
    }
    for y in 0..h {
        plane[row(y)].copy_from_slice(&rows[row(y)]);
    }
}

/// Each level's size: the whole tile, then what's left to split each time
fn sizes(width: usize, height: usize) -> Vec<(usize, usize)> {
    let mut list = vec![(width, height)];
    while list.len() <= LEVELS {
        let &(w, h) = list.last().unwrap();
        if w < SMALLEST || h < SMALLEST {
            break;
        }
        list.push((w.div_ceil(2), h.div_ceil(2)));
    }
    list
}

fn split(plane: &mut [i32], width: usize, levels: &[(usize, usize)]) {
    let mut scratch = vec![0; width];
    let mut rows = vec![0; plane.len()];
    for &(w, h) in &levels[..levels.len() - 1] {
        for y in 0..h {
            split_row(&mut plane[y * width..y * width + w], &mut scratch);
        }
        split_columns(plane, width, w, h, &mut rows);
    }
}

fn join(plane: &mut [i32], width: usize, levels: &[(usize, usize)]) {
    let mut scratch = vec![0; width];
    let mut rows = vec![0; plane.len()];
    for &(w, h) in levels[..levels.len() - 1].iter().rev() {
        join_columns(plane, width, w, h, &mut rows);
        for y in 0..h {
            join_row(&mut plane[y * width..y * width + w], &mut scratch);
        }
    }
}

/// The bands coarsest first: left, top, width, height, and how many splits from the top they are (the
/// coarsest band being 0)
fn bands(levels: &[(usize, usize)]) -> Vec<(usize, usize, usize, usize, usize)> {
    let deepest = levels.len() - 1;
    let (w, h) = levels[deepest];
    let mut list = vec![(0, 0, w, h, 0)];
    for depth in (1..=deepest).rev() {
        let (whole_w, whole_h) = levels[depth - 1];
        let (low_w, low_h) = levels[depth];
        let level = depth;
        list.push((low_w, 0, whole_w - low_w, low_h, level));
        list.push((0, low_h, low_w, whole_h - low_h, level));
        list.push((low_w, low_h, whole_w - low_w, whole_h - low_h, level));
    }
    list
}

#[derive(Default)]
struct BitWriter {
    out: Vec<u8>,
    held: u64,
    count: u32,
}

impl BitWriter {
    #[inline(always)]
    fn put(&mut self, value: u32, bits: u32) {
        self.held |= (value as u64) << self.count;
        self.count += bits;
        if self.count >= 32 {
            self.out.extend_from_slice(&(self.held as u32).to_le_bytes());
            self.held >>= 32;
            self.count -= 32;
        }
    }

    fn finish(mut self) -> Vec<u8> {
        while self.count > 0 {
            self.out.push(self.held as u8);
            self.held >>= 8;
            self.count = self.count.saturating_sub(8);
        }
        self.out
    }
}

struct BitReader<'a> {
    bytes: &'a [u8],
    at: usize,
    held: u64,
    count: u32,
}

impl BitReader<'_> {
    /// Past the end reads as zeros: what comes off the network can make a wrong picture, nothing worse
    #[inline(always)]
    fn take(&mut self, bits: u32) -> u32 {
        if self.count < bits {
            let word = match self.bytes.get(self.at..self.at + 4) {
                Some(word) => u32::from_le_bytes(word.try_into().unwrap()),
                None => {
                    let mut last = [0u8; 4];
                    if let Some(left) = self.bytes.get(self.at..) {
                        last[..left.len()].copy_from_slice(left);
                    }
                    u32::from_le_bytes(last)
                }
            };
            self.at += 4;
            self.held |= (word as u64) << self.count;
            self.count += 32;
        }
        let value = (self.held & ((1u64 << bits) - 1)) as u32;
        self.held >>= bits;
        self.count -= bits;
        value
    }
}

fn code_plane(plane: &[i32], width: usize, levels: &[(usize, usize)], quality: &Quality, chroma: bool, bits: &mut BitWriter) {
    for (left, top, w, h, level) in bands(levels) {
        let drop = bits_of(quality, level, chroma);
        let mut y = 0;
        while y < h {
            let down = (h - y).min(4);
            let mut x = 0;
            while x < w {
                let across = (w - x).min(4);
                let mut folded = [0u32; 16];
                let mut any = 0;
                for j in 0..down {
                    let start = (top + y + j) * width + left + x;
                    for (i, value) in plane[start..start + across].iter().enumerate() {
                        // Towards zero, as a coefficient near zero is mostly noise
                        let sign = value >> 31;
                        let scaled = (((*value ^ sign) - sign) >> drop ^ sign) - sign;
                        let fold = ((scaled << 1) ^ (scaled >> 31)) as u32;
                        folded[j * 4 + i] = fold;
                        any |= fold;
                    }
                }
                if any == 0 {
                    bits.put(0, 1);
                } else {
                    let needed = 32 - any.leading_zeros();
                    bits.put(1 | ((needed - 1) << 1), 6);
                    for j in 0..down {
                        for fold in &folded[j * 4..j * 4 + across] {
                            bits.put(*fold, needed);
                        }
                    }
                }
                x += 4;
            }
            y += 4;
        }
    }
}

fn read_plane(plane: &mut [i32], width: usize, levels: &[(usize, usize)], quality: &Quality, chroma: bool, bits: &mut BitReader) {
    plane.fill(0);
    for (left, top, w, h, level) in bands(levels) {
        let drop = bits_of(quality, level, chroma);
        let half = (1i32 << drop) >> 1;
        let mut y = 0;
        while y < h {
            let down = (h - y).min(4);
            let mut x = 0;
            while x < w {
                if bits.take(1) != 0 {
                    let needed = bits.take(5) + 1;
                    for j in 0..down {
                        let start = (top + y + j) * width + left + x;
                        for out in &mut plane[start..start + (w - x).min(4)] {
                            let fold = bits.take(needed);
                            let value = (((fold >> 1) as i32) ^ -((fold & 1) as i32)).clamp(-COEFFICIENT_MOST >> drop, COEFFICIENT_MOST >> drop);
                            *out = match value {
                                0 => 0,
                                v if v > 0 => (v << drop) + half,
                                v => (v << drop) - half,
                            };
                        }
                    }
                }
                x += 4;
            }
            y += 4;
        }
    }
}

/// Whether a kind's colours go as YCoCg
fn as_colour(kind: Kind, srgb: bool) -> bool {
    srgb && kind.channels() >= 3
}

/// Chroma's size, for a tile of this size
fn half(width: usize, height: usize) -> (usize, usize) {
    (width.div_ceil(2), height.div_ceil(2))
}

fn encode_tile(pixels: &[u8], width: usize, height: usize, kind: Kind, srgb: bool, quality: &Quality) -> Vec<u8> {
    let levels = sizes(width, height);
    let texels = width * height;
    let mut bits = BitWriter::default();
    let has_alpha = kind.channels() == 4 && pixels.chunks_exact(4).any(|pixel| pixel[3] != 255);
    bits.put(has_alpha as u32, 1);
    if as_colour(kind, srgb) {
        let (cw, ch) = half(width, height);
        let chroma_levels = sizes(cw, ch);
        let (mut y, mut co, mut cg) = (vec![0; texels], vec![0; cw * ch], vec![0; cw * ch]);
        for row in 0..height {
            let at = row * width;
            let (co_row, cg_row) = (&mut co[(row / 2) * cw..(row / 2) * cw + cw], &mut cg[(row / 2) * cw..(row / 2) * cw + cw]);
            for (x, pixel) in pixels[at * 4..(at + width) * 4].chunks_exact(4).enumerate() {
                let (r, g, b) = (pixel[0] as i32, pixel[1] as i32, pixel[2] as i32);
                let orange = r - b;
                let t = b + (orange >> 1);
                let green = g - t;
                y[at + x] = t + (green >> 1) - 128;
                co_row[x / 2] += orange;
                cg_row[x / 2] += green;
            }
        }
        // Each is the sum of up to four. Only the last row or column of an odd size has fewer, which
        // loses nothing that matters.
        for value in co.iter_mut().chain(cg.iter_mut()) {
            *value >>= 2;
        }
        split(&mut y, width, &levels);
        code_plane(&y, width, &levels, quality, false, &mut bits);
        for plane in [&mut co, &mut cg] {
            split(plane, cw, &chroma_levels);
            code_plane(plane, cw, &chroma_levels, quality, true, &mut bits);
        }
    } else {
        for channel in 0..kind.channels().min(3) {
            let mut plane: Vec<i32> = pixels.chunks_exact(4).map(|pixel| pixel[channel] as i32 - 128).collect();
            split(&mut plane, width, &levels);
            code_plane(&plane, width, &levels, quality, false, &mut bits);
        }
    }
    if has_alpha {
        let mut plane: Vec<i32> = pixels.chunks_exact(4).map(|pixel| pixel[3] as i32 - 128).collect();
        split(&mut plane, width, &levels);
        code_plane(&plane, width, &levels, quality, false, &mut bits);
    }
    bits.finish()
}

fn decode_tile(coded: &[u8], pixels: &mut [u8], width: usize, height: usize, kind: Kind, srgb: bool, quality: &Quality) {
    let levels = sizes(width, height);
    let mut bits = BitReader { bytes: coded, at: 0, held: 0, count: 0 };
    let has_alpha = bits.take(1) != 0 && kind.channels() == 4;
    let mut plane = vec![0; width * height];
    if as_colour(kind, srgb) {
        read_plane(&mut plane, width, &levels, quality, false, &mut bits);
        join(&mut plane, width, &levels);
        let (cw, ch) = half(width, height);
        let chroma_levels = sizes(cw, ch);
        let (mut co, mut cg) = (vec![0; cw * ch], vec![0; cw * ch]);
        for chroma in [&mut co, &mut cg] {
            read_plane(chroma, cw, &chroma_levels, quality, true, &mut bits);
            join(chroma, cw, &chroma_levels);
        }
        for row in 0..height {
            let at = row * width;
            let (co_row, cg_row) = (&co[(row / 2) * cw..(row / 2) * cw + cw], &cg[(row / 2) * cw..(row / 2) * cw + cw]);
            for (x, pixel) in pixels[at * 4..(at + width) * 4].chunks_exact_mut(4).enumerate() {
                let (luma, orange, green) = (plane[at + x] + 128, co_row[x / 2], cg_row[x / 2]);
                let t = luma - (green >> 1);
                let g = green + t;
                let b = t - (orange >> 1);
                let r = b + orange;
                pixel[0] = r.clamp(0, 255) as u8;
                pixel[1] = g.clamp(0, 255) as u8;
                pixel[2] = b.clamp(0, 255) as u8;
                pixel[3] = 255;
            }
        }
    } else {
        for pixel in pixels.chunks_exact_mut(4) {
            pixel.copy_from_slice(&[0, 0, 0, 255]);
        }
        for channel in 0..kind.channels().min(3) {
            read_plane(&mut plane, width, &levels, quality, false, &mut bits);
            join(&mut plane, width, &levels);
            for (pixel, value) in pixels.chunks_exact_mut(4).zip(&plane) {
                pixel[channel] = (value + 128).clamp(0, 255) as u8;
            }
        }
    }
    if has_alpha {
        read_plane(&mut plane, width, &levels, quality, false, &mut bits);
        join(&mut plane, width, &levels);
        for (pixel, value) in pixels.chunks_exact_mut(4).zip(&plane) {
            pixel[3] = (value + 128).clamp(0, 255) as u8;
        }
    }
}

/// Runs `job(0)..job(count)` on up to `threads` threads
fn on_threads(threads: usize, count: usize, job: impl Fn(usize) + Sync) {
    let next = AtomicUsize::new(0);
    let work = || {
        loop {
            let index = next.fetch_add(1, Ordering::Relaxed);
            if index >= count {
                return;
            }
            job(index);
        }
    };
    if threads <= 1 || count <= 1 {
        work();
        return;
    }
    thread::scope(|scope| {
        for _ in 0..threads.min(count) {
            scope.spawn(work);
        }
    });
}

/// A mip's tiles: where each starts and its size, in blocks
fn tiles(kind: Kind, across: usize, down: usize) -> Vec<(usize, usize, usize, usize)> {
    let side = if kind.block_side() == 1 { TILE_TEXELS } else { TILE_BLOCKS };
    let mut list = Vec::new();
    let mut top = 0;
    while top < down {
        let mut left = 0;
        while left < across {
            list.push((left, top, (across - left).min(side), (down - top).min(side)));
            left += side;
        }
        top += side;
    }
    list
}

/// Codes a mip: `bytes` holds `across` by `down` blocks of `kind`. At the start, each tile's length as a
/// u32, then the tiles.
pub fn encode_mip(kind: Kind, srgb: bool, bytes: &[u8], across: usize, down: usize, quality: &Quality, threads: usize) -> Vec<u8> {
    let list = tiles(kind, across, down);
    let done: Vec<std::sync::Mutex<Vec<u8>>> = list.iter().map(|_| std::sync::Mutex::new(Vec::new())).collect();
    let (block, side) = (kind.block_bytes(), kind.block_side());
    on_threads(threads, list.len(), |index| {
        let (left, top, wide, high) = list[index];
        let (width, height) = (wide * side, high * side);
        let mut pixels = vec![0u8; width * height * 4];
        let mut texels = [0u8; 64];
        for y in 0..high {
            for x in 0..wide {
                let at = ((top + y) * across + left + x) * block;
                bc::decode_block(kind, &bytes[at..at + block], &mut texels);
                for row in 0..side {
                    let to = ((y * side + row) * width + x * side) * 4;
                    pixels[to..to + side * 4].copy_from_slice(&texels[row * 16..row * 16 + side * 4]);
                }
            }
        }
        *done[index].lock().unwrap_or_else(|p| p.into_inner()) = encode_tile(&pixels, width, height, kind, srgb, quality);
    });
    let done: Vec<Vec<u8>> = done.into_iter().map(|tile| tile.into_inner().unwrap_or_else(|p| p.into_inner())).collect();
    let mut out = Vec::with_capacity(done.len() * 4 + done.iter().map(Vec::len).sum::<usize>());
    for tile in &done {
        out.extend_from_slice(&(tile.len() as u32).to_le_bytes());
    }
    for tile in &done {
        out.extend_from_slice(tile);
    }
    out
}

/// A pointer to where tiles put their blocks, each its own
struct Blocks(*mut u8);
unsafe impl Sync for Blocks {}

impl Blocks {
    fn at(&self, at: usize) -> *mut u8 {
        self.0.wrapping_add(at)
    }
}

/// Undoes encode_mip into new blocks of `kind`. False if `coded` can't be what it says it is. `out` may
/// be slow to read, as it's only written.
///
/// # Safety
/// `out` must be valid for writing `across * down * kind.block_bytes()` bytes.
#[allow(clippy::too_many_arguments)]
pub unsafe fn decode_mip(kind: Kind, srgb: bool, coded: &[u8], across: usize, down: usize, quality: &Quality, threads: usize, out: *mut u8) -> bool {
    let list = tiles(kind, across, down);
    let Some(table) = list.len().checked_mul(4).filter(|&table| table <= coded.len()) else { return false };
    let mut spans = Vec::with_capacity(list.len());
    let mut at = table;
    for length in coded[..table].chunks_exact(4) {
        let length = u32::from_le_bytes(length.try_into().unwrap()) as usize;
        let Some(end) = at.checked_add(length).filter(|&end| end <= coded.len()) else { return false };
        spans.push((at, end));
        at = end;
    }
    let (block, side) = (kind.block_bytes(), kind.block_side());
    let blocks = Blocks(out);
    on_threads(threads, list.len(), |index| {
        let (left, top, wide, high) = list[index];
        let (width, height) = (wide * side, high * side);
        let mut pixels = vec![0u8; width * height * 4];
        decode_tile(&coded[spans[index].0..spans[index].1], &mut pixels, width, height, kind, srgb, quality);
        let mut texels = [0u8; 64];
        for y in 0..high {
            for x in 0..wide {
                for row in 0..side {
                    let from = ((y * side + row) * width + x * side) * 4;
                    texels[row * 16..row * 16 + side * 4].copy_from_slice(&pixels[from..from + side * 4]);
                }
                let at = ((top + y) * across + left + x) * block;
                // Tiles don't overlap, and the caller says it's all there
                bc::encode_block(kind, &texels, unsafe { std::slice::from_raw_parts_mut(blocks.at(at), block) });
            }
        }
    });
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smooth areas and edges, as a texture has
    fn picture(width: usize, height: usize) -> Vec<u8> {
        let mut pixels = vec![255u8; width * height * 4];
        for y in 0..height {
            for x in 0..width {
                let pixel = &mut pixels[(y * width + x) * 4..][..4];
                pixel[0] = (x * 255 / width) as u8;
                pixel[1] = (y * 255 / height) as u8;
                pixel[2] = if (x / 32 + y / 32) % 2 == 0 { 200 } else { 60 };
            }
        }
        pixels
    }

    fn psnr(a: &[u8], b: &[u8]) -> f64 {
        let error: f64 = a.iter().zip(b).map(|(x, y)| (*x as f64 - *y as f64).powi(2)).sum::<f64>() / a.len() as f64;
        10.0 * (255.0 * 255.0 / error.max(1e-9)).log10()
    }

    #[test]
    fn splitting_undoes_exactly_at_any_size() {
        for (width, height) in [(1, 1), (2, 2), (3, 5), (7, 8), (8, 8), (16, 9), (33, 64), (256, 256), (1024, 512), (255, 1)] {
            let levels = sizes(width, height);
            let original: Vec<i32> = (0..width * height).map(|at| ((at * 37) % 251) as i32 - 128).collect();
            let mut plane = original.clone();
            split(&mut plane, width, &levels);
            join(&mut plane, width, &levels);
            assert_eq!(plane, original, "{width}x{height}");
        }
    }

    #[test]
    fn bands_cover_every_coefficient_once() {
        for (width, height) in [(1, 1), (9, 8), (64, 64), (33, 90), (1024, 1024)] {
            let levels = sizes(width, height);
            let mut seen = vec![0u8; width * height];
            for (left, top, w, h, _) in bands(&levels) {
                for y in top..top + h {
                    for x in left..left + w {
                        seen[y * width + x] += 1;
                    }
                }
            }
            assert!(seen.iter().all(|count| *count == 1), "{width}x{height}");
        }
    }

    #[test]
    fn a_tile_comes_back_close_with_any_kind_of_channels() {
        let (width, height) = (256, 192);
        let pixels = picture(width, height);
        let finest = Quality { bits: 1, extra: [0; LEVELS] };
        for (kind, srgb, least) in [(Kind::Bc7, true, 32.0), (Kind::Bc7, false, 40.0), (Kind::Bc1, true, 32.0), (Kind::Rgba8, false, 40.0)] {
            let coded = encode_tile(&pixels, width, height, kind, srgb, &finest);
            let mut back = vec![0u8; pixels.len()];
            decode_tile(&coded, &mut back, width, height, kind, srgb, &finest);
            let quality = psnr(&pixels, &back);
            assert!(quality >= least, "{kind:?} srgb {srgb}: {quality:.1} dB");
        }
    }

    #[test]
    fn alpha_is_kept_where_a_kind_has_it() {
        let (width, height) = (64, 64);
        let mut pixels = picture(width, height);
        for pixel in pixels.chunks_exact_mut(4) {
            pixel[3] = 40;
        }
        let coded = encode_tile(&pixels, width, height, Kind::Bc7, true, &QUALITY);
        let mut back = vec![0u8; pixels.len()];
        decode_tile(&coded, &mut back, width, height, Kind::Bc7, true, &QUALITY);
        assert!(back.chunks_exact(4).all(|pixel| (pixel[3] as i32 - 40).abs() <= 2));
    }

    fn blocks_of(kind: Kind, width: usize, height: usize) -> Vec<u8> {
        let pixels = picture(width, height);
        let (across, down) = (width / 4, height / 4);
        let mut out = vec![0u8; across * down * kind.block_bytes()];
        let mut texels = [0u8; 64];
        for y in 0..down {
            for x in 0..across {
                for row in 0..4 {
                    let from = ((y * 4 + row) * width + x * 4) * 4;
                    texels[row * 16..row * 16 + 16].copy_from_slice(&pixels[from..from + 16]);
                }
                let at = (y * across + x) * kind.block_bytes();
                bc::encode_block(kind, &texels, &mut out[at..at + kind.block_bytes()]);
            }
        }
        out
    }

    fn decoded(kind: Kind, blocks: &[u8], across: usize, down: usize) -> Vec<u8> {
        let mut pixels = vec![0u8; across * down * 16 * 4];
        let mut texels = [0u8; 64];
        for y in 0..down {
            for x in 0..across {
                bc::decode_block(kind, &blocks[(y * across + x) * kind.block_bytes()..][..kind.block_bytes()], &mut texels);
                for row in 0..4 {
                    let to = ((y * 4 + row) * across * 4 + x * 4) * 4;
                    pixels[to..to + 16].copy_from_slice(&texels[row * 16..row * 16 + 16]);
                }
            }
        }
        pixels
    }

    #[test]
    fn a_mip_goes_smaller_and_comes_back_near_it() {
        // Larger than a tile, so tiles join up
        let (width, height) = (1536, 1100);
        let (across, down) = (width / 4, height / 4);
        for (kind, srgb, smaller, least) in [(Kind::Bc7, true, 5, 28.0), (Kind::Bc1, true, 2, 26.0), (Kind::Bc5, false, 2, 26.0)] {
            let blocks = blocks_of(kind, across * 4, down * 4);
            let coded = encode_mip(kind, srgb, &blocks, across, down, &QUALITY, 4);
            assert!(coded.len() * smaller <= blocks.len(), "{kind:?}: {} from {}", coded.len(), blocks.len());
            let mut back = vec![0u8; blocks.len()];
            assert!(unsafe { decode_mip(kind, srgb, &coded, across, down, &QUALITY, 4, back.as_mut_ptr()) });
            let (before, after) = (decoded(kind, &blocks, across, down), decoded(kind, &back, across, down));
            let quality = psnr(&before, &after);
            assert!(quality >= least, "{kind:?}: {quality:.1} dB");
            // However many threads
            assert_eq!(encode_mip(kind, srgb, &blocks, across, down, &QUALITY, 1), coded);
        }
    }

    #[test]
    fn small_mips_and_textures_without_blocks_work() {
        for (across, down) in [(1, 1), (2, 1), (3, 5), (16, 16)] {
            let blocks = vec![0x55u8; across * down * 16];
            let coded = encode_mip(Kind::Bc7, true, &blocks, across, down, &QUALITY, 2);
            let mut back = vec![0u8; blocks.len()];
            assert!(unsafe { decode_mip(Kind::Bc7, true, &coded, across, down, &QUALITY, 2, back.as_mut_ptr()) });
        }
        let pixels = picture(1100, 40);
        let coded = encode_mip(Kind::Rgba8, false, &pixels, 1100, 40, &QUALITY, 2);
        let mut back = vec![0u8; pixels.len()];
        assert!(unsafe { decode_mip(Kind::Rgba8, false, &coded, 1100, 40, &QUALITY, 2, back.as_mut_ptr()) });
        assert!(psnr(&pixels, &back) > 28.0);
    }

    #[test]
    fn what_comes_off_the_network_cannot_reach_outside() {
        let (across, down) = (64, 64);
        let blocks = blocks_of(Kind::Bc7, across * 4, down * 4);
        let coded = encode_mip(Kind::Bc7, true, &blocks, across, down, &QUALITY, 2);
        let mut back = vec![0u8; blocks.len()];
        unsafe {
            // Cut short
            assert!(!decode_mip(Kind::Bc7, true, &coded[..coded.len() / 2], across, down, &QUALITY, 2, back.as_mut_ptr()));
            assert!(!decode_mip(Kind::Bc7, true, &[], across, down, &QUALITY, 2, back.as_mut_ptr()));
            // Nonsense in the bits only makes a wrong picture
            let mut garbage = coded.clone();
            for byte in &mut garbage[4..] {
                *byte = byte.wrapping_mul(167).wrapping_add(13);
            }
            assert!(decode_mip(Kind::Bc7, true, &garbage, across, down, &QUALITY, 2, back.as_mut_ptr()));
        }
    }
}
