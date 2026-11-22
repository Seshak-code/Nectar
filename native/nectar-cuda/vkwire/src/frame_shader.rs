//! frame.rs's encoder as compute shaders, so frames are coded on the GPU they were rendered on, which
//! has far more threads for it than the CPU beside it. They write the same bits, so the same decoder
//! reads them.
//!
//! Three passes, since a GPU runs threads in groups that must take both sides of any branch one of
//! them takes, and it's branches per coefficient that would cost the time:
//!
//!   compare    one invocation a tile: has it changed since the last frame?
//!   transform  one invocation an 8x8 block: its colour, cosine transform and division by the steps
//!              have no branches to speak of, and the results go to a work buffer, in writing order,
//!              with a mask of which aren't zero
//!   write      one invocation a tile: its bits. It only visits the coefficients the masks name.
//!
//! Each tile writes to a slot of its own, as a tile's coded length is only known once it's coded and
//! tiles run at once. Writing a word at a time to memory the CPU can read crosses PCIe a word at a
//! time, at about 540 MB/s, so the start of each slot, which is all most tiles need, is in the GPU's
//! own memory (the hot buffer), and is copied across in one go when the shaders are done. Only what
//! won't fit there goes straight to the output.
//!
//! The output is, in order: a u32 per tile, its length in bytes or 0 if it hasn't changed; the start of
//! each tile's slot; the rest of each. The first two are the hot buffer's copy. The CPU then only has
//! to gather what's in use (`gather`), which takes little.
//!
//! Bindings: 0 this frame's BGRA, 1 the last frame's, 2 the output, 3 the work buffer, 4 the hot
//! buffer. Push constants:
//! width, height, the step scale, 1 if there's no last frame to compare with, and the first plane of
//! this dispatch of the transform pass. Dispatch by `Pass`.

use crate::frame::{C1, C2, C3, C4, C5, C6, C7, TILE, ZIGZAG, step_scale};
use crate::spirv::{Buffer, Id, Push, Shader, Variable};

/// A slot's size. A coefficient takes at most 25 bits: orthonormal 8x8 cosine coefficients of 8-bit
/// pixels stay within about +/-1,300, and at the finest step of 1 that's 2 * log2(2 * 1,300) + 1
/// bits, with a bit for the zeros before it. 96 blocks of 64 of them is 19,200 bytes.
pub const SLOT_BYTES: usize = 20 << 10;
const SLOT_WORDS: u32 = (SLOT_BYTES / 4) as u32;
/// How much of a slot is in the GPU's memory. 1080p at the quality the link can carry at 120 frames
/// a second averages under 1 KB a tile.
pub const HOT_BYTES: usize = 4 << 10;
const HOT_WORDS: u32 = (HOT_BYTES / 4) as u32;
const COLD_WORDS: u32 = SLOT_WORDS - HOT_WORDS;
/// The bytes of push constants
pub const PUSH_BYTES: u32 = 20;
/// Values past this would make codes longer than the 31 bits one write can take, or not fit in 16
const LARGEST_VALUE: i32 = 16_000;
/// Blocks in a tile: 64 of luma and 16 each of chroma
const BLOCKS: u32 = 96;
/// In the work buffer, for a block: two words of mask, then 16 bits a coefficient
const BLOCK_WORDS: u32 = 2 + 32;

/// Tiles across and down
pub fn tiles(width: u32, height: u32) -> (u32, u32) {
    (width.div_ceil(TILE as u32), height.div_ceil(TILE as u32))
}

/// How large the output is
pub fn output_size(width: u32, height: u32) -> u64 {
    let (across, down) = tiles(width, height);
    let count = across as u64 * down as u64;
    count * 4 + count * SLOT_BYTES as u64
}

/// How large the hot buffer is, which is also how much of the output is copied from it
pub fn hot_size(width: u32, height: u32) -> u64 {
    let (across, down) = tiles(width, height);
    let count = across as u64 * down as u64;
    count * 4 + count * HOT_BYTES as u64
}

/// How large the work buffer is: a word a tile, then every block
pub fn work_size(width: u32, height: u32) -> u64 {
    let (across, down) = tiles(width, height);
    let count = across as u64 * down as u64;
    count * 4 + count * (BLOCKS * BLOCK_WORDS * 4) as u64
}

/// The push constants for a frame; the transform pass's first plane goes in the last
pub fn push_constants(width: u32, height: u32, quality: u32, first: bool, first_plane: u32) -> [u32; 5] {
    [width, height, step_scale(quality).to_bits(), first as u32, first_plane]
}

/// One dispatch of the frame's coding, in order; a barrier between shader writes and reads goes
/// between each of them
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pass {
    /// Which of `shaders()`
    pub shader: usize,
    pub first_plane: u32,
    pub groups: [u32; 3],
}

/// The dispatches for a frame of this size
pub fn passes(width: u32, height: u32) -> [Pass; 4] {
    let (across, down) = tiles(width, height);
    [
        Pass { shader: 0, first_plane: 0, groups: [across, down, 1] },
        Pass { shader: 1, first_plane: 0, groups: [across * 8, down * 8, 1] },
        Pass { shader: 1, first_plane: 1, groups: [across * 4, down * 4, 2] },
        Pass { shader: 2, first_plane: 0, groups: [across, down, 1] },
    ]
}

/// Makes a frame of frame.rs's format from the shaders' output, which is at least `output_size`
pub fn gather(output: &[u8], width: u32, height: u32, quality: u32) -> Vec<u8> {
    let (across, down) = tiles(width, height);
    let count = across as usize * down as usize;
    let length = |tile: usize| u32::from_le_bytes(output[tile * 4..tile * 4 + 4].try_into().unwrap()) as usize;
    let changed = (0..count).filter(|&tile| length(tile) != 0);
    let total: usize = changed.clone().map(|tile| 8 + length(tile)).sum();
    let mut out = Vec::with_capacity(16 + total);
    out.extend_from_slice(&width.to_le_bytes());
    out.extend_from_slice(&height.to_le_bytes());
    out.extend_from_slice(&quality.to_le_bytes());
    out.extend_from_slice(&(changed.clone().count() as u32).to_le_bytes());
    for tile in changed {
        let bytes = length(tile).min(SLOT_BYTES);
        let hot_start = count * 4 + tile * HOT_BYTES;
        out.extend_from_slice(&((tile % across as usize) as u16).to_le_bytes());
        out.extend_from_slice(&((tile / across as usize) as u16).to_le_bytes());
        out.extend_from_slice(&(bytes as u32).to_le_bytes());
        out.extend_from_slice(&output[hot_start..hot_start + bytes.min(HOT_BYTES)]);
        if bytes > HOT_BYTES {
            let cold_start = count * 4 + count * HOT_BYTES + tile * (SLOT_BYTES - HOT_BYTES);
            out.extend_from_slice(&output[cold_start..cold_start + bytes - HOT_BYTES]);
        }
    }
    out
}

/// The three shaders, as SPIR-V
pub fn shaders() -> [Vec<u32>; 3] {
    [compare_shader(), transform_shader(), write_shader()]
}

/// What every pass declares, at the same bindings
struct Common {
    current: Buffer,
    previous: Buffer,
    output: Buffer,
    work: Buffer,
    hot: Buffer,
    push: Push,
}

fn common(s: &mut Shader) -> Common {
    let current = s.storage_buffer(0);
    let previous = s.storage_buffer(1);
    let output = s.storage_buffer(2);
    let work = s.storage_buffer(3);
    let hot = s.storage_buffer(4);
    let push = s.push_constants(&[s.uint, s.uint, s.float, s.uint, s.uint]);
    Common { current, previous, output, work, hot, push }
}

/// Tiles across, and in all
fn tile_counts(s: &mut Shader, push: &Push) -> (Id, Id) {
    let width = s.read_push(push, 0);
    let height = s.read_push(push, 1);
    let (tile, last) = (s.u(TILE as u32), s.u(TILE as u32 - 1));
    let sum = s.add(width, last);
    let across = s.div(sum, tile);
    let sum = s.add(height, last);
    let down = s.div(sum, tile);
    let count = s.mul(across, down);
    (across, count)
}

fn compare_shader() -> Vec<u32> {
    let mut s = Shader::new();
    let c = common(&mut s);
    s.main(|s| {
        let tile = s.u(TILE as u32);
        let frame_width = s.read_push(&c.push, 0);
        let frame_height = s.read_push(&c.push, 1);
        let first = s.read_push(&c.push, 3);
        let (across, _) = tile_counts(s, &c.push);
        let tile_x = s.invocation(0);
        let tile_y = s.invocation(1);
        let ox = s.mul(tile_x, tile);
        let oy = s.mul(tile_y, tile);
        let room_x = s.sub(frame_width, ox);
        let room_y = s.sub(frame_height, oy);
        let tw = s.umin(tile, room_x);
        let th = s.umin(tile, room_y);
        let line = s.mul(tile_y, across);
        let index = s.add(line, tile_x);
        let differs = s.local_of(s.uint, first);
        let no = s.u(0);
        let row = s.local_of(s.uint, no);
        s.while_(
            |s| {
                let at = s.load(row);
                let inside = s.lt(at, th);
                let found = s.load(differs);
                let zero = s.u(0);
                let none = s.eq(found, zero);
                s.both(inside, none)
            },
            |s| {
                let at = s.load(row);
                let y = s.add(oy, at);
                let line = s.mul(y, frame_width);
                let start = s.add(line, ox);
                let zero = s.u(0);
                let column = s.local_of(s.uint, zero);
                s.while_(
                    |s| {
                        let at = s.load(column);
                        let inside = s.lt(at, tw);
                        let found = s.load(differs);
                        let zero = s.u(0);
                        let none = s.eq(found, zero);
                        s.both(inside, none)
                    },
                    |s| {
                        let at = s.load(column);
                        let word_index = s.add(start, at);
                        let now = s.read(c.current, word_index);
                        let then = s.read(c.previous, word_index);
                        let changed = s.ne(now, then);
                        s.if_(changed, |s| {
                            let one = s.u(1);
                            s.store(differs, one);
                        });
                        let one = s.u(1);
                        let next = s.add(at, one);
                        s.store(column, next);
                    },
                );
                let one = s.u(1);
                let next = s.add(at, one);
                s.store(row, next);
            },
        );
        let found = s.load(differs);
        s.write(c.work, index, found);
    });
    s.finish()
}

fn transform_shader() -> Vec<u32> {
    let mut s = Shader::new();
    let (uint, float, void) = (s.uint, s.float, s.void);
    let c = common(&mut s);
    let zigzag: Vec<u32> = ZIGZAG.iter().map(|&at| at as u32).collect();
    let zigzag = s.table(uint, 64, Some(&zigzag));
    // One over each coefficient's step, before the scale: a division costs a GPU far more than this
    let inverse_of = |base: f32, slope: f32| -> Vec<u32> {
        (0..64u32).map(|at| (1.0 / (base + ((at >> 3) + (at & 7)) as f32 * slope)).to_bits()).collect()
    };
    let inverse_luma = s.table(float, 64, Some(&inverse_of(1.0, 0.5)));
    let inverse_chroma = s.table(float, 64, Some(&inverse_of(2.0, 1.0)));
    let block = s.table(float, 64, None);
    let origin_x = s.private(uint);
    let origin_y = s.private(uint);
    let tile_width = s.private(uint);
    let tile_height = s.private(uint);

    // A pixel of the frame, at (x, y) in the tile
    let pixel = s.function(uint, &[uint, uint], |s, parameters| {
        let frame_width = s.read_push(&c.push, 0);
        let ox = s.load(origin_x);
        let oy = s.load(origin_y);
        let y = s.add(oy, parameters[1]);
        let line = s.mul(y, frame_width);
        let x = s.add(ox, parameters[0]);
        let at = s.add(line, x);
        Some(s.read(c.current, at))
    });

    let channel = |s: &mut Shader, word: Id, shift: u32| -> Id {
        let bits = s.u(shift);
        let moved = s.shr(word, bits);
        let mask = s.u(255);
        let byte = s.and(moved, mask);
        s.to_float(byte)
    };

    // A sample of a plane: 0 is Y, at full resolution; 1 and 2 are Co and Cg, at half
    let sample = s.function(float, &[uint, uint, uint], |s, parameters| {
        let (plane, x, y) = (parameters[0], parameters[1], parameters[2]);
        let result = s.local(float);
        let zero = s.u(0);
        let half_value = s.f(0.5);
        let is_luma = s.eq(plane, zero);
        s.if_else(
            is_luma,
            |s| {
                let word = s.call(uint, pixel, &[x, y]);
                let b = channel(s, word, 0);
                let g = channel(s, word, 8);
                let r = channel(s, word, 16);
                let sum = s.fadd(r, b);
                let t = s.fmul(sum, half_value);
                let cg = s.fsub(g, t);
                let half_cg = s.fmul(cg, half_value);
                let luma = s.fadd(t, half_cg);
                let middle = s.f(128.0);
                let centred = s.fsub(luma, middle);
                s.store(result, centred);
            },
            |s| {
                // A pixel past the tile's edge repeats the last, which averages as if it weren't there
                let one = s.u(1);
                let x0 = s.shl(x, one);
                let y0 = s.shl(y, one);
                let tw = s.load(tile_width);
                let th = s.load(tile_height);
                let last_x = s.sub(tw, one);
                let last_y = s.sub(th, one);
                let x_next = s.add(x0, one);
                let y_next = s.add(y0, one);
                let x1 = s.umin(x_next, last_x);
                let y1 = s.umin(y_next, last_y);
                let (mut b, mut g, mut r) = (s.f(0.0), s.f(0.0), s.f(0.0));
                for (px, py) in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)] {
                    let word = s.call(uint, pixel, &[px, py]);
                    let value = channel(s, word, 0);
                    b = s.fadd(b, value);
                    let value = channel(s, word, 8);
                    g = s.fadd(g, value);
                    let value = channel(s, word, 16);
                    r = s.fadd(r, value);
                }
                let quarter = s.f(0.25);
                let b = s.fmul(b, quarter);
                let g = s.fmul(g, quarter);
                let r = s.fmul(r, quarter);
                let is_co = s.eq(plane, one);
                s.if_else(
                    is_co,
                    |s| {
                        let co = s.fsub(r, b);
                        s.store(result, co);
                    },
                    |s| {
                        let sum = s.fadd(r, b);
                        let t = s.fmul(sum, half_value);
                        let cg = s.fsub(g, t);
                        s.store(result, cg);
                    },
                );
            },
        );
        Some(s.load(result))
    });

    // An 8-point cosine transform in place, of `block` from `start`, `stride` apart, by sums and
    // differences as frame.rs's forward_8
    let forward_8 = s.function(void, &[uint, uint], |s, parameters| {
        let (start, stride) = (parameters[0], parameters[1]);
        let at: Vec<Id> = (0..8)
            .map(|index| {
                let index = s.u(index);
                let distance = s.mul(index, stride);
                s.add(start, distance)
            })
            .collect();
        let x: Vec<Id> = at.iter().map(|&at| s.get(block, at)).collect();
        let (s07, s16, s25, s34) = (s.fadd(x[0], x[7]), s.fadd(x[1], x[6]), s.fadd(x[2], x[5]), s.fadd(x[3], x[4]));
        let (d07, d16, d25, d34) = (s.fsub(x[0], x[7]), s.fsub(x[1], x[6]), s.fsub(x[2], x[5]), s.fsub(x[3], x[4]));
        let (a, b, c, d) = (s.fadd(s07, s34), s.fadd(s16, s25), s.fsub(s16, s25), s.fsub(s07, s34));
        let (c1, c2, c3, c4, c5, c6, c7) = (s.f(C1), s.f(C2), s.f(C3), s.f(C4), s.f(C5), s.f(C6), s.f(C7));
        let four = |s: &mut Shader, terms: [(Id, Id, bool); 4]| -> Id {
            let mut total = s.fmul(terms[0].0, terms[0].1);
            for (value, constant, negative) in &terms[1..] {
                let product = s.fmul(*value, *constant);
                total = if *negative { s.fsub(total, product) } else { s.fadd(total, product) };
            }
            total
        };
        let sum = s.fadd(a, b);
        let x0 = s.fmul(sum, c4);
        let x1 = four(s, [(d07, c1, false), (d16, c3, false), (d25, c5, false), (d34, c7, false)]);
        let (product, other) = (s.fmul(d, c2), s.fmul(c, c6));
        let x2 = s.fadd(product, other);
        let x3 = four(s, [(d07, c3, false), (d16, c7, true), (d25, c1, true), (d34, c5, true)]);
        let difference = s.fsub(a, b);
        let x4 = s.fmul(difference, c4);
        let x5 = four(s, [(d07, c5, false), (d16, c1, true), (d25, c7, false), (d34, c3, false)]);
        let (product, other) = (s.fmul(d, c6), s.fmul(c, c2));
        let x6 = s.fsub(product, other);
        let x7 = four(s, [(d07, c7, false), (d16, c5, true), (d25, c3, false), (d34, c1, true)]);
        for (&at, value) in at.iter().zip([x0, x1, x2, x3, x4, x5, x6, x7]) {
            s.set(block, at, value);
        }
        None
    });

    // The coefficient at a zigzag position divided by its step, as frame.rs's `steps` has them, in 16
    // bits. `inverse_scale` is one over the step scale.
    let quantized = s.function(s.int, &[uint, s.boolean, float], |s, parameters| {
        let (position, is_luma, inverse_scale) = (parameters[0], parameters[1], parameters[2]);
        let at = s.get(zigzag, position);
        let luma = s.get(inverse_luma, at);
        let chroma = s.get(inverse_chroma, at);
        let inverse = s.select(s.float, is_luma, luma, chroma);
        let value = s.get(block, at);
        let quotient = s.fmul(value, inverse);
        let quotient = s.fmul(quotient, inverse_scale);
        let rounded = s.round(quotient);
        let (low, high) = (s.i(-LARGEST_VALUE), s.i(LARGEST_VALUE));
        Some(s.iclamp(rounded, low, high))
    });

    s.main(|s| {
        let tile = s.u(TILE as u32);
        let frame_width = s.read_push(&c.push, 0);
        let frame_height = s.read_push(&c.push, 1);
        let first_plane = s.read_push(&c.push, 4);
        let (across, count) = tile_counts(s, &c.push);
        let block_x = s.invocation(0);
        let block_y = s.invocation(1);
        let layer = s.invocation(2);
        let plane = s.add(layer, first_plane);
        let zero = s.u(0);
        let one = s.u(1);
        let is_luma = s.eq(plane, zero);
        // Blocks across a tile: 8 of luma, 4 of chroma
        let (three, two) = (s.u(3), s.u(2));
        let log_blocks = s.select(s.uint, is_luma, three, two);
        let tile_x = s.shr(block_x, log_blocks);
        let tile_y = s.shr(block_y, log_blocks);
        let blocks_across = s.shl(one, log_blocks);
        let last_block = s.sub(blocks_across, one);
        let local_x = s.and(block_x, last_block);
        let local_y = s.and(block_y, last_block);
        let line = s.mul(tile_y, across);
        let tile_index = s.add(line, tile_x);
        let changed = s.read(c.work, tile_index);
        let has_changed = s.ne(changed, zero);
        s.if_(has_changed, |s| {
            let ox = s.mul(tile_x, tile);
            let oy = s.mul(tile_y, tile);
            s.store(origin_x, ox);
            s.store(origin_y, oy);
            let room_x = s.sub(frame_width, ox);
            let room_y = s.sub(frame_height, oy);
            let tw = s.umin(tile, room_x);
            let th = s.umin(tile, room_y);
            s.store(tile_width, tw);
            s.store(tile_height, th);
            let half_w = s.add(tw, one);
            let half_w = s.shr(half_w, one);
            let half_h = s.add(th, one);
            let half_h = s.shr(half_h, one);
            let plane_w = s.select(s.uint, is_luma, tw, half_w);
            let plane_h = s.select(s.uint, is_luma, th, half_h);
            let left = s.shl(local_x, three);
            let top = s.shl(local_y, three);
            let inside_x = s.lt(left, plane_w);
            let inside_y = s.lt(top, plane_h);
            let inside = s.both(inside_x, inside_y);
            s.if_(inside, |s| {
                let eight = s.u(8);
                let last_x = s.sub(plane_w, one);
                let last_y = s.sub(plane_h, one);
                s.for_(0, eight, 1, |s, y| {
                    let three = s.u(3);
                    let line = s.shl(y, three);
                    let down = s.add(top, y);
                    let row = s.umin(down, last_y);
                    s.for_(0, eight, 1, |s, x| {
                        let across_block = s.add(left, x);
                        let column = s.umin(across_block, last_x);
                        let value = s.call(s.float, sample, &[plane, column, row]);
                        let at = s.add(line, x);
                        s.set(block, at, value);
                    });
                });
                s.for_(0, eight, 1, |s, y| {
                    let three = s.u(3);
                    let start = s.shl(y, three);
                    let one = s.u(1);
                    s.call(s.void, forward_8, &[start, one]);
                });
                s.for_(0, eight, 1, |s, x| {
                    let eight = s.u(8);
                    s.call(s.void, forward_8, &[x, eight]);
                });

                let scale = s.read_push(&c.push, 2);
                let whole = s.f(1.0);
                let inverse_scale = s.fdiv(whole, scale);

                // Where this block's results go: luma's 64, then Co's 16, then Cg's 16
                let in_tile = s.shl(local_y, log_blocks);
                let in_tile = s.add(in_tile, local_x);
                let (sixty_four, sixteen) = (s.u(64), s.u(16));
                let chroma_start = s.sub(plane, one);
                let chroma_start = s.mul(chroma_start, sixteen);
                let chroma_start = s.add(sixty_four, chroma_start);
                let first_block = s.select(s.uint, is_luma, zero, chroma_start);
                let block_index = s.add(first_block, in_tile);
                let blocks = s.u(BLOCKS);
                let before = s.mul(tile_index, blocks);
                let before = s.add(before, block_index);
                let block_words = s.u(BLOCK_WORDS);
                let before = s.mul(before, block_words);
                let start = s.add(count, before);

                let mask_low = s.local_of(s.uint, zero);
                let mask_high = s.local_of(s.uint, zero);
                let thirty_two = s.u(32);
                s.for_(0, thirty_two, 1, |s, pair| {
                    let (one, two) = (s.u(1), s.u(2));
                    let first_position = s.shl(pair, one);
                    let second_position = s.add(first_position, one);
                    let v0 = s.call(s.int, quantized, &[first_position, is_luma, inverse_scale]);
                    let v1 = s.call(s.int, quantized, &[second_position, is_luma, inverse_scale]);
                    let (low_bits, sixteen) = (s.u(0xFFFF), s.u(16));
                    let u0 = s.as_uint(v0);
                    let u1 = s.as_uint(v1);
                    let low = s.and(u0, low_bits);
                    let high = s.shl(u1, sixteen);
                    let word = s.or(low, high);
                    let at = s.add(start, two);
                    let at = s.add(at, pair);
                    s.write(c.work, at, word);
                    let zero = s.u(0);
                    let zero_int = s.i(0);
                    let nonzero0 = s.ne(v0, zero_int);
                    let nonzero1 = s.ne(v1, zero_int);
                    let bit0 = s.select(s.uint, nonzero0, one, zero);
                    let bit1 = s.select(s.uint, nonzero1, two, zero);
                    let bits = s.or(bit0, bit1);
                    let is_low = s.lt(pair, sixteen);
                    let thirty_one = s.u(31);
                    let shift = s.and(first_position, thirty_one);
                    let moved = s.shl(bits, shift);
                    let was_low = s.load(mask_low);
                    let was_high = s.load(mask_high);
                    let joined_low = s.or(was_low, moved);
                    let joined_high = s.or(was_high, moved);
                    let next_low = s.select(s.uint, is_low, joined_low, was_low);
                    let next_high = s.select(s.uint, is_low, was_high, joined_high);
                    s.store(mask_low, next_low);
                    s.store(mask_high, next_high);
                });
                // The first is the DC, which always goes
                let low = s.load(mask_low);
                let without_dc = s.u(!1);
                let low = s.and(low, without_dc);
                s.write(c.work, start, low);
                let at = s.add(start, one);
                let high = s.load(mask_high);
                s.write(c.work, at, high);
            });
        });
    });
    s.finish()
}

/// Where a tile writes: words so far, and where its slot's two parts start
#[derive(Clone, Copy)]
struct Slot {
    hot: Buffer,
    output: Buffer,
    word_at: Variable,
    hot_start: Variable,
    cold_start: Variable,
}

/// Stores a finished word, its first bit in the first byte's top, if the slot has room
fn write_word(s: &mut Shader, word: Id, slot: Slot) {
    let at = s.load(slot.word_at);
    let (hot_words, slot_words) = (s.u(HOT_WORDS), s.u(SLOT_WORDS));
    let is_hot = s.lt(at, hot_words);
    let room = s.lt(at, slot_words);
    s.if_(room, |s| {
        let (byte, high, eight, twenty_four) = (s.u(0xFF), s.u(0xFF00), s.u(8), s.u(24));
        let first = s.shr(word, twenty_four);
        let moved = s.shr(word, eight);
        let second = s.and(moved, high);
        let moved = s.shl(word, eight);
        let high_third = s.u(0xFF_0000);
        let third = s.and(moved, high_third);
        let low = s.and(word, byte);
        let fourth = s.shl(low, twenty_four);
        let swapped = s.or(first, second);
        let swapped = s.or(swapped, third);
        let swapped = s.or(swapped, fourth);
        s.if_else(
            is_hot,
            |s| {
                let start = s.load(slot.hot_start);
                let place = s.add(start, at);
                s.write(slot.hot, place, swapped);
            },
            |s| {
                let start = s.load(slot.cold_start);
                let beyond = s.sub(at, hot_words);
                let place = s.add(start, beyond);
                s.write(slot.output, place, swapped);
            },
        );
    });
    let one = s.u(1);
    let next = s.add(at, one);
    s.store(slot.word_at, next);
}

fn write_shader() -> Vec<u32> {
    let mut s = Shader::new();
    let (uint, int, void) = (s.uint, s.int, s.void);
    let c = common(&mut s);
    let held = s.private(uint);
    let used = s.private(uint);
    let slot = Slot { hot: c.hot, output: c.output, word_at: s.private(uint), hot_start: s.private(uint), cold_start: s.private(uint) };
    let tile_base = s.private(uint);

    // Writes the low `width` bits of `value`, most significant first; 1 to 31 of them. Both outcomes
    // are worked out, as a branch would have every thread in the group take both anyway.
    let put = s.function(void, &[uint, uint], |s, parameters| {
        let (value, width) = (parameters[0], parameters[1]);
        let now = s.load(used);
        let before = s.load(held);
        let total = s.add(now, width);
        let (word_bits, mask_bits, one) = (s.u(32), s.u(31), s.u(1));
        let fits = s.lt(total, word_bits);
        let moved = s.shl(before, width);
        let joined = s.or(moved, value);
        // If it doesn't fit: what's room for finishes the word
        let left = s.sub(total, word_bits);
        let left = s.and(left, mask_bits);
        let room = s.sub(word_bits, now);
        let room = s.and(room, mask_bits);
        let top = s.shr(value, left);
        let moved = s.shl(before, room);
        let finished = s.or(moved, top);
        let mask = s.shl(one, left);
        let mask = s.sub(mask, one);
        let rest = s.and(value, mask);
        let next_held = s.select(uint, fits, joined, rest);
        let next_used = s.select(uint, fits, total, left);
        s.store(held, next_held);
        s.store(used, next_used);
        let full = s.not(fits);
        s.if_(full, |s| write_word(s, finished, slot));
        None
    });

    // Exponential Golomb: small values short
    let number = s.function(void, &[uint], |s, parameters| {
        let one = s.u(1);
        let shifted = s.add(parameters[0], one);
        let top = s.msb(shifted);
        let doubled = s.shl(top, one);
        let width = s.add(doubled, one);
        s.call(void, put, &[shifted, width]);
        None
    });

    let signed = s.function(void, &[int], |s, parameters| {
        let value = parameters[0];
        let bits = s.as_uint(value);
        let one = s.u(1);
        let doubled = s.shl(bits, one);
        let sign = s.sign_mask(value);
        let folded = s.xor(doubled, sign);
        s.call(void, number, &[folded]);
        None
    });

    // The coefficient at a position, from a block's words
    let coefficient = s.function(int, &[uint, uint], |s, parameters| {
        let (start, position) = (parameters[0], parameters[1]);
        let (one, two, sixteen) = (s.u(1), s.u(2), s.u(16));
        let pair = s.shr(position, one);
        let at = s.add(start, two);
        let at = s.add(at, pair);
        let word = s.read(c.work, at);
        // Up to the top of the word, then back down with its sign
        let odd = s.and(position, one);
        let shift = s.mul(odd, sixteen);
        let up = s.xor(shift, sixteen);
        let raised = s.shl(word, up);
        Some(s.arithmetic_shift_16(raised))
    });

    // Codes a plane's blocks: `first_block` is where they start in the tile, of `across` a row
    let code_plane = s.function(void, &[uint, uint, uint, uint], |s, parameters| {
        let (width, height, first_block, log_across) = (parameters[0], parameters[1], parameters[2], parameters[3]);
        let zero_int = s.i(0);
        let previous_dc = s.local_of(int, zero_int);
        let last_position = s.local(uint);
        s.for_(0, height, 8, |s, top| {
            s.for_(0, width, 8, |s, left| {
                let three = s.u(3);
                let row = s.shr(top, three);
                let row = s.shl(row, log_across);
                let column = s.shr(left, three);
                let index = s.add(first_block, row);
                let index = s.add(index, column);
                let block_words = s.u(BLOCK_WORDS);
                let offset = s.mul(index, block_words);
                let base = s.load(tile_base);
                let start = s.add(base, offset);
                let zero = s.u(0);
                let dc = s.call(int, coefficient, &[start, zero]);
                let before = s.load(previous_dc);
                let difference = s.isub(dc, before);
                s.call(void, signed, &[difference]);
                s.store(previous_dc, dc);
                s.store(last_position, zero);

                // The two words of the mask: each set bit is a coefficient, and the zeros before it
                // are the distance from the last
                let one = s.u(1);
                let mask_at = s.add(start, one);
                let low = s.read(c.work, start);
                let high = s.read(c.work, mask_at);
                for (word, offset) in [(low, 0u32), (high, 32u32)] {
                    let mask = s.local_of(uint, word);
                    let offset = s.u(offset);
                    s.while_(
                        |s| {
                            let bits = s.load(mask);
                            let zero = s.u(0);
                            s.ne(bits, zero)
                        },
                        |s| {
                            let bits = s.load(mask);
                            let bit = s.lsb(bits);
                            let position = s.add(offset, bit);
                            let last = s.load(last_position);
                            let one = s.u(1);
                            let distance = s.sub(position, last);
                            let zeros = s.sub(distance, one);
                            s.call(void, number, &[zeros]);
                            let value = s.call(int, coefficient, &[start, position]);
                            s.call(void, signed, &[value]);
                            s.store(last_position, position);
                            let lowered = s.sub(bits, one);
                            let rest = s.and(bits, lowered);
                            s.store(mask, rest);
                        },
                    );
                }
                // A run reaching the end says so
                let last = s.load(last_position);
                let end = s.u(63);
                let short = s.lt(last, end);
                s.if_(short, |s| {
                    let zeros = s.sub(end, last);
                    s.call(void, number, &[zeros]);
                });
            });
        });
        None
    });

    s.main(|s| {
        let tile = s.u(TILE as u32);
        let frame_width = s.read_push(&c.push, 0);
        let frame_height = s.read_push(&c.push, 1);
        let (across, count) = tile_counts(s, &c.push);
        let tile_x = s.invocation(0);
        let tile_y = s.invocation(1);
        let line = s.mul(tile_y, across);
        let index = s.add(line, tile_x);
        let changed = s.read(c.work, index);
        let zero = s.u(0);
        let is_same = s.eq(changed, zero);
        s.if_else(
            is_same,
            |s| {
                let zero = s.u(0);
                s.write(c.hot, index, zero);
            },
            |s| {
                let ox = s.mul(tile_x, tile);
                let oy = s.mul(tile_y, tile);
                let room_x = s.sub(frame_width, ox);
                let room_y = s.sub(frame_height, oy);
                let tw = s.umin(tile, room_x);
                let th = s.umin(tile, room_y);

                let (hot_words, cold_words) = (s.u(HOT_WORDS), s.u(COLD_WORDS));
                let offset = s.mul(index, hot_words);
                let hot_start = s.add(count, offset);
                s.store(slot.hot_start, hot_start);
                let all_hot = s.mul(count, hot_words);
                let offset = s.mul(index, cold_words);
                let cold_start = s.add(count, all_hot);
                let cold_start = s.add(cold_start, offset);
                s.store(slot.cold_start, cold_start);
                let zero = s.u(0);
                s.store(slot.word_at, zero);
                s.store(held, zero);
                s.store(used, zero);
                let (blocks, block_words) = (s.u(BLOCKS), s.u(BLOCK_WORDS));
                let before = s.mul(index, blocks);
                let before = s.mul(before, block_words);
                let start = s.add(count, before);
                s.store(tile_base, start);

                let (one, two, three) = (s.u(1), s.u(2), s.u(3));
                s.call(void, code_plane, &[tw, th, zero, three]);
                let cw = s.add(tw, one);
                let cw = s.shr(cw, one);
                let ch = s.add(th, one);
                let ch = s.shr(ch, one);
                let (sixty_four, eighty) = (s.u(64), s.u(80));
                s.call(void, code_plane, &[cw, ch, sixty_four, two]);
                s.call(void, code_plane, &[cw, ch, eighty, two]);

                // What's left of the last word, padded
                let tail = s.load(used);
                let any = s.ne(tail, zero);
                s.if_(any, |s| {
                    let word_bits = s.u(32);
                    let pad = s.sub(word_bits, tail);
                    let before = s.load(held);
                    let word = s.shl(before, pad);
                    write_word(s, word, slot);
                });
                let written = s.load(slot.word_at);
                // Whole words before the last, then the bytes of it in use
                let bytes = s.shl(written, two);
                let seven = s.u(7);
                let tail_bytes = s.add(tail, seven);
                let tail_bytes = s.shr(tail_bytes, three);
                let four = s.u(4);
                let unused = s.sub(four, tail_bytes);
                let trimmed = s.sub(bytes, unused);
                let length = s.select(uint, any, trimmed, bytes);
                s.write(c.hot, index, length);
            },
        );
    });
    s.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shaders_are_well_formed() {
        for words in shaders() {
            assert_eq!(words[0], 0x0723_0203);
            let mut at = 5;
            while at < words.len() {
                let length = (words[at] >> 16) as usize;
                assert!(length > 0 && at + length <= words.len(), "at word {at}");
                at += length;
            }
            assert_eq!(at, words.len());
        }
    }

    #[test]
    fn gathering_leaves_out_tiles_that_havent_changed() {
        let (width, height) = (130u32, 70u32);
        assert_eq!(tiles(width, height), (3, 2));
        let mut output = vec![0u8; output_size(width, height) as usize];
        let tile = 4;
        output[tile * 4] = 3;
        let start = 6 * 4 + tile * HOT_BYTES;
        output[start..start + 3].copy_from_slice(&[9, 8, 7]);
        let frame = gather(&output, width, height, 70);
        assert_eq!(u32::from_le_bytes(frame[12..16].try_into().unwrap()), 1);
        assert_eq!(&frame[16..20], &[1, 0, 1, 0]);
        assert_eq!(&frame[20..24], &[3, 0, 0, 0]);
        assert_eq!(&frame[24..], &[9, 8, 7]);
    }

    #[test]
    fn a_long_tile_is_gathered_from_both_parts() {
        let (width, height) = (130u32, 70u32);
        let mut output = vec![0u8; output_size(width, height) as usize];
        let tile = 2;
        let length = HOT_BYTES + 5;
        output[tile * 4..tile * 4 + 4].copy_from_slice(&(length as u32).to_le_bytes());
        let hot = 6 * 4 + tile * HOT_BYTES;
        output[hot..hot + HOT_BYTES].fill(1);
        let cold = 6 * 4 + 6 * HOT_BYTES + tile * (SLOT_BYTES - HOT_BYTES);
        output[cold..cold + 5].fill(2);
        let frame = gather(&output, width, height, 70);
        assert_eq!(&frame[16..20], &[2, 0, 0, 0]);
        assert_eq!(u32::from_le_bytes(frame[20..24].try_into().unwrap()) as usize, length);
        assert!(frame[24..24 + HOT_BYTES].iter().all(|&byte| byte == 1));
        assert_eq!(&frame[24 + HOT_BYTES..], &[2; 5]);
        assert_eq!(hot_size(width, height), 6 * 4 + 6 * HOT_BYTES as u64);
    }

    #[test]
    fn constants_carry_the_frame() {
        let [width, height, scale, first, plane] = push_constants(640, 480, 78, true, 1);
        assert_eq!((width, height, first, plane), (640, 480, 1, 1));
        assert!((f32::from_bits(scale) - 2.0).abs() < 0.01);
    }

    #[test]
    fn every_block_is_visited_once() {
        let passes = passes(130, 70);
        assert_eq!(passes[1].groups, [24, 16, 1]);
        assert_eq!(passes[2].groups, [12, 8, 2]);
        assert_eq!(passes[2].first_plane, 1);
        assert_eq!(work_size(64, 64), 4 + 96 * 34 * 4);
    }
}
