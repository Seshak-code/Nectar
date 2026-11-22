//! Block-compressed textures on the CPU: every texel a game's BC1, BC2, BC3, BC4, BC5 or BC7 blocks hold,
//! and new blocks from texels. wavelet.rs codes a mip as texels, so one end has to decode what the game
//! gave and the other make blocks the image can take.
//!
//! The decoders follow the formats' public specifications, and `vk-probe --bc-check` compares them with
//! a real GPU's on random blocks. The encoders are quick and plain: BC7 only ever uses mode 6, its one
//! subset with colour and alpha together. What they make stands in for the real blocks for a moment.

#![allow(clippy::chunks_exact_to_as_chunks, clippy::needless_range_loop)]

use nectar_vk_sys::*;

/// What a format's blocks are
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Kind {
    Bc1,
    Bc2,
    Bc3,
    Bc4,
    Bc5,
    Bc7,
    /// 8 bits each of red, green, blue and alpha, not compressed
    Rgba8,
}

impl Kind {
    /// What this module handles of a format, and whether its colours are sRGB
    pub fn of(format: VkFormat) -> Option<(Kind, bool)> {
        Some(match format {
            // BC1 with one bit of alpha is left out: the encoder here never makes a texel transparent
            VK_FORMAT_BC1_RGB_UNORM_BLOCK => (Kind::Bc1, false),
            VK_FORMAT_BC1_RGB_SRGB_BLOCK => (Kind::Bc1, true),
            VK_FORMAT_BC2_UNORM_BLOCK => (Kind::Bc2, false),
            VK_FORMAT_BC2_SRGB_BLOCK => (Kind::Bc2, true),
            VK_FORMAT_BC3_UNORM_BLOCK => (Kind::Bc3, false),
            VK_FORMAT_BC3_SRGB_BLOCK => (Kind::Bc3, true),
            VK_FORMAT_BC4_UNORM_BLOCK => (Kind::Bc4, false),
            VK_FORMAT_BC5_UNORM_BLOCK => (Kind::Bc5, false),
            VK_FORMAT_BC7_UNORM_BLOCK => (Kind::Bc7, false),
            VK_FORMAT_BC7_SRGB_BLOCK => (Kind::Bc7, true),
            VK_FORMAT_R8G8B8A8_UNORM => (Kind::Rgba8, false),
            VK_FORMAT_R8G8B8A8_SRGB => (Kind::Rgba8, true),
            _ => return None,
        })
    }

    /// Bytes a block
    pub fn block_bytes(self) -> usize {
        match self {
            Kind::Bc1 | Kind::Bc4 => 8,
            Kind::Rgba8 => 4,
            _ => 16,
        }
    }

    /// Texels a block has across and down
    pub fn block_side(self) -> usize {
        if self == Kind::Rgba8 { 1 } else { 4 }
    }

    /// Channels that hold anything
    pub fn channels(self) -> usize {
        match self {
            Kind::Bc4 => 1,
            Kind::Bc5 => 2,
            Kind::Bc1 => 3,
            _ => 4,
        }
    }
}

fn from_565(color: u16) -> [i32; 3] {
    let (r, g, b) = ((color >> 11) as i32 & 31, (color >> 5) as i32 & 63, color as i32 & 31);
    [(r << 3) | (r >> 2), (g << 2) | (g >> 4), (b << 3) | (b >> 2)]
}

/// Colour blocks of BC1, BC2 and BC3. BC2 and BC3 always have four colours.
fn decode_colour(block: &[u8], out: &mut [u8; 64], one_bit_alpha: bool) {
    let (c0, c1) = (u16::from_le_bytes([block[0], block[1]]), u16::from_le_bytes([block[2], block[3]]));
    let (a, b) = (from_565(c0), from_565(c1));
    let mix = |m: i32, n: i32, d: i32| [(m * a[0] + n * b[0]) / d, (m * a[1] + n * b[1]) / d, (m * a[2] + n * b[2]) / d];
    let (palette, last_alpha) = if c0 > c1 || !one_bit_alpha {
        ([a, b, mix(2, 1, 3), mix(1, 2, 3)], 255)
    } else {
        ([a, b, mix(1, 1, 2), [0, 0, 0]], 0)
    };
    let bits = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    for (texel, pixel) in out.chunks_exact_mut(4).enumerate() {
        let index = (bits >> (texel * 2)) as usize & 3;
        for channel in 0..3 {
            pixel[channel] = palette[index][channel] as u8;
        }
        pixel[3] = if index == 3 { last_alpha } else { 255 };
    }
}

/// A BC4 block's values
fn decode_single(block: &[u8], out: &mut [u8; 64], channel: usize) {
    let (a0, a1) = (block[0] as i32, block[1] as i32);
    let palette = if a0 > a1 {
        [a0, a1, (6 * a0 + a1) / 7, (5 * a0 + 2 * a1) / 7, (4 * a0 + 3 * a1) / 7, (3 * a0 + 4 * a1) / 7, (2 * a0 + 5 * a1) / 7, (a0 + 6 * a1) / 7]
    } else {
        [a0, a1, (4 * a0 + a1) / 5, (3 * a0 + 2 * a1) / 5, (2 * a0 + 3 * a1) / 5, (a0 + 4 * a1) / 5, 0, 255]
    };
    let bits = u64::from_le_bytes([block[2], block[3], block[4], block[5], block[6], block[7], 0, 0]);
    for texel in 0..16 {
        out[texel * 4 + channel] = palette[(bits >> (texel * 3)) as usize & 7] as u8;
    }
}

/// Which subset each texel of a block is in, for BC7's two subsets: a bit a texel
const PARTITIONS_2: [u16; 64] = [
    0xCCCC, 0x8888, 0xEEEE, 0xECC8, 0xC880, 0xFEEC, 0xFEC8, 0xEC80, 0xC800, 0xFFEC, 0xFE80, 0xE800, 0xFFE8, 0xFF00, 0xFFF0, 0xF000, 0xF710, 0x008E, 0x7100, 0x08CE, 0x008C, 0x7310,
    0x3100, 0x8CCE, 0x088C, 0x3110, 0x6666, 0x366C, 0x17E8, 0x0FF0, 0x718E, 0x399C, 0xAAAA, 0xF0F0, 0x5A5A, 0x33CC, 0x3C3C, 0x55AA, 0x9696, 0xA55A, 0x73CE, 0x13C8, 0x324C, 0x3BDC,
    0x6996, 0xC33C, 0x9966, 0x0660, 0x0272, 0x04E4, 0x4E40, 0x2720, 0xC936, 0x936C, 0x39C6, 0x639C, 0x9336, 0x9CC6, 0x817E, 0xE718, 0xCCF0, 0x0FCC, 0x7744, 0xEE22,
];

/// And for three, row by row
const PARTITIONS_3: [[u8; 16]; 64] = [
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 1, 2, 2, 2, 2],
    [0, 0, 0, 1, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 2, 1],
    [0, 0, 0, 0, 2, 0, 0, 1, 2, 2, 1, 1, 2, 2, 1, 1],
    [0, 2, 2, 2, 0, 0, 2, 2, 0, 0, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2],
    [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 2, 2],
    [0, 0, 2, 2, 0, 0, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
    [0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2],
    [0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2, 0, 1, 1, 2],
    [0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2],
    [0, 0, 1, 1, 0, 1, 1, 2, 1, 1, 2, 2, 1, 2, 2, 2],
    [0, 0, 1, 1, 2, 0, 0, 1, 2, 2, 0, 0, 2, 2, 2, 0],
    [0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 2, 1, 1, 2, 2],
    [0, 1, 1, 1, 0, 0, 1, 1, 2, 0, 0, 1, 2, 2, 0, 0],
    [0, 0, 0, 0, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2],
    [0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2, 1, 1, 1, 1],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 2, 2, 2, 0, 2, 2, 2],
    [0, 0, 0, 1, 0, 0, 0, 1, 2, 2, 2, 1, 2, 2, 2, 1],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 2, 0, 1, 2, 2],
    [0, 0, 0, 0, 1, 1, 0, 0, 2, 2, 1, 0, 2, 2, 1, 0],
    [0, 1, 2, 2, 0, 1, 2, 2, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 2, 0, 0, 1, 2, 1, 1, 2, 2, 2, 2, 2, 2],
    [0, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 1, 2, 2, 1, 1, 2, 2, 1],
    [0, 0, 2, 2, 1, 1, 0, 2, 1, 1, 0, 2, 0, 0, 2, 2],
    [0, 1, 1, 0, 0, 1, 1, 0, 2, 0, 0, 2, 2, 2, 2, 2],
    [0, 0, 1, 1, 0, 1, 2, 2, 0, 1, 2, 2, 0, 0, 1, 1],
    [0, 0, 0, 0, 2, 0, 0, 0, 2, 2, 1, 1, 2, 2, 2, 1],
    [0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2, 2, 1, 2, 2, 2],
    [0, 2, 2, 2, 0, 0, 2, 2, 0, 0, 1, 2, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 0, 1, 2, 0, 0, 2, 2, 0, 2, 2, 2],
    [0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0, 0, 1, 2, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0],
    [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
    [0, 1, 2, 0, 2, 0, 1, 2, 1, 2, 0, 1, 0, 1, 2, 0],
    [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1],
    [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 1, 1],
    [0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 2, 1, 2, 1, 2, 1],
    [0, 0, 2, 2, 1, 1, 2, 2, 0, 0, 2, 2, 1, 1, 2, 2],
    [0, 0, 2, 2, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 1, 1],
    [0, 2, 2, 0, 1, 2, 2, 1, 0, 2, 2, 0, 1, 2, 2, 1],
    [0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 0, 1],
    [0, 0, 0, 0, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
    [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2],
    [0, 2, 2, 2, 0, 1, 1, 1, 0, 2, 2, 2, 0, 1, 1, 1],
    [0, 0, 0, 2, 1, 1, 1, 2, 0, 0, 0, 2, 1, 1, 1, 2],
    [0, 0, 0, 0, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2],
    [0, 2, 2, 2, 0, 1, 1, 1, 0, 1, 1, 1, 0, 2, 2, 2],
    [0, 0, 0, 2, 1, 1, 1, 2, 1, 1, 1, 2, 0, 0, 0, 2],
    [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2, 2, 1, 1, 2],
    [0, 1, 1, 0, 0, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2],
    [0, 0, 2, 2, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 2],
    [0, 0, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 0, 0, 2, 2],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1, 1, 2],
    [0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1],
    [0, 2, 2, 2, 1, 2, 2, 2, 0, 2, 2, 2, 1, 2, 2, 2],
    [0, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    [0, 1, 1, 1, 2, 0, 1, 1, 2, 2, 0, 1, 2, 2, 2, 0],
];

/// The texel of each subset after the first whose index has a bit less
const ANCHORS_2: [u8; 64] = [
    15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2, 8, 2, 2, 8, 8, 15, 2, 8, 2, 2, 8, 8, 2, 2, 15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6, 6, 2, 6, 8, 15, 15, 2, 2,
    15, 15, 15, 15, 15, 2, 2, 15,
];
const ANCHORS_3_A: [u8; 64] = [
    3, 3, 15, 15, 8, 3, 15, 15, 8, 8, 6, 6, 6, 5, 3, 3, 3, 3, 8, 15, 3, 3, 6, 10, 5, 8, 8, 6, 8, 5, 15, 15, 8, 15, 3, 5, 6, 10, 8, 15, 15, 3, 15, 5, 15, 15, 15, 15, 3, 15, 5, 5, 5, 8, 5, 10, 5, 10,
    8, 13, 15, 12, 3, 3,
];
const ANCHORS_3_B: [u8; 64] = [
    15, 8, 8, 3, 15, 15, 3, 8, 15, 15, 15, 15, 15, 15, 15, 8, 15, 8, 15, 3, 15, 8, 15, 8, 3, 15, 6, 10, 15, 15, 10, 8, 15, 3, 15, 10, 10, 8, 9, 10, 6, 15, 8, 15, 3, 6, 6, 8, 15, 3, 15, 15, 15, 15,
    15, 15, 15, 15, 15, 15, 3, 15, 15, 8,
];

const WEIGHTS_2: [i32; 4] = [0, 21, 43, 64];
const WEIGHTS_3: [i32; 8] = [0, 9, 18, 27, 37, 46, 55, 64];
const WEIGHTS_4: [i32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];
/// The index in WEIGHTS_4 nearest each weight from 0 to 64
const NEAREST_WEIGHT: [u8; 65] = {
    let mut table = [0u8; 65];
    let mut weight = 0;
    while weight < 65 {
        let (mut index, mut best, mut distance) = (0, 0, i32::MAX);
        while index < 16 {
            let gap = (WEIGHTS_4[index] - weight as i32).abs();
            if gap < distance {
                distance = gap;
                best = index;
            }
            index += 1;
        }
        table[weight] = best as u8;
        weight += 1;
    }
    table
};

/// A BC7 mode: subsets, partition bits, rotation bits, index selection bits, colour bits, alpha bits, a
/// p-bit for each endpoint, a p-bit shared by each subset's, index bits, second index bits
#[derive(Clone, Copy)]
struct Mode(usize, usize, usize, usize, usize, usize, usize, usize, usize, usize);

const MODES: [Mode; 8] = [
    Mode(3, 4, 0, 0, 4, 0, 1, 0, 3, 0),
    Mode(2, 6, 0, 0, 6, 0, 0, 1, 3, 0),
    Mode(3, 6, 0, 0, 5, 0, 0, 0, 2, 0),
    Mode(2, 6, 0, 0, 7, 0, 1, 0, 2, 0),
    Mode(1, 0, 2, 1, 5, 6, 0, 0, 2, 3),
    Mode(1, 0, 2, 0, 7, 8, 0, 0, 2, 2),
    Mode(1, 0, 0, 0, 7, 7, 1, 0, 4, 0),
    Mode(2, 6, 0, 0, 5, 5, 1, 0, 2, 0),
];

struct Bits {
    left: u128,
}

impl Bits {
    #[inline(always)]
    fn take(&mut self, count: usize) -> u32 {
        let value = (self.left & ((1u128 << count) - 1)) as u32;
        self.left >>= count;
        value
    }
}

fn expand(value: u32, bits: usize) -> i32 {
    let value = value << (8 - bits);
    (value | (value >> bits)) as i32
}

fn weight(bits: usize, index: usize) -> i32 {
    match bits {
        2 => WEIGHTS_2[index],
        3 => WEIGHTS_3[index],
        _ => WEIGHTS_4[index],
    }
}

fn decode_bc7(block: &[u8], out: &mut [u8; 64]) {
    let Some(mode_number) = (0..8).find(|&bit| block[0] & (1 << bit) != 0) else {
        // No mode: all zeros, by the specification
        out.fill(0);
        return;
    };
    let Mode(subsets, partition_bits, rotation_bits, selection_bits, colour_bits, alpha_bits, endpoint_p, shared_p, index_bits, index2_bits) = MODES[mode_number];
    let mut bits = Bits { left: u128::from_le_bytes(block[..16].try_into().unwrap()) >> (mode_number + 1) };
    let partition = bits.take(partition_bits) as usize;
    let rotation = bits.take(rotation_bits);
    let selection = bits.take(selection_bits);
    let mut endpoints = [[[0u32; 4]; 2]; 3];
    for channel in 0..3 {
        for subset in endpoints.iter_mut().take(subsets) {
            subset[0][channel] = bits.take(colour_bits);
            subset[1][channel] = bits.take(colour_bits);
        }
    }
    for subset in endpoints.iter_mut().take(subsets) {
        subset[0][3] = bits.take(alpha_bits);
        subset[1][3] = bits.take(alpha_bits);
    }
    // P-bits add a bit to every channel of the endpoint they're for
    let (colour_width, alpha_width) = (colour_bits + endpoint_p + shared_p, if alpha_bits > 0 { alpha_bits + endpoint_p + shared_p } else { 0 });
    if endpoint_p + shared_p > 0 {
        for subset in endpoints.iter_mut().take(subsets) {
            let shared = if shared_p > 0 { bits.take(1) } else { 0 };
            for endpoint in subset.iter_mut() {
                let p = if endpoint_p > 0 { bits.take(1) } else { shared };
                for value in endpoint.iter_mut() {
                    *value = (*value << 1) | p;
                }
            }
        }
    }
    let subset_of = |texel: usize| match subsets {
        1 => 0,
        2 => (PARTITIONS_2[partition] >> texel) as usize & 1,
        _ => PARTITIONS_3[partition][texel] as usize,
    };
    let is_anchor = |texel: usize| match subsets {
        1 => texel == 0,
        2 => texel == 0 || texel == ANCHORS_2[partition] as usize,
        _ => texel == 0 || texel == ANCHORS_3_A[partition] as usize || texel == ANCHORS_3_B[partition] as usize,
    };
    let mut first = [0usize; 16];
    for (texel, index) in first.iter_mut().enumerate() {
        *index = bits.take(index_bits - is_anchor(texel) as usize) as usize;
    }
    let mut second = [0usize; 16];
    if index2_bits > 0 {
        for (texel, index) in second.iter_mut().enumerate() {
            *index = bits.take(index2_bits - (texel == 0) as usize) as usize;
        }
    }
    for (texel, pixel) in out.chunks_exact_mut(4).enumerate() {
        let [from, to] = endpoints[subset_of(texel)];
        let (colour_weight, alpha_weight) = if index2_bits == 0 {
            (weight(index_bits, first[texel]), weight(index_bits, first[texel]))
        } else if selection == 0 {
            (weight(index_bits, first[texel]), weight(index2_bits, second[texel]))
        } else {
            (weight(index2_bits, second[texel]), weight(index_bits, first[texel]))
        };
        let mix = |channel: usize, width: usize, w: i32| {
            let (a, b) = (expand(from[channel], width), expand(to[channel], width));
            ((64 - w) * a + w * b + 32) >> 6
        };
        for channel in 0..3 {
            pixel[channel] = mix(channel, colour_width, colour_weight) as u8;
        }
        pixel[3] = if alpha_width > 0 { mix(3, alpha_width, alpha_weight) as u8 } else { 255 };
        match rotation {
            1 => pixel.swap(0, 3),
            2 => pixel.swap(1, 3),
            3 => pixel.swap(2, 3),
            _ => {}
        }
    }
}

/// A block's texels, RGBA, row by row. What a format doesn't hold reads as 0, and alpha as 255.
pub fn decode_block(kind: Kind, block: &[u8], out: &mut [u8; 64]) {
    match kind {
        Kind::Bc1 => decode_colour(block, out, true),
        Kind::Bc2 => {
            decode_colour(&block[8..], out, false);
            let bits = u64::from_le_bytes(block[..8].try_into().unwrap());
            for texel in 0..16 {
                out[texel * 4 + 3] = ((bits >> (texel * 4)) as u8 & 15) * 17;
            }
        }
        Kind::Bc3 => {
            decode_colour(&block[8..], out, false);
            decode_single(block, out, 3);
        }
        Kind::Bc4 => {
            for pixel in out.chunks_exact_mut(4) {
                pixel.copy_from_slice(&[0, 0, 0, 255]);
            }
            decode_single(block, out, 0);
        }
        Kind::Bc5 => {
            for pixel in out.chunks_exact_mut(4) {
                pixel.copy_from_slice(&[0, 0, 0, 255]);
            }
            decode_single(block, out, 0);
            decode_single(&block[8..], out, 1);
        }
        Kind::Bc7 => decode_bc7(block, out),
        Kind::Rgba8 => out[..4].copy_from_slice(&block[..4]),
    }
}

/// The two ends of the line most of `count` points lie along, in `channels` dimensions, pulled in by
/// `inset` of the distance between them
fn line_ends(points: &[[f32; 4]; 16], channels: usize, inset: f32) -> ([f32; 4], [f32; 4]) {
    let mut mean = [0f32; 4];
    for point in points {
        for channel in 0..channels {
            mean[channel] += point[channel] / 16.0;
        }
    }
    let mut spread = [[0f32; 4]; 4];
    for point in points {
        for a in 0..channels {
            for b in 0..channels {
                spread[a][b] += (point[a] - mean[a]) * (point[b] - mean[b]);
            }
        }
    }
    let mut axis = [1f32; 4];
    for _ in 0..4 {
        let mut next = [0f32; 4];
        for a in 0..channels {
            for b in 0..channels {
                next[a] += spread[a][b] * axis[b];
            }
        }
        let length = next.iter().map(|value| value * value).sum::<f32>().sqrt();
        if length < 1e-3 {
            return (mean, mean);
        }
        axis = next.map(|value| value / length);
    }
    let (mut low, mut high) = (f32::MAX, f32::MIN);
    for point in points {
        let along: f32 = (0..channels).map(|channel| (point[channel] - mean[channel]) * axis[channel]).sum();
        low = low.min(along);
        high = high.max(along);
    }
    let gap = (high - low) * inset;
    let at = |distance: f32| {
        let mut point = mean;
        for channel in 0..channels {
            point[channel] += axis[channel] * distance;
        }
        point
    };
    (at(high - gap), at(low + gap))
}

fn to_565(point: [f32; 4]) -> u16 {
    let scale = |value: f32, most: f32| ((value / 255.0 * most + 0.5).clamp(0.0, most)) as u16;
    (scale(point[0], 31.0) << 11) | (scale(point[1], 63.0) << 5) | scale(point[2], 31.0)
}

fn points_of(pixels: &[u8; 64]) -> [[f32; 4]; 16] {
    let mut points = [[0f32; 4]; 16];
    for (point, pixel) in points.iter_mut().zip(pixels.chunks_exact(4)) {
        *point = [pixel[0] as f32, pixel[1] as f32, pixel[2] as f32, pixel[3] as f32];
    }
    points
}

fn encode_colour(pixels: &[u8; 64], out: &mut [u8]) {
    let (from, to) = line_ends(&points_of(pixels), 3, 1.0 / 16.0);
    let (mut c0, mut c1) = (to_565(from), to_565(to));
    if c0 < c1 {
        std::mem::swap(&mut c0, &mut c1);
    }
    // Four colours need c0 greater; a colour with nothing to tell apart has indices all 0
    if c0 == c1 {
        if c0 > 0 {
            c1 = c0 - 1;
        } else {
            c0 = 1;
        }
    }
    let (a, b) = (from_565(c0), from_565(c1));
    let mix = |m: i32, n: i32| [(m * a[0] + n * b[0]) / 3, (m * a[1] + n * b[1]) / 3, (m * a[2] + n * b[2]) / 3];
    let palette = [a, b, mix(2, 1), mix(1, 2)];
    let mut bits = 0u32;
    for (texel, pixel) in pixels.chunks_exact(4).enumerate() {
        let mut best = (i32::MAX, 0);
        for (index, colour) in palette.iter().enumerate() {
            let distance = (0..3).map(|channel| (colour[channel] - pixel[channel] as i32).pow(2)).sum();
            if distance < best.0 {
                best = (distance, index as u32);
            }
        }
        bits |= best.1 << (texel * 2);
    }
    out[..2].copy_from_slice(&c0.to_le_bytes());
    out[2..4].copy_from_slice(&c1.to_le_bytes());
    out[4..8].copy_from_slice(&bits.to_le_bytes());
}

fn encode_single(pixels: &[u8; 64], channel: usize, out: &mut [u8]) {
    let values = || pixels.chunks_exact(4).map(|pixel| pixel[channel] as i32);
    let (high, low) = (values().max().unwrap_or(0), values().min().unwrap_or(0));
    out[0] = high as u8;
    out[1] = low as u8;
    let palette = if high > low {
        [high, low, (6 * high + low) / 7, (5 * high + 2 * low) / 7, (4 * high + 3 * low) / 7, (3 * high + 4 * low) / 7, (2 * high + 5 * low) / 7, (high + 6 * low) / 7]
    } else {
        [high; 8]
    };
    let mut bits = 0u64;
    for (texel, value) in values().enumerate() {
        let mut best = (i32::MAX, 0);
        for (index, entry) in palette.iter().enumerate() {
            let distance = (entry - value).abs();
            if distance < best.0 {
                best = (distance, index as u64);
            }
        }
        bits |= best.1 << (texel * 3);
    }
    out[2..8].copy_from_slice(&bits.to_le_bytes()[..6]);
}

/// BC7's mode 6: one subset, colour and alpha along one line, 4 bits a texel
fn encode_bc7(pixels: &[u8; 64], out: &mut [u8]) {
    let points = points_of(pixels);
    let (from, to) = line_ends(&points, 4, 1.0 / 64.0);
    // 7 bits and a p-bit of 1 for both
    let quantise = |point: [f32; 4]| point.map(|value| ((value / 2.0).clamp(0.0, 127.0) + 0.5) as i32);
    let mut ends = [quantise(from), quantise(to)];
    let palette = |ends: &[[i32; 4]; 2]| {
        let mut colours = [[0i32; 4]; 16];
        for (colour, w) in colours.iter_mut().zip(WEIGHTS_4) {
            for channel in 0..4 {
                let (a, b) = ((ends[0][channel] << 1) | 1, (ends[1][channel] << 1) | 1);
                colour[channel] = ((64 - w) * a + w * b + 32) >> 6;
            }
        }
        colours
    };
    // How far along the line each texel is says its index, with no need to try all 16
    let colours = palette(&ends);
    let line: [f32; 4] = std::array::from_fn(|channel| (colours[15][channel] - colours[0][channel]) as f32);
    let length: f32 = line.iter().map(|value| value * value).sum();
    let mut indices = [0u32; 16];
    if length > 0.0 {
        for (index, point) in indices.iter_mut().zip(&points) {
            let along: f32 = (0..4).map(|channel| (point[channel] - colours[0][channel] as f32) * line[channel]).sum();
            *index = NEAREST_WEIGHT[((along / length * 64.0 + 0.5).clamp(0.0, 64.0)) as usize] as u32;
        }
    }
    // The first texel's index has no top bit
    if indices[0] >= 8 {
        ends.swap(0, 1);
        for index in &mut indices {
            *index = 15 - *index;
        }
    }
    let mut low = 1u64 << 6;
    let mut at = 7;
    for channel in 0..4 {
        for end in &ends {
            low |= (end[channel] as u64) << at;
            at += 7;
        }
    }
    // Bit 63 is the first p-bit, and the second begins the next word, with the indices
    low |= 1 << 63;
    let mut high = 1u64;
    let mut at = 1;
    for (texel, index) in indices.iter().enumerate() {
        high |= (*index as u64) << at;
        at += if texel == 0 { 3 } else { 4 };
    }
    out[..8].copy_from_slice(&low.to_le_bytes());
    out[8..16].copy_from_slice(&high.to_le_bytes());
}

/// A block all of one colour, the average of `block`: repeated across a mip it looks like a small
/// picture, where the block itself would look like a pattern
pub fn flatten_block(kind: Kind, block: &mut [u8]) {
    if kind == Kind::Rgba8 {
        return;
    }
    let mut pixels = [0u8; 64];
    decode_block(kind, block, &mut pixels);
    let mut totals = [0u32; 4];
    for pixel in pixels.chunks_exact(4) {
        for channel in 0..4 {
            totals[channel] += pixel[channel] as u32;
        }
    }
    let average = totals.map(|total| ((total + 8) / 16) as u8);
    for pixel in pixels.chunks_exact_mut(4) {
        pixel.copy_from_slice(&average);
    }
    encode_block(kind, &pixels, block);
}

/// A block from its texels, RGBA, row by row
pub fn encode_block(kind: Kind, pixels: &[u8; 64], out: &mut [u8]) {
    match kind {
        Kind::Bc1 => encode_colour(pixels, out),
        Kind::Bc2 => {
            let mut bits = 0u64;
            for (texel, pixel) in pixels.chunks_exact(4).enumerate() {
                bits |= ((pixel[3] as u64 + 8) / 17) << (texel * 4);
            }
            out[..8].copy_from_slice(&bits.to_le_bytes());
            encode_colour(pixels, &mut out[8..]);
        }
        Kind::Bc3 => {
            encode_single(pixels, 3, out);
            encode_colour(pixels, &mut out[8..]);
        }
        Kind::Bc4 => encode_single(pixels, 0, out),
        Kind::Bc5 => {
            encode_single(pixels, 0, out);
            encode_single(pixels, 1, &mut out[8..]);
        }
        Kind::Bc7 => encode_bc7(pixels, out),
        Kind::Rgba8 => out[..4].copy_from_slice(&pixels[..4]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn picture_block(seed: u32) -> [u8; 64] {
        let mut pixels = [0u8; 64];
        for (texel, pixel) in pixels.chunks_exact_mut(4).enumerate() {
            let (x, y) = ((texel % 4) as u32, (texel / 4) as u32);
            pixel[0] = (seed.wrapping_mul(7) + x * 9) as u8;
            pixel[1] = (seed.wrapping_mul(13) + y * 11) as u8;
            pixel[2] = (seed.wrapping_mul(29) + (x + y) * 5) as u8;
            pixel[3] = (seed.wrapping_mul(3) + y * 7) as u8;
        }
        pixels
    }

    #[test]
    fn what_is_encoded_decodes_near_it() {
        // On average: colours that don't lie along a line can't all be near
        for (kind, most) in [(Kind::Bc1, 8), (Kind::Bc2, 8), (Kind::Bc3, 8), (Kind::Bc4, 3), (Kind::Bc5, 3), (Kind::Bc7, 8)] {
            let (mut total, mut count) = (0, 0);
            for seed in 0..200 {
                let pixels = picture_block(seed);
                let mut block = [0u8; 16];
                let mut back = [0u8; 64];
                encode_block(kind, &pixels, &mut block);
                decode_block(kind, &block, &mut back);
                for (wanted, got) in pixels.chunks_exact(4).zip(back.chunks_exact(4)) {
                    for channel in 0..kind.channels() {
                        total += (wanted[channel] as i32 - got[channel] as i32).abs();
                        count += 1;
                    }
                }
            }
            assert!(total / count <= most, "{kind:?}: {}", total / count);
        }
    }

    #[test]
    fn one_colour_comes_back_as_it_was() {
        let pixels: [u8; 64] = std::array::from_fn(|at| [200u8, 100, 50, 255][at % 4]);
        let mut block = [0u8; 16];
        let mut back = [0u8; 64];
        encode_block(Kind::Bc7, &pixels, &mut block);
        decode_block(Kind::Bc7, &block, &mut back);
        for (wanted, got) in pixels.iter().zip(&back) {
            assert!((*wanted as i32 - *got as i32).abs() <= 1);
        }
    }

    #[test]
    fn a_flat_block_is_one_colour_the_average() {
        for kind in [Kind::Bc1, Kind::Bc3, Kind::Bc5, Kind::Bc7] {
            let pixels = picture_block(9);
            let mut block = [0u8; 16];
            encode_block(kind, &pixels, &mut block);
            let mut before = [0u8; 64];
            decode_block(kind, &block, &mut before);
            flatten_block(kind, &mut block[..kind.block_bytes()]);
            let mut after = [0u8; 64];
            decode_block(kind, &block, &mut after);
            for channel in 0..kind.channels() {
                let average = before.chunks_exact(4).map(|pixel| pixel[channel] as i32).sum::<i32>() / 16;
                for pixel in after.chunks_exact(4) {
                    assert!((pixel[channel] as i32 - average).abs() <= 6, "{kind:?} channel {channel}: {} against {average}", pixel[channel]);
                }
            }
        }
    }

    #[test]
    fn a_block_with_no_mode_is_all_zeros() {
        let mut out = [9u8; 64];
        decode_block(Kind::Bc7, &[0u8; 16], &mut out);
        assert_eq!(out, [0u8; 64]);
    }

    #[test]
    fn formats_are_told_apart() {
        assert_eq!(Kind::of(VK_FORMAT_BC7_SRGB_BLOCK), Some((Kind::Bc7, true)));
        assert_eq!(Kind::of(VK_FORMAT_BC5_UNORM_BLOCK), Some((Kind::Bc5, false)));
        assert_eq!(Kind::of(VK_FORMAT_BC6H_UFLOAT_BLOCK), None);
        assert_eq!(Kind::Bc1.block_bytes(), 8);
        assert_eq!(Kind::Rgba8.block_side(), 1);
    }
}
