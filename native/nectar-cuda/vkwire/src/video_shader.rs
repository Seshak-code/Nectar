//! The colours a video encoder takes, made on the GPU: a compute shader that turns a frame's 8-bit
//! red, green and blue into NV12, which is a plane of luma and a half-size plane of the two chromas side
//! by side. BT.709 in full range, which `nv12_to_bgra` undoes on the other side.
//!
//! Each invocation does 4 pixels across and 2 down: 2 words of luma, and 1 word holding 2 pairs of
//! chromas, each the average of its 4 pixels. Pixels past the frame's edge repeat the edge, so the
//! encoder's picture, which is whole 16-pixel blocks, has nothing sharp to code there.
//!
//! ```text
//! binding 0: the frame, a word a pixel, `width` across
//! binding 1: luma, coded_width * coded_height bytes, then chroma, half as many
//! push constants: u32 width, height, coded width, coded height, red's shift, blue's shift
//! ```

use crate::spirv::{Id, Shader};

/// Bytes of push constants
pub const PUSH_BYTES: u32 = 24;

/// How large the encoder's picture is for a frame: whole macroblocks
pub fn coded_size(width: u32, height: u32) -> (u32, u32) {
    (width.next_multiple_of(16), height.next_multiple_of(16))
}

/// Bytes of NV12 for a coded size
pub fn nv12_size(coded_width: u32, coded_height: u32) -> u64 {
    coded_width as u64 * coded_height as u64 * 3 / 2
}

/// The workgroups to dispatch
pub fn groups(coded_width: u32, coded_height: u32) -> (u32, u32) {
    (coded_width / 4, coded_height / 2)
}

pub fn push_constants(width: u32, height: u32, bgra: bool) -> [u32; 6] {
    let (coded_width, coded_height) = coded_size(width, height);
    let (red, blue) = if bgra { (16, 0) } else { (0, 16) };
    [width, height, coded_width, coded_height, red, blue]
}

pub const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];
/// What blue less luma, and red less luma, are scaled by to make the chromas
pub const CHROMA: [f32; 2] = [1.0 / 1.8556, 1.0 / 1.5748];

pub fn shader() -> Vec<u32> {
    let mut s = Shader::new();
    let frame = s.storage_buffer(0);
    let out = s.storage_buffer(1);
    let uint = s.uint;
    let push = s.push_constants(&[uint; 6]);
    s.main(|s| {
        let (gx, gy) = (s.invocation(0), s.invocation(1));
        let [width, height, coded_width, coded_height, red_shift, blue_shift] = std::array::from_fn(|index| s.read_push(&push, index as u32));
        let (one, two, four, eight, sixteen, mask) = (s.u(1), s.u(2), s.u(4), s.u(8), s.u(16), s.u(255));
        let x0 = s.mul(gx, four);
        let y0 = s.mul(gy, two);
        let last_x = s.sub(width, one);
        let last_y = s.sub(height, one);
        let (zero_i, top_i) = (s.i(0), s.i(255));
        let byte = |s: &mut Shader, value: Id| -> Id {
            let rounded = s.round(value);
            let clamped = s.iclamp(rounded, zero_i, top_i);
            s.as_uint(clamped)
        };
        // Sums of each channel over each 2x2 block, for the chromas
        let mut sums: [[Option<Id>; 3]; 2] = [[None; 3]; 2];
        for row in 0..2u32 {
            let row_id = s.u(row);
            let y = s.add(y0, row_id);
            let source_y = s.umin(y, last_y);
            let source_row = s.mul(source_y, width);
            let mut word = s.u(0);
            for column in 0..4u32 {
                let column_id = s.u(column);
                let x = s.add(x0, column_id);
                let source_x = s.umin(x, last_x);
                let at = s.add(source_row, source_x);
                let pixel = s.read(frame, at);
                let channel = |s: &mut Shader, shift: Id| -> Id {
                    let moved = s.shr(pixel, shift);
                    let value = s.and(moved, mask);
                    s.to_float(value)
                };
                let rgb = [channel(s, red_shift), channel(s, eight), channel(s, blue_shift)];
                let mut luma = s.f(0.0);
                for (value, weight) in rgb.iter().zip(LUMA) {
                    let weight = s.f(weight);
                    let part = s.fmul(*value, weight);
                    luma = s.fadd(luma, part);
                }
                let luma = byte(s, luma);
                let shift = s.u(column * 8);
                let placed = s.shl(luma, shift);
                word = s.or(word, placed);
                let block = (column / 2) as usize;
                for (sum, value) in sums[block].iter_mut().zip(rgb) {
                    *sum = Some(match *sum {
                        Some(before) => s.fadd(before, value),
                        None => value,
                    });
                }
            }
            // This row's 4 bytes of luma
            let row_start = s.mul(y, coded_width);
            let byte_at = s.add(row_start, x0);
            let word_at = s.shr(byte_at, two);
            s.write(out, word_at, word);
        }
        let quarter = s.f(0.25);
        let half = s.f(128.0);
        let mut word = s.u(0);
        for (block, sums) in sums.iter().enumerate() {
            let [red, green, blue] = sums.map(|sum| {
                let sum = sum.expect("every block has pixels");
                s.fmul(sum, quarter)
            });
            let mut luma = s.f(0.0);
            for (value, weight) in [red, green, blue].iter().zip(LUMA) {
                let weight = s.f(weight);
                let part = s.fmul(*value, weight);
                luma = s.fadd(luma, part);
            }
            for (index, (value, scale)) in [(blue, CHROMA[0]), (red, CHROMA[1])].into_iter().enumerate() {
                let difference = s.fsub(value, luma);
                let scale = s.f(scale);
                let scaled = s.fmul(difference, scale);
                let chroma = s.fadd(scaled, half);
                let chroma = byte(s, chroma);
                let shift = s.u((block as u32 * 2 + index as u32) * 8);
                let placed = s.shl(chroma, shift);
                word = s.or(word, placed);
            }
        }
        // The chroma plane follows the luma, a row of it as many bytes as a row of luma
        let plane = s.mul(coded_width, coded_height);
        let row_start = s.mul(gy, coded_width);
        let in_plane = s.add(row_start, x0);
        let byte_at = s.add(plane, in_plane);
        let word_at = s.shr(byte_at, two);
        s.write(out, word_at, word);
        let _ = sixteen;
    });
    s.finish()
}

/// What the shader computes, on the CPU: for tests, and for a lender whose GPU can't
pub fn bgra_to_nv12(pixels: &[u8], width: u32, height: u32, bgra: bool) -> Vec<u8> {
    let (coded_width, coded_height) = coded_size(width, height);
    let (cw, ch) = (coded_width as usize, coded_height as usize);
    let mut out = vec![0u8; nv12_size(coded_width, coded_height) as usize];
    let rgb = |x: usize, y: usize| -> [f32; 3] {
        let at = (y.min(height as usize - 1) * width as usize + x.min(width as usize - 1)) * 4;
        let (first, green, third) = (pixels[at] as f32, pixels[at + 1] as f32, pixels[at + 2] as f32);
        if bgra { [third, green, first] } else { [first, green, third] }
    };
    let luma = |rgb: [f32; 3]| rgb[0] * LUMA[0] + rgb[1] * LUMA[1] + rgb[2] * LUMA[2];
    let byte = |value: f32| value.round().clamp(0.0, 255.0) as u8;
    for y in 0..ch {
        for x in 0..cw {
            out[y * cw + x] = byte(luma(rgb(x, y)));
        }
    }
    for y in 0..ch / 2 {
        for x in 0..cw / 2 {
            let mut sum = [0.0f32; 3];
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let pixel = rgb(x * 2 + dx, y * 2 + dy);
                for (total, value) in sum.iter_mut().zip(pixel) {
                    *total += value;
                }
            }
            let mean = sum.map(|total| total * 0.25);
            let l = luma(mean);
            let at = cw * ch + y * cw + x * 2;
            out[at] = byte((mean[2] - l) * CHROMA[0] + 128.0);
            out[at + 1] = byte((mean[0] - l) * CHROMA[1] + 128.0);
        }
    }
    out
}

/// NV12 back to the 4 bytes a pixel the window takes (blue, green, red, 255), for the top-left `width` by
/// `height` of a picture whose rows are `stride` bytes apart and whose chroma starts `chroma_offset` in.
/// On every core, as a frame is a few million pixels.
pub fn nv12_to_bgra(nv12: &[u8], stride: usize, chroma_offset: usize, width: usize, height: usize, out: &mut [u8]) {
    assert!(out.len() >= width * height * 4);
    assert!(stride >= width && nv12.len() >= chroma_offset + stride * (height.div_ceil(2) - 1) + width.next_multiple_of(2));
    // 16 bits after the point
    const RED: i32 = 103_206;
    const GREEN_BLUE: i32 = 12_276;
    const GREEN_RED: i32 = 30_679;
    const BLUE: i32 = 121_609;
    static THREADS: nectar_cuda_protocol::pool::Pool = nectar_cuda_protocol::pool::Pool::new();
    struct Out(*mut u8);
    unsafe impl Sync for Out {}
    let target = Out(out.as_mut_ptr());
    let target = &target;
    let next = std::sync::atomic::AtomicUsize::new(0);
    const ROWS: usize = 32;
    let threads = std::thread::available_parallelism().map_or(1, |count| count.get()).min(8).min(height.div_ceil(ROWS));
    THREADS.run(threads.saturating_sub(1), &|| {
        loop {
            let first = next.fetch_add(ROWS, std::sync::atomic::Ordering::Relaxed);
            if first >= height {
                return;
            }
            for y in first..(first + ROWS).min(height) {
                let luma = &nv12[y * stride..y * stride + width];
                let chroma = &nv12[chroma_offset + (y / 2) * stride..];
                // Rows don't overlap, so the threads write different pixels
                let row = unsafe { std::slice::from_raw_parts_mut(target.0.add(y * width * 4), width * 4) };
                for (x, pixel) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    let l = (luma[x] as i32) << 16;
                    let (cb, cr) = (chroma[x & !1] as i32 - 128, chroma[x | 1] as i32 - 128);
                    let clamp = |value: i32| ((value + 0x8000) >> 16).clamp(0, 255) as u8;
                    *pixel = [clamp(l + BLUE * cb), clamp(l - GREEN_BLUE * cb - GREEN_RED * cr), clamp(l + RED * cr), 255];
                }
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shader_is_well_formed() {
        let words = shader();
        assert_eq!(words[0], 0x0723_0203);
        let mut at = 5;
        while at < words.len() {
            let length = (words[at] >> 16) as usize;
            assert!(length > 0);
            at += length;
        }
        assert_eq!(at, words.len());
    }

    #[test]
    fn colours_come_back_nearly_the_same() {
        // Flat blocks of every sort of colour, as chroma is shared by 4 pixels
        let (width, height) = (70usize, 38usize);
        let mut pixels = vec![255u8; width * height * 4];
        for y in 0..height {
            for x in 0..width {
                let block = (x / 2 * 7 + y / 2 * 13) as u32;
                let at = (y * width + x) * 4;
                pixels[at] = (block * 37 % 256) as u8;
                pixels[at + 1] = (block * 91 % 256) as u8;
                pixels[at + 2] = (block * 53 % 256) as u8;
            }
        }
        for bgra in [true, false] {
            let nv12 = bgra_to_nv12(&pixels, width as u32, height as u32, bgra);
            let (coded_width, coded_height) = coded_size(width as u32, height as u32);
            assert_eq!((coded_width, coded_height), (80, 48));
            let mut back = vec![0u8; pixels.len()];
            nv12_to_bgra(&nv12, coded_width as usize, (coded_width * coded_height) as usize, width, height, &mut back);
            for (index, (was, now)) in pixels.chunks(4).zip(back.chunks(4)).enumerate() {
                // nv12_to_bgra always gives blue first
                let was = if bgra { [was[0], was[1], was[2]] } else { [was[2], was[1], was[0]] };
                for channel in 0..3 {
                    assert!((was[channel] as i32 - now[channel] as i32).abs() <= 3, "pixel {index}: {was:?} came back {:?}", &now[..3]);
                }
            }
        }
    }

    #[test]
    fn the_edge_repeats_into_the_padding() {
        let (width, height) = (18u32, 18u32);
        let pixels = vec![200u8; (width * height * 4) as usize];
        let nv12 = bgra_to_nv12(&pixels, width, height, true);
        let (coded_width, coded_height) = coded_size(width, height);
        assert!(nv12[..(coded_width * coded_height) as usize].iter().all(|&luma| luma == 200));
        assert!(nv12[(coded_width * coded_height) as usize..].iter().all(|&chroma| chroma == 128));
    }
}
