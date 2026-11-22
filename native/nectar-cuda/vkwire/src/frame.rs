//! The frame codec: how a frame crosses the link, for the swapchain. Portable, with nothing to
//! install and no hardware it depends on, so any lender can send frames and any borrower can show
//! them. A hardware video encoder can do better where one exists; this always works.
//!
//! A frame is cut into 64x64 tiles. A tile that hasn't changed since the last frame isn't sent. A
//! changed tile is coded much as JPEG codes a picture: YCoCg colour with half-resolution chroma,
//! an 8x8 cosine transform, coefficients divided by a step set by the quality, and those that are
//! left written with a code that adapts to their size. Tiles are independent, so they're coded on
//! every core and a lost one couldn't take others with it.
//!
//! ```text
//! frame: u32 width, u32 height, u32 quality, u32 tile count, then per changed tile
//!        u16 tile x, u16 tile y, u32 length, coded bytes
//! ```

/// Pixels on a tile's side
pub const TILE: usize = 64;
/// Quality runs from this (smallest) to 100 (largest and sharpest)
pub const LOWEST_QUALITY: u32 = 10;

const BLOCK: usize = 8;

/// The order coefficients are written in, low frequencies first
pub(crate) const ZIGZAG: [u8; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56,
    57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// What the steps are scaled by: 1 at quality 100, 4.6 at 50 and 23 at 10; it halves about every 22
/// points
pub(crate) fn step_scale(quality: u32) -> f32 {
    let quality = quality.clamp(LOWEST_QUALITY, 100) as f32;
    (((100.0 - quality) / 22.0).exp2() * 1.0).max(1.0)
}

/// How coarsely each coefficient is kept: the eye misses high frequencies, and chroma more so
fn steps(quality: u32) -> ([f32; 64], [f32; 64]) {
    let scale = step_scale(quality);
    let (mut luma, mut chroma) = ([0.0; 64], [0.0; 64]);
    for y in 0..BLOCK {
        for x in 0..BLOCK {
            let frequency = (x + y) as f32;
            luma[y * BLOCK + x] = scale * (1.0 + frequency * 0.5);
            chroma[y * BLOCK + x] = scale * (2.0 + frequency * 1.0);
        }
    }
    (luma, chroma)
}

#[cfg(test)]
fn cosines() -> &'static [[f32; BLOCK]; BLOCK] {
    static COSINES: std::sync::OnceLock<[[f32; BLOCK]; BLOCK]> = std::sync::OnceLock::new();
    COSINES.get_or_init(|| {
        let mut table = [[0.0; BLOCK]; BLOCK];
        for (u, row) in table.iter_mut().enumerate() {
            let scale = if u == 0 { (1.0f32 / 8.0).sqrt() } else { 0.5 };
            for (x, value) in row.iter_mut().enumerate() {
                *value = scale * (std::f32::consts::PI * (2 * x + 1) as f32 * u as f32 / 16.0).cos();
            }
        }
        table
    })
}

/// cos(k * pi / 16) / 2, the transform's constants. It's orthonormal: c[4] is 1 / sqrt(8).
pub(crate) const C1: f32 = 0.490_392_64;
pub(crate) const C2: f32 = 0.461_939_77;
pub(crate) const C3: f32 = 0.415_734_8;
pub(crate) const C4: f32 = 0.353_553_4;
pub(crate) const C5: f32 = 0.277_785_12;
pub(crate) const C6: f32 = 0.191_341_72;
pub(crate) const C7: f32 = 0.097_545_16;

/// An 8-point cosine transform, by sums and differences, so it takes a quarter of the multiplications
#[inline(always)]
fn forward_8(x: [f32; 8]) -> [f32; 8] {
    let (s07, s16, s25, s34) = (x[0] + x[7], x[1] + x[6], x[2] + x[5], x[3] + x[4]);
    let (d07, d16, d25, d34) = (x[0] - x[7], x[1] - x[6], x[2] - x[5], x[3] - x[4]);
    let (a, b, c, d) = (s07 + s34, s16 + s25, s16 - s25, s07 - s34);
    [
        (a + b) * C4,
        d07 * C1 + d16 * C3 + d25 * C5 + d34 * C7,
        d * C2 + c * C6,
        d07 * C3 - d16 * C7 - d25 * C1 - d34 * C5,
        (a - b) * C4,
        d07 * C5 - d16 * C1 + d25 * C7 + d34 * C3,
        d * C6 - c * C2,
        d07 * C7 - d16 * C5 + d25 * C3 - d34 * C1,
    ]
}

/// Its inverse
#[inline(always)]
fn inverse_8(x: [f32; 8]) -> [f32; 8] {
    let (a, b) = ((x[0] + x[4]) * C4, (x[0] - x[4]) * C4);
    let (d, c) = (x[2] * C2 + x[6] * C6, x[2] * C6 - x[6] * C2);
    let (s07, s34, s16, s25) = (a + d, a - d, b + c, b - c);
    let d07 = x[1] * C1 + x[3] * C3 + x[5] * C5 + x[7] * C7;
    let d16 = x[1] * C3 - x[3] * C7 - x[5] * C1 - x[7] * C5;
    let d25 = x[1] * C5 - x[3] * C1 + x[5] * C7 + x[7] * C3;
    let d34 = x[1] * C7 - x[3] * C5 + x[5] * C3 - x[7] * C1;
    [s07 + d07, s16 + d16, s25 + d25, s34 + d34, s34 - d34, s25 - d25, s16 - d16, s07 - d07]
}

/// The 2D cosine transform: rows, then columns
fn forward(block: &mut [f32; 64]) {
    for y in 0..BLOCK {
        let row: &mut [f32; 8] = (&mut block[y * BLOCK..y * BLOCK + BLOCK]).try_into().unwrap();
        *row = forward_8(*row);
    }
    for x in 0..BLOCK {
        let column = forward_8(std::array::from_fn(|y| block[y * BLOCK + x]));
        for (y, value) in column.into_iter().enumerate() {
            block[y * BLOCK + x] = value;
        }
    }
}

fn inverse(block: &mut [f32; 64]) {
    for x in 0..BLOCK {
        let column = inverse_8(std::array::from_fn(|y| block[y * BLOCK + x]));
        for (y, value) in column.into_iter().enumerate() {
            block[y * BLOCK + x] = value;
        }
    }
    for y in 0..BLOCK {
        let row: &mut [f32; 8] = (&mut block[y * BLOCK..y * BLOCK + BLOCK]).try_into().unwrap();
        *row = inverse_8(*row);
    }
}

/// Writes bits, most significant first
#[derive(Default)]
struct BitWriter {
    bytes: Vec<u8>,
    held: u64,
    count: u32,
}

impl BitWriter {
    fn put(&mut self, value: u32, bits: u32) {
        self.held = (self.held << bits) | (value as u64 & ((1u64 << bits) - 1));
        self.count += bits;
        while self.count >= 8 {
            self.count -= 8;
            self.bytes.push((self.held >> self.count) as u8);
        }
    }

    /// Exponential Golomb: small values short
    fn number(&mut self, value: u32) {
        let shifted = value + 1;
        let bits = 32 - shifted.leading_zeros();
        self.put(0, bits - 1);
        self.put(shifted, bits);
    }

    fn signed(&mut self, value: i32) {
        self.number(((value << 1) ^ (value >> 31)) as u32);
    }

    fn finish(mut self) -> Vec<u8> {
        if self.count > 0 {
            let pad = 8 - self.count;
            self.put(0, pad);
        }
        self.bytes
    }
}

struct BitReader<'a> {
    bytes: &'a [u8],
    at: usize,
    held: u64,
    count: u32,
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0, held: 0, count: 0 }
    }

    /// Past the end it reads zeros, so a damaged tile can't run away
    fn take(&mut self, bits: u32) -> u32 {
        while self.count < bits {
            let byte = self.bytes.get(self.at).copied().unwrap_or(0);
            self.at += 1;
            self.held = (self.held << 8) | byte as u64;
            self.count += 8;
        }
        self.count -= bits;
        ((self.held >> self.count) & ((1u64 << bits) - 1)) as u32
    }

    fn number(&mut self) -> u32 {
        let mut zeros = 0;
        while self.take(1) == 0 {
            zeros += 1;
            if zeros >= 31 || self.at > self.bytes.len() + 8 {
                return 0;
            }
        }
        ((1u64 << zeros) | self.take(zeros) as u64) as u32 - 1
    }

    fn signed(&mut self) -> i32 {
        let value = self.number();
        ((value >> 1) as i32) ^ -((value & 1) as i32)
    }
}

/// One tile's planes, as floats: Y at full resolution, Co and Cg at half
struct Planes {
    width: usize,
    height: usize,
    y: Vec<f32>,
    co: Vec<f32>,
    cg: Vec<f32>,
}

fn half(size: usize) -> usize {
    size.div_ceil(2)
}

/// `pixels`: the whole frame's BGRA; the tile is at (x, y)
fn planes_of(pixels: &[u8], stride: usize, x: usize, y: usize, width: usize, height: usize) -> Planes {
    let (cw, ch) = (half(width), half(height));
    let mut planes = Planes { width, height, y: vec![0.0; width * height], co: vec![0.0; cw * ch], cg: vec![0.0; cw * ch] };
    let mut counts = vec![0.0f32; cw * ch];
    for row in 0..height {
        let line = &pixels[(y + row) * stride + x * 4..];
        for column in 0..width {
            let pixel = &line[column * 4..column * 4 + 4];
            let (b, g, r) = (pixel[0] as f32, pixel[1] as f32, pixel[2] as f32);
            let co = r - b;
            let t = b + co / 2.0;
            let cg = g - t;
            planes.y[row * width + column] = t + cg / 2.0 - 128.0;
            let at = (row / 2) * cw + column / 2;
            planes.co[at] += co;
            planes.cg[at] += cg;
            counts[at] += 1.0;
        }
    }
    for (at, &count) in counts.iter().enumerate() {
        planes.co[at] /= count;
        planes.cg[at] /= count;
    }
    planes
}

/// Codes a plane in 8x8 blocks. A block past the plane's edge repeats its last pixels.
fn code_plane(plane: &[f32], width: usize, height: usize, steps: &[f32; 64], out: &mut BitWriter) {
    let mut previous_dc = 0i32;
    for top in (0..height).step_by(BLOCK) {
        for left in (0..width).step_by(BLOCK) {
            let mut block = [0.0f32; 64];
            for y in 0..BLOCK {
                let row = (top + y).min(height - 1);
                for x in 0..BLOCK {
                    block[y * BLOCK + x] = plane[row * width + (left + x).min(width - 1)];
                }
            }
            forward(&mut block);
            let dc = (block[0] / steps[0]).round() as i32;
            out.signed(dc - previous_dc);
            previous_dc = dc;
            // Runs of zeros, then the value; a run reaching the end says so
            let mut zeros = 0u32;
            for &at in &ZIGZAG[1..] {
                let value = (block[at as usize] / steps[at as usize]).round() as i32;
                if value == 0 {
                    zeros += 1;
                } else {
                    out.number(zeros);
                    out.signed(value);
                    zeros = 0;
                }
            }
            if zeros > 0 {
                out.number(zeros);
            }
        }
    }
}

fn decode_plane(plane: &mut [f32], width: usize, height: usize, steps: &[f32; 64], r: &mut BitReader) {
    let mut previous_dc = 0i32;
    for top in (0..height).step_by(BLOCK) {
        for left in (0..width).step_by(BLOCK) {
            let mut block = [0.0f32; 64];
            previous_dc = previous_dc.wrapping_add(r.signed());
            block[0] = previous_dc as f32 * steps[0];
            let mut position = 1;
            while position < 64 {
                position += r.number() as usize;
                if position >= 64 {
                    break;
                }
                let at = ZIGZAG[position] as usize;
                block[at] = r.signed() as f32 * steps[at];
                position += 1;
            }
            inverse(&mut block);
            for y in 0..BLOCK.min(height - top) {
                for x in 0..BLOCK.min(width - left) {
                    plane[(top + y) * width + left + x] = block[y * BLOCK + x];
                }
            }
        }
    }
}

fn code_tile(pixels: &[u8], stride: usize, x: usize, y: usize, width: usize, height: usize, quality: u32) -> Vec<u8> {
    let planes = planes_of(pixels, stride, x, y, width, height);
    let (luma, chroma) = steps(quality);
    let mut out = BitWriter::default();
    code_plane(&planes.y, planes.width, planes.height, &luma, &mut out);
    let (cw, ch) = (half(width), half(height));
    code_plane(&planes.co, cw, ch, &chroma, &mut out);
    code_plane(&planes.cg, cw, ch, &chroma, &mut out);
    out.finish()
}

/// Decodes a tile into the frame's BGRA. `output` points at the frame's first pixel; only this
/// tile's pixels are written.
///
/// # Safety
/// `output` must be valid for `stride * frame_height` bytes, and (x, y, width, height) within them.
#[allow(clippy::too_many_arguments)]
unsafe fn decode_tile(coded: &[u8], output: *mut u8, stride: usize, x: usize, y: usize, width: usize, height: usize, quality: u32) {
    let (luma, chroma) = steps(quality);
    let (cw, ch) = (half(width), half(height));
    let mut planes = Planes { width, height, y: vec![0.0; width * height], co: vec![0.0; cw * ch], cg: vec![0.0; cw * ch] };
    let mut r = BitReader::new(coded);
    decode_plane(&mut planes.y, width, height, &luma, &mut r);
    decode_plane(&mut planes.co, cw, ch, &chroma, &mut r);
    decode_plane(&mut planes.cg, cw, ch, &chroma, &mut r);
    let clamp = |value: f32| (value + 0.5).clamp(0.0, 255.0) as u8;
    for row in 0..height {
        let line = unsafe { output.add((y + row) * stride + x * 4) };
        for column in 0..width {
            let luminance = planes.y[row * width + column] + 128.0;
            let at = (row / 2) * cw + column / 2;
            let (co, cg) = (planes.co[at], planes.cg[at]);
            let t = luminance - cg / 2.0;
            let g = cg + t;
            let b = t - co / 2.0;
            let r = b + co;
            unsafe {
                let pixel = line.add(column * 4);
                *pixel = clamp(b);
                *pixel.add(1) = clamp(g);
                *pixel.add(2) = clamp(r);
                *pixel.add(3) = 255;
            }
        }
    }
}

fn workers(jobs: usize) -> usize {
    std::thread::available_parallelism().map_or(1, |count| count.get()).min(jobs.max(1)).min(16)
}

static THREADS: nectar_cuda_protocol::pool::Pool = nectar_cuda_protocol::pool::Pool::new();

/// What's kept between frames to find the tiles that changed
#[derive(Default)]
pub struct Encoder {
    previous: Vec<u8>,
}

impl Encoder {
    /// Codes a frame of BGRA `pixels`, `width * height * 4` bytes, leaving out unchanged tiles
    pub fn encode(&mut self, pixels: &[u8], width: u32, height: u32, quality: u32) -> Vec<u8> {
        let (width_px, height_px, stride) = (width as usize, height as usize, width as usize * 4);
        assert!(pixels.len() >= stride * height_px);
        let same_size = self.previous.len() == pixels.len();
        let mut changed = Vec::new();
        for top in (0..height_px).step_by(TILE) {
            for left in (0..width_px).step_by(TILE) {
                let (w, h) = (TILE.min(width_px - left), TILE.min(height_px - top));
                let differs = !same_size
                    || (0..h).any(|row| {
                        let start = (top + row) * stride + left * 4;
                        pixels[start..start + w * 4] != self.previous[start..start + w * 4]
                    });
                if differs {
                    changed.push((left, top, w, h));
                }
            }
        }

        let next = std::sync::atomic::AtomicUsize::new(0);
        let coded = std::sync::Mutex::new(Vec::with_capacity(changed.len()));
        THREADS.run(workers(changed.len()) - 1, &|| {
            let mut done = Vec::new();
            loop {
                let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(&(x, y, w, h)) = changed.get(index) else { break };
                done.push((index, code_tile(pixels, stride, x, y, w, h, quality)));
            }
            coded.lock().unwrap_or_else(|p| p.into_inner()).append(&mut done);
        });
        let mut tiles: Vec<(usize, Vec<u8>)> = coded.into_inner().unwrap_or_else(|p| p.into_inner());
        tiles.sort_unstable_by_key(|tile| tile.0);

        let mut out = Vec::with_capacity(16 + tiles.iter().map(|tile| 8 + tile.1.len()).sum::<usize>());
        out.extend_from_slice(&width.to_le_bytes());
        out.extend_from_slice(&height.to_le_bytes());
        out.extend_from_slice(&quality.to_le_bytes());
        out.extend_from_slice(&(tiles.len() as u32).to_le_bytes());
        for (index, bytes) in &tiles {
            let (x, y, _, _) = changed[*index];
            out.extend_from_slice(&((x / TILE) as u16).to_le_bytes());
            out.extend_from_slice(&((y / TILE) as u16).to_le_bytes());
            out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(bytes);
        }
        self.previous.clear();
        self.previous.extend_from_slice(&pixels[..stride * height_px]);
        out
    }
}

/// Applies a coded frame to `frame`, the BGRA of the frames before it. False if it's malformed or
/// another size.
pub fn decode(coded: &[u8], frame: &mut [u8], width: u32, height: u32) -> bool {
    let word = |at: usize| coded.get(at..at + 4).map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()));
    let (Some(coded_width), Some(coded_height), Some(quality), Some(count)) = (word(0), word(4), word(8), word(12)) else {
        return false;
    };
    let (width_px, height_px, stride) = (width as usize, height as usize, width as usize * 4);
    if coded_width != width || coded_height != height || frame.len() < stride * height_px {
        return false;
    }
    let mut tiles = Vec::with_capacity(count as usize);
    let mut at = 16;
    for _ in 0..count {
        let Some(header) = coded.get(at..at + 8) else { return false };
        let x = u16::from_le_bytes([header[0], header[1]]) as usize * TILE;
        let y = u16::from_le_bytes([header[2], header[3]]) as usize * TILE;
        let length = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
        let Some(bytes) = coded.get(at + 8..at + 8 + length) else { return false };
        if x >= width_px || y >= height_px {
            return false;
        }
        tiles.push((x, y, bytes));
        at += 8 + length;
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    // The tiles don't overlap, so the threads write different pixels
    struct Frame(*mut u8);
    unsafe impl Sync for Frame {}
    let output = Frame(frame.as_mut_ptr());
    let output = &output;
    let (next, tiles_ref) = (&next, &tiles);
    THREADS.run(workers(tiles.len()) - 1, &move || {
        loop {
            let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let Some(&(x, y, bytes)) = tiles_ref.get(index) else { return };
            let (w, h) = (TILE.min(width_px - x), TILE.min(height_px - y));
            unsafe { decode_tile(bytes, output.0, stride, x, y, w, h, quality) };
        }
    });
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A smooth picture with some edges, as rendered frames are
    fn picture(width: usize, height: usize, phase: usize) -> Vec<u8> {
        let mut pixels = vec![255u8; width * height * 4];
        for y in 0..height {
            for x in 0..width {
                let pixel = &mut pixels[(y * width + x) * 4..][..4];
                pixel[0] = ((x + phase) * 255 / width.max(1)) as u8;
                pixel[1] = (y * 255 / height.max(1)) as u8;
                pixel[2] = if (x + phase) / 40 % 2 == (y / 40) % 2 { 220 } else { 40 };
            }
        }
        pixels
    }

    fn psnr(a: &[u8], b: &[u8]) -> f64 {
        let error: f64 = a
            .chunks(4)
            .zip(b.chunks(4))
            .map(|(a, b)| (0..3).map(|i| (a[i] as f64 - b[i] as f64).powi(2)).sum::<f64>())
            .sum::<f64>()
            / (a.len() as f64 / 4.0 * 3.0);
        10.0 * (255.0f64 * 255.0 / error.max(1e-9)).log10()
    }

    /// How fast and how small, on a 1080p picture with fine detail, as a game's frames have:
    /// cargo test --release -p nectar-vk-wire frame::tests::speed -- --ignored --nocapture
    #[test]
    #[ignore]
    fn speed() {
        let (width, height) = (1920usize, 1080usize);
        let detailed = |phase: usize| {
            let mut pixels = picture(width, height, phase);
            let mut noise = 0x2545_f491u32;
            for pixel in pixels.chunks_mut(4) {
                noise ^= noise << 13;
                noise ^= noise >> 17;
                noise ^= noise << 5;
                for channel in &mut pixel[..3] {
                    *channel = channel.saturating_add((noise & 15) as u8);
                }
            }
            pixels
        };
        let frames = [detailed(0), detailed(7)];
        for quality in [90, 75, 50, 30] {
            let mut encoder = Encoder::default();
            let mut shown = vec![0u8; frames[0].len()];
            let started = std::time::Instant::now();
            let mut coded = Vec::new();
            for index in 0..10 {
                coded = encoder.encode(&frames[index % 2], width as u32, height as u32, quality);
            }
            let encode = started.elapsed().as_secs_f64() * 100.0;
            let started = std::time::Instant::now();
            for _ in 0..10 {
                assert!(decode(&coded, &mut shown, width as u32, height as u32));
            }
            let decode_ms = started.elapsed().as_secs_f64() * 100.0;
            println!(
                "quality {quality}: encode {encode:.1} ms, decode {decode_ms:.1} ms, {} KB ({:.2} bits per pixel), {:.1} dB",
                coded.len() / 1000,
                coded.len() as f64 * 8.0 / (width * height) as f64,
                psnr(&frames[1], &shown)
            );
        }
    }

    #[test]
    fn frames_can_be_decoded_from_many_threads_at_once() {
        let (width, height) = (256u32, 256u32);
        let pixels = picture(width as usize, height as usize, 0);
        let coded = Encoder::default().encode(&pixels, width, height, 90);
        let expected = {
            let mut frame = vec![0u8; pixels.len()];
            assert!(decode(&coded, &mut frame, width, height));
            frame
        };
        std::thread::scope(|scope| {
            for _ in 0..6 {
                scope.spawn(|| {
                    for _ in 0..40 {
                        let mut frame = vec![0u8; expected.len()];
                        assert!(decode(&coded, &mut frame, width, height));
                        assert!(frame == expected);
                    }
                });
            }
        });
    }

    #[test]
    fn bits_round_trip() {
        let mut w = BitWriter::default();
        let values = [0i32, 1, -1, 2, -2, 7, -128, 1000, -32768];
        for &value in &values {
            w.signed(value);
        }
        w.number(0);
        w.number(70000);
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        for &value in &values {
            assert_eq!(r.signed(), value);
        }
        assert_eq!(r.number(), 0);
        assert_eq!(r.number(), 70000);
    }

    /// The fast transform is the transform: the definition, worked out in full
    #[test]
    fn the_fast_transform_matches_the_definition() {
        let table = cosines();
        let mut block = [0.0f32; 64];
        for (i, value) in block.iter_mut().enumerate() {
            *value = ((i * 37 + 11) % 255) as f32 - 128.0;
        }
        let mut expected = [0.0f32; 64];
        for v in 0..BLOCK {
            for u in 0..BLOCK {
                let mut total = 0.0;
                for y in 0..BLOCK {
                    for x in 0..BLOCK {
                        total += block[y * BLOCK + x] * table[u][x] * table[v][y];
                    }
                }
                expected[v * BLOCK + u] = total;
            }
        }
        forward(&mut block);
        for (a, b) in block.iter().zip(&expected) {
            assert!((a - b).abs() < 0.01, "{a} and {b}");
        }
    }

    #[test]
    fn the_transform_inverts() {
        let mut block = [0.0f32; 64];
        for (i, value) in block.iter_mut().enumerate() {
            *value = (i as f32 * 7.0) % 255.0 - 128.0;
        }
        let original = block;
        forward(&mut block);
        inverse(&mut block);
        for (a, b) in block.iter().zip(&original) {
            assert!((a - b).abs() < 0.01, "{a} and {b}");
        }
    }

    #[test]
    fn frames_come_back_looking_the_same_and_smaller() {
        let (width, height) = (333u32, 217u32);
        let pixels = picture(width as usize, height as usize, 0);
        for (quality, least_db, most_share) in [(95, 38.0, 0.40), (70, 33.0, 0.20), (40, 28.0, 0.10)] {
            let coded = Encoder::default().encode(&pixels, width, height, quality);
            let mut frame = vec![0u8; pixels.len()];
            assert!(decode(&coded, &mut frame, width, height));
            let db = psnr(&pixels, &frame);
            let share = coded.len() as f64 / pixels.len() as f64;
            assert!(db >= least_db, "quality {quality}: {db:.1} dB");
            assert!(share <= most_share, "quality {quality}: {share:.3} of the size");
        }
    }

    #[test]
    fn only_changed_tiles_cross() {
        let (width, height) = (256u32, 128u32);
        let mut pixels = picture(width as usize, height as usize, 0);
        let mut encoder = Encoder::default();
        let first = encoder.encode(&pixels, width, height, 80);
        let mut frame = vec![0u8; pixels.len()];
        assert!(decode(&first, &mut frame, width, height));

        let unchanged = encoder.encode(&pixels, width, height, 80);
        assert_eq!(unchanged.len(), 16);
        let copy = frame.clone();
        assert!(decode(&unchanged, &mut frame, width, height));
        assert_eq!(frame, copy);

        // One pixel, in the tile at (1, 1)
        pixels[(70 * 256 + 70) * 4] ^= 0xff;
        let one = encoder.encode(&pixels, width, height, 80);
        assert_eq!(u32::from_le_bytes(one[12..16].try_into().unwrap()), 1);
        assert_eq!(&one[16..20], &[1, 0, 1, 0]);
        assert!(decode(&one, &mut frame, width, height));
        assert_eq!(frame[..64 * 4], copy[..64 * 4]);
    }

    #[test]
    fn damaged_frames_are_refused_or_harmless() {
        let (width, height) = (128u32, 128u32);
        let pixels = picture(128, 128, 0);
        let coded = Encoder::default().encode(&pixels, width, height, 80);
        let mut frame = vec![0u8; pixels.len()];
        assert!(!decode(&coded[..10], &mut frame, width, height));
        assert!(!decode(&coded[..coded.len() - 1], &mut frame, width, height));
        assert!(!decode(&coded, &mut frame, 64, 64));
        // Wrong bits inside a tile decode to the wrong picture, not a crash
        let mut damaged = coded.clone();
        let length = u32::from_le_bytes(coded[20..24].try_into().unwrap()) as usize;
        for byte in damaged[24..24 + length].iter_mut().step_by(3) {
            *byte ^= 0x5a;
        }
        assert!(decode(&damaged, &mut frame, width, height));
        // A tile claimed outside the frame
        let mut outside = coded;
        outside[16..18].copy_from_slice(&999u16.to_le_bytes());
        assert!(!decode(&outside, &mut frame, width, height));
    }
}
