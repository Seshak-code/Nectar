//! H.264 back into pictures, with the decoder Windows comes with (Media Foundation's H.264 transform), so
//! a borrower needs nothing installed. It's asked for low latency, and the lender's stream has no frames
//! out of order, so each frame in gives its picture out.
//!
//! Media Foundation is COM, which the bindings crate leaves out, so the few interfaces used are called
//! through their tables here, by the positions `mftransform.h` and `mfobjects.h` give.
//!
//! A decoder belongs to the thread that made it.

use std::ffi::c_void;
use std::time::{Duration, Instant};
use std::{mem, ptr};

use crate::video_shader::nv12_to_bgra;

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
struct Guid(u32, u16, u16, [u8; 8]);

const fn media(four: &[u8; 4]) -> Guid {
    Guid(u32::from_le_bytes(*four), 0x0000, 0x0010, [0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71])
}

const CLSID_H264_DECODER: Guid = Guid(0x62CE7E72, 0x4C71, 0x4D20, [0xB1, 0x5D, 0x45, 0x28, 0x31, 0xA8, 0x7D, 0x9D]);
const IID_TRANSFORM: Guid = Guid(0xBF94C121, 0x5B05, 0x4E6F, [0x80, 0x00, 0xBA, 0x59, 0x89, 0x61, 0x41, 0x4D]);
const MAJOR_TYPE: Guid = Guid(0x48EBA18E, 0xF8C9, 0x4687, [0xBF, 0x11, 0x0A, 0x74, 0xC9, 0xF9, 0x6A, 0x8F]);
const SUBTYPE: Guid = Guid(0xF7E34C9A, 0x42E8, 0x4714, [0xB7, 0x4B, 0xCB, 0x29, 0xD7, 0x2C, 0x35, 0xE5]);
const FRAME_SIZE: Guid = Guid(0x1652C33D, 0xD6B2, 0x4012, [0xB8, 0x34, 0x72, 0x03, 0x08, 0x49, 0xA3, 0x7D]);
const INTERLACE_MODE: Guid = Guid(0xE2724BB8, 0xE676, 0x4806, [0xB4, 0xB2, 0xA8, 0xD6, 0xEF, 0xB4, 0x4C, 0xCD]);
const DEFAULT_STRIDE: Guid = Guid(0x644B4E48, 0x1E02, 0x4516, [0xB0, 0xEB, 0xC0, 0x1C, 0xA9, 0xD4, 0x9A, 0xC6]);
const LOW_LATENCY: Guid = Guid(0x9C27891A, 0xED7A, 0x40E1, [0x88, 0xE8, 0xB2, 0x27, 0x27, 0xA0, 0x24, 0xEE]);
const VIDEO: Guid = media(b"vids");
const H264: Guid = media(b"H264");
const NV12: Guid = media(b"NV12");

const NEED_MORE_INPUT: i32 = 0xC00D6D72u32 as i32;
const STREAM_CHANGE: i32 = 0xC00D6D61u32 as i32;
const NOT_ACCEPTING: i32 = 0xC00D36B5u32 as i32;
const PROVIDES_SAMPLES: u32 = 0x100 | 0x200;
const BEGIN_STREAMING: u32 = 0x1000_0000;
const START_OF_STREAM: u32 = 0x1000_0003;
/// A frame's time as the decoder is told it, in 100 ns: it only needs them to go up
const FRAME_TIME: i64 = 83_333;

#[link(name = "ole32")]
unsafe extern "system" {
    fn CoInitializeEx(reserved: *mut c_void, flags: u32) -> i32;
    fn CoCreateInstance(class: *const Guid, outer: *mut c_void, context: u32, interface: *const Guid, out: *mut *mut c_void) -> i32;
}

#[link(name = "mfplat")]
unsafe extern "system" {
    fn MFStartup(version: u32, flags: u32) -> i32;
    fn MFCreateMediaType(out: *mut *mut c_void) -> i32;
    fn MFCreateSample(out: *mut *mut c_void) -> i32;
    fn MFCreateMemoryBuffer(most: u32, out: *mut *mut c_void) -> i32;
}

/// A COM object, released when dropped
struct Com(*mut c_void);

impl Com {
    /// The method at `index` of the object's table
    unsafe fn method<F: Copy>(&self, index: usize) -> F {
        unsafe {
            let table = *(self.0 as *const *const *const c_void);
            mem::transmute_copy(&*table.add(index))
        }
    }
}

impl Com {
    /// The same object, held once more
    unsafe fn again(other: &Com) -> Com {
        unsafe { other.method::<unsafe extern "system" fn(*mut c_void) -> u32>(1)(other.0) };
        Com(other.0)
    }
}

impl Drop for Com {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { self.method::<unsafe extern "system" fn(*mut c_void) -> u32>(2)(self.0) };
        }
    }
}

type Get = unsafe extern "system" fn(*mut c_void, *mut *mut c_void) -> i32;
type GetAt = unsafe extern "system" fn(*mut c_void, u32, *mut *mut c_void) -> i32;
type SetType = unsafe extern "system" fn(*mut c_void, u32, *mut c_void, u32) -> i32;

fn check(what: &str, result: i32) -> Result<(), String> {
    if result < 0 { Err(format!("{what}: {:#010x}", result as u32)) } else { Ok(()) }
}

#[repr(C)]
#[derive(Default)]
struct StreamInfo {
    flags: u32,
    size: u32,
    alignment: u32,
}

#[repr(C)]
struct OutputBuffer {
    stream: u32,
    sample: *mut c_void,
    status: u32,
    events: *mut c_void,
}

pub struct Decoder {
    transform: Com,
    width: usize,
    height: usize,
    /// Of the decoder's pictures: bytes from one row to the next, and rows of luma before the chroma
    stride: usize,
    rows: usize,
    provides_samples: bool,
    sample_size: u32,
    frames: i64,
    /// The sample and buffer the decoder's pictures are taken in, kept from one to the next: a new one
    /// each time is 5 MB of memory to find and touch for a 2560x1440 frame
    spare: Option<(Com, Com)>,
    /// How long the decoder has taken over its pictures, and turning them into the window's colours
    pub decoding: Duration,
    pub colours: Duration,
}

impl Decoder {
    pub fn new(width: u32, height: u32) -> Result<Self, String> {
        unsafe {
            // Already set up some other way on this thread is fine too
            CoInitializeEx(ptr::null_mut(), 0);
            check("MFStartup", MFStartup(0x0002_0070, 1))?;
            let mut transform = ptr::null_mut();
            check("no H.264 decoder", CoCreateInstance(&CLSID_H264_DECODER, ptr::null_mut(), 1, &IID_TRANSFORM, &mut transform))?;
            let transform = Com(transform);
            let mut attributes = ptr::null_mut();
            if transform.method::<Get>(8)(transform.0, &mut attributes) >= 0 {
                let attributes = Com(attributes);
                attributes.method::<unsafe extern "system" fn(*mut c_void, *const Guid, u32) -> i32>(21)(attributes.0, &LOW_LATENCY, 1);
            }
            let mut input = ptr::null_mut();
            check("MFCreateMediaType", MFCreateMediaType(&mut input))?;
            let input = Com(input);
            let set_guid = input.method::<unsafe extern "system" fn(*mut c_void, *const Guid, *const Guid) -> i32>(24);
            check("major type", set_guid(input.0, &MAJOR_TYPE, &VIDEO))?;
            check("subtype", set_guid(input.0, &SUBTYPE, &H264))?;
            let size = (width as u64) << 32 | height as u64;
            check("frame size", input.method::<unsafe extern "system" fn(*mut c_void, *const Guid, u64) -> i32>(22)(input.0, &FRAME_SIZE, size))?;
            // Progressive
            check("interlace mode", input.method::<unsafe extern "system" fn(*mut c_void, *const Guid, u32) -> i32>(21)(input.0, &INTERLACE_MODE, 2))?;
            check("the decoder's input type", transform.method::<SetType>(15)(transform.0, 0, input.0, 0))?;
            let mut decoder = Decoder { transform, width: width as usize, height: height as usize, stride: 0, rows: 0, provides_samples: false, sample_size: 0, frames: 0, spare: None, decoding: Duration::ZERO, colours: Duration::ZERO };
            decoder.choose_output()?;
            let message = decoder.transform.method::<unsafe extern "system" fn(*mut c_void, u32, usize) -> i32>(23);
            check("begin streaming", message(decoder.transform.0, BEGIN_STREAMING, 0))?;
            check("start of stream", message(decoder.transform.0, START_OF_STREAM, 0))?;
            Ok(decoder)
        }
    }

    /// Asks for NV12, and notes how its pictures are laid out. Again whenever the decoder says its output
    /// has changed, which it does once it has read the stream's real size.
    unsafe fn choose_output(&mut self) -> Result<(), String> {
        unsafe {
            let transform = &self.transform;
            for index in 0.. {
                let mut kind = ptr::null_mut();
                check("the decoder offers no NV12", transform.method::<unsafe extern "system" fn(*mut c_void, u32, u32, *mut *mut c_void) -> i32>(14)(transform.0, 0, index, &mut kind))?;
                let kind = Com(kind);
                let mut subtype = Guid(0, 0, 0, [0; 8]);
                kind.method::<unsafe extern "system" fn(*mut c_void, *const Guid, *mut Guid) -> i32>(10)(kind.0, &SUBTYPE, &mut subtype);
                if subtype != NV12 {
                    continue;
                }
                check("the decoder's output type", transform.method::<SetType>(16)(transform.0, 0, kind.0, 0))?;
                break;
            }
            let mut current = ptr::null_mut();
            check("the decoder's output type", transform.method::<GetAt>(18)(transform.0, 0, &mut current))?;
            let current = Com(current);
            let mut size = 0u64;
            check("the picture's size", current.method::<unsafe extern "system" fn(*mut c_void, *const Guid, *mut u64) -> i32>(8)(current.0, &FRAME_SIZE, &mut size))?;
            let (picture_width, picture_height) = ((size >> 32) as usize, size as u32 as usize);
            let mut stride = 0u32;
            let has_stride = current.method::<unsafe extern "system" fn(*mut c_void, *const Guid, *mut u32) -> i32>(7)(current.0, &DEFAULT_STRIDE, &mut stride) >= 0;
            self.stride = if has_stride && (stride as i32) > 0 { stride as usize } else { picture_width };
            self.rows = picture_height;
            if picture_width < self.width || picture_height < self.height {
                return Err(format!("the decoder's pictures are {picture_width}x{picture_height}, smaller than the frame"));
            }
            let mut info = StreamInfo::default();
            check("the output stream", transform.method::<unsafe extern "system" fn(*mut c_void, u32, *mut StreamInfo) -> i32>(7)(transform.0, 0, &mut info))?;
            self.provides_samples = info.flags & PROVIDES_SAMPLES != 0;
            self.sample_size = info.size.max((self.stride * self.rows * 3 / 2) as u32);
            self.spare = None;
            Ok(())
        }
    }

    /// A sample with one buffer of `size` bytes, holding `bytes` if there are any
    unsafe fn sample_of(bytes: Option<&[u8]>, size: u32) -> Result<(Com, Com), String> {
        unsafe {
            let mut buffer = ptr::null_mut();
            check("MFCreateMemoryBuffer", MFCreateMemoryBuffer(size, &mut buffer))?;
            let buffer = Com(buffer);
            if let Some(bytes) = bytes {
                let (mut data, mut most, mut length) = (ptr::null_mut(), 0u32, 0u32);
                check("lock", buffer.method::<unsafe extern "system" fn(*mut c_void, *mut *mut u8, *mut u32, *mut u32) -> i32>(3)(buffer.0, &mut data, &mut most, &mut length))?;
                ptr::copy_nonoverlapping(bytes.as_ptr(), data, bytes.len());
                buffer.method::<unsafe extern "system" fn(*mut c_void) -> i32>(4)(buffer.0);
                check("length", buffer.method::<unsafe extern "system" fn(*mut c_void, u32) -> i32>(6)(buffer.0, bytes.len() as u32))?;
            }
            let mut sample = ptr::null_mut();
            check("MFCreateSample", MFCreateSample(&mut sample))?;
            let sample = Com(sample);
            check("add buffer", sample.method::<unsafe extern "system" fn(*mut c_void, *mut c_void) -> i32>(42)(sample.0, buffer.0))?;
            Ok((sample, buffer))
        }
    }

    /// Takes whatever pictures the decoder has ready, putting the last into `out`. Returns how many.
    unsafe fn take_pictures(&mut self, out: &mut [u8]) -> Result<usize, String> {
        unsafe {
            let mut taken = 0;
            loop {
                let started = Instant::now();
                let ours = match (self.provides_samples, self.spare.take()) {
                    (true, _) => None,
                    (false, Some(kept)) => Some(kept),
                    (false, None) => Some(Self::sample_of(None, self.sample_size)?),
                };
                let mut output = OutputBuffer { stream: 0, sample: ours.as_ref().map_or(ptr::null_mut(), |(sample, _)| sample.0), status: 0, events: ptr::null_mut() };
                let mut status = 0u32;
                let process = self.transform.method::<unsafe extern "system" fn(*mut c_void, u32, u32, *mut OutputBuffer, *mut u32) -> i32>(25);
                let result = process(self.transform.0, 0, 1, &mut output, &mut status);
                if !output.events.is_null() {
                    drop(Com(output.events));
                }
                // One the decoder made is ours to release
                let theirs = (ours.is_none() && !output.sample.is_null()).then(|| Com(output.sample));
                match result {
                    NEED_MORE_INPUT => {
                        self.spare = ours;
                        return Ok(taken);
                    }
                    STREAM_CHANGE => {
                        self.choose_output()?;
                        continue;
                    }
                    result => check("decoding", result)?,
                }
                let buffer = match (&ours, &theirs) {
                    (Some((_, buffer)), _) => Com::again(buffer),
                    (None, Some(sample)) => {
                        let mut buffer = ptr::null_mut();
                        check("the picture's bytes", sample.method::<Get>(41)(sample.0, &mut buffer))?;
                        Com(buffer)
                    }
                    (None, None) => return Err("the decoder gave no picture".to_string()),
                };
                let (mut data, mut most, mut length) = (ptr::null_mut(), 0u32, 0u32);
                check("lock", buffer.method::<unsafe extern "system" fn(*mut c_void, *mut *mut u8, *mut u32, *mut u32) -> i32>(3)(buffer.0, &mut data, &mut most, &mut length))?;
                let picture = std::slice::from_raw_parts(data, length as usize);
                let chroma = self.stride * self.rows;
                let fits = picture.len() >= chroma + self.stride * (self.height.div_ceil(2) - 1) + self.width.next_multiple_of(2);
                let decoded = Instant::now();
                self.decoding += decoded - started;
                if fits {
                    nv12_to_bgra(picture, self.stride, chroma, self.width, self.height, out);
                }
                self.colours += decoded.elapsed();
                buffer.method::<unsafe extern "system" fn(*mut c_void) -> i32>(4)(buffer.0);
                if !fits {
                    return Err(format!("a picture of {length} bytes, too few for {}x{} at {} a row", self.width, self.height, self.stride));
                }
                // Empty again for the next
                if ours.is_some() {
                    buffer.method::<unsafe extern "system" fn(*mut c_void, u32) -> i32>(6)(buffer.0, 0);
                }
                self.spare = ours;
                taken += 1;
            }
        }
    }

    /// Decodes one frame's bytes, as the lender sent them, into `out`: 4 bytes a pixel, blue first, `width`
    /// by `height`. Returns whether a picture came out; if not, `out` is as it was.
    pub fn decode(&mut self, frame: &[u8], out: &mut [u8]) -> Result<bool, String> {
        assert!(out.len() >= self.width * self.height * 4);
        unsafe {
            let started = Instant::now();
            let (sample, _) = Self::sample_of(Some(frame), frame.len() as u32)?;
            self.frames += 1;
            check("time", sample.method::<unsafe extern "system" fn(*mut c_void, i64) -> i32>(36)(sample.0, self.frames * FRAME_TIME))?;
            check("duration", sample.method::<unsafe extern "system" fn(*mut c_void, i64) -> i32>(38)(sample.0, FRAME_TIME))?;
            let input = self.transform.method::<unsafe extern "system" fn(*mut c_void, u32, *mut c_void, u32) -> i32>(24);
            let mut taken = 0;
            let mut result = input(self.transform.0, 0, sample.0, 0);
            if result == NOT_ACCEPTING {
                // It wants its pictures taken first
                taken += self.take_pictures(out)?;
                result = input(self.transform.0, 0, sample.0, 0);
            }
            check("the decoder's input", result)?;
            self.decoding += started.elapsed();
            taken += self.take_pictures(out)?;
            Ok(taken > 0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The picture `vk-probe --video` makes: blue across, green down, and squares of red
    fn picture(width: usize, height: usize, phase: usize) -> Vec<u8> {
        let mut pixels = vec![255u8; width * height * 4];
        for y in 0..height {
            for x in 0..width {
                let pixel = &mut pixels[(y * width + x) * 4..][..4];
                pixel[0] = ((x + phase) * 255 / width) as u8;
                pixel[1] = (y * 255 / height) as u8;
                pixel[2] = if (x + phase) / 40 % 2 == (y / 40) % 2 { 220 } else { 40 };
            }
        }
        pixels
    }

    fn psnr(a: &[u8], b: &[u8]) -> f64 {
        let error: f64 = a.chunks(4).zip(b.chunks(4)).map(|(a, b)| (0..3).map(|i| (a[i] as f64 - b[i] as f64).powi(2)).sum::<f64>()).sum::<f64>() / (a.len() as f64 / 4.0 * 3.0);
        10.0 * (255.0f64 * 255.0 / error.max(1e-9)).log10()
    }

    /// Two frames an RTX 5070's encoder made of that picture, 323 by 243, which is neither whole blocks
    /// nor even: a key frame and the one after it, 6 pixels on. `NECTAR_VK_PROBE_DUMP=<folder> vk-probe
    /// --video <size>` writes them.
    #[test]
    fn the_systems_decoder_gives_each_picture_as_it_comes() {
        let (width, height) = (323, 243);
        let mut decoder = Decoder::new(width, height).expect("Windows has an H.264 decoder");
        let mut shown = vec![0u8; (width * height * 4) as usize];
        for (coded, phase) in [(&include_bytes!("../tests/video/key.h264")[..], 0), (&include_bytes!("../tests/video/next.h264")[..], 6)] {
            assert_eq!(decoder.decode(coded, &mut shown), Ok(true));
            let close = psnr(&picture(width as usize, height as usize, phase), &shown);
            assert!(close > 30.0, "the picture {phase} pixels on came back at {close:.1} dB");
        }
        // Not the picture before, which is what a decoder a frame behind would give
        assert!(psnr(&picture(width as usize, height as usize, 0), &shown) < 25.0);
    }
}
