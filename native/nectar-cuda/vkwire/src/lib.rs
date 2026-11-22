//! Vulkan calls on the wire, for Nectar's Vulkan driver (the borrower) and the lender.
//!
//! - `tables` and `generated`: what every struct and command is made of, from vk.xml
//! - `wire`: the interpreter that encodes calls on the borrower and decodes them on the lender
//! - `thunks`: the lender's calls into the real driver, with arguments in u64 slots (`slots`)

mod generated;
pub mod slots;
pub mod tables;
pub mod thunks;

pub use generated::{COMMANDS, EXTENSIONS, TYPES, by_stype, command_name};
pub mod bc;
pub mod descriptor_copies;
pub mod frame;
pub mod frame_shader;
pub mod h264;
#[cfg(windows)]
pub mod video_decode;
/// Where no decoder of the system's is used yet, there's none, and frames come the other way
#[cfg(not(windows))]
pub mod video_decode {
    pub struct Decoder {
        pub decoding: std::time::Duration,
        pub colours: std::time::Duration,
    }

    impl Decoder {
        pub fn new(_width: u32, _height: u32) -> Result<Self, String> {
            Err("no H.264 decoder is used on this system yet".to_string())
        }

        pub fn decode(&mut self, _frame: &[u8], _out: &mut [u8]) -> Result<bool, String> {
            Err("no H.264 decoder is used on this system yet".to_string())
        }
    }
}
pub mod video_encode;
pub mod video_shader;
pub mod spirv;
pub mod wavelet;
pub mod wire;

#[cfg(test)]
mod tests;

/// Requests of Nectar's own, beside the Vulkan commands: mapped memory, which the borrower
/// shadows and keeps in step with the lender's mapping
pub mod messages {
    use nectar_cuda_protocol::function_id;

    /// Maps a memory object on the lender: its id, and u8 1 if its bytes are to come only as they
    /// are touched. Answer: its size (u64) then, with the lender taking the borrower to hold zeros;
    /// otherwise its contents.
    pub const MAP: u32 = function_id("nectarVkMap");
    /// Bytes the program wrote: memory id, offset, bytes. Sent without waiting.
    pub const WRITE: u32 = function_id("nectarVkWrite");
    /// What the GPU wrote since: the answer is a count of (memory id, offset, bytes)
    pub const REFRESH: u32 = function_id("nectarVkRefresh");
    /// What the GPU wrote in parts of one mapping: memory id, a count of (offset, length). The
    /// answer is as REFRESH's
    pub const FETCH: u32 = function_id("nectarVkFetch");
    /// Ends a mapping
    pub const UNMAP: u32 = function_id("nectarVkUnmap");
    /// As WRITE, for the lender to keep by content hash: memory id, offset, the 16-byte hash, bytes.
    /// So the next load of the same model or textures needs only the hash. Sent without waiting.
    pub const KEEP: u32 = function_id("nectarVkKeep");
    /// As WRITE, of bytes the lender keeps: memory id, offset, the hash, u64 size. Sent without
    /// waiting, as what the lender was given or said it had, it has: if it can't find them, nothing
    /// correct can follow and the connection ends.
    pub const KEPT: u32 = function_id("nectarVkKept");
    /// Comes from the lender in place of an answer: its cache is full or failing, so send no more KEEPs.
    /// What it was already sent, it holds in memory until the connection ends.
    pub const STOP_KEEPING: u32 = function_id("nectarVkStopKeeping");
    /// A frame the swapchain is copying to mapped memory: the swapchain's first such memory id (which
    /// names its coder), this frame's memory id, device, fence, u32 width, u32 height, u32 quality,
    /// u8 1 if that memory holds frame_shader.rs's output and 0 if it holds the frame's BGRA.
    /// Sent straight after the submit that copies it, and needs no wait for that first: the lender
    /// waits for the fence itself, then codes frames in the order they came, off the ordered stream.
    /// Answer: u32 microseconds on the lender, u32 microseconds of those coding, the frame coded by
    /// frame.rs.
    pub const FRAME: u32 = function_id("nectarVkFrame");
    /// Asks for a swapchain's frames as H.264 from the card's encoder (video_encode.rs): a key the
    /// borrower chooses for the stream, device, u32 width, u32 height. Answer, if the lender's device can:
    /// u32 the queue family that encodes, which the buffers the frames' NV12 is written to have to be
    /// shared with.
    pub const VIDEO_OPEN: u32 = function_id("nectarVkVideoOpen");
    /// One frame of such a stream: the key, the buffer its NV12 is in (video_shader.rs), device, the fence
    /// of the submit writing it, u32 quality, u8 1 to ask for a key frame. Sent straight after that submit,
    /// as FRAME is. Answer: u32 microseconds on the lender, u32 microseconds of those encoding, u8 1 if
    /// it's a key frame, the picture's bytes.
    pub const VIDEO_FRAME: u32 = function_id("nectarVkVideoFrame");
    /// Ends a stream once the frames asked for are done: the key. Answered when the encoder is gone.
    pub const VIDEO_CLOSE: u32 = function_id("nectarVkVideoClose");
    /// Waits for fences in the ordered stream, holding up what comes after: device, u32 count, the
    /// fences. Sent without waiting, for copies whose wait the borrower answered itself.
    pub const HOLD: u32 = function_id("nectarVkHold");
    /// How busy the lender's GPU is. Answer: u32 percent, or u32::MAX if it can't say.
    pub const GPU_USAGE: u32 = function_id("nectarVkGpuUsage");
    /// Asks to hear when a timeline semaphore reaches a value a submit before this signals: queue,
    /// semaphore, u64 value, then what that work writes in mapped memory, to come ahead of the news: u32
    /// count, then memory id, u64 start, u64 end. Sent without waiting; the lender sends back PUSHED if
    /// there's anything to come, then REACHED.
    pub const WATCH: u32 = function_id("nectarVkWatch");
    /// Many descriptor copies, as descriptor_copies.rs says. Sent without waiting.
    pub const DESCRIPTOR_COPIES: u32 = function_id("nectarVkDescriptorCopies");
    /// Comes from the lender in place of an answer: semaphore, u64 value
    pub const REACHED: u32 = function_id("nectarVkReached");
    /// Comes from the lender in place of an answer, just before the REACHED it's for, with what changed in
    /// what WATCH named: semaphore, u64 value, then as REFRESH's answer
    pub const PUSHED: u32 = function_id("nectarVkPushed");
    /// From here on, what the borrower sends is in protocol::pack's blocks. Sent without waiting.
    pub const PACKED: u32 = function_id("nectarVkPacked");
    /// Makes a mip that's to come later out of the one below, by repeating its blocks, in mapped memory:
    /// the one below's memory id and offset, u32 blocks across and down; this mip's memory id and
    /// offset, u32 blocks across and down; u32 bytes a block; u32 a format whose blocks are to be made flat
    /// first, or 0 where they're not to be, as they'd be if they're already a stand-in. Sent without waiting.
    pub const FILL_MIP: u32 = function_id("nectarVkFillMip");
    /// As FILL_MIP where there's nothing below to use yet, from a sample of the mip's own blocks: its
    /// memory id and offset, u32 blocks across and down; u32 bytes a block; u32 how many blocks each
    /// sampled block stands for, across and down; the samples. Sent without waiting.
    pub const FILL_MIP_FROM: u32 = function_id("nectarVkFillMipFrom");
    /// Part of a mip coded by wavelet.rs: the memory id it's to end up in, the bytes. Sent without waiting.
    pub const CODED_PART: u32 = function_id("nectarVkCodedPart");
    /// Decodes what CODED_PARTs brought into that memory as blocks, off the ordered stream: memory id,
    /// u32 format, u32 blocks across and down. Answered when it's done, with a result that isn't 0 if
    /// it couldn't be.
    pub const DECODE_MIP: u32 = function_id("nectarVkDecodeMip");
    /// Answered when everything before it has been dealt with. Answer: nothing. For telling how much
    /// is queued along the way.
    pub const ECHO: u32 = function_id("nectarVkEcho");
    /// Bytes per page that memory is compared and sent in
    pub const PAGE: usize = 4096;
}

/// Fills `to` by repeating blocks of `from`, each standing for `scale` blocks across and down. Sizes are
/// in blocks; `block` is bytes a block, a texel if it's not compressed. Where a size is odd the last
/// block stands in for what's past it.
#[allow(clippy::too_many_arguments)]
pub fn expand_mip(from: &[u8], from_across: usize, from_down: usize, to: &mut [u8], across: usize, down: usize, block: usize, scale: usize) {
    let (from_pitch, pitch) = (from_across * block, across * block);
    if pitch == 0 || from_pitch == 0 || scale == 0 || from.len() < from_pitch * from_down {
        return;
    }
    for (y, row) in to.chunks_exact_mut(pitch).take(down).enumerate() {
        let source = &from[(y / scale).min(from_down - 1) * from_pitch..][..from_pitch];
        for (x, piece) in row.chunks_exact_mut(block).enumerate() {
            piece.copy_from_slice(&source[(x / scale).min(from_across - 1) * block..][..block]);
        }
    }
}

/// Every `scale`th block of a mip across and down, which expand_mip makes do with. Returns the samples
/// and how many there are across and down.
pub fn sample_mip(bytes: &[u8], across: usize, down: usize, block: usize, scale: usize) -> (Vec<u8>, usize, usize) {
    let (sampled_across, sampled_down) = (across.div_ceil(scale), down.div_ceil(scale));
    let mut samples = Vec::with_capacity(sampled_across * sampled_down * block);
    for y in (0..down).step_by(scale) {
        for x in (0..across).step_by(scale) {
            samples.extend_from_slice(&bytes[(y * across + x) * block..][..block]);
        }
    }
    (samples, sampled_across, sampled_down)
}
