//! What's being copied out of mapped memory, and where to. A texture or a model's tensor goes through
//! staging memory at whatever offset was free, beside whatever else was going, so the bytes around it
//! say nothing about where it starts. The copy commands do: each names the exact bytes, so memory.rs
//! can hash those and the same texture has the same hash wherever it sits.
//!
//! Regions only divide what memory.rs sends. It still sends every byte the program wrote, so a size
//! worked out wrong here costs a repeat its hash, and nothing else.

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::sync::Mutex;

use nectar_vk_sys::*;

use crate::memory;

/// Bytes a copy reads from mapped memory
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Region {
    pub memory: u64,
    pub offset: usize,
    pub size: usize,
    /// Where the copy goes, which says whether this is new data or something being rewritten
    pub place: (u64, u64),
    /// The mip of the image `place` names, if this is all of one that could come later
    pub mip: Option<u32>,
    /// A block's bytes, and its width and height
    pub block: (usize, usize, usize),
    /// The size in texels of what's copied
    pub extent: (u32, u32),
}

#[derive(Default)]
struct Uploads {
    /// What each command buffer copies, since it began recording
    recording: HashMap<u64, Vec<Region>>,
    /// Each image's format
    formats: HashMap<u64, VkFormat>,
    /// When each command buffer that's finished recording finished, by memory.rs's count
    ended: HashMap<u64, u64>,
    /// Those that have been submitted since they began
    submitted: HashSet<u64>,
}

/// What a submit means for the memory that's waiting to go
#[derive(Default)]
pub struct Submit {
    /// What it copies
    pub regions: Vec<Region>,
    /// The latest its command buffers finished recording, if writes after that can wait
    pub due: Option<u64>,
    /// What command buffers not yet submitted copy, which can wait for them, if writes can wait
    pub elsewhere: Vec<Region>,
    /// Its command buffers
    pub commands: Vec<u64>,
    /// Its queue
    pub queue: usize,
}

/// A copy as uploads.rs takes it in
struct Copy {
    offset: u64,
    size: u64,
    place: (u64, u64),
    mip: Option<u32>,
    block: (usize, usize, usize),
    extent: (u32, u32),
}

fn buffer_copy(offset: u64, size: u64, place: (u64, u64)) -> Copy {
    Copy { offset, size, place, mip: None, block: (1, 1, 1), extent: (0, 0) }
}

/// A copy to an image, if the size of what it reads can be worked out
#[allow(clippy::too_many_arguments)]
fn image_copy(format: VkFormat, image: u64, offset: u64, row_length: u32, image_height: u32, subresource: &VkImageSubresourceLayers, to: VkOffset3D, extent: VkExtent3D) -> Option<Copy> {
    let size = image_copy_size(format, row_length, image_height, extent, subresource.layerCount)?;
    Some(Copy {
        offset,
        size: size as u64,
        place: (image, image_place(subresource.mipLevel, subresource.baseArrayLayer, to)),
        mip: crate::later::whole_mip(image, subresource.mipLevel, row_length, image_height, to, extent),
        block: block_of(format)?,
        extent: (extent.width, extent.height),
    })
}

static UPLOADS: Mutex<Option<Uploads>> = Mutex::new(None);

fn uploads<R>(f: impl FnOnce(&mut Uploads) -> R) -> R {
    f(UPLOADS.lock().unwrap_or_else(|p| p.into_inner()).get_or_insert_with(Uploads::default))
}

/// A format's texel block: its bytes, width and height. None for formats whose copies don't follow
/// these rules (depth, stencil, several planes), or that aren't known.
fn block_of(format: VkFormat) -> Option<(usize, usize, usize)> {
    const ASTC: [(usize, usize); 14] =
        [(4, 4), (5, 4), (5, 5), (6, 5), (6, 6), (8, 5), (8, 6), (8, 8), (10, 5), (10, 6), (10, 8), (10, 10), (12, 10), (12, 12)];
    let plain = |bytes| Some((bytes, 1, 1));
    match format {
        1 | 9..=15 => plain(1),
        2..=8 | 16..=22 | 70..=76 => plain(2),
        23..=36 => plain(3),
        37..=69 | 77..=83 | 98..=100 | 122 | 123 => plain(4),
        84..=90 => plain(6),
        91..=97 | 101..=103 | 110..=112 => plain(8),
        104..=106 => plain(12),
        107..=109 | 113..=115 => plain(16),
        116..=118 => plain(24),
        119..=121 => plain(32),
        // BC1 and BC4, then BC2, BC3, BC5, BC6H and BC7
        131..=134 | 139 | 140 => Some((8, 4, 4)),
        135..=138 | 141..=146 => Some((16, 4, 4)),
        // ETC2 and EAC
        147..=150 | 153 | 154 => Some((8, 4, 4)),
        151 | 152 | 155 | 156 => Some((16, 4, 4)),
        157..=184 => {
            let (width, height) = ASTC[(format as usize - 157) / 2];
            Some((16, width, height))
        }
        _ => None,
    }
}

/// An image's format, if it's been seen being created
pub fn image_format(image: u64) -> Option<VkFormat> {
    uploads(|uploads| uploads.formats.get(&image).copied())
}

/// The bytes a copy to an image reads, by Vulkan's rules for where a buffer's texels are. The last
/// row of the last slice ends after its last texel, not after a whole row's pitch.
pub fn image_copy_size(format: VkFormat, row_length: u32, image_height: u32, extent: VkExtent3D, layers: u32) -> Option<usize> {
    let (bytes, block_width, block_height) = block_of(format)?;
    if extent.width == 0 || extent.height == 0 || extent.depth == 0 || layers == 0 {
        return None;
    }
    let across = (if row_length == 0 { extent.width } else { row_length }) as usize;
    let down = (if image_height == 0 { extent.height } else { image_height }) as usize;
    let pitch = across.div_ceil(block_width) * bytes;
    let slice = down.div_ceil(block_height) * pitch;
    let rows = (extent.height as usize).div_ceil(block_height);
    let width = (extent.width as usize).div_ceil(block_width) * bytes;
    Some((extent.depth as usize * layers as usize - 1) * slice + (rows - 1) * pitch + width)
}

/// Mip, layer and corner together, so a part of an image rewritten is told from another part
fn image_place(mip: u32, layer: u32, offset: VkOffset3D) -> u64 {
    ((mip as u64) << 56) ^ ((layer as u64) << 40) ^ ((offset.z as u32 as u64) << 32) ^ ((offset.y as u32 as u64) << 16) ^ offset.x as u32 as u64
}

/// Notes a command before it goes out
///
/// # Safety
/// `args` must be the command's arguments, each by its address.
pub unsafe fn note(name: &str, args: &[*const c_void]) {
    let value = |index: usize| unsafe { *(args[index] as *const u64) };
    unsafe {
        match name {
            "vkBeginCommandBuffer" => {
                let commands = value(0);
                uploads(|uploads| {
                    uploads.recording.insert(commands, Vec::new());
                    uploads.ended.remove(&commands);
                    uploads.submitted.remove(&commands);
                });
            }
            "vkEndCommandBuffer" => {
                let commands = value(0);
                let ended = memory::command_buffer_ended();
                uploads(|uploads| uploads.ended.insert(commands, ended));
            }
            "vkFreeCommandBuffers" => {
                let list = std::slice::from_raw_parts(*(args[3] as *const *const u64), *(args[2] as *const u32) as usize);
                uploads(|uploads| {
                    for commands in list {
                        uploads.recording.remove(commands);
                        uploads.ended.remove(commands);
                        uploads.submitted.remove(commands);
                    }
                });
            }
            "vkCmdExecuteCommands" => {
                let list = std::slice::from_raw_parts(*(args[2] as *const *const u64), *(args[1] as *const u32) as usize);
                let commands = value(0);
                uploads(|uploads| {
                    let inner: Vec<Region> = list.iter().filter_map(|inner| uploads.recording.get(inner)).flatten().copied().collect();
                    uploads.recording.entry(commands).or_default().extend(inner);
                });
            }
            "vkCmdCopyBuffer" => {
                let list = std::slice::from_raw_parts(*(args[4] as *const *const VkBufferCopy), *(args[3] as *const u32) as usize);
                add(value(0), value(1), list.iter().map(|copy| buffer_copy(copy.srcOffset, copy.size, (value(2), copy.dstOffset))));
            }
            "vkCmdCopyBuffer2" => {
                let info = &**(args[1] as *const *const VkCopyBufferInfo2);
                let list = std::slice::from_raw_parts(info.pRegions, info.regionCount as usize);
                add(value(0), info.srcBuffer, list.iter().map(|copy| buffer_copy(copy.srcOffset, copy.size, (info.dstBuffer, copy.dstOffset))));
            }
            "vkCmdCopyBufferToImage" => {
                let list = std::slice::from_raw_parts(*(args[5] as *const *const VkBufferImageCopy), *(args[4] as *const u32) as usize);
                let image = value(2);
                let Some(format) = uploads(|uploads| uploads.formats.get(&image).copied()) else { return };
                add(
                    value(0),
                    value(1),
                    list.iter().filter_map(|copy| image_copy(format, image, copy.bufferOffset, copy.bufferRowLength, copy.bufferImageHeight, &copy.imageSubresource, copy.imageOffset, copy.imageExtent)),
                );
            }
            "vkCmdCopyBufferToImage2" => {
                let info = &**(args[1] as *const *const VkCopyBufferToImageInfo2);
                let list = std::slice::from_raw_parts(info.pRegions, info.regionCount as usize);
                let image = info.dstImage;
                let Some(format) = uploads(|uploads| uploads.formats.get(&image).copied()) else { return };
                add(
                    value(0),
                    info.srcBuffer,
                    list.iter().filter_map(|copy| image_copy(format, image, copy.bufferOffset, copy.bufferRowLength, copy.bufferImageHeight, &copy.imageSubresource, copy.imageOffset, copy.imageExtent)),
                );
            }
            "vkDestroyImage" => {
                let image = value(1);
                uploads(|uploads| uploads.formats.remove(&image));
            }
            _ => {}
        }
    }
}

fn add(commands: u64, source: u64, copies: impl Iterator<Item = Copy>) {
    let Some((memory, base)) = memory::buffer_place(source) else { return };
    let regions: Vec<Region> = copies
        .map(|copy| Region {
            memory,
            offset: (base + copy.offset) as usize,
            size: copy.size as usize,
            place: copy.place,
            mip: copy.mip,
            block: copy.block,
            extent: copy.extent,
        })
        .collect();
    // A command buffer not seen beginning has an unknown past, but this much of it is known
    uploads(|uploads| uploads.recording.entry(commands).or_default().extend(regions));
}

/// What command buffers not yet submitted copy
pub fn elsewhere() -> Vec<Region> {
    uploads(|uploads| {
        uploads
            .recording
            .iter()
            .filter(|(commands, _)| !uploads.submitted.contains(commands))
            .flat_map(|(_, regions)| regions.iter().copied())
            .collect()
    })
}

/// An image was created, which chose its id
pub fn image_created(image: u64, format: VkFormat) {
    uploads(|uploads| uploads.formats.insert(image, format));
}

/// The command buffers of a submit
///
/// # Safety
/// `args` must be the arguments of `name`, each by its address.
pub unsafe fn commands_of(name: &str, args: &[*const c_void]) -> Vec<u64> {
    let mut commands = Vec::new();
    unsafe {
        let count = *(args[1] as *const u32) as usize;
        match name {
            "vkQueueSubmit" if count > 0 => {
                for submit in std::slice::from_raw_parts(*(args[2] as *const *const VkSubmitInfo), count) {
                    if submit.commandBufferCount > 0 {
                        let list = std::slice::from_raw_parts(submit.pCommandBuffers, submit.commandBufferCount as usize);
                        commands.extend(list.iter().map(|&commands| commands as usize as u64));
                    }
                }
            }
            "vkQueueSubmit2" if count > 0 => {
                for submit in std::slice::from_raw_parts(*(args[2] as *const *const VkSubmitInfo2), count) {
                    if submit.commandBufferInfoCount > 0 {
                        let list = std::slice::from_raw_parts(submit.pCommandBufferInfos, submit.commandBufferInfoCount as usize);
                        commands.extend(list.iter().map(|info| info.commandBuffer as usize as u64));
                    }
                }
            }
            _ => {}
        }
    }
    commands
}

/// What a submit copies, and what can wait
///
/// # Safety
/// `args` must be the arguments of `name`, each by its address.
pub unsafe fn of_submit(name: &str, args: &[*const c_void]) -> Submit {
    let commands = unsafe { commands_of(name, args) };
    let holding = memory::holding_writes();
    uploads(|uploads| {
        let regions = commands.iter().filter_map(|commands| uploads.recording.get(commands)).flatten().copied().collect();
        let queue = unsafe { *(args[0] as *const u64) } as usize;
        let mut submit = Submit { regions, commands: commands.clone(), queue, ..Submit::default() };
        if holding {
            submit.due = Some(commands.iter().filter_map(|commands| uploads.ended.get(commands)).copied().max().unwrap_or(0));
            uploads.submitted.extend(commands.iter().copied());
            submit.elsewhere = uploads
                .recording
                .iter()
                .filter(|(commands, _)| !uploads.submitted.contains(commands))
                .flat_map(|(_, regions)| regions.iter().copied())
                .collect();
        }
        submit
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extent(width: u32, height: u32, depth: u32) -> VkExtent3D {
        VkExtent3D { width, height, depth }
    }

    #[test]
    fn formats_have_the_right_blocks() {
        assert_eq!(block_of(VK_FORMAT_R4G4_UNORM_PACK8), Some((1, 1, 1)));
        assert_eq!(block_of(VK_FORMAT_R8G8B8A8_UNORM), Some((4, 1, 1)));
        assert_eq!(block_of(VK_FORMAT_R64G64B64A64_SFLOAT), Some((32, 1, 1)));
        assert_eq!(block_of(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32), Some((4, 1, 1)));
        assert_eq!(block_of(VK_FORMAT_BC1_RGB_UNORM_BLOCK), Some((8, 4, 4)));
        assert_eq!(block_of(VK_FORMAT_BC4_UNORM_BLOCK), Some((8, 4, 4)));
        assert_eq!(block_of(VK_FORMAT_BC5_UNORM_BLOCK), Some((16, 4, 4)));
        assert_eq!(block_of(VK_FORMAT_BC6H_UFLOAT_BLOCK), Some((16, 4, 4)));
        assert_eq!(block_of(VK_FORMAT_BC7_SRGB_BLOCK), Some((16, 4, 4)));
        assert_eq!(block_of(VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK), Some((8, 4, 4)));
        assert_eq!(block_of(VK_FORMAT_EAC_R11G11_SNORM_BLOCK), Some((16, 4, 4)));
        assert_eq!(block_of(VK_FORMAT_ASTC_4x4_UNORM_BLOCK), Some((16, 4, 4)));
        assert_eq!(block_of(VK_FORMAT_ASTC_12x12_SRGB_BLOCK), Some((16, 12, 12)));
        assert_eq!(block_of(VK_FORMAT_D32_SFLOAT), None);
        assert_eq!(block_of(VK_FORMAT_UNDEFINED), None);
    }

    #[test]
    fn copy_sizes_follow_vulkans_rules() {
        // Tightly packed
        assert_eq!(image_copy_size(VK_FORMAT_R8G8B8A8_UNORM, 0, 0, extent(16, 8, 1), 1), Some(512));
        // A longer row's pitch counts, except after the last row
        assert_eq!(image_copy_size(VK_FORMAT_R8G8B8A8_UNORM, 32, 0, extent(16, 8, 1), 1), Some(7 * 128 + 64));
        // Blocks round up: 5x5 texels of BC7 are 2x2 blocks
        assert_eq!(image_copy_size(VK_FORMAT_BC7_UNORM_BLOCK, 0, 0, extent(5, 5, 1), 1), Some(64));
        assert_eq!(image_copy_size(VK_FORMAT_BC1_RGB_UNORM_BLOCK, 0, 0, extent(1024, 1024, 1), 1), Some(256 * 256 * 8));
        // Layers and slices each take a whole slice's room, by the image height given
        assert_eq!(image_copy_size(VK_FORMAT_R8G8B8A8_UNORM, 0, 16, extent(16, 8, 1), 3), Some(2 * 16 * 64 + 8 * 64));
        assert_eq!(image_copy_size(VK_FORMAT_R8G8B8A8_UNORM, 0, 0, extent(4, 4, 2), 1), Some(128));
        assert_eq!(image_copy_size(VK_FORMAT_D32_SFLOAT, 0, 0, extent(4, 4, 1), 1), None);
        assert_eq!(image_copy_size(VK_FORMAT_R8G8B8A8_UNORM, 0, 0, extent(0, 4, 1), 1), None);
    }

    #[test]
    fn a_submit_gets_what_its_command_buffers_copy() {
        let (commands, other, source, target) = (7701u64, 7702u64, 7703u64, 7704u64);
        memory::buffer_created(source, 1 << 20, 0x1, false);
        memory::buffer_bound(source, 7705, 4096);
        let begin = |commands: &u64| unsafe { note("vkBeginCommandBuffer", &[commands as *const u64 as *const c_void]) };
        begin(&commands);
        begin(&other);
        let copies = [VkBufferCopy { srcOffset: 100, dstOffset: 200, size: 300 }];
        let (count, list) = (1u32, copies.as_ptr());
        let args = [
            &commands as *const u64 as *const c_void,
            &source as *const u64 as *const c_void,
            &target as *const u64 as *const c_void,
            &count as *const u32 as *const c_void,
            &list as *const *const VkBufferCopy as *const c_void,
        ];
        unsafe { note("vkCmdCopyBuffer", &args) };
        let expected = vec![Region { memory: 7705, offset: 4196, size: 300, place: (target, 200), mip: None, block: (1, 1, 1), extent: (0, 0) }];

        let submit = |commands: &u64| VkSubmitInfo {
            sType: VK_STRUCTURE_TYPE_SUBMIT_INFO,
            pNext: std::ptr::null(),
            waitSemaphoreCount: 0,
            pWaitSemaphores: std::ptr::null(),
            pWaitDstStageMask: std::ptr::null(),
            commandBufferCount: 1,
            pCommandBuffers: commands as *const u64 as *const VkCommandBuffer,
            signalSemaphoreCount: 0,
            pSignalSemaphores: std::ptr::null(),
        };
        let of = |info: VkSubmitInfo| {
            let (queue, count, list) = (0u64, 1u32, &info as *const VkSubmitInfo);
            let args = [&queue as *const u64 as *const c_void, &count as *const u32 as *const c_void, &list as *const *const VkSubmitInfo as *const c_void];
            unsafe { of_submit("vkQueueSubmit", &args) }.regions
        };
        assert_eq!(of(submit(&commands)), expected);
        assert_eq!(of(submit(&other)), Vec::new());
        // Submitted again, it copies the same; begun again, nothing yet
        assert_eq!(of(submit(&commands)), expected);
        begin(&commands);
        assert_eq!(of(submit(&commands)), Vec::new());
    }
}
