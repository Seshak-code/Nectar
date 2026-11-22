//! vk-probe --bc-check: vkwire's block decoders against this GPU's, on random blocks. Blocks are copied
//! to an image, the GPU blits that to plain 8-bit texels, and both must agree. BC7 is defined exactly, so
//! it must match exactly. The others leave GPUs freedom in how they work out colours between their
//! endpoints, and the RTX 5070 differs by up to 7 of 255, so a little more than that is allowed.

use std::ffi::CStr;
use std::{mem, ptr};

use nectar_vk_sys::*;
use nectar_vk_wire::bc::{self, Kind};

use crate::{check, fail};

const HOST_VISIBLE: u32 = 0x2;
const HOST_COHERENT: u32 = 0x4;
const DEVICE_LOCAL: u32 = 0x1;
const SIDE: u32 = 512;

/// # Safety
/// The handles must be valid, and `queue` of `family`.
pub unsafe fn run(
    physical: VkPhysicalDevice,
    device: VkDevice,
    queue: VkQueue,
    family: u32,
    get_memory_properties: PFN_vkGetPhysicalDeviceMemoryProperties,
    device_function: &dyn Fn(&CStr) -> PFN_vkVoidFunction,
) {
    macro_rules! f {
        ($name:ident, $pfn:ty) => {
            mem::transmute::<PFN_vkVoidFunction, Option<$pfn>>(device_function(&std::ffi::CString::new(stringify!($name)).unwrap()))
                .unwrap_or_else(|| panic!("no {}", stringify!($name)))
        };
    }
    unsafe {
        let mut properties: VkPhysicalDeviceMemoryProperties = mem::zeroed();
        get_memory_properties(physical, &mut properties);
        let (types, count) = (properties.memoryTypes, properties.memoryTypeCount as usize);
        let allocate = |needs: VkMemoryRequirements, wanted: u32| -> VkDeviceMemory {
            let kind = (0..count)
                .find(|&i| needs.memoryTypeBits & (1 << i) != 0 && types[i].propertyFlags & wanted == wanted)
                .unwrap_or_else(|| fail("no memory type", VK_ERROR_OUT_OF_DEVICE_MEMORY));
            let info = VkMemoryAllocateInfo { sType: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, pNext: ptr::null(), allocationSize: needs.size, memoryTypeIndex: kind as u32 };
            let mut memory = 0;
            check("vkAllocateMemory", f!(vkAllocateMemory, PFN_vkAllocateMemory)(device, &info, ptr::null(), &mut memory));
            memory
        };
        let make_buffer = |size: u64, usage: u32| -> (VkBuffer, *mut u8) {
            let info = VkBufferCreateInfo {
                sType: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                size,
                usage,
                sharingMode: VK_SHARING_MODE_EXCLUSIVE,
                queueFamilyIndexCount: 0,
                pQueueFamilyIndices: ptr::null(),
            };
            let mut buffer = 0;
            check("vkCreateBuffer", f!(vkCreateBuffer, PFN_vkCreateBuffer)(device, &info, ptr::null(), &mut buffer));
            let mut needs: VkMemoryRequirements = mem::zeroed();
            f!(vkGetBufferMemoryRequirements, PFN_vkGetBufferMemoryRequirements)(device, buffer, &mut needs);
            let memory = allocate(needs, HOST_VISIBLE | HOST_COHERENT);
            check("vkBindBufferMemory", f!(vkBindBufferMemory, PFN_vkBindBufferMemory)(device, buffer, memory, 0));
            let mut pointer = ptr::null_mut();
            check("vkMapMemory", f!(vkMapMemory, PFN_vkMapMemory)(device, memory, 0, size, 0, &mut pointer));
            (buffer, pointer as *mut u8)
        };
        let make_image = |format: VkFormat, usage: u32| -> VkImage {
            let info = VkImageCreateInfo {
                sType: VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                imageType: 1,
                format,
                extent: VkExtent3D { width: SIDE, height: SIDE, depth: 1 },
                mipLevels: 1,
                arrayLayers: 1,
                samples: 1,
                tiling: 0,
                usage,
                sharingMode: VK_SHARING_MODE_EXCLUSIVE,
                queueFamilyIndexCount: 0,
                pQueueFamilyIndices: ptr::null(),
                initialLayout: VK_IMAGE_LAYOUT_UNDEFINED,
            };
            let mut image = 0;
            check("vkCreateImage", f!(vkCreateImage, PFN_vkCreateImage)(device, &info, ptr::null(), &mut image));
            let mut needs: VkMemoryRequirements = mem::zeroed();
            f!(vkGetImageMemoryRequirements, PFN_vkGetImageMemoryRequirements)(device, image, &mut needs);
            let memory = allocate(needs, DEVICE_LOCAL);
            check("vkBindImageMemory", f!(vkBindImageMemory, PFN_vkBindImageMemory)(device, image, memory, 0));
            image
        };
        let pool_info = VkCommandPoolCreateInfo { sType: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, pNext: ptr::null(), flags: VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, queueFamilyIndex: family };
        let mut pool = 0;
        check("vkCreateCommandPool", f!(vkCreateCommandPool, PFN_vkCreateCommandPool)(device, &pool_info, ptr::null(), &mut pool));
        let allocate_info = VkCommandBufferAllocateInfo { sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, pNext: ptr::null(), commandPool: pool, level: VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount: 1 };
        let mut commands: VkCommandBuffer = mem::zeroed();
        check("vkAllocateCommandBuffers", f!(vkAllocateCommandBuffers, PFN_vkAllocateCommandBuffers)(device, &allocate_info, &mut commands));

        let target = make_image(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        let (out, out_pointer) = make_buffer(SIDE as u64 * SIDE as u64 * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT);

        let cases = [
            ("BC1", VK_FORMAT_BC1_RGBA_UNORM_BLOCK, Kind::Bc1, 8),
            ("BC2", VK_FORMAT_BC2_UNORM_BLOCK, Kind::Bc2, 8),
            ("BC3", VK_FORMAT_BC3_UNORM_BLOCK, Kind::Bc3, 8),
            ("BC4", VK_FORMAT_BC4_UNORM_BLOCK, Kind::Bc4, 8),
            ("BC5", VK_FORMAT_BC5_UNORM_BLOCK, Kind::Bc5, 8),
            ("BC7", VK_FORMAT_BC7_UNORM_BLOCK, Kind::Bc7, 0),
        ];
        let mut noise = 0x2545_f491_4f6c_dd1du64;
        let mut failed = false;
        for (name, format, kind, allowed) in cases {
            let blocks = (SIDE / 4) as usize;
            let size = blocks * blocks * kind.block_bytes();
            let (source, source_pointer) = make_buffer(size as u64, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
            let image = make_image(format, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
            let bytes = std::slice::from_raw_parts_mut(source_pointer, size);
            for word in bytes.as_chunks_mut::<8>().0 {
                noise ^= noise << 13;
                noise ^= noise >> 7;
                noise ^= noise << 17;
                *word = noise.to_le_bytes();
            }
            if kind == Kind::Bc7 {
                // Random bytes are nearly all mode 0: an even share of each
                for (at, block) in bytes.as_chunks_mut::<16>().0.iter_mut().enumerate() {
                    let mode = at % 8;
                    block[0] = (block[0] & !(((1u16 << (mode + 1)) - 1) as u8)) | (1 << mode);
                }
            }
            let begin = VkCommandBufferBeginInfo { sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, pNext: ptr::null(), flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, pInheritanceInfo: ptr::null() };
            check("vkBeginCommandBuffer", f!(vkBeginCommandBuffer, PFN_vkBeginCommandBuffer)(commands, &begin));
            let range = VkImageSubresourceRange { aspectMask: VK_IMAGE_ASPECT_COLOR_BIT, baseMipLevel: 0, levelCount: 1, baseArrayLayer: 0, layerCount: 1 };
            let barrier = |image: VkImage, from: VkImageLayout, to: VkImageLayout| VkImageMemoryBarrier {
                sType: VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                pNext: ptr::null(),
                srcAccessMask: VK_ACCESS_TRANSFER_WRITE_BIT,
                dstAccessMask: VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                oldLayout: from,
                newLayout: to,
                srcQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
                image,
                subresourceRange: range,
            };
            let transfer = VK_PIPELINE_STAGE_TRANSFER_BIT;
            let pipeline_barrier = f!(vkCmdPipelineBarrier, PFN_vkCmdPipelineBarrier);
            let start = [barrier(image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL), barrier(target, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)];
            pipeline_barrier(commands, transfer, transfer, 0, 0, ptr::null(), 0, ptr::null(), 2, start.as_ptr());
            let layers = VkImageSubresourceLayers { aspectMask: VK_IMAGE_ASPECT_COLOR_BIT, mipLevel: 0, baseArrayLayer: 0, layerCount: 1 };
            let whole = VkBufferImageCopy {
                bufferOffset: 0,
                bufferRowLength: 0,
                bufferImageHeight: 0,
                imageSubresource: layers,
                imageOffset: VkOffset3D { x: 0, y: 0, z: 0 },
                imageExtent: VkExtent3D { width: SIDE, height: SIDE, depth: 1 },
            };
            f!(vkCmdCopyBufferToImage, PFN_vkCmdCopyBufferToImage)(commands, source, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &whole);
            let to_read = barrier(image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            pipeline_barrier(commands, transfer, transfer, 0, 0, ptr::null(), 0, ptr::null(), 1, &to_read);
            let corners = [VkOffset3D { x: 0, y: 0, z: 0 }, VkOffset3D { x: SIDE as i32, y: SIDE as i32, z: 1 }];
            let blit = VkImageBlit { srcSubresource: layers, srcOffsets: corners, dstSubresource: layers, dstOffsets: corners };
            f!(vkCmdBlitImage, PFN_vkCmdBlitImage)(commands, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
            let to_copy = barrier(target, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            pipeline_barrier(commands, transfer, transfer, 0, 0, ptr::null(), 0, ptr::null(), 1, &to_copy);
            f!(vkCmdCopyImageToBuffer, PFN_vkCmdCopyImageToBuffer)(commands, target, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, out, 1, &whole);
            check("vkEndCommandBuffer", f!(vkEndCommandBuffer, PFN_vkEndCommandBuffer)(commands));
            let submit = VkSubmitInfo {
                sType: VK_STRUCTURE_TYPE_SUBMIT_INFO,
                pNext: ptr::null(),
                waitSemaphoreCount: 0,
                pWaitSemaphores: ptr::null(),
                pWaitDstStageMask: ptr::null(),
                commandBufferCount: 1,
                pCommandBuffers: &commands,
                signalSemaphoreCount: 0,
                pSignalSemaphores: ptr::null(),
            };
            check("vkQueueSubmit", f!(vkQueueSubmit, PFN_vkQueueSubmit)(queue, 1, &submit, 0));
            check("vkQueueWaitIdle", f!(vkQueueWaitIdle, PFN_vkQueueWaitIdle)(queue));

            let shown = std::slice::from_raw_parts(out_pointer, SIDE as usize * SIDE as usize * 4);
            let mut pixels = [0u8; 64];
            let (mut wrong_blocks, mut worst) = (0usize, 0i32);
            let mut wrong_modes = [0usize; 9];
            let mut first_wrong = None;
            for (at, block) in bytes.chunks_exact(kind.block_bytes()).enumerate() {
                bc::decode_block(kind, block, &mut pixels);
                let (x, y) = (at % blocks, at / blocks);
                let mut wrong = false;
                for texel in 0..16 {
                    let row = (y * 4 + texel / 4) * SIDE as usize + x * 4 + texel % 4;
                    for channel in 0..kind.channels() {
                        let difference = (shown[row * 4 + channel] as i32 - pixels[texel * 4 + channel] as i32).abs();
                        worst = worst.max(difference);
                        wrong |= difference > allowed;
                    }
                }
                if wrong {
                    wrong_blocks += 1;
                    if kind == Kind::Bc7 {
                        wrong_modes[(0..8).find(|&bit| block[0] & (1 << bit) != 0).unwrap_or(8)] += 1;
                    }
                    first_wrong.get_or_insert(at);
                }
            }
            println!("{name}: {wrong_blocks} of {} blocks differ by more than {allowed}, the most by {worst}", blocks * blocks);
            if wrong_blocks > 0 {
                failed = true;
                if kind == Kind::Bc7 {
                    println!("  by mode: {wrong_modes:?}");
                }
                if let Some(at) = first_wrong {
                    let block = &bytes[at * kind.block_bytes()..][..kind.block_bytes()];
                    bc::decode_block(kind, block, &mut pixels);
                    let (x, y) = (at % blocks, at / blocks);
                    let row = y * 4 * SIDE as usize + x * 4;
                    println!("  block {block:02x?}");
                    println!("  here {:?}", &pixels[..16]);
                    println!("  GPU  {:?}", &shown[row * 4..row * 4 + 16]);
                }
            }
        }
        if failed {
            println!("FAIL the block decoders don't match this GPU");
            std::process::exit(1);
        }
        println!("PASS the block decoders match this GPU");
    }
}
