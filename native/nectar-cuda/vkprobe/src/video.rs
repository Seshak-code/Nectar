//! vk-probe --video: frames through the card's H.264 encoder (video_shader.rs and video_encode.rs) and
//! back through the decoder Windows has (video_decode.rs). Every frame in must give its own picture out,
//! at once, and close to what went in; a key frame must be enough for a decoder that has seen nothing.

use std::ffi::CStr;
use std::time::Instant;
use std::{mem, process, ptr};

use nectar_vk_sys::*;
use nectar_vk_wire::video_decode::Decoder;
use nectar_vk_wire::video_encode::{DEVICE_EXTENSIONS, Encoder, encode_family};
use nectar_vk_wire::video_shader::{self, PUSH_BYTES};

use crate::coder::{picture, psnr};
use crate::{check, fail};

const HOST_VISIBLE: u32 = 0x2;
const HOST_COHERENT: u32 = 0x4;
const DEVICE_LOCAL: u32 = 0x1;
const FRAMES: usize = 60;

fn stop(what: &str) -> ! {
    println!("FAIL video: {what}");
    process::exit(1);
}

pub unsafe fn run(gipa: PFN_vkGetInstanceProcAddr, instance: VkInstance, physical: VkPhysicalDevice, width: u32, height: u32) {
    unsafe {
        let find_instance = |name: &CStr| gipa(instance, name.as_ptr());
        macro_rules! i {
            ($name:ident, $pfn:ty) => {
                mem::transmute::<PFN_vkVoidFunction, Option<$pfn>>(find_instance(&std::ffi::CString::new(stringify!($name)).unwrap())).unwrap_or_else(|| panic!("no {}", stringify!($name)))
            };
        }
        let Some(encode) = encode_family(&find_instance, physical) else { stop("the card has no H.264 encoder") };
        let mut count = 0;
        i!(vkGetPhysicalDeviceQueueFamilyProperties, PFN_vkGetPhysicalDeviceQueueFamilyProperties)(physical, &mut count, ptr::null_mut());
        let mut families = vec![mem::zeroed::<VkQueueFamilyProperties>(); count as usize];
        i!(vkGetPhysicalDeviceQueueFamilyProperties, PFN_vkGetPhysicalDeviceQueueFamilyProperties)(physical, &mut count, families.as_mut_ptr());
        let work = families.iter().position(|family| family.queueFlags & VK_QUEUE_COMPUTE_BIT != 0).unwrap_or_else(|| stop("no compute queue")) as u32;
        println!("video        encoding on queue family {encode}, converting on {work}");

        let priority = 1.0f32;
        let queue_info = |family: u32| VkDeviceQueueCreateInfo {
            sType: VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            queueFamilyIndex: family,
            queueCount: 1,
            pQueuePriorities: &priority,
        };
        let queues = [queue_info(work), queue_info(encode)];
        let extensions: Vec<_> = DEVICE_EXTENSIONS.iter().map(|name| name.as_ptr()).collect();
        let device_info = VkDeviceCreateInfo {
            sType: VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            queueCreateInfoCount: if work == encode { 1 } else { 2 },
            pQueueCreateInfos: queues.as_ptr(),
            enabledLayerCount: 0,
            ppEnabledLayerNames: ptr::null(),
            enabledExtensionCount: extensions.len() as u32,
            ppEnabledExtensionNames: extensions.as_ptr(),
            pEnabledFeatures: ptr::null(),
        };
        let mut device = ptr::null_mut();
        check("vkCreateDevice", i!(vkCreateDevice, PFN_vkCreateDevice)(physical, &device_info, ptr::null(), &mut device));
        let gdpa = i!(vkGetDeviceProcAddr, PFN_vkGetDeviceProcAddr);
        let find_device = |name: &CStr| gdpa(device, name.as_ptr());
        macro_rules! f {
            ($name:ident, $pfn:ty) => {
                mem::transmute::<PFN_vkVoidFunction, Option<$pfn>>(find_device(&std::ffi::CString::new(stringify!($name)).unwrap())).unwrap_or_else(|| panic!("no {}", stringify!($name)))
            };
        }
        let (mut work_queue, mut encode_queue) = (ptr::null_mut(), ptr::null_mut());
        f!(vkGetDeviceQueue, PFN_vkGetDeviceQueue)(device, work, 0, &mut work_queue);
        f!(vkGetDeviceQueue, PFN_vkGetDeviceQueue)(device, encode, 0, &mut encode_queue);

        let mut properties: VkPhysicalDeviceMemoryProperties = mem::zeroed();
        i!(vkGetPhysicalDeviceMemoryProperties, PFN_vkGetPhysicalDeviceMemoryProperties)(physical, &mut properties);
        let shared = [work, encode];
        let make_buffer = |size: u64, usage: u32, wanted: &[u32], both: bool| -> (VkBuffer, VkDeviceMemory) {
            let both = both && work != encode;
            let info = VkBufferCreateInfo {
                sType: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                size,
                usage,
                sharingMode: if both { VK_SHARING_MODE_CONCURRENT } else { VK_SHARING_MODE_EXCLUSIVE },
                queueFamilyIndexCount: if both { 2 } else { 0 },
                pQueueFamilyIndices: shared.as_ptr(),
            };
            let mut buffer = 0;
            check("vkCreateBuffer", f!(vkCreateBuffer, PFN_vkCreateBuffer)(device, &info, ptr::null(), &mut buffer));
            let mut needs: VkMemoryRequirements = mem::zeroed();
            f!(vkGetBufferMemoryRequirements, PFN_vkGetBufferMemoryRequirements)(device, buffer, &mut needs);
            let kind = wanted
                .iter()
                .find_map(|&flags| (0..properties.memoryTypeCount as usize).find(|&index| needs.memoryTypeBits & (1 << index) != 0 && properties.memoryTypes[index].propertyFlags & flags == flags))
                .unwrap_or_else(|| fail("no memory type for a buffer", VK_ERROR_OUT_OF_DEVICE_MEMORY));
            let allocate = VkMemoryAllocateInfo { sType: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, pNext: ptr::null(), allocationSize: needs.size, memoryTypeIndex: kind as u32 };
            let mut memory = 0;
            check("vkAllocateMemory", f!(vkAllocateMemory, PFN_vkAllocateMemory)(device, &allocate, ptr::null(), &mut memory));
            check("vkBindBufferMemory", f!(vkBindBufferMemory, PFN_vkBindBufferMemory)(device, buffer, memory, 0));
            (buffer, memory)
        };
        let frame_size = width as u64 * height as u64 * 4;
        let (coded_width, coded_height) = video_shader::coded_size(width, height);
        let nv12_size = video_shader::nv12_size(coded_width, coded_height);
        let (staging, staging_memory) = make_buffer(frame_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &[HOST_VISIBLE | HOST_COHERENT], false);
        let (frame, frame_memory) = make_buffer(frame_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, &[DEVICE_LOCAL, 0], false);
        let (nv12, nv12_memory) = make_buffer(nv12_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &[DEVICE_LOCAL, 0], true);
        let mut staging_pointer = ptr::null_mut();
        check("vkMapMemory", f!(vkMapMemory, PFN_vkMapMemory)(device, staging_memory, 0, frame_size, 0, &mut staging_pointer));
        let staged = std::slice::from_raw_parts_mut(staging_pointer as *mut u8, frame_size as usize);

        // The pipeline that makes NV12 of a frame
        let bindings: Vec<VkDescriptorSetLayoutBinding> = (0..2)
            .map(|binding| VkDescriptorSetLayoutBinding { binding, descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount: 1, stageFlags: VK_SHADER_STAGE_COMPUTE_BIT, pImmutableSamplers: ptr::null() })
            .collect();
        let set_info = VkDescriptorSetLayoutCreateInfo { sType: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, pNext: ptr::null(), flags: 0, bindingCount: 2, pBindings: bindings.as_ptr() };
        let mut set_layout = 0;
        check("vkCreateDescriptorSetLayout", f!(vkCreateDescriptorSetLayout, PFN_vkCreateDescriptorSetLayout)(device, &set_info, ptr::null(), &mut set_layout));
        let range = VkPushConstantRange { stageFlags: VK_SHADER_STAGE_COMPUTE_BIT, offset: 0, size: PUSH_BYTES };
        let layout_info = VkPipelineLayoutCreateInfo {
            sType: VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            setLayoutCount: 1,
            pSetLayouts: &set_layout,
            pushConstantRangeCount: 1,
            pPushConstantRanges: &range,
        };
        let mut layout = 0;
        check("vkCreatePipelineLayout", f!(vkCreatePipelineLayout, PFN_vkCreatePipelineLayout)(device, &layout_info, ptr::null(), &mut layout));
        let code = video_shader::shader();
        let module_info = VkShaderModuleCreateInfo { sType: VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, pNext: ptr::null(), flags: 0, codeSize: code.len() * 4, pCode: code.as_ptr() };
        let mut module = 0;
        check("vkCreateShaderModule", f!(vkCreateShaderModule, PFN_vkCreateShaderModule)(device, &module_info, ptr::null(), &mut module));
        let pipeline_info = VkComputePipelineCreateInfo {
            sType: VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            stage: VkPipelineShaderStageCreateInfo {
                sType: VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                stage: VK_SHADER_STAGE_COMPUTE_BIT,
                module,
                pName: c"main".as_ptr(),
                pSpecializationInfo: ptr::null(),
            },
            layout,
            basePipelineHandle: 0,
            basePipelineIndex: -1,
        };
        let mut pipeline = 0;
        check("vkCreateComputePipelines", f!(vkCreateComputePipelines, PFN_vkCreateComputePipelines)(device, 0, 1, &pipeline_info, ptr::null(), &mut pipeline));
        f!(vkDestroyShaderModule, PFN_vkDestroyShaderModule)(device, module, ptr::null());
        let pool_size = VkDescriptorPoolSize { r#type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount: 2 };
        let pool_info = VkDescriptorPoolCreateInfo { sType: VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, pNext: ptr::null(), flags: 0, maxSets: 1, poolSizeCount: 1, pPoolSizes: &pool_size };
        let mut descriptor_pool = 0;
        check("vkCreateDescriptorPool", f!(vkCreateDescriptorPool, PFN_vkCreateDescriptorPool)(device, &pool_info, ptr::null(), &mut descriptor_pool));
        let allocate_set = VkDescriptorSetAllocateInfo { sType: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, pNext: ptr::null(), descriptorPool: descriptor_pool, descriptorSetCount: 1, pSetLayouts: &set_layout };
        let mut set = 0;
        check("vkAllocateDescriptorSets", f!(vkAllocateDescriptorSets, PFN_vkAllocateDescriptorSets)(device, &allocate_set, &mut set));
        let infos = [VkDescriptorBufferInfo { buffer: frame, offset: 0, range: frame_size }, VkDescriptorBufferInfo { buffer: nv12, offset: 0, range: nv12_size }];
        let writes: Vec<VkWriteDescriptorSet> = (0..2)
            .map(|binding| VkWriteDescriptorSet {
                sType: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                pNext: ptr::null(),
                dstSet: set,
                dstBinding: binding as u32,
                dstArrayElement: 0,
                descriptorCount: 1,
                descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pImageInfo: ptr::null(),
                pBufferInfo: &infos[binding],
                pTexelBufferView: ptr::null(),
            })
            .collect();
        f!(vkUpdateDescriptorSets, PFN_vkUpdateDescriptorSets)(device, 2, writes.as_ptr(), 0, ptr::null());

        let command_pool_info = VkCommandPoolCreateInfo { sType: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, pNext: ptr::null(), flags: 0, queueFamilyIndex: work };
        let mut command_pool = 0;
        check("vkCreateCommandPool", f!(vkCreateCommandPool, PFN_vkCreateCommandPool)(device, &command_pool_info, ptr::null(), &mut command_pool));
        let command_info = VkCommandBufferAllocateInfo { sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, pNext: ptr::null(), commandPool: command_pool, level: VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount: 1 };
        let mut commands = ptr::null_mut();
        check("vkAllocateCommandBuffers", f!(vkAllocateCommandBuffers, PFN_vkAllocateCommandBuffers)(device, &command_info, &mut commands));
        let fence_info = VkFenceCreateInfo { sType: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, pNext: ptr::null(), flags: 0 };
        let mut fence = 0;
        check("vkCreateFence", f!(vkCreateFence, PFN_vkCreateFence)(device, &fence_info, ptr::null(), &mut fence));
        let begin = VkCommandBufferBeginInfo { sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, pNext: ptr::null(), flags: 0, pInheritanceInfo: ptr::null() };
        let barrier = |source: u32, target: u32, source_stage: u32, target_stage: u32| {
            let barrier = VkMemoryBarrier { sType: VK_STRUCTURE_TYPE_MEMORY_BARRIER, pNext: ptr::null(), srcAccessMask: source, dstAccessMask: target };
            f!(vkCmdPipelineBarrier, PFN_vkCmdPipelineBarrier)(commands, source_stage, target_stage, 0, 1, &barrier, 0, ptr::null(), 0, ptr::null());
        };
        // Copies the staged frame in and makes its NV12. Returns how long the GPU took over it.
        let convert = |frame_width: u32, frame_height: u32| -> f64 {
            let timer = Instant::now();
            check("vkResetCommandPool", f!(vkResetCommandPool, PFN_vkResetCommandPool)(device, command_pool, 0));
            check("vkBeginCommandBuffer", f!(vkBeginCommandBuffer, PFN_vkBeginCommandBuffer)(commands, &begin));
            let region = VkBufferCopy { srcOffset: 0, dstOffset: 0, size: frame_width as u64 * frame_height as u64 * 4 };
            f!(vkCmdCopyBuffer, PFN_vkCmdCopyBuffer)(commands, staging, frame, 1, &region);
            barrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            f!(vkCmdBindPipeline, PFN_vkCmdBindPipeline)(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            f!(vkCmdBindDescriptorSets, PFN_vkCmdBindDescriptorSets)(commands, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, ptr::null());
            let constants = video_shader::push_constants(frame_width, frame_height, true);
            f!(vkCmdPushConstants, PFN_vkCmdPushConstants)(commands, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, PUSH_BYTES, constants.as_ptr().cast());
            let (coded_width, coded_height) = video_shader::coded_size(frame_width, frame_height);
            let (across, down) = video_shader::groups(coded_width, coded_height);
            f!(vkCmdDispatch, PFN_vkCmdDispatch)(commands, across, down, 1);
            barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            check("vkEndCommandBuffer", f!(vkEndCommandBuffer, PFN_vkEndCommandBuffer)(commands));
            let info = VkSubmitInfo {
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
            check("vkQueueSubmit", f!(vkQueueSubmit, PFN_vkQueueSubmit)(work_queue, 1, &info, fence));
            check("vkWaitForFences", f!(vkWaitForFences, PFN_vkWaitForFences)(device, 1, &fence, 1, u64::MAX));
            check("vkResetFences", f!(vkResetFences, PFN_vkResetFences)(device, 1, &fence));
            timer.elapsed().as_secs_f64() * 1000.0
        };

        // NECTAR_VK_PROBE_DUMP names a folder for the small size's first two coded frames, which
        // video_decode.rs's test decodes
        let dump = std::env::var_os("NECTAR_VK_PROBE_DUMP").map(std::path::PathBuf::from);
        let mut failed = false;
        // The second size isn't whole 16-pixel blocks, nor even: the decoder has to cut the picture down
        let small = (width.min(323), height.min(243));
        for (frame_width, frame_height) in [(width, height), small] {
            let timer = Instant::now();
            let mut encoder = Encoder::new(&find_instance, &find_device, physical, device, encode, encode_queue, frame_width, frame_height).unwrap_or_else(|error| stop(&error));
            let made = timer.elapsed().as_secs_f64() * 1000.0;
            let mut decoder = Decoder::new(frame_width, frame_height).unwrap_or_else(|error| stop(&error));
            println!("video        {frame_width}x{frame_height}: encoder ready in {made:.0} ms, {} bytes of headers", encoder.headers().len());
            let length = frame_width as usize * frame_height as usize * 4;
            let mut shown = vec![0u8; length];
            let mut frames = 0usize;
            for (kind, noisy) in [("smooth", false), ("detailed", true)] {
                for quality in [90, 70, 50, 30] {
                    let (mut convert_ms, mut encode_ms, mut decode_ms, mut bytes, mut worst, mut key_bytes) = (0.0, 0.0, 0.0, 0usize, f64::MAX, 0usize);
                    let (decoding, colours) = (decoder.decoding, decoder.colours);
                    for index in 0..FRAMES {
                        // Each frame moves on from the last, as a game's do, so each is told from its neighbours
                        let original = picture(frame_width as usize, frame_height as usize, frames * 6, noisy);
                        frames += 1;
                        staged[..length].copy_from_slice(&original);
                        convert_ms += convert(frame_width, frame_height);
                        let timer = Instant::now();
                        let (coded, key) = match encoder.encode(nv12, quality, false) {
                            Ok((coded, key)) => (coded.to_vec(), key),
                            Err(error) => stop(&error),
                        };
                        encode_ms += timer.elapsed().as_secs_f64() * 1000.0;
                        if let Some(folder) = dump.as_ref().filter(|_| (frame_width, frame_height) == small && frames <= 2) {
                            let name = if key { "key.h264" } else { "next.h264" };
                            std::fs::write(folder.join(name), &coded).unwrap_or_else(|error| stop(&format!("{name} can't be written: {error}")));
                        }
                        if key {
                            key_bytes = coded.len();
                        } else {
                            bytes += coded.len();
                        }
                        let timer = Instant::now();
                        match decoder.decode(&coded, &mut shown) {
                            Ok(true) => {}
                            Ok(false) => {
                                println!("FAIL video: frame {index} at quality {quality} gave no picture when it came");
                                failed = true;
                                continue;
                            }
                            Err(error) => stop(&error),
                        }
                        decode_ms += timer.elapsed().as_secs_f64() * 1000.0;
                        worst = worst.min(psnr(&original, &shown));
                    }
                    let count = FRAMES as f64;
                    let later = (FRAMES - (key_bytes > 0) as usize).max(1);
                    println!(
                        "video        {kind:<8} quality {quality}: {:.1} KB a frame{}, worst {worst:.1} dB; convert {:.2} ms, encode {:.2} ms, decode {:.2} ms ({:.2} the decoder, {:.2} colours)",
                        bytes as f64 / later as f64 / 1000.0,
                        if key_bytes > 0 { format!(" after a key frame of {:.1} KB", key_bytes as f64 / 1000.0) } else { String::new() },
                        convert_ms / count,
                        encode_ms / count,
                        decode_ms / count,
                        (decoder.decoding - decoding).as_secs_f64() * 1000.0 / count,
                        (decoder.colours - colours).as_secs_f64() * 1000.0 / count
                    );
                    // A wrong or late picture is 15 dB or less on these, which move 6 pixels a frame
                    let least = if quality >= 50 { 28.0 } else { 24.0 };
                    if worst < least {
                        println!("FAIL video: a picture came back at {worst:.1} dB, under {least}");
                        failed = true;
                    }
                }
            }

            // A decoder that has seen nothing takes it up from a key frame
            let original = picture(frame_width as usize, frame_height as usize, frames * 6, false);
            staged[..length].copy_from_slice(&original);
            convert(frame_width, frame_height);
            let (coded, key) = match encoder.encode(nv12, 70, true) {
                Ok((coded, key)) => (coded.to_vec(), key),
                Err(error) => stop(&error),
            };
            let mut fresh = Decoder::new(frame_width, frame_height).unwrap_or_else(|error| stop(&error));
            let mut fresh_shown = vec![0u8; length];
            match fresh.decode(&coded, &mut fresh_shown) {
                Ok(true) if key && psnr(&original, &fresh_shown) >= 28.0 => println!("video        a key frame asked for mid-stream, {:.1} KB, is enough for a new decoder", coded.len() as f64 / 1000.0),
                Ok(shown) => {
                    println!("FAIL video: a key frame asked for mid-stream: key {key}, picture {shown}, {:.1} dB", psnr(&original, &fresh_shown));
                    failed = true;
                }
                Err(error) => stop(&error),
            }
            // And the one that was there carries on
            if !matches!(decoder.decode(&coded, &mut shown), Ok(true)) || psnr(&original, &shown) < 28.0 {
                println!("FAIL video: the decoder that was there didn't take the key frame");
                failed = true;
            }
            encoder.destroy();
        }

        f!(vkDeviceWaitIdle, PFN_vkDeviceWaitIdle)(device);
        f!(vkDestroyFence, PFN_vkDestroyFence)(device, fence, ptr::null());
        f!(vkDestroyCommandPool, PFN_vkDestroyCommandPool)(device, command_pool, ptr::null());
        f!(vkDestroyDescriptorPool, PFN_vkDestroyDescriptorPool)(device, descriptor_pool, ptr::null());
        f!(vkDestroyPipeline, PFN_vkDestroyPipeline)(device, pipeline, ptr::null());
        f!(vkDestroyPipelineLayout, PFN_vkDestroyPipelineLayout)(device, layout, ptr::null());
        f!(vkDestroyDescriptorSetLayout, PFN_vkDestroyDescriptorSetLayout)(device, set_layout, ptr::null());
        for (buffer, memory) in [(staging, staging_memory), (frame, frame_memory), (nv12, nv12_memory)] {
            f!(vkDestroyBuffer, PFN_vkDestroyBuffer)(device, buffer, ptr::null());
            f!(vkFreeMemory, PFN_vkFreeMemory)(device, memory, ptr::null());
        }
        f!(vkDestroyDevice, PFN_vkDestroyDevice)(device, ptr::null());
        if failed {
            process::exit(1);
        }
        println!("video        every frame gave its own picture at once, and a key frame starts a new decoder");
    }
}
