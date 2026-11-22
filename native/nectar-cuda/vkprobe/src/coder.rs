//! vk-probe --coder: frame_shader.rs against frame.rs, on the same pictures. The shader's frames must
//! decode with the same decoder to about the same picture and size, and it should take less time.

use std::ffi::CStr;
use std::time::Instant;
use std::{mem, ptr};

use nectar_vk_sys::*;
use nectar_vk_wire::frame::{self, Encoder};
use nectar_vk_wire::frame_shader::{self, PUSH_BYTES};

use crate::{check, fail};

const HOST_VISIBLE: u32 = 0x2;
const HOST_COHERENT: u32 = 0x4;
const HOST_CACHED: u32 = 0x8;
const DEVICE_LOCAL: u32 = 0x1;

/// A picture with smooth areas, edges, and noise as fine detail, as a game's frames have
pub(crate) fn picture(width: usize, height: usize, phase: usize, noisy: bool) -> Vec<u8> {
    let mut pixels = vec![255u8; width * height * 4];
    let mut noise = 0x2545_f491u32;
    for y in 0..height {
        for x in 0..width {
            let pixel = &mut pixels[(y * width + x) * 4..][..4];
            pixel[0] = ((x + phase) * 255 / width.max(1)) as u8;
            pixel[1] = (y * 255 / height.max(1)) as u8;
            pixel[2] = if (x + phase) / 40 % 2 == (y / 40) % 2 { 220 } else { 40 };
            if noisy {
                noise ^= noise << 13;
                noise ^= noise >> 17;
                noise ^= noise << 5;
                for channel in &mut pixel[..3] {
                    *channel = channel.saturating_add((noise & 15) as u8);
                }
            }
        }
    }
    pixels
}

pub(crate) fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let error: f64 = a
        .chunks(4)
        .zip(b.chunks(4))
        .map(|(a, b)| (0..3).map(|i| (a[i] as f64 - b[i] as f64).powi(2)).sum::<f64>())
        .sum::<f64>()
        / (a.len() as f64 / 4.0 * 3.0);
    10.0 * (255.0f64 * 255.0 / error.max(1e-9)).log10()
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn run(
    physical: VkPhysicalDevice,
    device: VkDevice,
    queue: VkQueue,
    family: u32,
    (width, height): (u32, u32),
    get_memory_properties: PFN_vkGetPhysicalDeviceMemoryProperties,
    nanoseconds_per_tick: f32,
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
        let types = properties.memoryTypes;
        let count = properties.memoryTypeCount as usize;
        let make_buffer = |size: u64, usage: u32, wanted: &[u32]| -> (VkBuffer, VkDeviceMemory) {
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
            let kind = wanted
                .iter()
                .find_map(|&flags| (0..count).find(|&i| needs.memoryTypeBits & (1 << i) != 0 && types[i].propertyFlags & flags == flags))
                .unwrap_or_else(|| fail("no memory type for a buffer", VK_ERROR_OUT_OF_DEVICE_MEMORY));
            let allocate = VkMemoryAllocateInfo {
                sType: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                pNext: ptr::null(),
                allocationSize: needs.size,
                memoryTypeIndex: kind as u32,
            };
            let mut memory = 0;
            check("vkAllocateMemory", f!(vkAllocateMemory, PFN_vkAllocateMemory)(device, &allocate, ptr::null(), &mut memory));
            check("vkBindBufferMemory", f!(vkBindBufferMemory, PFN_vkBindBufferMemory)(device, buffer, memory, 0));
            (buffer, memory)
        };
        let frame_size = width as u64 * height as u64 * 4;
        let output_size = frame_shader::output_size(width, height);
        let work_size = frame_shader::work_size(width, height);
        let hot_size = frame_shader::hot_size(width, height);
        const STORAGE: u32 = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        const TO: u32 = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        const FROM: u32 = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        let (current, current_memory) = make_buffer(frame_size, STORAGE | TO, &[DEVICE_LOCAL, 0]);
        let (previous, previous_memory) = make_buffer(frame_size, STORAGE | TO, &[DEVICE_LOCAL, 0]);
        let (work, work_memory) = make_buffer(work_size, STORAGE, &[DEVICE_LOCAL, 0]);
        let (hot, hot_memory) = make_buffer(hot_size, STORAGE | FROM, &[DEVICE_LOCAL, 0]);
        let (staging, staging_memory) = make_buffer(frame_size, FROM, &[HOST_VISIBLE | HOST_COHERENT]);
        let (output, output_memory) = make_buffer(output_size, STORAGE | TO, &[HOST_VISIBLE | HOST_COHERENT | HOST_CACHED, HOST_VISIBLE | HOST_COHERENT]);
        let map = f!(vkMapMemory, PFN_vkMapMemory);
        let mut staging_pointer = ptr::null_mut();
        check("vkMapMemory", map(device, staging_memory, 0, frame_size, 0, &mut staging_pointer));
        let mut output_pointer = ptr::null_mut();
        check("vkMapMemory", map(device, output_memory, 0, output_size, 0, &mut output_pointer));
        let staged = std::slice::from_raw_parts_mut(staging_pointer as *mut u8, frame_size as usize);
        let out = std::slice::from_raw_parts(output_pointer as *const u8, output_size as usize);

        // The pipeline
        let bindings: Vec<VkDescriptorSetLayoutBinding> = (0..5)
            .map(|binding| VkDescriptorSetLayoutBinding {
                binding,
                descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount: 1,
                stageFlags: VK_SHADER_STAGE_COMPUTE_BIT,
                pImmutableSamplers: ptr::null(),
            })
            .collect();
        let set_info = VkDescriptorSetLayoutCreateInfo {
            sType: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            bindingCount: 5,
            pBindings: bindings.as_ptr(),
        };
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
        let compile = Instant::now();
        let mut words = 0;
        let mut pipelines = Vec::new();
        for code in frame_shader::shaders() {
            words += code.len();
            let module_info = VkShaderModuleCreateInfo {
                sType: VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                codeSize: code.len() * 4,
                pCode: code.as_ptr(),
            };
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
            pipelines.push(pipeline);
        }
        println!("coder        {words} words of SPIR-V, compiled in {:.0} ms", compile.elapsed().as_secs_f64() * 1000.0);

        let pool_size = VkDescriptorPoolSize { r#type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount: 5 };
        let pool_info = VkDescriptorPoolCreateInfo {
            sType: VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            maxSets: 1,
            poolSizeCount: 1,
            pPoolSizes: &pool_size,
        };
        let mut descriptor_pool = 0;
        check("vkCreateDescriptorPool", f!(vkCreateDescriptorPool, PFN_vkCreateDescriptorPool)(device, &pool_info, ptr::null(), &mut descriptor_pool));
        let allocate_set = VkDescriptorSetAllocateInfo {
            sType: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            pNext: ptr::null(),
            descriptorPool: descriptor_pool,
            descriptorSetCount: 1,
            pSetLayouts: &set_layout,
        };
        let mut set = 0;
        check("vkAllocateDescriptorSets", f!(vkAllocateDescriptorSets, PFN_vkAllocateDescriptorSets)(device, &allocate_set, &mut set));
        let infos = [
            VkDescriptorBufferInfo { buffer: current, offset: 0, range: frame_size },
            VkDescriptorBufferInfo { buffer: previous, offset: 0, range: frame_size },
            VkDescriptorBufferInfo { buffer: output, offset: 0, range: output_size },
            VkDescriptorBufferInfo { buffer: work, offset: 0, range: work_size },
            VkDescriptorBufferInfo { buffer: hot, offset: 0, range: hot_size },
        ];
        let writes: Vec<VkWriteDescriptorSet> = (0..5)
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
        f!(vkUpdateDescriptorSets, PFN_vkUpdateDescriptorSets)(device, 5, writes.as_ptr(), 0, ptr::null());

        let command_pool_info = VkCommandPoolCreateInfo {
            sType: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            queueFamilyIndex: family,
        };
        let mut command_pool = 0;
        check("vkCreateCommandPool", f!(vkCreateCommandPool, PFN_vkCreateCommandPool)(device, &command_pool_info, ptr::null(), &mut command_pool));
        let command_info = VkCommandBufferAllocateInfo {
            sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            pNext: ptr::null(),
            commandPool: command_pool,
            level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount: 1,
        };
        let mut commands = ptr::null_mut();
        check("vkAllocateCommandBuffers", f!(vkAllocateCommandBuffers, PFN_vkAllocateCommandBuffers)(device, &command_info, &mut commands));
        let query_info = VkQueryPoolCreateInfo {
            sType: VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            queryType: VK_QUERY_TYPE_TIMESTAMP,
            queryCount: 5,
            pipelineStatistics: 0,
        };
        let mut queries = 0;
        check("vkCreateQueryPool", f!(vkCreateQueryPool, PFN_vkCreateQueryPool)(device, &query_info, ptr::null(), &mut queries));
        let fence_info = VkFenceCreateInfo { sType: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, pNext: ptr::null(), flags: 0 };
        let mut fence = 0;
        check("vkCreateFence", f!(vkCreateFence, PFN_vkCreateFence)(device, &fence_info, ptr::null(), &mut fence));

        let begin = VkCommandBufferBeginInfo {
            sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext: ptr::null(),
            flags: 0,
            pInheritanceInfo: ptr::null(),
        };
        let submit_and_wait = || {
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
            check("vkQueueSubmit", f!(vkQueueSubmit, PFN_vkQueueSubmit)(queue, 1, &info, fence));
            check("vkWaitForFences", f!(vkWaitForFences, PFN_vkWaitForFences)(device, 1, &fence, 1, u64::MAX));
            check("vkResetFences", f!(vkResetFences, PFN_vkResetFences)(device, 1, &fence));
        };
        let barrier_all = |source: u32, target: u32, source_stage: u32, target_stage: u32| {
            let barrier = VkMemoryBarrier { sType: VK_STRUCTURE_TYPE_MEMORY_BARRIER, pNext: ptr::null(), srcAccessMask: source, dstAccessMask: target };
            f!(vkCmdPipelineBarrier, PFN_vkCmdPipelineBarrier)(commands, source_stage, target_stage, 0, 1, &barrier, 0, ptr::null(), 0, ptr::null());
        };
        let upload = |pixels: &[u8], staged: &mut [u8], target: VkBuffer| {
            staged.copy_from_slice(pixels);
            check("vkResetCommandPool", f!(vkResetCommandPool, PFN_vkResetCommandPool)(device, command_pool, 0));
            check("vkBeginCommandBuffer", f!(vkBeginCommandBuffer, PFN_vkBeginCommandBuffer)(commands, &begin));
            let region = VkBufferCopy { srcOffset: 0, dstOffset: 0, size: frame_size };
            f!(vkCmdCopyBuffer, PFN_vkCmdCopyBuffer)(commands, staging, target, 1, &region);
            barrier_all(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            check("vkEndCommandBuffer", f!(vkEndCommandBuffer, PFN_vkEndCommandBuffer)(commands));
            submit_and_wait();
        };
        let code_frame = |quality: u32, first: bool| {
            check("vkResetCommandPool", f!(vkResetCommandPool, PFN_vkResetCommandPool)(device, command_pool, 0));
            check("vkBeginCommandBuffer", f!(vkBeginCommandBuffer, PFN_vkBeginCommandBuffer)(commands, &begin));
            f!(vkCmdResetQueryPool, PFN_vkCmdResetQueryPool)(commands, queries, 0, 5);
            f!(vkCmdWriteTimestamp, PFN_vkCmdWriteTimestamp)(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries, 0);
            f!(vkCmdBindDescriptorSets, PFN_vkCmdBindDescriptorSets)(commands, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, ptr::null());
            for (index, pass) in frame_shader::passes(width, height).into_iter().enumerate() {
                f!(vkCmdBindPipeline, PFN_vkCmdBindPipeline)(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[pass.shader]);
                let constants = frame_shader::push_constants(width, height, quality, first, pass.first_plane);
                f!(vkCmdPushConstants, PFN_vkCmdPushConstants)(commands, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, PUSH_BYTES, constants.as_ptr().cast());
                f!(vkCmdDispatch, PFN_vkCmdDispatch)(commands, pass.groups[0], pass.groups[1], pass.groups[2]);
                barrier_all(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
                if index != 1 {
                    let stamp = if index == 0 { 1 } else { index as u32 };
                    f!(vkCmdWriteTimestamp, PFN_vkCmdWriteTimestamp)(commands, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, stamp);
                }
            }
            barrier_all(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            let region = VkBufferCopy { srcOffset: 0, dstOffset: 0, size: hot_size };
            f!(vkCmdCopyBuffer, PFN_vkCmdCopyBuffer)(commands, hot, output, 1, &region);
            barrier_all(VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT);
            f!(vkCmdWriteTimestamp, PFN_vkCmdWriteTimestamp)(commands, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries, 4);
            check("vkEndCommandBuffer", f!(vkEndCommandBuffer, PFN_vkEndCommandBuffer)(commands));
            submit_and_wait();
            let mut ticks = [0u64; 5];
            check(
                "vkGetQueryPoolResults",
                f!(vkGetQueryPoolResults, PFN_vkGetQueryPoolResults)(device, queries, 0, 5, 40, ticks.as_mut_ptr().cast(), 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
            );
            let ms = |from: usize, to: usize| ticks[to].saturating_sub(ticks[from]) as f64 * nanoseconds_per_tick as f64 / 1e6;
            (ms(0, 4), [ms(0, 1), ms(1, 2), ms(2, 3), ms(3, 4)])
        };

        let mut failed = false;
        for (kind, noisy) in [("smooth", false), ("detailed", true)] {
            let a = picture(width as usize, height as usize, 0, noisy);
            upload(&a, staged, current);
            for quality in [90, 75, 50, 30] {
                let (mut gpu_ms, mut stages) = (f64::MAX, [0.0; 4]);
                for _ in 0..8 {
                    let (this, these) = code_frame(quality, true);
                    if this < gpu_ms {
                        (gpu_ms, stages) = (this, these);
                    }
                }
                let gather = Instant::now();
                let coded = frame_shader::gather(out, width, height, quality);
                let gather_ms = gather.elapsed().as_secs_f64() * 1000.0;
                let mut shown = vec![0u8; a.len()];
                if !frame::decode(&coded, &mut shown, width, height) {
                    println!("FAIL coder: the shader's frame doesn't decode");
                    failed = true;
                    continue;
                }
                let gpu_db = psnr(&a, &shown);

                let mut cpu_ms = f64::MAX;
                let mut cpu_coded = Vec::new();
                for _ in 0..4 {
                    let timer = Instant::now();
                    cpu_coded = Encoder::default().encode(&a, width, height, quality);
                    cpu_ms = cpu_ms.min(timer.elapsed().as_secs_f64() * 1000.0);
                }
                let mut cpu_shown = vec![0u8; a.len()];
                frame::decode(&cpu_coded, &mut cpu_shown, width, height);
                let cpu_db = psnr(&a, &cpu_shown);
                let share = coded.len() as f64 / cpu_coded.len().max(1) as f64;
                println!(
                    "coder        {kind:<8} quality {quality}: GPU {gpu_ms:.2} ms ({:.2} compare, {:.2} transform, {:.2} write, {:.2} copy) + gather {gather_ms:.2} ms, {} KB, {gpu_db:.2} dB; CPU {cpu_ms:.2} ms, {} KB, {cpu_db:.2} dB",
                    stages[0],
                    stages[1],
                    stages[2],
                    stages[3],
                    coded.len() / 1000,
                    cpu_coded.len() / 1000
                );
                if (gpu_db - cpu_db).abs() > 0.5 || !(0.9..1.1).contains(&share) {
                    println!("FAIL coder: the shader's frame differs too much from the CPU's");
                    failed = true;
                }
            }

            // Only what changed crosses
            upload(&a, staged, previous);
            let mut b = a.clone();
            let (changed_x, changed_y) = (70usize, 70usize.min(height as usize - 1));
            b[(changed_y * width as usize + changed_x) * 4] ^= 0x40;
            upload(&b, staged, current);
            code_frame(80, false);
            let coded = frame_shader::gather(out, width, height, 80);
            let tiles = u32::from_le_bytes(coded[12..16].try_into().unwrap());
            if tiles != 1 || coded[16..20] != [1, 0, 1, 0] {
                println!("FAIL coder: one pixel changed, and {tiles} tiles came, the first being {:?}", &coded[16..20]);
                failed = true;
            }
            upload(&a, staged, current);
            code_frame(80, false);
            let coded = frame_shader::gather(out, width, height, 80);
            if coded.len() != 16 {
                println!("FAIL coder: nothing changed, and {} bytes came", coded.len());
                failed = true;
            }
            let blank = vec![0u8; a.len()];
            upload(&blank, staged, previous);
        }

        f!(vkDeviceWaitIdle, PFN_vkDeviceWaitIdle)(device);
        f!(vkDestroyQueryPool, PFN_vkDestroyQueryPool)(device, queries, ptr::null());
        f!(vkDestroyFence, PFN_vkDestroyFence)(device, fence, ptr::null());
        f!(vkDestroyCommandPool, PFN_vkDestroyCommandPool)(device, command_pool, ptr::null());
        f!(vkDestroyDescriptorPool, PFN_vkDestroyDescriptorPool)(device, descriptor_pool, ptr::null());
        for pipeline in pipelines {
            f!(vkDestroyPipeline, PFN_vkDestroyPipeline)(device, pipeline, ptr::null());
        }
        f!(vkDestroyPipelineLayout, PFN_vkDestroyPipelineLayout)(device, layout, ptr::null());
        f!(vkDestroyDescriptorSetLayout, PFN_vkDestroyDescriptorSetLayout)(device, set_layout, ptr::null());
        for (buffer, memory) in [(current, current_memory), (previous, previous_memory), (staging, staging_memory), (output, output_memory), (work, work_memory), (hot, hot_memory)] {
            f!(vkDestroyBuffer, PFN_vkDestroyBuffer)(device, buffer, ptr::null());
            f!(vkFreeMemory, PFN_vkFreeMemory)(device, memory, ptr::null());
        }
        if failed {
            std::process::exit(1);
        }
        println!("coder        the shader's frames decode, match the CPU's in size and picture, and leave out what hasn't changed");
    }
}
