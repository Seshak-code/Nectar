//! vk-probe --read-back: what copies write to memory the CPU caches comes back right, however it gets
//! here. Each round copies a new pattern over part of one page or several, then reads all of the memory:
//! the last thing copied to each byte must be there. Some rounds wait on a timeline semaphore, so the
//! bytes come ahead of the news, and others on a fence, so they're asked for; some overlap the last copy,
//! and some send two copies before waiting.

use std::ffi::CStr;
use std::{mem, process, ptr};

use nectar_vk_sys::*;

use crate::{check, fail};

const HOST_VISIBLE: u32 = 0x2;
const HOST_CACHED: u32 = 0x8;
const SIZE: usize = 1 << 20;
const ROUNDS: u32 = 400;

/// The same small generator on both sides of a comparison
fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// # Safety
/// The handles must be valid, `queue` of `family`, and the device made with timeline semaphores on.
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
        let make_buffer = |usage: u32, wanted: u32| -> (VkBuffer, *mut u8) {
            let info = VkBufferCreateInfo {
                sType: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                size: SIZE as u64,
                usage,
                sharingMode: VK_SHARING_MODE_EXCLUSIVE,
                queueFamilyIndexCount: 0,
                pQueueFamilyIndices: ptr::null(),
            };
            let mut buffer = 0;
            check("vkCreateBuffer", f!(vkCreateBuffer, PFN_vkCreateBuffer)(device, &info, ptr::null(), &mut buffer));
            let mut needs: VkMemoryRequirements = mem::zeroed();
            f!(vkGetBufferMemoryRequirements, PFN_vkGetBufferMemoryRequirements)(device, buffer, &mut needs);
            let kind = (0..count)
                .find(|&i| needs.memoryTypeBits & (1 << i) != 0 && types[i].propertyFlags & wanted == wanted)
                .unwrap_or_else(|| fail("no memory type", VK_ERROR_OUT_OF_DEVICE_MEMORY));
            let allocate = VkMemoryAllocateInfo { sType: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, pNext: ptr::null(), allocationSize: needs.size, memoryTypeIndex: kind as u32 };
            let mut memory = 0;
            check("vkAllocateMemory", f!(vkAllocateMemory, PFN_vkAllocateMemory)(device, &allocate, ptr::null(), &mut memory));
            check("vkBindBufferMemory", f!(vkBindBufferMemory, PFN_vkBindBufferMemory)(device, buffer, memory, 0));
            let mut pointer = ptr::null_mut();
            check("vkMapMemory", f!(vkMapMemory, PFN_vkMapMemory)(device, memory, 0, SIZE as u64, 0, &mut pointer));
            (buffer, pointer as *mut u8)
        };
        let (source, source_bytes) = make_buffer(VK_BUFFER_USAGE_TRANSFER_SRC_BIT, HOST_VISIBLE);
        let (target, target_bytes) = make_buffer(VK_BUFFER_USAGE_TRANSFER_DST_BIT, HOST_VISIBLE | HOST_CACHED);
        let source_bytes = std::slice::from_raw_parts_mut(source_bytes, SIZE);
        let target_bytes = std::slice::from_raw_parts(target_bytes as *const u8, SIZE);

        let pool_info = VkCommandPoolCreateInfo {
            sType: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            pNext: ptr::null(),
            flags: VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex: family,
        };
        let mut pool = 0;
        check("vkCreateCommandPool", f!(vkCreateCommandPool, PFN_vkCreateCommandPool)(device, &pool_info, ptr::null(), &mut pool));
        let allocate = VkCommandBufferAllocateInfo {
            sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            pNext: ptr::null(),
            commandPool: pool,
            level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount: 2,
        };
        let mut commands = [ptr::null_mut(); 2];
        check("vkAllocateCommandBuffers", f!(vkAllocateCommandBuffers, PFN_vkAllocateCommandBuffers)(device, &allocate, commands.as_mut_ptr()));

        let kind = VkSemaphoreTypeCreateInfo { sType: VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, pNext: ptr::null(), semaphoreType: VK_SEMAPHORE_TYPE_TIMELINE, initialValue: 0 };
        let semaphore_info = VkSemaphoreCreateInfo { sType: VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, pNext: (&raw const kind).cast(), flags: 0 };
        let mut timeline = 0;
        check("vkCreateSemaphore", f!(vkCreateSemaphore, PFN_vkCreateSemaphore)(device, &semaphore_info, ptr::null(), &mut timeline));
        let fence_info = VkFenceCreateInfo { sType: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, pNext: ptr::null(), flags: 0 };
        let mut fence = 0;
        check("vkCreateFence", f!(vkCreateFence, PFN_vkCreateFence)(device, &fence_info, ptr::null(), &mut fence));

        let begin = f!(vkBeginCommandBuffer, PFN_vkBeginCommandBuffer);
        let end = f!(vkEndCommandBuffer, PFN_vkEndCommandBuffer);
        let copy = f!(vkCmdCopyBuffer, PFN_vkCmdCopyBuffer);
        let submit = f!(vkQueueSubmit, PFN_vkQueueSubmit);
        let wait_timeline = f!(vkWaitSemaphores, PFN_vkWaitSemaphores);
        let wait_fence = f!(vkWaitForFences, PFN_vkWaitForFences);
        let reset_fence = f!(vkResetFences, PFN_vkResetFences);

        let mut expected = vec![0u8; SIZE];
        // The memory starts as nothing in particular
        let _ = target_bytes[0];
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut value = 0u64;
        for round in 0..ROUNDS {
            // Two copies some rounds, the second maybe over the first
            let copies = if round % 5 == 4 { 2 } else { 1 };
            let mut placed = (0usize, 0usize);
            for which in 0..copies {
                let size = match next(&mut state) % 3 {
                    0 => 8 + next(&mut state) as usize % 64,
                    1 => 4096 + next(&mut state) as usize % 4096,
                    _ => 64 << 10,
                };
                let offset = if which == 1 && round % 10 == 4 {
                    // Over the copy before
                    (placed.0 + placed.1 / 2).min(SIZE - size)
                } else {
                    next(&mut state) as usize % (SIZE - size)
                };
                placed = (offset, size);
                let from = (which as usize) * (SIZE / 2) % (SIZE - size);
                for byte in &mut source_bytes[from..from + size] {
                    *byte = next(&mut state) as u8;
                }
                expected[offset..offset + size].copy_from_slice(&source_bytes[from..from + size]);
                let commands = commands[which as usize];
                let info = VkCommandBufferBeginInfo { sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, pNext: ptr::null(), flags: 0, pInheritanceInfo: ptr::null() };
                check("vkBeginCommandBuffer", begin(commands, &info));
                let region = VkBufferCopy { srcOffset: from as u64, dstOffset: offset as u64, size: size as u64 };
                copy(commands, source, target, 1, &region);
                check("vkEndCommandBuffer", end(commands));
                // Each waits for the last, as work that overlaps has to
                value += 1;
                let values = [value];
                let timeline_info = VkTimelineSemaphoreSubmitInfo {
                    sType: VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
                    pNext: ptr::null(),
                    waitSemaphoreValueCount: 0,
                    pWaitSemaphoreValues: ptr::null(),
                    signalSemaphoreValueCount: 1,
                    pSignalSemaphoreValues: values.as_ptr(),
                };
                let wait_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
                let before = [value - 1];
                let waiting = VkTimelineSemaphoreSubmitInfo { waitSemaphoreValueCount: 1, pWaitSemaphoreValues: before.as_ptr(), ..timeline_info };
                let by_fence = round % 3 == 2 && which == copies - 1;
                let info = VkSubmitInfo {
                    sType: VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    pNext: if value > 1 { (&raw const waiting).cast() } else { (&raw const timeline_info).cast() },
                    waitSemaphoreCount: if value > 1 { 1 } else { 0 },
                    pWaitSemaphores: &timeline,
                    pWaitDstStageMask: &wait_stage,
                    commandBufferCount: 1,
                    pCommandBuffers: &commands,
                    signalSemaphoreCount: 1,
                    pSignalSemaphores: &timeline,
                };
                check("vkQueueSubmit", submit(queue, 1, &info, if by_fence { fence } else { 0 }));
                if by_fence {
                    check("vkWaitForFences", wait_fence(device, 1, &fence, 1, u64::MAX));
                    check("vkResetFences", reset_fence(device, 1, &fence));
                }
            }
            let info = VkSemaphoreWaitInfo { sType: VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, pNext: ptr::null(), flags: 0, semaphoreCount: 1, pSemaphores: &timeline, pValues: &value };
            check("vkWaitSemaphores", wait_timeline(device, &info, u64::MAX));
            // The first read sees what the driver made of it all. Only what's been copied into is compared,
            // since the rest was never written.
            if let Some(at) = (0..SIZE).find(|&at| expected[at] != 0 && target_bytes[at] != expected[at]) {
                println!("FAIL read-back: round {round}, byte {at} is {} and should be {}", target_bytes[at], expected[at]);
                process::exit(1);
            }
        }
        check("vkDeviceWaitIdle", f!(vkDeviceWaitIdle, PFN_vkDeviceWaitIdle)(device));
        println!("read-back    {ROUNDS} rounds of copies to cached memory, byte for byte");
    }
}
