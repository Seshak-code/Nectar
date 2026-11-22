//! Nectar's Vulkan driver for a borrowed GPU: a loader driver (ICD) whose every call is carried
//! out on the lender's GPU (nectar-server, Vulkan personality).
//!
//! The Vulkan loader finds it through its manifest (nectar_icd.json): registered by the
//! installer, or named by VK_DRIVER_FILES / VK_ADD_DRIVER_FILES for testing. While nothing
//! is borrowed it lists no device, so programs carry on with the computer's own GPUs.
//!
//! The lender is NECTAR_VK_SERVER (with NECTAR_VK_TOKEN) for testing, or the GPU nectar-svc
//! has borrowed when `borrow.cfg` lists vulkan.
//!
//! Most commands are generated (generated.rs) and carried by nectar-vk-wire's interpreter.
//! This file holds the loader interface and the commands that need more: the instance and
//! device, the extension list (only what can be carried), mapped memory, and the points where
//! memory is kept in step (before a submit, after a wait). present.rs puts frames on screen.

#![allow(non_snake_case, clippy::missing_safety_doc)]

mod connection;
mod copies;
mod later;
mod uploads;
mod descriptors;
mod generated;
mod memory;
mod objects;
#[cfg(any(windows, target_os = "linux"))]
mod present;
mod requirements;
mod templates;
mod timelines;
#[cfg(any(windows, target_os = "linux"))]
mod window;

use std::collections::HashMap;
use std::ffi::{CStr, c_char, c_void};
use std::sync::Mutex;
use std::time::Duration;
use std::sync::atomic::Ordering::Relaxed;
use std::{env, ptr};

use nectar_cuda_protocol::Writer;
use nectar_vk_sys::*;
use nectar_vk_wire::tables::{Mode, Ret};
use nectar_vk_wire::{COMMANDS, EXTENSIONS, command_name, wire};

use objects::{Borrower, id_of};

pub(crate) const DEVICE_LOST: i32 = VK_ERROR_DEVICE_LOST;
/// The loader interface version this driver speaks
const LOADER_INTERFACE: u32 = 5;
/// What the device's name ends with, so programs and nectar-detect can tell it's borrowed
const SUFFIX: &str = " (Nectar)";

pub(crate) fn log(message: &str) {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let at = START.get_or_init(std::time::Instant::now).elapsed();
    eprintln!("nectar-vk {:>10.3}: {message}", at.as_secs_f64() * 1000.0);
}

/// NECTAR_VK_LOG=1 logs every call
pub(crate) fn logging() -> bool {
    static LOGGING: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *LOGGING.get_or_init(|| env::var_os("NECTAR_VK_LOG").is_some())
}

fn index_of(name: &str) -> Option<usize> {
    COMMANDS.iter().position(|command| command.name == name)
}

thread_local! {
    /// This thread is doing the driver's own work, which reads nothing the program wrote
    static OWN_WORK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs the driver's own Vulkan calls
pub(crate) fn own_work<R>(work: impl FnOnce() -> R) -> R {
    OWN_WORK.set(true);
    let result = work();
    OWN_WORK.set(false);
    result
}

/// Carries out COMMANDS[index] on the lender; each argument is given by its address
pub(crate) unsafe fn call(index: usize, args: &[*const c_void]) -> u64 {
    let command = &COMMANDS[index];
    let failed = |result: i32| if command.ret == Ret::Result { result as i64 as u64 } else { 0 };
    let Some(connection) = connection::get() else {
        return failed(DEVICE_LOST);
    };
    let own = OWN_WORK.get();
    if command.name == "vkUpdateDescriptorSets" && unsafe { descriptors::copy_later(&connection, args) } {
        return 0;
    }
    descriptors::before_command(&connection, command.name);
    let mut pushing = memory::Pushing::default();
    if matches!(command.name, "vkQueueSubmit" | "vkQueueSubmit2") {
        unsafe {
            let (commands, signals) = (uploads::commands_of(command.name, args), timelines::signals_of(command.name, args));
            pushing = memory::work_submitted(*(args[0] as *const u64), &commands, *(args[3] as *const u64), &signals, timelines::waits_of(command.name, args));
        }
    }
    // Held until the submit is on its way
    let mut _order = None;
    if !own && matches!(command.name, "vkQueueSubmit" | "vkQueueSubmit2") {
        let mut submit = if connection.keep_uploads { unsafe { uploads::of_submit(command.name, args) } } else { uploads::Submit::default() };
        if connection.keep_uploads {
            if later::enabled() && submit.due.is_some() {
                memory::wait_for_copies(&submit, uploads::elsewhere);
                submit.elsewhere = uploads::elsewhere();
            }
            later::before_submit(&submit.commands);
            _order = Some(later::order());
            later::submitted(&submit.commands);
        }
        memory::flush(&connection, &submit);
    }
    // These could let work already on its way see anything
    if !own && matches!(command.name, "vkQueueBindSparse" | "vkSetEvent" | "vkSignalSemaphore") {
        memory::flush(&connection, &uploads::Submit::default());
    }
    if command.name == "vkFreeMemory" {
        memory::forget(&connection, unsafe { *(args[1] as *const u64) });
    }
    // What the driver does for itself touches nothing the program can see
    if !own {
        unsafe { note_writers(command.name, args) };
    }
    unsafe { copies::note(command.name, args) };
    if connection.keep_uploads && !own {
        unsafe { uploads::note(command.name, args) };
        unsafe { later::note(command.name, args) };
    }
    // What an image needs of memory was answered for the last one created the same way
    if unsafe { requirements::answer(command.name, args) } {
        return 0;
    }
    if command.name == "vkDestroyImage" {
        requirements::image_destroyed(unsafe { *(args[1] as *const u64) });
    }
    // A wait only copies were behind is answered here
    if matches!(command.name, "vkWaitForFences" | "vkGetFenceStatus") {
        let device = unsafe { objects::id_of(*(args[0] as *const u64)) };
        let fences: &[u64] = unsafe {
            if command.name == "vkGetFenceStatus" {
                std::slice::from_raw_parts(args[1] as *const u64, 1)
            } else {
                std::slice::from_raw_parts(*(args[2] as *const *const u64), *(args[1] as *const u32) as usize)
            }
        };
        if copies::answer_wait(&connection, device, fences) {
            return VK_SUCCESS as u64;
        }
    }

    // A wait whose answer is on its way here needn't ask for it
    if command.name == "vkWaitSemaphores"
        && let Some(result) = unsafe { timelines::wait(&connection, &**(args[1] as *const *const VkSemaphoreWaitInfo), *(args[2] as *const u64)) }
    {
        if result == VK_SUCCESS {
            let unseen = unsafe { timelines::unseen(&**(args[1] as *const *const VkSemaphoreWaitInfo)) };
            if !unseen.is_empty() {
                memory::refresh(&connection);
                for (semaphore, value) in unseen {
                    timelines::told(semaphore, value);
                }
            }
        }
        return result as i64 as u64;
    }
    if command.name == "vkGetSemaphoreCounterValue"
        && let Some((value, newer)) = timelines::value_on_its_way(unsafe { *(args[1] as *const u64) })
    {
        let semaphore = unsafe { *(args[1] as *const u64) };
        let _ = connection.flush();
        unsafe { **(args[2] as *const *mut u64) = value };
        if newer {
            memory::refresh(&connection);
            timelines::told(semaphore, value);
        }
        return VK_SUCCESS as u64;
    }
    if command.name == "vkDestroySemaphore" {
        timelines::destroyed(unsafe { *(args[1] as *const u64) });
    }

    // An allocation whose answer is known here needn't wait for it
    let mut known_allocation = false;
    if command.name == "vkAllocateDescriptorSets" {
        match unsafe { descriptors::allocation(args) } {
            descriptors::Allocation::Refuse => return VK_ERROR_OUT_OF_POOL_MEMORY as i64 as u64,
            descriptors::Allocation::Send => known_allocation = true,
            descriptors::Allocation::Ask => {}
        }
    }

    // So is a pipeline, unless the program wants to hear something about how it went
    let compiled_later = matches!(command.name, "vkCreateGraphicsPipelines" | "vkCreateComputePipelines") && unsafe { compiles_later(command.name, args) };

    let code_order = memory::code_order(&connection, command.name);
    let mut w = Writer::new();
    if let Err(error) = unsafe { wire::encode_call(command, args, &mut w, &mut Borrower) } {
        log(&format!("{} can't be carried: {error:?}", command.name));
        return failed(VK_ERROR_FEATURE_NOT_PRESENT);
    }
    // Encoding it chose the buffer's id
    if command.name == "vkCreateBuffer" {
        unsafe {
            let info = &**(args[1] as *const *const VkBufferCreateInfo);
            memory::buffer_created(**(args[3] as *const *const u64), info.size, info.usage, !info.pNext.is_null());
        }
    }
    if command.name == "vkCreateImage" {
        unsafe {
            let (image, info) = (**(args[3] as *const *const u64), &**(args[1] as *const *const VkImageCreateInfo));
            uploads::image_created(image, info.format);
            requirements::image_created(*(args[0] as *const u64), image, info);
            if connection.keep_uploads {
                later::image_created(image, *(args[0] as *const u64) as usize, info);
            }
        }
    }
    if command.name == "vkAllocateMemory" {
        unsafe { memory::memory_allocated(**(args[3] as *const *const u64), (**(args[1] as *const *const VkMemoryAllocateInfo)).memoryTypeIndex) };
    }
    if command.name == "vkAllocateMemory" && connection.keep_uploads {
        unsafe { later::memory_allocated(**(args[3] as *const *const u64), (**(args[1] as *const *const VkMemoryAllocateInfo)).memoryTypeIndex) };
    }
    if command.name == "vkCreateSemaphore" {
        unsafe { timelines::created(args) };
    }
    if command.name == "vkCreateBufferView" {
        unsafe {
            let info = &**(args[1] as *const *const VkBufferViewCreateInfo);
            memory::view_created(**(args[3] as *const *const u64), info.buffer, info.offset, info.range);
        }
    }
    unsafe { descriptors::note(command.name, args) };
    // A device that can't be created must say so at once
    let mode = if command.name == "vkCreateDevice" {
        Mode::Sync
    } else if known_allocation || compiled_later {
        Mode::Async
    } else {
        command.mode
    };
    if mode == Mode::Async {
        if logging() {
            log(command.name);
        }
        let sent = connection.send(command.id, w.as_slice());
        drop(code_order);
        if matches!(command.name, "vkQueueSubmit" | "vkQueueSubmit2") {
            unsafe { timelines::submitted(&connection, command.name, args, &pushing) };
        }
        // Work on the GPU shouldn't wait for the flushing thread
        if matches!(command.name, "vkQueueSubmit" | "vkQueueSubmit2" | "vkQueueBindSparse") {
            let _ = connection.flush();
        }
        return match sent {
            Ok(()) => 0,
            Err(_) => failed(DEVICE_LOST),
        };
    }
    let asked = std::time::Instant::now();
    let answer = if code_order.is_some() {
        let later = connection.ask_later(command.id, w.as_slice());
        drop(code_order);
        later.and_then(connection::Later::wait).ok()
    } else {
        answer_of(&connection, command, w.as_slice())
    };
    note_wait(command.name, asked.elapsed());
    let Some(answer) = answer else {
        return failed(DEVICE_LOST);
    };
    if answer.deferred != 0 {
        log(&format!("an earlier call failed on the lender with {}", answer.deferred));
    }
    if logging() {
        let (sent, received) = (connection.sent.load(Relaxed), connection.received.load(Relaxed));
        log(&format!("{} = {} (sent {sent}, received {received})", command.name, answer.result as i64));
    }
    let succeeded = command.ret != Ret::Result || (answer.result as i64) >= 0;
    if succeeded && let Err(error) = unsafe { wire::decode_outputs(command, args, &mut answer.outputs(), &mut Borrower) } {
        log(&format!("{}: bad answer: {error:?}", command.name));
        return failed(DEVICE_LOST);
    }
    if succeeded {
        unsafe { after(command.name, args) };
        // The GPU's writes come back once the work is known to be done: a wait or a fence query
        // that says so (VK_SUCCESS, not VK_NOT_READY or VK_TIMEOUT)
        let done = answer.result == VK_SUCCESS as u64;
        if done {
            unsafe { note_finished(command.name, args) };
        }
        if done && (mode == Mode::Wait || matches!(command.name, "vkGetFenceStatus" | "vkGetSemaphoreCounterValue")) {
            memory::refresh(&connection);
        }
    }
    answer.result
}

/// What the program has waited for the lender over, for NECTAR_VK_STATS: every 10 s, the commands
/// that cost the most, as these are what a game spends a link's round trips on while it loads
fn note_wait(name: &'static str, took: std::time::Duration) {
    static REPORT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*REPORT.get_or_init(|| env::var_os("NECTAR_VK_STATS").is_some()) {
        return;
    }
    type Waits = (std::time::Instant, std::collections::HashMap<&'static str, (u32, f64)>);
    static WAITS: std::sync::Mutex<Option<Waits>> = std::sync::Mutex::new(None);
    let mut waits = WAITS.lock().unwrap_or_else(|p| p.into_inner());
    let (since, by_name) = waits.get_or_insert_with(|| (std::time::Instant::now(), Default::default()));
    let entry = by_name.entry(name).or_default();
    entry.0 += 1;
    entry.1 += took.as_secs_f64() * 1000.0;
    if since.elapsed().as_secs() >= 10 {
        let mut list: Vec<_> = by_name.drain().collect();
        list.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
        let top: Vec<String> = list.iter().take(8).map(|(name, (count, ms))| format!("{name} {count} for {ms:.0} ms")).collect();
        let total = list.iter().fold((0, 0.0), |sum, (_, (count, ms))| (sum.0 + count, sum.1 + ms));
        log(&format!("waits: {} for {:.0} ms in {:.0} s: {}", total.0, total.1, since.elapsed().as_secs_f64(), top.join(", ")));
        *since = std::time::Instant::now();
    }
}

/// Notes what an answer of success says about the work that's finished, for memory.rs
unsafe fn note_finished(name: &str, args: &[*const c_void]) {
    unsafe {
        match name {
            "vkWaitForFences" => {
                let (count, all) = (*(args[1] as *const u32) as usize, *(args[3] as *const u32) != 0);
                if all || count == 1 {
                    for fence in std::slice::from_raw_parts(*(args[2] as *const *const u64), count) {
                        memory::fence_signalled(*fence);
                    }
                }
            }
            "vkGetFenceStatus" => memory::fence_signalled(*(args[1] as *const u64)),
            "vkQueueWaitIdle" => memory::queue_idle(*(args[0] as *const u64)),
            "vkDeviceWaitIdle" => memory::queue_idle(0),
            _ => {}
        }
    }
}

/// Whether these pipelines can be left to the lender to compile, without waiting: not if the program
/// asks to be told a compile would be needed, or how long one took. A failure then is reported by
/// ending the connection, if the program goes on to use the pipeline. NECTAR_VK_WAIT_PIPELINES=1 waits
/// for them all the same.
unsafe fn compiles_later(name: &str, args: &[*const c_void]) -> bool {
    static WAIT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *WAIT.get_or_init(|| std::env::var_os("NECTAR_VK_WAIT_PIPELINES").is_some()) {
        return false;
    }
    unsafe fn quiet(flags: VkPipelineCreateFlags, mut next: *const VkBaseInStructure) -> bool {
        if flags & VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT != 0 {
            return false;
        }
        unsafe {
            while let Some(link) = next.as_ref() {
                match link.sType {
                    VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO => return false,
                    VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO
                        if (*(next as *const VkPipelineCreateFlags2CreateInfo)).flags & VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT != 0 =>
                    {
                        return false;
                    }
                    _ => {}
                }
                next = link.pNext;
            }
        }
        true
    }
    unsafe {
        let count = *(args[2] as *const u32) as usize;
        if name == "vkCreateGraphicsPipelines" {
            std::slice::from_raw_parts(*(args[3] as *const *const VkGraphicsPipelineCreateInfo), count)
                .iter()
                .all(|info| quiet(info.flags, info.pNext.cast()))
        } else {
            std::slice::from_raw_parts(*(args[3] as *const *const VkComputePipelineCreateInfo), count)
                .iter()
                .all(|info| quiet(info.flags, info.pNext.cast()))
        }
    }
}

/// Notes what lets the GPU write which memory (memory.rs), before a command goes out
unsafe fn note_writers(name: &str, args: &[*const c_void]) {
    // Commands that write by address, so nothing can be said about what they wrote
    const BY_ADDRESS: &[&str] = &[
        "vkCmdBuild",
        "vkCmdPreprocess",
        "vkCmdCopyMemory",
        "vkCmdDecompress",
        "vkCmdWriteAccel",
        "vkCmdWriteMicromaps",
        "vkCmdCopyMicromap",
        "vkCmdCopyAccel",
        "vkCmdCopyTensor",
        "vkCmdConvertCooperative",
    ];
    if BY_ADDRESS.iter().any(|prefix| name.starts_with(prefix)) {
        memory::anything_written();
        return;
    }
    let value = |index: usize| unsafe { *(args[index] as *const u64) };
    unsafe {
        match name {
            "vkBeginCommandBuffer" => memory::recording_began(value(0)),
            "vkFreeCommandBuffers" => {
                for commands in std::slice::from_raw_parts(*(args[3] as *const *const u64), *(args[2] as *const u32) as usize) {
                    memory::recording_freed(*commands);
                }
            }
            "vkCmdExecuteCommands" => {
                memory::recording_includes(value(0), std::slice::from_raw_parts(*(args[2] as *const *const u64), *(args[1] as *const u32) as usize));
            }
            "vkCmdCopyBuffer" => {
                let list = std::slice::from_raw_parts(*(args[4] as *const *const VkBufferCopy), *(args[3] as *const u32) as usize);
                for copy in list {
                    memory::writes_once(value(0), value(2), copy.dstOffset, copy.size);
                }
            }
            "vkCmdCopyBuffer2" => {
                let info = &**(args[1] as *const *const VkCopyBufferInfo2);
                for copy in std::slice::from_raw_parts(info.pRegions, info.regionCount as usize) {
                    memory::writes_once(value(0), info.dstBuffer, copy.dstOffset, copy.size);
                }
            }
            "vkCmdFillBuffer" | "vkCmdUpdateBuffer" => memory::writes_once(value(0), value(1), value(2), value(3)),
            "vkCmdCopyImageToBuffer" => {
                let list = std::slice::from_raw_parts(*(args[5] as *const *const VkBufferImageCopy), *(args[4] as *const u32) as usize);
                for copy in list {
                    let size = uploads::image_format(value(1)).and_then(|format| {
                        uploads::image_copy_size(format, copy.bufferRowLength, copy.bufferImageHeight, copy.imageExtent, copy.imageSubresource.layerCount)
                    });
                    // Depth, stencil or an image not known: all of it
                    match size {
                        Some(size) => memory::writes_once(value(0), value(3), copy.bufferOffset, size as u64),
                        None => memory::writes_once(value(0), value(3), 0, u64::MAX),
                    }
                }
            }
            "vkCmdCopyImageToBuffer2" => {
                let info = &**(args[1] as *const *const VkCopyImageToBufferInfo2);
                for copy in std::slice::from_raw_parts(info.pRegions, info.regionCount as usize) {
                    let size = uploads::image_format(info.srcImage).and_then(|format| {
                        uploads::image_copy_size(format, copy.bufferRowLength, copy.bufferImageHeight, copy.imageExtent, copy.imageSubresource.layerCount)
                    });
                    match size {
                        Some(size) => memory::writes_once(value(0), info.dstBuffer, copy.bufferOffset, size as u64),
                        None => memory::writes_once(value(0), info.dstBuffer, 0, u64::MAX),
                    }
                }
            }
            "vkCmdCopyQueryPoolResults" => {
                // The room each takes, and at most a value and an availability value for the last
                let (count, offset, stride) = (*(args[3] as *const u32) as u64, value(5), value(6));
                memory::writes_once(value(0), value(4), offset, count.saturating_mul(stride).saturating_add(16));
            }
            "vkCmdWriteBufferMarkerAMD" => memory::writes_once(value(0), value(2), value(3), 4),
            "vkCmdWriteBufferMarker2AMD" => memory::writes_once(value(0), value(2), value(3), 4),
            "vkCmdBindTransformFeedbackBuffersEXT" | "vkCmdBindTransformFeedbackBuffers2EXT" => {
                let count = *(args[2] as *const u32) as usize;
                for &buffer in std::slice::from_raw_parts(*(args[3] as *const *const u64), count) {
                    memory::written(buffer);
                }
            }
            "vkCmdBeginTransformFeedbackEXT" | "vkCmdEndTransformFeedbackEXT" => {
                let (count, buffers) = (*(args[2] as *const u32) as usize, *(args[3] as *const *const u64));
                if !buffers.is_null() {
                    for &buffer in std::slice::from_raw_parts(buffers, count) {
                        memory::written(buffer);
                    }
                }
            }
            "vkUpdateDescriptorSets" => {
                let count = *(args[1] as *const u32) as usize;
                if count > 0 {
                    note_storage(std::slice::from_raw_parts(*(args[2] as *const *const VkWriteDescriptorSet), count));
                }
            }
            "vkCmdPushDescriptorSet" => {
                let count = *(args[4] as *const u32) as usize;
                if count > 0 {
                    note_storage(std::slice::from_raw_parts(*(args[5] as *const *const VkWriteDescriptorSet), count));
                }
            }
            "vkCmdPushDescriptorSet2" => {
                let info = &**(args[1] as *const *const VkPushDescriptorSetInfo);
                if info.descriptorWriteCount > 0 {
                    note_storage(std::slice::from_raw_parts(info.pDescriptorWrites, info.descriptorWriteCount as usize));
                }
            }
            "vkCreateShaderModule" => {
                let info = &**(args[1] as *const *const VkShaderModuleCreateInfo);
                note_shader(info);
            }
            "vkCreateGraphicsPipelines" => {
                let count = *(args[2] as *const u32) as usize;
                for pipeline in std::slice::from_raw_parts(*(args[3] as *const *const VkGraphicsPipelineCreateInfo), count) {
                    if pipeline.stageCount > 0 {
                        note_stages(std::slice::from_raw_parts(pipeline.pStages, pipeline.stageCount as usize));
                    }
                }
            }
            "vkCreateComputePipelines" => {
                let count = *(args[2] as *const u32) as usize;
                for pipeline in std::slice::from_raw_parts(*(args[3] as *const *const VkComputePipelineCreateInfo), count) {
                    note_stages(std::slice::from_ref(&pipeline.stage));
                }
            }
            "vkCreateShadersEXT" | "vkCreatePipelineBinariesKHR" => memory::addresses_write(),
            "vkBindBufferMemory" => memory::buffer_bound(value(1), value(2), value(3)),
            "vkBindImageMemory" => memory::image_bound(value(2)),
            "vkBindBufferMemory2" => {
                let count = *(args[1] as *const u32) as usize;
                for info in std::slice::from_raw_parts(*(args[2] as *const *const VkBindBufferMemoryInfo), count) {
                    memory::buffer_bound(info.buffer, info.memory, info.memoryOffset);
                }
            }
            "vkBindImageMemory2" => {
                let count = *(args[1] as *const u32) as usize;
                for info in std::slice::from_raw_parts(*(args[2] as *const *const VkBindImageMemoryInfo), count) {
                    memory::image_bound(info.memory);
                }
            }
            "vkDestroyBuffer" => memory::buffer_destroyed(value(1)),
            _ => {}
        }
    }
}

/// Storage descriptors let the GPU write what they name
unsafe fn note_storage(writes: &[VkWriteDescriptorSet]) {
    for write in writes {
        let count = write.descriptorCount as usize;
        unsafe {
            match write.descriptorType {
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER => {
                    for info in std::slice::from_raw_parts(write.pBufferInfo, count) {
                        memory::storage_named(write.dstBinding, info.buffer, info.offset, info.range);
                    }
                }
                // Where in the buffer is only said when it's bound
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC => {
                    for info in std::slice::from_raw_parts(write.pBufferInfo, count) {
                        memory::storage_named(write.dstBinding, info.buffer, 0, u64::MAX);
                    }
                }
                VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER => {
                    for &view in std::slice::from_raw_parts(write.pTexelBufferView, count) {
                        memory::view_written(write.dstBinding, view);
                    }
                }
                _ => {}
            }
        }
    }
}

unsafe fn note_shader(info: &VkShaderModuleCreateInfo) {
    let code = unsafe { std::slice::from_raw_parts(info.pCode, info.codeSize / 4) };
    if memory::writes_through_addresses(code) {
        memory::addresses_write();
    }
    memory::storage_writes(&memory::storage_writes_of(code));
}

/// A stage's shader is a module seen earlier, or chained on the stage. One that's neither, known
/// only by an identifier, can't be looked inside.
unsafe fn note_stages(stages: &[VkPipelineShaderStageCreateInfo]) {
    for stage in stages {
        if stage.module != 0 {
            continue;
        }
        let mut next = stage.pNext as *const VkBaseInStructure;
        let mut found = false;
        unsafe {
            while let Some(link) = next.as_ref() {
                if link.sType == VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO {
                    note_shader(&*(next as *const VkShaderModuleCreateInfo));
                    found = true;
                }
                next = link.pNext;
            }
        }
        if !found {
            memory::addresses_write();
        }
    }
}

/// Answers that don't change, kept by the request: what a physical device is and can do. DXVK asks
/// for a format's properties each frame. How much memory is free does change, so that's asked again
/// after a second.
fn kept_for(name: &str) -> Option<Option<Duration>> {
    match name {
        "vkGetPhysicalDeviceFormatProperties"
        | "vkGetPhysicalDeviceFormatProperties2"
        | "vkGetPhysicalDeviceImageFormatProperties"
        | "vkGetPhysicalDeviceImageFormatProperties2"
        | "vkGetPhysicalDeviceSparseImageFormatProperties"
        | "vkGetPhysicalDeviceSparseImageFormatProperties2"
        | "vkGetPhysicalDeviceQueueFamilyProperties"
        | "vkGetPhysicalDeviceQueueFamilyProperties2"
        | "vkGetPhysicalDeviceProperties"
        | "vkGetPhysicalDeviceProperties2"
        | "vkGetPhysicalDeviceFeatures"
        | "vkGetPhysicalDeviceFeatures2" => Some(None),
        "vkGetPhysicalDeviceMemoryProperties" | "vkGetPhysicalDeviceMemoryProperties2" => Some(Some(Duration::from_secs(1))),
        _ => None,
    }
}

type KeptAnswer = (std::time::Instant, u64, Vec<u8>);
type KeptAnswers = HashMap<(u32, Vec<u8>), KeptAnswer>;
static KEPT_ANSWERS: Mutex<Option<KeptAnswers>> = Mutex::new(None);

/// The answer to a request, kept or from the lender
fn answer_of(connection: &connection::Connection, command: &nectar_vk_wire::tables::CommandInfo, request: &[u8]) -> Option<connection::Answer> {
    let Some(lasts) = kept_for(command.name) else {
        return connection.ask(command.id, request).ok();
    };
    let key = (command.id, request.to_vec());
    {
        let kept = KEPT_ANSWERS.lock().unwrap_or_else(|p| p.into_inner());
        if let Some((at, result, body)) = kept.as_ref().and_then(|kept| kept.get(&key))
            && lasts.is_none_or(|lasts| at.elapsed() < lasts)
        {
            return Some(connection::Answer { result: *result, deferred: 0, body: body.clone(), at: 12, crossing: Duration::ZERO });
        }
    }
    let answer = connection.ask(command.id, request).ok()?;
    KEPT_ANSWERS
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .get_or_insert_with(HashMap::new)
        .insert(key, (std::time::Instant::now(), answer.result, answer.body.clone()));
    Some(answer)
}

/// What the borrower adds to a command's answer
unsafe fn after(name: &str, args: &[*const c_void]) {
    unsafe {
        match name {
            "vkGetPhysicalDeviceProperties" => rename(*(args[1] as *const *mut VkPhysicalDeviceProperties)),
            "vkGetPhysicalDeviceProperties2" => {
                rename(&mut (**(args[1] as *const *mut VkPhysicalDeviceProperties2)).properties)
            }
            "vkGetPhysicalDeviceMemoryProperties" => {
                remote_memory(*(args[1] as *const *mut VkPhysicalDeviceMemoryProperties))
            }
            "vkGetPhysicalDeviceMemoryProperties2" => {
                remote_memory(&mut (**(args[1] as *const *mut VkPhysicalDeviceMemoryProperties2)).memoryProperties)
            }
            "vkGetDeviceImageMemoryRequirements" | "vkGetImageMemoryRequirements" | "vkGetImageMemoryRequirements2" => requirements::answered(name, args),
            _ => {}
        }
    }
}

/// Offers the GPU's own memory as device-only, as on a GPU without resizable BAR. The lender's CPU
/// reads it over PCIe, slowly, and each wait would read it all to find what the GPU changed, so
/// programs stage through system memory instead. A GPU whose only memory the CPU can map is its
/// own (integrated or unified) is left as it is.
unsafe fn remote_memory(properties: *mut VkPhysicalDeviceMemoryProperties) {
    const HOST: VkMemoryPropertyFlags =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    const STAGING: VkMemoryPropertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    let Some(properties) = (unsafe { properties.as_mut() }) else { return };
    let count = (properties.memoryTypeCount as usize).min(properties.memoryTypes.len());
    let types = &mut properties.memoryTypes[..count];
    let device = |flags: VkMemoryPropertyFlags| flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT != 0;
    if !types.iter().any(|t| !device(t.propertyFlags) && t.propertyFlags & STAGING == STAGING) {
        return;
    }
    for t in types.iter_mut().filter(|t| device(t.propertyFlags)) {
        t.propertyFlags &= !HOST;
    }
    memory::memory_types_told(&types.iter().map(|t| t.propertyFlags).collect::<Vec<_>>());
}

/// Marks the device's name as borrowed
unsafe fn rename(properties: *mut VkPhysicalDeviceProperties) {
    if properties.is_null() {
        return;
    }
    let name = unsafe { &mut (*properties).deviceName };
    let length = name.iter().position(|&c| c == 0).unwrap_or(name.len());
    let suffix = SUFFIX.as_bytes();
    let start = length.min(name.len() - 1 - suffix.len());
    for (i, &byte) in suffix.iter().enumerate() {
        name[start + i] = byte as c_char;
    }
    name[start + suffix.len()] = 0;
}

// ---- The loader interface ----

#[unsafe(no_mangle)]
pub unsafe extern "system" fn vk_icdNegotiateLoaderICDInterfaceVersion(version: *mut u32) -> VkResult {
    unsafe {
        if version.is_null() || *version < LOADER_INTERFACE {
            return VK_ERROR_INCOMPATIBLE_DRIVER;
        }
        *version = LOADER_INTERFACE;
    }
    VK_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn vk_icdGetInstanceProcAddr(instance: VkInstance, name: *const c_char) -> PFN_vkVoidFunction {
    unsafe { vkGetInstanceProcAddr(instance, name) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn vk_icdGetPhysicalDeviceProcAddr(instance: VkInstance, name: *const c_char) -> PFN_vkVoidFunction {
    unsafe { vkGetInstanceProcAddr(instance, name) }
}

/// A command by name: written here, or generated
fn lookup(name: &str) -> PFN_vkVoidFunction {
    macro_rules! own {
        ($($f:ident),*) => {
            match name {
                $(stringify!($f) => return Some(unsafe { std::mem::transmute::<*const (), unsafe extern "system" fn()>($f as *const ()) }),)*
                _ => {}
            }
        };
    }
    let name = command_name(name);
    #[cfg(any(windows, target_os = "linux"))]
    if let Some(function) = present::lookup(name).or_else(|| window::lookup(name)) {
        return Some(function);
    }
    if let Some(function) = templates::lookup(name) {
        return Some(function);
    }
    own!(
        vkGetInstanceProcAddr,
        vkGetDeviceProcAddr,
        vkCreateInstance,
        vkDestroyInstance,
        vkEnumerateInstanceVersion,
        vkEnumerateInstanceExtensionProperties,
        vkEnumerateInstanceLayerProperties,
        vkEnumerateDeviceLayerProperties,
        vkEnumeratePhysicalDevices,
        vkEnumerateDeviceExtensionProperties,
        vkCreateDevice,
        vkGetDeviceQueue,
        vkGetDeviceQueue2,
        vkMapMemory,
        vkMapMemory2,
        vkUnmapMemory,
        vkUnmapMemory2,
        vkFlushMappedMemoryRanges,
        vkInvalidateMappedMemoryRanges
    );
    // What can't be carried isn't offered, so the loader answers it or the program goes without
    match index_of(name) {
        Some(index) if COMMANDS[index].unsupported.is_none() => generated::entry(name),
        _ => None,
    }
}

pub unsafe extern "system" fn vkGetInstanceProcAddr(_instance: VkInstance, name: *const c_char) -> PFN_vkVoidFunction {
    if name.is_null() {
        return None;
    }
    lookup(unsafe { CStr::from_ptr(name) }.to_str().unwrap_or_default())
}

pub unsafe extern "system" fn vkGetDeviceProcAddr(_device: VkDevice, name: *const c_char) -> PFN_vkVoidFunction {
    if name.is_null() {
        return None;
    }
    lookup(unsafe { CStr::from_ptr(name) }.to_str().unwrap_or_default())
}

// ---- Instances: made on the borrower at once, on the lender once a device is asked for ----

/// The program's instance: its id, the version it asked for, and whether the lender has it
struct Instance {
    id: u64,
    api_version: u32,
    created: bool,
}

static INSTANCES: Mutex<Vec<Instance>> = Mutex::new(Vec::new());

/// The instance extensions offered: core since Vulkan 1.1, so any lender has them
const INSTANCE_EXTENSIONS: &[(&str, u32)] = &[
    ("VK_KHR_get_physical_device_properties2", 2),
    ("VK_KHR_external_memory_capabilities", 1),
    ("VK_KHR_external_semaphore_capabilities", 1),
    ("VK_KHR_external_fence_capabilities", 1),
];

pub unsafe extern "system" fn vkCreateInstance(
    pCreateInfo: *const VkInstanceCreateInfo,
    _pAllocator: *const VkAllocationCallbacks,
    pInstance: *mut VkInstance,
) -> VkResult {
    unsafe {
        if pCreateInfo.is_null() || pInstance.is_null() {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        let application = (*pCreateInfo).pApplicationInfo;
        if !application.is_null() && !(*application).pEngineName.is_null() {
            memory::engine_named(std::ffi::CStr::from_ptr((*application).pEngineName).to_bytes());
        }
        let api_version = if application.is_null() || (*application).apiVersion == 0 {
            VK_API_VERSION_1_0
        } else {
            (*application).apiVersion
        };
        let id = objects::new_id();
        *pInstance = objects::object(id) as usize as VkInstance;
        INSTANCES.lock().unwrap_or_else(|p| p.into_inner()).push(Instance { id, api_version, created: false });
    }
    VK_SUCCESS
}

const VK_API_VERSION_1_0: u32 = 1 << 22;

/// Creates the program's instance on the lender, once
unsafe fn create_remote(instance: VkInstance) -> bool {
    let id = unsafe { id_of(instance as usize as u64) };
    let mut instances = INSTANCES.lock().unwrap_or_else(|p| p.into_inner());
    let Some(entry) = instances.iter_mut().find(|i| i.id == id) else { return false };
    if entry.created {
        return true;
    }
    let Some(connection) = connection::get() else { return false };
    let application = VkApplicationInfo {
        sType: VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pNext: ptr::null(),
        pApplicationName: c"Nectar".as_ptr(),
        applicationVersion: 1,
        pEngineName: ptr::null(),
        engineVersion: 0,
        apiVersion: entry.api_version,
    };
    let info = VkInstanceCreateInfo {
        sType: VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pNext: ptr::null(),
        flags: 0,
        pApplicationInfo: &application,
        enabledLayerCount: 0,
        ppEnabledLayerNames: ptr::null(),
        enabledExtensionCount: 0,
        ppEnabledExtensionNames: ptr::null(),
    };
    let info_ptr = &info as *const VkInstanceCreateInfo;
    let allocator: *const VkAllocationCallbacks = ptr::null();
    let mut handle = instance;
    let out = &mut handle as *mut VkInstance;
    let args = [
        &info_ptr as *const _ as *const c_void,
        &allocator as *const _ as *const c_void,
        &out as *const _ as *const c_void,
    ];
    // The instance's id is the program's object's, not a new one
    struct Same(u64);
    impl wire::Handles for Same {
        fn outgoing(&mut self, d: bool, v: u64) -> u64 {
            Borrower.outgoing(d, v)
        }
        fn incoming(&mut self, d: bool, id: u64) -> u64 {
            Borrower.incoming(d, id)
        }
        fn create(&mut self, _: bool) -> (u64, u64) {
            (self.0, objects::object(self.0))
        }
        fn bind(&mut self, _: u64, _: u64) {}
    }
    let Some(index) = index_of("vkCreateInstance") else { return false };
    let mut w = Writer::new();
    if unsafe { wire::encode_call(&COMMANDS[index], &args, &mut w, &mut Same(id)) }.is_err() {
        return false;
    }
    match connection.ask(COMMANDS[index].id, w.as_slice()) {
        Ok(answer) if answer.result as i64 as i32 == VK_SUCCESS => {
            entry.created = true;
            true
        }
        Ok(answer) => {
            log(&format!("the lender couldn't create an instance: {}", answer.result as i64));
            false
        }
        Err(_) => false,
    }
}

pub unsafe extern "system" fn vkDestroyInstance(instance: VkInstance, pAllocator: *const VkAllocationCallbacks) {
    let id = unsafe { id_of(instance as usize as u64) };
    let created = {
        let mut instances = INSTANCES.lock().unwrap_or_else(|p| p.into_inner());
        let created = instances.iter().any(|i| i.id == id && i.created);
        instances.retain(|i| i.id != id);
        created
    };
    if created && let Some(index) = index_of("vkDestroyInstance") {
        unsafe {
            call(index, &[&instance as *const _ as *const c_void, &pAllocator as *const _ as *const c_void]);
        }
    }
}

pub unsafe extern "system" fn vkEnumerateInstanceVersion(pApiVersion: *mut u32) -> VkResult {
    unsafe { *pApiVersion = nectar_vk_sys::API_VERSION_1_4 };
    VK_SUCCESS
}

/// Fills a two-call list: the count, or as many items as fit, then VK_INCOMPLETE if some didn't
pub(crate) unsafe fn fill<T: Copy>(items: &[T], count: *mut u32, out: *mut T) -> VkResult {
    unsafe {
        if out.is_null() {
            *count = items.len() as u32;
            return VK_SUCCESS;
        }
        let fits = (*count as usize).min(items.len());
        ptr::copy_nonoverlapping(items.as_ptr(), out, fits);
        *count = fits as u32;
        if fits < items.len() { VK_INCOMPLETE } else { VK_SUCCESS }
    }
}

fn extension(name: &str, version: u32) -> VkExtensionProperties {
    let mut properties: VkExtensionProperties = unsafe { std::mem::zeroed() };
    for (i, byte) in name.bytes().take(255).enumerate() {
        properties.extensionName[i] = byte as c_char;
    }
    properties.specVersion = version;
    properties
}

pub unsafe extern "system" fn vkEnumerateInstanceExtensionProperties(
    pLayerName: *const c_char,
    pPropertyCount: *mut u32,
    pProperties: *mut VkExtensionProperties,
) -> VkResult {
    if !pLayerName.is_null() {
        return VK_ERROR_LAYER_NOT_PRESENT;
    }
    #[cfg(any(windows, target_os = "linux"))]
    let surfaces = window::SURFACE_EXTENSIONS;
    #[cfg(not(any(windows, target_os = "linux")))]
    let surfaces: &[(&str, u32)] = &[];
    let list: Vec<VkExtensionProperties> = INSTANCE_EXTENSIONS.iter().chain(surfaces).map(|(n, v)| extension(n, *v)).collect();
    unsafe { fill(&list, pPropertyCount, pProperties) }
}

pub unsafe extern "system" fn vkEnumerateInstanceLayerProperties(pPropertyCount: *mut u32, _: *mut VkLayerProperties) -> VkResult {
    unsafe { *pPropertyCount = 0 };
    VK_SUCCESS
}

pub unsafe extern "system" fn vkEnumerateDeviceLayerProperties(_: VkPhysicalDevice, pPropertyCount: *mut u32, _: *mut VkLayerProperties) -> VkResult {
    unsafe { *pPropertyCount = 0 };
    VK_SUCCESS
}

/// No device while nothing is borrowed; the lender's devices when something is
pub unsafe extern "system" fn vkEnumeratePhysicalDevices(
    instance: VkInstance,
    pPhysicalDeviceCount: *mut u32,
    pPhysicalDevices: *mut VkPhysicalDevice,
) -> VkResult {
    unsafe {
        if !create_remote(instance) {
            *pPhysicalDeviceCount = 0;
            return VK_SUCCESS;
        }
        let Some(index) = index_of("vkEnumeratePhysicalDevices") else { return VK_ERROR_INITIALIZATION_FAILED };
        call(
            index,
            &[
                &instance as *const _ as *const c_void,
                &pPhysicalDeviceCount as *const _ as *const c_void,
                &pPhysicalDevices as *const _ as *const c_void,
            ],
        ) as i64 as VkResult
    }
}

/// The lender's device extensions, all of them
unsafe fn lender_extensions(physicalDevice: VkPhysicalDevice) -> Result<Vec<VkExtensionProperties>, VkResult> {
    let Some(index) = index_of("vkEnumerateDeviceExtensionProperties") else { return Err(VK_ERROR_INITIALIZATION_FAILED) };
    let mut count = 0u32;
    let layer: *const c_char = ptr::null();
    let none: *mut VkExtensionProperties = ptr::null_mut();
    let count_ptr = &mut count as *mut u32;
    unsafe {
        let args = |list: &*mut VkExtensionProperties| {
            [
                &physicalDevice as *const _ as *const c_void,
                &layer as *const _ as *const c_void,
                &count_ptr as *const _ as *const c_void,
                list as *const _ as *const c_void,
            ]
        };
        let result = call(index, &args(&none)) as i64 as VkResult;
        if result < 0 {
            return Err(result);
        }
        let mut all: Vec<VkExtensionProperties> = vec![std::mem::zeroed(); count as usize];
        let list = all.as_mut_ptr();
        let result = call(index, &args(&list)) as i64 as VkResult;
        if result < 0 {
            return Err(result);
        }
        all.truncate(count as usize);
        Ok(all)
    }
}

/// The lender's device extensions that Nectar can carry, and the swapchain, which it does itself
pub unsafe extern "system" fn vkEnumerateDeviceExtensionProperties(
    physicalDevice: VkPhysicalDevice,
    pLayerName: *const c_char,
    pPropertyCount: *mut u32,
    pProperties: *mut VkExtensionProperties,
) -> VkResult {
    if !pLayerName.is_null() {
        return VK_ERROR_LAYER_NOT_PRESENT;
    }
    unsafe {
        let all = match lender_extensions(physicalDevice) {
            Ok(all) => all,
            Err(result) => return result,
        };
        #[allow(unused_mut)]
        let mut carried: Vec<VkExtensionProperties> = all
            .into_iter()
            .filter(|e| {
                let name = CStr::from_ptr(e.extensionName.as_ptr()).to_str().unwrap_or_default();
                EXTENSIONS.iter().any(|(known, _)| *known == name) && !NOT_WORTH_ASKING.contains(&name) && !HOST_ADDRESSES.contains(&name) && !of_the_window_system(name)
            })
            .collect();
        #[cfg(any(windows, target_os = "linux"))]
        carried.push(extension(SWAPCHAIN, 70));
        fill(&carried, pPropertyCount, pProperties)
    }
}

/// Whether an extension works on the swapchain or the display, which are this side's: the lender
/// never sees a swapchain or a present, and anything chained on them here wouldn't be honoured
fn of_the_window_system(name: &str) -> bool {
    ["swapchain", "present", "surface", "display", "low_latency", "anti_lag", "hdr_metadata", "full_screen", "frame_boundary"]
        .iter()
        .any(|part| name.contains(part))
}

/// An extension that only saves a program compiling what it already has, which takes a round trip for each
/// shader to ask: vkd3d-proton does without.
const NOT_WORTH_ASKING: &[&str] = &["VK_EXT_shader_module_identifier"];

/// Extensions whose commands take addresses in the program's memory inside structs, which mean
/// nothing on the lender: the tables carry them as numbers, and it would write through them.
/// DXVK takes its ordinary descriptor path without them.
const HOST_ADDRESSES: &[&str] = &[
    "VK_EXT_descriptor_heap",
    "VK_EXT_descriptor_buffer",
    "VK_EXT_host_image_copy",
    "VK_EXT_external_memory_host",
    "VK_KHR_acceleration_structure",
    "VK_KHR_deferred_host_operations",
    "VK_KHR_ray_tracing_pipeline",
    "VK_KHR_ray_query",
];

#[cfg(any(windows, target_os = "linux"))]
const SWAPCHAIN: &str = "VK_KHR_swapchain";

/// Creates the device on the lender. The swapchain is done here, so a lender without it (one with
/// no display, say) is still asked for the rest.
pub unsafe extern "system" fn vkCreateDevice(
    physicalDevice: VkPhysicalDevice,
    pCreateInfo: *const VkDeviceCreateInfo,
    pAllocator: *const VkAllocationCallbacks,
    pDevice: *mut VkDevice,
) -> VkResult {
    unsafe {
        let mut info = *pCreateInfo;
        let names = if info.enabledExtensionCount > 0 {
            std::slice::from_raw_parts(info.ppEnabledExtensionNames, info.enabledExtensionCount as usize)
        } else {
            &[]
        };
        let is_swapchain = |name: &*const c_char| CStr::from_ptr(*name).to_bytes() == b"VK_KHR_swapchain";
        let lender_has = names.iter().any(is_swapchain)
            && lender_extensions(physicalDevice).is_ok_and(|all| {
                all.iter().any(|e| CStr::from_ptr(e.extensionName.as_ptr()).to_bytes() == b"VK_KHR_swapchain")
            });
        let kept: Vec<*const c_char> = names.iter().filter(|name| lender_has || !is_swapchain(name)).copied().collect();
        info.enabledExtensionCount = kept.len() as u32;
        info.ppEnabledExtensionNames = kept.as_ptr();
        let result = generated::vkCreateDevice(physicalDevice, &info, pAllocator, pDevice);
        #[cfg(any(windows, target_os = "linux"))]
        if result == VK_SUCCESS {
            present::device_created(*pDevice, physicalDevice, pCreateInfo);
        }
        result
    }
}

pub unsafe extern "system" fn vkGetDeviceQueue(device: VkDevice, queueFamilyIndex: u32, queueIndex: u32, pQueue: *mut VkQueue) {
    unsafe {
        generated::vkGetDeviceQueue(device, queueFamilyIndex, queueIndex, pQueue);
        later::queue_found(*pQueue as usize, queueFamilyIndex);
        #[cfg(any(windows, target_os = "linux"))]
        present::queue_found(*pQueue, queueFamilyIndex);
    }
}

pub unsafe extern "system" fn vkGetDeviceQueue2(device: VkDevice, pQueueInfo: *const VkDeviceQueueInfo2, pQueue: *mut VkQueue) {
    unsafe {
        generated::vkGetDeviceQueue2(device, pQueueInfo, pQueue);
        later::queue_found(*pQueue as usize, (*pQueueInfo).queueFamilyIndex);
        #[cfg(any(windows, target_os = "linux"))]
        present::queue_found(*pQueue, (*pQueueInfo).queueFamilyIndex);
    }
}

// ---- Mapped memory ----

pub unsafe extern "system" fn vkMapMemory(
    _device: VkDevice,
    memory: VkDeviceMemory,
    offset: VkDeviceSize,
    _size: VkDeviceSize,
    _flags: VkMemoryMapFlags,
    ppData: *mut *mut c_void,
) -> VkResult {
    let Some(connection) = connection::get() else { return DEVICE_LOST };
    match memory::map(&connection, memory) {
        Ok(shadow) => {
            unsafe { *ppData = shadow.add(offset as usize).cast() };
            VK_SUCCESS
        }
        Err(result) => result,
    }
}

pub unsafe extern "system" fn vkMapMemory2(device: VkDevice, pMemoryMapInfo: *const VkMemoryMapInfo, ppData: *mut *mut c_void) -> VkResult {
    unsafe {
        let info = &*pMemoryMapInfo;
        vkMapMemory(device, info.memory, info.offset, info.size, info.flags, ppData)
    }
}

/// The shadow stays until the memory is freed, so mapping again keeps the program's pointer. Writes
/// go with the next submit, which is when the copy they're for is known.
pub unsafe extern "system" fn vkUnmapMemory(_device: VkDevice, _memory: VkDeviceMemory) {}

pub unsafe extern "system" fn vkUnmapMemory2(device: VkDevice, pMemoryUnmapInfo: *const VkMemoryUnmapInfo) -> VkResult {
    unsafe { vkUnmapMemory(device, (*pMemoryUnmapInfo).memory) };
    VK_SUCCESS
}

/// Writes go with the next submit or host signal, the first thing that could let the GPU see them, by
/// when the copy they're for is known
pub unsafe extern "system" fn vkFlushMappedMemoryRanges(_device: VkDevice, _count: u32, _ranges: *const VkMappedMemoryRange) -> VkResult {
    VK_SUCCESS
}

pub unsafe extern "system" fn vkInvalidateMappedMemoryRanges(_device: VkDevice, _count: u32, _ranges: *const VkMappedMemoryRange) -> VkResult {
    if let Some(connection) = connection::get() {
        memory::refresh(&connection);
    }
    VK_SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    fn properties(flags: &[VkMemoryPropertyFlags]) -> VkPhysicalDeviceMemoryProperties {
        let mut properties: VkPhysicalDeviceMemoryProperties = unsafe { std::mem::zeroed() };
        properties.memoryTypeCount = flags.len() as u32;
        for (t, &f) in properties.memoryTypes.iter_mut().zip(flags) {
            t.propertyFlags = f;
        }
        properties
    }

    fn flags(properties: &VkPhysicalDeviceMemoryProperties) -> Vec<VkMemoryPropertyFlags> {
        properties.memoryTypes[..properties.memoryTypeCount as usize].iter().map(|t| t.propertyFlags).collect()
    }

    #[test]
    fn window_system_extensions_are_not_offered() {
        for name in ["VK_KHR_swapchain_maintenance1", "VK_EXT_swapchain_maintenance1", "VK_KHR_present_wait", "VK_KHR_present_id", "VK_NV_low_latency2", "VK_EXT_hdr_metadata"] {
            assert!(of_the_window_system(name), "{name}");
        }
        for name in ["VK_KHR_timeline_semaphore", "VK_EXT_robustness2", "VK_KHR_dynamic_rendering"] {
            assert!(!of_the_window_system(name), "{name}");
        }
    }

    #[test]
    fn discrete_gpu_memory_is_device_only() {
        // The RTX 5070's types: its BAR window (type 4) loses its host bits
        let mut p = properties(&[0, 0x1, 0x6, 0xe, 0x7]);
        unsafe { remote_memory(&mut p) };
        assert_eq!(flags(&p), [0, 0x1, 0x6, 0xe, 0x1]);
    }

    #[test]
    fn unified_memory_is_left_alone() {
        // An integrated GPU: every type is the GPU's own and mappable
        let mut p = properties(&[0x1, 0x7, 0xf]);
        unsafe { remote_memory(&mut p) };
        assert_eq!(flags(&p), [0x1, 0x7, 0xf]);
    }
}
