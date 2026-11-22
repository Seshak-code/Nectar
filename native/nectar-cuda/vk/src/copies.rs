//! Waits that needn't cross the network: for submits that only copy into memory the program hasn't
//! mapped. Nothing the CPU can see depends on such a copy being done, so the wait is answered here,
//! and the lender holds everything that comes after it until the copy has finished, by waiting on
//! the same fence in its ordered stream (messages::HOLD). llama.cpp uploads its small inputs for
//! each token this way: a copy, then a wait.
//!
//! A submit counts only if every command buffer holds nothing but copies between buffers, fills and
//! updates of buffers and barriers, it waits on no semaphore, and no destination is mapped. An error
//! such a copy met is reported by the next answer, as errors of commands sent without waiting are.
//! NECTAR_VK_WAIT_COPIES=1 waits for them all the same.

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::sync::Mutex;

use nectar_cuda_protocol::Writer;
use nectar_vk_sys::*;
use nectar_vk_wire::messages::HOLD;

use crate::connection::Connection;
use crate::memory;

#[derive(Default)]
struct Copies {
    /// Command buffers holding only copies, with the buffers they write. One holding anything else
    /// isn't here.
    recording: HashMap<u64, Vec<u64>>,
    /// Fences of submits that count, not reset since, and whether the lender has been told to hold
    pending: HashMap<u64, bool>,
}

static COPIES: Mutex<Option<Copies>> = Mutex::new(None);

fn copies<R>(f: impl FnOnce(&mut Copies) -> R) -> R {
    f(COPIES.lock().unwrap_or_else(|p| p.into_inner()).get_or_insert_with(Copies::default))
}

fn enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NECTAR_VK_WAIT_COPIES").is_none())
}

/// What a command means for its command buffer: the buffer it writes, if it may stay copy-only
/// (Some(None) when it writes none), or None if it can't
fn writes(name: &str, value: impl Fn(usize) -> u64, info: impl Fn() -> u64) -> Option<Option<u64>> {
    match name {
        "vkCmdPipelineBarrier" | "vkCmdPipelineBarrier2" => Some(None),
        "vkCmdCopyBuffer" => Some(Some(value(2))),
        "vkCmdCopyBuffer2" => Some(Some(info())),
        "vkCmdFillBuffer" | "vkCmdUpdateBuffer" => Some(Some(value(1))),
        _ => None,
    }
}

/// Notes a command before it goes out. Most are `vkCmd` commands.
///
/// # Safety
/// `args` must be the command's arguments, each by its address.
pub unsafe fn note(name: &str, args: &[*const c_void]) {
    if !enabled() {
        return;
    }
    let commands = || unsafe { *(args[0] as *const u64) };
    match name {
        "vkBeginCommandBuffer" => {
            let commands = commands();
            copies(|copies| copies.recording.insert(commands, Vec::new()));
        }
        _ if name.starts_with("vkCmd") => {
            let value = |index: usize| unsafe { *(args[index] as *const u64) };
            let info = || unsafe { (**(args[1] as *const *const VkCopyBufferInfo2)).dstBuffer };
            let written = writes(name, value, info);
            let commands = commands();
            copies(|copies| match written {
                None => {
                    copies.recording.remove(&commands);
                }
                Some(buffer) => {
                    if let (Some(list), Some(buffer)) = (copies.recording.get_mut(&commands), buffer) {
                        list.push(buffer);
                    }
                }
            });
        }
        "vkQueueSubmit" => unsafe {
            let count = *(args[1] as *const u32) as usize;
            let fence = *(args[3] as *const u64);
            if fence != 0 && count > 0 {
                let submits = std::slice::from_raw_parts(*(args[2] as *const *const VkSubmitInfo), count);
                if counts(submits) {
                    copies(|copies| copies.pending.insert(fence, false));
                }
            }
        },
        "vkResetFences" => unsafe {
            let fences = std::slice::from_raw_parts(*(args[2] as *const *const u64), *(args[1] as *const u32) as usize);
            copies(|copies| {
                for fence in fences {
                    copies.pending.remove(fence);
                }
            });
        },
        "vkDestroyFence" => unsafe {
            let fence = *(args[1] as *const u64);
            copies(|copies| copies.pending.remove(&fence));
        },
        _ => {}
    }
}

/// Whether every command buffer of these submits only copies into memory that isn't mapped, with
/// no semaphore to wait on
unsafe fn counts(submits: &[VkSubmitInfo]) -> bool {
    let mut written = HashSet::new();
    for submit in submits {
        if submit.waitSemaphoreCount != 0 || submit.commandBufferCount == 0 {
            return false;
        }
        let list = unsafe { std::slice::from_raw_parts(submit.pCommandBuffers, submit.commandBufferCount as usize) };
        for &commands in list {
            let id = commands as usize as u64;
            let Some(buffers) = copies(|copies| copies.recording.get(&id).cloned()) else {
                return false;
            };
            written.extend(buffers);
        }
    }
    !written.into_iter().any(memory::buffer_is_mapped)
}

/// Answers a wait for `fences` here if all of them can be. True when it did.
pub fn answer_wait(connection: &Connection, device: u64, fences: &[u64]) -> bool {
    if !enabled() || fences.is_empty() {
        return false;
    }
    let to_hold: Option<Vec<u64>> = copies(|copies| {
        if !fences.iter().all(|fence| copies.pending.contains_key(fence)) {
            return None;
        }
        let new = fences.iter().copied().filter(|fence| !copies.pending[fence]).collect();
        for fence in fences {
            copies.pending.insert(*fence, true);
        }
        Some(new)
    });
    let Some(to_hold) = to_hold else { return false };
    if !to_hold.is_empty() {
        let mut w = Writer::new();
        w.u64(device).u32(to_hold.len() as u32);
        for fence in to_hold {
            w.u64(fence);
        }
        let _ = connection.send(HOLD, w.as_slice());
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_stay_and_anything_else_does_not() {
        let value = |_: usize| 7u64;
        let info = || 9u64;
        assert_eq!(writes("vkCmdPipelineBarrier", value, info), Some(None));
        assert_eq!(writes("vkCmdCopyBuffer", value, info), Some(Some(7)));
        assert_eq!(writes("vkCmdCopyBuffer2", value, info), Some(Some(9)));
        assert_eq!(writes("vkCmdFillBuffer", value, info), Some(Some(7)));
        for other in ["vkCmdDispatch", "vkCmdDraw", "vkCmdCopyImageToBuffer", "vkCmdCopyQueryPoolResults", "vkCmdExecuteCommands"] {
            assert_eq!(writes(other, value, info), None, "{other}");
        }
    }

    #[test]
    fn a_wait_is_answered_only_for_fences_of_copies() {
        // Recording: copies only, then a command buffer that dispatches
        let (only_copies, dispatches) = (101u64, 102u64);
        let begin = |commands: &u64| unsafe { note("vkBeginCommandBuffer", &[commands as *const u64 as *const c_void]) };
        begin(&only_copies);
        begin(&dispatches);
        let src = 1u64;
        let dst = 8802u64;
        memory::buffer_created(dst, 1 << 20, 0x2, false);
        memory::buffer_bound(dst, 8803, 0);
        let count = 1u32;
        let regions = std::ptr::null::<c_void>();
        let copy_args = |commands: &u64| {
            [
                commands as *const u64 as *const c_void,
                &src as *const u64 as *const c_void,
                &dst as *const u64 as *const c_void,
                &count as *const u32 as *const c_void,
                &regions as *const *const c_void as *const c_void,
            ]
        };
        unsafe { note("vkCmdCopyBuffer", &copy_args(&only_copies)) };
        unsafe { note("vkCmdCopyBuffer", &copy_args(&dispatches)) };
        unsafe { note("vkCmdDispatch", &copy_args(&dispatches)) };

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
        unsafe {
            assert!(counts(&[submit(&only_copies)]));
            assert!(!counts(&[submit(&dispatches)]));
            // Never recorded here
            assert!(!counts(&[submit(&999u64)]));
            let mut waits = submit(&only_copies);
            waits.waitSemaphoreCount = 1;
            assert!(!counts(&[waits]));
        }
    }
}
