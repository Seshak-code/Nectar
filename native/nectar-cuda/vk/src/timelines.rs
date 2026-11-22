//! Waits on timeline semaphores that needn't cross the network and come back. DXVK waits on one
//! several times a frame to know its work is done. Each submit that signals a timeline also asks
//! the lender to say when that value is reached (messages::WATCH), so the news is on its way before
//! anyone waits: a wait then costs the trip one way at most, and nothing if the GPU is done already.
//!
//! A wait for a value no submit was seen to signal goes to the lender as usual.
//! NECTAR_VK_WAIT_SEMAPHORES=1 sends them all.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

use nectar_cuda_protocol::Writer;
use nectar_vk_sys::*;
use nectar_vk_wire::messages::WATCH;

use crate::connection::Connection;
use crate::memory::Pushing;
use crate::objects;

#[derive(Default)]
struct Timeline {
    /// The highest value the lender was asked to report
    watched: u64,
    /// The highest it has reported
    reached: u64,
    /// The highest the program has been told, by `value_on_its_way`
    told: u64,
}

#[derive(Default)]
struct Known {
    timelines: HashMap<u64, Timeline>,
    lost: bool,
}

static KNOWN: Mutex<Option<Known>> = Mutex::new(None);
static CHANGED: Condvar = Condvar::new();

fn known() -> MutexGuard<'static, Option<Known>> {
    KNOWN.lock().unwrap_or_else(|p| p.into_inner())
}

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("NECTAR_VK_WAIT_SEMAPHORES").is_none())
}

/// Encoding vkCreateSemaphore chose its id: notes it if it's a timeline
///
/// # Safety
/// `args` must be vkCreateSemaphore's arguments, each by its address.
pub unsafe fn created(args: &[*const c_void]) {
    if !enabled() {
        return;
    }
    unsafe {
        let info = &**(args[1] as *const *const VkSemaphoreCreateInfo);
        let mut next = info.pNext as *const VkBaseInStructure;
        while let Some(link) = next.as_ref() {
            if link.sType == VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO
                && (*(next as *const VkSemaphoreTypeCreateInfo)).semaphoreType == VK_SEMAPHORE_TYPE_TIMELINE
            {
                let semaphore = **(args[3] as *const *const u64);
                known().get_or_insert_with(Known::default).timelines.insert(semaphore, Timeline::default());
                return;
            }
            next = link.pNext;
        }
    }
}

pub fn destroyed(semaphore: u64) {
    if let Some(known) = known().as_mut() {
        known.timelines.remove(&semaphore);
    }
}

/// A submit went out: asks to hear about the timeline values it signals, and with the last of those, as
/// everything before a signal in a queue is done when it happens, for `pushing` ahead of the news
///
/// # Safety
/// `args` must be vkQueueSubmit's or vkQueueSubmit2's arguments, each by its address.
pub unsafe fn submitted(connection: &Connection, name: &str, args: &[*const c_void], pushing: &Pushing) {
    if !enabled() {
        return;
    }
    let signals = unsafe { signals_of(name, args) };
    if signals.is_empty() {
        return;
    }
    let queue = unsafe { objects::id_of(*(args[0] as *const u64)) };
    let mut known = known();
    let Some(known) = known.as_mut() else { return };
    let last = signals.iter().rposition(|(semaphore, value)| known.timelines.get(semaphore).is_some_and(|timeline| *value > timeline.watched));
    for (at, (semaphore, value)) in signals.into_iter().enumerate() {
        // Binary semaphores aren't here
        let Some(timeline) = known.timelines.get_mut(&semaphore) else { continue };
        if value > timeline.watched {
            timeline.watched = value;
            let mut w = Writer::new();
            w.u64(queue).u64(semaphore).u64(value);
            if Some(at) == last && !pushing.ranges.is_empty() {
                crate::memory::push_awaited(semaphore, value, pushing);
                w.u32(pushing.ranges.len() as u32);
                for (memory, start, end) in &pushing.ranges {
                    w.u64(*memory).u64(*start).u64(*end);
                }
            } else {
                w.u32(0);
            }
            let _ = connection.send(WATCH, w.as_slice());
        }
    }
}

/// The timeline values a submit signals, and for all it knows other semaphores with values of 0
///
/// # Safety
/// `args` must be vkQueueSubmit's or vkQueueSubmit2's arguments, each by its address.
pub unsafe fn signals_of(name: &str, args: &[*const c_void]) -> Vec<(u64, u64)> {
    let mut signals: Vec<(u64, u64)> = Vec::new();
    unsafe {
        let count = *(args[1] as *const u32) as usize;
        if count == 0 {
            return signals;
        }
        if name == "vkQueueSubmit2" {
            for submit in std::slice::from_raw_parts(*(args[2] as *const *const VkSubmitInfo2), count) {
                if submit.signalSemaphoreInfoCount > 0 {
                    for signal in std::slice::from_raw_parts(submit.pSignalSemaphoreInfos, submit.signalSemaphoreInfoCount as usize) {
                        signals.push((signal.semaphore, signal.value));
                    }
                }
            }
        } else {
            for submit in std::slice::from_raw_parts(*(args[2] as *const *const VkSubmitInfo), count) {
                let mut next = submit.pNext as *const VkBaseInStructure;
                while let Some(link) = next.as_ref() {
                    if link.sType == VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO {
                        let values = &*(next as *const VkTimelineSemaphoreSubmitInfo);
                        let signalled = values.signalSemaphoreValueCount.min(submit.signalSemaphoreCount) as usize;
                        if signalled > 0 && !values.pSignalSemaphoreValues.is_null() {
                            let semaphores = std::slice::from_raw_parts(submit.pSignalSemaphores, signalled);
                            let values = std::slice::from_raw_parts(values.pSignalSemaphoreValues, signalled);
                            signals.extend(semaphores.iter().copied().zip(values.iter().copied()));
                        }
                    }
                    next = link.pNext;
                }
            }
        }
    }
    signals
}

/// The lender says a value is reached
pub fn reached(semaphore: u64, value: u64) {
    crate::memory::value_reached(semaphore, value);
    if let Some(known) = known().as_mut()
        && let Some(timeline) = known.timelines.get_mut(&semaphore)
    {
        timeline.reached = timeline.reached.max(value);
    }
    CHANGED.notify_all();
}

/// What a timeline has reached, while news of a higher value is on its way: the program can't tell this
/// from asking a moment earlier, and vkd3d-proton asks all the time. With the value is whether it's more
/// than what's been `told`, so that whoever sees it first makes memory ready for it. None if the lender
/// has to be asked: nothing is on its way, so a value signalled some other way wouldn't be heard of here.
pub fn value_on_its_way(semaphore: u64) -> Option<(u64, bool)> {
    if !enabled() {
        return None;
    }
    let mut known = known();
    let timeline = known.as_mut()?.timelines.get_mut(&semaphore)?;
    if timeline.watched <= timeline.reached {
        return None;
    }
    Some((timeline.reached, timeline.reached > timeline.told))
}

/// What's been reached of these and memory hasn't been made ready for
///
/// # Safety
/// `info` must be what vkWaitSemaphores allows.
pub unsafe fn unseen(info: &VkSemaphoreWaitInfo) -> Vec<(u64, u64)> {
    let semaphores = unsafe { std::slice::from_raw_parts(info.pSemaphores, info.semaphoreCount as usize) };
    let known = known();
    semaphores.iter().filter_map(|semaphore| known.as_ref()?.timelines.get(semaphore).filter(|timeline| timeline.reached > timeline.told).map(|timeline| (*semaphore, timeline.reached))).collect()
}

/// Whether any of a submit's batches waits on a semaphore
///
/// # Safety
/// `args` must be vkQueueSubmit's or vkQueueSubmit2's arguments, each by its address.
pub unsafe fn waits_of(name: &str, args: &[*const c_void]) -> bool {
    unsafe {
        let count = *(args[1] as *const u32) as usize;
        if count == 0 {
            return false;
        }
        if name == "vkQueueSubmit2" {
            std::slice::from_raw_parts(*(args[2] as *const *const VkSubmitInfo2), count).iter().any(|submit| submit.waitSemaphoreInfoCount > 0)
        } else {
            std::slice::from_raw_parts(*(args[2] as *const *const VkSubmitInfo), count).iter().any(|submit| submit.waitSemaphoreCount > 0)
        }
    }
}

/// Memory is ready for a value. Until then, everyone who sees it does that too, as one of them mustn't go
/// on to read while another is still at it.
pub fn told(semaphore: u64, value: u64) {
    if let Some(known) = known().as_mut()
        && let Some(timeline) = known.timelines.get_mut(&semaphore)
    {
        timeline.told = timeline.told.max(value);
    }
}

/// The lender is gone: no news will come
pub fn lost() {
    known().get_or_insert_with(Known::default).lost = true;
    CHANGED.notify_all();
}

/// Waits here if the news of every value is on its way; None if the lender has to be asked
///
/// # Safety
/// `info` must be what vkWaitSemaphores allows.
pub unsafe fn wait(connection: &Connection, info: &VkSemaphoreWaitInfo, timeout: u64) -> Option<VkResult> {
    if !enabled() || info.semaphoreCount == 0 {
        return None;
    }
    let count = info.semaphoreCount as usize;
    let semaphores = unsafe { std::slice::from_raw_parts(info.pSemaphores, count) };
    let values = unsafe { std::slice::from_raw_parts(info.pValues, count) };
    let any = info.flags & VK_SEMAPHORE_WAIT_ANY_BIT != 0;
    let wanted: Vec<(u64, u64)> = semaphores.iter().copied().zip(values.iter().copied()).collect();
    {
        let known = known();
        let known = known.as_ref()?;
        let covered = wanted.iter().all(|(semaphore, value)| known.timelines.get(semaphore).is_some_and(|timeline| *value <= timeline.watched));
        if !covered {
            return None;
        }
    }
    // What the news depends on must be on its way
    let _ = connection.flush();
    let deadline = Instant::now().checked_add(Duration::from_nanos(timeout));
    let mut guard = known();
    loop {
        let known = guard.as_ref()?;
        if known.lost {
            return Some(VK_ERROR_DEVICE_LOST);
        }
        let done = |(semaphore, value): &(u64, u64)| known.timelines.get(semaphore).is_some_and(|timeline| *value <= timeline.reached);
        if if any { wanted.iter().any(done) } else { wanted.iter().all(done) } {
            return Some(VK_SUCCESS);
        }
        match deadline {
            None => guard = CHANGED.wait(guard).unwrap_or_else(|p| p.into_inner()),
            Some(deadline) => {
                let left = deadline.saturating_duration_since(Instant::now());
                if left.is_zero() {
                    return Some(VK_TIMEOUT);
                }
                guard = CHANGED.wait_timeout(guard, left).unwrap_or_else(|p| p.into_inner()).0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wait_info<'a>(semaphores: &'a [u64], values: &'a [u64]) -> VkSemaphoreWaitInfo {
        VkSemaphoreWaitInfo {
            sType: VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            pNext: std::ptr::null(),
            flags: 0,
            semaphoreCount: semaphores.len() as u32,
            pSemaphores: semaphores.as_ptr(),
            pValues: values.as_ptr(),
        }
    }

    #[test]
    fn a_value_is_answered_here_only_while_news_is_on_its_way() {
        let semaphore = 977_001;
        known().get_or_insert_with(Known::default).timelines.insert(semaphore, Timeline { watched: 5, reached: 2, told: 0 });
        assert_eq!(value_on_its_way(semaphore), Some((2, true)));
        assert_eq!(value_on_its_way(semaphore), Some((2, true)), "not until it's been dealt with");
        told(semaphore, 2);
        assert_eq!(value_on_its_way(semaphore), Some((2, false)));
        reached(semaphore, 4);
        assert_eq!(value_on_its_way(semaphore), Some((4, true)));
        reached(semaphore, 5);
        assert_eq!(value_on_its_way(semaphore), None);
        assert_eq!(value_on_its_way(semaphore + 1), None);
    }

    #[test]
    fn news_reaches_those_waiting_and_only_counts_forward() {
        let id = 0x7F00_0001;
        known().get_or_insert_with(Known::default).timelines.insert(id, Timeline { watched: 5, reached: 0, told: 0 });
        reached(id, 3);
        reached(id, 2);
        assert_eq!(known().as_ref().unwrap().timelines[&id].reached, 3);
        destroyed(id);
        // A semaphore that's gone is no trouble
        reached(id, 4);
        assert!(!known().as_ref().unwrap().timelines.contains_key(&id));
        let _ = wait_info(&[id], &[1]);
    }
}
