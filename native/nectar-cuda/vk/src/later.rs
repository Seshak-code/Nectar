//! A texture's largest mips, sent after the texture is in use. Each mip holds about three quarters of
//! what's left of a chain, so the small ones cross in moments and the large ones take seconds. A game
//! shouldn't stop for that, so its copy goes ahead with something standing in: the lender makes each
//! waiting mip by repeating the blocks of the one below (messages::FILL_MIP) or, as a game may well send
//! the largest first, of a small sample of its own (FILL_MIP_FROM). The real one follows on this thread,
//! smallest first, leaving room for the game's own traffic. Once it's all there, it's copied into the
//! image. The picture starts coarse and sharpens, and nothing is lost in the end.
//!
//! On a slow link a large mip takes long enough that it's worth sending twice. First it's coded by
//! wavelet.rs, many times smaller, and the lender decodes that into blocks off the ordered stream, which
//! replaces the stand-in with something close to the real thing. Then the real bytes follow as above. On
//! a quick link the real bytes get there about as soon as coding them would, and coding only costs the
//! game CPU, so how much is getting through decides. NECTAR_VK_WAVELET=1 always codes, and 0 never.
//!
//! Only plain 2D textures qualify: one layer, only ever sampled, and each mip copied whole. If the game
//! is about to copy from one, or write to one, its submit waits here until the real mips are in place,
//! so it can't tell. NECTAR_VK_MIPS_LATER=0 turns this off.

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::mem;
use std::ptr;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use nectar_cuda_protocol::Writer;
use nectar_vk_sys::*;
use nectar_vk_wire::bc::Kind;
use nectar_vk_wire::messages::{CODED_PART, DECODE_MIP, ECHO, MAP, UNMAP};
use nectar_vk_wire::wavelet;

use crate::connection::{self, Connection};
use crate::{generated, memory, own_work};

/// Bytes going to the lender in one submit before the rest of a chain can wait: about half of what the
/// measured link carries in a frame at 120 fps
pub const FIRST_BUDGET: usize = 256 << 10;
/// A chain with less than this to send isn't worth the trouble
pub const LATER_LEAST: usize = 1 << 20;
/// The most of a mip's own blocks sent to stand in for it when nothing below it has come yet
pub const SAMPLES_MOST: usize = 64 << 10;
/// The most held here waiting to go. Past this, a texture goes the ordinary way, holding up the game.
const HELD_MOST: usize = 2 << 30;
/// A mip smaller than this isn't worth coding: its real bytes will be along in moments
const CODED_LEAST: usize = 1 << 20;
/// How many mips the lender may be decoding at once
const DECODING_MOST: usize = 2;
/// What can get through, beside the game's traffic, in bytes a second, past which mips aren't coded. On
/// the measured link this reads 23 to 31 MB/s, where coding gets to a good picture only 1.4 times as
/// soon and the real one takes 1.7 times as long. On links reading 9 to 20 MB/s it's 5 times as soon
/// and costs nothing at the end. It's a first estimate, and those two ranges are close.
const CODE_BELOW: f64 = 20e6;

const IMAGE_TYPE_2D: VkImageType = 1;
const USAGE_ALLOWED: u32 = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
/// Mutable format, block texel view, extended usage: none changes how a mip is laid out
const FLAGS_ALLOWED: u32 = 0x8 | 0x80 | 0x100;

#[derive(Clone)]
struct Image {
    device: usize,
    format: VkFormat,
    width: u32,
    height: u32,
    /// Each mip's layout, as of the last submit
    layouts: Vec<VkImageLayout>,
    /// Mips still to arrive
    waiting: u32,
    /// Which, a bit each
    mips: u64,
    /// Mips that have had real bytes sent, a bit each. Sending them again means the game animates this
    /// texture, and what it's sending now is what it wants to see.
    sent: u64,
    /// Don't carry on: the image is going away
    cancelled: bool,
}

/// A mip on its way
struct Level {
    image: u64,
    mip: u32,
    queue: usize,
    /// The memory type its bytes wait in on the lender
    memory_type: u32,
    bytes: Vec<u8>,
    /// Where it's kept, as the ordinary path would say
    place: (u64, u64),
    /// Its buffer and memory on the lender, once made
    side: Option<(VkBuffer, u64)>,
    /// How much has gone
    at: usize,
    stage: Stage,
    /// The last thing done for it was something other than sending a block, which took its own time
    did_other: bool,
}

/// How far a mip has got
enum Stage {
    /// Being coded and sent: what's coded so far, and how much of it has gone
    Coding(Option<Vec<u8>>, usize),
    /// The lender is decoding it
    Decoding(connection::Later),
    /// It's been decoded, or not, and is to go in place
    Decoded(bool),
    /// The real bytes are going
    Real,
}

#[derive(Default)]
struct Notes {
    /// (image, first mip, how many, layout), in order
    layouts: Vec<(u64, u32, u32, VkImageLayout)>,
    /// Images it copies from or writes to, other than by copying to a mip
    touches: HashSet<u64>,
    /// Mips it copies to
    copies_to: HashSet<(u64, u32)>,
    /// Something it does changes layouts in a way not followed here
    lost_track: bool,
}

#[derive(Default)]
struct Later {
    images: HashMap<u64, Image>,
    families: HashMap<usize, u32>,
    memory_types: HashMap<u64, u32>,
    recording: HashMap<u64, Notes>,
    /// Layouts can't be known any more, so nothing more waits, and nothing is applied
    lost_track: bool,
    levels: Vec<Level>,
    held: usize,
    /// The thread's running
    started: bool,
    /// Images a submit is waiting for, which go first
    urgent: HashSet<u64>,
    /// Command pools of the driver's own, by device and family
    pools: HashMap<(usize, u32), VkCommandPool>,
}

static LATER: Mutex<Option<Later>> = Mutex::new(None);
static CHANGED: Condvar = Condvar::new();
/// Held from a submit's layouts being taken into account until it's sent, and while a mip is put in
/// place: the layout a mip is left in has to be the one the game's next submit thinks it's in
static ORDER: Mutex<()> = Mutex::new(());

fn later() -> MutexGuard<'static, Option<Later>> {
    LATER.lock().unwrap_or_else(|p| p.into_inner())
}

fn with<R>(f: impl FnOnce(&mut Later) -> R) -> R {
    f(later().get_or_insert_with(Later::default))
}

pub fn enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("NECTAR_VK_MIPS_LATER").map_or(true, |value| value != "0"))
}

pub fn order() -> MutexGuard<'static, ()> {
    ORDER.lock().unwrap_or_else(|p| p.into_inner())
}

/// An image was created: is it one whose mips can wait?
pub fn image_created(image: u64, device: usize, info: &VkImageCreateInfo) {
    let qualifies = enabled()
        && info.imageType == IMAGE_TYPE_2D
        && info.arrayLayers == 1
        && info.mipLevels >= 2
        && info.samples == 1
        && info.tiling == 0
        && info.usage & !USAGE_ALLOWED == 0
        && info.flags & !FLAGS_ALLOWED == 0
        && info.sharingMode == 0;
    if qualifies {
        let image_info = Image {
            device,
            format: info.format,
            width: info.extent.width,
            height: info.extent.height,
            layouts: vec![info.initialLayout; info.mipLevels as usize],
            waiting: 0,
            mips: 0,
            sent: 0,
            cancelled: false,
        };
        with(|later| later.images.insert(image, image_info));
    }
}

/// The mip a copy fills, if it's one that could wait: all of the mip, tightly packed
pub fn whole_mip(image: u64, mip: u32, row_length: u32, image_height: u32, offset: VkOffset3D, extent: VkExtent3D) -> Option<u32> {
    if offset.x != 0 || offset.y != 0 || offset.z != 0 || extent.depth != 1 {
        return None;
    }
    with(|later| {
        let image = later.images.get(&image)?;
        let (width, height) = ((image.width >> mip).max(1), (image.height >> mip).max(1));
        let tight = (row_length == 0 || row_length == width) && (image_height == 0 || image_height == height);
        (tight && extent.width == width && extent.height == height && (mip as usize) < image.layouts.len()).then_some(mip)
    })
}

pub fn queue_found(queue: usize, family: u32) {
    with(|later| later.families.insert(queue, family));
}

pub fn memory_allocated(memory: u64, memory_type: u32) {
    with(|later| later.memory_types.insert(memory, memory_type));
}

pub fn memory_freed(memory: u64) {
    with(|later| later.memory_types.remove(&memory));
}

/// Whether this mip's bytes may wait. Either way they count as sent.
pub fn may_wait(image: u64, mip: u32, queue: usize, memory: u64, size: usize, held_now: usize) -> bool {
    with(|later| {
        let room = later.held + held_now + size <= HELD_MOST;
        let known = !later.lost_track && later.families.contains_key(&queue) && later.memory_types.contains_key(&memory);
        let Some(image) = later.images.get_mut(&image) else { return false };
        let first = image.sent & (1 << mip) == 0;
        image.sent |= 1 << mip;
        first && room && known && !image.cancelled
    })
}

/// Hands a mip's bytes to the thread
pub fn wait(connection: &Arc<Connection>, image: u64, mip: u32, queue: usize, memory: u64, bytes: Vec<u8>, place: (u64, u64)) {
    with(|later| {
        let Some(memory_type) = later.memory_types.get(&memory).copied() else { return };
        let Some(info) = later.images.get_mut(&image) else { return };
        info.waiting += 1;
        info.mips |= 1 << mip;
        later.held += bytes.len();
        let stage = if bytes.len() >= CODED_LEAST && coded_size(info, mip, bytes.len()).is_some() && coding() != Some(false) { Stage::Coding(None, 0) } else { Stage::Real };
        later.levels.push(Level { image, mip, queue, memory_type, bytes, place, side: None, at: 0, stage, did_other: false });
        if !later.started {
            later.started = true;
            let connection = Arc::clone(connection);
            std::thread::spawn(move || deliver(&connection));
        }
    });
    CHANGED.notify_all();
}

/// What this module's codecs make of an image's blocks, if anything
pub fn kind_of(image: u64) -> Option<Kind> {
    with(|later| later.images.get(&image).and_then(|image| Kind::of(image.format))).map(|(kind, _)| kind)
}

/// An image's format, or 0 if it isn't known here
pub fn format_of(image: u64) -> VkFormat {
    with(|later| later.images.get(&image).map_or(0, |image| image.format))
}

/// NECTAR_VK_WAVELET: whether to always code mips, or never, or None to go by the link
fn coding() -> Option<bool> {
    static CODING: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    *CODING.get_or_init(|| std::env::var("NECTAR_VK_WAVELET").ok().and_then(|value| match value.as_str() {
        "0" => Some(false),
        "1" => Some(true),
        _ => None,
    }))
}

/// Whether to code a mip, given what's been getting through. None means it isn't known yet.
fn worth_coding(forced: Option<bool>, getting_through: Option<f64>) -> Option<bool> {
    match forced {
        Some(forced) => Some(forced),
        None => getting_through.map(|rate| rate < CODE_BELOW),
    }
}

/// What a mip is as far as wavelet.rs goes: its kind, whether it's sRGB, and its blocks across and down,
/// if it can code it and `size` is all of the mip
fn coded_size(image: &Image, mip: u32, size: usize) -> Option<(Kind, bool, usize, usize)> {
    let (kind, srgb) = Kind::of(image.format)?;
    let side = kind.block_side();
    let (across, down) = (((image.width >> mip).max(1) as usize).div_ceil(side), ((image.height >> mip).max(1) as usize).div_ceil(side));
    (across * down * kind.block_bytes() == size).then_some((kind, srgb, across, down))
}

/// Notes a command before it goes out
///
/// # Safety
/// `args` must be the command's arguments, each by its address.
pub unsafe fn note(name: &str, args: &[*const c_void]) {
    let value = |index: usize| unsafe { *(args[index] as *const u64) };
    let notes = |later: &mut Later| -> *mut Notes { later.recording.entry(value(0)).or_default() };
    unsafe {
        match name {
            "vkBeginCommandBuffer" => {
                with(|later| later.recording.insert(value(0), Notes::default()));
            }
            "vkFreeCommandBuffers" => {
                let list = std::slice::from_raw_parts(*(args[3] as *const *const u64), *(args[2] as *const u32) as usize);
                with(|later| {
                    for commands in list {
                        later.recording.remove(commands);
                    }
                });
            }
            "vkCmdExecuteCommands" => {
                let list = std::slice::from_raw_parts(*(args[2] as *const *const u64), *(args[1] as *const u32) as usize);
                with(|later| {
                    let mut merged = Notes::default();
                    for inner in list {
                        if let Some(notes) = later.recording.get(inner) {
                            merged.layouts.extend(notes.layouts.iter().copied());
                            merged.touches.extend(notes.touches.iter().copied());
                            merged.copies_to.extend(notes.copies_to.iter().copied());
                            merged.lost_track |= notes.lost_track;
                        }
                    }
                    let notes = later.recording.entry(value(0)).or_default();
                    notes.layouts.extend(merged.layouts);
                    notes.touches.extend(merged.touches);
                    notes.copies_to.extend(merged.copies_to);
                    notes.lost_track |= merged.lost_track;
                });
            }
            "vkCmdPipelineBarrier" => {
                let count = *(args[8] as *const u32) as usize;
                if count > 0 {
                    let barriers = std::slice::from_raw_parts(*(args[9] as *const *const VkImageMemoryBarrier), count);
                    with(|later| {
                        let tracked: Vec<bool> = barriers.iter().map(|barrier| later.images.contains_key(&barrier.image)).collect();
                        let notes = &mut *notes(later);
                        for (barrier, _) in barriers.iter().zip(tracked).filter(|(_, tracked)| *tracked) {
                            layout_changed(notes, barrier.image, &barrier.subresourceRange, barrier.newLayout, barrier.srcQueueFamilyIndex != barrier.dstQueueFamilyIndex);
                        }
                    });
                }
            }
            "vkCmdPipelineBarrier2" => {
                let info = &**(args[1] as *const *const VkDependencyInfo);
                if info.imageMemoryBarrierCount > 0 {
                    let barriers = std::slice::from_raw_parts(info.pImageMemoryBarriers, info.imageMemoryBarrierCount as usize);
                    with(|later| {
                        let tracked: Vec<bool> = barriers.iter().map(|barrier| later.images.contains_key(&barrier.image)).collect();
                        let notes = &mut *notes(later);
                        for (barrier, _) in barriers.iter().zip(tracked).filter(|(_, tracked)| *tracked) {
                            layout_changed(notes, barrier.image, &barrier.subresourceRange, barrier.newLayout, barrier.srcQueueFamilyIndex != barrier.dstQueueFamilyIndex);
                        }
                    });
                }
            }
            "vkCmdWaitEvents" | "vkCmdWaitEvents2" => with(|later| (*notes(later)).lost_track = true),
            "vkCmdCopyImageToBuffer" | "vkCmdResolveImage" => touched(notes, &[value(1)]),
            "vkCmdCopyImage" | "vkCmdBlitImage" => touched(notes, &[value(1), value(3)]),
            "vkCmdClearColorImage" => touched(notes, &[value(1)]),
            "vkCmdCopyBufferToImage" => {
                let list = std::slice::from_raw_parts(*(args[5] as *const *const VkBufferImageCopy), *(args[4] as *const u32) as usize);
                copied_to(notes, value(2), list.iter().map(|copy| copy.imageSubresource.mipLevel));
            }
            "vkCmdCopyImageToBuffer2" => touched(notes, &[(**(args[1] as *const *const VkCopyImageToBufferInfo2)).srcImage]),
            "vkCmdResolveImage2" => touched(notes, &[(**(args[1] as *const *const VkResolveImageInfo2)).srcImage]),
            "vkCmdCopyImage2" => {
                let info = &**(args[1] as *const *const VkCopyImageInfo2);
                touched(notes, &[info.srcImage, info.dstImage]);
            }
            "vkCmdBlitImage2" => {
                let info = &**(args[1] as *const *const VkBlitImageInfo2);
                touched(notes, &[info.srcImage, info.dstImage]);
            }
            "vkCmdCopyBufferToImage2" => {
                let info = &**(args[1] as *const *const VkCopyBufferToImageInfo2);
                let list = std::slice::from_raw_parts(info.pRegions, info.regionCount as usize);
                copied_to(notes, info.dstImage, list.iter().map(|copy| copy.imageSubresource.mipLevel));
            }
            "vkDestroyImage" => image_destroyed(value(1)),
            "vkFreeMemory" => memory_freed(value(1)),
            "vkDestroyDevice" => device_destroyed(),
            _ => {}
        }
    }
}

fn touched(notes: impl Fn(&mut Later) -> *mut Notes, images: &[u64]) {
    with(|later| {
        if images.iter().any(|image| later.images.contains_key(image)) {
            let notes = unsafe { &mut *notes(later) };
            notes.touches.extend(images.iter().copied());
        }
    });
}

fn copied_to(notes: impl Fn(&mut Later) -> *mut Notes, image: u64, mips: impl Iterator<Item = u32>) {
    with(|later| {
        if later.images.contains_key(&image) {
            let notes = unsafe { &mut *notes(later) };
            notes.copies_to.extend(mips.map(|mip| (image, mip)));
        }
    });
}

fn layout_changed(notes: &mut Notes, image: u64, range: &VkImageSubresourceRange, layout: VkImageLayout, family_changes: bool) {
    notes.layouts.push((image, range.baseMipLevel, range.levelCount, layout));
    // Moving to another queue family: DXVK does this before it's used, on a texture it clears first.
    // Mips still to come were to be copied on the old one, so they'd have to be in place by then.
    if family_changes {
        notes.touches.insert(image);
    }
}

/// Follows the layouts in a submit's command buffers
fn apply_layouts(images: &mut HashMap<u64, Image>, notes: &Notes) {
    for &(image, first, count, layout) in &notes.layouts {
        if let Some(image) = images.get_mut(&image) {
            let end = if count == VK_REMAINING_MIP_LEVELS { image.layouts.len() } else { (first as usize + count as usize).min(image.layouts.len()) };
            if let Some(mips) = image.layouts.get_mut(first as usize..end) {
                mips.fill(layout);
            }
        }
    }
}

/// Before a submit: waits for the real mips of any image its command buffers copy from or write to
pub fn before_submit(commands: &[u64]) {
    let mut guard = later();
    let later = guard.get_or_insert_with(Later::default);
    let mut touched: Vec<u64> = commands.iter().filter_map(|commands| later.recording.get(commands)).flat_map(|notes| notes.touches.iter().copied()).collect();
    // Copying over a mip that's still to come
    touched.extend(
        commands
            .iter()
            .filter_map(|commands| later.recording.get(commands))
            .flat_map(|notes| notes.copies_to.iter())
            .filter(|(image, mip)| later.images.get(image).is_some_and(|image| image.mips & (1 << mip) != 0))
            .map(|(image, _)| *image),
    );
    let lost = commands.iter().filter_map(|commands| later.recording.get(commands)).any(|notes| notes.lost_track);
    let mut waiting_for: Vec<u64> = touched.into_iter().filter(|image| later.images.get(image).is_some_and(|image| image.waiting > 0)).collect();
    if lost {
        later.lost_track = true;
        waiting_for = later.images.iter().filter(|(_, image)| image.waiting > 0).map(|(&id, _)| id).collect();
    }
    if waiting_for.is_empty() {
        return;
    }
    later.urgent.extend(waiting_for.iter().copied());
    CHANGED.notify_all();
    drop(guard);
    wait_for(&waiting_for);
}

/// Waits until these images have no mips to come
fn wait_for(images: &[u64]) {
    let mut guard = later();
    loop {
        let later = guard.get_or_insert_with(Later::default);
        if !images.iter().any(|image| later.images.get(image).is_some_and(|image| image.waiting > 0)) {
            return;
        }
        guard = CHANGED.wait(guard).unwrap_or_else(|p| p.into_inner());
    }
}

/// A submit is going: call holding order()
pub fn submitted(commands: &[u64]) {
    with(|later| {
        let Later { images, recording, .. } = later;
        for commands in commands {
            if let Some(notes) = recording.get(commands) {
                apply_layouts(images, notes);
            }
        }
    });
}

/// The image is going away. What's on its way to it isn't wanted, and what's being put in place has to
/// finish first.
fn image_destroyed(image: u64) {
    let mut guard = later();
    let later = guard.get_or_insert_with(Later::default);
    let Some(info) = later.images.get_mut(&image) else { return };
    info.cancelled = true;
    let waiting = info.waiting > 0;
    if waiting {
        later.urgent.insert(image);
        CHANGED.notify_all();
    }
    drop(guard);
    if waiting {
        wait_for(&[image]);
    }
    with(|later| later.images.remove(&image));
}

fn device_destroyed() {
    let waiting: Vec<u64> = with(|later| {
        for image in later.images.values_mut() {
            image.cancelled = true;
        }
        later.urgent.extend(later.images.keys().copied());
        later.images.iter().filter(|(_, image)| image.waiting > 0).map(|(&id, _)| id).collect()
    });
    CHANGED.notify_all();
    wait_for(&waiting);
    with(|later| {
        later.images.clear();
        later.pools.clear();
    });
}

/// Which goes next: what a submit is waiting for; then what's decoded; then what's being coded, unless
/// the lender has its hands full; then the smallest. What the lender's decoding can't go on until it's
/// done, which this finds out.
fn next_level(levels: &mut [Level], urgent: &HashSet<u64>) -> Option<usize> {
    for level in levels.iter_mut() {
        if let Stage::Decoding(answer) = &level.stage
            && let Some(result) = answer.arrived()
        {
            level.stage = Stage::Decoded(matches!(result, Ok(0)));
        }
    }
    let decoding = levels.iter().filter(|level| matches!(level.stage, Stage::Decoding(_))).count();
    let rank = |level: &Level| match level.stage {
        Stage::Decoded(_) => 0,
        Stage::Coding(..) => 1,
        _ => 2,
    };
    levels
        .iter()
        .enumerate()
        .filter(|(_, level)| match level.stage {
            Stage::Decoding(_) => false,
            Stage::Coding(..) => decoding < DECODING_MOST || urgent.contains(&level.image),
            _ => true,
        })
        .min_by_key(|(_, level)| (!urgent.contains(&level.image), rank(level), level.bytes.len()))
        .map(|(index, _)| index)
}

/// How much goes before seeing how long it took: the least, the first, and the most
const WINDOW_LEAST: usize = memory::KEPT_BLOCK;
const WINDOW_FIRST: usize = 4 * memory::KEPT_BLOCK;
const WINDOW_MOST: usize = 256 * memory::KEPT_BLOCK;
/// Delay past the quickest seen that says the bytes are queuing somewhere along the way, in the game's
/// way, about a frame at 120 fps
const DELAY_MOST: Duration = Duration::from_millis(8);

/// What the next window should be, given how long the last took and the quickest seen. It grows while
/// nothing's queuing along the way and shrinks when something is, so the game's traffic stays ahead on
/// any link, however much it can hold.
fn next_window(window: usize, took: Duration, quickest: Duration) -> usize {
    if took > quickest + DELAY_MOST {
        (window - window / 4).max(WINDOW_LEAST)
    } else {
        (window + memory::KEPT_BLOCK).min(WINDOW_MOST)
    }
}

/// The thread: sends the smallest mip waiting, a block at a time, as smaller ones may turn up, and puts
/// each in place when it's all there
fn deliver(connection: &Arc<Connection>) {
    let (mut window, mut quickest, mut since_look) = (WINDOW_FIRST, Duration::MAX, 0usize);
    let mut getting_through = None::<f64>;
    let mut window_started = Instant::now();
    let (mut busy_since, mut delivered) = (None::<Instant>, 0usize);
    let (mut coding_since, mut coded_any) = (None::<Instant>, false);
    let report = crate::logging() || std::env::var_os("NECTAR_VK_STATS").is_some();
    loop {
        let mut level = {
            let mut guard = later();
            loop {
                let later = guard.get_or_insert_with(Later::default);
                if coding_since.is_some() && !later.levels.iter().any(|level| matches!(level.stage, Stage::Coding(..) | Stage::Decoding(_) | Stage::Decoded(_))) {
                    let since: Instant = coding_since.take().unwrap_or_else(Instant::now);
                    if report && coded_any {
                        crate::log(&format!("textures close: coded mips in place in {:.2} s", since.elapsed().as_secs_f64()));
                    }
                    coded_any = false;
                }
                let Later { levels, urgent, .. } = later;
                if let Some(index) = next_level(levels, urgent) {
                    break later.levels.swap_remove(index);
                }
                if later.levels.is_empty() {
                    if let Some(since) = busy_since.take() {
                        if report {
                            crate::log(&format!("textures sharp: {:.1} MB of mips came later, in {:.2} s", delivered as f64 / 1e6, since.elapsed().as_secs_f64()));
                        }
                        delivered = 0;
                    }
                    guard = CHANGED.wait(guard).unwrap_or_else(|p| p.into_inner());
                } else {
                    // Waiting for the lender to decode something
                    guard = CHANGED.wait_timeout(guard, Duration::from_millis(1)).unwrap_or_else(|p| p.into_inner()).0;
                }
            }
        };
        busy_since.get_or_insert_with(Instant::now);
        if matches!(level.stage, Stage::Coding(..)) {
            coding_since.get_or_insert_with(Instant::now);
        }
        coded_any |= matches!(level.stage, Stage::Coding(Some(_), _));
        let image = level.image;
        let urgent = with(|later| later.urgent.contains(&image));
        let decision = worth_coding(coding(), getting_through);
        if since_look == 0 {
            window_started = Instant::now();
        }
        let result = own_work(|| unsafe { send_next_block(connection, &mut level, urgent, decision) });
        // Only time spent sending says what's getting through
        if std::mem::take(&mut level.did_other) {
            since_look = 0;
            with(|later| later.levels.push(level));
            continue;
        }
        match result {
            Ok(false) => {
                since_look += memory::KEPT_BLOCK;
                with(|later| later.levels.push(level));
                // What a submit is waiting for doesn't stop to look
                if since_look >= window && !urgent {
                    let started = Instant::now();
                    if connection.ask(ECHO, &[]).is_err() {
                        continue;
                    }
                    let took = started.elapsed();
                    quickest = quickest.min(took);
                    window = next_window(window, took, quickest);
                    let rate = since_look as f64 / window_started.elapsed().as_secs_f64().max(1e-6);
                    getting_through = Some(getting_through.map_or(rate, |before| before * 0.75 + rate * 0.25));
                    since_look = 0;
                }
            }
            done => {
                delivered += level.bytes.len();
                if let Err(error) = done {
                    crate::log(&format!("a texture's mip {} stays coarse: {error}", level.mip));
                }
                with(|later| {
                    later.held -= level.bytes.len();
                    if let Some(info) = later.images.get_mut(&image) {
                        info.waiting -= 1;
                        info.mips &= !(1 << level.mip);
                        if info.waiting == 0 {
                            later.urgent.remove(&image);
                        }
                    }
                });
                CHANGED.notify_all();
            }
        }
    }
}

fn failed(what: &str, result: VkResult) -> String {
    format!("{what} failed with {result}")
}

/// Memory of its own on the lender for a mip's bytes to wait in
unsafe fn make_side(connection: &Arc<Connection>, device: VkDevice, level: &Level) -> Result<(VkBuffer, u64), String> {
    unsafe {
        let buffer_info = VkBufferCreateInfo {
            sType: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            size: level.bytes.len() as u64,
            usage: VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            sharingMode: 0,
            queueFamilyIndexCount: 0,
            pQueueFamilyIndices: ptr::null(),
        };
        let mut buffer = 0;
        let result = generated::vkCreateBuffer(device, &buffer_info, ptr::null(), &mut buffer);
        if result != VK_SUCCESS {
            return Err(failed("vkCreateBuffer", result));
        }
        let mut needs: VkMemoryRequirements = mem::zeroed();
        generated::vkGetBufferMemoryRequirements(device, buffer, &mut needs);
        let memory_info = VkMemoryAllocateInfo {
            sType: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            pNext: ptr::null(),
            allocationSize: needs.size,
            memoryTypeIndex: level.memory_type,
        };
        let mut memory = 0;
        let result = generated::vkAllocateMemory(device, &memory_info, ptr::null(), &mut memory);
        if result != VK_SUCCESS {
            generated::vkDestroyBuffer(device, buffer, ptr::null());
            return Err(failed("vkAllocateMemory", result));
        }
        generated::vkBindBufferMemory(device, buffer, memory, 0);
        let mut w = Writer::new();
        w.u64(memory).u8(1);
        let answer = connection.ask(MAP, w.as_slice()).map_err(|error| error.to_string())?;
        if answer.result != 0 {
            free_side(device, buffer, memory);
            return Err(failed("mapping", answer.result as i64 as i32));
        }
        Ok((buffer, memory))
    }
}

unsafe fn free_side(device: VkDevice, buffer: VkBuffer, memory: u64) {
    unsafe {
        generated::vkDestroyBuffer(device, buffer, ptr::null());
        generated::vkFreeMemory(device, memory, ptr::null());
    }
}

/// Does the next thing a mip needs; once its real bytes have all gone, puts it in place. True when it's
/// finished with.
unsafe fn send_next_block(connection: &Arc<Connection>, level: &mut Level, urgent: bool, decision: Option<bool>) -> Result<bool, String> {
    let Some((device_handle, cancelled)) = with(|later| later.images.get(&level.image).map(|image| (image.device, image.cancelled))) else {
        return Ok(true);
    };
    let device = device_handle as VkDevice;
    unsafe {
        if cancelled {
            if let Some((buffer, memory)) = level.side {
                unmap_side(connection, memory);
                free_side(device, buffer, memory);
            }
            return Ok(true);
        }
        let (buffer, memory) = match level.side {
            Some(side) => side,
            None => {
                let side = make_side(connection, device, level)?;
                level.side = Some(side);
                level.did_other = true;
                return Ok(false);
            }
        };
        // What a submit is waiting for needs the real thing, and nothing on the way to it
        if urgent && matches!(level.stage, Stage::Coding(..) | Stage::Decoded(_)) {
            level.stage = Stage::Real;
        }
        // On a link that's quick enough, it's not worth it
        if decision == Some(false) && matches!(level.stage, Stage::Coding(None, _)) {
            level.stage = Stage::Real;
        }
        // Until it's known how much gets through, real bytes find out, and they won't have to go again
        if decision.is_none() && matches!(level.stage, Stage::Coding(None, _)) && level.at < level.bytes.len() {
            let end = (level.at + memory::KEPT_BLOCK).min(level.bytes.len());
            memory::send_later_block(connection, memory, level.at, &level.bytes[level.at..end], level.place, level.at / memory::KEPT_BLOCK);
            level.at = end;
            return Ok(false);
        }
        match &mut level.stage {
            Stage::Coding(coded, at) => {
                let Some(bytes) = coded else {
                    let (image, threads) = (
                        with(|later| later.images.get(&level.image).cloned()).ok_or("the image has gone")?,
                        std::thread::available_parallelism().map_or(1, |count| (count.get() / 2).max(1)),
                    );
                    let (kind, srgb, across, down) = coded_size(&image, level.mip, level.bytes.len()).ok_or("the mip isn't what it was")?;
                    *coded = Some(wavelet::encode_mip(kind, srgb, &level.bytes, across, down, &wavelet::QUALITY, threads));
                    level.did_other = true;
                    return Ok(false);
                };
                let end = (*at + memory::KEPT_BLOCK).min(bytes.len());
                let mut w = Writer::new();
                w.u64(memory).bytes(&bytes[*at..end]);
                let _ = connection.send(CODED_PART, w.as_slice());
                *at = end;
                if *at == bytes.len() {
                    let image = with(|later| later.images.get(&level.image).cloned()).ok_or("the image has gone")?;
                    let (_, _, across, down) = coded_size(&image, level.mip, level.bytes.len()).ok_or("the mip isn't what it was")?;
                    let mut w = Writer::new();
                    w.u64(memory).u32(image.format as u32).u32(across as u32).u32(down as u32);
                    let answer = connection.ask_later(DECODE_MIP, w.as_slice()).map_err(|error| error.to_string())?;
                    level.stage = Stage::Decoding(answer);
                    level.did_other = true;
                }
                return Ok(false);
            }
            Stage::Decoded(decoded) => {
                let decoded = *decoded;
                level.stage = Stage::Real;
                // Decoding wrote over any real bytes that went first, to see what was getting through.
                // Going again costs only their hashes, as they're kept.
                level.at = 0;
                level.did_other = true;
                if decoded {
                    apply(device, device_handle, buffer, level)?;
                }
                return Ok(false);
            }
            // Doesn't come here: it's not chosen until it's answered
            Stage::Decoding(_) => return Ok(false),
            Stage::Real => {}
        }
        let block = memory::KEPT_BLOCK;
        if level.at < level.bytes.len() {
            let index = level.at / block;
            let end = (level.at + block).min(level.bytes.len());
            memory::send_later_block(connection, memory, level.at, &level.bytes[level.at..end], level.place, index);
            level.at = end;
            if level.at < level.bytes.len() {
                return Ok(false);
            }
        }
        unmap_side(connection, memory);
        let result = apply(device, device_handle, buffer, level);
        free_side(device, buffer, memory);
        result.map(|()| true)
    }
}

fn unmap_side(connection: &Connection, memory: u64) {
    let mut w = Writer::new();
    w.u64(memory);
    let _ = connection.send(UNMAP, w.as_slice());
}

/// Copies a mip into its image, and waits for that to finish
unsafe fn apply(device: VkDevice, device_handle: usize, buffer: VkBuffer, level: &Level) -> Result<(), String> {
    unsafe {
        // The layout it's left in is the one the game's next submit thinks it's in
        let (fence, commands, family) = {
            let _order = order();
            let family = with(|later| later.families.get(&level.queue).copied()).ok_or("its queue isn't known")?;
            let (known, width, height, layout) = with(|later| {
                let image = later.images.get(&level.image);
                let layout = image.map_or(VK_IMAGE_LAYOUT_UNDEFINED, |image| image.layouts[level.mip as usize]);
                (!later.lost_track && image.is_some_and(|image| !image.cancelled), image.map_or(0, |image| image.width), image.map_or(0, |image| image.height), layout)
            });
            if !known {
                return Ok(());
            }
            if layout == VK_IMAGE_LAYOUT_UNDEFINED {
                return Err("its layout isn't known".into());
            }
            let commands = record_copy(device, device_handle, family, buffer, level, (width >> level.mip).max(1), (height >> level.mip).max(1), layout)?;
            let fence_info = VkFenceCreateInfo { sType: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, pNext: ptr::null(), flags: 0 };
            let mut fence = 0;
            let result = generated::vkCreateFence(device, &fence_info, ptr::null(), &mut fence);
            if result != VK_SUCCESS {
                return Err(failed("vkCreateFence", result));
            }
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
            let result = generated::vkQueueSubmit(level.queue as VkQueue, 1, &submit, fence);
            if result != VK_SUCCESS {
                generated::vkDestroyFence(device, fence, ptr::null());
                return Err(failed("vkQueueSubmit", result));
            }
            (fence, commands, family)
        };
        generated::vkWaitForFences(device, 1, &fence, 1, u64::MAX);
        generated::vkDestroyFence(device, fence, ptr::null());
        if let Some(pool) = with(|later| later.pools.get(&(device_handle, family)).copied()) {
            generated::vkFreeCommandBuffers(device, pool, 1, &commands);
        }
    }
    Ok(())
}

/// Records: the mip's old contents dropped, the copy, then the mip left in `layout`
#[allow(clippy::too_many_arguments)]
unsafe fn record_copy(device: VkDevice, device_handle: usize, family: u32, buffer: VkBuffer, level: &Level, width: u32, height: u32, layout: VkImageLayout) -> Result<VkCommandBuffer, String> {
    unsafe {
        let pool = match with(|later| later.pools.get(&(device_handle, family)).copied()) {
            Some(pool) => pool,
            None => {
                let info = VkCommandPoolCreateInfo {
                    sType: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                    pNext: ptr::null(),
                    flags: 0,
                    queueFamilyIndex: family,
                };
                let mut pool = 0;
                let result = generated::vkCreateCommandPool(device, &info, ptr::null(), &mut pool);
                if result != VK_SUCCESS {
                    return Err(failed("vkCreateCommandPool", result));
                }
                with(|later| later.pools.insert((device_handle, family), pool));
                pool
            }
        };
        let allocate = VkCommandBufferAllocateInfo {
            sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            pNext: ptr::null(),
            commandPool: pool,
            level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount: 1,
        };
        let mut commands: VkCommandBuffer = mem::zeroed();
        let result = generated::vkAllocateCommandBuffers(device, &allocate, &mut commands);
        if result != VK_SUCCESS {
            return Err(failed("vkAllocateCommandBuffers", result));
        }
        let begin = VkCommandBufferBeginInfo {
            sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext: ptr::null(),
            flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            pInheritanceInfo: ptr::null(),
        };
        let result = generated::vkBeginCommandBuffer(commands, &begin);
        if result != VK_SUCCESS {
            return Err(failed("vkBeginCommandBuffer", result));
        }
        let range = VkImageSubresourceRange { aspectMask: VK_IMAGE_ASPECT_COLOR_BIT, baseMipLevel: level.mip, levelCount: 1, baseArrayLayer: 0, layerCount: 1 };
        let barrier = |from, to, source, target| VkImageMemoryBarrier {
            sType: VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            pNext: ptr::null(),
            srcAccessMask: source,
            dstAccessMask: target,
            oldLayout: from,
            newLayout: to,
            srcQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
            image: level.image,
            subresourceRange: range,
        };
        let all = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        let before = barrier(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        generated::vkCmdPipelineBarrier(commands, all, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, ptr::null(), 0, ptr::null(), 1, &before);
        let copy = VkBufferImageCopy {
            bufferOffset: 0,
            bufferRowLength: 0,
            bufferImageHeight: 0,
            imageSubresource: VkImageSubresourceLayers { aspectMask: VK_IMAGE_ASPECT_COLOR_BIT, mipLevel: level.mip, baseArrayLayer: 0, layerCount: 1 },
            imageOffset: VkOffset3D { x: 0, y: 0, z: 0 },
            imageExtent: VkExtent3D { width, height, depth: 1 },
        };
        generated::vkCmdCopyBufferToImage(commands, buffer, level.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        let after = barrier(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, layout, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        generated::vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, all, 0, 0, ptr::null(), 0, ptr::null(), 1, &after);
        let result = generated::vkEndCommandBuffer(commands);
        if result != VK_SUCCESS {
            return Err(failed("vkEndCommandBuffer", result));
        }
        Ok(commands)
    }
}

/// Of the mips of one image copied in one submit, smallest first, which should wait. `sizes` holds each
/// mip's number and how much of it has to cross. `spent` is what this submit has sent already.
pub fn choose(sizes: &[(u32, usize)], spent: &mut usize, may_wait: &mut dyn FnMut(u32) -> bool) -> Vec<u32> {
    let mut waiting = Vec::new();
    let total: usize = sizes.iter().map(|(_, size)| size).sum();
    if total < LATER_LEAST {
        *spent += total;
        return waiting;
    }
    for &(mip, size) in sizes {
        if size > 0 && *spent + size > FIRST_BUDGET && may_wait(mip) {
            waiting.push(mip);
            *spent += SAMPLES_MOST;
        } else {
            *spent += size;
        }
    }
    waiting
}

/// How many blocks each sampled block stands for, so that the samples are at most SAMPLES_MOST: a power
/// of 2, as what's sampled is then in step with the mips
pub fn sample_scale(across: usize, down: usize, block: usize) -> usize {
    let mut scale = 1;
    while across.div_ceil(scale) * down.div_ceil(scale) * block > SAMPLES_MOST {
        scale *= 2;
    }
    scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_mips_wait_and_small_ones_go() {
        // A chain of 1024x1024 BC7, smallest mips first
        let chain: Vec<(u32, usize)> = vec![(8, 16), (7, 64), (6, 256), (5, 1024), (4, 4096), (3, 16 << 10), (2, 64 << 10), (1, 256 << 10), (0, 1 << 20)];
        let mut spent = 0;
        assert_eq!(choose(&chain, &mut spent, &mut |_| true), vec![1, 0]);
        assert!(spent <= FIRST_BUDGET + 2 * SAMPLES_MOST);
        // What's kept already has nothing to send, so needn't wait
        let mut kept = chain.clone();
        kept[8].1 = 0;
        kept[7].1 = 0;
        kept[6].1 = LATER_LEAST;
        let mut spent = 0;
        assert_eq!(choose(&kept, &mut spent, &mut |_| true), vec![2]);
        // A small chain goes as it is
        let mut spent = 0;
        assert_eq!(choose(&chain[..7], &mut spent, &mut |_| true), Vec::<u32>::new());
        // Once it can't wait, it doesn't
        let mut spent = 0;
        assert_eq!(choose(&chain, &mut spent, &mut |mip| mip != 0), vec![1]);
        // A mip alone in its submit can wait too, as DXVK sends large ones first
        let mut spent = 0;
        assert_eq!(choose(&[(0, 1 << 20)], &mut spent, &mut |_| true), vec![0]);
        // Budget shared with what the submit has already sent
        let mut spent = FIRST_BUDGET;
        assert_eq!(choose(&chain[3..], &mut spent, &mut |_| true), vec![5, 4, 3, 2, 1, 0]);
    }

    #[test]
    fn only_the_first_bytes_for_a_mip_wait() {
        let image = 77001u64;
        let info = VkImageCreateInfo {
            sType: VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            imageType: IMAGE_TYPE_2D,
            format: VK_FORMAT_BC7_UNORM_BLOCK,
            extent: VkExtent3D { width: 64, height: 64, depth: 1 },
            mipLevels: 3,
            arrayLayers: 1,
            samples: 1,
            tiling: 0,
            usage: VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            sharingMode: 0,
            queueFamilyIndexCount: 0,
            pQueueFamilyIndices: ptr::null(),
            initialLayout: 0,
        };
        image_created(image, 1, &info);
        queue_found(77002, 0);
        memory_allocated(77003, 0);
        assert!(may_wait(image, 0, 77002, 77003, 100, 0));
        assert!(!may_wait(image, 0, 77002, 77003, 100, 0));
        assert!(may_wait(image, 1, 77002, 77003, 100, 0));
        // Not an image that qualifies, or a queue or memory that isn't known, or too much held
        assert!(!may_wait(77004, 0, 77002, 77003, 100, 0));
        assert!(!may_wait(image, 2, 77005, 77003, 100, 0));
        let another = 77006u64;
        image_created(another, 1, &info);
        assert!(!may_wait(another, 0, 77002, 77003, HELD_MOST + 1, 0));
        // With a layer more, or if it can be drawn to, it doesn't qualify
        let (layered, drawn) = (77007u64, 77008u64);
        image_created(layered, 1, &VkImageCreateInfo { arrayLayers: 2, ..info });
        image_created(drawn, 1, &VkImageCreateInfo { usage: VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, ..info });
        assert!(!may_wait(layered, 0, 77002, 77003, 100, 0));
        assert!(!may_wait(drawn, 0, 77002, 77003, 100, 0));
        // Only a whole mip, tightly packed, counts
        let (zero, whole) = (VkOffset3D { x: 0, y: 0, z: 0 }, VkExtent3D { width: 32, height: 32, depth: 1 });
        assert_eq!(whole_mip(image, 1, 0, 0, zero, whole), Some(1));
        assert_eq!(whole_mip(image, 1, 32, 32, zero, whole), Some(1));
        assert_eq!(whole_mip(image, 1, 64, 0, zero, whole), None);
        assert_eq!(whole_mip(image, 0, 0, 0, zero, whole), None);
        assert_eq!(whole_mip(image, 1, 0, 0, VkOffset3D { x: 4, y: 0, z: 0 }, whole), None);
    }

    #[test]
    fn samples_are_kept_small() {
        assert_eq!(sample_scale(16, 16, 16), 1);
        // 4096x4096 texels of BC7 is 268 MB at the top of a 16384 texture
        let scale = sample_scale(4096, 4096, 16);
        assert_eq!(scale, 64);
        assert!(4096usize.div_ceil(scale).pow(2) * 16 <= SAMPLES_MOST);
    }

    #[test]
    fn only_a_slow_link_is_worth_coding_for() {
        assert_eq!(worth_coding(None, None), None);
        assert_eq!(worth_coding(None, Some(6e6)), Some(true));
        assert_eq!(worth_coding(None, Some(37e6)), Some(false));
        // It can be forced either way, whatever's getting through
        assert_eq!(worth_coding(Some(true), Some(37e6)), Some(true));
        assert_eq!(worth_coding(Some(false), None), Some(false));
        assert_eq!(worth_coding(Some(true), None), Some(true));
    }

    #[test]
    fn the_window_grows_until_something_queues() {
        let quickest = Duration::from_millis(9);
        assert_eq!(next_window(WINDOW_FIRST, quickest, quickest), WINDOW_FIRST + memory::KEPT_BLOCK);
        assert_eq!(next_window(WINDOW_FIRST, quickest + DELAY_MOST + Duration::from_millis(1), quickest), WINDOW_FIRST - WINDOW_FIRST / 4);
        assert_eq!(next_window(WINDOW_LEAST, Duration::from_secs(1), quickest), WINDOW_LEAST);
        assert_eq!(next_window(WINDOW_MOST, quickest, quickest), WINDOW_MOST);
    }

    #[test]
    fn layouts_follow_the_last_barrier_in_order() {
        let mut images = HashMap::new();
        let image = Image { device: 0, format: VK_FORMAT_BC7_UNORM_BLOCK, width: 8, height: 8, layouts: vec![0; 4], waiting: 0, mips: 0, sent: 0, cancelled: false };
        images.insert(5u64, image);
        let notes = Notes {
            layouts: vec![(5, 0, VK_REMAINING_MIP_LEVELS, 7), (5, 1, 2, 5), (9, 0, 1, 5), (5, 3, 9, 2)],
            ..Notes::default()
        };
        apply_layouts(&mut images, &notes);
        assert_eq!(images[&5].layouts, vec![7, 5, 5, 2]);
    }

    #[test]
    fn the_next_to_go_is_what_is_waited_for_then_the_smallest() {
        let level = |image: u64, size: usize, stage: Stage| Level { image, mip: 0, queue: 0, memory_type: 0, bytes: vec![0; size], place: (0, 0), side: None, at: 0, stage, did_other: false };
        let mut levels = [level(1, 300, Stage::Real), level(2, 200, Stage::Real), level(3, 400, Stage::Real)];
        assert_eq!(next_level(&mut levels, &HashSet::new()), Some(1));
        assert_eq!(next_level(&mut levels, &HashSet::from([3])), Some(2));
        assert_eq!(next_level(&mut [], &HashSet::new()), None);
        // What's been decoded, then what's being coded, then the real bytes, whatever their size
        let mut levels = [level(1, 100, Stage::Real), level(2, 500, Stage::Coding(None, 0)), level(3, 900, Stage::Decoded(true))];
        assert_eq!(next_level(&mut levels, &HashSet::new()), Some(2));
        let mut levels = [level(1, 100, Stage::Real), level(2, 500, Stage::Coding(None, 0))];
        assert_eq!(next_level(&mut levels, &HashSet::new()), Some(1));
    }
}
