//! Frames on screen: VK_KHR_surface and VK_KHR_swapchain, in this computer's own window
//! (window.rs says which kinds of window).
//!
//! The program's swapchain images are ordinary images on the lender, which it renders into as
//! usual. Presenting one copies it to a buffer the lender's CPU can read, and asks for the frame in
//! the same breath: the lender waits for the copy itself and has the frame coded, so a frame costs
//! one trip across the link, not one for the copy and one for the frame. A thread here draws the
//! answers in the window as they come.
//!
//! The coding is done on the GPU (GpuCoder, nectar-vk-wire's frame_shader.rs), which leaves the
//! lender's CPU only gathering the result; where that can't be done, or for a lender with no real GPU,
//! or with NECTAR_VK_FRAME_CODER=cpu, the frame is copied where the lender's CPU can read it and coded
//! there by frame.rs. Both write the same format.
//!
//! A few frames are on their way at a time, so that the time one takes there doesn't limit the frame
//! rate: with 2, a frame that takes 50 ms to be asked for, rendered, coded, carried and drawn held The
//! Witcher 3 to 40 fps, whatever else was quick. How many follows the program: while it has to wait in
//! present for one to finish, one more is tried, up to 4, and kept if its frames then come faster. If
//! they don't, something on the way can't keep up, and more at once would only queue there: it goes
//! back, and the program waits in present. NECTAR_VK_FRAMES fixes how many (1 to 4).
//!
//! The quality follows what it can change: how long a frame takes to code, and how long its bytes
//! take to cross. The time for a trip across is the same at any quality, and is left out. What that
//! may take is measured against the time between the program's frames, not a frame rate it isn't
//! reaching: a game at 35 fps can spend more on each frame than one at 120. Frames cross one after
//! another, so as long as one takes less than the time between them, they aren't what holds the program
//! back, and if they were, that time would be theirs and the quality would come down.
//! NECTAR_VK_FRAME_QUALITY fixes it.
//!
//! All of that is ordinary Vulkan on the lender, so it works on any lender.
//!
//! Where the lender's card has a video encoder and this computer a decoder for it, frames go as H.264
//! instead (VideoCoder): a shader here makes the frame's NV12 (video_shader.rs), the lender's encoder
//! makes a picture of it (video_encode.rs) coded against the one before, and the decoder Windows comes
//! with turns it back (video_decode.rs). A frame is then a few tens of KB where the other coder's is
//! hundreds. NECTAR_VK_FRAME_CODER=tiles keeps the other coder, and =cpu its CPU form.
//!

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use std::{mem, ptr};

use nectar_cuda_protocol::Writer;
use nectar_vk_sys::*;
use nectar_vk_wire::frame::{self, LOWEST_QUALITY};
use nectar_vk_wire::frame_shader;
use nectar_vk_wire::messages::{FRAME, GPU_USAGE, VIDEO_CLOSE, VIDEO_FRAME, VIDEO_OPEN};
use nectar_vk_wire::video_decode::Decoder;
use nectar_vk_wire::video_shader;

use crate::connection::Later;
use crate::window::{Show, Window, window_of};
use crate::{connection, generated, logging, memory, objects};

const WAIT_NOTHING: u64 = 0;
const COLOR_ASPECT: VkImageAspectFlags = VK_IMAGE_ASPECT_COLOR_BIT;
/// A frame's time at 120 frames a second: frames coming faster than that aren't given up quality for
const FASTEST_FRAME: Duration = Duration::from_micros(8333);
/// Frames on their way at a time to begin with, and the fewest there are room for, unless
/// NECTAR_VK_FRAMES says otherwise
const FRAMES_ON_THE_WAY: usize = 2;
/// The most there's room for
const MOST_FRAMES_ON_THE_WAY: usize = 4;
const FIRST_QUALITY: u32 = 75;
const BEST_QUALITY: u32 = 90;
/// The most swapchain images the surface offers
const MAX_IMAGES: usize = 8;

/// The largest frames the decoder Windows comes with takes
const LARGEST_VIDEO: (u32, u32) = (4096, 2304);

/// What present needs to know about a device
#[derive(Clone)]
struct Device {
    physical: VkPhysicalDevice,
    /// A queue family the program asked for, for signalling acquires
    family: u32,
    /// All it asked for
    families: Vec<u32>,
}

// A handle is only a number here
unsafe impl Send for Device {}

/// By device, and each queue's family by queue
#[derive(Default)]
struct Known {
    devices: HashMap<usize, Device>,
    families: HashMap<usize, u32>,
    swapchains: HashMap<u64, Arc<Swapchain>>,
}

static KNOWN: Mutex<Option<Known>> = Mutex::new(None);

fn known<R>(f: impl FnOnce(&mut Known) -> R) -> R {
    f(KNOWN.lock().unwrap_or_else(|p| p.into_inner()).get_or_insert_with(Known::default))
}

/// A device was created: the first queue family it asked for
pub(crate) unsafe fn device_created(device: VkDevice, physical: VkPhysicalDevice, info: *const VkDeviceCreateInfo) {
    let families: Vec<u32> = unsafe {
        let info = &*info;
        (0..info.queueCreateInfoCount as usize).map(|index| (*info.pQueueCreateInfos.add(index)).queueFamilyIndex).collect()
    };
    let family = families.first().copied().unwrap_or(0);
    known(|known| known.devices.insert(device as usize, Device { physical, family, families }));
}

pub(crate) fn queue_found(queue: VkQueue, family: u32) {
    known(|known| known.families.insert(queue as usize, family));
}

// ---- Surfaces ----

pub unsafe extern "system" fn vkGetPhysicalDeviceSurfaceSupportKHR(
    _physical: VkPhysicalDevice,
    _family: u32,
    surface: VkSurfaceKHR,
    pSupported: *mut VkBool32,
) -> VkResult {
    unsafe { *pSupported = window_of(surface).is_some() as VkBool32 };
    VK_SUCCESS
}

pub unsafe extern "system" fn vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
    _physical: VkPhysicalDevice,
    surface: VkSurfaceKHR,
    pCapabilities: *mut VkSurfaceCapabilitiesKHR,
) -> VkResult {
    let Some(window) = (unsafe { window_of(surface) }) else { return VK_ERROR_SURFACE_LOST_KHR };
    let (width, height) = window.size();
    let capabilities = VkSurfaceCapabilitiesKHR {
        minImageCount: 2,
        maxImageCount: MAX_IMAGES as u32,
        currentExtent: VkExtent2D { width, height },
        minImageExtent: VkExtent2D { width: 1, height: 1 },
        maxImageExtent: VkExtent2D { width: 16384, height: 16384 },
        maxImageArrayLayers: 1,
        supportedTransforms: VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
        currentTransform: VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
        supportedCompositeAlpha: VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        supportedUsageFlags: VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
            | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
            | VK_IMAGE_USAGE_TRANSFER_DST_BIT
            | VK_IMAGE_USAGE_SAMPLED_BIT
            | VK_IMAGE_USAGE_STORAGE_BIT,
    };
    unsafe { *pCapabilities = capabilities };
    VK_SUCCESS
}

pub unsafe extern "system" fn vkGetPhysicalDeviceSurfaceFormatsKHR(
    _physical: VkPhysicalDevice,
    _surface: VkSurfaceKHR,
    pCount: *mut u32,
    pFormats: *mut VkSurfaceFormatKHR,
) -> VkResult {
    // In the byte order a Windows bitmap and a 32-bit X image have
    let formats = [VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_SRGB]
        .map(|format| VkSurfaceFormatKHR { format, colorSpace: VK_COLOR_SPACE_SRGB_NONLINEAR_KHR });
    unsafe { crate::fill(&formats, pCount, pFormats) }
}

pub unsafe extern "system" fn vkGetPhysicalDeviceSurfacePresentModesKHR(
    _physical: VkPhysicalDevice,
    _surface: VkSurfaceKHR,
    pCount: *mut u32,
    pModes: *mut VkPresentModeKHR,
) -> VkResult {
    unsafe { crate::fill(&[VK_PRESENT_MODE_FIFO_KHR, VK_PRESENT_MODE_MAILBOX_KHR, VK_PRESENT_MODE_IMMEDIATE_KHR], pCount, pModes) }
}

pub unsafe extern "system" fn vkGetPhysicalDevicePresentRectanglesKHR(
    _physical: VkPhysicalDevice,
    surface: VkSurfaceKHR,
    pCount: *mut u32,
    pRects: *mut VkRect2D,
) -> VkResult {
    let Some(window) = (unsafe { window_of(surface) }) else { return VK_ERROR_SURFACE_LOST_KHR };
    let (width, height) = window.size();
    let rect = VkRect2D { offset: VkOffset2D { x: 0, y: 0 }, extent: VkExtent2D { width, height } };
    unsafe { crate::fill(&[rect], pCount, pRects) }
}

pub unsafe extern "system" fn vkGetDeviceGroupPresentCapabilitiesKHR(
    _device: VkDevice,
    pCapabilities: *mut VkDeviceGroupPresentCapabilitiesKHR,
) -> VkResult {
    unsafe {
        let capabilities = &mut *pCapabilities;
        capabilities.presentMask = [0; 32];
        capabilities.presentMask[0] = 1;
        capabilities.modes = VK_DEVICE_GROUP_PRESENT_MODE_LOCAL_BIT_KHR;
    }
    VK_SUCCESS
}

pub unsafe extern "system" fn vkGetDeviceGroupSurfacePresentModesKHR(
    _device: VkDevice,
    _surface: VkSurfaceKHR,
    pModes: *mut VkDeviceGroupPresentModeFlagsKHR,
) -> VkResult {
    unsafe { *pModes = VK_DEVICE_GROUP_PRESENT_MODE_LOCAL_BIT_KHR };
    VK_SUCCESS
}

// ---- Swapchains ----

struct Job {
    image: usize,
    slot: usize,
    answer: Later,
    asked: Instant,
    /// How long the program waited in present for a frame before this to finish
    waited: Duration,
}

/// Where one frame on its way is copied to
struct Slot {
    buffer: VkBuffer,
    memory: VkDeviceMemory,
    fence: VkFence,
}

/// Which images the program holds or are on their way, and how many frames are
#[derive(Default)]
struct State {
    held: Vec<bool>,
    next: usize,
    /// Frames being copied, coded, carried or drawn
    on_the_way: usize,
    /// How many of those there may be for now
    limit: usize,
    next_slot: usize,
    frames: u64,
}

/// Coding frames on the GPU: the frame and the one before it in memory the shader reads quickly, by
/// turns, since each frame is copied over the one before last. The queue keeps them in order.
struct GpuCoder {
    frames: [(VkBuffer, VkDeviceMemory); 2],
    work: (VkBuffer, VkDeviceMemory),
    hot: (VkBuffer, VkDeviceMemory),
    set_layout: VkDescriptorSetLayout,
    layout: VkPipelineLayout,
    pipelines: [VkPipeline; 3],
    descriptor_pool: VkDescriptorPool,
    /// For each slot, reading frame 0 then frame 1 as this one
    sets: Vec<[VkDescriptorSet; 2]>,
}

/// Frames as video: each is copied where the shader reads it, which writes its NV12 into the frame's slot
/// for the lender's encoder. The queue keeps the frames in order.
struct VideoCoder {
    /// Names the stream to the lender
    key: u64,
    frame: (VkBuffer, VkDeviceMemory),
    set_layout: VkDescriptorSetLayout,
    layout: VkPipelineLayout,
    pipeline: VkPipeline,
    descriptor_pool: VkDescriptorPool,
    /// For each slot
    sets: Vec<VkDescriptorSet>,
    /// The next frame has to be one that needs none before it
    need_key: AtomicBool,
}

/// What the drawing thread decodes video with. The decoder belongs to that thread.
struct Showing {
    decoder: Option<Decoder>,
    /// Pictures are no use until one comes that needs none before it
    waiting_for_key: bool,
    complaints: u32,
    /// The least a picture has taken to decode lately, in seconds
    floor: f64,
}

impl Showing {
    /// How much longer a picture took to decode than the quickest have lately, which is the part its
    /// quality can change: a finer picture has more bits to read. The rest is what a picture of this size
    /// takes however coarse, or a busy computer, and bringing the quality down for those gains nothing:
    /// flat 2560x1440 frames took 11 ms each at any quality, and counting all of it took them to the
    /// coarsest there is. The floor creeps up by a hundredth a picture, so it follows a computer getting
    /// busier, and comes down at once.
    fn over_floor(&mut self, took: Duration) -> Duration {
        let took = took.as_secs_f64();
        self.floor = if self.floor == 0.0 { took } else { (self.floor * 1.01).min(took) };
        Duration::from_secs_f64(took - self.floor)
    }

    /// Decodes a picture into `shown`. Returns whether there's a new picture to show.
    fn picture(&mut self, video: &VideoCoder, key: bool, coded: &[u8], shown: &mut [u8], width: u32, height: u32) -> bool {
        if self.waiting_for_key && !key {
            return false;
        }
        let outcome = match &mut self.decoder {
            Some(decoder) => decoder.decode(coded, shown),
            None => Decoder::new(width, height).and_then(|mut decoder| {
                let outcome = decoder.decode(coded, shown);
                self.decoder = Some(decoder);
                outcome
            }),
        };
        match outcome {
            Ok(picture) => {
                self.waiting_for_key = false;
                picture
            }
            Err(error) => {
                // A new decoder, and a picture it can start from
                self.decoder = None;
                self.waiting_for_key = true;
                video.need_key.store(true, Ordering::Relaxed);
                if self.complaints < 5 {
                    self.complaints += 1;
                    crate::log(&format!("a frame's picture couldn't be decoded: {error}"));
                }
                false
            }
        }
    }
}

struct Swapchain {
    device: VkDevice,
    window: Window,
    width: u32,
    height: u32,
    images: Vec<VkImage>,
    image_memory: Vec<VkDeviceMemory>,
    slots: Vec<Slot>,
    gpu: Option<GpuCoder>,
    video: Option<VideoCoder>,
    /// For the next frame
    quality: AtomicU32,
    /// A queue for signalling acquires
    queue: VkQueue,
    /// The command pool and the copies of each image to each slot, by queue family
    copies: Mutex<HashMap<u32, (VkCommandPool, Vec<VkCommandBuffer>)>>,
    state: Mutex<State>,
    changed: Condvar,
    jobs: Mutex<Option<Sender<Job>>>,
    thread: Mutex<Option<JoinHandle<()>>>,
}

// The handles are only numbers here; Vulkan takes them from any thread
unsafe impl Send for Swapchain {}
unsafe impl Sync for Swapchain {}

/// A memory type among `allowed` with `wanted` set, or failing that with `needed` set
unsafe fn memory_type(physical: VkPhysicalDevice, allowed: u32, wanted: VkMemoryPropertyFlags, needed: VkMemoryPropertyFlags) -> Option<u32> {
    let mut properties: VkPhysicalDeviceMemoryProperties = unsafe { mem::zeroed() };
    unsafe { generated::vkGetPhysicalDeviceMemoryProperties(physical, &mut properties) };
    let find = |flags: VkMemoryPropertyFlags| {
        (0..properties.memoryTypeCount).find(|&index| {
            allowed & (1 << index) != 0 && properties.memoryTypes[index as usize].propertyFlags & flags == flags
        })
    };
    find(wanted).or_else(|| find(needed))
}

unsafe fn allocate(device: VkDevice, size: VkDeviceSize, memoryTypeIndex: u32) -> Result<VkDeviceMemory, VkResult> {
    let info = VkMemoryAllocateInfo {
        sType: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        pNext: ptr::null(),
        allocationSize: size,
        memoryTypeIndex,
    };
    let mut memory = 0;
    match unsafe { generated::vkAllocateMemory(device, &info, ptr::null(), &mut memory) } {
        VK_SUCCESS => Ok(memory),
        error => Err(error),
    }
}

fn memory_barrier(source: VkAccessFlags, target: VkAccessFlags) -> VkMemoryBarrier {
    VkMemoryBarrier { sType: VK_STRUCTURE_TYPE_MEMORY_BARRIER, pNext: ptr::null(), srcAccessMask: source, dstAccessMask: target }
}

/// Asks the lender for frames of this size as video. Returns the stream's key and the queue family that
/// encodes, if both ends can: the lender's card has an encoder, and this computer a decoder.
fn open_video(connection: &connection::Connection, device: VkDevice, width: u32, height: u32) -> Option<(u64, u32)> {
    if std::env::var("NECTAR_VK_FRAME_CODER").is_ok_and(|coder| coder == "cpu" || coder == "tiles") {
        return None;
    }
    // Which coder frames get is worth a line wherever anything is being said
    let say = |what: &str| {
        if logging() || std::env::var_os("NECTAR_VK_STATS").is_some() {
            crate::log(&format!("frames of {width}x{height} {what}"));
        }
    };
    if width > LARGEST_VIDEO.0 || height > LARGEST_VIDEO.1 {
        say("are larger than the video decoder takes, so they come as tiles");
        return None;
    }
    // On a thread that ends, as making a decoder sets up things for its thread that aren't the program's to find
    match std::thread::spawn(move || Decoder::new(width, height).map(|_| ())).join() {
        Ok(Ok(())) => {}
        Ok(Err(error)) => {
            say(&format!("come as tiles, as there's no video decoder here: {error}"));
            return None;
        }
        Err(_) => {
            say("come as tiles, as trying the video decoder failed");
            return None;
        }
    }
    let key = objects::new_id();
    let mut w = Writer::new();
    w.u64(key).u64(unsafe { objects::id_of(device as u64) }).u32(width).u32(height);
    let family = match connection.ask(VIDEO_OPEN, w.as_slice()) {
        Ok(answer) if answer.result == 0 => answer.outputs().u32().ok(),
        Ok(answer) => {
            say(&format!("come as tiles, as the lender has no video encoder for them ({})", answer.result as i64 as i32));
            return None;
        }
        Err(error) => {
            say(&format!("come as tiles, as the lender couldn't be asked for video: {error}"));
            return None;
        }
    };
    say("come as H.264 from the lender's encoder");
    Some((key, family?))
}

/// Sets up the shader that makes NV12; `outputs` are the slots' buffers
unsafe fn create_video_coder(device: VkDevice, physical: VkPhysicalDevice, key: u64, width: u32, height: u32, outputs: &[VkBuffer]) -> Result<VideoCoder, VkResult> {
    let frame_size = width as u64 * height as u64 * 4;
    let (coded_width, coded_height) = video_shader::coded_size(width, height);
    let nv12_size = video_shader::nv12_size(coded_width, coded_height);
    macro_rules! made {
        ($call:expr) => {
            match $call {
                VK_SUCCESS => {}
                error => return Err(error),
            }
        };
    }
    unsafe {
        let info = VkBufferCreateInfo {
            sType: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            size: frame_size,
            usage: VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            sharingMode: VK_SHARING_MODE_EXCLUSIVE,
            queueFamilyIndexCount: 0,
            pQueueFamilyIndices: ptr::null(),
        };
        let mut frame = (0, 0);
        made!(generated::vkCreateBuffer(device, &info, ptr::null(), &mut frame.0));
        let mut needs: VkMemoryRequirements = mem::zeroed();
        generated::vkGetBufferMemoryRequirements(device, frame.0, &mut needs);
        let kind = memory_type(physical, needs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0).ok_or(VK_ERROR_OUT_OF_DEVICE_MEMORY)?;
        frame.1 = allocate(device, needs.size, kind)?;
        generated::vkBindBufferMemory(device, frame.0, frame.1, 0);

        let bindings: Vec<VkDescriptorSetLayoutBinding> = (0..2)
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
            bindingCount: 2,
            pBindings: bindings.as_ptr(),
        };
        let mut set_layout = 0;
        made!(generated::vkCreateDescriptorSetLayout(device, &set_info, ptr::null(), &mut set_layout));
        let range = VkPushConstantRange { stageFlags: VK_SHADER_STAGE_COMPUTE_BIT, offset: 0, size: video_shader::PUSH_BYTES };
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
        made!(generated::vkCreatePipelineLayout(device, &layout_info, ptr::null(), &mut layout));
        let code = video_shader::shader();
        let module_info = VkShaderModuleCreateInfo {
            sType: VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            codeSize: code.len() * 4,
            pCode: code.as_ptr(),
        };
        let mut module = 0;
        made!(crate::own_work(|| generated::vkCreateShaderModule(device, &module_info, ptr::null(), &mut module)));
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
        made!(crate::own_work(|| generated::vkCreateComputePipelines(device, 0, 1, &pipeline_info, ptr::null(), &mut pipeline)));
        generated::vkDestroyShaderModule(device, module, ptr::null());

        let set_count = outputs.len() as u32;
        let pool_size = VkDescriptorPoolSize { r#type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount: set_count * 2 };
        let pool_info = VkDescriptorPoolCreateInfo {
            sType: VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            maxSets: set_count,
            poolSizeCount: 1,
            pPoolSizes: &pool_size,
        };
        let mut descriptor_pool = 0;
        made!(generated::vkCreateDescriptorPool(device, &pool_info, ptr::null(), &mut descriptor_pool));
        let mut sets = Vec::with_capacity(outputs.len());
        for &output in outputs {
            let allocate_info = VkDescriptorSetAllocateInfo {
                sType: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                pNext: ptr::null(),
                descriptorPool: descriptor_pool,
                descriptorSetCount: 1,
                pSetLayouts: &set_layout,
            };
            let mut set = 0;
            made!(generated::vkAllocateDescriptorSets(device, &allocate_info, &mut set));
            let infos = [VkDescriptorBufferInfo { buffer: frame.0, offset: 0, range: frame_size }, VkDescriptorBufferInfo { buffer: output, offset: 0, range: nv12_size }];
            let writes: Vec<VkWriteDescriptorSet> = infos
                .iter()
                .enumerate()
                .map(|(binding, info)| VkWriteDescriptorSet {
                    sType: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    pNext: ptr::null(),
                    dstSet: set,
                    dstBinding: binding as u32,
                    dstArrayElement: 0,
                    descriptorCount: 1,
                    descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pImageInfo: ptr::null(),
                    pBufferInfo: info,
                    pTexelBufferView: ptr::null(),
                })
                .collect();
            crate::own_work(|| generated::vkUpdateDescriptorSets(device, 2, writes.as_ptr(), 0, ptr::null()));
            sets.push(set);
        }
        Ok(VideoCoder { key, frame, set_layout, layout, pipeline, descriptor_pool, sets, need_key: AtomicBool::new(false) })
    }
}

/// Whether frames can be coded on this GPU
unsafe fn can_code_on_gpu(physical: VkPhysicalDevice, family: u32, output_size: u64) -> bool {
    if std::env::var("NECTAR_VK_FRAME_CODER").is_ok_and(|coder| coder == "cpu") {
        return false;
    }
    unsafe {
        let mut properties: VkPhysicalDeviceProperties = mem::zeroed();
        generated::vkGetPhysicalDeviceProperties(physical, &mut properties);
        let mut count = 0;
        generated::vkGetPhysicalDeviceQueueFamilyProperties(physical, &mut count, ptr::null_mut());
        let mut families: Vec<VkQueueFamilyProperties> = vec![mem::zeroed(); count as usize];
        generated::vkGetPhysicalDeviceQueueFamilyProperties(physical, &mut count, families.as_mut_ptr());
        // A CPU pretending to be a GPU does better without the pretence
        properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU
            && families.get(family as usize).is_some_and(|family| family.queueFlags & VK_QUEUE_COMPUTE_BIT != 0)
            && output_size <= properties.limits.maxStorageBufferRange as u64
    }
}

/// Sets up the shader; `outputs` are the slots' buffers
unsafe fn create_gpu_coder(device: VkDevice, physical: VkPhysicalDevice, width: u32, height: u32, outputs: &[VkBuffer]) -> Result<GpuCoder, VkResult> {
    let frame_size = width as u64 * height as u64 * 4;
    let output_size = frame_shader::output_size(width, height);
    let work_size = frame_shader::work_size(width, height);
    let hot_size = frame_shader::hot_size(width, height);
    macro_rules! made {
        ($call:expr) => {
            match $call {
                VK_SUCCESS => {}
                error => return Err(error),
            }
        };
    }
    unsafe {
        let mut made_buffers = [(0, 0); 4];
        for (index, frame) in made_buffers.iter_mut().enumerate() {
            let info = VkBufferCreateInfo {
                sType: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                size: [frame_size, frame_size, work_size, hot_size][index],
                usage: VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                sharingMode: VK_SHARING_MODE_EXCLUSIVE,
                queueFamilyIndexCount: 0,
                pQueueFamilyIndices: ptr::null(),
            };
            made!(generated::vkCreateBuffer(device, &info, ptr::null(), &mut frame.0));
            let mut needs: VkMemoryRequirements = mem::zeroed();
            generated::vkGetBufferMemoryRequirements(device, frame.0, &mut needs);
            let kind = memory_type(physical, needs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0).ok_or(VK_ERROR_OUT_OF_DEVICE_MEMORY)?;
            frame.1 = allocate(device, needs.size, kind)?;
            generated::vkBindBufferMemory(device, frame.0, frame.1, 0);
        }
        let [first_frame, second_frame, work, hot] = made_buffers;
        let frames = [first_frame, second_frame];
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
        made!(generated::vkCreateDescriptorSetLayout(device, &set_info, ptr::null(), &mut set_layout));
        let range = VkPushConstantRange { stageFlags: VK_SHADER_STAGE_COMPUTE_BIT, offset: 0, size: frame_shader::PUSH_BYTES };
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
        made!(generated::vkCreatePipelineLayout(device, &layout_info, ptr::null(), &mut layout));
        let mut pipelines = [0; 3];
        for (pipeline, code) in pipelines.iter_mut().zip(frame_shader::shaders()) {
            let module_info = VkShaderModuleCreateInfo {
                sType: VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                codeSize: code.len() * 4,
                pCode: code.as_ptr(),
            };
            let mut module = 0;
            made!(crate::own_work(|| generated::vkCreateShaderModule(device, &module_info, ptr::null(), &mut module)));
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
            made!(crate::own_work(|| generated::vkCreateComputePipelines(device, 0, 1, &pipeline_info, ptr::null(), pipeline)));
            generated::vkDestroyShaderModule(device, module, ptr::null());
        }

        let set_count = outputs.len() as u32 * 2;
        let pool_size = VkDescriptorPoolSize { r#type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount: set_count * 5 };
        let pool_info = VkDescriptorPoolCreateInfo {
            sType: VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            maxSets: set_count,
            poolSizeCount: 1,
            pPoolSizes: &pool_size,
        };
        let mut descriptor_pool = 0;
        made!(generated::vkCreateDescriptorPool(device, &pool_info, ptr::null(), &mut descriptor_pool));
        let mut sets = Vec::with_capacity(outputs.len());
        for &output in outputs {
            let mut pair = [0, 0];
            for (this, set) in pair.iter_mut().enumerate() {
                let allocate_info = VkDescriptorSetAllocateInfo {
                    sType: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                    pNext: ptr::null(),
                    descriptorPool: descriptor_pool,
                    descriptorSetCount: 1,
                    pSetLayouts: &set_layout,
                };
                made!(generated::vkAllocateDescriptorSets(device, &allocate_info, set));
                let infos = [
                    VkDescriptorBufferInfo { buffer: frames[this].0, offset: 0, range: frame_size },
                    VkDescriptorBufferInfo { buffer: frames[1 - this].0, offset: 0, range: frame_size },
                    VkDescriptorBufferInfo { buffer: output, offset: 0, range: output_size },
                    VkDescriptorBufferInfo { buffer: work.0, offset: 0, range: work_size },
                    VkDescriptorBufferInfo { buffer: hot.0, offset: 0, range: hot_size },
                ];
                let writes: Vec<VkWriteDescriptorSet> = infos
                    .iter()
                    .enumerate()
                    .map(|(binding, info)| VkWriteDescriptorSet {
                        sType: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                        pNext: ptr::null(),
                        dstSet: *set,
                        dstBinding: binding as u32,
                        dstArrayElement: 0,
                        descriptorCount: 1,
                        descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        pImageInfo: ptr::null(),
                        pBufferInfo: info,
                        pTexelBufferView: ptr::null(),
                    })
                    .collect();
                crate::own_work(|| generated::vkUpdateDescriptorSets(device, 5, writes.as_ptr(), 0, ptr::null()));
            }
            sets.push(pair);
        }
        Ok(GpuCoder { frames, work, hot, set_layout, layout, pipelines, descriptor_pool, sets })
    }
}

unsafe fn create_swapchain(device: VkDevice, info: &VkSwapchainCreateInfoKHR) -> Result<Arc<Swapchain>, VkResult> {
    unsafe {
        let window = window_of(info.surface).ok_or(VK_ERROR_SURFACE_LOST_KHR)?;
        let Device { physical, family, families } = known(|known| known.devices.get(&(device as usize)).cloned()).ok_or(VK_ERROR_DEVICE_LOST)?;
        let (width, height) = (info.imageExtent.width, info.imageExtent.height);

        let image_info = VkImageCreateInfo {
            sType: VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            imageType: VK_IMAGE_TYPE_2D,
            format: info.imageFormat,
            extent: VkExtent3D { width, height, depth: 1 },
            mipLevels: 1,
            arrayLayers: 1,
            samples: VK_SAMPLE_COUNT_1_BIT,
            tiling: VK_IMAGE_TILING_OPTIMAL,
            usage: info.imageUsage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            sharingMode: info.imageSharingMode,
            queueFamilyIndexCount: info.queueFamilyIndexCount,
            pQueueFamilyIndices: info.pQueueFamilyIndices,
            initialLayout: VK_IMAGE_LAYOUT_UNDEFINED,
        };
        // Room for the most there can be: how many there are at a time follows the program's frames
        let fixed_frames = fixed_frames_on_the_way();
        let on_the_way = fixed_frames.unwrap_or(MOST_FRAMES_ON_THE_WAY);
        let first_limit = fixed_frames.unwrap_or(FRAMES_ON_THE_WAY);
        // Frames on their way hold their images, and the program needs one to render into
        let asked = info.minImageCount.max(2) as usize;
        let count = (asked + on_the_way - 1).min(MAX_IMAGES).max(asked);
        let (mut images, mut image_memory) = (Vec::new(), Vec::new());
        for _ in 0..count {
            let mut image = 0;
            let result = generated::vkCreateImage(device, &image_info, ptr::null(), &mut image);
            if result != VK_SUCCESS {
                return Err(result);
            }
            let mut needs: VkMemoryRequirements = mem::zeroed();
            generated::vkGetImageMemoryRequirements(device, image, &mut needs);
            let kind = memory_type(physical, needs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)
                .ok_or(VK_ERROR_OUT_OF_DEVICE_MEMORY)?;
            let memory = allocate(device, needs.size, kind)?;
            generated::vkBindImageMemory(device, image, memory, 0);
            images.push(image);
            image_memory.push(memory);
        }

        let connection = connection::get().ok_or(VK_ERROR_DEVICE_LOST)?;
        let mut slots = Vec::with_capacity(on_the_way);
        let output_size = frame_shader::output_size(width, height);
        let on_gpu = can_code_on_gpu(physical, family, output_size);
        // As video, if a shader can make its colours here and both ends have the coder for it
        let (coded_width, coded_height) = video_shader::coded_size(width, height);
        let nv12_size = video_shader::nv12_size(coded_width, coded_height);
        let as_video = if on_gpu { open_video(&connection, device, width, height) } else { None };
        // The encoder reads the frames' NV12 from its own queue family
        let mut sharing = families.clone();
        sharing.extend(as_video.map(|(_, encodes)| encodes));
        sharing.sort_unstable();
        sharing.dedup();
        let shared = as_video.is_some() && sharing.len() > 1;
        for _ in 0..on_the_way {
            let buffer_info = VkBufferCreateInfo {
                sType: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                size: if as_video.is_some() {
                    nv12_size
                } else if on_gpu {
                    output_size
                } else {
                    width as u64 * height as u64 * 4
                },
                usage: if as_video.is_some() {
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                } else if on_gpu {
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                } else {
                    VK_BUFFER_USAGE_TRANSFER_DST_BIT
                },
                sharingMode: if shared { VK_SHARING_MODE_CONCURRENT } else { VK_SHARING_MODE_EXCLUSIVE },
                queueFamilyIndexCount: if shared { sharing.len() as u32 } else { 0 },
                pQueueFamilyIndices: if shared { sharing.as_ptr() } else { ptr::null() },
            };
            let mut buffer = 0;
            let result = generated::vkCreateBuffer(device, &buffer_info, ptr::null(), &mut buffer);
            if result != VK_SUCCESS {
                return Err(result);
            }
            let mut needs: VkMemoryRequirements = mem::zeroed();
            generated::vkGetBufferMemoryRequirements(device, buffer, &mut needs);
            const READABLE: VkMemoryPropertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            // The encoder takes video's from the GPU, so nothing reads it here or on the lender's CPU
            let kind = if as_video.is_some() {
                memory_type(physical, needs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0)
            } else {
                memory_type(physical, needs.memoryTypeBits, READABLE | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, READABLE)
            }
            .ok_or(VK_ERROR_OUT_OF_DEVICE_MEMORY)?;
            let memory = allocate(device, needs.size, kind)?;
            generated::vkBindBufferMemory(device, buffer, memory, 0);
            if as_video.is_none() {
                memory::map_readback(&connection, memory)?;
            }

            let fence_info = VkFenceCreateInfo { sType: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, pNext: ptr::null(), flags: 0 };
            let mut fence = 0;
            let result = generated::vkCreateFence(device, &fence_info, ptr::null(), &mut fence);
            if result != VK_SUCCESS {
                return Err(result);
            }
            slots.push(Slot { buffer, memory, fence });
        }
        let outputs: Vec<VkBuffer> = slots.iter().map(|slot| slot.buffer).collect();
        let video = match as_video {
            Some((key, _)) => Some(create_video_coder(device, physical, key, width, height, &outputs)?),
            None => None,
        };
        let gpu = if on_gpu && video.is_none() { Some(create_gpu_coder(device, physical, width, height, &outputs)?) } else { None };
        if as_video.is_none() && !on_gpu && (logging() || std::env::var_os("NECTAR_VK_STATS").is_some()) {
            crate::log(&format!("frames of {width}x{height} are coded on the lender's CPU"));
        }
        let mut queue: VkQueue = mem::zeroed();
        crate::vkGetDeviceQueue(device, family, 0, &mut queue);

        let (jobs, received) = channel::<Job>();
        let swapchain = Arc::new(Swapchain {
            device,
            window,
            width,
            height,
            images,
            image_memory,
            slots,
            gpu,
            video,
            quality: AtomicU32::new(FIRST_QUALITY),
            queue,
            copies: Mutex::new(HashMap::new()),
            state: Mutex::new(State { held: vec![false; count], limit: first_limit, ..State::default() }),
            changed: Condvar::new(),
            jobs: Mutex::new(Some(jobs)),
            thread: Mutex::new(None),
        });
        let drawing = Arc::clone(&swapchain);
        *swapchain.thread.lock().unwrap_or_else(|p| p.into_inner()) = Some(std::thread::spawn(move || {
            let fixed = std::env::var("NECTAR_VK_FRAME_QUALITY").ok().and_then(|value| value.parse().ok());
            if let Some(quality) = fixed {
                drawing.quality.store(quality, Ordering::Relaxed);
            }
            let mut shown = vec![0u8; drawing.width as usize * drawing.height as usize * 4];
            let report = logging() || std::env::var_os("NECTAR_VK_STATS").is_some();
            let (mut frames, mut bytes) = (0u32, 0usize);
            let mut pacing = Pacing::default();
            let mut on_the_way = OnTheWay::new(drawing.state.lock().unwrap_or_else(|p| p.into_inner()).limit);
            let mut stages = Stages::default();
            let mut since = Instant::now();
            let mut before = Totals::now();
            let mut showing = Showing { decoder: None, waiting_for_key: true, complaints: 0, floor: 0.0 };
            let mut decoded_before = (Duration::ZERO, Duration::ZERO);
            for job in received {
                let (asked, waited) = (job.asked, job.waited);
                let (coded, cost) = drawing.draw(job, &mut shown, &mut stages, &mut showing);
                bytes += coded;
                frames += 1;
                if let Some(cost) = cost {
                    pacing.frame(asked, cost, waited);
                    if fixed.is_none() {
                        let quality = drawing.quality.load(Ordering::Relaxed);
                        drawing.quality.store(pacing.next_quality(quality), Ordering::Relaxed);
                    }
                    if fixed_frames.is_none() {
                        let wanted = on_the_way.frame(&pacing, drawing.slots.len());
                        let mut state = drawing.state.lock().unwrap_or_else(|p| p.into_inner());
                        if state.limit != wanted {
                            state.limit = wanted;
                            drawing.changed.notify_all();
                        }
                    }
                }
                if report && since.elapsed() >= Duration::from_secs(2) {
                    let now = Totals::now();
                    let quality = drawing.quality.load(Ordering::Relaxed);
                    let limit = drawing.state.lock().unwrap_or_else(|p| p.into_inner()).limit;
                    // Of video's decoding: the decoder itself, and turning its picture into the window's colours
                    let video = showing.decoder.as_ref().map_or(String::new(), |decoder| {
                        let now = (decoder.decoding, decoder.colours);
                        let per_frame = |now: Duration, before: Duration| now.saturating_sub(before).as_secs_f64() * 1000.0 / frames.max(1) as f64;
                        let said = format!(", video's decoder {:.1} ms and colours {:.1} ms", per_frame(now.0, decoded_before.0), per_frame(now.1, decoded_before.1));
                        decoded_before = now;
                        said
                    });
                    crate::log(&format!("{}, up to {limit} frames on their way{video}", stages.report(frames, since.elapsed(), bytes, quality, &before, &now)));
                    before = now;
                    frames = 0;
                    bytes = 0;
                    stages = Stages::default();
                    since = Instant::now();
                }
            }
        }));
        Ok(swapchain)
    }
}

impl Swapchain {
    /// A command buffer per slot, for `family`, begun for a frame: the image is copied to `target`, where a
    /// shader reads it
    unsafe fn begin_frame(&self, family: u32, image: usize, slot: usize, target: VkBuffer) -> Result<VkCommandBuffer, VkResult> {
        let mut copies = self.copies.lock().unwrap_or_else(|p| p.into_inner());
        unsafe {
            if let std::collections::hash_map::Entry::Vacant(entry) = copies.entry(family) {
                let pool_info = VkCommandPoolCreateInfo {
                    sType: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                    pNext: ptr::null(),
                    flags: VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                    queueFamilyIndex: family,
                };
                let mut pool = 0;
                let result = generated::vkCreateCommandPool(self.device, &pool_info, ptr::null(), &mut pool);
                if result != VK_SUCCESS {
                    return Err(result);
                }
                let allocate_info = VkCommandBufferAllocateInfo {
                    sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                    pNext: ptr::null(),
                    commandPool: pool,
                    level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                    commandBufferCount: self.slots.len() as u32,
                };
                let mut buffers: Vec<VkCommandBuffer> = vec![mem::zeroed(); self.slots.len()];
                let result = generated::vkAllocateCommandBuffers(self.device, &allocate_info, buffers.as_mut_ptr());
                if result != VK_SUCCESS {
                    return Err(result);
                }
                entry.insert((pool, buffers));
            }
            let commands = copies[&family].1[slot];
            let begin = VkCommandBufferBeginInfo {
                sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                pNext: ptr::null(),
                flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                pInheritanceInfo: ptr::null(),
            };
            let result = generated::vkBeginCommandBuffer(commands, &begin);
            if result != VK_SUCCESS {
                return Err(result);
            }
            let whole = VkImageSubresourceRange {
                aspectMask: COLOR_ASPECT,
                baseMipLevel: 0,
                levelCount: 1,
                baseArrayLayer: 0,
                layerCount: 1,
            };
            let image_barrier = |from, to, source, target| VkImageMemoryBarrier {
                sType: VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                pNext: ptr::null(),
                srcAccessMask: source,
                dstAccessMask: target,
                oldLayout: from,
                newLayout: to,
                srcQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
                image: self.images[image],
                subresourceRange: whole,
            };
            // The shader last read what's copied over now
            let before = image_barrier(
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_ACCESS_MEMORY_WRITE_BIT,
                VK_ACCESS_TRANSFER_READ_BIT,
            );
            let overwrite = memory_barrier(VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            generated::vkCmdPipelineBarrier(
                commands,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                1,
                &overwrite,
                0,
                ptr::null(),
                1,
                &before,
            );
            let region = VkBufferImageCopy {
                bufferOffset: 0,
                bufferRowLength: 0,
                bufferImageHeight: 0,
                imageSubresource: VkImageSubresourceLayers {
                    aspectMask: COLOR_ASPECT,
                    mipLevel: 0,
                    baseArrayLayer: 0,
                    layerCount: 1,
                },
                imageOffset: VkOffset3D { x: 0, y: 0, z: 0 },
                imageExtent: VkExtent3D { width: self.width, height: self.height, depth: 1 },
            };
            generated::vkCmdCopyImageToBuffer(commands, self.images[image], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, target, 1, &region);
            let after = image_barrier(
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_ACCESS_TRANSFER_READ_BIT,
                0,
            );
            let read = memory_barrier(VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
            generated::vkCmdPipelineBarrier(
                commands,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                0,
                1,
                &read,
                0,
                ptr::null(),
                1,
                &after,
            );
            Ok(commands)
        }
    }

    /// The frame's commands when it goes as video: its NV12 is made in the slot's buffer
    unsafe fn video_frame(&self, coder: &VideoCoder, family: u32, image: usize, slot: usize) -> Result<VkCommandBuffer, VkResult> {
        unsafe {
            let commands = self.begin_frame(family, image, slot, coder.frame.0)?;
            generated::vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, coder.pipeline);
            generated::vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, coder.layout, 0, 1, &coder.sets[slot], 0, ptr::null());
            // The surface's formats all have blue first
            let constants = video_shader::push_constants(self.width, self.height, true);
            generated::vkCmdPushConstants(commands, coder.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, video_shader::PUSH_BYTES, constants.as_ptr().cast());
            let (coded_width, coded_height) = video_shader::coded_size(self.width, self.height);
            let (across, down) = video_shader::groups(coded_width, coded_height);
            generated::vkCmdDispatch(commands, across, down, 1);
            let written = memory_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            generated::vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &written, 0, ptr::null(), 0, ptr::null());
            match crate::own_work(|| generated::vkEndCommandBuffer(commands)) {
                VK_SUCCESS => Ok(commands),
                error => Err(error),
            }
        }
    }

    /// A command buffer per slot, for `family`, recorded for each frame: copies the image to where the
    /// shader reads it, and has it coded into the slot
    unsafe fn gpu_frame(&self, coder: &GpuCoder, family: u32, image: usize, slot: usize, frame_number: u64) -> Result<VkCommandBuffer, VkResult> {
        let this = (frame_number % 2) as usize;
        unsafe {
            let commands = self.begin_frame(family, image, slot, coder.frames[this].0)?;
            generated::vkCmdBindDescriptorSets(commands, VK_PIPELINE_BIND_POINT_COMPUTE, coder.layout, 0, 1, &coder.sets[slot][this], 0, ptr::null());
            let quality = self.quality.load(Ordering::Relaxed);
            let between = memory_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
            for pass in frame_shader::passes(self.width, self.height) {
                generated::vkCmdBindPipeline(commands, VK_PIPELINE_BIND_POINT_COMPUTE, coder.pipelines[pass.shader]);
                let constants = frame_shader::push_constants(self.width, self.height, quality, frame_number == 0, pass.first_plane);
                generated::vkCmdPushConstants(commands, coder.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, frame_shader::PUSH_BYTES, constants.as_ptr().cast());
                generated::vkCmdDispatch(commands, pass.groups[0], pass.groups[1], pass.groups[2]);
                generated::vkCmdPipelineBarrier(
                    commands,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,
                    1,
                    &between,
                    0,
                    ptr::null(),
                    0,
                    ptr::null(),
                );
            }
            let to_copy = memory_barrier(VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            generated::vkCmdPipelineBarrier(
                commands,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                1,
                &to_copy,
                0,
                ptr::null(),
                0,
                ptr::null(),
            );
            let hot_region = VkBufferCopy { srcOffset: 0, dstOffset: 0, size: frame_shader::hot_size(self.width, self.height) };
            generated::vkCmdCopyBuffer(commands, coder.hot.0, self.slots[slot].buffer, 1, &hot_region);
            let readable = memory_barrier(VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
            generated::vkCmdPipelineBarrier(
                commands,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT,
                0,
                1,
                &readable,
                0,
                ptr::null(),
                0,
                ptr::null(),
            );
            match crate::own_work(|| generated::vkEndCommandBuffer(commands)) {
                VK_SUCCESS => Ok(commands),
                error => Err(error),
            }
        }
    }

    /// A command buffer per image and slot, for `family`: copies the image to the buffer the CPU reads
    unsafe fn copy_of(&self, family: u32, image: usize, slot: usize) -> Result<VkCommandBuffer, VkResult> {
        let per_image = self.slots.len();
        let mut copies = self.copies.lock().unwrap_or_else(|p| p.into_inner());
        if let Some((_, buffers)) = copies.get(&family) {
            return Ok(buffers[image * per_image + slot]);
        }
        unsafe {
            let pool_info = VkCommandPoolCreateInfo {
                sType: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                pNext: ptr::null(),
                flags: 0,
                queueFamilyIndex: family,
            };
            let mut pool = 0;
            let result = generated::vkCreateCommandPool(self.device, &pool_info, ptr::null(), &mut pool);
            if result != VK_SUCCESS {
                return Err(result);
            }
            let allocate_info = VkCommandBufferAllocateInfo {
                sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                pNext: ptr::null(),
                commandPool: pool,
                level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount: (self.images.len() * per_image) as u32,
            };
            let mut buffers: Vec<VkCommandBuffer> = vec![mem::zeroed(); self.images.len() * per_image];
            let result = generated::vkAllocateCommandBuffers(self.device, &allocate_info, buffers.as_mut_ptr());
            if result != VK_SUCCESS {
                return Err(result);
            }
            for (position, &commands) in buffers.iter().enumerate() {
                self.record(commands, self.images[position / per_image], self.slots[position % per_image].buffer)?;
            }
            let chosen = buffers[image * per_image + slot];
            copies.insert(family, (pool, buffers));
            Ok(chosen)
        }
    }

    unsafe fn record(&self, commands: VkCommandBuffer, image: VkImage, target: VkBuffer) -> Result<(), VkResult> {
        let whole = VkImageSubresourceRange {
            aspectMask: COLOR_ASPECT,
            baseMipLevel: 0,
            levelCount: 1,
            baseArrayLayer: 0,
            layerCount: 1,
        };
        let barrier = |from, to, source, destination| VkImageMemoryBarrier {
            sType: VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            pNext: ptr::null(),
            srcAccessMask: source,
            dstAccessMask: destination,
            oldLayout: from,
            newLayout: to,
            srcQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
            image,
            subresourceRange: whole,
        };
        let begin = VkCommandBufferBeginInfo {
            sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext: ptr::null(),
            flags: 0,
            pInheritanceInfo: ptr::null(),
        };
        unsafe {
            let result = generated::vkBeginCommandBuffer(commands, &begin);
            if result != VK_SUCCESS {
                return Err(result);
            }
            let before = barrier(
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_ACCESS_MEMORY_WRITE_BIT,
                VK_ACCESS_TRANSFER_READ_BIT,
            );
            generated::vkCmdPipelineBarrier(
                commands,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0,
                ptr::null(),
                0,
                ptr::null(),
                1,
                &before,
            );
            let region = VkBufferImageCopy {
                bufferOffset: 0,
                bufferRowLength: 0,
                bufferImageHeight: 0,
                imageSubresource: VkImageSubresourceLayers {
                    aspectMask: COLOR_ASPECT,
                    mipLevel: 0,
                    baseArrayLayer: 0,
                    layerCount: 1,
                },
                imageOffset: VkOffset3D { x: 0, y: 0, z: 0 },
                imageExtent: VkExtent3D { width: self.width, height: self.height, depth: 1 },
            };
            generated::vkCmdCopyImageToBuffer(commands, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, target, 1, &region);
            let readable = VkMemoryBarrier {
                sType: VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                pNext: ptr::null(),
                srcAccessMask: VK_ACCESS_TRANSFER_WRITE_BIT,
                dstAccessMask: VK_ACCESS_HOST_READ_BIT,
            };
            let after = barrier(
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                VK_ACCESS_TRANSFER_READ_BIT,
                0,
            );
            generated::vkCmdPipelineBarrier(
                commands,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                0,
                1,
                &readable,
                0,
                ptr::null(),
                1,
                &after,
            );
            match crate::own_work(|| generated::vkEndCommandBuffer(commands)) {
                VK_SUCCESS => Ok(()),
                error => Err(error),
            }
        }
    }

    /// Asks for the frame the submit just made is copying to `slot`
    fn ask(&self, slot: usize) -> std::io::Result<Later> {
        let connection = connection::get().ok_or(std::io::ErrorKind::NotConnected)?;
        let mut w = Writer::new();
        if let Some(video) = &self.video {
            w.u64(video.key)
                .u64(self.slots[slot].buffer)
                .u64(unsafe { objects::id_of(self.device as u64) })
                .u64(self.slots[slot].fence)
                .u32(self.quality.load(Ordering::Relaxed))
                .u8(video.need_key.swap(false, Ordering::Relaxed) as u8);
            return connection.ask_later(VIDEO_FRAME, w.as_slice());
        }
        w.u64(self.slots[0].memory)
            .u64(self.slots[slot].memory)
            .u64(unsafe { objects::id_of(self.device as u64) })
            .u64(self.slots[slot].fence)
            .u32(self.width)
            .u32(self.height)
            .u32(self.quality.load(Ordering::Relaxed))
            .u8(self.gpu.is_some() as u8);
        connection.ask_later(FRAME, w.as_slice())
    }

    /// On the drawing thread: waits for the frame, and draws it. Returns the bytes that crossed, and
    /// what the quality can change about the time it took.
    fn draw(&self, job: Job, shown: &mut [u8], stages: &mut Stages, showing: &mut Showing) -> (usize, Option<Duration>) {
        let (mut bytes, mut cost) = (0, None);
        let Job { image, slot, answer, asked, .. } = job;
        let answer = answer.wait().ok().filter(|answer| answer.result == 0);
        let came = Instant::now();
        match answer {
            Some(answer) => {
                let mut outputs = answer.outputs();
                let lender = Duration::from_micros(outputs.u32().unwrap_or(0) as u64);
                let coding = Duration::from_micros(outputs.u32().unwrap_or(0) as u64);
                let key = self.video.is_some() && outputs.u8().unwrap_or(0) != 0;
                let coded = outputs.bytes().unwrap_or_default();
                bytes = coded.len();
                stages.lender += lender.as_secs_f64();
                stages.coding += coding.as_secs_f64();
                stages.crossing += answer.crossing.as_secs_f64();
                stages.trip += (came - asked).saturating_sub(lender).saturating_sub(answer.crossing).as_secs_f64();
                cost = Some(answer.crossing.max(coding));
                let picture = match &self.video {
                    Some(video) => showing.picture(video, key, coded, shown, self.width, self.height),
                    None => frame::decode(coded, shown, self.width, self.height),
                };
                // A video picture takes longer to decode the finer it is, on the one thread that draws, so
                // that counts with what quality can change. The other coder's decoding barely follows it.
                if self.video.is_some() && picture {
                    let over = showing.over_floor(came.elapsed());
                    cost = cost.map(|cost| cost.max(over));
                }
                if picture {
                    let decoded = Instant::now();
                    stages.decode += (decoded - came).as_secs_f64();
                    unsafe { self.window.show(shown.as_ptr(), self.width, self.height) };
                    stages.show += decoded.elapsed().as_secs_f64();
                }
            }
            None => {
                // The pictures after one that's missing are coded against it
                if let Some(video) = &self.video {
                    showing.waiting_for_key = true;
                    video.need_key.store(true, Ordering::Relaxed);
                }
                if logging() {
                    crate::log("a frame didn't come");
                }
            }
        }
        unsafe { generated::vkResetFences(self.device, 1, &self.slots[slot].fence) };
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        state.held[image] = false;
        state.on_the_way -= 1;
        self.changed.notify_all();
        (bytes, cost)
    }

    /// Whether the window still is the size this swapchain was made for
    fn fits(&self) -> bool {
        self.window.size() == (self.width, self.height)
    }
}

/// Seconds spent on frames, for NECTAR_VK_STATS
#[derive(Default)]
struct Stages {
    /// From the question reaching the lender to the answer leaving: rendering, the copy, and coding
    lender: f64,
    coding: f64,
    /// The frame's bytes arriving
    crossing: f64,
    /// The rest of the time it took: across and back, behind whatever else was on the link
    trip: f64,
    decode: f64,
    show: f64,
}

/// What's counted over the whole connection, to take differences of
struct Totals {
    sent: u64,
    sent_packed: u64,
    packing: u64,
    flush: u64,
    touch: u64,
    touches: u64,
    asks: u64,
}

impl Totals {
    fn now() -> Totals {
        use std::sync::atomic::Ordering::Relaxed;
        Totals {
            sent: connection::get().map_or(0, |connection| connection.sent.load(Relaxed)),
            sent_packed: connection::get().map_or(0, |connection| connection.sent_packed().0),
            packing: connection::get().map_or(0, |connection| connection.sent_packed().1),
            flush: memory::FLUSH_NANOS.load(Relaxed),
            touch: memory::TOUCH_NANOS.load(Relaxed),
            touches: memory::TOUCHES.load(Relaxed),
            asks: memory::ASKS.load(Relaxed),
        }
    }
}

impl Stages {
    fn report(&self, frames: u32, over: Duration, bytes: usize, quality: u32, before: &Totals, now: &Totals) -> String {
        let count = frames.max(1) as f64;
        let per_frame = |nanos: u64| nanos as f64 / 1e6 / count;
        let gpu = connection::get()
            .and_then(|connection| connection.ask(GPU_USAGE, &[]).ok())
            .and_then(|answer| answer.outputs().u32().ok())
            .filter(|&percent| percent != u32::MAX)
            .map_or("the lender can't say".to_string(), |percent| format!("{percent}%"));
        let packed = now.sent_packed.saturating_sub(before.sent_packed);
        let packing = now.packing.saturating_sub(before.packing);
        let packed = if packed > 0 {
            format!(" ({:.0} KB compressed in {:.2} ms)", packed as f64 / count / 1e3, packing as f64 / count / 1e6)
        } else {
            String::new()
        };
        format!(
            "present: {:.1} frames/s, GPU {gpu}, {:.0} KB down and {:.0} KB up{packed} per frame, quality {quality}; \
             ms per frame: on the lender {:.1} (coding {:.1}), across and back {:.1}, frame crossing {:.1}, decode {:.1}, show {:.1}, sending writes {:.1}, {:.1} touches ({:.1} asking) {:.1}",
            frames as f64 / over.as_secs_f64(),
            bytes as f64 / count / 1e3,
            now.sent.saturating_sub(before.sent) as f64 / count / 1e3,
            self.lender * 1000.0 / count,
            self.coding * 1000.0 / count,
            self.trip * 1000.0 / count,
            self.crossing * 1000.0 / count,
            self.decode * 1000.0 / count,
            self.show * 1000.0 / count,
            per_frame(now.flush.saturating_sub(before.flush)),
            now.touches.saturating_sub(before.touches) as f64 / count,
            now.asks.saturating_sub(before.asks) as f64 / count,
            per_frame(now.touch.saturating_sub(before.touch)),
        )
    }
}

/// How many frames NECTAR_VK_FRAMES says are on their way at a time, if it does
fn fixed_frames_on_the_way() -> Option<usize> {
    std::env::var("NECTAR_VK_FRAMES").ok().and_then(|value| value.parse::<usize>().ok()).map(|frames| frames.clamp(1, MOST_FRAMES_ON_THE_WAY))
}

/// What the quality and the frames on their way go by, smoothed over the last several frames, in seconds
#[derive(Default)]
struct Pacing {
    last: Option<Instant>,
    /// Between the program's frames
    interval: f64,
    /// What quality can change about a frame: the longer of coding it and its bytes crossing, as each of
    /// those limits the frame rate on its own
    cost: f64,
    /// The program waiting in present for a frame to finish
    waited: f64,
}

impl Pacing {
    fn frame(&mut self, asked: Instant, cost: Duration, waited: Duration) {
        let smooth = |old: f64, new: f64| if old == 0.0 { new } else { old + (new - old) / 8.0 };
        if let Some(last) = self.last {
            self.interval = smooth(self.interval, asked.saturating_duration_since(last).as_secs_f64());
        }
        self.last = Some(asked);
        self.cost = smooth(self.cost, cost.as_secs_f64());
        self.waited = smooth(self.waited, waited.as_secs_f64());
    }

    /// The quality for the next frame. A frame may take three quarters of the time between the program's
    /// frames, or of a 120 fps frame if they come faster, leaving the link room for what else crosses.
    /// It used to be held to 12 ms whatever the program's rate, one frame at a time, coming down 5 for a
    /// frame over and going up 1 for a frame well under: a link's jitter alone walked it down to about 20
    /// at 2560x1440, for a game at 35 fps with 28 ms between its frames.
    fn next_quality(&self, quality: u32) -> u32 {
        // Not before the time between frames is known
        if self.interval <= 0.0 {
            return quality;
        }
        let budget = self.interval.max(FASTEST_FRAME.as_secs_f64());
        if self.cost > budget * 3.0 / 4.0 {
            quality.saturating_sub(5).max(LOWEST_QUALITY)
        } else if self.cost < budget * 3.0 / 8.0 {
            (quality + 1).min(BEST_QUALITY)
        } else {
            quality
        }
    }
}

/// How many frames may be on their way at a time, found by trying. Reasoning it out from how long a frame
/// takes on its way doesn't work while the program is being held back, as then that time and the time
/// between frames are the same thing.
struct OnTheWay {
    limit: usize,
    /// Frames since it last looked
    frames: u32,
    /// The frame rate before one more was tried, while it's being tried
    trying: Option<f64>,
    /// Looks to go before one more is tried again, after it didn't help
    hold: u32,
}

/// Frames between looks: long enough for the smoothed times to follow a change
const LOOK_EVERY: u32 = 30;
/// Looks before trying again what didn't help
const HOLD_LOOKS: u32 = 20;

impl OnTheWay {
    fn new(limit: usize) -> Self {
        Self { limit, frames: 0, trying: None, hold: 0 }
    }

    /// Counts a frame, and returns how many may be on their way from here on
    fn frame(&mut self, pacing: &Pacing, most: usize) -> usize {
        self.frames += 1;
        if self.frames < LOOK_EVERY || pacing.interval <= 0.0 {
            return self.limit;
        }
        self.frames = 0;
        let rate = 1.0 / pacing.interval;
        if let Some(before) = self.trying.take() {
            // Worth it only if frames come noticeably faster
            if rate < before * 1.05 {
                self.limit -= 1;
                self.hold = HOLD_LOOKS;
            }
        } else if self.hold > 0 {
            self.hold -= 1;
        } else if pacing.waited > pacing.interval / 20.0 && self.limit < most {
            self.trying = Some(rate);
            self.limit += 1;
        }
        self.limit
    }
}

fn swapchain(handle: VkSwapchainKHR) -> Option<Arc<Swapchain>> {
    known(|known| known.swapchains.get(&handle).cloned())
}

pub unsafe extern "system" fn vkCreateSwapchainKHR(
    device: VkDevice,
    pCreateInfo: *const VkSwapchainCreateInfoKHR,
    _pAllocator: *const VkAllocationCallbacks,
    pSwapchain: *mut VkSwapchainKHR,
) -> VkResult {
    match unsafe { create_swapchain(device, &*pCreateInfo) } {
        Ok(created) => {
            let handle = crate::objects::new_id();
            known(|known| known.swapchains.insert(handle, created));
            unsafe { *pSwapchain = handle };
            VK_SUCCESS
        }
        Err(error) => error,
    }
}

pub unsafe extern "system" fn vkDestroySwapchainKHR(device: VkDevice, handle: VkSwapchainKHR, _pAllocator: *const VkAllocationCallbacks) {
    let Some(swapchain) = known(|known| known.swapchains.remove(&handle)) else { return };
    // Ends the drawing thread once its frames are done
    drop(swapchain.jobs.lock().unwrap_or_else(|p| p.into_inner()).take());
    if let Some(thread) = swapchain.thread.lock().unwrap_or_else(|p| p.into_inner()).take() {
        let _ = thread.join();
    }
    unsafe {
        // The lender's encoder goes first, as it reads the slots' buffers
        if let Some(video) = &swapchain.video {
            let mut w = Writer::new();
            w.u64(video.key);
            let _ = connection::get().map(|connection| connection.ask(VIDEO_CLOSE, w.as_slice()));
        }
        // The frames still being coded finished before their answers came, so nothing is in use
        for (pool, _) in swapchain.copies.lock().unwrap_or_else(|p| p.into_inner()).values() {
            generated::vkDestroyCommandPool(device, *pool, ptr::null());
        }
        if let Some(coder) = &swapchain.video {
            generated::vkDestroyDescriptorPool(device, coder.descriptor_pool, ptr::null());
            generated::vkDestroyPipeline(device, coder.pipeline, ptr::null());
            generated::vkDestroyPipelineLayout(device, coder.layout, ptr::null());
            generated::vkDestroyDescriptorSetLayout(device, coder.set_layout, ptr::null());
            generated::vkDestroyBuffer(device, coder.frame.0, ptr::null());
            generated::vkFreeMemory(device, coder.frame.1, ptr::null());
        }
        for (&image, &memory) in swapchain.images.iter().zip(&swapchain.image_memory) {
            generated::vkDestroyImage(device, image, ptr::null());
            generated::vkFreeMemory(device, memory, ptr::null());
        }
        if let Some(coder) = &swapchain.gpu {
            generated::vkDestroyDescriptorPool(device, coder.descriptor_pool, ptr::null());
            for pipeline in coder.pipelines {
                generated::vkDestroyPipeline(device, pipeline, ptr::null());
            }
            for (buffer, memory) in [coder.work, coder.hot] {
                generated::vkDestroyBuffer(device, buffer, ptr::null());
                generated::vkFreeMemory(device, memory, ptr::null());
            }
            generated::vkDestroyPipelineLayout(device, coder.layout, ptr::null());
            generated::vkDestroyDescriptorSetLayout(device, coder.set_layout, ptr::null());
            for (buffer, memory) in coder.frames {
                generated::vkDestroyBuffer(device, buffer, ptr::null());
                generated::vkFreeMemory(device, memory, ptr::null());
            }
        }
        for slot in &swapchain.slots {
            generated::vkDestroyBuffer(device, slot.buffer, ptr::null());
            generated::vkFreeMemory(device, slot.memory, ptr::null());
            generated::vkDestroyFence(device, slot.fence, ptr::null());
        }
    }
}

pub unsafe extern "system" fn vkGetSwapchainImagesKHR(_device: VkDevice, handle: VkSwapchainKHR, pCount: *mut u32, pImages: *mut VkImage) -> VkResult {
    let Some(swapchain) = swapchain(handle) else { return VK_ERROR_SURFACE_LOST_KHR };
    unsafe { crate::fill(&swapchain.images, pCount, pImages) }
}

pub unsafe extern "system" fn vkAcquireNextImageKHR(
    _device: VkDevice,
    handle: VkSwapchainKHR,
    timeout: u64,
    semaphore: VkSemaphore,
    fence: VkFence,
    pImageIndex: *mut u32,
) -> VkResult {
    let Some(swapchain) = swapchain(handle) else { return VK_ERROR_SURFACE_LOST_KHR };
    let started = Instant::now();
    let limit = Duration::from_nanos(timeout);
    let mut state = swapchain.state.lock().unwrap_or_else(|p| p.into_inner());
    let index = loop {
        let count = state.held.len();
        if let Some(index) = (0..count).map(|step| (state.next + step) % count).find(|&index| !state.held[index]) {
            break index;
        }
        if timeout == WAIT_NOTHING {
            return VK_NOT_READY;
        }
        let left = limit.saturating_sub(started.elapsed());
        if left.is_zero() {
            return VK_TIMEOUT;
        }
        state = swapchain.changed.wait_timeout(state, left).unwrap_or_else(|p| p.into_inner()).0;
    };
    state.held[index] = true;
    state.next = (index + 1) % state.held.len();
    drop(state);
    unsafe { *pImageIndex = index as u32 };

    // The image is ready at once: an empty submit signals what the program waits on
    if semaphore != 0 || fence != 0 {
        let submit = VkSubmitInfo {
            sType: VK_STRUCTURE_TYPE_SUBMIT_INFO,
            pNext: ptr::null(),
            waitSemaphoreCount: 0,
            pWaitSemaphores: ptr::null(),
            pWaitDstStageMask: ptr::null(),
            commandBufferCount: 0,
            pCommandBuffers: ptr::null(),
            signalSemaphoreCount: (semaphore != 0) as u32,
            pSignalSemaphores: &semaphore,
        };
        let result = crate::own_work(|| unsafe { generated::vkQueueSubmit(swapchain.queue, 1, &submit, fence) });
        if result != VK_SUCCESS {
            return result;
        }
    }
    if swapchain.fits() { VK_SUCCESS } else { VK_SUBOPTIMAL_KHR }
}

pub unsafe extern "system" fn vkAcquireNextImage2KHR(device: VkDevice, pInfo: *const VkAcquireNextImageInfoKHR, pImageIndex: *mut u32) -> VkResult {
    let info = unsafe { &*pInfo };
    unsafe { vkAcquireNextImageKHR(device, info.swapchain, info.timeout, info.semaphore, info.fence, pImageIndex) }
}

pub unsafe extern "system" fn vkQueuePresentKHR(queue: VkQueue, pPresentInfo: *const VkPresentInfoKHR) -> VkResult {
    let info = unsafe { &*pPresentInfo };
    let family = known(|known| known.families.get(&(queue as usize)).copied()).unwrap_or(0);
    let handles = unsafe { std::slice::from_raw_parts(info.pSwapchains, info.swapchainCount as usize) };
    let indices = unsafe { std::slice::from_raw_parts(info.pImageIndices, info.swapchainCount as usize) };
    let mut worst = VK_SUCCESS;
    for (position, (&handle, &index)) in handles.iter().zip(indices).enumerate() {
        let result = (|| {
            let swapchain = swapchain(handle).ok_or(VK_ERROR_SURFACE_LOST_KHR)?;
            let image = index as usize;
            // The program waits here for the link
            let waiting = Instant::now();
            let (slot, frame_number) = {
                let mut state = swapchain.state.lock().unwrap_or_else(|p| p.into_inner());
                while state.on_the_way >= state.limit.min(swapchain.slots.len()) {
                    state = swapchain.changed.wait(state).unwrap_or_else(|p| p.into_inner());
                }
                state.on_the_way += 1;
                let slot = state.next_slot;
                state.next_slot = (slot + 1) % swapchain.slots.len();
                state.frames += 1;
                (slot, state.frames - 1)
            };
            let waited = waiting.elapsed();
            let commands = unsafe {
                match (&swapchain.video, &swapchain.gpu) {
                    (Some(coder), _) => swapchain.video_frame(coder, family, image, slot),
                    (None, Some(coder)) => swapchain.gpu_frame(coder, family, image, slot, frame_number),
                    (None, None) => swapchain.copy_of(family, image, slot),
                }
            }?;
            // What the program asked to wait for, on the first swapchain's copy
            let waits = if position == 0 { info.waitSemaphoreCount } else { 0 };
            let stages = vec![VK_PIPELINE_STAGE_ALL_COMMANDS_BIT; waits as usize];
            let submit = VkSubmitInfo {
                sType: VK_STRUCTURE_TYPE_SUBMIT_INFO,
                pNext: ptr::null(),
                waitSemaphoreCount: waits,
                pWaitSemaphores: info.pWaitSemaphores,
                pWaitDstStageMask: stages.as_ptr(),
                commandBufferCount: 1,
                pCommandBuffers: &commands,
                signalSemaphoreCount: 0,
                pSignalSemaphores: ptr::null(),
            };
            let result = crate::own_work(|| unsafe { generated::vkQueueSubmit(queue, 1, &submit, swapchain.slots[slot].fence) });
            if result != VK_SUCCESS {
                return Err(result);
            }
            let asked = Instant::now();
            let answer = swapchain.ask(slot).map_err(|_| VK_ERROR_DEVICE_LOST)?;
            let jobs = swapchain.jobs.lock().unwrap_or_else(|p| p.into_inner());
            let sent = jobs.as_ref().map(|jobs| jobs.send(Job { image, slot, answer, asked, waited }));
            if !matches!(sent, Some(Ok(()))) {
                return Err(VK_ERROR_SURFACE_LOST_KHR);
            }
            Ok(if swapchain.fits() { VK_SUCCESS } else { VK_SUBOPTIMAL_KHR })
        })()
        .unwrap_or_else(|error| error);
        if !info.pResults.is_null() {
            unsafe { *info.pResults.add(position) = result };
        }
        if result < 0 || worst == VK_SUCCESS {
            worst = result;
        }
    }
    worst
}

/// A command of this file, by name
pub(crate) fn lookup(name: &str) -> PFN_vkVoidFunction {
    macro_rules! own {
        ($($f:ident),*) => {
            match name {
                $(stringify!($f) => Some(unsafe { mem::transmute::<*const (), unsafe extern "system" fn()>($f as *const ()) }),)*
                _ => None,
            }
        };
    }
    own!(
        vkGetPhysicalDeviceSurfaceSupportKHR,
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR,
        vkGetPhysicalDeviceSurfaceFormatsKHR,
        vkGetPhysicalDeviceSurfacePresentModesKHR,
        vkGetPhysicalDevicePresentRectanglesKHR,
        vkGetDeviceGroupPresentCapabilitiesKHR,
        vkGetDeviceGroupSurfacePresentModesKHR,
        vkCreateSwapchainKHR,
        vkDestroySwapchainKHR,
        vkGetSwapchainImagesKHR,
        vkAcquireNextImageKHR,
        vkAcquireNextImage2KHR,
        vkQueuePresentKHR
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Frames at this many a second, each costing this long
    fn settle(per_second: f64, cost_ms: f64, from: u32) -> u32 {
        let mut pacing = Pacing::default();
        let mut at = Instant::now();
        let mut quality = from;
        for _ in 0..400 {
            at += Duration::from_secs_f64(1.0 / per_second);
            pacing.frame(at, Duration::from_secs_f64(cost_ms / 1000.0), Duration::ZERO);
            quality = pacing.next_quality(quality);
        }
        quality
    }

    #[test]
    fn a_slow_game_gets_better_frames_than_a_fast_one_for_the_same_cost() {
        // 10 ms a frame is more than a 120 fps game can spare, and little to a 35 fps one
        assert_eq!(settle(35.0, 10.0, 30), BEST_QUALITY);
        assert_eq!(settle(120.0, 10.0, 60), LOWEST_QUALITY);
        // Faster than 120 fps, frames get what a 120 fps game's would
        assert_eq!(settle(400.0, 2.5, 30), BEST_QUALITY);
    }

    #[test]
    fn one_slow_frame_doesnt_bring_it_down() {
        let mut pacing = Pacing::default();
        let mut at = Instant::now();
        let mut quality = BEST_QUALITY;
        for frame in 0..400 {
            at += Duration::from_millis(28);
            // Every tenth takes 3 times as long to cross, as a link's do
            let cost = if frame % 10 == 9 { 24 } else { 8 };
            pacing.frame(at, Duration::from_millis(cost), Duration::ZERO);
            quality = pacing.next_quality(quality);
        }
        assert_eq!(quality, BEST_QUALITY);
    }

    #[test]
    fn frames_cost_no_more_than_the_time_between_them_can_carry() {
        assert_eq!(settle(35.0, 25.0, 90), LOWEST_QUALITY);
        // Between raising and lowering it stays where it is
        assert_eq!(settle(35.0, 15.0, 60), 60);
    }

    /// A program that could render `could` frames a second, when a frame takes `on_its_way_ms` there and
    /// whatever is slowest on the way carries `carried` a second. Returns how many were on their way at a
    /// time for most of 6000 frames, and the frame rate at the end.
    fn frames_settle(could: f64, on_its_way_ms: f64, carried: f64, most: usize) -> (usize, f64) {
        let mut pacing = Pacing::default();
        let mut on_the_way = OnTheWay::new(FRAMES_ON_THE_WAY.min(most));
        let mut at = Instant::now();
        let mut limit = on_the_way.limit;
        let mut frames_at = [0u32; MOST_FRAMES_ON_THE_WAY + 1];
        let mut rate = 0.0;
        for _ in 0..6000 {
            // Held to what that many on their way allow, and to what's carried
            rate = could.min(limit as f64 * 1000.0 / on_its_way_ms).min(carried);
            let interval = 1.0 / rate;
            at += Duration::from_secs_f64(interval);
            let waited = interval - 1.0 / could;
            pacing.frame(at, Duration::from_millis(1), Duration::from_secs_f64(waited.max(0.0)));
            limit = on_the_way.frame(&pacing, most);
            frames_at[limit] += 1;
        }
        let most_often = (0..frames_at.len()).max_by_key(|&limit| frames_at[limit]).unwrap();
        (most_often, rate)
    }

    #[test]
    fn as_many_frames_are_on_their_way_as_make_the_program_faster() {
        let endless = f64::INFINITY;
        // 50 ms on its way holds a game that could do 47 fps to 40 with 2: one more frees it
        let (limit, rate) = frames_settle(47.0, 50.0, endless, 4);
        assert_eq!((limit, rate.round()), (3, 47.0));
        // A game at 30 fps never waits, and a quick link needs no more either
        assert_eq!(frames_settle(30.0, 50.0, endless, 4).0, 2);
        assert_eq!(frames_settle(120.0, 10.0, endless, 4).0, 2);
        // One that could do 120 with 27 ms on the way takes all there's room for
        let (limit, rate) = frames_settle(120.0, 27.0, endless, 4);
        assert_eq!((limit, rate.round()), (4, 120.0));
        assert_eq!(frames_settle(120.0, 27.0, endless, 3).0, 3);
        assert_eq!(frames_settle(120.0, 27.0, endless, 1).0, 1);
    }

    #[test]
    fn more_at_once_arent_kept_when_something_on_the_way_cant_keep_up() {
        // The link carries 74 frames a second, which 2 on their way reach: 3 would only queue there
        assert_eq!(frames_settle(120.0, 27.0, 74.0, 4).0, 2);
        // It carries enough for 3 to help, and not for 4 to
        assert_eq!(frames_settle(120.0, 27.0, 100.0, 4).0, 3);
    }

    #[test]
    fn only_decoding_time_that_quality_can_change_counts() {
        let mut showing = Showing { decoder: None, waiting_for_key: true, complaints: 0, floor: 0.0 };
        let ms = |ms: f64| Duration::from_secs_f64(ms / 1000.0);
        // Pictures that take 11 ms however coarse cost nothing
        for _ in 0..50 {
            assert!(showing.over_floor(ms(11.0)) < ms(0.2));
        }
        // Fine ones with a lot to read cost what they take over that
        let over = showing.over_floor(ms(18.0));
        assert!(over > ms(6.5) && over < ms(7.5), "{over:?}");
        // A busier computer stops counting after a while
        for _ in 0..100 {
            showing.over_floor(ms(18.0));
        }
        assert!(showing.over_floor(ms(18.0)) < ms(0.2));
        // And a quieter one counts from where it is at once
        assert!(showing.over_floor(ms(5.0)) < ms(0.01));
        assert!(showing.over_floor(ms(7.0)) > ms(1.9));
    }

    #[test]
    fn frames_that_set_the_pace_bring_the_quality_down() {
        // If frames crossing were what held the program back, the time between them would be what one takes
        assert_eq!(settle(1000.0 / 14.0, 14.0, 90), LOWEST_QUALITY);
    }
}
