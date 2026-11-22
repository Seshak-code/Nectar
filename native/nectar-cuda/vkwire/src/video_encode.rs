//! Frames into H.264 on the lender's GPU, with the encoder the card has (Vulkan Video), which any
//! vendor's driver offers the same way. A frame comes as NV12 in a buffer, which `video_shader.rs` fills,
//! and leaves as the bytes of one picture, which `video_decode.rs` or any H.264 decoder takes.
//!
//! The stream is as plain as a game's frames allow: one picture in, one out, each coded against the one
//! before, with no pictures out of order, so a decoder can show each as it arrives. A whole picture (a key
//! frame) comes first and whenever asked for, which is what a borrower does after losing one.
//!
//! How fine the pictures are is a fixed quantizer chosen from the frame's quality, which keeps a frame's
//! size following the picture and leaves the choice of quality where it is for the other coder.
//!
//! Everything here runs on the queue family that encodes, which can also copy: the NV12 comes into the
//! encoder's own picture there. The buffer it comes from has to be shared with that family.

use std::ffi::CStr;
use std::{mem, ptr};

use nectar_vk_sys::*;

use crate::h264::{
    CHROMA_420, LEVEL_5_2, NO_REFERENCE_PICTURE, PICTURE_IDR, PICTURE_IDR_FLAG, PICTURE_IS_REFERENCE, PICTURE_P, POC_TYPE_2, PPS_DEBLOCKING_FILTER_CONTROL_PRESENT, PPS_ENTROPY_CODING_MODE,
    PPS_TRANSFORM_8X8_MODE, PROFILE_HIGH, SLICE_I, SLICE_P, SPS_DIRECT_8X8_INFERENCE, SPS_FRAME_CROPPING, SPS_FRAME_MBS_ONLY, SPS_VUI_PARAMETERS_PRESENT, StdVideoEncodeH264PictureInfo,
    StdVideoEncodeH264ReferenceInfo, StdVideoEncodeH264ReferenceListsInfo, StdVideoEncodeH264SliceHeader, StdVideoH264PictureParameterSet, StdVideoH264SequenceParameterSet,
    StdVideoH264SequenceParameterSetVui, VUI_BITSTREAM_RESTRICTION, VUI_COLOR_DESCRIPTION_PRESENT, VUI_VIDEO_FULL_RANGE, VUI_VIDEO_SIGNAL_TYPE_PRESENT, VkVideoEncodeH264CapabilitiesKHR,
    VkVideoEncodeH264DpbSlotInfoKHR, VkVideoEncodeH264NaluSliceInfoKHR, VkVideoEncodeH264PictureInfoKHR, VkVideoEncodeH264ProfileInfoKHR, VkVideoEncodeH264SessionParametersAddInfoKHR,
    VkVideoEncodeH264SessionParametersCreateInfoKHR, VkVideoEncodeH264SessionParametersGetInfoKHR,
};
use crate::video_shader::{coded_size, nv12_size};

/// What the device has to be created with
pub const DEVICE_EXTENSIONS: [&CStr; 4] = [c"VK_KHR_synchronization2", c"VK_KHR_video_queue", c"VK_KHR_video_encode_queue", c"VK_KHR_video_encode_h264"];

/// Finds a function of the instance's, or of the device's, by name
pub type Lookup<'a> = &'a dyn Fn(&CStr) -> PFN_vkVoidFunction;

const HOST_VISIBLE: u32 = 0x2;
const HOST_COHERENT: u32 = 0x4;
const HOST_CACHED: u32 = 0x8;
const NV12: VkFormat = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
/// How long a picture may take before the encoder is given up on
const PICTURE_WAIT: u64 = 2_000_000_000;
/// The quantizers the two ends of quality stand for. 18 is where differences stop being seen; each 6 more
/// halves the detail kept.
const FINEST: f32 = 18.0;
const COARSEST: f32 = 45.0;
/// frame_num counts in 16 bits
const LOG2_FRAME_NUMBERS: u8 = 16;

macro_rules! calls {
    ($kind:ident { $($name:ident: $pfn:ty,)* }) => {
        #[allow(non_snake_case)]
        struct $kind {
            $($name: $pfn,)*
        }

        impl $kind {
            unsafe fn load(find: Lookup) -> Result<Self, String> {
                Ok(Self {
                    $($name: unsafe { mem::transmute::<PFN_vkVoidFunction, Option<$pfn>>(find(CStr::from_bytes_with_nul(concat!(stringify!($name), "\0").as_bytes()).unwrap())) }
                        .ok_or_else(|| format!("no {}", stringify!($name)))?,)*
                })
            }
        }
    };
}

calls!(InstanceCalls {
    vkGetPhysicalDeviceQueueFamilyProperties: PFN_vkGetPhysicalDeviceQueueFamilyProperties,
    vkGetPhysicalDeviceMemoryProperties: PFN_vkGetPhysicalDeviceMemoryProperties,
    vkGetPhysicalDeviceVideoCapabilitiesKHR: PFN_vkGetPhysicalDeviceVideoCapabilitiesKHR,
    vkGetPhysicalDeviceVideoFormatPropertiesKHR: PFN_vkGetPhysicalDeviceVideoFormatPropertiesKHR,
});

calls!(Calls {
    vkCreateVideoSessionKHR: PFN_vkCreateVideoSessionKHR,
    vkDestroyVideoSessionKHR: PFN_vkDestroyVideoSessionKHR,
    vkGetVideoSessionMemoryRequirementsKHR: PFN_vkGetVideoSessionMemoryRequirementsKHR,
    vkBindVideoSessionMemoryKHR: PFN_vkBindVideoSessionMemoryKHR,
    vkCreateVideoSessionParametersKHR: PFN_vkCreateVideoSessionParametersKHR,
    vkDestroyVideoSessionParametersKHR: PFN_vkDestroyVideoSessionParametersKHR,
    vkGetEncodedVideoSessionParametersKHR: PFN_vkGetEncodedVideoSessionParametersKHR,
    vkCmdBeginVideoCodingKHR: PFN_vkCmdBeginVideoCodingKHR,
    vkCmdControlVideoCodingKHR: PFN_vkCmdControlVideoCodingKHR,
    vkCmdEncodeVideoKHR: PFN_vkCmdEncodeVideoKHR,
    vkCmdEndVideoCodingKHR: PFN_vkCmdEndVideoCodingKHR,
    vkCreateImage: PFN_vkCreateImage,
    vkDestroyImage: PFN_vkDestroyImage,
    vkGetImageMemoryRequirements: PFN_vkGetImageMemoryRequirements,
    vkBindImageMemory: PFN_vkBindImageMemory,
    vkCreateImageView: PFN_vkCreateImageView,
    vkDestroyImageView: PFN_vkDestroyImageView,
    vkCreateBuffer: PFN_vkCreateBuffer,
    vkDestroyBuffer: PFN_vkDestroyBuffer,
    vkGetBufferMemoryRequirements: PFN_vkGetBufferMemoryRequirements,
    vkBindBufferMemory: PFN_vkBindBufferMemory,
    vkAllocateMemory: PFN_vkAllocateMemory,
    vkFreeMemory: PFN_vkFreeMemory,
    vkMapMemory: PFN_vkMapMemory,
    vkCreateCommandPool: PFN_vkCreateCommandPool,
    vkDestroyCommandPool: PFN_vkDestroyCommandPool,
    vkAllocateCommandBuffers: PFN_vkAllocateCommandBuffers,
    vkResetCommandPool: PFN_vkResetCommandPool,
    vkBeginCommandBuffer: PFN_vkBeginCommandBuffer,
    vkEndCommandBuffer: PFN_vkEndCommandBuffer,
    vkCmdPipelineBarrier: PFN_vkCmdPipelineBarrier,
    vkCmdCopyBufferToImage: PFN_vkCmdCopyBufferToImage,
    vkCreateQueryPool: PFN_vkCreateQueryPool,
    vkDestroyQueryPool: PFN_vkDestroyQueryPool,
    vkCmdResetQueryPool: PFN_vkCmdResetQueryPool,
    vkCmdBeginQuery: PFN_vkCmdBeginQuery,
    vkCmdEndQuery: PFN_vkCmdEndQuery,
    vkGetQueryPoolResults: PFN_vkGetQueryPoolResults,
    vkCreateFence: PFN_vkCreateFence,
    vkDestroyFence: PFN_vkDestroyFence,
    vkWaitForFences: PFN_vkWaitForFences,
    vkResetFences: PFN_vkResetFences,
    vkQueueSubmit: PFN_vkQueueSubmit,
});

fn check(what: &str, result: VkResult) -> Result<(), String> {
    if result < 0 { Err(format!("{what}: VkResult {result}")) } else { Ok(()) }
}

/// What kind of video this is, which every object made for it has to be told the same way. The parts
/// point at each other, so they stay where they are.
struct Profile {
    usage: VkVideoEncodeUsageInfoKHR,
    h264: VkVideoEncodeH264ProfileInfoKHR,
    profile: VkVideoProfileInfoKHR,
    list: VkVideoProfileListInfoKHR,
}

impl Profile {
    fn new() -> Box<Self> {
        let mut profile = Box::new(Profile {
            usage: VkVideoEncodeUsageInfoKHR {
                sType: VK_STRUCTURE_TYPE_VIDEO_ENCODE_USAGE_INFO_KHR,
                pNext: ptr::null(),
                videoUsageHints: VK_VIDEO_ENCODE_USAGE_STREAMING_BIT_KHR,
                videoContentHints: VK_VIDEO_ENCODE_CONTENT_RENDERED_BIT_KHR,
                tuningMode: VK_VIDEO_ENCODE_TUNING_MODE_ULTRA_LOW_LATENCY_KHR,
            },
            h264: VkVideoEncodeH264ProfileInfoKHR { sType: VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_PROFILE_INFO_KHR, pNext: ptr::null(), stdProfileIdc: PROFILE_HIGH },
            profile: VkVideoProfileInfoKHR {
                sType: VK_STRUCTURE_TYPE_VIDEO_PROFILE_INFO_KHR,
                pNext: ptr::null(),
                videoCodecOperation: VK_VIDEO_CODEC_OPERATION_ENCODE_H264_BIT_KHR,
                chromaSubsampling: VK_VIDEO_CHROMA_SUBSAMPLING_420_BIT_KHR,
                lumaBitDepth: VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR,
                chromaBitDepth: VK_VIDEO_COMPONENT_BIT_DEPTH_8_BIT_KHR,
            },
            list: VkVideoProfileListInfoKHR { sType: VK_STRUCTURE_TYPE_VIDEO_PROFILE_LIST_INFO_KHR, pNext: ptr::null(), profileCount: 1, pProfiles: ptr::null() },
        });
        profile.h264.pNext = (&raw const profile.usage).cast();
        profile.profile.pNext = (&raw const profile.h264).cast();
        profile.list.pProfiles = &raw const profile.profile;
        profile
    }
}

/// What the card's encoder can do for this kind of video
struct Capabilities {
    video: VkVideoCapabilitiesKHR,
    encode: VkVideoEncodeCapabilitiesKHR,
    h264: VkVideoEncodeH264CapabilitiesKHR,
}

unsafe fn capabilities(calls: &InstanceCalls, physical: VkPhysicalDevice, profile: &Profile) -> Result<Box<Capabilities>, String> {
    unsafe {
        let mut all: Box<Capabilities> = Box::new(mem::zeroed());
        all.h264.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_CAPABILITIES_KHR;
        all.encode.sType = VK_STRUCTURE_TYPE_VIDEO_ENCODE_CAPABILITIES_KHR;
        all.encode.pNext = (&raw mut all.h264).cast();
        all.video.sType = VK_STRUCTURE_TYPE_VIDEO_CAPABILITIES_KHR;
        all.video.pNext = (&raw mut all.encode).cast();
        check("the card doesn't encode H.264", (calls.vkGetPhysicalDeviceVideoCapabilitiesKHR)(physical, &profile.profile, &mut all.video))?;
        Ok(all)
    }
}

/// The queue family that encodes H.264 on this card, if one does
///
/// # Safety
/// `find_instance` gives the functions of the instance `physical` belongs to.
pub unsafe fn encode_family(find_instance: Lookup, physical: VkPhysicalDevice) -> Option<u32> {
    unsafe {
        let calls = InstanceCalls::load(find_instance).ok()?;
        let mut count = 0;
        (calls.vkGetPhysicalDeviceQueueFamilyProperties)(physical, &mut count, ptr::null_mut());
        let mut families: Vec<VkQueueFamilyProperties> = vec![mem::zeroed(); count as usize];
        (calls.vkGetPhysicalDeviceQueueFamilyProperties)(physical, &mut count, families.as_mut_ptr());
        let family = families.iter().take(count as usize).position(|family| family.queueFlags & VK_QUEUE_VIDEO_ENCODE_BIT_KHR != 0 && family.queueCount > 0)?;
        capabilities(&calls, physical, &Profile::new()).ok()?;
        Some(family as u32)
    }
}

/// The quantizer a frame's quality stands for
pub fn quantizer(quality: u32, least: i32, most: i32) -> i32 {
    let quality = quality.clamp(10, 100) as f32;
    let qp = COARSEST - (quality - 10.0) * (COARSEST - FINEST) / 90.0;
    (qp.round() as i32).clamp(least.max(0), most.min(51))
}

/// One reference picture's place: both of them are layers of one image
fn picture_resource(view: VkImageView, width: u32, height: u32) -> VkVideoPictureResourceInfoKHR {
    VkVideoPictureResourceInfoKHR {
        sType: VK_STRUCTURE_TYPE_VIDEO_PICTURE_RESOURCE_INFO_KHR,
        pNext: ptr::null(),
        codedOffset: VkOffset2D { x: 0, y: 0 },
        codedExtent: VkExtent2D { width, height },
        baseArrayLayer: 0,
        imageViewBinding: view,
    }
}

pub struct Encoder {
    calls: Calls,
    device: VkDevice,
    queue: VkQueue,
    profile: Box<Profile>,
    session: VkVideoSessionKHR,
    session_memory: Vec<VkDeviceMemory>,
    parameters: VkVideoSessionParametersKHR,
    /// The picture being coded
    source: VkImage,
    source_memory: VkDeviceMemory,
    source_view: VkImageView,
    /// The pictures as a decoder will have them: the last, and the one being made
    pictures: VkImage,
    pictures_memory: VkDeviceMemory,
    picture_views: [VkImageView; 2],
    coded: VkBuffer,
    coded_memory: VkDeviceMemory,
    coded_pointer: *const u8,
    coded_size: u64,
    pool: VkCommandPool,
    commands: VkCommandBuffer,
    queries: VkQueryPool,
    fence: VkFence,
    coded_width: u32,
    coded_height: u32,
    quantizers: (i32, i32),
    headers: Vec<u8>,
    frames: u64,
    since_key: u32,
    keys: u16,
    last: StdVideoEncodeH264ReferenceInfo,
    /// A picture was submitted and hasn't been seen to finish
    pending: bool,
    out: Vec<u8>,
}

// Its pointers are to memory and objects of its own
unsafe impl Send for Encoder {}

impl Encoder {
    /// An encoder for frames of this size, on a device created with `DEVICE_EXTENSIONS` and a queue of
    /// `family`, which `encode_family` chose.
    ///
    /// # Safety
    /// The lookups give the functions of the instance and of the device, and the handles are theirs.
    /// `queue` is used by nothing else while a frame is being encoded.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new(find_instance: Lookup, find_device: Lookup, physical: VkPhysicalDevice, device: VkDevice, family: u32, queue: VkQueue, width: u32, height: u32) -> Result<Self, String> {
        unsafe {
            let instance = InstanceCalls::load(find_instance)?;
            let calls = Calls::load(find_device)?;
            let (coded_width, coded_height) = coded_size(width, height);
            let mut encoder = Encoder {
                calls,
                device,
                queue,
                profile: Profile::new(),
                session: 0,
                session_memory: Vec::new(),
                parameters: 0,
                source: 0,
                source_memory: 0,
                source_view: 0,
                pictures: 0,
                pictures_memory: 0,
                picture_views: [0; 2],
                coded: 0,
                coded_memory: 0,
                coded_pointer: ptr::null(),
                coded_size: 0,
                pool: 0,
                commands: ptr::null_mut(),
                queries: 0,
                fence: 0,
                coded_width,
                coded_height,
                quantizers: (0, 51),
                headers: Vec::new(),
                frames: 0,
                since_key: 0,
                keys: 0,
                last: StdVideoEncodeH264ReferenceInfo::default(),
                pending: false,
                out: Vec::new(),
            };
            match encoder.build(&instance, physical, family, width, height) {
                Ok(()) => Ok(encoder),
                Err(error) => {
                    encoder.destroy();
                    Err(error)
                }
            }
        }
    }

    unsafe fn build(&mut self, instance: &InstanceCalls, physical: VkPhysicalDevice, family: u32, width: u32, height: u32) -> Result<(), String> {
        unsafe {
            let (device, coded_width, coded_height) = (self.device, self.coded_width, self.coded_height);
            let capabilities = capabilities(instance, physical, &self.profile)?;
            let (least, most) = (capabilities.video.minCodedExtent, capabilities.video.maxCodedExtent);
            if coded_width < least.width || coded_height < least.height || coded_width > most.width || coded_height > most.height {
                return Err(format!("the encoder takes {}x{} to {}x{}, and the frames are {width}x{height}", least.width, least.height, most.width, most.height));
            }
            if capabilities.encode.rateControlModes & VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR == 0 {
                return Err("the encoder has no fixed quantizer".to_string());
            }
            if capabilities.video.maxDpbSlots < 2 || capabilities.video.maxActiveReferencePictures < 1 {
                return Err("the encoder keeps no picture to code the next against".to_string());
            }
            self.quantizers = (capabilities.h264.minQp, capabilities.h264.maxQp);

            // The formats its pictures can have, for what's done with each
            for (usage, what) in [
                (VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR | VK_IMAGE_USAGE_TRANSFER_DST_BIT, "the encoder's pictures can't be copied into"),
                (VK_IMAGE_USAGE_VIDEO_ENCODE_DPB_BIT_KHR, "the encoder keeps no pictures of this kind"),
            ] {
                let info = VkPhysicalDeviceVideoFormatInfoKHR { sType: VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VIDEO_FORMAT_INFO_KHR, pNext: (&raw const self.profile.list).cast(), imageUsage: usage };
                let mut count = 0;
                check(what, (instance.vkGetPhysicalDeviceVideoFormatPropertiesKHR)(physical, &info, &mut count, ptr::null_mut()))?;
                let mut formats: Vec<VkVideoFormatPropertiesKHR> = vec![mem::zeroed(); count as usize];
                for format in &mut formats {
                    format.sType = VK_STRUCTURE_TYPE_VIDEO_FORMAT_PROPERTIES_KHR;
                }
                check(what, (instance.vkGetPhysicalDeviceVideoFormatPropertiesKHR)(physical, &info, &mut count, formats.as_mut_ptr()))?;
                if !formats.iter().take(count as usize).any(|format| format.format == NV12 && format.imageTiling == VK_IMAGE_TILING_OPTIMAL) {
                    return Err(format!("{what}: no NV12 among {count} formats"));
                }
            }

            let session_info = VkVideoSessionCreateInfoKHR {
                sType: VK_STRUCTURE_TYPE_VIDEO_SESSION_CREATE_INFO_KHR,
                pNext: ptr::null(),
                queueFamilyIndex: family,
                flags: 0,
                pVideoProfile: &self.profile.profile,
                pictureFormat: NV12,
                maxCodedExtent: VkExtent2D { width: coded_width, height: coded_height },
                referencePictureFormat: NV12,
                maxDpbSlots: 2,
                maxActiveReferencePictures: 1,
                pStdHeaderVersion: &capabilities.video.stdHeaderVersion,
            };
            check("vkCreateVideoSessionKHR", (self.calls.vkCreateVideoSessionKHR)(device, &session_info, ptr::null(), &mut self.session))?;

            let mut memory_properties: VkPhysicalDeviceMemoryProperties = mem::zeroed();
            (instance.vkGetPhysicalDeviceMemoryProperties)(physical, &mut memory_properties);
            let memory_type = |bits: u32, wanted: &[u32]| -> Result<u32, String> {
                wanted
                    .iter()
                    .find_map(|&flags| (0..memory_properties.memoryTypeCount).find(|&index| bits & (1 << index) != 0 && memory_properties.memoryTypes[index as usize].propertyFlags & flags == flags))
                    .ok_or_else(|| "no memory of the kind the encoder needs".to_string())
            };
            let allocate = |calls: &Calls, needs: &VkMemoryRequirements, wanted: &[u32]| -> Result<VkDeviceMemory, String> {
                let info = VkMemoryAllocateInfo {
                    sType: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                    pNext: ptr::null(),
                    allocationSize: needs.size,
                    memoryTypeIndex: memory_type(needs.memoryTypeBits, wanted)?,
                };
                let mut memory = 0;
                check("vkAllocateMemory", (calls.vkAllocateMemory)(device, &info, ptr::null(), &mut memory))?;
                Ok(memory)
            };

            // The session's own memory
            let mut count = 0;
            check("vkGetVideoSessionMemoryRequirementsKHR", (self.calls.vkGetVideoSessionMemoryRequirementsKHR)(device, self.session, &mut count, ptr::null_mut()))?;
            let mut needs: Vec<VkVideoSessionMemoryRequirementsKHR> = vec![mem::zeroed(); count as usize];
            for need in &mut needs {
                need.sType = VK_STRUCTURE_TYPE_VIDEO_SESSION_MEMORY_REQUIREMENTS_KHR;
            }
            check("vkGetVideoSessionMemoryRequirementsKHR", (self.calls.vkGetVideoSessionMemoryRequirementsKHR)(device, self.session, &mut count, needs.as_mut_ptr()))?;
            let mut binds = Vec::new();
            for need in needs.iter().take(count as usize) {
                let memory = allocate(&self.calls, &need.memoryRequirements, &[0])?;
                self.session_memory.push(memory);
                binds.push(VkBindVideoSessionMemoryInfoKHR {
                    sType: VK_STRUCTURE_TYPE_BIND_VIDEO_SESSION_MEMORY_INFO_KHR,
                    pNext: ptr::null(),
                    memoryBindIndex: need.memoryBindIndex,
                    memory,
                    memoryOffset: 0,
                    memorySize: need.memoryRequirements.size,
                });
            }
            if !binds.is_empty() {
                check("vkBindVideoSessionMemoryKHR", (self.calls.vkBindVideoSessionMemoryKHR)(device, self.session, binds.len() as u32, binds.as_ptr()))?;
            }

            // What a decoder needs to know before the first picture
            let crop = (coded_width != width || coded_height != height) as u32 * SPS_FRAME_CROPPING;
            let vui = StdVideoH264SequenceParameterSetVui {
                flags: VUI_VIDEO_SIGNAL_TYPE_PRESENT | VUI_VIDEO_FULL_RANGE | VUI_COLOR_DESCRIPTION_PRESENT | VUI_BITSTREAM_RESTRICTION,
                aspect_ratio_idc: 0,
                sar_width: 0,
                sar_height: 0,
                // No format, and BT.709 for all three
                video_format: 5,
                colour_primaries: 1,
                transfer_characteristics: 1,
                matrix_coefficients: 1,
                num_units_in_tick: 0,
                time_scale: 0,
                // No pictures out of order, and one kept
                max_num_reorder_frames: 0,
                max_dec_frame_buffering: 1,
                chroma_sample_loc_type_top_field: 0,
                chroma_sample_loc_type_bottom_field: 0,
                reserved1: 0,
                pHrdParameters: ptr::null(),
            };
            let sps = StdVideoH264SequenceParameterSet {
                flags: SPS_DIRECT_8X8_INFERENCE | SPS_FRAME_MBS_ONLY | SPS_VUI_PARAMETERS_PRESENT | crop,
                profile_idc: PROFILE_HIGH,
                level_idc: LEVEL_5_2.min(capabilities.h264.maxLevelIdc),
                chroma_format_idc: CHROMA_420,
                seq_parameter_set_id: 0,
                bit_depth_luma_minus8: 0,
                bit_depth_chroma_minus8: 0,
                log2_max_frame_num_minus4: LOG2_FRAME_NUMBERS - 4,
                pic_order_cnt_type: POC_TYPE_2,
                offset_for_non_ref_pic: 0,
                offset_for_top_to_bottom_field: 0,
                log2_max_pic_order_cnt_lsb_minus4: 0,
                num_ref_frames_in_pic_order_cnt_cycle: 0,
                max_num_ref_frames: 1,
                reserved1: 0,
                pic_width_in_mbs_minus1: coded_width / 16 - 1,
                pic_height_in_map_units_minus1: coded_height / 16 - 1,
                frame_crop_left_offset: 0,
                // In pairs of pixels
                frame_crop_right_offset: (coded_width - width) / 2,
                frame_crop_top_offset: 0,
                frame_crop_bottom_offset: (coded_height - height) / 2,
                reserved2: 0,
                pOffsetForRefFrame: ptr::null(),
                pScalingLists: ptr::null(),
                pSequenceParameterSetVui: &vui,
            };
            let can = capabilities.h264.stdSyntaxFlags;
            let mut pps_flags = PPS_DEBLOCKING_FILTER_CONTROL_PRESENT;
            if can & VK_VIDEO_ENCODE_H264_STD_TRANSFORM_8X8_MODE_FLAG_SET_BIT_KHR != 0 {
                pps_flags |= PPS_TRANSFORM_8X8_MODE;
            }
            if can & VK_VIDEO_ENCODE_H264_STD_ENTROPY_CODING_MODE_FLAG_SET_BIT_KHR != 0 {
                pps_flags |= PPS_ENTROPY_CODING_MODE;
            }
            let pps = StdVideoH264PictureParameterSet {
                flags: pps_flags,
                seq_parameter_set_id: 0,
                pic_parameter_set_id: 0,
                num_ref_idx_l0_default_active_minus1: 0,
                num_ref_idx_l1_default_active_minus1: 0,
                weighted_bipred_idc: 0,
                pic_init_qp_minus26: 0,
                pic_init_qs_minus26: 0,
                chroma_qp_index_offset: 0,
                second_chroma_qp_index_offset: 0,
                pScalingLists: ptr::null(),
            };
            let add = VkVideoEncodeH264SessionParametersAddInfoKHR {
                sType: VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_ADD_INFO_KHR,
                pNext: ptr::null(),
                stdSPSCount: 1,
                pStdSPSs: &sps,
                stdPPSCount: 1,
                pStdPPSs: &pps,
            };
            let h264_parameters = VkVideoEncodeH264SessionParametersCreateInfoKHR {
                sType: VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_CREATE_INFO_KHR,
                pNext: ptr::null(),
                maxStdSPSCount: 1,
                maxStdPPSCount: 1,
                pParametersAddInfo: &add,
            };
            let parameters_info = VkVideoSessionParametersCreateInfoKHR {
                sType: VK_STRUCTURE_TYPE_VIDEO_SESSION_PARAMETERS_CREATE_INFO_KHR,
                pNext: (&raw const h264_parameters).cast(),
                flags: 0,
                videoSessionParametersTemplate: 0,
                videoSession: self.session,
            };
            check("vkCreateVideoSessionParametersKHR", (self.calls.vkCreateVideoSessionParametersKHR)(device, &parameters_info, ptr::null(), &mut self.parameters))?;

            // As the driver writes them, which is what its pictures go with
            let which = VkVideoEncodeH264SessionParametersGetInfoKHR {
                sType: VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_SESSION_PARAMETERS_GET_INFO_KHR,
                pNext: ptr::null(),
                writeStdSPS: 1,
                writeStdPPS: 1,
                stdSPSId: 0,
                stdPPSId: 0,
            };
            let get = VkVideoEncodeSessionParametersGetInfoKHR {
                sType: VK_STRUCTURE_TYPE_VIDEO_ENCODE_SESSION_PARAMETERS_GET_INFO_KHR,
                pNext: (&raw const which).cast(),
                videoSessionParameters: self.parameters,
            };
            let mut size = 0usize;
            check("the stream's headers", (self.calls.vkGetEncodedVideoSessionParametersKHR)(device, &get, ptr::null_mut(), &mut size, ptr::null_mut()))?;
            self.headers = vec![0u8; size];
            check("the stream's headers", (self.calls.vkGetEncodedVideoSessionParametersKHR)(device, &get, ptr::null_mut(), &mut size, self.headers.as_mut_ptr().cast()))?;
            self.headers.truncate(size);
            if !self.headers.starts_with(&[0, 0, 1]) && !self.headers.starts_with(&[0, 0, 0, 1]) {
                return Err("the stream's headers aren't a byte stream".to_string());
            }

            // The picture being coded, and the two kept
            let image = |calls: &Calls, usage: u32, layers: u32| -> Result<(VkImage, VkDeviceMemory), String> {
                let info = VkImageCreateInfo {
                    sType: VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                    pNext: (&raw const self.profile.list).cast(),
                    flags: 0,
                    imageType: VK_IMAGE_TYPE_2D,
                    format: NV12,
                    extent: VkExtent3D { width: coded_width, height: coded_height, depth: 1 },
                    mipLevels: 1,
                    arrayLayers: layers,
                    samples: VK_SAMPLE_COUNT_1_BIT,
                    tiling: VK_IMAGE_TILING_OPTIMAL,
                    usage,
                    sharingMode: VK_SHARING_MODE_EXCLUSIVE,
                    queueFamilyIndexCount: 0,
                    pQueueFamilyIndices: ptr::null(),
                    initialLayout: VK_IMAGE_LAYOUT_UNDEFINED,
                };
                let mut image = 0;
                check("vkCreateImage", (calls.vkCreateImage)(device, &info, ptr::null(), &mut image))?;
                let mut needs: VkMemoryRequirements = mem::zeroed();
                (calls.vkGetImageMemoryRequirements)(device, image, &mut needs);
                let memory = match allocate(calls, &needs, &[0]) {
                    Ok(memory) => memory,
                    Err(error) => {
                        (calls.vkDestroyImage)(device, image, ptr::null());
                        return Err(error);
                    }
                };
                if let Err(error) = check("vkBindImageMemory", (calls.vkBindImageMemory)(device, image, memory, 0)) {
                    (calls.vkDestroyImage)(device, image, ptr::null());
                    (calls.vkFreeMemory)(device, memory, ptr::null());
                    return Err(error);
                }
                Ok((image, memory))
            };
            let view = |calls: &Calls, image: VkImage, layer: u32| -> Result<VkImageView, String> {
                let info = VkImageViewCreateInfo {
                    sType: VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                    pNext: ptr::null(),
                    flags: 0,
                    image,
                    viewType: VK_IMAGE_VIEW_TYPE_2D,
                    format: NV12,
                    components: mem::zeroed(),
                    subresourceRange: VkImageSubresourceRange { aspectMask: VK_IMAGE_ASPECT_COLOR_BIT, baseMipLevel: 0, levelCount: 1, baseArrayLayer: layer, layerCount: 1 },
                };
                let mut view = 0;
                check("vkCreateImageView", (calls.vkCreateImageView)(device, &info, ptr::null(), &mut view))?;
                Ok(view)
            };
            (self.source, self.source_memory) = image(&self.calls, VK_IMAGE_USAGE_VIDEO_ENCODE_SRC_BIT_KHR | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 1)?;
            self.source_view = view(&self.calls, self.source, 0)?;
            (self.pictures, self.pictures_memory) = image(&self.calls, VK_IMAGE_USAGE_VIDEO_ENCODE_DPB_BIT_KHR, 2)?;
            for layer in 0..2 {
                self.picture_views[layer] = view(&self.calls, self.pictures, layer as u32)?;
            }

            // Where the coded picture comes out: no picture codes to more than it started as
            let alignment = capabilities.video.minBitstreamBufferSizeAlignment.max(1);
            self.coded_size = nv12_size(coded_width, coded_height).next_multiple_of(alignment);
            let buffer_info = VkBufferCreateInfo {
                sType: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                pNext: (&raw const self.profile.list).cast(),
                flags: 0,
                size: self.coded_size,
                usage: VK_BUFFER_USAGE_VIDEO_ENCODE_DST_BIT_KHR,
                sharingMode: VK_SHARING_MODE_EXCLUSIVE,
                queueFamilyIndexCount: 0,
                pQueueFamilyIndices: ptr::null(),
            };
            check("vkCreateBuffer", (self.calls.vkCreateBuffer)(device, &buffer_info, ptr::null(), &mut self.coded))?;
            let mut needs: VkMemoryRequirements = mem::zeroed();
            (self.calls.vkGetBufferMemoryRequirements)(device, self.coded, &mut needs);
            self.coded_memory = allocate(&self.calls, &needs, &[HOST_VISIBLE | HOST_COHERENT | HOST_CACHED, HOST_VISIBLE | HOST_COHERENT])?;
            check("vkBindBufferMemory", (self.calls.vkBindBufferMemory)(device, self.coded, self.coded_memory, 0))?;
            let mut pointer = ptr::null_mut();
            check("vkMapMemory", (self.calls.vkMapMemory)(device, self.coded_memory, 0, self.coded_size, 0, &mut pointer))?;
            self.coded_pointer = pointer as *const u8;

            let pool_info = VkCommandPoolCreateInfo { sType: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, pNext: ptr::null(), flags: 0, queueFamilyIndex: family };
            check("vkCreateCommandPool", (self.calls.vkCreateCommandPool)(device, &pool_info, ptr::null(), &mut self.pool))?;
            let commands_info = VkCommandBufferAllocateInfo {
                sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                pNext: ptr::null(),
                commandPool: self.pool,
                level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount: 1,
            };
            check("vkAllocateCommandBuffers", (self.calls.vkAllocateCommandBuffers)(device, &commands_info, &mut self.commands))?;
            // Where in the buffer the picture is, and how long
            let feedback = VkQueryPoolVideoEncodeFeedbackCreateInfoKHR {
                sType: VK_STRUCTURE_TYPE_QUERY_POOL_VIDEO_ENCODE_FEEDBACK_CREATE_INFO_KHR,
                pNext: (&raw const self.profile.profile).cast(),
                encodeFeedbackFlags: VK_VIDEO_ENCODE_FEEDBACK_BITSTREAM_BUFFER_OFFSET_BIT_KHR | VK_VIDEO_ENCODE_FEEDBACK_BITSTREAM_BYTES_WRITTEN_BIT_KHR,
            };
            let query_info = VkQueryPoolCreateInfo {
                sType: VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                pNext: (&raw const feedback).cast(),
                flags: 0,
                queryType: VK_QUERY_TYPE_VIDEO_ENCODE_FEEDBACK_KHR,
                queryCount: 1,
                pipelineStatistics: 0,
            };
            check("vkCreateQueryPool", (self.calls.vkCreateQueryPool)(device, &query_info, ptr::null(), &mut self.queries))?;
            let fence_info = VkFenceCreateInfo { sType: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, pNext: ptr::null(), flags: 0 };
            check("vkCreateFence", (self.calls.vkCreateFence)(device, &fence_info, ptr::null(), &mut self.fence))?;
            Ok(())
        }
    }

    /// The size of the NV12 a frame comes as: whole 16-pixel blocks
    pub fn coded_size(&self) -> (u32, u32) {
        (self.coded_width, self.coded_height)
    }

    /// What a decoder needs before the first picture. Every key frame starts with them.
    pub fn headers(&self) -> &[u8] {
        &self.headers
    }

    unsafe fn layout(&self, image: VkImage, layers: u32, from: VkImageLayout, to: VkImageLayout, wrote: u32, writes: u32) {
        let barrier = VkImageMemoryBarrier {
            sType: VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            pNext: ptr::null(),
            srcAccessMask: wrote,
            dstAccessMask: writes,
            oldLayout: from,
            newLayout: to,
            srcQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
            image,
            subresourceRange: VkImageSubresourceRange { aspectMask: VK_IMAGE_ASPECT_COLOR_BIT, baseMipLevel: 0, levelCount: 1, baseArrayLayer: 0, layerCount: layers },
        };
        unsafe {
            (self.calls.vkCmdPipelineBarrier)(self.commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, ptr::null(), 0, ptr::null(), 1, &barrier);
        }
    }

    /// Codes the frame whose NV12 is in `nv12`, which whatever wrote it has finished with. Returns the
    /// picture's bytes and whether it's a key frame: one that needs none before it, which the first is,
    /// and any `key` asks for.
    ///
    /// # Safety
    /// `nv12` is a buffer of this device's, at least as large as the coded size's NV12, shared with the
    /// encoder's queue family.
    pub unsafe fn encode(&mut self, nv12: VkBuffer, quality: u32, key: bool) -> Result<(&[u8], bool), String> {
        unsafe {
            let calls = &self.calls;
            let (device, commands) = (self.device, self.commands);
            if self.pending {
                return Err("the encoder is still on a picture that took too long".to_string());
            }
            let first = self.frames == 0;
            let key = key || first || self.since_key >= 1 << 30;
            if key {
                self.since_key = 0;
                self.keys = self.keys.wrapping_add(1);
            }
            let (setup, reference) = ((self.frames % 2) as usize, ((self.frames + 1) % 2) as usize);
            let (width, height) = (self.coded_width, self.coded_height);

            check("vkResetCommandPool", (calls.vkResetCommandPool)(device, self.pool, 0))?;
            let begin = VkCommandBufferBeginInfo { sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, pNext: ptr::null(), flags: 0, pInheritanceInfo: ptr::null() };
            check("vkBeginCommandBuffer", (calls.vkBeginCommandBuffer)(commands, &begin))?;
            self.layout(self.source, 1, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
            if first {
                self.layout(self.pictures, 2, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_VIDEO_ENCODE_DPB_KHR, 0, 0);
            }
            let plane = |aspect: u32, offset: u64, width: u32, height: u32| VkBufferImageCopy {
                bufferOffset: offset,
                bufferRowLength: 0,
                bufferImageHeight: 0,
                imageSubresource: VkImageSubresourceLayers { aspectMask: aspect, mipLevel: 0, baseArrayLayer: 0, layerCount: 1 },
                imageOffset: VkOffset3D { x: 0, y: 0, z: 0 },
                imageExtent: VkExtent3D { width, height, depth: 1 },
            };
            // The luma, then the two chromas side by side at half the size each way
            let planes = [plane(VK_IMAGE_ASPECT_PLANE_0_BIT, 0, width, height), plane(VK_IMAGE_ASPECT_PLANE_1_BIT, width as u64 * height as u64, width / 2, height / 2)];
            (calls.vkCmdCopyBufferToImage)(commands, nv12, self.source, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 2, planes.as_ptr());
            self.layout(self.source, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_VIDEO_ENCODE_SRC_KHR, VK_ACCESS_TRANSFER_WRITE_BIT, 0);
            (calls.vkCmdResetQueryPool)(commands, self.queries, 0, 1);

            let kind = if key { PICTURE_IDR } else { PICTURE_P };
            let this = StdVideoEncodeH264ReferenceInfo {
                flags: 0,
                primary_pic_type: kind,
                FrameNum: self.since_key % (1 << LOG2_FRAME_NUMBERS),
                PicOrderCnt: (self.since_key * 2) as i32,
                long_term_pic_num: 0,
                long_term_frame_idx: 0,
                temporal_id: 0,
            };
            let resources = [picture_resource(self.picture_views[0], width, height), picture_resource(self.picture_views[1], width, height)];
            let setup_info = VkVideoEncodeH264DpbSlotInfoKHR { sType: VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_DPB_SLOT_INFO_KHR, pNext: ptr::null(), pStdReferenceInfo: &this };
            let reference_info = VkVideoEncodeH264DpbSlotInfoKHR { sType: VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_DPB_SLOT_INFO_KHR, pNext: ptr::null(), pStdReferenceInfo: &self.last };
            let slot = |index: i32, at: usize, info: &VkVideoEncodeH264DpbSlotInfoKHR| VkVideoReferenceSlotInfoKHR {
                sType: VK_STRUCTURE_TYPE_VIDEO_REFERENCE_SLOT_INFO_KHR,
                pNext: (info as *const VkVideoEncodeH264DpbSlotInfoKHR).cast(),
                slotIndex: index,
                pPictureResource: &resources[at],
            };
            let reference_slot = slot(reference as i32, reference, &reference_info);
            let setup_slot = slot(setup as i32, setup, &setup_info);
            // What's bound while coding: the picture coded against, if there is one, and the one being
            // made, which has no slot until it's made
            let bound = [reference_slot, slot(-1, setup, &setup_info)];
            let bound = if key { &bound[1..] } else { &bound[..] };
            let fixed = VkVideoEncodeRateControlInfoKHR {
                sType: VK_STRUCTURE_TYPE_VIDEO_ENCODE_RATE_CONTROL_INFO_KHR,
                pNext: ptr::null(),
                flags: 0,
                rateControlMode: VK_VIDEO_ENCODE_RATE_CONTROL_MODE_DISABLED_BIT_KHR,
                layerCount: 0,
                pLayers: ptr::null(),
                virtualBufferSizeInMs: 0,
                initialVirtualBufferSizeInMs: 0,
            };
            let begin_coding = VkVideoBeginCodingInfoKHR {
                sType: VK_STRUCTURE_TYPE_VIDEO_BEGIN_CODING_INFO_KHR,
                // What the session was last told, which at first is nothing
                pNext: if first { ptr::null() } else { (&raw const fixed).cast() },
                flags: 0,
                videoSession: self.session,
                videoSessionParameters: self.parameters,
                referenceSlotCount: bound.len() as u32,
                pReferenceSlots: bound.as_ptr(),
            };
            (calls.vkCmdBeginVideoCodingKHR)(commands, &begin_coding);
            if first {
                let control = VkVideoCodingControlInfoKHR {
                    sType: VK_STRUCTURE_TYPE_VIDEO_CODING_CONTROL_INFO_KHR,
                    pNext: (&raw const fixed).cast(),
                    flags: VK_VIDEO_CODING_CONTROL_RESET_BIT_KHR | VK_VIDEO_CODING_CONTROL_ENCODE_RATE_CONTROL_BIT_KHR,
                };
                (calls.vkCmdControlVideoCodingKHR)(commands, &control);
            }

            let mut lists: StdVideoEncodeH264ReferenceListsInfo = mem::zeroed();
            lists.RefPicList0 = [NO_REFERENCE_PICTURE; 32];
            lists.RefPicList1 = [NO_REFERENCE_PICTURE; 32];
            if !key {
                lists.RefPicList0[0] = reference as u8;
            }
            let picture = StdVideoEncodeH264PictureInfo {
                flags: PICTURE_IS_REFERENCE | if key { PICTURE_IDR_FLAG } else { 0 },
                seq_parameter_set_id: 0,
                pic_parameter_set_id: 0,
                idr_pic_id: self.keys,
                primary_pic_type: kind,
                frame_num: this.FrameNum,
                PicOrderCnt: this.PicOrderCnt,
                temporal_id: 0,
                reserved1: [0; 3],
                pRefLists: &lists,
            };
            let header = StdVideoEncodeH264SliceHeader {
                flags: 0,
                first_mb_in_slice: 0,
                slice_type: if key { SLICE_I } else { SLICE_P },
                slice_alpha_c0_offset_div2: 0,
                slice_beta_offset_div2: 0,
                slice_qp_delta: 0,
                reserved1: 0,
                cabac_init_idc: 0,
                disable_deblocking_filter_idc: 0,
                pWeightTable: ptr::null(),
            };
            let slice = VkVideoEncodeH264NaluSliceInfoKHR {
                sType: VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_NALU_SLICE_INFO_KHR,
                pNext: ptr::null(),
                constantQp: quantizer(quality, self.quantizers.0, self.quantizers.1),
                pStdSliceHeader: &header,
            };
            let h264_picture = VkVideoEncodeH264PictureInfoKHR {
                sType: VK_STRUCTURE_TYPE_VIDEO_ENCODE_H264_PICTURE_INFO_KHR,
                pNext: ptr::null(),
                naluSliceEntryCount: 1,
                pNaluSliceEntries: &slice,
                pStdPictureInfo: &picture,
                generatePrefixNalu: 0,
            };
            let encode = VkVideoEncodeInfoKHR {
                sType: VK_STRUCTURE_TYPE_VIDEO_ENCODE_INFO_KHR,
                pNext: (&raw const h264_picture).cast(),
                flags: 0,
                dstBuffer: self.coded,
                dstBufferOffset: 0,
                dstBufferRange: self.coded_size,
                srcPictureResource: picture_resource(self.source_view, width, height),
                pSetupReferenceSlot: &setup_slot,
                referenceSlotCount: !key as u32,
                pReferenceSlots: &reference_slot,
                precedingExternallyEncodedBytes: 0,
            };
            (calls.vkCmdBeginQuery)(commands, self.queries, 0, 0);
            (calls.vkCmdEncodeVideoKHR)(commands, &encode);
            (calls.vkCmdEndQuery)(commands, self.queries, 0);
            let end_coding = VkVideoEndCodingInfoKHR { sType: VK_STRUCTURE_TYPE_VIDEO_END_CODING_INFO_KHR, pNext: ptr::null(), flags: 0 };
            (calls.vkCmdEndVideoCodingKHR)(commands, &end_coding);
            check("vkEndCommandBuffer", (calls.vkEndCommandBuffer)(commands))?;

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
            check("vkQueueSubmit", (calls.vkQueueSubmit)(self.queue, 1, &submit, self.fence))?;
            self.pending = true;
            let waited = (calls.vkWaitForFences)(device, 1, &self.fence, 1, PICTURE_WAIT);
            if waited == VK_TIMEOUT {
                return Err("the encoder took too long over a picture".to_string());
            }
            check("vkWaitForFences", waited)?;
            self.pending = false;
            check("vkResetFences", (calls.vkResetFences)(device, 1, &self.fence))?;
            // Where it starts, how many bytes, and how it went
            let mut feedback = [0u32; 3];
            check(
                "vkGetQueryPoolResults",
                (calls.vkGetQueryPoolResults)(device, self.queries, 0, 1, 12, feedback.as_mut_ptr().cast(), 12, VK_QUERY_RESULT_WITH_STATUS_BIT_KHR),
            )?;
            let [offset, bytes, status] = feedback;
            if status as i32 != VK_QUERY_RESULT_STATUS_COMPLETE_KHR {
                return Err(format!("the encoder didn't finish a picture: status {}", status as i32));
            }
            if offset as u64 + bytes as u64 > self.coded_size {
                return Err(format!("the encoder says its picture is {bytes} bytes at {offset}, past its buffer's end"));
            }
            self.out.clear();
            if key {
                self.out.extend_from_slice(&self.headers);
            }
            self.out.extend_from_slice(std::slice::from_raw_parts(self.coded_pointer.add(offset as usize), bytes as usize));
            self.last = this;
            self.frames += 1;
            self.since_key += 1;
            Ok((&self.out, key))
        }
    }

    /// Gives back everything it made, once a picture that timed out has finished or been given longer
    ///
    /// # Safety
    /// The device is still there.
    pub unsafe fn destroy(&mut self) {
        unsafe {
            let (calls, device) = (&self.calls, self.device);
            if mem::take(&mut self.pending) {
                (calls.vkWaitForFences)(device, 1, &self.fence, 1, PICTURE_WAIT * 5);
            }
            if self.fence != 0 {
                (calls.vkDestroyFence)(device, self.fence, ptr::null());
            }
            if self.queries != 0 {
                (calls.vkDestroyQueryPool)(device, self.queries, ptr::null());
            }
            if self.pool != 0 {
                (calls.vkDestroyCommandPool)(device, self.pool, ptr::null());
            }
            if self.coded != 0 {
                (calls.vkDestroyBuffer)(device, self.coded, ptr::null());
            }
            for view in self.picture_views.into_iter().chain([self.source_view]) {
                if view != 0 {
                    (calls.vkDestroyImageView)(device, view, ptr::null());
                }
            }
            for image in [self.pictures, self.source] {
                if image != 0 {
                    (calls.vkDestroyImage)(device, image, ptr::null());
                }
            }
            if self.parameters != 0 {
                (calls.vkDestroyVideoSessionParametersKHR)(device, self.parameters, ptr::null());
            }
            if self.session != 0 {
                (calls.vkDestroyVideoSessionKHR)(device, self.session, ptr::null());
            }
            for memory in self.session_memory.drain(..).chain([self.coded_memory, self.pictures_memory, self.source_memory]) {
                if memory != 0 {
                    (calls.vkFreeMemory)(device, memory, ptr::null());
                }
            }
            (self.fence, self.queries, self.pool, self.coded, self.picture_views, self.source_view) = (0, 0, 0, 0, [0; 2], 0);
            (self.pictures, self.source, self.parameters, self.session) = (0, 0, 0, 0);
            (self.coded_memory, self.pictures_memory, self.source_memory) = (0, 0, 0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn better_quality_is_a_finer_quantizer_within_what_the_card_takes() {
        assert_eq!(quantizer(100, 0, 51), 18);
        assert_eq!(quantizer(10, 0, 51), 45);
        assert_eq!(quantizer(0, 0, 51), 45);
        assert!(quantizer(60, 0, 51) < quantizer(40, 0, 51));
        assert_eq!(quantizer(100, 20, 40), 20);
        assert_eq!(quantizer(10, 20, 40), 40);
        assert_eq!(quantizer(50, -5, 80), 33);
    }
}
