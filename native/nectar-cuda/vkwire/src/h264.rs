//! The H.264 structures Vulkan Video's encoding takes, which the generated types leave out: those of the
//! codec's own header (`vulkan_video_codec_h264std.h` and `..._encode.h`), and the Vulkan structures that
//! point at them. Written from the registry's `video.xml` and `vk.xml`, field for field.
//!
//! C packs a structure's one-bit flags into a word from the lowest bit up, so each flags structure here is
//! a `u32` with the bits named.

#![allow(non_snake_case)]

use std::ffi::c_void;

use nectar_vk_sys::{VkBool32, VkStructureType};

pub const PROFILE_HIGH: u32 = 100;
pub const LEVEL_5_1: u32 = 14;
pub const LEVEL_5_2: u32 = 15;
pub const CHROMA_420: u32 = 1;
pub const POC_TYPE_2: u32 = 2;
pub const PICTURE_P: u32 = 0;
pub const PICTURE_IDR: u32 = 5;
pub const SLICE_P: u32 = 0;
pub const SLICE_I: u32 = 2;
pub const MAX_NUM_LIST_REF: usize = 32;
pub const NO_REFERENCE_PICTURE: u8 = 0xFF;

pub const SPS_DIRECT_8X8_INFERENCE: u32 = 1 << 6;
pub const SPS_FRAME_MBS_ONLY: u32 = 1 << 8;
pub const SPS_FRAME_CROPPING: u32 = 1 << 13;
pub const SPS_VUI_PARAMETERS_PRESENT: u32 = 1 << 15;

pub const VUI_VIDEO_SIGNAL_TYPE_PRESENT: u32 = 1 << 3;
pub const VUI_VIDEO_FULL_RANGE: u32 = 1 << 4;
pub const VUI_COLOR_DESCRIPTION_PRESENT: u32 = 1 << 5;
pub const VUI_BITSTREAM_RESTRICTION: u32 = 1 << 9;

pub const PPS_TRANSFORM_8X8_MODE: u32 = 1 << 0;
pub const PPS_DEBLOCKING_FILTER_CONTROL_PRESENT: u32 = 1 << 3;
pub const PPS_ENTROPY_CODING_MODE: u32 = 1 << 6;

pub const PICTURE_IDR_FLAG: u32 = 1 << 0;
pub const PICTURE_IS_REFERENCE: u32 = 1 << 1;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct StdVideoH264SequenceParameterSetVui {
    pub flags: u32,
    pub aspect_ratio_idc: u32,
    pub sar_width: u16,
    pub sar_height: u16,
    pub video_format: u8,
    pub colour_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
    pub num_units_in_tick: u32,
    pub time_scale: u32,
    pub max_num_reorder_frames: u8,
    pub max_dec_frame_buffering: u8,
    pub chroma_sample_loc_type_top_field: u8,
    pub chroma_sample_loc_type_bottom_field: u8,
    pub reserved1: u32,
    pub pHrdParameters: *const c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct StdVideoH264SequenceParameterSet {
    pub flags: u32,
    pub profile_idc: u32,
    pub level_idc: u32,
    pub chroma_format_idc: u32,
    pub seq_parameter_set_id: u8,
    pub bit_depth_luma_minus8: u8,
    pub bit_depth_chroma_minus8: u8,
    pub log2_max_frame_num_minus4: u8,
    pub pic_order_cnt_type: u32,
    pub offset_for_non_ref_pic: i32,
    pub offset_for_top_to_bottom_field: i32,
    pub log2_max_pic_order_cnt_lsb_minus4: u8,
    pub num_ref_frames_in_pic_order_cnt_cycle: u8,
    pub max_num_ref_frames: u8,
    pub reserved1: u8,
    pub pic_width_in_mbs_minus1: u32,
    pub pic_height_in_map_units_minus1: u32,
    pub frame_crop_left_offset: u32,
    pub frame_crop_right_offset: u32,
    pub frame_crop_top_offset: u32,
    pub frame_crop_bottom_offset: u32,
    pub reserved2: u32,
    pub pOffsetForRefFrame: *const i32,
    pub pScalingLists: *const c_void,
    pub pSequenceParameterSetVui: *const StdVideoH264SequenceParameterSetVui,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct StdVideoH264PictureParameterSet {
    pub flags: u32,
    pub seq_parameter_set_id: u8,
    pub pic_parameter_set_id: u8,
    pub num_ref_idx_l0_default_active_minus1: u8,
    pub num_ref_idx_l1_default_active_minus1: u8,
    pub weighted_bipred_idc: u32,
    pub pic_init_qp_minus26: i8,
    pub pic_init_qs_minus26: i8,
    pub chroma_qp_index_offset: i8,
    pub second_chroma_qp_index_offset: i8,
    pub pScalingLists: *const c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct StdVideoEncodeH264SliceHeader {
    pub flags: u32,
    pub first_mb_in_slice: u32,
    pub slice_type: u32,
    pub slice_alpha_c0_offset_div2: i8,
    pub slice_beta_offset_div2: i8,
    pub slice_qp_delta: i8,
    pub reserved1: u8,
    pub cabac_init_idc: u32,
    pub disable_deblocking_filter_idc: u32,
    pub pWeightTable: *const c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct StdVideoEncodeH264ReferenceListsInfo {
    pub flags: u32,
    pub num_ref_idx_l0_active_minus1: u8,
    pub num_ref_idx_l1_active_minus1: u8,
    pub RefPicList0: [u8; MAX_NUM_LIST_REF],
    pub RefPicList1: [u8; MAX_NUM_LIST_REF],
    pub refList0ModOpCount: u8,
    pub refList1ModOpCount: u8,
    pub refPicMarkingOpCount: u8,
    pub reserved1: [u8; 7],
    pub pRefList0ModOperations: *const c_void,
    pub pRefList1ModOperations: *const c_void,
    pub pRefPicMarkingOperations: *const c_void,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct StdVideoEncodeH264PictureInfo {
    pub flags: u32,
    pub seq_parameter_set_id: u8,
    pub pic_parameter_set_id: u8,
    pub idr_pic_id: u16,
    pub primary_pic_type: u32,
    pub frame_num: u32,
    pub PicOrderCnt: i32,
    pub temporal_id: u8,
    pub reserved1: [u8; 3],
    pub pRefLists: *const StdVideoEncodeH264ReferenceListsInfo,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct StdVideoEncodeH264ReferenceInfo {
    pub flags: u32,
    pub primary_pic_type: u32,
    pub FrameNum: u32,
    pub PicOrderCnt: i32,
    pub long_term_pic_num: u16,
    pub long_term_frame_idx: u16,
    pub temporal_id: u8,
}

#[repr(C)]
pub struct VkVideoEncodeH264ProfileInfoKHR {
    pub sType: VkStructureType,
    pub pNext: *const c_void,
    pub stdProfileIdc: u32,
}

#[repr(C)]
pub struct VkVideoEncodeH264CapabilitiesKHR {
    pub sType: VkStructureType,
    pub pNext: *mut c_void,
    pub flags: u32,
    pub maxLevelIdc: u32,
    pub maxSliceCount: u32,
    pub maxPPictureL0ReferenceCount: u32,
    pub maxBPictureL0ReferenceCount: u32,
    pub maxL1ReferenceCount: u32,
    pub maxTemporalLayerCount: u32,
    pub expectDyadicTemporalLayerPattern: VkBool32,
    pub minQp: i32,
    pub maxQp: i32,
    pub prefersGopRemainingFrames: VkBool32,
    pub requiresGopRemainingFrames: VkBool32,
    pub stdSyntaxFlags: u32,
}

#[repr(C)]
pub struct VkVideoEncodeH264SessionParametersAddInfoKHR {
    pub sType: VkStructureType,
    pub pNext: *const c_void,
    pub stdSPSCount: u32,
    pub pStdSPSs: *const StdVideoH264SequenceParameterSet,
    pub stdPPSCount: u32,
    pub pStdPPSs: *const StdVideoH264PictureParameterSet,
}

#[repr(C)]
pub struct VkVideoEncodeH264SessionParametersCreateInfoKHR {
    pub sType: VkStructureType,
    pub pNext: *const c_void,
    pub maxStdSPSCount: u32,
    pub maxStdPPSCount: u32,
    pub pParametersAddInfo: *const VkVideoEncodeH264SessionParametersAddInfoKHR,
}

#[repr(C)]
pub struct VkVideoEncodeH264SessionParametersGetInfoKHR {
    pub sType: VkStructureType,
    pub pNext: *const c_void,
    pub writeStdSPS: VkBool32,
    pub writeStdPPS: VkBool32,
    pub stdSPSId: u32,
    pub stdPPSId: u32,
}

#[repr(C)]
pub struct VkVideoEncodeH264NaluSliceInfoKHR {
    pub sType: VkStructureType,
    pub pNext: *const c_void,
    pub constantQp: i32,
    pub pStdSliceHeader: *const StdVideoEncodeH264SliceHeader,
}

#[repr(C)]
pub struct VkVideoEncodeH264PictureInfoKHR {
    pub sType: VkStructureType,
    pub pNext: *const c_void,
    pub naluSliceEntryCount: u32,
    pub pNaluSliceEntries: *const VkVideoEncodeH264NaluSliceInfoKHR,
    pub pStdPictureInfo: *const StdVideoEncodeH264PictureInfo,
    pub generatePrefixNalu: VkBool32,
}

#[repr(C)]
pub struct VkVideoEncodeH264DpbSlotInfoKHR {
    pub sType: VkStructureType,
    pub pNext: *const c_void,
    pub pStdReferenceInfo: *const StdVideoEncodeH264ReferenceInfo,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    /// The sizes and offsets a C compiler gives these on a 64-bit target, worked out from the header by the
    /// usual rules: each field at the next multiple of its own alignment
    #[test]
    fn layouts_are_cs() {
        assert_eq!(size_of::<StdVideoH264SequenceParameterSet>(), 88);
        assert_eq!(offset_of!(StdVideoH264SequenceParameterSet, pic_order_cnt_type), 20);
        assert_eq!(offset_of!(StdVideoH264SequenceParameterSet, pic_width_in_mbs_minus1), 36);
        assert_eq!(offset_of!(StdVideoH264SequenceParameterSet, pOffsetForRefFrame), 64);
        assert_eq!(size_of::<StdVideoH264SequenceParameterSetVui>(), 40);
        assert_eq!(offset_of!(StdVideoH264SequenceParameterSetVui, num_units_in_tick), 16);
        assert_eq!(offset_of!(StdVideoH264SequenceParameterSetVui, pHrdParameters), 32);
        assert_eq!(size_of::<StdVideoH264PictureParameterSet>(), 24);
        assert_eq!(offset_of!(StdVideoH264PictureParameterSet, pScalingLists), 16);
        assert_eq!(size_of::<StdVideoEncodeH264SliceHeader>(), 32);
        assert_eq!(offset_of!(StdVideoEncodeH264SliceHeader, cabac_init_idc), 16);
        assert_eq!(size_of::<StdVideoEncodeH264ReferenceListsInfo>(), 104);
        assert_eq!(offset_of!(StdVideoEncodeH264ReferenceListsInfo, RefPicList1), 38);
        assert_eq!(offset_of!(StdVideoEncodeH264ReferenceListsInfo, pRefList0ModOperations), 80);
        assert_eq!(size_of::<StdVideoEncodeH264PictureInfo>(), 32);
        assert_eq!(offset_of!(StdVideoEncodeH264PictureInfo, pRefLists), 24);
        assert_eq!(size_of::<StdVideoEncodeH264ReferenceInfo>(), 24);
        assert_eq!(size_of::<VkVideoEncodeH264PictureInfoKHR>(), 48);
        assert_eq!(size_of::<VkVideoEncodeH264NaluSliceInfoKHR>(), 32);
    }
}
