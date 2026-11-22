//! Vulkan types, constants and command signatures, generated from the Vulkan registry by
//! nectar-vk-codegen (vkcodegen). Version helpers are written here.

mod generated;

pub use generated::*;

pub const fn make_api_version(variant: u32, major: u32, minor: u32, patch: u32) -> u32 {
    (variant << 29) | (major << 22) | (minor << 12) | patch
}

pub const API_VERSION_1_4: u32 = make_api_version(0, 1, 4, 0);

#[cfg(test)]
mod tests {
    use super::*;

    // Sizes on 64-bit targets, from the C headers
    #[test]
    fn layouts_match_c() {
        assert_eq!(size_of::<VkPhysicalDeviceLimits>(), 504);
        assert_eq!(size_of::<VkPhysicalDeviceProperties>(), 824);
        assert_eq!(size_of::<VkPhysicalDeviceFeatures>(), 220);
        assert_eq!(size_of::<VkPhysicalDeviceMemoryProperties>(), 520);
        assert_eq!(size_of::<VkImageCreateInfo>(), 88);
        assert_eq!(size_of::<VkAccelerationStructureInstanceKHR>(), 64);
        assert_eq!(size_of::<VkClearValue>(), 16);
        assert_eq!(size_of::<VkWriteDescriptorSet>(), 64);
        assert_eq!(VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES, 1000196000);
        assert_eq!(VK_ERROR_OUT_OF_DATE_KHR, -1000001004);
    }
}
