//! What an image needs of memory depends only on how it's created, so each way of creating one is
//! asked about once. vkd3d-proton asks about every texture, more than once: The Witcher 3 spent about
//! 100 s of its load on these alone, a whole round trip each.
//!
//! NECTAR_VK_ASK_IMAGES=1 asks every time, and says if an answer ever differs from the one kept.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use nectar_vk_sys::*;

type Key = Vec<u32>;

#[derive(Clone, Copy, PartialEq)]
struct Answer {
    size: u64,
    alignment: u64,
    types: u32,
    /// Prefers and requires a dedicated allocation, if that's been asked
    dedicated: Option<(u32, u32)>,
}

#[derive(Default)]
struct Needs {
    images: HashMap<u64, Key>,
    answers: HashMap<Key, Answer>,
}

static NEEDS: Mutex<Option<Needs>> = Mutex::new(None);

fn needs<R>(work: impl FnOnce(&mut Needs) -> R) -> R {
    work(NEEDS.lock().unwrap_or_else(|p| p.into_inner()).get_or_insert_with(Needs::default))
}

fn always_asking() -> bool {
    static ASKING: OnceLock<bool> = OnceLock::new();
    *ASKING.get_or_init(|| std::env::var_os("NECTAR_VK_ASK_IMAGES").is_some())
}

/// Everything about how an image is created, or nothing if it has something chained that isn't known
/// here. The planes of a disjoint image each have their own answer, so those always ask.
unsafe fn key_of(device: u64, info: &VkImageCreateInfo) -> Option<Key> {
    if info.flags & VK_IMAGE_CREATE_DISJOINT_BIT != 0 {
        return None;
    }
    let mut key = vec![
        device as u32,
        (device >> 32) as u32,
        info.flags,
        info.imageType as u32,
        info.format as u32,
        info.extent.width,
        info.extent.height,
        info.extent.depth,
        info.mipLevels,
        info.arrayLayers,
        info.samples,
        info.tiling as u32,
        info.usage,
        info.sharingMode as u32,
        info.initialLayout as u32,
    ];
    unsafe {
        if info.sharingMode == VK_SHARING_MODE_CONCURRENT && !info.pQueueFamilyIndices.is_null() {
            key.extend_from_slice(std::slice::from_raw_parts(info.pQueueFamilyIndices, info.queueFamilyIndexCount as usize));
        }
        let mut next = info.pNext as *const VkBaseInStructure;
        while let Some(link) = next.as_ref() {
            key.push(link.sType as u32);
            match link.sType {
                VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO => {
                    let list = &*(next as *const VkImageFormatListCreateInfo);
                    if !list.pViewFormats.is_null() {
                        key.extend(std::slice::from_raw_parts(list.pViewFormats, list.viewFormatCount as usize).iter().map(|&format| format as u32));
                    }
                }
                VK_STRUCTURE_TYPE_IMAGE_STENCIL_USAGE_CREATE_INFO => key.push((*(next as *const VkImageStencilUsageCreateInfo)).stencilUsage),
                _ => return None,
            }
            next = link.pNext;
        }
    }
    Some(key)
}

pub unsafe fn image_created(device: u64, image: u64, info: &VkImageCreateInfo) {
    if let Some(key) = unsafe { key_of(device, info) } {
        needs(|needs| needs.images.insert(image, key));
    }
}

pub fn image_destroyed(image: u64) {
    needs(|needs| needs.images.remove(&image));
}

/// The way of creating an image that this call is about, if that's known
unsafe fn key_of_call(needs: &Needs, name: &str, args: &[*const c_void]) -> Option<Key> {
    unsafe {
        match name {
            "vkGetDeviceImageMemoryRequirements" => {
                let info = &**(args[1] as *const *const VkDeviceImageMemoryRequirements);
                if info.pNext.is_null() && info.planeAspect == 0 { key_of(*(args[0] as *const u64), &*info.pCreateInfo) } else { None }
            }
            "vkGetImageMemoryRequirements" => needs.images.get(&*(args[1] as *const u64)).cloned(),
            "vkGetImageMemoryRequirements2" => {
                let info = &**(args[1] as *const *const VkImageMemoryRequirementsInfo2);
                if info.pNext.is_null() { needs.images.get(&info.image).cloned() } else { None }
            }
            _ => None,
        }
    }
}

/// Where a call's answer goes: the requirements, then whether a dedicated answer is wanted and where.
/// Nothing if something else is wanted that isn't kept.
unsafe fn out_of_call(name: &str, args: &[*const c_void]) -> Option<(*mut VkMemoryRequirements, *mut VkMemoryDedicatedRequirements)> {
    unsafe {
        if name == "vkGetImageMemoryRequirements" {
            return Some((*(args[2] as *const *mut VkMemoryRequirements), std::ptr::null_mut()));
        }
        let out = *(args[2] as *const *mut VkMemoryRequirements2);
        let mut dedicated = std::ptr::null_mut();
        let mut next = (*out).pNext as *const VkBaseInStructure;
        while let Some(link) = next.as_ref() {
            if link.sType != VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS {
                return None;
            }
            dedicated = next as *mut VkMemoryDedicatedRequirements;
            next = link.pNext;
        }
        Some((&raw mut (*out).memoryRequirements, dedicated))
    }
}

fn is_about_images(name: &str) -> bool {
    matches!(name, "vkGetDeviceImageMemoryRequirements" | "vkGetImageMemoryRequirements" | "vkGetImageMemoryRequirements2")
}

/// Answers a call from what's kept. False if it has to go to the lender.
pub unsafe fn answer(name: &str, args: &[*const c_void]) -> bool {
    if !is_about_images(name) || always_asking() {
        return false;
    }
    unsafe {
        let Some((out, dedicated)) = out_of_call(name, args) else { return false };
        let Some(kept) = needs(|needs| key_of_call(needs, name, args).and_then(|key| needs.answers.get(&key).copied())) else { return false };
        if !dedicated.is_null() {
            let Some((prefers, requires)) = kept.dedicated else { return false };
            (*dedicated).prefersDedicatedAllocation = prefers;
            (*dedicated).requiresDedicatedAllocation = requires;
        }
        (*out).size = kept.size;
        (*out).alignment = kept.alignment;
        (*out).memoryTypeBits = kept.types;
    }
    true
}

/// The lender answered a call: keeps it for the next image created the same way
pub unsafe fn answered(name: &str, args: &[*const c_void]) {
    unsafe {
        let Some((out, dedicated)) = out_of_call(name, args) else { return };
        let dedicated = dedicated.as_ref().map(|d| (d.prefersDedicatedAllocation, d.requiresDedicatedAllocation));
        let new = Answer { size: (*out).size, alignment: (*out).alignment, types: (*out).memoryTypeBits, dedicated };
        needs(|needs| {
            let Some(key) = key_of_call(needs, name, args) else { return };
            let kept = needs.answers.entry(key).or_insert(new);
            if always_asking() && (kept.size, kept.alignment, kept.types) != (new.size, new.alignment, new.types) {
                crate::log(&format!("{name}: {} and {} bytes for images created the same way", kept.size, new.size));
            }
            // What an answer leaves out, it keeps from the one before
            let dedicated = new.dedicated.or(kept.dedicated);
            *kept = Answer { dedicated, ..new };
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info() -> VkImageCreateInfo {
        let mut info: VkImageCreateInfo = unsafe { std::mem::zeroed() };
        info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        info.extent = VkExtent3D { width: 64, height: 64, depth: 1 };
        info.mipLevels = 1;
        info.arrayLayers = 1;
        info
    }

    fn ask(device: u64, image: u64, out: &mut VkMemoryRequirements) -> bool {
        let (image_arg, out_arg) = (image, out as *mut VkMemoryRequirements);
        let args = [&raw const device as *const c_void, &raw const image_arg as *const c_void, &raw const out_arg as *const c_void];
        unsafe { answer("vkGetImageMemoryRequirements", &args) }
    }

    fn told(device: u64, image: u64, out: &mut VkMemoryRequirements) {
        let (image_arg, out_arg) = (image, out as *mut VkMemoryRequirements);
        let args = [&raw const device as *const c_void, &raw const image_arg as *const c_void, &raw const out_arg as *const c_void];
        unsafe { answered("vkGetImageMemoryRequirements", &args) }
    }

    #[test]
    fn images_created_the_same_way_share_an_answer() {
        let device = 7001;
        let (first, second, wider) = (70011, 70012, 70013);
        let mut other = info();
        other.extent.width = 128;
        unsafe {
            image_created(device, first, &info());
            image_created(device, second, &info());
            image_created(device, wider, &other);
        }
        let mut out: VkMemoryRequirements = unsafe { std::mem::zeroed() };
        assert!(!ask(device, first, &mut out));
        out.size = 4096;
        out.alignment = 1024;
        out.memoryTypeBits = 3;
        told(device, first, &mut out);

        let mut again: VkMemoryRequirements = unsafe { std::mem::zeroed() };
        assert!(ask(device, second, &mut again));
        assert_eq!((again.size, again.alignment, again.memoryTypeBits), (4096, 1024, 3));
        assert!(!ask(device, wider, &mut again));

        // Another device's images are its own
        unsafe { image_created(device + 1, 70014, &info()) };
        assert!(!ask(device + 1, 70014, &mut again));

        image_destroyed(second);
        assert!(!ask(device, second, &mut again));
    }

    #[test]
    fn what_the_key_covers() {
        let plain = unsafe { key_of(1, &info()) }.unwrap();
        let formats = [VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SRGB];
        let list = VkImageFormatListCreateInfo {
            sType: VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,
            pNext: std::ptr::null(),
            viewFormatCount: 2,
            pViewFormats: formats.as_ptr(),
        };
        let mut listed = info();
        listed.pNext = &raw const list as *const c_void;
        let with_list = unsafe { key_of(1, &listed) }.unwrap();
        assert_ne!(plain, with_list);

        let one = VkImageFormatListCreateInfo { viewFormatCount: 1, ..list };
        listed.pNext = &raw const one as *const c_void;
        assert_ne!(with_list, unsafe { key_of(1, &listed) }.unwrap());

        // Something chained that isn't known here
        let unknown = VkImageFormatListCreateInfo { sType: 1, ..list };
        listed.pNext = &raw const unknown as *const c_void;
        assert!(unsafe { key_of(1, &listed) }.is_none());

        let mut disjoint = info();
        disjoint.flags = VK_IMAGE_CREATE_DISJOINT_BIT;
        assert!(unsafe { key_of(1, &disjoint) }.is_none());
    }

    #[test]
    fn a_dedicated_answer_has_to_have_been_asked() {
        let device = 7101;
        let image = 71011;
        unsafe { image_created(device, image, &info()) };
        let mut plain: VkMemoryRequirements = unsafe { std::mem::zeroed() };
        plain.size = 65536;
        told(device, image, &mut plain);

        let mut dedicated: VkMemoryDedicatedRequirements = unsafe { std::mem::zeroed() };
        dedicated.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS;
        let mut two: VkMemoryRequirements2 = unsafe { std::mem::zeroed() };
        two.pNext = &raw mut dedicated as *mut c_void;
        let request = VkImageMemoryRequirementsInfo2 { sType: 0, pNext: std::ptr::null(), image };
        let (request_arg, two_arg) = (&raw const request, &raw mut two);
        let args = [&raw const device as *const c_void, &raw const request_arg as *const c_void, &raw const two_arg as *const c_void];
        assert!(!unsafe { answer("vkGetImageMemoryRequirements2", &args) });

        two.memoryRequirements.size = 65536;
        dedicated.prefersDedicatedAllocation = 1;
        unsafe { answered("vkGetImageMemoryRequirements2", &args) };
        dedicated.prefersDedicatedAllocation = 0;
        two.memoryRequirements.size = 0;
        assert!(unsafe { answer("vkGetImageMemoryRequirements2", &args) });
        assert_eq!((two.memoryRequirements.size, dedicated.prefersDedicatedAllocation), (65536, 1));
    }
}
