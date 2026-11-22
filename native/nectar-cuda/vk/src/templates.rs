//! Descriptor update templates, which DXVK uses for every draw's descriptors. A template says how
//! to read descriptors out of the program's own memory, in a layout only it knows. It stays on this
//! side: each update is read here and goes to the lender as the plain update it stands for.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;
use std::{mem, ptr};

use nectar_vk_sys::*;

use crate::{generated, objects};

struct Template {
    entries: Vec<VkDescriptorUpdateTemplateEntry>,
    bind_point: VkPipelineBindPoint,
}

static TEMPLATES: Mutex<Option<HashMap<u64, Template>>> = Mutex::new(None);

fn with_template<R>(handle: VkDescriptorUpdateTemplate, f: impl FnOnce(&Template) -> R) -> Option<R> {
    TEMPLATES.lock().unwrap_or_else(|p| p.into_inner()).as_ref()?.get(&handle).map(f)
}

/// What an update's descriptors are read as
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Image,
    Buffer,
    TexelBuffer,
    /// Bytes: the entry's count is their number
    InlineBlock,
    Unknown,
}

fn kind(descriptor: VkDescriptorType) -> Kind {
    match descriptor {
        VK_DESCRIPTOR_TYPE_SAMPLER
        | VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
        | VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE
        | VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
        | VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT => Kind::Image,
        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
        | VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
        | VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC
        | VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC => Kind::Buffer,
        VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER | VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER => Kind::TexelBuffer,
        VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK => Kind::InlineBlock,
        _ => Kind::Unknown,
    }
}

/// The plain update a template and the program's data stand for. The writes point into the vectors
/// here and into `data`.
struct Update {
    writes: Vec<VkWriteDescriptorSet>,
    _images: Vec<Vec<VkDescriptorImageInfo>>,
    _buffers: Vec<Vec<VkDescriptorBufferInfo>>,
    _views: Vec<Vec<VkBufferView>>,
    // Boxed so that they stay where the writes point as this grows
    #[allow(clippy::vec_box)]
    _blocks: Vec<Box<VkWriteDescriptorSetInlineUniformBlock>>,
}

/// Reads the `index`th descriptor of an entry out of the program's data
unsafe fn read<T: Copy>(data: *const c_void, entry: &VkDescriptorUpdateTemplateEntry, index: usize) -> T {
    unsafe { (data as *const u8).add(entry.offset + entry.stride * index).cast::<T>().read_unaligned() }
}

unsafe fn update_of(template: &Template, set: VkDescriptorSet, data: *const c_void) -> Update {
    let mut update = Update { writes: Vec::new(), _images: Vec::new(), _buffers: Vec::new(), _views: Vec::new(), _blocks: Vec::new() };
    for entry in &template.entries {
        let count = entry.descriptorCount as usize;
        let mut write = VkWriteDescriptorSet {
            sType: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            pNext: ptr::null(),
            dstSet: set,
            dstBinding: entry.dstBinding,
            dstArrayElement: entry.dstArrayElement,
            descriptorCount: entry.descriptorCount,
            descriptorType: entry.descriptorType,
            pImageInfo: ptr::null(),
            pBufferInfo: ptr::null(),
            pTexelBufferView: ptr::null(),
        };
        unsafe {
            match kind(entry.descriptorType) {
                Kind::Image => {
                    let list: Vec<VkDescriptorImageInfo> = (0..count).map(|i| read(data, entry, i)).collect();
                    write.pImageInfo = list.as_ptr();
                    update._images.push(list);
                }
                Kind::Buffer => {
                    let list: Vec<VkDescriptorBufferInfo> = (0..count).map(|i| read(data, entry, i)).collect();
                    write.pBufferInfo = list.as_ptr();
                    update._buffers.push(list);
                }
                Kind::TexelBuffer => {
                    let list: Vec<VkBufferView> = (0..count).map(|i| read(data, entry, i)).collect();
                    write.pTexelBufferView = list.as_ptr();
                    update._views.push(list);
                }
                Kind::InlineBlock => {
                    let block = Box::new(VkWriteDescriptorSetInlineUniformBlock {
                        sType: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK,
                        pNext: ptr::null(),
                        dataSize: entry.descriptorCount,
                        pData: (data as *const u8).add(entry.offset).cast(),
                    });
                    write.pNext = (&*block as *const VkWriteDescriptorSetInlineUniformBlock).cast();
                    update._blocks.push(block);
                }
                Kind::Unknown => {
                    crate::log(&format!("a template's descriptor type {} can't be carried", entry.descriptorType));
                    continue;
                }
            }
        }
        update.writes.push(write);
    }
    update
}

pub unsafe extern "system" fn vkCreateDescriptorUpdateTemplate(
    _device: VkDevice,
    pCreateInfo: *const VkDescriptorUpdateTemplateCreateInfo,
    _pAllocator: *const VkAllocationCallbacks,
    pTemplate: *mut VkDescriptorUpdateTemplate,
) -> VkResult {
    unsafe {
        let info = &*pCreateInfo;
        let entries = std::slice::from_raw_parts(info.pDescriptorUpdateEntries, info.descriptorUpdateEntryCount as usize).to_vec();
        let handle = objects::new_id();
        let template = Template { entries, bind_point: info.pipelineBindPoint };
        TEMPLATES.lock().unwrap_or_else(|p| p.into_inner()).get_or_insert_with(HashMap::new).insert(handle, template);
        *pTemplate = handle;
    }
    VK_SUCCESS
}

pub unsafe extern "system" fn vkDestroyDescriptorUpdateTemplate(
    _device: VkDevice,
    template: VkDescriptorUpdateTemplate,
    _pAllocator: *const VkAllocationCallbacks,
) {
    if let Some(templates) = TEMPLATES.lock().unwrap_or_else(|p| p.into_inner()).as_mut() {
        templates.remove(&template);
    }
}

pub unsafe extern "system" fn vkUpdateDescriptorSetWithTemplate(
    device: VkDevice,
    set: VkDescriptorSet,
    template: VkDescriptorUpdateTemplate,
    pData: *const c_void,
) {
    if let Some(update) = with_template(template, |template| unsafe { update_of(template, set, pData) }) {
        unsafe { generated::vkUpdateDescriptorSets(device, update.writes.len() as u32, update.writes.as_ptr(), 0, ptr::null()) };
    }
}

pub unsafe extern "system" fn vkCmdPushDescriptorSetWithTemplate(
    commands: VkCommandBuffer,
    template: VkDescriptorUpdateTemplate,
    layout: VkPipelineLayout,
    set: u32,
    pData: *const c_void,
) {
    let found = with_template(template, |template| (unsafe { update_of(template, 0, pData) }, template.bind_point));
    if let Some((update, bind_point)) = found {
        unsafe { generated::vkCmdPushDescriptorSet(commands, bind_point, layout, set, update.writes.len() as u32, update.writes.as_ptr()) };
    }
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
        vkCreateDescriptorUpdateTemplate,
        vkDestroyDescriptorUpdateTemplate,
        vkUpdateDescriptorSetWithTemplate,
        vkCmdPushDescriptorSetWithTemplate
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_update_is_read_out_of_the_programs_layout() {
        // As a program lays out its data: a header, then buffers 32 bytes apart, then a view
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct Slot {
            info: VkDescriptorBufferInfo,
            padding: u64,
        }
        #[repr(C)]
        struct Data {
            header: [u64; 3],
            slots: [Slot; 2],
            view: VkBufferView,
            block: [u8; 8],
        }
        let info = |buffer| VkDescriptorBufferInfo { buffer, offset: 16, range: 64 };
        let data = Data {
            header: [0; 3],
            slots: [Slot { info: info(11), padding: 0 }, Slot { info: info(22), padding: 0 }],
            view: 33,
            block: *b"inline!!",
        };
        let entry = |binding, count, descriptor, offset, stride| VkDescriptorUpdateTemplateEntry {
            dstBinding: binding,
            dstArrayElement: 0,
            descriptorCount: count,
            descriptorType: descriptor,
            offset,
            stride,
        };
        let template = Template {
            entries: vec![
                entry(1, 2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, mem::offset_of!(Data, slots), size_of::<Slot>()),
                entry(2, 1, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, mem::offset_of!(Data, view), 0),
                entry(3, 8, VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK, mem::offset_of!(Data, block), 0),
            ],
            bind_point: VK_PIPELINE_BIND_POINT_GRAPHICS,
        };
        let update = unsafe { update_of(&template, 77, (&data as *const Data).cast()) };
        assert_eq!(update.writes.len(), 3);
        let buffers = unsafe { std::slice::from_raw_parts(update.writes[0].pBufferInfo, 2) };
        assert_eq!((buffers[0].buffer, buffers[1].buffer, buffers[1].range), (11, 22, 64));
        assert_eq!(update.writes[0].dstSet, 77);
        assert_eq!(unsafe { *update.writes[1].pTexelBufferView }, 33);
        let block = unsafe { &*update.writes[2].pNext.cast::<VkWriteDescriptorSetInlineUniformBlock>() };
        assert_eq!(block.dataSize, 8);
        assert_eq!(unsafe { std::slice::from_raw_parts(block.pData.cast::<u8>(), 8) }, b"inline!!");
    }
}
