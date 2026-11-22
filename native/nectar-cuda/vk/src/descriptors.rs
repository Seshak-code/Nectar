//! Descriptor sets without a wait each. DXVK allocates sets for every draw, and an allocation can
//! only be sent without waiting if its answer is known here. So pools and layouts are counted here,
//! as the specification counts them: an allocation from a pool never freed from individually can
//! fail only by going over what the pool was created with. Anything less plain (chained create
//! info, inline blocks, freeing individually) still asks the lender.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use nectar_cuda_protocol::Writer;
use nectar_vk_sys::*;
use nectar_vk_wire::descriptor_copies::{self, DescriptorCopy};
use nectar_vk_wire::messages::DESCRIPTOR_COPIES;

use crate::connection::Connection;
use crate::objects;

/// Copies waiting to go, with the device they're on. vkd3d-proton copies descriptors one at a time, many
/// thousand a frame, so they go together: see descriptor_copies.rs. NECTAR_VK_COPY_DESCRIPTORS=1 sends each
/// as it comes.
static COPIES: Mutex<(u64, Vec<DescriptorCopy>)> = Mutex::new((0, Vec::new()));
static COPIES_WAITING: AtomicBool = AtomicBool::new(false);
/// Copies that go at once however many more could join them
const COPIES_MOST: usize = 4096;

fn send_copies_locked(connection: &Connection, device: u64, copies: &mut Vec<DescriptorCopy>) {
    if copies.is_empty() {
        return;
    }
    let mut w = Writer::new();
    descriptor_copies::encode(device, copies, &mut w);
    let _ = connection.send(DESCRIPTOR_COPIES, w.as_slice());
    copies.clear();
    COPIES_WAITING.store(false, Ordering::Relaxed);
}

/// Takes a vkUpdateDescriptorSets that only copies, to go later. False if it's anything else.
///
/// # Safety
/// `args` must be vkUpdateDescriptorSets' arguments, each by its address.
pub unsafe fn copy_later(connection: &Connection, args: &[*const c_void]) -> bool {
    static BATCHING: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*BATCHING.get_or_init(|| std::env::var_os("NECTAR_VK_COPY_DESCRIPTORS").is_none()) {
        return false;
    }
    unsafe {
        let (writes, count) = (*(args[1] as *const u32), *(args[3] as *const u32) as usize);
        if writes != 0 || count == 0 {
            return false;
        }
        let list = std::slice::from_raw_parts(*(args[4] as *const *const VkCopyDescriptorSet), count);
        if list.iter().any(|copy| !copy.pNext.is_null()) {
            return false;
        }
        let device = objects::id_of(*(args[0] as *const u64));
        let mut waiting = COPIES.lock().unwrap_or_else(|p| p.into_inner());
        let (for_device, copies) = &mut *waiting;
        if *for_device != device {
            send_copies_locked(connection, *for_device, copies);
            *for_device = device;
        }
        for copy in list {
            descriptor_copies::add(
                copies,
                DescriptorCopy {
                    source_set: copy.srcSet,
                    target_set: copy.dstSet,
                    source_binding: copy.srcBinding,
                    target_binding: copy.dstBinding,
                    source_element: copy.srcArrayElement,
                    target_element: copy.dstArrayElement,
                    count: copy.descriptorCount,
                },
            );
        }
        COPIES_WAITING.store(true, Ordering::Relaxed);
        if copies.len() >= COPIES_MOST {
            send_copies_locked(connection, *for_device, copies);
        }
    }
    true
}

/// Whether a command can go ahead of copies waiting. Recording only says what to do, and what it uses is
/// read when it runs, except that a set must be as it should be when it's bound without update-after-bind.
fn copies_can_wait(name: &str) -> bool {
    (name.starts_with("vkCmd") && !name.contains("Descriptor")) || matches!(name, "vkBeginCommandBuffer" | "vkEndCommandBuffer" | "vkGetPhysicalDeviceMemoryProperties2")
}

/// Before a command goes out, what it could depend on does
pub fn before_command(connection: &Connection, name: &str) {
    if COPIES_WAITING.load(Ordering::Relaxed) && !copies_can_wait(name) {
        let mut waiting = COPIES.lock().unwrap_or_else(|p| p.into_inner());
        let (device, copies) = &mut *waiting;
        send_copies_locked(connection, *device, copies);
    }
}

#[derive(Default)]
struct Pool {
    sets_left: i64,
    left: HashMap<VkDescriptorType, i64>,
    sets: u32,
    sizes: Vec<VkDescriptorPoolSize>,
}

impl Pool {
    fn restore(&mut self) {
        self.sets_left = self.sets as i64;
        self.left.clear();
        for size in &self.sizes {
            *self.left.entry(size.r#type).or_default() += size.descriptorCount as i64;
        }
    }
}

#[derive(Default)]
struct Counted {
    /// Plain pools by handle, and plain layouts with how many descriptors of each type they hold
    pools: HashMap<u64, Pool>,
    layouts: HashMap<u64, Vec<(VkDescriptorType, u32)>>,
}

static COUNTED: Mutex<Option<Counted>> = Mutex::new(None);

fn counted<R>(f: impl FnOnce(&mut Counted) -> R) -> R {
    f(COUNTED.lock().unwrap_or_else(|p| p.into_inner()).get_or_insert_with(Counted::default))
}

/// What to do with an allocation
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Allocation {
    /// Ask the lender, as it isn't known here
    Ask,
    /// It fits: send it without waiting
    Send,
    /// It doesn't fit
    Refuse,
}

/// Counts an allocation from `pool` of sets with `layouts`
fn allocate_from(pool: u64, layouts: &[u64]) -> Allocation {
    counted(|counted| {
        let Counted { pools, layouts: known } = counted;
        let Some(pool) = pools.get_mut(&pool) else { return Allocation::Ask };
        let mut wanted: HashMap<VkDescriptorType, i64> = HashMap::new();
        for layout in layouts {
            let Some(bindings) = known.get(layout) else { return Allocation::Ask };
            for &(kind, count) in bindings {
                *wanted.entry(kind).or_default() += count as i64;
            }
        }
        let fits = pool.sets_left >= layouts.len() as i64
            && wanted.iter().all(|(kind, count)| *count == 0 || pool.left.get(kind).is_some_and(|left| left >= count));
        if !fits {
            return Allocation::Refuse;
        }
        pool.sets_left -= layouts.len() as i64;
        for (kind, count) in wanted {
            if count > 0 {
                *pool.left.entry(kind).or_default() -= count;
            }
        }
        Allocation::Send
    })
}

/// Before a vkAllocateDescriptorSets goes out
///
/// # Safety
/// `args` must be the call's arguments, each by its address.
pub unsafe fn allocation(args: &[*const c_void]) -> Allocation {
    unsafe {
        let info = &**(args[1] as *const *const VkDescriptorSetAllocateInfo);
        if !info.pNext.is_null() {
            return Allocation::Ask;
        }
        let layouts = std::slice::from_raw_parts(info.pSetLayouts, info.descriptorSetCount as usize);
        allocate_from(info.descriptorPool, layouts)
    }
}

/// Notes a pool or layout once its handle is chosen, and what happens to them
///
/// # Safety
/// `args` must be the command's arguments, each by its address.
pub unsafe fn note(name: &str, args: &[*const c_void]) {
    unsafe {
        match name {
            "vkCreateDescriptorSetLayout" => {
                let info = &**(args[1] as *const *const VkDescriptorSetLayoutCreateInfo);
                let handle = **(args[3] as *const *const u64);
                let bindings = if info.bindingCount > 0 {
                    std::slice::from_raw_parts(info.pBindings, info.bindingCount as usize)
                } else {
                    &[]
                };
                let plain = info.pNext.is_null() && bindings.iter().all(|b| b.descriptorType != VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK);
                if plain {
                    let counts = bindings.iter().map(|b| (b.descriptorType, b.descriptorCount)).collect();
                    counted(|counted| counted.layouts.insert(handle, counts));
                }
            }
            "vkDestroyDescriptorSetLayout" => {
                let handle = *(args[1] as *const u64);
                counted(|counted| counted.layouts.remove(&handle));
            }
            "vkCreateDescriptorPool" => {
                let info = &**(args[1] as *const *const VkDescriptorPoolCreateInfo);
                let handle = **(args[3] as *const *const u64);
                let frees_singly = info.flags & VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT != 0;
                if info.pNext.is_null() && !frees_singly {
                    let sizes = if info.poolSizeCount > 0 {
                        std::slice::from_raw_parts(info.pPoolSizes, info.poolSizeCount as usize).to_vec()
                    } else {
                        Vec::new()
                    };
                    let mut pool = Pool { sets: info.maxSets, sizes, ..Pool::default() };
                    pool.restore();
                    counted(|counted| counted.pools.insert(handle, pool));
                }
            }
            "vkResetDescriptorPool" => {
                let handle = *(args[1] as *const u64);
                counted(|counted| {
                    if let Some(pool) = counted.pools.get_mut(&handle) {
                        pool.restore();
                    }
                });
            }
            "vkDestroyDescriptorPool" => {
                let handle = *(args[1] as *const u64);
                counted(|counted| counted.pools.remove(&handle));
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(handle: u64, sets: u32, sizes: &[(VkDescriptorType, u32)]) {
        let sizes = sizes.iter().map(|&(r#type, descriptorCount)| VkDescriptorPoolSize { r#type, descriptorCount }).collect();
        let mut pool = Pool { sets, sizes, ..Pool::default() };
        pool.restore();
        counted(|counted| counted.pools.insert(handle, pool));
    }

    #[test]
    fn allocations_count_down_and_reset() {
        const UNIFORM: VkDescriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        pool(7001, 3, &[(UNIFORM, 4)]);
        counted(|counted| {
            counted.layouts.insert(7002, vec![(UNIFORM, 2)]);
            counted.layouts.insert(7003, vec![(VK_DESCRIPTOR_TYPE_SAMPLER, 1)]);
            counted.layouts.insert(7004, Vec::new());
        });
        assert_eq!(allocate_from(7001, &[7002, 7002]), Allocation::Send);
        // Out of uniform buffers, though a set is left
        assert_eq!(allocate_from(7001, &[7002]), Allocation::Refuse);
        // A type the pool never had
        assert_eq!(allocate_from(7001, &[7003]), Allocation::Refuse);
        // A set with no descriptors takes only a set; then there are none left
        assert_eq!(allocate_from(7001, &[7004]), Allocation::Send);
        assert_eq!(allocate_from(7001, &[7004]), Allocation::Refuse);

        counted(|counted| counted.pools.get_mut(&7001).unwrap().restore());
        assert_eq!(allocate_from(7001, &[7002]), Allocation::Send);
    }

    #[test]
    fn what_is_not_known_asks_the_lender() {
        pool(7101, 1, &[]);
        assert_eq!(allocate_from(7199, &[]), Allocation::Ask);
        assert_eq!(allocate_from(7101, &[7198]), Allocation::Ask);
    }
}
