//! Calls carried end to end through the interpreter: encoded from real Vulkan structures as a
//! program builds them, decoded as the lender's driver would see them, and outputs written back.

use std::ffi::{CStr, c_void};
use std::{mem, ptr};

use nectar_cuda_protocol::{Reader, Writer};
use nectar_vk_sys::*;

use crate::COMMANDS;
use crate::tables::CommandInfo;
use crate::wire::*;

/// The borrower's side: dispatchable objects are fake pointers 0x1000_0000 above their id
#[derive(Default)]
struct Borrower {
    next: u64,
}

impl Handles for Borrower {
    fn outgoing(&mut self, dispatchable: bool, value: u64) -> u64 {
        if dispatchable { value - 0x1000_0000 } else { value }
    }
    fn incoming(&mut self, dispatchable: bool, id: u64) -> u64 {
        if dispatchable { id + 0x1000_0000 } else { id }
    }
    fn create(&mut self, dispatchable: bool) -> (u64, u64) {
        self.next += 1;
        (self.next, if dispatchable { self.next + 0x1000_0000 } else { self.next })
    }
    fn bind(&mut self, _: u64, _: u64) {}
}

/// The lender's side: real handles are 0xA000 above their id
#[derive(Default)]
struct Lender {
    bound: Vec<(u64, u64)>,
}

impl Handles for Lender {
    fn outgoing(&mut self, _: bool, value: u64) -> u64 {
        value - 0xA000
    }
    fn incoming(&mut self, _: bool, id: u64) -> u64 {
        id + 0xA000
    }
    fn create(&mut self, _: bool) -> (u64, u64) {
        unreachable!()
    }
    fn bind(&mut self, id: u64, real: u64) {
        self.bound.push((id, real));
    }
}

fn command(name: &str) -> &'static CommandInfo {
    COMMANDS.iter().find(|c| c.name == name).unwrap()
}

fn arg<T>(value: &T) -> *const c_void {
    value as *const T as *const c_void
}

/// Encodes on the borrower, decodes on the lender
fn carry(command: &CommandInfo, args: &[*const c_void], arena: &mut Arena, lender: &mut Lender) -> Call {
    let mut w = Writer::new();
    unsafe { encode_call(command, args, &mut w, &mut Borrower::default()).unwrap() };
    let bytes = w.as_slice().to_vec();
    let mut r = Reader::new(&bytes);
    let call = unsafe { decode_call(command, &mut r, arena, lender).unwrap() };
    assert!(r.take(1).is_err(), "everything encoded was decoded");
    call
}

#[test]
fn instance_creation_carries_strings() {
    let application = VkApplicationInfo {
        sType: VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pNext: ptr::null(),
        pApplicationName: c"llama.cpp".as_ptr(),
        applicationVersion: 1,
        pEngineName: ptr::null(),
        engineVersion: 0,
        apiVersion: crate::tests::API_1_3,
    };
    let extensions = [c"VK_KHR_get_physical_device_properties2".as_ptr(), c"VK_EXT_debug_utils".as_ptr()];
    let info = VkInstanceCreateInfo {
        sType: VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pNext: ptr::null(),
        flags: 0,
        pApplicationInfo: &application,
        enabledLayerCount: 0,
        ppEnabledLayerNames: ptr::null(),
        enabledExtensionCount: 2,
        ppEnabledExtensionNames: extensions.as_ptr(),
    };
    let allocator: *const VkAllocationCallbacks = ptr::null();
    let mut instance: VkInstance = ptr::null_mut();
    let out = &mut instance as *mut VkInstance;
    let info_ptr = &info as *const VkInstanceCreateInfo;
    let mut arena = Arena::default();
    let call = carry(command("vkCreateInstance"), &[arg(&info_ptr), arg(&allocator), arg(&out)], &mut arena, &mut Lender::default());

    let decoded = unsafe { &*(call.slots[0] as *const VkInstanceCreateInfo) };
    assert_eq!(decoded.enabledExtensionCount, 2);
    let name = unsafe { CStr::from_ptr(*decoded.ppEnabledExtensionNames.add(1)) };
    assert_eq!(name, c"VK_EXT_debug_utils");
    let application = unsafe { &*decoded.pApplicationInfo };
    assert_eq!(unsafe { CStr::from_ptr(application.pApplicationName) }, c"llama.cpp");
    assert!(application.pEngineName.is_null());
    assert_eq!(call.slots[1], 0, "the allocator is never sent");
}

pub const API_1_3: u32 = (1 << 22) | (3 << 12);

#[test]
fn buffer_creation_carries_chains_arrays_and_chosen_handles() {
    let usage2 = VkBufferUsageFlags2CreateInfo {
        sType: VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO,
        pNext: ptr::null(),
        usage: 0x20,
    };
    let families = [3u32, 5];
    let info = VkBufferCreateInfo {
        sType: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        pNext: &usage2 as *const _ as *const c_void,
        flags: 0,
        size: 1 << 20,
        usage: 0x80,
        sharingMode: VK_SHARING_MODE_CONCURRENT,
        queueFamilyIndexCount: 2,
        pQueueFamilyIndices: families.as_ptr(),
    };
    let device = 0x1000_0007usize as VkDevice;
    let info_ptr = &info as *const VkBufferCreateInfo;
    let allocator: *const VkAllocationCallbacks = ptr::null();
    let mut buffer: VkBuffer = 0;
    let out = &mut buffer as *mut VkBuffer;
    let mut arena = Arena::default();
    let mut lender = Lender::default();
    let command = command("vkCreateBuffer");
    let call = carry(command, &[arg(&device), arg(&info_ptr), arg(&allocator), arg(&out)], &mut arena, &mut lender);

    assert_ne!(buffer, 0, "the borrower chose the handle at once");
    assert_eq!(call.slots[0], 7 + 0xA000, "the device is the lender's");
    let decoded = unsafe { &*(call.slots[1] as *const VkBufferCreateInfo) };
    assert_eq!((decoded.size, decoded.usage, decoded.queueFamilyIndexCount), (1 << 20, 0x80, 2));
    assert_eq!(unsafe { *decoded.pQueueFamilyIndices.add(1) }, 5);
    let next = unsafe { &*(decoded.pNext as *const VkBufferUsageFlags2CreateInfo) };
    assert_eq!((next.sType, next.usage), (VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO, 0x20));

    // The driver writes its handle; the lender binds it to the borrower's id
    unsafe { *(call.slots[3] as *mut u64) = 0xBEEF };
    let mut w = Writer::new();
    unsafe { encode_outputs(command, &call, true, &mut w, &mut lender).unwrap() };
    assert_eq!(lender.bound, vec![(buffer, 0xBEEF)]);
    assert!(w.is_empty(), "nothing to send back");
}

#[test]
fn properties_come_back_into_the_programs_chain() {
    let mut driver: VkPhysicalDeviceDriverProperties = unsafe { mem::zeroed() };
    driver.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    // A structure the program chains that has nothing to do with this call is passed over
    let mut unknown = [0u64; 4];
    unknown[0] = 0x7fff_0000;
    driver.pNext = unknown.as_mut_ptr().cast();
    let mut properties: VkPhysicalDeviceProperties2 = unsafe { mem::zeroed() };
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &mut driver as *mut _ as *mut c_void;

    let physical = 0x1000_0002usize as VkPhysicalDevice;
    let out = &mut properties as *mut VkPhysicalDeviceProperties2;
    let args = [arg(&physical), arg(&out)];
    let command = command("vkGetPhysicalDeviceProperties2");
    let mut arena = Arena::default();
    let mut lender = Lender::default();
    let call = carry(command, &args, &mut arena, &mut lender);

    // The lender's driver sees the same chain, less the unknown structure, and fills it
    let filled = unsafe { &mut *(call.slots[1] as *mut VkPhysicalDeviceProperties2) };
    let filled_driver = unsafe { &mut *(filled.pNext as *mut VkPhysicalDeviceDriverProperties) };
    assert_eq!(filled_driver.sType, VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES);
    assert!(filled_driver.pNext.is_null());
    filled.properties.vendorID = 0x10de;
    filled.properties.deviceName[..4].copy_from_slice(&[b'R' as _, b'T' as _, b'X' as _, 0]);
    filled.properties.limits.maxComputeWorkGroupInvocations = 1024;
    filled_driver.driverID = VK_DRIVER_ID_NVIDIA_PROPRIETARY;
    filled_driver.conformanceVersion.major = 1;

    let mut w = Writer::new();
    unsafe { encode_outputs(command, &call, true, &mut w, &mut lender).unwrap() };
    let bytes = w.as_slice().to_vec();
    unsafe { decode_outputs(command, &args, &mut Reader::new(&bytes), &mut Borrower::default()).unwrap() };

    assert_eq!(properties.properties.vendorID, 0x10de);
    assert_eq!(properties.properties.limits.maxComputeWorkGroupInvocations, 1024);
    assert_eq!(unsafe { CStr::from_ptr(properties.properties.deviceName.as_ptr()) }, c"RTX");
    assert_eq!(driver.driverID, VK_DRIVER_ID_NVIDIA_PROPRIETARY);
    assert_eq!(driver.conformanceVersion.major, 1);
    assert_eq!(properties.pNext, &mut driver as *mut _ as *mut c_void, "the program's pointers stay");
    assert_eq!(driver.pNext, unknown.as_mut_ptr().cast(), "and so does the unknown structure");
}

#[test]
fn enumerations_take_two_calls() {
    let command = command("vkEnumerateDeviceExtensionProperties");
    let physical = 0x1000_0002usize as VkPhysicalDevice;
    let layer: *const i8 = ptr::null();

    // First call: the count only
    let mut count = 0u32;
    let count_ptr = &mut count as *mut u32;
    let none: *mut VkExtensionProperties = ptr::null_mut();
    let args = [arg(&physical), arg(&layer), arg(&count_ptr), arg(&none)];
    let mut arena = Arena::default();
    let mut lender = Lender::default();
    let call = carry(command, &args, &mut arena, &mut lender);
    assert_eq!(call.slots[3], 0);
    unsafe { *(call.slots[2] as *mut u32) = 3 };
    let mut w = Writer::new();
    unsafe { encode_outputs(command, &call, true, &mut w, &mut lender).unwrap() };
    let bytes = w.as_slice().to_vec();
    unsafe { decode_outputs(command, &args, &mut Reader::new(&bytes), &mut Borrower::default()).unwrap() };
    assert_eq!(count, 3);

    // Second call: room for 3, and the lender has 2
    let mut list: [VkExtensionProperties; 3] = unsafe { mem::zeroed() };
    let list_ptr = list.as_mut_ptr();
    let args = [arg(&physical), arg(&layer), arg(&count_ptr), arg(&list_ptr)];
    let mut arena = Arena::default();
    let call = carry(command, &args, &mut arena, &mut lender);
    assert_eq!(unsafe { *(call.slots[2] as *const u32) }, 3, "the capacity reaches the lender");
    unsafe {
        *(call.slots[2] as *mut u32) = 2;
        let filled = call.slots[3] as *mut VkExtensionProperties;
        (*filled.add(1)).specVersion = 7;
        (*filled.add(1)).extensionName[0] = b'X' as _;
    }
    let mut w = Writer::new();
    unsafe { encode_outputs(command, &call, true, &mut w, &mut lender).unwrap() };
    let bytes = w.as_slice().to_vec();
    unsafe { decode_outputs(command, &args, &mut Reader::new(&bytes), &mut Borrower::default()).unwrap() };
    assert_eq!(count, 2);
    assert_eq!(list[1].specVersion, 7);
    assert_eq!(list[1].extensionName[0], b'X' as _);
}

#[test]
fn array_parameters_and_floats() {
    let command = command("vkCmdSetBlendConstants");
    let buffer = 0x1000_0009usize as VkCommandBuffer;
    let constants = [0.25f32, 0.5, 0.75, 1.0];
    let constants_ptr = constants.as_ptr();
    let mut arena = Arena::default();
    let call = carry(command, &[arg(&buffer), arg(&constants_ptr)], &mut arena, &mut Lender::default());
    let decoded = unsafe { std::slice::from_raw_parts(call.slots[1] as *const f32, 4) };
    assert_eq!(decoded, &constants);

    let command = self::command("vkCmdSetLineWidth");
    let width = 2.5f32;
    let call = carry(command, &[arg(&buffer), arg(&width)], &mut arena, &mut Lender::default());
    assert_eq!(f32::from_bits(call.slots[1] as u32), 2.5);
}

/// Shader code by hash: kept the first time, and only the hash after that
#[test]
fn shader_code_goes_by_hash() {
    use nectar_cuda_protocol::{Hash, content_hash};
    use std::collections::HashMap;

    #[derive(Default)]
    struct KeepingBorrower {
        inner: Borrower,
        kept: std::collections::HashSet<Hash>,
    }
    impl Handles for KeepingBorrower {
        fn outgoing(&mut self, dispatchable: bool, value: u64) -> u64 {
            self.inner.outgoing(dispatchable, value)
        }
        fn incoming(&mut self, dispatchable: bool, id: u64) -> u64 {
            self.inner.incoming(dispatchable, id)
        }
        fn create(&mut self, dispatchable: bool) -> (u64, u64) {
            self.inner.create(dispatchable)
        }
        fn bind(&mut self, _: u64, _: u64) {}
        fn code_out(&mut self, code: &[u8]) -> CodeWay {
            let hash = content_hash(code);
            if self.kept.insert(hash) { CodeWay::Keep(hash) } else { CodeWay::Kept(hash) }
        }
    }
    #[derive(Default)]
    struct KeepingLender {
        inner: Lender,
        store: HashMap<Hash, Vec<u8>>,
    }
    impl Handles for KeepingLender {
        fn outgoing(&mut self, dispatchable: bool, value: u64) -> u64 {
            self.inner.outgoing(dispatchable, value)
        }
        fn incoming(&mut self, dispatchable: bool, id: u64) -> u64 {
            self.inner.incoming(dispatchable, id)
        }
        fn create(&mut self, dispatchable: bool) -> (u64, u64) {
            self.inner.create(dispatchable)
        }
        fn bind(&mut self, id: u64, real: u64) {
            self.inner.bind(id, real);
        }
        fn code_keep(&mut self, hash: &Hash, code: &[u8]) {
            self.store.insert(*hash, code.to_vec());
        }
        fn code_in(&mut self, hash: &Hash, size: u64) -> Option<Vec<u8>> {
            self.store.get(hash).filter(|code| code.len() as u64 == size).cloned()
        }
    }

    let command = COMMANDS.iter().find(|command| command.name == "vkCreateShaderModule").unwrap();
    let code: Vec<u32> = (0..2048).collect();
    let info = VkShaderModuleCreateInfo {
        sType: VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        pNext: ptr::null(),
        flags: 0,
        codeSize: code.len() * 4,
        pCode: code.as_ptr(),
    };
    let device: u64 = 0x1000_0000 + 7;
    let info_pointer = &info as *const VkShaderModuleCreateInfo;
    let allocator: *const c_void = ptr::null();
    let (mut borrower, mut lender) = (KeepingBorrower::default(), KeepingLender::default());
    let mut sizes = Vec::new();
    for _ in 0..2 {
        let mut module = 0u64;
        let module_pointer = &mut module as *mut u64;
        let args = [
            &device as *const u64 as *const c_void,
            &info_pointer as *const _ as *const c_void,
            &allocator as *const _ as *const c_void,
            &module_pointer as *const _ as *const c_void,
        ];
        let mut w = Writer::new();
        let mut arena = Arena::default();
        let call = unsafe {
            encode_call(command, &args, &mut w, &mut borrower).unwrap();
            let mut r = Reader::new(w.as_slice());
            decode_call(command, &mut r, &mut arena, &mut lender).unwrap()
        };
        sizes.push(w.as_slice().len());
        let decoded = unsafe { &*(call.slots[1] as usize as *const VkShaderModuleCreateInfo) };
        let carried = unsafe { std::slice::from_raw_parts(decoded.pCode, decoded.codeSize / 4) };
        assert_eq!(carried, &code[..]);
    }
    assert!(sizes[0] > code.len() * 4);
    assert!(sizes[1] < 200, "{}", sizes[1]);

    // Code the lender doesn't have fails, rather than arriving wrong
    lender.store.clear();
    let mut module = 0u64;
    let module_pointer = &mut module as *mut u64;
    let args = [
        &device as *const u64 as *const c_void,
        &info_pointer as *const _ as *const c_void,
        &allocator as *const _ as *const c_void,
        &module_pointer as *const _ as *const c_void,
    ];
    let mut w = Writer::new();
    let mut arena = Arena::default();
    unsafe {
        encode_call(command, &args, &mut w, &mut borrower).unwrap();
        let mut r = Reader::new(w.as_slice());
        assert!(decode_call(command, &mut r, &mut arena, &mut lender).is_err());
    }
}

#[test]
fn a_mip_is_made_from_the_one_below_by_repeating_blocks() {
    // 2x1 blocks of 2 bytes become 4x2
    let from = [1u8, 2, 3, 4];
    let mut to = [0u8; 16];
    crate::expand_mip(&from, 2, 1, &mut to, 4, 2, 2, 2);
    assert_eq!(to, [1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4]);
    // Odd sizes: 2x2 below 5x3, whose last column and row repeat what's there
    let from = [1u8, 2, 3, 4];
    let mut to = [0u8; 15];
    crate::expand_mip(&from, 2, 2, &mut to, 5, 3, 1, 2);
    assert_eq!(to, [1, 1, 2, 2, 2, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4]);
    // Nothing to do
    crate::expand_mip(&[], 0, 0, &mut [], 0, 0, 16, 2);
    // Too little to read leaves it as it was
    let mut untouched = [9u8; 4];
    crate::expand_mip(&[1], 2, 2, &mut untouched, 2, 2, 1, 2);
    assert_eq!(untouched, [9u8; 4]);
}

#[test]
fn a_sample_of_a_mip_stands_in_for_it() {
    // 5x3 blocks of 1 byte, sampling every other one
    let bytes: Vec<u8> = (0..15).collect();
    let (samples, across, down) = crate::sample_mip(&bytes, 5, 3, 1, 2);
    assert_eq!((across, down), (3, 2));
    assert_eq!(samples, [0, 2, 4, 10, 12, 14]);
    let mut made = vec![0u8; 15];
    crate::expand_mip(&samples, across, down, &mut made, 5, 3, 1, 2);
    assert_eq!(made, [0, 0, 2, 2, 4, 0, 0, 2, 2, 4, 10, 10, 12, 12, 14]);
}

#[test]
fn what_the_arena_gives_stays_apart_and_starts_clear() {
    use crate::wire::Arena;
    let mut arena = Arena::default();
    let mut given: Vec<(*mut u8, usize)> = Vec::new();
    // A large one first, then small ones, past a block's end, then large and small mixed
    for (round, size) in [100_000usize, 8, 24, 1, 16_384, 16_000, 900, 200_000, 4].into_iter().enumerate() {
        let pointer = arena.alloc(size);
        assert_eq!(pointer as usize % 8, 0);
        for at in 0..size {
            assert_eq!(unsafe { *pointer.add(at) }, 0);
        }
        unsafe { std::ptr::write_bytes(pointer, round as u8 + 1, size) };
        given.push((pointer, size));
    }
    for (round, &(pointer, size)) in given.iter().enumerate() {
        for at in 0..size {
            assert_eq!(unsafe { *pointer.add(at) }, round as u8 + 1, "alloc {round} was written over");
        }
    }
}
