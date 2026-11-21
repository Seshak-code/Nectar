//! Nectar's Vulkan driver for a borrowed GPU: a loader driver (ICD) whose every call is carried
//! out on the lender's GPU (nectar-server, Vulkan personality).
//!
//! The Vulkan loader finds it through its manifest (nectar_icd.json): registered by the
//! installer, or named by VK_DRIVER_FILES / VK_ADD_DRIVER_FILES for testing. While nothing
//! is borrowed it lists no device, so programs carry on with the computer's own GPUs.
//!
//! The lender is NECTAR_VK_SERVER (with NECTAR_VK_TOKEN) for testing, or the GPU nectar-svc
//! has borrowed when `borrow.cfg` lists vulkan.
//!
//! Most commands are generated (generated.rs) and carried by nectar-vk-wire's interpreter.
//! This file holds the loader interface and the commands that need more: the instance and
//! device, the extension list (only what can be carried), mapped memory, and the points where
//! memory is kept in step (before a submit, after a wait).

#![allow(non_snake_case, clippy::missing_safety_doc)]

mod connection;
mod generated;
mod memory;
mod objects;

use std::ffi::{CStr, c_char, c_void};
use std::sync::Mutex;
use std::{env, ptr};

use nectar_cuda_protocol::{Reader, Writer};
use nectar_vk_sys::*;
use nectar_vk_wire::tables::{Mode, Ret};
use nectar_vk_wire::{COMMANDS, EXTENSIONS, command_name, wire};

use objects::{Borrower, id_of};

pub(crate) const DEVICE_LOST: i32 = VK_ERROR_DEVICE_LOST;
/// The loader interface version this driver speaks
const LOADER_INTERFACE: u32 = 5;
/// What the device's name ends with, so programs and nectar-detect can tell it's borrowed
const SUFFIX: &str = " (Nectar)";

pub(crate) fn log(message: &str) {
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let at = START.get_or_init(std::time::Instant::now).elapsed();
    eprintln!("nectar-vk {:>10.3}: {message}", at.as_secs_f64() * 1000.0);
}

/// NECTAR_VK_LOG=1 logs every call
fn logging() -> bool {
    static LOGGING: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *LOGGING.get_or_init(|| env::var_os("NECTAR_VK_LOG").is_some())
}

fn index_of(name: &str) -> Option<usize> {
    COMMANDS.iter().position(|command| command.name == name)
}

/// Carries out COMMANDS[index] on the lender; each argument is given by its address
pub(crate) unsafe fn call(index: usize, args: &[*const c_void]) -> u64 {
    let command = &COMMANDS[index];
    let failed = |result: i32| if command.ret == Ret::Result { result as i64 as u64 } else { 0 };
    let Some(connection) = connection::get() else {
        return failed(DEVICE_LOST);
    };
    if matches!(command.name, "vkQueueSubmit" | "vkQueueSubmit2" | "vkQueueBindSparse") {
        memory::flush(&connection);
    }
    if command.name == "vkFreeMemory" {
        memory::forget(&connection, unsafe { *(args[1] as *const u64) });
    }

    let mut w = Writer::new();
    if let Err(error) = unsafe { wire::encode_call(command, args, &mut w, &mut Borrower) } {
        log(&format!("{} can't be carried: {error:?}", command.name));
        return failed(VK_ERROR_FEATURE_NOT_PRESENT);
    }
    // A device that can't be created must say so at once
    let mode = if command.name == "vkCreateDevice" { Mode::Sync } else { command.mode };
    if mode == Mode::Async {
        if logging() {
            log(command.name);
        }
        return match connection.send(command.id, w.as_slice()) {
            Ok(()) => 0,
            Err(_) => failed(DEVICE_LOST),
        };
    }
    let answer = match connection.ask(command.id, w.as_slice()) {
        Ok(answer) => answer,
        Err(_) => return failed(DEVICE_LOST),
    };
    if answer.deferred != 0 {
        log(&format!("an earlier call failed on the lender with {}", answer.deferred));
    }
    if logging() {
        let (sent, received) = (connection.sent.load(Relaxed), connection.received.load(Relaxed));
        log(&format!("{} = {} (sent {sent}, received {received})", command.name, answer.result as i64));
    }
    let succeeded = command.ret != Ret::Result || (answer.result as i64) >= 0;
    if succeeded && let Err(error) = unsafe { wire::decode_outputs(command, args, &mut answer.outputs(), &mut Borrower) } {
        log(&format!("{}: bad answer: {error:?}", command.name));
        return failed(DEVICE_LOST);
    }
    if succeeded {
        unsafe { after(command.name, args) };
        // The GPU's writes come back once the work is known to be done: a wait or a fence query
        // that says so (VK_SUCCESS, not VK_NOT_READY or VK_TIMEOUT)
        let done = answer.result == VK_SUCCESS as u64;
        if done && (mode == Mode::Wait || matches!(command.name, "vkGetFenceStatus" | "vkGetSemaphoreCounterValue")) {
            memory::refresh(&connection);
        }
    }
    answer.result
}

/// What the borrower adds to a command's answer
unsafe fn after(name: &str, args: &[*const c_void]) {
    unsafe {
        match name {
            "vkGetPhysicalDeviceProperties" => rename(*(args[1] as *const *mut VkPhysicalDeviceProperties)),
            "vkGetPhysicalDeviceProperties2" => {
                rename(&mut (**(args[1] as *const *mut VkPhysicalDeviceProperties2)).properties)
            }
            "vkGetPhysicalDeviceMemoryProperties" => {
                remote_memory(*(args[1] as *const *mut VkPhysicalDeviceMemoryProperties))
            }
            "vkGetPhysicalDeviceMemoryProperties2" => {
                remote_memory(&mut (**(args[1] as *const *mut VkPhysicalDeviceMemoryProperties2)).memoryProperties)
            }
            _ => {}
        }
    }
}

/// Offers the GPU's own memory as device-only, as on a GPU without resizable BAR. The lender's CPU
/// reads it over PCIe, slowly, and each wait would read it all to find what the GPU changed, so
/// programs stage through system memory instead. A GPU whose only memory the CPU can map is its
/// own (integrated or unified) is left as it is.
unsafe fn remote_memory(properties: *mut VkPhysicalDeviceMemoryProperties) {
    const HOST: VkMemoryPropertyFlags =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    const STAGING: VkMemoryPropertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    let Some(properties) = (unsafe { properties.as_mut() }) else { return };
    let count = (properties.memoryTypeCount as usize).min(properties.memoryTypes.len());
    let types = &mut properties.memoryTypes[..count];
    let device = |flags: VkMemoryPropertyFlags| flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT != 0;
    if !types.iter().any(|t| !device(t.propertyFlags) && t.propertyFlags & STAGING == STAGING) {
        return;
    }
    for t in types.iter_mut().filter(|t| device(t.propertyFlags)) {
        t.propertyFlags &= !HOST;
    }
}

/// Marks the device's name as borrowed
unsafe fn rename(properties: *mut VkPhysicalDeviceProperties) {
    if properties.is_null() {
        return;
    }
    let name = unsafe { &mut (*properties).deviceName };
    let length = name.iter().position(|&c| c == 0).unwrap_or(name.len());
    let suffix = SUFFIX.as_bytes();
    let start = length.min(name.len() - 1 - suffix.len());
    for (i, &byte) in suffix.iter().enumerate() {
        name[start + i] = byte as c_char;
    }
    name[start + suffix.len()] = 0;
}

// ---- The loader interface ----

#[unsafe(no_mangle)]
pub unsafe extern "system" fn vk_icdNegotiateLoaderICDInterfaceVersion(version: *mut u32) -> VkResult {
    unsafe {
        if version.is_null() || *version < LOADER_INTERFACE {
            return VK_ERROR_INCOMPATIBLE_DRIVER;
        }
        *version = LOADER_INTERFACE;
    }
    VK_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn vk_icdGetInstanceProcAddr(instance: VkInstance, name: *const c_char) -> PFN_vkVoidFunction {
    unsafe { vkGetInstanceProcAddr(instance, name) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn vk_icdGetPhysicalDeviceProcAddr(instance: VkInstance, name: *const c_char) -> PFN_vkVoidFunction {
    unsafe { vkGetInstanceProcAddr(instance, name) }
}

/// A command by name: written here, or generated
fn lookup(name: &str) -> PFN_vkVoidFunction {
    macro_rules! own {
        ($($f:ident),*) => {
            match name {
                $(stringify!($f) => return unsafe { std::mem::transmute::<PFN_vkVoidFunction, PFN_vkVoidFunction>(Some(std::mem::transmute::<*const (), unsafe extern "system" fn()>($f as *const ()))) },)*
                _ => {}
            }
        };
    }
    let name = command_name(name);
    own!(
        vkGetInstanceProcAddr,
        vkGetDeviceProcAddr,
        vkCreateInstance,
        vkDestroyInstance,
        vkEnumerateInstanceVersion,
        vkEnumerateInstanceExtensionProperties,
        vkEnumerateInstanceLayerProperties,
        vkEnumerateDeviceLayerProperties,
        vkEnumeratePhysicalDevices,
        vkEnumerateDeviceExtensionProperties,
        vkMapMemory,
        vkMapMemory2,
        vkUnmapMemory,
        vkUnmapMemory2,
        vkFlushMappedMemoryRanges,
        vkInvalidateMappedMemoryRanges
    );
    // What can't be carried isn't offered, so the loader answers it or the program goes without
    match index_of(name) {
        Some(index) if COMMANDS[index].unsupported.is_none() => generated::entry(name),
        _ => None,
    }
}

pub unsafe extern "system" fn vkGetInstanceProcAddr(_instance: VkInstance, name: *const c_char) -> PFN_vkVoidFunction {
    if name.is_null() {
        return None;
    }
    lookup(unsafe { CStr::from_ptr(name) }.to_str().unwrap_or_default())
}

pub unsafe extern "system" fn vkGetDeviceProcAddr(_device: VkDevice, name: *const c_char) -> PFN_vkVoidFunction {
    if name.is_null() {
        return None;
    }
    lookup(unsafe { CStr::from_ptr(name) }.to_str().unwrap_or_default())
}

// ---- Instances: made on the borrower at once, on the lender once a device is asked for ----

/// The program's instance: its id, the version it asked for, and whether the lender has it
struct Instance {
    id: u64,
    api_version: u32,
    created: bool,
}

static INSTANCES: Mutex<Vec<Instance>> = Mutex::new(Vec::new());

/// The instance extensions offered: core since Vulkan 1.1, so any lender has them
const INSTANCE_EXTENSIONS: &[(&str, u32)] = &[
    ("VK_KHR_get_physical_device_properties2", 2),
    ("VK_KHR_external_memory_capabilities", 1),
    ("VK_KHR_external_semaphore_capabilities", 1),
    ("VK_KHR_external_fence_capabilities", 1),
];

pub unsafe extern "system" fn vkCreateInstance(
    pCreateInfo: *const VkInstanceCreateInfo,
    _pAllocator: *const VkAllocationCallbacks,
    pInstance: *mut VkInstance,
) -> VkResult {
    unsafe {
        if pCreateInfo.is_null() || pInstance.is_null() {
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        let application = (*pCreateInfo).pApplicationInfo;
        let api_version = if application.is_null() || (*application).apiVersion == 0 {
            VK_API_VERSION_1_0
        } else {
            (*application).apiVersion
        };
        let id = objects::new_id();
        *pInstance = objects::object(id) as usize as VkInstance;
        INSTANCES.lock().unwrap_or_else(|p| p.into_inner()).push(Instance { id, api_version, created: false });
    }
    VK_SUCCESS
}

const VK_API_VERSION_1_0: u32 = 1 << 22;

/// Creates the program's instance on the lender, once
unsafe fn create_remote(instance: VkInstance) -> bool {
    let id = unsafe { id_of(instance as usize as u64) };
    let mut instances = INSTANCES.lock().unwrap_or_else(|p| p.into_inner());
    let Some(entry) = instances.iter_mut().find(|i| i.id == id) else { return false };
    if entry.created {
        return true;
    }
    let Some(connection) = connection::get() else { return false };
    let application = VkApplicationInfo {
        sType: VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pNext: ptr::null(),
        pApplicationName: c"Nectar".as_ptr(),
        applicationVersion: 1,
        pEngineName: ptr::null(),
        engineVersion: 0,
        apiVersion: entry.api_version,
    };
    let info = VkInstanceCreateInfo {
        sType: VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pNext: ptr::null(),
        flags: 0,
        pApplicationInfo: &application,
        enabledLayerCount: 0,
        ppEnabledLayerNames: ptr::null(),
        enabledExtensionCount: 0,
        ppEnabledExtensionNames: ptr::null(),
    };
    let info_ptr = &info as *const VkInstanceCreateInfo;
    let allocator: *const VkAllocationCallbacks = ptr::null();
    let mut handle = instance;
    let out = &mut handle as *mut VkInstance;
    let args = [
        &info_ptr as *const _ as *const c_void,
        &allocator as *const _ as *const c_void,
        &out as *const _ as *const c_void,
    ];
    // The instance's id is the program's object's, not a new one
    struct Same(u64);
    impl wire::Handles for Same {
        fn outgoing(&mut self, d: bool, v: u64) -> u64 {
            Borrower.outgoing(d, v)
        }
        fn incoming(&mut self, d: bool, id: u64) -> u64 {
            Borrower.incoming(d, id)
        }
        fn create(&mut self, _: bool) -> (u64, u64) {
            (self.0, objects::object(self.0))
        }
        fn bind(&mut self, _: u64, _: u64) {}
    }
    let Some(index) = index_of("vkCreateInstance") else { return false };
    let mut w = Writer::new();
    if unsafe { wire::encode_call(&COMMANDS[index], &args, &mut w, &mut Same(id)) }.is_err() {
        return false;
    }
    match connection.ask(COMMANDS[index].id, w.as_slice()) {
        Ok(answer) if answer.result as i64 as i32 == VK_SUCCESS => {
            entry.created = true;
            true
        }
        Ok(answer) => {
            log(&format!("the lender couldn't create an instance: {}", answer.result as i64));
            false
        }
        Err(_) => false,
    }
}

pub unsafe extern "system" fn vkDestroyInstance(instance: VkInstance, pAllocator: *const VkAllocationCallbacks) {
    let id = unsafe { id_of(instance as usize as u64) };
    let created = {
        let mut instances = INSTANCES.lock().unwrap_or_else(|p| p.into_inner());
        let created = instances.iter().any(|i| i.id == id && i.created);
        instances.retain(|i| i.id != id);
        created
    };
    if created && let Some(index) = index_of("vkDestroyInstance") {
        unsafe {
            call(index, &[&instance as *const _ as *const c_void, &pAllocator as *const _ as *const c_void]);
        }
    }
}

pub unsafe extern "system" fn vkEnumerateInstanceVersion(pApiVersion: *mut u32) -> VkResult {
    unsafe { *pApiVersion = nectar_vk_sys::API_VERSION_1_4 };
    VK_SUCCESS
}

/// Fills a two-call list: the count, or as many items as fit, then VK_INCOMPLETE if some didn't
unsafe fn fill<T: Copy>(items: &[T], count: *mut u32, out: *mut T) -> VkResult {
    unsafe {
        if out.is_null() {
            *count = items.len() as u32;
            return VK_SUCCESS;
        }
        let fits = (*count as usize).min(items.len());
        ptr::copy_nonoverlapping(items.as_ptr(), out, fits);
        *count = fits as u32;
        if fits < items.len() { VK_INCOMPLETE } else { VK_SUCCESS }
    }
}

fn extension(name: &str, version: u32) -> VkExtensionProperties {
    let mut properties: VkExtensionProperties = unsafe { std::mem::zeroed() };
    for (i, byte) in name.bytes().take(255).enumerate() {
        properties.extensionName[i] = byte as c_char;
    }
    properties.specVersion = version;
    properties
}

pub unsafe extern "system" fn vkEnumerateInstanceExtensionProperties(
    pLayerName: *const c_char,
    pPropertyCount: *mut u32,
    pProperties: *mut VkExtensionProperties,
) -> VkResult {
    if !pLayerName.is_null() {
        return VK_ERROR_LAYER_NOT_PRESENT;
    }
    let list: Vec<VkExtensionProperties> = INSTANCE_EXTENSIONS.iter().map(|(n, v)| extension(n, *v)).collect();
    unsafe { fill(&list, pPropertyCount, pProperties) }
}

pub unsafe extern "system" fn vkEnumerateInstanceLayerProperties(pPropertyCount: *mut u32, _: *mut VkLayerProperties) -> VkResult {
    unsafe { *pPropertyCount = 0 };
    VK_SUCCESS
}

pub unsafe extern "system" fn vkEnumerateDeviceLayerProperties(_: VkPhysicalDevice, pPropertyCount: *mut u32, _: *mut VkLayerProperties) -> VkResult {
    unsafe { *pPropertyCount = 0 };
    VK_SUCCESS
}

/// No device while nothing is borrowed; the lender's devices when something is
pub unsafe extern "system" fn vkEnumeratePhysicalDevices(
    instance: VkInstance,
    pPhysicalDeviceCount: *mut u32,
    pPhysicalDevices: *mut VkPhysicalDevice,
) -> VkResult {
    unsafe {
        if !create_remote(instance) {
            *pPhysicalDeviceCount = 0;
            return VK_SUCCESS;
        }
        let Some(index) = index_of("vkEnumeratePhysicalDevices") else { return VK_ERROR_INITIALIZATION_FAILED };
        call(
            index,
            &[
                &instance as *const _ as *const c_void,
                &pPhysicalDeviceCount as *const _ as *const c_void,
                &pPhysicalDevices as *const _ as *const c_void,
            ],
        ) as i64 as VkResult
    }
}

/// The lender's device extensions that Nectar can carry
pub unsafe extern "system" fn vkEnumerateDeviceExtensionProperties(
    physicalDevice: VkPhysicalDevice,
    pLayerName: *const c_char,
    pPropertyCount: *mut u32,
    pProperties: *mut VkExtensionProperties,
) -> VkResult {
    if !pLayerName.is_null() {
        return VK_ERROR_LAYER_NOT_PRESENT;
    }
    let Some(index) = index_of("vkEnumerateDeviceExtensionProperties") else { return VK_ERROR_INITIALIZATION_FAILED };
    let mut count = 0u32;
    let layer: *const c_char = ptr::null();
    let none: *mut VkExtensionProperties = ptr::null_mut();
    let count_ptr = &mut count as *mut u32;
    unsafe {
        let args = |list: &*mut VkExtensionProperties| {
            [
                &physicalDevice as *const _ as *const c_void,
                &layer as *const _ as *const c_void,
                &count_ptr as *const _ as *const c_void,
                list as *const _ as *const c_void,
            ]
        };
        let result = call(index, &args(&none)) as i64 as VkResult;
        if result < 0 {
            return result;
        }
        let mut all: Vec<VkExtensionProperties> = vec![std::mem::zeroed(); count as usize];
        let list = all.as_mut_ptr();
        let result = call(index, &args(&list)) as i64 as VkResult;
        if result < 0 {
            return result;
        }
        all.truncate(count as usize);
        let carried: Vec<VkExtensionProperties> = all
            .into_iter()
            .filter(|e| {
                let name = CStr::from_ptr(e.extensionName.as_ptr()).to_str().unwrap_or_default();
                EXTENSIONS.iter().any(|(known, _)| *known == name)
            })
            .collect();
        fill(&carried, pPropertyCount, pProperties)
    }
}

// ---- Mapped memory ----

pub unsafe extern "system" fn vkMapMemory(
    _device: VkDevice,
    memory: VkDeviceMemory,
    offset: VkDeviceSize,
    _size: VkDeviceSize,
    _flags: VkMemoryMapFlags,
    ppData: *mut *mut c_void,
) -> VkResult {
    let Some(connection) = connection::get() else { return DEVICE_LOST };
    match memory::map(&connection, memory) {
        Ok(shadow) => {
            unsafe { *ppData = shadow.add(offset as usize).cast() };
            VK_SUCCESS
        }
        Err(result) => result,
    }
}

pub unsafe extern "system" fn vkMapMemory2(device: VkDevice, pMemoryMapInfo: *const VkMemoryMapInfo, ppData: *mut *mut c_void) -> VkResult {
    unsafe {
        let info = &*pMemoryMapInfo;
        vkMapMemory(device, info.memory, info.offset, info.size, info.flags, ppData)
    }
}

/// The shadow stays until the memory is freed, so mapping again keeps the program's pointer
pub unsafe extern "system" fn vkUnmapMemory(_device: VkDevice, _memory: VkDeviceMemory) {
    if let Some(connection) = connection::get() {
        memory::flush(&connection);
    }
}

pub unsafe extern "system" fn vkUnmapMemory2(device: VkDevice, pMemoryUnmapInfo: *const VkMemoryUnmapInfo) -> VkResult {
    unsafe { vkUnmapMemory(device, (*pMemoryUnmapInfo).memory) };
    VK_SUCCESS
}

pub unsafe extern "system" fn vkFlushMappedMemoryRanges(_device: VkDevice, _count: u32, _ranges: *const VkMappedMemoryRange) -> VkResult {
    if let Some(connection) = connection::get() {
        memory::flush(&connection);
    }
    VK_SUCCESS
}

pub unsafe extern "system" fn vkInvalidateMappedMemoryRanges(_device: VkDevice, _count: u32, _ranges: *const VkMappedMemoryRange) -> VkResult {
    if let Some(connection) = connection::get() {
        memory::refresh(&connection);
    }
    VK_SUCCESS
}
