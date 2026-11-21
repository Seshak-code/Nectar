//! vk-probe: checks a Vulkan driver end to end through the system's Vulkan loader: the
//! device, a storage buffer in mapped memory, a compute shader, and a fence. Run it as is for
//! the computer's own GPUs, and with Nectar's driver (VK_DRIVER_FILES=nectar_icd.json and a
//! lender) to compare what a program sees on a borrowed GPU.
//!
//! Usage: vk-probe [--device <part of its name>] [--mib <buffer size>] [--calls <latency samples>]
//!
//! The shader is SPIR-V written out by hand below, so no shader compiler is needed:
//! `data[i] = data[i] * 2 + i` over a storage buffer, 64 invocations per workgroup.

use std::ffi::{CStr, c_char, c_void};
use std::time::Instant;
use std::{env, mem, process, ptr};

use libloading::Library;
use nectar_vk_sys::*;

/// The shader, as SPIR-V 1.0 words. Ids: 1 main, 2 gl_GlobalInvocationID, 3 void, 4 fn type,
/// 5 uint, 6 uvec3, 7 Input uvec3*, 8 uint[], 9 struct { uint[] }, 10 Uniform struct*,
/// 11 the buffer, 12 Uniform uint*, 14 0u, 15 2u, 16 label, 17-22 values.
const SHADER: &[u32] = &[
    0x0723_0203, 0x0001_0000, 0, 23, 0,
    (2 << 16) | 17, 1,                                  // OpCapability Shader
    (3 << 16) | 14, 0, 1,                               // OpMemoryModel Logical GLSL450
    (6 << 16) | 15, 5, 1, 0x6E69_616D, 0, 2,            // OpEntryPoint GLCompute %1 "main" %2
    (6 << 16) | 16, 1, 17, 64, 1, 1,                    // OpExecutionMode %1 LocalSize 64 1 1
    (4 << 16) | 71, 2, 11, 28,                          // OpDecorate %2 BuiltIn GlobalInvocationId
    (4 << 16) | 71, 8, 6, 4,                            // OpDecorate %8 ArrayStride 4
    (5 << 16) | 72, 9, 0, 35, 0,                        // OpMemberDecorate %9 0 Offset 0
    (3 << 16) | 71, 9, 3,                               // OpDecorate %9 BufferBlock
    (4 << 16) | 71, 11, 34, 0,                          // OpDecorate %11 DescriptorSet 0
    (4 << 16) | 71, 11, 33, 0,                          // OpDecorate %11 Binding 0
    (2 << 16) | 19, 3,                                  // OpTypeVoid %3
    (3 << 16) | 33, 4, 3,                               // OpTypeFunction %4 %3
    (4 << 16) | 21, 5, 32, 0,                           // OpTypeInt %5 32 unsigned
    (4 << 16) | 23, 6, 5, 3,                            // OpTypeVector %6 %5 3
    (4 << 16) | 32, 7, 1, 6,                            // OpTypePointer %7 Input %6
    (3 << 16) | 29, 8, 5,                               // OpTypeRuntimeArray %8 %5
    (3 << 16) | 30, 9, 8,                               // OpTypeStruct %9 %8
    (4 << 16) | 32, 10, 2, 9,                           // OpTypePointer %10 Uniform %9
    (4 << 16) | 32, 12, 2, 5,                           // OpTypePointer %12 Uniform %5
    (4 << 16) | 43, 5, 14, 0,                           // OpConstant %5 %14 0
    (4 << 16) | 43, 5, 15, 2,                           // OpConstant %5 %15 2
    (4 << 16) | 59, 7, 2, 1,                            // OpVariable %7 %2 Input
    (4 << 16) | 59, 10, 11, 2,                          // OpVariable %10 %11 Uniform
    (5 << 16) | 54, 3, 1, 0, 4,                         // OpFunction %3 %1 None %4
    (2 << 16) | 248, 16,                                // OpLabel %16
    (4 << 16) | 61, 6, 17, 2,                           // %17 = OpLoad %6 %2
    (5 << 16) | 81, 5, 18, 17, 0,                       // %18 = OpCompositeExtract %5 %17 0
    (6 << 16) | 65, 12, 19, 11, 14, 18,                 // %19 = OpAccessChain %12 %11 %14 %18
    (4 << 16) | 61, 5, 20, 19,                          // %20 = OpLoad %5 %19
    (5 << 16) | 132, 5, 21, 20, 15,                     // %21 = OpIMul %5 %20 %15
    (5 << 16) | 128, 5, 22, 21, 18,                     // %22 = OpIAdd %5 %21 %18
    (3 << 16) | 62, 19, 22,                             // OpStore %19 %22
    (1 << 16) | 253,                                    // OpReturn
    (1 << 16) | 56,                                     // OpFunctionEnd
];

const HOST_VISIBLE: u32 = 0x2;
const HOST_COHERENT: u32 = 0x4;
const HOST_CACHED: u32 = 0x8;

fn fail(what: &str, result: VkResult) -> ! {
    println!("FAIL {what}: VkResult {result}");
    process::exit(1);
}

fn check(what: &str, result: VkResult) {
    if result != VK_SUCCESS {
        fail(what, result);
    }
}

fn text(chars: &[c_char]) -> String {
    unsafe { CStr::from_ptr(chars.as_ptr()) }.to_string_lossy().into_owned()
}

fn main() {
    let mut wanted = String::new();
    let mut mib = 64usize;
    let mut calls = 200usize;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match (arg.as_str(), args.next()) {
            ("--device", Some(value)) => wanted = value,
            ("--mib", Some(value)) => mib = value.parse().expect("--mib needs a number"),
            ("--calls", Some(value)) => calls = value.parse().expect("--calls needs a number"),
            _ => {
                eprintln!("usage: vk-probe [--device <part of its name>] [--mib <size>] [--calls <count>]");
                process::exit(2);
            }
        }
    }
    unsafe { run(&wanted, mib, calls) }
}

unsafe fn run(wanted: &str, mib: usize, calls: usize) {
    let names: &[&str] = if cfg!(windows) {
        &["vulkan-1.dll"]
    } else if cfg!(target_os = "macos") {
        &["libvulkan.1.dylib", "libvulkan.dylib"]
    } else {
        &["libvulkan.so.1"]
    };
    let Some((name, library)) = names.iter().find_map(|n| unsafe { Library::new(n) }.ok().map(|l| (n, l))) else {
        println!("FAIL no Vulkan loader ({})", names.join(", "));
        process::exit(1);
    };
    println!("library      {name} (the Vulkan loader)");
    let started = Instant::now();
    unsafe {
        let gipa: PFN_vkGetInstanceProcAddr = *library.get(b"vkGetInstanceProcAddr\0").expect("no vkGetInstanceProcAddr");
        macro_rules! instance_fn {
            ($instance:expr, $name:ident) => {{
                let name = concat!(stringify!($name), "\0");
                let f = gipa($instance, name.as_ptr().cast());
                mem::transmute::<PFN_vkVoidFunction, Option<concat_idents_pfn!($name)>>(f)
                    .unwrap_or_else(|| panic!("no {}", stringify!($name)))
            }};
        }
        macro_rules! concat_idents_pfn {
            (vkCreateInstance) => { PFN_vkCreateInstance };
            (vkEnumeratePhysicalDevices) => { PFN_vkEnumeratePhysicalDevices };
            (vkGetPhysicalDeviceProperties2) => { PFN_vkGetPhysicalDeviceProperties2 };
            (vkGetPhysicalDeviceQueueFamilyProperties) => { PFN_vkGetPhysicalDeviceQueueFamilyProperties };
            (vkGetPhysicalDeviceMemoryProperties) => { PFN_vkGetPhysicalDeviceMemoryProperties };
            (vkCreateDevice) => { PFN_vkCreateDevice };
            (vkGetDeviceProcAddr) => { PFN_vkGetDeviceProcAddr };
            (vkDestroyInstance) => { PFN_vkDestroyInstance };
        }

        let application = VkApplicationInfo {
            sType: VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pNext: ptr::null(),
            pApplicationName: c"vk-probe".as_ptr(),
            applicationVersion: 1,
            pEngineName: ptr::null(),
            engineVersion: 0,
            apiVersion: make_api_version(0, 1, 2, 0),
        };
        let create_info = VkInstanceCreateInfo {
            sType: VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            pApplicationInfo: &application,
            enabledLayerCount: 0,
            ppEnabledLayerNames: ptr::null(),
            enabledExtensionCount: 0,
            ppEnabledExtensionNames: ptr::null(),
        };
        let mut instance = ptr::null_mut();
        check("vkCreateInstance", instance_fn!(ptr::null_mut(), vkCreateInstance)(&create_info, ptr::null(), &mut instance));

        let enumerate = instance_fn!(instance, vkEnumeratePhysicalDevices);
        let mut count = 0;
        check("vkEnumeratePhysicalDevices", enumerate(instance, &mut count, ptr::null_mut()));
        let mut devices = vec![ptr::null_mut(); count as usize];
        check("vkEnumeratePhysicalDevices", enumerate(instance, &mut count, devices.as_mut_ptr()));
        let properties2 = instance_fn!(instance, vkGetPhysicalDeviceProperties2);
        let mut chosen = None;
        for (index, &device) in devices.iter().enumerate() {
            let mut driver: VkPhysicalDeviceDriverProperties = mem::zeroed();
            driver.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
            let mut properties: VkPhysicalDeviceProperties2 = mem::zeroed();
            properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
            properties.pNext = (&mut driver as *mut VkPhysicalDeviceDriverProperties).cast();
            properties2(device, &mut properties);
            let name = text(&properties.properties.deviceName);
            let api = properties.properties.apiVersion;
            println!(
                "device {index}     {name}, Vulkan {}.{}.{}, driver {} {}",
                (api >> 22) & 0x7f,
                (api >> 12) & 0x3ff,
                api & 0xfff,
                text(&driver.driverName),
                text(&driver.driverInfo)
            );
            if chosen.is_none() && name.contains(wanted) {
                chosen = Some((device, name));
            }
        }
        let Some((physical, device_name)) = chosen else {
            println!("FAIL no device{}", if wanted.is_empty() { String::new() } else { format!(" named like {wanted}") });
            process::exit(1);
        };

        let mut families = 0;
        let queue_properties = instance_fn!(instance, vkGetPhysicalDeviceQueueFamilyProperties);
        queue_properties(physical, &mut families, ptr::null_mut());
        let mut family_list = vec![mem::zeroed::<VkQueueFamilyProperties>(); families as usize];
        queue_properties(physical, &mut families, family_list.as_mut_ptr());
        let Some(family) = family_list.iter().position(|f| f.queueFlags & VK_QUEUE_COMPUTE_BIT != 0) else {
            println!("FAIL no compute queue");
            process::exit(1);
        };
        let priority = 1.0f32;
        let queue_info = VkDeviceQueueCreateInfo {
            sType: VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            queueFamilyIndex: family as u32,
            queueCount: 1,
            pQueuePriorities: &priority,
        };
        let device_info = VkDeviceCreateInfo {
            sType: VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            queueCreateInfoCount: 1,
            pQueueCreateInfos: &queue_info,
            enabledLayerCount: 0,
            ppEnabledLayerNames: ptr::null(),
            enabledExtensionCount: 0,
            ppEnabledExtensionNames: ptr::null(),
            pEnabledFeatures: ptr::null(),
        };
        let mut device = ptr::null_mut();
        check("vkCreateDevice", instance_fn!(instance, vkCreateDevice)(physical, &device_info, ptr::null(), &mut device));
        let gdpa = instance_fn!(instance, vkGetDeviceProcAddr);
        macro_rules! device_fn {
            ($name:ident, $pfn:ty) => {{
                let name = concat!(stringify!($name), "\0");
                mem::transmute::<PFN_vkVoidFunction, Option<$pfn>>(gdpa(device, name.as_ptr().cast()))
                    .unwrap_or_else(|| panic!("no {}", stringify!($name)))
            }};
        }

        let mut queue = ptr::null_mut();
        device_fn!(vkGetDeviceQueue, PFN_vkGetDeviceQueue)(device, family as u32, 0, &mut queue);

        // The buffer, in memory the host sees: cached if there is such, for reading back
        let size = (mib << 20) as u64;
        let elements = (size / 4) as usize;
        let buffer_info = VkBufferCreateInfo {
            sType: VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            size,
            usage: VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            sharingMode: VK_SHARING_MODE_EXCLUSIVE,
            queueFamilyIndexCount: 0,
            pQueueFamilyIndices: ptr::null(),
        };
        let mut buffer = 0;
        check("vkCreateBuffer", device_fn!(vkCreateBuffer, PFN_vkCreateBuffer)(device, &buffer_info, ptr::null(), &mut buffer));
        let mut requirements: VkMemoryRequirements = mem::zeroed();
        device_fn!(vkGetBufferMemoryRequirements, PFN_vkGetBufferMemoryRequirements)(device, buffer, &mut requirements);
        let mut memory_properties: VkPhysicalDeviceMemoryProperties = mem::zeroed();
        instance_fn!(instance, vkGetPhysicalDeviceMemoryProperties)(physical, &mut memory_properties);
        let types = &memory_properties.memoryTypes[..memory_properties.memoryTypeCount as usize];
        let fits = |i: usize, flags: u32| requirements.memoryTypeBits & (1 << i) != 0 && types[i].propertyFlags & flags == flags;
        let Some(memory_type) = (0..types.len())
            .find(|&i| fits(i, HOST_VISIBLE | HOST_COHERENT | HOST_CACHED))
            .or_else(|| (0..types.len()).find(|&i| fits(i, HOST_VISIBLE | HOST_COHERENT)))
        else {
            println!("FAIL no host-visible memory for the buffer");
            process::exit(1);
        };
        let allocate_info = VkMemoryAllocateInfo {
            sType: VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            pNext: ptr::null(),
            allocationSize: requirements.size,
            memoryTypeIndex: memory_type as u32,
        };
        let mut memory = 0;
        check("vkAllocateMemory", device_fn!(vkAllocateMemory, PFN_vkAllocateMemory)(device, &allocate_info, ptr::null(), &mut memory));
        check("vkBindBufferMemory", device_fn!(vkBindBufferMemory, PFN_vkBindBufferMemory)(device, buffer, memory, 0));
        let mut mapped = ptr::null_mut();
        check("vkMapMemory", device_fn!(vkMapMemory, PFN_vkMapMemory)(device, memory, 0, size, 0, &mut mapped));
        let data = std::slice::from_raw_parts_mut(mapped as *mut u32, elements);

        // The pipeline
        let binding = VkDescriptorSetLayoutBinding {
            binding: 0,
            descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount: 1,
            stageFlags: VK_SHADER_STAGE_COMPUTE_BIT,
            pImmutableSamplers: ptr::null(),
        };
        let layout_info = VkDescriptorSetLayoutCreateInfo {
            sType: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            bindingCount: 1,
            pBindings: &binding,
        };
        let mut set_layout = 0;
        check(
            "vkCreateDescriptorSetLayout",
            device_fn!(vkCreateDescriptorSetLayout, PFN_vkCreateDescriptorSetLayout)(device, &layout_info, ptr::null(), &mut set_layout),
        );
        let pipeline_layout_info = VkPipelineLayoutCreateInfo {
            sType: VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            setLayoutCount: 1,
            pSetLayouts: &set_layout,
            pushConstantRangeCount: 0,
            pPushConstantRanges: ptr::null(),
        };
        let mut pipeline_layout = 0;
        check(
            "vkCreatePipelineLayout",
            device_fn!(vkCreatePipelineLayout, PFN_vkCreatePipelineLayout)(device, &pipeline_layout_info, ptr::null(), &mut pipeline_layout),
        );
        let module_info = VkShaderModuleCreateInfo {
            sType: VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            codeSize: SHADER.len() * 4,
            pCode: SHADER.as_ptr(),
        };
        let mut module = 0;
        check("vkCreateShaderModule", device_fn!(vkCreateShaderModule, PFN_vkCreateShaderModule)(device, &module_info, ptr::null(), &mut module));
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
            layout: pipeline_layout,
            basePipelineHandle: 0,
            basePipelineIndex: -1,
        };
        let mut pipeline = 0;
        check(
            "vkCreateComputePipelines",
            device_fn!(vkCreateComputePipelines, PFN_vkCreateComputePipelines)(device, 0, 1, &pipeline_info, ptr::null(), &mut pipeline),
        );
        let pool_size = VkDescriptorPoolSize { r#type: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount: 1 };
        let pool_info = VkDescriptorPoolCreateInfo {
            sType: VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            maxSets: 1,
            poolSizeCount: 1,
            pPoolSizes: &pool_size,
        };
        let mut descriptor_pool = 0;
        check(
            "vkCreateDescriptorPool",
            device_fn!(vkCreateDescriptorPool, PFN_vkCreateDescriptorPool)(device, &pool_info, ptr::null(), &mut descriptor_pool),
        );
        let set_info = VkDescriptorSetAllocateInfo {
            sType: VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            pNext: ptr::null(),
            descriptorPool: descriptor_pool,
            descriptorSetCount: 1,
            pSetLayouts: &set_layout,
        };
        let mut set = 0;
        check("vkAllocateDescriptorSets", device_fn!(vkAllocateDescriptorSets, PFN_vkAllocateDescriptorSets)(device, &set_info, &mut set));
        let buffer_descriptor = VkDescriptorBufferInfo { buffer, offset: 0, range: size };
        let write = VkWriteDescriptorSet {
            sType: VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            pNext: ptr::null(),
            dstSet: set,
            dstBinding: 0,
            dstArrayElement: 0,
            descriptorCount: 1,
            descriptorType: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pImageInfo: ptr::null(),
            pBufferInfo: &buffer_descriptor,
            pTexelBufferView: ptr::null(),
        };
        device_fn!(vkUpdateDescriptorSets, PFN_vkUpdateDescriptorSets)(device, 1, &write, 0, ptr::null());

        // A command buffer that runs the shader over the whole buffer
        let command_pool_info = VkCommandPoolCreateInfo {
            sType: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            pNext: ptr::null(),
            flags: 0,
            queueFamilyIndex: family as u32,
        };
        let mut command_pool = 0;
        check("vkCreateCommandPool", device_fn!(vkCreateCommandPool, PFN_vkCreateCommandPool)(device, &command_pool_info, ptr::null(), &mut command_pool));
        let command_info = VkCommandBufferAllocateInfo {
            sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            pNext: ptr::null(),
            commandPool: command_pool,
            level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount: 1,
        };
        let mut commands = ptr::null_mut();
        check("vkAllocateCommandBuffers", device_fn!(vkAllocateCommandBuffers, PFN_vkAllocateCommandBuffers)(device, &command_info, &mut commands));
        let begin = VkCommandBufferBeginInfo {
            sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext: ptr::null(),
            flags: 0,
            pInheritanceInfo: ptr::null(),
        };
        check("vkBeginCommandBuffer", device_fn!(vkBeginCommandBuffer, PFN_vkBeginCommandBuffer)(commands, &begin));
        device_fn!(vkCmdBindPipeline, PFN_vkCmdBindPipeline)(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        device_fn!(vkCmdBindDescriptorSets, PFN_vkCmdBindDescriptorSets)(
            commands,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout,
            0,
            1,
            &set,
            0,
            ptr::null(),
        );
        device_fn!(vkCmdDispatch, PFN_vkCmdDispatch)(commands, (elements / 64) as u32, 1, 1);
        // Makes the shader's writes visible to the host once the fence signals
        let barrier = VkMemoryBarrier {
            sType: VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            pNext: ptr::null(),
            srcAccessMask: VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask: VK_ACCESS_HOST_READ_BIT,
        };
        device_fn!(vkCmdPipelineBarrier, PFN_vkCmdPipelineBarrier)(
            commands,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT,
            0,
            1,
            &barrier,
            0,
            ptr::null(),
            0,
            ptr::null(),
        );
        check("vkEndCommandBuffer", device_fn!(vkEndCommandBuffer, PFN_vkEndCommandBuffer)(commands));

        let fence_info = VkFenceCreateInfo { sType: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, pNext: ptr::null(), flags: 0 };
        let mut fence = 0;
        check("vkCreateFence", device_fn!(vkCreateFence, PFN_vkCreateFence)(device, &fence_info, ptr::null(), &mut fence));
        println!("setup        {:.1} ms (instance, device, buffer, pipeline)", started.elapsed().as_secs_f64() * 1000.0);

        // Round trips: a call whose answer the program needs
        let status = device_fn!(vkGetFenceStatus, PFN_vkGetFenceStatus)(device, fence);
        if status != VK_NOT_READY {
            fail("vkGetFenceStatus of a new fence", status);
        }
        let get_status = device_fn!(vkGetFenceStatus, PFN_vkGetFenceStatus);
        let timer = Instant::now();
        for _ in 0..calls {
            get_status(device, fence);
        }
        println!("latency      {:.1} us per call ({calls} calls of vkGetFenceStatus)", timer.elapsed().as_secs_f64() * 1e6 / calls as f64);

        // The program writes the buffer, the GPU computes over it, the program reads it back
        let submit = device_fn!(vkQueueSubmit, PFN_vkQueueSubmit);
        let wait = device_fn!(vkWaitForFences, PFN_vkWaitForFences);
        let reset = device_fn!(vkResetFences, PFN_vkResetFences);
        let submit_info = VkSubmitInfo {
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
        for (i, value) in data.iter_mut().enumerate() {
            *value = i as u32;
        }
        let timer = Instant::now();
        check("vkQueueSubmit", submit(queue, 1, &submit_info, fence));
        check("vkWaitForFences", wait(device, 1, &fence, VK_TRUE, u64::MAX));
        let first = timer.elapsed().as_secs_f64();
        let wrong = data.iter().enumerate().filter(|&(i, &v)| v != (i as u32).wrapping_mul(3)).count();
        if wrong != 0 {
            let at = data.iter().enumerate().position(|(i, &v)| v != (i as u32).wrapping_mul(3)).unwrap();
            println!("FAIL compute: {wrong} of {elements} wrong, first at {at}: {} instead of {}", data[at], at * 3);
            process::exit(1);
        }
        println!(
            "compute      {elements} elements correct; write, dispatch, wait and read back {:.1} ms ({:.0} MB/s each way)",
            first * 1000.0,
            2.0 * size as f64 / first / 1e6 / 2.0
        );

        // Again on what is now there: only the GPU's changes come back
        check("vkResetFences", reset(device, 1, &fence));
        let timer = Instant::now();
        check("vkQueueSubmit", submit(queue, 1, &submit_info, fence));
        check("vkWaitForFences", wait(device, 1, &fence, VK_TRUE, u64::MAX));
        let second = timer.elapsed().as_secs_f64();
        let wrong = data.iter().enumerate().filter(|&(i, &v)| v != (i as u32).wrapping_mul(7)).count();
        if wrong != 0 {
            println!("FAIL second dispatch: {wrong} of {elements} wrong");
            process::exit(1);
        }
        println!("again        {elements} elements correct; dispatch, wait and read back {:.1} ms", second * 1000.0);

        device_fn!(vkDeviceWaitIdle, PFN_vkDeviceWaitIdle)(device);
        device_fn!(vkDestroyFence, PFN_vkDestroyFence)(device, fence, ptr::null());
        device_fn!(vkDestroyCommandPool, PFN_vkDestroyCommandPool)(device, command_pool, ptr::null());
        device_fn!(vkDestroyDescriptorPool, PFN_vkDestroyDescriptorPool)(device, descriptor_pool, ptr::null());
        device_fn!(vkDestroyPipeline, PFN_vkDestroyPipeline)(device, pipeline, ptr::null());
        device_fn!(vkDestroyShaderModule, PFN_vkDestroyShaderModule)(device, module, ptr::null());
        device_fn!(vkDestroyPipelineLayout, PFN_vkDestroyPipelineLayout)(device, pipeline_layout, ptr::null());
        device_fn!(vkDestroyDescriptorSetLayout, PFN_vkDestroyDescriptorSetLayout)(device, set_layout, ptr::null());
        device_fn!(vkUnmapMemory, PFN_vkUnmapMemory)(device, memory);
        device_fn!(vkDestroyBuffer, PFN_vkDestroyBuffer)(device, buffer, ptr::null());
        device_fn!(vkFreeMemory, PFN_vkFreeMemory)(device, memory, ptr::null());
        device_fn!(vkDestroyDevice, PFN_vkDestroyDevice)(device, ptr::null());
        instance_fn!(instance, vkDestroyInstance)(instance, ptr::null());
        println!("PASS {device_name}: {mib} MiB through mapped memory and a compute shader, correct both times");
    }
}
