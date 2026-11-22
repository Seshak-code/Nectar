//! vk-probe --present: a window, a swapchain, and frames of a known colour, checked on screen.

use std::ffi::CStr;
use std::time::Instant;
use std::{mem, ptr};

use nectar_vk_sys::*;

use crate::{check, fail};

#[cfg(windows)]
mod platform {
    use std::ffi::{CStr, c_void};
    use std::{mem, ptr};

    use nectar_vk_sys::*;
    use windows_sys::Win32::Foundation::{HWND, LPARAM, LRESULT, RECT, WPARAM};
    use windows_sys::Win32::Graphics::Gdi::{GetDC, GetPixel, ReleaseDC};
    use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows_sys::Win32::UI::WindowsAndMessaging::*;

    unsafe extern "system" fn window_proc(window: HWND, message: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
        unsafe { DefWindowProcW(window, message, wparam, lparam) }
    }

    pub struct Window {
        window: HWND,
        module: *mut c_void,
    }

    /// A window whose client area is `width` by `height`
    pub unsafe fn open(width: i32, height: i32) -> Window {
        unsafe {
            let module = GetModuleHandleW(ptr::null());
            let name: Vec<u16> = "vk-probe".encode_utf16().chain(Some(0)).collect();
            let class = WNDCLASSW {
                style: 0,
                lpfnWndProc: Some(window_proc),
                cbClsExtra: 0,
                cbWndExtra: 0,
                hInstance: module,
                hIcon: ptr::null_mut(),
                hCursor: ptr::null_mut(),
                hbrBackground: ptr::null_mut(),
                lpszMenuName: ptr::null(),
                lpszClassName: name.as_ptr(),
            };
            RegisterClassW(&class);
            let mut rect = RECT { left: 0, top: 0, right: width, bottom: height };
            AdjustWindowRect(&mut rect, WS_OVERLAPPEDWINDOW, 0);
            let window = CreateWindowExW(
                WS_EX_TOPMOST,
                name.as_ptr(),
                name.as_ptr(),
                WS_OVERLAPPEDWINDOW & !WS_THICKFRAME,
                0,
                0,
                rect.right - rect.left,
                rect.bottom - rect.top,
                ptr::null_mut(),
                ptr::null_mut(),
                module,
                ptr::null(),
            );
            ShowWindow(window, SW_SHOWNOACTIVATE);
            Window { window, module }
        }
    }

    impl Window {
        pub unsafe fn surface(&self, instance: VkInstance, function: &dyn Fn(&CStr) -> PFN_vkVoidFunction) -> (VkResult, VkSurfaceKHR) {
            #[repr(C)]
            struct Info {
                s_type: VkStructureType,
                next: *const c_void,
                flags: u32,
                instance: *mut c_void,
                window: HWND,
            }
            type Create = unsafe extern "system" fn(VkInstance, *const Info, *const c_void, *mut VkSurfaceKHR) -> VkResult;
            const VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR: VkStructureType = 1000009000;
            unsafe {
                let create = mem::transmute::<PFN_vkVoidFunction, Option<Create>>(function(c"vkCreateWin32SurfaceKHR")).expect("no vkCreateWin32SurfaceKHR");
                let info = Info {
                    s_type: VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
                    next: ptr::null(),
                    flags: 0,
                    instance: self.module,
                    window: self.window,
                };
                let mut surface = 0;
                (create(instance, &info, ptr::null(), &mut surface), surface)
            }
        }

        pub unsafe fn pump(&self) {
            unsafe {
                let mut message: MSG = mem::zeroed();
                while PeekMessageW(&mut message, ptr::null_mut(), 0, 0, PM_REMOVE) != 0 {
                    DispatchMessageW(&message);
                }
            }
        }

        /// The colour on screen at a point, as red, green and blue
        pub unsafe fn pixel(&self, x: i32, y: i32) -> (u32, u32, u32) {
            unsafe {
                let dc = GetDC(self.window);
                let pixel = GetPixel(dc, x, y);
                ReleaseDC(self.window, dc);
                (pixel & 0xff, (pixel >> 8) & 0xff, (pixel >> 16) & 0xff)
            }
        }

        pub unsafe fn close(&self) {
            unsafe { DestroyWindow(self.window) };
        }
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use std::ffi::{CStr, c_void};
    use std::{mem, ptr};

    use libloading::Library;
    use nectar_vk_sys::*;

    type Pointer = *mut c_void;

    /// An X window, through libX11
    pub struct Window {
        library: Library,
        display: Pointer,
        window: u64,
    }

    macro_rules! x {
        ($self:expr, $name:literal, $type:ty) => {
            *$self.library.get::<$type>($name).expect(stringify!($name))
        };
    }

    /// A window `width` by `height`. Exits if there's no X server to show it on.
    pub unsafe fn open(width: i32, height: i32) -> Window {
        unsafe {
            let Ok(library) = Library::new("libX11.so.6") else {
                println!("FAIL no libX11.so.6 for a window");
                std::process::exit(1);
            };
            let open_display: unsafe extern "C" fn(*const i8) -> Pointer = *library.get(b"XOpenDisplay\0").expect("XOpenDisplay");
            let display = open_display(ptr::null());
            if display.is_null() {
                println!("FAIL no X server to open a window on (DISPLAY)");
                std::process::exit(1);
            }
            let root: unsafe extern "C" fn(Pointer) -> u64 = *library.get(b"XDefaultRootWindow\0").expect("XDefaultRootWindow");
            let create: unsafe extern "C" fn(Pointer, u64, i32, i32, u32, u32, u32, u64, u64) -> u64 =
                *library.get(b"XCreateSimpleWindow\0").expect("XCreateSimpleWindow");
            let map: unsafe extern "C" fn(Pointer, u64) -> i32 = *library.get(b"XMapWindow\0").expect("XMapWindow");
            let sync: unsafe extern "C" fn(Pointer, i32) -> i32 = *library.get(b"XSync\0").expect("XSync");
            let window = create(display, root(display), 0, 0, width as u32, height as u32, 0, 0, 0);
            map(display, window);
            sync(display, 0);
            std::thread::sleep(std::time::Duration::from_millis(500));
            Window { library, display, window }
        }
    }

    impl Window {
        pub unsafe fn surface(&self, instance: VkInstance, function: &dyn Fn(&CStr) -> PFN_vkVoidFunction) -> (VkResult, VkSurfaceKHR) {
            #[repr(C)]
            struct Info {
                s_type: VkStructureType,
                next: *const c_void,
                flags: u32,
                display: Pointer,
                window: u64,
            }
            type Create = unsafe extern "system" fn(VkInstance, *const Info, *const c_void, *mut VkSurfaceKHR) -> VkResult;
            const VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR: VkStructureType = 1000004000;
            unsafe {
                let create = mem::transmute::<PFN_vkVoidFunction, Option<Create>>(function(c"vkCreateXlibSurfaceKHR")).expect("no vkCreateXlibSurfaceKHR");
                let info = Info {
                    s_type: VK_STRUCTURE_TYPE_XLIB_SURFACE_CREATE_INFO_KHR,
                    next: ptr::null(),
                    flags: 0,
                    display: self.display,
                    window: self.window,
                };
                let mut surface = 0;
                (create(instance, &info, ptr::null(), &mut surface), surface)
            }
        }

        pub unsafe fn pump(&self) {}

        /// The colour on screen at a point, as red, green and blue
        pub unsafe fn pixel(&self, x: i32, y: i32) -> (u32, u32, u32) {
            const ALL_PLANES: u64 = !0;
            const Z_PIXMAP: i32 = 2;
            unsafe {
                let get_image = x!(self, b"XGetImage\0", unsafe extern "C" fn(Pointer, u64, i32, i32, u32, u32, u64, i32) -> Pointer);
                let get_pixel = x!(self, b"XGetPixel\0", unsafe extern "C" fn(Pointer, i32, i32) -> u64);
                let free = x!(self, b"XFree\0", unsafe extern "C" fn(Pointer) -> i32);
                let image = get_image(self.display, self.window, x, y, 1, 1, ALL_PLANES, Z_PIXMAP);
                if image.is_null() {
                    return (0, 0, 0);
                }
                let pixel = get_pixel(image, 0, 0) as u32;
                free(image);
                ((pixel >> 16) & 0xff, (pixel >> 8) & 0xff, pixel & 0xff)
            }
        }

        pub unsafe fn close(&self) {
            unsafe {
                let destroy = x!(self, b"XDestroyWindow\0", unsafe extern "C" fn(Pointer, u64) -> i32);
                destroy(self.display, self.window);
            }
        }
    }
}

/// What the frame numbered `frame` shows
fn colour_of(frame: u32) -> [f32; 3] {
    match frame % 3 {
        0 => [0.9, 0.1, 0.1],
        1 => [0.1, 0.8, 0.2],
        _ => [0.1, 0.2, 0.9],
    }
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn run(
    instance: VkInstance,
    physical: VkPhysicalDevice,
    device: VkDevice,
    queue: VkQueue,
    family: u32,
    frames: u32,
    (width, height): (i32, i32),
    instance_function: &dyn Fn(&CStr) -> PFN_vkVoidFunction,
    device_function: &dyn Fn(&CStr) -> PFN_vkVoidFunction,
) {
    macro_rules! get {
        ($from:ident, $name:ident, $pfn:ty) => {
            mem::transmute::<PFN_vkVoidFunction, Option<$pfn>>($from(&std::ffi::CString::new(stringify!($name)).unwrap()))
                .unwrap_or_else(|| panic!("no {}", stringify!($name)))
        };
    }
    unsafe {
        let window = platform::open(width, height);
        window.pump();
        let (created, surface) = window.surface(instance, instance_function);
        check("creating the surface", created);

        let mut supported = 0;
        check(
            "vkGetPhysicalDeviceSurfaceSupportKHR",
            get!(instance_function, vkGetPhysicalDeviceSurfaceSupportKHR, PFN_vkGetPhysicalDeviceSurfaceSupportKHR)(
                physical, family, surface, &mut supported,
            ),
        );
        if supported == 0 {
            fail("the surface isn't supported on this queue", 0);
        }
        let mut capabilities: VkSurfaceCapabilitiesKHR = mem::zeroed();
        check(
            "vkGetPhysicalDeviceSurfaceCapabilitiesKHR",
            get!(instance_function, vkGetPhysicalDeviceSurfaceCapabilitiesKHR, PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR)(
                physical, surface, &mut capabilities,
            ),
        );

        let swapchain_info = VkSwapchainCreateInfoKHR {
            sType: VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            pNext: ptr::null(),
            flags: 0,
            surface,
            minImageCount: capabilities.minImageCount.max(3),
            imageFormat: VK_FORMAT_B8G8R8A8_UNORM,
            imageColorSpace: VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
            imageExtent: capabilities.currentExtent,
            imageArrayLayers: 1,
            imageUsage: VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            imageSharingMode: VK_SHARING_MODE_EXCLUSIVE,
            queueFamilyIndexCount: 0,
            pQueueFamilyIndices: ptr::null(),
            preTransform: VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
            compositeAlpha: VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            presentMode: VK_PRESENT_MODE_FIFO_KHR,
            clipped: 0,
            oldSwapchain: 0,
        };
        let mut swapchain = 0;
        check("vkCreateSwapchainKHR", get!(device_function, vkCreateSwapchainKHR, PFN_vkCreateSwapchainKHR)(device, &swapchain_info, ptr::null(), &mut swapchain));
        let get_images = get!(device_function, vkGetSwapchainImagesKHR, PFN_vkGetSwapchainImagesKHR);
        let mut count = 0;
        check("vkGetSwapchainImagesKHR", get_images(device, swapchain, &mut count, ptr::null_mut()));
        let mut images = vec![0; count as usize];
        check("vkGetSwapchainImagesKHR", get_images(device, swapchain, &mut count, images.as_mut_ptr()));

        let pool_info = VkCommandPoolCreateInfo {
            sType: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            pNext: ptr::null(),
            flags: VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex: family,
        };
        let mut pool = 0;
        check("vkCreateCommandPool", get!(device_function, vkCreateCommandPool, PFN_vkCreateCommandPool)(device, &pool_info, ptr::null(), &mut pool));
        let allocate_info = VkCommandBufferAllocateInfo {
            sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            pNext: ptr::null(),
            commandPool: pool,
            level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount: 1,
        };
        let mut commands = ptr::null_mut();
        check("vkAllocateCommandBuffers", get!(device_function, vkAllocateCommandBuffers, PFN_vkAllocateCommandBuffers)(device, &allocate_info, &mut commands));

        let semaphore_info = VkSemaphoreCreateInfo { sType: VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, pNext: ptr::null(), flags: 0 };
        let create_semaphore = get!(device_function, vkCreateSemaphore, PFN_vkCreateSemaphore);
        let (mut acquired, mut rendered) = (0, 0);
        check("vkCreateSemaphore", create_semaphore(device, &semaphore_info, ptr::null(), &mut acquired));
        check("vkCreateSemaphore", create_semaphore(device, &semaphore_info, ptr::null(), &mut rendered));
        let fence_info = VkFenceCreateInfo { sType: VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, pNext: ptr::null(), flags: 0 };
        let mut fence = 0;
        check("vkCreateFence", get!(device_function, vkCreateFence, PFN_vkCreateFence)(device, &fence_info, ptr::null(), &mut fence));

        let acquire = get!(device_function, vkAcquireNextImageKHR, PFN_vkAcquireNextImageKHR);
        let present = get!(device_function, vkQueuePresentKHR, PFN_vkQueuePresentKHR);
        let begin_commands = get!(device_function, vkBeginCommandBuffer, PFN_vkBeginCommandBuffer);
        let end_commands = get!(device_function, vkEndCommandBuffer, PFN_vkEndCommandBuffer);
        let barrier_fn = get!(device_function, vkCmdPipelineBarrier, PFN_vkCmdPipelineBarrier);
        let clear = get!(device_function, vkCmdClearColorImage, PFN_vkCmdClearColorImage);
        let submit = get!(device_function, vkQueueSubmit, PFN_vkQueueSubmit);
        let wait = get!(device_function, vkWaitForFences, PFN_vkWaitForFences);
        let reset = get!(device_function, vkResetFences, PFN_vkResetFences);
        let begin = VkCommandBufferBeginInfo {
            sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext: ptr::null(),
            flags: VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            pInheritanceInfo: ptr::null(),
        };
        let whole = VkImageSubresourceRange { aspectMask: VK_IMAGE_ASPECT_COLOR_BIT, baseMipLevel: 0, levelCount: 1, baseArrayLayer: 0, layerCount: 1 };

        let timer = Instant::now();
        for frame in 0..frames {
            window.pump();
            let mut index = 0;
            let acquired_result = acquire(device, swapchain, u64::MAX, acquired, 0, &mut index);
            if acquired_result < 0 {
                fail("vkAcquireNextImageKHR", acquired_result);
            }
            let image = images[index as usize];
            let barrier = |old, new, source, destination| VkImageMemoryBarrier {
                sType: VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                pNext: ptr::null(),
                srcAccessMask: source,
                dstAccessMask: destination,
                oldLayout: old,
                newLayout: new,
                srcQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex: VK_QUEUE_FAMILY_IGNORED,
                image,
                subresourceRange: whole,
            };
            check("vkBeginCommandBuffer", begin_commands(commands, &begin));
            let to_clear = barrier(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
            barrier_fn(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, ptr::null(), 0, ptr::null(), 1, &to_clear);
            let [r, g, b] = colour_of(frame);
            let colour = VkClearColorValue { float32: [r, g, b, 1.0] };
            clear(commands, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &colour, 1, &whole);
            let to_present = barrier(VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_WRITE_BIT, 0);
            barrier_fn(commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, ptr::null(), 0, ptr::null(), 1, &to_present);
            check("vkEndCommandBuffer", end_commands(commands));
            let stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            let submit_info = VkSubmitInfo {
                sType: VK_STRUCTURE_TYPE_SUBMIT_INFO,
                pNext: ptr::null(),
                waitSemaphoreCount: 1,
                pWaitSemaphores: &acquired,
                pWaitDstStageMask: &stage,
                commandBufferCount: 1,
                pCommandBuffers: &commands,
                signalSemaphoreCount: 1,
                pSignalSemaphores: &rendered,
            };
            check("vkQueueSubmit", submit(queue, 1, &submit_info, fence));
            let present_info = VkPresentInfoKHR {
                sType: VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                pNext: ptr::null(),
                waitSemaphoreCount: 1,
                pWaitSemaphores: &rendered,
                swapchainCount: 1,
                pSwapchains: &swapchain,
                pImageIndices: &index,
                pResults: ptr::null_mut(),
            };
            let presented = present(queue, &present_info);
            if presented < 0 {
                fail("vkQueuePresentKHR", presented);
            }
            // The command buffer is used again for the next frame
            check("vkWaitForFences", wait(device, 1, &fence, 1, u64::MAX));
            check("vkResetFences", reset(device, 1, &fence));


        }
        let seconds = timer.elapsed().as_secs_f64();
        get!(device_function, vkDeviceWaitIdle, PFN_vkDeviceWaitIdle)(device);

        // Everything presented has been drawn once its swapchain is destroyed
        get!(device_function, vkDestroySwapchainKHR, PFN_vkDestroySwapchainKHR)(device, swapchain, ptr::null());
        window.pump();
        // Frames are drawn on another connection on some systems, and a moment behind
        std::thread::sleep(std::time::Duration::from_millis(100));
        let (r, g, b) = window.pixel(width / 2, height / 2);
        // A driver may still be showing an earlier frame; any frame's colour says frames arrive
        let close = |got: u32, want: f32| (got as i32 - (want * 255.0) as i32).abs() <= 24;
        let shown = (0..3).any(|frame| {
            let [want_r, want_g, want_b] = colour_of(frame);
            close(r, want_r) && close(g, want_g) && close(b, want_b)
        });
        window.close();

        for semaphore in [acquired, rendered] {
            get!(device_function, vkDestroySemaphore, PFN_vkDestroySemaphore)(device, semaphore, ptr::null());
        }
        get!(device_function, vkDestroyFence, PFN_vkDestroyFence)(device, fence, ptr::null());
        get!(device_function, vkDestroyCommandPool, PFN_vkDestroyCommandPool)(device, pool, ptr::null());
        get!(instance_function, vkDestroySurfaceKHR, PFN_vkDestroySurfaceKHR)(instance, surface, ptr::null());

        println!(
            "present      {frames} frames of {}x{} in {seconds:.2} s: {:.1} frames/s",
            capabilities.currentExtent.width,
            capabilities.currentExtent.height,
            frames as f64 / seconds
        );
        if !shown {
            println!("FAIL the window shows ({r}, {g}, {b}), which is no frame's colour");
            std::process::exit(1);
        }
        println!("on screen    ({r}, {g}, {b}), a frame's colour");
    }
}
