//! The program's window, for present.rs: surfaces, the window's size, and drawing a frame in it.
//!
//! - Windows: a surface is a struct laid out as the loader's own VkIcdSurfaceWin32, since older
//!   loaders make theirs and pass it in, and newer ones ask this driver to make one. Frames are
//!   drawn with GDI.
//! - Linux: Xlib and XCB surfaces, whichever the program made, name an X window. Frames are drawn
//!   over a connection of this driver's own to the X server, so its thread never shares the
//!   program's, through libX11, loaded here since a computer without X has no use for it.
//! - Wayland and macOS (Metal) surfaces aren't offered yet.

#![allow(non_snake_case)]

use std::ffi::c_void;
use std::mem;

use nectar_vk_sys::*;

pub use platform::*;

/// A command of this file, by name
pub(crate) fn lookup(name: &str) -> PFN_vkVoidFunction {
    if name == "vkDestroySurfaceKHR" {
        return Some(unsafe { mem::transmute::<*const (), unsafe extern "system" fn()>(platform::vkDestroySurfaceKHR as *const ()) });
    }
    platform::lookup_surface(name)
}

/// Runs of pixels are BGRA, top row first, `width * 4` bytes a row
pub trait Show {
    /// The window's size now
    fn size(&self) -> (u32, u32);
    /// Draws a frame, scaled to the window where the system can
    ///
    /// # Safety
    /// `pixels` must hold `width * height * 4` bytes.
    unsafe fn show(&self, pixels: *const u8, width: u32, height: u32);
}

type VoidPointer = *mut c_void;

#[cfg(windows)]
mod platform {
    use super::*;
    use windows_sys::Win32::Foundation::{HWND, RECT};
    use windows_sys::Win32::Graphics::Gdi::{
        BITMAPINFO, BITMAPINFOHEADER, DIB_RGB_COLORS, GetDC, ReleaseDC, SRCCOPY, SetDIBitsToDevice, StretchDIBits,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::GetClientRect;

    const VK_ICD_WSI_PLATFORM_WIN32: i32 = 2;

    pub const SURFACE_EXTENSIONS: &[(&str, u32)] = &[("VK_KHR_surface", 25), ("VK_KHR_win32_surface", 6)];

    #[repr(C)]
    struct IcdSurfaceWin32 {
        platform: i32,
        instance: VoidPointer,
        window: HWND,
    }

    #[repr(C)]
    pub struct VkWin32SurfaceCreateInfoKHR {
        sType: VkStructureType,
        pNext: *const c_void,
        flags: u32,
        hinstance: VoidPointer,
        hwnd: HWND,
    }

    pub unsafe extern "system" fn vkCreateWin32SurfaceKHR(
        _instance: VkInstance,
        pCreateInfo: *const VkWin32SurfaceCreateInfoKHR,
        _pAllocator: *const VkAllocationCallbacks,
        pSurface: *mut VkSurfaceKHR,
    ) -> VkResult {
        unsafe {
            let info = &*pCreateInfo;
            let surface = Box::new(IcdSurfaceWin32 { platform: VK_ICD_WSI_PLATFORM_WIN32, instance: info.hinstance, window: info.hwnd });
            *pSurface = Box::into_raw(surface) as usize as VkSurfaceKHR;
        }
        VK_SUCCESS
    }

    pub unsafe extern "system" fn vkDestroySurfaceKHR(_instance: VkInstance, surface: VkSurfaceKHR, _pAllocator: *const VkAllocationCallbacks) {
        if surface != 0 {
            drop(unsafe { Box::from_raw(surface as usize as *mut IcdSurfaceWin32) });
        }
    }

    pub(super) fn lookup_surface(name: &str) -> PFN_vkVoidFunction {
        (name == "vkCreateWin32SurfaceKHR")
            .then(|| unsafe { mem::transmute::<*const (), unsafe extern "system" fn()>(vkCreateWin32SurfaceKHR as *const ()) })
    }

    pub struct Window(HWND);

    // A window handle is only a number here; GDI takes it from any thread
    unsafe impl Send for Window {}
    unsafe impl Sync for Window {}

    /// The window behind a surface
    ///
    /// # Safety
    /// `surface` must be a surface of this driver or the loader.
    pub unsafe fn window_of(surface: VkSurfaceKHR) -> Option<Window> {
        let surface = unsafe { (surface as usize as *const IcdSurfaceWin32).as_ref() }?;
        (surface.platform == VK_ICD_WSI_PLATFORM_WIN32).then_some(Window(surface.window))
    }

    impl Show for Window {
        fn size(&self) -> (u32, u32) {
            let mut rect = RECT { left: 0, top: 0, right: 0, bottom: 0 };
            unsafe { GetClientRect(self.0, &mut rect) };
            ((rect.right - rect.left).max(0) as u32, (rect.bottom - rect.top).max(0) as u32)
        }

        unsafe fn show(&self, pixels: *const u8, width: u32, height: u32) {
            let info = BITMAPINFO {
                bmiHeader: BITMAPINFOHEADER {
                    biSize: size_of::<BITMAPINFOHEADER>() as u32,
                    biWidth: width as i32,
                    // Top-down, as the image is
                    biHeight: -(height as i32),
                    biPlanes: 1,
                    biBitCount: 32,
                    biCompression: 0,
                    biSizeImage: 0,
                    biXPelsPerMeter: 0,
                    biYPelsPerMeter: 0,
                    biClrUsed: 0,
                    biClrImportant: 0,
                },
                bmiColors: unsafe { mem::zeroed() },
            };
            let (client_width, client_height) = self.size();
            unsafe {
                let dc = GetDC(self.0);
                if dc.is_null() {
                    return;
                }
                if (client_width, client_height) == (width, height) {
                    SetDIBitsToDevice(dc, 0, 0, width, height, 0, 0, 0, height, pixels.cast(), &info, DIB_RGB_COLORS);
                } else {
                    StretchDIBits(
                        dc,
                        0,
                        0,
                        client_width as i32,
                        client_height as i32,
                        0,
                        0,
                        width as i32,
                        height as i32,
                        pixels.cast(),
                        &info,
                        DIB_RGB_COLORS,
                        SRCCOPY,
                    );
                }
                ReleaseDC(self.0, dc);
            }
        }
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use std::sync::{Mutex, OnceLock};

    use libloading::Library;

    use super::*;


    pub const SURFACE_EXTENSIONS: &[(&str, u32)] = &[("VK_KHR_surface", 25), ("VK_KHR_xlib_surface", 6), ("VK_KHR_xcb_surface", 6)];

    /// A surface: the X window it draws in
    struct Surface {
        window: u64,
    }

    #[repr(C)]
    pub struct VkXlibSurfaceCreateInfoKHR {
        sType: VkStructureType,
        pNext: *const c_void,
        flags: u32,
        dpy: VoidPointer,
        window: u64,
    }

    #[repr(C)]
    pub struct VkXcbSurfaceCreateInfoKHR {
        sType: VkStructureType,
        pNext: *const c_void,
        flags: u32,
        connection: VoidPointer,
        window: u32,
    }

    unsafe fn make(window: u64, pSurface: *mut VkSurfaceKHR) -> VkResult {
        unsafe { *pSurface = Box::into_raw(Box::new(Surface { window })) as usize as VkSurfaceKHR };
        VK_SUCCESS
    }

    pub unsafe extern "system" fn vkCreateXlibSurfaceKHR(
        _instance: VkInstance,
        pCreateInfo: *const VkXlibSurfaceCreateInfoKHR,
        _pAllocator: *const VkAllocationCallbacks,
        pSurface: *mut VkSurfaceKHR,
    ) -> VkResult {
        unsafe { make((*pCreateInfo).window, pSurface) }
    }

    pub unsafe extern "system" fn vkCreateXcbSurfaceKHR(
        _instance: VkInstance,
        pCreateInfo: *const VkXcbSurfaceCreateInfoKHR,
        _pAllocator: *const VkAllocationCallbacks,
        pSurface: *mut VkSurfaceKHR,
    ) -> VkResult {
        unsafe { make((*pCreateInfo).window as u64, pSurface) }
    }

    pub unsafe extern "system" fn vkDestroySurfaceKHR(_instance: VkInstance, surface: VkSurfaceKHR, _pAllocator: *const VkAllocationCallbacks) {
        if surface != 0 {
            drop(unsafe { Box::from_raw(surface as usize as *mut Surface) });
        }
    }

    pub(super) fn lookup_surface(name: &str) -> PFN_vkVoidFunction {
        let function = match name {
            "vkCreateXlibSurfaceKHR" => vkCreateXlibSurfaceKHR as *const (),
            "vkCreateXcbSurfaceKHR" => vkCreateXcbSurfaceKHR as *const (),
            _ => return None,
        };
        Some(unsafe { mem::transmute::<*const (), unsafe extern "system" fn()>(function) })
    }

    type OpenDisplay = unsafe extern "C" fn(*const std::ffi::c_char) -> VoidPointer;
    type DefaultScreen = unsafe extern "C" fn(VoidPointer) -> i32;
    type DefaultVisual = unsafe extern "C" fn(VoidPointer, i32) -> VoidPointer;
    type DefaultDepth = unsafe extern "C" fn(VoidPointer, i32) -> i32;
    type CreateGc = unsafe extern "C" fn(VoidPointer, u64, u64, VoidPointer) -> VoidPointer;
    type FreeGc = unsafe extern "C" fn(VoidPointer, VoidPointer) -> i32;
    type CreateImage = unsafe extern "C" fn(VoidPointer, VoidPointer, u32, i32, i32, *mut i8, u32, u32, i32, i32) -> VoidPointer;
    type PutImage = unsafe extern "C" fn(VoidPointer, u64, VoidPointer, VoidPointer, i32, i32, i32, i32, u32, u32) -> i32;
    type Free = unsafe extern "C" fn(VoidPointer) -> i32;
    type Flush = unsafe extern "C" fn(VoidPointer) -> i32;
    type GetGeometry = unsafe extern "C" fn(VoidPointer, u64, *mut u64, *mut i32, *mut i32, *mut u32, *mut u32, *mut u32, *mut u32) -> i32;

    /// libX11, and this driver's own connection to the X server
    struct X {
        _library: Library,
        display: usize,

        depth: i32,
        visual: usize,
        create_gc: CreateGc,
        free_gc: FreeGc,
        create_image: CreateImage,
        put_image: PutImage,
        free: Free,
        flush: Flush,
        get_geometry: GetGeometry,
        /// Xlib calls on one connection go one at a time
        lock: Mutex<()>,
    }

    fn x() -> Option<&'static X> {
        static X_SERVER: OnceLock<Option<X>> = OnceLock::new();
        X_SERVER
            .get_or_init(|| unsafe {
                let library = Library::new("libX11.so.6").ok()?;
                let open: OpenDisplay = *library.get(b"XOpenDisplay\0").ok()?;
                let default_screen: DefaultScreen = *library.get(b"XDefaultScreen\0").ok()?;
                let default_visual: DefaultVisual = *library.get(b"XDefaultVisual\0").ok()?;
                let default_depth: DefaultDepth = *library.get(b"XDefaultDepth\0").ok()?;
                let display = open(std::ptr::null());
                if display.is_null() {
                    return None;
                }
                let screen = default_screen(display);
                Some(X {
                    display: display as usize,

                    depth: default_depth(display, screen),
                    visual: default_visual(display, screen) as usize,
                    create_gc: *library.get(b"XCreateGC\0").ok()?,
                    free_gc: *library.get(b"XFreeGC\0").ok()?,
                    create_image: *library.get(b"XCreateImage\0").ok()?,
                    put_image: *library.get(b"XPutImage\0").ok()?,
                    free: *library.get(b"XFree\0").ok()?,
                    flush: *library.get(b"XFlush\0").ok()?,
                    get_geometry: *library.get(b"XGetGeometry\0").ok()?,
                    lock: Mutex::new(()),
                    _library: library,
                })
            })
            .as_ref()
    }

    pub struct Window(u64);

    /// The window behind a surface
    ///
    /// # Safety
    /// `surface` must be a surface of this driver.
    pub unsafe fn window_of(surface: VkSurfaceKHR) -> Option<Window> {
        x()?;
        let surface = unsafe { (surface as usize as *const Surface).as_ref() }?;
        Some(Window(surface.window))
    }

    impl Show for Window {
        fn size(&self) -> (u32, u32) {
            let Some(x) = x() else { return (0, 0) };
            let _one_at_a_time = x.lock.lock().unwrap_or_else(|p| p.into_inner());
            let (mut root, mut left, mut top, mut width, mut height, mut border, mut depth) = (0u64, 0, 0, 0u32, 0u32, 0u32, 0u32);
            let found = unsafe {
                (x.get_geometry)(x.display as VoidPointer, self.0, &mut root, &mut left, &mut top, &mut width, &mut height, &mut border, &mut depth)
            };
            if found == 0 { (0, 0) } else { (width, height) }
        }

        unsafe fn show(&self, pixels: *const u8, width: u32, height: u32) {
            const Z_PIXMAP: i32 = 2;
            let Some(x) = x() else { return };
            let _one_at_a_time = x.lock.lock().unwrap_or_else(|p| p.into_inner());
            let display = x.display as VoidPointer;
            unsafe {
                let image = (x.create_image)(
                    display,
                    x.visual as VoidPointer,
                    x.depth as u32,
                    Z_PIXMAP,
                    0,
                    pixels as *mut i8,
                    width,
                    height,
                    32,
                    (width * 4) as i32,
                );
                if image.is_null() {
                    return;
                }
                let gc = (x.create_gc)(display, self.0, 0, std::ptr::null_mut());
                (x.put_image)(display, self.0, gc, image, 0, 0, 0, 0, width, height);
                (x.free_gc)(display, gc);
                (x.flush)(display);
                // Only the struct: the pixels aren't its to free
                (x.free)(image);
            }

        }
    }
}
