//! nectar-server: runs CUDA driver API calls from borrowers on this machine's GPU.
//!
//! Each borrower connection gets its own thread, so the context it makes current stays
//! current for all of its calls. What a borrower leaves behind when it disconnects
//! (allocations, retained primary contexts) is released.
//!
//! Most calls are carried out by generated code (generated.rs). This file holds the
//! connection and the functions written by hand on both sides. cuGetProcAddress never
//! reaches the lender: the client answers it from cudaTypedefs.h.
//!
//! Not yet: isolation between borrowers, which share this process and its primary
//! contexts, and authentication. Listen on 127.0.0.1 until there is a shared key.
//!
//! Uploads of 1 MiB or more are kept by content hash (cache.rs), by default in
//! %LOCALAPPDATA%\Nectar\cuda-cache up to 20 GB, so loading the same model again sends only
//! hashes. Borrowers on loopback do not use it.
//!
//! Usage: nectar-server [--listen <address:port>] [--lend] [--cache <folder>]
//! [--cache-limit-gb <size, 0 for none>] [--cuda-driver <library>]

mod cache;
mod generated;
mod lend;
mod vulkan;

use std::sync::atomic::Ordering;

use cache::Cache;
use lend::Lending;
use nectar_lending::discovery::{LenderInfo, Offer, State};

use std::collections::{HashMap, HashSet};
use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::io::{self, BufReader, BufWriter, Write};
use std::marker::PhantomData;
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::{env, mem, ptr, thread};

use libloading::Library;
use nectar_cuda_protocol::{self as protocol, Reader, Writer, id};
use nectar_cuda_sys::*;

/// The real driver: nvcuda.dll from System32, libcuda.so.1, or the library --cuda-driver
/// names (such as ZLUDA's, which runs CUDA on AMD GPUs)
struct Driver {
    _library: Library,
    /// Function id to address, for every export the generated code knows
    functions: HashMap<u32, usize>,
}

/// The CUDA driver library a lender uses unless told otherwise
fn default_driver() -> PathBuf {
    if cfg!(windows) {
        // By full path, so a Nectar nvcuda.dll in the current folder is never picked up
        env::var_os("SystemRoot")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(r"C:\Windows"))
            .join(r"System32\nvcuda.dll")
    } else {
        PathBuf::from("libcuda.so.1")
    }
}

impl Driver {
    fn load(path: &std::path::Path) -> Result<Self, String> {
        let library = unsafe { Library::new(path) }.map_err(|error| error.to_string())?;
        if unsafe { library.get::<*const c_void>(b"nectarIdentify\0") }.is_ok() {
            return Err(format!(
                "{} is Nectar's own driver for borrowing, not a GPU driver",
                path.display()
            ));
        }

        let mut functions = HashMap::new();
        for &name in protocol::NAMES {
            let symbol = format!("{name}\0");
            // A symbol typed as a pointer dereferences to its own address
            if let Ok(address) = unsafe { library.get::<*const c_void>(symbol.as_bytes()) } {
                let address = *address as usize;
                if address != 0 {
                    functions.insert(protocol::function_id(name), address);
                }
            }
        }

        if let Some(&address) = functions.get(&id::cuPointerGetAttribute) {
            let _ = POINTER_GET_ATTRIBUTE.set(address);
        }

        Ok(Self {
            _library: library,
            functions,
        })
    }

    fn address(&self, function: u32) -> Option<*const c_void> {
        self.functions
            .get(&function)
            .map(|&address| address as *const c_void)
    }

    /// A driver function by id, as the function pointer type F
    ///
    /// # Safety
    /// F must be the function's real signature.
    unsafe fn get<F: Copy>(&self, function: u32) -> Option<F> {
        assert_eq!(mem::size_of::<F>(), mem::size_of::<usize>());
        self.functions
            .get(&function)
            .map(|address| unsafe { mem::transmute_copy(address) })
    }
}

/// What one borrower holds, released when it disconnects
#[derive(Default)]
pub(crate) struct Session {
    allocations: HashSet<CUdeviceptr>,
    /// Device to (primary context, times retained)
    contexts: HashMap<CUdevice, (usize, u32)>,
    /// Handles from cuLibraryGetKernel, which launches take in place of functions
    pub(crate) kernels: HashSet<usize>,
    /// Virtual address ranges reserved with cuMemAddressReserve, and their sizes
    reserved: HashMap<CUdeviceptr, usize>,
    /// Physical allocations from cuMemCreate. Unlike cuMemAlloc memory they belong to no
    /// context, so nothing frees them when contexts go
    memory: HashSet<CUmemGenericAllocationHandle>,
    /// Uploads kept by content hash, shared by every borrower
    cache: Option<Arc<Cache>>,
}

/// Size of the upload cache unless --cache-limit-gb says otherwise: two or three models
const DEFAULT_CACHE_GB: u64 = 20;

impl Session {
    pub(crate) fn reserved(&mut self, pointer: CUdeviceptr, size: usize) {
        self.reserved.insert(pointer, size);
    }

    pub(crate) fn unreserved(&mut self, pointer: CUdeviceptr) {
        self.reserved.remove(&pointer);
    }

    pub(crate) fn created(&mut self, handle: CUmemGenericAllocationHandle) {
        self.memory.insert(handle);
    }

    pub(crate) fn released_memory(&mut self, handle: CUmemGenericAllocationHandle) {
        self.memory.remove(&handle);
    }

    pub(crate) fn allocated(&mut self, pointer: CUdeviceptr) {
        self.allocations.insert(pointer);
    }

    pub(crate) fn freed(&mut self, pointer: CUdeviceptr) {
        self.allocations.remove(&pointer);
    }

    pub(crate) fn retained(&mut self, device: CUdevice, context: CUcontext) {
        self.contexts
            .entry(device)
            .or_insert((context as usize, 0))
            .1 += 1;
    }

    pub(crate) fn released(&mut self, device: CUdevice) {
        if let Some(entry) = self.contexts.get_mut(&device) {
            entry.1 -= 1;
            if entry.1 == 0 {
                self.contexts.remove(&device);
            }
        }
    }

    fn release(&mut self, driver: &Driver) {
        type SetCurrent = unsafe extern "system" fn(CUcontext) -> CUresult;
        type Free = unsafe extern "system" fn(CUdeviceptr) -> CUresult;
        type Release = unsafe extern "system" fn(CUdevice) -> CUresult;

        unsafe {
            let (Some(set_current), Some(free), Some(release)) = (
                driver.get::<SetCurrent>(id::cuCtxSetCurrent),
                driver.get::<Free>(id::cuMemFree_v2),
                driver.get::<Release>(id::cuDevicePrimaryCtxRelease_v2),
            ) else {
                return;
            };

            // Allocations belong to a primary context, which must be current to free them
            if let Some(&(context, _)) = self.contexts.values().next() {
                set_current(context as CUcontext);
            }
            for pointer in self.allocations.drain() {
                free(pointer);
            }

            type Unmap = unsafe extern "system" fn(CUdeviceptr, usize) -> CUresult;
            type AddressFree = unsafe extern "system" fn(CUdeviceptr, usize) -> CUresult;
            type MemRelease = unsafe extern "system" fn(CUmemGenericAllocationHandle) -> CUresult;
            if let (Some(unmap), Some(address_free), Some(memory_release)) = (
                driver.get::<Unmap>(id::cuMemUnmap),
                driver.get::<AddressFree>(id::cuMemAddressFree),
                driver.get::<MemRelease>(id::cuMemRelease),
            ) {
                // Unmapping a range with nothing mapped only fails, which is fine here
                for (pointer, size) in self.reserved.drain() {
                    unmap(pointer, size);
                    address_free(pointer, size);
                }
                for handle in self.memory.drain() {
                    memory_release(handle);
                }
            }

            for (device, (_, count)) in self.contexts.drain() {
                for _ in 0..count {
                    release(device);
                }
            }
        }
    }
}

/// Space for a value the driver reads or writes through a pointer. It is padded well past
/// the value, so a function that writes more than the header declares cannot overrun it.
pub(crate) struct Slot<T> {
    buffer: Option<Vec<u128>>,
    _value: PhantomData<T>,
}

impl<T: Copy> Slot<T> {
    const PADDED: usize = 512;

    /// Reads an optional value: absent means the caller passed NULL
    pub(crate) fn read(request: &mut Reader) -> io::Result<Self> {
        let buffer = if request.u8()? == 0 {
            None
        } else {
            let bytes = request.take(mem::size_of::<T>())?;
            let mut buffer = vec![0u128; mem::size_of::<T>().max(Self::PADDED).div_ceil(16)];
            unsafe {
                ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    buffer.as_mut_ptr().cast::<u8>(),
                    bytes.len(),
                )
            };
            Some(buffer)
        };
        Ok(Self {
            buffer,
            _value: PhantomData,
        })
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
        self.buffer
            .as_mut()
            .map_or(ptr::null_mut(), |buffer| buffer.as_mut_ptr().cast())
    }

    pub(crate) fn as_ptr(&self) -> *const T {
        self.buffer
            .as_ref()
            .map_or(ptr::null(), |buffer| buffer.as_ptr().cast())
    }

    pub(crate) fn get(&self) -> Option<T> {
        self.buffer
            .as_ref()
            .map(|buffer| unsafe { buffer.as_ptr().cast::<T>().read() })
    }

    /// Writes the value back for the caller, or that it was absent
    pub(crate) fn write(&self, reply: &mut Writer) {
        match self.get() {
            Some(value) => reply.u8(1).value(&value),
            None => reply.u8(0),
        };
    }
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.to_string())
}

/// cuPointerGetAttribute, for `gpu_memory`, which the generated code calls without a driver
static POINTER_GET_ATTRIBUTE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

/// Whether every address is GPU memory on this machine. Copies that take any address would
/// otherwise treat a borrower's host address as host memory in this process.
pub(crate) fn gpu_memory(pointers: &[CUdeviceptr]) -> bool {
    type PointerGetAttribute =
        unsafe extern "system" fn(*mut c_void, CUpointer_attribute, CUdeviceptr) -> CUresult;

    let Some(&address) = POINTER_GET_ATTRIBUTE.get() else {
        return false;
    };
    let get = unsafe { mem::transmute::<usize, PointerGetAttribute>(address) };
    pointers.iter().all(|&pointer| {
        let mut memory_type: CUmemorytype = 0;
        let result = unsafe {
            get(
                (&mut memory_type as *mut CUmemorytype).cast(),
                CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                pointer,
            )
        };
        result == CUDA_SUCCESS && memory_type == CU_MEMORYTYPE_DEVICE
    })
}

/// Waits for everything queued on a stream
fn synchronize(driver: &Driver, stream: CUstream) -> CUresult {
    type StreamSynchronize = unsafe extern "system" fn(CUstream) -> CUresult;
    match unsafe { driver.get::<StreamSynchronize>(id::cuStreamSynchronize) } {
        Some(synchronize) => unsafe { synchronize(stream) },
        None => CUDA_ERROR_NOT_SUPPORTED,
    }
}

/// Bytes copied to 16-byte aligned memory, as the driver may expect of code and arguments
fn aligned(bytes: &[u8]) -> Vec<u128> {
    let mut buffer = vec![0u128; bytes.len().div_ceil(16).max(1)];
    unsafe {
        ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            buffer.as_mut_ptr().cast::<u8>(),
            bytes.len(),
        )
    };
    buffer
}

/// The size of each parameter of a kernel, for the client to copy arguments by value.
/// Handles from cuLibraryGetKernel are kernels; the rest are functions.
fn param_layout(
    driver: &Driver,
    session: &Session,
    request: &mut Reader,
    reply: &mut Writer,
) -> io::Result<CUresult> {
    type ParamInfo =
        unsafe extern "system" fn(*mut c_void, usize, *mut usize, *mut usize) -> CUresult;

    let handle = request.u64()? as usize;
    let query = if session.kernels.contains(&handle) {
        id::cuKernelGetParamInfo
    } else {
        id::cuFuncGetParamInfo
    };
    let Some(info) = (unsafe { driver.get::<ParamInfo>(query) }) else {
        return Ok(CUDA_ERROR_NOT_SUPPORTED);
    };

    let mut sizes = Vec::new();
    loop {
        let (mut offset, mut size) = (0, 0);
        match unsafe { info(handle as *mut c_void, sizes.len(), &mut offset, &mut size) } {
            CUDA_SUCCESS => sizes.push(size as u32),
            // Asked past the last parameter
            CUDA_ERROR_INVALID_VALUE => break,
            error => return Ok(error),
        }
    }

    reply.u32(sizes.len() as u32);
    for size in sizes {
        reply.u32(size);
    }
    Ok(CUDA_SUCCESS)
}

/// cuLaunchKernel with the arguments the client copied: one per parameter, or one buffer
/// for `extra`. With `with_config`, cuLaunchKernelEx, which takes the settings and the
/// launch attributes in a struct.
unsafe fn launch(
    address: *const c_void,
    with_config: bool,
    request: &mut Reader,
) -> io::Result<CUresult> {
    type LaunchKernel = unsafe extern "system" fn(
        *mut c_void,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        CUstream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CUresult;

    let function = request.u64()? as usize as *mut c_void;
    let mut dimensions = [0u32; 6];
    for dimension in &mut dimensions {
        *dimension = request.u32()?;
    }
    let shared_memory = request.u32()?;
    let stream = request.u64()? as usize as CUstream;
    let mut attributes = (0..request.u32()?)
        .map(|_| unsafe { request.value::<CUlaunchAttribute>() })
        .collect::<io::Result<Vec<_>>>()?;

    // Each argument's memory, which must stay put until the launch returns
    let mut storage: Vec<Vec<u128>> = Vec::new();
    let mut arguments: Vec<*mut c_void> = Vec::new();
    let mut buffer_size: usize;
    let mut extra: Vec<*mut c_void> = Vec::new();

    let mode = request.u8()?;
    match mode {
        1 => {
            for _ in 0..request.u32()? {
                let mut argument = aligned(request.bytes()?);
                arguments.push(argument.as_mut_ptr().cast());
                storage.push(argument);
            }
        }
        2 => {
            let bytes = request.bytes()?;
            buffer_size = bytes.len();
            let mut buffer = aligned(bytes);
            extra = vec![
                CU_LAUNCH_PARAM_BUFFER_POINTER_AS_INT as usize as *mut c_void,
                buffer.as_mut_ptr().cast(),
                CU_LAUNCH_PARAM_BUFFER_SIZE_AS_INT as usize as *mut c_void,
                (&mut buffer_size as *mut usize).cast(),
                CU_LAUNCH_PARAM_END_AS_INT as usize as *mut c_void,
            ];
            storage.push(buffer);
        }
        _ => {}
    }

    let arguments = if mode == 1 {
        arguments.as_mut_ptr()
    } else {
        ptr::null_mut()
    };
    let extra = if mode == 2 {
        extra.as_mut_ptr()
    } else {
        ptr::null_mut()
    };
    let [grid_x, grid_y, grid_z, block_x, block_y, block_z] = dimensions;

    let result = if with_config {
        type LaunchKernelEx = unsafe extern "system" fn(
            *const CUlaunchConfig,
            *mut c_void,
            *mut *mut c_void,
            *mut *mut c_void,
        ) -> CUresult;
        let config = CUlaunchConfig {
            gridDimX: grid_x,
            gridDimY: grid_y,
            gridDimZ: grid_z,
            blockDimX: block_x,
            blockDimY: block_y,
            blockDimZ: block_z,
            sharedMemBytes: shared_memory,
            hStream: stream,
            attrs: attributes.as_mut_ptr(),
            numAttrs: attributes.len() as c_uint,
        };
        unsafe {
            mem::transmute::<*const c_void, LaunchKernelEx>(address)(
                &config, function, arguments, extra,
            )
        }
    } else {
        unsafe {
            mem::transmute::<*const c_void, LaunchKernel>(address)(
                function,
                grid_x,
                grid_y,
                grid_z,
                block_x,
                block_y,
                block_z,
                shared_memory,
                stream,
                arguments,
                extra,
            )
        }
    };
    drop(storage);
    Ok(result)
}

/// Copies bytes from the borrower to GPU memory as `function` would: the synchronous copies
/// at once, the asynchronous ones in the stream's order and then waited for, so the
/// borrower's memory is read at the right moment
fn upload(
    driver: &Driver,
    function: u32,
    destination: CUdeviceptr,
    stream: CUstream,
    data: &[u8],
) -> CUresult {
    type MemcpyHtoD = unsafe extern "system" fn(CUdeviceptr, *const c_void, usize) -> CUresult;
    type MemcpyHtoDAsync =
        unsafe extern "system" fn(CUdeviceptr, *const c_void, usize, CUstream) -> CUresult;

    unsafe {
        match function {
            id::cuMemcpyHtoD_v2 | id::cuMemcpyHtoD_v2_ptds => {
                match driver.get::<MemcpyHtoD>(function) {
                    Some(copy) => copy(destination, data.as_ptr().cast(), data.len()),
                    None => CUDA_ERROR_NOT_SUPPORTED,
                }
            }
            id::cuMemcpyHtoDAsync_v2 | id::cuMemcpyHtoDAsync_v2_ptsz => {
                let result = match driver.get::<MemcpyHtoDAsync>(function) {
                    Some(copy) => copy(destination, data.as_ptr().cast(), data.len(), stream),
                    None => CUDA_ERROR_NOT_SUPPORTED,
                };
                if result == CUDA_SUCCESS {
                    synchronize(driver, stream)
                } else {
                    result
                }
            }
            _ => CUDA_ERROR_INVALID_VALUE,
        }
    }
}

/// protocol::UPLOAD: an upload kept by its content hash, sent with its bytes or without
fn cached_upload(
    driver: &Driver,
    session: &Session,
    request: &mut Reader,
    reply: &mut Writer,
) -> io::Result<CUresult> {
    let function = request.u32()?;
    let destination = request.u64()?;
    let stream = request.u64()? as usize as CUstream;
    let hash: cache::Hash = request.take(16)?.try_into().unwrap();
    let Some(cache) = &session.cache else {
        return Ok(CUDA_ERROR_NOT_SUPPORTED);
    };

    // The result says whether the bytes are kept, so the borrower knows it can send the
    // hash alone next time
    if request.u8()? != 0 {
        let data = request.bytes()?;
        let kept = match cache.put(&hash, data) {
            Ok(()) => true,
            Err(error) => {
                eprintln!("cannot keep an upload: {error}");
                false
            }
        };
        reply.u8(kept as u8);
        return Ok(upload(driver, function, destination, stream, data));
    }

    let size = request.u64()?;
    match cache.get(&hash, size) {
        Ok(data) => {
            reply.u8(1);
            Ok(upload(driver, function, destination, stream, &data))
        }
        Err(error) => {
            eprintln!("a kept upload is missing: {error}");
            Ok(CUDA_ERROR_NOT_FOUND)
        }
    }
}

/// protocol::ATTRIBUTES: every attribute the driver answers for a device, kernel or function
fn attributes(driver: &Driver, request: &mut Reader, reply: &mut Writer) -> io::Result<CUresult> {
    type Device = unsafe extern "system" fn(*mut c_int, CUdevice_attribute, CUdevice) -> CUresult;
    type Kernel = unsafe extern "system" fn(
        *mut c_int,
        CUfunction_attribute,
        CUkernel,
        CUdevice,
    ) -> CUresult;
    type Function =
        unsafe extern "system" fn(*mut c_int, CUfunction_attribute, CUfunction) -> CUresult;

    /// Asks for one attribute
    type GetOne = Box<dyn Fn(i32, &mut c_int) -> CUresult>;

    let function = request.u32()?;
    let get: GetOne = unsafe {
        match function {
            id::cuDeviceGetAttribute => {
                let device = request.i32()?;
                let Some(get) = driver.get::<Device>(function) else {
                    return Ok(CUDA_ERROR_NOT_SUPPORTED);
                };
                Box::new(move |attribute, value| get(value, attribute, device))
            }
            id::cuKernelGetAttribute => {
                let kernel = request.u64()? as usize as CUkernel;
                let device = request.i32()?;
                let Some(get) = driver.get::<Kernel>(function) else {
                    return Ok(CUDA_ERROR_NOT_SUPPORTED);
                };
                Box::new(move |attribute, value| get(value, attribute, kernel, device))
            }
            id::cuFuncGetAttribute => {
                let handle = request.u64()? as usize as CUfunction;
                let Some(get) = driver.get::<Function>(function) else {
                    return Ok(CUDA_ERROR_NOT_SUPPORTED);
                };
                Box::new(move |attribute, value| get(value, attribute, handle))
            }
            _ => return Ok(CUDA_ERROR_INVALID_VALUE),
        }
    };
    let count = if function == id::cuDeviceGetAttribute {
        CU_DEVICE_ATTRIBUTE_MAX
    } else {
        CU_FUNC_ATTRIBUTE_MAX
    };

    let mut answered = Vec::new();
    for attribute in 0..count {
        let mut value = 0;
        if get(attribute, &mut value) == CUDA_SUCCESS {
            answered.push((attribute, value));
        }
    }
    reply.u32(answered.len() as u32);
    for (attribute, value) in answered {
        reply.i32(attribute).i32(value);
    }
    Ok(CUDA_SUCCESS)
}

/// Carries out one call. Results are written only on success.
fn call(
    driver: &Driver,
    session: &mut Session,
    function: u32,
    request: &mut Reader,
    reply: &mut Writer,
) -> io::Result<CUresult> {
    type GetText = unsafe extern "system" fn(CUresult, *mut *const c_char) -> CUresult;
    type GetName = unsafe extern "system" fn(*mut c_char, c_int, CUdevice) -> CUresult;
    type Init = unsafe extern "system" fn(c_uint) -> CUresult;
    type MemcpyDtoH = unsafe extern "system" fn(*mut c_void, CUdeviceptr, usize) -> CUresult;
    type MemcpyDtoHAsync =
        unsafe extern "system" fn(*mut c_void, CUdeviceptr, usize, CUstream) -> CUresult;

    type GetLuid = unsafe extern "system" fn(*mut c_char, *mut c_uint, CUdevice) -> CUresult;
    type Granularity = unsafe extern "system" fn(
        *mut usize,
        *const CUmemAllocationProp,
        CUmemAllocationGranularity_flags,
    ) -> CUresult;
    type MemCreate = unsafe extern "system" fn(
        *mut CUmemGenericAllocationHandle,
        usize,
        *const CUmemAllocationProp,
        u64,
    ) -> CUresult;
    type SetAccess =
        unsafe extern "system" fn(CUdeviceptr, usize, *const CUmemAccessDesc, usize) -> CUresult;
    type ModuleLoadData = unsafe extern "system" fn(*mut CUmodule, *const c_void) -> CUresult;
    type LibraryLoadData = unsafe extern "system" fn(
        *mut CUlibrary,
        *const c_void,
        *mut CUjit_option,
        *mut *mut c_void,
        c_uint,
        *mut CUlibraryOption,
        *mut *mut c_void,
        c_uint,
    ) -> CUresult;

    match function {
        protocol::PARAM_LAYOUT => return param_layout(driver, session, request, reply),
        protocol::UPLOAD => return cached_upload(driver, session, request, reply),
        protocol::ATTRIBUTES => return attributes(driver, request, reply),
        protocol::CACHE_INDEX => {
            let Some(cache) = &session.cache else {
                return Ok(CUDA_ERROR_NOT_SUPPORTED);
            };
            let index = cache.index();
            reply.u64(index.len() as u64);
            for hash in &index {
                reply.value(hash);
            }
            return Ok(CUDA_SUCCESS);
        }
        _ => {}
    }

    let Some(address) = driver.address(function) else {
        eprintln!(
            "{} is missing from this driver",
            protocol::function_name(function).unwrap_or("an unknown function")
        );
        return Ok(CUDA_ERROR_NOT_SUPPORTED);
    };

    let result = unsafe {
        match function {
            id::cuInit => mem::transmute::<*const c_void, Init>(address)(request.u32()?),

            id::cuGetErrorName | id::cuGetErrorString => {
                let error = request.i32()?;
                let mut text: *const c_char = ptr::null();
                let result = mem::transmute::<*const c_void, GetText>(address)(error, &mut text);
                if result == CUDA_SUCCESS && !text.is_null() {
                    reply.bytes(CStr::from_ptr(text).to_bytes());
                }
                result
            }

            id::cuDeviceGetName => {
                let length = request.i32()?;
                let device = request.i32()?;
                if !(1..=4096).contains(&length) {
                    return Ok(CUDA_ERROR_INVALID_VALUE);
                }

                let mut name = vec![0u8; length as usize];
                let result = mem::transmute::<*const c_void, GetName>(address)(
                    name.as_mut_ptr().cast(),
                    length,
                    device,
                );
                let end = name
                    .iter()
                    .position(|&byte| byte == 0)
                    .unwrap_or(name.len());
                reply.bytes(&name[..end]);
                result
            }

            id::cuMemcpyHtoD_v2 | id::cuMemcpyHtoD_v2_ptds => {
                let destination = request.u64()?;
                let data = request.bytes()?;
                upload(driver, function, destination, ptr::null_mut(), data)
            }

            id::cuMemcpyDtoH_v2 | id::cuMemcpyDtoH_v2_ptds => {
                let source = request.u64()?;
                let size =
                    usize::try_from(request.u64()?).map_err(|_| invalid("size too large"))?;
                if size > protocol::MAX_MESSAGE - 64 {
                    return Ok(CUDA_ERROR_INVALID_VALUE);
                }

                let mut data = vec![0u8; size];
                let result = mem::transmute::<*const c_void, MemcpyDtoH>(address)(
                    data.as_mut_ptr().cast(),
                    source,
                    size,
                );
                reply.bytes(&data);
                result
            }

            id::cuDeviceGetLuid => {
                let device = request.i32()?;
                let mut luid = [0 as c_char; 8];
                let mut mask: c_uint = 0;
                let result = mem::transmute::<*const c_void, GetLuid>(address)(
                    luid.as_mut_ptr(),
                    &mut mask,
                    device,
                );
                reply.value(&luid).u32(mask);
                result
            }

            id::cuDeviceGetPCIBusId => {
                let length = request.i32()?;
                let device = request.i32()?;
                if !(1..=4096).contains(&length) {
                    return Ok(CUDA_ERROR_INVALID_VALUE);
                }
                let mut text = vec![0u8; length as usize];
                let result = mem::transmute::<*const c_void, GetName>(address)(
                    text.as_mut_ptr().cast(),
                    length,
                    device,
                );
                let end = text
                    .iter()
                    .position(|&byte| byte == 0)
                    .unwrap_or(text.len());
                reply.bytes(&text[..end]);
                result
            }

            id::cuMemGetAllocationGranularity => {
                let prop: CUmemAllocationProp = request.value()?;
                let option: CUmemAllocationGranularity_flags = request.value()?;
                let mut granularity = 0usize;
                let result = mem::transmute::<*const c_void, Granularity>(address)(
                    &mut granularity,
                    &prop,
                    option,
                );
                reply.u64(granularity as u64);
                result
            }

            id::cuMemCreate => {
                let size =
                    usize::try_from(request.u64()?).map_err(|_| invalid("size too large"))?;
                let prop: CUmemAllocationProp = request.value()?;
                let flags = request.u64()?;
                let mut handle = 0;
                let result = mem::transmute::<*const c_void, MemCreate>(address)(
                    &mut handle,
                    size,
                    &prop,
                    flags,
                );
                if result == CUDA_SUCCESS {
                    session.created(handle);
                }
                reply.u64(handle);
                result
            }

            id::cuMemSetAccess => {
                let pointer = request.u64()?;
                let size =
                    usize::try_from(request.u64()?).map_err(|_| invalid("size too large"))?;
                let count =
                    usize::try_from(request.u64()?).map_err(|_| invalid("count too large"))?;
                if count > 256 {
                    return Ok(CUDA_ERROR_INVALID_VALUE);
                }
                let descriptors = (0..count)
                    .map(|_| request.value::<CUmemAccessDesc>())
                    .collect::<io::Result<Vec<_>>>()?;
                mem::transmute::<*const c_void, SetAccess>(address)(
                    pointer,
                    size,
                    descriptors.as_ptr(),
                    count,
                )
            }

            id::cuModuleLoadData | id::cuModuleLoadDataEx => {
                let image = aligned(request.bytes()?);
                let mut module = ptr::null_mut();
                let result = mem::transmute::<*const c_void, ModuleLoadData>(
                    driver.address(id::cuModuleLoadData).unwrap_or(address),
                )(&mut module, image.as_ptr().cast());
                reply.u64(module as u64);
                result
            }

            id::cuLibraryLoadData => {
                // The driver copies the code, so no options need to keep it alive
                let image = aligned(request.bytes()?);
                let mut library = ptr::null_mut();
                let result = mem::transmute::<*const c_void, LibraryLoadData>(address)(
                    &mut library,
                    image.as_ptr().cast(),
                    ptr::null_mut(),
                    ptr::null_mut(),
                    0,
                    ptr::null_mut(),
                    ptr::null_mut(),
                    0,
                );
                reply.u64(library as u64);
                result
            }

            id::cuLaunchKernel | id::cuLaunchKernel_ptsz => launch(address, false, request)?,
            id::cuLaunchKernelEx | id::cuLaunchKernelEx_ptsz => launch(address, true, request)?,

            id::cuMemcpyHtoDAsync_v2 | id::cuMemcpyHtoDAsync_v2_ptsz => {
                let destination = request.u64()?;
                let stream = request.u64()? as usize as CUstream;
                let data = request.bytes()?;
                upload(driver, function, destination, stream, data)
            }

            // Copied in the stream's order, then waited for, so the borrower's memory is
            // written at the right moment
            id::cuMemcpyDtoHAsync_v2 | id::cuMemcpyDtoHAsync_v2_ptsz => {
                let source = request.u64()?;
                let size =
                    usize::try_from(request.u64()?).map_err(|_| invalid("size too large"))?;
                let stream = request.u64()? as usize as CUstream;
                if size > protocol::MAX_MESSAGE - 64 {
                    return Ok(CUDA_ERROR_INVALID_VALUE);
                }

                let mut data = vec![0u8; size];
                let mut result = mem::transmute::<*const c_void, MemcpyDtoHAsync>(address)(
                    data.as_mut_ptr().cast(),
                    source,
                    size,
                    stream,
                );
                if result == CUDA_SUCCESS {
                    result = synchronize(driver, stream);
                }
                reply.bytes(&data);
                result
            }

            _ => match generated::dispatch(address, session, function, request, reply)? {
                Some(result) => result,
                None => {
                    eprintln!(
                        "{} is not supported yet",
                        protocol::function_name(function).unwrap_or("an unknown function")
                    );
                    CUDA_ERROR_NOT_SUPPORTED
                }
            },
        }
    };

    Ok(result)
}

/// Notices a borrower that vanished (Wi-Fi gone, laptop asleep) within about 20 s, so its
/// lease can run out instead of waiting on a connection that will never close: probes after
/// 10 s idle, every 2 s, 5 times
#[cfg(windows)]
fn keep_alive(stream: &TcpStream) {
    use std::os::windows::io::AsRawSocket;
    use windows_sys::Win32::Networking::WinSock::{
        IPPROTO_TCP, SO_KEEPALIVE, SOL_SOCKET, TCP_KEEPCNT, TCP_KEEPIDLE, TCP_KEEPINTVL, setsockopt,
    };

    let socket = stream.as_raw_socket() as usize;
    let set = |level: i32, option: i32, value: u32| unsafe {
        setsockopt(socket, level, option, (&value as *const u32).cast(), 4)
    };
    set(SOL_SOCKET, SO_KEEPALIVE, 1);
    set(IPPROTO_TCP, TCP_KEEPIDLE, 10);
    set(IPPROTO_TCP, TCP_KEEPINTVL, 2);
    set(IPPROTO_TCP, TCP_KEEPCNT, 5);
}

#[cfg(unix)]
fn keep_alive(stream: &TcpStream) {
    use std::os::fd::AsRawFd;

    let socket = stream.as_raw_fd();
    let set = |level: i32, option: i32, value: i32| unsafe {
        libc::setsockopt(
            socket,
            level,
            option,
            (&value as *const i32).cast(),
            size_of::<i32>() as libc::socklen_t,
        )
    };
    #[cfg(target_vendor = "apple")]
    let idle = libc::TCP_KEEPALIVE;
    #[cfg(not(target_vendor = "apple"))]
    let idle = libc::TCP_KEEPIDLE;
    set(libc::SOL_SOCKET, libc::SO_KEEPALIVE, 1);
    set(libc::IPPROTO_TCP, idle, 10);
    set(libc::IPPROTO_TCP, libc::TCP_KEEPINTVL, 2);
    set(libc::IPPROTO_TCP, libc::TCP_KEEPCNT, 5);
}

fn serve(
    cuda: Option<&Driver>,
    stream: TcpStream,
    cache: Option<Arc<Cache>>,
    lending: Option<Arc<Lending>>,
    peer: &str,
) -> io::Result<()> {
    type DriverGetVersion = unsafe extern "system" fn(*mut c_int) -> CUresult;

    stream.set_nodelay(true)?;
    keep_alive(&stream);
    let mut reader = BufReader::with_capacity(1 << 16, stream.try_clone()?);
    let mut writer = BufWriter::with_capacity(1 << 16, stream);

    let kind = protocol::read_hello(&mut reader).map_err(|error| match error.kind() {
        // Such as the borrower's test checking that the port answers
        io::ErrorKind::UnexpectedEof => {
            io::Error::new(error.kind(), "closed without a hello (a port check)")
        }
        _ => error,
    })?;

    let mut version = 0;
    if let Some(get_version) = cuda.and_then(|driver| unsafe { driver.get::<DriverGetVersion>(id::cuDriverGetVersion) }) {
        unsafe { get_version(&mut version) };
    }

    let token = match &lending {
        None => {
            protocol::write_lender_hello(&mut writer, version, &[])?;
            None
        }
        Some(lending) => match lending.admit(kind, &mut reader, &mut writer, version, peer)? {
            Some(token) => Some(token),
            None => return Ok(()),
        },
    };

    if kind == protocol::HELLO_CONTROL {
        let (Some(lending), Some(token)) = (&lending, &token) else {
            return Err(invalid("a control connection to a server that is not lending"));
        };
        // While it is lent over Wi-Fi, this PC's Wi-Fi keeps its latency low
        let held = match (reader.get_ref().local_addr(), reader.get_ref().peer_addr()) {
            (Ok(local), Ok(remote)) => nectar_lending::wifi::hold(local.ip(), remote.ip()),
            _ => None,
        };
        if held.is_some() {
            eprintln!("{peer}: this PC's Wi-Fi is in low-latency mode while the GPU is lent");
        }
        // A borrower silent for the lease's grace is gone
        reader.get_ref().set_read_timeout(Some(nectar_lending::lease::GRACE))?;
        let outcome = lending.control(token, &mut reader, &mut writer, peer);
        lending.leave(token);
        return outcome;
    }

    let profile = lending.as_ref().map_or_else(nectar_lending::Profile::current, |lending| lending.profile());
    nectar_lending::profile::tune(reader.get_ref(), profile);

    if kind == protocol::HELLO_LINK {
        let stream = reader.get_ref().try_clone()?;
        let outcome = nectar_lending::link::serve(&mut reader, &mut writer, &stream);
        if let (Some(lending), Some(token)) = (&lending, &token) {
            lending.leave(token);
        }
        return outcome;
    }

    if kind == protocol::HELLO_VULKAN {
        let outcome = vulkan::serve(reader, writer, cache, peer);
        if let (Some(lending), Some(token)) = (&lending, &token) {
            lending.leave(token);
        }
        return outcome;
    }

    let Some(driver) = cuda else {
        if let (Some(lending), Some(token)) = (&lending, &token) {
            lending.leave(token);
        }
        return Err(invalid("CUDA asked of a lender without CUDA"));
    };

    let mut session = Session {
        cache,
        ..Session::default()
    };
    let outcome = loop {
        let (function, body) = match protocol::read_message(&mut reader) {
            Ok(message) => message,
            // Windows resets the connection when the borrower's process ends
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::UnexpectedEof
                        | io::ErrorKind::ConnectionReset
                        | io::ErrorKind::ConnectionAborted
                ) =>
            {
                break Ok(());
            }
            Err(error) => break Err(error),
        };

        let mut reply = Writer::new();
        let result = match call(
            driver,
            &mut session,
            function,
            &mut Reader::new(&body),
            &mut reply,
        ) {
            Ok(result) => result,
            Err(error) => break Err(error),
        };

        let results = if result == CUDA_SUCCESS {
            reply.as_slice()
        } else {
            &[]
        };
        if let Some(lending) = &lending {
            lending.calls.fetch_add(1, Ordering::Relaxed);
            lending.received.fetch_add(8 + body.len() as u64, Ordering::Relaxed);
            lending.sent.fetch_add(8 + results.len() as u64, Ordering::Relaxed);
        }
        if let Err(error) = protocol::write_message(&mut writer, result as u32, &[results]) {
            break Err(error);
        }
        // The borrower sends many requests without waiting for their replies, so the
        // replies go out together once no more requests have arrived
        if reader.buffer().is_empty()
            && let Err(error) = writer.flush()
        {
            break Err(error);
        }
    };

    session.release(driver);
    if let (Some(lending), Some(token)) = (&lending, &token) {
        lending.leave(token);
    }
    outcome
}

/// This lender as discovery describes it (name, GPU 0, its memory, its platform and the APIs
/// it lends: CUDA when it has a CUDA driver, Vulkan when it has a Vulkan device), and the
/// driver's CUDA version as cuDriverGetVersion gives it
fn lender_info(cuda: Option<&Driver>, port: u16) -> (LenderInfo, i32) {
    type GetName = unsafe extern "system" fn(*mut c_char, c_int, CUdevice) -> CUresult;
    type TotalMem = unsafe extern "system" fn(*mut usize, CUdevice) -> CUresult;
    type DriverGetVersion = unsafe extern "system" fn(*mut c_int) -> CUresult;

    let mut info = LenderInfo {
        name: nectar_lending::computer_name(),
        gpu: String::new(),
        vram_mib: 0,
        vendor: String::new(),
        os: env::consts::OS.to_string(),
        arch: env::consts::ARCH.to_string(),
        apis: Vec::new(),
        port,
        state: State::Available,
        borrower: String::new(),
        hybrid: String::new(),
    };
    let mut version = 0;
    if let Some(driver) = cuda {
        let mut name = [0u8; 256];
        let mut memory = 0usize;
        unsafe {
            if let Some(get) = driver.get::<GetName>(id::cuDeviceGetName) {
                get(name.as_mut_ptr().cast(), name.len() as c_int, 0);
            }
            if let Some(get) = driver.get::<TotalMem>(id::cuDeviceTotalMem_v2) {
                get(&mut memory, 0);
            }
            if let Some(get) = driver.get::<DriverGetVersion>(id::cuDriverGetVersion) {
                get(&mut version);
            }
        }
        let end = name.iter().position(|&byte| byte == 0).unwrap_or(name.len());
        info.gpu = String::from_utf8_lossy(&name[..end]).into_owned();
        info.vram_mib = (memory >> 20) as u64;
        // ZLUDA runs CUDA on AMD GPUs and marks the device's name
        let zluda = info.gpu.contains("ZLUDA");
        info.vendor = if zluda { "AMD" } else { "NVIDIA" }.to_string();
        info.apis.push(Offer {
            api: "cuda".to_string(),
            version: format!("{}.{}", version / 1000, version % 1000 / 10),
            origin: if zluda { "layered" } else { "official" }.to_string(),
            driver: if zluda { "ZLUDA" } else { "NVIDIA" }.to_string(),
        });
    }
    // The first Vulkan device that is a GPU here: a real one before a software one
    let mut devices: Vec<nectar_detect::Api> = nectar_detect::vulkan::detect()
        .into_iter()
        .filter(|api| api.origin != nectar_detect::Origin::Remote)
        .collect();
    devices.sort_by_key(|api| api.origin == nectar_detect::Origin::Software);
    if let Some(device) = devices.first() {
        if info.gpu.is_empty() {
            info.gpu = device.device.clone();
            info.vram_mib = device.memory_mib;
            info.vendor = nectar_detect::vendor_name(device.vendor_id).to_string();
        }
        info.apis.push(Offer {
            api: "vulkan".to_string(),
            version: device.version.clone(),
            origin: device.origin.name().to_string(),
            driver: device.driver.clone(),
        });
    }
    (info, version)
}

/// Where uploads are kept unless told otherwise: %LOCALAPPDATA%\Nectar\cuda-cache, or
/// $XDG_CACHE_HOME/nectar/cuda-cache (~/.cache by default)
fn default_cache() -> Option<PathBuf> {
    if cfg!(windows) {
        env::var_os("LOCALAPPDATA").map(|folder| PathBuf::from(folder).join(r"Nectar\cuda-cache"))
    } else {
        env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
            .map(|folder| folder.join("nectar").join("cuda-cache"))
    }
}

fn main() {
    type Init = unsafe extern "system" fn(c_uint) -> CUresult;

    let mut listen = format!("0.0.0.0:{}", protocol::DEFAULT_PORT);
    let mut cache_folder = default_cache();
    let mut cache_limit_gb = DEFAULT_CACHE_GB;
    let mut cuda_driver = default_driver();
    let mut lend = false;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--lend" {
            lend = true;
            continue;
        }
        match (arg.as_str(), args.next()) {
            ("--listen", Some(value)) => listen = value,
            ("--cache", Some(value)) => cache_folder = Some(PathBuf::from(value)),
            ("--cuda-driver", Some(value)) => cuda_driver = PathBuf::from(value),
            ("--cache-limit-gb", Some(value)) if value.parse::<u64>().is_ok() => {
                cache_limit_gb = value.parse().unwrap()
            }
            _ => {
                eprintln!(
                    "usage: nectar-server [--listen <address:port>] [--lend] [--cache <folder>] [--cache-limit-gb <size, 0 for none>] [--cuda-driver <library>]"
                );
                std::process::exit(2);
            }
        }
    }
    // Not on the command line, where other users could read it
    let secret = env::var("NECTAR_LEND_SECRET").unwrap_or_default();
    if lend && secret.is_empty() {
        eprintln!("--lend needs the shared secret in NECTAR_LEND_SECRET");
        std::process::exit(2);
    }

    // Without CUDA (a Mac, an AMD or Intel GPU), the lender still lends Vulkan
    let cuda = match Driver::load(&cuda_driver) {
        Ok(driver) => {
            let result = unsafe {
                driver
                    .get::<Init>(id::cuInit)
                    .map_or(CUDA_ERROR_NOT_SUPPORTED, |init| init(0))
            };
            if result == CUDA_SUCCESS {
                Some(Arc::new(driver))
            } else {
                eprintln!("no CUDA: cuInit failed with {result}");
                None
            }
        }
        Err(error) => {
            eprintln!("no CUDA: cannot load {}: {error}", cuda_driver.display());
            None
        }
    };

    let listener = TcpListener::bind(&listen).unwrap_or_else(|error| {
        eprintln!("cannot listen on {listen}: {error}");
        std::process::exit(1);
    });
    match &cuda {
        Some(driver) => eprintln!(
            "nectar-server listening on {listen} ({} of {} CUDA driver functions found)",
            driver.functions.len(),
            protocol::NAMES.len()
        ),
        None => eprintln!("nectar-server listening on {listen}"),
    }

    let cache = match cache_folder.filter(|_| cache_limit_gb > 0) {
        None => None,
        Some(folder) => match Cache::open(&folder, cache_limit_gb << 30) {
            Ok((cache, total)) => {
                eprintln!(
                    "keeping uploads in {} ({:.1} of {cache_limit_gb} GB used)",
                    folder.display(),
                    total as f64 / (1u64 << 30) as f64
                );
                Some(Arc::new(cache))
            }
            Err(error) => {
                eprintln!("not keeping uploads: {}: {error}", folder.display());
                None
            }
        },
    };

    let lending = if lend {
        let local = listener.local_addr().expect("a bound listener has an address");
        let port = local.port();
        let (mut info, cuda_version) = lender_info(cuda.as_deref(), port);
        let hybrid = lend::find_hybrid(cuda_version);
        if let Some((key, _)) = &hybrid {
            info.hybrid = key.clone();
        }
        if hybrid.is_none() && cfg!(windows) {
            eprintln!("no nvcudart_hybrid64.dll in this PC's NVIDIA driver; borrowers must bring their own");
        }
        let lending = Arc::new(Lending::new(secret.into_bytes(), info.clone(), hybrid));
        if let Err(error) = lending.announce(local.ip()) {
            eprintln!(
                "cannot answer discovery on UDP port {}: {error}",
                nectar_lending::discovery::PORT
            );
            std::process::exit(1);
        }
        eprintln!(
            "lending {} ({} MiB, {}) as {}; answering discovery on UDP port {}",
            info.gpu,
            info.vram_mib,
            info.lends(),
            info.name,
            nectar_lending::discovery::PORT
        );
        Some(lending)
    } else {
        None
    };

    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let peer = stream
            .peer_addr()
            .map(|address| address.to_string())
            .unwrap_or_default();
        let cuda = cuda.clone();
        let cache = cache.clone();
        let lending = lending.clone();

        thread::spawn(move || {
            eprintln!("{peer} connected");
            match serve(cuda.as_deref(), stream, cache, lending, &peer) {
                Ok(()) => eprintln!("{peer} disconnected"),
                Err(error) => eprintln!("{peer} dropped: {error}"),
            }
        });
    }
}
