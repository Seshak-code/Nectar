//! cuda-probe: checks a CUDA driver library end to end. Run it against the real driver and
//! against Nectar's nvcuda.dll to compare what an application would see.
//!
//! Usage: cuda-probe [path to nvcuda.dll] [--mib <copy size>] [--calls <latency samples>]
//! Without a path it uses the real driver in System32.

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::path::PathBuf;
use std::time::Instant;
use std::{env, process, ptr};

use libloading::Library;

type CUresult = c_int;
type CUdevice = c_int;
type CUdeviceptr = u64;
type CUcontext = *mut c_void;

const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

struct Cuda {
    _library: Library,
    init: unsafe extern "system" fn(c_uint) -> CUresult,
    driver_get_version: unsafe extern "system" fn(*mut c_int) -> CUresult,
    get_error_name: unsafe extern "system" fn(CUresult, *mut *const c_char) -> CUresult,
    device_get_count: unsafe extern "system" fn(*mut c_int) -> CUresult,
    device_get: unsafe extern "system" fn(*mut CUdevice, c_int) -> CUresult,
    device_get_name: unsafe extern "system" fn(*mut c_char, c_int, CUdevice) -> CUresult,
    device_total_mem: unsafe extern "system" fn(*mut usize, CUdevice) -> CUresult,
    device_get_attribute: unsafe extern "system" fn(*mut c_int, c_int, CUdevice) -> CUresult,
    primary_ctx_retain: unsafe extern "system" fn(*mut CUcontext, CUdevice) -> CUresult,
    primary_ctx_release: unsafe extern "system" fn(CUdevice) -> CUresult,
    ctx_set_current: unsafe extern "system" fn(CUcontext) -> CUresult,
    ctx_synchronize: unsafe extern "system" fn() -> CUresult,
    mem_get_info: unsafe extern "system" fn(*mut usize, *mut usize) -> CUresult,
    mem_alloc: unsafe extern "system" fn(*mut CUdeviceptr, usize) -> CUresult,
    mem_free: unsafe extern "system" fn(CUdeviceptr) -> CUresult,
    memcpy_htod: unsafe extern "system" fn(CUdeviceptr, *const c_void, usize) -> CUresult,
    memcpy_dtoh: unsafe extern "system" fn(*mut c_void, CUdeviceptr, usize) -> CUresult,
    memcpy: unsafe extern "system" fn(CUdeviceptr, CUdeviceptr, usize) -> CUresult,
    module_load_data: unsafe extern "system" fn(*mut *mut c_void, *const c_void) -> CUresult,
    module_get_function:
        unsafe extern "system" fn(*mut *mut c_void, *mut c_void, *const c_char) -> CUresult,
    module_unload: unsafe extern "system" fn(*mut c_void) -> CUresult,
    library_load_data: unsafe extern "system" fn(
        *mut *mut c_void,
        *const c_void,
        *mut c_void,
        *mut *mut c_void,
        c_uint,
        *mut c_void,
        *mut *mut c_void,
        c_uint,
    ) -> CUresult,
    library_get_kernel:
        unsafe extern "system" fn(*mut *mut c_void, *mut c_void, *const c_char) -> CUresult,
    library_unload: unsafe extern "system" fn(*mut c_void) -> CUresult,
    launch_kernel: unsafe extern "system" fn(
        *mut c_void,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CUresult,
}

/// The test kernel, c[i] = a[i] + b[i] (kernels/add.cu)
const ADD_PTX: &str = concat!(include_str!("../kernels/add.ptx"), "\0");

impl Cuda {
    fn load(path: &PathBuf) -> Result<Self, libloading::Error> {
        let library = unsafe { Library::new(path)? };

        macro_rules! symbol {
            ($name:literal) => {
                *unsafe { library.get(concat!($name, "\0").as_bytes())? }
            };
        }

        Ok(Self {
            init: symbol!("cuInit"),
            driver_get_version: symbol!("cuDriverGetVersion"),
            get_error_name: symbol!("cuGetErrorName"),
            device_get_count: symbol!("cuDeviceGetCount"),
            device_get: symbol!("cuDeviceGet"),
            device_get_name: symbol!("cuDeviceGetName"),
            device_total_mem: symbol!("cuDeviceTotalMem_v2"),
            device_get_attribute: symbol!("cuDeviceGetAttribute"),
            primary_ctx_retain: symbol!("cuDevicePrimaryCtxRetain"),
            primary_ctx_release: symbol!("cuDevicePrimaryCtxRelease_v2"),
            ctx_set_current: symbol!("cuCtxSetCurrent"),
            ctx_synchronize: symbol!("cuCtxSynchronize"),
            mem_get_info: symbol!("cuMemGetInfo_v2"),
            mem_alloc: symbol!("cuMemAlloc_v2"),
            mem_free: symbol!("cuMemFree_v2"),
            memcpy_htod: symbol!("cuMemcpyHtoD_v2"),
            memcpy_dtoh: symbol!("cuMemcpyDtoH_v2"),
            memcpy: symbol!("cuMemcpy"),
            module_load_data: symbol!("cuModuleLoadData"),
            module_get_function: symbol!("cuModuleGetFunction"),
            module_unload: symbol!("cuModuleUnload"),
            library_load_data: symbol!("cuLibraryLoadData"),
            library_get_kernel: symbol!("cuLibraryGetKernel"),
            library_unload: symbol!("cuLibraryUnload"),
            launch_kernel: symbol!("cuLaunchKernel"),
            _library: library,
        })
    }

    /// Runs the test kernel and checks every element of the result
    unsafe fn run_add(&self, what: &str, function: *mut c_void) {
        const N: usize = 1 << 20;
        let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..N).map(|i| 2.0 * i as f32).collect();
        let mut c = vec![0f32; N];
        let bytes = N * std::mem::size_of::<f32>();

        unsafe {
            let (mut da, mut db, mut dc) = (0, 0, 0);
            self.check("cuMemAlloc", (self.mem_alloc)(&mut da, bytes));
            self.check("cuMemAlloc", (self.mem_alloc)(&mut db, bytes));
            self.check("cuMemAlloc", (self.mem_alloc)(&mut dc, bytes));
            self.check("cuMemcpyHtoD", (self.memcpy_htod)(da, a.as_ptr().cast(), bytes));
            self.check("cuMemcpyHtoD", (self.memcpy_htod)(db, b.as_ptr().cast(), bytes));

            let mut n = N as c_int;
            let mut arguments: [*mut c_void; 4] = [
                (&mut da as *mut u64).cast(),
                (&mut db as *mut u64).cast(),
                (&mut dc as *mut u64).cast(),
                (&mut n as *mut c_int).cast(),
            ];
            // On a borrowed GPU the copies may still be on their way; time only the kernel
            self.check("cuCtxSynchronize", (self.ctx_synchronize)());
            let started = Instant::now();
            self.check(
                "cuLaunchKernel",
                (self.launch_kernel)(
                    function,
                    N.div_ceil(256) as c_uint,
                    1,
                    1,
                    256,
                    1,
                    1,
                    0,
                    ptr::null_mut(),
                    arguments.as_mut_ptr(),
                    ptr::null_mut(),
                ),
            );
            self.check("cuCtxSynchronize", (self.ctx_synchronize)());
            let elapsed = started.elapsed();

            self.check("cuMemcpyDtoH", (self.memcpy_dtoh)(c.as_mut_ptr().cast(), dc, bytes));
            for pointer in [da, db, dc] {
                self.check("cuMemFree", (self.mem_free)(pointer));
            }

            if let Some(i) = (0..N).find(|&i| c[i] != 3.0 * i as f32) {
                println!("FAIL {what}: c[{i}] = {}, expected {}", c[i], 3.0 * i as f32);
                process::exit(1);
            }
            println!(
                "kernel       {what}: {N} elements correct, launch and sync {:.2} ms",
                elapsed.as_secs_f64() * 1e3
            );
        }
    }

    /// Ends the probe with the error's name when a call fails
    fn check(&self, what: &str, result: CUresult) {
        if result != 0 {
            let mut name: *const c_char = ptr::null();
            let text = if unsafe { (self.get_error_name)(result, &mut name) } == 0 && !name.is_null() {
                unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned()
            } else {
                "unknown error".to_string()
            };
            println!("FAIL {what}: {text} ({result})");
            process::exit(1);
        }
    }
}

fn main() {
    let mut path = None;
    let mut mib = 64usize;
    let mut calls = 1000u32;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--mib" => mib = args.next().and_then(|value| value.parse().ok()).expect("--mib needs a number"),
            "--calls" => calls = args.next().and_then(|value| value.parse().ok()).expect("--calls needs a number"),
            _ => path = Some(PathBuf::from(arg)),
        }
    }

    // On Linux, by name through the library path, as programs find it
    let path = path.unwrap_or_else(|| {
        if cfg!(windows) {
            env::var_os("SystemRoot")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(r"C:\Windows"))
                .join(r"System32\nvcuda.dll")
        } else {
            PathBuf::from("libcuda.so.1")
        }
    });

    let cuda = Cuda::load(&path).unwrap_or_else(|error| {
        println!("FAIL loading {}: {error}", path.display());
        process::exit(1);
    });
    println!("library      {}", path.display());

    unsafe {
        let started = Instant::now();
        cuda.check("cuInit", (cuda.init)(0));
        println!("cuInit       {:.1} ms", started.elapsed().as_secs_f64() * 1e3);

        let mut version = 0;
        cuda.check("cuDriverGetVersion", (cuda.driver_get_version)(&mut version));
        println!("driver       CUDA {}.{}", version / 1000, version % 1000 / 10);

        let mut count = 0;
        cuda.check("cuDeviceGetCount", (cuda.device_get_count)(&mut count));
        for ordinal in 0..count {
            let mut device = 0;
            cuda.check("cuDeviceGet", (cuda.device_get)(&mut device, ordinal));

            let mut name = [0 as c_char; 256];
            cuda.check("cuDeviceGetName", (cuda.device_get_name)(name.as_mut_ptr(), name.len() as c_int, device));

            let mut memory = 0;
            cuda.check("cuDeviceTotalMem", (cuda.device_total_mem)(&mut memory, device));

            let attribute = |attribute| {
                let mut value = 0;
                cuda.check("cuDeviceGetAttribute", (cuda.device_get_attribute)(&mut value, attribute, device));
                value
            };

            println!(
                "device {ordinal}     {}, {:.1} GiB, compute {}.{}, {} SMs",
                CStr::from_ptr(name.as_ptr()).to_string_lossy(),
                memory as f64 / (1u64 << 30) as f64,
                attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
                attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR),
                attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
            );
        }

        if count == 0 {
            println!("FAIL no devices");
            process::exit(1);
        }

        let mut context = ptr::null_mut();
        cuda.check("cuDevicePrimaryCtxRetain", (cuda.primary_ctx_retain)(&mut context, 0));
        cuda.check("cuCtxSetCurrent", (cuda.ctx_set_current)(context));

        let (mut free, mut total) = (0, 0);
        cuda.check("cuMemGetInfo", (cuda.mem_get_info)(&mut free, &mut total));
        println!("memory       {:.1} of {:.1} GiB free", free as f64 / (1u64 << 30) as f64, total as f64 / (1u64 << 30) as f64);

        // Small calls show the cost of one round trip. Free memory changes, so Nectar always
        // asks the lender for it, where it keeps answers such as the device count.
        let started = Instant::now();
        for _ in 0..calls {
            cuda.check("cuMemGetInfo", (cuda.mem_get_info)(&mut free, &mut total));
        }
        println!("latency      {:.1} us per call ({calls} calls)", started.elapsed().as_secs_f64() * 1e6 / calls as f64);

        let size = mib << 20;
        let sent: Vec<u32> = (0..size as u32 / 4).map(|i| i.wrapping_mul(2_654_435_761)).collect();
        let mut received = vec![0u32; sent.len()];

        let mut device_memory = 0;
        cuda.check("cuMemAlloc", (cuda.mem_alloc)(&mut device_memory, size));

        let started = Instant::now();
        cuda.check("cuMemcpyHtoD", (cuda.memcpy_htod)(device_memory, sent.as_ptr().cast(), size));
        cuda.check("cuCtxSynchronize", (cuda.ctx_synchronize)());
        let upload = started.elapsed().as_secs_f64();

        let started = Instant::now();
        cuda.check("cuMemcpyDtoH", (cuda.memcpy_dtoh)(received.as_mut_ptr().cast(), device_memory, size));
        let download = started.elapsed().as_secs_f64();

        cuda.check("cuMemFree", (cuda.mem_free)(device_memory));

        let megabytes = size as f64 / 1e6;
        println!("upload       {mib} MiB at {:.0} MB/s", megabytes / upload);
        println!("download     {mib} MiB at {:.0} MB/s", megabytes / download);

        if sent != received {
            println!("FAIL the data read back differs from what was written");
            process::exit(1);
        }

        // The same kernel loaded as a module, then as a library, whose launches take a
        // CUkernel in place of a CUfunction
        let mut module = ptr::null_mut();
        cuda.check("cuModuleLoadData", (cuda.module_load_data)(&mut module, ADD_PTX.as_ptr().cast()));
        let mut function = ptr::null_mut();
        cuda.check("cuModuleGetFunction", (cuda.module_get_function)(&mut function, module, c"add".as_ptr()));
        cuda.run_add("module", function);
        cuda.check("cuModuleUnload", (cuda.module_unload)(module));

        let mut library = ptr::null_mut();
        cuda.check(
            "cuLibraryLoadData",
            (cuda.library_load_data)(
                &mut library,
                ADD_PTX.as_ptr().cast(),
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                ptr::null_mut(),
                ptr::null_mut(),
                0,
            ),
        );
        let mut kernel = ptr::null_mut();
        cuda.check("cuLibraryGetKernel", (cuda.library_get_kernel)(&mut kernel, library, c"add".as_ptr()));
        cuda.run_add("library", kernel);
        cuda.check("cuLibraryUnload", (cuda.library_unload)(library));

        cuda.check("cuDevicePrimaryCtxRelease", (cuda.primary_ctx_release)(0));
        // cuMemcpy takes any unified address: GPU to GPU must work. A host address is valid
        // on the real driver, while Nectar refuses it rather than let the lender's driver
        // read its own process memory.
        let size = 1 << 20;
        let pattern: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();
        let mut copied = vec![0u8; size];
        let (mut from, mut to) = (0, 0);
        cuda.check("cuMemAlloc", (cuda.mem_alloc)(&mut from, size));
        cuda.check("cuMemAlloc", (cuda.mem_alloc)(&mut to, size));
        cuda.check("cuMemcpyHtoD", (cuda.memcpy_htod)(from, pattern.as_ptr().cast(), size));
        cuda.check("cuMemcpy", (cuda.memcpy)(to, from, size));
        cuda.check("cuMemcpyDtoH", (cuda.memcpy_dtoh)(copied.as_mut_ptr().cast(), to, size));
        if copied != pattern {
            println!("FAIL cuMemcpy between GPU buffers changed the data");
            process::exit(1);
        }
        let host = (cuda.memcpy)(to, pattern.as_ptr() as CUdeviceptr, size);
        println!("cuMemcpy     GPU to GPU correct; from a host address: result {host}");
        cuda.check("cuMemFree", (cuda.mem_free)(from));
        cuda.check("cuMemFree", (cuda.mem_free)(to));

        println!("PASS {mib} MiB round trip matches and kernels compute correctly");
    }
}
