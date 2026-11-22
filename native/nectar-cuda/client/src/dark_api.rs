//! The CUDA runtime's undocumented driver interface. cuGetExportTable hands out tables of
//! function pointers, each named by a UUID, and the runtime will not start without them.
//!
//! Which tables exist, where their entries sit and what they do are adapted from ZLUDA
//! (dark_api/src/lib.rs and zluda/src/impl/driver.rs, https://github.com/vosen/ZLUDA, MIT;
//! see LICENSE-ZLUDA). Nothing here comes from NVIDIA's binaries.
//!
//! Most entries are answered here on the borrower. Those that need the GPU call this
//! library's own exports, so they reach the lender like any other call.

use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::ffi::{CStr, c_char, c_uchar, c_uint, c_void};
use std::sync::Mutex;
use std::{env, mem, process, ptr};

use nectar_cuda_sys::*;

use crate::code::load_module;
use crate::integrity::{self, DeviceHashInfo};
use crate::{generated, log};

/// A table as the runtime reads it: function pointers, most led by the table's size in bytes
#[repr(transparent)]
struct Table<const N: usize>([*const c_void; N]);

unsafe impl<const N: usize> Sync for Table<N> {}

impl<const N: usize> Table<N> {
    const SIZE: *const c_void = (N * mem::size_of::<usize>()) as *const c_void;

    fn start(&self) -> *const c_void {
        self.0.as_ptr().cast()
    }
}

const NULL: *const c_void = ptr::null();

/// A UUID written as text, as the bytes cuGetExportTable is given
const fn uuid(text: &str) -> [u8; 16] {
    let text = text.as_bytes();
    let mut bytes = [0u8; 16];
    let (mut index, mut digits) = (0, 0);
    while index < text.len() {
        let digit = match text[index] {
            b'-' => {
                index += 1;
                continue;
            }
            c @ b'0'..=b'9' => c - b'0',
            c @ b'a'..=b'f' => c - b'a' + 10,
            c @ b'A'..=b'F' => c - b'A' + 10,
            _ => panic!("not a UUID"),
        };
        bytes[digits / 2] = bytes[digits / 2] << 4 | digit;
        digits += 1;
        index += 1;
    }
    assert!(digits == 32);
    bytes
}

static CUDART_INTERFACE: Table<13> = Table([
    Table::<13>::SIZE,
    get_module_from_cubin as *const c_void,
    cudart_interface_fn2 as *const c_void,
    NULL,
    primary_ctx_retain_with_flags as *const c_void,
    NULL,
    get_module_from_cubin_ext1 as *const c_void,
    cudart_interface_fn7 as *const c_void,
    get_module_from_cubin_ext2 as *const c_void,
    NULL,
    NULL,
    NULL,
    load_compilers as *const c_void,
]);

/// The real driver's layout (3 entries); ZLUDA leaves its entries empty because its own
/// cuBLAS never calls them. NVIDIA's cuBLAS does, around each of its calls.
static TOOLS_TLS: Table<3> = Table([
    Table::<3>::SIZE,
    tools_tls_entry1 as *const c_void,
    tools_tls_enter as *const c_void,
]);

static TOOLS_RUNTIME_CALLBACK_HOOKS: Table<7> = Table([
    Table::<7>::SIZE,
    NULL,
    get_unknown_buffer1 as *const c_void,
    NULL,
    NULL,
    NULL,
    get_unknown_buffer2 as *const c_void,
]);

/// The only table without its size in front
static CONTEXT_LOCAL_STORAGE: Table<4> = Table([
    context_local_storage_put as *const c_void,
    context_local_storage_delete as *const c_void,
    context_local_storage_get as *const c_void,
    NULL,
]);

static CTX_CREATE_BYPASS: Table<2> =
    Table([Table::<2>::SIZE, ctx_create_v2_bypass as *const c_void]);

static HEAP_ACCESS: Table<3> = Table([
    Table::<3>::SIZE,
    heap_alloc as *const c_void,
    heap_free as *const c_void,
]);

static DEVICE_EXTENDED_RT: Table<26> = {
    let mut entries = [NULL; 26];
    entries[0] = Table::<26>::SIZE;
    entries[5] = device_get_attribute_ext as *const c_void;
    entries[13] = device_get_something as *const c_void;
    Table(entries)
};

static INTEGRITY_CHECK: Table<3> =
    Table([Table::<3>::SIZE, integrity_check as *const c_void, NULL]);

static CONTEXT_CHECKS: Table<4> = Table([
    Table::<4>::SIZE,
    NULL,
    context_check as *const c_void,
    check_fn3 as *const c_void,
]);

/// On Windows the static runtime loads nvcudart_hybrid64.dll through this
static HYBRID_CUDART: Table<3> = Table([
    Table::<3>::SIZE,
    hybrid_runtime_load_get_proc_address as *const c_void,
    hybrid_runtime_free as *const c_void,
]);

/// The table for a UUID, or None for one this library does not provide
pub(crate) fn export_table(id: &[u8; 16]) -> Option<*const c_void> {
    let tables: [([u8; 16], *const c_void); 10] = [
        (
            uuid("6BD5FB6C-5BF4-E74A-8987-D93912FD9DF9"),
            CUDART_INTERFACE.start(),
        ),
        (
            uuid("42D85A81-23F6-CB47-8298-F6E78A3AECDC"),
            TOOLS_TLS.start(),
        ),
        (
            uuid("A094798C-2E74-2E74-93F2-0800200C0A66"),
            TOOLS_RUNTIME_CALLBACK_HOOKS.start(),
        ),
        (
            uuid("C693336E-1121-DF11-A8C3-68F355D89593"),
            CONTEXT_LOCAL_STORAGE.start(),
        ),
        (
            uuid("0CA50B8C-1004-929A-89A7-D0DF10E77286"),
            CTX_CREATE_BYPASS.start(),
        ),
        (
            uuid("195BCBF4-D67D-024A-ACC5-1D29CEA631AE"),
            HEAP_ACCESS.start(),
        ),
        (
            uuid("B10541E1-F7C7-C74A-9F64-F223BE99F1E2"),
            DEVICE_EXTENDED_RT.start(),
        ),
        (
            uuid("D4082055-BDE6-704B-8D34-BA123C66E1F2"),
            INTEGRITY_CHECK.start(),
        ),
        (
            uuid("263E8860-7CD2-6143-92F6-BBD5006DFA7E"),
            CONTEXT_CHECKS.start(),
        ),
        (
            uuid("DD9E98A4-F410-4844-99B0-E2AEB7B432B5"),
            HYBRID_CUDART.start(),
        ),
    ];
    tables
        .iter()
        .find(|(uuid, _)| uuid == id)
        .map(|&(_, table)| table)
        .or_else(|| unknown::table(id))
}

/// Tables the runtime asks for but, in every program traced so far, never calls into. Their
/// layout is the real driver's, as the tracer records it (trace/src/tables.rs). Every entry
/// is a stub that logs the call once and returns CUDA_ERROR_NOT_SUPPORTED. Returning without
/// reading the arguments is safe whatever the signature, in the x64 calling convention.
mod unknown {
    use std::arch::naked_asm;
    use std::collections::HashSet;
    use std::ffi::c_void;
    use std::sync::{Mutex, OnceLock};

    use nectar_cuda_sys::*;

    use super::uuid;
    use crate::log;

    /// UUID, number of entries including the size in entry 0, entries that are NULL
    const TABLES: [(&str, usize, &[usize]); 2] = [
        ("F8CFF951-2146-8B4E-B9E2-FB469E7C0DD9", 3, &[]),
        ("21318C60-9714-3248-8CA6-41FF7324C8F2", 100, &[91]),
    ];

    /// Entries whose behavior the tracer has observed, implemented to match
    fn implemented(table: &str, entry: usize) -> Option<*const c_void> {
        match (table, entry) {
            ("21318C60-9714-3248-8CA6-41FF7324C8F2", 4) => {
                Some(super::tools_check as *const c_void)
            }
            ("F8CFF951-2146-8B4E-B9E2-FB469E7C0DD9", 2) => {
                Some(super::timing_mark as *const c_void)
            }
            _ => None,
        }
    }

    const STUBS: usize = 128;

    /// Stub N loads N into r11 and jumps to `stub`; the stubs are 16 bytes apart
    #[unsafe(naked)]
    unsafe extern "C" fn stubs() {
        naked_asm!(
            ".p2align 4",
            ".set nectar_stub, 0",
            ".rept {count}",
            ".p2align 4",
            "movl $nectar_stub, %r11d",
            "jmp {stub}",
            ".set nectar_stub, nectar_stub + 1",
            ".endr",
            count = const STUBS,
            stub = sym stub,
            options(att_syntax),
        )
    }

    #[unsafe(naked)]
    unsafe extern "C" fn stub() {
        naked_asm!(
            "sub $0x28, %rsp",
            "mov %r11d, %ecx",
            "call {called}",
            "add $0x28, %rsp",
            "ret",
            called = sym called,
            options(att_syntax),
        )
    }

    fn address(stub: usize) -> *const c_void {
        (((stubs as *const () as usize + 15) & !15) + stub * 16) as *const c_void
    }

    /// (table, entry) of each stub handed out
    static OWNERS: Mutex<Vec<(&'static str, usize)>> = Mutex::new(Vec::new());

    extern "system" fn called(stub: u32) -> CUresult {
        static REPORTED: Mutex<Option<HashSet<u32>>> = Mutex::new(None);

        let mut reported = REPORTED
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if reported.get_or_insert_with(HashSet::new).insert(stub) {
            let owners = OWNERS
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some((table, entry)) = owners.get(stub as usize) {
                log(&format!(
                    "entry {entry} of export table {table} was called; it is not supported yet"
                ));
            }
        }
        CUDA_ERROR_NOT_SUPPORTED
    }

    pub(super) fn table(id: &[u8; 16]) -> Option<*const c_void> {
        static BUILT: OnceLock<Vec<([u8; 16], usize)>> = OnceLock::new();

        let built = BUILT.get_or_init(|| {
            let mut owners = OWNERS
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            TABLES
                .iter()
                .map(|&(name, count, nulls)| {
                    let mut entries = vec![(count * size_of::<usize>()) as *const c_void];
                    for entry in 1..count {
                        if let Some(function) = implemented(name, entry) {
                            entries.push(function);
                        } else if nulls.contains(&entry) || owners.len() >= STUBS {
                            entries.push(std::ptr::null());
                        } else {
                            entries.push(address(owners.len()));
                            owners.push((name, entry));
                        }
                    }
                    (
                        uuid(name),
                        Box::leak(entries.into_boxed_slice()).as_ptr() as usize,
                    )
                })
                .collect()
        });

        built
            .iter()
            .find(|(uuid, _)| uuid == id)
            .map(|&(_, table)| table as *const c_void)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn stubs_report_not_supported() {
            let table =
                table(&uuid("21318C60-9714-3248-8CA6-41FF7324C8F2")).unwrap() as *const usize;
            unsafe {
                assert_eq!(*table, 800);
                assert_eq!(*table.add(91), 0);
                let entry: extern "C" fn(u64, u64) -> CUresult = std::mem::transmute(*table.add(5));
                assert_eq!(entry(1, 2), CUDA_ERROR_NOT_SUPPORTED);
            }
        }
    }
}

// CUDART_INTERFACE

unsafe extern "system" fn get_module_from_cubin(
    module: *mut CUmodule,
    fatbinc_wrapper: *const c_void,
) -> CUresult {
    unsafe { load_module(module, fatbinc_wrapper) }
}

/// The device's primary context. Each is retained once and kept for the life of the process.
unsafe extern "system" fn cudart_interface_fn2(
    context: *mut CUcontext,
    device: CUdevice,
) -> CUresult {
    static PRIMARY: Mutex<Option<HashMap<CUdevice, usize>>> = Mutex::new(None);

    if context.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    let mut primary = PRIMARY
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let primary = primary.get_or_insert_with(HashMap::new);
    let handle = match primary.get(&device) {
        Some(&handle) => handle,
        None => {
            let mut retained = ptr::null_mut();
            let result = unsafe { generated::cuDevicePrimaryCtxRetain(&mut retained, device) };
            if result != CUDA_SUCCESS {
                return result;
            }
            primary.insert(device, retained as usize);
            retained as usize
        }
    };

    unsafe { *context = handle as CUcontext };
    CUDA_SUCCESS
}

unsafe extern "system" fn primary_ctx_retain_with_flags(
    device: CUdevice,
    flags: c_uint,
) -> CUresult {
    let mut context = ptr::null_mut();
    let result = unsafe { generated::cuDevicePrimaryCtxRetain(&mut context, device) };
    if result != CUDA_SUCCESS {
        return result;
    }
    unsafe { generated::cuDevicePrimaryCtxSetFlags_v2(device, flags) }
}

unsafe extern "system" fn get_module_from_cubin_ext1(
    module: *mut CUmodule,
    fatbinc_wrapper: *const c_void,
    arg3: *mut c_void,
    arg4: *mut c_void,
    arg5: c_uint,
) -> CUresult {
    if !arg3.is_null() || !arg4.is_null() || arg5 != 0 {
        log("loading a module with extra arguments is not supported yet");
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    unsafe { load_module(module, fatbinc_wrapper) }
}

unsafe extern "system" fn cudart_interface_fn7(_arg1: usize) {}

unsafe extern "system" fn get_module_from_cubin_ext2(
    fatbin_header: *const c_void,
    module: *mut CUmodule,
    arg3: *mut c_void,
    arg4: *mut c_void,
    arg5: c_uint,
) -> CUresult {
    if !arg3.is_null() || !arg4.is_null() || arg5 != 0 {
        log("loading a module with extra arguments is not supported yet");
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    unsafe { load_module(module, fatbin_header) }
}

/// The real driver loads its PTX compilers here; the lender's driver has its own
unsafe extern "system" fn load_compilers() -> CUresult {
    CUDA_SUCCESS
}

// TOOLS_TLS and entry 4 of table 21318c60: the profiling hooks NVIDIA's cuBLAS calls around
// each of its calls. What they do was seen with the tracer on the real driver, with no
// profiler attached:
//   TOOLS_TLS[2](record, domain, id, data) writes a per-thread tools object into the
//   record's first field and returns 0;
//   21318c60[4](object, state, 0x100, 0x100) writes 1 into the state's first field and
//   returns 0.
// With no profiler on the borrower, the object only needs to exist.

static TOOLS_OBJECT: Buffer<64> = Buffer(UnsafeCell::new([0; 64]));

unsafe extern "system" fn tools_tls_entry1() -> CUresult {
    log("entry 1 of the TOOLS_TLS export table was called; it is not supported yet");
    CUDA_ERROR_NOT_SUPPORTED
}

unsafe extern "system" fn tools_tls_enter(
    record: *mut *mut c_void,
    _domain: usize,
    _id: usize,
    _data: *mut c_void,
) -> CUresult {
    if !record.is_null() {
        unsafe { *record = TOOLS_OBJECT.0.get().cast() };
    }
    CUDA_SUCCESS
}

unsafe extern "system" fn tools_check(
    _object: *mut c_void,
    state: *mut u64,
    _arg3: usize,
    _arg4: usize,
) -> CUresult {
    if !state.is_null() {
        unsafe { *state = 1 };
    }
    CUDA_SUCCESS
}

/// f8cff951[2], which llama.cpp's static runtime calls around launches. In the tracer it
/// never writes memory, and its arguments look like a phase (0 or 1), a value, a
/// QueryPerformanceCounter reading and the counter's frequency, so it is a timing mark.
/// What it returns stays the same across calls and looks like a leftover pointer.
unsafe extern "system" fn timing_mark(
    _phase: usize,
    _value: usize,
    _time: u64,
    _frequency: u64,
) -> usize {
    0
}

// TOOLS_RUNTIME_CALLBACK_HOOKS: two buffers the runtime writes into and never reads back

struct Buffer<const N: usize>(UnsafeCell<[u32; N]>);

unsafe impl<const N: usize> Sync for Buffer<N> {}

static UNKNOWN_BUFFER1: Buffer<1024> = Buffer(UnsafeCell::new([0; 1024]));
static UNKNOWN_BUFFER2: Buffer<14> = Buffer(UnsafeCell::new([0; 14]));

unsafe extern "system" fn get_unknown_buffer1(buffer: *mut *mut c_void, size: *mut usize) {
    unsafe {
        *buffer = UNKNOWN_BUFFER1.0.get().cast();
        *size = 1024;
    }
}

unsafe extern "system" fn get_unknown_buffer2(buffer: *mut *mut c_void, size: *mut usize) {
    unsafe {
        *buffer = UNKNOWN_BUFFER2.0.get().cast();
        *size = 14;
    }
}

// CONTEXT_LOCAL_STORAGE: values the runtime keeps per context and key. Contexts are the
// lender's handles, which are stable, so the values can stay here.

static STORAGE: Mutex<Option<HashMap<(usize, usize), usize>>> = Mutex::new(None);

/// The given context, or the current one for NULL. With neither, the real driver answers
/// CUDA_ERROR_INVALID_CONTEXT (seen with the tracer).
fn context_or_current(context: CUcontext) -> Result<usize, CUresult> {
    if !context.is_null() {
        return Ok(context as usize);
    }
    let mut current = ptr::null_mut();
    match unsafe { generated::cuCtxGetCurrent(&mut current) } {
        CUDA_SUCCESS if current.is_null() => Err(CUDA_ERROR_INVALID_CONTEXT),
        CUDA_SUCCESS => Ok(current as usize),
        error => Err(error),
    }
}

/// The destructor would run when a primary context is reset, which Nectar does not track yet
unsafe extern "system" fn context_local_storage_put(
    context: CUcontext,
    key: *mut c_void,
    value: *mut c_void,
    _destructor: Option<extern "system" fn(CUcontext, *mut c_void, *mut c_void)>,
) -> CUresult {
    let context = match context_or_current(context) {
        Ok(context) => context,
        Err(error) => return error,
    };
    let mut storage = STORAGE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    storage
        .get_or_insert_with(HashMap::new)
        .insert((context, key as usize), value as usize);
    CUDA_SUCCESS
}

unsafe extern "system" fn context_local_storage_delete(
    context: CUcontext,
    key: *mut c_void,
) -> CUresult {
    let context = match context_or_current(context) {
        Ok(context) => context,
        Err(error) => return error,
    };
    let mut storage = STORAGE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    storage
        .get_or_insert_with(HashMap::new)
        .remove(&(context, key as usize));
    CUDA_SUCCESS
}

unsafe extern "system" fn context_local_storage_get(
    value: *mut *mut c_void,
    context: CUcontext,
    key: *mut c_void,
) -> CUresult {
    if value.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let context = match context_or_current(context) {
        Ok(context) => context,
        Err(error) => return error,
    };
    let mut storage = STORAGE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    match storage
        .get_or_insert_with(HashMap::new)
        .get(&(context, key as usize))
    {
        Some(&stored) => {
            unsafe { *value = stored as *mut c_void };
            CUDA_SUCCESS
        }
        None => CUDA_ERROR_INVALID_HANDLE,
    }
}

// CTX_CREATE_BYPASS

unsafe extern "system" fn ctx_create_v2_bypass(
    context: *mut CUcontext,
    flags: c_uint,
    device: CUdevice,
) -> CUresult {
    unsafe { generated::cuCtxCreate_v2(context, flags, device) }
}

// HEAP_ACCESS: records the runtime allocates through the driver

#[repr(C)]
struct AllocRecord {
    /// Kept as the runtime gave it; only `y` is ever handed back
    #[allow(dead_code)]
    x: usize,
    y: usize,
}

unsafe extern "system" fn heap_alloc(
    record: *mut *const c_void,
    arg2: usize,
    arg3: usize,
) -> CUresult {
    if record.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    unsafe { *record = Box::into_raw(Box::new(AllocRecord { x: arg2, y: arg3 })).cast() };
    CUDA_SUCCESS
}

unsafe extern "system" fn heap_free(record: *const c_void, arg2: *mut usize) -> CUresult {
    if record.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let record = unsafe { Box::from_raw(record.cast_mut().cast::<AllocRecord>()) };
    if !arg2.is_null() {
        unsafe { *arg2 = record.y };
    }

    CUDA_SUCCESS
}

// DEVICE_EXTENDED_RT: ZLUDA leaves these unimplemented too

unsafe extern "system" fn device_get_attribute_ext(
    _device: CUdevice,
    _attribute: c_uint,
    _unknown: i32,
    _result: *mut [usize; 2],
) -> CUresult {
    log("the extended device attribute query is not supported yet");
    CUDA_ERROR_NOT_SUPPORTED
}

unsafe extern "system" fn device_get_something(
    _result: *mut c_uchar,
    _device: CUdevice,
) -> CUresult {
    log("the extended device query is not supported yet");
    CUDA_ERROR_NOT_SUPPORTED
}

// INTEGRITY_CHECK

fn device_hash_info() -> Result<Vec<DeviceHashInfo>, CUresult> {
    let check = |result: CUresult| {
        if result == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(result)
        }
    };

    let mut count = 0;
    check(unsafe { generated::cuDeviceGetCount(&mut count) })?;

    (0..count)
        .map(|ordinal| {
            let mut device = 0;
            check(unsafe { generated::cuDeviceGet(&mut device, ordinal) })?;

            let mut guid = CUuuid { bytes: [0; 16] };
            check(unsafe { generated::cuDeviceGetUuid_v2(&mut guid, device) })?;

            let attribute = |attribute| {
                let mut value = 0;
                check(unsafe { generated::cuDeviceGetAttribute(&mut value, attribute, device) })
                    .map(|()| value)
            };

            Ok(DeviceHashInfo {
                guid,
                pci_domain: attribute(CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID)?,
                pci_bus: attribute(CU_DEVICE_ATTRIBUTE_PCI_BUS_ID)?,
                pci_device: attribute(CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)?,
            })
        })
        .collect()
}

/// Hashes what the runtime hashes itself. The driver version must be the one this library
/// reports, which is the lender's.
unsafe extern "system" fn integrity_check(
    version: u32,
    unix_seconds: u64,
    result: *mut [u64; 2],
) -> CUresult {
    if result.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }

    let mut driver_version = 0;
    let status = unsafe { generated::cuDriverGetVersion(&mut driver_version) };
    if status != CUDA_SUCCESS {
        return status;
    }
    let devices = match device_hash_info() {
        Ok(devices) => devices,
        Err(error) => return error,
    };

    let hash = integrity::compute(
        version,
        unix_seconds,
        driver_version as u32,
        process::id(),
        thread_id(),
        INTEGRITY_CHECK.start(),
        CUDART_INTERFACE.start(),
        INTEGRITY_CHECK.0[1],
        &devices,
    );
    unsafe { *result = hash };
    CUDA_SUCCESS
}

/// The thread id the integrity check hashes: Windows' own, or pthread_self on Unix (as ZLUDA
/// does)
#[cfg(windows)]
fn thread_id() -> u32 {
    unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() }
}

#[cfg(unix)]
fn thread_id() -> u32 {
    unsafe { libc::pthread_self() as u32 }
}

// CONTEXT_CHECKS: bits that are never observably set

unsafe extern "system" fn context_check(
    _context: CUcontext,
    result1: *mut u32,
    _result2: *mut *const c_void,
) -> CUresult {
    if !result1.is_null() {
        unsafe { *result1 = 0 };
    }
    CUDA_SUCCESS
}

unsafe extern "system" fn check_fn3() -> u32 {
    0
}

// HYBRID_CUDART

const HYBRID_TOKEN: usize = 302100128;

/// Where the hybrid runtime may be: NECTAR_CUDA_HYBRID; the sandbox nectar-svc fills from
/// the lender while a GPU is borrowed (`%ProgramData%\Nectar\drivers\<key>\`), matching its
/// driver exactly; next to this library; System32; then the NVIDIA display driver's
/// DriverStore folders, newest first. It ships with NVIDIA's display driver only, not with
/// the CUDA toolkit, and only on Windows.
#[cfg(windows)]
fn hybrid_candidates(name: &str) -> Vec<std::path::PathBuf> {
    use std::path::PathBuf;

    let mut candidates = Vec::new();
    if let Some(path) = env::var_os("NECTAR_CUDA_HYBRID") {
        candidates.push(PathBuf::from(path));
    }
    if let Some(config) = nectar_lending::BorrowConfig::read()
        && !config.driver.is_empty()
    {
        candidates.push(nectar_lending::driver_folder(&config.driver).join(name));
    }

    // This library's own folder
    let mut module = ptr::null_mut();
    let mut path = [0u16; 1024];
    unsafe {
        use windows_sys::Win32::System::LibraryLoader::{
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            GetModuleFileNameW, GetModuleHandleExW,
        };
        if GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            hybrid_candidates as *const u16,
            &mut module,
        ) != 0
        {
            let length = GetModuleFileNameW(module, path.as_mut_ptr(), path.len() as u32) as usize;
            if let Some(folder) = PathBuf::from(String::from_utf16_lossy(&path[..length])).parent()
            {
                candidates.push(folder.join(name));
            }
        }
    }

    let system = env::var_os("SystemRoot")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Windows"));
    candidates.push(system.join("System32").join(name));

    let mut stored: Vec<(std::time::SystemTime, PathBuf)> =
        std::fs::read_dir(system.join(r"System32\DriverStore\FileRepository"))
            .into_iter()
            .flatten()
            .flatten()
            .filter(|entry| entry.file_name().to_string_lossy().starts_with("nv"))
            .map(|entry| entry.path().join(name))
            .filter_map(|path| Some((std::fs::metadata(&path).ok()?.modified().ok()?, path)))
            .collect();
    stored.sort();
    candidates.extend(stored.into_iter().rev().map(|(_, path)| path));
    candidates
}

/// The first hybrid runtime that loads, from hybrid_candidates
#[cfg(windows)]
fn load_hybrid(name: &str) -> Option<libloading::Library> {
    // Its own dependencies are found next to it
    const LOAD_WITH_ALTERED_SEARCH_PATH: u32 = 0x8;
    hybrid_candidates(name)
        .into_iter()
        .filter(|path| path.is_file())
        .find_map(|path| {
            unsafe { libloading::os::windows::Library::load_with_flags(&path, LOAD_WITH_ALTERED_SEARCH_PATH) }
                .ok()
        })
        .map(Into::into)
}

/// Only Windows' CUDA runtime asks for one
#[cfg(not(windows))]
fn load_hybrid(_name: &str) -> Option<libloading::Library> {
    None
}

/// The static CUDA runtime on Windows hands much of its work to the driver's own runtime,
/// nvcudart_hybrid64.dll, through this. NECTAR_CUDA_NO_HYBRID=1 tests what happens without it.
unsafe extern "system" fn hybrid_runtime_load_get_proc_address(
    name: *const c_char,
    function: *mut *const c_void,
    token: *mut usize,
) -> CUresult {
    static HYBRID: Mutex<Option<libloading::Library>> = Mutex::new(None);

    if name.is_null() || function.is_null() || token.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let name = match unsafe { CStr::from_ptr(name) }.to_str() {
        Ok(name @ ("nvcudart_hybrid64.dll" | "nvcudart_hybrid64a.dll")) => name,
        _ => return CUDA_ERROR_INVALID_VALUE,
    };
    if env::var_os("NECTAR_CUDA_NO_HYBRID").is_some() {
        return CUDA_ERROR_FILE_NOT_FOUND;
    }

    let mut hybrid = HYBRID
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if hybrid.is_none() {
        match load_hybrid(name) {
            Some(library) => *hybrid = Some(library),
            None => {
                log(&format!(
                    "{name} was not found; it comes with NVIDIA's display driver"
                ));
                return CUDA_ERROR_FILE_NOT_FOUND;
            }
        }
    }

    let library = hybrid.as_ref().unwrap();
    match unsafe { library.get::<*const c_void>(b"__cudaGetProcAddress\0") } {
        Ok(address) => {
            unsafe {
                *function = *address;
                *token = HYBRID_TOKEN;
            }
            CUDA_SUCCESS
        }
        Err(_) => CUDA_ERROR_OPERATING_SYSTEM,
    }
}

unsafe extern "system" fn hybrid_runtime_free(token: usize) -> CUresult {
    if token == HYBRID_TOKEN {
        CUDA_SUCCESS
    } else {
        CUDA_ERROR_INVALID_HANDLE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uuids_read_as_cuda_reports_them() {
        assert_eq!(
            uuid("DD9E98A4-F410-4844-99B0-E2AEB7B432B5"),
            [
                0xdd, 0x9e, 0x98, 0xa4, 0xf4, 0x10, 0x48, 0x44, 0x99, 0xb0, 0xe2, 0xae, 0xb7, 0xb4,
                0x32, 0xb5
            ]
        );
    }

    #[test]
    fn tables_lead_with_their_size() {
        let table = export_table(&uuid("D4082055-BDE6-704B-8D34-BA123C66E1F2")).unwrap();
        assert_eq!(
            unsafe { *table.cast::<usize>() },
            3 * mem::size_of::<usize>()
        );
        assert!(export_table(&[0; 16]).is_none());
    }
}
