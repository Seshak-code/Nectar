//! Pass-through nvcuda.dll that records every CUDA driver call an application makes: what
//! Nectar must support, and the call patterns worth batching or replaying.
//!
//! The build makes nvcudatrace.dll; copy it as nvcuda.dll next to the application. Every
//! call goes on to the real driver in System32. When the process ends, the report is
//! written to the folder in NECTAR_CUDA_TRACE, or %TEMP%, as nectar-cuda-trace-<pid>.txt.
//!
//! # Safety
//!
//! Every export has the contract cuda.h documents for it.

//! Windows only, for now.

#![cfg(windows)]
#![allow(non_snake_case, clippy::missing_safety_doc)]

mod generated;
mod tables;

use std::ffi::{CStr, c_char, c_int, c_void};
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;
use std::{env, fs, process};

use libloading::Library;
use nectar_cuda_protocol::{self as protocol, NAMES};
use nectar_cuda_sys::*;

/// How many calls are kept in order
const SEQUENCE_LIMIT: usize = 20_000;

struct Real {
    _library: Library,
    /// Address of each export, by its index in NAMES (0 when the driver lacks it)
    addresses: Vec<usize>,
}

fn driver() -> Option<&'static Real> {
    static REAL: OnceLock<Option<Real>> = OnceLock::new();

    REAL.get_or_init(|| {
        let system = env::var_os("SystemRoot")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(r"C:\Windows"));
        // Past the redirect, which would answer with the tracer itself
        let module = nectar_cuda_redirect::load_real(&system.join(r"System32\nvcuda.dll")).ok()?;
        let library: Library =
            unsafe { libloading::os::windows::Library::from_raw(module as isize) }.into();

        let mut addresses = vec![0; NAMES.len()];
        for (index, name) in NAMES.iter().enumerate() {
            let symbol = format!("{name}\0");
            // A symbol typed as a pointer dereferences to its own address
            if let Ok(address) = unsafe { library.get::<*const c_void>(symbol.as_bytes()) } {
                addresses[index] = *address as usize;
            }
        }

        Some(Real {
            _library: library,
            addresses,
        })
    })
    .as_ref()
}

/// The real driver's function for the export at `index` in NAMES
pub(crate) fn real(index: usize) -> Option<*const c_void> {
    driver()?
        .addresses
        .get(index)
        .copied()
        .filter(|&address| address != 0)
        .map(|address| address as *const c_void)
}

fn index_of(name: &str) -> Option<usize> {
    NAMES.binary_search(&name).ok()
}

#[derive(Clone, Copy, Default)]
struct Stat {
    calls: u64,
    errors: u64,
    nanos: u64,
}

struct Trace {
    started: Instant,
    stats: Vec<Stat>,
    sequence: Vec<(u32, CUresult)>,
    /// symbol, CUDA version, flags, the export handed out
    lookups: Vec<(String, c_int, cuuint64_t, String)>,
    export_tables: Vec<String>,
}

static TRACE: Mutex<Option<Trace>> = Mutex::new(None);

fn with_trace(update: impl FnOnce(&mut Trace)) {
    let mut trace = TRACE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let trace = trace.get_or_insert_with(|| Trace {
        started: Instant::now(),
        stats: vec![Stat::default(); NAMES.len()],
        sequence: Vec::new(),
        lookups: Vec::new(),
        export_tables: Vec::new(),
    });
    update(trace);
}

pub(crate) fn record(index: usize, started: Instant, result: CUresult) {
    let nanos = started.elapsed().as_nanos() as u64;
    with_trace(|trace| {
        let stat = &mut trace.stats[index];
        stat.calls += 1;
        stat.nanos += nanos;
        if result != CUDA_SUCCESS {
            stat.errors += 1;
        }
        if trace.sequence.len() < SEQUENCE_LIMIT {
            trace.sequence.push((index as u32, result));
        }
    });
}

/// Hands out this library's wrapper in place of what the driver returned, so calls made
/// through the pointer are traced too. The driver returns internal addresses, not its
/// exports, so the export is picked from cudaTypedefs.h as Nectar's client does.
unsafe fn substitute(
    symbol: *const c_char,
    pfn: *mut *mut c_void,
    version: c_int,
    flags: cuuint64_t,
) {
    let symbol = unsafe { CStr::from_ptr(symbol) }
        .to_string_lossy()
        .into_owned();
    let export = protocol::proc_address(&symbol, version, protocol::per_thread_stream(flags)).ok();

    let handed_out = match export.and_then(|export| generated::exports().get(export)) {
        Some(&wrapper) => {
            unsafe { *pfn = wrapper.cast_mut() };
            export.unwrap().to_string()
        }
        None => "(the driver's own, untraced)".to_string(),
    };
    with_trace(|trace| trace.lookups.push((symbol, version, flags, handed_out)));
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGetProcAddress_v2(
    symbol: *const c_char,
    pfn: *mut *mut c_void,
    cudaVersion: c_int,
    flags: cuuint64_t,
    symbolStatus: *mut CUdriverProcAddressQueryResult,
) -> CUresult {
    type GetProcAddress = unsafe extern "system" fn(
        *const c_char,
        *mut *mut c_void,
        c_int,
        cuuint64_t,
        *mut CUdriverProcAddressQueryResult,
    ) -> CUresult;

    let index = index_of("cuGetProcAddress_v2").unwrap();
    let Some(address) = real(index) else {
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    let call: GetProcAddress = unsafe { std::mem::transmute(address) };

    let started = Instant::now();
    let result = unsafe { call(symbol, pfn, cudaVersion, flags, symbolStatus) };
    record(index, started, result);

    if result == CUDA_SUCCESS && !symbol.is_null() && !pfn.is_null() {
        unsafe { substitute(symbol, pfn, cudaVersion, flags) };
    }
    result
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGetProcAddress(
    symbol: *const c_char,
    pfn: *mut *mut c_void,
    cudaVersion: c_int,
    flags: cuuint64_t,
) -> CUresult {
    type GetProcAddress =
        unsafe extern "system" fn(*const c_char, *mut *mut c_void, c_int, cuuint64_t) -> CUresult;

    let index = index_of("cuGetProcAddress").unwrap();
    let Some(address) = real(index) else {
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    let call: GetProcAddress = unsafe { std::mem::transmute(address) };

    let started = Instant::now();
    let result = unsafe { call(symbol, pfn, cudaVersion, flags) };
    record(index, started, result);

    if result == CUDA_SUCCESS && !symbol.is_null() && !pfn.is_null() {
        unsafe { substitute(symbol, pfn, cudaVersion, flags) };
    }
    result
}

/// Which undocumented tables are asked for is recorded, and the chosen ones are handed out
/// with their entries traced (tables.rs)
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGetExportTable(
    table: *mut *const c_void,
    table_id: *const CUuuid,
) -> CUresult {
    type GetExportTable = unsafe extern "system" fn(*mut *const c_void, *const CUuuid) -> CUresult;

    let index = index_of("cuGetExportTable").unwrap();
    let Some(address) = real(index) else {
        return CUDA_ERROR_NOT_SUPPORTED;
    };
    let call: GetExportTable = unsafe { std::mem::transmute(address) };

    let started = Instant::now();
    let result = unsafe { call(table, table_id) };
    record(index, started, result);

    if !table_id.is_null() {
        let uuid = unsafe { (*table_id).bytes }.map(|byte| byte as u8);
        with_trace(|trace| {
            trace
                .export_tables
                .push(format!("{} -> {result}", tables::hex(&uuid)))
        });
        if result == CUDA_SUCCESS && !table.is_null() && unsafe { !(*table).is_null() } {
            unsafe { *table = tables::wrap(&uuid, *table) };
        }
    }
    result
}

fn report(trace: &Trace) -> String {
    let mut out = String::new();
    let program = env::current_exe()
        .map(|path| path.display().to_string())
        .unwrap_or_default();
    let _ = writeln!(
        out,
        "Nectar CUDA trace of {program}, process {}, {:.1} s",
        process::id(),
        trace.started.elapsed().as_secs_f64()
    );

    let mut called: Vec<(usize, Stat)> = trace
        .stats
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, stat)| stat.calls > 0)
        .collect();
    called.sort_by_key(|(_, stat)| std::cmp::Reverse(stat.calls));
    let total: u64 = called.iter().map(|(_, stat)| stat.calls).sum();
    let _ = writeln!(out, "\n{} functions, {total} calls\n", called.len());
    let _ = writeln!(
        out,
        "{:>10} {:>8} {:>12} {:>10}  function",
        "calls", "errors", "total ms", "avg us"
    );
    for (index, stat) in &called {
        let _ = writeln!(
            out,
            "{:>10} {:>8} {:>12.3} {:>10.2}  {}",
            stat.calls,
            stat.errors,
            stat.nanos as f64 / 1e6,
            stat.nanos as f64 / 1e3 / stat.calls as f64,
            NAMES[*index]
        );
    }

    let _ = writeln!(out, "\nExport tables asked for (cuGetExportTable):");
    for table in &trace.export_tables {
        let _ = writeln!(out, "  {table}");
    }

    let _ = writeln!(
        out,
        "\nSymbols looked up (cuGetProcAddress), symbol version flags -> export:"
    );
    for (symbol, version, flags, export) in &trace.lookups {
        let _ = writeln!(out, "  {symbol} {version} {flags} -> {export}");
    }

    tables::report(&mut out);

    // Runs of the same call collapse to one line
    let _ = writeln!(out, "\nFirst {} calls in order:", trace.sequence.len());
    let mut index = 0;
    while index < trace.sequence.len() {
        let current = trace.sequence[index];
        let mut run = 1;
        while index + run < trace.sequence.len() && trace.sequence[index + run] == current {
            run += 1;
        }
        let suffix = if run > 1 {
            format!(" x{run}")
        } else {
            String::new()
        };
        let _ = writeln!(
            out,
            "  {} = {}{suffix}",
            NAMES[current.0 as usize], current.1
        );
        index += run;
    }
    out
}

fn write_report() {
    // The process is ending, possibly with other threads stopped mid-call: never wait
    let Ok(trace) = TRACE.try_lock() else { return };
    let Some(trace) = trace.as_ref() else { return };

    let folder = env::var_os("NECTAR_CUDA_TRACE")
        .map(PathBuf::from)
        .unwrap_or_else(env::temp_dir);
    let path = folder.join(format!("nectar-cuda-trace-{}.txt", process::id()));
    let _ = fs::write(path, report(trace));
}

#[unsafe(no_mangle)]
extern "system" fn DllMain(module: *mut c_void, reason: u32, _reserved: *mut c_void) -> i32 {
    const DLL_PROCESS_DETACH: u32 = 0;
    const DLL_PROCESS_ATTACH: u32 = 1;
    match reason {
        DLL_PROCESS_ATTACH => {
            if let Err(error) = unsafe { nectar_cuda_redirect::install(module, false) } {
                eprintln!("nectar-cuda-trace: cannot redirect loads of nvcuda.dll ({error})");
            }
        }
        DLL_PROCESS_DETACH => write_report(),
        _ => {}
    }
    1
}
