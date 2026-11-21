//! nectar-svc: the borrower's service. It finds lenders, borrows the one the user picks in
//! nectar.exe, keeps the lease alive, and writes `%ProgramData%\Nectar\borrow.cfg` while a
//! GPU is borrowed, which Nectar's nvcuda.dll in System32 reads. It runs as LocalSystem, so
//! nectar.exe needs no administrator rights.
//!
//! The shared secret is `%ProgramData%\Nectar\secret`, which the installer writes readable by
//! SYSTEM and Administrators only.
//!
//! Usage:
//!   nectar-svc run       as the Windows service (the installer registers it so)
//!   nectar-svc console   the same in a console, for testing; NECTAR_HOME moves its folder

#![allow(clippy::missing_safety_doc)]

mod borrow;
mod engine;
mod pipe;

use std::fs::OpenOptions;
use std::io::Write;
use std::sync::{Arc, Condvar, Mutex};
use std::{env, fs, process, ptr, thread};

use windows_sys::Win32::System::Services::{
    RegisterServiceCtrlHandlerExW, SERVICE_ACCEPT_SHUTDOWN, SERVICE_ACCEPT_STOP,
    SERVICE_CONTROL_INTERROGATE, SERVICE_CONTROL_SHUTDOWN, SERVICE_CONTROL_STOP, SERVICE_RUNNING,
    SERVICE_STATUS, SERVICE_STATUS_HANDLE, SERVICE_STOP_PENDING, SERVICE_STOPPED,
    SERVICE_TABLE_ENTRYW, SERVICE_WIN32_OWN_PROCESS, SetServiceStatus,
    StartServiceCtrlDispatcherW,
};

use engine::Engine;

const SERVICE_NAME: &str = "NectarBorrow";

/// Logs to `%ProgramData%\Nectar\svc.log`, and to the console in console mode
pub fn log(message: &str) {
    let line = format!(
        "{} {message}\n",
        chrono_like_now()
    );
    eprint!("{line}");
    let path = nectar_lending::home().join("svc.log");
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = file.write_all(line.as_bytes());
    }
}

/// The time of day in UTC, enough to line up with the lender's log
fn chrono_like_now() -> String {
    let seconds = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| since.as_secs());
    format!(
        "{:02}:{:02}:{:02}Z",
        seconds / 3600 % 24,
        seconds / 60 % 60,
        seconds % 60
    )
}

fn secret() -> Vec<u8> {
    let path = nectar_lending::home().join("secret");
    match fs::read_to_string(&path) {
        Ok(secret) if !secret.trim().is_empty() => secret.trim().as_bytes().to_vec(),
        _ => {
            log(&format!(
                "no shared secret in {}: install with Install-Borrower.ps1 -Secret",
                path.display()
            ));
            process::exit(1);
        }
    }
}

fn main() {
    let _ = fs::create_dir_all(nectar_lending::home());
    match env::args().nth(1).as_deref() {
        Some("console") => {
            let engine = Engine::new(secret());
            let lending = engine.start();
            log(&format!("borrowing service in a console; the app talks to {}", pipe::NAME));
            let served = Arc::clone(&engine);
            thread::spawn(move || pipe::serve(served));
            let _ = lending.join();
        }
        Some("run") => {
            let mut name = wide(SERVICE_NAME);
            let table = [
                SERVICE_TABLE_ENTRYW {
                    lpServiceName: name.as_mut_ptr(),
                    lpServiceProc: Some(service_main),
                },
                SERVICE_TABLE_ENTRYW {
                    lpServiceName: ptr::null_mut(),
                    lpServiceProc: None,
                },
            ];
            if unsafe { StartServiceCtrlDispatcherW(table.as_ptr()) } == 0 {
                eprintln!("nectar-svc run is for the Windows service; use nectar-svc console to test");
                process::exit(1);
            }
        }
        _ => {
            eprintln!("usage: nectar-svc run | console");
            process::exit(2);
        }
    }
}

fn wide(text: &str) -> Vec<u16> {
    text.encode_utf16().chain(Some(0)).collect()
}

/// Set when Windows asks the service to stop
static STOP: Mutex<bool> = Mutex::new(false);
static STOPPING: Condvar = Condvar::new();

unsafe extern "system" fn handler(
    control: u32,
    _event: u32,
    _data: *mut std::ffi::c_void,
    _context: *mut std::ffi::c_void,
) -> u32 {
    match control {
        SERVICE_CONTROL_STOP | SERVICE_CONTROL_SHUTDOWN => {
            *STOP.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = true;
            STOPPING.notify_all();
            0
        }
        SERVICE_CONTROL_INTERROGATE => 0,
        // ERROR_CALL_NOT_IMPLEMENTED
        _ => 120,
    }
}

fn report(handle: SERVICE_STATUS_HANDLE, state: u32, accepts: u32) {
    let status = SERVICE_STATUS {
        dwServiceType: SERVICE_WIN32_OWN_PROCESS,
        dwCurrentState: state,
        dwControlsAccepted: accepts,
        dwWin32ExitCode: 0,
        dwServiceSpecificExitCode: 0,
        dwCheckPoint: 0,
        dwWaitHint: 5000,
    };
    unsafe { SetServiceStatus(handle, &status) };
}

unsafe extern "system" fn service_main(_argc: u32, _argv: *mut windows_sys::core::PWSTR) {
    let name = wide(SERVICE_NAME);
    let handle = unsafe { RegisterServiceCtrlHandlerExW(name.as_ptr(), Some(handler), ptr::null()) };
    if handle.is_null() {
        return;
    }
    report(handle, SERVICE_RUNNING, SERVICE_ACCEPT_STOP | SERVICE_ACCEPT_SHUTDOWN);

    let engine = Engine::new(secret());
    let lending = engine.start();
    let served = Arc::clone(&engine);
    thread::spawn(move || pipe::serve(served));
    log("borrowing service started");

    let mut stop = STOP.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    while !*stop {
        stop = STOPPING.wait(stop).unwrap_or_else(|poisoned| poisoned.into_inner());
    }
    drop(stop);

    report(handle, SERVICE_STOP_PENDING, 0);
    // Returns the GPU and removes borrow.cfg
    engine.stop();
    let _ = lending.join();
    log("borrowing service stopped");
    report(handle, SERVICE_STOPPED, 0);
}
