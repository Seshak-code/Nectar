//! nectar-svc: the borrower's service. It finds lenders, borrows the one the user picks in the
//! Nectar app, keeps the lease alive, and writes `borrow.cfg` in Nectar's folder while a GPU
//! is borrowed, which Nectar's CUDA driver reads (nvcuda.dll in System32, libcuda.so.1 on
//! Linux). It runs as LocalSystem or root, so the app needs no administrator rights.
//!
//! The shared secret is `secret` in Nectar's folder (`%ProgramData%\Nectar`,
//! `/var/lib/nectar`), which the installer writes readable by administrators (root) only.
//!
//! Usage:
//!   nectar-svc run       as the service: the Windows service, or under systemd
//!   nectar-svc console   in a console, for testing; NECTAR_HOME moves its folder
//!   nectar-svc ask <command>   asks the running service, as the app does, and prints the answer

#![allow(clippy::missing_safety_doc)]

mod borrow;
mod engine;
#[cfg(windows)]
mod pipe;
#[cfg(unix)]
mod socket;

use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Read, Write};
use std::sync::Arc;
use std::{env, fs, process, thread};

#[cfg(windows)]
use pipe as ipc;
#[cfg(unix)]
use socket as ipc;

use engine::Engine;

#[cfg(windows)]
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
                "no shared secret in {}: install Nectar with its installer and a secret",
                path.display()
            ));
            process::exit(1);
        }
    }
}

/// Answers one client of the pipe or socket: one command per line
pub fn answer(engine: &Engine, reader: impl Read, mut writer: impl Write) {
    for line in BufReader::new(reader).lines() {
        let Ok(line) = line else { return };
        let reply = engine.command(&line);
        if writer.write_all(format!("{reply}\n").as_bytes()).is_err() {
            return;
        }
    }
}

fn main() {
    let _ = fs::create_dir_all(nectar_lending::home());
    match env::args().nth(1).as_deref() {
        Some("console") => {
            let engine = Engine::new(secret());
            let lending = engine.start();
            log(&format!("borrowing service in a console; the app talks to {}", ipc::name()));
            let served = Arc::clone(&engine);
            thread::spawn(move || ipc::serve(served));
            let _ = lending.join();
        }
        Some("run") => run(),
        Some("ask") => ask(&env::args().skip(2).collect::<Vec<_>>().join(" ")),
        _ => {
            eprintln!("usage: nectar-svc run | console | ask <command>");
            process::exit(2);
        }
    }
}

/// Asks the running service, for scripts: the commands are status, borrow <lender>, return,
/// add <address>, profile <gaming or ai> and check
fn ask(command: &str) {
    let address = nectar_lending::service_address();
    #[cfg(windows)]
    let stream = OpenOptions::new().read(true).write(true).open(&address);
    #[cfg(unix)]
    let stream = std::os::unix::net::UnixStream::connect(&address);
    let answered = stream.and_then(|mut stream| {
        stream.write_all(format!("{command}\n").as_bytes())?;
        for line in BufReader::new(stream).lines() {
            let line = line?;
            if line.is_empty() {
                break;
            }
            println!("{line}");
        }
        Ok(())
    });
    if let Err(error) = answered {
        eprintln!("cannot ask the service at {}: {error}", address.display());
        process::exit(1);
    }
}

/// Under systemd: runs until SIGTERM (or SIGINT), then returns the GPU and removes borrow.cfg
#[cfg(unix)]
fn run() {
    // Blocked here, before any thread starts, so every thread inherits the mask and only the
    // sigwait below receives them
    let signals = unsafe {
        let mut set: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut set);
        libc::sigaddset(&mut set, libc::SIGTERM);
        libc::sigaddset(&mut set, libc::SIGINT);
        libc::pthread_sigmask(libc::SIG_BLOCK, &set, std::ptr::null_mut());
        set
    };
    let engine = Engine::new(secret());
    let lending = engine.start();
    let served = Arc::clone(&engine);
    thread::spawn(move || ipc::serve(served));
    log(&format!("borrowing service started; the app talks to {}", ipc::name()));

    let mut signal = 0;
    unsafe { libc::sigwait(&signals, &mut signal) };
    engine.stop();
    let _ = lending.join();
    let _ = fs::remove_file(nectar_lending::service_address());
    log("borrowing service stopped");
}

#[cfg(windows)]
fn run() {
    use windows_service::*;
    let mut name = wide(SERVICE_NAME);
    let table = [
        SERVICE_TABLE_ENTRYW {
            lpServiceName: name.as_mut_ptr(),
            lpServiceProc: Some(service_main),
        },
        SERVICE_TABLE_ENTRYW {
            lpServiceName: std::ptr::null_mut(),
            lpServiceProc: None,
        },
    ];
    if unsafe { StartServiceCtrlDispatcherW(table.as_ptr()) } == 0 {
        eprintln!("nectar-svc run is for the Windows service; use nectar-svc console to test");
        process::exit(1);
    }
}

/// The Windows service: Service Control Manager callbacks
#[cfg(windows)]
mod windows_service {
    use std::sync::{Arc, Condvar, Mutex};
    use std::{ptr, thread};

    pub use windows_sys::Win32::System::Services::{
        RegisterServiceCtrlHandlerExW, SERVICE_ACCEPT_SHUTDOWN, SERVICE_ACCEPT_STOP,
        SERVICE_CONTROL_INTERROGATE, SERVICE_CONTROL_SHUTDOWN, SERVICE_CONTROL_STOP, SERVICE_RUNNING,
        SERVICE_STATUS, SERVICE_STATUS_HANDLE, SERVICE_STOP_PENDING, SERVICE_STOPPED,
        SERVICE_TABLE_ENTRYW, SERVICE_WIN32_OWN_PROCESS, SetServiceStatus,
        StartServiceCtrlDispatcherW,
    };

    use crate::engine::Engine;
    use crate::{SERVICE_NAME, ipc, log, secret};

    pub fn wide(text: &str) -> Vec<u16> {
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

    pub unsafe extern "system" fn service_main(_argc: u32, _argv: *mut windows_sys::core::PWSTR) {
        let name = wide(SERVICE_NAME);
        let handle = unsafe { RegisterServiceCtrlHandlerExW(name.as_ptr(), Some(handler), ptr::null()) };
        if handle.is_null() {
            return;
        }
        report(handle, SERVICE_RUNNING, SERVICE_ACCEPT_STOP | SERVICE_ACCEPT_SHUTDOWN);

        let engine = Engine::new(secret());
        let lending = engine.start();
        let served = Arc::clone(&engine);
        thread::spawn(move || ipc::serve(served));
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
}
