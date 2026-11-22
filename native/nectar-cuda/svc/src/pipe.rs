//! `\\.\pipe\nectar`: how nectar.exe talks to the service on Windows. A request is one line
//! (status, borrow <lender>, return, add <address>); the answer is key=value lines and a blank
//! line.
//!
//! The service runs as LocalSystem, so the pipe lets the people signed in to this PC read
//! and write it, and nobody from the network.

use std::fs::File;
use std::os::windows::io::FromRawHandle;
use std::sync::Arc;
use std::{ptr, thread};

use windows_sys::Win32::Foundation::{ERROR_PIPE_CONNECTED, GetLastError, INVALID_HANDLE_VALUE};
use windows_sys::Win32::Security::Authorization::{
    ConvertStringSecurityDescriptorToSecurityDescriptorW, SDDL_REVISION_1,
};
use windows_sys::Win32::Security::{PSECURITY_DESCRIPTOR, SECURITY_ATTRIBUTES};
use windows_sys::Win32::Storage::FileSystem::PIPE_ACCESS_DUPLEX;
use windows_sys::Win32::System::Pipes::{
    ConnectNamedPipe, CreateNamedPipeW, PIPE_READMODE_BYTE, PIPE_REJECT_REMOTE_CLIENTS,
    PIPE_TYPE_BYTE, PIPE_UNLIMITED_INSTANCES, PIPE_WAIT,
};

use crate::engine::Engine;
use crate::log;

pub fn name() -> String {
    nectar_lending::service_address().display().to_string()
}

fn wide(text: &str) -> Vec<u16> {
    text.encode_utf16().chain(Some(0)).collect()
}

/// Serves the pipe until the process ends, each client on a thread of its own
pub fn serve(engine: Arc<Engine>) {
    // SYSTEM and Administrators: all access; interactive users: read and write
    let mut descriptor: PSECURITY_DESCRIPTOR = ptr::null_mut();
    let sddl = wide("D:(A;;GA;;;SY)(A;;GA;;;BA)(A;;GRGW;;;IU)");
    let ok = unsafe {
        ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl.as_ptr(),
            SDDL_REVISION_1,
            &mut descriptor,
            ptr::null_mut(),
        )
    };
    if ok == 0 {
        log("cannot set up the pipe's security");
        return;
    }
    let attributes = SECURITY_ATTRIBUTES {
        nLength: size_of::<SECURITY_ATTRIBUTES>() as u32,
        lpSecurityDescriptor: descriptor,
        bInheritHandle: 0,
    };
    let pipe_name = wide(&name());

    loop {
        let handle = unsafe {
            CreateNamedPipeW(
                pipe_name.as_ptr(),
                PIPE_ACCESS_DUPLEX,
                PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
                PIPE_UNLIMITED_INSTANCES,
                4096,
                4096,
                0,
                &attributes,
            )
        };
        if handle == INVALID_HANDLE_VALUE {
            log(&format!("cannot create {}: error {}", name(), unsafe { GetLastError() }));
            return;
        }
        let connected = unsafe { ConnectNamedPipe(handle, ptr::null_mut()) } != 0
            || unsafe { GetLastError() } == ERROR_PIPE_CONNECTED;
        let pipe = unsafe { File::from_raw_handle(handle) };
        if !connected {
            continue;
        }
        let engine = Arc::clone(&engine);
        thread::spawn(move || answer(&engine, pipe));
    }
}

fn answer(engine: &Engine, pipe: File) {
    if let Ok(writer) = pipe.try_clone() {
        crate::answer(engine, pipe, writer);
    }
}
