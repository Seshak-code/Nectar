//! The Unix socket the `nectar` command talks to the service over (/run/nectar.sock), with the
//! same one-line requests and key=value answers as the Windows pipe (pipe.rs).
//!
//! The service runs as root. The socket is open to root and to the `nectar` group, which the
//! installer creates for the people allowed to borrow; without that group, to root only.

use std::ffi::CString;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixListener;
use std::sync::Arc;
use std::thread;

use crate::engine::Engine;
use crate::log;

pub fn name() -> String {
    nectar_lending::service_address().display().to_string()
}

/// The `nectar` group's id, if it exists
fn nectar_group() -> Option<libc::gid_t> {
    let name = CString::new("nectar").ok()?;
    let group = unsafe { libc::getgrnam(name.as_ptr()) };
    (!group.is_null()).then(|| unsafe { (*group).gr_gid })
}

/// Serves the socket until the process ends, each client on a thread of its own
pub fn serve(engine: Arc<Engine>) {
    let path = nectar_lending::service_address();
    // Left by a service that didn't stop cleanly
    let _ = fs::remove_file(&path);
    let listener = match UnixListener::bind(&path) {
        Ok(listener) => listener,
        Err(error) => {
            log(&format!("cannot listen on {}: {error}", path.display()));
            return;
        }
    };
    let mode = match nectar_group() {
        Some(group) => {
            let c_path = CString::new(path.as_os_str().as_encoded_bytes()).unwrap_or_default();
            unsafe { libc::chown(c_path.as_ptr(), u32::MAX, group) };
            0o660
        }
        None => {
            log("no nectar group: only root can use the service");
            0o600
        }
    };
    if let Err(error) = fs::set_permissions(&path, fs::Permissions::from_mode(mode)) {
        log(&format!("cannot restrict {}: {error}", path.display()));
        return;
    }
    for stream in listener.incoming() {
        let Ok(stream) = stream else { continue };
        let Ok(writer) = stream.try_clone() else { continue };
        let engine = Arc::clone(&engine);
        thread::spawn(move || crate::answer(&engine, stream, writer));
    }
}
