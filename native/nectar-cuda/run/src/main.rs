//! nectar-cuda-run: starts a program with Nectar's nvcuda.dll in place of the real driver.
//!
//! The CUDA runtime loads nvcuda.dll from System32 only, so a copy next to the program is
//! never used. But Windows hands back an already loaded DLL of the same name, whatever the
//! search flags. So the program starts suspended, this loads Nectar's nvcuda.dll into it,
//! and only then does the program run.
//!
//! Processes the program starts get the DLL too, through a job object, since launchers such
//! as the Store's python.exe, venvs and py.exe run the real program as a child.
//!
//! Usage: nectar-cuda-run [--dll <nvcuda.dll>] <program> [arguments...]
//! Without --dll it uses the nvcuda.dll next to this launcher. The tracer works the same
//! way: pass its DLL with --dll.

use std::ffi::OsStr;
use std::os::windows::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::{env, mem, process, ptr};

use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE, WAIT_OBJECT_0};
use windows_sys::Win32::System::Console::{
    GetStdHandle, STD_ERROR_HANDLE, STD_INPUT_HANDLE, STD_OUTPUT_HANDLE,
};
use windows_sys::Win32::System::Diagnostics::Debug::WriteProcessMemory;
use windows_sys::Win32::System::IO::{CreateIoCompletionPort, GetQueuedCompletionStatus};
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JOBOBJECT_ASSOCIATE_COMPLETION_PORT,
    JobObjectAssociateCompletionPortInformation, SetInformationJobObject,
};
use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress};
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAllocEx, VirtualFreeEx,
};
use windows_sys::Win32::System::Threading::{
    CREATE_SUSPENDED, CreateProcessW, CreateRemoteThread, GetExitCodeProcess, GetExitCodeThread,
    INFINITE, IsWow64Process, OpenProcess, PROCESS_CREATE_THREAD, PROCESS_INFORMATION,
    PROCESS_QUERY_INFORMATION, PROCESS_VM_OPERATION, PROCESS_VM_READ, PROCESS_VM_WRITE,
    ResumeThread, STARTF_USESTDHANDLES, STARTUPINFOW, TerminateProcess, WaitForSingleObject,
};

/// Job notifications (winnt.h)
const JOB_OBJECT_MSG_ACTIVE_PROCESS_ZERO: u32 = 4;
const JOB_OBJECT_MSG_NEW_PROCESS: u32 = 6;

fn wide(text: &OsStr) -> Vec<u16> {
    text.encode_wide().chain(Some(0)).collect()
}

/// Quotes one argument the way the C runtime splits a command line
fn quote(argument: &str) -> String {
    if !argument.is_empty() && !argument.contains([' ', '\t', '"']) {
        return argument.to_string();
    }

    let mut quoted = String::from("\"");
    let mut backslashes = 0;
    for c in argument.chars() {
        match c {
            '\\' => backslashes += 1,
            '"' => {
                quoted.extend(std::iter::repeat_n('\\', backslashes * 2 + 1));
                quoted.push('"');
                backslashes = 0;
            }
            _ => {
                quoted.extend(std::iter::repeat_n('\\', backslashes));
                quoted.push(c);
                backslashes = 0;
            }
        }
    }
    quoted.extend(std::iter::repeat_n('\\', backslashes * 2));
    quoted.push('"');
    quoted
}

fn fail(message: &str) -> ! {
    eprintln!("nectar-cuda-run: {message}");
    process::exit(1);
}

/// Loads `dll` into the suspended process by running LoadLibraryW on a thread of its own
unsafe fn load_into(process: HANDLE, dll: &Path) -> Result<(), String> {
    let path = wide(dll.as_os_str());
    let size = path.len() * mem::size_of::<u16>();

    unsafe {
        let remote = VirtualAllocEx(
            process,
            ptr::null(),
            size,
            MEM_COMMIT | MEM_RESERVE,
            PAGE_READWRITE,
        );
        if remote.is_null() {
            return Err("cannot allocate memory in the program".into());
        }

        let result = (|| {
            if WriteProcessMemory(process, remote, path.as_ptr().cast(), size, ptr::null_mut()) == 0
            {
                return Err("cannot write to the program".to_string());
            }

            // kernel32 sits at the same address in every process of this boot
            let kernel32 = GetModuleHandleW(wide(OsStr::new("kernel32.dll")).as_ptr());
            let load_library = GetProcAddress(kernel32, c"LoadLibraryW".as_ptr().cast())
                .ok_or("no LoadLibraryW")?;

            let thread = CreateRemoteThread(
                process,
                ptr::null(),
                0,
                Some(mem::transmute::<
                    unsafe extern "system" fn() -> isize,
                    unsafe extern "system" fn(*mut core::ffi::c_void) -> u32,
                >(load_library)),
                remote,
                0,
                ptr::null_mut(),
            );
            if thread.is_null() {
                return Err("cannot start a thread in the program".into());
            }

            let finished = WaitForSingleObject(thread, 30_000) == WAIT_OBJECT_0;
            let mut module = 0u32;
            GetExitCodeThread(thread, &mut module);
            CloseHandle(thread);

            // The thread's exit code is the low half of the module handle, zero on failure
            if !finished || module == 0 {
                return Err(format!("{} did not load in the program", dll.display()));
            }
            Ok(())
        })();

        VirtualFreeEx(process, remote, 0, MEM_RELEASE);
        result
    }
}

fn main() {
    let mut args = env::args().skip(1).peekable();
    let mut dll = None;
    if args.peek().map(String::as_str) == Some("--dll") {
        args.next();
        dll = Some(PathBuf::from(
            args.next().unwrap_or_else(|| fail("--dll needs a path")),
        ));
    }
    let arguments: Vec<String> = args.collect();
    if arguments.is_empty() {
        fail("usage: nectar-cuda-run [--dll <nvcuda.dll>] <program> [arguments...]");
    }

    let dll = dll.unwrap_or_else(|| env::current_exe().unwrap().with_file_name("nvcuda.dll"));
    let dll = dll
        .canonicalize()
        .unwrap_or_else(|error| fail(&format!("{}: {error}", dll.display())));

    let command_line = arguments
        .iter()
        .map(|argument| quote(argument))
        .collect::<Vec<_>>()
        .join(" ");
    let mut command_line = wide(OsStr::new(&command_line));

    unsafe {
        let mut startup: STARTUPINFOW = mem::zeroed();
        startup.cb = mem::size_of::<STARTUPINFOW>() as u32;
        startup.dwFlags = STARTF_USESTDHANDLES;
        startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
        startup.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
        startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);

        let mut info: PROCESS_INFORMATION = mem::zeroed();
        if CreateProcessW(
            ptr::null(),
            command_line.as_mut_ptr(),
            ptr::null(),
            ptr::null(),
            1,
            CREATE_SUSPENDED,
            ptr::null(),
            ptr::null(),
            &startup,
            &mut info,
        ) == 0
        {
            fail(&format!(
                "cannot start {}: {}",
                arguments[0],
                std::io::Error::last_os_error()
            ));
        }

        // Every process the program starts joins the job, and the job reports each new one
        let job = CreateJobObjectW(ptr::null(), ptr::null());
        let port = CreateIoCompletionPort(INVALID_HANDLE_VALUE, ptr::null_mut(), 0, 1);
        let association = JOBOBJECT_ASSOCIATE_COMPLETION_PORT { CompletionKey: job, CompletionPort: port };
        let following = !job.is_null()
            && !port.is_null()
            && SetInformationJobObject(
                job,
                JobObjectAssociateCompletionPortInformation,
                (&association as *const JOBOBJECT_ASSOCIATE_COMPLETION_PORT).cast(),
                mem::size_of::<JOBOBJECT_ASSOCIATE_COMPLETION_PORT>() as u32,
            ) != 0
            && AssignProcessToJobObject(job, info.hProcess) != 0;
        if !following {
            eprintln!("nectar-cuda-run: child processes will not get Nectar's nvcuda.dll");
        }

        if let Err(message) = load_into(info.hProcess, &dll) {
            TerminateProcess(info.hProcess, 1);
            fail(&message);
        }

        let verbose = env::var_os("NECTAR_CUDA_RUN_VERBOSE").is_some();
        if verbose {
            eprintln!("nectar-cuda-run: started process {}", info.dwProcessId);
        }
        ResumeThread(info.hThread);
        CloseHandle(info.hThread);

        if following {
            follow(port, info.dwProcessId, &dll, verbose);
        }
        WaitForSingleObject(info.hProcess, INFINITE);

        let mut code = 1u32;
        GetExitCodeProcess(info.hProcess, &mut code);
        CloseHandle(info.hProcess);
        process::exit(code as i32);
    }
}

/// Loads the DLL into each process the program starts, until the last of them has ended.
/// A child is already running when the job reports it, so the DLL arrives within moments of
/// its start rather than before it; a child that loads CUDA at once could still beat it.
unsafe fn follow(port: HANDLE, first: u32, dll: &Path, verbose: bool) {
    loop {
        let (mut message, mut key, mut overlapped) = (0u32, 0usize, ptr::null_mut());
        if unsafe { GetQueuedCompletionStatus(port, &mut message, &mut key, &mut overlapped, INFINITE) } == 0 {
            return;
        }

        match message {
            JOB_OBJECT_MSG_NEW_PROCESS => {
                let id = overlapped as usize as u32;
                if id != first {
                    unsafe { load_into_child(id, dll, verbose) };
                }
            }
            JOB_OBJECT_MSG_ACTIVE_PROCESS_ZERO => return,
            _ => {}
        }
    }
}

unsafe fn load_into_child(id: u32, dll: &Path, verbose: bool) {
    let access = PROCESS_CREATE_THREAD | PROCESS_QUERY_INFORMATION | PROCESS_VM_OPERATION | PROCESS_VM_WRITE | PROCESS_VM_READ;
    let process = unsafe { OpenProcess(access, 0, id) };
    if process.is_null() {
        eprintln!("nectar-cuda-run: cannot open child process {id}");
        return;
    }

    // A 64-bit DLL cannot go into a 32-bit process
    let mut wow64 = 0;
    unsafe { IsWow64Process(process, &mut wow64) };
    if wow64 != 0 {
        if verbose {
            eprintln!("nectar-cuda-run: child process {id} is 32-bit, left alone");
        }
    } else {
        match unsafe { load_into(process, dll) } {
            Ok(()) if verbose => eprintln!("nectar-cuda-run: loaded into child process {id}"),
            Ok(()) => {}
            Err(message) => eprintln!("nectar-cuda-run: child process {id}: {message}"),
        }
    }
    unsafe { CloseHandle(process) };
}

