//! Black-box tracing of the driver's undocumented export tables (cuGetExportTable).
//!
//! The entries of the chosen tables are replaced by trampolines. Each one records the call's
//! register arguments, 32 bytes of the memory each argument points to before and after,
//! and the result, then calls the real entry with the arguments untouched. This observes
//! the interface at run time, as any API tracer does. No driver code is read.
//!
//! NECTAR_CUDA_TRACE_TABLES picks the tables:
//!   unknown  (default) those Nectar does not provide, so their behavior can be worked out
//!   all      every table except CUDART_INTERFACE and INTEGRITY_CHECK, whose addresses the
//!            runtime hashes in its integrity check, so they must not move
//!   none

use std::arch::naked_asm;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ffi::c_void;
use std::fmt::Write as _;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{env, mem};

use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE,
    PAGE_EXECUTE_WRITECOPY, PAGE_GUARD, PAGE_NOACCESS, VirtualQuery,
};

/// How many entries can be wrapped in all
const SLOTS: usize = 512;
/// Bytes kept of the memory an argument points to
const SNAPSHOT: usize = 32;
/// How many calls are kept in order
const CALL_LIMIT: usize = 3000;

/// Tables Nectar provides (client/src/dark_api.rs), as ZLUDA documents them
const KNOWN: [&str; 10] = [
    "6bd5fb6c5bf4e74a8987d93912fd9df9",
    "42d85a8123f6cb478298f6e78a3aecdc",
    "a094798c2e742e7493f20800200c0a66",
    "c693336e1121df11a8c368f355d89593",
    "0ca50b8c1004929a89a7d0df10e77286",
    "195bcbf4d67d024aacc51d29cea631ae",
    "b10541e1f7c7c74a9f64f223be99f1e2",
    "d4082055bde6704b8d34ba123c66e1f2",
    "263e88607cd2614392f6bbd5006dfa7e",
    "dd9e98a4f410484499b0e2aeb7b432b5",
];

/// The runtime hashes the addresses of these, so they are never replaced
const HASHED: [&str; 2] = [
    "6bd5fb6c5bf4e74a8987d93912fd9df9",
    "d4082055bde6704b8d34ba123c66e1f2",
];

pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn should_wrap(table: &str) -> bool {
    let mode = env::var("NECTAR_CUDA_TRACE_TABLES").unwrap_or_else(|_| "unknown".into());
    match mode.as_str() {
        "none" => false,
        "all" => !HASHED.contains(&table),
        _ => !KNOWN.contains(&table),
    }
}

fn region(address: usize) -> Option<MEMORY_BASIC_INFORMATION> {
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { mem::zeroed() };
    let size = unsafe {
        VirtualQuery(
            address as *const c_void,
            &mut info,
            mem::size_of::<MEMORY_BASIC_INFORMATION>(),
        )
    };
    (size != 0 && info.State == MEM_COMMIT && info.Protect & (PAGE_NOACCESS | PAGE_GUARD) == 0)
        .then_some(info)
}

/// How many bytes from `address` can be read, up to `wanted`
fn readable(address: usize, wanted: usize) -> usize {
    if address < 0x10000 {
        return 0;
    }
    match region(address) {
        Some(info) => (info.BaseAddress as usize + info.RegionSize - address).min(wanted),
        None => 0,
    }
}

fn executable(address: usize) -> bool {
    let code = PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY;
    address >= 0x10000 && region(address).is_some_and(|info| info.Protect & code != 0)
}

// Trampolines. Thunk N loads N into r11 and jumps to `common`, which saves the argument
// registers, asks `before` for the real entry, copies 16 stack arguments, calls the entry,
// hands the result to `after` and returns it. The thunks are 16 bytes apart.

/// What `common` saves on its stack, from offset 0xA0
#[repr(C)]
struct Frame {
    args: [u64; 4],
    xmm: [[u64; 2]; 4],
    slot: u64,
    result: u64,
    real: u64,
}

#[unsafe(naked)]
unsafe extern "C" fn thunks() {
    naked_asm!(
        ".p2align 4",
        ".set nectar_slot, 0",
        ".rept {slots}",
        ".p2align 4",
        "movl $nectar_slot, %r11d",
        "jmp {common}",
        ".set nectar_slot, nectar_slot + 1",
        ".endr",
        slots = const SLOTS,
        common = sym common,
        options(att_syntax),
    )
}

#[unsafe(naked)]
unsafe extern "C" fn common() {
    naked_asm!(
        "push %rbp",
        "mov %rsp, %rbp",
        "sub $0x140, %rsp",
        "mov %rcx, 0xA0(%rsp)",
        "mov %rdx, 0xA8(%rsp)",
        "mov %r8, 0xB0(%rsp)",
        "mov %r9, 0xB8(%rsp)",
        "movdqu %xmm0, 0xC0(%rsp)",
        "movdqu %xmm1, 0xD0(%rsp)",
        "movdqu %xmm2, 0xE0(%rsp)",
        "movdqu %xmm3, 0xF0(%rsp)",
        "mov %r11, 0x100(%rsp)",
        "lea 0xA0(%rsp), %rcx",
        "call {before}",
        "mov %rax, 0x110(%rsp)",
        // The caller's stack arguments start after its 32 bytes of shadow space
        ".set nectar_argument, 0",
        ".rept 16",
        "mov 0x30 + nectar_argument * 8(%rbp), %r10",
        "mov %r10, 0x20 + nectar_argument * 8(%rsp)",
        ".set nectar_argument, nectar_argument + 1",
        ".endr",
        "mov 0xA0(%rsp), %rcx",
        "mov 0xA8(%rsp), %rdx",
        "mov 0xB0(%rsp), %r8",
        "mov 0xB8(%rsp), %r9",
        "movdqu 0xC0(%rsp), %xmm0",
        "movdqu 0xD0(%rsp), %xmm1",
        "movdqu 0xE0(%rsp), %xmm2",
        "movdqu 0xF0(%rsp), %xmm3",
        "call *0x110(%rsp)",
        "mov %rax, 0x108(%rsp)",
        "movdqu %xmm0, 0x120(%rsp)",
        "lea 0xA0(%rsp), %rcx",
        "call {after}",
        "movdqu 0x120(%rsp), %xmm0",
        "mov 0x108(%rsp), %rax",
        "mov %rbp, %rsp",
        "pop %rbp",
        "ret",
        before = sym before,
        after = sym after,
        options(att_syntax),
    )
}

fn thunk(slot: usize) -> usize {
    ((thunks as *const () as usize + 15) & !15) + slot * 16
}

/// The real entry behind each thunk
static REAL: [AtomicUsize; SLOTS] = [const { AtomicUsize::new(0) }; SLOTS];

struct Recording {
    /// (table, index) of each slot
    slots: Vec<(String, usize)>,
    /// Each table's entries when first handed out
    layouts: Vec<String>,
    calls: Vec<Call>,
    /// Per (table, index): calls and the results seen
    summary: BTreeMap<(String, usize), (u64, BTreeSet<u64>)>,
}

static RECORDING: Mutex<Recording> = Mutex::new(Recording {
    slots: Vec::new(),
    layouts: Vec::new(),
    calls: Vec::new(),
    summary: BTreeMap::new(),
});

fn recording() -> std::sync::MutexGuard<'static, Recording> {
    RECORDING
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

type Snapshot = [Option<Vec<u8>>; 4];

struct Call {
    slot: usize,
    thread: u32,
    args: [u64; 4],
    xmm0: u64,
    before: Snapshot,
    after: Snapshot,
    result: u64,
}

struct Pending {
    slot: usize,
    args: [u64; 4],
    xmm0: u64,
    before: Snapshot,
}

thread_local! {
    static PENDING: RefCell<Vec<Pending>> = const { RefCell::new(Vec::new()) };
}

fn snapshot(args: &[u64; 4]) -> Snapshot {
    args.map(|value| {
        let length = readable(value as usize, SNAPSHOT);
        (length > 0)
            .then(|| unsafe { std::slice::from_raw_parts(value as *const u8, length) }.to_vec())
    })
}

extern "system" fn before(frame: *mut Frame) -> usize {
    let frame = unsafe { &*frame };
    let slot = frame.slot as usize;
    let pending = Pending {
        slot,
        args: frame.args,
        xmm0: frame.xmm[0][0],
        before: snapshot(&frame.args),
    };
    let _ = PENDING.try_with(|stack| stack.borrow_mut().push(pending));
    REAL[slot].load(Ordering::Relaxed)
}

extern "system" fn after(frame: *mut Frame) {
    let frame = unsafe { &*frame };
    let Some(pending) = PENDING
        .try_with(|stack| stack.borrow_mut().pop())
        .ok()
        .flatten()
    else {
        return;
    };

    let call = Call {
        slot: pending.slot,
        thread: unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() },
        args: pending.args,
        xmm0: pending.xmm0,
        after: snapshot(&pending.args),
        before: pending.before,
        result: frame.result,
    };

    let mut recording = recording();
    let key = recording.slots[call.slot].clone();
    let entry = recording.summary.entry(key).or_default();
    entry.0 += 1;
    if entry.1.len() < 8 {
        entry.1.insert(call.result);
    }
    if recording.calls.len() < CALL_LIMIT {
        recording.calls.push(call);
    }
}

/// The table's entries: as many as its leading size says, or up to the first NULL
unsafe fn entries(table: *const c_void) -> Vec<usize> {
    let table = table as usize;
    if readable(table, 8) < 8 {
        return Vec::new();
    }

    let first = unsafe { *(table as *const usize) };
    let sized = first % 8 == 0 && (8..=0x1000).contains(&first);
    let count = if sized { first / 8 } else { 64 };

    let mut entries = Vec::new();
    for index in 0..count {
        let address = table + index * 8;
        if readable(address, 8) < 8 {
            break;
        }
        let value = unsafe { *(address as *const usize) };
        if !sized && value == 0 {
            break;
        }
        entries.push(value);
    }
    entries
}

/// Hands out the table, with its entries wrapped when it is one being traced
///
/// # Safety
/// `table` must be what the real cuGetExportTable returned for `uuid`.
pub(crate) unsafe fn wrap(uuid: &[u8; 16], table: *const c_void) -> *const c_void {
    static WRAPPED: Mutex<Option<HashMap<[u8; 16], usize>>> = Mutex::new(None);

    let mut wrapped = WRAPPED
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let wrapped = wrapped.get_or_insert_with(HashMap::new);
    if let Some(&copy) = wrapped.get(uuid) {
        return copy as *const c_void;
    }

    let name = hex(uuid);
    let entries = unsafe { entries(table) };
    let trace = should_wrap(&name);

    let mut layout = format!(
        "  {name} at {table:p}, {} entries{}",
        entries.len(),
        if trace { ", traced" } else { "" }
    );
    for (index, &value) in entries.iter().enumerate() {
        let kind = if value == 0 {
            "null".to_string()
        } else if executable(value) {
            "function".to_string()
        } else if value < 0x10000 {
            format!("{value}")
        } else {
            let length = readable(value, SNAPSHOT);
            format!(
                "data {}",
                hex(unsafe { std::slice::from_raw_parts(value as *const u8, length) })
            )
        };
        let _ = write!(layout, "\n    [{index}] {kind}");
    }

    let mut copy = entries.clone();
    if trace {
        let mut recording = recording();
        for (index, &value) in entries.iter().enumerate() {
            if !executable(value) || recording.slots.len() >= SLOTS {
                continue;
            }
            let slot = recording.slots.len();
            REAL[slot].store(value, Ordering::Relaxed);
            recording.slots.push((name.clone(), index));
            copy[index] = thunk(slot);
        }
    }
    recording().layouts.push(layout);

    let handed_out = if trace && !copy.is_empty() {
        Box::leak(copy.into_boxed_slice()).as_ptr() as usize
    } else {
        table as usize
    };
    wrapped.insert(*uuid, handed_out);
    handed_out as *const c_void
}

/// Appends what was seen to the trace report
pub(crate) fn report(out: &mut String) {
    let Ok(recording) = RECORDING.try_lock() else {
        return;
    };

    let _ = writeln!(out, "\nExport table layouts (when first handed out):");
    for layout in &recording.layouts {
        let _ = writeln!(out, "{layout}");
    }

    let _ = writeln!(
        out,
        "\nExport table entries called (table [index]: calls, results seen):"
    );
    for ((table, index), (calls, results)) in &recording.summary {
        let results: Vec<String> = results
            .iter()
            .map(|result| format!("{result:#x}"))
            .collect();
        let _ = writeln!(out, "  {table} [{index}]: {calls}, {}", results.join(" "));
    }

    let _ = writeln!(
        out,
        "\nExport table calls in order (args rcx rdx r8 r9; memory they point to before -> after):"
    );
    for call in &recording.calls {
        let (table, index) = &recording.slots[call.slot];
        let args: Vec<String> = call.args.iter().map(|arg| format!("{arg:#x}")).collect();
        let _ = writeln!(
            out,
            "  {} [{index}] thread {} ({}) xmm0 {:#x} = {:#x}",
            &table[..8],
            call.thread,
            args.join(", "),
            call.xmm0,
            call.result
        );
        for (position, (before, after)) in call.before.iter().zip(&call.after).enumerate() {
            if let (Some(before), Some(after)) = (before, after) {
                let change = if before == after {
                    "unchanged".to_string()
                } else {
                    hex(after)
                };
                let _ = writeln!(out, "      arg{position}: {} -> {change}", hex(before));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    extern "C" fn sample(a: u64, b: u64, c: u64, d: u64, e: u64, f: u64) -> u64 {
        a + 10 * b + 100 * c + 1000 * d + 10000 * e + 100000 * f
    }

    #[test]
    fn thunks_are_16_bytes_apart_and_carry_their_slot() {
        for slot in [0usize, 1, 7, SLOTS - 1] {
            let code = unsafe { std::slice::from_raw_parts(thunk(slot) as *const u8, 6) };
            // movl $slot, %r11d
            assert_eq!(&code[..2], &[0x41, 0xBB]);
            assert_eq!(
                u32::from_le_bytes(code[2..6].try_into().unwrap()),
                slot as u32
            );
        }
    }

    #[test]
    fn a_call_through_a_thunk_reaches_the_entry_and_is_recorded() {
        let slot = {
            let mut recording = recording();
            let slot = recording.slots.len();
            REAL[slot].store(sample as *const () as usize, Ordering::Relaxed);
            recording.slots.push(("test".into(), 3));
            slot
        };

        // Six arguments: four in registers, two on the stack
        let traced: extern "C" fn(u64, u64, u64, u64, u64, u64) -> u64 =
            unsafe { mem::transmute(thunk(slot)) };
        assert_eq!(traced(1, 2, 3, 4, 5, 6), 654321);

        let recording = recording();
        let call = recording
            .calls
            .iter()
            .rev()
            .find(|call| call.slot == slot)
            .unwrap();
        assert_eq!(call.args, [1, 2, 3, 4]);
        assert_eq!(call.result, 654321);
        assert_eq!(recording.summary[&("test".to_string(), 3)].0, 1);
    }
}
