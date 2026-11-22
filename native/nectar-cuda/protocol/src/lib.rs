//! Wire protocol between the borrower's `nvcuda.dll` and the lender's `nectar-server`.
//!
//! A connection starts with a hello in each direction, then carries one reply per request,
//! in order. The borrower may send many requests before it reads their replies:
//!
//! ```text
//! hello:   "NECTARCU", u32 protocol version, u32 generation, i32 value
//!          (the borrower's value is HELLO_CUDA or HELLO_CONTROL; the lender's is its
//!          driver version, followed by u8 challenge length and the challenge)
//! proof:   only after a challenge: u8 length, the HMAC of the challenge, u32 length and
//!          the borrower's name; then from the lender u8 1 (accepted) or 0, u32 length and
//!          a text: the lease token for a control connection, or why it was refused
//! request: u32 length, u32 function id, arguments
//! reply:   u32 length, i32 CUresult, results (only when the result is CUDA_SUCCESS)
//! ```
//!
//! A lender started to lend (nectar-lending) sends a challenge. A control connection proves
//! the shared secret and takes the lease; a CUDA connection proves the lease's token. A
//! lender without a secret sends no challenge, as on loopback in tests. Only the hello is
//! authenticated: what follows it is neither encrypted nor signed.
//!
//! Lengths count the bytes after the length field. Integers are little-endian. Byte
//! strings are a u64 length followed by the bytes. Values of CUDA types are their raw
//! bytes, and optional ones (pointers that may be NULL) are preceded by a u8 0 or 1.
//!
//! A function id is the FNV-1a hash of the versioned driver API symbol, such as
//! `cuMemAlloc_v2`. Both sides derive it from the name, so supporting a new function
//! never renumbers the others. The generation (functions.rs) changes whenever generated
//! code would put different bytes on the wire, and the hello refuses a mismatch.
//!
//! Nothing is compressed: what crosses in volume, quantized model weights and float results
//! such as logits, does not shrink under LZ4 (99.6% and 100.4% of their size), and weights
//! cross once anyway (UPLOAD).

pub mod pack;
pub mod pool;

use std::ffi::{CStr, CString, c_char};
use std::io::{self, Read, Write};
use std::{mem, ptr, slice};

pub const MAGIC: &[u8; 8] = b"NECTARCU";
/// 7: the link check, and the control connection names the profile. 6: the hello may carry a
/// challenge (lending). 5: version 4 could compress messages.
pub const VERSION: u32 = 15;

/// The borrower's hello value: a connection that carries CUDA calls
pub const HELLO_CUDA: i32 = 0;
/// The borrower's hello value: the lease's control connection (nectar-lending)
pub const HELLO_CONTROL: i32 = 1;
/// The borrower's hello value: a connection that carries Vulkan calls (nectar-vk-wire)
pub const HELLO_VULKAN: i32 = 2;
/// The borrower's hello value: a link check (nectar-lending's link.rs)
pub const HELLO_LINK: i32 = 3;
pub const DEFAULT_PORT: u16 = 43212;

/// Largest message either side accepts
pub const MAX_MESSAGE: usize = 1 << 30;

/// Not a driver function: asks the lender for the size of each parameter of a kernel
/// (a CUfunction or CUkernel), so the client can copy kernel arguments by value
pub const PARAM_LAYOUT: u32 = function_id("nectarParamLayout");

/// Not a driver function: asks for the content hash of every upload the lender keeps.
/// Results: u64 count, then the 16-byte hashes.
pub const CACHE_INDEX: u32 = function_id("nectarCacheIndex");

/// Not a driver function: an upload to GPU memory the lender keeps by content hash, so a
/// later upload of the same bytes, such as the next load of a model, needs only the hash.
/// Arguments: u32 id of the copy function it stands for (cuMemcpyHtoD_v2 and the like), u64
/// destination, u64 stream (for the asynchronous ones), the 16-byte hash, then u8 1 and the
/// bytes, or u8 0 and the u64 size to take from what the lender keeps.
pub const UPLOAD: u32 = function_id("nectarUpload");

/// Uploads smaller than this are sent as they are
pub const CACHE_MIN: usize = 1 << 20;

/// Not a driver function: every attribute of a device, kernel or function at once, where
/// the CUDA runtime asks for them one by one. Arguments: u32 id of cuDeviceGetAttribute,
/// cuKernelGetAttribute or cuFuncGetAttribute, then that function's arguments after the
/// attribute (the device; the kernel and device; the function). Results: u32 count, then
/// an i32 attribute and its i32 value for each the driver answers.
pub const ATTRIBUTES: u32 = function_id("nectarAttributes");

pub type Hash = [u8; 16];

/// The content hash uploads are kept by: XXH3, 128 bits
pub fn content_hash(data: &[u8]) -> Hash {
    xxhash_rust::xxh3::xxh3_128(data).to_le_bytes()
}

/// On a control connection: the borrower's heartbeat, once a second. Results: the lender's
/// totals for the lease so far: u64 calls, u64 bytes received, u64 bytes sent; then how busy its
/// GPU is: u32 percent, or u32::MAX if it can't say.
pub const BEAT: u32 = function_id("nectarBeat");

/// On a control connection: the lender's nvcudart_hybrid64.dll. Results: its version key
/// as a byte string, then its bytes.
pub const HYBRID: u32 = function_id("nectarHybrid");

/// On a control connection: gives the lease back
pub const RETURN: u32 = function_id("nectarReturn");

/// On a control connection: what the user chose the GPU for, as its key (gaming, ai). The
/// lender tunes the lease's connections by it.
pub const PROFILE: u32 = function_id("nectarProfile");

/// On a link check: answers with nothing
pub const LINK_ECHO: u32 = function_id("nectarLinkEcho");
/// On a link check: bytes to throw away, not answered
pub const LINK_SINK: u32 = function_id("nectarLinkSink");
/// On a link check: answers once everything sent before it has arrived
pub const LINK_SINK_END: u32 = function_id("nectarLinkSinkEnd");
/// On a link check: u64 count; answers with that many messages of a mebibyte
pub const LINK_SOURCE: u32 = function_id("nectarLinkSource");
/// On a link check: what the lender's end is connected by
pub const LINK_INFO: u32 = function_id("nectarLinkInfo");

/// The name of a driver function or of one of the requests above, for logs
pub fn call_name(id: u32) -> &'static str {
    match id {
        BEAT => "nectarBeat",
        HYBRID => "nectarHybrid",
        RETURN => "nectarReturn",
        PROFILE => "nectarProfile",
        PARAM_LAYOUT => "nectarParamLayout",
        CACHE_INDEX => "nectarCacheIndex",
        UPLOAD => "nectarUpload",
        ATTRIBUTES => "nectarAttributes",
        _ => function_name(id).unwrap_or("an unknown function"),
    }
}

/// Values of CUdriverProcAddressQueryResult
pub const SYMBOL_FOUND: i32 = 0;
pub const SYMBOL_NOT_FOUND: i32 = 1;
pub const VERSION_NOT_SUFFICIENT: i32 = 2;

/// The export cuGetProcAddress hands out for `symbol` when asked for CUDA `version`: the
/// newest variant that is not newer than it, the per-thread default stream one when asked
/// and there is one. Otherwise the CUdriverProcAddressQueryResult saying why not.
///
/// The real driver answers with internal addresses rather than its exports, so this is
/// worked out from cudaTypedefs.h instead (PROC_ADDRESSES).
pub fn proc_address(symbol: &str, version: i32, per_thread: bool) -> Result<&'static str, i32> {
    let start = PROC_ADDRESSES.partition_point(|entry| entry.0 < symbol);
    let variants = PROC_ADDRESSES[start..]
        .iter()
        .take_while(|entry| entry.0 == symbol);

    let newest = |stream: bool| {
        variants
            .clone()
            .filter(|entry| entry.2 == stream && entry.1 as i64 <= version as i64)
            .max_by_key(|entry| entry.1)
    };

    if variants.clone().next().is_none() {
        return Err(SYMBOL_NOT_FOUND);
    }
    let found = if per_thread {
        newest(true).or_else(|| newest(false))
    } else {
        newest(false)
    };
    found.map(|entry| entry.3).ok_or(VERSION_NOT_SUFFICIENT)
}

/// Whether cuGetProcAddress flags ask for per-thread default stream variants. With neither
/// CU_GET_PROC_ADDRESS_LEGACY_STREAM (1) nor _PER_THREAD_DEFAULT_STREAM (2), the process's
/// own setting decides, as with the real driver.
pub fn per_thread_stream(flags: u64) -> bool {
    match flags & 3 {
        2 => true,
        0 => std::env::var("CUDA_API_PER_THREAD_DEFAULT_STREAM").is_ok_and(|value| value == "1"),
        _ => false,
    }
}

pub const fn function_id(symbol: &str) -> u32 {
    let bytes = symbol.as_bytes();
    let mut hash: u32 = 0x811c_9dc5;
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u32;
        hash = hash.wrapping_mul(0x0100_0193);
        i += 1;
    }
    hash
}

macro_rules! functions {
    ($($symbol:ident),* $(,)?) => {
        /// Ids of the driver API functions
        #[allow(non_upper_case_globals)]
        pub mod id {
            $(pub const $symbol: u32 = super::function_id(stringify!($symbol));)*
        }

        /// Every export of the real driver, in the order the generated tables use
        pub const NAMES: &[&str] = &[$(stringify!($symbol)),*];

        pub fn function_name(id: u32) -> Option<&'static str> {
            $(if id == id::$symbol { return Some(stringify!($symbol)); })*
            None
        }
    };
}

include!("functions.rs");

pub fn write_hello(stream: &mut impl Write, value: i32) -> io::Result<()> {
    stream.write_all(MAGIC)?;
    stream.write_all(&VERSION.to_le_bytes())?;
    stream.write_all(&GENERATION.to_le_bytes())?;
    stream.write_all(&value.to_le_bytes())?;
    stream.flush()
}

/// Reads the other side's hello and returns its value
pub fn read_hello(stream: &mut impl Read) -> io::Result<i32> {
    let mut hello = [0u8; 20];
    stream.read_exact(&mut hello)?;

    if &hello[..8] != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a Nectar CUDA connection",
        ));
    }

    let version = u32::from_le_bytes(hello[8..12].try_into().unwrap());
    let generation = u32::from_le_bytes(hello[12..16].try_into().unwrap());
    if version != VERSION || generation != GENERATION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "protocol {version} generation {generation:#010x}, expected {VERSION} generation {GENERATION:#010x}"
            ),
        ));
    }

    Ok(i32::from_le_bytes(hello[16..20].try_into().unwrap()))
}

/// The lender's hello: its value, then the challenge (empty when it has no secret)
pub fn write_lender_hello(stream: &mut impl Write, value: i32, challenge: &[u8]) -> io::Result<()> {
    stream.write_all(MAGIC)?;
    stream.write_all(&VERSION.to_le_bytes())?;
    stream.write_all(&GENERATION.to_le_bytes())?;
    stream.write_all(&value.to_le_bytes())?;
    stream.write_all(&[challenge.len() as u8])?;
    stream.write_all(challenge)?;
    stream.flush()
}

/// Reads the lender's hello: its value and its challenge
pub fn read_lender_hello(stream: &mut impl Read) -> io::Result<(i32, Vec<u8>)> {
    let value = read_hello(stream)?;
    let mut length = [0u8];
    stream.read_exact(&mut length)?;
    let mut challenge = vec![0u8; length[0] as usize];
    stream.read_exact(&mut challenge)?;
    Ok((value, challenge))
}

fn write_text(stream: &mut impl Write, text: &[u8]) -> io::Result<()> {
    let length = u32::try_from(text.len())
        .ok()
        .filter(|&length| length <= 4096)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "text too long"))?;
    stream.write_all(&length.to_le_bytes())?;
    stream.write_all(text)
}

fn read_text(stream: &mut impl Read) -> io::Result<Vec<u8>> {
    let mut length = [0u8; 4];
    stream.read_exact(&mut length)?;
    let length = u32::from_le_bytes(length) as usize;
    if length > 4096 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "text too long"));
    }
    let mut text = vec![0u8; length];
    stream.read_exact(&mut text)?;
    Ok(text)
}

/// The borrower's answer to a challenge: the HMAC, and who it is
pub fn write_proof(stream: &mut impl Write, mac: &[u8], name: &str) -> io::Result<()> {
    stream.write_all(&[mac.len() as u8])?;
    stream.write_all(mac)?;
    write_text(stream, name.as_bytes())?;
    stream.flush()
}

pub fn read_proof(stream: &mut impl Read) -> io::Result<(Vec<u8>, String)> {
    let mut length = [0u8];
    stream.read_exact(&mut length)?;
    let mut mac = vec![0u8; length[0] as usize];
    stream.read_exact(&mut mac)?;
    let name = String::from_utf8_lossy(&read_text(stream)?).into_owned();
    Ok((mac, name))
}

/// The lender's verdict on a proof: accepted with a text (a control connection's token),
/// or refused with the reason
pub fn write_verdict(stream: &mut impl Write, accepted: bool, text: &str) -> io::Result<()> {
    stream.write_all(&[accepted as u8])?;
    write_text(stream, text.as_bytes())?;
    stream.flush()
}

/// The text of an accepted proof, or an error of kind PermissionDenied with the reason
pub fn read_verdict(stream: &mut impl Read) -> io::Result<String> {
    let mut accepted = [0u8];
    stream.read_exact(&mut accepted)?;
    let text = String::from_utf8_lossy(&read_text(stream)?).into_owned();
    if accepted[0] == 1 {
        Ok(text)
    } else {
        Err(io::Error::new(io::ErrorKind::PermissionDenied, text))
    }
}

/// Writes a message from parts, so large data is never copied into one buffer first.
/// `head` is the function id of a request or the CUresult of a reply. The caller flushes,
/// so several messages can go out together.
pub fn write_message(stream: &mut impl Write, head: u32, parts: &[&[u8]]) -> io::Result<()> {
    let length = 4 + parts.iter().map(|part| part.len()).sum::<usize>();
    if length > MAX_MESSAGE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "message too large",
        ));
    }

    stream.write_all(&(length as u32).to_le_bytes())?;
    stream.write_all(&head.to_le_bytes())?;
    for part in parts {
        stream.write_all(part)?;
    }
    Ok(())
}

/// Reads a message and returns its head and body
pub fn read_message(stream: &mut impl Read) -> io::Result<(u32, Vec<u8>)> {
    read_message_timed(stream).map(|(head, body, _)| (head, body))
}

/// As read_message, into a buffer that can be used again, as a game sends tens of thousands of these a
/// frame. Returns the head.
pub fn read_message_into(stream: &mut impl Read, body: &mut Vec<u8>) -> io::Result<u32> {
    let mut header = [0u8; 8];
    stream.read_exact(&mut header)?;
    let length = u32::from_le_bytes(header[..4].try_into().unwrap()) as usize;
    if !(4..=MAX_MESSAGE).contains(&length) {
        return Err(io::Error::new(io::ErrorKind::InvalidData, format!("message length {length}")));
    }
    body.resize(length - 4, 0);
    stream.read_exact(body)?;
    Ok(u32::from_le_bytes(header[4..].try_into().unwrap()))
}

/// As read_message, and how long the body took to arrive after its header: what a message costs on
/// the link, whatever else delayed it on the way
pub fn read_message_timed(stream: &mut impl Read) -> io::Result<(u32, Vec<u8>, std::time::Duration)> {
    let mut header = [0u8; 8];
    stream.read_exact(&mut header)?;

    let length = u32::from_le_bytes(header[..4].try_into().unwrap()) as usize;
    if !(4..=MAX_MESSAGE).contains(&length) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("message length {length}"),
        ));
    }

    let head = u32::from_le_bytes(header[4..].try_into().unwrap());
    let mut body = vec![0u8; length - 4];
    let started = std::time::Instant::now();
    stream.read_exact(&mut body)?;

    Ok((head, body, started.elapsed()))
}

/// Encodes arguments or results
#[derive(Default)]
pub struct Writer {
    buffer: Vec<u8>,
}

impl Writer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn u8(&mut self, value: u8) -> &mut Self {
        self.buffer.push(value);
        self
    }

    pub fn u32(&mut self, value: u32) -> &mut Self {
        self.buffer.extend_from_slice(&value.to_le_bytes());
        self
    }

    pub fn i32(&mut self, value: i32) -> &mut Self {
        self.buffer.extend_from_slice(&value.to_le_bytes());
        self
    }

    pub fn u64(&mut self, value: u64) -> &mut Self {
        self.buffer.extend_from_slice(&value.to_le_bytes());
        self
    }

    pub fn bytes(&mut self, value: &[u8]) -> &mut Self {
        self.u64(value.len() as u64);
        self.buffer.extend_from_slice(value);
        self
    }

    /// The raw bytes of a value of a CUDA type
    pub fn value<T: Copy>(&mut self, value: &T) -> &mut Self {
        let bytes =
            unsafe { slice::from_raw_parts((value as *const T).cast::<u8>(), mem::size_of::<T>()) };
        self.buffer.extend_from_slice(bytes);
        self
    }

    /// The value a pointer points to, or that it is NULL
    ///
    /// # Safety
    /// A non-null `pointer` must be valid for reading one `T`.
    pub unsafe fn optional<T: Copy>(&mut self, pointer: *const T) -> &mut Self {
        if pointer.is_null() {
            self.u8(0)
        } else {
            let value = unsafe { pointer.read_unaligned() };
            self.u8(1).value(&value)
        }
    }

    /// A C string, or that it is NULL
    ///
    /// # Safety
    /// A non-null `text` must point to a NUL-terminated string.
    pub unsafe fn string(&mut self, text: *const c_char) -> &mut Self {
        if text.is_null() {
            self.u8(0)
        } else {
            let bytes = unsafe { CStr::from_ptr(text) }.to_bytes();
            self.u8(1).bytes(bytes)
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.buffer
    }

    /// Bytes whose length the reader knows, without a length before them
    pub fn raw(&mut self, value: &[u8]) -> &mut Self {
        self.buffer.extend_from_slice(value);
        self
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

/// Decodes arguments or results, failing on truncated input
pub struct Reader<'a> {
    data: &'a [u8],
}

impl<'a> Reader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data }
    }

    /// The next `count` bytes
    pub fn take(&mut self, count: usize) -> io::Result<&'a [u8]> {
        if self.data.len() < count {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "message is truncated",
            ));
        }

        let (taken, rest) = self.data.split_at(count);
        self.data = rest;
        Ok(taken)
    }

    pub fn u8(&mut self) -> io::Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn i32(&mut self) -> io::Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    pub fn bytes(&mut self) -> io::Result<&'a [u8]> {
        let length = usize::try_from(self.u64()?)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "length too large"))?;
        self.take(length)
    }

    /// A value of a CUDA type from its raw bytes
    ///
    /// # Safety
    /// Every bit pattern must be a valid `T`, as for C integers, enums and plain structs.
    pub unsafe fn value<T: Copy>(&mut self) -> io::Result<T> {
        let bytes = self.take(mem::size_of::<T>())?;
        Ok(unsafe { ptr::read_unaligned(bytes.as_ptr().cast::<T>()) })
    }

    /// Reads an optional value and stores it through `destination` when both are present
    ///
    /// # Safety
    /// As for `value`, and a non-null `destination` must be valid for writing one `T`.
    pub unsafe fn optional_into<T: Copy>(&mut self, destination: *mut T) -> io::Result<()> {
        if self.u8()? != 0 {
            let value: T = unsafe { self.value()? };
            if !destination.is_null() {
                unsafe { destination.write_unaligned(value) };
            }
        }
        Ok(())
    }

    /// An optional C string
    pub fn string(&mut self) -> io::Result<Option<CString>> {
        if self.u8()? == 0 {
            return Ok(None);
        }
        CString::new(self.bytes()?)
            .map(Some)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "string holds a NUL"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn function_ids_are_fnv1a_of_the_symbol() {
        // FNV-1a("a") is a published test vector
        assert_eq!(function_id("a"), 0xe40c292c);
        assert_eq!(function_name(id::cuMemAlloc_v2), Some("cuMemAlloc_v2"));
        assert_ne!(id::cuMemAlloc_v2, function_id("cuMemAlloc"));
    }

    #[test]
    fn function_ids_are_unique() {
        let mut ids: Vec<u32> = NAMES.iter().map(|name| function_id(name)).collect();
        ids.push(PARAM_LAYOUT);
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), NAMES.len() + 1);
    }

    #[test]
    fn proc_addresses_follow_cuda_versions() {
        assert_eq!(
            proc_address("cuMemAlloc", 13040, false),
            Ok("cuMemAlloc_v2")
        );
        assert_eq!(proc_address("cuMemAlloc", 2000, false), Ok("cuMemAlloc"));
        assert_eq!(proc_address("cuCtxCreate", 2000, false), Ok("cuCtxCreate"));
        assert_eq!(
            proc_address("cuCtxCreate", 3020, false),
            Ok("cuCtxCreate_v2")
        );
        assert_eq!(
            proc_address("cuCtxCreate", 13040, false),
            Ok("cuCtxCreate_v4")
        );
        assert_eq!(
            proc_address("cuGetProcAddress", 11030, false),
            Ok("cuGetProcAddress")
        );
        assert_eq!(
            proc_address("cuGetProcAddress", 12000, false),
            Ok("cuGetProcAddress_v2")
        );
        assert_eq!(
            proc_address("cuMemcpyHtoD", 13040, false),
            Ok("cuMemcpyHtoD_v2")
        );
        assert_eq!(
            proc_address("cuMemcpyHtoD", 13040, true),
            Ok("cuMemcpyHtoD_v2_ptds")
        );
        assert_eq!(
            proc_address("cuInit", 13040, true),
            Ok("cuInit"),
            "no per-thread variant: the usual one"
        );
        assert_eq!(
            proc_address("cuMemAllocAsync", 11000, false),
            Err(VERSION_NOT_SUFFICIENT)
        );
        assert_eq!(
            proc_address("cuNotAFunction", 13040, false),
            Err(SYMBOL_NOT_FOUND)
        );
    }

    #[test]
    fn messages_round_trip() {
        let mut args = Writer::new();
        args.u64(0x7f00_0000_1000).i32(-5);
        let data = [1u8, 2, 3];

        let mut wire = Vec::new();
        write_message(
            &mut wire,
            id::cuMemcpyHtoD_v2,
            &[args.as_slice(), &(data.len() as u64).to_le_bytes(), &data],
        )
        .unwrap();

        let (head, body) = read_message(&mut wire.as_slice()).unwrap();
        assert_eq!(head, id::cuMemcpyHtoD_v2);

        let mut reader = Reader::new(&body);
        assert_eq!(reader.u64().unwrap(), 0x7f00_0000_1000);
        assert_eq!(reader.i32().unwrap(), -5);
        assert_eq!(reader.bytes().unwrap(), &data);
        assert!(reader.u32().is_err());
    }

    #[test]
    fn optional_values_and_strings_round_trip() {
        let present: u64 = 42;
        let mut writer = Writer::new();
        unsafe {
            writer
                .optional(&present)
                .optional(ptr::null::<u64>())
                .string(c"cuInit".as_ptr())
                .string(ptr::null());
        }

        let mut reader = Reader::new(writer.as_slice());
        let mut out: u64 = 0;
        unsafe { reader.optional_into(&mut out).unwrap() };
        assert_eq!(out, 42);
        unsafe { reader.optional_into(&mut out).unwrap() };
        assert_eq!(out, 42, "an absent value leaves the destination alone");
        assert_eq!(reader.string().unwrap().as_deref(), Some(c"cuInit"));
        assert_eq!(reader.string().unwrap(), None);
    }

    #[test]
    fn hello_rejects_other_protocols() {
        let mut wire = Vec::new();
        write_hello(&mut wire, 13050).unwrap();
        assert_eq!(read_hello(&mut wire.as_slice()).unwrap(), 13050);

        wire[12] ^= 1;
        assert!(
            read_hello(&mut wire.as_slice()).is_err(),
            "a different generation"
        );

        wire[12] ^= 1;
        wire[0] = b'X';
        assert!(read_hello(&mut wire.as_slice()).is_err());
    }
}
