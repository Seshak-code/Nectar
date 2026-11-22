//! The interpreter that carries Vulkan calls, driven by the generated tables.
//!
//! The borrower encodes a call's inputs from the program's memory (`encode_call`), and later
//! writes the lender's outputs back into the program's memory (`decode_outputs`). The lender
//! rebuilds the arguments in an arena (`decode_call`), calls the real driver, and encodes the
//! outputs (`encode_outputs`).
//!
//! Format: values are little-endian and fixed-size, handles are 64-bit ids, a pointer is a
//! presence byte and then an element count and the elements, and a pNext chain is each known
//! structure's sType and members, ending with 0. Structures the tables don't know are left
//! out of a chain, as a driver ignores what it doesn't know.
//!
//! An output is sent as a request first: its capacity, and for structures their skeleton
//! (sType, pNext chain and counts) so the lender's driver sees the same chain the program
//! built. The answer carries the filled values, written into the program's own structures
//! without touching their pointers.
//!
//! Shader code (VkShaderModuleCreateInfo's pCode, on its own or chained on a pipeline's stage) can
//! go by content hash, so that a game's shaders cross the link once: its presence byte is then 3
//! for code to keep, the element count, the hash and the code, or 2 for code kept earlier, the
//! element count and the hash.

use std::ffi::c_void;
use std::{io, ptr, slice};

use nectar_cuda_protocol::{Hash, Reader, Writer};
use nectar_vk_sys::VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;

use crate::tables::*;
use crate::{TYPES, by_stype};

#[derive(Debug)]
pub enum Error {
    /// Something the tables say can't be carried
    Unsupported(&'static str),
    /// A message that doesn't decode
    Io(io::Error),
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Error::Io(error)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// The most memory one output may ask the lender for
const MAX_OUTPUT: usize = 1 << 30;

/// How shader code goes
pub enum CodeWay {
    Whole,
    /// With its hash, for the lender to keep
    Keep(Hash),
    /// Only its hash: the lender has it
    Kept(Hash),
}

/// Translates handles at either end, and looks after shader code
pub trait Handles {
    /// A handle as it goes out: a dispatchable object (or a real handle, on the lender) to
    /// the id the other side knows it by
    fn outgoing(&mut self, dispatchable: bool, value: u64) -> u64;
    /// A handle as it comes in: an id to the object (borrower) or real handle (lender)
    fn incoming(&mut self, dispatchable: bool, id: u64) -> u64;
    /// The borrower: a new id for a handle it creates, and the value the program gets
    fn create(&mut self, dispatchable: bool) -> (u64, u64);
    /// The lender: a handle created under the id the borrower chose
    fn bind(&mut self, id: u64, real: u64);
    /// The borrower: how this shader code goes
    fn code_out(&mut self, _code: &[u8]) -> CodeWay {
        CodeWay::Whole
    }
    /// The lender: code to keep under its hash
    fn code_keep(&mut self, _hash: &Hash, _code: &[u8]) {}
    /// The lender: code kept earlier
    fn code_in(&mut self, _hash: &Hash, _size: u64) -> Option<Vec<u8>> {
        None
    }
}

/// How the members of a structure are read: whole, or only what an output needs
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Full,
    Skeleton,
}

/// What a length is evaluated against
#[derive(Clone, Copy)]
enum Context<'a> {
    Struct(*const u8, &'static [Member]),
    Params(&'a [*const c_void], &'static [Param]),
}

unsafe fn read_width(at: *const u8, width: usize) -> usize {
    unsafe {
        match width {
            1 => *at as usize,
            2 => (at as *const u16).read_unaligned() as usize,
            4 => (at as *const u32).read_unaligned() as usize,
            _ => (at as *const u64).read_unaligned() as usize,
        }
    }
}

fn field_width(field: &Field) -> usize {
    match field {
        Field::Raw(size) => *size,
        _ => 8,
    }
}

fn elem_width(elem: &Elem) -> usize {
    match elem {
        Elem::Raw(size) => *size,
        _ => 8,
    }
}

fn elem_size(elem: &Elem) -> usize {
    match elem {
        Elem::Raw(size) => *size,
        Elem::Struct(index) => TYPES[*index as usize].size,
        _ => 8,
    }
}

/// How many elements a pointer of `len` points to
unsafe fn evaluate(len: &Len, context: Context, pointer: *const u8) -> usize {
    unsafe {
        match (*len, context) {
            (Len::One, _) => 1,
            (Len::Fixed(count), _) => count,
            (Len::NullTerminated, _) => {
                if pointer.is_null() {
                    0
                } else {
                    std::ffi::CStr::from_ptr(pointer.cast()).to_bytes().len()
                }
            }
            (Len::Sibling(index), Context::Struct(base, members)) => {
                let member = &members[index as usize];
                read_width(base.add(member.offset), field_width(&member.field))
            }
            (Len::Sibling(index), Context::Params(args, params)) => {
                read_width(args[index as usize].cast(), field_width(&params[index as usize].field))
            }
            (Len::SiblingDeref(index), Context::Params(args, params)) => {
                let target = *(args[index as usize] as *const *const u8);
                let width = match &params[index as usize].field {
                    Field::Ptr(p) => elem_width(&p.elem),
                    _ => 4,
                };
                if target.is_null() { 0 } else { read_width(target, width) }
            }
            (Len::SiblingField(index, offset, width), Context::Params(args, _)) => {
                let target = *(args[index as usize] as *const *const u8);
                if target.is_null() { 0 } else { read_width(target.add(offset), width as usize) }
            }
            (Len::StructExpr(function), Context::Struct(base, _)) => function(base),
            (Len::ParamExpr(function), Context::Params(args, _)) => function(args),
            _ => 0,
        }
    }
}

// ---- Encoding: the borrower's inputs, and the lender's outputs ----

unsafe fn encode_struct(w: &mut Writer, index: Index, base: *const u8, h: &mut dyn Handles, mode: Mode) -> Result<()> {
    let info = &TYPES[index as usize];
    let shader = info.stype == VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO && mode == Mode::Full;
    for member in info.members {
        unsafe {
            let at = base.add(member.offset);
            let context = Context::Struct(base, info.members);
            match &member.field {
                Field::Ptr(p) if shader && member.name == "pCode" => encode_code(w, p, at, context, h)?,
                field => encode_field(w, field, at, context, h, mode)?,
            }
        }
    }
    Ok(())
}

/// Shader code, by hash where that's what `h` wants
unsafe fn encode_code(w: &mut Writer, p: &Pointer, at: *const u8, context: Context, h: &mut dyn Handles) -> Result<()> {
    unsafe {
        let pointer = (at as *const *const u8).read_unaligned();
        let count = evaluate(&p.len, context, pointer);
        if pointer.is_null() {
            w.u8(0);
            return Ok(());
        }
        let code = slice::from_raw_parts(pointer, elem_size(&p.elem) * count);
        match h.code_out(code) {
            CodeWay::Whole => return encode_pointer(w, p, pointer, count, h),
            CodeWay::Keep(hash) => {
                w.u8(CODE_TO_KEEP).u64(count as u64).raw(&hash).raw(code);
            }
            CodeWay::Kept(hash) => {
                w.u8(CODE_KEPT).u64(count as u64).raw(&hash);
            }
        }
    }
    Ok(())
}

const CODE_KEPT: u8 = 2;
const CODE_TO_KEEP: u8 = 3;

/// Shader code, however it came
unsafe fn decode_code(r: &mut Reader, p: &Pointer, arena: &mut Arena, h: &mut dyn Handles) -> Result<*mut u8> {
    let way = r.u8()?;
    if way == 0 {
        return Ok(ptr::null_mut());
    }
    let count = r.u64()? as usize;
    let size = elem_size(&p.elem);
    let length = size * count;
    let pointer = arena.alloc(size * (count + 1));
    unsafe {
        match way {
            CODE_KEPT => {
                let hash: Hash = r.take(16)?.try_into().map_err(|_| io::Error::from(io::ErrorKind::InvalidData))?;
                let code = h.code_in(&hash, length as u64).ok_or(Error::Unsupported("shader code that isn't kept here"))?;
                if code.len() != length {
                    return Err(io::Error::from(io::ErrorKind::InvalidData).into());
                }
                ptr::copy_nonoverlapping(code.as_ptr(), pointer, length);
            }
            CODE_TO_KEEP => {
                let hash: Hash = r.take(16)?.try_into().map_err(|_| io::Error::from(io::ErrorKind::InvalidData))?;
                let code = r.take(length)?;
                h.code_keep(&hash, code);
                ptr::copy_nonoverlapping(code.as_ptr(), pointer, length);
            }
            _ => ptr::copy_nonoverlapping(r.take(length)?.as_ptr(), pointer, length),
        }
    }
    Ok(pointer)
}

/// A pNext chain: each structure the tables know, by sType, ending with 0
unsafe fn encode_chain(w: &mut Writer, mut node: *const u8, h: &mut dyn Handles, mode: Mode) -> Result<()> {
    unsafe {
        while !node.is_null() {
            let stype = (node as *const i32).read_unaligned();
            if let Some(index) = by_stype(stype) {
                w.i32(stype);
                // Its own pNext member carries the rest of the chain
                return encode_struct(w, index, node, h, mode);
            }
            node = (node.add(8) as *const *const u8).read_unaligned();
        }
    }
    w.i32(0);
    Ok(())
}

unsafe fn encode_field(w: &mut Writer, field: &Field, at: *const u8, context: Context, h: &mut dyn Handles, mode: Mode) -> Result<()> {
    unsafe {
        match field {
            Field::SType => {
                w.raw(slice::from_raw_parts(at, 4));
            }
            Field::Next => encode_chain(w, (at as *const *const u8).read_unaligned(), h, mode)?,
            Field::Raw(size) => {
                w.raw(slice::from_raw_parts(at, *size));
            }
            Field::Float => {
                w.raw(slice::from_raw_parts(at, 4));
            }
            Field::SizeT => {
                w.u64((at as *const usize).read_unaligned() as u64);
            }
            Field::Handle(dispatchable) => {
                let value = (at as *const u64).read_unaligned();
                w.u64(if mode == Mode::Skeleton { value } else { h.outgoing(*dispatchable, value) });
            }
            Field::HandleArray(dispatchable, count) => {
                for i in 0..*count {
                    let value = (at.add(i * 8) as *const u64).read_unaligned();
                    w.u64(if mode == Mode::Skeleton { value } else { h.outgoing(*dispatchable, value) });
                }
            }
            Field::Struct(index) => encode_struct(w, *index, at, h, mode)?,
            Field::StructArray(index, count) => {
                let size = TYPES[*index as usize].size;
                for i in 0..*count {
                    encode_struct(w, *index, at.add(i * size), h, mode)?;
                }
            }
            Field::Ptr(p) => {
                let pointer = (at as *const *const u8).read_unaligned();
                let count = evaluate(&p.len, context, pointer);
                if mode == Mode::Skeleton {
                    encode_capacity(w, p, pointer, count, h)?;
                } else {
                    encode_pointer(w, p, pointer, count, h)?;
                }
            }
            Field::Unsupported(why) => {
                // A NULL pointer of a kind that can't be carried is fine
                if (at as *const usize).read_unaligned() != 0 {
                    return Err(Error::Unsupported(why));
                }
                w.u8(0);
            }
        }
    }
    Ok(())
}

/// A pointer's elements
unsafe fn encode_pointer(w: &mut Writer, p: &Pointer, pointer: *const u8, count: usize, h: &mut dyn Handles) -> Result<()> {
    if pointer.is_null() {
        w.u8(0);
        return Ok(());
    }
    w.u8(1).u64(count as u64);
    unsafe { encode_elements(w, &p.elem, pointer, count, h, Mode::Full) }
}

unsafe fn encode_elements(w: &mut Writer, elem: &Elem, pointer: *const u8, count: usize, h: &mut dyn Handles, mode: Mode) -> Result<()> {
    unsafe {
        match elem {
            Elem::Raw(size) => {
                w.raw(slice::from_raw_parts(pointer, size * count));
            }
            Elem::SizeT => {
                for i in 0..count {
                    w.u64((pointer.add(i * 8) as *const usize).read_unaligned() as u64);
                }
            }
            Elem::Handle(dispatchable) => {
                for i in 0..count {
                    let value = (pointer.add(i * 8) as *const u64).read_unaligned();
                    w.u64(if mode == Mode::Skeleton { value } else { h.outgoing(*dispatchable, value) });
                }
            }
            Elem::Struct(index) => {
                let size = TYPES[*index as usize].size;
                for i in 0..count {
                    encode_struct(w, *index, pointer.add(i * size), h, mode)?;
                }
            }
            Elem::Ptr(inner) => {
                for i in 0..count {
                    let element = (pointer.add(i * 8) as *const *const u8).read_unaligned();
                    let inner_count = evaluate(&inner.len, Context::Struct(ptr::null(), &[]), element);
                    encode_pointer(w, inner, element, inner_count, h)?;
                }
            }
            Elem::Unsupported(why) => return Err(Error::Unsupported(why)),
        }
    }
    Ok(())
}

/// An output pointer as a request: whether there is one, its capacity, and the skeleton of
/// its structures. Counts the caller sets (inout) are sent as they are.
unsafe fn encode_capacity(w: &mut Writer, p: &Pointer, pointer: *const u8, count: usize, h: &mut dyn Handles) -> Result<()> {
    if pointer.is_null() {
        w.u8(0);
        return Ok(());
    }
    w.u8(1).u64(count as u64);
    unsafe {
        match &p.elem {
            Elem::Struct(_) => encode_elements(w, &p.elem, pointer, count, h, Mode::Skeleton)?,
            Elem::Raw(size) if p.inout => {
                w.raw(slice::from_raw_parts(pointer, size * count));
            }
            Elem::SizeT if p.inout => encode_elements(w, &p.elem, pointer, count, h, Mode::Full)?,
            _ => {}
        }
    }
    Ok(())
}

// ---- Decoding: the lender's inputs, and the borrower's outputs ----

/// Memory for one call's decoded arguments, freed with it
#[derive(Default)]
pub struct Arena {
    blocks: Vec<Box<[u64]>>,
    /// Words used of the last block
    used: usize,
    /// A call's slots, to use again
    slots: Vec<u64>,
}

/// Words in a block. A call rarely needs more than a few, and they're used again call after call.
const ARENA_BLOCK: usize = 2048;

impl Arena {
    pub fn alloc(&mut self, bytes: usize) -> *mut u8 {
        let words = bytes.div_ceil(8).max(1);
        if self.blocks.is_empty() {
            self.blocks.push(vec![0u64; ARENA_BLOCK].into_boxed_slice());
            self.used = 0;
        }
        if words > ARENA_BLOCK {
            // Of its own, below the block in use. It doesn't stay, being rare.
            let mut block = vec![0u64; words].into_boxed_slice();
            let pointer = block.as_mut_ptr().cast();
            let at = self.blocks.len().saturating_sub(1);
            self.blocks.insert(at, block);
            return pointer;
        }
        if self.blocks.last().is_none_or(|block| block.len() < self.used + words) {
            self.blocks.push(vec![0u64; ARENA_BLOCK].into_boxed_slice());
            self.used = 0;
        }
        let at = self.used;
        self.used += words;
        let block = self.blocks.last_mut().expect("one was just made");
        block[at..at + words].fill(0);
        block[at..].as_mut_ptr().cast()
    }

    /// Everything from it is finished with, so it can all be used again
    pub fn reuse(&mut self, call: Call) {
        self.slots = call.slots;
        if self.blocks.len() > 1 {
            let last = self.blocks.pop();
            self.blocks.clear();
            self.blocks.extend(last);
        }
        self.used = 0;
    }
}

unsafe fn decode_struct(r: &mut Reader, index: Index, base: *mut u8, arena: &mut Arena, h: &mut dyn Handles, mode: Mode) -> Result<()> {
    let info = &TYPES[index as usize];
    let shader = info.stype == VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO && mode == Mode::Full;
    for member in info.members {
        unsafe {
            let at = base.add(member.offset);
            match &member.field {
                Field::Ptr(p) if shader && member.name == "pCode" => {
                    (at as *mut *mut u8).write_unaligned(decode_code(r, p, arena, h)?);
                }
                field => decode_field(r, field, at, arena, h, mode)?,
            }
        }
    }
    Ok(())
}

unsafe fn decode_field(r: &mut Reader, field: &Field, at: *mut u8, arena: &mut Arena, h: &mut dyn Handles, mode: Mode) -> Result<()> {
    unsafe {
        match field {
            Field::SType | Field::Float => ptr::copy_nonoverlapping(r.take(4)?.as_ptr(), at, 4),
            Field::Next => {
                let stype = r.i32()?;
                let node = if stype == 0 {
                    ptr::null_mut()
                } else {
                    let index = by_stype(stype).ok_or(Error::Unsupported("a structure this lender doesn't know"))?;
                    let node = arena.alloc(TYPES[index as usize].size);
                    decode_struct(r, index, node, arena, h, mode)?;
                    node
                };
                (at as *mut *mut u8).write_unaligned(node);
            }
            Field::Raw(size) => ptr::copy_nonoverlapping(r.take(*size)?.as_ptr(), at, *size),
            Field::SizeT => (at as *mut usize).write_unaligned(r.u64()? as usize),
            Field::Handle(dispatchable) => {
                let id = r.u64()?;
                (at as *mut u64).write_unaligned(if mode == Mode::Skeleton { id } else { h.incoming(*dispatchable, id) });
            }
            Field::HandleArray(dispatchable, count) => {
                for i in 0..*count {
                    let id = r.u64()?;
                    (at.add(i * 8) as *mut u64)
                        .write_unaligned(if mode == Mode::Skeleton { id } else { h.incoming(*dispatchable, id) });
                }
            }
            Field::Struct(index) => decode_struct(r, *index, at, arena, h, mode)?,
            Field::StructArray(index, count) => {
                let size = TYPES[*index as usize].size;
                for i in 0..*count {
                    decode_struct(r, *index, at.add(i * size), arena, h, mode)?;
                }
            }
            Field::Ptr(p) => {
                let pointer = if mode == Mode::Skeleton {
                    decode_capacity(r, p, arena, h)?.0
                } else {
                    decode_pointer(r, p, arena, h)?
                };
                (at as *mut *mut u8).write_unaligned(pointer);
            }
            Field::Unsupported(_) => {
                r.u8()?;
                (at as *mut usize).write_unaligned(0);
            }
        }
    }
    Ok(())
}

unsafe fn decode_pointer(r: &mut Reader, p: &Pointer, arena: &mut Arena, h: &mut dyn Handles) -> Result<*mut u8> {
    if r.u8()? == 0 {
        return Ok(ptr::null_mut());
    }
    let count = r.u64()? as usize;
    let size = elem_size(&p.elem);
    // One more element, zeroed, ends a C string
    let pointer = arena.alloc(size * (count + 1));
    unsafe { decode_elements(r, &p.elem, pointer, count, arena, h, Mode::Full)? };
    Ok(pointer)
}

unsafe fn decode_elements(r: &mut Reader, elem: &Elem, pointer: *mut u8, count: usize, arena: &mut Arena, h: &mut dyn Handles, mode: Mode) -> Result<()> {
    unsafe {
        match elem {
            Elem::Raw(size) => ptr::copy_nonoverlapping(r.take(size * count)?.as_ptr(), pointer, size * count),
            Elem::SizeT => {
                for i in 0..count {
                    (pointer.add(i * 8) as *mut usize).write_unaligned(r.u64()? as usize);
                }
            }
            Elem::Handle(dispatchable) => {
                for i in 0..count {
                    let id = r.u64()?;
                    (pointer.add(i * 8) as *mut u64)
                        .write_unaligned(if mode == Mode::Skeleton { id } else { h.incoming(*dispatchable, id) });
                }
            }
            Elem::Struct(index) => {
                let size = TYPES[*index as usize].size;
                for i in 0..count {
                    decode_struct(r, *index, pointer.add(i * size), arena, h, mode)?;
                }
            }
            Elem::Ptr(inner) => {
                for i in 0..count {
                    let element = decode_pointer(r, inner, arena, h)?;
                    (pointer.add(i * 8) as *mut *mut u8).write_unaligned(element);
                }
            }
            Elem::Unsupported(why) => return Err(Error::Unsupported(why)),
        }
    }
    Ok(())
}

/// An output request: memory for the outputs, with the skeleton of its structures
unsafe fn decode_capacity(r: &mut Reader, p: &Pointer, arena: &mut Arena, h: &mut dyn Handles) -> Result<(*mut u8, usize)> {
    if r.u8()? == 0 {
        return Ok((ptr::null_mut(), 0));
    }
    let count = r.u64()? as usize;
    // An output never needs more; a larger request is a count the program didn't set
    if elem_size(&p.elem).saturating_mul(count) > MAX_OUTPUT {
        return Err(Error::Unsupported("an output larger than 1 GiB"));
    }
    let pointer = arena.alloc(elem_size(&p.elem) * count.max(1));
    unsafe {
        match &p.elem {
            Elem::Struct(_) => decode_elements(r, &p.elem, pointer, count, arena, h, Mode::Skeleton)?,
            Elem::Raw(size) if p.inout => ptr::copy_nonoverlapping(r.take(size * count)?.as_ptr(), pointer, size * count),
            Elem::SizeT if p.inout => decode_elements(r, &p.elem, pointer, count, arena, h, Mode::Full)?,
            _ => {}
        }
    }
    Ok((pointer, count))
}

/// Writes a structure the lender filled into the program's own structure, following the
/// program's pointers (its pNext chain and arrays) rather than replacing them
unsafe fn decode_into_struct(r: &mut Reader, index: Index, base: *mut u8, h: &mut dyn Handles) -> Result<()> {
    for member in TYPES[index as usize].members {
        let at = unsafe { base.add(member.offset) };
        unsafe {
            match &member.field {
                Field::SType => {
                    r.take(4)?;
                }
                Field::Next => {
                    let stype = r.i32()?;
                    if stype != 0 {
                        let index = by_stype(stype).ok_or(Error::Unsupported("a structure this borrower doesn't know"))?;
                        // The program's node with that sType; the request skipped unknown ones
                        let mut node = (at as *const *mut u8).read_unaligned();
                        while !node.is_null() && (node as *const i32).read_unaligned() != stype {
                            node = (node.add(8) as *const *mut u8).read_unaligned();
                        }
                        if node.is_null() {
                            let mut scratch = Arena::default();
                            let node = scratch.alloc(TYPES[index as usize].size);
                            decode_into_struct(r, index, node, h)?;
                        } else {
                            decode_into_struct(r, index, node, h)?;
                        }
                    }
                }
                Field::Ptr(p) => {
                    let pointer = (at as *const *mut u8).read_unaligned();
                    decode_into_pointer(r, p, pointer, h)?;
                }
                Field::Struct(inner) => decode_into_struct(r, *inner, at, h)?,
                Field::StructArray(inner, count) => {
                    let size = TYPES[*inner as usize].size;
                    for i in 0..*count {
                        decode_into_struct(r, *inner, at.add(i * size), h)?;
                    }
                }
                field => {
                    let mut scratch = Arena::default();
                    decode_field(r, field, at, &mut scratch, h, Mode::Full)?;
                }
            }
        }
    }
    Ok(())
}

/// An output array into the program's memory, at most what the program made room for
unsafe fn decode_into_pointer(r: &mut Reader, p: &Pointer, pointer: *mut u8, h: &mut dyn Handles) -> Result<()> {
    if r.u8()? == 0 {
        return Ok(());
    }
    let count = r.u64()? as usize;
    let mut scratch = Arena::default();
    let target = if pointer.is_null() { scratch.alloc(elem_size(&p.elem) * count.max(1)) } else { pointer };
    unsafe {
        match &p.elem {
            Elem::Struct(index) => {
                let size = TYPES[*index as usize].size;
                for i in 0..count {
                    decode_into_struct(r, *index, target.add(i * size), h)?;
                }
            }
            elem => decode_elements(r, elem, target, count, &mut scratch, h, Mode::Full)?,
        }
    }
    Ok(())
}

// ---- Calls ----

/// The borrower: a call's inputs and output requests, from the program's arguments (each
/// argument given by its address)
///
/// # Safety
/// The arguments must be what the command's Vulkan specification allows.
pub unsafe fn encode_call(command: &CommandInfo, args: &[*const c_void], w: &mut Writer, h: &mut dyn Handles) -> Result<()> {
    if let Some(why) = command.unsupported {
        return Err(Error::Unsupported(why));
    }
    let context = Context::Params(args, command.params);
    for (index, param) in command.params.iter().enumerate() {
        let at = args[index] as *const u8;
        unsafe {
            match &param.field {
                Field::Ptr(p) if p.out => {
                    let pointer = (at as *const *mut u8).read_unaligned();
                    let count = evaluate(&p.len, context, pointer);
                    if let Elem::Handle(dispatchable) = p.elem
                        && command.client_handles
                    {
                        // Handles the borrower chooses, so the call needn't wait
                        if pointer.is_null() {
                            w.u8(0);
                            continue;
                        }
                        w.u8(1).u64(count as u64);
                        for i in 0..count {
                            let (id, value) = h.create(dispatchable);
                            (pointer.add(i * 8) as *mut u64).write_unaligned(value);
                            w.u64(id);
                        }
                    } else {
                        encode_capacity(w, p, pointer, count, h)?;
                    }
                }
                field => encode_field(w, field, at, context, h, Mode::Full)?,
            }
        }
    }
    Ok(())
}

/// The borrower: writes the lender's outputs into the program's memory
///
/// # Safety
/// As for encode_call, with the same arguments.
pub unsafe fn decode_outputs(command: &CommandInfo, args: &[*const c_void], r: &mut Reader, h: &mut dyn Handles) -> Result<()> {
    for (index, param) in command.params.iter().enumerate() {
        if let Field::Ptr(p) = &param.field
            && p.out
            && !(command.client_handles && matches!(p.elem, Elem::Handle(_)))
        {
            unsafe {
                let pointer = (args[index] as *const *mut u8).read_unaligned();
                decode_into_pointer(r, p, pointer, h)?;
            }
        }
    }
    Ok(())
}

/// A call decoded on the lender: its arguments as u64 slots, ready for thunks::call
pub struct Call {
    pub slots: Vec<u64>,
    /// Handles the borrower chose: where the driver writes each, and its id
    created: Vec<(*mut u8, Vec<u64>)>,
}

impl Call {
    /// The ids of the handles the borrower chose
    pub fn created_ids(&self) -> impl Iterator<Item = u64> + '_ {
        self.created.iter().flat_map(|(_, ids)| ids.iter().copied())
    }
}

/// The lender: a call's arguments, with memory for its outputs
///
/// # Safety
/// The tables must describe the command, as generated.
pub unsafe fn decode_call(command: &CommandInfo, r: &mut Reader, arena: &mut Arena, h: &mut dyn Handles) -> Result<Call> {
    let mut slots = std::mem::take(&mut arena.slots);
    slots.clear();
    slots.resize(command.params.len(), 0);
    let mut call = Call { slots, created: Vec::new() };
    for (index, param) in command.params.iter().enumerate() {
        let slot = &mut call.slots[index] as *mut u64 as *mut u8;
        unsafe {
            match &param.field {
                Field::Ptr(p) if p.out => {
                    if matches!(p.elem, Elem::Handle(_)) && command.client_handles {
                        if r.u8()? == 0 {
                            continue;
                        }
                        let count = r.u64()? as usize;
                        let ids = (0..count).map(|_| r.u64()).collect::<io::Result<Vec<u64>>>()?;
                        let pointer = arena.alloc(8 * count.max(1));
                        call.slots[index] = pointer as u64;
                        call.created.push((pointer, ids));
                    } else {
                        let (pointer, _) = decode_capacity(r, p, arena, h)?;
                        call.slots[index] = pointer as u64;
                    }
                }
                Field::Raw(size) => {
                    ptr::copy_nonoverlapping(r.take(*size)?.as_ptr(), slot, *size);
                }
                field => decode_field(r, field, slot, arena, h, Mode::Full)?,
            }
        }
    }
    Ok(call)
}

/// The lender, after the call: binds the handles the borrower chose, and encodes the outputs
///
/// # Safety
/// `call` must be what decode_call made for this command, after the driver ran it.
pub unsafe fn encode_outputs(command: &CommandInfo, call: &Call, succeeded: bool, w: &mut Writer, h: &mut dyn Handles) -> Result<()> {
    if succeeded {
        for (pointer, ids) in &call.created {
            for (i, id) in ids.iter().enumerate() {
                h.bind(*id, unsafe { (pointer.add(i * 8) as *const u64).read_unaligned() });
            }
        }
    }
    // Nearly all commands have nothing to send back, and a game sends a great many
    if !command.params.iter().any(|param| matches!(&param.field, Field::Ptr(p) if p.out && !(command.client_handles && matches!(p.elem, Elem::Handle(_))))) {
        return Ok(());
    }
    let args: Vec<*const c_void> = call.slots.iter().map(|slot| slot as *const u64 as *const c_void).collect();
    let context = Context::Params(&args, command.params);
    for (index, param) in command.params.iter().enumerate() {
        if let Field::Ptr(p) = &param.field
            && p.out
            && !(command.client_handles && matches!(p.elem, Elem::Handle(_)))
        {
            let pointer = call.slots[index] as usize as *const u8;
            unsafe {
                let count = if succeeded { evaluate(&p.len, context, pointer) } else { 0 };
                encode_pointer(w, p, if succeeded { pointer } else { ptr::null() }, count, h)?;
            }
        }
    }
    Ok(())
}
