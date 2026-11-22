//! What the generated tables say about each Vulkan struct and command, for the interpreter in
//! wire.rs to encode and decode them. nectar-vk-codegen writes the tables (generated.rs).

use std::ffi::c_void;

/// A struct's position in TYPES
pub type Index = u16;

/// How many elements a pointer points to
#[derive(Clone, Copy, Debug)]
pub enum Len {
    One,
    /// The value of another member (of the same struct) or parameter
    Sibling(u16),
    /// The value another parameter points to (a count passed by pointer)
    SiblingDeref(u16),
    /// A member of the struct another parameter points to: parameter, offset, width
    SiblingField(u16, usize, u8),
    /// A C string's length
    NullTerminated,
    /// A C array parameter of fixed size, such as blendConstants[4]
    Fixed(usize),
    /// An expression over the struct's members, such as `codeSize / 4`
    StructExpr(fn(*const u8) -> usize),
    /// An expression over the parameters
    ParamExpr(fn(&[*const c_void]) -> usize),
}

/// What one element behind a pointer is
#[derive(Clone, Copy, Debug)]
pub enum Elem {
    /// Copied as it is: scalars, enums, flags, data
    Raw(usize),
    /// size_t, sent as 64 bits
    SizeT,
    /// A handle; true when dispatchable (a pointer to the driver's object)
    Handle(bool),
    Struct(Index),
    /// Pointers to further elements, such as an array of strings
    Ptr(&'static Pointer),
    /// Can't be sent: the reason
    Unsupported(&'static str),
}

#[derive(Debug)]
pub struct Pointer {
    pub elem: Elem,
    pub len: Len,
    /// The pointer's target is written by the call (not const)
    pub out: bool,
    /// An output whose value is also read: a count the caller sets to the array's capacity
    pub inout: bool,
}

/// A struct member or a command parameter
#[derive(Clone, Copy, Debug)]
pub enum Field {
    SType,
    Next,
    /// Copied as it is: scalars, fixed arrays of them, unions, bitfields
    Raw(usize),
    SizeT,
    Handle(bool),
    /// A fixed array of handles
    HandleArray(bool, usize),
    Struct(Index),
    StructArray(Index, usize),
    Ptr(&'static Pointer),
    /// A float passed by value, which a call carries in a different register
    Float,
    Unsupported(&'static str),
}

#[derive(Debug)]
pub struct Member {
    pub name: &'static str,
    pub offset: usize,
    pub field: Field,
}

#[derive(Debug)]
pub struct TypeInfo {
    pub name: &'static str,
    pub size: usize,
    /// Its VkStructureType value, or 0 when it has none
    pub stype: i32,
    pub members: &'static [Member],
}

/// Whose dispatch a command goes through
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    /// vkCreateInstance, vkEnumerateInstance*: the loader's own
    Global,
    /// The first parameter is a VkInstance or VkPhysicalDevice
    Instance,
    /// The first parameter is a VkDevice, VkQueue or VkCommandBuffer
    Device,
}

/// How the borrower sends a command
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Sent without waiting: no outputs besides handles the borrower chooses
    Async,
    /// Sent, then its answer awaited
    Sync,
    /// Waits for the GPU: the lender runs it beside the ordered stream, so a wait can't hold
    /// up the work it waits for
    Wait,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ret {
    Void,
    Result,
    U32,
    U64,
    PFN,
}

#[derive(Debug)]
pub struct Param {
    pub name: &'static str,
    pub field: Field,
}

#[derive(Debug)]
pub struct CommandInfo {
    pub name: &'static str,
    /// protocol::function_id of the name
    pub id: u32,
    pub params: &'static [Param],
    pub ret: Ret,
    pub level: Level,
    pub mode: Mode,
    /// Output handles are chosen by the borrower (vkCreate*, vkAllocate*), so the call can be
    /// sent without waiting; otherwise the lender names them
    pub client_handles: bool,
    /// Why it can't be carried, when it can't
    pub unsupported: Option<&'static str>,
}
