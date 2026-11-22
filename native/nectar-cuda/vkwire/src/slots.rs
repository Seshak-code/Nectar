//! Arguments and results held in u64 slots: every Vulkan parameter and result fits in 64 bits
//! (scalars, floats, handles and pointers), so one generic interpreter can carry them all.

use nectar_vk_sys::PFN_vkVoidFunction;

pub trait FromSlot {
    fn from_slot(slot: u64) -> Self;
}

pub trait ToSlot {
    fn to_slot(self) -> u64;
}

macro_rules! integers {
    ($($t:ty),*) => {$(
        impl FromSlot for $t {
            fn from_slot(slot: u64) -> Self {
                slot as $t
            }
        }
        impl ToSlot for $t {
            fn to_slot(self) -> u64 {
                self as u64
            }
        }
    )*};
}
integers!(u8, i8, u16, i16, u32, i32, u64, i64, usize);

impl FromSlot for f32 {
    fn from_slot(slot: u64) -> Self {
        f32::from_bits(slot as u32)
    }
}

impl ToSlot for f32 {
    fn to_slot(self) -> u64 {
        self.to_bits() as u64
    }
}

impl<T> FromSlot for *const T {
    fn from_slot(slot: u64) -> Self {
        slot as usize as *const T
    }
}

impl<T> FromSlot for *mut T {
    fn from_slot(slot: u64) -> Self {
        slot as usize as *mut T
    }
}

impl ToSlot for () {
    fn to_slot(self) -> u64 {
        0
    }
}

impl FromSlot for () {
    fn from_slot(_: u64) -> Self {}
}

impl FromSlot for PFN_vkVoidFunction {
    fn from_slot(slot: u64) -> Self {
        unsafe { std::mem::transmute::<usize, PFN_vkVoidFunction>(slot as usize) }
    }
}

impl ToSlot for PFN_vkVoidFunction {
    fn to_slot(self) -> u64 {
        unsafe { std::mem::transmute::<PFN_vkVoidFunction, usize>(self) as u64 }
    }
}
