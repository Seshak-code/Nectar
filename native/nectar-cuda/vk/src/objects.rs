//! Handles on the borrower. Dispatchable ones (instances, physical devices, devices, queues,
//! command buffers) are objects of this driver, which the Vulkan loader writes its dispatch
//! table into; each holds the id the lender knows it by. Other handles are the ids
//! themselves. Ids the borrower chooses count up from 1; the lender's have the top bit set.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use nectar_vk_wire::wire::Handles;

/// What the loader expects in the first word of a dispatchable object it hasn't seen yet
const ICD_LOADER_MAGIC: usize = 0x01CD_C0DE;

#[repr(C)]
pub struct Object {
    /// The loader's dispatch table, once the loader has seen the object
    loader: usize,
    pub id: u64,
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
/// Dispatchable objects by id, so an id from the lender finds the same object
static OBJECTS: Mutex<Option<HashMap<u64, usize>>> = Mutex::new(None);

pub fn new_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

/// The object for an id, made on first sight
pub fn object(id: u64) -> u64 {
    if id == 0 {
        return 0;
    }
    let mut objects = OBJECTS.lock().unwrap_or_else(|p| p.into_inner());
    *objects.get_or_insert_with(HashMap::new).entry(id).or_insert_with(|| {
        Box::into_raw(Box::new(Object { loader: ICD_LOADER_MAGIC, id })) as usize
    }) as u64
}

/// The id of a dispatchable handle
///
/// # Safety
/// `handle` must be 0 or an object of this driver.
pub unsafe fn id_of(handle: u64) -> u64 {
    if handle == 0 { 0 } else { unsafe { (*(handle as usize as *const Object)).id } }
}

pub struct Borrower;

impl Handles for Borrower {
    fn outgoing(&mut self, dispatchable: bool, value: u64) -> u64 {
        if dispatchable { unsafe { id_of(value) } } else { value }
    }

    fn incoming(&mut self, dispatchable: bool, id: u64) -> u64 {
        if dispatchable { object(id) } else { id }
    }

    fn create(&mut self, dispatchable: bool) -> (u64, u64) {
        let id = new_id();
        (id, if dispatchable { object(id) } else { id })
    }

    fn bind(&mut self, _: u64, _: u64) {}
}
