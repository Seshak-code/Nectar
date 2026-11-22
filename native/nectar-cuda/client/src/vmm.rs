//! CUDA's virtual memory management (cuMemCreate, cuMemMap, cuMemSetAccess...), which memory
//! pools such as llama.cpp's use. Most of it is plain values and is generated. These take an
//! allocation properties struct, whose one pointer, win32HandleMetaData, only matters for
//! sharing memory between processes, or an array of access descriptors.
//!
//! Also cuDeviceGetPCIBusId, which fills a string buffer like cuDeviceGetName.

use std::ffi::{c_char, c_int, c_ulonglong};
use std::{ptr, slice};

use nectar_cuda_protocol::{Writer, id};
use nectar_cuda_sys::*;

use crate::{call, finish, log};

/// The properties to send, or why they cannot be
fn properties(prop: *const CUmemAllocationProp) -> Result<CUmemAllocationProp, CUresult> {
    if prop.is_null() {
        return Err(CUDA_ERROR_INVALID_VALUE);
    }
    let prop = unsafe { prop.read_unaligned() };
    if !prop.win32HandleMetaData.is_null() {
        log("memory shared between processes (win32HandleMetaData) is not supported yet");
        return Err(CUDA_ERROR_NOT_SUPPORTED);
    }
    Ok(prop)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemGetAllocationGranularity(
    granularity: *mut usize,
    prop: *const CUmemAllocationProp,
    option: CUmemAllocationGranularity_flags,
) -> CUresult {
    if granularity.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let prop = match properties(prop) {
        Ok(prop) => prop,
        Err(error) => return error,
    };

    let mut request = Writer::new();
    request.value(&prop).value(&option);
    finish(
        call(id::cuMemGetAllocationGranularity, &[request.as_slice()]),
        |reply| {
            unsafe { *granularity = reply.u64()? as usize };
            Ok(())
        },
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemCreate(
    handle: *mut CUmemGenericAllocationHandle,
    size: usize,
    prop: *const CUmemAllocationProp,
    flags: c_ulonglong,
) -> CUresult {
    if handle.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let prop = match properties(prop) {
        Ok(prop) => prop,
        Err(error) => return error,
    };

    let mut request = Writer::new();
    request.u64(size as u64).value(&prop).u64(flags);
    finish(call(id::cuMemCreate, &[request.as_slice()]), |reply| {
        unsafe { *handle = reply.u64()? };
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemSetAccess(
    ptr: CUdeviceptr,
    size: usize,
    desc: *const CUmemAccessDesc,
    count: usize,
) -> CUresult {
    if desc.is_null() && count > 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let descriptors = if count == 0 {
        &[][..]
    } else {
        unsafe { slice::from_raw_parts(desc, count) }
    };

    let mut request = Writer::new();
    request.u64(ptr).u64(size as u64).u64(count as u64);
    for descriptor in descriptors {
        request.value(descriptor);
    }
    finish(call(id::cuMemSetAccess, &[request.as_slice()]), |_| Ok(()))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuDeviceGetPCIBusId(
    pciBusId: *mut c_char,
    len: c_int,
    dev: CUdevice,
) -> CUresult {
    if pciBusId.is_null() || len <= 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }

    let mut request = Writer::new();
    request.i32(len).i32(dev);
    finish(
        call(id::cuDeviceGetPCIBusId, &[request.as_slice()]),
        |reply| {
            let text = reply.bytes()?;
            let count = text.len().min(len as usize - 1);
            unsafe {
                ptr::copy_nonoverlapping(text.as_ptr(), pciBusId.cast(), count);
                *pciBusId.add(count) = 0;
            }
            Ok(())
        },
    )
}
