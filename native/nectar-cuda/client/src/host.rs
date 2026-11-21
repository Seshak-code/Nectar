//! Host memory. Pinned memory exists so the GPU can copy to and from it directly; on a
//! borrowed GPU every copy crosses the network anyway, so "pinned" memory is ordinary memory
//! on the borrower, page aligned. Registering memory as pinned has nothing to do.
//!
//! Copies between host memory and the GPU on a stream take effect in the stream's order on
//! the lender, which copies and then waits for the stream. Neither waits on the borrower
//! (calls.rs): a copy to the GPU sends the bytes as they are when it is called, and a copy
//! from the GPU writes host memory when its reply is read, by the synchronize the program
//! needs before it reads them.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::collections::HashMap;
use std::ffi::{c_uint, c_void};
use std::sync::Mutex;
use std::slice;

use nectar_cuda_protocol::{Writer, id};
use nectar_cuda_sys::*;

use crate::{call, copy_to_host, finish, log};

/// Address to (layout, flags) of each block handed out
static BLOCKS: Mutex<Option<HashMap<usize, (Layout, c_uint)>>> = Mutex::new(None);

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemHostAlloc(pp: *mut *mut c_void, bytesize: usize, Flags: c_uint) -> CUresult {
    if pp.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if Flags & CU_MEMHOSTALLOC_DEVICEMAP != 0 {
        log("mapped host memory (CU_MEMHOSTALLOC_DEVICEMAP) is not supported yet; the memory is not visible to the GPU");
    }

    let Ok(layout) = Layout::from_size_align(bytesize.max(1), 4096) else {
        return CUDA_ERROR_INVALID_VALUE;
    };
    let block = unsafe { alloc_zeroed(layout) };
    if block.is_null() {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }

    BLOCKS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get_or_insert_with(HashMap::new)
        .insert(block as usize, (layout, Flags));
    unsafe { *pp = block.cast() };
    CUDA_SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemAllocHost_v2(pp: *mut *mut c_void, bytesize: usize) -> CUresult {
    unsafe { cuMemHostAlloc(pp, bytesize, 0) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemFreeHost(p: *mut c_void) -> CUresult {
    let block = BLOCKS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get_or_insert_with(HashMap::new)
        .remove(&(p as usize));
    match block {
        Some((layout, _)) => {
            unsafe { dealloc(p.cast(), layout) };
            CUDA_SUCCESS
        }
        None => CUDA_ERROR_INVALID_VALUE,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemHostGetFlags(pFlags: *mut c_uint, p: *mut c_void) -> CUresult {
    if pFlags.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let blocks = BLOCKS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    match blocks.as_ref().and_then(|blocks| blocks.get(&(p as usize))) {
        Some(&(_, flags)) => {
            unsafe { *pFlags = flags };
            CUDA_SUCCESS
        }
        None => CUDA_ERROR_INVALID_VALUE,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemHostRegister_v2(p: *mut c_void, bytesize: usize, _Flags: c_uint) -> CUresult {
    if p.is_null() || bytesize == 0 { CUDA_ERROR_INVALID_VALUE } else { CUDA_SUCCESS }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemHostUnregister(p: *mut c_void) -> CUresult {
    if p.is_null() { CUDA_ERROR_INVALID_VALUE } else { CUDA_SUCCESS }
}

fn memcpy_htod_async(function: u32, destination: CUdeviceptr, source: *const c_void, size: usize, stream: CUstream) -> CUresult {
    if source.is_null() && size > 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }

    // The bytes are taken now, which is always safe: the caller may not change them before
    // the copy happens, and may change them after
    let data = if size == 0 { &[][..] } else { unsafe { slice::from_raw_parts(source.cast::<u8>(), size) } };
    let mut head = Writer::new();
    head.u64(destination).u64(stream as u64);
    finish(
        upload(function, head.as_slice(), destination, stream as u64, data),
        |_| Ok(()),
    )
}

fn memcpy_dtoh_async(function: u32, destination: *mut c_void, source: CUdeviceptr, size: usize, stream: CUstream) -> CUresult {
    if destination.is_null() && size > 0 {
        return CUDA_ERROR_INVALID_VALUE;
    }

    let mut request = Writer::new();
    request.u64(source).u64(size as u64).u64(stream as u64);
    copy_to_host(function, request.as_slice(), destination, size)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpyHtoDAsync_v2(
    dstDevice: CUdeviceptr,
    srcHost: *const c_void,
    ByteCount: usize,
    hStream: CUstream,
) -> CUresult {
    memcpy_htod_async(id::cuMemcpyHtoDAsync_v2, dstDevice, srcHost, ByteCount, hStream)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpyHtoDAsync_v2_ptsz(
    dstDevice: CUdeviceptr,
    srcHost: *const c_void,
    ByteCount: usize,
    hStream: CUstream,
) -> CUresult {
    memcpy_htod_async(id::cuMemcpyHtoDAsync_v2_ptsz, dstDevice, srcHost, ByteCount, hStream)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpyDtoHAsync_v2(
    dstHost: *mut c_void,
    srcDevice: CUdeviceptr,
    ByteCount: usize,
    hStream: CUstream,
) -> CUresult {
    memcpy_dtoh_async(id::cuMemcpyDtoHAsync_v2, dstHost, srcDevice, ByteCount, hStream)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpyDtoHAsync_v2_ptsz(
    dstHost: *mut c_void,
    srcDevice: CUdeviceptr,
    ByteCount: usize,
    hStream: CUstream,
) -> CUresult {
    memcpy_dtoh_async(id::cuMemcpyDtoHAsync_v2_ptsz, dstHost, srcDevice, ByteCount, hStream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    #[test]
    fn host_memory_is_page_aligned_and_freed_once() {
        let mut block = ptr::null_mut();
        unsafe {
            assert_eq!(cuMemHostAlloc(&mut block, 100, 0), CUDA_SUCCESS);
            assert_eq!(block as usize % 4096, 0);
            let mut flags = 7;
            assert_eq!(cuMemHostGetFlags(&mut flags, block), CUDA_SUCCESS);
            assert_eq!(flags, 0);
            assert_eq!(cuMemFreeHost(block), CUDA_SUCCESS);
            assert_eq!(cuMemFreeHost(block), CUDA_ERROR_INVALID_VALUE);
        }
    }
}
