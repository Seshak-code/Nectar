//! Which calls wait for the lender.
//!
//! Over Wi-Fi a round trip takes milliseconds. llama.cpp makes about 60 calls for each
//! token it generates and over a thousand for each pass over a prompt, and nearly all of
//! them need no answer from the lender:
//!
//! - Calls that only queue work on the GPU are sent without waiting: kernel and graph
//!   launches, memsets, copies from GPU memory or from bytes the request already carries,
//!   event records and stream waits. They return CUDA_SUCCESS at once. The lender carries
//!   them out in order, and an error one of them meets is reported by the next synchronize,
//!   as errors of work running on a GPU are.
//! - cuStreamSynchronize waits only when work sent may still be running on the GPU, or a
//!   copy to host memory is on its way. Otherwise it is still sent, so the lender keeps the
//!   same order, but not waited for. This is tracked for the whole process, not per stream.
//!   cuCtxSynchronize always waits, since programs use it to time their work.
//! - Answers that cannot change are kept: the driver version and devices, device and kernel
//!   attributes (all of one device's or kernel's in one round trip), occupancy, and the
//!   current context and its device, until a call that could change them.
//! - Uploads of 1 MiB or more to a lender over the network go by content hash, so the bytes
//!   cross once and the lender keeps them for the next load (protocol::UPLOAD).
//!
//! Requests that are not waited for go out together: when a call waits, once 16 KiB have
//! gathered, or after a millisecond (lib.rs). Whatever the host learns about the GPU's work
//! comes back in replies read in order, so it sees the same results as with every call
//! waited for.

use nectar_cuda_protocol::id;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// Sends the request and waits for the reply
    Wait,
    /// Sends the request without waiting. `runs`: the GPU may still be running its work
    /// once the lender has carried it out.
    Send { runs: bool },
    /// Waits when work may be running or a copy to host memory is on its way, or `always`.
    /// Reports the first error of a request sent without waiting.
    Synchronize { always: bool },
    /// Keeps the answer, keyed by the request after its first parameter, which is the
    /// `size`-byte value it writes (see `key`)
    Keep { size: usize },
    /// Forgets every kept answer, then waits
    Forget,
}

pub(crate) fn kind(function: u32) -> Kind {
    match function {
        id::cuLaunchKernel
        | id::cuLaunchKernel_ptsz
        | id::cuLaunchKernelEx
        | id::cuLaunchKernelEx_ptsz
        | id::cuGraphLaunch
        | id::cuGraphLaunch_ptsz
        | id::cuMemcpyDtoD_v2
        | id::cuMemcpyDtoD_v2_ptds
        | id::cuMemcpyDtoDAsync_v2
        | id::cuMemcpyDtoDAsync_v2_ptsz
        | id::cuMemcpyAsync
        | id::cuMemcpyAsync_ptsz
        | id::cuEventRecord
        | id::cuEventRecord_ptsz
        | id::cuEventRecordWithFlags
        | id::cuEventRecordWithFlags_ptsz
        | id::cuStreamWaitEvent
        | id::cuStreamWaitEvent_ptsz => Kind::Send { runs: true },

        // The request carries the bytes, and the lender waits for asynchronous copies of
        // them itself (server/src/main.rs). Memsets take microseconds: a synchronize after
        // them alone, as llama.cpp makes for each tensor it loads, need not wait.
        id::cuMemsetD8_v2
        | id::cuMemsetD8_v2_ptds
        | id::cuMemsetD16_v2
        | id::cuMemsetD16_v2_ptds
        | id::cuMemsetD32_v2
        | id::cuMemsetD32_v2_ptds
        | id::cuMemsetD8Async
        | id::cuMemsetD8Async_ptsz
        | id::cuMemsetD16Async
        | id::cuMemsetD16Async_ptsz
        | id::cuMemsetD32Async
        | id::cuMemsetD32Async_ptsz
        | id::cuMemsetD2D8_v2
        | id::cuMemsetD2D8_v2_ptds
        | id::cuMemsetD2D16_v2
        | id::cuMemsetD2D32_v2
        | id::cuMemsetD2D8Async
        | id::cuMemsetD2D8Async_ptsz
        | id::cuMemsetD2D16Async
        | id::cuMemsetD2D32Async
        | id::cuMemcpyHtoD_v2
        | id::cuMemcpyHtoD_v2_ptds
        | id::cuMemcpyHtoDAsync_v2
        | id::cuMemcpyHtoDAsync_v2_ptsz
        | id::cuStreamBeginCapture
        | id::cuStreamBeginCapture_ptsz
        | id::cuStreamBeginCapture_v2
        | id::cuStreamBeginCapture_v2_ptsz
        | id::cuGraphDestroy
        | id::cuGraphExecDestroy
        | id::cuEventDestroy_v2 => Kind::Send { runs: false },

        id::cuStreamSynchronize | id::cuStreamSynchronize_ptsz => {
            Kind::Synchronize { always: false }
        }
        id::cuCtxSynchronize | id::cuCtxSynchronize_v2 => Kind::Synchronize { always: true },

        // The CUDA runtime asks for the current context before nearly every call
        id::cuCtxGetCurrent => Kind::Keep { size: 8 },
        // Device and kernel attributes are fetched all at once on the first miss (lib.rs)
        id::cuCtxGetDevice
        | id::cuCtxGetDevice_v2
        | id::cuDriverGetVersion
        | id::cuDeviceGet
        | id::cuDeviceGetCount
        | id::cuDeviceGetAttribute
        | id::cuFuncGetAttribute
        | id::cuKernelGetAttribute
        | id::cuOccupancyMaxActiveBlocksPerMultiprocessor
        | id::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags => Kind::Keep { size: 4 },

        // Change the current context, attributes or occupancy, or free handles a kept
        // answer may name
        id::cuCtxCreate_v2
        | id::cuCtxCreate_v3
        | id::cuCtxCreate_v4
        | id::cuCtxDestroy_v2
        | id::cuCtxSetCurrent
        | id::cuCtxPushCurrent_v2
        | id::cuCtxPopCurrent_v2
        | id::cuDevicePrimaryCtxRelease
        | id::cuDevicePrimaryCtxRelease_v2
        | id::cuDevicePrimaryCtxReset
        | id::cuDevicePrimaryCtxReset_v2
        | id::cuGreenCtxDestroy
        | id::cuCtxSetCacheConfig
        | id::cuCtxSetSharedMemConfig
        | id::cuCtxSetLimit
        | id::cuFuncSetAttribute
        | id::cuFuncSetCacheConfig
        | id::cuFuncSetSharedMemConfig
        | id::cuFuncSetBlockShape
        | id::cuFuncSetSharedSize
        | id::cuKernelSetAttribute
        | id::cuKernelSetCacheConfig
        | id::cuModuleUnload
        | id::cuLibraryUnload => Kind::Forget,

        _ => Kind::Wait,
    }
}

/// What identifies the answer to a Keep request: everything after its first parameter. That
/// parameter is the only one these functions write, a value of `size` bytes, which the
/// generated code sends as an optional value. None when its pointer is NULL, which the
/// lender refuses.
pub(crate) fn key(request: &[u8], size: usize) -> Option<&[u8]> {
    match request.first() {
        Some(1) => request.get(1 + size..),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nectar_cuda_protocol::Writer;
    use nectar_cuda_sys::CUcontext;
    use std::ptr;

    #[test]
    fn the_key_leaves_out_the_value_written() {
        // As generated.rs encodes cuDeviceGetAttribute(pi, attrib, dev)
        let encode = |value: i32| {
            let mut request = Writer::new();
            unsafe { request.optional(&value as *const i32) };
            request.value(&16i32).value(&0i32);
            request.as_slice().to_vec()
        };
        let (first, second) = (encode(7), encode(-1));
        assert_eq!(key(&first, 4), Some(&[16, 0, 0, 0, 0, 0, 0, 0][..]));
        assert_eq!(key(&first, 4), key(&second, 4));

        let mut null = Writer::new();
        unsafe { null.optional(ptr::null::<i32>()) };
        null.value(&16i32).value(&0i32);
        assert_eq!(key(null.as_slice(), 4), None);

        // cuCtxGetCurrent(pctx) has nothing after its 8-byte context
        let context = 0x1234usize as CUcontext;
        let mut current = Writer::new();
        unsafe { current.optional(&context as *const CUcontext) };
        assert_eq!(key(current.as_slice(), 8), Some(&[][..]));
    }

    #[test]
    fn calls_are_classified() {
        assert_eq!(kind(id::cuCtxGetDevice_v2), Kind::Keep { size: 4 });
        assert_eq!(kind(id::cuCtxGetCurrent), Kind::Keep { size: 8 });
        assert_eq!(kind(id::cuCtxSetCurrent), Kind::Forget);
        assert_eq!(kind(id::cuMemcpyDtoH_v2), Kind::Wait);
        assert_eq!(kind(id::cuMemcpy), Kind::Wait);
    }
}
