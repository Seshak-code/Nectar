//! Kernel launches. Each kernelParams[i] points at argument i, copied to the lender by value.
//! Argument sizes come from the lender's driver (cuFuncGetParamInfo), once per kernel.
//! The other way to pass arguments, one buffer through `extra`, is sent as that buffer.
//!
//! cuLaunchKernelEx takes its launch settings in a struct with a list of attributes, such as
//! the programmatic stream serialization llama.cpp asks for. Every attribute value is plain
//! data or a lender handle, so the list is sent as it is.

use std::collections::HashMap;
use std::ffi::{c_uint, c_void};
use std::slice;
use std::sync::Mutex;

use nectar_cuda_protocol::{self as protocol, Reader, Writer, id};
use nectar_cuda_sys::*;

use crate::{call, finish};

/// Argument sizes of each kernel seen, by the lender's handle
fn argument_sizes(function: CUfunction) -> Result<Vec<u32>, CUresult> {
    static SIZES: Mutex<Option<HashMap<usize, Vec<u32>>>> = Mutex::new(None);

    let key = function as usize;
    if let Some(sizes) = SIZES
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get_or_insert_with(HashMap::new)
        .get(&key)
    {
        return Ok(sizes.clone());
    }

    let mut request = Writer::new();
    request.u64(key as u64);
    let reply = call(protocol::PARAM_LAYOUT, &[request.as_slice()])?;

    let mut reader = Reader::new(&reply);
    let decode = |reader: &mut Reader| -> std::io::Result<Vec<u32>> {
        let count = reader.u32()?;
        (0..count).map(|_| reader.u32()).collect()
    };
    let sizes = decode(&mut reader).map_err(|_| CUDA_ERROR_UNKNOWN)?;

    SIZES
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get_or_insert_with(HashMap::new)
        .insert(key, sizes.clone());
    Ok(sizes)
}

/// The argument buffer described by `extra`: CU_LAUNCH_PARAM_BUFFER_POINTER, the buffer,
/// CU_LAUNCH_PARAM_BUFFER_SIZE, a pointer to its size, then CU_LAUNCH_PARAM_END
unsafe fn extra_buffer<'a>(extra: *mut *mut c_void) -> Option<&'a [u8]> {
    let (mut buffer, mut size) = (None, None);
    let mut entry = extra;
    unsafe {
        loop {
            match *entry as usize as u32 {
                CU_LAUNCH_PARAM_END_AS_INT => break,
                CU_LAUNCH_PARAM_BUFFER_POINTER_AS_INT => buffer = Some(*entry.add(1) as *const u8),
                CU_LAUNCH_PARAM_BUFFER_SIZE_AS_INT => size = Some(*(*entry.add(1) as *const usize)),
                _ => return None,
            }
            entry = entry.add(2);
        }
        Some(slice::from_raw_parts(buffer?, size?))
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn launch(
    function_id: u32,
    function: CUfunction,
    grid: [c_uint; 3],
    block: [c_uint; 3],
    shared_memory: c_uint,
    stream: CUstream,
    attributes: &[CUlaunchAttribute],
    params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> CUresult {
    let mut request = Writer::new();
    request.u64(function as u64);
    for dimension in grid.into_iter().chain(block) {
        request.u32(dimension);
    }
    request.u32(shared_memory).u64(stream as u64);
    request.u32(attributes.len() as u32);
    for attribute in attributes {
        request.value(attribute);
    }

    if !params.is_null() {
        let sizes = match argument_sizes(function) {
            Ok(sizes) => sizes,
            Err(result) => return result,
        };
        request.u8(1).u32(sizes.len() as u32);
        for (index, &size) in sizes.iter().enumerate() {
            let argument = unsafe { *params.add(index) };
            if argument.is_null() {
                return CUDA_ERROR_INVALID_VALUE;
            }
            request.bytes(unsafe { slice::from_raw_parts(argument.cast::<u8>(), size as usize) });
        }
    } else if !extra.is_null() {
        let Some(buffer) = (unsafe { extra_buffer(extra) }) else {
            return CUDA_ERROR_INVALID_VALUE;
        };
        request.u8(2).bytes(buffer);
    } else {
        request.u8(0);
    }

    finish(call(function_id, &[request.as_slice()]), |_| Ok(()))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuLaunchKernel(
    f: CUfunction,
    gridDimX: c_uint,
    gridDimY: c_uint,
    gridDimZ: c_uint,
    blockDimX: c_uint,
    blockDimY: c_uint,
    blockDimZ: c_uint,
    sharedMemBytes: c_uint,
    hStream: CUstream,
    kernelParams: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> CUresult {
    unsafe {
        launch(
            id::cuLaunchKernel,
            f,
            [gridDimX, gridDimY, gridDimZ],
            [blockDimX, blockDimY, blockDimZ],
            sharedMemBytes,
            hStream,
            &[],
            kernelParams,
            extra,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuLaunchKernel_ptsz(
    f: CUfunction,
    gridDimX: c_uint,
    gridDimY: c_uint,
    gridDimZ: c_uint,
    blockDimX: c_uint,
    blockDimY: c_uint,
    blockDimZ: c_uint,
    sharedMemBytes: c_uint,
    hStream: CUstream,
    kernelParams: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> CUresult {
    unsafe {
        launch(
            id::cuLaunchKernel_ptsz,
            f,
            [gridDimX, gridDimY, gridDimZ],
            [blockDimX, blockDimY, blockDimZ],
            sharedMemBytes,
            hStream,
            &[],
            kernelParams,
            extra,
        )
    }
}

unsafe fn launch_with_config(
    function_id: u32,
    config: *const CUlaunchConfig,
    function: CUfunction,
    params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> CUresult {
    let Some(config) = (unsafe { config.as_ref() }) else {
        return CUDA_ERROR_INVALID_VALUE;
    };
    let attributes = match (config.numAttrs, config.attrs.is_null()) {
        (0, _) => &[][..],
        (_, true) => return CUDA_ERROR_INVALID_VALUE,
        (count, false) => unsafe { slice::from_raw_parts(config.attrs, count as usize) },
    };
    unsafe {
        launch(
            function_id,
            function,
            [config.gridDimX, config.gridDimY, config.gridDimZ],
            [config.blockDimX, config.blockDimY, config.blockDimZ],
            config.sharedMemBytes,
            config.hStream,
            attributes,
            params,
            extra,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuLaunchKernelEx(
    config: *const CUlaunchConfig,
    f: CUfunction,
    kernelParams: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> CUresult {
    unsafe { launch_with_config(id::cuLaunchKernelEx, config, f, kernelParams, extra) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuLaunchKernelEx_ptsz(
    config: *const CUlaunchConfig,
    f: CUfunction,
    kernelParams: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> CUresult {
    unsafe { launch_with_config(id::cuLaunchKernelEx_ptsz, config, f, kernelParams, extra) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extra_describes_one_buffer() {
        let mut arguments = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let mut size = arguments.len();
        let mut extra = [
            CU_LAUNCH_PARAM_BUFFER_POINTER_AS_INT as usize as *mut c_void,
            arguments.as_mut_ptr().cast(),
            CU_LAUNCH_PARAM_BUFFER_SIZE_AS_INT as usize as *mut c_void,
            (&mut size as *mut usize).cast(),
            CU_LAUNCH_PARAM_END_AS_INT as usize as *mut c_void,
        ];
        let buffer = unsafe { extra_buffer(extra.as_mut_ptr()) }.unwrap();
        assert_eq!(buffer, &arguments);
    }
}

