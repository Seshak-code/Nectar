//! Loading GPU code. The image goes to the lender as bytes, so its length has to be found
//! from its own headers, all of them public formats:
//! - the fatbin wrapper the CUDA runtime embeds (fatbinary_section.h, FATBINC_MAGIC)
//! - a fatbin (magic 0xBA55ED50, then its header size and the size of its files)
//! - an ELF cubin (its program and section tables give where the file ends)
//! - PTX text, which ends at its NUL

use std::ffi::{CStr, c_uint, c_void};
use std::slice;

use nectar_cuda_protocol::{Writer, id};
use nectar_cuda_sys::*;

use crate::{call, finish, log};

const FATBINC_MAGIC: u32 = 0x466243B1;
const FATBIN_MAGIC: u32 = 0xBA55ED50;
const ELF_MAGIC: &[u8; 4] = b"\x7fELF";

/// The bytes of a GPU code image, or None for a format that cannot be sent yet
///
/// # Safety
/// `image` must point to an image the driver would accept.
pub(crate) unsafe fn image_bytes<'a>(image: *const c_void) -> Option<&'a [u8]> {
    if image.is_null() {
        return None;
    }

    unsafe {
        match image.cast::<u32>().read_unaligned() {
            FATBINC_MAGIC => {
                // { u32 magic, u32 version, const void* data, const void* filename_or_fatbins }
                let version = image.cast::<u32>().add(1).read_unaligned();
                if version != 1 {
                    // Version 2 lists several fatbins that the driver links together
                    log("a fatbin wrapper of version 2 cannot be loaded yet");
                    return None;
                }
                let data = image.cast::<u8>().add(8).cast::<*const c_void>().read_unaligned();
                fatbin_bytes(data)
            }
            FATBIN_MAGIC => fatbin_bytes(image),
            _ if slice::from_raw_parts(image.cast::<u8>(), 4) == ELF_MAGIC => elf_bytes(image.cast()),
            _ => Some(CStr::from_ptr(image.cast()).to_bytes_with_nul()),
        }
    }
}

/// { u32 magic, u16 version, u16 header size, u64 size of the files that follow }
unsafe fn fatbin_bytes<'a>(data: *const c_void) -> Option<&'a [u8]> {
    unsafe {
        if data.is_null() || data.cast::<u32>().read_unaligned() != FATBIN_MAGIC {
            return None;
        }
        let header_size = data.cast::<u8>().add(6).cast::<u16>().read_unaligned() as usize;
        let files_size = data.cast::<u8>().add(8).cast::<u64>().read_unaligned() as usize;
        Some(slice::from_raw_parts(data.cast(), header_size + files_size))
    }
}

/// An ELF64 file ends where the furthest of its header tables or section contents ends
unsafe fn elf_bytes<'a>(data: *const u8) -> Option<&'a [u8]> {
    const SHT_NOBITS: u32 = 8;

    unsafe {
        let u16_at = |offset: usize| data.add(offset).cast::<u16>().read_unaligned() as usize;
        let u32_at = |offset: usize| data.add(offset).cast::<u32>().read_unaligned();
        let u64_at = |offset: usize| data.add(offset).cast::<u64>().read_unaligned() as usize;

        if *data.add(4) != 2 {
            log("only 64-bit ELF cubins can be loaded");
            return None;
        }

        let (program_offset, program_size, program_count) = (u64_at(0x20), u16_at(0x36), u16_at(0x38));
        let (section_offset, section_size, section_count) = (u64_at(0x28), u16_at(0x3A), u16_at(0x3C));

        let mut end = u16_at(0x34)
            .max(program_offset + program_size * program_count)
            .max(section_offset + section_size * section_count);
        for index in 0..section_count {
            let section = section_offset + index * section_size;
            if u32_at(section + 4) != SHT_NOBITS {
                end = end.max(u64_at(section + 0x18) + u64_at(section + 0x20));
            }
        }

        Some(slice::from_raw_parts(data, end))
    }
}

/// Loads an image as a module on the lender
///
/// # Safety
/// As for cuModuleLoadData.
pub(crate) unsafe fn load_module(module: *mut CUmodule, image: *const c_void) -> CUresult {
    if module.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    let Some(bytes) = (unsafe { image_bytes(image) }) else {
        return CUDA_ERROR_NOT_SUPPORTED;
    };

    let mut request = Writer::new();
    request.bytes(bytes);
    finish(call(id::cuModuleLoadData, &[request.as_slice()]), |reply| {
        unsafe { *module = reply.u64()? as usize as CUmodule };
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult {
    unsafe { load_module(module, image) }
}

/// JIT options only tune or log the PTX compiler, so the lender compiles with its defaults
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuModuleLoadDataEx(
    module: *mut CUmodule,
    image: *const c_void,
    _numOptions: c_uint,
    _options: *mut CUjit_option,
    _optionValues: *mut *mut c_void,
) -> CUresult {
    unsafe { load_module(module, image) }
}

/// The lender keeps its own copy of the code, so the only library option, that the caller
/// keeps the binary alive, does not matter there. JIT options are left to the lender's
/// defaults, as for modules.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuLibraryLoadData(
    library: *mut CUlibrary,
    code: *const c_void,
    _jitOptions: *mut CUjit_option,
    _jitOptionsValues: *mut *mut c_void,
    _numJitOptions: c_uint,
    libraryOptions: *mut CUlibraryOption,
    _libraryOptionValues: *mut *mut c_void,
    numLibraryOptions: c_uint,
) -> CUresult {
    if library.is_null() {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if numLibraryOptions > 0 {
        if libraryOptions.is_null() {
            return CUDA_ERROR_INVALID_VALUE;
        }
        let options = unsafe { slice::from_raw_parts(libraryOptions, numLibraryOptions as usize) };
        if let Some(option) = options.iter().find(|&&option| option != CU_LIBRARY_BINARY_IS_PRESERVED) {
            log(&format!("library option {option} is not supported yet"));
            return CUDA_ERROR_NOT_SUPPORTED;
        }
    }

    let Some(bytes) = (unsafe { image_bytes(code) }) else {
        return CUDA_ERROR_NOT_SUPPORTED;
    };

    let mut request = Writer::new();
    request.bytes(bytes);
    finish(call(id::cuLibraryLoadData, &[request.as_slice()]), |reply| {
        unsafe { *library = reply.u64()? as usize as CUlibrary };
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fatbin_size_comes_from_its_header() {
        let mut fatbin = vec![0u8; 64];
        fatbin[..4].copy_from_slice(&FATBIN_MAGIC.to_le_bytes());
        fatbin[4..6].copy_from_slice(&1u16.to_le_bytes());
        fatbin[6..8].copy_from_slice(&16u16.to_le_bytes());
        fatbin[8..16].copy_from_slice(&40u64.to_le_bytes());

        let bytes = unsafe { image_bytes(fatbin.as_ptr().cast()) }.unwrap();
        assert_eq!(bytes.len(), 56);

        // The runtime's wrapper points at the fatbin
        let wrapper: [usize; 3] = [(1usize << 32) | FATBINC_MAGIC as usize, fatbin.as_ptr() as usize, 0];
        let bytes = unsafe { image_bytes(wrapper.as_ptr().cast()) }.unwrap();
        assert_eq!(bytes.as_ptr(), fatbin.as_ptr());
        assert_eq!(bytes.len(), 56);
    }

    #[test]
    fn ptx_ends_at_its_nul() {
        let ptx = b".version 8.0\n.target sm_120\n\0trailing";
        let bytes = unsafe { image_bytes(ptx.as_ptr().cast()) }.unwrap();
        assert_eq!(bytes, b".version 8.0\n.target sm_120\n\0");
    }

    #[test]
    fn elf_ends_after_its_last_section() {
        let mut elf = vec![0u8; 0x200];
        elf[..4].copy_from_slice(ELF_MAGIC);
        elf[4] = 2;
        elf[0x34..0x36].copy_from_slice(&64u16.to_le_bytes());
        // One section header at 0x100, 64 bytes, describing contents at 0x80..0x1C0
        elf[0x28..0x30].copy_from_slice(&0x100u64.to_le_bytes());
        elf[0x3A..0x3C].copy_from_slice(&64u16.to_le_bytes());
        elf[0x3C..0x3E].copy_from_slice(&1u16.to_le_bytes());
        elf[0x100 + 4..0x100 + 8].copy_from_slice(&1u32.to_le_bytes());
        elf[0x100 + 0x18..0x100 + 0x20].copy_from_slice(&0x80u64.to_le_bytes());
        elf[0x100 + 0x20..0x100 + 0x28].copy_from_slice(&0x140u64.to_le_bytes());

        let bytes = unsafe { image_bytes(elf.as_ptr().cast()) }.unwrap();
        assert_eq!(bytes.len(), 0x1C0);
    }
}
