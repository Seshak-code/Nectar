//! HMAC-SHA256 and random bytes from Windows' own cryptography (BCrypt), so no crate is
//! needed for them.

use windows_sys::Win32::Security::Cryptography::{
    BCRYPT_HMAC_SHA256_ALG_HANDLE, BCRYPT_USE_SYSTEM_PREFERRED_RNG, BCryptGenRandom, BCryptHash,
};

pub const MAC_SIZE: usize = 32;

/// HMAC-SHA256 of `message` under `key`
pub fn hmac(key: &[u8], message: &[u8]) -> [u8; MAC_SIZE] {
    let mut mac = [0u8; MAC_SIZE];
    let status = unsafe {
        BCryptHash(
            BCRYPT_HMAC_SHA256_ALG_HANDLE,
            key.as_ptr(),
            key.len() as u32,
            message.as_ptr(),
            message.len() as u32,
            mac.as_mut_ptr(),
            MAC_SIZE as u32,
        )
    };
    // Fails only for arguments that cannot happen here, or on Windows before 10
    assert!(status >= 0, "BCryptHash failed with {status:#x}");
    mac
}

/// Random bytes from the system's generator
pub fn random<const N: usize>() -> [u8; N] {
    let mut bytes = [0u8; N];
    let status = unsafe {
        BCryptGenRandom(
            std::ptr::null_mut(),
            bytes.as_mut_ptr(),
            N as u32,
            BCRYPT_USE_SYSTEM_PREFERRED_RNG,
        )
    };
    assert!(status >= 0, "BCryptGenRandom failed with {status:#x}");
    bytes
}

/// Compares in time that does not depend on where the bytes differ
pub fn same(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub fn unhex(text: &str) -> Option<Vec<u8>> {
    if !text.len().is_multiple_of(2) {
        return None;
    }
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(text.get(i..i + 2)?, 16).ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // RFC 4231 test cases 1, 2 and 6 (a key longer than the block)
    #[test]
    fn hmac_matches_rfc_4231() {
        assert_eq!(
            hex(&hmac(&[0x0b; 20], b"Hi There")),
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
        );
        assert_eq!(
            hex(&hmac(b"Jefe", b"what do ya want for nothing?")),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
        assert_eq!(
            hex(&hmac(
                &[0xaa; 131],
                b"Test Using Larger Than Block-Size Key - Hash Key First"
            )),
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54"
        );
    }

    #[test]
    fn random_bytes_differ_and_hex_round_trips() {
        let (a, b) = (random::<16>(), random::<16>());
        assert_ne!(a, b);
        assert_eq!(unhex(&hex(&a)).unwrap(), a);
        assert_eq!(unhex("abc"), None);
        assert_eq!(unhex("zz"), None);
        assert!(same(b"abc", b"abc"));
        assert!(!same(b"abc", b"abd"));
        assert!(!same(b"abc", b"ab"));
    }
}
