//! Uploads kept by content hash (protocol::UPLOAD), so the next load of the same model sends
//! only hashes over the network.
//!
//! Each upload is a file in the cache folder named by its hash. When the server starts, the
//! least recently used files are removed until the folder fits its limit. Nothing is removed
//! while it runs, so a borrower can rely on the index it was given.
//!
//! The hash of every upload is checked before it is kept, so a borrower cannot store bytes
//! under another upload's hash.

use std::collections::HashSet;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::SystemTime;

use nectar_cuda_protocol as protocol;

pub(crate) type Hash = [u8; 16];

pub(crate) struct Cache {
    folder: PathBuf,
    kept: Mutex<HashSet<Hash>>,
}

impl Cache {
    /// Opens the folder, keeping the most recently used files that fit in `limit` bytes.
    /// Returns the cache and the bytes it holds.
    pub(crate) fn open(folder: &Path, limit: u64) -> io::Result<(Self, u64)> {
        fs::create_dir_all(folder)?;

        let mut files = Vec::new();
        for entry in fs::read_dir(folder)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().into_owned();
            match name.strip_suffix(".bin").and_then(parse) {
                Some(hash) => {
                    let metadata = entry.metadata()?;
                    let used = metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH);
                    files.push((used, metadata.len(), hash, entry.path()));
                }
                // Left by an upload cut short
                None if name.ends_with(".part") => {
                    let _ = fs::remove_file(entry.path());
                }
                None => {}
            }
        }

        files.sort_by(|a, b| b.0.cmp(&a.0));
        let (mut kept, mut total) = (HashSet::new(), 0u64);
        for (_, size, hash, path) in files {
            if total + size <= limit {
                total += size;
                kept.insert(hash);
            } else {
                fs::remove_file(path)?;
            }
        }

        let cache = Self {
            folder: folder.to_path_buf(),
            kept: Mutex::new(kept),
        };
        Ok((cache, total))
    }

    fn kept(&self) -> std::sync::MutexGuard<'_, HashSet<Hash>> {
        self.kept
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn path(&self, hash: &Hash) -> PathBuf {
        let name: String = hash.iter().map(|byte| format!("{byte:02x}")).collect();
        self.folder.join(format!("{name}.bin"))
    }

    pub(crate) fn index(&self) -> Vec<Hash> {
        self.kept().iter().copied().collect()
    }

    /// The bytes kept under `hash`, which must be `size` long
    pub(crate) fn get(&self, hash: &Hash, size: u64) -> io::Result<Vec<u8>> {
        let path = self.path(hash);
        let data = fs::read(&path)?;
        if data.len() as u64 != size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{} holds {} bytes, not {size}", path.display(), data.len()),
            ));
        }
        // Marks it used, for the next trim
        let _ = File::options()
            .write(true)
            .open(&path)
            .and_then(|file| file.set_modified(SystemTime::now()));
        Ok(data)
    }

    /// Keeps `data` under `hash`, if that is its hash and it is not kept already
    pub(crate) fn put(&self, hash: &Hash, data: &[u8]) -> io::Result<()> {
        if self.kept().contains(hash) {
            return Ok(());
        }
        if protocol::content_hash(data) != *hash {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "an upload's bytes do not match its hash",
            ));
        }

        let path = self.path(hash);
        let part = path.with_extension("part");
        fs::write(&part, data)?;
        fs::rename(&part, &path)?;
        self.kept().insert(*hash);
        Ok(())
    }
}

fn parse(name: &str) -> Option<Hash> {
    if name.len() != 32 {
        return None;
    }
    let mut hash = [0u8; 16];
    for (i, byte) in hash.iter_mut().enumerate() {
        *byte = u8::from_str_radix(name.get(2 * i..2 * i + 2)?, 16).ok()?;
    }
    Some(hash)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uploads_are_kept_checked_and_trimmed() {
        let folder = std::env::temp_dir().join(format!("nectar-cache-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&folder);

        let (cache, total) = Cache::open(&folder, 1 << 20).unwrap();
        assert_eq!(total, 0);

        let data = vec![5u8; 4096];
        let hash = protocol::content_hash(&data);
        assert!(cache.put(&protocol::content_hash(b"other"), &data).is_err());
        cache.put(&hash, &data).unwrap();
        assert_eq!(cache.index(), vec![hash]);
        assert_eq!(cache.get(&hash, 4096).unwrap(), data);
        assert!(cache.get(&hash, 4095).is_err());

        // Reopened, it is still there; with too small a limit it is removed
        let (cache, total) = Cache::open(&folder, 1 << 20).unwrap();
        assert_eq!((cache.index(), total), (vec![hash], 4096));
        let (cache, total) = Cache::open(&folder, 100).unwrap();
        assert_eq!((cache.index().len(), total), (0, 0));

        fs::remove_dir_all(&folder).unwrap();
    }
}
