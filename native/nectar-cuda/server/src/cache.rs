//! Uploads kept by content hash (protocol::UPLOAD), so the next load of the same model sends
//! only hashes over the network.
//!
//! Each upload is a file in the cache folder named by its hash. When the server starts, the
//! least recently used files are removed until the folder is at most 3/4 of its limit, which leaves
//! room for what's new. Nothing is removed while it runs, so a borrower can rely on the index it was
//! given; once the limit is reached nothing more is kept until the next start.
//!
//! The hash of every upload is checked before it is kept, so a borrower cannot store bytes
//! under another upload's hash.
//!
//! Anything small, such as shader code and a texture's small mips, is kept the same way but in one file
//! (Pack): a game has thousands of them, and opening a file for each takes longer than sending it.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

use nectar_cuda_protocol as protocol;

pub(crate) use protocol::Hash;

pub(crate) struct Cache {
    folder: PathBuf,
    kept: Mutex<HashSet<Hash>>,
    pack: Mutex<Pack>,
    /// Bytes in files, and the most there may be
    total: AtomicU64,
    limit: u64,
}

/// The share of the cache's limit the pack may take
const PACK_SHARE: u64 = 2;
/// A record in the pack: the hash, u32 length, then the bytes
const RECORD_HEAD: usize = 20;

/// What's left of a limit on starting, so that there's room for what's new
fn room_left(limit: u64) -> u64 {
    limit - limit / 4
}

/// Small things one record after another. If it's fuller than room_left when opened it starts over:
/// what it held then comes again once, which costs little.
struct Pack {
    file: File,
    limit: u64,
    size: u64,
    /// Where each is: its code's offset, and length
    index: HashMap<Hash, (u64, u32)>,
}

impl Pack {
    fn open(path: &Path, limit: u64) -> io::Result<Self> {
        let mut file = File::options().read(true).write(true).create(true).truncate(false).open(path)?;
        let length = file.metadata()?.len();
        let (mut index, mut at) = (HashMap::new(), 0u64);
        if length <= room_left(limit) {
            let mut head = [0u8; RECORD_HEAD];
            while at + RECORD_HEAD as u64 <= length {
                file.seek(SeekFrom::Start(at))?;
                file.read_exact(&mut head)?;
                let size = u32::from_le_bytes(head[16..].try_into().unwrap());
                let code = at + RECORD_HEAD as u64;
                if code + size as u64 > length {
                    break;
                }
                index.insert(head[..16].try_into().unwrap(), (code, size));
                at = code + size as u64;
            }
        }
        // Anything after the last whole record was cut short
        file.set_len(at)?;
        Ok(Self { file, limit, size: at, index })
    }

    fn get(&mut self, hash: &Hash, size: u64) -> io::Result<Option<Vec<u8>>> {
        let Some(&(at, length)) = self.index.get(hash).filter(|(_, length)| *length as u64 == size) else {
            return Ok(None);
        };
        let mut data = vec![0u8; length as usize];
        self.file.seek(SeekFrom::Start(at))?;
        self.file.read_exact(&mut data)?;
        Ok(Some(data))
    }

    fn put(&mut self, hash: &Hash, data: &[u8]) -> io::Result<()> {
        if self.index.contains_key(hash) {
            return Ok(());
        }
        let needs = (RECORD_HEAD + data.len()) as u64;
        if needs > self.limit {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "larger than the cache"));
        }
        if self.size + needs > self.limit {
            return Err(io::Error::other("the cache's pack is full"));
        }
        self.file.seek(SeekFrom::Start(self.size))?;
        let mut record = Vec::with_capacity(needs as usize);
        record.extend_from_slice(hash);
        record.extend_from_slice(&(data.len() as u32).to_le_bytes());
        record.extend_from_slice(data);
        self.file.write_all(&record)?;
        self.index.insert(*hash, (self.size + RECORD_HEAD as u64, data.len() as u32));
        self.size += needs;
        Ok(())
    }
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

        files.sort_by_key(|file| std::cmp::Reverse(file.0));
        let (mut kept, mut total) = (HashSet::new(), 0u64);
        for (_, size, hash, path) in files {
            if total + size <= room_left(limit) {
                total += size;
                kept.insert(hash);
            } else {
                fs::remove_file(path)?;
            }
        }

        let cache = Self {
            folder: folder.to_path_buf(),
            kept: Mutex::new(kept),
            pack: Mutex::new(Pack::open(&folder.join("shaders.pack"), limit / PACK_SHARE)?),
            total: AtomicU64::new(total),
            limit,
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

    fn pack(&self) -> std::sync::MutexGuard<'_, Pack> {
        self.pack.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub(crate) fn index(&self) -> Vec<Hash> {
        let mut index: Vec<Hash> = self.kept().iter().copied().collect();
        index.extend(self.pack().index.keys());
        index
    }

    /// What's kept under `hash`, in the pack or a file, which must be `size` long
    pub(crate) fn find(&self, hash: &Hash, size: u64) -> io::Result<Vec<u8>> {
        match self.pack().get(hash, size)? {
            Some(bytes) => Ok(bytes),
            None => self.get(hash, size),
        }
    }

    /// Keeps something small in the pack under `hash`, if that is its hash
    pub(crate) fn keep_small(&self, hash: &Hash, bytes: &[u8]) -> io::Result<()> {
        if protocol::content_hash(bytes) != *hash {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bytes do not match their hash"));
        }
        self.pack().put(hash, bytes)
    }

    /// Keeps `bytes` under `hash` where something its size goes
    pub(crate) fn keep(&self, hash: &Hash, bytes: &[u8]) -> io::Result<()> {
        if bytes.len() < protocol::CACHE_MIN { self.keep_small(hash, bytes) } else { self.put(hash, bytes) }
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

        if self.total.load(Ordering::Relaxed) + data.len() as u64 > self.limit {
            return Err(io::Error::other("the cache is full"));
        }

        let path = self.path(hash);
        let part = path.with_extension("part");
        fs::write(&part, data)?;
        fs::rename(&part, &path)?;
        self.total.fetch_add(data.len() as u64, Ordering::Relaxed);
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

        // Over the limit nothing more is kept, and what's kept stays
        assert!(cache.put(&hash, &data).is_err());
        let (cache, _) = Cache::open(&folder, 6000).unwrap();
        cache.put(&hash, &data).unwrap();
        let more = vec![6u8; 4096];
        assert!(cache.put(&protocol::content_hash(&more), &more).is_err());
        assert_eq!(cache.get(&hash, 4096).unwrap(), data);
        // Reopened it leaves room for what's new: 4096 is more than 3/4 of 5000
        let (cache, total) = Cache::open(&folder, 5000).unwrap();
        assert_eq!((cache.index().len(), total), (0, 0));

        fs::remove_dir_all(&folder).unwrap();
    }

    #[test]
    fn small_things_and_large_go_to_their_places() {
        let folder = std::env::temp_dir().join(format!("nectar-keep-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&folder);
        let (cache, _) = Cache::open(&folder, 64 << 20).unwrap();
        let (small, large) = (vec![1u8; 4096], vec![2u8; protocol::CACHE_MIN]);
        let (small_hash, large_hash) = (protocol::content_hash(&small), protocol::content_hash(&large));
        cache.keep(&small_hash, &small).unwrap();
        cache.keep(&large_hash, &large).unwrap();
        assert_eq!(cache.find(&small_hash, 4096).unwrap(), small);
        assert_eq!(cache.find(&large_hash, large.len() as u64).unwrap(), large);
        assert!(cache.keep(&large_hash, &small).is_err());
        assert_eq!(cache.index().len(), 2);
        // The small one is in the pack, not a file of its own
        assert_eq!(fs::read_dir(&folder).unwrap().count(), 2);
        drop(cache);
        fs::remove_dir_all(&folder).unwrap();
    }

    #[test]
    fn shader_code_is_kept_in_one_file() {
        let folder = std::env::temp_dir().join(format!("nectar-pack-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&folder);
        let code = |seed: u8| vec![seed; 3000];
        let hash = |seed: u8| protocol::content_hash(&code(seed));

        let (cache, _) = Cache::open(&folder, 40_000).unwrap();
        assert!(cache.keep_small(&hash(2), &code(1)).is_err());
        cache.keep_small(&hash(1), &code(1)).unwrap();
        cache.keep_small(&hash(2), &code(2)).unwrap();
        // Again changes nothing
        cache.keep_small(&hash(1), &code(1)).unwrap();
        assert_eq!(cache.find(&hash(1), 3000).unwrap(), code(1));
        assert_eq!(cache.find(&hash(2), 3000).unwrap(), code(2));
        assert!(cache.find(&hash(2), 2999).is_err());
        assert!(cache.find(&hash(3), 3000).is_err());
        assert_eq!(cache.index().len(), 2);
        drop(cache);
        assert_eq!(fs::read_dir(&folder).unwrap().count(), 1);

        // Reopened it's still there, and a record cut short is dropped
        let path = folder.join("shaders.pack");
        let file = File::options().write(true).open(&path).unwrap();
        file.set_len(file.metadata().unwrap().len() - 10).unwrap();
        drop(file);
        let (cache, _) = Cache::open(&folder, 40_000).unwrap();
        assert_eq!(cache.index(), vec![hash(1)]);
        assert_eq!(cache.find(&hash(1), 3000).unwrap(), code(1));

        // Full, it keeps no more, and what it holds stays, as the borrower relies on it
        for seed in 3..8 {
            cache.keep_small(&hash(seed), &code(seed)).unwrap();
        }
        assert!(cache.keep_small(&hash(8), &code(8)).is_err());
        assert_eq!(cache.find(&hash(1), 3000).unwrap(), code(1));
        assert_eq!(cache.find(&hash(4), 3000).unwrap(), code(4));
        drop(cache);

        // Fuller than the room it leaves, it starts over when opened
        let (cache, _) = Cache::open(&folder, 40_000).unwrap();
        assert_eq!(cache.index().len(), 0);
        cache.keep_small(&hash(8), &code(8)).unwrap();
        assert_eq!(cache.find(&hash(8), 3000).unwrap(), code(8));
        drop(cache);

        fs::remove_dir_all(&folder).unwrap();
    }
}
