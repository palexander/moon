use crate::{HashEngine, StateEngine, merge_clean_results, resolve_path};
use moon_cache_item::*;
use moon_common::path::encode_component;
use moon_env_var::GlobalEnvBag;
use moon_time::parse_duration;
use serde::Serialize;
use serde::de::DeserializeOwned;
use starbase_utils::fs::{FileLock, RemoveDirContentsResult};
use starbase_utils::{fs, json};
use std::collections::HashMap;
use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tracing::{debug, instrument};

/// A file lock that serializes both intra-process (tokio) and inter-process
/// (OS-level flock) access to a shared lock file.
///
/// On Linux, flock(2) locks are per-open-file-description (OFD), not
/// per-process: two open() calls from the same PID produce independent OFDs
/// that deadlock each other on flock(LOCK_EX). The `_process_guard` prevents
/// concurrent tokio tasks in the same process from ever reaching the flock
/// layer simultaneously, while `file_lock` retains the inter-process guarantee.
///
/// Drop order matters: `file_lock` releases the OS lock first, then
/// `_process_guard` wakes the next in-process waiter. Rust drops fields in
/// declaration order.
pub struct InProcessFileLock {
    pub file_lock: FileLock,
    _process_guard: OwnedSemaphorePermit,
}

#[derive(Debug)]
pub struct CacheEngine {
    /// The `.moon/cache` directory relative to workspace root.
    /// Contains cached items pertaining to runs and processes.
    pub cache_dir: PathBuf,

    /// Manages reading and writing of content hashable items.
    pub hash: HashEngine,

    /// Manages states of projects, tasks, tools, and more.
    pub state: StateEngine,

    /// A temporary directory for random artifacts.
    pub temp_dir: PathBuf,

    mode: CacheMode,
    forced_mode: RwLock<Option<CacheMode>>,
    in_process_locks: Mutex<HashMap<PathBuf, Arc<Semaphore>>>,
}

impl CacheEngine {
    pub fn new(config_dir: impl AsRef<Path>) -> miette::Result<CacheEngine> {
        let dir = config_dir.as_ref().join("cache");
        let cache_tag = dir.join("CACHEDIR.TAG");

        debug!(
            cache_dir = ?dir,
            "Creating cache engine",
        );

        fs::create_dir_all(&dir)?;

        // Create a cache directory tag
        if !cache_tag.exists() {
            fs::write_file(
                cache_tag,
                r#"Signature: 8a477f597d28d172789f06886806bc55
# This file is a cache directory tag created by moon.
# For information see https://bford.info/cachedir"#,
            )?;
        }

        Ok(CacheEngine {
            hash: HashEngine::new(&dir)?,
            state: StateEngine::new(&dir)?,
            temp_dir: dir.join("temp"),
            cache_dir: dir,
            mode: get_cache_mode(),
            forced_mode: RwLock::new(None),
            in_process_locks: Mutex::new(HashMap::new()),
        })
    }

    pub fn force_mode(&self, mode: CacheMode) {
        let _ = self.forced_mode.write().unwrap().insert(mode);

        GlobalEnvBag::instance().set("MOON_CACHE", mode.to_string());
    }

    pub fn cache<T>(&self, path: impl AsRef<OsStr>) -> miette::Result<CacheItem<T>>
    where
        T: Default + DeserializeOwned + Serialize,
    {
        CacheItem::<T>::load(self.resolve_path(path))
    }

    #[instrument(skip(self))]
    pub fn clean_stale_cache(&self, lifetime: &str, all: bool) -> miette::Result<(usize, u64)> {
        let duration = self.parse_lifetime(lifetime)?;

        debug!(
            "Cleaning up and deleting stale cached artifacts older than \"{}\"",
            lifetime
        );

        let mut dirs = vec![&self.hash.hashes_dir, &self.hash.outputs_dir];

        if all {
            dirs.push(&self.state.states_dir);
            dirs.push(&self.temp_dir);
        }

        let mut result = RemoveDirContentsResult {
            files_deleted: 0,
            bytes_saved: 0,
        };

        for dir in dirs {
            result = merge_clean_results(result, fs::remove_dir_stale_contents(dir, duration)?);
        }

        debug!(
            "Deleted {} artifacts and saved {} bytes",
            result.files_deleted, result.bytes_saved
        );

        Ok((result.files_deleted, result.bytes_saved))
    }

    pub async fn create_lock<T: AsRef<str>>(&self, name: T) -> miette::Result<InProcessFileLock> {
        let mut name = encode_component(name.as_ref());

        if !name.ends_with(".lock") {
            name.push_str(".lock");
        }

        let lock_path = self.cache_dir.join("locks").join(&name);

        // Serialize intra-process access before calling flock(2).
        // Linux flock(2) semantics are per-open-file-description (OFD): two
        // open() calls from the same PID produce independent OFDs that
        // deadlock each other on flock(LOCK_EX). The tokio semaphore below
        // prevents concurrent tokio tasks in the same process from ever racing
        // for the same lock file, while the subsequent flock still provides the
        // inter-process guarantee for the rare case of two moon binaries sharing
        // the same workspace.
        let sem = {
            let mut map = self.in_process_locks.lock().expect("in_process_locks poisoned");
            Arc::clone(
                map.entry(lock_path.clone())
                    .or_insert_with(|| Arc::new(Semaphore::new(1))),
            )
        };

        let process_guard = sem
            .acquire_owned()
            .await
            .expect("CacheEngine lock semaphore closed unexpectedly");

        let file_lock = fs::lock_file(&lock_path)?;

        Ok(InProcessFileLock {
            file_lock,
            _process_guard: process_guard,
        })
    }

    pub fn write<K, T>(&self, path: K, data: &T) -> miette::Result<()>
    where
        K: AsRef<OsStr>,
        T: ?Sized + Serialize,
    {
        let path = self.resolve_path(path);

        debug!(cache = ?path, "Writing cache");

        // This purposefully ignores the cache mode and always writes!
        json::write_file(path, &data, false)?;

        Ok(())
    }

    pub async fn execute_if_changed<K, T, F, R>(
        &self,
        label: K,
        fingerprint: T,
        op: F,
    ) -> miette::Result<Option<R>>
    where
        K: AsRef<str>,
        T: Serialize,
        F: AsyncFnOnce(&str) -> miette::Result<R>,
    {
        let mut hasher = self.hash.create_hasher(label.as_ref());
        hasher.hash_content(fingerprint)?;

        let hash = hasher.generate_hash()?;

        // If the hash manifest exists, then it has ran before,
        // otherwise run and write the manifest
        if !self.hash.get_manifest_path(&hash).exists() {
            let result = op(&hash).await?;

            self.hash.save_manifest(&mut hasher)?;

            return Ok(Some(result));
        }

        Ok(None)
    }

    pub fn parse_lifetime(&self, lifetime: &str) -> miette::Result<Duration> {
        parse_duration(lifetime).map_err(|error| miette::miette!("Invalid lifetime: {error}"))
    }

    pub fn resolve_path(&self, path: impl AsRef<OsStr>) -> PathBuf {
        resolve_path(&self.cache_dir, path)
    }

    pub fn is_readable(&self) -> bool {
        self.get_mode().is_readable()
    }

    pub fn is_read_only(&self) -> bool {
        self.get_mode().is_read_only()
    }

    pub fn is_writable(&self) -> bool {
        self.get_mode().is_writable()
    }

    pub fn is_write_only(&self) -> bool {
        self.get_mode().is_write_only()
    }

    fn get_mode(&self) -> CacheMode {
        if let Ok(lock) = self.forced_mode.read()
            && let Some(mode) = &*lock
        {
            return *mode;
        }

        self.mode
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use starbase_sandbox::create_empty_sandbox;
    use std::time::Duration;

    /// Regression test: two concurrent tokio tasks calling create_lock with the
    /// same name must both complete. Before the in-process semaphore was added,
    /// the second task would block on flock(LOCK_EX) against the first task's
    /// open FD (per-OFD flock semantics on Linux), starving the tokio runtime
    /// and deadlocking indefinitely.
    #[tokio::test]
    async fn create_lock_concurrent_same_name_does_not_deadlock() {
        let sandbox = create_empty_sandbox();
        let engine = Arc::new(CacheEngine::new(sandbox.path()).unwrap());

        let e1 = Arc::clone(&engine);
        let e2 = Arc::clone(&engine);

        let t1 = tokio::spawn(async move {
            let _lock = e1.create_lock("concurrent-lock-test").await.unwrap();
            tokio::time::sleep(Duration::from_millis(50)).await;
        });

        let t2 = tokio::spawn(async move {
            let _lock = e2.create_lock("concurrent-lock-test").await.unwrap();
            tokio::time::sleep(Duration::from_millis(50)).await;
        });

        tokio::time::timeout(Duration::from_secs(5), async {
            t1.await.unwrap();
            t2.await.unwrap();
        })
        .await
        .expect("both lock acquisitions must complete — intra-process deadlock detected");
    }

    #[tokio::test]
    async fn create_lock_different_names_run_concurrently() {
        let sandbox = create_empty_sandbox();
        let engine = Arc::new(CacheEngine::new(sandbox.path()).unwrap());

        let e1 = Arc::clone(&engine);
        let e2 = Arc::clone(&engine);

        let start = tokio::time::Instant::now();

        let t1 = tokio::spawn(async move {
            let _lock = e1.create_lock("lock-a").await.unwrap();
            tokio::time::sleep(Duration::from_millis(100)).await;
        });

        let t2 = tokio::spawn(async move {
            let _lock = e2.create_lock("lock-b").await.unwrap();
            tokio::time::sleep(Duration::from_millis(100)).await;
        });

        t1.await.unwrap();
        t2.await.unwrap();

        // Both tasks held different locks and ran concurrently, so elapsed
        // should be closer to 100ms than 200ms.
        assert!(start.elapsed() < Duration::from_millis(180));
    }
}
