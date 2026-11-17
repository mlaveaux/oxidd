use std::fmt;
use std::io;
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::BroadcastContext;

/// Worker thread pool
///
/// Dropping it waits until every worker thread has exited, including the
/// destructors of its thread-local state, so that no worker outlives the
/// manager. Thread-affine libraries used by the workers (e.g. ones that move
/// per-thread state to the main thread when a thread exits) rely on that.
pub struct Workers {
    pool: ManuallyDrop<rayon::ThreadPool>,
    split_depth: AtomicU32,
    /// The worker threads, joined on drop
    threads: Mutex<Vec<JoinHandle<()>>>,
}

// spell-checker:ignore rlim
fn getrlimit_as() -> usize {
    #[cfg(all(target_family = "unix", not(miri)))]
    {
        let mut rlimit = std::mem::MaybeUninit::uninit();
        // SAFETY: `getrlimit` is MT-Safe, the second argument is a valid
        // pointer
        if unsafe { libc::getrlimit(libc::RLIMIT_AS, rlimit.as_mut_ptr()) } == 0 {
            // SAFETY: on success (return value 0), `rlimit` is initialized
            let limit = unsafe { rlimit.assume_init_ref() }.rlim_cur;
            return limit.try_into().unwrap_or(usize::MAX);
        }
    }
    usize::MAX
}

#[derive(Debug)]
struct WorkerBuildError(rayon::ThreadPoolBuildError);

impl fmt::Display for WorkerBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "failed to build the manager's worker pool: {}", self.0)
    }
}

impl std::error::Error for WorkerBuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

impl From<WorkerBuildError> for io::Error {
    fn from(value: WorkerBuildError) -> Self {
        io::Error::other(value)
    }
}

impl Workers {
    /// Create a new worker pool with the given number of threads
    ///
    /// If `threads` is 0, the thread count will be determined automatically
    /// based on the environment variable `OXIDD_NUM_THREADS` if present or
    /// [`std::thread::available_parallelism()`].
    pub fn new(threads: u32) -> io::Result<Self> {
        let threads = if threads > 0 {
            threads
        } else if let Ok(str) = std::env::var("OXIDD_NUM_THREADS")
            && let Ok(threads) = str.parse()
            && threads > 0
        {
            threads
        } else if let Ok(n) = std::thread::available_parallelism() {
            n.get().try_into().unwrap_or(u32::MAX)
        } else {
            1
        };
        debug_assert_ne!(threads, 0);

        let stack_size: usize = if let Ok(str) = std::env::var("OXIDD_STACK_SIZE")
            && let Ok(size) = str.parse()
        {
            size
        } else {
            let default = 1024 * 1024 * 1024; // 1 GiB
            // Never use more than 1/16 of the available address space for
            // worker pool stacks. This parameter is based on experiments on a
            // Linux machine with an RLIMIT_AS of 512 MiB and up to 16 threads.
            std::cmp::min(default, getrlimit_as() / (16 * threads as usize))
        };

        static POOL_COUNTER: AtomicU64 = AtomicU64::new(0);
        let p = POOL_COUNTER.fetch_add(1, Relaxed);

        let handles = Arc::new(Mutex::new(Vec::with_capacity(threads as usize)));
        let spawned = handles.clone();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads as usize)
            .thread_name(move |i| format!("oxidd/rayon {p}-{i}"))
            .stack_size(stack_size)
            .spawn_handler(move |thread| {
                let mut builder = std::thread::Builder::new();
                if let Some(name) = thread.name() {
                    builder = builder.name(name.to_owned());
                }
                if let Some(stack_size) = thread.stack_size() {
                    builder = builder.stack_size(stack_size);
                }
                let handle = builder.spawn(|| thread.run())?;
                spawned
                    .lock()
                    .unwrap_or_else(|err| err.into_inner())
                    .push(handle);
                Ok(())
            })
            .build()
            .map_err(WorkerBuildError)?;

        let split_depth = AtomicU32::new(Workers::auto_split_depth(&pool));
        let threads = std::mem::take(&mut *handles.lock().unwrap_or_else(|err| err.into_inner()));
        Ok(Self {
            pool: ManuallyDrop::new(pool),
            split_depth,
            threads: Mutex::new(threads),
        })
    }

    fn auto_split_depth(pool: &rayon::ThreadPool) -> u32 {
        let threads = pool.current_num_threads();
        if threads > 1 {
            (4096 * threads).ilog2()
        } else {
            0
        }
    }
}

impl Drop for Workers {
    fn drop(&mut self) {
        // SAFETY: `self.pool` is not used after this.
        unsafe { ManuallyDrop::drop(&mut self.pool) };
        // Dropping the pool only tells the workers to stop; wait until they
        // did. A worker that drops the pool itself cannot wait for itself.
        let current = std::thread::current().id();
        for handle in std::mem::take(
            self.threads
                .get_mut()
                .unwrap_or_else(|err| err.into_inner()),
        ) {
            if handle.thread().id() != current {
                let _ = handle.join();
            }
        }
    }
}

impl crate::WorkerPool for Workers {
    #[inline]
    fn current_num_threads(&self) -> usize {
        self.pool.current_num_threads()
    }

    #[inline(always)]
    fn split_depth(&self) -> u32 {
        self.split_depth.load(Relaxed)
    }

    fn set_split_depth(&self, depth: Option<u32>) {
        let depth = match depth {
            Some(d) => d,
            None => Self::auto_split_depth(&self.pool),
        };
        self.split_depth.store(depth, Relaxed);
    }

    #[inline]
    fn install<RA: Send>(&self, op: impl FnOnce() -> RA + Send) -> RA {
        self.pool.install(op)
    }

    #[inline]
    fn join<RA: Send, RB: Send>(
        &self,
        op_a: impl FnOnce() -> RA + Send,
        op_b: impl FnOnce() -> RB + Send,
    ) -> (RA, RB) {
        self.pool.join(op_a, op_b)
    }

    #[inline]
    fn broadcast<RA: Send>(&self, op: impl Fn(BroadcastContext) -> RA + Sync) -> Vec<RA> {
        self.pool.broadcast(move |ctx| {
            op(BroadcastContext {
                index: ctx.index() as u32,
                num_threads: ctx.num_threads() as u32,
            })
        })
    }

    #[inline]
    fn spawn_broadcast(&self, op: impl Fn(BroadcastContext) + Send + Sync + 'static) {
        self.pool.spawn_broadcast(move |ctx| {
            op(BroadcastContext {
                index: ctx.index() as u32,
                num_threads: ctx.num_threads() as u32,
            })
        });
    }
}
