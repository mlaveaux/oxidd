//! Keep idle rayon workers busy instead of sleeping
//!
//! This is a workaround for rayon's sleep policy, without changing rayon.
//! rayon puts a worker to sleep after 32 rounds without work, and lets a
//! worker whose half of a `join` was stolen sleep until the thief is done.
//! With many small fork-join operations in quick succession, as in decision
//! diagram operations, the workers are then woken (a `futex` call) again and
//! again, which costs more than the tasks themselves.
//!
//! [`Spinners`] provides two remedies for a [`rayon::ThreadPool`]:
//!
//! - While an operation entered from outside the pool runs, and shortly after,
//!   every worker runs a *spinner* job that keeps looking for work in user
//!   space.
//! - A `join` on a worker waits by running other work ([`spin_join`]) instead
//!   of sleeping, as Lace's leapfrogging does.
//!
//! Both are off for a pool with a single worker.

#![warn(missing_docs)]

use std::cell::UnsafeCell;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::Arc;
use std::sync::atomic::{
    AtomicBool, AtomicU64, AtomicUsize,
    Ordering::{Acquire, Relaxed, Release},
};
use std::time::Instant;

use rayon::ThreadPool;

/// How long the spinners keep searching for work after the last operation
/// ended, in microseconds
pub const SPINNER_LINGER_US: u64 = 1000;

/// Spinner jobs and [`spin_join`] for one [`ThreadPool`]
///
/// Every method takes the pool that this was created for. Call
/// [`Spinners::stop`] before dropping the pool.
pub struct Spinners {
    state: Arc<State>,
}

/// The state shared with the spinner jobs
struct State {
    /// How long spinners keep going after the last operation ended (0: off)
    linger_ns: u64,
    /// Number of operations entered from outside the pool that are running
    active: AtomicUsize,
    /// Whether worker `i` runs a spinner
    running: Box<[AtomicBool]>,
    /// End of the last operation, see [`State::now`]
    last_end: AtomicU64,
    /// Set by [`Spinners::stop`]
    stop: AtomicBool,
    /// Whether `join` on a worker uses [`spin_join`]
    spin_join: bool,
    start: Instant,
}

impl Spinners {
    /// Spinners for `pool`; with a single worker there is nobody to keep
    /// busy, so everything is off.
    pub fn new(pool: &ThreadPool) -> Self {
        let threads = pool.current_num_threads();
        let parallel = threads > 1;
        Self {
            state: Arc::new(State {
                linger_ns: if parallel {
                    SPINNER_LINGER_US * 1000
                } else {
                    0
                },
                active: AtomicUsize::new(0),
                running: (0..threads).map(|_| AtomicBool::new(false)).collect(),
                last_end: AtomicU64::new(0),
                stop: AtomicBool::new(false),
                spin_join: parallel,
                start: Instant::now(),
            }),
        }
    }

    /// Runs `op`, which enters `pool` (e.g. [`ThreadPool::install`]), with
    /// spinners on the workers while it runs
    ///
    /// Called on a worker of `pool`, this just runs `op`.
    #[inline]
    pub fn entered<R>(&self, pool: &ThreadPool, op: impl FnOnce() -> R) -> R {
        let state = &*self.state;
        if state.linger_ns == 0 || pool.current_thread_index().is_some() {
            return op();
        }
        state.active.fetch_add(1, Relaxed);
        self.start(pool);
        let result = op();
        state.last_end.store(state.now(), Relaxed);
        state.active.fetch_sub(1, Relaxed);
        result
    }

    /// [`ThreadPool::join`] on `pool`, using [`spin_join`] on its workers and
    /// running with spinners when called from outside
    #[inline]
    pub fn join<RA: Send, RB: Send>(
        &self,
        pool: &ThreadPool,
        op_a: impl FnOnce() -> RA + Send,
        op_b: impl FnOnce() -> RB + Send,
    ) -> (RA, RB) {
        if self.state.spin_join && pool.current_thread_index().is_some() {
            return spin_join(op_a, op_b);
        }
        self.entered(pool, || pool.join(op_a, op_b))
    }

    /// Ends all spinners, call this before dropping the pool
    pub fn stop(&self) {
        self.state.stop.store(true, Relaxed);
    }

    /// Makes sure that every worker runs a spinner job
    ///
    /// A broadcast reaches every worker exactly once; a worker that already
    /// runs a spinner (it may pick up the broadcast inside its spinner) returns
    /// right away.
    fn start(&self, pool: &ThreadPool) {
        if self.state.running.iter().all(|r| r.load(Relaxed)) {
            return;
        }
        let state = self.state.clone();
        pool.spawn_broadcast(move |ctx| state.run(ctx.index()));
    }
}

impl State {
    fn now(&self) -> u64 {
        self.start.elapsed().as_nanos() as u64
    }

    /// The body of a spinner job on worker `index`: run or steal pending work
    /// (including broadcasts and injected jobs) with [`rayon::yield_now`],
    /// until no operation has been running for [`SPINNER_LINGER_US`]
    fn run(&self, index: usize) {
        if self.running[index].swap(true, Relaxed) {
            return; // this worker already spins further down its stack
        }
        loop {
            if rayon::yield_now() == Some(rayon::Yield::Executed) {
                continue;
            }
            if self.stop.load(Relaxed)
                || (self.active.load(Relaxed) == 0
                    && self.now().saturating_sub(self.last_end.load(Relaxed)) > self.linger_ns)
            {
                break;
            }
            std::hint::spin_loop();
        }
        self.running[index].store(false, Relaxed);
    }
}

/// A `join` that waits by running other work instead of sleeping
///
/// rayon's `join` lets a worker whose second half was stolen look for other
/// work for 32 rounds of `sched_yield` and then sleep until the thief is done.
/// This version hands `op_b` to rayon with [`rayon::spawn`], runs `op_a`, and
/// then calls [`rayon::yield_now`] until `op_b` has finished: it pops `op_b`
/// itself if nobody stole it, and otherwise keeps running or stealing other
/// work, as Lace's leapfrogging does.
///
/// Must be called on a worker of a pool: elsewhere, [`rayon::spawn`] would
/// use the global pool.
pub fn spin_join<RA: Send, RB: Send>(
    op_a: impl FnOnce() -> RA + Send,
    op_b: impl FnOnce() -> RB + Send,
) -> (RA, RB) {
    struct Slot<R> {
        result: UnsafeCell<Option<std::thread::Result<R>>>,
        done: AtomicBool,
    }
    // SAFETY: `result` is written once by the job before `done` is set
    // (`Release`), and read by the joining thread only after it observed
    // `done` (`Acquire`).
    unsafe impl<R: Send> Sync for Slot<R> {}

    let slot = Slot {
        result: UnsafeCell::new(None),
        done: AtomicBool::new(false),
    };
    let shared = &slot;
    let job: Box<dyn FnOnce() + Send + '_> = Box::new(move || {
        let slot = shared;
        let result = catch_unwind(AssertUnwindSafe(op_b));
        // SAFETY: see `Slot`; nobody reads `result` before `done` is set.
        unsafe { *slot.result.get() = Some(result) };
        slot.done.store(true, Release);
    });
    // SAFETY: the job borrows `slot` and `op_b` from this frame. This function
    // does not return or unwind before the job has finished (`done`): `op_a`
    // runs under `catch_unwind`, and the loop below waits for `done`.
    let job: Box<dyn FnOnce() + Send + 'static> = unsafe { std::mem::transmute(job) };
    rayon::spawn(job);

    let result_a = catch_unwind(AssertUnwindSafe(op_a));
    while !slot.done.load(Acquire) {
        if rayon::yield_now() != Some(rayon::Yield::Executed) {
            std::hint::spin_loop();
        }
    }
    // SAFETY: `done` is set, so the job has written `result` and finished.
    let result_b = unsafe { (*slot.result.get()).take() }.expect("the job sets the result");
    match (result_a, result_b) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(error), _) | (_, Err(error)) => resume_unwind(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(threads: usize) -> ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
    }

    fn fib(spin: &Spinners, pool: &ThreadPool, n: u64) -> u64 {
        if n < 2 {
            return n;
        }
        let (a, b) = spin.join(pool, || fib(spin, pool, n - 1), || fib(spin, pool, n - 2));
        a + b
    }

    #[test]
    fn join_computes_both_halves() {
        for threads in [1, 4] {
            let pool = pool(threads);
            let spin = Spinners::new(&pool);
            assert_eq!(fib(&spin, &pool, 20), 6765);
            let result = spin.entered(&pool, || pool.install(|| fib(&spin, &pool, 20)));
            assert_eq!(result, 6765);
            spin.stop();
        }
    }

    #[test]
    fn spin_join_propagates_panics() {
        let pool = pool(2);
        let result = pool
            .install(|| catch_unwind(AssertUnwindSafe(|| spin_join(|| 1, || panic!("in op_b")))));
        assert!(result.is_err());
    }
}
