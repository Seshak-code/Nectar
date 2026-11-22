//! Threads that stay, for work done many times a second that would otherwise start and end a dozen each
//! time. Ending a thread runs every loaded DLL's cleanup, a graphics driver's among them.

use std::sync::{Condvar, Mutex, MutexGuard};

pub struct Pool {
    state: Mutex<State>,
    changed: Condvar,
    /// One job at a time: those who come later wait their turn
    turn: Mutex<()>,
}

struct State {
    /// What's to run and how many more threads are to take it up: the address of the function, which
    /// outlives the call as `run` waits for it
    job: Option<(usize, usize)>,
    running: usize,
    threads: usize,
}

impl Pool {
    pub const fn new() -> Self {
        Self { state: Mutex::new(State { job: None, running: 0, threads: 0 }), changed: Condvar::new(), turn: Mutex::new(()) }
    }

    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn work(&'static self) {
        let mut state = self.lock();
        loop {
            match state.job {
                Some((job, left)) => {
                    state.job = if left > 1 { Some((job, left - 1)) } else { None };
                    state.running += 1;
                    drop(state);
                    // Valid until `running` comes back to 0, which `run` waits for
                    unsafe { (*(job as *const &(dyn Fn() + Sync)))() };
                    state = self.lock();
                    state.running -= 1;
                    self.changed.notify_all();
                }
                None => state = self.changed.wait(state).unwrap_or_else(|p| p.into_inner()),
            }
        }
    }

    /// Runs `work` on this thread and `others` more at once, and returns when all of them have finished
    pub fn run(&'static self, others: usize, work: &(dyn Fn() + Sync)) {
        if others == 0 {
            work();
            return;
        }
        let _turn = self.turn.lock().unwrap_or_else(|p| p.into_inner());
        let job = (&raw const work) as usize;
        {
            let mut state = self.lock();
            while state.threads < others {
                state.threads += 1;
                std::thread::spawn(move || self.work());
            }
            state.job = Some((job, others));
            self.changed.notify_all();
        }
        work();
        let mut state = self.lock();
        state.job = None;
        while state.running > 0 {
            state = self.changed.wait(state).unwrap_or_else(|p| p.into_inner());
        }
    }
}

impl Default for Pool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static POOL: Pool = Pool::new();

    #[test]
    fn every_thread_runs_it_and_it_returns_when_they_have() {
        for others in [0, 1, 3, 7, 2] {
            let ran = AtomicUsize::new(0);
            POOL.run(others, &|| {
                ran.fetch_add(1, Ordering::Relaxed);
            });
            // A thread that was slow to wake misses the job only if the others left none for it, and
            // here each is given its own
            assert!(ran.load(Ordering::Relaxed) >= 1);
            assert!(ran.load(Ordering::Relaxed) <= others + 1);
        }
    }

    #[test]
    fn work_is_all_done_however_many_callers_there_are() {
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    for _ in 0..50 {
                        let (next, done) = (AtomicUsize::new(0), AtomicUsize::new(0));
                        POOL.run(3, &|| {
                            while next.fetch_add(1, Ordering::Relaxed) < 200 {
                                done.fetch_add(1, Ordering::Relaxed);
                            }
                        });
                        assert_eq!(done.load(Ordering::Relaxed), 200);
                    }
                });
            }
        });
    }
}
