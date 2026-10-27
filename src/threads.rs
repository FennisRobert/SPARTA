//! The threads SPARTA's own parallel work runs on.
//!
//! [`Config::threads`](crate::Config::threads) of zero, the default, runs on
//! Rayon's global pool, which `RAYON_NUM_THREADS` sizes for the whole process.
//! A positive count gives each analysis and each CPU solver a pool of its own
//! with that many threads, so SPARTA can be bounded without touching anything
//! else in the process that uses Rayon. `SPARTA_NUM_THREADS`, when set to a
//! positive count, overrides whatever count the caller passed: it is the one
//! switch that reaches SPARTA through any binding and any wrapper.
use crate::{Error, Result};
use std::sync::Arc;

/// The thread count a caller's `requested` count resolves to:
/// `SPARTA_NUM_THREADS` when it holds a positive count, otherwise `requested`.
/// Zero means Rayon's global pool.
pub(crate) fn resolve(requested: usize) -> usize {
    pick(std::env::var("SPARTA_NUM_THREADS").ok().as_deref(), requested)
}

/// [`resolve`] with the variable's value passed in: a positive count wins, and
/// anything else -- unset, empty, zero or not a number -- leaves `requested`.
fn pick(variable: Option<&str>, requested: usize) -> usize {
    match variable.and_then(|v| v.trim().parse::<usize>().ok()) {
        Some(n) if n > 0 => n,
        _ => requested,
    }
}

/// A private Rayon pool, or the global one. Clones share the pool.
#[derive(Clone, Default)]
pub(crate) struct Pool(Option<Arc<rayon::ThreadPool>>);

impl Pool {
    /// The pool for a caller's `requested` count; see [`resolve`].
    pub(crate) fn new(requested: usize) -> Result<Self> {
        let threads = resolve(requested);
        if threads == 0 {
            return Ok(Self(None));
        }
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("sparta-{i}"))
            .build()
            .map(|pool| Self(Some(Arc::new(pool))))
            .map_err(|e| Error::Unsupported(format!("SPARTA thread pool: {e}")))
    }

    /// Whether this is a pool of its own, which bounds SPARTA's thread count.
    /// BLAS's own threading would escape that bound, so a bounded solver never
    /// hands a product to the library to thread.
    pub(crate) fn bounded(&self) -> bool {
        self.0.is_some()
    }

    /// Run `op` on this pool: every Rayon call inside it, and
    /// `rayon::current_num_threads`, then refer to this pool.
    pub(crate) fn run<R: Send>(&self, op: impl FnOnce() -> R + Send) -> R {
        match &self.0 {
            Some(pool) => pool.install(op),
            None => op(),
        }
    }

    /// The number of threads [`Pool::run`] runs on.
    pub(crate) fn threads(&self) -> usize {
        self.run(rayon::current_num_threads).max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_variable_overrides_any_positive_count() {
        assert_eq!(pick(Some("6"), 0), 6);
        assert_eq!(pick(Some(" 6 "), 3), 6);
        for ignored in [None, Some(""), Some("0"), Some("-2"), Some("many")] {
            assert_eq!(pick(ignored, 3), 3, "{ignored:?}");
            assert_eq!(pick(ignored, 0), 0, "{ignored:?}");
        }
    }

    #[test]
    fn a_count_builds_a_pool_of_that_size() {
        let pool = Pool(Some(Arc::new(rayon::ThreadPoolBuilder::new().num_threads(3).build().unwrap())));
        assert!(pool.bounded());
        assert_eq!(pool.threads(), 3);
        assert!(pool.run(|| rayon::current_thread_index().is_some()));
        let global = Pool::default();
        assert!(!global.bounded());
        assert_eq!(global.threads(), rayon::current_num_threads());
    }
}
