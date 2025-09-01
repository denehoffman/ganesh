use crate::traits::{Algorithm, Callback, Status};
use parking_lot::Once;
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicBool, Ordering};

/// A trait for abort signals.
/// This trait is used in minimizers to check if the user has requested to abort the calculation.
pub trait AbortSignal {
    /// Return `true` if the user has requested to abort the calculation.
    fn is_aborted(&self) -> bool;
    /// Abort the calculation. Make `is_aborted()` return `true`.
    fn abort(&self);
    /// Reset the abort signal. Make `is_aborted()` return `false`.
    fn reset(&self);
}

impl<T, A, P, S, U, E> Callback<A, P, S, U, E> for T
where
    T: AbortSignal,
    A: Algorithm<P, S, U, E>,
    S: Status,
{
    fn callback(
        &mut self,
        _current_step: usize,
        _algorithm: &mut A,
        _problem: &mut P,
        _status: &mut S,
        _user_data: &mut U,
    ) -> ControlFlow<()> {
        if self.is_aborted() {
            self.reset();
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }
}

static INIT: Once = Once::new();
static CTRL_C_PRESSED: AtomicBool = AtomicBool::new(false);

/// A signal that is triggered when the user presses `Ctrl-C`.
/// <div class="warning">This signal takes over the `Ctrl-C` handler for the whole process and can interfere with
/// other libraries that use `Ctrl-C` (e.g. `tokio`).</div>
#[derive(Default)]
pub struct CtrlCAbortSignal;
impl CtrlCAbortSignal {
    /// Create a new `CtrlCAbortSignal` and register a ctrl-c handler.
    pub fn new() -> Self {
        let signal = Self {};
        signal.init_handler();
        signal
    }

    fn init_handler(&self) {
        INIT.call_once(|| {
            #[allow(clippy::expect_used)]
            ctrlc::set_handler(move || {
                println!("Ctrl-C pressed");
                CTRL_C_PRESSED.store(true, Ordering::SeqCst);
            })
            .expect("Error setting Ctrl-C handler");
        });
    }
}

impl AbortSignal for CtrlCAbortSignal {
    fn is_aborted(&self) -> bool {
        CTRL_C_PRESSED.load(Ordering::SeqCst)
    }

    fn abort(&self) {
        CTRL_C_PRESSED.store(true, Ordering::SeqCst)
    }

    fn reset(&self) {
        CTRL_C_PRESSED.store(false, Ordering::SeqCst);
    }
}

/// A signal that is triggered by setting an atomic boolean.
#[derive(Default)]
pub struct AtomicAbortSignal {
    abort: AtomicBool,
}

impl AtomicAbortSignal {
    /// Create a new `AtomicAbortSignal`.
    pub const fn new() -> Self {
        Self {
            abort: AtomicBool::new(false),
        }
    }
}

impl AbortSignal for AtomicAbortSignal {
    fn is_aborted(&self) -> bool {
        self.abort.load(Ordering::SeqCst)
    }

    fn abort(&self) {
        self.abort.store(true, Ordering::SeqCst);
    }

    fn reset(&self) {
        self.abort.store(false, Ordering::SeqCst);
    }
}
