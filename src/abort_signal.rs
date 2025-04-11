use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::Once;

/// A trait for abort signals.
/// This trait is used in minimizers to check if the user has requested to abort the calculation.
pub trait AbortSignal {
    /// Return `true` if the user has requested to abort the calculation.
    fn is_aborted(&self) -> bool;
    /// Abort the calculation. Make `is_aborted()` return `true`.
    fn abort(&self);
    /// Reset the abort signal. Make `is_aborted()` return `false`.
    fn reset(&self);
    /// Return a boxed version of the signal.
    fn boxed(self) -> Box<Self>
    where
        Self: Sized,
    {
        Box::new(self)
    }
}

static INIT: Once = Once::new();
static CTRL_C_PRESSED: AtomicBool = AtomicBool::new(false);

/// A signal that is triggered when the user presses `Ctrl-C`.
/// <div class="warning">This signal takes over the `Ctrl-C` handler for the whole process and can interfere with
/// other libraries that use `Ctrl-C` (e.g. `tokio`).</div>
pub struct CtrlCAbortSignal {}
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

/// A signal that is never triggered.
pub struct NopAbortSignal;

impl NopAbortSignal {
    /// Create a new `NopAbortSignal`.
    pub fn new() -> Self {
        Self {}
    }
}

impl AbortSignal for NopAbortSignal {
    fn is_aborted(&self) -> bool {
        false
    }

    fn abort(&self) {}

    fn reset(&self) {}
}

/// A signal that is triggered by setting an atomic boolean.
pub struct AtomicAbortSignal {
    abort: AtomicBool,
}

impl AtomicAbortSignal {
    /// Create a new `AtomicAbortSignal`.
    pub fn new() -> Self {
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
