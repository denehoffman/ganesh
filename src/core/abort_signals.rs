use parking_lot::Once;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::traits::AbortSignal;
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
