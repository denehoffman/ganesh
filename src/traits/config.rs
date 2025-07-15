/// A trait used to imply an [`Algorithm`](`crate::traits::Algorithm`) is configurable.
pub trait Configurable {
    /// The configuration struct for the algorithm.
    type Config;

    /// Convenience method to use chainable methods to setup the configuration struct.
    fn setup_config<F>(&mut self, mut f: F) -> &mut Self
    where
        F: FnMut(&mut Self::Config) -> &mut Self::Config,
        Self: Sized,
    {
        f(self.get_config_mut());
        self
    }
    /// A helper method to get the mutable internal configuration struct.
    fn get_config_mut(&mut self) -> &mut Self::Config;
}
