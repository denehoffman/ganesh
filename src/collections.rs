struct LimitedDeque<T>(VecDeque<T>, usize);
impl<T> LimitedDeque<T> {
    fn new(size: usize) -> Self {
        Self(VecDeque::default(), size)
    }
    fn push_back(&mut self, value: T) {
        if self.is_full() {
            self.0.pop_front();
        }
        self.0.push_back(value);
    }
    fn push_front(&mut self, value: T) {
        if self.is_full() {
            self.0.pop_back();
        }
        self.0.push_front(value);
    }
    pub fn pop_back(&mut self) -> Option<T> {
        self.0.pop_back()
    }

    pub fn pop_front(&mut self) -> Option<T> {
        self.0.pop_front()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn is_full(&self) -> bool {
        self.0.len() == self.1
    }

    pub fn clear(&mut self) {
        self.0.clear();
    }
}

impl<T> std::ops::Deref for LimitedDeque<T> {
    type Target = VecDeque<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> std::ops::DerefMut for LimitedDeque<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
