//! Indexed max-heap with in-place key changes, for the FM move queue.
//! A lazy heap that pushes a fresh entry on every gain change grows with stale
//! entries and pops each one only to discard it; here every handle owns at
//! most one entry, so the heap holds only live moves.

/// Marks a handle with no entry in the heap.
const ABSENT: u32 = u32::MAX;

/// Children per heap node. A wider node halves the depth that a pop walks.
const ARITY: usize = 4;

/// Max-heap of `u128` keys, each owned by a handle below the capacity given
/// to [`IndexedHeap::new`]. Setting a handle that already has a key moves that
/// key instead of adding a second one.
pub(crate) struct IndexedHeap {
    keys: Vec<u128>,
    handles: Vec<u32>,
    at: Vec<u32>,
}
impl IndexedHeap {
    /// Creates an empty heap for handles `0..capacity`.
    pub fn new(capacity: usize) -> Self {
        Self {
            keys: Vec::new(),
            handles: Vec::new(),
            at: vec![ABSENT; capacity],
        }
    }
    /// Gives handle `h` the key `key`, inserting it or moving its entry.
    #[inline]
    pub fn set(&mut self, h: usize, key: u128) {
        let i = self.at[h];
        if i == ABSENT {
            self.keys.push(key);
            self.handles.push(h as u32);
            self.up(self.keys.len() - 1, key, h as u32);
        } else {
            let i = i as usize;
            if key > self.keys[i] {
                self.up(i, key, h as u32);
            } else {
                self.down(i, key, h as u32);
            }
        }
    }
    /// Removes and returns the largest key and its handle.
    #[inline]
    pub fn pop(&mut self) -> Option<(u128, usize)> {
        let top = *self.keys.first()?;
        let h = self.handles[0];
        self.at[h as usize] = ABSENT;
        let key = self.keys.pop().unwrap();
        let last = self.handles.pop().unwrap();
        if !self.keys.is_empty() {
            self.down(0, key, last);
        }
        Some((top, h as usize))
    }
    /// Removes the entry of handle `h`, if it has one.
    #[inline]
    pub fn remove(&mut self, h: usize) {
        let i = self.at[h];
        if i == ABSENT {
            return;
        }
        self.at[h] = ABSENT;
        let (i, key, last) = (i as usize, self.keys.pop().unwrap(), self.handles.pop().unwrap());
        if i < self.keys.len() {
            if key > self.keys[i] {
                self.up(i, key, last);
            } else {
                self.down(i, key, last);
            }
        }
    }
    /// Removes every entry, calling `f` with each removed handle.
    pub fn clear(&mut self, mut f: impl FnMut(usize)) {
        for &h in &self.handles {
            self.at[h as usize] = ABSENT;
            f(h as usize);
        }
        self.keys.clear();
        self.handles.clear();
    }
    /// Places `key` of handle `h` at slot `i` or above, moving smaller
    /// parents down into the hole it leaves.
    #[inline]
    fn up(&mut self, mut i: usize, key: u128, h: u32) {
        while i > 0 {
            let parent = (i - 1) / ARITY;
            if self.keys[parent] >= key {
                break;
            }
            self.keys[i] = self.keys[parent];
            self.handles[i] = self.handles[parent];
            self.at[self.handles[i] as usize] = i as u32;
            i = parent;
        }
        self.keys[i] = key;
        self.handles[i] = h;
        self.at[h as usize] = i as u32;
    }
    /// Places `key` of handle `h` at slot `i` or below, moving larger
    /// children up into the hole it leaves.
    #[inline]
    fn down(&mut self, mut i: usize, key: u128, h: u32) {
        let n = self.keys.len();
        loop {
            let first = i * ARITY + 1;
            if first >= n {
                break;
            }
            let mut best = first;
            for c in first + 1..(first + ARITY).min(n) {
                if self.keys[c] > self.keys[best] {
                    best = c;
                }
            }
            if self.keys[best] <= key {
                break;
            }
            self.keys[i] = self.keys[best];
            self.handles[i] = self.handles[best];
            self.at[self.handles[i] as usize] = i as u32;
            i = best;
        }
        self.keys[i] = key;
        self.handles[i] = h;
        self.at[h as usize] = i as u32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    /// Random sets, removals and pops match an ordered map of live keys.
    fn matches_ordered_map() {
        let mut heap = IndexedHeap::new(64);
        let mut live: BTreeMap<u128, usize> = BTreeMap::new();
        let mut owned = vec![None; 64];
        let mut x = 12345u64;
        for _ in 0..20000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            if x % 7 == 1 {
                let h = (x >> 8) as usize % 64;
                if let Some(old) = owned[h].take() {
                    live.remove(&old);
                }
                heap.remove(h);
            } else if x % 3 == 0 {
                let got = heap.pop();
                let want = live.pop_last();
                assert_eq!(got, want);
                if let Some((_, h)) = got {
                    owned[h] = None;
                }
            } else {
                let h = (x >> 8) as usize % 64;
                // Keys stay unique across handles, as FM's packed keys do.
                let key = ((x >> 20) % 1000) as u128 * 64 + h as u128;
                if let Some(old) = owned[h] {
                    live.remove(&old);
                }
                owned[h] = Some(key);
                live.insert(key, h);
                heap.set(h, key);
            }
        }
    }
}
