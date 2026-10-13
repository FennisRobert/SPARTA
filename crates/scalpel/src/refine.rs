use crate::{graph::Graph, Options};
use crate::heap::IndexedHeap;

/// Per-call FM scratch: lazily computed child-neighbor weights and stamps.
/// `conn[v][c]` is the weight of `v`'s neighbors in child `c`, valid only while
/// `valid[v]` equals the current pass stamp; label changes update every valid
/// neighbor, so a gain costs O(1) once its vertex has been seen in a pass.
/// `queued[v][side]` is the gain of the heap entry for that pair, handle
/// `2 * v + side`, or `NONE` when it has none. A touch that leaves a gain
/// unchanged therefore does not touch the heap.
/// `blocked[v][side] == pass` when moving `v` to `side` would pull a locked
/// vertex out of the other child. Locked vertices keep their child for the
/// rest of the pass, so such a pair stays out of the heap until the next one.
struct Scratch {
    conn: Vec<[usize; 2]>,
    valid: Vec<u32>,
    locked: Vec<bool>,
    blocked: Vec<[u32; 2]>,
    queued: Vec<[i64; 2]>,
    touched: Vec<u32>,
    pass: u32,
    step: u32,
}

/// Marks a vertex/side pair with no live heap entry. Gains are bounded by the
/// total vertex weight, so no real gain takes this value.
const NONE: i64 = i64::MIN;

impl Scratch {
    /// Allocates scratch for a graph of `n` vertices.
    fn new(n: usize) -> Self {
        Self {
            conn: vec![[0; 2]; n],
            valid: vec![0; n],
            locked: vec![false; n],
            blocked: vec![[0; 2]; n],
            queued: vec![[NONE; 2]; n],
            touched: vec![0; n],
            pass: 0,
            step: 0,
        }
    }
    /// Returns the separator weight saved by moving `v` of `g` to child `side`
    /// under labels `s`, computing `v`'s neighbor weights on first use.
    #[inline]
    fn gain(&mut self, g: &Graph, s: &[u8], v: usize, side: u8) -> i64 {
        if self.valid[v] != self.pass {
            let mut c = [0usize; 3];
            for &u in g.neighbors(v) {
                let u = u as usize;
                // Separator neighbors land in the unused third slot, without a branch.
                c[s[u] as usize] += g.weight[u];
            }
            self.conn[v] = [c[0], c[1]];
            self.valid[v] = self.pass;
        }
        g.weight[v] as i64 - self.conn[v][1 - side as usize] as i64
    }
    /// Queues vertex `v` of `g` for target `side` at its current gain in `heap`,
    /// unless its live entry already carries that gain.
    #[inline]
    fn queue(&mut self, g: &Graph, s: &[u8], v: usize, side: u8, heap: &mut IndexedHeap) {
        if self.blocked[v][side as usize] == self.pass {
            return;
        }
        let gain = self.gain(g, s, v, side);
        let slot = &mut self.queued[v][side as usize];
        if *slot == gain {
            return;
        }
        *slot = gain;
        heap.set(2 * v + side as usize, pack(gain, v, side));
    }
    /// Records that vertex `x` of `g` left child `from` and/or joined child `to`
    /// (2 for the separator) in every valid neighbor count.
    #[inline]
    fn relabel(&mut self, g: &Graph, x: usize, from: u8, to: u8) {
        let w = g.weight[x];
        for &y in g.neighbors(x) {
            let y = y as usize;
            if self.valid[y] == self.pass {
                if from < 2 {
                    self.conn[y][from as usize] -= w;
                }
                if to < 2 {
                    self.conn[y][to as usize] += w;
                }
            }
        }
    }
}

/// Heap entry ordered as (gain, lower vertex index first, higher side first):
/// the gain with its sign bit flipped fills the high 64 bits, then the
/// reversed vertex index and the side.
type Entry = u128;

/// Packs `gain`, vertex `v` and target `side` into an ordered heap entry.
#[inline]
fn pack(gain: i64, v: usize, side: u8) -> Entry {
    let high = (gain as u64 ^ (1 << 63)) as u128;
    (high << 64) | (((u32::MAX - v as u32) as u128) << 1) | side as u128
}

/// Recovers the gain, vertex and side packed by [`pack`].
#[inline]
fn unpack(e: Entry) -> (i64, usize, u8) {
    let gain = ((e >> 64) as u64 ^ (1 << 63)) as i64;
    let v = u32::MAX - ((e as u64 >> 1) as u32);
    (gain, v as usize, (e & 1) as u8)
}

/// FM refinement of weighted graph `g`, with rollback to the best balanced prefix.
/// Labels `s` are modified in place; `o` bounds sweeps and partition imbalance.
pub(crate) fn fm(g: &Graph, s: &mut [u8], o: &Options) {
    if o.fm_passes == 0 {
        return;
    }
    let mut w = crate::separator::weights(g, s);
    let limit = ((w[0] + w[1] + w[2]) as f64 * o.imbalance).ceil() as usize;
    let mut x = Scratch::new(g.len());
    let mut changes: Vec<(u32, u8)> = Vec::new();
    let mut affected: Vec<usize> = Vec::new();
    let mut heap = IndexedHeap::new(2 * g.len());
    for _ in 0..o.fm_passes {
        let preferred = if w[0] <= w[1] { 0 } else { 1 };
        let sides: &[u8] = if !o.one_sided_refinement {
            &[0, 1]
        } else if preferred == 0 {
            &[0]
        } else {
            &[1]
        };
        x.pass += 1;
        for &(v, old) in &changes {
            if old == 2 {
                x.locked[v as usize] = false;
            }
        }
        changes.clear();
        for (v, &label) in s.iter().enumerate() {
            if label == 2 {
                for &side in sides {
                    x.queue(g, s, v, side, &mut heap);
                }
            }
        }
        let mut best = crate::separator::quality(w, o);
        let initial = best;
        let mut best_w = w;
        let mut best_at = 0;
        let mut stale = 0;
        while let Some((entry, _)) = heap.pop() {
            let (priority, v, side) = unpack(entry);
            x.queued[v][side as usize] = NONE;
            if x.locked[v] || s[v] != 2 || w[side as usize] + g.weight[v] > limit {
                continue;
            }
            let other = 1 - side;
            debug_assert!(!g
                .neighbors(v)
                .iter()
                .any(|&u| s[u as usize] == other && x.locked[u as usize]));
            let actual = x.gain(g, s, v, side);
            if actual != priority {
                x.queue(g, s, v, side, &mut heap);
                continue;
            }
            let pulled = (g.weight[v] as i64 - actual) as usize;
            if pulled >= w[other as usize] {
                continue;
            }
            x.locked[v] = true;
            changes.push((v as u32, 2));
            s[v] = side;
            // `v` is locked in `side` now: no neighbor may move to `other`.
            let pass = x.pass;
            x.queued[v][other as usize] = NONE;
            heap.remove(2 * v + other as usize);
            for &u in g.neighbors(v) {
                let u = u as usize;
                x.blocked[u][other as usize] = pass;
                x.queued[u][other as usize] = NONE;
                heap.remove(2 * u + other as usize);
            }
            x.relabel(g, v, 2, side);
            w[2] -= g.weight[v];
            w[side as usize] += g.weight[v];
            affected.clear();
            affected.push(v);
            for &u in g.neighbors(v) {
                let u = u as usize;
                if s[u] == other {
                    changes.push((u as u32, other));
                    s[u] = 2;
                    affected.push(u);
                    w[other as usize] -= g.weight[u];
                    w[2] += g.weight[u];
                }
            }
            for &u in &affected[1..] {
                x.relabel(g, u, other, 2);
            }
            x.step += 1;
            for &u in &affected {
                for a in std::iter::once(u).chain(g.neighbors(u).iter().map(|&a| a as usize)) {
                    if s[a] == 2 && !x.locked[a] && x.touched[a] != x.step {
                        x.touched[a] = x.step;
                        for &target in sides {
                            x.queue(g, s, a, target, &mut heap);
                        }
                    }
                }
            }
            let key = crate::separator::quality(w, o);
            if w[0].max(w[1]) <= limit && w[0] > 0 && w[1] > 0 && key < best {
                best = key;
                best_w = w;
                best_at = changes.len();
                stale = 0;
            } else {
                stale += 1;
            }
            if stale >= 64 {
                break;
            }
        }
        // Entries still queued would read as live next pass; clear only those.
        heap.clear(|h| x.queued[h / 2][h % 2] = NONE);
        for &(v, old) in changes[best_at..].iter().rev() {
            s[v as usize] = old;
        }
        w = best_w;
        if best >= initial {
            break;
        }
    }
}
