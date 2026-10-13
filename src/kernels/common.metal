// Shared foundation for every SPARTA kernel: the descriptors that cross the
// Rust/Metal boundary, complex arithmetic, threadgroup reductions, the pivot
// failure and perturbation policy, the tiling constants, and the two kernels
// that belong to no single factorization.
//
// Everything here is used by both the LDL^T and the LU path. Anything that
// only one of them needs lives in that path's own file.
//
// Concatenated first by build.rs; the remaining files depend on it.
#include <metal_stdlib>
using namespace metal;
// Must match analysis::Node.
// Must match analysis::Node. `soffset` locates this front's contribution
// block once store_factors has compacted it out of the front; the front slot
// itself is recycled from the next level on, so the parent reads the block
// there and never touches `offset`.
struct Node {
    uint offset, nr, nc, rows, children, count, entries, nentries, first, loffset, woffset,
        uoffset, soffset, sld;
};
struct Child {
    uint node, map;
};
struct Entry {
    uint source, position;
};
// Per-front progress, shared by every numeric kernel. next: pivots taken so
// far. start/width: the most recent pivot, driving the fully summed update.
// panel: first pivot whose Schur contribution is still pending, so
// [panel, next) is the rank-k block the Schur update applies. perturbed:
// pivots floored rather than failed on. Kept deliberately small: update_strip
// and update_schur read one of these per threadgroup, hundreds of thousands of
// times per factorization, so the reference scale a perturbation needs is
// recomputed on demand instead of being stored here.
// Must match analysis::FrontState.
struct FrontState {
    uint next, start, width, panel;
    uint perturbed;
};
// Four multiplies and two adds, and it stays that way. metal.m builds with
// fastMathEnabled = NO, so the compiler will not contract these into fused
// multiply-adds on its own, and two hand-written alternatives were measured
// against this one on an M4 Pro:
//
//   acc.x = fma(-a.x, b.x, acc.x); acc.x = fma(a.y, b.y, acc.x);   (and .y)
//     Four operations instead of eight for a multiply-and-accumulate. Worth
//     4.9-6.7% off update_schur -- and 30% more residual on a curl-curl grid
//     (2.84e-7 -> 3.71e-7 at 24^3, 3.19e-7 -> 4.17e-7 at 30^3). Folding the
//     products straight into the accumulator defers the cancellation between
//     them, so the accumulator's low bits are lost instead of the products'.
//     The unstructured EMerge systems were indifferent, within +-2%.
//
//   re = fma(-a.y, b.y, a.x * b.x); im = fma(a.y, b.x, a.x * b.y);
//     Keeps the subtraction intact, so accuracy matches this form within 1%
//     everywhere. Six operations rather than eight, and worth nothing at all:
//     update_schur moved +0.1 to +0.5%, inside noise. The negated operand
//     costs an instruction the count does not show.
//
// These kernels are not ALU-bound enough for either to pay. Before trying a
// third variant, measure the kernels' occupancy (SPARTA_PROFILE=2): the
// update kernels run thousands of threadgroups deep and are limited by
// threadgroup-memory bandwidth, not by this expression.
float2 mul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
float mag(float2 a) {
    return length(a);
}
float2 cdiv(float2 a, float2 b) {
    if (abs(b.x) >= abs(b.y)) {
        float r = b.y / b.x;
        float d = b.x + b.y * r;
        return float2((a.x + a.y * r) / d, (a.y - a.x * r) / d);
    }
    float r = b.x / b.y;
    float d = b.y + b.x * r;
    return float2((a.x * r + a.y) / d, (a.y * r - a.x) / d);
}
// Upper bound on SIMD groups in one threadgroup, sizing every reduction's
// scratch. 1024 threads is the Metal maximum and 32 the Apple SIMD width, so a
// threadgroup never holds more than this many SIMD groups however wide the host
// dispatches it.
#define MAX_SIMDS 32
// status[s] is 0 on success, 0xffffffff when a child front failed, 0xfffffffe
// when a value went non-finite, and otherwise (reason<<24)|(k+1) for a pivot
// breakdown at fully summed column k. Fronts are bounded by a u32 element
// count, so nr < 65536 and the column always fits in 24 bits. The reason is
// what separates "this matrix is rank deficient" from "single precision ran
// out of resolution here", which need different responses from the caller.
#define PIVOT_ZERO_COLUMN 0x01000000u
#define PIVOT_NONFINITE 0x02000000u
#define PIVOT_TINY_2X2 0x03000000u
// Relative floor on a 2x2 pivot block's determinant. The block is scaled by its
// own largest magnitude first, so this is a pure relative test, and single
// precision carries about 1.19e-7. Below the floor the 2x2 inverse has no
// correct digits left.
#define PIVOT_EPS 1e-7f
// Static pivot perturbation. A pivot with no significant digits left in single
// precision is replaced by this fraction of the front's largest magnitude
// instead of failing the front. The factorization then returns L D L^T for
// A + dA with a small dA, which is still a good preconditioner -- the caller is
// told how many pivots were perturbed and can iterate to recover the accuracy.
// This is the same static pivoting SuperLU_DIST and MUMPS use.
//
// This is emphatically NOT a blanket floor on every pivot. Legitimate pivots in
// an indefinite Helmholtz front sit many orders of magnitude below the front's
// largest entry, and flooring those wrecks the factorization instead of
// rescuing it -- measured: a 3.45e-4 relative floor perturbed ~6500 pivots on a
// healthy 88.8k EMerge system and drove the residual from 1.4e-4 to 1.2e1.
// Perturbation fires only where the factorization would otherwise have failed
// outright: a pivot of exactly zero, or a 2x2 block with no usable inverse.
// Everywhere else the pivot is used exactly as found.
//
// sqrt(f32 epsilon) is the replacement value. Perturbing to the noise floor
// instead would leave multipliers of order 1/epsilon, risking overflow in the
// trailing updates; at this floor they stay under about 3e3.
#define PIVOT_FLOOR 3.45e-4f
// Replacement for a pivot that has genuinely died, keeping its phase when it has
// one so the perturbation scales the original entry rather than pointing
// somewhere new. `dead` is only ever reached from a previously fatal branch.
static float2 floor_pivot(float2 d, float floor) {
    float v = mag(d);
    return v > 0 ? d * (floor / v) : float2(floor, 0);
}
// A pivot the factorization cannot divide by at all.
static bool pivot_dead(float2 d) {
    return !(mag(d) > 0);
}
// --- Hierarchical threadgroup reductions ------------------------------------
// Every one of these replaces a log2(N)-stage barriered tree over threadgroup
// memory with one hardware SIMD reduction plus a short serial combine over the
// per-SIMD-group results: two barriers instead of seven at 128 threads, eight
// at 256. `simds` is how many SIMD groups the threadgroup holds; the scratch
// arrays are the callers' existing per-thread buffers, so only the first
// `simds` slots are used and nothing new is allocated.
//
// Each returns the same value on every lane, so callers need no broadcast pass.
// The caller still owns the barrier that makes the scratch safe to reuse.
static float reduce_max(float v, threadgroup float* scratch, uint sg, uint sl, uint simds) {
    v = simd_max(v);
    if (sl == 0)
        scratch[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = scratch[0];
    for (uint i = 1; i < simds; i++)
        r = max(r, scratch[i]);
    return r;
}
// Flags are 0/1, so a max is the OR.
static uint reduce_or(uint v, threadgroup uint* scratch, uint sg, uint sl, uint simds) {
    v = simd_max(v);
    if (sl == 0)
        scratch[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint r = scratch[0];
    for (uint i = 1; i < simds; i++)
        r = max(r, scratch[i]);
    return r;
}
static float2 reduce_sum(float2 v, threadgroup float2* scratch, uint sg, uint sl, uint simds) {
    v = float2(simd_sum(v.x), simd_sum(v.y));
    if (sl == 0)
        scratch[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float2 r = scratch[0];
    for (uint i = 1; i < simds; i++)
        r += scratch[i];
    return r;
}
// Largest magnitude, ties broken toward the lowest index. Callers pre-filter
// their own candidates with the identical rule, so a NaN never reaches here:
// it fails both `v>cm` and `v==cm`, exactly as it fails the serial scan. An
// infinity does reach here and survives the max, which is what the isfinite
// checks downstream rely on.
static void reduce_argmax(float v, uint idx, threadgroup float* fs, threadgroup uint* us, uint sg,
                          uint sl, uint simds, thread float* value, thread uint* at) {
    float best = simd_max(v);
    // Every lane compares against the group's own maximum, so at least one
    // matches and the min is a real index rather than the sentinel.
    uint where = simd_min(v == best ? idx : 0xffffffffu);
    if (sl == 0) {
        fs[sg] = best;
        us[sg] = where;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float bv = fs[0];
    uint bi = us[0];
    for (uint i = 1; i < simds; i++)
        if (fs[i] > bv || (fs[i] == bv && us[i] < bi)) {
            bv = fs[i];
            bi = us[i];
        }
    *value = bv;
    *at = bi;
}
// Largest magnitude in the front's fully summed columns, reduced across the
// threadgroup. Reached only from a branch that used to be a hard failure, so a
// healthy front never pays for this sweep. Group-uniform: every lane returns
// the same value.
//
// The sweep deliberately stops at column `n` instead of covering all `m*m`
// entries. The trailing-by-trailing Schur block is the one region whose
// staleness depends on the Schur drain cadence (see update_schur), so including
// it would make the replacement value a pivot is floored to depend on how often
// the host chose to drain -- the same matrix would perturb to a different
// number at cadence 16 and at final-only, and no A/B of the cadence could then
// be held to bitwise equality. Columns `< n` are covered by update_strip on the
// pivot-panel cadence, which is fixed, so this scale is cadence-invariant.
//
// It is also the more defensible reference: the pivot being rescued lives in
// the fully summed block, and the trailing block contributes no candidate.
//
// The effect is invariance, not accuracy. Measured against the old whole-front
// sweep on rescued two-level systems, the end-to-end solution difference was
// below the run-to-run spread of the ordering itself -- the floored value is a
// ~3.45e-4 relative perturbation either way, and which maximum it is relative
// to barely moves it. Do not re-widen this loop expecting a numerical change;
// the reason it is narrow is that the cadence above must not be able to reach
// it. It is also the cheaper sweep, m*n rather than m*m.
static float front_scale(device const float2* a, uint m, uint n, uint lane, uint threads,
                         threadgroup float* scratch, uint sg, uint sl, uint simds) {
    float big = 0;
    for (uint idx = lane; idx < m * n; idx += threads) {
        uint i = idx / n, j = idx - i * n;
        big = max(big, mag(a[i * m + j]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return reduce_max(big, scratch, sg, sl, simds);
}
// Where a front's local index lives in the global solution vector.
//
// Two mappings, and on the LU path they differ. `perms` records the pivot
// order, which on the LDL^T path is a symmetric permutation -- local index k
// names the same row and the same column -- so `global_row` serves for both.
// LU pivots by row interchange alone: pivot k solves the equation that started
// at local row `perms[k]`, but it eliminates the variable of local *column* k.
// So an LU solve reads its right-hand side through `global_row` and writes its
// answer through `front_row`. Trailing indices are unpermuted, and the two
// agree there.
uint global_row(Node node, uint local, device const uint* rows, device const uint* perms) {
    return rows[node.rows + (local < node.nc ? perms[node.first + local] : local)];
}
uint front_row(Node node, uint local, device const uint* rows) {
    return rows[node.rows + local];
}
// Threads per front in the solve sweeps, and the panel of the solved block
// staged per pass. Fixed rather than taken from the dispatch: a level here is
// mostly small fronts, so a wider group only leaves lanes idle, and a
// compile-time stride lets the compiler optimise these inner loops.
#ifndef SOLVE_THREADS
#define SOLVE_THREADS 256
#endif
// The wide variant, for levels whose fronts are tall enough to use it. Measured
// solve time on this machine at one fixed width across the whole tree: an 86.0k
// LU system took 0.033s at 256 and 0.027s at 512; a 68.7k LDL^T system whose
// tallest front is 644 rows took 0.004s at 256 and 0.006s at 512.
#ifndef SOLVE_WIDE
#define SOLVE_WIDE 512
#endif
#define SOLVE_STAGE 256
// The solve runs one threadgroup per front, level by level, rather than one
// threadgroup for the whole tree.
//
// A front's fully summed rows are its own: no other front writes them, and every
// front that contributes to them is a strict descendant. So within a level the
// fronts are independent apart from one case -- forward substitution writes a
// front's trailing rows, which belong to shared ancestors, so siblings can
// collide there and those updates go through float atomics. Backward
// substitution writes only fully summed rows and reads ancestor rows that are
// already final in reverse level order, so it needs no atomics at all.
static void atomic_subtract(device float2* x, uint at, float2 v) {
    device atomic_float* slot = (device atomic_float*)(x + at);
    atomic_fetch_sub_explicit(slot, v.x, memory_order_relaxed);
    atomic_fetch_sub_explicit(slot + 1, v.y, memory_order_relaxed);
}
struct UpdateTile {
    uint node, row, col;
};
#ifndef PIVOT_THREADS
#define PIVOT_THREADS 128
#endif
// Pivots accumulated per panel. Overridden by Config::panel, which the
// host injects as a macro; this value applies only if it does not.
#ifndef PANEL
#define PANEL 16
#endif
// Per-thread register block edge; the 16x16 threadgroup covers
// SCHUR_TILE = 16*SCHUR_BLOCK. Overridden by Config::schur_block.
#ifndef SCHUR_BLOCK
#define SCHUR_BLOCK 4
#endif
#define SCHUR_TILE (16 * SCHUR_BLOCK)
// Rows one strip threadgroup owns, as a per-thread register block of
// STRIP_BLOCK rows over the same 16x16 group. The strip keeps its 16-column
// edge -- a panel advances 16 columns, so a wider column tile would spend half
// its arithmetic on columns the panel has already finished -- and grows in the
// row direction instead, where a front is long and nothing is masked.
//
// Without the register block each thread owned one output and read two staged
// operands per multiply-add; at STRIP_BLOCK = 4 it reads five for four, and one
// threadgroup covers four times the rows, so the column operand is staged once
// per four row blocks and there are four times fewer groups to launch. This is
// the same ratio argument SCHUR_BLOCK is chosen by, applied to the one update
// kernel that did not have it.
//
// Measured on an M4 Pro: strip threadgroups over a whole factorization fall
// 3.7x, 439,468 to 117,291 on a 109.4k EMerge LDL^T system and 4,779,246 to
// 1,289,494 on an 86.0k LU one, and this kernel's own device time falls 1.60x
// and 1.42x. Overridden by Config::strip_block.
#ifndef STRIP_BLOCK
#define STRIP_BLOCK 4
#endif
#define STRIP_ROWS (16 * STRIP_BLOCK)
// Panel columns staged per pass. SCHUR_TILE*SCHUR_CHUNK*8 bytes per array, two
// arrays, must stay inside the 32 KiB threadgroup limit.
#ifndef SCHUR_CHUNK
#define SCHUR_CHUNK 16
#endif
#define PANEL_COLS (PANEL + 1)
// Closes a panel once its Schur contribution has been applied.
kernel void advance_panel(device const Node* nodes [[buffer(0)]],
                          device const uint* ids [[buffer(5)]],
                          constant uint2& range [[buffer(14)]],
                          device FrontState* state [[buffer(15)]],
                          uint tid [[thread_position_in_grid]]) {
    if (tid >= range.y)
        return;
    uint s = ids[range.x + tid];
    state[s].panel = state[s].next;
}
