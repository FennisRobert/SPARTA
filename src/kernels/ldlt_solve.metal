// Complex-symmetric LDL^T: the triangular solve.
//
// Three sweeps over the assembly tree, one threadgroup per front: forward
// substitution with L, the block-diagonal solve with D, and backward
// substitution with L^T. Forward runs leaves first, backward roots first.
//
// Both sweeps are serial in the column index and spend most of their time at
// the top of the tree, where a level holds one or two fronts: on a 109k EMerge
// system the six levels with fourteen threadgroups or fewer are about two
// thirds of the solve, and the root front alone -- 1582 x 614 -- costs 0.7 ms
// forward and 0.9 ms backward against an arithmetic bound near 1.4 us. So the
// cost is latency, not work.
//
// Blocking the column loop was the obvious answer and it does not pay. Taking
// SOLVE_BLOCK columns per barrier divides the barrier count and the traffic in
// the solution vector by that width, and it was measured at widths 2, 4, 8 and
// 16 on both sweeps, with the block's own triangle staged in threadgroup memory
// so the serial tail is not a chain of dependent device loads. Every width was
// slower than the loop below. Forward improved on the mid-levels and lost more
// than that at the root; backward lost everywhere, the per-lane accumulator
// block costing more occupancy than the barriers were worth.
//
// Hoisting work out of the serial loop does not help either. The trailing rows
// are final before the backward sweep begins, so the whole L21^T z2 product can
// be taken once ahead of the triangle instead of from inside every column of
// it -- on the root front that lifts a 614 x 968 product out of a 614-step loop
// and leaves it the 614 x 614 triangle. Implemented with one simdgroup per
// column, so the lifted phase needs no threadgroup barrier at all, it measured
// neutral to slightly negative.
//
// Those two results together say the cost is the serial chain itself, not what
// is inside it and not the barrier count. Roughly 614 sequential
// barrier-and-reduce steps cost what they cost whether the work between them is
// a 968-row product or nothing. Making this faster means removing the chain,
// not feeding it better, and the way to remove it is to store each supernode's
// diagonal block already inverted -- selective inversion -- so both sweeps
// become dense products. That changes the factorization and the error
// behaviour, so it is a separate piece of work, but it is the only one of these
// three that can win. Do not re-block or re-hoist this loop without measuring.
//
// Both sweeps stride over a front's rows, so the useful threadgroup width
// follows how tall the fronts in a level are. The width is a template
// parameter rather than a dispatch argument: these inner loops are tight
// enough that a compile-time stride is worth measurably more than the
// flexibility, most of all on a level of short fronts. The host
// picks a pipeline per level; see solve_width in metal.m.

template <uint WIDTH>
static void forward_impl(device const Node* nodes, device const uint* rows, device const uint* ids,
                         device const uint* perms, device float2* x, constant uint4& range,
                         device const float2* factors, threadgroup float2* xs, uint2 group,
                         uint lane) {
    uint s = ids[range.x + group.x];
    Node node = nodes[s];
    uint m = node.nr, nc = node.nc;
    // One threadgroup per (front, right-hand side). Columns of the block are
    // independent here, so batching them is extra grid, not extra algorithm.
    x += group.y * range.z;
    device const float2* l = factors + node.loffset;
    // Phase 1: the triangular solve, confined to this front's own fully summed
    // rows. Those rows belong to the front alone, each lane owns a distinct one,
    // and no sibling can touch them, so none of this needs atomics.
    for (uint k = 0; k < nc; k++) {
        float2 v = x[global_row(node, k, rows, perms)];
        for (uint i = k + 1 + lane; i < nc; i += WIDTH)
            x[global_row(node, i, rows, perms)] -= mul(l[k * m + i], v);
        threadgroup_barrier(mem_flags::mem_device);
    }
    // Phase 2: the trailing rows, which belong to shared ancestors and so do
    // need atomics. They feed no part of the solve above -- ancestors consume
    // them at later levels -- so each one can accumulate its whole dot product
    // in registers and pay a single complex atomic, rather than one per pivot.
    // The solved block is staged in chunks so the indirection through perms and
    // rows is read once per group instead of once per lane per pivot.
    uint tail = m - nc;
    for (uint r0 = 0; r0 < tail; r0 += WIDTH) {
        uint i = nc + r0 + lane;
        float2 acc = float2(0);
        for (uint base = 0; base < nc; base += SOLVE_STAGE) {
            uint count = min((uint)SOLVE_STAGE, nc - base);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Strided: the stage is SOLVE_STAGE wide whatever WIDTH is, so a
            // narrow group fills it in several passes.
            for (uint t = lane; t < count; t += WIDTH)
                xs[t] = x[global_row(node, base + t, rows, perms)];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (i < m)
                for (uint c = 0; c < count; c++)
                    acc += mul(l[(base + c) * m + i], xs[c]);
        }
        if (i < m)
            atomic_subtract(x, global_row(node, i, rows, perms), acc);
    }
}

template <uint WIDTH>
static void backward_impl(device const Node* nodes, device const uint* rows, device const uint* ids,
                          device const uint* perms, device float2* x, constant uint4& range,
                          device const float2* factors, threadgroup float2* partial, uint2 group,
                          uint lane, uint sg, uint sl, uint simdw) {
    uint s = ids[range.x + group.x];
    Node node = nodes[s];
    uint m = node.nr;
    x += group.y * range.z;
    uint simds = (WIDTH + simdw - 1) / simdw;
    device const float2* l = factors + node.loffset;
    for (uint kk = node.nc; kk > 0; kk--) {
        uint k = kk - 1;
        float2 acc = float2(0);
        for (uint j = k + 1 + lane; j < m; j += WIDTH)
            acc -= mul(l[k * m + j], x[global_row(node, j, rows, perms)]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc = reduce_sum(acc, partial, sg, sl, simds);
        if (lane == 0)
            x[global_row(node, k, rows, perms)] += acc;
        threadgroup_barrier(mem_flags::mem_device);
    }
}

#define SOLVE_FORWARD(name, width)                                                                 \
    kernel void name(                                                                              \
        device const Node* nodes [[buffer(0)]], device const uint* rows [[buffer(1)]],             \
        device const uint* ids [[buffer(5)]], device const uint* perms [[buffer(10)]],             \
        device float2* x [[buffer(13)]], constant uint4& range [[buffer(14)]],                     \
        device const float2* factors [[buffer(17)]], uint2 group [[threadgroup_position_in_grid]], \
        uint lane [[thread_index_in_threadgroup]]) {                                               \
        threadgroup float2 xs[SOLVE_STAGE];                                                        \
        forward_impl<width>(nodes, rows, ids, perms, x, range, factors, xs, group, lane);          \
    }
#define SOLVE_BACKWARD(name, width)                                                                \
    kernel void name(                                                                              \
        device const Node* nodes [[buffer(0)]], device const uint* rows [[buffer(1)]],             \
        device const uint* ids [[buffer(5)]], device const uint* perms [[buffer(10)]],             \
        device float2* x [[buffer(13)]], constant uint4& range [[buffer(14)]],                     \
        device const float2* factors [[buffer(17)]], uint2 group [[threadgroup_position_in_grid]], \
        uint lane [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],     \
        uint sl [[thread_index_in_simdgroup]], uint simdw [[threads_per_simdgroup]]) {             \
        threadgroup float2 partial[MAX_SIMDS];                                                     \
        backward_impl<width>(nodes, rows, ids, perms, x, range, factors, partial, group, lane, sg, \
                             sl, simdw);                                                           \
    }

SOLVE_FORWARD(solve_forward, SOLVE_THREADS)
SOLVE_FORWARD(solve_forward_wide, SOLVE_WIDE)
SOLVE_BACKWARD(solve_backward, SOLVE_THREADS)
SOLVE_BACKWARD(solve_backward_wide, SOLVE_WIDE)

kernel void solve_diagonal(device const Node* nodes [[buffer(0)]],
                           device const uint* rows [[buffer(1)]],
                           device const float2* diagonal [[buffer(8)]],
                           device const float2* subdiag [[buffer(9)]],
                           device const uint* perms [[buffer(10)]],
                           device const uint* two [[buffer(11)]], device float2* x [[buffer(13)]],
                           constant uint4& range [[buffer(14)]],
                           uint2 tid [[thread_position_in_grid]]) {
    if (tid.x >= range.y || tid.y >= range.w)
        return;
    Node node = nodes[range.x + tid.x];
    uint k = 0;
    x += tid.y * range.z;
    while (k < node.nc) {
        uint i = global_row(node, k, rows, perms);
        uint d = node.first + k;
        if (two[d]) {
            uint j = global_row(node, k + 1, rows, perms);
            float2 a = diagonal[d], b = subdiag[d], c = diagonal[d + 1];
            float scale = max(mag(a), max(mag(b), mag(c)));
            a /= scale;
            b /= scale;
            c /= scale;
            float2 det = mul(a, c) - mul(b, b), u = x[i] / scale, v = x[j] / scale;
            x[i] = cdiv(mul(u, c) - mul(v, b), det);
            x[j] = cdiv(mul(v, a) - mul(u, b), det);
            k += 2;
        } else {
            x[i] = cdiv(x[i], diagonal[d]);
            k++;
        }
    }
}

// Large fronts use cooperative pivot preparation followed by tiled updates
// across independent threadgroups. No group waits for another group in a kernel.
// next: pivots taken so far. start/width: the most recent pivot, driving the
// fully summed update. panel: first pivot whose Schur contribution is still
// pending, so [panel, next) is the rank-k block the Schur update applies.
