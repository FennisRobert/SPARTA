// Unsymmetric LU: the triangular solve.
//
// Two sweeps rather than three -- there is no block diagonal between them --
// but the bookkeeping is heavier than the symmetric case in one specific way.
// The pivot order is a row permutation only: pivot k solves the equation that
// began at local row `perms[k]` and eliminates the variable of local column k.
// So forward substitution reads its input through `global_row` and has to hand
// its result over at `front_row`, and those are different slots.
//
// The handover uses the `diagonal` buffer as scratch. It is exactly n complex
// values indexed by `node.first + k`, which partitions the variables one front
// at a time, and it holds the LDL^T block diagonal that LU never forms -- so it
// is free, correctly sized, and private to the front that is using it.

// Width is a template parameter for the same reason as the LDL^T sweeps: a
// compile-time stride in these loops is worth more than a runtime one.
template <uint WIDTH>
static void forward_lu_impl(device const Node* nodes, device const uint* rows,
                            device const uint* ids, device float2* scratch,
                            device const uint* perms, device float2* x, constant uint4& range,
                            device const float2* factors, threadgroup float2* xs, uint2 group,
                            uint lane) {
    uint s = ids[range.x + group.x];
    Node node = nodes[s];
    uint m = node.nr, nc = node.nc;
    // One threadgroup per (front, right-hand side). The pivot-order handover
    // scratch is per column as well, so it is offset with x.
    x += group.y * range.z;
    scratch += group.y * range.z;
    device const float2* l = factors + node.loffset;
    device float2* y = scratch + node.first;
    // Gather P b. Every one of these slots belongs to this front alone: its
    // fully summed rows are its own, and every front that contributes to them
    // is a strict descendant, already finished at an earlier level.
    for (uint k = lane; k < nc; k += WIDTH)
        y[k] = x[global_row(node, k, rows, perms)];
    threadgroup_barrier(mem_flags::mem_device);
    // The triangular solve, in pivot order, entirely inside the scratch.
    for (uint k = 0; k < nc; k++) {
        float2 v = y[k];
        for (uint i = k + 1 + lane; i < nc; i += WIDTH)
            y[i] -= mul(l[k * m + i], v);
        threadgroup_barrier(mem_flags::mem_device);
    }
    // Hand over to column order. Legal in place because every one of the nc
    // slots was read into the scratch above before any of them was written.
    for (uint k = lane; k < nc; k += WIDTH)
        x[front_row(node, k, rows)] = y[k];
    threadgroup_barrier(mem_flags::mem_device);
    // The trailing rows belong to shared ancestors, so siblings can collide
    // there and those updates go through float atomics. They feed no part of
    // the solve above, so each accumulates its whole dot product in registers
    // and pays a single complex atomic rather than one per pivot.
    uint tail = m - nc;
    for (uint r0 = 0; r0 < tail; r0 += WIDTH) {
        uint i = nc + r0 + lane;
        float2 acc = float2(0);
        for (uint base = 0; base < nc; base += SOLVE_STAGE) {
            uint count = min((uint)SOLVE_STAGE, nc - base);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // Strided: the stage is SOLVE_STAGE wide whatever the dispatch is.
            for (uint t = lane; t < count; t += WIDTH)
                xs[t] = y[base + t];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (i < m)
                for (uint c = 0; c < count; c++)
                    acc += mul(l[(base + c) * m + i], xs[c]);
        }
        if (i < m)
            atomic_subtract(x, front_row(node, i, rows), acc);
    }
}
template <uint WIDTH>
static void backward_lu_impl(device const Node* nodes, device const uint* rows,
                             device const uint* ids, device float2* x, constant uint4& range,
                             device const float2* factors, device const float2* upper,
                             threadgroup float2* partial, uint2 group, uint lane, uint sg, uint sl,
                             uint simdw) {
    uint s = ids[range.x + group.x];
    Node node = nodes[s];
    uint m = node.nr, nc = node.nc, tail = m - nc;
    x += group.y * range.z;
    uint simds = (WIDTH + simdw - 1) / simdw;
    device const float2* l = factors + node.loffset;
    device const float2* u = upper + node.uoffset;
    // Indices are unpermuted throughout: this sweep is over columns, and only
    // rows were ever interchanged.
    for (uint kk = nc; kk > 0; kk--) {
        uint k = kk - 1;
        float2 acc = float2(0);
        // U11 sits in the upper part of the stored L block -- column j, row k --
        // because the first nc columns of a finished front hold both triangles.
        for (uint j = k + 1 + lane; j < nc; j += WIDTH)
            acc -= mul(l[j * m + k], x[front_row(node, j, rows)]);
        // U12 is stored row-major, so row k of the trailing columns is contiguous.
        for (uint j = nc + lane; j < m; j += WIDTH)
            acc -= mul(u[k * tail + j - nc], x[front_row(node, j, rows)]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc = reduce_sum(acc, partial, sg, sl, simds);
        // U has a real diagonal to divide by; L's is the implicit unit one.
        if (lane == 0) {
            uint at = front_row(node, k, rows);
            x[at] = cdiv(x[at] + acc, l[k * m + k]);
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
}

#define SOLVE_FORWARD_LU(name, width)                                                              \
    kernel void name(                                                                              \
        device const Node* nodes [[buffer(0)]], device const uint* rows [[buffer(1)]],             \
        device const uint* ids [[buffer(5)]], device float2* scratch [[buffer(8)]],                \
        device const uint* perms [[buffer(10)]], device float2* x [[buffer(13)]],                  \
        constant uint4& range [[buffer(14)]], device const float2* factors [[buffer(17)]],         \
        uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]]) { \
        threadgroup float2 xs[SOLVE_STAGE];                                                        \
        forward_lu_impl<width>(nodes, rows, ids, scratch, perms, x, range, factors, xs, group,     \
                               lane);                                                              \
    }
#define SOLVE_BACKWARD_LU(name, width)                                                            \
    kernel void name(                                                                             \
        device const Node* nodes [[buffer(0)]], device const uint* rows [[buffer(1)]],            \
        device const uint* ids [[buffer(5)]], device float2* x [[buffer(13)]],                    \
        constant uint4& range [[buffer(14)]], device const float2* factors [[buffer(17)]],        \
        device const float2* upper [[buffer(20)]], uint2 group [[threadgroup_position_in_grid]],  \
        uint lane [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],    \
        uint sl [[thread_index_in_simdgroup]], uint simdw [[threads_per_simdgroup]]) {            \
        threadgroup float2 partial[MAX_SIMDS];                                                    \
        backward_lu_impl<width>(nodes, rows, ids, x, range, factors, upper, partial, group, lane, \
                                sg, sl, simdw);                                                   \
    }

SOLVE_FORWARD_LU(solve_forward_lu, SOLVE_THREADS)
SOLVE_FORWARD_LU(solve_forward_lu_wide, SOLVE_WIDE)
SOLVE_BACKWARD_LU(solve_backward_lu, SOLVE_THREADS)
SOLVE_BACKWARD_LU(solve_backward_lu_wide, SOLVE_WIDE)
