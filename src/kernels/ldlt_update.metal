// Complex-symmetric LDL^T: the deferred trailing updates.
//
// `update_strip` refreshes the fully summed columns after every panel, because
// that is what the next pivot search reads. `update_schur` drains the
// trailing-by-trailing block on the host's own cadence, since nothing reads it
// until the parent assembles it. Both fold the block diagonal into one operand.
// The strip writes each result twice inside the fully summed block, where the
// pivot row search reads the mirror; the Schur block is kept lower only
// (SPARTA_LOWER in common.metal).
// Folds the block diagonal into one row of the factor panel. Reads the paired
// column straight from device memory, which keeps the staged panel to the two
// arrays actually consumed by the inner product.
static T fold_diagonal(device const T* row, uint first, uint k, device const uint* two,
                            device const float2* diagonal, device const float2* subdiag) {
    if (two[first + k])
        return mul(row[k], narrow(diagonal[first + k])) + mul(row[k + 1], narrow(subdiag[first + k]));
    if (k > 0 && two[first + k - 1])
        return mul(row[k - 1], narrow(subdiag[first + k - 1])) + mul(row[k], narrow(diagonal[first + k]));
    return mul(row[k], narrow(diagonal[first + k]));
}

// Deferred rank-`width` update of the fully summed columns past the panel.
// Same shape as update_schur, over the strip that later pivot searches read:
// one STRIP_ROWS x 16 tile per threadgroup, a STRIP_BLOCK x 1 register block
// per thread.
kernel void update_strip(
    device const Node* nodes [[buffer(0)]], device float2* fronts [[buffer(7)]],
    device const float2* diagonal [[buffer(8)]], device const float2* subdiag [[buffer(9)]],
    device const uint* two [[buffer(11)]], device atomic_uint* status [[buffer(12)]],
    constant uint2& range [[buffer(14)]], device const FrontState* state [[buffer(15)]],
    device const UpdateTile* jobs [[buffer(16)]], uint2 group [[threadgroup_position_in_grid]],
    uint2 lane [[thread_position_in_threadgroup]]) {
    UpdateTile job = jobs[range.x + group.x];
    uint s = job.node;
    Node node = nodes[s];
    FrontState st = state[s];
    uint end = st.start + st.width;
    // Every tile is scheduled for the whole front, but a tile whose columns are
    // all factored already writes nothing: the per-element guard below discards
    // j < end, and j >= nc belongs to the Schur complement. Deciding that here,
    // before the first barrier, skips the staging and the rank-k product for a
    // tile that would throw the result away. Group-uniform: `end` and `job.col`
    // are identical on every lane.
    if (st.width == 0 || end >= node.nc || (job.col + 1u) * 16u <= end ||
        atomic_load_explicit(status + s, memory_order_relaxed) != 0)
        return;
    uint m = node.nr, first = node.first;
    device T* a = (device T*)(fronts + node.offset);
    // The row operand spans STRIP_ROWS rows and the column operand only 16, so
    // the two stages are separate loops rather than one fused pass: the column
    // block is staged once for four row blocks' worth of output.
    threadgroup T w[STRIP_ROWS][SCHUR_CHUNK], lj[16][SCHUR_CHUNK];
    uint rowbase = job.row * STRIP_ROWS, colbase = job.col * 16;
    uint linear = lane.y * 16 + lane.x;
    T acc[STRIP_BLOCK];
    for (uint d = 0; d < STRIP_BLOCK; d++)
        acc[d] = T(0);
    for (uint c0 = st.start; c0 < end;) {
        uint width = min((uint)SCHUR_CHUNK, end - c0);
        if (c0 + width < end && two[first + c0 + width - 1])
            width--;
        for (uint idx = linear; idx < STRIP_ROWS * width; idx += 256) {
            uint r = idx / width, c = idx - r * width, ri = rowbase + r;
            w[r][c] = ri < m ? fold_diagonal(a + ri * m, first, c0 + c, two, diagonal, subdiag)
                             : T(0);
        }
        for (uint idx = linear; idx < 16 * width; idx += 256) {
            uint r = idx / width, c = idx - r * width, rj = colbase + r;
            lj[r][c] = rj < m ? a[rj * m + c0 + c] : T(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = 0; c < width; c++) {
            T lv = lj[lane.x][c];
            for (uint d = 0; d < STRIP_BLOCK; d++)
                acc[d] -= mul(w[STRIP_BLOCK * lane.y + d][c], lv);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        c0 += width;
    }
    uint j = colbase + lane.x;
    // Columns inside the panel already hold their factor; columns at or past nc
    // are the Schur complement and are drained separately.
    if (j >= node.nc || j < end)
        return;
    for (uint d = 0; d < STRIP_BLOCK; d++) {
        uint i = rowbase + STRIP_BLOCK * lane.y + d;
        if (i >= m || j > i)
            continue;
        T v = a[i * m + j] + acc[d];
        if (!finite(v))
            atomic_store_explicit(status + s, 0xfffffffeu, memory_order_relaxed);
        a[i * m + j] = v;
        // The mirror is worth writing only where something reads it. Rows
        // inside the fully summed block are read by the next pivot's row
        // search; past it the mirror lands in U12, which nothing on this path
        // reads -- see the note in cooperative_swap.
        if (i < node.nc)
            a[j * m + i] = v;
    }
}

// Blocked Schur update: one rank-(next-panel) product per drain instead of one
// rank-1 product per pivot. Each threadgroup owns a 16x16 block of the Schur
// complement, stages both factor row blocks in threadgroup memory, and folds
// the block diagonal into one side. This is the step that moves the trailing
// update off the memory roofline: traffic per element drops by the drained
// width.
//
// `[state.panel, state.next)` is a whole batch of pivot panels, not one panel:
// the host drains on its own cadence (SPARTA_SCHUR_BATCH in metal.m), so this
// range is as wide as the host let it grow. Nothing here assumes otherwise --
// the chunk loop already walks an arbitrary range in SCHUR_CHUNK columns and
// already refuses to split a 2x2 pivot across chunks. Accumulating the whole
// batch in `acc` before touching `a` is what saves the traffic, and it rounds
// the destination once per batch instead of once per panel.
#if SCHUR_TILE == 64 && SCHUR_CHUNK % 8 == 0 && SCHUR_CHUNK <= 16
// The same update on the simdgroup matrix units. The tile is an 8 x 8 grid of
// 8 x 8 blocks, and simdgroup `sg` owns block row `sg`, accumulating its
// eight blocks in registers. A complex product is four real ones, with the
// real and imaginary parts staged separately and the imaginary part of the
// row operand staged negated as well, since the units only add:
//   re += Wr Lr^T + (-Wi) Li^T,   im += Wr Li^T + Wi Lr^T.
// The real copy stages and multiplies the real parts alone.
// Staging, the chunk loop and the 2x2 rule are those of the scalar kernel
// below, which remains for the geometries this one does not cover.
kernel void update_schur(
    device const Node* nodes [[buffer(0)]], device float2* fronts [[buffer(7)]],
    device const float2* diagonal [[buffer(8)]], device const float2* subdiag [[buffer(9)]],
    device const uint* two [[buffer(11)]], device atomic_uint* status [[buffer(12)]],
    constant uint2& range [[buffer(14)]], device const FrontState* state [[buffer(15)]],
    device const UpdateTile* jobs [[buffer(18)]], uint2 group [[threadgroup_position_in_grid]],
    uint2 lane [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint sl [[thread_index_in_simdgroup]]) {
    UpdateTile job = jobs[range.x + group.x];
    uint s = job.node;
    Node node = nodes[s];
    FrontState st = state[s];
    if (st.next <= st.panel || atomic_load_explicit(status + s, memory_order_relaxed) != 0)
        return;
    uint m = node.nr, base = node.nc, first = node.first;
    device T* a = (device T*)(fronts + node.offset);
    threadgroup float wr[SCHUR_TILE][SCHUR_CHUNK], lr[SCHUR_TILE][SCHUR_CHUNK];
#if !SPARTA_REAL
    threadgroup float wi[SCHUR_TILE][SCHUR_CHUNK], wn[SCHUR_TILE][SCHUR_CHUNK],
        li[SCHUR_TILE][SCHUR_CHUNK];
#endif
    // One 8 x 8 result block per simdgroup at a time, real then imaginary.
    threadgroup float out[8][2][64];
    uint rowbase = base + job.row * SCHUR_TILE, colbase = base + job.col * SCHUR_TILE;
    uint linear = lane.y * 16 + lane.x;
    bool diag = job.row == job.col;
    // Group-uniform per simdgroup: a block row past the front's last row, or
    // a block above the diagonal of a diagonal tile, has nothing to compute.
    bool rowlive = rowbase + 8 * sg < m;
    simdgroup_float8x8 cr[8], ci[8];
#pragma unroll
    for (uint b = 0; b < 8; b++) {
        cr[b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
#if !SPARTA_REAL
        ci[b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
#endif
    }
    for (uint c0 = st.panel; c0 < st.next;) {
        uint width = min((uint)SCHUR_CHUNK, st.next - c0);
        if (c0 + width < st.next && two[first + c0 + width - 1])
            width--;
        // Padded with zeros to whole 8-column steps.
        uint padded = (width + 7u) & ~7u;
        for (uint idx = linear; idx < SCHUR_TILE * padded; idx += 256) {
            uint r = idx / padded, c = idx - r * padded, k = c0 + c;
            uint ri = rowbase + r, rj = colbase + r;
            T wv = c < width && ri < m ? fold_diagonal(a + ri * m, first, k, two, diagonal, subdiag)
                                       : T(0);
            T lv = c < width && rj < m ? a[rj * m + k] : T(0);
#if SPARTA_REAL
            wr[r][c] = wv;
            lr[r][c] = lv;
#else
            wr[r][c] = wv.x;
            wi[r][c] = wv.y;
            wn[r][c] = -wv.y;
            lr[r][c] = lv.x;
            li[r][c] = lv.y;
#endif
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (rowlive) {
            for (uint k8 = 0; k8 < padded; k8 += 8) {
                simdgroup_float8x8 ar;
                simdgroup_load(ar, &wr[8 * sg][k8], SCHUR_CHUNK);
#if !SPARTA_REAL
                simdgroup_float8x8 ai, an;
                simdgroup_load(ai, &wi[8 * sg][k8], SCHUR_CHUNK);
                simdgroup_load(an, &wn[8 * sg][k8], SCHUR_CHUNK);
#endif
#pragma unroll
                for (uint b = 0; b < 8; b++) {
                    if ((diag && b > sg) || colbase + 8 * b >= m)
                        continue;
                    simdgroup_float8x8 br;
                    simdgroup_load(br, &lr[8 * b][k8], SCHUR_CHUNK, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(cr[b], ar, br, cr[b]);
#if !SPARTA_REAL
                    simdgroup_float8x8 bi;
                    simdgroup_load(bi, &li[8 * b][k8], SCHUR_CHUNK, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(cr[b], an, bi, cr[b]);
                    simdgroup_multiply_accumulate(ci[b], ar, bi, ci[b]);
                    simdgroup_multiply_accumulate(ci[b], ai, br, ci[b]);
#endif
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        c0 += width;
    }
    if (!rowlive)
        return;
#pragma unroll
    for (uint b = 0; b < 8; b++) {
        if ((diag && b > sg) || colbase + 8 * b >= m)
            continue;
        simdgroup_store(cr[b], &out[sg][0][0], 8);
#if !SPARTA_REAL
        simdgroup_store(ci[b], &out[sg][1][0], 8);
#endif
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = sl; e < 64; e += 32) {
            uint i = rowbase + 8 * sg + e / 8, j = colbase + 8 * b + e % 8;
            if (i >= m || j > i)
                continue;
#if SPARTA_REAL
            T v = a[i * m + j] - out[sg][0][e];
#else
            T v = a[i * m + j] - float2(out[sg][0][e], out[sg][1][e]);
#endif
            if (!finite(v))
                atomic_store_explicit(status + s, 0xfffffffeu, memory_order_relaxed);
            a[i * m + j] = v;
            if (!SPARTA_LOWER)
                a[j * m + i] = v;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}
#else
kernel void update_schur(
    device const Node* nodes [[buffer(0)]], device float2* fronts [[buffer(7)]],
    device const float2* diagonal [[buffer(8)]], device const float2* subdiag [[buffer(9)]],
    device const uint* two [[buffer(11)]], device atomic_uint* status [[buffer(12)]],
    constant uint2& range [[buffer(14)]], device const FrontState* state [[buffer(15)]],
    device const UpdateTile* jobs [[buffer(18)]], uint2 group [[threadgroup_position_in_grid]],
    uint2 lane [[thread_position_in_threadgroup]]) {
    UpdateTile job = jobs[range.x + group.x];
    uint s = job.node;
    Node node = nodes[s];
    FrontState st = state[s];
    if (st.next <= st.panel || atomic_load_explicit(status + s, memory_order_relaxed) != 0)
        return;
    uint m = node.nr, base = node.nc, first = node.first;
    device T* a = (device T*)(fronts + node.offset);
    // Each thread owns a SCHUR_BLOCK x SCHUR_BLOCK register block, so one pass
    // over the staged panel feeds BLOCK^2 multiply-adds from 2*BLOCK
    // threadgroup reads. That ratio is what decides this kernel: at 2x2 it is
    // one complex multiply-add per staged operand read, and the inner loop
    // spends its time on threadgroup traffic rather than arithmetic.
    threadgroup T w[SCHUR_TILE][SCHUR_CHUNK], lj[SCHUR_TILE][SCHUR_CHUNK];
    uint rowbase = base + job.row * SCHUR_TILE, colbase = base + job.col * SCHUR_TILE;
    uint linear = lane.y * 16 + lane.x;
    // The tile is a 4 x 4 grid of SUB x SUB blocks, sixteen threads each, so
    // one simdgroup owns two blocks. The lower triangle's ten come first: on a
    // diagonal tile the last three simdgroups own only blocks above the
    // diagonal and skip the product, leaving the core's ALUs to the rest. A
    // block wholly past the front's last row or column skips it the same way.
    // The skip is uniform over a block, so it costs no divergence.
    constexpr uint SUB = SCHUR_TILE / 4;
    constexpr uchar order[16] = {0x00, 0x10, 0x11, 0x20, 0x21, 0x22, 0x30, 0x31,
                                 0x32, 0x33, 0x01, 0x02, 0x03, 0x12, 0x13, 0x23};
    uint block = order[linear / 16], within = linear % 16;
    uint myrow = SUB * (block >> 4) + SCHUR_BLOCK * (within / 4);
    uint mycol = SUB * (block & 15) + SCHUR_BLOCK * (within % 4);
    bool idle = (job.row == job.col && linear / 16 >= 10) ||
                rowbase + SUB * (block >> 4) >= m || colbase + SUB * (block & 15) >= m;
    T acc[SCHUR_BLOCK][SCHUR_BLOCK];
    for (uint di = 0; di < SCHUR_BLOCK; di++)
        for (uint dj = 0; dj < SCHUR_BLOCK; dj++)
            acc[di][dj] = T(0);
    for (uint c0 = st.panel; c0 < st.next;) {
        uint width = min((uint)SCHUR_CHUNK, st.next - c0);
        // Never split a 2x2 pivot across chunks; its two columns must be
        // folded with the block diagonal together.
        if (c0 + width < st.next && two[first + c0 + width - 1])
            width--;
        for (uint idx = linear; idx < SCHUR_TILE * width; idx += 256) {
            uint r = idx / width, c = idx - r * width, k = c0 + c;
            uint ri = rowbase + r, rj = colbase + r;
            w[r][c] =
                ri < m ? fold_diagonal(a + ri * m, first, k, two, diagonal, subdiag) : T(0);
            lj[r][c] = rj < m ? a[rj * m + k] : T(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = 0; c < (idle ? 0u : width); c++) {
            T wv[SCHUR_BLOCK], lv[SCHUR_BLOCK];
            for (uint d = 0; d < SCHUR_BLOCK; d++) {
                wv[d] = w[myrow + d][c];
                lv[d] = lj[mycol + d][c];
            }
            for (uint di = 0; di < SCHUR_BLOCK; di++)
                for (uint dj = 0; dj < SCHUR_BLOCK; dj++)
                    acc[di][dj] -= mul(wv[di], lv[dj]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        c0 += width;
    }
    for (uint di = 0; di < SCHUR_BLOCK; di++)
        for (uint dj = 0; dj < SCHUR_BLOCK; dj++) {
            uint i = rowbase + myrow + di, j = colbase + mycol + dj;
            if (idle || i >= m || j > i)
                continue;
            T v = a[i * m + j] + acc[di][dj];
            if (!finite(v))
                atomic_store_explicit(status + s, 0xfffffffeu, memory_order_relaxed);
            a[i * m + j] = v;
            a[j * m + i] = v;
        }
}
#endif
