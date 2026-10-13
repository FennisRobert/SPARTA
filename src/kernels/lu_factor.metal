// A positive threshold accepts the diagonal when |a_kk| >= threshold*column_max.
// Zero preserves strict partial pivoting. This is an experiment, not a stability
// guarantee: lowering the threshold permits larger elimination multipliers.
#ifndef SPARTA_LU_DIAG_THRESHOLD
#define SPARTA_LU_DIAG_THRESHOLD 0
#endif
static uint lu_prefer_diagonal(float2 diagonal, float largest, uint k, uint best) {
    float d = mag(diagonal);
    return SPARTA_LU_DIAG_THRESHOLD > 0 && isfinite(d) && d > 0 &&
                   d >= float(SPARTA_LU_DIAG_THRESHOLD) * largest
               ? k
               : best;
}

// Unsymmetric LU with partial pivoting: the pivoting itself.
//
// Three routes mirroring the LDL^T ones, sharing one pivot rule so the pivot
// sequence never depends on which route a front took:
//
//   factor_nodes_lu    one thread owns a whole short front, start to finish
//   factor_narrow_lu   every pivot of a thin front inside one threadgroup
//   factor_panel_lu    one PANEL-wide panel of a wide front, LAPACK getrf shape
//
// Two things separate this from the symmetric path.
//
// **Pivots are row interchanges.** The pivot for column k is the largest entry
// of that column, and the whole row is swapped across every column of the
// front, so the permutation is one-sided: P A = L U. The solve has to read its
// right-hand side through that permutation and write its answer without it --
// see `global_row` and `front_row` in common.
//
// **U is not free.** LDL^T reads the whole upper triangle off L, because the
// factorization is its own transpose. Here U12 has to be computed, and the only
// place it can be computed is where L11 exists: the panel routine solves
// L11 U12 = A12 for the panel's own rows across every trailing column before it
// returns. That is the `getrf` TRSM, and it is why the LU front is not
// uniformly stale the way the LDL^T front is -- rows the panel has finished are
// *final*, not stale, and only rows at or past the panel end carry pending
// updates. Nothing reads the finished rows again, so the invariant holds.
//
// Pivots are confined to the fully summed rows, which is what makes them
// eliminable at all. A column with no usable entry there is not delayed to the
// parent as MUMPS would delay it; it is floored by the same static
// perturbation the LDL^T path uses, and counted.

// Cooperative row interchange across the whole front. Unlike the symmetric
// `cooperative_swap` this touches rows only -- columns keep their identity, and
// so do the variables they eliminate.
static void swap_front_rows(device float2* a, uint m, uint p, uint q, device uint* perm, uint lane,
                            uint threads) {
    if (p != q) {
        for (uint j = lane; j < m; j += threads) {
            float2 t = a[p * m + j];
            a[p * m + j] = a[q * m + j];
            a[q * m + j] = t;
        }
        threadgroup_barrier(mem_flags::mem_device);
        if (lane == 0) {
            uint t = perm[p];
            perm[p] = perm[q];
            perm[q] = t;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
}

// Correctness prototype and the short-front path: one GPU thread owns a
// complete front. Independent fronts share a dispatch; no spin waits, no
// atomics. Right-looking, so the trailing block is updated in full after each
// pivot rather than deferred.
kernel void factor_nodes_lu(
    device const Node* nodes [[buffer(0)]], device const uint* rows [[buffer(1)]],
    device const Child* children [[buffer(2)]], device const uint* maps [[buffer(3)]],
    device const Entry* entries [[buffer(4)]], device const uint* ids [[buffer(5)]],
    device const float2* values [[buffer(6)]], device float2* fronts [[buffer(7)]],
    device uint* perms [[buffer(10)]], device uint* status [[buffer(12)]],
    constant uint2& range [[buffer(14)]], device FrontState* state [[buffer(15)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= range.y)
        return;
    uint s = ids[range.x + tid];
    Node node = nodes[s];
    uint m = node.nr, n = node.nc;
    device float2* a = fronts + node.offset;
    device uint* perm = perms + node.first;
    status[s] = 0;
    for (uint i = 0; i < m * m; i++)
        a[i] = float2(0);
    for (uint i = 0; i < n; i++)
        perm[i] = i;
    // Entry positions are already row*nr + column, and nothing is mirrored.
    for (uint e = 0; e < node.nentries; e++) {
        Entry entry = entries[node.entries + e];
        a[entry.position] = values[entry.source];
    }
    for (uint c = 0; c < node.count; c++) {
        Child child = children[node.children + c];
        Node cn = nodes[child.node];
        if (status[child.node] != 0) {
            status[s] = 0xffffffffu;
            return;
        }
        uint tail = cn.nr - cn.nc;
        // `soffset`/`sld` locate the block whether or not it was compacted.
        for (uint i = 0; i < tail; i++)
            for (uint j = 0; j < tail; j++)
                a[maps[child.map + i] * m + maps[child.map + j]] +=
                    fronts[cn.soffset + i * cn.sld + j];
    }
    // The reference scale for a perturbed pivot is computed only if a pivot
    // actually dies, so a healthy front never pays for the extra sweep.
    float pfloor = -1.0f;
    uint bumped = 0;
    state[s] = FrontState{0, 0, 0, 0, 0};
    for (uint k = 0; k < n; k++) {
        // Partial pivoting, restricted to the fully summed rows: a row past nc
        // still has contributions to come from this front's ancestors, so it
        // cannot be eliminated here however large its entry is.
        float best = 0;
        uint pk = k;
        for (uint i = k; i < n; i++) {
            float v = mag(a[i * m + k]);
            if (v > best) {
                best = v;
                pk = i;
            }
        }
        if (!isfinite(best)) {
            status[s] = PIVOT_NONFINITE | (k + 1);
            return;
        }
        pk = lu_prefer_diagonal(a[k * m + k], best, k, pk);
        if (pk != k) {
            for (uint j = 0; j < m; j++) {
                float2 t = a[k * m + j];
                a[k * m + j] = a[pk * m + j];
                a[pk * m + j] = t;
            }
            uint t = perm[k];
            perm[k] = perm[pk];
            perm[pk] = t;
        }
        float2 d = a[k * m + k];
        if (pivot_dead(d)) {
            if (pfloor < 0) {
                float big = 0;
                for (uint i = 0; i < m * m; i++)
                    big = max(big, mag(a[i]));
                pfloor = PIVOT_FLOOR * big;
            }
            if (pfloor <= 0) {
                status[s] = PIVOT_ZERO_COLUMN | (k + 1);
                return;
            }
            d = floor_pivot(d, pfloor);
            a[k * m + k] = d;
            bumped++;
        }
        for (uint i = k + 1; i < m; i++)
            a[i * m + k] = cdiv(a[i * m + k], d);
        for (uint i = k + 1; i < m; i++) {
            float2 li = a[i * m + k];
            for (uint j = k + 1; j < m; j++)
                a[i * m + j] -= mul(li, a[k * m + j]);
        }
    }
    for (uint i = 0; i < m * m; i++)
        if (!all(isfinite(a[i]))) {
            status[s] = 0xfffffffeu;
            return;
        }
    state[s].perturbed = bumped;
}

// A front whose fully summed block is narrow takes every one of its pivots in
// one threadgroup, as in the symmetric case. The trailing update is confined to
// the *cross*: the remaining fully summed columns over every row, and the
// remaining fully summed rows over every column. The square block past nc is
// the Schur complement and is left to the blocked update.
kernel void factor_narrow_lu(
    device const Node* nodes [[buffer(0)]], device const uint* ids [[buffer(5)]],
    device float2* fronts [[buffer(7)]], device uint* perms [[buffer(10)]],
    device uint* status [[buffer(12)]], constant uint2& range [[buffer(14)]],
    device FrontState* state [[buffer(15)]], uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint sl [[thread_index_in_simdgroup]], uint simdw [[threads_per_simdgroup]]) {
    uint s = ids[range.x + group];
    Node node = nodes[s];
    uint m = node.nr, n = node.nc;
    uint simds = (PIVOT_THREADS + simdw - 1) / simdw;
    if (status[s] != 0)
        return; // Group-uniform: assembly already failed.
    device float2* a = fronts + node.offset;
    device uint* perm = perms + node.first;
    threadgroup float maxima[MAX_SIMDS];
    threadgroup uint indices[MAX_SIMDS], bad[MAX_SIMDS];
    for (uint k = 0; k < n; k++) {
        float cm = 0;
        uint imax = k;
        for (uint i = k + lane; i < n; i += PIVOT_THREADS) {
            float v = mag(a[i * m + k]);
            if (v > cm || (v == cm && i < imax)) {
                cm = v;
                imax = i;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        reduce_argmax(cm, imax, maxima, indices, sg, sl, simds, &cm, &imax);
        if (lane == 0 && !isfinite(cm))
            status[s] = PIVOT_NONFINITE | (k + 1);
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        if (status[s] != 0)
            return; // Group-uniform: read after a barrier.
        imax = lu_prefer_diagonal(a[k * m + k], cm, k, imax);
        swap_front_rows(a, m, k, imax, perm, lane, PIVOT_THREADS);
        float2 d = a[k * m + k];
        if (pivot_dead(d)) {
            // An all-zero fully summed column would have been fatal; floor it
            // to a fraction of the front's own scale instead and report it.
            float big = front_scale(a, m, n, lane, PIVOT_THREADS, maxima, sg, sl, simds);
            if (big <= 0) {
                if (lane == 0)
                    status[s] = PIVOT_ZERO_COLUMN | (k + 1);
                return;
            }
            d = floor_pivot(d, PIVOT_FLOOR * big);
            if (lane == 0) {
                a[k * m + k] = d;
                state[s].perturbed++;
            }
            threadgroup_barrier(mem_flags::mem_device);
        }
        for (uint i = k + 1 + lane; i < m; i += PIVOT_THREADS)
            a[i * m + k] = cdiv(a[i * m + k], d);
        threadgroup_barrier(mem_flags::mem_device);
        uint invalid = 0;
        for (uint i = k + 1 + lane; i < m; i += PIVOT_THREADS)
            invalid |= uint(!all(isfinite(a[i * m + k])));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (reduce_or(invalid, bad, sg, sl, simds)) {
            if (lane == 0)
                status[s] = 0xfffffffeu;
            return;
        }
        // Remaining fully summed columns, over every row below the pivot.
        for (uint j = k + 1; j < n; j++) {
            float2 u = a[k * m + j];
            for (uint i = k + 1 + lane; i < m; i += PIVOT_THREADS)
                a[i * m + j] -= mul(a[i * m + k], u);
        }
        // Remaining fully summed rows, over the trailing columns. Disjoint from
        // the loop above, and together they are the whole cross.
        for (uint i = k + 1; i < n; i++) {
            float2 li = a[i * m + k];
            for (uint j = n + lane; j < m; j += PIVOT_THREADS)
                a[i * m + j] -= mul(li, a[k * m + j]);
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
    if (lane == 0) {
        state[s].next = n;
        state[s].start = 0;
        state[s].width = 0;
    }
}

// ---------------------------------------------------------------------------
// Panelled wide fronts (LAPACK getrf shape).
//
// Invariant: rows before the panel start `p` are *final* -- both their L
// multipliers and their U entries, all the way out to the last column -- and
// rows at or past `p` are uniformly current with respect to every pivot before
// `p`. The panel's own columns are formed in a separate workspace `w`,
// column-major with leading dimension nr, exactly as in the symmetric path.
//
// The reason for the workspace is different here, and weaker. LDL^T needs it
// because a Bunch-Kaufman row search can name a column outside the panel. LU's
// pivot search reads one column, the one being formed, so the workspace buys
// only the deferred rank-k update -- but that is the whole point of a panel, so
// the shape is the same.

// Brings front column `col` current with respect to the panel's first `count`
// pivots and writes it to panel column `dest`. Rows at or past `rows` are left
// to `factor_panel_tail_lu`: the pivot search reads only the fully summed rows,
// so nothing in this kernel reads the ones below them back.
//
// Rows p..p+count-1 are the panel's own pivot rows, and their entries in this
// column are U values. Getting them needs a forward substitution against the
// unit lower-triangular L11 the panel has built so far: this is the one step
// LDL^T gets for free, where `form_column` reads the same numbers straight out
// of the panel because U is D L^T. It is `count` dependent steps over at most
// PANEL values. Lane 0 used to run it alone, one device load after another,
// while the rest waited; spreading each step over the lanes cut factor_panel_lu
// from 0.063 s to 0.050 s on em_m3_asym1 and em_m3_asym2, measured on an M4 Pro.
//
// Each lane also scans the rows it forms for the largest magnitude, ties to the
// lowest row, into `cm` and `imax`: the partial-pivoting search, done while the
// value is still in a register instead of re-reading the column afterwards.
// The lanes' results reduce to the same row whichever lane scanned which.
static void form_column_lu(device const float2* a, device float2* w, uint m, uint count, uint dest,
                           uint col, uint p, threadgroup float2* coef, uint lane, uint threads,
                           uint rows, thread float& cm, thread uint& imax) {
    for (uint i = lane; i < count; i += threads)
        coef[i] = a[(p + i) * m + col];
    // One column per pass: entry t2 takes column t's term once coef[t] is
    // final. Each entry subtracts its terms in the same order as a serial
    // sweep, so the result is bit-identical.
    for (uint t = 0; t + 1 < count; t++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint t2 = t + 1 + lane; t2 < count; t2 += threads)
            coef[t2] -= mul(w[t * m + p + t2], coef[t]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lane; i < count; i += threads)
        w[dest * m + p + i] = coef[i];
    for (uint i = p + count + lane; i < rows; i += threads) {
        float2 v = a[i * m + col];
        for (uint t = 0; t < count; t++)
            v -= mul(w[t * m + i], coef[t]);
        w[dest * m + i] = v;
        float mv = mag(v);
        if (mv > cm || (mv == cm && i < imax)) {
            cm = mv;
            imax = i;
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
}
// Swaps rows p and q across the panel's formed columns, matching the row swap
// applied to the front. Unlike the symmetric path there is nothing to re-form
// afterwards: a row interchange permutes a formed column, it does not
// invalidate it.
static void swap_panel_rows_lu(device float2* w, uint m, uint columns, uint p, uint q, uint lane,
                               uint threads) {
    if (p != q)
        for (uint t = lane; t < columns; t += threads) {
            float2 v = w[t * m + p];
            w[t * m + p] = w[t * m + q];
            w[t * m + q] = v;
        }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
}
kernel void factor_panel_lu(
    device const Node* nodes [[buffer(0)]], device const uint* ids [[buffer(5)]],
    device float2* fronts [[buffer(7)]], device uint* perms [[buffer(10)]],
    device uint* status [[buffer(12)]], constant uint2& range [[buffer(14)]],
    device FrontState* state [[buffer(15)]], device float2* panels [[buffer(19)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint sl [[thread_index_in_simdgroup]],
    uint simdw [[threads_per_simdgroup]], uint width [[threads_per_threadgroup]]) {
    uint s = ids[range.x + group];
    Node node = nodes[s];
    uint simds = (width + simdw - 1) / simdw;
    uint m = node.nr, n = node.nc, p = state[s].next;
    if (lane == 0) {
        state[s].start = p;
        state[s].width = 0;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (status[s] != 0 || p >= n)
        return; // Group-uniform.
    device float2* a = fronts + node.offset;
    device float2* w = panels + node.woffset;
    device uint* perm = perms + node.first;
    threadgroup float maxima[MAX_SIMDS];
    threadgroup uint indices[MAX_SIMDS], bad[MAX_SIMDS];
    threadgroup float2 coef[PANEL_COLS];
    uint t = 0, k = p;
    while (t < PANEL && k < n) {
        // Partial pivoting over the fully summed rows of the formed column,
        // searched while forming it.
        float cm = 0;
        uint imax = k;
        form_column_lu(a, w, m, t, t, k, p, coef, lane, width, n, cm, imax);
        reduce_argmax(cm, imax, maxima, indices, sg, sl, simds, &cm, &imax);
        // Every lane holds the same cm, so the failure test needs no broadcast
        // and no re-read of the status word.
        if (!isfinite(cm)) {
            if (lane == 0)
                status[s] = PIVOT_NONFINITE | (k + 1);
            return;
        }
        imax = lu_prefer_diagonal(w[t * m + k], cm, k, imax);
        swap_front_rows(a, m, k, imax, perm, lane, width);
        swap_panel_rows_lu(w, m, t + 1, k, imax, lane, width);
        float2 d = w[t * m + k];
        if (pivot_dead(d)) {
            float big = front_scale(a, m, n, lane, width, maxima, sg, sl, simds);
            if (big <= 0) {
                if (lane == 0)
                    status[s] = PIVOT_ZERO_COLUMN | (k + 1);
                return;
            }
            d = floor_pivot(d, PIVOT_FLOOR * big);
            if (lane == 0) {
                w[t * m + k] = d;
                state[s].perturbed++;
            }
            threadgroup_barrier(mem_flags::mem_device);
        }
        // Each lane checks the multipliers it writes, so no pass re-reads them.
        uint invalid = 0;
        for (uint i = k + 1 + lane; i < n; i += width) {
            float2 v = cdiv(w[t * m + i], d);
            invalid |= uint(!all(isfinite(v)));
            w[t * m + i] = v;
        }
        if (reduce_or(invalid, bad, sg, sl, simds)) {
            if (lane == 0)
                status[s] = 0xfffffffeu;
            return;
        }
        t++;
        k++;
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
    // Publish the panel's columns: rows p up to the end of the fully summed
    // block, U above the diagonal and L below it. Rows before p are finished U
    // rows from earlier panels and must not be touched; the rows past nc are
    // written straight into the front by factor_panel_tail_lu, which never puts
    // them in the panel workspace at all.
    uint rows_here = n - p;
    for (uint idx = lane; idx < t * rows_here; idx += width) {
        uint t2 = idx / rows_here, i = p + idx - t2 * rows_here;
        a[i * m + p + t2] = w[t2 * m + i];
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (lane == 0) {
        state[s].next = p + t;
        state[s].start = p;
        state[s].width = t;
    }
}

// Applies a finished LU panel to the rows below the fully summed block.
//
// The same split as the symmetric path, and a simpler one. LU's pivot search
// reads only the fully summed rows, so the rows below them carry no dependency
// on it; and because LU interchanges rows rather than rows and columns, and
// every pivot it chooses lies inside the fully summed block, those rows are
// never permuted at all. There are no 2x2 pivots either, so each is a plain
// unit-lower-triangular solve of PANEL steps.
//
// The coefficients are already in the panel workspace: form_column_lu writes
// the panel's own rows of column t to w[t*m + p + t2], which is the U entry
// column t consumes from column t2, and the diagonal of that is the pivot
// itself. So nothing has to be recomputed here, only read.
//
// A block of rows is staged in threadgroup memory and solved in place, one
// thread per row, so the accumulation reads no device memory and the front is
// read along its rows, which is the only way to read a row-major front
// coalesced. The arithmetic is form_column_lu's in its order, so the factors
// are bit-identical to taking the whole panel in one threadgroup.
kernel void factor_panel_tail_lu(
    device const Node* nodes [[buffer(0)]], device const uint* ids [[buffer(5)]],
    device float2* fronts [[buffer(7)]], device uint* status [[buffer(12)]],
    constant uint4& range [[buffer(14)]], device const FrontState* state [[buffer(15)]],
    device const float2* panels [[buffer(19)]], uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint sl [[thread_index_in_simdgroup]], uint simdw [[threads_per_simdgroup]],
    uint width [[threads_per_threadgroup]]) {
    threadgroup uint bad[MAX_SIMDS];
    threadgroup float2 coefficient[PANEL_COLS][PANEL_COLS];
    threadgroup float2 block[TAIL_ROWS][PANEL_COLS];
    threadgroup float2 l11[PANEL * PANEL];
    uint splits = range.z;
    uint s = ids[range.x + group / splits];
    uint slice = group - (group / splits) * splits;
    Node node = nodes[s];
    FrontState st = state[s];
    uint m = node.nr, n = node.nc;
    uint p = st.start, taken = st.width;
    // Group-uniform: a front whose panel took nothing, or that has already
    // failed, has nothing below to solve.
    if (taken == 0 || status[s] != 0)
        return;
    device float2* a = fronts + node.offset;
    device const float2* w = panels + node.woffset;
    for (uint idx = lane; idx < taken * taken; idx += width) {
        uint t = idx / taken, t2 = idx - t * taken;
        coefficient[t][t2] = t2 <= t ? w[t * m + p + t2] : float2(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint invalid = 0;
    for (uint base = n + slice * TAIL_ROWS; base < m && n < m; base += splits * TAIL_ROWS) {
        uint rows = min((uint)TAIL_ROWS, m - base);
        // Staged along the front's rows: consecutive lanes take consecutive
        // columns of one row, which is contiguous device memory.
        for (uint idx = lane; idx < rows * taken; idx += width) {
            uint r = idx / taken, t = idx - r * taken;
            block[r][t] = a[(base + r) * m + p + t];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane < rows) {
            threadgroup float2* row = &block[lane][0];
            for (uint t = 0; t < taken; t++) {
                float2 v = row[t];
                for (uint t2 = 0; t2 < t; t2++)
                    v -= mul(row[t2], coefficient[t][t2]);
                row[t] = cdiv(v, coefficient[t][t]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = lane; idx < rows * taken; idx += width) {
            uint r = idx / taken, t = idx - r * taken;
            float2 v = block[r][t];
            invalid |= uint(!all(isfinite(v)));
            a[(base + r) * m + p + t] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // The TRSM: solve L11 U12 = A12 for the panel's rows across every trailing
    // column, which is what makes those rows final and what the deferred rank-t
    // update reads as its second operand. It is parallel over columns and
    // touches rows p..p+taken only, so it shares this dispatch with the row
    // solve above -- they are disjoint -- rather than staying behind the serial
    // pivot loop where it was half the panel's remaining arithmetic.
    //
    // L11 is at most PANEL x PANEL and shared by every column, so it is staged
    // once. Loop bounds are the compile-time PANEL with a guard rather than the
    // dynamic `taken`, so the substitution unrolls and `u` stays in registers.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = lane; idx < taken * taken; idx += width) {
        uint r2 = idx / taken, r = idx - r2 * taken;
        l11[r2 * PANEL + r] = r2 > r ? a[(p + r2) * m + p + r] : float2(0);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = p + taken + slice * width + lane; j < m; j += splits * width) {
        float2 u[PANEL];
        for (uint r = 0; r < PANEL; r++)
            u[r] = r < taken ? a[(p + r) * m + j] : float2(0);
        for (uint r = 0; r < PANEL; r++) {
            float2 v = u[r];
            for (uint r2 = 0; r2 < PANEL; r2++)
                if (r2 > r && r2 < taken)
                    u[r2] -= mul(l11[r2 * PANEL + r], v);
        }
        for (uint r = 0; r < PANEL; r++)
            if (r < taken)
                a[(p + r) * m + j] = u[r];
    }
    // Every lane runs the reduction: it barriers internally, so it must not sit
    // behind a lane-uniform condition.
    if (reduce_or(invalid, bad, sg, sl, (width + simdw - 1) / simdw) && lane == 0)
        status[s] = 0xfffffffeu;
}
