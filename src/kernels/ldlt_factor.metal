// Complex-symmetric LDL^T: the pivoting itself.
//
// Three routes, all sharing one Bunch-Kaufman pivot rule so that the pivot
// sequence never depends on which route a front took:
//
//   factor_nodes    one thread owns a whole short front, start to finish
//   factor_narrow   every pivot of a thin front inside one threadgroup
//   factor_panel    one PANEL-wide panel of a wide front, LAPACK LASYF shape
//
// Pivots are confined to the fully summed columns. A pivot that has died is
// floored rather than failed on -- see the perturbation policy in common.
void swap_pivot(device float2* a, uint m, uint p, uint q, device uint* perm) {
    if (p == q)
        return;
    for (uint j = 0; j < m; j++) {
        float2 t = a[p * m + j];
        a[p * m + j] = a[q * m + j];
        a[q * m + j] = t;
    }
    for (uint i = 0; i < m; i++) {
        float2 t = a[i * m + p];
        a[i * m + p] = a[i * m + q];
        a[i * m + q] = t;
    }
    uint t = perm[p];
    perm[p] = perm[q];
    perm[q] = t;
}
// Correctness prototype: one GPU thread owns a complete front. Independent
// fronts share a dispatch; there are no inter-thread spin waits or atomics.
kernel void factor_nodes(
    device const Node* nodes [[buffer(0)]], device const uint* rows [[buffer(1)]],
    device const Child* children [[buffer(2)]], device const uint* maps [[buffer(3)]],
    device const Entry* entries [[buffer(4)]], device const uint* ids [[buffer(5)]],
    device const float2* values [[buffer(6)]], device float2* fronts [[buffer(7)]],
    device float2* diagonal [[buffer(8)]], device float2* subdiag [[buffer(9)]],
    device uint* perms [[buffer(10)]], device uint* two [[buffer(11)]],
    device uint* status [[buffer(12)]], constant uint2& range [[buffer(14)]],
    device FrontState* state [[buffer(15)]], uint tid [[thread_position_in_grid]]) {
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
    for (uint i = 0; i < n; i++) {
        perm[i] = i;
        two[node.first + i] = 0;
        subdiag[node.first + i] = float2(0);
    }
    for (uint e = 0; e < node.nentries; e++) {
        Entry entry = entries[node.entries + e];
        uint i = entry.position / m, j = entry.position % m;
        a[i * m + j] = values[entry.source];
        a[j * m + i] = values[entry.source];
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
    uint k = 0;
    while (k < n) {
        float md = 0, mo = 0;
        uint pd = k, po = k, qo = k;
        for (uint i = k; i < n; i++) {
            float v = mag(a[i * m + i]);
            if (v > md) {
                md = v;
                pd = i;
            }
            for (uint j = k; j < i; j++) {
                v = mag(a[i * m + j]);
                if (v > mo) {
                    mo = v;
                    po = j;
                    qo = i;
                }
            }
        }
        if (!isfinite(md) || !isfinite(mo)) {
            status[s] = PIVOT_NONFINITE | (k + 1);
            return;
        }
        if (max(md, mo) == 0) {
            // An all-zero remaining block falls through to a floored 1x1 pivot,
            // unless the whole front is zero and there is no scale to floor
            // against, which stays fatal as before.
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
        }
        if (md >= 0.6403882f * mo || k + 1 == n) {
            swap_pivot(a, m, k, pd, perm);
            float2 raw = a[k * m + k], d = raw;
            if (pivot_dead(raw)) {
                if (pfloor < 0) {
                    float big = 0;
                    for (uint i = 0; i < m * m; i++)
                        big = max(big, mag(a[i]));
                    pfloor = PIVOT_FLOOR * big;
                }
                if (pfloor > 0) {
                    d = floor_pivot(raw, pfloor);
                    a[k * m + k] = d;
                    bumped++;
                }
            }
            diagonal[node.first + k] = d;
            for (uint i = k + 1; i < m; i++)
                a[i * m + k] = cdiv(a[i * m + k], d);
            for (uint i = k + 1; i < m; i++)
                for (uint j = k + 1; j <= i; j++) {
                    float2 v = a[i * m + j] - mul(mul(a[i * m + k], d), a[j * m + k]);
                    a[i * m + j] = v;
                    a[j * m + i] = v;
                }
            k++;
        } else {
            // po < qo, so swapping po with k cannot displace qo.
            swap_pivot(a, m, k, po, perm);
            swap_pivot(a, m, k + 1, qo, perm);
            float2 da = a[k * m + k], db = a[(k + 1) * m + k], dc = a[(k + 1) * m + k + 1];
            float scale = max(mag(da), max(mag(db), mag(dc)));
            float2 aa = da / scale, bb = db / scale, cc = dc / scale,
                   det = mul(aa, cc) - mul(bb, bb);
            if (!all(isfinite(det))) {
                status[s] = PIVOT_NONFINITE | (k + 1);
                return;
            }
            if (mag(det) <= PIVOT_EPS) {
                // No usable 2x2 inverse; take a floored 1x1 on column k instead
                // and leave the partner to the next pivot.
                if (pfloor < 0) {
                    float big = 0;
                    for (uint i = 0; i < m * m; i++)
                        big = max(big, mag(a[i]));
                    pfloor = PIVOT_FLOOR * big;
                }
                if (pfloor <= 0) {
                    status[s] = PIVOT_TINY_2X2 | (k + 1);
                    return;
                }
                float2 d = pivot_dead(da) ? floor_pivot(da, pfloor) : da;
                a[k * m + k] = d;
                diagonal[node.first + k] = d;
                bumped++;
                for (uint i = k + 1; i < m; i++)
                    a[i * m + k] = cdiv(a[i * m + k], d);
                for (uint i = k + 1; i < m; i++)
                    for (uint j = k + 1; j <= i; j++) {
                        float2 v = a[i * m + j] - mul(mul(a[i * m + k], d), a[j * m + k]);
                        a[i * m + j] = v;
                        a[j * m + i] = v;
                    }
                k++;
                continue;
            }
            diagonal[node.first + k] = da;
            diagonal[node.first + k + 1] = dc;
            subdiag[node.first + k] = db;
            two[node.first + k] = 1;
            for (uint i = k + 2; i < m; i++) {
                float2 x = a[i * m + k] / scale, y = a[i * m + k + 1] / scale;
                a[i * m + k] = cdiv(mul(x, cc) - mul(y, bb), det);
                a[i * m + k + 1] = cdiv(mul(y, aa) - mul(x, bb), det);
            }
            for (uint i = k + 2; i < m; i++)
                for (uint j = k + 2; j <= i; j++) {
                    float2 li = a[i * m + k], ri = a[i * m + k + 1], lj = a[j * m + k],
                           rj = a[j * m + k + 1];
                    float2 v = a[i * m + j] - mul(mul(li, da) + mul(ri, db), lj) -
                               mul(mul(li, db) + mul(ri, dc), rj);
                    a[i * m + j] = v;
                    a[j * m + i] = v;
                }
            a[(k + 1) * m + k] = float2(0);
            k += 2;
        }
    }
    for (uint i = 0; i < m * m; i++)
        if (!all(isfinite(a[i]))) {
            status[s] = 0xfffffffeu;
            return;
        }
    state[s].perturbed = bumped;
}
// Applies a symmetric interchange of `p` and `q` to the front.
//
// `threads` is the caller's threadgroup width: the narrow path runs at the
// fixed PIVOT_THREADS, the wide path at whatever the host dispatched.
void cooperative_swap(device float2* a, uint m, uint n, uint p, uint q, device uint* perm,
                      uint lane, uint threads) {
    if (p != q) {
        // The row swap stops at the fully summed block. Past it lies U12, rows
        // inside the block over the trailing columns, and nothing on this path
        // ever reads U12: the pivot searches, form_column, front_scale and
        // store_factors all stay in columns below nc, and both update kernels
        // read their operands there too. It is written only because the front
        // is held symmetric. The column swap below is a different matter -- it
        // moves L21, which factor_panel_tail and both update kernels do read --
        // and stays over every row.
        for (uint j = lane; j < n; j += threads) {
            float2 t = a[p * m + j];
            a[p * m + j] = a[q * m + j];
            a[q * m + j] = t;
        }
        threadgroup_barrier(mem_flags::mem_device);
        for (uint i = lane; i < m; i += threads) {
            float2 t = a[i * m + p];
            a[i * m + p] = a[i * m + q];
            a[i * m + q] = t;
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
// Takes one Bunch-Kaufman pivot at column k and returns its width, or 0 if the
// front failed. Shared by the step-at-a-time path for wide fronts and the
// whole-front path for narrow ones, so both take the identical pivot sequence.
// Every early return is group-uniform: the conditions are read from device
// memory by all lanes after a barrier.
static uint take_pivot(device float2* a, uint m, uint n, uint k, uint first, device uint* perm,
                       device float2* diagonal, device float2* subdiag, device uint* two,
                       device uint* status, uint s, uint lane, threadgroup float* maxima,
                       threadgroup uint* indices, threadgroup uint* choice, threadgroup uint* bad,
                       uint sg, uint sl, uint simds, device FrontState* state) {
    // Bunch-Kaufman search scans a column and (only if necessary) one row.
    // All candidate pivots remain inside the fully summed node columns.
    float cm = 0;
    uint imax = k;
    for (uint i = k + 1 + lane; i < n; i += PIVOT_THREADS) {
        float v = mag(a[i * m + k]);
        if (v > cm || (v == cm && i < imax)) {
            cm = v;
            imax = i;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    reduce_argmax(cm, imax, maxima, indices, sg, sl, simds, &cm, &imax);
    float akk = mag(a[k * m + k]);
    // Every lane already holds the result; this only guards reuse of the scratch.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        choice[0] = 1;
        choice[1] = k;
        choice[2] = imax;
        if (!isfinite(cm) || !isfinite(akk))
            status[s] = PIVOT_NONFINITE | (k + 1);
        // An all-zero column is no longer fatal here: it reaches the 1x1 branch
        // below with a dead diagonal, which floors it and reports the
        // perturbation. Only a wholly zero front still fails, and it fails
        // there, where the scale that proves it is already computed.
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (status[s] != 0)
        return 0;
    if (akk < 0.6403882f * cm) {
        float rm = 0;
        for (uint j = k + lane; j < n; j += PIVOT_THREADS)
            if (j != imax)
                rm = max(rm, mag(a[imax * m + j]));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        rm = reduce_max(rm, maxima, sg, sl, simds);
        if (lane == 0) {
            float aii = mag(a[imax * m + imax]);
            if (!isfinite(rm) || !isfinite(aii))
                status[s] = PIVOT_NONFINITE | (k + 1);
            else if (akk >= 0.6403882f * cm * (cm / rm))
                choice[1] = k;
            else if (aii >= 0.6403882f * rm)
                choice[1] = imax;
            else {
                choice[0] = 2;
                choice[1] = k;
                choice[2] = imax;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (status[s] != 0)
        return 0;
    uint step = choice[0];
    cooperative_swap(a, m, n, k, choice[1], perm, lane, PIVOT_THREADS);
    if (step == 2)
        cooperative_swap(a, m, n, k + 1, choice[2], perm, lane, PIVOT_THREADS);
    if (step == 1) {
        float2 raw = a[k * m + k], d = raw;
        if (pivot_dead(raw)) {
            float big = front_scale(a, m, n, lane, PIVOT_THREADS, maxima, sg, sl, simds);
            if (big <= 0) {
                if (lane == 0)
                    status[s] = PIVOT_ZERO_COLUMN | (k + 1);
                return 0;
            }
            d = floor_pivot(raw, PIVOT_FLOOR * big);
            if (lane == 0) {
                a[k * m + k] = d;
                state[s].perturbed++;
            }
            threadgroup_barrier(mem_flags::mem_device);
        }
        if (lane == 0)
            diagonal[first + k] = d;
        for (uint i = k + 1 + lane; i < m; i += PIVOT_THREADS)
            a[i * m + k] = cdiv(a[i * m + k], d);
    } else {
        float2 da = a[k * m + k], db = a[(k + 1) * m + k], dc = a[(k + 1) * m + k + 1];
        float scale = max(mag(da), max(mag(db), mag(dc)));
        float2 aa = da / scale, bb = db / scale, cc = dc / scale, det = mul(aa, cc) - mul(bb, bb);
        if (!all(isfinite(det))) {
            if (lane == 0)
                status[s] = PIVOT_NONFINITE | (k + 1);
            return 0;
        }
        if (mag(det) <= PIVOT_EPS) {
            // The 2x2 block has no usable inverse left in single precision.
            // Demote to a floored 1x1 on column k and leave the partner column
            // to the next pivot. Group-uniform: every lane read the same block.
            step = 1;
            // Reached only from what was a hard failure, so flooring here can
            // only improve on aborting the front.
            float2 d = da;
            if (pivot_dead(da)) {
                float big = front_scale(a, m, n, lane, PIVOT_THREADS, maxima, sg, sl, simds);
                if (big <= 0) {
                    if (lane == 0)
                        status[s] = PIVOT_TINY_2X2 | (k + 1);
                    return 0;
                }
                d = floor_pivot(da, PIVOT_FLOOR * big);
            }
            if (lane == 0) {
                diagonal[first + k] = d;
                a[k * m + k] = d;
                state[s].perturbed++;
            }
            threadgroup_barrier(mem_flags::mem_device);
            for (uint i = k + 1 + lane; i < m; i += PIVOT_THREADS)
                a[i * m + k] = cdiv(a[i * m + k], d);
        } else {
            if (lane == 0) {
                diagonal[first + k] = da;
                diagonal[first + k + 1] = dc;
                subdiag[first + k] = db;
                two[first + k] = 1;
            }
            for (uint i = k + 2 + lane; i < m; i += PIVOT_THREADS) {
                float2 x = a[i * m + k] / scale, y = a[i * m + k + 1] / scale;
                a[i * m + k] = cdiv(mul(x, cc) - mul(y, bb), det);
                a[i * m + k + 1] = cdiv(mul(y, aa) - mul(x, bb), det);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    // Multipliers may overflow even when there is no remaining trailing entry.
    uint invalid = 0;
    for (uint i = k + step + lane; i < m; i += PIVOT_THREADS) {
        invalid |= uint(!all(isfinite(a[i * m + k])));
        if (step == 2)
            invalid |= uint(!all(isfinite(a[i * m + k + 1])));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (reduce_or(invalid, bad, sg, sl, simds)) {
        if (lane == 0)
            status[s] = 0xfffffffeu;
        return 0;
    }
    if (lane == 0 && step == 2)
        a[(k + 1) * m + k] = float2(0);
    threadgroup_barrier(mem_flags::mem_device);
    return step;
}
// A front whose fully summed block is narrow takes every one of its pivots in
// one threadgroup. The trailing update it applies is confined to the remaining
// fully summed columns, which is a thin strip, so the whole front costs one
// dispatch instead of one per pivot. The Schur complement is still deferred to
// the blocked update. Wide fronts keep the step-at-a-time path, where each
// pivot's strip is large enough to be worth its own grid.
kernel void factor_narrow(
    device const Node* nodes [[buffer(0)]], device const uint* ids [[buffer(5)]],
    device float2* fronts [[buffer(7)]], device float2* diagonal [[buffer(8)]],
    device float2* subdiag [[buffer(9)]], device uint* perms [[buffer(10)]],
    device uint* two [[buffer(11)]], device uint* status [[buffer(12)]],
    constant uint2& range [[buffer(14)]], device FrontState* state [[buffer(15)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint sl [[thread_index_in_simdgroup]],
    uint simdw [[threads_per_simdgroup]]) {
    uint s = ids[range.x + group];
    Node node = nodes[s];
    uint m = node.nr, n = node.nc, first = node.first;
    uint simds = (PIVOT_THREADS + simdw - 1) / simdw;
    if (status[s] != 0)
        return; // Group-uniform: assembly already failed.
    device float2* a = fronts + node.offset;
    threadgroup float maxima[PIVOT_THREADS];
    threadgroup uint indices[PIVOT_THREADS], choice[3], bad[PIVOT_THREADS];
    uint k = 0;
    while (k < n) {
        uint step = take_pivot(a, m, n, k, first, perms + first, diagonal, subdiag, two, status, s,
                               lane, maxima, indices, choice, bad, sg, sl, simds, state);
        if (!step)
            return;
        // Columns past this pivot stay current for the searches that follow;
        // columns at or past nc are the Schur complement and are left alone.
        float2 da = diagonal[first + k], db = subdiag[first + k];
        float2 dc = step == 2 ? diagonal[first + k + 1] : float2(0);
        for (uint j = k + step; j < n; j++) {
            float2 lj = a[j * m + k], rj = step == 2 ? a[j * m + k + 1] : float2(0);
            for (uint i = j + lane; i < m; i += PIVOT_THREADS) {
                float2 li = a[i * m + k], v = a[i * m + j];
                if (step == 1)
                    v -= mul(mul(li, da), lj);
                else {
                    float2 ri = a[i * m + k + 1];
                    v -= mul(mul(li, da) + mul(ri, db), lj);
                    v -= mul(mul(li, db) + mul(ri, dc), rj);
                }
                a[i * m + j] = v;
                a[j * m + i] = v;
            }
        }
        threadgroup_barrier(mem_flags::mem_device);
        k += step;
    }
    if (lane == 0) {
        state[s].next = n;
        state[s].start = 0;
        state[s].width = 0;
    }
}
// ---------------------------------------------------------------------------
// Panelled wide fronts (LAPACK LASYF shape).
//
// Invariant: the front `a` is uniformly current with respect to every pivot
// before the panel start `p`, and is never partially updated. The panel's
// updated columns live in a separate workspace `w`, column-major with leading
// dimension nr, where column `t` holds the factor column for pivot `p+t`.
//
// That separation is what makes pivoting legal. A Bunch-Kaufman row search can
// name a column outside the panel, and if that column becomes the pivot it gets
// swapped to the front of the panel. Updating columns of `a` in place would
// leave that one column current while its neighbours were stale, and no choice
// of swap can repair it -- the "current" column has to go somewhere. Keeping `a`
// uniformly stale means a symmetric swap is always consistent, and the deferred
// rank-k update at panel close applies to every trailing column alike.
// c[t] = (D * row)[t] over the panel's completed pivot blocks.
static void apply_block_diagonal(threadgroup float2* out, threadgroup const float2* row, uint t,
                                 uint first, uint p, device const uint* two,
                                 device const float2* diagonal, device const float2* subdiag,
                                 uint lane, uint threads) {
    for (uint i = lane; i < t; i += threads) {
        uint k = p + i;
        if (two[first + k])
            out[i] = mul(row[i], diagonal[first + k]) + mul(row[i + 1], subdiag[first + k]);
        else if (i > 0 && two[first + k - 1])
            out[i] = mul(row[i - 1], subdiag[first + k - 1]) + mul(row[i], diagonal[first + k]);
        else
            out[i] = mul(row[i], diagonal[first + k]);
    }
}
// Brings column `col` of the front current with respect to the panel's pivots
// so far and writes it to panel column `dest`. Rows below `k` are never read,
// and rows at or past `rows` are left to `factor_panel_tail`: they feed no
// pivot search, so nothing in this kernel reads them back.
//
// `count` is how many *completed* pivots to subtract, which is not always the
// same as `dest`: the second column of a 2x2 pivot is written to `dest = t+1`
// but must still subtract only columns `0..t`, since column `t` holds the raw
// first column of the pivot being formed and is not a factor column yet.
//
// With SEARCH set, each lane also scans the rows it formed below `k` for the
// largest magnitude, ties to the lowest row, into `cm` and `imax`, while the
// value is still in a register. A separate pass re-read the column from device
// memory after the barrier. The lanes' results reduce to the same row whichever
// lane scanned which, so the pivot sequence is unchanged.
template <bool SEARCH>
static void form_column(device const float2* a, device float2* w, uint m, uint count, uint dest,
                        uint col, uint k, uint first, uint p, device const uint* two,
                        device const float2* diagonal, device const float2* subdiag,
                        threadgroup float2* rowbuf, threadgroup float2* coef, uint lane,
                        uint threads, uint rows, thread float* cm = nullptr,
                        thread uint* imax = nullptr) {
    for (uint i = lane; i < count; i += threads)
        rowbuf[i] = w[i * m + col];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    apply_block_diagonal(coef, rowbuf, count, first, p, two, diagonal, subdiag, lane, threads);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = k + lane; i < rows; i += threads) {
        float2 v = a[i * m + col];
        for (uint t2 = 0; t2 < count; t2++)
            v -= mul(w[t2 * m + i], coef[t2]);
        w[dest * m + i] = v;
        if (SEARCH && i > k) {
            float mv = mag(v);
            if (mv > *cm || (mv == *cm && i < *imax)) {
                *cm = mv;
                *imax = i;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
}
// Swaps rows p and q across the panel's formed columns, matching the symmetric
// swap applied to the front.
static void swap_panel_rows(device float2* w, uint m, uint columns, uint p, uint q, uint lane,
                            uint threads) {
    if (p != q)
        for (uint t = lane; t < columns; t += threads) {
            float2 v = w[t * m + p];
            w[t * m + p] = w[t * m + q];
            w[t * m + q] = v;
        }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
}
kernel void factor_panel(
    device const Node* nodes [[buffer(0)]], device const uint* ids [[buffer(5)]],
    device float2* fronts [[buffer(7)]], device float2* diagonal [[buffer(8)]],
    device float2* subdiag [[buffer(9)]], device uint* perms [[buffer(10)]],
    device uint* two [[buffer(11)]], device uint* status [[buffer(12)]],
    constant uint2& range [[buffer(14)]], device FrontState* state [[buffer(15)]],
    device float2* panels [[buffer(19)]], uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint sl [[thread_index_in_simdgroup]], uint simdw [[threads_per_simdgroup]],
    uint width [[threads_per_threadgroup]]) {
    uint s = ids[range.x + group];
    Node node = nodes[s];
    // Dispatched as wide as the pipeline allows rather than at a fixed width.
    // The pivot loop is serial in k, but everything inside one pivot -- forming
    // the column over m rows, the column and row searches, the multiplier
    // divide -- is a flat stride over the front, and near the root of the tree
    // there is often a single front in flight, so a narrow group leaves the
    // device idle. Past a point more lanes only cost occupancy, because the
    // pivot loop is serial in k, so the width is capped at 512, and
    // SPARTA_PANEL_THREADS overrides it.
    //
    // factor_narrow deliberately keeps the fixed PIVOT_THREADS: its fronts have
    // nc <= 32, so extra lanes have no row to own. While the two kernels shared
    // one constant, widening for this one dragged factor_narrow with it and made
    // it several times slower, which swallowed most of the gain.
    uint simds = (width + simdw - 1) / simdw;
    uint m = node.nr, n = node.nc, first = node.first, p = state[s].next;
    if (lane == 0) {
        state[s].start = p;
        state[s].width = 0;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (status[s] != 0 || p >= n)
        return; // Group-uniform.
    device float2* a = fronts + node.offset;
    device float2* w = panels + node.woffset;
    device uint* perm = perms + first;
    // Reduction scratch is written one slot per SIMD group, never one per
    // lane, so it is sized by MAX_SIMDS and is independent of the dispatch.
    threadgroup float maxima[MAX_SIMDS];
    threadgroup uint indices[MAX_SIMDS], choice[3], bad[MAX_SIMDS];
    threadgroup float2 rowbuf[PANEL_COLS], coef[PANEL_COLS];
    threadgroup float candidate[1];
    uint t = 0, k = p;
    while (t < PANEL && k < n) {
        // Forming the column also searches the fully summed rows below the
        // diagonal.
        float cm = 0;
        uint imax = k;
        form_column<true>(a, w, m, t, t, k, k, first, p, two, diagonal, subdiag, rowbuf, coef,
                          lane, width, n, &cm, &imax);
        reduce_argmax(cm, imax, maxima, indices, sg, sl, simds, &cm, &imax);
        float akk = mag(w[t * m + k]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
            choice[0] = 1;
            choice[1] = k;
            choice[2] = imax;
            if (!isfinite(cm) || !isfinite(akk)) {
                status[s] = PIVOT_NONFINITE | (k + 1);
                choice[0] = 0;
            }
            // An all-zero column reaches the 1x1 branch with a dead diagonal,
            // which floors it there; only a wholly zero front still fails.
        }
        // A failure is broadcast through choice[0] rather than re-read from
        // status: within this kernel only lane 0 writes the front's status word,
        // so the barrier need not fence device memory for it.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (choice[0] == 0)
            return;
        if (akk < 0.6403882f * cm) {
            // Row imax equals column imax by symmetry, and that column may lie
            // outside the panel, so its pending updates are applied on the fly.
            for (uint i = lane; i < t; i += width)
                rowbuf[i] = w[i * m + imax];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            apply_block_diagonal(coef, rowbuf, t, first, p, two, diagonal, subdiag, lane, width);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // The sweep also forms the candidate's own diagonal entry, on
            // whichever lane owns row imax, and hands its magnitude to lane 0
            // through `candidate`. Lane 0 used to form it alone, which put a
            // chain of up to PANEL dependent loads behind the decision.
            float rm = 0;
            for (uint j = k + lane; j < n; j += width) {
                float2 v = a[imax * m + j];
                for (uint t2 = 0; t2 < t; t2++)
                    v -= mul(w[t2 * m + j], coef[t2]);
                if (j == imax)
                    candidate[0] = mag(v);
                else
                    rm = max(rm, mag(v));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            // The reduction's own barrier also publishes `candidate`.
            rm = reduce_max(rm, maxima, sg, sl, simds);
            if (lane == 0) {
                float aii = candidate[0];
                if (!isfinite(rm) || !isfinite(aii)) {
                    status[s] = PIVOT_NONFINITE | (k + 1);
                    choice[0] = 0;
                } else if (akk >= 0.6403882f * cm * (cm / rm))
                    choice[1] = k;
                else if (aii >= 0.6403882f * rm)
                    choice[1] = imax;
                else {
                    choice[0] = 2;
                    choice[1] = k;
                    choice[2] = imax;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (choice[0] == 0)
            return;
        uint step = choice[0], pick = choice[1], second = choice[2];
        // Set by the multiplier divide below on any non-finite value it writes.
        uint invalid = 0;
        if (step == 1) {
            if (pick != k) {
                cooperative_swap(a, m, n, k, pick, perm, lane, width);
                swap_panel_rows(w, m, t + 1, k, pick, lane, width);
                // Position k now holds a different column of the stale front.
                form_column<false>(a, w, m, t, t, k, k, first, p, two, diagonal, subdiag,
                                   rowbuf, coef, lane, width, n);
            }
            float2 raw = w[t * m + k], d = raw;
            if (pivot_dead(raw)) {
                float big = front_scale(a, m, n, lane, width, maxima, sg, sl, simds);
                if (big <= 0) {
                    if (lane == 0)
                        status[s] = PIVOT_ZERO_COLUMN | (k + 1);
                    return;
                }
                d = floor_pivot(raw, PIVOT_FLOOR * big);
                if (lane == 0) {
                    w[t * m + k] = d;
                    state[s].perturbed++;
                }
                threadgroup_barrier(mem_flags::mem_device);
            }
            if (lane == 0)
                diagonal[first + k] = d;
            for (uint i = k + 1 + lane; i < n; i += width) {
                float2 v = cdiv(w[t * m + i], d);
                invalid |= uint(!all(isfinite(v)));
                w[t * m + i] = v;
            }
        } else {
            // 2x2 on (k, k+1) after bringing the partner to k+1. Column k keeps
            // its formed values; only its rows move.
            cooperative_swap(a, m, n, k + 1, second, perm, lane, width);
            swap_panel_rows(w, m, t + 1, k + 1, second, lane, width);
            form_column<false>(a, w, m, t, t + 1, k + 1, k, first, p, two, diagonal, subdiag,
                               rowbuf, coef, lane, width, n);
            float2 da = w[t * m + k], db = w[t * m + k + 1], dc = w[(t + 1) * m + k + 1];
            float scale = max(mag(da), max(mag(db), mag(dc)));
            float2 aa = da / scale, bb = db / scale, cc = dc / scale,
                   det = mul(aa, cc) - mul(bb, bb);
            if (!all(isfinite(det))) {
                if (lane == 0)
                    status[s] = PIVOT_NONFINITE | (k + 1);
                return;
            }
            if (mag(det) <= PIVOT_EPS) {
                // No usable 2x2 inverse left. Demote to a floored 1x1 on column
                // k; the partner column was already formed at t+1 but is simply
                // not consumed, and the next pivot re-forms it. Group-uniform.
                step = 1;
                float2 d = da;
                if (pivot_dead(da)) {
                    float big = front_scale(a, m, n, lane, width, maxima, sg, sl, simds);
                    if (big <= 0) {
                        if (lane == 0)
                            status[s] = PIVOT_TINY_2X2 | (k + 1);
                        return;
                    }
                    d = floor_pivot(da, PIVOT_FLOOR * big);
                }
                if (lane == 0) {
                    diagonal[first + k] = d;
                    w[t * m + k] = d;
                    state[s].perturbed++;
                }
                threadgroup_barrier(mem_flags::mem_device);
                for (uint i = k + 1 + lane; i < n; i += width) {
                    float2 v = cdiv(w[t * m + i], d);
                    invalid |= uint(!all(isfinite(v)));
                    w[t * m + i] = v;
                }
            } else {
                if (lane == 0) {
                    diagonal[first + k] = da;
                    diagonal[first + k + 1] = dc;
                    subdiag[first + k] = db;
                    two[first + k] = 1;
                }
                for (uint i = k + 2 + lane; i < n; i += width) {
                    float2 x = w[t * m + i] / scale, y = w[(t + 1) * m + i] / scale;
                    float2 u = cdiv(mul(x, cc) - mul(y, bb), det);
                    float2 v = cdiv(mul(y, aa) - mul(x, bb), det);
                    invalid |= uint(!all(isfinite(u)) || !all(isfinite(v)));
                    w[t * m + i] = u;
                    w[(t + 1) * m + i] = v;
                }
                // Every lane read `db` out of this slot above, and lanes are not in
                // lockstep: without this barrier lane 0 can reach the store
                // while another lane is still loading, handing that lane a zero
                // off-diagonal. It then inverts a different 2x2 block and
                // writes wrong multipliers -- but only for the rows it owns, so
                // the pivot search never sees it and the factorization looks
                // healthy while the rows below nc are silently wrong.
                // factor_narrow is safe from this only by accident: its
                // reduce_or falls between the same read and write.
                threadgroup_barrier(mem_flags::mem_device);
                if (lane == 0)
                    w[t * m + k + 1] = float2(0);
            }
        }
        // Each lane checked the multipliers it wrote, so no pass re-reads them.
        if (reduce_or(invalid, bad, sg, sl, simds)) {
            if (lane == 0)
                status[s] = 0xfffffffeu;
            return;
        }
        t += step;
        k += step;
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
    // Publish the panel's factor columns so the deferred updates and the solve
    // read them from the front like any other column. Only the fully summed
    // rows: `factor_panel_tail` writes the rows below them straight into the
    // front, without ever putting them in the panel workspace.
    for (uint idx = lane; idx < t * n; idx += width) {
        uint t2 = idx / n, i = idx - t2 * n, col = p + t2;
        if (i >= col)
            a[i * m + col] = w[t2 * m + i];
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (lane == 0) {
        state[s].next = p + t;
        state[s].start = p;
        state[s].width = t;
    }
}

// Rows one factor_panel_tail threadgroup solves at a time, and the width it is
// dispatched at: one thread per row of the block. The block is staged in
// threadgroup memory so the front is read along its rows, which is the only way
// to read it coalesced -- a front is row-major, so the column of it that each
// row of the panel needs is strided by nr, and one thread per row reading its
// own row directly wastes about eight bytes in sixty-four.
#ifndef TAIL_ROWS
#define TAIL_ROWS 128
#endif
// Applies a finished panel to the rows below the fully summed block.
//
// factor_panel is serial in the pivot index and can only be one threadgroup per
// front, because each pivot's search reads what the previous pivot wrote.
// Forming the panel's columns over the rows *below* nc has no such dependency:
// those rows feed no search, so once the pivot sequence and the block diagonal
// are fixed, each of them is an independent triangular solve against the panel
// the search already produced. On the measured EMerge systems that is 71-74% of
// the panel's arithmetic over 134k-169k rows, and it was running at the two
// threadgroups one-group-per-front allows.
//
// A block of TAIL_ROWS rows is staged, solved in place and written back. Each
// thread owns one row of the block, so the rank-k accumulation reads
// threadgroup memory rather than device memory: factor_panel re-read the panel
// workspace once per (column, earlier column) pair, which is PANEL/2 times the
// traffic. Staging is what makes the device traffic coalesced, and it is worth
// more than the arithmetic saving.
//
// The arithmetic is form_column's, in form_column's order, against the same
// coefficients: `coefficient[t]` is built by the same apply_block_diagonal over
// the same panel row. So the factors are bit-identical to taking the whole
// panel in one threadgroup, and an A/B of this change compares like with like.
kernel void factor_panel_tail(
    device const Node* nodes [[buffer(0)]], device const uint* ids [[buffer(5)]],
    device float2* fronts [[buffer(7)]], device const float2* diagonal [[buffer(8)]],
    device const float2* subdiag [[buffer(9)]], device const uint* two [[buffer(11)]],
    device uint* status [[buffer(12)]], constant uint4& range [[buffer(14)]],
    device const FrontState* state [[buffer(15)]], device const float2* panels [[buffer(19)]],
    uint group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint sl [[thread_index_in_simdgroup]],
    uint simdw [[threads_per_simdgroup]], uint width [[threads_per_threadgroup]]) {
    threadgroup uint bad[MAX_SIMDS];
    threadgroup float2 rowbuf[PANEL_COLS];
    // Column t of the panel subtracts coefficient[t][t2] times panel column t2,
    // for t2 < consumed[t]. Shared by every row this threadgroup owns.
    threadgroup float2 coefficient[PANEL_COLS][PANEL_COLS];
    threadgroup uint consumed[PANEL_COLS];
    threadgroup float2 block[TAIL_ROWS][PANEL_COLS];
    uint splits = range.z;
    uint s = ids[range.x + group / splits];
    uint slice = group - (group / splits) * splits;
    Node node = nodes[s];
    FrontState st = state[s];
    uint m = node.nr, n = node.nc, first = node.first;
    uint p = st.start, taken = st.width;
    // Group-uniform: a front whose panel took nothing, or that has already
    // failed, has nothing below to solve.
    if (taken == 0 || status[s] != 0 || n >= m)
        return;
    device float2* a = fronts + node.offset;
    device const float2* w = panels + node.woffset;
    for (uint t = 0; t < taken; t++) {
        uint col = p + t;
        // The second column of a 2x2 consumes one fewer than its position: the
        // pair's first column is not a factor column when the second is formed.
        uint count = (t > 0 && two[first + col - 1]) ? t - 1 : t;
        if (lane == 0)
            consumed[t] = count;
        for (uint i = lane; i < count; i += width)
            rowbuf[i] = w[i * m + col];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        apply_block_diagonal(&coefficient[t][0], rowbuf, count, first, p, two, diagonal, subdiag,
                             lane, width);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint invalid = 0;
    for (uint base = n + slice * TAIL_ROWS; base < m; base += splits * TAIL_ROWS) {
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
            uint t = 0;
            while (t < taken) {
                uint pivot = p + t, count = consumed[t];
                if (two[first + pivot] && t + 1 < taken) {
                    float2 v0 = row[t], v1 = row[t + 1];
                    for (uint t2 = 0; t2 < count; t2++) {
                        v0 -= mul(row[t2], coefficient[t][t2]);
                        v1 -= mul(row[t2], coefficient[t + 1][t2]);
                    }
                    float2 da = diagonal[first + pivot], db = subdiag[first + pivot],
                           dc = diagonal[first + pivot + 1];
                    float scale = max(mag(da), max(mag(db), mag(dc)));
                    float2 aa = da / scale, bb = db / scale, cc = dc / scale,
                           det = mul(aa, cc) - mul(bb, bb);
                    float2 x = v0 / scale, y = v1 / scale;
                    row[t] = cdiv(mul(x, cc) - mul(y, bb), det);
                    row[t + 1] = cdiv(mul(y, aa) - mul(x, bb), det);
                    t += 2;
                } else {
                    float2 v = row[t];
                    for (uint t2 = 0; t2 < count; t2++)
                        v -= mul(row[t2], coefficient[t][t2]);
                    row[t] = cdiv(v, diagonal[first + pivot]);
                    t += 1;
                }
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
    // Every lane runs the reduction: it barriers internally, so it must not sit
    // behind a lane-uniform condition.
    if (reduce_or(invalid, bad, sg, sl, (width + simdw - 1) / simdw) && lane == 0)
        status[s] = 0xfffffffeu;
}
