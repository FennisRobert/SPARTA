// Unsymmetric LU: the deferred trailing updates.
//
// Same two-cadence split as the symmetric path, over a different region.
//
// `update_strip_lu` refreshes the **cross**: the fully summed columns over
// every row, and the fully summed rows over every column. LDL^T gets the second
// arm free by mirroring each write; here it has to be computed, and it is not
// optional -- the next panel's TRSM reads exactly those rows out to the last
// column before it can finish.
//
// `update_schur_lu` drains the square trailing block, which nobody reads until
// the parent assembles it, on the host's own cadence. That is the same
// deferral argument as the symmetric path, and it survives for the same reason:
// pivot searches, the TRSM and `store_factors` all stay inside the cross.
//
// Both operands come straight out of the front -- L from the columns the panel
// published, U from the rows its TRSM finished -- so there is no block diagonal
// to fold and no `two` array to consult. That makes the inner product a plain
// complex multiply-add, cheaper per element than the symmetric one; the cost is
// that the Schur block is a full square rather than a triangle, so there are
// close to twice as many tiles for the same amount of arithmetic per element.

// Deferred rank-`width` update of the cross past the panel. One
// STRIP_ROWS x 16 tile per threadgroup, a STRIP_BLOCK x 1 register block per
// thread, exactly as on the symmetric path.
kernel void update_strip_lu(
    device const Node* nodes [[buffer(0)]], device float2* fronts [[buffer(7)]],
    device atomic_uint* status [[buffer(12)]], constant uint2& range [[buffer(14)]],
    device const FrontState* state [[buffer(15)]], device const UpdateTile* jobs [[buffer(16)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 lane [[thread_position_in_threadgroup]]) {
    UpdateTile job = jobs[range.x + group.x];
    uint s = job.node;
    Node node = nodes[s];
    FrontState st = state[s];
    uint end = st.start + st.width;
    // A tile can only write where both its rows and its columns are past the
    // panel, so the earlier of the two edges decides it -- the same index the
    // host ordered the tiles by. Once `end` reaches nc there is no cross left.
    // Group-uniform: every lane reads the same `end` and the same tile.
    if (st.width == 0 || end >= node.nc ||
        min((job.col + 1u) * 16u, (job.row + 1u) * (uint)STRIP_ROWS) <= end ||
        atomic_load_explicit(status + s, memory_order_relaxed) != 0)
        return;
    uint m = node.nr;
    device float2* a = fronts + node.offset;
    threadgroup float2 li[STRIP_ROWS][SCHUR_CHUNK], uj[16][SCHUR_CHUNK];
    uint rowbase = job.row * STRIP_ROWS, colbase = job.col * 16;
    uint linear = lane.y * 16 + lane.x;
    float2 acc[STRIP_BLOCK];
    for (uint d = 0; d < STRIP_BLOCK; d++)
        acc[d] = float2(0);
    for (uint c0 = st.start; c0 < end;) {
        uint span = min((uint)SCHUR_CHUNK, end - c0);
        // Two staging loops rather than one, because the two operands run in
        // opposite directions: L is read down a column of the front and U along
        // a row, so each wants its own index as the fast one.
        for (uint idx = linear; idx < STRIP_ROWS * span; idx += 256) {
            uint r = idx / span, c = idx - r * span, i = rowbase + r;
            li[r][c] = i < m ? a[i * m + c0 + c] : float2(0);
        }
        for (uint idx = linear; idx < 16 * span; idx += 256) {
            uint c = idx / 16, r = idx - c * 16, j = colbase + r;
            uj[r][c] = j < m ? a[(c0 + c) * m + j] : float2(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = 0; c < span; c++) {
            float2 uv = uj[lane.x][c];
            for (uint d = 0; d < STRIP_BLOCK; d++)
                acc[d] -= mul(li[STRIP_BLOCK * lane.y + d][c], uv);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        c0 += span;
    }
    uint j = colbase + lane.x;
    if (j >= m || j < end)
        return;
    for (uint d = 0; d < STRIP_BLOCK; d++) {
        uint i = rowbase + STRIP_BLOCK * lane.y + d;
        // Inside the panel the factor is already final; outside the cross the
        // element belongs to the Schur complement and is drained separately.
        if (i >= m || i < end || (i >= node.nc && j >= node.nc))
            continue;
        float2 v = a[i * m + j] + acc[d];
        if (!all(isfinite(v)))
            atomic_store_explicit(status + s, 0xfffffffeu, memory_order_relaxed);
        a[i * m + j] = v;
    }
}

// Blocked Schur update: one rank-(next - panel) product per drain instead of
// one rank-1 product per pivot. Each threadgroup owns a SCHUR_TILE block of the
// trailing square, stages both operands in threadgroup memory, and accumulates
// the whole pending range in registers before touching the front once.
kernel void update_schur_lu(
    device const Node* nodes [[buffer(0)]], device float2* fronts [[buffer(7)]],
    device atomic_uint* status [[buffer(12)]], constant uint2& range [[buffer(14)]],
    device const FrontState* state [[buffer(15)]], device const UpdateTile* jobs [[buffer(18)]],
    uint2 group [[threadgroup_position_in_grid]], uint2 lane [[thread_position_in_threadgroup]]) {
    UpdateTile job = jobs[range.x + group.x];
    uint s = job.node;
    Node node = nodes[s];
    FrontState st = state[s];
    if (st.next <= st.panel || atomic_load_explicit(status + s, memory_order_relaxed) != 0)
        return;
    uint m = node.nr, base = node.nc;
    device float2* a = fronts + node.offset;
    threadgroup float2 li[SCHUR_TILE][SCHUR_CHUNK], uj[SCHUR_TILE][SCHUR_CHUNK];
    uint rowbase = base + job.row * SCHUR_TILE, colbase = base + job.col * SCHUR_TILE;
    uint linear = lane.y * 16 + lane.x;
    // As in update_schur, the tile is a 4 x 4 grid of SUB x SUB blocks, sixteen
    // threads each and two per simdgroup, and a block wholly past the front's
    // last row or column skips the product. Row-major block order pairs the
    // two blocks of a simdgroup along a row, so a tile that runs past the
    // front's last row idles whole simdgroups. There is no triangle to skip.
    constexpr uint SUB = SCHUR_TILE / 4;
    uint block = linear / 16, within = linear % 16;
    uint myrow = SUB * (block / 4) + SCHUR_BLOCK * (within / 4);
    uint mycol = SUB * (block % 4) + SCHUR_BLOCK * (within % 4);
    bool idle = rowbase + SUB * (block / 4) >= m || colbase + SUB * (block % 4) >= m;
    float2 acc[SCHUR_BLOCK][SCHUR_BLOCK];
    for (uint di = 0; di < SCHUR_BLOCK; di++)
        for (uint dj = 0; dj < SCHUR_BLOCK; dj++)
            acc[di][dj] = float2(0);
    for (uint c0 = st.panel; c0 < st.next;) {
        uint span = min((uint)SCHUR_CHUNK, st.next - c0);
        for (uint idx = linear; idx < SCHUR_TILE * span; idx += 256) {
            uint r = idx / span, c = idx - r * span, i = rowbase + r;
            li[r][c] = i < m ? a[i * m + c0 + c] : float2(0);
        }
        for (uint idx = linear; idx < SCHUR_TILE * span; idx += 256) {
            uint c = idx / SCHUR_TILE, r = idx - c * SCHUR_TILE, j = colbase + r;
            uj[r][c] = j < m ? a[(c0 + c) * m + j] : float2(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = 0; c < (idle ? 0u : span); c++) {
            float2 lv[SCHUR_BLOCK], uv[SCHUR_BLOCK];
            for (uint d = 0; d < SCHUR_BLOCK; d++) {
                lv[d] = li[myrow + d][c];
                uv[d] = uj[mycol + d][c];
            }
            for (uint di = 0; di < SCHUR_BLOCK; di++)
                for (uint dj = 0; dj < SCHUR_BLOCK; dj++)
                    acc[di][dj] -= mul(lv[di], uv[dj]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        c0 += span;
    }
    for (uint di = 0; di < SCHUR_BLOCK; di++)
        for (uint dj = 0; dj < SCHUR_BLOCK; dj++) {
            uint i = rowbase + myrow + di, j = colbase + mycol + dj;
            if (idle || i >= m || j >= m)
                continue;
            float2 v = a[i * m + j] + acc[di][dj];
            if (!all(isfinite(v)))
                atomic_store_explicit(status + s, 0xfffffffeu, memory_order_relaxed);
            a[i * m + j] = v;
        }
}
