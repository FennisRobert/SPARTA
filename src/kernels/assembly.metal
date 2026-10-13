// Front assembly and factor extraction.
//
// `zero_fronts` clears the front workspace, `initialize_large` scatters the
// original matrix entries into it, `assemble_children` adds the child Schur
// blocks; `store_factors` copies a finished
// front's factor columns into the permanent store before the workspace is
// recycled. Both come in an LDL^T and an LU form, differing only in whether
// the front is mirrored about its diagonal and whether the row block of U is
// kept as well.
// `mirror` is the only difference between the two factorizations here. LDL^T
// stores one triangle of the input and reflects each scattered entry across the
// diagonal; LU is handed both triangles and puts every entry exactly where the
// plan says. Both callers pass a compile-time constant, so the branch folds
// away. Child Schur blocks are square either way, which is why
// `assemble_children` needs no mirrored form and serves both paths.
static void assemble_front(bool mirror, device const Node* nodes, device const Child* children,
                           device const uint* maps, device const Entry* entries,
                           device const uint* ids, device const float2* values,
                           device float2* fronts, device float2* subdiag, device uint* perms,
                           device uint* two, device uint* status, constant uint2& range,
                           device FrontState* state, threadgroup uint* bad, uint group, uint lane,
                           uint sg, uint sl, uint simdw, uint width) {
    // Dispatched as wide as the pipeline allows rather than at a fixed width:
    // every loop here is a flat stride over the front, with no serial
    // dependency to limit the threadgroup, and near the root there are too few
    // fronts for one narrow group each to keep the device busy. `width` is the
    // real threadgroup size, so the strides stay correct at whatever the host
    // dispatched.
    uint s = ids[range.x + group];
    Node node = nodes[s];
    uint m = node.nr;
    device float2* a = fronts + node.offset;
    if (lane == 0) {
        status[s] = 0;
        state[s] = FrontState{0, 0, 0, 0, 0};
    }
    // The front itself is cleared by zero_fronts, in its own dispatch: it is by
    // far the largest write here, it is trivially parallel, and one threadgroup
    // per front leaves it to a single GPU core.
    // two/subdiag describe the LDL^T block diagonal and are dead weight on the
    // LU path, but clearing them costs one pass over nc and keeps one loop.
    for (uint i = lane; i < node.nc; i += width) {
        perms[node.first + i] = i;
        two[node.first + i] = 0;
        subdiag[node.first + i] = float2(0);
    }
    threadgroup_barrier(mem_flags::mem_device);
    // Input and assembly overflow must fail even if a bad value lies outside
    // the fully-summed pivot block. No racing writes to status across lanes.
    //
    // The check rides along with the two loops that write, instead of being a
    // separate sweep of the assembled front afterwards. That is the same
    // coverage, not a weaker one: the front was zeroed by the preceding
    // dispatch, so every entry is
    // either still zero, a scattered original value, or a sum of child
    // contributions, and both of the latter are tested here at the moment they
    // are written. A position touched by several children is tested after each,
    // so the last test sees the final value -- and non-finiteness is absorbing
    // under addition anyway, so an intermediate that goes bad cannot come back.
    //
    // It is worth the care: the sweep it replaces was a full m*m read of a
    // front that had just been written, and it cost 25.7% of assembly time --
    // 0.233s of 0.907s on a 30^3 curl-curl grid, 11.7% of the whole
    // factorization. This version adds no memory traffic at all.
    uint invalid = 0;
    for (uint e = lane; e < node.nentries; e += width) {
        Entry v = entries[node.entries + e];
        uint i = v.position / m, j = v.position % m;
        float2 val = values[v.source];
        invalid |= uint(!all(isfinite(val)));
        a[i * m + j] = val;
        if (mirror)
            a[j * m + i] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Every lane runs the reduction: it barriers internally, so it must not sit
    // behind a lane-uniform condition.
    uint any = reduce_or(invalid, bad, sg, sl, (width + simdw - 1) / simdw);
    if (lane == 0 && any)
        status[s] = 0xfffffffeu;
}
// One parameter list, shared by the two kernels that differ only in whether
// they mirror. Hand-formatted and fenced off: clang-format cannot tell a
// pointer declaration from a multiplication inside a macro body, so left alone
// it writes `Node *nodes` here and `Node* nodes` everywhere else.
// clang-format off
#define SPARTA_ASSEMBLY_ARGS                                       \
    device const Node* nodes        [[buffer(0)]],                 \
    device const Child* children    [[buffer(2)]],                 \
    device const uint* maps         [[buffer(3)]],                 \
    device const Entry* entries     [[buffer(4)]],                 \
    device const uint* ids          [[buffer(5)]],                 \
    device const float2* values     [[buffer(6)]],                 \
    device float2* fronts           [[buffer(7)]],                 \
    device float2* subdiag          [[buffer(9)]],                 \
    device uint* perms              [[buffer(10)]],                \
    device uint* two                [[buffer(11)]],                \
    device uint* status             [[buffer(12)]],                \
    constant uint2& range           [[buffer(14)]],                \
    device FrontState* state        [[buffer(15)]],                \
    uint group                      [[threadgroup_position_in_grid]],   \
    uint lane                       [[thread_index_in_threadgroup]],    \
    uint sg                         [[simdgroup_index_in_threadgroup]], \
    uint sl                         [[thread_index_in_simdgroup]],      \
    uint simdw                      [[threads_per_simdgroup]],          \
    uint width                      [[threads_per_threadgroup]]

#define SPARTA_ASSEMBLY_CALL(mirror)                                              \
    assemble_front(mirror, nodes, children, maps, entries, ids, values, fronts,   \
                   subdiag, perms, two, status, range, state, bad, group, lane,   \
                   sg, sl, simdw, width)
// clang-format on
// Metal forbids a threadgroup declaration outside a kernel, so the reduction
// scratch is allocated here and passed down.
kernel void initialize_large(SPARTA_ASSEMBLY_ARGS) {
    threadgroup uint bad[MAX_SIMDS];
    SPARTA_ASSEMBLY_CALL(true);
}
kernel void initialize_large_lu(SPARTA_ASSEMBLY_ARGS) {
    threadgroup uint bad[MAX_SIMDS];
    SPARTA_ASSEMBLY_CALL(false);
}
// Adds one child's Schur block into each of a level's fronts.
//
// Split out of `initialize_large`, and dispatched once per child ordinal rather
// than once per level, because two children of the same front accumulate into
// overlapping regions of it and so cannot run at the same time. Ordinal `c` is
// a whole dispatch: every front in the level applies its `c`th child, and no
// front applies two at once, so the only ordering needed is between dispatches
// and the encoder already provides it. A front with fewer children than the
// ordinal simply returns.
//
// Within one child the mapping (i,j) -> (maps[i], maps[j]) is injective, so no
// two lanes ever accumulate into the same destination and the read-modify-write
// needs no atomic. That is what lets the work be sliced: `range.z` threadgroups
// share each front, strided.
//
// This is worth the extra dispatches. The extend-add moves three times the
// traffic of the front clear -- it reads the child block, reads the parent
// entry and writes it back -- and at one threadgroup per front it ran at 111
// GB/s on an M4 Pro. Sliced across threadgroups it reaches 208 GB/s, which is
// the same roofline the clear sits at, and the kernel's own time falls 1.88x.
// Measured slice counts 1/4/8/16 on a 30^3 curl-curl grid: 0.522s, 0.295s,
// 0.277s, 0.278s -- it saturates by 8.
kernel void assemble_children(device const Node* nodes [[buffer(0)]],
                              device const Child* children [[buffer(2)]],
                              device const uint* maps [[buffer(3)]],
                              device const uint* ids [[buffer(5)]],
                              device float2* fronts [[buffer(7)]],
                              device uint* status [[buffer(12)]],
                              constant uint4& range [[buffer(14)]],
                              uint group [[threadgroup_position_in_grid]],
                              uint lane [[thread_index_in_threadgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]],
                              uint sl [[thread_index_in_simdgroup]],
                              uint simdw [[threads_per_simdgroup]],
                              uint width [[threads_per_threadgroup]]) {
    threadgroup uint bad[MAX_SIMDS];
    uint splits = range.z, ordinal = range.w;
    uint s = ids[range.x + group / splits];
    uint slice = group - (group / splits) * splits;
    Node node = nodes[s];
    // Group-uniform: a front with no `ordinal`th child, or one already failed,
    // has nothing to add.
    if (ordinal >= node.count || status[s] == 0xffffffffu)
        return;
    Child child = children[node.children + ordinal];
    if (status[child.node] != 0) {
        if (lane == 0 && slice == 0)
            status[s] = 0xffffffffu;
        return;
    }
    uint m = node.nr;
    Node cn = nodes[child.node];
    uint tail = cn.nr - cn.nc;
    device float2* a = fronts + node.offset;
    // `soffset`/`sld` locate the child's block whether it was compacted out of
    // its front -- in which case the front slot has since been reused -- or
    // left strided inside it.
    device const float2* source = fronts + cn.soffset;
    uint sld = cn.sld;
    uint invalid = 0;
    for (uint index = slice * width + lane; index < tail * tail; index += splits * width) {
        uint i = index / tail, j = index - i * tail;
        uint dest = maps[child.map + i] * m + maps[child.map + j];
        float2 v = a[dest] + source[i * sld + j];
        invalid |= uint(!all(isfinite(v)));
        a[dest] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Every lane runs the reduction: it barriers internally, so it must not sit
    // behind a lane-uniform condition.
    uint any = reduce_or(invalid, bad, sg, sl, (width + simdw - 1) / simdw);
    if (lane == 0 && any)
        status[s] = 0xfffffffeu;
}
// Clears the fronts a level is about to assemble.
//
// Split out of `initialize_large` because of how it is dispatched rather than
// what it computes. Assembly runs one threadgroup per front, which is right for
// the scatter and the child extend-add -- both need the whole front visible to
// one group so their phases can be ordered by a threadgroup barrier. Clearing
// needs no ordering at all, and it is the largest write in the kernel: a
// 5181-row front is 26.8 million complex entries, 214 MB, and at one
// threadgroup per front a single GPU core wrote all of it while the rest of the
// device waited.
//
// Here each front is cut into `range.z` slices and one threadgroup takes each,
// so a level with a handful of huge fronts still fills the machine. The host
// picks the slice count per level from the level's largest front and how many
// fronts it holds; a small front over-split simply finds its slice empty and
// exits.
kernel void zero_fronts(device const Node* nodes [[buffer(0)]], device const uint* ids [[buffer(5)]],
                        device float2* fronts [[buffer(7)]], constant uint4& range [[buffer(14)]],
                        uint group [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_threadgroup]],
                        uint width [[threads_per_threadgroup]]) {
    uint splits = range.z;
    uint s = ids[range.x + group / splits];
    uint slice = group - (group / splits) * splits;
    Node node = nodes[s];
    uint total = node.nr * node.nr;
    device float2* a = fronts + node.offset;
    for (uint i = slice * width + lane; i < total; i += splits * width)
        a[i] = float2(0);
}
// Compacts the trailing `nr-nc` square of a finished front into the node's
// contribution slot, densely, leading dimension `nr-nc`.
//
// This is what lets the front slot be recycled one level later instead of being
// held until the parent has assembled: the block the parent needs is a strided
// window on an `nr*nr` front, and moving it to its own slot shrinks what has to
// survive the gap from `nr*nr` to `(nr-nc)^2`. Peak workspace on
// `em_bandcond_sym1` falls 1.00 GiB -> 0.77 GiB.
//
// The destination is disjoint from every front live in this level -- the
// allocator places it so -- which is what makes this safe to run in the same
// dispatch as the factor copy, with no barrier between the two.
//
// Fronts the analysis judged not worth compacting keep their block where it is
// and are held until the parent has assembled instead; those are exactly the
// ones whose block already starts at its place inside the front, which is what
// the offset test below recognises. A fully summed front has no block at all.
static void compact_block(Node node, device float2* fronts, uint lane, uint width) {
    // `lane` and `width` are a thread's position and the stride over every
    // slice of the front, so a sliced dispatch divides the copy between them.
    uint m = node.nr, nc = node.nc, tail = m - nc;
    if (!tail || node.soffset == node.offset + nc * m + nc)
        return;
    device const float2* a = fronts + node.offset;
    device float2* block = fronts + node.soffset;
    for (uint idx = lane; idx < tail * tail; idx += width) {
        uint i = idx / tail, j = idx - i * tail;
        block[idx] = a[(nc + i) * m + nc + j];
    }
}
// Fronts are recycled between levels, so the nc factor columns each node
// contributes to the solve are copied into the permanent store the moment the
// node's level completes. Stored column-major (leading dimension nr) to match
// the solve's inner loop over rows.
//
// Dispatched like zero_fronts: `range.z` threadgroups share each front. Near
// the root a level holds one to three fronts, and at one threadgroup each this
// copy ran on that many GPU cores, which made it most of this kernel's time.
kernel void store_factors(
    device const Node* nodes [[buffer(0)]], device const uint* ids [[buffer(5)]],
    device float2* fronts [[buffer(7)]], constant uint4& range [[buffer(14)]],
    device float2* factors [[buffer(17)]], uint group [[threadgroup_position_in_grid]],
    uint local [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
    uint splits = range.z;
    uint s = ids[range.x + group / splits];
    uint lane = (group - (group / splits) * splits) * threads + local, width = splits * threads;
    Node node = nodes[s];
    uint m = node.nr;
    device const float2* a = fronts + node.offset;
    device float2* l = factors + node.loffset;
    // A flat strided copy, so it takes whatever width the host dispatched.
    for (uint idx = lane; idx < node.nc * m; idx += width) {
        uint k = idx / m, i = idx - k * m;
        l[idx] = a[i * m + k];
    }
    compact_block(node, fronts, lane, width);
}
// The LU form of the same copy.
//
// The first nc columns are kept exactly as above, and they are already the
// whole of L: a finished front holds L11 strictly below its diagonal, U11 on
// and above it, and L21 beneath -- so one nr x nc block carries L11, U11 and
// L21 together, and the backward solve reads U11 out of it transposed. Only
// U12, the nc fully summed rows over the trailing columns, needs a second
// block. Storing it row-major with leading dimension nr - nc keeps row k of U
// contiguous, which is the order the backward solve sweeps it in.
kernel void store_factors_lu(
    device const Node* nodes [[buffer(0)]], device const uint* ids [[buffer(5)]],
    device float2* fronts [[buffer(7)]], constant uint4& range [[buffer(14)]],
    device float2* factors [[buffer(17)]], device float2* upper [[buffer(20)]],
    uint group [[threadgroup_position_in_grid]], uint local [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    uint splits = range.z;
    uint s = ids[range.x + group / splits];
    uint lane = (group - (group / splits) * splits) * threads + local, width = splits * threads;
    Node node = nodes[s];
    uint m = node.nr, nc = node.nc, tail = m - nc;
    device const float2* a = fronts + node.offset;
    device float2* l = factors + node.loffset;
    device float2* u = upper + node.uoffset;
    for (uint idx = lane; idx < nc * m; idx += width) {
        uint k = idx / m, i = idx - k * m;
        l[idx] = a[i * m + k];
    }
    for (uint idx = lane; idx < nc * tail; idx += width) {
        uint k = idx / tail, j = idx - k * tail;
        u[idx] = a[k * m + nc + j];
    }
    compact_block(node, fronts, lane, width);
}
