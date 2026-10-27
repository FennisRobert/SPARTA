/*==============================================================================

  SPARTA  --  sparse direct solver for Apple Silicon GPUs

  Solves A x = b for large sparse A, with the factorization and the triangular
  solves running on the GPU of an Apple Silicon Mac.

  The work happens in three steps:

      1. analyze    Study the pattern of nonzeros and plan the computation.
                    Runs on the CPU. Depends only on where the nonzeros are,
                    never on their values.

      2. factor     Break the matrix into factors on the GPU. Depends on the
                    values. Repeat it for any matrix sharing the pattern.

      3. solve      Use those factors to find x. Repeat it for any b.

  Steps 2 and 3 reuse the plan and the GPU memory from step 1, so a frequency
  sweep or a time loop pays for the analysis once.

  Requires Apple Silicon and macOS 13 or later. Arithmetic is single-precision
  complex (complex64). There is no CPU fallback: if the GPU is unavailable,
  calls fail rather than running slowly.

  Link with -lsparta. For C++, sparta.hpp wraps this header in a class.

==============================================================================*/

#ifndef SPARTA_H
#define SPARTA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*------------------------------------------------------------------------------
  Version
------------------------------------------------------------------------------*/

#define SPARTA_VERSION_MAJOR 0
#define SPARTA_VERSION_MINOR 1
#define SPARTA_VERSION_PATCH 4

/** Returns the library version as text, for example "0.3.1". */
const char *sparta_version(void);

/** Returns the GPU backend this library was built for: "metal" or "vulkan".
 *
 *  One build carries one backend's kernels, so this describes the library and
 *  not the machine. The string is static and outlives any call. Pair it with
 *  sparta_gpu_available(), which reports whether this machine can run them. */
const char *sparta_device(void);

/** Returns 1 when a usable GPU is visible to this process, 0 otherwise.
 *  Analysis works either way; factoring and solving need the GPU. */
int sparta_gpu_available(void);


/*------------------------------------------------------------------------------
  Numbers and matrices
------------------------------------------------------------------------------*/

/** One complex number. The layout matches C99 `float _Complex` and C++
 *  `std::complex<float>`, so an existing array can be cast rather than copied. */
typedef struct {
    float re;
    float im;
} sparta_complex;

/** What kind of matrix is being solved, which decides how it must be stored. */
typedef enum {
    /** Symmetric: A equals its own transpose. Supply the lower triangle only
     *  (entries on or below the diagonal). Factored as L D L^T. */
    SPARTA_SYMMETRIC = 0,

    /** General square matrix. Supply every nonzero entry. Factored as L U with
     *  partial pivoting. Expect roughly twice the memory and time of the
     *  symmetric path, so prefer SPARTA_SYMMETRIC when the matrix allows it. */
    SPARTA_GENERAL = 1
} sparta_matrix;

/*
  Matrices are given in compressed sparse column (CSC) form, as three arrays:

      col_ptr    n + 1 offsets. Column j occupies positions
                 col_ptr[j] .. col_ptr[j + 1] - 1. col_ptr[0] is 0.
      row_index  one row index per nonzero, ascending within each column and
                 without repeats.
      values     one number per nonzero, in the same order.

  The 2x2 matrix [[4, -1], [-1, 4]] stored as SPARTA_SYMMETRIC (lower triangle)
  is:

      col_ptr   = { 0, 2, 3 }
      row_index = { 0, 1, 1 }
      values    = { 4, -1, 4 }

  The pattern is fixed by sparta_analyze. Every later call to sparta_factor
  must pass a `values` array of the same length and in the same order.
*/


/*------------------------------------------------------------------------------
  Errors
------------------------------------------------------------------------------*/

/** Result of a call. Every function that can fail returns one of these. */
typedef enum {
    /** The call succeeded. */
    SPARTA_OK = 0,

    /** A input was rejected: a malformed matrix, a size that does not match the
     *  analyzed one, a setting outside its allowed range, or a null pointer. */
    SPARTA_ERROR_INVALID = 1,

    /** This machine or build cannot run the request, usually because no GPU
     *  is available, or because this build carries no GPU backend. */
    SPARTA_ERROR_UNSUPPORTED = 2,

    /** The GPU rejected or failed the work. */
    SPARTA_ERROR_DEVICE = 3,
    /** Former name of SPARTA_ERROR_DEVICE, from when Metal was the only
     *  backend. Same value; kept so existing callers keep compiling. */
    SPARTA_ERROR_METAL = SPARTA_ERROR_DEVICE,

    /** The factorization broke down at a pivot, which means the matrix is
     *  singular, nearly singular, or badly scaled. */
    SPARTA_ERROR_PIVOT = 4,

    /** A fault inside the library. Please report it. */
    SPARTA_ERROR_INTERNAL = 5
} sparta_status;

/** Describes the most recent failure on the calling thread.
 *
 *  Returns a sentence naming what went wrong, or an empty string if nothing has
 *  failed yet. The text belongs to the library: do not free it, and copy it if
 *  it must outlive the next SPARTA call on this thread. */
const char *sparta_last_error(void);


/*------------------------------------------------------------------------------
  Settings
------------------------------------------------------------------------------*/

/** Which algorithm chooses the order the unknowns are eliminated in. The order
 *  decides how much fill-in the factors carry, and so most of the cost. */
typedef enum {
    /** SCALPEL, SPARTA's own parallel nested dissection: the only ordering,
     *  and tunable through the ordering fields below. */
    SPARTA_ORDER_SCALPEL = 0
} sparta_ordering;

/** A named starting point for the settings, chosen to suit a kind of workload.
 *  Each is a starting point rather than a recommendation: the best settings
 *  depend on the matrix, so measure before keeping a change. */
typedef enum {
    /** The measured defaults. One factorization of an unstructured mesh. */
    SPARTA_PRESET_BALANCED = 0,

    /** One pattern factored many times, such as a frequency sweep. Spends more
     *  time on ordering quality because every factorization gets the benefit. */
    SPARTA_PRESET_SWEEP = 1,

    /** Factor once and discard. Orders quickly and accepts more fill-in. */
    SPARTA_PRESET_ONE_SHOT = 2,

    /** Regular grids, where grouping more columns together keeps paying off. */
    SPARTA_PRESET_STRUCTURED_GRID = 3,

    /** Ill-conditioned systems. Keeps strict pivoting and a balanced tree. */
    SPARTA_PRESET_CAREFUL = 4
} sparta_preset;

/** Every tuning choice available when analyzing a matrix.
 *
 *  Fill it with sparta_config_default or sparta_config_preset first, then
 *  change only what a workload needs. Always set `struct_size` to
 *  sizeof(sparta_config); it lets a newer library accept a program built
 *  against an older header. */
typedef struct {
    /** Must be sizeof(sparta_config). */
    size_t struct_size;

    /* ---- Ordering ---- */

    /** Which ordering algorithm to use. See sparta_ordering. */
    sparta_ordering ordering;

    /** Smallest number of columns a group must eliminate before it may merge
     *  with its neighbour. Larger values give fewer, wider blocks, which suits
     *  the GPU but stores more fill-in. Default 16. */
    size_t nemin;

    /** Largest block the plan may build, in rows, or 0 for no limit. A block
     *  larger than this signals a bad ordering, so the limit exists to fail
     *  early instead of allocating for it. Default 0. */
    size_t max_front;

    /* ---- Ordering quality. ---- */

    /** How unevenly the ordering may split a region, from 0 (perfectly even)
     *  to 1. Even splits balance the work; uneven ones can cut fewer edges.
     *  Default 0.8. */
    double imbalance;

    /** Extra coarsening cycles spent looking for a smaller split. More cycles
     *  cost ordering time and usually save factorization time. Default 4. */
    uint32_t wide_cycles;

    /** Refinement passes over each split once it is found. Default 2. */
    uint32_t fm_passes;

    /** How many candidate splits to try per region, keeping the best.
     *  Default 4. */
    uint32_t separator_trials;

    /* ---- Accuracy ---- */

    /** For SPARTA_GENERAL only: how much the factorization may prefer the
     *  diagonal when choosing a pivot, from 0 to 1. 0 keeps strict partial
     *  pivoting, which is the most accurate. A larger value accepts a diagonal
     *  pivot within that fraction of the best candidate, which is faster but
     *  looser. Default 0. */
    double lu_diag_threshold;

    /* ---- GPU layout. Advanced; the defaults are measured. ---- */

    /** Height, in rows, at which a block switches from the simple GPU kernel to
     *  the cooperative one. Default 32. */
    uint32_t small_front_rows;

    /** Pivot panels completed between two updates of the remaining matrix, or 0
     *  to update once per level of the tree. Default 0. */
    uint32_t schur_batch;

    /** GPU threads per block in the panel factorization. Default 512. */
    uint32_t panel_threads;

    /** GPU threads per block in the triangular solves, or 0 to choose per
     *  block. Default 0. */
    uint32_t solve_threads;

    /** Pivots accumulated before the trailing update runs. Must be 16; the GPU
     *  kernels are written for that width. Default 16. */
    uint32_t panel;

    /** Register block edge in the trailing update. Each GPU thread group owns a
     *  tile of 16 * schur_block squared. Between 1 and 8. Default 4. */
    uint32_t schur_block;

    /** Register block height in the strip update, between 1 and 8. Default 4. */
    uint32_t strip_block;

    /** Width above which a block takes the wide kernel path. Default 32. */
    uint32_t narrow_cols;

    /** GPU threads cooperating on one pivot search. A multiple of 32.
     *  Default 128. */
    uint32_t pivot_threads;

    /** Panel columns staged per pass of the trailing update. Default 16. */
    uint32_t schur_chunk;

    /* ---- CPU and GPU together. ---- */

    /** Share of the factorization work, from 0 to below 1, that the CPU does
     *  while the GPU does the rest at the same time. 0 is the GPU alone; 0.3 is
     *  a good start. The share then adapts after every factorization to the
     *  measured speed of each. Read when the solver is created. Only the Metal
     *  backend has a hybrid path; elsewhere it is ignored. Default 0, or the
     *  SPARTA_HYBRID_CPU environment variable. */
    double hybrid_cpu;

    /* ---- Threads. ---- */

    /** CPU threads for SPARTA's own parallel work: the analysis and the
     *  hybrid's CPU share. 0 uses the process-wide pool, which the
     *  RAYON_NUM_THREADS environment variable sizes; a positive count, up to
     *  4096, gives each analysis and solver a pool of its own. The
     *  SPARTA_NUM_THREADS environment variable, when set to a positive count,
     *  overrides this. Default 0. */
    uint32_t threads;
} sparta_config;

/** Fills `config` with the default settings. */
void sparta_config_default(sparta_config *config);

/** Fills `config` with the settings a named preset stands for.
 *  Returns SPARTA_ERROR_INVALID if `preset` is not one of sparta_preset. */
sparta_status sparta_config_preset(sparta_config *config, sparta_preset preset);


/*------------------------------------------------------------------------------
  Step 1: analyze the pattern
------------------------------------------------------------------------------*/

/** A plan for solving matrices with one particular pattern of nonzeros.
 *  Create it with sparta_analyze; release it with sparta_analysis_free. */
typedef struct sparta_analysis sparta_analysis;

/** Plans the computation for a matrix pattern. Runs on the CPU.
 *
 *  @param n          Number of rows, which equals the number of columns.
 *  @param col_ptr    n + 1 column offsets.
 *  @param row_index  col_ptr[n] row indices, ascending within each column.
 *  @param kind       Whether the lower triangle or the whole matrix was given.
 *  @param config     Settings, or NULL for the defaults.
 *  @param out        Receives the new plan. Set to NULL if the call fails, so
 *                    do not point it at a variable already holding a plan.
 *
 *  @return SPARTA_OK, or an error with the reason in sparta_last_error. */
sparta_status sparta_analyze(int64_t n,
                             const int64_t *col_ptr,
                             const int64_t *row_index,
                             sparta_matrix kind,
                             const sparta_config *config,
                             sparta_analysis **out);

/** Returns the number of rows the plan was built for, or 0 if `analysis` is
 *  NULL. */
int64_t sparta_analysis_rows(const sparta_analysis *analysis);

/** Returns the number of nonzeros the plan expects in each `values` array, or 0
 *  if `analysis` is NULL. */
int64_t sparta_analysis_nonzeros(const sparta_analysis *analysis);

/** Returns how many entries the factors hold per nonzero of the matrix. This is
 *  the fill-in: 1.0 would mean the factors are no denser than the matrix, and a
 *  lower number means a better ordering. Returns 0 if `analysis` is NULL. */
double sparta_analysis_fill_ratio(const sparta_analysis *analysis);

/** Returns the GPU memory the factors will occupy, in bytes, or 0 if `analysis`
 *  is NULL. Check this before factoring a matrix that may not fit. */
size_t sparta_analysis_factor_bytes(const sparta_analysis *analysis);

/** Releases a plan. Does nothing if `analysis` is NULL. Not needed after
 *  sparta_solver_create has taken the plan. */
void sparta_analysis_free(sparta_analysis *analysis);


/*------------------------------------------------------------------------------
  Steps 2 and 3: factor and solve
------------------------------------------------------------------------------*/

/** A plan together with the GPU memory it runs in. Create it with
 *  sparta_solver_create; release it with sparta_solver_free. */
typedef struct sparta_solver sparta_solver;

/** Reserves GPU memory for a plan and prepares the GPU kernels.
 *
 *  The solver takes ownership of the plan whether or not the call succeeds, so
 *  `*analysis` is set to NULL and must not be freed or reused afterwards.
 *
 *  @param analysis  The plan to take over. Set to NULL by the call.
 *  @param out       Receives the new solver. Set to NULL if the call fails, so
 *                   do not point it at a variable already holding a solver. */
sparta_status sparta_solver_create(sparta_analysis **analysis,
                                   sparta_solver **out);

/** Factors a complex matrix on the GPU.
 *
 *  Call it again with different numbers to factor another matrix sharing the
 *  analyzed pattern; the plan and the GPU memory are reused.
 *
 *  @param solver  A solver from sparta_solver_create.
 *  @param values  One number per nonzero, in the order sparta_analyze was given
 *                 them. Its length must be sparta_analysis_nonzeros.
 *
 *  @return SPARTA_OK, or SPARTA_ERROR_PIVOT when the matrix is singular or too
 *          badly scaled to factor. Any earlier factors are discarded either
 *          way, so a failed call leaves nothing to solve with. */
sparta_status sparta_factor(sparta_solver *solver, const sparta_complex *values);

/** Factors a real matrix on the GPU. Same as sparta_factor, with one float per
 *  nonzero. For now the GPU runs it through the complex kernels with a zero
 *  imaginary part, so it is as fast as sparta_factor, not faster.
 *
 *  Solve against the result with sparta_solve_real, or with sparta_solve for
 *  a complex right-hand side. */
sparta_status sparta_factor_real(sparta_solver *solver, const float *values);

/** Solves A x = b for a complex right-hand side, using the factors from the
 *  last sparta_factor or sparta_factor_real.
 *
 *  @param solver  A solver whose last factorization succeeded.
 *  @param b       The right-hand side: n numbers.
 *  @param x       Receives the solution: n numbers. May be the same array
 *                 as `b`, which solves in place. */
sparta_status sparta_solve(sparta_solver *solver,
                           const sparta_complex *b,
                           sparta_complex *x);

/** Solves for several right-hand sides at once, which is faster than solving
 *  them one by one.
 *
 *  @param solver  A solver whose last sparta_factor succeeded.
 *  @param b       `count` columns end to end: column j starts at b[j * n].
 *  @param x       Receives the solutions in the same layout. May be `b`.
 *  @param count   How many right-hand sides. At least 1. */
sparta_status sparta_solve_many(sparta_solver *solver,
                                const sparta_complex *b,
                                sparta_complex *x,
                                int64_t count);

/** Solves A x = b for a real right-hand side, using the factors from the last
 *  sparta_factor_real. Returns SPARTA_ERROR_INVALID after sparta_factor: the
 *  solution of a complex matrix is complex.
 *
 *  @param b  n floats.
 *  @param x  Receives n floats. May be the same array as `b`. */
sparta_status sparta_solve_real(sparta_solver *solver, const float *b, float *x);

/** Solves for several real right-hand sides at once; see sparta_solve_many
 *  for the layout and sparta_solve_real for when it applies. */
sparta_status sparta_solve_many_real(sparta_solver *solver,
                                     const float *b,
                                     float *x,
                                     int64_t count);

/** Returns the number of rows the solver was built for, or 0 if `solver` is
 *  NULL. */
int64_t sparta_solver_rows(const sparta_solver *solver);

/** Returns the number of nonzeros each `values` array must hold, or 0 if
 *  `solver` is NULL. */
int64_t sparta_solver_nonzeros(const sparta_solver *solver);

/** Returns how long the GPU spent on the last factorization, in seconds. Queue
 *  waits and host work are not counted. */
double sparta_factor_seconds(const sparta_solver *solver);

/** Returns how many pivots the last factorization had to nudge away from zero
 *  to continue.
 *
 *  Zero is the normal result. Anything else means the factors belong to a
 *  slightly altered matrix, so the answer is worth checking against the
 *  residual, and is better used to drive iterative refinement than trusted
 *  outright. */
uint64_t sparta_perturbed_pivots(const sparta_solver *solver);

/** Releases a solver and the GPU memory it holds. Does nothing if `solver` is
 *  NULL. */
void sparta_solver_free(sparta_solver *solver);


/*------------------------------------------------------------------------------
  Threads

  A solver belongs to the thread that created it, and every call taking it must
  come from that thread. Different solvers are independent and may run on
  different threads at the same time. sparta_analyze is free of this rule and
  may be called from any thread, including several at once.
------------------------------------------------------------------------------*/

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* SPARTA_H */
