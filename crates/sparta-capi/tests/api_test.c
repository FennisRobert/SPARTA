/*==============================================================================

  Checks the C API from C: what it accepts, what it rejects, and whether a
  rejection leaves the handles usable.

  Every check is a line of output. The program exits non-zero if any fails.

  Build and run:  make -C .. test

==============================================================================*/

#include "sparta.h"

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*------------------------------------------------------------------------------
  Checking
------------------------------------------------------------------------------*/

static int failures = 0;

/** Prints a section heading. */
static void section(const char *title) { printf("\n%s\n", title); }

/** Records whether `condition` held, printing the outcome either way. */
static void expect(const char *what, int condition) {
    printf("  %-58s %s\n", what, condition ? "ok" : "FAILED");
    failures += !condition;
}

/** Records whether a call returned the status it should have. A wrong status
 *  prints SPARTA's own explanation, which is usually enough to see why. */
static void expect_status(const char *what, sparta_status got, sparta_status want) {
    const int ok = got == want;
    printf("  %-58s %s\n", what, ok ? "ok" : "FAILED");
    if (!ok) {
        printf("      wanted status %d, got %d: %s\n", want, got, sparta_last_error());
        failures += 1;
    }
}

/*------------------------------------------------------------------------------
  Small matrices to test against
------------------------------------------------------------------------------*/

enum { SMALL = 8 };

/** Lower triangle of a symmetric tridiagonal matrix: 4 + i/10 on the diagonal,
 *  -1 below it. Well conditioned, so a failure to factor it is the solver's. */
typedef struct {
    int64_t col_ptr[SMALL + 1];
    int64_t row_index[2 * SMALL - 1];
    float complex values[2 * SMALL - 1];
} tridiagonal;

static void build_tridiagonal(tridiagonal *a) {
    int64_t at = 0;
    for (int64_t j = 0; j < SMALL; ++j) {
        a->col_ptr[j] = at;
        a->row_index[at] = j;
        a->values[at++] = 4.0f + 0.1f * I;
        if (j + 1 < SMALL) {
            a->row_index[at] = j + 1;
            a->values[at++] = -1.0f;
        }
    }
    a->col_ptr[SMALL] = at;
}

/** Multiplies the tridiagonal by a vector of ones, giving each row's sum. */
static void ones_product(const tridiagonal *a, float complex *b) {
    for (int64_t i = 0; i < SMALL; ++i) {
        b[i] = 0.0f;
    }
    for (int64_t j = 0; j < SMALL; ++j) {
        for (int64_t k = a->col_ptr[j]; k < a->col_ptr[j + 1]; ++k) {
            const int64_t i = a->row_index[k];
            b[i] += a->values[k];
            if (i != j) {
                b[j] += a->values[k];
            }
        }
    }
}

/** Returns the largest distance between any entry of `x` and 1. */
static double error_against_ones(const float complex *x, int64_t rows) {
    double worst = 0.0;
    for (int64_t i = 0; i < rows; ++i) {
        const double off = cabsf(x[i] - 1.0f);
        worst = off > worst ? off : worst;
    }
    return worst;
}

/*------------------------------------------------------------------------------
  The checks
------------------------------------------------------------------------------*/

/** Settings, the preset table, and the struct_size guard. */
static void check_config(void) {
    section("settings");

    sparta_config config;
    sparta_config_default(&config);
    expect("defaults set struct_size", config.struct_size == sizeof config);
    expect("defaults choose SCALPEL ordering", config.ordering == SPARTA_ORDER_SCALPEL);

    expect_status("a named preset is accepted",
                  sparta_config_preset(&config, SPARTA_PRESET_SWEEP), SPARTA_OK);
    expect("the sweep preset keeps SCALPEL", config.ordering == SPARTA_ORDER_SCALPEL);

    expect_status("a preset number out of range is rejected",
                  sparta_config_preset(&config, (sparta_preset)99), SPARTA_ERROR_INVALID);
    expect_status("a null config is rejected",
                  sparta_config_preset(NULL, SPARTA_PRESET_BALANCED), SPARTA_ERROR_INVALID);

    sparta_config_default(&config);
    expect("a null config is survivable", (sparta_config_default(NULL), 1));
}

/** Inputs sparta_analyze must refuse. */
static void check_analyze_rejects(void) {
    section("rejected input");

    tridiagonal a;
    build_tridiagonal(&a);
    sparta_analysis *plan = NULL;

    expect_status("nowhere to put the plan",
                  sparta_analyze(SMALL, a.col_ptr, a.row_index, SPARTA_SYMMETRIC, NULL, NULL),
                  SPARTA_ERROR_INVALID);
    expect_status("a dimension of zero",
                  sparta_analyze(0, a.col_ptr, a.row_index, SPARTA_SYMMETRIC, NULL, &plan),
                  SPARTA_ERROR_INVALID);
    expect_status("a negative dimension",
                  sparta_analyze(-4, a.col_ptr, a.row_index, SPARTA_SYMMETRIC, NULL, &plan),
                  SPARTA_ERROR_INVALID);
    expect_status("a null col_ptr",
                  sparta_analyze(SMALL, NULL, a.row_index, SPARTA_SYMMETRIC, NULL, &plan),
                  SPARTA_ERROR_INVALID);
    expect_status("a null row_index",
                  sparta_analyze(SMALL, a.col_ptr, NULL, SPARTA_SYMMETRIC, NULL, &plan),
                  SPARTA_ERROR_INVALID);
    expect_status("a matrix kind that does not exist",
                  sparta_analyze(SMALL, a.col_ptr, a.row_index, (sparta_matrix)7, NULL, &plan),
                  SPARTA_ERROR_INVALID);
    sparta_config unknown;
    sparta_config_default(&unknown);
    unknown.ordering = (sparta_ordering)5;
    expect_status("an ordering that does not exist",
                  sparta_analyze(SMALL, a.col_ptr, a.row_index, SPARTA_SYMMETRIC, &unknown, &plan),
                  SPARTA_ERROR_INVALID);
    expect("no plan was produced by any of those", plan == NULL);

    /* col_ptr must start at zero and never decrease. */
    tridiagonal bad = a;
    bad.col_ptr[0] = 1;
    expect_status("col_ptr not starting at zero",
                  sparta_analyze(SMALL, bad.col_ptr, bad.row_index, SPARTA_SYMMETRIC, NULL, &plan),
                  SPARTA_ERROR_INVALID);

    bad = a;
    bad.col_ptr[3] = bad.col_ptr[4] + 1;
    expect_status("col_ptr going backwards",
                  sparta_analyze(SMALL, bad.col_ptr, bad.row_index, SPARTA_SYMMETRIC, NULL, &plan),
                  SPARTA_ERROR_INVALID);

    /* The symmetric path takes the lower triangle, so an entry above the
       diagonal is a mistake rather than extra information. */
    bad = a;
    bad.row_index[1] = 0;
    expect_status("an entry above the diagonal in a symmetric matrix",
                  sparta_analyze(SMALL, bad.col_ptr, bad.row_index, SPARTA_SYMMETRIC, NULL, &plan),
                  SPARTA_ERROR_INVALID);

    sparta_config config;
    sparta_config_default(&config);
    config.schur_block = 99;
    expect_status("a GPU setting outside its range",
                  sparta_analyze(SMALL, a.col_ptr, a.row_index, SPARTA_SYMMETRIC, &config, &plan),
                  SPARTA_ERROR_INVALID);
    expect("the message names the setting", strstr(sparta_last_error(), "schur_block") != NULL);

    sparta_config_default(&config);
    config.hybrid_cpu = 1.5;
    expect_status("a hybrid share of 1 or more",
                  sparta_analyze(SMALL, a.col_ptr, a.row_index, SPARTA_SYMMETRIC, &config, &plan),
                  SPARTA_ERROR_INVALID);

    sparta_config_default(&config);
    config.threads = 5000;
    expect_status("more threads than any machine has",
                  sparta_analyze(SMALL, a.col_ptr, a.row_index, SPARTA_SYMMETRIC, &config, &plan),
                  SPARTA_ERROR_INVALID);
    expect("the message names the setting", strstr(sparta_last_error(), "threads") != NULL);

    sparta_config_default(&config);
    config.struct_size = sizeof config + 8;
    expect_status("a config from a newer header than this library",
                  sparta_analyze(SMALL, a.col_ptr, a.row_index, SPARTA_SYMMETRIC, &config, &plan),
                  SPARTA_ERROR_INVALID);

    sparta_config_default(&config);
    config.struct_size = 0;
    expect_status("a config with no declared size",
                  sparta_analyze(SMALL, a.col_ptr, a.row_index, SPARTA_SYMMETRIC, &config, &plan),
                  SPARTA_ERROR_INVALID);
}

/** Reading a plan, and handing it over exactly once. */
static void check_handles(void) {
    section("handles");

    tridiagonal a;
    build_tridiagonal(&a);

    sparta_analysis *plan = NULL;
    expect_status("a well-formed matrix is analyzed",
                  sparta_analyze(SMALL, a.col_ptr, a.row_index, SPARTA_SYMMETRIC, NULL, &plan),
                  SPARTA_OK);
    expect("the plan reports its rows", sparta_analysis_rows(plan) == SMALL);
    expect("the plan reports its nonzeros",
           sparta_analysis_nonzeros(plan) == 2 * SMALL - 1);
    expect("the plan reports a fill ratio", sparta_analysis_fill_ratio(plan) >= 1.0);
    expect("the plan reports its GPU memory", sparta_analysis_factor_bytes(plan) > 0);

    expect("reading a null plan gives zero, not a crash",
           sparta_analysis_rows(NULL) == 0 && sparta_analysis_nonzeros(NULL) == 0 &&
               sparta_analysis_fill_ratio(NULL) == 0.0 &&
               sparta_analysis_factor_bytes(NULL) == 0);

    sparta_solver *solver = NULL;
    expect_status("the plan is handed to the GPU", sparta_solver_create(&plan, &solver), SPARTA_OK);
    expect("the handover cleared the caller's handle", plan == NULL);

    /* A second handover has nothing to take. It is given its own `again` to
       write into, because a failed call clears the handle it was pointing at
       and would otherwise lose the solver that succeeded. */
    sparta_solver *again = NULL;
    expect_status("handing over the same plan twice is refused",
                  sparta_solver_create(&plan, &again), SPARTA_ERROR_INVALID);
    expect("the refused call produced no solver", again == NULL);

    expect("freeing a null plan is survivable", (sparta_analysis_free(NULL), 1));
    expect("freeing a null solver is survivable", (sparta_solver_free(NULL), 1));
    sparta_solver_free(solver);
}

/** Factoring and solving, including the order the two must happen in. */
static void check_solve(void) {
    section("factor and solve");

    tridiagonal a;
    build_tridiagonal(&a);
    float complex b[SMALL];
    float complex x[SMALL];
    ones_product(&a, b);

    sparta_analysis *plan = NULL;
    sparta_solver *solver = NULL;
    sparta_analyze(SMALL, a.col_ptr, a.row_index, SPARTA_SYMMETRIC, NULL, &plan);
    sparta_solver_create(&plan, &solver);

    expect_status("solving before factoring is refused",
                  sparta_solve(solver, (const sparta_complex *)b, (sparta_complex *)x),
                  SPARTA_ERROR_INVALID);
    expect_status("factoring without values is refused",
                  sparta_factor(solver, NULL), SPARTA_ERROR_INVALID);
    expect_status("factoring a null solver is refused",
                  sparta_factor(NULL, (const sparta_complex *)a.values), SPARTA_ERROR_INVALID);

    expect_status("the matrix factors",
                  sparta_factor(solver, (const sparta_complex *)a.values), SPARTA_OK);
    expect("no pivot needed perturbing", sparta_perturbed_pivots(solver) == 0);
    expect("the GPU time was recorded", sparta_factor_seconds(solver) > 0.0);

    expect_status("solving without room for the answer is refused",
                  sparta_solve(solver, (const sparta_complex *)b, NULL), SPARTA_ERROR_INVALID);
    expect_status("asking for zero right-hand sides is refused",
                  sparta_solve_many(solver, (const sparta_complex *)b, (sparta_complex *)x, 0),
                  SPARTA_ERROR_INVALID);

    expect_status("the system solves",
                  sparta_solve(solver, (const sparta_complex *)b, (sparta_complex *)x), SPARTA_OK);
    expect("the answer is a vector of ones", error_against_ones(x, SMALL) < 1.0e-4);

    /* The header promises x may be b, which solves in place. */
    float complex in_place[SMALL];
    memcpy(in_place, b, sizeof in_place);
    expect_status("solving in place",
                  sparta_solve(solver, (const sparta_complex *)in_place,
                               (sparta_complex *)in_place),
                  SPARTA_OK);
    expect("in place gives the same answer", error_against_ones(in_place, SMALL) < 1.0e-4);

    sparta_solver_free(solver);

    /* The same system with the CPU taking part of the work, on two threads. */
    sparta_config config;
    sparta_config_default(&config);
    config.hybrid_cpu = 0.3;
    config.threads = 2;
    sparta_analyze(SMALL, a.col_ptr, a.row_index, SPARTA_SYMMETRIC, &config, &plan);
    expect_status("a hybrid solver is created", sparta_solver_create(&plan, &solver), SPARTA_OK);
    expect_status("it factors", sparta_factor(solver, (const sparta_complex *)a.values), SPARTA_OK);
    expect_status("it solves", sparta_solve(solver, (const sparta_complex *)b, (sparta_complex *)x),
                  SPARTA_OK);
    expect("its answer is a vector of ones", error_against_ones(x, SMALL) < 1.0e-4);
    sparta_solver_free(solver);
}

/** A real matrix factored with real arithmetic, and which right-hand sides
 *  each kind of factors accepts. */
static void check_real(void) {
    section("a real matrix");

    /* The tridiagonal with 4 + i/10 on the diagonal, made real: 4 - 1.5 on the
       diagonal, which is indefinite, and -1 below it. */
    tridiagonal a;
    build_tridiagonal(&a);
    float values[2 * SMALL - 1];
    for (int k = 0; k < 2 * SMALL - 1; ++k) {
        values[k] = crealf(a.values[k]) == 4.0f ? 2.5f : crealf(a.values[k]);
    }
    /* Row sums of the real matrix: A times a vector of ones. */
    float b[SMALL];
    float x[SMALL];
    for (int64_t i = 0; i < SMALL; ++i) {
        b[i] = 2.5f - (i > 0) - (i + 1 < SMALL);
    }

    sparta_analysis *plan = NULL;
    sparta_solver *solver = NULL;
    sparta_analyze(SMALL, a.col_ptr, a.row_index, SPARTA_SYMMETRIC, NULL, &plan);
    sparta_solver_create(&plan, &solver);

    expect_status("the real matrix factors", sparta_factor_real(solver, values), SPARTA_OK);
    expect_status("a real right-hand side solves", sparta_solve_real(solver, b, x), SPARTA_OK);
    float worst = 0.0f;
    for (int64_t i = 0; i < SMALL; ++i) {
        worst = fmaxf(worst, fabsf(x[i] - 1.0f));
    }
    expect("the answer is a vector of ones", worst < 1.0e-4f);

    /* A complex right-hand side against real factors: ones times (1 + 2i). */
    float complex z[SMALL];
    float complex y[SMALL];
    for (int64_t i = 0; i < SMALL; ++i) {
        z[i] = b[i] * (1.0f + 2.0f * I);
    }
    expect_status("a complex right-hand side solves too",
                  sparta_solve(solver, (const sparta_complex *)z, (sparta_complex *)y), SPARTA_OK);
    float far = 0.0f;
    for (int64_t i = 0; i < SMALL; ++i) {
        far = fmaxf(far, cabsf(y[i] - (1.0f + 2.0f * I)));
    }
    expect("its answer is ones times 1 + 2i", far < 1.0e-4f);

    expect_status("complex factors refuse a real right-hand side",
                  (sparta_factor(solver, (const sparta_complex *)a.values),
                   sparta_solve_real(solver, b, x)),
                  SPARTA_ERROR_INVALID);
    sparta_solver_free(solver);
}

/** A matrix that cannot be factored, and what the caller is told about it. */
static void check_singular(void) {
    section("a singular matrix");

    /* Diagonal, with a zero in the third column. */
    const int64_t col_ptr[5] = {0, 1, 2, 3, 4};
    const int64_t row_index[4] = {0, 1, 2, 3};
    const float complex values[4] = {1.0f, 1.0f, 0.0f, 1.0f};

    sparta_analysis *plan = NULL;
    sparta_solver *solver = NULL;
    expect_status("the pattern analyzes, since nothing is zero structurally",
                  sparta_analyze(4, col_ptr, row_index, SPARTA_SYMMETRIC, NULL, &plan), SPARTA_OK);
    expect_status("and reaches the GPU", sparta_solver_create(&plan, &solver), SPARTA_OK);

    expect_status("factoring reports a pivot failure",
                  sparta_factor(solver, (const sparta_complex *)values), SPARTA_ERROR_PIVOT);
    expect("the message says which column broke down",
           strstr(sparta_last_error(), "column") != NULL);

    float complex b[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float complex x[4];
    expect_status("and there is nothing to solve with afterwards",
                  sparta_solve(solver, (const sparta_complex *)b, (sparta_complex *)x),
                  SPARTA_ERROR_INVALID);

    sparta_solver_free(solver);
}

/** The unsymmetric path, which needs the whole matrix rather than a triangle. */
static void check_general(void) {
    section("a general matrix");

    /* 4 on the diagonal, -1 below it, -2 above it: unsymmetric by design. */
    enum { N = 6 };
    int64_t col_ptr[N + 1];
    int64_t row_index[3 * N];
    float complex values[3 * N];

    int64_t at = 0;
    for (int64_t j = 0; j < N; ++j) {
        col_ptr[j] = at;
        if (j > 0) {
            row_index[at] = j - 1;
            values[at++] = -2.0f;
        }
        row_index[at] = j;
        values[at++] = 4.0f + 0.1f * I;
        if (j + 1 < N) {
            row_index[at] = j + 1;
            values[at++] = -1.0f;
        }
    }
    col_ptr[N] = at;

    /* Row sums, so the answer is again a vector of ones. */
    float complex b[N] = {0};
    for (int64_t j = 0; j < N; ++j) {
        for (int64_t k = col_ptr[j]; k < col_ptr[j + 1]; ++k) {
            b[row_index[k]] += values[k];
        }
    }

    sparta_analysis *plan = NULL;
    sparta_solver *solver = NULL;
    float complex x[N];

    expect_status("the whole matrix analyzes on the general path",
                  sparta_analyze(N, col_ptr, row_index, SPARTA_GENERAL, NULL, &plan), SPARTA_OK);
    expect_status("it reaches the GPU", sparta_solver_create(&plan, &solver), SPARTA_OK);
    expect_status("it factors", sparta_factor(solver, (const sparta_complex *)values), SPARTA_OK);
    expect_status("it solves",
                  sparta_solve(solver, (const sparta_complex *)b, (sparta_complex *)x), SPARTA_OK);
    expect("the answer is a vector of ones", error_against_ones(x, N) < 1.0e-4);

    sparta_solver_free(solver);
}

int main(void) {
    printf("SPARTA %s\n", sparta_version());
    if (!sparta_gpu_available()) {
        fprintf(stderr, "no Metal GPU is visible to this process\n");
        return EXIT_FAILURE;
    }

    check_config();
    check_analyze_rejects();
    check_handles();
    check_solve();
    check_real();
    check_singular();
    check_general();

    printf("\n%s\n", failures ? "FAILURES" : "all checks passed");
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
