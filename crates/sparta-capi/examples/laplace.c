/*==============================================================================

  Solving a system with SPARTA from C.

  The system is a 2D Laplacian on a square grid with a small imaginary shift,
  which is complex-symmetric and so takes the SPARTA_SYMMETRIC path. The
  right-hand side is chosen so that the exact answer is a vector of ones, which
  makes the error easy to read.

  Build and run:  make -C .. run-c

==============================================================================*/

#include "sparta.h"

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/*------------------------------------------------------------------------------
  A test matrix
------------------------------------------------------------------------------*/

/** A square matrix as SPARTA wants it: compressed sparse column, and for the
 *  symmetric path holding the lower triangle only. */
typedef struct {
    int64_t rows;
    int64_t nonzeros;
    int64_t *col_ptr;
    int64_t *row_index;
    float complex *values;
} matrix;

/** Builds the lower triangle of a 2D Laplacian on a `side` by `side` grid.
 *
 *  Grid point (x, y) is unknown number y * side + x, and is coupled to its
 *  neighbours to the right and below. Column j therefore holds the diagonal
 *  first, then j + 1, then j + side, which is already ascending order.
 *
 *  @param side   Grid points along one edge.
 *  @param shift  Imaginary part added to the diagonal. Makes the matrix
 *                complex-symmetric rather than merely real symmetric.
 *  @return       The matrix. Release it with matrix_free. */
static matrix laplacian(int side, float shift) {
    const int64_t rows = (int64_t)side * side;
    const int64_t nonzeros = rows + 2 * (int64_t)side * (side - 1);

    matrix a;
    a.rows = rows;
    a.nonzeros = nonzeros;
    a.col_ptr = malloc((rows + 1) * sizeof *a.col_ptr);
    a.row_index = malloc(nonzeros * sizeof *a.row_index);
    a.values = malloc(nonzeros * sizeof *a.values);

    int64_t at = 0;
    for (int64_t j = 0; j < rows; ++j) {
        const int64_t x = j % side;
        const int64_t y = j / side;
        a.col_ptr[j] = at;

        a.row_index[at] = j;
        a.values[at++] = 4.0f + shift * I;

        if (x + 1 < side) {
            a.row_index[at] = j + 1;
            a.values[at++] = -1.0f;
        }
        if (y + 1 < side) {
            a.row_index[at] = j + side;
            a.values[at++] = -1.0f;
        }
    }
    a.col_ptr[rows] = at;
    return a;
}

static void matrix_free(matrix *a) {
    free(a->col_ptr);
    free(a->row_index);
    free(a->values);
}

/** Computes y = A x, reading the stored lower triangle as a symmetric matrix.
 *  Every off-diagonal entry is used twice: once where it is stored, once
 *  mirrored across the diagonal. */
static void multiply(const matrix *a, const float complex *x, float complex *y) {
    for (int64_t i = 0; i < a->rows; ++i) {
        y[i] = 0.0f;
    }
    for (int64_t j = 0; j < a->rows; ++j) {
        for (int64_t k = a->col_ptr[j]; k < a->col_ptr[j + 1]; ++k) {
            const int64_t i = a->row_index[k];
            y[i] += a->values[k] * x[j];
            if (i != j) {
                y[j] += a->values[k] * x[i];
            }
        }
    }
}

/*------------------------------------------------------------------------------
  Measuring the answer
------------------------------------------------------------------------------*/

/** Returns the largest distance between any entry of `x` and 1. */
static double error_against_ones(const float complex *x, int64_t rows) {
    double worst = 0.0;
    for (int64_t i = 0; i < rows; ++i) {
        const double off = cabsf(x[i] - 1.0f);
        if (off > worst) {
            worst = off;
        }
    }
    return worst;
}

/** Returns ||b - A x|| / ||b||, the relative residual, computed in double. */
static double residual(const matrix *a, const float complex *x, const float complex *b) {
    float complex *ax = malloc(a->rows * sizeof *ax);
    multiply(a, x, ax);

    double left = 0.0;
    double right = 0.0;
    for (int64_t i = 0; i < a->rows; ++i) {
        left += (double)cabsf(b[i] - ax[i]) * cabsf(b[i] - ax[i]);
        right += (double)cabsf(b[i]) * cabsf(b[i]);
    }
    free(ax);
    return sqrt(left) / sqrt(right);
}

/*------------------------------------------------------------------------------
  Running the solver
------------------------------------------------------------------------------*/

/** Stops the program with SPARTA's own explanation when a call fails. */
static void check(const char *step, sparta_status status) {
    if (status != SPARTA_OK) {
        fprintf(stderr, "%s failed: %s\n", step, sparta_last_error());
        exit(EXIT_FAILURE);
    }
}

int main(void) {
    printf("SPARTA %s\n", sparta_version());
    if (!sparta_gpu_available()) {
        fprintf(stderr, "no Metal GPU is visible to this process\n");
        return EXIT_FAILURE;
    }

    const int side = 64;
    matrix a = laplacian(side, 0.05f);
    printf("%lld unknowns, %lld nonzeros in the lower triangle\n\n",
           (long long)a.rows, (long long)a.nonzeros);

    /* A right-hand side whose exact answer is a vector of ones. */
    float complex *want = malloc(a.rows * sizeof *want);
    float complex *b = malloc(a.rows * sizeof *b);
    float complex *x = malloc(a.rows * sizeof *x);
    for (int64_t i = 0; i < a.rows; ++i) {
        want[i] = 1.0f;
    }
    multiply(&a, want, b);

    /* Step 1: plan the computation from the pattern alone. */
    sparta_config config;
    sparta_config_default(&config);
    check("preset", sparta_config_preset(&config, SPARTA_PRESET_STRUCTURED_GRID));

    sparta_analysis *plan = NULL;
    check("analyze", sparta_analyze(a.rows, a.col_ptr, a.row_index, SPARTA_SYMMETRIC, &config, &plan));
    printf("analyze   fill ratio %.2f, factors need %.1f MB on the GPU\n",
           sparta_analysis_fill_ratio(plan),
           sparta_analysis_factor_bytes(plan) / 1.0e6);

    /* Step 2: hand the plan to the GPU and factor. */
    sparta_solver *solver = NULL;
    check("reserve GPU memory", sparta_solver_create(&plan, &solver));
    check("factor", sparta_factor(solver, (const sparta_complex *)a.values));
    printf("factor    %.4f s on the GPU, %llu pivots perturbed\n",
           sparta_factor_seconds(solver),
           (unsigned long long)sparta_perturbed_pivots(solver));

    /* Step 3: solve. */
    check("solve", sparta_solve(solver, (const sparta_complex *)b, (sparta_complex *)x));
    printf("solve     largest error %.2e, relative residual %.2e\n\n",
           error_against_ones(x, a.rows), residual(&a, x, b));

    /* The plan and the GPU memory are reused for a second matrix with the same
       pattern: only the diagonal shift changes. This is the frequency-sweep
       case, and it skips the analysis entirely. */
    for (int64_t j = 0; j < a.rows; ++j) {
        a.values[a.col_ptr[j]] = 4.0f + 0.25f * I;
    }
    multiply(&a, want, b);
    check("refactor", sparta_factor(solver, (const sparta_complex *)a.values));
    check("solve again", sparta_solve(solver, (const sparta_complex *)b, (sparta_complex *)x));
    printf("reuse     refactored in %.4f s, largest error %.2e\n",
           sparta_factor_seconds(solver), error_against_ones(x, a.rows));

    /* Several right-hand sides in one call, held end to end. */
    const int64_t count = 4;
    float complex *many = malloc(a.rows * count * sizeof *many);
    float complex *answers = malloc(a.rows * count * sizeof *answers);
    for (int64_t j = 0; j < count; ++j) {
        for (int64_t i = 0; i < a.rows; ++i) {
            want[i] = (float)(j + 1);
        }
        multiply(&a, want, many + j * a.rows);
    }
    check("solve many", sparta_solve_many(solver, (const sparta_complex *)many,
                                          (sparta_complex *)answers, count));

    double worst = 0.0;
    for (int64_t j = 0; j < count; ++j) {
        for (int64_t i = 0; i < a.rows; ++i) {
            const double off = cabsf(answers[j * a.rows + i] - (float)(j + 1)) / (j + 1);
            if (off > worst) {
                worst = off;
            }
        }
    }
    printf("batch     %lld right-hand sides at once, largest relative error %.2e\n",
           (long long)count, worst);

    sparta_solver_free(solver);
    matrix_free(&a);
    free(want);
    free(b);
    free(x);
    free(many);
    free(answers);
    return EXIT_SUCCESS;
}
