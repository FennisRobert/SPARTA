/*==============================================================================

  sparta_mex  --  the MATLAB gateway to the SPARTA C library.

  Called by SpartaSolver.m, not by users. Every call names a command first:

      id   = sparta_mex('create', A, kind, settings)  plan A's pattern, reserve GPU memory
      [t, p] = sparta_mex('factor', id, A)             factor A's values on the GPU
      X    = sparta_mex('solve', id, B)                solve A X = B, one column per B column
      info = sparta_mex('info', id)                    sizes and statistics, as a struct
             sparta_mex('free', id)                    release the solver
      s    = sparta_mex('version' | 'device')          library version, GPU backend name
      tf   = sparta_mex('gpu_available')               whether this machine has a usable GPU

  A is a square sparse double matrix holding exactly the entries to factor: for
  'symmetric' that is the lower triangle. `settings` is a struct whose fields
  are sparta_config field names, plus an optional 'preset'.

  Solvers live in a table inside this file and are named by a positive integer.
  The file locks itself in memory while any solver exists, so `clear mex`
  cannot drop the table out from under live GPU memory, and releases them all
  when MATLAB exits.

  Builds against either complex layout: interleaved (mex -R2018a) or separate
  real and imaginary arrays (the older default, and GNU Octave's).

==============================================================================*/

#include "mex.h"
#include "sparta.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if defined(MX_HAS_INTERLEAVED_COMPLEX) && MX_HAS_INTERLEAVED_COMPLEX
#define INTERLEAVED 1
#else
#define INTERLEAVED 0
#endif

/*------------------------------------------------------------------------------
  The solver table
------------------------------------------------------------------------------*/

/** One live solver, with the pattern it was planned for. */
typedef struct {
    sparta_solver *solver;      /* NULL marks a free slot */
    int64_t rows;
    int64_t nonzeros;
    int64_t *col_ptr;           /* the analyzed pattern, to place new values in */
    int64_t *row_index;
    sparta_complex *values;     /* scratch: one value per analyzed nonzero */
    double fill_ratio;
    double factor_bytes;
    int factored;               /* 1 after a successful factor */
} slot;

static slot *table = NULL;
static size_t table_size = 0;
static size_t live = 0;

static void release(slot *s) {
    sparta_solver_free(s->solver);
    mxFree(s->col_ptr);
    mxFree(s->row_index);
    mxFree(s->values);
    memset(s, 0, sizeof *s);
}

/** Frees every solver. Registered with mexAtExit. */
static void release_all(void) {
    for (size_t i = 0; i < table_size; ++i) {
        if (table[i].solver != NULL) {
            release(&table[i]);
        }
    }
    mxFree(table);
    table = NULL;
    table_size = 0;
    live = 0;
}

/** Returns a free slot and its id, growing the table when it is full. */
static slot *free_slot(double *id) {
    for (size_t i = 0; i < table_size; ++i) {
        if (table[i].solver == NULL) {
            *id = (double)(i + 1);
            return &table[i];
        }
    }
    const size_t grown = table_size == 0 ? 8 : 2 * table_size;
    table = mxRealloc(table, grown * sizeof *table);
    mexMakeMemoryPersistent(table);
    memset(table + table_size, 0, (grown - table_size) * sizeof *table);
    *id = (double)(table_size + 1);
    table_size = grown;
    return &table[(size_t)*id - 1];
}

/** Returns the live solver an id names, or stops with an error. */
static slot *lookup(const mxArray *id) {
    if (!mxIsDouble(id) || mxIsComplex(id) || mxGetNumberOfElements(id) != 1) {
        mexErrMsgIdAndTxt("sparta:handle", "the solver id must be a real scalar");
    }
    const double value = mxGetScalar(id);
    const size_t index = (size_t)value;
    if (value < 1 || (double)index != value || index > table_size || table[index - 1].solver == NULL) {
        mexErrMsgIdAndTxt("sparta:handle", "this SpartaSolver has been released; create a new one");
    }
    return &table[index - 1];
}

/*------------------------------------------------------------------------------
  Errors
------------------------------------------------------------------------------*/

/** Stops with SPARTA's own explanation when a call failed. The error id names
 *  the kind of failure, for example sparta:pivot. */
static void check(const char *step, sparta_status status) {
    static const char *const names[] = {"ok", "invalid", "unsupported", "device", "pivot", "internal"};
    if (status == SPARTA_OK) {
        return;
    }
    char id[32] = "sparta:internal";
    if ((unsigned)status < sizeof names / sizeof *names) {
        snprintf(id, sizeof id, "sparta:%s", names[status]);
    }
    mexErrMsgIdAndTxt(id, "%s failed: %s", step, sparta_last_error());
}

/*------------------------------------------------------------------------------
  Moving numbers between MATLAB and SPARTA
------------------------------------------------------------------------------*/

/** Copies `count` numbers of a real or complex double array into single
 *  precision complex, the only precision the GPU computes in. */
static void read_values(const mxArray *a, sparta_complex *out, size_t count) {
#if INTERLEAVED
    if (mxIsComplex(a)) {
        const mxComplexDouble *z = mxGetComplexDoubles(a);
        for (size_t i = 0; i < count; ++i) {
            out[i].re = (float)z[i].real;
            out[i].im = (float)z[i].imag;
        }
        return;
    }
    const mxDouble *re = mxGetDoubles(a);
    for (size_t i = 0; i < count; ++i) {
        out[i].re = (float)re[i];
        out[i].im = 0.0f;
    }
#else
    const double *re = mxGetPr(a);
    const double *im = mxIsComplex(a) ? mxGetPi(a) : NULL;
    for (size_t i = 0; i < count; ++i) {
        out[i].re = (float)re[i];
        out[i].im = im != NULL ? (float)im[i] : 0.0f;
    }
#endif
}

/** Returns a new complex double rows-by-columns array holding `data`. */
static mxArray *write_values(const sparta_complex *data, size_t rows, size_t columns) {
    mxArray *a = mxCreateDoubleMatrix(rows, columns, mxCOMPLEX);
    const size_t count = rows * columns;
#if INTERLEAVED
    mxComplexDouble *z = mxGetComplexDoubles(a);
    for (size_t i = 0; i < count; ++i) {
        z[i].real = data[i].re;
        z[i].imag = data[i].im;
    }
#else
    double *re = mxGetPr(a);
    double *im = mxGetPi(a);
    for (size_t i = 0; i < count; ++i) {
        re[i] = data[i].re;
        im[i] = data[i].im;
    }
#endif
    return a;
}

/** Stops unless `a` is a square sparse double matrix, `rows` by `rows` when
 *  `rows` is not negative. */
static void check_matrix(const mxArray *a, int64_t rows) {
    if (!mxIsSparse(a) || !mxIsDouble(a)) {
        mexErrMsgIdAndTxt("sparta:invalid", "the matrix must be sparse and double");
    }
    const size_t m = mxGetM(a);
    if (m != mxGetN(a) || m == 0) {
        mexErrMsgIdAndTxt("sparta:invalid", "the matrix must be square and nonempty");
    }
    if (rows >= 0 && (int64_t)m != rows) {
        mexErrMsgIdAndTxt("sparta:invalid", "the matrix has %zu rows; this solver was made for %lld",
                          m, (long long)rows);
    }
}

/** Places the values of `a` into the analyzed pattern of `s`, in s->values.
 *  Entries of the pattern that `a` lacks become zero: MATLAB drops entries
 *  that compute to exactly zero, so a later matrix of a sweep can hold fewer.
 *  Returns 0 when `a` holds an entry outside the pattern. */
static int place_values(slot *s, const mxArray *a) {
    const mwIndex *jc = mxGetJc(a);
    const mwIndex *ir = mxGetIr(a);
    const size_t count = jc[s->rows];
    sparta_complex *given = mxMalloc((count > 0 ? count : 1) * sizeof *given);
    read_values(a, given, count);

    const sparta_complex zero = {0.0f, 0.0f};
    for (int64_t j = 0; j < s->rows; ++j) {
        int64_t p = s->col_ptr[j];
        const int64_t end = s->col_ptr[j + 1];
        for (mwIndex q = jc[j]; q < jc[j + 1]; ++q) {
            while (p < end && s->row_index[p] < (int64_t)ir[q]) {
                s->values[p++] = zero;
            }
            if (p == end || s->row_index[p] != (int64_t)ir[q]) {
                mxFree(given);
                return 0;
            }
            s->values[p++] = given[q];
        }
        while (p < end) {
            s->values[p++] = zero;
        }
    }
    mxFree(given);
    return 1;
}

/*------------------------------------------------------------------------------
  Settings
------------------------------------------------------------------------------*/

typedef enum { SIZE, U32, F64 } field_type;

/** A sparta_config field a settings struct may name. */
typedef struct {
    const char *name;
    size_t offset;
    field_type type;
} field;

#define FIELD(name, type) {#name, offsetof(sparta_config, name), type}
static const field fields[] = {
    FIELD(nemin, SIZE),           FIELD(max_front, SIZE),
    FIELD(imbalance, F64),        FIELD(wide_cycles, U32),
    FIELD(fm_passes, U32),        FIELD(separator_trials, U32),
    FIELD(lu_diag_threshold, F64),
    FIELD(small_front_rows, U32), FIELD(schur_batch, U32),
    FIELD(panel_threads, U32),    FIELD(solve_threads, U32),
    FIELD(panel, U32),            FIELD(schur_block, U32),
    FIELD(strip_block, U32),      FIELD(narrow_cols, U32),
    FIELD(pivot_threads, U32),    FIELD(schur_chunk, U32),
    FIELD(hybrid_cpu, F64),
};
#undef FIELD

static const char *const presets[] = {"balanced", "sweep", "one_shot", "structured_grid", "careful"};

/** Returns the text of a char row vector, in memory that MATLAB frees. */
static char *text(const mxArray *a, const char *what) {
    char *s = mxIsChar(a) ? mxArrayToString(a) : NULL;
    if (s == NULL) {
        mexErrMsgIdAndTxt("sparta:invalid", "%s must be text", what);
    }
    return s;
}

/** Stores one numeric setting, checking that it fits the field. */
static void set_field(sparta_config *config, const field *f, const mxArray *value) {
    if (!mxIsNumeric(value) || mxIsComplex(value) || mxGetNumberOfElements(value) != 1) {
        mexErrMsgIdAndTxt("sparta:invalid", "setting '%s' must be a real number", f->name);
    }
    const double v = mxGetScalar(value);
    char *at = (char *)config + f->offset;
    if (f->type == F64) {
        *(double *)at = v;
        return;
    }
    const double limit = f->type == U32 ? 4294967295.0 : 9007199254740992.0;
    if (!(v >= 0 && v <= limit) || v != (double)(uint64_t)v) {
        mexErrMsgIdAndTxt("sparta:invalid", "setting '%s' must be a whole number of at least 0", f->name);
    }
    if (f->type == U32) {
        *(uint32_t *)at = (uint32_t)v;
    } else {
        *(size_t *)at = (size_t)v;
    }
}

/** Builds a sparta_config from a struct of settings: the preset first, if
 *  named, then each field on top of it. sparta_analyze checks the ranges. */
static void read_config(const mxArray *settings, sparta_config *config) {
    sparta_config_default(config);
    if (mxIsEmpty(settings)) {
        return;
    }
    if (!mxIsStruct(settings) || mxGetNumberOfElements(settings) != 1) {
        mexErrMsgIdAndTxt("sparta:invalid", "settings must be a scalar struct");
    }
    const mxArray *preset = mxGetField(settings, 0, "preset");
    if (preset != NULL) {
        const char *name = text(preset, "preset");
        size_t p = 0;
        while (p < sizeof presets / sizeof *presets && strcmp(name, presets[p]) != 0) {
            ++p;
        }
        if (p == sizeof presets / sizeof *presets) {
            mexErrMsgIdAndTxt("sparta:invalid", "unknown preset '%s'; use balanced, sweep, "
                              "one_shot, structured_grid or careful", name);
        }
        check("preset", sparta_config_preset(config, (sparta_preset)p));
    }
    const int count = mxGetNumberOfFields(settings);
    for (int i = 0; i < count; ++i) {
        const char *name = mxGetFieldNameByNumber(settings, i);
        if (strcmp(name, "preset") == 0) {
            continue;
        }
        size_t f = 0;
        while (f < sizeof fields / sizeof *fields && strcmp(name, fields[f].name) != 0) {
            ++f;
        }
        if (f == sizeof fields / sizeof *fields) {
            mexErrMsgIdAndTxt("sparta:invalid", "unknown setting '%s'", name);
        }
        set_field(config, &fields[f], mxGetFieldByNumber(settings, 0, i));
    }
}

/*------------------------------------------------------------------------------
  Commands
------------------------------------------------------------------------------*/

/** id = sparta_mex('create', A, kind, settings) */
static void create(mxArray *out[], int in_count, const mxArray *in[]) {
    if (in_count != 4) {
        mexErrMsgIdAndTxt("sparta:invalid", "create takes a matrix, a kind and settings");
    }
    const mxArray *a = in[1];
    check_matrix(a, -1);
    const char *kind_name = text(in[2], "kind");
    sparta_matrix kind;
    if (strcmp(kind_name, "symmetric") == 0) {
        kind = SPARTA_SYMMETRIC;
    } else if (strcmp(kind_name, "general") == 0) {
        kind = SPARTA_GENERAL;
    } else {
        mexErrMsgIdAndTxt("sparta:invalid", "kind must be 'symmetric' or 'general'");
    }
    sparta_config config;
    read_config(in[3], &config);

    const int64_t rows = (int64_t)mxGetM(a);
    const mwIndex *jc = mxGetJc(a);
    const mwIndex *ir = mxGetIr(a);
    const int64_t nonzeros = (int64_t)jc[rows];
    int64_t *col_ptr = mxMalloc((size_t)(rows + 1) * sizeof *col_ptr);
    int64_t *row_index = mxMalloc((size_t)(nonzeros > 0 ? nonzeros : 1) * sizeof *row_index);
    for (int64_t j = 0; j <= rows; ++j) {
        col_ptr[j] = (int64_t)jc[j];
    }
    for (int64_t k = 0; k < nonzeros; ++k) {
        row_index[k] = (int64_t)ir[k];
    }

    sparta_analysis *plan = NULL;
    check("analyze", sparta_analyze(rows, col_ptr, row_index, kind, &config, &plan));
    const double fill_ratio = sparta_analysis_fill_ratio(plan);
    const double factor_bytes = (double)sparta_analysis_factor_bytes(plan);
    sparta_solver *solver = NULL;
    check("reserve GPU memory", sparta_solver_create(&plan, &solver));

    double id;
    slot *s = free_slot(&id);
    s->solver = solver;
    s->rows = rows;
    s->nonzeros = nonzeros;
    s->col_ptr = col_ptr;
    s->row_index = row_index;
    s->values = mxMalloc((size_t)(nonzeros > 0 ? nonzeros : 1) * sizeof *s->values);
    s->fill_ratio = fill_ratio;
    s->factor_bytes = factor_bytes;
    mexMakeMemoryPersistent(col_ptr);
    mexMakeMemoryPersistent(row_index);
    mexMakeMemoryPersistent(s->values);
    if (live++ == 0) {
        mexLock();
    }
    out[0] = mxCreateDoubleScalar(id);
}

/** [seconds, perturbed] = sparta_mex('factor', id, A) */
static void factor(mxArray *out[], int in_count, const mxArray *in[]) {
    if (in_count != 3) {
        mexErrMsgIdAndTxt("sparta:invalid", "factor takes a solver id and a matrix");
    }
    slot *s = lookup(in[1]);
    check_matrix(in[2], s->rows);
    if (!place_values(s, in[2])) {
        mexErrMsgIdAndTxt("sparta:pattern", "the matrix has an entry the analyzed pattern lacks");
    }
    s->factored = 0;
    check("factor", sparta_factor(s->solver, s->values));
    s->factored = 1;
    out[0] = mxCreateDoubleScalar(sparta_factor_seconds(s->solver));
    out[1] = mxCreateDoubleScalar((double)sparta_perturbed_pivots(s->solver));
}

/** X = sparta_mex('solve', id, B) */
static void solve(mxArray *out[], int in_count, const mxArray *in[]) {
    if (in_count != 3) {
        mexErrMsgIdAndTxt("sparta:invalid", "solve takes a solver id and right-hand sides");
    }
    slot *s = lookup(in[1]);
    if (!s->factored) {
        mexErrMsgIdAndTxt("sparta:invalid", "the last factorization failed; factor again before solving");
    }
    const mxArray *b = in[2];
    if (!mxIsDouble(b) || mxIsSparse(b) || mxGetNumberOfDimensions(b) != 2 || (int64_t)mxGetM(b) != s->rows) {
        mexErrMsgIdAndTxt("sparta:invalid", "the right-hand side must be a full double matrix with %lld rows",
                          (long long)s->rows);
    }
    const size_t columns = mxGetN(b);
    const size_t count = (size_t)s->rows * columns;
    sparta_complex *x = mxMalloc((count > 0 ? count : 1) * sizeof *x);
    read_values(b, x, count);
    if (columns > 0) {
        check("solve", sparta_solve_many(s->solver, x, x, (int64_t)columns));
    }
    out[0] = write_values(x, (size_t)s->rows, columns);
    mxFree(x);
}

/** info = sparta_mex('info', id) */
static void info(mxArray *out[], const mxArray *in[]) {
    static const char *names[] = {"rows", "nonzeros", "fill_ratio", "factor_bytes",
                                  "factor_seconds", "perturbed_pivots"};
    const slot *s = lookup(in[1]);
    mxArray *r = mxCreateStructMatrix(1, 1, 6, names);
    mxSetField(r, 0, "rows", mxCreateDoubleScalar((double)s->rows));
    mxSetField(r, 0, "nonzeros", mxCreateDoubleScalar((double)s->nonzeros));
    mxSetField(r, 0, "fill_ratio", mxCreateDoubleScalar(s->fill_ratio));
    mxSetField(r, 0, "factor_bytes", mxCreateDoubleScalar(s->factor_bytes));
    mxSetField(r, 0, "factor_seconds", mxCreateDoubleScalar(sparta_factor_seconds(s->solver)));
    mxSetField(r, 0, "perturbed_pivots", mxCreateDoubleScalar((double)sparta_perturbed_pivots(s->solver)));
    out[0] = r;
}

/** sparta_mex('free', id). Ignores an id that is already released. */
static void free_solver(const mxArray *in[]) {
    const double value = mxGetScalar(in[1]);
    const size_t index = (size_t)value;
    if (value >= 1 && (double)index == value && index <= table_size && table[index - 1].solver != NULL) {
        release(&table[index - 1]);
        if (--live == 0) {
            mexUnlock();
        }
    }
}

void mexFunction(int out_count, mxArray *out[], int in_count, const mxArray *in[]) {
    static int registered = 0;
    if (!registered) {
        mexAtExit(release_all);
        registered = 1;
    }
    if (in_count < 1) {
        mexErrMsgIdAndTxt("sparta:invalid", "sparta_mex needs a command; use SpartaSolver instead");
    }
    const char *command = text(in[0], "the command");
    const int has_id = in_count >= 2;
    (void)out_count;

    if (strcmp(command, "create") == 0) {
        create(out, in_count, in);
    } else if (strcmp(command, "factor") == 0) {
        factor(out, in_count, in);
    } else if (strcmp(command, "solve") == 0) {
        solve(out, in_count, in);
    } else if (strcmp(command, "info") == 0 && has_id) {
        info(out, in);
    } else if (strcmp(command, "free") == 0 && has_id) {
        free_solver(in);
    } else if (strcmp(command, "version") == 0) {
        out[0] = mxCreateString(sparta_version());
    } else if (strcmp(command, "device") == 0) {
        out[0] = mxCreateString(sparta_device());
    } else if (strcmp(command, "gpu_available") == 0) {
        out[0] = mxCreateLogicalScalar(sparta_gpu_available() != 0);
    } else {
        mexErrMsgIdAndTxt("sparta:invalid", "unknown command '%s'", command);
    }
}
