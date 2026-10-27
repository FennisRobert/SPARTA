/*==============================================================================

  SPARTA  --  C++ interface

  A thin layer over sparta.h. It owns the handles, reports failures as
  exceptions instead of return codes, and takes and returns std::vector.
  Nothing else differs, so sparta.h remains the reference for what each
  setting means.

      sparta::Solver solver(n, col_ptr, row_index, sparta::Matrix::Symmetric);
      solver.factor(values);
      std::vector<sparta::complex> x = solver.solve(b);

  Header-only, C++17 or later. Link with -lsparta.

==============================================================================*/

#ifndef SPARTA_HPP
#define SPARTA_HPP

#include "sparta.h"

#include <complex>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sparta {

/** The number type the solver works in: single-precision complex. */
using complex = std::complex<float>;

/** How the matrix was stored. See sparta_matrix in sparta.h. */
enum class Matrix : int {
    Symmetric = SPARTA_SYMMETRIC,
    General = SPARTA_GENERAL,
};

/** A named starting point for the settings. See sparta_preset in sparta.h. */
enum class Preset : int {
    Balanced = SPARTA_PRESET_BALANCED,
    Sweep = SPARTA_PRESET_SWEEP,
    OneShot = SPARTA_PRESET_ONE_SHOT,
    StructuredGrid = SPARTA_PRESET_STRUCTURED_GRID,
    Careful = SPARTA_PRESET_CAREFUL,
};

/** Thrown when a call fails. `status()` gives the machine-readable reason and
 *  `what()` the sentence from sparta_last_error. */
class Error : public std::runtime_error {
  public:
    Error(sparta_status status, const std::string &message)
        : std::runtime_error(message), status_(status) {}

    sparta_status status() const noexcept { return status_; }

  private:
    sparta_status status_;
};

/** Settings, starting from the defaults or from a preset. Every field of
 *  sparta_config can be set directly afterwards. */
struct Config : sparta_config {
    Config() { sparta_config_default(this); }

    explicit Config(Preset preset) {
        sparta_config_default(this);
        sparta_config_preset(this, static_cast<sparta_preset>(preset));
    }
};

/** Returns true when a usable GPU is visible to this process. */
inline bool gpu_available() { return sparta_gpu_available() != 0; }

/** Returns the library version, for example "0.3.0". */
inline std::string version() { return sparta_version(); }

namespace detail {

/** Turns a failed status into an exception. Successful ones pass through. */
inline void check(sparta_status status) {
    if (status != SPARTA_OK) {
        throw Error(status, sparta_last_error());
    }
}

}  // namespace detail

/** A matrix pattern, analyzed and resident on the GPU.
 *
 *  Construction performs step 1, the analysis, and reserves the GPU memory.
 *  factor() and solve() are steps 2 and 3, and each may be repeated. The
 *  solver belongs to the thread that created it. */
class Solver {
  public:
    /** Analyzes a pattern and reserves GPU memory for it.
     *
     *  @param rows       Number of rows, which equals the number of columns.
     *  @param col_ptr    rows + 1 column offsets.
     *  @param row_index  col_ptr[rows] row indices, ascending within a column.
     *  @param kind       Whether the lower triangle or the whole matrix is given.
     *  @param config     Settings; the defaults if omitted. */
    Solver(std::int64_t rows, const std::int64_t *col_ptr, const std::int64_t *row_index,
           Matrix kind, const Config &config = Config()) {
        sparta_analysis *plan = nullptr;
        detail::check(sparta_analyze(rows, col_ptr, row_index,
                                     static_cast<sparta_matrix>(kind), &config, &plan));
        fill_ratio_ = sparta_analysis_fill_ratio(plan);
        factor_bytes_ = sparta_analysis_factor_bytes(plan);
        // The plan is taken whether or not this succeeds, so there is nothing
        // left to release on the failing path.
        detail::check(sparta_solver_create(&plan, &solver_));
    }

    ~Solver() { sparta_solver_free(solver_); }

    Solver(Solver &&other) noexcept { swap(other); }

    Solver &operator=(Solver &&other) noexcept {
        swap(other);
        return *this;
    }

    Solver(const Solver &) = delete;
    Solver &operator=(const Solver &) = delete;

    /** Factors a matrix with the analyzed pattern. Repeatable with new numbers. */
    void factor(const std::vector<complex> &values) {
        expect(values.size(), nonzeros(), "values");
        detail::check(
            sparta_factor(solver_, reinterpret_cast<const sparta_complex *>(values.data())));
    }

    /** Factors a real matrix with the analyzed pattern, with real arithmetic. */
    void factor(const std::vector<float> &values) {
        expect(values.size(), nonzeros(), "values");
        detail::check(sparta_factor_real(solver_, values.data()));
    }

    /** Solves A x = b for one right-hand side. */
    std::vector<complex> solve(const std::vector<complex> &b) { return solve_many(b, 1); }

    /** Solves A x = b for one real right-hand side against real factors. */
    std::vector<float> solve(const std::vector<float> &b) { return solve_many(b, 1); }

    /** Solves for `count` real right-hand sides against real factors. */
    std::vector<float> solve_many(const std::vector<float> &b, std::int64_t count) {
        expect(b.size(), rows() * count, "b");
        std::vector<float> x(b.size());
        detail::check(sparta_solve_many_real(solver_, b.data(), x.data(), count));
        return x;
    }

    /** Solves for `count` right-hand sides held end to end, column j at b[j * rows]. */
    std::vector<complex> solve_many(const std::vector<complex> &b, std::int64_t count) {
        expect(b.size(), rows() * count, "b");
        std::vector<complex> x(b.size());
        detail::check(sparta_solve_many(solver_,
                                        reinterpret_cast<const sparta_complex *>(b.data()),
                                        reinterpret_cast<sparta_complex *>(x.data()), count));
        return x;
    }

    /** Number of rows the solver was built for. */
    std::int64_t rows() const { return sparta_solver_rows(solver_); }

    /** Number of nonzeros every `values` vector must hold. */
    std::int64_t nonzeros() const { return sparta_solver_nonzeros(solver_); }

    /** Entries the factors hold per nonzero of the matrix: the fill-in. */
    double fill_ratio() const { return fill_ratio_; }

    /** GPU memory the factors occupy, in bytes. */
    std::size_t factor_bytes() const { return factor_bytes_; }

    /** How long the GPU spent on the last factorization, in seconds. */
    double factor_seconds() const { return sparta_factor_seconds(solver_); }

    /** Pivots the last factorization had to nudge away from zero. Zero is the
     *  normal result; see sparta_perturbed_pivots. */
    std::uint64_t perturbed_pivots() const { return sparta_perturbed_pivots(solver_); }

  private:
    /** Rejects a vector of the wrong length before it reaches the C layer,
     *  where the length is trusted rather than checked. */
    static void expect(std::size_t given, std::int64_t wanted, const char *name) {
        if (given != static_cast<std::size_t>(wanted)) {
            throw Error(SPARTA_ERROR_INVALID, std::string(name) + " holds " +
                                                  std::to_string(given) + " numbers; " +
                                                  std::to_string(wanted) + " were expected");
        }
    }

    void swap(Solver &other) noexcept {
        std::swap(solver_, other.solver_);
        std::swap(fill_ratio_, other.fill_ratio_);
        std::swap(factor_bytes_, other.factor_bytes_);
    }

    sparta_solver *solver_ = nullptr;
    double fill_ratio_ = 0.0;
    std::size_t factor_bytes_ = 0;
};

}  // namespace sparta

#endif  // SPARTA_HPP
