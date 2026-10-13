/*==============================================================================

  The same system as laplace.c, solved through the C++ interface.

  sparta.hpp owns the handles and throws on failure, so the three steps read as
  a constructor and two calls.

  Build and run:  make -C .. run-cpp

==============================================================================*/

#include "sparta.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using sparta::complex;

/** The lower triangle of a 2D Laplacian on a square grid. */
struct Laplacian {
    std::int64_t rows = 0;
    std::vector<std::int64_t> col_ptr;
    std::vector<std::int64_t> row_index;
    std::vector<complex> values;

    /** @param side   Grid points along one edge.
     *  @param shift  Imaginary part added to the diagonal. */
    Laplacian(int side, float shift) : rows(std::int64_t(side) * side) {
        col_ptr.reserve(rows + 1);
        for (std::int64_t j = 0; j < rows; ++j) {
            col_ptr.push_back(std::int64_t(row_index.size()));

            row_index.push_back(j);
            values.push_back({4.0f, shift});

            if (j % side + 1 < side) {
                row_index.push_back(j + 1);
                values.push_back({-1.0f, 0.0f});
            }
            if (j / side + 1 < side) {
                row_index.push_back(j + side);
                values.push_back({-1.0f, 0.0f});
            }
        }
        col_ptr.push_back(std::int64_t(row_index.size()));
    }

    /** Returns A x, reading the stored triangle as a symmetric matrix. */
    std::vector<complex> operator*(const std::vector<complex> &x) const {
        std::vector<complex> y(rows);
        for (std::int64_t j = 0; j < rows; ++j) {
            for (std::int64_t k = col_ptr[j]; k < col_ptr[j + 1]; ++k) {
                const std::int64_t i = row_index[k];
                y[i] += values[k] * x[j];
                if (i != j) {
                    y[j] += values[k] * x[i];
                }
            }
        }
        return y;
    }
};

/** Returns the largest distance between any entry of `x` and 1. */
double error_against_ones(const std::vector<complex> &x) {
    double worst = 0.0;
    for (const complex &value : x) {
        worst = std::max(worst, double(std::abs(value - 1.0f)));
    }
    return worst;
}

}  // namespace

int main() {
    std::printf("SPARTA %s\n", sparta::version().c_str());
    if (!sparta::gpu_available()) {
        std::fprintf(stderr, "no Metal GPU is visible to this process\n");
        return 1;
    }

    const Laplacian a(64, 0.05f);
    const std::vector<complex> want(a.rows, complex(1.0f, 0.0f));
    const std::vector<complex> b = a * want;
    std::printf("%lld unknowns, %zu nonzeros in the lower triangle\n\n",
                static_cast<long long>(a.rows), a.values.size());

    try {
        sparta::Solver solver(a.rows, a.col_ptr.data(), a.row_index.data(),
                              sparta::Matrix::Symmetric);
        std::printf("analyze   fill ratio %.2f, factors need %.1f MB on the GPU\n",
                    solver.fill_ratio(), solver.factor_bytes() / 1.0e6);

        solver.factor(a.values);
        std::printf("factor    %.4f s on the GPU, %llu pivots perturbed\n",
                    solver.factor_seconds(),
                    static_cast<unsigned long long>(solver.perturbed_pivots()));

        const std::vector<complex> x = solver.solve(b);
        std::printf("solve     largest error %.2e\n", error_against_ones(x));
    } catch (const sparta::Error &error) {
        std::fprintf(stderr, "SPARTA failed (%d): %s\n", error.status(), error.what());
        return 1;
    }
    return 0;
}
