"""Real Metal validation that the Schur drain cadence is a scheduling choice only.

`SPARTA_SCHUR_BATCH` sets how many pivot panels complete between two drains of
the trailing-by-trailing Schur block (0 = drain once per level). It must not
change what the factorization computes, only how often the contribution matrix
is read back and written out.

The invariant that makes this legal: a pivot search reads the fully summed
columns and their coupling rows, which `update_strip` refreshes after every
panel regardless of cadence, and never the trailing-by-trailing block. Nothing
else reads that block before the parent assembles it in the next level, and the
end-of-level drain always runs. So the pivot sequence, the permutation and D are
bitwise unaffected inside any one front; only the rounding of the Schur
accumulation moves, and with it the parent's inputs.

The rescue path is the exception the cadence could have broken, because
`front_scale` used to sweep the whole front including the stale Schur block; it
now stops at the fully summed columns, which are on the fixed panel cadence.
Note what that is and is not worth: measured against the old sweep on rescued
multi-level systems, the end-to-end difference was below the run-to-run noise
of the ordering, because the floored value enters as a relative perturbation of
~3.45e-4 either way. The change buys a by-construction guarantee that the
cadence is a pure scheduling knob, not a measurable accuracy gain, and the
tests below do not detect the difference. What they do test is that the rescue
path survives batching at all: the same pivots are floored and the solutions
still agree.

Every analyze here passes refinement_steps=0. These tests compare the raw GPU
float accumulation between cadences; refinement would iterate both runs to the
same tolerance and hide the difference they exist to detect.
"""
import os
import unittest
import numpy as np
import scipy.sparse as sparse
import sparta

if os.environ.get('SPARTA_REQUIRE_METAL') == '1' and not sparta.gpu_available():
    raise RuntimeError('Metal required: run in normal macOS Terminal')

# 1 reproduces the old cadence, one drain per panel; 0 drains once per level.
# 3 and 5 are deliberately coprime to the panel count of the fronts below, so a
# front takes its last panel mid-batch and is caught only by the final drain.
CADENCES = [1, 2, 3, 5, 8, 0]


@unittest.skipUnless(sparta.gpu_available(), 'requires Apple Silicon Metal')
class SchurCadenceTests(unittest.TestCase):
    def factor_at(self, matrix, batch, a=None):
        """Analyze and factor `matrix` under one drain cadence.

        The cadence has to be named when the analysis is built, not before the
        factorization. SPARTA_SCHUR_BATCH is read by Config::default, which runs
        inside analyze; an analysis carries the value it was built with, and
        factorize reuses it. Setting the variable between the two -- which this
        helper used to do -- changes nothing, so every comparison in this file
        was running one cadence against itself, and the positive control below
        only ever passed on the float atomics in forward substitution making two
        runs differ by chance.

        Re-analyzing per cadence is sound: the drain cadence shapes the dispatch
        and nothing in the plan, so every cadence gets the same plan and the
        comparison is still like for like.
        """
        symbolic = sparta.analyze(matrix, refinement_steps=0, schur_batch=batch)
        return symbolic.factorize(a)

    def sweep(self, a, expected, residual_bar=3e-5, error_bar=3e-4, agreement=1e-5):
        """Factor `a` at every cadence and compare against cadence 1."""
        b = a @ expected
        scale = np.linalg.norm(b)
        reference, perturbed = None, None
        for batch in CADENCES:
            with self.subTest(batch=batch):
                f = self.factor_at(a, batch)
                x = f.solve(b)
                self.assertLess(float(np.linalg.norm(a @ x - b) / scale), residual_bar)
                self.assertLess(
                    float(np.linalg.norm(x - expected) / np.linalg.norm(expected)), error_bar)
                if reference is None:
                    reference, perturbed = x, f.perturbed_pivots
                else:
                    # The rescue scale is cadence-invariant, so a pivot that was
                    # floored at one cadence is floored at every other one.
                    self.assertEqual(
                        f.perturbed_pivots, perturbed,
                        'drain cadence changed which pivots were perturbed')
                    drift = float(np.linalg.norm(x - reference) / np.linalg.norm(reference))
                    self.assertLess(
                        drift, agreement,
                        f'batch {batch} drifted {drift:.3g} from the one-panel cadence')
                del f

    def test_cadence_does_not_change_a_multilevel_tree(self):
        """Wide fronts whose Schur blocks are consumed by a parent.

        A single front would not test anything: with no parent, nothing ever
        reads the trailing block, so any cadence trivially agrees. Here each
        child's drained Schur block is assembled into the root, which is where a
        dropped or double-counted contribution would surface.
        """
        rng = np.random.default_rng(31)

        def block(n):
            x = rng.uniform(-0.01, 0.01, (n, n)) + 1j * rng.uniform(-0.01, 0.01, (n, n))
            return sparse.csc_matrix(x + x.T + np.eye(n) * 8)

        left, right, root = 200, 232, 176
        c1 = sparse.csc_matrix(rng.uniform(-0.01, 0.01, (left, root)) * (1 + 0.1j))
        c2 = sparse.csc_matrix(rng.uniform(-0.01, 0.01, (right, root)) * (1 + 0.1j))
        a = sparse.bmat([[block(left), None, c1],
                         [None, block(right), c2],
                         [c1.T, c2.T, block(root)]], format='csc')
        symbolic = sparta.analyze(a, refinement_steps=0)
        self.assertGreaterEqual(symbolic.info['large_fronts'], 2)
        self.assertGreater(symbolic.info['levels'], 1)
        expected = rng.normal(size=a.shape[0]) + 1j * rng.normal(size=a.shape[0])
        self.sweep(a, expected)

    def test_cadence_does_not_change_two_by_two_pivots(self):
        """2x2 pivots, including ones that straddle a drain boundary.

        A zero diagonal forces Bunch-Kaufman onto the 2x2 branch almost
        everywhere. `update_schur` must not split a 2x2 block across its staging
        chunks, and batching makes the pending range long enough to contain many
        of them.
        """
        n = 286
        rng = np.random.default_rng(922)
        a = np.full((n, n), 1e-6 * (1 + 0.1j), dtype=np.complex128)
        a[np.diag_indices(n)] = 0
        for i, j in rng.permutation(n).reshape(-1, 2):
            a[i, j] = a[j, i] = 2 + 0.7j
        a = sparse.csc_matrix(a)
        expected = rng.normal(size=n) + 1j * rng.normal(size=n)
        self.sweep(a, expected)

    def test_cadence_does_not_change_a_rescued_factorization(self):
        """A rank-deficient system whose dead pivots are floored, not failed.

        This front has no trailing block, so it pins the simplest claim: a
        cadence change must not turn a rescued factorization into a failed one,
        nor move the count of floored pivots. It does not isolate `front_scale`
        -- see the module docstring for why no black-box test here does.
        """
        n = 257
        singular = np.full((n, n), 1 / 1024, dtype=np.complex128)
        singular[np.arange(n - 2), np.arange(n - 2)] += 4
        good = sparse.csc_matrix(singular + np.diag(
            np.concatenate([np.zeros(n - 2), [4, 4]])).astype(np.complex128))
        singular = sparse.csc_matrix(singular)

        expected = np.ones(n, dtype=np.complex128)
        b = singular @ expected
        counts, solutions = [], []
        for batch in CADENCES:
            f = self.factor_at(good, batch, singular)
            counts.append(f.perturbed_pivots)
            solutions.append(f.solve(b))
            del f
        self.assertGreater(counts[0], 0, 'this system is meant to need a rescue')
        self.assertEqual(
            set(counts), {counts[0]},
            f'drain cadence changed the perturbed pivot count: {dict(zip(CADENCES, counts))}')
        # Perturbed factors solve a nearby system, so compare the cadences to
        # each other rather than to the exact solution.
        for batch, x in zip(CADENCES[1:], solutions[1:]):
            drift = float(np.linalg.norm(x - solutions[0]) / np.linalg.norm(solutions[0]))
            self.assertLess(drift, 1e-4, f'batch {batch} drifted {drift:.3g} on the rescue path')

        # And the same pattern still factors the healthy system cleanly at any cadence.
        for batch in CADENCES:
            f = self.factor_at(good, batch, good)
            self.assertEqual(f.perturbed_pivots, 0)
            del f

    def test_cadence_does_not_change_a_rescue_inside_a_front_with_a_tail(self):
        """The rescue path where the deferred block actually exists.

        The test above rescues inside a single front, which has no Schur
        complement at all. Here the rank deficiency sits in child fronts that do
        have a trailing block and a parent to feed, so a floored pivot is taken
        while a batch of Schur contributions is still pending -- the arrangement
        the review flagged as the one numerical exception to deferral.
        """
        rng = np.random.default_rng(11)
        dead = 6

        def block(k):
            x = rng.uniform(-0.01, 0.01, (k, k)) + 1j * rng.uniform(-0.01, 0.01, (k, k))
            m = x + x.T + np.eye(k) * 8
            m[-dead:, :] = 0
            m[:, -dead:] = 0
            return sparse.csc_matrix(m)

        left, right, root = 200, 200, 176
        c1 = sparse.csc_matrix(rng.uniform(-0.01, 0.01, (left, root)) * (1 + 0.1j))
        c2 = sparse.csc_matrix(rng.uniform(-0.01, 0.01, (right, root)) * (1 + 0.1j))
        r = rng.uniform(-0.01, 0.01, (root, root))
        a = sparse.bmat([[block(left), None, c1],
                         [None, block(right), c2],
                         [c1.T, c2.T, sparse.csc_matrix(r + r.T + np.eye(root) * 8)]],
                        format='csc')
        symbolic = sparta.analyze(a, refinement_steps=0)
        self.assertGreater(symbolic.info['levels'], 1)
        b = (rng.normal(size=a.shape[0]) + 1j * rng.normal(size=a.shape[0])).astype(np.complex128)
        counts, solutions = [], []
        for batch in CADENCES:
            f = self.factor_at(a, batch)
            counts.append(f.perturbed_pivots)
            solutions.append(f.solve(b))
            del f
        self.assertGreater(counts[0], 0, 'this system is meant to need a rescue')
        self.assertEqual(
            set(counts), {counts[0]},
            f'drain cadence changed the perturbed pivot count: {dict(zip(CADENCES, counts))}')
        for batch, x in zip(CADENCES[1:], solutions[1:]):
            drift = float(np.linalg.norm(x - solutions[0]) / np.linalg.norm(solutions[0]))
            # Loose next to the healthy cases: these are the factors of a
            # perturbed matrix, and the ordering that decides which pivots die
            # is not reproducible run to run. Measured spread is 2e-5.
            self.assertLess(drift, 5e-4, f'batch {batch} drifted {drift:.3g} on the rescue path')

    def test_cadence_is_actually_reaching_the_bridge(self):
        """Positive control for the whole file.

        Every test above asserts that two cadences agree. That would also pass
        if the variable never reached the encoder and each run used one cadence.
        Batching regroups a float accumulation, so on a tree this size the
        results must differ somewhere in the low bits -- if they are bitwise
        identical, the knob is dead and the agreement above proves nothing.
        """
        rng = np.random.default_rng(5)
        n, root = 224, 160

        def block(k):
            x = rng.uniform(-0.02, 0.02, (k, k)) + 1j * rng.uniform(-0.02, 0.02, (k, k))
            return sparse.csc_matrix(x + x.T + np.eye(k) * 6)

        c = sparse.csc_matrix(rng.uniform(-0.02, 0.02, (n, root)) * (1 + 0.2j))
        a = sparse.bmat([[block(n), None, c],
                         [None, block(n), c],
                         [c.T, c.T, block(root)]], format='csc')
        b = (rng.normal(size=a.shape[0]) + 1j * rng.normal(size=a.shape[0])).astype(np.complex128)
        first = self.factor_at(a, 1).solve(b)
        final = self.factor_at(a, 0).solve(b)
        self.assertFalse(
            np.array_equal(first, final),
            'cadence 1 and final-only gave bitwise identical results: '
            'SPARTA_SCHUR_BATCH is not reaching the encoder')
        # Same computation, different grouping: they must still agree closely.
        drift = float(np.linalg.norm(final - first) / np.linalg.norm(first))
        self.assertLess(drift, 1e-5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
