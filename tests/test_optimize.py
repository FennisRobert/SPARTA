"""sparta.optimize and saved profiles.

The search itself is tested in Rust (src/optimize.rs); these check what the
Python layer adds: problem forms, progress, saving, loading by name and the
profile file, which each test points at a temporary one.
"""
import os
import tempfile
import unittest
import warnings

import numpy as np
from scipy import sparse
import sparta


def laplacian(k: int, shift: complex = 0.1j) -> sparse.csc_matrix:
    """A shifted 2D Laplacian on a k by k grid, complex symmetric."""
    line = sparse.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(k, k))
    a = sparse.kron(sparse.eye(k), line) + sparse.kron(line, sparse.eye(k))
    return (a + shift*sparse.eye(k*k)).astype(np.complex128).tocsc()


class ProfileFile(unittest.TestCase):
    """Points SPARTA_CONFIG at a fresh file for each test."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self._saved = os.environ.get('SPARTA_CONFIG')
        os.environ['SPARTA_CONFIG'] = os.path.join(self._dir.name, 'configs.json')

    def tearDown(self):
        if self._saved is None:
            os.environ.pop('SPARTA_CONFIG', None)
        else:
            os.environ['SPARTA_CONFIG'] = self._saved
        self._dir.cleanup()


class ProfileTests(ProfileFile):
    def test_save_load_list_delete(self):
        config = sparta.Config.defaults().with_(nemin=32, cpu_panel=48)
        config.save('mine', note='hand tuned')
        self.assertEqual(sparta.Config.load('mine'), config)
        self.assertIn('hand tuned', sparta.profiles()['mine'])
        self.assertEqual(sparta.Config.store(), os.environ['SPARTA_CONFIG'])
        sparta.Config.delete('mine')
        self.assertEqual(sparta.profiles(), {})
        with self.assertRaises(KeyError):
            sparta.Config.load('mine')
        with self.assertRaises(KeyError):
            sparta.Config.delete('mine')

    def test_preset_names_are_refused(self):
        with self.assertRaises(ValueError):
            sparta.Config.defaults().save('sweep')
        with self.assertRaises(ValueError):
            sparta.optimize(laplacian(4), name='one_shot', verbose=False)

    def test_analyze_takes_a_profile_or_preset_name(self):
        a = laplacian(12)
        sparta.Config.defaults().with_(nemin=8).save('small')
        for config in ('small', 'sweep'):
            with sparta.analyze(a, config=config) as solver:
                x = solver.factorize().solve(np.ones(a.shape[0]))
                self.assertLess(np.linalg.norm(a @ x - 1)/np.sqrt(a.shape[0]), 1e-5)

    def test_a_profile_from_another_chip_warns(self):
        import json
        sparta.Config.defaults().save('carried')
        path = sparta.Config.store()
        with open(path) as f:
            data = json.load(f)
        data['profiles']['carried']['machine']['chip'] = 'Some Other Chip'
        # An older profile lacks settings added since; they take defaults.
        del data['profiles']['carried']['config']['cpu_panel']
        with open(path, 'w') as f:
            json.dump(data, f)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            config = sparta.Config.load('carried')
        self.assertTrue(any('Some Other Chip' in str(w.message) for w in caught))
        self.assertEqual(config, sparta.Config.defaults())


class OptimizeTests(ProfileFile):
    def test_cpu_optimization_saves_and_reports(self):
        a, b = laplacian(20), np.ones(400)
        reports = []
        result = sparta.optimize([(a, b), laplacian(16)], name='tuned', gpu=False,
                                 time_limit=20, verbose=False, progress=reports.append)
        self.assertIs(result.backend, sparta.Backend.CPU)
        self.assertTrue(result.config.force_cpu)
        self.assertGreater(result.speedup, 0)
        self.assertEqual(set(result.seconds), {sparta.Backend.CPU})
        self.assertEqual(sparta.Config.load('tuned'), result.config)
        # Progress counts up to a total it kept honest.
        self.assertTrue(reports)
        self.assertEqual(reports[-1].done, reports[-1].total)
        self.assertTrue(all(r.done <= r.total for r in reports))
        self.assertIn('default', str(result))
        for name, (before, after) in result.changes.items():
            self.assertEqual(getattr(result.config, name), after)

    def test_lu_problems_and_block_right_hand_sides(self):
        a = laplacian(14).tolil()
        a[0, 1] = 3.0  # unsymmetric, so the LU path
        rhs = np.random.default_rng(1).standard_normal((a.shape[0], 3))
        result = sparta.optimize((a.tocsc(), rhs), gpu=False, time_limit=5, verbose=False)
        self.assertIsNone(result.name)
        self.assertEqual(sparta.profiles(), {})

    def test_a_raising_callback_stops_it(self):
        def stop(report):
            if report.done >= 2:
                raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            sparta.optimize(laplacian(10), gpu=False, verbose=False, progress=stop)

    def test_iterations_restart_until_settled(self):
        lines = []
        result = sparta.optimize(laplacian(12), gpu=False, iterations=3, verbose=False,
                                 progress=lambda r: lines.append(r.line))
        ran, asked = result.iterations
        self.assertEqual(asked, 3)
        self.assertTrue(1 <= ran <= 3)
        self.assertEqual(sum('iteration' in l and 'of 3' in l for l in lines), ran)
        if ran < 3:
            self.assertTrue(any('settled' in l for l in lines))
        with self.assertRaises(ValueError):
            sparta.optimize(laplacian(4), iterations=0, verbose=False)

    def test_a_high_minimum_gain_keeps_the_defaults(self):
        result = sparta.optimize(laplacian(10), gpu=False, min_gain=0.9, verbose=False)
        self.assertEqual(result.changes, {})
        with self.assertRaises(ValueError):
            sparta.optimize(laplacian(4), min_gain=1.5, verbose=False)

    def test_tuning_settings_go_in_start(self):
        with self.assertRaises(TypeError):
            sparta.optimize(laplacian(4), nemin=8)

    @unittest.skipUnless(sparta.gpu_available(), "Metal device required")
    def test_both_backends(self):
        result = sparta.optimize(laplacian(24), time_limit=10, verbose=False)
        self.assertEqual(set(result.seconds), set(sparta.Backend))
        self.assertEqual(result.config.force_cpu, result.backend is sparta.Backend.CPU)


if __name__ == '__main__':
    unittest.main()
