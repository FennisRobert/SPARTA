"""Unknown keywords are ignored, never fatal.

analyze(), factorize() and Config.with_() take any keyword, so one settings
dict can serve both calls, older or newer SPARTA versions, and other solvers.
"""
import unittest
import warnings
import numpy as np
import scipy.sparse as sparse
import sparta


class UnusedSettingTests(unittest.TestCase):
    def setUp(self):
        self.a = (sparse.eye(6, dtype=complex, format='csc') * 4
                  + sparse.eye(6, k=1, dtype=complex) + sparse.eye(6, k=-1, dtype=complex)).tocsc()

    def test_analyze_warns_and_runs(self):
        with self.assertWarns(sparta.UnusedSettingWarning) as caught:
            f = sparta.analyze(self.a, force_cpu=True, some_old_option=3, pardiso_iparm=[1])
        self.assertIn('pardiso_iparm, some_old_option', str(caught.warning))
        self.assertEqual(caught.filename, __file__)
        f.factorize()
        np.testing.assert_allclose(self.a @ f.solve(np.ones(6)), np.ones(6))

    def test_one_settings_dict_for_both_calls(self):
        """Analysis-only settings are ignored silently by factorize()."""
        settings = dict(force_cpu=True, nemin=4, supernodal=False, static_pivot=1e-10,
                        config=sparta.Config.preset('balanced'), separator_trials=1, refinement_steps=0)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            f = sparta.analyze(self.a, **settings).factorize(self.a, **settings)
        np.testing.assert_allclose(self.a @ f.solve(np.ones(6)), np.ones(6))
        with self.assertWarns(sparta.UnusedSettingWarning) as caught:
            f.factorize(self.a, nemin=8, made_up=1)
        self.assertIn('made_up', str(caught.warning))
        self.assertNotIn('nemin', str(caught.warning))

    def test_config_with_ignores_unknown_names(self):
        with self.assertWarns(sparta.UnusedSettingWarning):
            config = sparta.Config.defaults().with_(nemin=8, not_a_setting=1)
        self.assertEqual(config.nemin, 8)
        self.assertFalse(hasattr(config, 'not_a_setting'))

    def test_bad_values_still_fail(self):
        """Only unknown names are forgiven; a wrong value for a known one is not."""
        with self.assertRaises(ValueError):
            sparta.analyze(self.a, force_cpu=True, static_pivot=2.0)


if __name__ == '__main__':
    unittest.main()
