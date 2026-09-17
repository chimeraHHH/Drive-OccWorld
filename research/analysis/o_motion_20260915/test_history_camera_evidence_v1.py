"""Analytic CPU QA only. Synthetic images/poses are not experimental results."""
import copy
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
import unittest

import numpy as np
import history_camera_evidence_v1 as core


def image(size=96):
    y, x = np.indices((size, size))
    gray = ((x * 37 + y * 19 + x * y * 3) % 256).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def frame(rgb, timestamp=1000000, pose=None):
    h, w = rgb.shape[:2]
    return dict(rgb=rgb, timestamp_us=timestamp,
                camera_to_global=np.eye(4) if pose is None else pose,
                intrinsic=np.asarray([[20., 0., w / 2], [0., 20., h / 2], [0., 0., 1.]]))


def pairs(current, past):
    result = {k: dict(current=None, past=None) for k in core.CAMERA_ORDER}
    result['CAM_FRONT'] = dict(current=current, past=past)
    return result


def evaluate(camera_pairs, p=None, v=None, **kwargs):
    p = np.asarray([[0., 0., 10.]]) if p is None else np.asarray(p, dtype=float)
    v = np.asarray([[2., 0., 0.]]) if v is None else np.asarray(v, dtype=float)
    return core.evaluate_history_correspondence(p, v, np.eye(4), 1000000,
                                               kwargs.pop('availability', 1000000),
                                               camera_pairs, **kwargs)


class AnalyticQA(unittest.TestCase):
    def test_lsq_uses_all_fixed_nominal_horizons(self):
        v = np.asarray([[2., -3., .5], [0., 0., 0.]])
        h = np.asarray(core.HORIZONS_SECONDS)
        d = h[:, None, None] * v[None]
        np.testing.assert_array_equal(core.nominal_lsq_velocity(d), v)
        d[3, 0, 0] += 1.
        self.assertAlmostEqual(core.nominal_lsq_velocity(d)[0, 0], 2. + 2. / 7.5)

    def test_actual_two_camera_times_and_column_transforms(self):
        rgb = image(128)
        g0 = np.eye(4)
        g0[:3, 3] = [5., 1., 0.]
        current_pose, past_pose = np.eye(4), np.eye(4)
        current_pose[0, 3], past_pose[0, 3] = 1., .5
        p, v = np.asarray([[0., 0., 10.]]), np.asarray([[2., 0., 0.]])
        for f, expected in ((frame(rgb, 1100000, current_pose), [72.4, 66.]),
                            (frame(rgb, 600000, past_pose), [71.4, 66.])):
            uv, depth, _, valid = core.project(p, v, g0, 1000000, f)
            np.testing.assert_allclose(uv[0], expected, rtol=0, atol=1e-12)
            self.assertEqual(depth[0], 10.)
            self.assertTrue(valid[0])
        rz = np.asarray([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        g0 = np.eye(4)
        g0[:3, :3] = rz
        uv, _, _, _ = core.project(np.asarray([[1., 0., 10.]]), v, g0, 1000000,
                                   frame(rgb, 1100000))
        np.testing.assert_allclose(uv[0], [64., 66.4], atol=1e-12, rtol=0)

    def test_patch_xy_bilinear_and_broken_roll(self):
        y, x = np.indices((40, 42))
        gray = (x + 10 * y).astype(float)
        uv = np.asarray([[10.25, 11.5], [21.1, 20.2]])
        p = core.sample_patches(gray, uv)
        self.assertAlmostEqual(p[0, 24], 125.25)
        self.assertAlmostEqual(p[0, 25] - p[0, 24], 1.)
        self.assertAlmostEqual(p[0, 31] - p[0, 24], 10.)
        actual = core.sample_patches(gray, uv, 21)
        expected = core.sample_patches(np.roll(gray, 21, axis=1), uv)
        np.testing.assert_array_equal(actual, expected)

    def test_zncc_fixed_regularizer_and_flat_patches(self):
        a = np.arange(49, dtype=float)[None] / 49.
        variance = a.var()
        expected = variance / (variance + core.ZNCC_EPS**2)
        self.assertAlmostEqual(core.regularized_zncc(a, a)[0], expected)
        self.assertAlmostEqual(core.regularized_zncc(a, 1. - a)[0], -expected)
        np.testing.assert_array_equal(core.regularized_zncc(a, np.zeros_like(a)), [0.])

    def test_motion_score_sign_and_zero_hypothesis_identity(self):
        current = image()
        # Object moved +2 pixels since the past frame at t0-.5s.
        past = np.roll(current, -2, axis=1)
        result = evaluate(pairs(frame(current), frame(past, 500000)))
        self.assertTrue(result['valid'][0])
        self.assertGreater(result['score'][0], .1)
        result = evaluate(pairs(frame(current), frame(past, 500000)), v=[[0., 0., 0.]])
        self.assertEqual(result['score'][0], 0.)
        self.assertEqual(result['broken_score'][0], 0.)

    def test_camera_tie_is_fixed_no_retry_and_availability(self):
        rgb = image()
        cam = pairs(frame(rgb), None)
        cam['CAM_FRONT_LEFT'] = dict(current=frame(rgb), past=frame(rgb, 500000))
        result = evaluate(cam)
        self.assertEqual(result['camera_index'][0], 0)
        self.assertEqual(result['reason_code'][0], 2)  # No retry to a usable past view.
        cam = pairs(frame(rgb, 1100000), frame(rgb, 500000))
        self.assertTrue(evaluate(cam, availability=1100000)['valid'][0])
        with self.assertRaises(ValueError):
            evaluate(cam, availability=1000000)

    def test_one_texture_mask_for_real_and_broken(self):
        current = image()
        flat = np.zeros_like(current)
        result = evaluate(pairs(frame(current), frame(flat, 500000)))
        self.assertTrue(result['valid'][0])  # No control/past-specific exclusion.
        self.assertEqual(result['score'][0], 0.)
        self.assertEqual(result['broken_score'][0], 0.)
        result = evaluate(pairs(frame(flat), frame(current, 500000)))
        self.assertEqual(result['reason_code'][0], 4)
        self.assertTrue(np.isnan(result['score'][0]))
        self.assertTrue(np.isnan(result['broken_score'][0]))

    def test_chunking_empty_input_and_readonly_rng(self):
        rgb = image()
        cam = pairs(frame(rgb), frame(np.roll(rgb, -2, axis=1), 500000))
        p = np.asarray([[0., 0., 10.], [1., 0., 10.], [0., 0., -10.]])
        v = np.repeat([[2., 0., 0.]], 3, axis=0)
        before = copy.deepcopy(cam)
        rng_before = np.random.get_state()
        a, b = evaluate(cam, p, v, chunk_size=1), evaluate(cam, p, v, chunk_size=4096)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        for name in cam:
            for which in ('current', 'past'):
                if cam[name][which] is not None:
                    for key in cam[name][which]:
                        np.testing.assert_array_equal(cam[name][which][key], before[name][which][key])
        rng_after = np.random.get_state()
        for left, right in zip(rng_before, rng_after):
            np.testing.assert_array_equal(left, right)
        result = evaluate(cam, np.empty((0, 3)), np.empty((0, 3)))
        self.assertEqual(result['score'].shape, (0,))
        self.assertEqual(a['reason_code'][2], 1)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--receipt':
        receipt = Path(sys.argv[2])
        start = time.monotonic()
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AnalyticQA))
        payload = dict(schema='history-camera-evidence-analytic-qa-v1',
                       status='PASS' if result.wasSuccessful() else 'FAIL',
                       tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                       synthetic_analytic_only=True, real_dataset_read=False, model_forward=False,
                       optimizer_updates=0, torch_imported=False,
                       python=sys.executable, python_version=platform.python_version(), numpy=np.__version__,
                       seconds=time.monotonic() - start,
                       source_sha256={str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in (Path(core.__file__), Path(__file__))})
        with receipt.open('x') as f:
            json.dump(payload, f, indent=2, allow_nan=False)
            f.write('\n')
        raise SystemExit(0 if result.wasSuccessful() else 1)
    unittest.main()
