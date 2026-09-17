"""Synthetic NumPy tests only: no real images, GT, RAFT weights or inference."""
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import history_frozen_flow_evidence_v1 as core


def field(x=0., y=0.):
    a = np.empty((core.HEIGHT, core.WIDTH, 2), dtype=np.float32)
    a[..., 0], a[..., 1] = x, y
    return a


def flows(forward=(2., -1.), reverse=(-2., 1.), broken=(7., 4.)):
    return dict(zip(core.FLOW_KEYS, (field(*v) for v in (forward, reverse, broken))))


def camera(reasons=(0,)):
    reason = np.asarray(reasons, dtype=np.uint8); n = len(reason)
    uv = np.tile(np.array([[[10.25, 20.5], [10.25, 20.5]],
                           [[30.25, 40.5], [32.25, 39.5]]]), (n, 1, 1, 1))
    uv[(reason != 0) & (reason != 4)] = np.nan
    ci = np.zeros(n, dtype=np.int8); ci[reason == 1] = -1
    return dict(source_flat_indices=np.arange(n, dtype=np.int64), camera_index=ci,
                reason_code=reason, valid=reason == 0, uv=uv,
                score=np.full(n, np.nan))  # Cached ZNCC scores must not be needed.


class SyntheticTests(unittest.TestCase):
    def test_01_direction_units_and_independent_broken(self):
        old = camera(); f = flows(); out = core.evaluate_flow_evidence(old, {'CAM_FRONT': f})
        np.testing.assert_allclose(out['residual'][0], [np.sqrt(5), 0.], rtol=0, atol=0)
        self.assertEqual(out['score'][0], np.sqrt(5))
        np.testing.assert_allclose(out['broken_residual'][0], [np.sqrt(65), np.sqrt(50)], rtol=0, atol=0)
        self.assertEqual(out['broken_score'][0], np.sqrt(65)-np.sqrt(50))
        np.testing.assert_array_equal(out['fb_valid'], [[True, True]])
        np.testing.assert_array_equal(out['fb_residual'], [[0., 0.]])
        np.testing.assert_array_equal(out['predicted_past_uv'][0], old['uv'][0, :, 0]+[2., -1.])
        self.assertEqual(out['hypothesis_flow_separation_px'][0], np.sqrt(5))
        # A rolled address would introduce W/2. The independently inferred broken
        # field is compared with the unchanged candidate address, as above.

    def test_02_fractional_bilinear_xy_axes_and_closed_boundary(self):
        y, x = np.indices((core.HEIGHT, core.WIDTH), dtype=np.float32)
        f = np.stack([x/16+y/32, y/8-x/64], axis=-1)
        q = np.array([[10.25, 20.5], [0., 0.], [1599., 899.], [1600., 20.], [-.1, 20.], [np.nan, 0.]])
        value, valid = core.bilinear_sample_flow(f, q)
        np.testing.assert_array_equal(valid, [True, True, True, False, False, False])
        expected = np.stack([q[:3, 0]/16+q[:3, 1]/32, q[:3, 1]/8-q[:3, 0]/64], axis=-1)
        np.testing.assert_array_equal(value[:3], expected)
        self.assertTrue(np.isnan(value[3:]).all()); self.assertEqual(value.dtype, np.float64)

    def test_03_primary_extension_missing_and_original_bytes(self):
        old = camera([0, 4, 1, 2, 3]); before = {k:(v.dtype.str, v.shape, v.tobytes()) for k,v in old.items()}
        for v in old.values():v.setflags(write=False)
        f = flows()
        for v in f.values():v.setflags(write=False)
        out = core.evaluate_flow_evidence(old, {'CAM_FRONT': f})
        np.testing.assert_array_equal(out['flow_valid'], [True, True, False, False, False])
        np.testing.assert_array_equal(out['primary_valid'], [True, False, False, False, False])
        for key in ('source_flat_indices', 'camera_index', 'reason_code'):
            self.assertEqual(out[key].tobytes(), old[key].tobytes())
            self.assertFalse(np.shares_memory(out[key], old[key]))
        for key, value in out.items():
            if value.dtype == np.float64:self.assertTrue(np.isnan(value[2:]).all(), key)
        self.assertEqual(before, {k:(v.dtype.str, v.shape, v.tobytes()) for k,v in old.items()})

    def test_04_endpoint_outside_keeps_primary_and_FB_only_missing(self):
        old = camera(); old['uv'][0, 0] = [[1595., 30.], [1595., 30.]]
        old['uv'][0, 1] = [[10., 30.], [20., 30.]]
        f = flows(forward=(10., 0.), reverse=(-10., 0.))
        out = core.evaluate_flow_evidence(old, {'CAM_FRONT': f})
        self.assertTrue(out['primary_valid'][0]); self.assertEqual(out['score'][0], 10.)
        np.testing.assert_array_equal(out['fb_valid'], [[False, True]])
        self.assertTrue(np.isnan(out['fb_residual'][0, 0])); self.assertEqual(out['fb_residual'][0, 1], 0.)
        self.assertEqual(out['predicted_past_uv'][0, 0, 0], 1605.)
        # Candidate past coordinate is arithmetic only; never a forward sampler.
        old['uv'][0, 0, 1] = [2000., 30.]
        more = core.evaluate_flow_evidence(old, {'CAM_FRONT': f})
        self.assertTrue(more['primary_valid'][0]); self.assertEqual(more['residual'][0, 0], 395.)

    def test_05_FB_is_continuous_not_a_gate(self):
        old = camera(); f = flows(); good = core.evaluate_flow_evidence(old, {'CAM_FRONT': f})
        f['past_to_current'] = field(10000., -5000.)
        bad = core.evaluate_flow_evidence(old, {'CAM_FRONT': f})
        np.testing.assert_array_equal(good['score'], bad['score'])
        np.testing.assert_array_equal(good['primary_valid'], bad['primary_valid'])
        self.assertTrue((bad['fb_residual'] > 1000.).all())

    def test_06_streamed_camera_merge_and_no_view_rows(self):
        old = camera([0, 4, 1, 3]); old['camera_index'][1] = 1
        fields = {'CAM_FRONT': flows(), 'CAM_FRONT_RIGHT': flows(forward=(-1., 3.), reverse=(1., -3.))}
        full = core.evaluate_flow_evidence(old, fields); merged = core.initialize_flow_evidence(old)
        for ci,name in enumerate(core.CAMERA_ORDER):
            rows,part = core.evaluate_camera_flow_evidence(old, ci, fields.get(name))
            for key,value in part.items():merged[key][rows] = value
        for key in full:np.testing.assert_array_equal(full[key], merged[key])
        self.assertFalse(merged['flow_valid'][2]); self.assertTrue(np.isnan(merged['score'][2]))
        empty = core.evaluate_flow_evidence(camera([]), {})
        self.assertEqual(empty['sampled_flow'].shape, (0,2,2)); self.assertEqual(empty['fb_valid'].shape, (0,2))

    def test_07_geometric_control_uses_both_time_addresses(self):
        old = camera(); old['uv'][0] = [[[10.,10.],[14.,14.]], [[20.,20.],[24.,24.]]]
        out = core.evaluate_flow_evidence(old, {'CAM_FRONT': flows()})
        self.assertEqual(out['hypothesis_flow_separation_px'][0], 0.)
        self.assertGreater(np.linalg.norm(old['uv'][0,1,1]-old['uv'][0,0,1]), 0.)

    def test_08_contract_failures_not_silent_reselection(self):
        old = camera()
        with self.assertRaises(ValueError):core.evaluate_flow_evidence(old, {})
        with self.assertRaises(ValueError):core.evaluate_flow_evidence(old, {'wrong_camera': flows()})
        f = flows(); f['current_to_past'] = f['current_to_past'].transpose(2,0,1)
        with self.assertRaises(ValueError):core.evaluate_flow_evidence(old, {'CAM_FRONT': f})
        f = flows(); f['current_to_past'][0,0,0] = np.nan
        with self.assertRaises(ValueError):core.evaluate_flow_evidence(old, {'CAM_FRONT': f})
        old['uv'][0,0,0,0] = -1.
        with self.assertRaises(ValueError):core.evaluate_flow_evidence(old, {'CAM_FRONT': flows()})
        extension = camera([4]); extension['uv'][0,0,1,0] = np.nan
        with self.assertRaises(ValueError):core.initialize_flow_evidence(extension)


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticTests))
    print(json.dumps(dict(status='PASS_SYNTHETIC_ONLY' if result.wasSuccessful() else 'FAIL_SYNTHETIC_ONLY',
        tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        core_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        numpy=np.__version__, real_inference=False, GT_read=False)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
