"""Eight synthetic CPU tests. No real dataset, image files or model execution."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np
import history_pose_address_control_v1 as core


def zquat(angle):
    return [np.cos(angle/2), 0., 0., np.sin(angle/2)]


def fixture(n=3):
    rng = np.random.default_rng(11)
    image = rng.integers(0, 256, (96, 96, 3), dtype=np.uint8)
    k = np.array([[20., 0., 48.], [0., 20., 48.], [0., 0., 1.]])
    def frame(t):
        return dict(timestamp_us=t, camera_to_global=np.eye(4), intrinsic=k.copy(), rgb=image.copy())
    pairs = {name: dict(current=frame(900000), past=frame(400000)) for name in core.CAMERA_ORDER}
    p = np.array([[0., 0., 10.], [.4, .2, 10.], [-.2, .3, 11.]])[:n].copy()
    velocities = np.tile([1., 0., 0.], (n, 1))
    old = core.PHOTO.evaluate_history_correspondence(p, velocities, np.eye(4), 1000000, 1000000, pairs)
    track = dict(instance_token='same-instance', valid_mask=[True]*3,
        annotation_tokens=['a-2', 'a-1', 'a0'], annotation_prev_tokens=['', 'a-2', 'a-1'],
        annotation_next_tokens=['a-1', 'a0', 'unused'],
        global_centers_m=[[-1., 0., 10.], [-.5, 0., 10.], [0., 0., 10.]],
        global_rotations_wxyz=[[1., 0., 0., 0.]]*3)
    raw = dict(frames=[dict(relative_frame_index=i-2, timestamp_us=i*500000) for i in range(3)], tracks=[track])
    return dict(points_R=p, lidar_to_global=np.eye(4), object_index=np.zeros(n, dtype=np.int64),
                instance_tokens=np.array(['same-instance']), raw=raw, camera_result=old, camera_pairs=pairs)


def call(x, **kwargs):
    return core.evaluate_pose_address_control(**x, **kwargs)


def snapshot(x):
    """Recursive immutable-content signature, including dtype/shape/NaN bytes."""
    if isinstance(x, np.ndarray):
        return ('array', str(x.dtype), x.shape, x.tobytes())
    if isinstance(x, dict):
        return tuple((k, snapshot(v)) for k, v in sorted(x.items()))
    if isinstance(x, (list, tuple)):
        return tuple(snapshot(v) for v in x)
    return x


class SyntheticTests(unittest.TestCase):
    def test_01_translation_material_geometry(self):
        x = fixture(); out = call(x)
        self.assertTrue(out['reference_valid'].all())
        expected = np.broadcast_to([[-.1, 0., 0.], [-.6, 0., 0.]], (3, 2, 3))
        np.testing.assert_allclose(out['reference_displacement_R_m'], expected, atol=2e-15, rtol=0)
        np.testing.assert_array_equal(out['reference_bracket_indices'], np.tile([[1, 2], [0, 1]], (3, 1, 1)))
        np.testing.assert_allclose(out['reference_alpha'], .8, atol=0, rtol=0)
        expected_uv = np.empty((3, 2, 2))
        for ti, dx in enumerate((-.1, -.6)):
            expected_uv[:, ti, 0] = 48.+20.*(x['points_R'][:, 0]+dx)/x['points_R'][:, 2]
            expected_uv[:, ti, 1] = 48.+20.*x['points_R'][:, 1]/x['points_R'][:, 2]
        np.testing.assert_allclose(out['reference_uv'], expected_uv, atol=1e-14, rtol=0)

    def test_02_slerp_shortest_arc_sign_and_near_coincidence(self):
        q = core.slerp(zquat(0.), zquat(np.pi), .5)
        np.testing.assert_allclose(core.quaternion_matrix(q) @ [1., 0., 0.], [0., 1., 0.], atol=5e-16, rtol=0)
        for a in (0., .2, 1.):
            qa = core.slerp(zquat(.3), zquat(1.9), a)
            qb = core.slerp(-np.array(zquat(.3)), -np.array(zquat(1.9)), a)
            np.testing.assert_array_equal(qa, qb)
            self.assertAlmostEqual(np.linalg.norm(qa), 1., places=15)
        q0 = np.array(zquat(.3)); np.testing.assert_allclose(core.slerp(q0, -q0, .4), q0, atol=2e-16)
        near = core.slerp(zquat(.2), zquat(.20000001), .3)
        expected = core.normalized_quaternion(.7*np.array(zquat(.2))+.3*np.array(zquat(.20000001)))
        np.testing.assert_allclose(near, expected, atol=2e-16, rtol=0)
        q = core.slerp(zquat(np.deg2rad(170)), zquat(np.deg2rad(-170)), .5)
        np.testing.assert_allclose(core.quaternion_matrix(q) @ [1., 0., 0.], [-1., 0., 0.], atol=1e-14, rtol=0)

    def test_03_full_sensor_rotation_and_box_rotation(self):
        x = fixture(1); angle = .35
        rg = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.], [-np.sin(angle), 0., np.cos(angle)]])
        g = np.eye(4); g[:3, :3] = rg; g[:3, 3] = [3., -2., 1.]
        x['lidar_to_global'] = g
        # Camera and R share the same full pitched pose; no global-Z shortcut.
        for pair in x['camera_pairs'].values():
            for f in pair.values(): f['camera_to_global'] = g.copy()
        c0 = g[:3, 3] + rg @ [0., 0., 10.]
        t = x['raw']['tracks'][0]
        t['global_centers_m'] = [c0.tolist()]*3
        t['global_rotations_wxyz'] = [zquat(-np.pi/2), zquat(-np.pi/4), zquat(0.)]
        x['points_R'][0] = [1., 0., 10.]
        x['camera_result'] = core.PHOTO.evaluate_history_correspondence(x['points_R'], np.zeros((1, 3)), g, 1000000, 1000000, x['camera_pairs'])
        out = call(x); self.assertTrue(out['reference_valid'][0])
        for ti, a in enumerate((-.1*np.pi/2, -.6*np.pi/2)):
            r = np.array([[np.cos(a), -np.sin(a), 0.], [np.sin(a), np.cos(a), 0.], [0., 0., 1.]])
            expected = rg.T @ (r @ (rg @ [1., 0., 0.])) + [0., 0., 10.]
            np.testing.assert_allclose(out['reference_displacement_R_m'][0, ti], expected-x['points_R'][0], atol=3e-15, rtol=0)
            np.testing.assert_allclose(out['reference_uv'][0, ti], 48.+20.*expected[:2]/expected[2], atol=2e-14, rtol=0)

    def test_04_interval_endpoints_and_no_availability_fallback(self):
        t = fixture()['raw']['tracks'][0]; times = [0, 500000, 1000000]
        for stamp, bracket, alpha in [(0, (0, 1), 0.), (500000, (0, 1), 1.), (1000000, (1, 2), 1.)]:
            status, b, a, _, _ = core.interpolate_history_pose(t, times, stamp)
            self.assertEqual((status, b, a), (0, bracket, alpha))
        for stamp in (-1, 1000001):
            status, b, a, _, _ = core.interpolate_history_pose(t, times, stamp)
            self.assertEqual((status, b), (1, (-1, -1))); self.assertTrue(np.isnan(a))
        t['valid_mask'][0] = False
        self.assertEqual(core.interpolate_history_pose(t, times, 500000)[0], 2)
        self.assertEqual(core.interpolate_history_pose(t, times, 500001)[0], 0)

    def test_05_missing_link_time_and_priority(self):
        x = fixture(1); x['raw']['tracks'][0]['valid_mask'][0] = False
        out = call(x); self.assertEqual(out['reference_reason_code'][0], 3)
        np.testing.assert_array_equal(out['pose_reason_code'][0], [0, 2])
        self.assertTrue(np.isfinite(out['reference_uv'][0, 0]).all())
        self.assertTrue(np.isnan(out['reference_correlation'][0]))
        x = fixture(1); x['raw']['tracks'][0]['annotation_next_tokens'][1] = 'wrong'
        out = call(x); self.assertEqual(out['reference_reason_code'][0], 2)
        np.testing.assert_array_equal(out['pose_reason_code'][0], [3, 0])
        x = fixture(1); x['camera_pairs']['CAM_FRONT']['current']['timestamp_us'] = 1000001
        x['camera_result']['timestamps_us'][0, 0] = 1000001
        x['raw']['tracks'][0]['valid_mask'][0] = False
        out = call(x); self.assertEqual(out['reference_reason_code'][0], 2)
        np.testing.assert_array_equal(out['pose_reason_code'][0], [1, 2])

    def test_06_identity_history_only_and_cache_time_contracts(self):
        for mutate in (lambda x:x.update(instance_tokens=np.array(['other'])),
                       lambda x:x.update(object_index=np.array([1],dtype=np.int64)),
                       lambda x:x['raw']['tracks'].append(copy.deepcopy(x['raw']['tracks'][0])),
                       lambda x:x['raw']['frames'].append(dict(relative_frame_index=1,timestamp_us=1500000)),
                       lambda x:x['raw']['tracks'][0]['global_centers_m'].append([99.,99.,99.]),
                       lambda x:x['camera_result']['timestamps_us'].__setitem__((0,0), 900001)):
            x = fixture(1); mutate(x)
            with self.assertRaises(ValueError):call(x)
        x = fixture(1)
        class Forbidden:
            def __array__(self, *args):raise AssertionError('Forbidden GT size/extra field consumed')
        x['raw']['tracks'][0]['sizes_wlh_m'] = Forbidden()
        x['raw']['future_labels'] = Forbidden()
        self.assertTrue(call(x)['reference_valid'][0])

    def test_07_original_view_and_texture_mask_are_fixed(self):
        x = fixture(); x['camera_result']['camera_index'][:] = 1
        # All original cameras tie, but the cached choice is authoritative.
        x['raw']['tracks'][0]['global_centers_m'][1][0] = 400.
        # Another view has wide enough FOV for the new addresses, but is not used.
        for f in x['camera_pairs']['CAM_FRONT'].values():
            f['intrinsic'][0, 0] = f['intrinsic'][1, 1] = .01
        out = call(x)
        np.testing.assert_array_equal(out['camera_index'], np.ones(3, dtype=np.int8))
        self.assertTrue((out['reference_reason_code'] == 4).all())
        x = fixture(1); x['camera_result']['reason_code'][0] = 4; x['camera_result']['valid'][0] = False
        out = call(x); self.assertEqual(out['reference_reason_code'][0], 1)
        np.testing.assert_array_equal(out['pose_reason_code'][0], [4, 4])
        # A reference patch can be flat while the cached original mask remains valid.
        x = fixture(1)
        for pair in x['camera_pairs'].values():
            for f in pair.values():f['rgb'][:] = 128
        out = call(x); self.assertTrue(out['reference_valid'][0])
        self.assertEqual(out['reference_correlation'][0], 0.)

    def test_08_broken_shared_support_immutability_chunk_empty(self):
        x = fixture(); before = snapshot(x); out = call(x, chunk_size=1)
        self.assertEqual(snapshot(x), before)
        full = call(x, chunk_size=4096)
        for key in out:np.testing.assert_array_equal(out[key], full[key])
        gc = core.PHOTO.grayscale(x['camera_pairs']['CAM_FRONT']['current']['rgb'])
        gp = core.PHOTO.grayscale(x['camera_pairs']['CAM_FRONT']['past']['rgb'])
        a = core.PHOTO.sample_patches(gc, out['reference_uv'][:, 0])
        b = core.PHOTO.sample_patches(np.roll(gp, gp.shape[1]//2, axis=1), out['reference_uv'][:, 1])
        np.testing.assert_array_equal(out['reference_broken_correlation'], core.PHOTO.regularized_zncc(a, b))
        for prefix in ('', 'broken_'):
            corr = out['reference_'+prefix+'correlation']; old = x['camera_result'][prefix+'correlation']
            np.testing.assert_array_equal(out[prefix+'delta_eD_minus_eGT'], corr-old[:, 1])
            np.testing.assert_array_equal(out[prefix+'delta_eZero_minus_eGT'], corr-old[:, 0])
        empty = call(fixture(0))
        self.assertEqual(empty['reference_displacement_R_m'].shape, (0, 2, 3))
        self.assertEqual(empty['reference_uv'].shape, (0, 2, 2))
        self.assertTrue(all(len(v) == 0 for v in empty.values()))


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps(dict(status='PASS_SYNTHETIC_ONLY' if result.wasSuccessful() else 'FAIL_SYNTHETIC_ONLY',
        tests=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        core_sha256=hashlib.sha256(Path(core.__file__).read_bytes()).hexdigest(),
        test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        frozen_photo_sha256=core.PHOTO_SOURCE_SHA256, real_data_opened=False, numpy=np.__version__)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
