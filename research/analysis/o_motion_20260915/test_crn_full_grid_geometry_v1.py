"""Small analytic tests; optional fixed FIRST real dev anchor is never default.

Real mode compares the entire 200x200x16 field, not a sampled score. It loads
only current predicted boxes and t0 G0 into build; raw GT tracks are not used.
"""
import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import unittest

import numpy as np
from crn_full_grid_geometry_v1 import build_full_grid_geometry, _geometry, REFERENCE_SHA, GMO

HERE = Path(__file__).resolve().parent


def reference():
    path = HERE / 'predicted_object_state_cv_diagnostic_v2.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == REFERENCE_SHA
    spec = importlib.util.spec_from_file_location('_grid_reference_cv', path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def box(center=(0., 0., 0.), size=(2., 4., 2.), rotation=(1., 0., 0., 0.),
        velocity=(2., 3.), score=.5, name='car'):
    return dict(translation=list(center), size=list(size), rotation=list(rotation),
                velocity=list(velocity), detection_score=score, detection_name=name)


def compare(boxes, G0, shape, extent, timings=None):
    before = copy.deepcopy(boxes);old_pose = G0.copy()
    tick = time.monotonic()
    actual = build_full_grid_geometry(boxes, G0, grid_shape_xyz=shape, extent_xyz=extent)
    build_seconds = time.monotonic() - tick
    points = _geometry().voxel_centers_xyz(shape, extent).reshape(-1, 3)
    tick = time.monotonic()
    velocity, owner = reference().predicted_velocity_field(boxes, G0, points)
    if timings is not None: timings.update(build_seconds=build_seconds, reference_seconds=time.monotonic()-tick)
    assert actual['velocity_R'].dtype == np.float64 and actual['owner'].dtype == np.int64
    # Bytes, including signed zero, plus full owner equality; no tolerance.
    assert actual['velocity_R'].tobytes() == velocity.reshape(*shape, 3).tobytes()
    assert actual['owner'].tobytes() == owner.reshape(shape).tobytes()
    assert boxes == before and G0.tobytes() == old_pose.tobytes()
    return actual


class FullGridTests(unittest.TestCase):
    def test_empty_and_non_square_xyz(self):
        result = compare([], np.eye(4), (3, 2, 4), (-3., -1., -2., 3., 1., 2.))
        self.assertEqual(result['velocity_R'].shape, (3, 2, 4, 3))
        self.assertTrue(np.all(result['owner'] == -1));self.assertTrue(np.all(result['velocity_R'] == 0))
        self.assertEqual(result['boxes']['R0_R'].shape, (0, 3, 3))
        # Hand-indexed non-symmetric centre: i=2,j=0,k=3 -> (2,-.5,1.5).
        result = compare([box(center=(2., -.5, 1.5), size=(.1, .1, .1))], np.eye(4),
                         (3, 2, 4), (-3., -1., -2., 3., 1., 2.))
        self.assertEqual(np.argwhere(result['owner'] >= 0).tolist(), [[2, 0, 3]])

    def test_scores_ties_all_classes_original_indices(self):
        boxes = [box(name='barrier', score=100.), box(score=-2., velocity=(1., 0.)),
                 box(score=.2, velocity=(3., 4.)), box(score=.2, velocity=(8., 9.))]
        result = compare(boxes, np.eye(4), (3, 2, 3), (-1.5, -1., -1.5, 1.5, 1., 1.5))
        self.assertTrue(np.all(result['owner'] == 2))
        self.assertEqual(result['boxes']['eligible_GMO'].tolist(), [False, True, True, True])
        self.assertEqual(result['boxes']['export_index'].tolist(), [0, 1, 2, 3])
        for name in GMO:
            result = compare([box(name=name, score=-99.)], np.eye(4), (1, 1, 1), (-.5, -.5, -.5, .5, .5, .5))
            self.assertEqual(result['owner'].item(), 0)

    def test_full_3d_face_tolerance_wlh(self):
        result = compare([box()], np.eye(4), (7, 5, 5), (-3.5, -2.5, -2.5, 3.5, 2.5, 2.5))
        # x half-length2,y half-width1,z half-height1; all closed faces.
        self.assertEqual(int(np.sum(result['owner'] == 0)), 5 * 3 * 3)
        for axis, face in enumerate((2., 1., 1.)):
            for sign in (-1., 1.):
                for offset, inside in ((0., True), (.5e-9, True), (2e-9, False)):
                    point = np.zeros(3);point[axis] = sign * (face + offset)
                    extent = tuple(point - .125) + tuple(point + .125)
                    result = compare([box()], np.eye(4), (1, 1, 1), extent)
                    self.assertEqual(result['owner'].item(), 0 if inside else -1)

    def test_translation_yaw_vector_and_metadata(self):
        pose = np.array([[0., -1., 0., 100.], [1., 0., 0., -20.],
                         [0., 0., 1., 3.], [0., 0., 0., 1.]])
        boxes = [box(center=(100., -20., 3.), velocity=(2., 0.), size=(20., 20., 20.))]
        result = compare(boxes, pose, (3, 4, 2), (-3., -2., -1., 3., 2., 1.))
        np.testing.assert_array_equal(result['velocity_R'], np.broadcast_to([0., -2., 0.], (3, 4, 2, 3)))
        np.testing.assert_array_equal(result['boxes']['c0_R'], [[0., 0., 0.]])
        np.testing.assert_array_equal(result['boxes']['R0_R'][0], pose[:3, :3].T)

    def test_pitch_roll_pose_and_box_rotation(self):
        geo = _geometry();pose = np.eye(4)
        pose[:3, :3] = geo.quaternion_wxyz_to_matrix([.7, .2, -.3, .1]);pose[:3, 3] = [53., -42., 1.3]
        boxes = [box(center=(53., -42., 1.3), rotation=(.5, .3, -.4, .6)),
                 box(center=(54., -43., 1.8), rotation=(-.8, -.2, .1, .3), score=.7),
                 box(center=(54., -43., 1.8), rotation=(-.8, -.2, .1, .3), score=.7, velocity=(99., 98.))]
        result = compare(boxes, pose, (9, 7, 5), (-4., -3., -2., 4., 3., 2.))
        self.assertTrue(np.any(result['owner'] >= 0));self.assertTrue(np.any(result['owner'] == -1))
        self.assertFalse(np.any(result['owner'] == 2))
        self.assertTrue(np.any(np.abs(result['velocity_R'][..., 2]) > 0))
        for i, b in enumerate(boxes):
            expected = np.linalg.inv(pose) @ geo.box_to_global(b['translation'], b['rotation'])
            np.testing.assert_array_equal(result['boxes']['c0_R'][i], expected[:3, 3])
            np.testing.assert_array_equal(result['boxes']['R0_R'][i], expected[:3, :3])

    def test_origin_and_grid_guard(self):
        with self.assertRaises(ValueError):
            build_full_grid_geometry([], np.eye(4), grid_shape_xyz=(1, 1, 1), extent_xyz=(-1, -1, -1, 1, 1, 1), box_origin='raw_bottom')
        with self.assertRaises(ValueError):
            build_full_grid_geometry([], np.eye(4), grid_shape_xyz=(1, 0, 1), extent_xyz=(-1, -1, -1, 1, 1, 1))


def real_first_dev_anchor():
    """Explicit optional, fixed dev[0]; no batch and no sample selection by score."""
    def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    root = HERE / 'crn_state_dev200_centered_v1';rawroot = HERE / 'motion_targets_v1'
    assert sha(root / 'complete.json') == '650f19ae2709bbb8af39ed8c70dcebb7cbb720b7a6bc5fdfa6fc7040a9361b7d'
    complete = json.loads((root / 'complete.json').read_bytes())
    for name, value in complete['files_sha256'].items(): assert sha(root / name) == value
    assert sha(rawroot / 'manifest.json') == '4d38d8feaa0c200620700fda43377fb20512c6232a30172a728fa558f57f6d71'
    selected = json.loads((root / 'predictions.json').read_bytes())['records'][0]
    descriptor = json.loads((rawroot / 'manifest.json').read_bytes())['records'][512]
    assert selected['ordinal'] == 0 and all(selected[k] == v for k, v in descriptor['identity'].items())
    rawpath = rawroot / descriptor['file'];assert sha(rawpath) == descriptor['sha256']
    rawbytes = gzip.decompress(rawpath.read_bytes());assert hashlib.sha256(rawbytes).hexdigest() == descriptor['uncompressed_json_sha256']
    carrier = json.loads(rawbytes);assert carrier['identity'] == descriptor['identity']
    current = carrier['frames'][2];assert current['relative_frame_index'] == 0 and current['sample_token'] == selected['sample_token']
    G0 = np.asarray(current['lidar_to_global_column_matrix'], dtype=np.float64)
    del carrier, rawbytes, current  # GT tracks/future frames never cross build API.
    started = time.monotonic()
    timings = {}
    result = compare(selected['boxes'], G0, (200, 200, 16), (-51.2, -51.2, -5., 51.2, 51.2, 3.), timings)
    return dict(status='EXACT_REAL_FIRST_DEV_ANCHOR_FULL_GRID',sample_token=selected['sample_token'],
                prediction_sha256=complete['files_sha256']['predictions.json'],raw_sha256=descriptor['sha256'],
                source_sha256=sha(HERE/'crn_full_grid_geometry_v1.py'),reference_sha256=REFERENCE_SHA,
                grid_shape_xyz=[200,200,16],points=640000,boxes=len(selected['boxes']),
                covered_points=int(np.sum(result['owner'] >= 0)),seconds_including_reference=time.monotonic()-started,
                timings=timings,owner_and_velocity_bytes_exact=True,GT_passed_to_build=False,model_inference=False,GPU_used=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-first-dev-anchor', action='store_true')
    args = parser.parse_args()
    if args.real_first_dev_anchor: print(json.dumps(real_first_dev_anchor(), indent=2))
    else:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(FullGridTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful(): sys.exit(1)
        assert 'torch' not in sys.modules
