"""Small owner-mapping tests and optional fixed dev[0] full-grid CPU check."""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
import unittest

import numpy as np
from shared_rigid_prediction_geometry_v1 import geometry_from_centered_boxes, _bound
from test_crn_full_grid_geometry_v1 import box, reference

HERE = Path(__file__).resolve().parent


class AdapterTests(unittest.TestCase):
    def test_noncontiguous_export_indices_and_tie(self):
        boxes = [box(name='barrier'), box(center=(-1., 0., 0.), size=(2., 2., 2.)),
                 box(name='traffic_cone', score=1.),
                 box(center=(-1., 0., 0.), size=(2., 2., 2.), velocity=(90., 90.)),
                 box(center=(2., 0., 0.), size=(2., 2., 2.), name='bus')]
        before = copy.deepcopy(boxes)
        result = geometry_from_centered_boxes(boxes, np.eye(4), grid_shape_xyz=(7, 1, 1), extent_xyz=(-3.5, -.5, -.5, 3.5, .5, .5))
        self.assertEqual(result['retained_original_box_indices'].tolist(), [1, 3, 4])
        self.assertEqual(result['owner_original_indices'].tolist(), [-1, 1, 1, 1, 4, 4, 4])
        self.assertEqual(result['owner'].tolist(), [-1, 0, 0, 0, 2, 2, 2])
        self.assertEqual(boxes, before)
        self.assertIs(result['c0_R'], result['states_numpy']['center'])
        self.assertIs(result['velocity_R'], result['states_numpy']['velocity'])

    def test_empty_and_non_GMO_only(self):
        for boxes in ([], [box(name='barrier')]):
            result = geometry_from_centered_boxes(boxes, np.eye(4), grid_shape_xyz=(2, 1, 3), extent_xyz=(-1., -.5, -1.5, 1., .5, 1.5))
            self.assertEqual(result['c0_R'].shape, (0, 3));self.assertEqual(result['R0_R'].shape, (0, 3, 3))
            self.assertEqual(result['owner'].tolist(), [-1]*6)
            self.assertTrue(np.all(result['cv_velocity_R'] == 0))

    def test_3d_rotation_original_state_and_field_paths(self):
        geo = _bound('motion_geometry.py');inputs = _bound('object_state_prediction_inputs_v1.py')
        pose = np.eye(4);pose[:3, :3] = geo.quaternion_wxyz_to_matrix([.7, .2, -.3, .1]);pose[:3, 3] = [53., -42., 1.3]
        boxes = [box(name='barrier'), box(center=(53., -42., 1.3), rotation=(.5, .3, -.4, .6)),
                 box(name='traffic_cone'),box(center=(54., -43., 1.8), rotation=(-.8, -.2, .1, .3), score=.7)]
        result = geometry_from_centered_boxes(boxes, pose, grid_shape_xyz=(9, 7, 5), extent_xyz=(-4., -3., -2., 4., 3., 2.))
        velocity, original_owner = reference().predicted_velocity_field(boxes, pose, result['points_R'])
        self.assertEqual(result['cv_velocity_R'].tobytes(), velocity.tobytes())
        self.assertEqual(result['owner_original_indices'].tobytes(), original_owner.tobytes())
        states, retained = inputs.current_states_numpy(boxes, pose)
        for key in states:self.assertEqual(states[key].tobytes(), result['states_numpy'][key].tobytes())
        self.assertEqual(retained, result['retained_original_box_indices'].tolist())
        covered = result['owner'] >= 0
        np.testing.assert_array_equal(result['retained_original_box_indices'][result['owner'][covered]], original_owner[covered])
        self.assertTrue(np.any(np.abs(result['velocity_R'][:, 2]) > 0))


def real_first_dev_anchor():
    def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    root = HERE / 'crn_state_dev200_centered_v1';rawroot = HERE / 'motion_targets_v1'
    assert sha(root/'complete.json') == '650f19ae2709bbb8af39ed8c70dcebb7cbb720b7a6bc5fdfa6fc7040a9361b7d'
    complete = json.loads((root/'complete.json').read_bytes())
    for name, value in complete['files_sha256'].items():assert sha(root/name) == value
    assert sha(rawroot/'manifest.json') == '4d38d8feaa0c200620700fda43377fb20512c6232a30172a728fa558f57f6d71'
    record = json.loads((root/'predictions.json').read_bytes())['records'][0]
    descriptor = json.loads((rawroot/'manifest.json').read_bytes())['records'][512]
    assert record['ordinal'] == 0 and all(record[k] == v for k, v in descriptor['identity'].items())
    rawpath = rawroot/descriptor['file'];assert sha(rawpath) == descriptor['sha256']
    content = gzip.decompress(rawpath.read_bytes());assert hashlib.sha256(content).hexdigest() == descriptor['uncompressed_json_sha256']
    carrier = json.loads(content);frame = carrier['frames'][2]
    assert carrier['identity'] == descriptor['identity'] and carrier['ordinal'] == 512
    assert frame['relative_frame_index'] == 0 and frame['sample_token'] == record['sample_token']
    G0 = np.asarray(frame['lidar_to_global_column_matrix'], dtype=np.float64)
    del carrier, frame, content
    tick = time.monotonic();result = geometry_from_centered_boxes(record['boxes'], G0)
    build_seconds = time.monotonic() - tick
    tick = time.monotonic();v, owner = reference().predicted_velocity_field(record['boxes'], G0, result['points_R'])
    reference_seconds = time.monotonic() - tick
    assert result['cv_velocity_R'].tobytes() == v.tobytes() and result['owner_original_indices'].tobytes() == owner.tobytes()
    states, retained = _bound('object_state_prediction_inputs_v1.py').current_states_numpy(record['boxes'], G0)
    assert retained == result['retained_original_box_indices'].tolist()
    for key in states:assert states[key].tobytes() == result['states_numpy'][key].tobytes()
    assert 'torch' not in sys.modules
    return dict(schema='shared-rigid-geometry-real-anchor-check-v1', status='PASS_REAL_DEV0_FULL_GRID',
                sample_token=record['sample_token'], ordinal=0, raw_global_ordinal=512,
                source_sha256=sha(HERE/'shared_rigid_prediction_geometry_v1.py'),test_sha256=sha(Path(__file__)),
                centered_complete_sha256=sha(root/'complete.json'),predictions_sha256=complete['files_sha256']['predictions.json'],
                raw_pose_carrier_sha256=descriptor['sha256'],raw_manifest_sha256=sha(rawroot/'manifest.json'),
                original_owner_bytes_exact=True,CV_field_bytes_exact=True,all_current_state_arrays_bytes_exact=True,
                build_seconds=build_seconds,reference_seconds=reference_seconds,geometry_receipt=result['receipt'],
                NumPy_version=np.__version__,Torch_imported=False,SSH_called=False,GPU_used=False,GT_passed_to_geometry=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--real-first-dev-anchor', action='store_true');a=p.parse_args()
    if a.real_first_dev_anchor: print(json.dumps(real_first_dev_anchor(), indent=2))
    else:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AdapterTests))
        if not result.wasSuccessful():sys.exit(1)
        assert 'torch' not in sys.modules
