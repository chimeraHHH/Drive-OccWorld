"""Analytic CPU tests; synthetic grids are tests, never research results.

Optional local raw-label test verifies real serialization and SE(3), not
fine-GT parity: the full GT tensors are not present in this local package.
Run: python3 -m unittest discover -s <this directory> -p test_common_occupancy_change_metrics_v1.py -v
"""
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_occupancy_change_metrics_v1 import (
    evaluate_common_occupancy_change as evaluate, speed_group, validate_raw_label)
from motion_geometry import box_to_global, unique_box_assignment


def make_track(token, centers, sizes=(.8, .8, .8), quaternion=(1., 0., 0., 0.)):
    return dict(instance_token=token, gmo_class=1,
        valid_mask=[c is not None for c in centers],
        global_centers_m=[None if c is None else list(c) for c in centers],
        global_rotations_wxyz=[None if c is None else list(quaternion) for c in centers],
        sizes_wlh_m=[None if c is None else list(sizes) for c in centers])


def raw_label(tracks=(), g0=None, times=(-1., -.5, 0., .5, 1., 1.5, 2.)):
    g0 = np.eye(4) if g0 is None else np.asarray(g0)
    frames = [dict(sequence_index=k, relative_frame_index=k-2,
        sample_token='anchor' if k == 2 else 'frame'+str(k), scene_token='scene',
        timestamp_us=10000000+int(round(t*1000000)), dt_seconds=t,
        lidar_to_global_column_matrix=g0.tolist()) for k, t in enumerate(times)]
    return dict(schema='raw-nuscenes-motion-target-v1', identity=dict(sample_token='anchor',
        scene_token='scene', split='train', official_index=0), frames=frames,
        tracks=list(tracks), missing_fill_applied=False, old_refine_applied=False,
        visibility_or_point_count_filter_applied=False, spatial_ROI_filter_applied=False,
        instance_union='all GMO instances in seven original sample annotations')


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.shape = (5, 4, 3, 2)
        self.extent = [0, 0, 0, 4, 3, 2]
        self.gt = np.zeros(self.shape, dtype=np.uint8)
        self.gt[0, 0, 0, 0] = 1
        self.gt[1, 1, 0, 0] = 1
        self.gt[2, 2, 1, 0] = 1
        self.gt[3, 2, 1, 0] = 1
        self.gt[4, 3, 2, 1] = 1

    def test_GT_copy_and_transition_all_classes(self):
        result = evaluate(self.gt == 1, self.gt, self.extent, raw_label())
        for row in result['horizons']:
            h = np.asarray(row['occupancy']['confusion'])
            self.assertEqual(int(h[0, 1]+h[1, 0]), 0)
        for row in result['transitions']:
            h = np.asarray(row['confusion'])
            self.assertEqual(int(h.sum()-h.trace()), 0)
            self.assertEqual(row['classes']['01']['IoU'], 1.)
            self.assertEqual(row['classes']['10']['IoU'], 1.)
            self.assertNotIn('accuracy', row)

    def test_all_empty_and_all_occupied_penalized(self):
        empty = evaluate(np.zeros(self.shape, bool), self.gt, self.extent, raw_label())
        full = evaluate(np.ones(self.shape, bool), self.gt, self.extent, raw_label())
        for a, b in zip(empty['horizons'], full['horizons']):
            self.assertEqual(a['occupancy']['classes']['1']['FN'], 1)
            self.assertEqual(a['occupancy']['classes']['1']['IoU'], 0.)
            self.assertEqual(b['global_negative'], dict(GT=23, FP=23, TN=0))
        for row in empty['transitions']:
            self.assertEqual(row['classes']['01']['FN'], 1)
            self.assertEqual(row['classes']['10']['FN'], 1)
        self.assertIsNone(empty['horizons'][0]['occupancy']['classes']['1']['precision'])

    def test_copy_t0_misses_arrival_and_vacating(self):
        prediction = np.broadcast_to(self.gt[0] == 1, self.shape).copy()
        result = evaluate(prediction, self.gt, self.extent, raw_label())
        for row in result['transitions']:
            self.assertEqual(row['confusion'][1][0], 1)  # arrival predicted still empty
            self.assertEqual(row['confusion'][2][3], 1)  # vacated site predicted persistent
            self.assertEqual(row['classes']['01']['recall'], 0.)
            self.assertEqual(row['classes']['10']['recall'], 0.)

    def test_added_false_arrival_and_false_vacating_both_count(self):
        pred = self.gt == 1
        pred[0, 3, 0, 0] = True
        pred[1, 3, 1, 0] = True
        row = evaluate(pred, self.gt, self.extent, raw_label())['transitions'][0]
        self.assertEqual(row['classes']['01']['FP'], 1)
        self.assertEqual(row['classes']['10']['FP'], 1)
        self.assertEqual(row['confusion'][0][2], 1)
        self.assertEqual(row['confusion'][0][1], 1)

    def test_joint_ignore_domain_is_separate_from_native_horizon(self):
        gt = np.zeros(self.shape, np.uint8)
        gt[0, 0, 0, 0] = 255; gt[1, 0, 0, 0] = 1
        gt[0, 1, 0, 0] = 1; gt[1, 1, 0, 0] = 255
        gt[:, 2, 0, 0] = 255
        pred = gt == 1
        result = evaluate(pred, gt, self.extent, raw_label())
        row = result['transitions'][0]
        self.assertEqual(row['domain'], dict(grid_voxels=24, both_valid=21,
            t0_valid_h_ignored=1, t0_ignored_h_valid=1, both_ignored=1,
            t0_valid=22, h_valid=22, t0_foreground_excluded=1, h_foreground_excluded=1))
        self.assertEqual(result['horizons'][1]['occupancy']['classes']['1']['TP'], 1)
        self.assertEqual(row['valid_voxels'], 21)

    def test_all_ignored_empty_union(self):
        gt = np.full(self.shape, 255, np.uint8)
        result = evaluate(np.ones(self.shape, bool), gt, self.extent, raw_label())
        self.assertEqual(result['horizons'][0]['occupancy']['valid_voxels'], 0)
        for row in result['transitions']:
            self.assertEqual(row['valid_voxels'], 0)
            self.assertTrue(all(c['IoU'] is None for c in row['classes'].values()))

    def test_future_box_assignment_source_missing_and_current_missing(self):
        centers = [[.5, .5, .5]]*3 + [[1.5, .5, .5]]*4
        moved = make_track('moving', centers)
        future_only = make_track('new_annotation', [None]*3+[[2.5, .5, .5]]*4)
        gap = make_track('t0_gap', [[3.5, .5, .5], None, None]+[[3.5, .5, .5]]*4)
        gone = make_track('missing_current', [[.5, 1.5, .5]]*3+[None]*4)
        gt = np.zeros(self.shape, np.uint8)
        gt[1:, 1:4, 0, 0] = 1
        gt[1:, 0, 1, 0] = 1  # old location not attributed with filled old box
        result = evaluate(gt == 1, gt, self.extent, raw_label([moved, future_only, gap, gone]))
        a = result['horizons'][1]['motion_positive_attribution']
        self.assertEqual(a['groups']['speed_gt_0.5_le_5']['GT'], 1)
        self.assertEqual(a['groups']['annotated_future_only']['GT'], 1)
        self.assertEqual(a['groups']['t0_missing_with_history']['GT'], 1)
        self.assertEqual(a['groups']['unknown_no_current_box']['GT'], 1)
        self.assertEqual(a['missing_current_instance_count'], 1)
        ledger = {r['instance_token']: r for r in a['instance_ledger']}
        self.assertEqual(ledger['moving']['global_xy_endpoint_speed_mps'], 2.)
        self.assertEqual(ledger['missing_current']['status'], 'missing_current_frame')
        self.assertIsNone(ledger['missing_current']['recall'])

    def test_static_hit_not_a_moving_FP_and_no_group_duplicates(self):
        tracks = [make_track('still', [[.5, .5, .5]]*7),
                  make_track('moving', [[1.5, 1.5, .5]]*3+[[2.5, 1.5, .5]]*4)]
        gt = np.zeros(self.shape, np.uint8); gt[:, 0, 0, 0] = 1; gt[1:, 2, 1, 0] = 1
        pred = np.zeros(self.shape, bool); pred[:, 0, 0, 0] = True; pred[1:, 3, 2, 1] = True
        result = evaluate(pred, gt, self.extent, raw_label(tracks))
        for row in result['horizons'][1:]:
            groups = row['motion_positive_attribution']['groups']
            self.assertEqual(groups['speed_le_0.1']['TP'], 1)
            self.assertEqual(sum(g['FN'] for g in groups.values()), 1)
            self.assertEqual(row['global_negative']['FP'], 1)
            self.assertTrue(all('FP' not in g for g in groups.values()))
            self.assertEqual(sum(g['GT'] for g in groups.values()), 2)

    def test_overlap_and_unknown_not_first_box_or_static(self):
        tracks = [make_track('a', [[.5, .5, .5]]*7), make_track('b', [[.5, .5, .5]]*7)]
        gt = np.zeros(self.shape, np.uint8); gt[:, 0, 0, 0] = 1; gt[:, 3, 2, 1] = 1
        row = evaluate(gt == 1, gt, self.extent, raw_label(tracks))['horizons'][1]
        groups = row['motion_positive_attribution']['groups']
        self.assertEqual(groups['overlap_current_boxes']['GT'], 1)
        self.assertEqual(groups['unknown_no_current_box']['GT'], 1)
        self.assertEqual(groups['speed_le_0.1']['GT'], 0)

    def test_non_square_XYZ_voxel_centers_real_SE3_and_global_xy_speed(self):
        # Rotated R has global X=Rz, global Y=Ry, global Z=-Rx.
        # Rotation makes local xy speed differ from required global xy speed.
        q = [np.sqrt(.5), 0., np.sqrt(.5), 0.]
        G0 = box_to_global([100., -20., 3.], q)
        r0 = np.array([1., 1.5, 3.]); rh = np.array([5., 4.5, 9.])
        global0 = G0[:3, :3] @ r0 + G0[:3, 3]
        globalh = G0[:3, :3] @ rh + G0[:3, 3]
        raw = raw_label([make_track('rotated', [global0]*3+[globalh]*4, quaternion=q)],
            g0=G0, times=(-1., -.5, 0., .4, 1., 1.5, 2.))
        # Future ego is unrelated to fixed-R GT attribution and speed.
        raw['frames'][3]['lidar_to_global_column_matrix'] = box_to_global([700., 50., -30.], [0.,0.,0.,1.]).tolist()
        gt = np.zeros(self.shape, np.uint8); gt[0, 0, 0, 0] = 1; gt[1:, 2, 1, 1] = 1
        row = evaluate(gt == 1, gt, [0,0,0,8,9,12], raw)['horizons'][1]
        a = row['motion_positive_attribution']
        self.assertEqual(a['unique_box_voxels'], 1)
        self.assertEqual(a['groups']['speed_gt_5']['TP'], 1)
        self.assertAlmostEqual(a['instance_ledger'][0]['global_xy_endpoint_speed_mps'], np.sqrt(45.)/.4)
        self.assertNotAlmostEqual(a['instance_ledger'][0]['global_xy_endpoint_speed_mps'], 12.5)
        self.assertEqual(row['actual_dt_seconds'], .4)

    def test_speed_boundaries(self):
        self.assertEqual([speed_group(s) for s in [0., .1, np.nextafter(.1,1), .5,
            np.nextafter(.5,1),5.,np.nextafter(5.,6)]],
            ['speed_le_0.1','speed_le_0.1','speed_gt_0.1_le_0.5','speed_gt_0.1_le_0.5',
             'speed_gt_0.5_le_5','speed_gt_0.5_le_5','speed_gt_5'])

    def test_input_immutable_json_safe_and_t0_cross_model_digest(self):
        gt = self.gt.copy(); pred = gt == 1; raw = raw_label()
        before = json.dumps(raw, sort_keys=True); gt0 = gt.copy(); pred0 = pred.copy()
        a = evaluate(pred, gt, self.extent, raw)
        changed = pred.copy(); changed[1:, 3, 0, 0] = True
        b = evaluate(changed, gt, self.extent, raw)
        self.assertEqual(a['t0_boundary'], b['t0_boundary'])
        changed[0, 3, 0, 0] = True
        c = evaluate(changed, gt, self.extent, raw)
        self.assertNotEqual(a['t0_boundary']['prediction_full_binary_sha256'],
                            c['t0_boundary']['prediction_full_binary_sha256'])
        self.assertTrue(np.array_equal(gt, gt0) and np.array_equal(pred, pred0))
        self.assertEqual(json.dumps(raw, sort_keys=True), before)
        json.dumps(a, allow_nan=False)

    def test_reject_invalid_or_filled_labels(self):
        with self.assertRaises(ValueError): evaluate(self.gt, self.gt, self.extent, raw_label())
        bad = self.gt.copy(); bad[0,0,0,0] = 2
        with self.assertRaises(ValueError): evaluate(bad == 1, bad, self.extent, raw_label())
        raw = raw_label(); raw['frames'][3]['dt_seconds'] = .6
        with self.assertRaises(ValueError): evaluate(self.gt == 1, self.gt, self.extent, raw)
        raw = raw_label(); raw['missing_fill_applied'] = True
        with self.assertRaises(ValueError): evaluate(self.gt == 1, self.gt, self.extent, raw)
        raw = raw_label([make_track('missing',[None]*3+[[.5,.5,.5]]*4)])
        raw['tracks'][0]['global_centers_m'][2] = [0,0,0]
        with self.assertRaises(ValueError): evaluate(self.gt == 1, self.gt, self.extent, raw)

    def test_real_training_label_only_no_claim_of_fineGT_parity(self):
        base = Path(__file__).resolve().parent/'motion_targets_v1'
        manifest_path = base/'manifest.json'
        if not manifest_path.exists():
            self.skipTest('No local raw-label package; no synthetic substitute for real-label test')
        manifest = json.loads(manifest_path.read_text())
        desc = next(r for r in manifest['records'] if r['identity']['split'] == 'train')
        path = base/desc['file']; data = path.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), desc['sha256'])
        raw = json.loads(gzip.decompress(data)); info = validate_raw_label(raw)
        self.assertEqual(info['identity'], desc['identity'])
        # Real pose/box serialization check; these are analytic box-centre
        # points, NOT actual occupancy voxels or a real fullGT evaluation.
        inverse = np.linalg.inv(info['G0'])
        for track in info['tracks']:
            for k in range(2,7):
                if not track['valid_mask'][k]: continue
                box = inverse @ box_to_global(track['global_centers_m'][k], track['global_rotations_wxyz'][k])
                own = unique_box_assignment(box[None,:3,3], box[None],
                    np.asarray(track['sizes_wlh_m'][k],dtype=float)[None])
                self.assertEqual(own.tolist(), [0])


if __name__ == '__main__':
    unittest.main()
