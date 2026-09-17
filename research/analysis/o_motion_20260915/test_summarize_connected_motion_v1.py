"""Analytic aggregation tests; fabricated unit-test counts, not research data."""
import copy
import unittest

from summarize_connected_motion_v1 import summary


def fixtures():
    anchors = []
    objects = []
    for sample, scene, tp, fn, error, npoints in [('a', 's0', 1, 1, 1., 100), ('b', 's1', 98, 0, 3., 1)]:
        h = [[[0, 0], [fn, tp]] for _ in range(5)]
        anchors.append(dict(sample_token=sample, scene_token=scene,
                            hist_by_arm={arm: copy.deepcopy(h) for arm in ('O', 'J', 'D')}))
        for arm in ('P', 'J', 'D'):
            for horizon in (.5, 1., 1.5, 2.):
                objects.append(dict(arm=arm, sample_token=sample, scene_token=scene,
                    instance_token='car', horizon_seconds=horizon, dt_seconds=horizon,
                    group='moving', source_points=npoints, epe_xy_m=error, epe_3d_m=error,
                    zero_epe_xy_m=4., zero_epe_3d_m=4.))
    return anchors, objects


class ConnectedSummaryTest(unittest.TestCase):
    def test_counts_pool_before_iou_and_equal_objects_before_epe(self):
        a, o = fixtures(); r = summary(a, o, 200)
        self.assertEqual(r['original_occupancy']['O']['future_GMO_IoU_percent'], 99.)
        group = next(x for x in r['physical'] if x['horizon_seconds'] == 2. and x['group'] == 'moving')
        self.assertEqual(group['metrics']['epe_xy_m']['values']['P']['mean'], 2.)
        self.assertEqual(group['source_point_occurrences'], 101)

    def test_identical_paired_models_zero_interval(self):
        a, o = fixtures(); r = summary(a, o, 200)
        for values in r['occupancy_comparisons'].values():
            for v in values.values():
                self.assertEqual((v['difference'], v['lower95'], v['upper95']), (0, 0, 0))
        for group in r['physical']:
            if group['object_anchor_pairs']:
                for m in group['metrics'].values():
                    v = m['comparisons']['J-minus-D']
                    self.assertEqual((v['difference'], v['lower95'], v['upper95']), (0, 0, 0))

    def test_object_support_and_gt_mismatch_rejected(self):
        a, o = fixtures(); o.pop()
        with self.assertRaisesRegex(ValueError, 'Physical support differs'):
            summary(a, o, 20)
        a, o = fixtures(); a[0]['hist_by_arm']['J'][1][1][0] += 1
        with self.assertRaisesRegex(ValueError, 'GT denominators differ'):
            summary(a, o, 20)

    def test_empty_group_is_undefined(self):
        r = summary(*fixtures(), 200)
        x = next(x for x in r['physical'] if x['group'] == 'stationary')
        self.assertIsNone(x['metrics']['epe_xy_m']['values']['P']['mean'])
        c = x['metrics']['epe_xy_m']['comparisons']['J-minus-P']
        self.assertIsNone(c['difference']); self.assertEqual(c['undefined_draws'], 200)

    def test_label_identity_and_duplicate_rejected(self):
        a, o = fixtures(); o[0]['source_points'] += 1
        with self.assertRaisesRegex(ValueError, 'Physical label/support mismatch'):
            summary(a, o, 20)
        a, o = fixtures(); o.append(copy.deepcopy(o[0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate physical identity'):
            summary(a, o, 20)


if __name__ == '__main__':
    unittest.main()
