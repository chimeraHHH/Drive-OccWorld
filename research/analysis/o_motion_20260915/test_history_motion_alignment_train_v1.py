"""Synthetic analytic checks only; no real images/labels/results are opened."""
import contextlib
import io
import unittest

import numpy as np


def run_analytic_qa(core, geometry, helper):
    def track(valid=True, linked=True):
        return dict(instance_token='synthetic_instance', valid_mask=[True, valid, True, True, True, True, True],
            annotation_tokens=['older', 'past' if valid else None, 'now', 'f1', 'f2', 'f3', 'f4'],
            annotation_next_tokens=['past', 'now' if linked else 'different', 'f1', 'f2', 'f3', 'f4', 'next'],
            annotation_prev_tokens=['prev', 'older', 'past', 'now', 'f1', 'f2', 'f3'],
            global_centers_m=[[0., 0., 0.], [-1., 0., 0.], [0., 0., 0.], [.5, 0., 0.],
                              [1., 0., 0.], [1.5, 0., 0.], [2., 0., 0.]],
            global_rotations_wxyz=[[1., 0., 0., 0.] for _ in range(7)])

    class Cases(unittest.TestCase):
        def test_signed_past_interval_and_translation(self):
            p = np.asarray([[2., 3., 4.], [-1., 0., 2.]])
            a0 = np.eye(4); previous = np.eye(4); previous[0, 3] = -1.
            b, historical, predicted = core.past_benefit(p, np.eye(4), a0, previous,
                np.asarray([[2., 0., 0.], [2., 0., 0.]]), -.5, geometry)
            np.testing.assert_array_equal(historical, [[-1., 0., 0.], [-1., 0., 0.]])
            np.testing.assert_array_equal(predicted, historical)
            np.testing.assert_array_equal(b, [1., 1.])
            bad_b, _, _ = core.past_benefit(p, np.eye(4), a0, previous,
                np.asarray([[-2., 0., 0.], [-2., 0., 0.]]), -.5, geometry)
            np.testing.assert_array_equal(bad_b, [-1., -1.])
            with self.assertRaises(ValueError):
                core.past_benefit(p, np.eye(4), a0, previous, np.zeros_like(p), .5, geometry)

        def test_full3d_rotation_and_material_point_not_center(self):
            p = np.asarray([[1., 0., 2.]])
            # R->global rotates +90deg about y; global [2,0,-1] is the point.
            g0 = np.eye(4); g0[:3, :3] = [[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]
            a0 = np.eye(4); ap = np.eye(4)
            # Object rotates +90deg about global z. Centre stays fixed.
            ap[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
            b, h, _ = core.past_benefit(p, g0, a0, ap, np.zeros_like(p), -.4, geometry)
            # global [0,2,-1] -> R [1,2,0], so displacement [0,2,-2].
            np.testing.assert_allclose(h, [[0., 2., -2.]], rtol=0, atol=1e-14)
            np.testing.assert_array_equal(b, [0.])

        def test_missing_link_no_minus2_fallback(self):
            self.assertEqual(core.past_link_reason(track()), 0)
            self.assertEqual(core.past_link_reason(track(valid=False)), 1)
            self.assertEqual(core.past_link_reason(track(linked=False)), 2)
            broken = track(); broken['annotation_prev_tokens'][2] = 'different'
            self.assertEqual(core.past_link_reason(broken), 2)

        def test_history_score_does_not_consume_future_pose(self):
            tr = track(); frames = [dict(relative_frame_index=i, timestamp_us=1000000 + i * 500000,
                        dt_seconds=i * .5, lidar_to_global_column_matrix=np.eye(4).tolist()) for i in range(-2, 5)]
            raw = dict(frames=frames, tracks=[tr])
            h = np.asarray(core.HORIZONS); velocity = np.asarray([[2., 0., 0.]])
            motion = dict(object_index=np.asarray([0]), source_flat_indices=np.asarray([0]),
                          instance_tokens=np.asarray(['synthetic_instance']), D_displacement_m=h[:, None, None] * velocity[None])
            points = np.asarray([[1., 2., 3.]])
            before = core.history_labels(raw, motion, points, geometry)
            for i in range(3, 7):
                tr['global_centers_m'][i] = [float('nan')] * 3
                tr['global_rotations_wxyz'][i] = None
            after = core.history_labels(raw, motion, points, geometry)
            for left, right in zip(before, after):
                np.testing.assert_array_equal(left, right)
            np.testing.assert_array_equal(before[1], [1.])

        def test_forward_target_reconstruction_gate_rejects_change(self):
            raw = dict(frames=[{}, {}, dict(lidar_to_global_column_matrix=np.eye(4).tolist())], tracks=[track()])
            motion = dict(instance_tokens=np.asarray(['synthetic_instance']), object_index=np.asarray([0, 0]),
                source_flat_indices=np.asarray([0, 1]), object_future_valid=np.ones((4, 1), dtype=bool),
                valid=np.ones((4, 2), dtype=bool), target_displacement_m=np.zeros((4, 2, 3), dtype=np.float32))
            motion['target_displacement_m'][3, :, 0] = 2.
            points = np.asarray([[0., 0., 0.], [1., 2., 3.]])
            gate = core.forward_2s_gate(raw, motion, points, geometry)
            self.assertEqual(gate['float32_bitwise_equal_points'], 2)
            motion['target_displacement_m'][3, 0, 0] += .01
            with self.assertRaises(ValueError):
                core.forward_2s_gate(raw, motion, points, geometry)

        def test_full_object_denominator_missing_and_covered_retained(self):
            b = np.asarray([2., -2., -4., 1.]); w = np.asarray([.25, .25, .5, 1.]); obj = np.asarray([0, 0, 0, 1])
            whole = core.empty_total(); core.accumulate(whole, b, w, obj, np.ones(4, dtype=bool))
            full = core.finish_total(whole, 2)
            self.assertEqual(full['FULL_denominator_positive_gain_xy_m'], .75)
            self.assertEqual(full['FULL_denominator_negative_cost_xy_m'], 1.25)
            part = core.empty_total(); core.accumulate(part, b, w, obj, np.asarray([True, False, False, False]))
            subset = core.finish_total(part, 2)
            self.assertEqual(subset['FULL_denominator_positive_gain_xy_m'], .25)
            self.assertEqual(subset['conditional_original_weight_mean_benefit_xy_m'], 2.)

        def test_common_population_and_explicit_zero_sign_table(self):
            e = dict(benefit=np.asarray([1., -1., 0., 3.]), past_benefit=np.asarray([2., -2., 0., np.nan]),
                     weight=np.asarray([.25, .25, .5, 1.]), object=np.asarray([0, 0, 0, 1]),
                     reason=np.asarray([0, 0, 0, 4]), past_reason=np.asarray([0, 0, 0, 1]),
                     camera_features=np.asarray([[1., -1., 2.], [-1., 1., 1.], [0., 0., 0.], [np.nan, np.nan, 1.]]))
            mask=np.asarray([True, True, True, False]); result=core.aucs(helper, e, mask)
            self.assertEqual(result['past_benefit_positive']['score_true']['auc'], 1.)
            self.assertEqual(result['past_benefit_positive']['score_broken']['auc'], 0.)
            self.assertEqual(result['future_benefit_positive']['Bpast_oracle_reference']['auc'], 1.)
            self.assertEqual(result['common_label_counts']['past']['zero_points'], 1)
            table=core.sign_table(helper, e, mask, 2)
            self.assertEqual(table['zero']['zero']['weight_sum'], .5)
            self.assertEqual(sum(v['points'] for row in table.values() for v in row.values()), 3)
            with self.assertRaises(ValueError):
                core.aucs(helper, e, np.ones(4, dtype=bool))

        def test_static_label_shared_D_magnitude_and_fixed_bin_boundaries(self):
            p=np.asarray([[0., 0., 0.], [1., 2., 3.]])
            v=np.asarray([[2., 0., 3.], [0., 4., 0.]])
            b, _, pred=core.past_benefit(p, np.eye(4), np.eye(4), np.eye(4), v, -.5, geometry)
            np.testing.assert_array_equal(b, -np.linalg.norm(pred[:, :2], axis=1))
            np.testing.assert_array_equal(np.searchsorted(core.EDGES, [0., .1, .5, 1., 2., 5., 10.], side='right')-1, np.arange(7))

    buffer=io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result=unittest.TextTestRunner(stream=buffer, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Cases))
    if not result.wasSuccessful():
        raise AssertionError(buffer.getvalue())
    return dict(schema='history-motion-alignment-synthetic-qa-v1', status='PASS_SYNTHETIC_ANALYTIC_QA_ONLY',
                tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                cases=unittest.defaultTestLoader.getTestCaseNames(Cases),
                real_data_opened=False, real_alignment_statistics_computed=False, model_forward=False,
                optimizer_updates=0, scene_bootstrap=False, note='Synthetic geometry/support checks, not dataset evidence.')
