"""Execute production detector methods without the CUDA/mmcv model stack.

The small decoder spies only replace expensive external components. The
train/test entry points, current-frame conditioning, future rollout loop, and
scientific evaluator are compiled verbatim from the production detector.
This is integration-contract coverage, not a complete training smoke test.
"""
import ast
import copy
from pathlib import Path
import runpy
import types
import unittest

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


ROOT = Path(__file__).resolve().parents[1]
DETECTOR_PATH = ROOT / ('projects/mmdet3d_plugin/bevformer/detectors/'
                        'drive_occworld.py')
TREE = ast.parse(DETECTOR_PATH.read_text())
DETECTOR = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and
                n.name == 'Drive_OccWorld')


def production_method(name):
    node = copy.deepcopy(next(n for n in DETECTOR.body
                              if isinstance(n, ast.FunctionDef) and n.name == name))
    node.decorator_list = []
    module = ast.Module(body=[node], type_ignores=[])
    scope = dict(torch=torch, F=F, np=np, copy=copy)
    exec(compile(ast.fix_missing_locations(module), str(DETECTOR_PATH), 'exec'), scope)
    return scope[name]


class M3InputContractTest(unittest.TestCase):
    def test_prior_uses_camera_reference_before_support_geometry_fusion(self):
        camera, observations, geometry, fused = object(), object(), object(), object()
        state = dict(geometry=geometry)
        calls = []

        def prepare(camera_arg, observation_arg):
            self.assertIs(camera_arg, camera)
            self.assertIs(observation_arg, observations)
            calls.append('prepare')
            return state

        def fuse(camera_arg, geometry_arg):
            self.assertIs(camera_arg, camera)
            self.assertIs(geometry_arg, geometry)
            calls.append('fuse')
            return fused

        fake = types.SimpleNamespace(
            doppler_posterior=types.SimpleNamespace(prepare=prepare),
            fuse_radar_bev=fuse)
        output, output_state = production_method('condition_reference')(
            fake, camera, radar_observations=observations)
        self.assertIs(output, fused)
        self.assertIs(output_state, state)
        self.assertEqual(calls, ['prepare', 'fuse'])

    def test_missing_observations_or_legacy_velocity_input_fail_before_fusion(self):
        def forbidden(*args, **kwargs):
            self.fail('Invalid M3 inputs reached model computation')

        fake = types.SimpleNamespace(
            doppler_posterior=types.SimpleNamespace(prepare=forbidden),
            fuse_radar_bev=forbidden)
        condition = production_method('condition_reference')
        with self.assertRaisesRegex(ValueError, 'requires radar_observations'):
            condition(fake, object())
        with self.assertRaisesRegex(ValueError, 'per-return observations'):
            condition(fake, object(), radar_bev=object(),
                      radar_observations=object())

    def test_m3_config_disables_legacy_flow_and_cached_velocity_raster(self):
        config = runpy.run_path(str(ROOT / ('projects/configs/radarflowocc/'
                                           'action_condition_GMO_radar_m3.py')))
        model = config['model']
        self.assertFalse(model['turn_on_flow'])
        self.assertFalse(model['turn_on_plan'])
        self.assertIsNone(model['doppler_flow_loss'])
        self.assertIsNone(model['doppler_advection'])
        self.assertTrue(model['scientific_eval'])
        self.assertEqual(model['doppler_posterior']['transport_mode'], 'sigma')
        for split in ('train', 'val', 'test'):
            self.assertIsNone(config['data'][split]['radar_cfg'])
            self.assertEqual(config['data'][split]['radar_observation_cfg']['nsweeps'], 1)
        self.assertIsNone(config['resume_from'])

    def test_physical_observations_are_not_cast_by_legacy_auto_fp16(self):
        method = next(n for n in DETECTOR.body if isinstance(n, ast.FunctionDef)
                      and n.name == 'forward_train')
        decorator = next(d for d in method.decorator_list
                         if isinstance(d, ast.Call) and
                         isinstance(d.func, ast.Name) and d.func.id == 'auto_fp16')
        cast_fields = ast.literal_eval(next(k.value for k in decorator.keywords
                                            if k.arg == 'apply_to'))
        self.assertNotIn('radar_observations', cast_fields)


@unittest.skipIf(torch is None, 'PyTorch required for production rollout/evaluator execution')
class M3ForwardIntegrationTest(unittest.TestCase):
    def make_detector(self):
        batch, height, width, channels = 2, 2, 3, 3
        camera = torch.arange(batch * height * width * channels).reshape(
            batch, height * width, channels).float()
        states, transport_calls, decoder_calls = [], [], []
        detector = types.SimpleNamespace(
            only_generate_dataset=False, random_drop_image_rate=0.0,
            random_drop_prev_rate=0.0, grid_mask_prev=False,
            turn_on_plan=False, turn_on_flow=False,
            doppler_advection=None, doppler_radial_flow_loss=None,
            only_train_cur_frame=False, supervise_all_future=True,
            future_pred_frame_num=2, test_future_frame_num=2,
            memory_queue_len=3, bev_h=height, bev_w=width,
            doppler_nll_weight=0.05, scientific_eval=True, training=True)
        detector.eval = lambda: setattr(detector, 'training', False)
        detector.obtain_history_bev = lambda *args, **kwargs: (
            camera * 0.0, [camera * 0.0, camera * 0.0])
        detector.extract_feat = lambda **kwargs: kwargs['img']
        detector.pts_bbox_head = lambda *args, **kwargs: camera

        def prepare(current_camera, observations):
            self.assertIs(current_camera, camera)
            state = dict(geometry=torch.ones(batch, 8, height, width),
                         loss_doppler_nll=torch.tensor(3.0),
                         phase=detector.training)
            states.append(state)
            return state

        def transport(reference, state, index, grid):
            torch.testing.assert_close(reference, camera + 100.0)
            self.assertIs(state, states[-1])
            prior = torch.full_like(reference, float(index))
            transport_calls.append((state, index, prior))
            return prior

        detector.doppler_posterior = types.SimpleNamespace(
            prepare=prepare, transport=transport)
        detector.fuse_radar_bev = lambda camera_arg, geometry: camera_arg + 100.0
        detector._get_history_ref_to_previous_transform = lambda *args: (
            torch.eye(4).reshape(1, 1, 4, 4).repeat(batch, 3, 1, 1))
        grid = torch.zeros(batch, height * width, 2)
        detector._align_bev_coordnates = lambda *args: (
            None, None, torch.eye(4).unsqueeze(0).repeat(batch, 1, 1), None, grid)

        class Decoder:
            bev_pred_head = [None]
            prev_render_neck = types.SimpleNamespace(sem_norm=False)

            def __call__(self, memory, metadata, index, action, condition, **kwargs):
                prior = kwargs['rollout_prior']
                self_call = (index, prior)
                decoder_calls.append(self_call)
                return (memory[:, -1] + prior).unsqueeze(0), None

            def forward_head(self, features):
                return features

        detector.future_pred_head = Decoder()
        detector.compute_occ_loss = lambda pred, gt: {'loss_occ': pred.sum()}
        detector.evaluate_occ_records = lambda *args: [dict(sample_token='sample0')]

        def legacy_flow_forbidden(*args, **kwargs):
            self.fail('M3 entered a legacy flow or aggregate evaluation path')

        detector.predict_current_flow = legacy_flow_forbidden
        detector.evaluate_instance = legacy_flow_forbidden
        detector.evaluate_occ = legacy_flow_forbidden
        for name in ('condition_reference', 'obtain_ref_bev', 'future_pred',
                     'forward_train', 'forward_test'):
            setattr(detector, name, types.MethodType(production_method(name), detector))
        return detector, states, transport_calls, decoder_calls

    def inputs(self):
        meta = dict(prev_bev_exists=True, sample_idx='sample', scene_token='scene')
        return dict(
            img=torch.zeros(2, 3, 1, 1, 1, 1),
            img_metas=[[dict(meta) for _ in range(3)] for _ in range(2)],
            radar_observations=torch.zeros(2, 4, 9),
            segmentation=torch.zeros(2, 5, 2, 3, 1),
            sdc_planning=torch.zeros(2, 3, 2),
            command=torch.zeros(2, 3), vel_steering=torch.zeros(2, 3, 4))

    def test_train_and_test_use_one_unchanged_state_at_every_future_horizon(self):
        detector, states, transport_calls, decoder_calls = self.make_detector()
        losses = detector.forward_train(**self.inputs())
        self.assertEqual(set(losses), {'loss_occ', 'loss_m3_doppler_nll'})
        self.assertAlmostEqual(losses['loss_m3_doppler_nll'].item(), 0.15, places=6)
        self.assertEqual(len(states), 1)
        self.assertTrue(states[0]['phase'])
        self.assertEqual([call[1] for call in transport_calls], [1, 2])
        for state, _, _ in transport_calls:
            self.assertIs(state, states[0])
        for decoder, transport in zip(decoder_calls, transport_calls):
            self.assertIs(decoder[1], transport[2])

        result = detector.forward_test(**self.inputs())
        self.assertEqual(set(result), {'occ_records'})
        self.assertNotIn('vpq', result)
        self.assertEqual(len(states), 2)
        self.assertFalse(states[1]['phase'])
        self.assertEqual([call[1] for call in transport_calls], [1, 2, 1, 2])
        for state, _, _ in transport_calls[2:]:
            self.assertIs(state, states[1])
        for decoder, transport in zip(decoder_calls[2:], transport_calls[2:]):
            self.assertIs(decoder[1], transport[2])

    def test_missing_motion_state_fails_in_production_future_loop(self):
        detector, _, _, _ = self.make_detector()
        metadata = self.inputs()['img_metas']
        with self.assertRaisesRegex(ValueError, 'posterior state is missing'):
            detector.future_pred(
                torch.zeros(2, 3, 6, 3), {}, {'occ_gts': None},
                {'ref_pose_pred': None, 'gt_traj': torch.zeros(2, 3, 2)},
                [0, 1, 2], [m[-1] for m in metadata], metadata, 3)


@unittest.skipIf(torch is None, 'PyTorch required for scientific evaluation')
class M3ScientificEvaluatorTest(unittest.TestCase):
    @staticmethod
    def prediction_tensor(labels):
        """Encode known voxel labels in the established WorldHead output layout."""
        frames, batch, height, width = labels.shape
        logits = torch.zeros(frames, 2, 1, batch, height * width, 1, 2)
        for t in range(frames):
            for b in range(batch):
                flat = labels[t, b].transpose(0, 1).reshape(-1)
                for cell, label in enumerate(flat.tolist()):
                    logits[t, 1, 0, b, cell, 0, label] = 10
                    # The first decoder layer deliberately predicts opposites.
                    logits[t, 0, 0, b, cell, 0, 1 - label] = 10
        return logits

    def fixture(self):
        labels = torch.tensor([
            [[[0, 0, 1], [1, 0, 1]], [[0, 0, 0], [0, 0, 0]]],
            [[[0, 0, 0], [0, 0, 0]], [[1, 1, 1], [1, 1, 1]]],
            [[[1, 1, 1], [1, 1, 1]], [[0, 0, 0], [0, 0, 0]]]])
        targets = torch.full((2, 5, 2, 3, 1), 255, dtype=torch.long)
        targets[0, 2, :, :, 0] = torch.tensor([[0, 1, 255], [1, 0, 1]])
        targets[0, 3] = 1
        targets[0, 4] = 0
        targets[1, 2] = 0
        targets[1, 3] = 1
        targets[1, 4] = 1
        fake = types.SimpleNamespace(
            bev_h=2, bev_w=3,
            future_pred_head=types.SimpleNamespace(history_queue_length=2))
        metadata = [dict(sample_idx='sample0', scene_token='scene0'),
                    dict(sample_token='sample1', scene_token='scene1')]
        return fake, self.prediction_tensor(labels), targets, metadata

    def test_batch_two_keeps_sample_horizon_identity_and_ignores_255(self):
        fake, predictions, targets, metadata = self.fixture()
        records = production_method('evaluate_occ_records')(
            fake, predictions, targets, metadata)
        self.assertEqual([r['sample_token'] for r in records], ['sample0', 'sample1'])
        self.assertEqual([r['scene_token'] for r in records], ['scene0', 'scene1'])
        self.assertEqual(records[0]['horizon_seconds'], [0.0, 0.5, 1.0])
        np.testing.assert_array_equal(records[0]['hist_by_horizon'], [
            [[2, 0], [1, 2]], [[0, 0], [6, 0]], [[0, 6], [0, 0]]])
        np.testing.assert_array_equal(records[1]['hist_by_horizon'], [
            [[6, 0], [0, 0]], [[0, 0], [0, 6]], [[0, 0], [6, 0]]])
        self.assertEqual(records[0]['hist_by_horizon'][0].sum(), 5)

    def test_missing_identity_or_mismatched_frames_fail(self):
        fake, predictions, targets, metadata = self.fixture()
        evaluator = production_method('evaluate_occ_records')
        with self.assertRaisesRegex(ValueError, 'frames mismatch'):
            evaluator(fake, predictions, targets[:, :-1], metadata)
        with self.assertRaisesRegex(ValueError, 'sample and scene tokens'):
            evaluator(fake, predictions, targets, [{}, metadata[1]])


if __name__ == '__main__':
    unittest.main()
