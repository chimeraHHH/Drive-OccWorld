"""CPU-only contract tests; AST isolation avoids importing the CUDA plugin."""
import ast
import copy
import inspect
from pathlib import Path
import random
from types import SimpleNamespace
import unittest

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / 'projects/mmdet3d_plugin/datasets'


class DataContainer:
    def __init__(self, data, cpu_only=False):
        self.data = data
        self.cpu_only = cpu_only


class Registry:
    def __init__(self):
        self.classes = {}

    def get(self, name):
        return self.classes.get(name)


def isolated_class(path, name, namespace):
    """Execute the actual source class without package imports/decorators."""
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == name)
    node.decorator_list = []
    node.bases = []
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace[name]


REGISTRY = Registry()
Dataset = isolated_class(
    DATASETS / 'nuscenes_world_dataset_template.py',
    'NuScenesWorldDatasetTemplate',
    dict(np=np, copy=copy, DC=DataContainer, PIPELINES=REGISTRY))
PhotoMetric = isolated_class(
    DATASETS / 'pipelines/transform_3d.py',
    'PhotoMetricDistortionMultiViewImage',
    dict(np=np, random=np.random, mmcv=SimpleNamespace(
        bgr2hsv=lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2HSV),
        hsv2bgr=lambda img: cv2.cvtColor(img, cv2.COLOR_HSV2BGR))))
CropResize = isolated_class(
    DATASETS / 'pipelines/augmentation.py', 'CropResizeFlipImage',
    dict(np=np, random=random, Image=Image))

TRAIN_NAMES = (
    'LoadMultiViewImageFromFiles', 'PhotoMetricDistortionMultiViewImage',
    'CropResizeFlipImage', 'NormalizeMultiviewImage', 'PadMultiViewImage',
    'LoadOccupancy', 'DefaultFormatBundle3D', 'CustomCollect3D')
for _name in TRAIN_NAMES:
    REGISTRY.classes[_name] = type(_name, (), {})
REGISTRY.classes['PhotoMetricDistortionMultiViewImage'] = PhotoMetric
REGISTRY.classes['CropResizeFlipImage'] = CropResize


def make_info(index=10, cameras=6):
    return dict(
        scene_token='scene-a',
        ego2global_translation=[float(index), 2.0, 3.0],
        ego2global_rotation=[1.0, 0.0, 0.0, 0.0],
        lidar2ego_translation=[0.3, 0.0, 1.8],
        lidar2ego_rotation=[1.0, 0.0, 0.0, 0.0],
        can_bus=np.arange(18, dtype=np.float64) + index,
        vel_steering=np.array([1., 2., 3., 4.]),
        img_filename=['unused-{}.jpg'.format(i) for i in range(cameras)])


def make_pipeline(height=32, width=48, resize=None, flip=True):
    photo = PhotoMetric()
    crop = CropResize(
        data_aug_conf=dict(
            reisze=resize or [height], crop=(0, 0, width, height),
            H=height, W=width, rand_flip=flip),
        training=True, debug=False)
    transforms = []
    for name in TRAIN_NAMES:
        if name == 'PhotoMetricDistortionMultiViewImage':
            transform = photo
        elif name == 'CropResizeFlipImage':
            transform = crop
        else:
            transform = REGISTRY.get(name)()
        transforms.append(transform)
    transforms[-1].meta_keys = Dataset._FUTURE_META_KEYS
    transforms[-1].keys = ['img', 'vel_steering']
    return SimpleNamespace(transforms=transforms), photo, crop


def make_dataset(fast=True, pipeline=None):
    dataset = Dataset.__new__(Dataset)
    dataset.future_metadata_only = fast
    dataset.pipeline = pipeline or make_pipeline()[0]
    dataset._future_metadata_rng_transforms = (
        dataset._validate_future_metadata_pipeline() if fast else ())
    dataset.get_data_info = make_info
    return dataset


def fail_if_called(*args, **kwargs):
    raise AssertionError('The future metadata path attempted heavy loading')


class FutureMetadataOnlyTest(unittest.TestCase):
    def assert_numpy_rng_equal(self, expected, actual):
        self.assertEqual(expected[0], actual[0])
        np.testing.assert_array_equal(expected[1], actual[1])
        self.assertEqual(expected[2:], actual[2:])

    def assert_metadata_equal(self, expected, actual):
        self.assertEqual(set(actual), set(Dataset._FUTURE_META_KEYS))
        for key in Dataset._FUTURE_META_KEYS:
            if isinstance(expected[key], np.ndarray):
                np.testing.assert_array_equal(expected[key], actual[key])
            else:
                self.assertEqual(expected[key], actual[key])

    def test_default_is_opt_out_and_config_only_enables_training(self):
        self.assertIs(
            inspect.signature(Dataset.__init__).parameters[
                'future_metadata_only'].default, False)
        config = {}
        path = ROOT / ('projects/configs/radarflowocc/'
                       'action_condition_GMO_radar_m1_fast_input.py')
        exec(compile(path.read_text(), str(path), 'exec'), config)
        self.assertEqual(config['_base_'], ['./action_condition_GMO_radar_m1.py'])
        self.assertEqual(config['data'], dict(train=dict(future_metadata_only=True)))
        self.assertNotIn('model', config)
        self.assertNotIn('optimizer', config)

    def test_metadata_contract_uses_get_data_info_without_loading(self):
        dataset = make_dataset()
        source = make_info()
        calls = []

        def get_data_info(index):
            calls.append(index)
            # Represent normalization performed by the original getter.
            source['can_bus'][-1] = 73.0
            return source

        dataset.get_data_info = get_data_info
        dataset.pipeline = fail_if_called
        dataset.radar_bev_loader = fail_if_called
        dataset.pre_pipeline = fail_if_called
        dataset.get_future_bboxes = fail_if_called
        dataset.record_instance = fail_if_called
        dataset.get_trajectory_sampling = fail_if_called
        result = dataset._prepare_future_data_info(10)
        self.assertEqual(calls, [10])
        self.assertEqual(set(result), {'img_metas', 'vel_steering'})
        self.assertTrue(result['img_metas'].cpu_only)
        self.assert_metadata_equal(source, result['img_metas'].data)
        self.assertEqual(result['img_metas'].data['can_bus'][-1], 73.0)
        np.testing.assert_array_equal(result['vel_steering'], source['vel_steering'])

    def test_missing_data_info_returns_none_without_rng_consumption(self):
        dataset = make_dataset()
        dataset.get_data_info = lambda index: None
        expected_np, expected_py = np.random.get_state(), random.getstate()
        self.assertIsNone(dataset._prepare_future_data_info(10))
        self.assert_numpy_rng_equal(expected_np, np.random.get_state())
        self.assertEqual(expected_py, random.getstate())

    def prepare_routes(self, fast):
        dataset = make_dataset(fast=fast)
        dataset.queue_length = 2
        dataset.future_length = 4
        dataset.data_infos = [None] * 30
        full_calls, metadata_calls = [], []

        def full(index, occ_load_flag=None, aug_param=None):
            full_calls.append((index, occ_load_flag))
            return dict(index=index, aug_param={'shared': 'history-augmentation'})

        def getter(index):
            metadata_calls.append(index)
            return make_info(index)

        dataset._prepare_data_info_single = full
        dataset.get_data_info = getter
        dataset.union2one = lambda previous, future: (previous, future)
        queues = dataset._prepare_data_info(10, rand_interval=1)
        return queues, full_calls, metadata_calls

    def test_history_current_pipeline_preserved_and_future_bypasses_it(self):
        queues, full_calls, metadata_calls = self.prepare_routes(True)
        self.assertEqual(full_calls, [(8, False), (9, False), (10, True)])
        self.assertEqual(metadata_calls, [10, 11, 12, 13, 14])
        self.assertEqual([item['index'] for item in queues[0]], [8, 9, 10])
        self.assertEqual(len(queues[1]), 5)
        self.assertTrue(all('img' not in item for item in queues[1]))

    def test_flag_disabled_keeps_all_eight_original_pipeline_calls(self):
        queues, full_calls, metadata_calls = self.prepare_routes(False)
        self.assertEqual(full_calls, [
            (8, False), (9, False), (10, True), (10, False), (11, False),
            (12, False), (13, False), (14, False)])
        self.assertEqual(metadata_calls, [])
        self.assertEqual([len(queue) for queue in queues], [3, 5])

    def test_unknown_or_reordered_pipeline_fails_closed(self):
        pipeline, _, _ = make_pipeline()
        pipeline.transforms.insert(1, SimpleNamespace())
        with self.assertRaisesRegex(ValueError, 'disable it'):
            make_dataset(pipeline=pipeline)
        pipeline, _, _ = make_pipeline()
        pipeline.transforms[1:3] = reversed(pipeline.transforms[1:3])
        with self.assertRaisesRegex(ValueError, 'disable it'):
            make_dataset(pipeline=pipeline)

    def test_same_named_unregistered_transform_is_rejected(self):
        pipeline, _, _ = make_pipeline()
        pipeline.transforms[0] = type('LoadMultiViewImageFromFiles', (), {})()
        with self.assertRaisesRegex(ValueError, 'disable it'):
            make_dataset(pipeline=pipeline)

    def test_debug_crop_or_missing_required_collect_keys_are_rejected(self):
        pipeline, _, crop = make_pipeline()
        crop.debug = True
        with self.assertRaisesRegex(ValueError, 'debug=False'):
            make_dataset(pipeline=pipeline)
        pipeline, _, _ = make_pipeline()
        pipeline.transforms[-1].meta_keys = ['scene_token']
        with self.assertRaisesRegex(ValueError, 'pose metadata'):
            make_dataset(pipeline=pipeline)
        pipeline, _, _ = make_pipeline()
        pipeline.transforms[-1].keys = ['img']
        with self.assertRaisesRegex(ValueError, 'vel_steering'):
            make_dataset(pipeline=pipeline)

    def check_rng_parity(self, seed, height, width, resize, cameras=6,
                         frames=1, flip=True):
        pipeline, photo, crop = make_pipeline(height, width, resize, flip)
        dataset = make_dataset(pipeline=pipeline)
        dataset.get_data_info = lambda index: make_info(index, cameras)
        np.random.seed(seed)
        random.seed(seed + 100)
        start_np, start_py = np.random.get_state(), random.getstate()
        for frame in range(frames):
            source = make_info(frame, cameras)
            original_meta = copy.deepcopy(source)
            source.update(
                img=[np.full((height, width, 3), 127, dtype=np.float32)
                     for _ in range(cameras)],
                cam2img=[np.eye(4) for _ in range(cameras)],
                lidar2cam=[np.eye(4) for _ in range(cameras)])
            # Run both real image transforms, including actual resize/flip.
            source = crop(photo(source))
            self.assert_metadata_equal(
                original_meta, {key: source[key] for key in Dataset._FUTURE_META_KEYS})
        expected_np, expected_py = np.random.get_state(), random.getstate()
        np.random.set_state(start_np)
        random.setstate(start_py)
        for frame in range(frames):
            result = dataset._prepare_future_data_info(frame)
            self.assert_metadata_equal(make_info(frame, cameras), result['img_metas'].data)
        self.assert_numpy_rng_equal(expected_np, np.random.get_state())
        self.assertEqual(expected_py, random.getstate())

    def test_original_transform_rng_parity_across_seeds_and_camera_counts(self):
        for seed in range(12):
            with self.subTest(seed=seed):
                self.check_rng_parity(
                    seed, 24, 40, [16, 24, 32],
                    cameras=1 if seed % 2 else 6, frames=5,
                    flip=bool(seed % 3))

    def test_original_transform_rng_parity_at_nuscenes_image_size(self):
        self.check_rng_parity(7, 900, 1600, [720], cameras=6)


if __name__ == '__main__':
    unittest.main()
