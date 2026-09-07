"""Protocol/path checks use the standard library; cache manifests add NumPy."""
import importlib.util
import copy
import json
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('m3_h200_config', ROOT / 'tools/m3_prepare_h200_config.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def plain_config(path):
    return {key: value for key, value in runpy.run_path(str(path)).items()
            if not key.startswith('__')}


class H200ConfigTest(unittest.TestCase):
    def setUp(self):
        self.reference = plain_config(ROOT / 'DOCS/m3_h200_20260907/l40s_resolved_reference.py')
        self.candidate = plain_config(ROOT / 'DOCS/m3_20260906/resolved_m3_l40s.py')

    def config(self, **paths):
        values = dict(data_root='/mounted/data', occ_root='/mounted/occupancy',
                      pretrained='/mounted/fcos3d.pth', work_dir='/new/output')
        values.update(paths)
        return MODULE.make_config(self.candidate, self.reference, **values)

    def test_common_model_and_training_pipeline_come_from_actual_l40s_run(self):
        actual = self.config()
        for key, value in self.reference['model'].items():
            if key not in MODULE.METHOD_MODEL_FIELDS:
                self.assertEqual(actual['model'][key], value)
        self.assertEqual(actual['optimizer'], self.reference['optimizer'])
        self.assertEqual(actual['optimizer_config']['grad_clip'],
                         self.reference['optimizer_config']['grad_clip'])
        augmentation = next(step for step in actual['data']['train']['pipeline']
                            if step['type'] == 'CropResizeFlipImage')
        self.assertEqual(augmentation['data_aug_conf']['reisze'], [720])
        self.assertTrue(augmentation['data_aug_conf']['rand_flip'])

    def test_new_run_uses_pretrained_initialization_and_no_m1_resume(self):
        self.assertIsNotNone(self.reference['resume_from'])
        actual = self.config()
        self.assertEqual(actual['load_from'], '/mounted/fcos3d.pth')
        self.assertIsNone(actual['resume_from'])
        self.assertIsNone(actual['fp16'])
        self.assertEqual(actual['tf32_policy'], dict(matmul=True, cudnn=True))
        self.assertEqual(actual['seed'], 0)

    def test_batch_sampler_lr_checkpoint_and_step_units_are_locked(self):
        actual = self.config()
        self.assertEqual(actual['data']['samples_per_gpu'], 1)
        self.assertEqual(actual['data']['workers_per_gpu'], 2)
        self.assertEqual(actual['data']['shuffler_sampler'], dict(
            type='VirtualDistributedGroupSampler', reference_world_size=4))
        self.assertEqual(actual['optimizer_config']['cumulative_iters'], 2)
        self.assertEqual(actual['optimizer_config']['expected_optimizer_steps_per_epoch'], 5983)
        self.assertEqual(actual['lr_config']['warmup_iters'], 500)
        self.assertEqual(actual['lr_config']['cumulative_iters'], 2)
        self.assertEqual(actual['runner']['max_epochs'], 24)
        self.assertEqual(actual['m3_protocol']['total_optimizer_updates'], 143592)
        self.assertEqual(actual['m3_protocol']['raw_microsteps_per_epoch'], 11966)
        self.assertEqual(actual['log_config']['interval'], 20)
        self.assertEqual(actual['checkpoint_config'], self.reference['checkpoint_config'])

    def test_m3_physics_keeps_one_sweep_and_rejects_old_raster_cache(self):
        actual = self.config()
        for split in ('train', 'val', 'test'):
            self.assertEqual(actual['data'][split]['radar_observation_cfg']['nsweeps'], 1)
            self.assertIsNone(actual['data'][split]['radar_cfg'])
            self.assertIsNone(actual['data'][split]['radar_observation_cfg']['cache_dir'])
        self.assertIsNone(actual['model']['doppler_advection'])
        self.assertEqual(actual['model']['doppler_posterior'], self.candidate['model']['doppler_posterior'])
        self.assertTrue(actual['model']['scientific_eval'])

    def test_modified_reference_budget_is_rejected(self):
        self.reference['optimizer']['lr'] = 0.0002
        with self.assertRaisesRegex(ValueError, 'optimizer differs'):
            self.config()

    def test_modified_m3_sweep_count_is_rejected(self):
        self.candidate['data']['train']['radar_observation_cfg']['nsweeps'] = 5
        with self.assertRaisesRegex(ValueError, 'one-sweep'):
            self.config()

    def prepared_paths(self, directory):
        root = Path(directory)
        data, occupancy, code = root / 'data', root / 'occupancy', root / 'code'
        for component in ('v1.0-trainval', 'samples', 'sweeps', 'can_bus'):
            (data / component).mkdir(parents=True)
        occupancy.mkdir()
        code.mkdir()
        train = data / 'nuscenes_infos_temporal_train_new.pkl'
        # Sparse files test byte-size guards without copying real user data.
        with train.open('wb') as handle:
            handle.truncate(MODULE.EXPECTED_TRAIN_ANN_BYTES)
        (data / 'nuscenes_infos_temporal_val_new.pkl').touch()
        pretrained = root / 'fcos3d.pth'
        with pretrained.open('wb') as handle:
            handle.truncate(MODULE.REFERENCE_CHECKPOINT_BYTES)
        config = self.config(data_root=data, occ_root=occupancy,
                             pretrained=pretrained, work_dir=root / 'output')
        return config, code

    def test_missing_links_require_explicit_creation_and_are_bound_correctly(self):
        with tempfile.TemporaryDirectory() as temp:
            config, code = self.prepared_paths(temp)
            with self.assertRaisesRegex(ValueError, 'create-data-links'):
                MODULE.verify_inputs(config, code)
            MODULE.verify_inputs(config, code, create_data_links=True)
            self.assertEqual((code / 'data/nuscenes').resolve(), Path(config['data_root']))
            self.assertEqual((code / 'data/nuScenes-Occupancy').resolve(), Path(config['occ_path']))
            MODULE.verify_inputs(config, code)

    def test_existing_wrong_link_and_nonempty_workdir_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            config, code = self.prepared_paths(temp)
            (code / 'data').mkdir()
            (code / 'data/nuscenes').symlink_to(Path(temp) / 'wrong')
            with self.assertRaisesRegex(ValueError, 'different dataset'):
                MODULE.verify_inputs(config, code, create_data_links=True)
            (code / 'data/nuscenes').unlink()
            MODULE.verify_inputs(config, code, create_data_links=True)
            Path(config['work_dir']).mkdir()
            (Path(config['work_dir']) / 'epoch_12.pth').touch()
            with self.assertRaisesRegex(ValueError, 'new empty work directory'):
                MODULE.verify_inputs(config, code)

    def test_wrong_checkpoint_size_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            config, code = self.prepared_paths(temp)
            Path(config['load_from']).write_bytes(b'wrong artifact')
            with self.assertRaisesRegex(ValueError, 'checkpoint missing or unexpected byte size'):
                MODULE.verify_inputs(config, code)

    def cache_record(self, config):
        return dict(status='COMPLETE', tokens_processed=100, tokens_required=100,
                    splits=['train', 'val'],
                    manifest=MODULE.observation_manifest(config['data']['train']))

    def write_status(self, directory, record):
        Path(directory, 'build_status.json').write_text(json.dumps(record))

    def test_completed_cache_binds_all_splits_readonly_without_database_import(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(sys.modules, {'nuscenes': None, 'mmcv': None, 'torch': None}):
                self.write_status(directory, self.cache_record(config))
                MODULE.bind_radar_cache(config, directory)
            for split in ('train', 'val', 'test'):
                self.assertEqual(config['data'][split]['radar_observation_cfg']['cache_dir'], str(Path(directory).resolve()))
                self.assertTrue(config['data'][split]['radar_observation_cfg']['cache_readonly'])
            self.assertEqual(config['m3_protocol']['radar_cache']['tokens_processed'], 100)
            self.assertTrue(config['radar_observation_cfg']['cache_readonly'])

    def test_missing_or_invalid_cache_build_record_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'existing build_status'):
                MODULE.bind_radar_cache(self.config(), directory)
            Path(directory, 'build_status.json').write_text('{')
            with self.assertRaisesRegex(ValueError, 'Cannot read'):
                MODULE.bind_radar_cache(self.config(), directory)

    def test_incomplete_and_limited_cache_are_rejected(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            for state in ('BUILDING', 'LIMITED_SMOKE_ONLY', 'FAILED', None):
                record = self.cache_record(config)
                record['status'] = state
                self.write_status(directory, record)
                with self.assertRaisesRegex(ValueError, 'must be COMPLETE'):
                    MODULE.bind_radar_cache(config, directory)

    def test_zero_inconsistent_or_nonnumeric_token_counts_are_rejected(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            for processed, required in ((0, 0), (99, 100), (101, 100),
                                        (-1, -1), ('100', 100), (True, True)):
                record = self.cache_record(config)
                record.update(tokens_processed=processed, tokens_required=required)
                self.write_status(directory, record)
                with self.assertRaisesRegex(ValueError, 'tokens_processed == tokens_required > 0'):
                    MODULE.bind_radar_cache(config, directory)

    def test_cache_must_cover_both_train_and_val(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            for splits in (['train'], ['val'], ['train', 'test'], [], 'train,val', None):
                record = self.cache_record(config)
                record['splits'] = splits
                self.write_status(directory, record)
                with self.assertRaisesRegex(ValueError, 'both requested splits'):
                    MODULE.bind_radar_cache(config, directory)

    def test_old_raster_schema_or_any_manifest_change_is_rejected(self):
        config = self.config()
        changes = dict(schema='old_radar_raster', nsweeps=5, max_returns=512,
                       max_time_lag=0.2, data_root='/different/dataset',
                       data_version='v1.0-mini', unexpected_key='wrong')
        with tempfile.TemporaryDirectory() as directory:
            for field, value in changes.items():
                record = self.cache_record(config)
                record['manifest'][field] = value
                self.write_status(directory, record)
                with self.assertRaisesRegex(ValueError, 'manifest does not match'):
                    MODULE.bind_radar_cache(config, directory)
            record = self.cache_record(config)
            record['manifest']['nsweeps'] = True  # bool == 1 in Python; schema must still reject it.
            self.write_status(directory, record)
            with self.assertRaisesRegex(ValueError, 'manifest does not match'):
                MODULE.bind_radar_cache(config, directory)

    def test_validation_split_manifest_mismatch_does_not_partially_bind(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            self.write_status(directory, self.cache_record(config))
            config['data']['val']['radar_observation_cfg']['max_returns'] = 1024
            before = copy.deepcopy(config)
            with self.assertRaisesRegex(ValueError, 'current val configuration'):
                MODULE.bind_radar_cache(config, directory)
            self.assertEqual(config, before)

    def test_legacy_radar_input_cannot_be_bound_as_m3_cache(self):
        config = self.config()
        with tempfile.TemporaryDirectory() as directory:
            self.write_status(directory, self.cache_record(config))
            config['data']['train']['radar_cfg'] = dict(nsweeps=5)
            with self.assertRaisesRegex(ValueError, 'Only the M3 per-return'):
                MODULE.bind_radar_cache(config, directory)

    @unittest.skipIf(importlib.util.find_spec('mmcv') is None, 'Real MMCV required for config roundtrip')
    def test_mmcv_dump_and_reload_new_filename_preserve_protocol(self):
        from mmcv import Config
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'new_config.py'
            config = self.config()
            MODULE.write_resolved_config(config, output)
            reloaded = Config.fromfile(str(output))
            self.assertEqual(reloaded.data.train.radar_observation_cfg.nsweeps, 1)
            self.assertEqual(reloaded.optimizer, config['optimizer'])
            self.assertEqual(reloaded.lr_config, config['lr_config'])
            self.assertEqual(reloaded.m3_protocol, config['m3_protocol'])
            self.assertEqual(reloaded.load_from, config['load_from'])
            self.assertIsNone(reloaded.resume_from)
            with self.assertRaises(FileExistsError):
                MODULE.write_resolved_config(config, output)


if __name__ == '__main__':
    unittest.main()
