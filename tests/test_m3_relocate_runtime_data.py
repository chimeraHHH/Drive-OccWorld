"""Small real-NPZ tests for local-only, recoverable M3 cache identity relocation."""
import copy
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import pickle
import runpy
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('m3_relocate', ROOT / 'tools/m3_relocate_runtime_data.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ORIGINAL_SERIALIZER = MODULE.serialize_config


def plain_config(path):
    return {k: v for k, v in runpy.run_path(str(path)).items() if not k.startswith('__')}


def simple_serialize(config, filename):
    return ('\n'.join('{} = {!r}'.format(k, v) for k, v in config.items()) + '\n').encode()


class RelocationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.local = self.base / 'local'
        self.old_data, self.old_occ, self.old_cache = (self.base / n for n in
                                                     ('source_data', 'source_occ', 'source_cache'))
        self.code = self.base / 'code'
        self.output = self.code / 'configs_runtime/local_data.py'
        self.output.parent.mkdir(parents=True)
        self.source_file = self.code / 'configs_runtime/active.py'
        self.addCleanup(patch.stopall)
        patch.object(MODULE, 'LOCAL_ROOT', self.local).start()
        patch.object(MODULE, 'CODE_ROOT', self.code).start()
        patch.object(MODULE, 'filesystem', return_value=['/dev/nvme0n1p2', 'ext4', '/']).start()
        self.serializer = patch.object(MODULE, 'serialize_config', side_effect=simple_serialize).start()
        paths = MODULE.destinations()
        self.new_data, self.new_occ, self.new_cache = paths['data'], paths['occupancy'], paths['cache']
        for directory in (self.old_data / 'samples', self.old_occ, self.old_cache,
                          self.new_data / 'samples', self.new_occ, self.new_cache):
            directory.mkdir(parents=True)
        (self.local / '.migration.lock').touch()
        reference = plain_config(ROOT / 'DOCS/m3_h200_20260907/l40s_resolved_reference.py')
        candidate = plain_config(ROOT / 'DOCS/m3_20260906/resolved_m3_l40s.py')
        self.source = MODULE.SUPPORT.make_config(candidate, reference,
            data_root=self.old_data, occ_root=self.old_occ,
            pretrained=self.base / 'pretrained.pth', work_dir=self.base / 'active_workdir')
        self.tokens = ['a', 'b', 'c']
        for split, tokens in [('train', ['a', 'b']), ('val', ['b', 'c'])]:
            name = 'nuscenes_infos_temporal_' + split + '_new.pkl'
            for directory in (self.old_data, self.new_data):
                (directory / name).write_bytes(pickle.dumps(dict(infos=[dict(
                    token=t, cams={camera: dict(data_path='./data/nuscenes/samples/{}/{}.jpg'.format(camera, t))
                                   for camera in MODULE.CAMERAS}) for t in tokens])))
        self.old_manifest = MODULE.manifests(self.source)
        options = copy.deepcopy(self.source['data']['train']['radar_observation_cfg'])
        options.update(cache_dir=str(self.old_cache), cache_readonly=False)
        loader = MODULE.OBSERVATIONS.NuScenesRadarObservations(
            nusc=MODULE.types.SimpleNamespace(version='v1.0-trainval', dataroot=str(self.old_data)),
            **options)
        for i, token in enumerate(self.tokens):
            values = np.zeros((4096, 9), np.float32)
            values[0] = [float(i), -0.0, 0., 3., 1., 0., 12.3 + i, 0.01, 1.]
            loader._write_cache(token, values)
        self.source_status = dict(status='COMPLETE', tokens_required=3, tokens_processed=3,
                                  tokens_selected=3, splits=['train', 'val'], limit=None,
                                  workers=8, manifest=self.old_manifest)
        self.write_json(self.old_cache / 'build_status.json', self.source_status)
        for path in self.old_cache.iterdir():
            shutil.copy2(path, self.new_cache / path.name)
        MODULE.SUPPORT.bind_radar_cache(self.source, self.old_cache)
        self.source_file.write_bytes(simple_serialize(self.source, self.source_file))
        self.source_bytes = {p.name: p.read_bytes() for p in self.old_cache.iterdir()}
        self.receipt_file = self.local / 'copy_manifest.json'
        expected = dict(images=(self.old_data / 'samples', self.new_data / 'samples'),
                        metadata=(self.old_data, self.new_data),
                        occupancy=(self.old_occ, self.new_occ),
                        radar_cache=(self.old_cache, self.new_cache))
        self.receipt = dict(schema='m3_local_data_copy_v1', status='COPY_VERIFIED',
                            destination_root=str(self.local), components={
            key: dict(source_root=str(old), destination_root=str(new), verified=True,
                      files=3, bytes=100, jobs=sorted(MODULE.CAMERAS) if key == 'images' else [key])
            for key, (old, new) in expected.items()})
        self.write_json(self.receipt_file, self.receipt)

    @staticmethod
    def write_json(path, value):
        Path(path).write_text(json.dumps(value))

    def run_tool(self):
        return MODULE.run(self.source, self.source_file, self.receipt_file, self.output)

    def status(self):
        return json.loads((self.new_cache / 'build_status.json').read_text())

    def assert_source_unchanged(self):
        self.assertEqual({p.name: p.read_bytes() for p in self.old_cache.iterdir()}, self.source_bytes)
        self.assertEqual(self.source_file.read_bytes(), simple_serialize(self.source, self.source_file))

    def replace_member(self, token, name, array):
        path = self.new_cache / (token + '.npz')
        with zipfile.ZipFile(path) as archive:
            members = {n: archive.read(n) for n in archive.namelist()}
        raw = io.BytesIO()
        np.save(raw, array, allow_pickle=False)
        members[name] = raw.getvalue()
        with zipfile.ZipFile(path, 'w') as archive:
            for n, data in members.items():
                archive.writestr(n, data)

    def test_production_cache_reads_new_identity_and_array_members_are_byte_identical(self):
        result = self.run_tool()
        self.assertEqual(result['status'], 'PREPARED_NOT_ACTIVATED')
        self.assertEqual(result['tokens_processed'], 3)
        self.assertEqual(result['entries_rewritten'], 3)
        config = plain_config(self.output)
        loader = MODULE.validator(config['data']['train'])
        loader.cache_dir = str(self.new_cache)
        loader.cache_readonly = True
        for token in self.tokens:
            observed = loader(token)
            self.assertEqual(observed.shape, (4096, 9))
            self.assertEqual(observed.dtype, np.float32)
            with zipfile.ZipFile(self.old_cache / (token + '.npz')) as old:
                with zipfile.ZipFile(self.new_cache / (token + '.npz')) as new:
                    for member in ('observations.npy', 'sample_token.npy'):
                        self.assertEqual(old.read(member), new.read(member))
        self.assertEqual(self.status()['status'], 'COMPLETE')
        ready = MODULE.read_json(self.new_cache / 'relocation_ready.json')
        self.assertEqual(ready['output_config_sha256'], hashlib.sha256(self.output.read_bytes()).hexdigest())
        self.assertFalse(ready['training_restarted'])
        self.assert_source_unchanged()

    def test_all_training_and_sampling_settings_are_unchanged(self):
        self.run_tool()
        actual = plain_config(self.output)
        self.assertEqual(actual['model'], self.source['model'])
        allowed = {'data', 'data_root', 'occ_path', 'train_pipeline', 'test_pipeline',
                   'evaluation', 'radar_observation_cfg', 'm3_protocol'}
        self.assertEqual({k: v for k, v in actual.items() if k not in allowed},
                         {k: v for k, v in self.source.items() if k not in allowed})
        restored = copy.deepcopy(actual)
        restored['data_root'], restored['occ_path'] = self.source['data_root'], self.source['occ_path']
        for split in ('train', 'val', 'test'):
            for key in ('data_root', 'ann_file'):
                restored['data'][split][key] = self.source['data'][split][key]
            restored['data'][split]['radar_observation_cfg']['cache_dir'] = str(self.old_cache)
            for step in restored['data'][split]['pipeline']:
                if step.get('type') == 'LoadOccupancy':
                    step['occ_path'] = str(self.old_occ)
        for pipeline in (restored['train_pipeline'], restored['test_pipeline'], restored['evaluation']['pipeline']):
            for step in pipeline:
                if step.get('type') == 'LoadOccupancy':
                    step['occ_path'] = str(self.old_occ)
        restored['radar_observation_cfg']['cache_dir'] = str(self.old_cache)
        restored['m3_protocol'].pop('data_relocation')
        restored['m3_protocol']['radar_cache'] = self.source['m3_protocol']['radar_cache']
        self.assertEqual(restored, self.source)

    def test_unverified_copy_or_wrong_component_identity_is_rejected_before_writes(self):
        for alter in (
                lambda r: r.update(status='COPYING'),
                lambda r: r.update(destination_root='/different'),
                lambda r: r['components']['metadata'].update(verified=False),
                lambda r: r['components']['radar_cache'].update(source_root='/different'),
                lambda r: r['components']['images'].update(jobs=['CAM_FRONT'])):
            receipt = copy.deepcopy(self.receipt)
            alter(receipt)
            self.write_json(self.receipt_file, receipt)
            with self.assertRaises(ValueError):
                self.run_tool()
            self.assertEqual(self.status(), self.source_status)
        self.assert_source_unchanged()

    def test_source_manifest_changes_including_boolean_for_integer_are_rejected(self):
        for field, value in [('nsweeps', True), ('nsweeps', 5), ('data_version', 'v1.0-mini')]:
            record = copy.deepcopy(self.source_status)
            record['manifest'][field] = value
            self.write_json(self.old_cache / 'build_status.json', record)
            with self.assertRaisesRegex(ValueError, 'manifest differs'):
                self.run_tool()
            self.assertEqual(self.status(), self.source_status)
        self.write_json(self.old_cache / 'build_status.json', self.source_status)

    def test_interrupted_mixed_cache_resumes_without_rewriting_completed_entry(self):
        original = MODULE.rebase_entry
        def interrupt(path, token, *args):
            if token == 'b':
                raise RuntimeError('simulated interruption')
            return original(path, token, *args)
        with patch.object(MODULE, 'rebase_entry', side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, 'interruption'):
                self.run_tool()
        self.assertEqual(self.status()['status'], 'FAILED')
        first = self.new_cache / 'a.npz'
        before = first.read_bytes(), first.stat().st_mtime_ns
        result = self.run_tool()
        self.assertEqual(result['entries_already_rebased'], 1)
        self.assertEqual(result['entries_rewritten'], 2)
        self.assertEqual(before, (first.read_bytes(), first.stat().st_mtime_ns))
        self.assert_source_unchanged()

    def test_config_serialization_failure_after_all_entries_is_recoverable(self):
        self.serializer.side_effect = RuntimeError('serializer unavailable')
        with self.assertRaisesRegex(RuntimeError, 'serializer'):
            self.run_tool()
        self.assertEqual(self.status()['status'], 'FAILED')
        self.assertEqual(self.status()['tokens_processed'], 3)
        self.assertFalse(self.output.exists())
        self.serializer.side_effect = simple_serialize
        result = self.run_tool()
        self.assertEqual(result['entries_already_rebased'], 3)
        self.assertEqual(result['entries_rewritten'], 0)
        self.assert_source_unchanged()

    def test_failure_after_config_publication_recovers_without_overwriting_it(self):
        original = MODULE.atomic_json
        def fail_final_ready(path, value):
            if Path(path).name == 'relocation_ready.json' and value.get('status') == 'PREPARED_NOT_ACTIVATED':
                raise RuntimeError('interrupted after config publication')
            return original(path, value)
        with patch.object(MODULE, 'atomic_json', side_effect=fail_final_ready):
            with self.assertRaisesRegex(RuntimeError, 'after config publication'):
                self.run_tool()
        self.assertTrue(self.output.is_file())
        before = self.output.read_bytes(), self.output.stat().st_mtime_ns
        result = self.run_tool()
        self.assertEqual(result['entries_already_rebased'], 3)
        self.assertEqual(result['status'], 'PREPARED_NOT_ACTIVATED')
        self.assertEqual(before, (self.output.read_bytes(), self.output.stat().st_mtime_ns))
        self.assert_source_unchanged()

    def test_third_manifest_or_token_mismatch_is_failed_not_complete(self):
        wrong = copy.deepcopy(self.old_manifest)
        wrong['subsample_seed'] = 88
        self.replace_member('a', 'manifest.npy', np.asarray(MODULE.canonical(wrong)))
        with self.assertRaisesRegex(ValueError, 'per-entry manifest'):
            self.run_tool()
        self.assertEqual(self.status()['status'], 'FAILED')
        shutil.copy2(self.old_cache / 'a.npz', self.new_cache / 'a.npz')
        self.replace_member('a', 'sample_token.npy', np.asarray('wrong'))
        with self.assertRaisesRegex(ValueError, 'token mismatch'):
            self.run_tool()
        self.assertEqual(self.status()['status'], 'FAILED')
        self.assertFalse(self.output.exists())
        self.assert_source_unchanged()

    def test_missing_and_extra_tokens_fail_exact_annotation_set_check(self):
        (self.new_cache / 'a.npz').unlink()
        with self.assertRaisesRegex(ValueError, 'token set/count'):
            self.run_tool()
        self.assertEqual(self.status()['status'], 'FAILED')
        shutil.copy2(self.old_cache / 'a.npz', self.new_cache / 'a.npz')
        shutil.copy2(self.old_cache / 'a.npz', self.new_cache / 'extra.npz')
        with self.assertRaisesRegex(ValueError, 'token set/count'):
            self.run_tool()
        self.assertEqual(self.status()['status'], 'FAILED')
        self.assert_source_unchanged()

    def test_invalid_array_dtype_and_corrupt_zip_are_rejected(self):
        self.replace_member('a', 'observations.npy', np.zeros((4096, 9), np.float64))
        with self.assertRaisesRegex(ValueError, 'dtype float32'):
            self.run_tool()
        (self.new_cache / 'a.npz').write_bytes(b'broken zip')
        with self.assertRaises(zipfile.BadZipFile):
            self.run_tool()
        self.assertEqual(self.status()['status'], 'FAILED')
        self.assert_source_unchanged()

    def test_camera_paths_must_use_the_private_relative_entry(self):
        annotation = self.new_data / 'nuscenes_infos_temporal_train_new.pkl'
        payload = pickle.loads(annotation.read_bytes())
        payload['infos'][0]['cams']['CAM_FRONT']['data_path'] = '/old/absolute/CAM_FRONT/a.jpg'
        annotation.write_bytes(pickle.dumps(payload))
        with self.assertRaisesRegex(ValueError, 'relative data/nuscenes entry'):
            self.run_tool()
        self.assertEqual(self.status()['status'], 'FAILED')
        self.assertFalse(self.output.exists())

    def test_destination_cache_symlink_cannot_redirect_writes_to_nas(self):
        self.new_cache.rename(self.new_cache.with_name('saved_copy'))
        self.new_cache.symlink_to(self.old_cache, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'without symlink ancestors'):
            self.run_tool()
        self.assert_source_unchanged()

    def test_npz_symlink_and_status_symlink_cannot_write_source(self):
        (self.new_cache / 'a.npz').unlink()
        (self.new_cache / 'a.npz').symlink_to(self.old_cache / 'a.npz')
        with self.assertRaisesRegex(ValueError, 'non-symlink'):
            self.run_tool()
        self.assert_source_unchanged()
        (self.new_cache / 'build_status.json').unlink()
        (self.new_cache / 'build_status.json').symlink_to(self.old_cache / 'build_status.json')
        with self.assertRaisesRegex(ValueError, 'non-symlink'):
            self.run_tool()
        self.assert_source_unchanged()

    def test_wrong_filesystem_is_rejected(self):
        with patch.object(MODULE, 'filesystem', return_value=['server:/nas', 'nfs4', '/storage/data']):
            with self.assertRaisesRegex(ValueError, 'system NVMe'):
                self.run_tool()
        self.assertEqual(self.status(), self.source_status)
        self.assert_source_unchanged()

    def test_existing_config_is_never_overwritten_or_rebased(self):
        self.output.write_bytes(b'user config')
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            self.run_tool()
        self.assertEqual(self.output.read_bytes(), b'user config')
        self.assertEqual(self.status(), self.source_status)
        self.output.unlink()
        self.output = self.source_file
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            self.run_tool()
        self.assert_source_unchanged()

    def test_exclusive_publish_refuses_a_racing_existing_config(self):
        self.output.write_bytes(b'concurrent file')
        with self.assertRaises(FileExistsError):
            MODULE.atomic_bytes(self.output, b'new config', overwrite=False)
        self.assertEqual(self.output.read_bytes(), b'concurrent file')

    def test_running_copy_lock_prevents_rebasing(self):
        import fcntl
        with (self.local / '.migration.lock').open() as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                self.run_tool()
        self.assertEqual(self.status(), self.source_status)
        self.assert_source_unchanged()

    def test_copy_receipt_is_rechecked_after_acquiring_the_copy_lock(self):
        original = MODULE.exclusive_locks
        @contextlib.contextmanager
        def invalidate_at_lock(cache):
            with original(cache):
                receipt = copy.deepcopy(self.receipt)
                receipt['status'] = 'COPYING'
                self.write_json(self.receipt_file, receipt)
                yield
        with patch.object(MODULE, 'exclusive_locks', side_effect=invalidate_at_lock):
            with self.assertRaisesRegex(ValueError, 'COPY_VERIFIED'):
                self.run_tool()
        self.assertEqual(self.status(), self.source_status)
        self.assert_source_unchanged()

    def test_resume_rejects_a_changed_source_config(self):
        self.serializer.side_effect = RuntimeError('stop before config')
        with self.assertRaises(RuntimeError):
            self.run_tool()
        self.source_file.write_bytes(self.source_file.read_bytes() + b'\n# changed\n')
        self.serializer.side_effect = simple_serialize
        with self.assertRaisesRegex(ValueError, 'different source, target or configuration'):
            self.run_tool()
        self.assertFalse(self.output.exists())

    def test_actual_mmcv_config_dump_roundtrip_when_available(self):
        try:
            import mmcv
        except ImportError:
            self.skipTest('MMCV is not installed in the lightweight local test environment')
        self.serializer.side_effect = ORIGINAL_SERIALIZER
        self.run_tool()
        restored = mmcv.Config.fromfile(str(self.output))._cfg_dict.to_dict()
        self.assertEqual(restored['optimizer'], self.source['optimizer'])
        self.assertEqual(restored['data']['train']['data_root'], str(self.new_data) + '/')


if __name__ == '__main__':
    unittest.main()
