"""Readiness means verified copy records plus complete local annotation image coverage."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CAMERA = load('m3_camera_check', ROOT / 'tools/m3_check_local_camera_paths.py')
_FIXTURE_MODULE = load('_m3_camera_fixtures', ROOT / 'tests/test_m3_relocate_runtime_data.py')


class CameraPathCheckTest(unittest.TestCase):
    def setUp(self):
        self.fixture = _FIXTURE_MODULE.RelocationTest(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.module = _FIXTURE_MODULE.MODULE
        self.camera_patch = patch.object(CAMERA, 'RELOCATION', self.module)
        self.camera_patch.start()
        self.addCleanup(self.camera_patch.stop)
        f = self.fixture
        for relative in CAMERA.META_DIRECTORIES:
            for root in (f.new_data, f.old_data):
                (root / relative).mkdir(parents=True, exist_ok=True)
        total_files, total_bytes = 0, 0
        for camera in self.module.CAMERAS:
            directory = f.new_data / 'samples' / camera
            directory.mkdir()
            for token in f.tokens:
                data = ('synthetic image file ' + token).encode()
                (directory / (token + '.jpg')).write_bytes(data)
                total_files += 1
                total_bytes += len(data)
        f.receipt['components']['images'].update(files=total_files, bytes=total_bytes)
        f.write_json(f.receipt_file, f.receipt)
        (f.code / 'data').mkdir()
        self.link = f.code / 'data/nuscenes'
        self.link.symlink_to(f.old_data, target_is_directory=True)

    def check(self):
        return CAMERA.check(self.fixture.source, self.fixture.code)

    def partial_jobs(self):
        f = self.fixture
        f.receipt_file.unlink()
        directory = f.local / 'migration_jobs'
        directory.mkdir()
        paths = {name: (f.old_data / 'samples' / name, f.new_data / 'samples' / name)
                 for name in self.module.CAMERAS}
        paths.update({'metadata_' + name.replace('/', '_'): (f.old_data / name, f.new_data / name)
                      for name in CAMERA.META_DIRECTORIES + CAMERA.META_FILES})
        for name, (source, target) in paths.items():
            if name in self.module.CAMERAS:
                files = list(target.iterdir())
                count, size = len(files), sum(p.stat().st_size for p in files)
            else:
                count, size = 1, 1
            f.write_json(directory / (name + '.json'), dict(status='VERIFIED', source=str(source.resolve()),
                destination=str(target), files=count, bytes=size))
        return directory

    def test_final_receipt_and_all_camera_files_pass_without_changing_private_link(self):
        result = self.check()
        self.assertEqual(result['status'], 'CAMERA_DATA_VERIFIED')
        self.assertEqual(result['required_image_files'], 18)
        self.assertEqual(result['annotation_tokens'], 3)
        self.assertTrue(result['link_switch_needed'])
        self.assertFalse(result['link_changed'])
        self.assertEqual(self.link.resolve(), self.fixture.old_data)
        self.fixture.assert_source_unchanged()

    def test_completed_camera_and_metadata_jobs_can_pass_before_occupancy_copy(self):
        self.partial_jobs()
        shutil.rmtree(self.fixture.new_occ)
        result = self.check()
        self.assertEqual(result['copy_evidence_kind'], 'VERIFIED_COMPONENT_JOBS')
        self.assertFalse(result['training_process_changed'])

    def test_missing_or_unfinished_metadata_record_cannot_pass(self):
        jobs = self.partial_jobs()
        path = jobs / 'metadata_lidarseg_v1.0-trainval.json'
        value = json.loads(path.read_text())
        value['status'] = 'COPYING'
        self.fixture.write_json(path, value)
        with self.assertRaisesRegex(ValueError, 'copy record'):
            self.check()
        path.unlink()
        with self.assertRaisesRegex(ValueError, 'non-symlink file'):
            self.check()

    def test_missing_annotation_image_is_detected_even_with_matching_aggregate_counts(self):
        f = self.fixture
        image = f.new_data / 'samples/CAM_FRONT/a.jpg'
        size = image.stat().st_size
        image.unlink()
        f.receipt['components']['images']['files'] -= 1
        f.receipt['components']['images']['bytes'] -= size
        f.write_json(f.receipt_file, f.receipt)
        with self.assertRaisesRegex(ValueError, 'missing from annotations'):
            self.check()

    def test_camera_file_symlink_is_rejected(self):
        image = self.fixture.new_data / 'samples/CAM_FRONT/a.jpg'
        image.unlink()
        image.symlink_to(self.fixture.old_cache / 'a.npz')
        with self.assertRaisesRegex(ValueError, 'non-regular entry'):
            self.check()
        self.fixture.assert_source_unchanged()

    def test_unrelated_private_link_and_missing_can_bus_are_rejected(self):
        self.link.unlink()
        self.link.symlink_to(self.fixture.old_occ, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'known source'):
            self.check()
        self.link.unlink()
        self.link.symlink_to(self.fixture.old_data, target_is_directory=True)
        shutil.rmtree(self.fixture.new_data / 'can_bus')
        with self.assertRaisesRegex(ValueError, 'real local directory'):
            self.check()

    def test_already_local_link_needs_no_switch(self):
        self.link.unlink()
        self.link.symlink_to(self.fixture.new_data, target_is_directory=True)
        self.assertFalse(self.check()['link_switch_needed'])

    def test_stale_copy_count_is_rejected(self):
        f = self.fixture
        f.receipt['components']['images']['bytes'] += 1
        f.write_json(f.receipt_file, f.receipt)
        with self.assertRaisesRegex(ValueError, 'counts/sizes'):
            self.check()


if __name__ == '__main__':
    unittest.main()
