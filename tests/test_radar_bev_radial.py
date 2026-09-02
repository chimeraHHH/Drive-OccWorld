import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1] /
    'projects/mmdet3d_plugin/datasets/radar_bev.py')
SPEC = importlib.util.spec_from_file_location('radar_bev', str(MODULE_PATH))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
NuScenesRadarBEV = MODULE.NuScenesRadarBEV


class RadarBEVRadialTest(unittest.TestCase):
    def test_projection_uses_actual_sensor_origin(self):
        points = np.array([[2.0], [1.0]], dtype=np.float32)
        velocity = np.array([[3.0], [9.0]], dtype=np.float32)
        sensor_origin = np.array([[1.0], [1.0]], dtype=np.float32)
        radial, direction = NuScenesRadarBEV._radial_measurement(
            points, velocity, sensor_origin)
        np.testing.assert_allclose(direction[:, 0], [1.0, 0.0])
        np.testing.assert_allclose(radial, [3.0])

    def test_zero_distance_has_no_artificial_direction(self):
        points = np.array([[1.0], [1.0]], dtype=np.float32)
        velocity = np.array([[3.0], [9.0]], dtype=np.float32)
        radial, direction = NuScenesRadarBEV._radial_measurement(
            points, velocity, points.copy())
        np.testing.assert_array_equal(direction, np.zeros_like(direction))
        np.testing.assert_array_equal(radial, np.zeros_like(radial))


if __name__ == '__main__':
    unittest.main()
