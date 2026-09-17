"""Analytic CPU geometry tests. These are not model/forecasting results."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from motion_geometry import (box_to_global, quaternion_wxyz_to_matrix,
                             rigid_displacement, unique_box_assignment,
                             voxel_centers_xyz)


def rotation_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1., 0., 0.], [0., c, -s], [0., s, c]])


def rotation_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def rotation_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def pose(rotation, translation):
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = translation
    return out


class MotionGeometryTests(unittest.TestCase):
    def test_quaternion_order_nonunit_and_sign(self):
        q = np.array([1., 0., 0., 1.])
        expected = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        old = q.copy()
        for scale in [1., -7., 1e-300, 1e300]:
            assert_allclose(quaternion_wxyz_to_matrix(q * scale), expected, atol=1e-14)
        assert_array_equal(q, old)

    def test_quaternion_full_3d_axis_rotation(self):
        # 120 degrees around (1,1,1): x -> y -> z -> x.
        expected = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        assert_allclose(quaternion_wxyz_to_matrix([1., 1., 1., 1.]), expected, atol=1e-14)

    def test_invalid_quaternions_rejected(self):
        for q in [[0., 0., 0., 0.], [1., 2., 3.], [np.nan, 0., 0., 1.], [np.inf, 0., 0., 1.]]:
            with self.assertRaises(ValueError):
                quaternion_wxyz_to_matrix(q)

    def test_box_column_transform_points_not_vectors(self):
        box = box_to_global([10., -3., 2.], [1., 0., 0., 1.])
        assert_allclose(box @ [2., 0., 0., 1.], [10., -1., 2., 1.], atol=1e-14)
        assert_allclose(box @ [2., 0., 0., 0.], [0., 2., 0., 0.], atol=1e-14)

    def test_stationary_object_no_ego_motion_in_displacement(self):
        points = np.array([[2., -5., 1.], [-3., 1., 4.]])
        object_global = pose(rotation_y(.3) @ rotation_x(-.7), [8., 2., -1.])
        # These represent different observing LiDAR frames. A fixed world
        # object has zero displacement in either fixed reference frame.
        for g0 in [np.eye(4), pose(rotation_z(.9) @ rotation_x(.2), [30., -6., 3.])]:
            assert_allclose(rigid_displacement(points, g0, object_global, object_global), 0., atol=2e-14)

    def test_asymmetric_translation_in_rotated_translated_reference(self):
        points = np.array([[1., 2., 3.], [-4., 6., -2.]])
        g0 = pose(rotation_z(np.pi/2), [21., -13., 7.])
        a0 = pose(rotation_x(.4), [7., 11., -5.])
        ah = a0.copy(); ah[:3, 3] += [3., -2., 5.]
        expected = np.tile([-2., -3., 5.], (len(points), 1))
        assert_allclose(rigid_displacement(points, g0, a0, ah), expected, atol=2e-14)

    def test_yaw_rotation_around_object_center_not_origin(self):
        a0 = pose(np.eye(3), [10., -7., 2.])
        ah = pose(rotation_z(np.pi/2), [13., -5., 3.])
        points = np.array([[12., -7., 2.], [10., -6., 4.]])
        # Same local points are (2,0,0) and (0,1,2).
        expected_future = np.array([[13., -3., 3.], [12., -5., 5.]])
        assert_allclose(rigid_displacement(points, np.eye(4), a0, ah), expected_future-points, atol=1e-14)

    def test_full_3d_tilt_from_material_point_construction(self):
        r0 = rotation_y(-.6) @ rotation_z(.2)
        rh = rotation_x(.8) @ rotation_y(.4)
        rg = rotation_z(-.3) @ rotation_x(.5)
        c0, ch, cg = np.array([8., -3., 4.]), np.array([9., 2., -1.]), np.array([1., 7., 2.])
        local = np.array([[2., 1., -1.], [-3., 0., 4.], [0., 0., 0.]])
        # Independent material-point construction, without the composed inverse formula.
        p0 = ((local @ r0.T + c0) - cg) @ rg
        ph = ((local @ rh.T + ch) - cg) @ rg
        result = rigid_displacement(p0, pose(rg, cg), pose(r0, c0), pose(rh, ch))
        assert_allclose(result, ph-p0, rtol=1e-14, atol=2e-14)
        self.assertGreater(abs(result[0, 2]), .1)  # Would detect a BEV-only implementation.

    def test_wlh_axis_mapping_and_closed_faces(self):
        # width=2, length=8, height=4 => x half4, y half1, z half2.
        pts = np.array([[3., 0., 0.], [0., 3., 0.], [0., 0., 1.5], [0., 0., 2.5],
                        [4., 1., 2.], [4.+1e-6, 0., 0.]])
        assert_array_equal(unique_box_assignment(pts, np.eye(4)[None], [[2., 8., 4.]]), [0, -1, 0, -1, 0, -1])

    def test_assignment_rotated_and_tilted_box(self):
        rot = rotation_z(np.pi/2) @ rotation_y(.6)
        box = pose(rot, [10., -4., 3.])
        local = np.array([[3., .5, 1.], [0., 1.5, 0.], [0., 0., 2.5]])
        pts = local @ rot.T + box[:3, 3]
        assert_array_equal(unique_box_assignment(pts, box[None], [[2., 8., 4.]]), [0, -1, -1])

    def test_overlaps_no_first_box_tie_break_and_shared_face(self):
        boxes = np.stack([pose(np.eye(3), [0., 0., 0.]), pose(np.eye(3), [2., 0., 0.])])
        pts = np.array([[-.5, 0., 0.], [1., 0., 0.], [2., 0., 0.], [5., 0., 0.]])
        assert_array_equal(unique_box_assignment(pts, boxes, np.ones((2, 3))*2), [0, -2, 1, -1])
        boxes = np.concatenate([boxes, boxes[:1]], axis=0)
        assert_array_equal(unique_box_assignment(pts, boxes, np.ones((3, 3))*2), [-2, -2, 1, -1])

    def test_no_input_mutation_and_empty_arrays(self):
        pts = np.array([[.2, -.3, .4]])
        boxes = np.eye(4)[None]; sizes = np.ones((1, 3))*2
        snapshots = [v.copy() for v in [pts, boxes, sizes]]
        rigid_displacement(pts, boxes[0], boxes[0], boxes[0])
        unique_box_assignment(pts, boxes, sizes)
        for actual, old in zip([pts, boxes, sizes], snapshots):
            assert_array_equal(actual, old)
        assert_array_equal(unique_box_assignment(pts, np.empty((0, 4, 4)), np.empty((0, 3))), [-1])
        self.assertEqual(unique_box_assignment(np.empty((0, 3)), boxes, sizes).shape, (0,))
        self.assertEqual(rigid_displacement(np.empty((0, 3)), boxes[0], boxes[0], boxes[0]).shape, (0, 3))

    def test_voxel_centers_anisotropic_xyz_cell_boundaries(self):
        centers = voxel_centers_xyz((2, 3, 4), [-1., 10., -4., 3., 16., 4.])
        self.assertEqual(centers.shape, (2, 3, 4, 3))
        self.assertEqual(centers.dtype, np.float64)
        assert_array_equal(centers[0, 0, 0], [0., 11., -3.])
        assert_array_equal(centers[1, 2, 3], [2., 15., 3.])
        assert_array_equal(centers[1, 0, 0]-centers[0, 0, 0], [2., 0., 0.])
        assert_array_equal(centers[0, 1, 0]-centers[0, 0, 0], [0., 2., 0.])
        assert_array_equal(centers[0, 0, 1]-centers[0, 0, 0], [0., 0., 2.])

    def test_invalid_pose_size_and_grid_rejected(self):
        for bad in [pose(np.diag([2., 1., 1.]), [0., 0., 0.]),
                    pose(np.diag([-1., 1., 1.]), [0., 0., 0.]), np.zeros((4, 4))]:
            with self.assertRaises(ValueError):
                rigid_displacement([[0., 0., 0.]], bad, np.eye(4), np.eye(4))
        for size in [[[0., 2., 3.]], [[1., -2., 3.]], [[1., 2., np.nan]]]:
            with self.assertRaises(ValueError):
                unique_box_assignment([[0., 0., 0.]], np.eye(4)[None], size)
        for shape, extent in [((2, 3, 0), [0., 0., 0., 1., 1., 1.]),
                              ((2., 3, 4), [0., 0., 0., 1., 1., 1.]),
                              ((True, 3, 4), [0., 0., 0., 1., 1., 1.]),
                              ((2, 3, 4), [0., 0., 0., 0., 1., 1.])]:
            with self.assertRaises(ValueError):
                voxel_centers_xyz(shape, extent)


if __name__ == "__main__":
    unittest.main(verbosity=2)
