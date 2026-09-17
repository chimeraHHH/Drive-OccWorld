"""Analytic, CPU-only matching tests; no simulated scientific performance."""

import itertools
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from common_output_gospa import bev_component_centers, gospa2, truth_group_breakdown


def exhaustive_cost(truth, pred, c):
    """Enumerate all PARTIAL pairings directly from Proposition 1 for tiny sets."""
    best = np.inf
    n, m = len(truth), len(pred)
    for k in range(min(n, m)+1):
        for ti in itertools.combinations(range(n), k):
            for pj in itertools.permutations(range(m), k):
                value = sum(float(np.sum((truth[i]-pred[j])**2)) for i,j in zip(ti,pj))
                value += c*c/2.*(n+m-2*k)
                best = min(best, value)
    return best


class CommonOutputGOSPATests(unittest.TestCase):
    def test_displacement_known_answer(self):
        r = gospa2([[0., 0.]], [[3., 0.]], cutoff_m=4.)
        self.assertEqual(r['distance_m'], 3.)
        self.assertEqual((r['localization_m2'],r['missed_m2'],r['false_m2']), (9.,0.,0.))

    def test_deletion_cost(self):
        r = gospa2([[0.,0.],[10.,0.]], [[0.,0.]], cutoff_m=4.)
        self.assertEqual(r['squared_cost_m2'], 8.)
        self.assertEqual(r['missed_truth_indices'], [1])
        self.assertEqual(r['false_m2'], 0.)

    def test_extra_false_and_no_cardinality_normalization(self):
        r = gospa2([[0.,0.]], [[0.,0.],[10.,0.],[20.,0.]], cutoff_m=4.)
        self.assertEqual(r['squared_cost_m2'], 16.)
        self.assertEqual(r['false_m2'], 16.)

    def test_outside_or_equal_cutoff_is_miss_plus_false(self):
        for x in [4., 20.]:
            r = gospa2([[0.,0.]], [[x,0.]], cutoff_m=4.)
            self.assertEqual(r['matches'], [])
            self.assertEqual((r['squared_cost_m2'],r['missed_m2'],r['false_m2']), (16.,8.,8.))

    def test_combined_localization_delete_and_false(self):
        r = gospa2([[0.,0.],[10.,0.]], [[1.,2.],[30.,0.]], cutoff_m=4.)
        self.assertEqual((r['localization_m2'],r['missed_m2'],r['false_m2']), (5.,8.,8.))
        self.assertEqual(r['squared_cost_m2'], 21.)

    def test_global_assignment_not_greedy_nearest(self):
        # Greedy truth0->pred0 costs 1+16; optimum uses 9+1=10.
        r = gospa2([[0.,0.],[2.,0.]], [[1.,0.],[-3.,0.]], cutoff_m=10.)
        self.assertEqual(r['squared_cost_m2'], 10.)

    def test_rectangular_solver_matches_partial_enumeration(self):
        rng = np.random.RandomState(11)
        for n,m in itertools.product(range(4), repeat=2):
            x,y=rng.uniform(-5,5,(n,2)),rng.uniform(-5,5,(m,2))
            self.assertAlmostEqual(gospa2(x,y,3.)['squared_cost_m2'],exhaustive_cost(x,y,3.),places=12)

    def test_empty_collections_and_symmetry(self):
        e = np.empty((0,2)); p = np.array([[1.,2.]])
        self.assertEqual(gospa2(e,e)['squared_cost_m2'], 0.)
        self.assertEqual(gospa2(e,p)['false_m2'], 8.)
        self.assertEqual(gospa2(p,e)['missed_m2'], 8.)
        self.assertEqual(gospa2(e,p)['distance_m'], gospa2(p,e)['distance_m'])

    def test_static_prediction_is_not_a_moving_false_positive(self):
        full = gospa2([[0.,0.],[10.,0.]], [[0.,0.],[30.,0.]], cutoff_m=4.)
        groups = truth_group_breakdown(full, ['stationary','moving'])
        self.assertEqual(groups['groups']['stationary']['matched_count'], 1)
        self.assertEqual(groups['groups']['moving']['missed_count'], 1)
        self.assertEqual(groups['global_false_count'], 1)
        self.assertFalse(groups['independent_group_gospa'])

    def test_same_locations_cannot_reveal_identity_switches(self):
        # Swapping two object identities changes no input point collection.
        r = gospa2([[-1.,0.],[1.,0.]], [[1.,0.],[-1.,0.]])
        self.assertEqual(r['squared_cost_m2'], 0.)
        self.assertTrue(r['no_temporal_identity_or_flow_evaluated'])

    def test_component_xyz_axes_and_four_not_eight_connectivity(self):
        fg=np.zeros((3,4,2),dtype=bool)
        fg[0,0,0]=True;fg[1,1,1]=True;fg[2,1,0]=True
        out=bev_component_centers(fg,[10.,-8.,0.,16.,4.,2.])
        assert_array_equal(out['area_cells'],[1,2])
        assert_allclose(out['centers_xy'],[[11.,-6.5],[14.,-3.5]])

    def test_shape_and_fragmentation_are_real_readout_limitations(self):
        # Two different connected shapes can share a centroid: GOSPA is blind
        # to the shape difference, while voxel IoU would still detect it.
        a=np.zeros((5,5,1),dtype=bool);b=np.zeros_like(a)
        a[2,2,0]=True;b[1:4,2,0]=True
        x=bev_component_centers(a,[0,0,0,5,5,1])
        y=bev_component_centers(b,[0,0,0,5,5,1])
        self.assertEqual(gospa2(x['centers_xy'],y['centers_xy'])['squared_cost_m2'],0.)
        b[2,2,0]=False
        y=bev_component_centers(b,[0,0,0,5,5,1])
        self.assertEqual(len(y['centers_xy']),2)
        self.assertGreater(gospa2(x['centers_xy'],y['centers_xy'])['false_m2'],0.)

    def test_invalid_or_excessive_inputs_rejected_without_pruning(self):
        for x in [[[float('nan'),0.]], [[0.,0.,0.]]]:
            with self.assertRaises(ValueError):gospa2(x,[[0.,0.]])
        with self.assertRaises(ValueError):gospa2([[0.,0.]],[[0.,0.]],cutoff_m=0.)
        with self.assertRaises(ValueError):gospa2(np.zeros((3,2)),np.zeros((3,2)),max_pairwise_entries=8)
        with self.assertRaises(ValueError):bev_component_centers(np.zeros((2,2,2)),[0,0,0,1,1,1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
