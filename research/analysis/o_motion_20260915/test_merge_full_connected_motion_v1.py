"""Tiny parser/coverage boundaries only; no synthetic scientific result files."""
import copy
from pathlib import Path
import unittest

import numpy as np
import merge_full_connected_motion_v1 as m


class Boundaries(unittest.TestCase):
    def setUp(self):
        # Two objects: one valid at h1/h2, one valid only h1. No EPE inference.
        self.label = dict(object_index=np.array([0,0,1]),
            valid=np.array([[1,1,1],[1,1,0],[0,0,0],[0,0,0]],dtype=bool),
            instance_tokens=np.array(['a','b']), object_speed_group=np.array([[0,2],[1,-1],[-1,-1],[-1,-1]]),
            dt_future_seconds=np.array([.49,1.01,1.52,2.02]))
        self.record = dict(sample_token='sample',scene_token='scene')
        self.expected = m.expected_support(self.label)

    def rows(self):
        return [dict(self.record, arm=arm,instance_token=key[0],horizon_seconds=key[1],**label,
            epe_xy_m=0.,epe_3d_m=0.,zero_epe_xy_m=0.,zero_epe_3d_m=0.)
            for arm in m.PHYSICAL_ARMS for key,label in self.expected.items()]

    def test_parsed_support(self):
        self.assertEqual(self.expected, {('a',.5):dict(group='stationary',source_points=2,dt_seconds=.49),
            ('b',.5):dict(group='moving',source_points=1,dt_seconds=.49),
            ('a',1.):dict(group='ambiguous',source_points=2,dt_seconds=1.01)})
        self.assertEqual(m.validate_object_anchor(self.rows(),self.record,self.expected)['physical_rows'],9)

    def test_common_omission_and_extra(self):
        missing = [r for r in self.rows() if r['instance_token'] != 'b']
        with self.assertRaises(ValueError): m.validate_object_anchor(missing,self.record,self.expected)
        rows=self.rows();rows.append(copy.deepcopy(rows[0]))
        with self.assertRaises(ValueError): m.validate_object_anchor(rows,self.record,self.expected)
        rows=self.rows();rows[0]['instance_token']='absent'
        with self.assertRaises(ValueError): m.validate_object_anchor(rows,self.record,self.expected)

    def test_empty_anchor(self):
        label=dict(object_index=np.empty(0,dtype=int),valid=np.empty((4,0),dtype=bool),
            instance_tokens=np.empty(0,dtype=str),object_speed_group=np.empty((4,0),dtype=int),
            dt_future_seconds=np.array([.5,1.,1.5,2.]))
        expected=m.expected_support(label)
        self.assertEqual(expected,{})
        self.assertTrue(m.validate_object_anchor([],self.record,expected)['empty_anchor'])
        with self.assertRaises(ValueError): m.validate_object_anchor(self.rows()[:1],self.record,expected)

    def test_group_points_and_one_ulp_dt(self):
        for key,value in [('group','moving'),('source_points',1),('dt_seconds',np.nextafter(.49,np.inf))]:
            rows=self.rows();rows[0][key]=value
            with self.assertRaises(ValueError): m.validate_object_anchor(rows,self.record,self.expected)

    def test_duplicate_or_missing_full_shard(self):
        def header(i,count):
            start,stop=5119*i//count,5119*(i+1)//count
            return dict(mode='full',status='COMPLETE_FULL_CONNECTED_MOTION_SHARD',shard_index=i,shard_count=count,
                ordinal_start=start,ordinal_stop_exclusive=stop,planned_ordinals=list(range(start,stop)),processed_samples=stop-start)
        m.validate_partition_headers([header(0,1)])
        m.validate_partition_headers([header(0,2),header(1,2)])
        for values in [[header(0,2)],[header(0,2),header(0,2)]]:
            with self.assertRaises(ValueError):m.validate_partition_headers(values)
        h=header(0,1);h['mode']='pilot'
        with self.assertRaises(ValueError):m.validate_partition_headers([h])
        h=header(0,1);h['processed_samples']-=1
        with self.assertRaises(ValueError):m.validate_partition_headers([h])


if __name__ == '__main__': unittest.main()
