"""Tiny NumPy patch contracts; no learned metrics or simulated result files."""
import copy
import unittest
import numpy as np
from partial_oracle_D_probe_v1 import patch_displacement, source_support, array_sha


class Patch(unittest.TestCase):
    def setup_arrays(self):
        shape=(2,3,4);field=np.arange(1*4*3*24,dtype=np.float32).reshape(1,4,3,*shape)
        # Distinct x/y/z and source index; no axis symmetry can hide transpose.
        ids=np.array([1,7,22],dtype=np.int64)
        valid=np.array([[1,0,1],[0,1,0],[0,0,0],[1,1,1]],dtype=bool)
        target=(1000+np.arange(4*3*3,dtype=np.float32)).reshape(4,3,3)
        return field,dict(source_flat_indices=ids,valid=valid,target_displacement_m=target,grid_shape_xyz=np.array(shape))

    def test_xyz_layout_valid_and_unchanged(self):
        field,label=self.setup_arrays();before=array_sha(field);old_label=copy.deepcopy(label)
        patched,audit=patch_displacement(field,label)
        for h in range(4):
            for i,index in enumerate(label['source_flat_indices']):
                xyz=np.unravel_index(index,field.shape[3:])
                actual=patched[(0,h,slice(None),*xyz)]
                expected=label['target_displacement_m'][h,i] if label['valid'][h,i] else field[(0,h,slice(None),*xyz)]
                np.testing.assert_array_equal(actual,expected)
            mask=np.ones(24,dtype=bool);mask[label['source_flat_indices'][label['valid'][h]]]=False
            np.testing.assert_array_equal(patched[0,h].reshape(3,-1)[:,mask].view(np.uint32),field[0,h].reshape(3,-1)[:,mask].view(np.uint32))
        self.assertEqual(array_sha(field),before)
        for key in label:np.testing.assert_array_equal(label[key],old_label[key])
        self.assertTrue(audit['unlabelled_and_invalid_displacement_bytes_unchanged'])

    def test_empty_and_all_invalid(self):
        field,label=self.setup_arrays();label['valid'][:]=False
        patched,_=patch_displacement(field,label);self.assertEqual(array_sha(patched),array_sha(field))
        label.update(source_flat_indices=np.empty(0,dtype=np.int64),valid=np.empty((4,0),dtype=bool),target_displacement_m=np.empty((4,0,3),dtype=np.float32))
        patched,_=patch_displacement(field,label);self.assertEqual(array_sha(patched),array_sha(field))

    def test_wrong_layout_duplicates_or_nonfinite(self):
        for key,value in [('source_flat_indices',np.array([1,1,22],dtype=np.int64)),('target_displacement_m',np.zeros((4,3,3),dtype=np.float64)),('grid_shape_xyz',np.array([3,2,4]))]:
            field,label=self.setup_arrays();label[key]=value
            with self.assertRaises(ValueError):patch_displacement(field,label)
        field,label=self.setup_arrays();field.flat[0]=np.nan
        with self.assertRaises(ValueError):patch_displacement(field,label)

    def test_source_support_keeps_missing_sources(self):
        field,label=self.setup_arrays();label.update(object_index=np.array([0,0,1]),instance_tokens=np.array(['a','b']),
            object_speed_group=np.array([[0,2],[1,-1],[-1,-1],[0,2]]),dt_future_seconds=np.array([.49,1.01,1.51,2.]))
        mask=np.zeros(field.shape[3:],dtype=bool);mask.reshape(-1)[1]=True
        r=source_support(label,mask)['horizons'][0]['groups']
        self.assertEqual(r['stationary']['predicted_source_points'],1)
        self.assertEqual(r['moving']['valid_points'],1)
        self.assertEqual(r['moving']['missing_source_points'],1)
        self.assertEqual(r['moving']['objects_without_predicted_source'],1)


if __name__=='__main__':unittest.main()
