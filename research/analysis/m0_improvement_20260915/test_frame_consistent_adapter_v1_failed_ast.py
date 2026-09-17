"""CPU source/geometry/shape tests; no benchmark or fabricated result output.

python test_frame_consistent_adapter.py
M0_SOURCE_FILE may point at the same SHA-locked detector in a deployment.
NumPy tests execute exact native rollout/alignment/grid-helper AST with a small
explicit tensor shim and a synthetic deterministic head. They verify routing,
not MMCV/CUDA numerics, learned quality, or real-data coordinate assumptions.
Optional PyTorch CPU tests check late-to-early autoregressive gradients. Missing
torch is an explicit SKIP; passing NumPy tests is not a GPU preflight receipt.
"""
import ast
import contextlib
import copy
import inspect
import os
from pathlib import Path
import types
import unittest

import numpy as np
import frame_consistent_adapter as fc

try:
    import torch
except ImportError:
    torch = None

BASE = Path(__file__).resolve().parents[2] / 'code/Drive-OccWorld-sota-p2/projects/mmdet3d_plugin/bevformer'
SOURCE = Path(os.environ.get('M0_SOURCE_FILE', str(BASE/'detectors/drive_occworld.py')))
GRID_SOURCE = SOURCE.parent.parent / 'utils/e2e_predictor_utils.py'


def source_function(path, name):
    text = path.read_text()
    nodes = [n for n in ast.walk(ast.parse(text)) if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1:
        raise ValueError('Source function is ambiguous: '+name)
    return nodes[0], ast.get_source_segment(text, nodes[0])


def execute_original(path, name, namespace):
    node, _ = source_function(path, name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class Tensor(np.ndarray):
    """Only the small tensor API reached by these locked source functions."""
    def unsqueeze(self, axis): return as_tensor(np.expand_dims(np.asarray(self), axis))
    def view(self, *shape): return as_tensor(np.asarray(self).reshape(*shape))
    def repeat(self, *counts): return as_tensor(np.tile(np.asarray(self), counts))
    def contiguous(self): return as_tensor(np.ascontiguousarray(self))
    def new_tensor(self, value): return as_tensor(value, self.dtype)
    def transpose(self, a, b): return as_tensor(np.swapaxes(np.asarray(self), a, b))
    def permute(self, *axes): return as_tensor(np.transpose(np.asarray(self), axes))
    def float(self): return self.astype(np.float64)


def as_tensor(value, dtype=np.float64):
    return np.asarray(value, dtype=dtype).view(Tensor)


class NumpyOps:
    float = np.float64
    @staticmethod
    def stack(values, dim=0): return as_tensor(np.stack([np.asarray(x) for x in values], axis=dim))
    @staticmethod
    def cat(values, dim=0): return as_tensor(np.concatenate([np.asarray(x) for x in values], axis=dim))
    @staticmethod
    def ones_like(value): return as_tensor(np.ones_like(np.asarray(value)))
    @staticmethod
    def matmul(left, right): return as_tensor(np.matmul(np.asarray(left), np.asarray(right)))
    @staticmethod
    def cumsum(value, dim): return as_tensor(np.cumsum(np.asarray(value), axis=dim))
    @staticmethod
    def allclose(a, b, **kw): return np.allclose(np.asarray(a), np.asarray(b), **kw)
    @staticmethod
    def linspace(start, stop, count, dtype=None, device=None):
        return as_tensor(np.linspace(start, stop, count, dtype=dtype))
    @staticmethod
    def meshgrid(a, b): return tuple(as_tensor(x) for x in np.meshgrid(a, b, indexing='ij'))
    @staticmethod
    def no_grad(): return contextlib.nullcontext()


def geometry_utils(ops):
    namespace = dict(torch=ops, np=np, copy=copy)
    functions = {name:execute_original(GRID_SOURCE, name, namespace) for name in
                 ('get_bev_grids', 'bev_grids_to_coordinates', 'bev_coords_to_grids')}
    if ops is not NumpyOps:
        original = functions['get_bev_grids']
        functions['get_bev_grids'] = lambda h,w,b:original(h,w,b,device='cpu',dtype=torch.float64)
    return types.SimpleNamespace(**functions)


def pose(x=0., y=0., yaw=0.):
    c, s = np.cos(yaw), np.sin(yaw)
    result = np.eye(4)
    result[:2,:2] = [[c,-s],[s,c]]; result[:2,3] = [x,y]
    return result


def metas(identity=False, batch=2):
    poses = [np.eye(4)]*5 if identity else [pose(),pose(2,0),pose(3,1,.2),pose(4,2,.6),pose(5,2,1.)]
    return [dict(future2ref_lidar_transform=[p.T for p in poses],
                 ref2future_lidar_transform=[np.linalg.inv(p).T for p in poses],
                 future_can_bus=np.arange(90).reshape(5,18)) for _ in range(batch)]


class NumpyHead:
    def __init__(self):
        self.embed_dims=2; self.use_plan_traj=False; self.bev_pred_head=[None]*3; self.trace=[]
    def __call__(self, memory, img_metas, step, action, condition, **kw):
        self.trace.append(dict(memory=memory.copy(),step=step,action=copy.deepcopy(action),
                               condition=copy.deepcopy(condition),refs=kw['ref_points'].copy(),
                               tgt=kw['tgt_points'].copy()))
        x=.7*memory[:,0]+.05*step+.02*kw['ref_points'][:,:,0]
        x=x+.001*condition['future2history'].sum(axis=(1,2,3))[:,None,None]
        return NumpyOps.stack([x,x+.01,x+.02]), None
    def forward_head(self, features): return features


def numpy_model(mode=None):
    utils=geometry_utils(NumpyOps)
    namespace=dict(torch=NumpyOps,np=np,e2e_predictor_utils=utils)
    original=execute_original(SOURCE,'future_pred',namespace)
    align=execute_original(SOURCE,'_align_bev_coordnates',namespace)
    model=types.SimpleNamespace(future_pred_head=NumpyHead(),bev_h=3,bev_w=5,
        point_cloud_range=[-4.,-3.,-1.,6.,3.,1.],training=True,future_pred_frame_num=4,
        test_future_frame_num=4,turn_on_plan=False,motion_residual=None,
        doppler_posterior=None,doppler_advection=None,physical_trace=[])
    def history(self, features, count, old, ref):
        if count!=1:raise ValueError('one slot')
        return as_tensor(np.tile(np.eye(4),(features.shape[0],1,1,1)))
    def traced_align(self, step, physical, metadata, plan):
        out=align(self,step,physical,metadata,plan)
        self.physical_trace.append((physical.copy(),out[2].copy(),out[3].copy()))
        return out
    model._get_history_ref_to_previous_transform=types.MethodType(history,model)
    model._align_bev_coordnates=types.MethodType(traced_align,model)
    if mode is not None:
        original,_=fc.derive_future_pred(source_function(SOURCE,'future_pred')[1],namespace,mode)
    model.future_pred=types.MethodType(original,model)
    return model


def run_numpy(model, identity=False, valid=(1,2,3,4), condition=None):
    batch=2
    x=as_tensor(np.arange(batch*15*2).reshape(batch,1,15,2)/100.)
    action=dict(command=as_tensor(np.arange(10).reshape(2,5)),
                vel_steering=as_tensor(np.arange(40).reshape(2,5,4)))
    args=(x,action,dict(occ_gts=None) if condition is None else condition,
          dict(ref_pose_pred=None,gt_traj=as_tensor(np.arange(16).reshape(2,4,2))),
          list(valid),metas(identity,batch),[[{}] for _ in range(batch)],1)
    return model.future_pred(*args)[0]


class GeometryAndSourceTests(unittest.TestCase):
    def test_locked_original_sources_and_exact_edit_count(self):
        self.assertEqual(fc.file_sha(SOURCE),fc.SOURCE_SHA256)
        self.assertEqual(fc.file_sha(GRID_SOURCE),fc.GRID_SOURCE_SHA256)
        _,source=source_function(SOURCE,'future_pred')
        _,align=source_function(SOURCE,'_align_bev_coordnates')
        fc.method_node(align,'_align_bev_coordnates',fc.ALIGN_AST_SHA256)
        node=source_function(SOURCE,'future_pred')[0]
        for mode in fc.MODES:
            derived,receipt=fc.derive_future_pred(source,{},mode)
            self.assertEqual(list(inspect.signature(derived).parameters),[a.arg for a in node.args.args])
            self.assertEqual(receipt['structural_changes'],dict(input_guard=1,
                separate_R_spatial_queue=int(mode=='reference'),replace_cross_attention_grid=int(mode=='reference')))
            # Remove only advertised inserted AST statements: original body must
            # be byte-for-byte identical in canonical form, including queue,
            # conditioning, output head and no_grad/BPTT branches.
            tree=ast.parse(inspect.getsource(fc.derive_future_pred))
            self.assertNotIn('detach(',ast.unparse(tree))
        with self.assertRaisesRegex(ValueError,'Unreviewed'):
            fc.derive_future_pred(source.replace('range(1, future_frame_num + 1)','range(2, future_frame_num + 1)'),{},'reference')

    def test_native_control_and_identity_geometry_full_rollout(self):
        for identity in (False,True):
            original=numpy_model(); control=numpy_model('native1')
            np.testing.assert_array_equal(run_numpy(original,identity),run_numpy(control,identity))
        control=numpy_model('native1');reference=numpy_model('reference')
        expected=run_numpy(control,True);actual=run_numpy(reference,True)
        self.assertEqual(actual.shape,(5,3,2,15,2))
        np.testing.assert_array_equal(actual,expected)

    def test_nonzero_pose_spatial_and_physical_queues_are_independent(self):
        control=numpy_model('native1');reference=numpy_model('reference')
        c=run_numpy(control);a=run_numpy(reference)
        self.assertFalse(np.array_equal(c,a)) # intentional function intervention
        for i,(ct,at) in enumerate(zip(control.future_pred_head.trace,reference.future_pred_head.trace)):
            self.assertEqual(ct['step'],at['step']);self.assertEqual(at['step'],i+1)
            np.testing.assert_array_equal(ct['condition']['future2history'],at['condition']['future2history'])
            for key in ct['action']:np.testing.assert_array_equal(ct['action'][key],at['action'][key])
            for left,right in zip(control.physical_trace[i],reference.physical_trace[i]):
                np.testing.assert_array_equal(left,right)
            np.testing.assert_array_equal(at['tgt'],ct['tgt'])
            np.testing.assert_allclose(at['refs'][:,:,0],at['tgt'],atol=2e-16,rtol=0)
            self.assertEqual(at['refs'].shape,(2,15,1,2))
            if i:np.testing.assert_array_equal(at['refs'],reference.future_pred_head.trace[0]['refs'])
        self.assertTrue(any(not np.array_equal(c['refs'],a['refs']) for c,a in
            zip(control.future_pred_head.trace,reference.future_pred_head.trace)))
        # Independent column-vector expression: physical ego conditioning has
        # the original transpose and successive time frames, not identity.
        poses=metas()[0]['future2ref_lidar_transform']
        for i,entry in enumerate(reference.physical_trace):
            expected=np.linalg.inv(poses[i].T)@poses[i+1].T
            np.testing.assert_allclose(entry[2][0,0],expected,atol=1e-14)

    def test_rectangle_centers_axis_order_and_static_point(self):
        utils=geometry_utils(NumpyOps);grid=utils.get_bev_grids(3,5,2)
        physical=as_tensor(np.tile(np.eye(4),(2,1,1,1)));spatial=fc._reference_memory(physical,NumpyOps)
        actual=fc._reference_grids(grid,spatial,3,5,[-4,-3,-1,6,3,1],utils,NumpyOps)
        self.assertEqual(actual.shape,(2,15,1,2))
        np.testing.assert_allclose(actual[0,0,0],[.1,1/6],atol=1e-15)
        np.testing.assert_allclose(actual[0,-1,0],[.9,5/6],atol=1e-15)
        # A world-stationary point at an exact cell center stays at the same R
        # index through arbitrary ego translations/yaw. No learned claim.
        point=np.array([1.,0.,1.,1.]);normalized=np.array([.5,.5])
        for ego in (pose(),pose(2),pose(yaw=np.pi/2),pose(2,yaw=np.pi/2)):
            np.testing.assert_array_equal(point@np.eye(4),point)
            if not np.array_equal(ego,np.eye(4)):
                self.assertFalse(np.allclose((point@ego.T)[:2],point[:2]))
        np.testing.assert_allclose(actual[0,7,0],normalized,atol=1e-15)

    def test_fail_closed_shape_labels_and_temporal_identity(self):
        for mode in fc.MODES:
            with self.assertRaisesRegex(ValueError,'Future GT'):
                run_numpy(numpy_model(mode),condition={'occ_gts':np.ones((1,7,2,2,2))})
            with self.assertRaisesRegex(ValueError,'all four'):
                run_numpy(numpy_model(mode),valid=(4,))
        with self.assertRaisesRegex(ValueError,'one observed'):
            fc._reference_memory(as_tensor(np.tile(np.eye(4),(2,2,1,1))),NumpyOps)
        wrong=as_tensor(np.eye(4)).view(1,1,4,4);wrong[0,0,3,0]=2.
        with self.assertRaisesRegex(ValueError,'not in R'):fc._reference_memory(wrong,NumpyOps)
        self.assertEqual([fc.slot_contract(t)['memory_time_index'] for t in range(1,5)],[0,1,2,3])
        self.assertTrue(all(fc.slot_contract(t)['memory_spatial_frame']=='R' for t in range(1,5)))


@unittest.skipIf(torch is None,'torch unavailable: CPU autograd and real GPU preflight remain unverified')
class TorchGradientTests(unittest.TestCase):
    def test_full_original_rollout_last_horizon_reaches_earlier_states(self):
        class Head(torch.nn.Module):
            def __init__(self):
                super().__init__();self.embed_dims=2;self.use_plan_traj=False;self.bev_pred_head=[None]*3
                self.linear=torch.nn.Linear(2,2,bias=False).double();self.states=[]
                with torch.no_grad():self.linear.weight.copy_(torch.eye(2,dtype=torch.float64)*.7)
            def forward(self,memory,metadata,step,action,condition,**kw):
                value=memory[:,0].reshape(2,3,5,2).permute(0,3,1,2)
                grid=(2*kw['ref_points'][:,:,0]-1).reshape(2,3,5,2)
                sampled=torch.nn.functional.grid_sample(value,grid,align_corners=False,padding_mode='zeros')
                x=self.linear(sampled.permute(0,2,3,1).reshape(2,15,2))+.01*step
                x.retain_grad();self.states.append(x)
                return torch.stack([x,x+.01,x+.02]),None
            def forward_head(self,features):return features
        for mode in fc.MODES:
            utils=geometry_utils(torch);namespace=dict(torch=torch,np=np,e2e_predictor_utils=utils)
            align=execute_original(SOURCE,'_align_bev_coordnates',namespace)
            forward,_=fc.derive_future_pred(source_function(SOURCE,'future_pred')[1],namespace,mode)
            model=types.SimpleNamespace(future_pred_head=Head(),bev_h=3,bev_w=5,
                point_cloud_range=[-40.,-30.,-1.,60.,30.,1.],training=True,
                future_pred_frame_num=4,test_future_frame_num=4,turn_on_plan=False,
                motion_residual=None,doppler_posterior=None,doppler_advection=None)
            model._align_bev_coordnates=types.MethodType(align,model)
            model._get_history_ref_to_previous_transform=lambda x,*args:torch.eye(4,dtype=x.dtype).repeat(2,1,1,1)
            model.future_pred=types.MethodType(forward,model)
            x=torch.ones(2,1,15,2,dtype=torch.float64,requires_grad=True)
            result=model.future_pred(x,{},dict(occ_gts=None),
                dict(ref_pose_pred=None,gt_traj=torch.zeros(2,4,2,dtype=torch.float64)),
                [1,2,3,4],metas(),[[{}],[{}]],1)[0]
            result[-1].square().mean().backward()
            self.assertGreater(float(x.grad.abs().sum()),0)
            self.assertGreater(float(model.future_pred_head.linear.weight.grad.abs().sum()),0)
            for state in model.future_pred_head.states:
                self.assertTrue(torch.isfinite(state.grad).all())
                self.assertGreater(float(state.grad.abs().sum()),0)


if __name__=='__main__':
    unittest.main(verbosity=2)
