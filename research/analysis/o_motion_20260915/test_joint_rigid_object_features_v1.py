"""CPU analytic tests only; no data, native model, optimizer, or CUDA.

Run explicitly in a Torch-equipped environment:
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python test_joint_rigid_object_features_v1.py
Float32 sampling/chunk tests allow declared roundoff; initial zero/arm/payload
and input nonmutation tests require exact bytes. These are synthetic algebraic
checks, never model performance evidence. This file does not auto-install Torch.
"""
import unittest

import torch

from joint_rigid_object_features_v1 import (
    JointRigidObjectFeatures, bev_query_centers, gaussian_object_aggregate, sample_current_context,
)
from rigid_object_motion_v1 import future_poses, so3_exp


def byte_equal(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


def fixture(count=2):
    centers=torch.tensor([[-.45,.25,.2],[.65,-.35,-.1]],dtype=torch.float64)[:count].clone()
    rotation=so3_exp(torch.tensor([[.2,-.1,.35],[-.15,.3,-.2]],dtype=torch.float64)[:count])
    size=torch.tensor([[.8,1.8,1.2],[1.2,.7,1.5]],dtype=torch.float64)[:count].clone()
    velocity=torch.tensor([[.3,-.1,.04],[-.15,.25,-.08]],dtype=torch.float64)[:count].clone()
    score=torch.tensor([.7,.3],dtype=torch.float64)[:count].clone()
    packed=torch.zeros((count,27),dtype=torch.float32)
    packed[:,:3]=centers.float()/51.2;packed[:,3:6]=size.float()/51.2
    packed[:,6:15]=rotation.float().reshape(count,9);packed[:,15]=1.
    packed[:,23]=score.float();packed[:,24:]=velocity.float()/20.
    bev=torch.linspace(-.5,.5,3*4*256,dtype=torch.float32).reshape(1,12,256)
    return dict(packed_states=packed,current_bev=bev,c0_R=centers,R0_R=rotation,wlh=size,velocity_R=velocity,score=score,
                grid_shape_yx=(3,4),extent_xy=(-2.,-1.5,2.,1.5))


def constant_values_and_open_projection(model):
    # Remove the value dependence only in this analytic test, to isolate the
    # address Jacobian instead of accidentally testing the shared value path.
    with torch.no_grad():
        for p in model.value_encoder.parameters(): p.zero_()
        model.value_encoder[2].bias.fill_(1.)
        model.native_projection.weight.zero_()
        model.native_projection.weight[0,0]=1.


class JointRigidFeaturesTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)

    def test_sampling_axes_cell_centres_rotation_and_zeros_padding(self):
        args=fixture(1)
        xy=bev_query_centers(grid_shape_yx=args['grid_shape_yx'],extent_xy=args['extent_xy'],device=torch.device('cpu'))
        expected=torch.tensor([[-1.5,-1.],[-.5,-1.],[.5,-1.],[1.5,-1.],[-1.5,0.]],dtype=torch.float64)
        self.assertTrue(byte_equal(xy[:5],expected))
        bev=torch.zeros((1,12,256),dtype=torch.float32)
        bev[0,:,0]=xy[:,0];bev[0,:,1]=xy[:,1];bev[0,:,2]=xy[:,0]+10*xy[:,1]
        centre=torch.tensor([[.2,.1,.4]],dtype=torch.float64)
        size=torch.tensor([[.2,.4,.8]],dtype=torch.float64)
        context,points=sample_current_context(bev,centre,args['R0_R'],size,
                                               grid_shape_yx=(3,4),extent_xy=args['extent_xy'])
        # All points are in the affine field interior, so the five-point mean
        # is exactly the centre analytically. align_corners=True would fail.
        # Float32 bilinear coordinates/features, with channel magnitude ~10.
        torch.testing.assert_close(context[0,:3],torch.tensor([.2,.1,1.2]),rtol=0.,atol=2e-6)
        local=torch.tensor([-.2,-.1,0.],dtype=torch.float64)
        torch.testing.assert_close(points[0,1],args['R0_R'][0]@local+centre[0],rtol=0.,atol=1e-15)
        far=centre+100.
        outside,_=sample_current_context(bev,far,args['R0_R'],size,grid_shape_yx=(3,4),extent_xy=args['extent_xy'])
        self.assertTrue(byte_equal(outside,torch.zeros_like(outside)))

    def test_initial_arms_zero_native_and_inputs_unmodified(self):
        model=JointRigidObjectFeatures();args=fixture()
        originals={k:v.clone() for k,v in args.items() if isinstance(v,torch.Tensor)}
        rng=torch.random.get_rng_state().clone()
        c=model(**args,coupled=True);f=model(**args,coupled=False)
        self.assertEqual(sum(p.numel() for p in model.parameters()),67366)
        self.assertIsNone(model.native_projection.bias)
        for k in ('delta_center_R','delta_rotvec_R','values','native_delta','query_centers_R','query_rotations_R'):
            self.assertTrue(byte_equal(c[k],f[k]),k)
        for k in ('delta_center_R','delta_rotvec_R','native_delta'):
            self.assertTrue(byte_equal(c[k],torch.zeros_like(c[k])),k)
        for k,v in originals.items():self.assertTrue(byte_equal(v,args[k]),k)
        self.assertTrue(byte_equal(rng,torch.random.get_rng_state()))
        self.assertEqual(c['native_delta'].shape,(4,1,12,256))

    def test_nonzero_pose_only_changes_address_edge_values_remain_identical(self):
        model=JointRigidObjectFeatures();args=fixture()
        with torch.no_grad():
            model.pose_head.bias.copy_(torch.tensor([.2,-.1,.05,.1,-.2,.15]))
            model.native_projection.weight[0,0]=1.
        c=model(**args,coupled=True);f=model(**args,coupled=False)
        for k in ('values','delta_center_R','delta_rotvec_R','future_centers_R','future_rotations_R'):
            self.assertTrue(byte_equal(c[k],f[k]),k)
        self.assertFalse(byte_equal(c['query_centers_R'],f['query_centers_R']))
        self.assertFalse(byte_equal(c['query_rotations_R'],f['query_rotations_R']))
        self.assertFalse(byte_equal(c['native_delta'],f['native_delta']))
        torch.testing.assert_close(c['delta_center_R'][:,0],torch.tensor([.5,1.,1.5,2.])[:,None]*model.pose_head.bias[:3],rtol=0.,atol=0.)
        payload_gradients=torch.autograd.grad(c['values'].sum(),(c['delta_center_R'],c['delta_rotvec_R']))
        self.assertTrue(all(bool(torch.isfinite(g).all()) and float(g.abs().sum())>0 for g in payload_gradients))

    def test_empty_all_outputs_finite_and_native_zero_after_projection_change(self):
        model=JointRigidObjectFeatures();constant_values_and_open_projection(model)
        args=fixture(0)
        for coupled in (False,True):
            result=model(**args,coupled=coupled)
            for value in result.values():self.assertTrue(bool(torch.isfinite(value).all()))
            self.assertEqual(result['delta_center_R'].shape,(4,0,3))
            self.assertEqual(result['delta_rotvec_R'].shape,(4,0,3))
            self.assertEqual(result['values'].shape,(4,0,32))
            self.assertEqual(result['query_centers_R'].shape,(4,0,3))
            self.assertEqual(result['query_rotations_R'].shape,(4,0,3,3))
            self.assertTrue(byte_equal(result['native_delta'],torch.zeros((4,1,12,256))))
            gradient,=torch.autograd.grad(result['native_delta'].sum(),model.pose_head.bias)
            self.assertTrue(bool(torch.isfinite(gradient).all()) and bool((gradient==0).all()))

    def test_pose_to_native_address_gradients_and_fix_control(self):
        model=JointRigidObjectFeatures();args=fixture();constant_values_and_open_projection(model)
        for coupled in (True,False):
            result=model(**args,coupled=coupled)
            # Off-centre query and anisotropic tilted boxes avoid a symmetric
            # stationary derivative; zero residual rotations remain tested.
            objective=result['native_delta'][3,0,0,0]
            grads=torch.autograd.grad(objective,(result['delta_center_R'],result['delta_rotvec_R']))
            for gradient in grads:
                self.assertTrue(bool(torch.isfinite(gradient).all()))
                if coupled:self.assertGreater(float(gradient.abs().sum()),0.)
                else:self.assertTrue(bool((gradient==0).all()))

    def test_chunking_and_gaussian_formula(self):
        args=fixture();dc=torch.full((4,2,3),.07,dtype=torch.float32,requires_grad=True)
        dr=torch.full((4,2,3),.12,dtype=torch.float32,requires_grad=True)
        ch,rh=future_poses(args['c0_R'],args['R0_R'],args['velocity_R'],dc,dr)
        xy=bev_query_centers(grid_shape_yx=(3,4),extent_xy=args['extent_xy'],device=torch.device('cpu'))
        values=torch.arange(4*2*32,dtype=torch.float32).reshape(4,2,32)/100.
        kwargs=dict(values=values,centers_R=ch,rotations_R=rh,wlh=args['wlh'],score=args['score'],query_xy=xy,cell_xy=(1.,1.))
        whole=gaussian_object_aggregate(**kwargs,query_chunk_size=12)
        for block in (1,5,2048):
            actual=gaussian_object_aggregate(**kwargs,query_chunk_size=block)
            torch.testing.assert_close(actual,whole,rtol=1e-6,atol=1e-7)
        # Exercise the actual production boundary and its three-query tail.
        long_xy=torch.stack((torch.linspace(-2.,2.,2051,dtype=torch.float64),
                             torch.linspace(-1.,1.,2051,dtype=torch.float64)),dim=-1)
        with torch.no_grad():
            long_kwargs=dict(kwargs,query_xy=long_xy)
            full=gaussian_object_aggregate(**long_kwargs,query_chunk_size=2051)
            blocked=gaussian_object_aggregate(**long_kwargs,query_chunk_size=2048)
        torch.testing.assert_close(blocked,full,rtol=1e-6,atol=1e-7)
        # Independent dense formula for one horizon, including l/w order,
        # rotated height contribution and the +1 zero-background denominator.
        covariance=rh[0]@torch.diag_embed((args['wlh'][:,[1,0,2]]/2).square())@rh[0].transpose(-1,-2)
        covariance=covariance[:,:2,:2]+torch.eye(2,dtype=torch.float64)/12.
        d=xy[:,None]-ch[0,None,:,:2]
        weight=(torch.exp(-.5*((d.unsqueeze(-2)@torch.linalg.inv(covariance)[None])@d.unsqueeze(-1)).squeeze(-1).squeeze(-1))*args['score']).float()
        expected=(weight[:,:,None]*values[0,None]).sum(1)/(1+weight.sum(1,keepdim=True))
        torch.testing.assert_close(whole[0],expected,rtol=1e-6,atol=1e-7)
        gradients=torch.autograd.grad(whole.square().sum(),(dc,dr))
        self.assertTrue(all(bool(torch.isfinite(g).all()) and float(g.abs().sum())>0 for g in gradients))

    def test_negative_score_is_rejected_not_clamped(self):
        args=fixture();args['score'][0]=-.1
        with self.assertRaisesRegex(ValueError,'nonnegative'):
            JointRigidObjectFeatures()(**args,coupled=True)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
