"""Current object context -> shared rigid corrections and future BEV features.

Pure single-sample module. It consumes no labels, ownership, sparse support,
future ego poses or timestamps. All 27 packed features and all eligible objects
remain in their caller-supplied order. Current BEV and float64 geometry are
fixed observations. Learned pose residuals and values retain their gradients.

BEV columns are y*X+x. grid_shape_yx=(Y,X); extent_xy=(xmin,ymin,xmax,ymax).
These are XY bounds, not the six-value XYZ extent used by material geometry.
The production forward uses exactly 2048 query columns per chunk, including
the final shorter chunk, and every object in every chunk; no top-k or mask.

Cpl/Fix differ ONLY in poses used for the occupancy spatial kernel. Both use
the same learned corrections in their object values and physical pose output.
The final projection has no bias: an empty object set remains zero after any
parameter update. Its weights start at zero, so the initial native delta is
zero. Consequently initial occupancy gradients into earlier new layers are
zero until this projection changes; the physical pose path is not gated here.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from rigid_object_motion_v1 import HORIZONS_SECONDS, future_poses


QUERY_CHUNK_SIZE = 2048


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _tensor(value, shape, dtype, device, name, *, fixed=False):
    _require(isinstance(value, torch.Tensor) and tuple(value.shape) == tuple(shape), name+' shape differs')
    _require(value.dtype == dtype and value.device == device, name+' dtype/device differs')
    _require(bool(torch.isfinite(value).all()), name+' contains nonfinite values')
    if fixed:
        _require(not value.requires_grad, name+' must be a fixed observation')


def _grid_contract(grid_shape_yx, extent_xy):
    _require(len(grid_shape_yx) == 2 and all(type(v) is int and v > 0 for v in grid_shape_yx),
             'grid_shape_yx must be positive integer (Y,X)')
    _require(len(extent_xy) == 4, 'extent_xy must be (xmin,ymin,xmax,ymax)')
    bounds = tuple(float(v) for v in extent_xy)
    _require(all(math.isfinite(v) for v in bounds) and bounds[2] > bounds[0] and bounds[3] > bounds[1],
             'extent_xy must be finite increasing bounds')
    y, x = grid_shape_yx
    return y, x, bounds, ((bounds[2]-bounds[0])/x, (bounds[3]-bounds[1])/y)


def sample_current_context(current_bev, c0_R, R0_R, wlh, *, grid_shape_yx, extent_xy):
    """Return [M,256] five-point means and [M,5,3] sampling positions in R.

    Points are centre plus four box-local footprint corners (local z=0).
    Complete current rotation maps each corner to R before taking XY.
    grid_sample is bilinear, zeros padding, align_corners=False. Objects are
    retained even when every point lies outside the BEV bounds.
    """
    y, x, bounds, _ = _grid_contract(grid_shape_yx, extent_xy)
    _require(isinstance(c0_R, torch.Tensor) and c0_R.ndim == 2 and c0_R.shape[-1] == 3,
             'c0_R must be [M,3]')
    count, device = c0_R.shape[0], c0_R.device
    _tensor(c0_R, (count,3), torch.float64, device, 'c0_R', fixed=True)
    _tensor(R0_R, (count,3,3), torch.float64, device, 'R0_R', fixed=True)
    _tensor(wlh, (count,3), torch.float64, device, 'wlh', fixed=True)
    _tensor(current_bev, (1,y*x,256), torch.float32, device, 'current_bev', fixed=True)
    _require(bool((wlh > 0).all()), 'wlh must be strictly positive')
    signs = c0_R.new_tensor([[0.,0.,0.], [-1.,-1.,0.], [-1.,1.,0.], [1.,-1.,0.], [1.,1.,0.]])
    half_lwh = wlh[:, [1,0,2]] / 2.
    local = half_lwh[:, None, :] * signs[None]
    points = torch.einsum('mij,mpj->mpi', R0_R, local) + c0_R[:, None, :]
    if count == 0:
        return current_bev.new_zeros((0,256)), points
    lower = c0_R.new_tensor(bounds[:2])
    span = c0_R.new_tensor([bounds[2]-bounds[0], bounds[3]-bounds[1]])
    normalized = (2. * (points[..., :2]-lower) / span - 1.).to(torch.float32)
    image = current_bev.transpose(1,2).reshape(1,256,y,x)
    samples = F.grid_sample(image, normalized.unsqueeze(0), mode='bilinear',
                            padding_mode='zeros', align_corners=False)
    return samples[0].mean(-1).transpose(0,1), points


def bev_query_centers(*, grid_shape_yx, extent_xy, device):
    """Fixed float64 [Y*X,2] cell centres, row-major y*X+x."""
    y, x, bounds, cell = _grid_contract(grid_shape_yx, extent_xy)
    xs = bounds[0] + (torch.arange(x, dtype=torch.float64, device=device)+.5)*cell[0]
    ys = bounds[1] + (torch.arange(y, dtype=torch.float64, device=device)+.5)*cell[1]
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    return torch.stack((xx,yy), dim=-1).reshape(y*x,2)


def gaussian_object_aggregate(values, centers_R, rotations_R, wlh, score, query_xy,
                              *, cell_xy, query_chunk_size=QUERY_CHUNK_SIZE):
    """Differentiable [4,Q,32] soft spatial feature field.

    w_j=score_j*exp(-.5*Mahalanobis^2), without Gaussian density normalization.
    Covariance is the XY block of R diag((l/2)^2,(w/2)^2,(h/2)^2) R^T plus
    diag(dx^2,dy^2)/12. Geometry/kernel arithmetic is float64; weights are
    cast to float32 for sum(w*value)/(1+sum(w)), retaining autograd. The 1
    is a zero-valued background contribution, not a clipping parameter.

    query_chunk_size is exposed for the analytic equivalence test only. The
    model forward always passes QUERY_CHUNK_SIZE. Chunking does not prune or
    approximate the object sum, but different float32 GEMM shapes need not be
    bit-identical. Empty objects have a finite zero/empty gradient path.
    """
    _require(isinstance(values, torch.Tensor) and values.ndim == 3 and values.shape[0] == 4
             and values.shape[2] == 32, 'values must be [4,M,32]')
    count, device = values.shape[1], values.device
    _tensor(values, (4,count,32), torch.float32, device, 'values')
    _tensor(centers_R, (4,count,3), torch.float64, device, 'centers_R')
    _tensor(rotations_R, (4,count,3,3), torch.float64, device, 'rotations_R')
    _tensor(wlh, (count,3), torch.float64, device, 'wlh', fixed=True)
    _tensor(score, (count,), torch.float64, device, 'score', fixed=True)
    _require(bool((wlh>0).all()) and bool((score>=0).all()), 'positive wlh and nonnegative score required; no clamping')
    _require(isinstance(query_xy, torch.Tensor) and query_xy.ndim == 2 and query_xy.shape[1] == 2,
             'query_xy must be [Q,2]')
    queries = query_xy.shape[0]
    _tensor(query_xy, (queries,2), torch.float64, device, 'query_xy', fixed=True)
    _require(type(query_chunk_size) is int and query_chunk_size>0, 'positive integer query chunk size required')
    _require(len(cell_xy)==2 and all(math.isfinite(float(v)) and float(v)>0 for v in cell_xy), 'positive finite cell_xy required')
    if count == 0 or queries == 0:
        zero_link = (values.sum() + centers_R.sum().to(torch.float32) + rotations_R.sum().to(torch.float32))*0.
        return values.new_zeros((4,queries,32)) + zero_link
    variance = (wlh[:, [1,0,2]]/2.).square()
    rotated = rotations_R * variance[None,:,None,:]
    covariance = (rotated @ rotations_R.transpose(-1,-2))[...,:2,:2]
    cell_variance = torch.diag(wlh.new_tensor(cell_xy).square()/12.)
    inverse = torch.linalg.inv(covariance + cell_variance)
    _require(bool(torch.isfinite(inverse).all()), 'nonfinite spatial covariance inverse')
    horizon_fields = []
    for h in range(4):
        chunks = []
        for start in range(0,queries,query_chunk_size):
            offset = query_xy[start:start+query_chunk_size,None,:] - centers_R[h,None,:,:2]
            quadratic = torch.einsum('qmi,mij,qmj->qm', offset, inverse[h], offset)
            weights = (score[None,:] * torch.exp(-.5*quadratic)).to(torch.float32)
            _require(bool(torch.isfinite(weights).all()), 'nonfinite spatial weights after float32 cast')
            chunks.append((weights @ values[h]) / (1. + weights.sum(-1,keepdim=True)))
        horizon_fields.append(torch.cat(chunks,dim=0))
    field = torch.stack(horizon_fields)
    _require(bool(torch.isfinite(field).all()), 'nonfinite spatial feature field')
    return field


class JointRigidObjectFeatures(nn.Module):
    """67,366 parameters; no native hooks, caches, model modes or RNG control."""
    def __init__(self):
        super().__init__()
        self.object_encoder = nn.Sequential(nn.Linear(284,128), nn.SiLU(), nn.Linear(128,128), nn.SiLU())
        self.pose_head = nn.Linear(128,6)
        nn.init.zeros_(self.pose_head.weight)
        nn.init.zeros_(self.pose_head.bias)
        self.value_encoder = nn.Sequential(nn.Linear(135,32), nn.SiLU(), nn.Linear(32,32), nn.SiLU())
        # Deliberately no bias: zero/empty aggregation must remain zero.
        self.native_projection = nn.Linear(32,256,bias=False)
        nn.init.zeros_(self.native_projection.weight)

    def forward(self, packed_states, current_bev, c0_R, R0_R, wlh, velocity_R, score,
                *, grid_shape_yx, extent_xy, coupled):
        _require(type(coupled) is bool, 'coupled must explicitly choose Cpl=True or Fix=False')
        _require(isinstance(packed_states,torch.Tensor) and packed_states.ndim==2 and packed_states.shape[1]==27,
                 'packed_states must be [M,27]')
        count,device = packed_states.shape[0],packed_states.device
        _tensor(packed_states,(count,27),torch.float32,device,'packed_states',fixed=True)
        for name,value,shape in (('c0_R',c0_R,(count,3)),('R0_R',R0_R,(count,3,3)),
                                 ('wlh',wlh,(count,3)),('velocity_R',velocity_R,(count,3)),('score',score,(count,))):
            _tensor(value,shape,torch.float64,device,name,fixed=True)
        _require(bool((score>=0).all()), 'score must be nonnegative; no implicit clamp')
        eye=torch.eye(3,dtype=torch.float64,device=device)
        _require(torch.allclose(R0_R.transpose(-1,-2)@R0_R,eye.expand(count,3,3),rtol=0.,atol=1e-6)
                 and torch.allclose(torch.linalg.det(R0_R),torch.ones(count,dtype=torch.float64,device=device),rtol=0.,atol=1e-6),
                 'R0_R must contain proper rotations')
        _require(all(p.dtype==torch.float32 and p.device==device for p in self.parameters()), 'module must use float32 input device')
        y,x,_,cell = _grid_contract(grid_shape_yx,extent_xy)
        context, sampling_points = sample_current_context(current_bev,c0_R,R0_R,wlh,
                                                          grid_shape_yx=grid_shape_yx,extent_xy=extent_xy)
        horizon=packed_states.new_tensor(HORIZONS_SECONDS).reshape(4,1,1)
        times=(horizon/2.).expand(4,count,1)
        inputs=torch.cat((packed_states[None].expand(4,-1,-1),context[None].expand(4,-1,-1),times),dim=-1)
        hidden=self.object_encoder(inputs)
        residual=self.pose_head(hidden)*horizon
        delta_center_R,delta_rotvec_R=residual[...,:3],residual[...,3:]
        values=self.value_encoder(torch.cat((hidden,delta_center_R,delta_rotvec_R,times),dim=-1))
        # The physical state always uses the learned residuals in both arms.
        centers,rotations=future_poses(c0_R,R0_R,velocity_R,delta_center_R,delta_rotvec_R)
        if coupled:
            query_centers,query_rotations=centers,rotations
        else:
            # Only this address edge is replaced. No learned payload detaches.
            query_centers,query_rotations=future_poses(c0_R,R0_R,velocity_R,
                                                       torch.zeros_like(delta_center_R),torch.zeros_like(delta_rotvec_R))
        xy=bev_query_centers(grid_shape_yx=grid_shape_yx,extent_xy=extent_xy,device=device)
        field=gaussian_object_aggregate(values,query_centers,query_rotations,wlh,score,xy,
                                        cell_xy=cell,query_chunk_size=QUERY_CHUNK_SIZE)
        native_delta=self.native_projection(field).unsqueeze(1)
        _require(tuple(native_delta.shape)==(4,1,y*x,256) and bool(torch.isfinite(native_delta).all()), 'native delta invalid')
        return dict(delta_center_R=delta_center_R,delta_rotvec_R=delta_rotvec_R,values=values,native_delta=native_delta,
                    future_centers_R=centers,future_rotations_R=rotations,
                    query_centers_R=query_centers,query_rotations_R=query_rotations,
                    current_context=context,current_sampling_points_R=sampling_points)
