"""Separate recurrent state from rendered source content.

The fixed-material arm keeps exactly the evolving arm's recurrent dynamics.
Only renderer content is held at the encoded current value. This is a dense
source-indexed latent, not a claim of recovered physical object identity.
"""
import torch

from dense_task_state_v2 import DenseTaskState
from transport_ops import forward_splat_3d


class DenseMaterialState(DenseTaskState):
    def __init__(self, *args, material_mode='evolving', **kwargs):
        super().__init__(*args, **kwargs)
        if material_mode not in ('evolving', 'fixed'):
            raise ValueError(material_mode)
        self.material_mode = material_mode

    def forward(self, tokens, arm='transport', intervention=None):
        if arm not in ('transport', 'direct') or intervention not in (None, 'zero', 'reverse'):
            raise ValueError('Unknown readout or intervention')
        origin = state = self.encode(tokens)
        current_material = self.to_xyz(origin, self.channels)
        logits = [self.decode(current_material, torch.ones_like(current_material[:, :1]))]
        displacement = current_material.new_zeros(current_material.shape[0], 3, *self.shape)
        flows, coverages = [], []
        for h in range(4):
            time = state.new_full((state.shape[0], 1, *state.shape[2:]), (h + 1) / 4.)
            hidden = self.dynamics(torch.cat((origin, state, time), dim=1))
            state = state + self.content_increment(hidden)
            displacement = displacement + .5 * self.to_xyz(self.velocity(hidden), 3)
            # Do not detach recurrence: it can affect later displacement.
            material = current_material if self.material_mode == 'fixed' else self.to_xyz(state, self.channels)
            flows.append(displacement)
            if arm == 'transport' and intervention != 'zero':
                address = -displacement if intervention == 'reverse' else displacement
                splat = forward_splat_3d(material, address, extent=self.extent)
                feature, coverage = splat['normalized'], splat['coverage']
            else:
                feature, coverage = material, torch.ones_like(material[:, :1])
            logits.append(self.decode(feature, coverage)); coverages.append(coverage)
        return dict(logits=torch.stack(logits, dim=1),
                    displacement=torch.stack(flows, dim=1),
                    coverage=torch.stack(coverages, dim=1))


def mechanism_checks():
    """CPU nonzero-state parity and graph checks, independent of train GT."""
    torch.manual_seed(731)
    kwargs=dict(shape=(7,9,4),extent=(-3.5,-4.5,-2.,3.5,4.5,2.))
    base=DenseTaskState(**kwargs)
    with torch.no_grad():
        base.content_increment.weight.normal_(0,.01)
        base.velocity.weight.normal_(0,.01)
    free=DenseMaterialState(**kwargs);free.load_state_dict(base.state_dict())
    fixed=DenseMaterialState(**kwargs,material_mode='fixed');fixed.load_state_dict(base.state_dict())
    tokens=torch.randn(1,63,256)
    with torch.no_grad():
        a=base(tokens);b=free(tokens);c=fixed(tokens);z=fixed(tokens,intervention='zero')
        assert torch.equal(a['logits'],b['logits'])
        assert torch.equal(a['displacement'],c['displacement'])
        assert torch.equal(c['logits'][:,0],a['logits'][:,0])
        assert torch.equal(z['logits'],z['logits'][:,:1].expand_as(z['logits']))
        assert not torch.equal(c['logits'],z['logits'])
    # Last increment has no path to a later velocity, isolating rendered
    # content from recurrent-state paths without removing recurrent capacity.
    gradient_records={}
    for name,model in [('fixed',fixed),('evolving',free)]:
        deltas=[]
        hook=model.content_increment.register_forward_hook(lambda mod,args,value:deltas.append(value))
        try:output=model(tokens)
        finally:hook.remove()
        loss=output['logits'][:,1:].square().mean()
        grad=torch.autograd.grad(loss,[deltas[-1],model.velocity.weight,model.content_increment.weight],allow_unused=True)
        if name=='fixed':assert grad[0] is None
        else:assert grad[0] is not None and grad[0].norm()>0
        assert all(g is not None and torch.isfinite(g).all() and g.norm()>0 for g in grad[1:])
        gradient_records[name]=dict(last_increment_gradient=None if grad[0] is None else float(grad[0].norm()),
            velocity_gradient=float(grad[1].norm()),recurrent_increment_gradient=float(grad[2].norm()))
    return dict(native_evolving_logits_exact=True,same_parameter_motion_exact=True,
        fixed_zero_is_current_persistence=True,last_increment_direct_readout_path_removed=True,
        recurrent_increment_indirect_path_retained=True,gradients=gradient_records)
