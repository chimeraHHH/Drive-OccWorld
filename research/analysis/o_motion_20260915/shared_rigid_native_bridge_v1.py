"""Explicit layout conversion and feature injection for the shared-state probe.

No model construction, weights, inputs, labels, optimizer, or scheduler are
owned here. The caller computes the four object feature fields from current
observations, then replays the original native model inside the context.
This is an implementation component, not a completed native preflight.
"""
from contextlib import contextmanager

import torch

from rigid_object_motion_v1 import _ExactZeroResidual


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_field_to_native_layout(field, grid_shape_xyz=(200, 200, 16)):
    """[4,X*Y*Z,3], Z-fastest source points -> [1,4,3,X,Y,Z]."""
    shape = tuple(grid_shape_xyz)
    require(len(shape) == 3 and all(type(x) is int and x > 0 for x in shape),
            'Expected a positive XYZ grid shape')
    nx, ny, nz = shape
    require(isinstance(field, torch.Tensor)
            and tuple(field.shape) == (4, nx * ny * nz, 3)
            and field.dtype == torch.float64,
            'Expected the complete float64 material field')
    return field.reshape(4, nx, ny, nz, 3).permute(0, 4, 1, 2, 3).unsqueeze(0).contiguous()


@contextmanager
def inject_future_object_features(head, native_delta):
    """Add one shared object field at the three native decoder entrances.

    native_delta is float32 [4,1,HW,256], in column order y*X+x.
    Each branch must be called once with [5,1,HW,256]. t0 is copied verbatim;
    four future features receive additive corrections with live gradients.
    A zero correction byte-preserves the original feature, including -0.
    No logits are modified, and native transformer/pose routing is untouched.
    The context removes every installed hook on success and on exception.
    """
    require(isinstance(native_delta, torch.Tensor)
            and native_delta.ndim == 4
            and tuple(native_delta.shape[:2]) == (4, 1)
            and native_delta.shape[2] > 0 and native_delta.shape[3] == 256
            and native_delta.dtype == torch.float32,
            'Expected native_delta float32 [4,1,HW,256]')
    require(hasattr(head, 'bev_pred_head') and len(head.bev_pred_head) == 3
            and not head.soft_weight,
            'Only the original three independent native readout branches are supported')
    branches = list(head.bev_pred_head)
    require(all(not branch._forward_pre_hooks and not branch._forward_hooks
                for branch in branches), 'A native readout already has a hook')
    counts = [0, 0, 0]
    handles = []
    expected_shape = (5, 1, native_delta.shape[2], 256)

    def make_hook(index):
        def hook(_module, inputs):
            require(len(inputs) == 1 and isinstance(inputs[0], torch.Tensor)
                    and tuple(inputs[0].shape) == expected_shape
                    and inputs[0].dtype == native_delta.dtype
                    and inputs[0].device == native_delta.device,
                    'The original native readout feature boundary changed')
            require(counts[index] == 0, 'Native branch was called more than once')
            counts[index] += 1
            current, future = inputs[0][:1], inputs[0][1:]
            corrected = _ExactZeroResidual.apply(future, native_delta)
            return (torch.cat((current, corrected), dim=0),)
        return hook

    try:
        for index, branch in enumerate(branches):
            handles.append(branch.register_forward_pre_hook(make_hook(index)))
        yield counts
        require(counts == [1, 1, 1], 'Native replay did not call all three readout branches exactly once')
    finally:
        for handle in handles:
            handle.remove()
