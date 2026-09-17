"""Physical supervision readout on the native O future-state graph.

This is an untrained research component, not an occupancy-output refiner.
The native O forward/occupancy decoder is unchanged. K and B use identical
modules: K retains the feature graph, B detaches only the physical readout's
feature input. The fixed D field is a baseline displacement, never a label.
No object boxes, GT positions, future timestamps, support masks, or transport
operator enter forward. All displacements use fixed t0 LiDAR XYZ in metres.
"""
import torch
from torch import nn


HORIZONS_SECONDS = (.5, 1., 1.5, 2.)
CHANNELS = 256
HIDDEN_CHANNELS = 128


def require(ok, message):
    if not ok:
        raise ValueError(message)


def grid_tuple(shape):
    shape = tuple(shape)
    require(len(shape) == 3 and all(type(x) is int and x > 0 for x in shape),
            'Expected positive integer physical XYZ grid shape')
    return shape


def native_state_pair_images(features, grid_shape):
    """[5,B,Y*X,256] -> [4*B,513,Y,X], horizon-major then batch.

    Native column index is y*X+x. Input groups are t0[256], future[256],
    and the fixed nominal horizon[1]. The last group is model metadata,
    not the example's future annotation time.
    """
    nx, ny, _ = grid_tuple(grid_shape)
    require(isinstance(features, torch.Tensor) and features.ndim == 4
            and features.shape[0] == 5 and features.shape[1] > 0
            and tuple(features.shape[2:]) == (ny * nx, CHANNELS),
            'Expected native current plus four future column features')
    require(features.dtype == torch.float32, 'Use the audited FP32 feature boundary')
    batch = features.shape[1]
    current = features[0:1].expand(4, -1, -1, -1)
    paired = torch.cat((current, features[1:]), dim=-1)
    images = paired.reshape(4, batch, ny, nx, 2 * CHANNELS).permute(0, 1, 4, 2, 3)
    times = features.new_tensor(HORIZONS_SECONDS).reshape(4, 1, 1, 1, 1)
    times = times.expand(4, batch, 1, ny, nx)
    return torch.cat((images, times), dim=2).reshape(4 * batch, 2 * CHANNELS + 1, ny, nx)


def readout_images_to_xyz(raw, batch, grid_shape):
    """[4*B,3*Z,Y,X] -> [B,4,3,X,Y,Z]; channel index axis*Z+z."""
    nx, ny, nz = grid_tuple(grid_shape)
    require(type(batch) is int and batch > 0 and isinstance(raw, torch.Tensor)
            and tuple(raw.shape) == (4 * batch, 3 * nz, ny, nx),
            'Displacement readout shape differs')
    return raw.reshape(4, batch, 3, nz, ny, nx).permute(1, 0, 2, 5, 4, 3).contiguous()


class _ExactZeroResidual(torch.autograd.Function):
    """Byte-preserve base at zero residual, with ordinary additive gradients."""
    @staticmethod
    def forward(ctx, base, residual):
        return torch.where(residual == 0, base, base + residual)

    @staticmethod
    def backward(ctx, gradient):
        return gradient, gradient


class FutureStateMotionReadout(nn.Module):
    """Shared spatial readout, zero-initialized correction to frozen D.

    Both arms have 745392 parameters for the native (200,200,16) grid.
    No occupancy prediction is modified by this module. A first-step zero
    gradient at its trunk/input is expected; the output receives gradients
    and can enable state supervision after a real readout update.
    """
    def __init__(self, grid_shape=(200, 200, 16)):
        super().__init__()
        self.grid_shape = grid_tuple(grid_shape)
        self.trunk = nn.Sequential(
            nn.Conv2d(2 * CHANNELS + 1, HIDDEN_CHANNELS, 3, padding=1),
            nn.GroupNorm(8, HIDDEN_CHANNELS),
            nn.ReLU(inplace=False),
            nn.Conv2d(HIDDEN_CHANNELS, HIDDEN_CHANNELS, 3, padding=1),
            nn.GroupNorm(8, HIDDEN_CHANNELS),
            nn.ReLU(inplace=False),
        )
        self.readout = nn.Conv2d(HIDDEN_CHANNELS, 3 * self.grid_shape[2], 1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, features, base_displacement, *, detach_features=False):
        require(type(detach_features) is bool, 'Choose the physical feature-graph boundary explicitly')
        require(isinstance(base_displacement, torch.Tensor) and not base_displacement.requires_grad,
                'The D displacement field must be frozen')
        inputs = features.detach() if detach_features else features
        images = native_state_pair_images(inputs, self.grid_shape)
        require(base_displacement.device == images.device and base_displacement.dtype == torch.float32,
                'Fixed D and native feature dtype/device differ')
        delta = readout_images_to_xyz(self.readout(self.trunk(images)), features.shape[1], self.grid_shape)
        require(base_displacement.shape == delta.shape, 'Fixed D XYZ field shape differs')
        return _ExactZeroResidual.apply(base_displacement, delta)


def capture_native_terminal(model, sample, native, *, training,
                            expected_shape=(5, 1, 40000, 256)):
    """Clone the native last-readout input while retaining its autograd graph.

    This changes neither forward values nor the model mode. The caller owns
    train/eval mode, parameter scope, dropout RNG and unchanged-source checks.
    The prehook is always removed, including on errors. Occupancy output is
    returned verbatim from native.replay. There is no new detach in K.
    """
    require(type(training) is bool and model.training == training,
            'Replay/model mode mismatch')
    head = model.future_pred_head
    require(len(head.bev_pred_head) == 3 and not head.soft_weight,
            'Only the original native three-layer occupancy readout is supported')
    branch = head.bev_pred_head[-1]
    require(not branch._forward_pre_hooks and not branch._forward_hooks,
            'A different capture hook is already installed')
    captured = []

    def capture(_branch, inputs):
        require(len(inputs) == 1 and tuple(inputs[0].shape) == tuple(expected_shape)
                and inputs[0].dtype == torch.float32,
                'Native terminal feature boundary changed')
        captured.append(inputs[0].clone())

    handle = branch.register_forward_pre_hook(capture)
    try:
        prediction = native.replay(model, sample, training=training)[0]
    finally:
        handle.remove()
    require(len(captured) == 1, 'Expected exactly one native terminal readout call')
    return prediction, captured[0]
