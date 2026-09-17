"""Untrained source-displacement readout for frozen O t0 BEV features.

Only t0_tokens enters forward. No future labels, past GT, velocity prior,
transport, probability/motion mask, ego transform, or bounded activation is
implemented here. Four horizons are separate outputs in metres, not scaled
copies of a constant velocity. This module alone establishes no motion
accuracy, occupancy improvement, or methodological novelty.

Native token index = y * X + x. Conv2d uses [B,C,Y,X]. Output uses the
explicit physical-axis layout [B,horizon,dx/dy/dz,X,Y,Z]. Consumers must
freeze O separately and establish the physical frame of the source grid.
"""

import torch
from torch import nn


HORIZONS_SECONDS = (0.5, 1.0, 1.5, 2.0)
INPUT_CHANNELS = 256
HIDDEN_CHANNELS = 128
GROUPS = 8


def _check_shape(grid_shape):
    shape = tuple(grid_shape)
    if len(shape) != 3 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in shape):
        raise ValueError('grid_shape must be positive integer (X,Y,Z)')
    return shape


def native_tokens_to_conv_bev(t0_tokens, grid_shape):
    """[B,Y*X,256], token y*X+x -> [B,256,Y,X]; no input mutation."""
    nx, ny, _ = _check_shape(grid_shape)
    if not isinstance(t0_tokens, torch.Tensor):
        raise TypeError('t0_tokens must be a torch tensor')
    if t0_tokens.ndim != 3 or t0_tokens.shape[0] <= 0:
        raise ValueError('t0_tokens layout must be [B,Y*X,256], B>0')
    if tuple(t0_tokens.shape[1:]) != (ny * nx, INPUT_CHANNELS):
        raise ValueError('native token count/channels disagree with grid_shape/256')
    if not t0_tokens.is_floating_point():
        raise TypeError('t0_tokens must be floating point')
    return t0_tokens.reshape(t0_tokens.shape[0], ny, nx, INPUT_CHANNELS).permute(0, 3, 1, 2).contiguous()


def readout_to_source_xyz(raw_readout, height_bins):
    """[B,4*3*Z,Y,X] -> [B,4,3,X,Y,Z].

    Channel index = (horizon * 3 + axis) * Z + height_index.
    Axis channels 0,1,2 mean metric dx,dy,dz respectively.
    """
    if isinstance(height_bins, bool) or not isinstance(height_bins, int) or height_bins <= 0:
        raise ValueError('height_bins must be a positive integer')
    if not isinstance(raw_readout, torch.Tensor) or raw_readout.ndim != 4:
        raise ValueError('raw_readout layout must be [B,4*3*Z,Y,X]')
    batch, channels, ny, nx = raw_readout.shape
    if batch <= 0 or nx <= 0 or ny <= 0 or channels != 4 * 3 * height_bins:
        raise ValueError('raw readout dimensions do not match 4 horizons, 3 axes and Z')
    return raw_readout.reshape(batch, 4, 3, height_bins, ny, nx).permute(0, 1, 2, 5, 4, 3).contiguous()


class MotionPredictionHead(nn.Module):
    """Small Conv2d trunk with a precisely zero-initialized displacement head.

    Args:
        grid_shape: physical-axis (X,Y,Z), default (200,200,16).

    forward(t0_tokens) returns [B,4,3,X,Y,Z] in metres at
    (0.5,1.0,1.5,2.0) seconds. All final weights AND biases start at zero.
    Consequently the first-step trunk/input gradients can be zero; the
    last layer receives gradients and can enable trunk gradients thereafter.
    No RNG seed or global training/device policy is changed on construction.
    """

    def __init__(self, grid_shape=(200, 200, 16)):
        super().__init__()
        self.grid_shape = _check_shape(grid_shape)
        self.horizons_seconds = HORIZONS_SECONDS
        self.trunk = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, HIDDEN_CHANNELS, kernel_size=3, padding=1),
            nn.GroupNorm(GROUPS, HIDDEN_CHANNELS),
            nn.ReLU(inplace=False),
            nn.Conv2d(HIDDEN_CHANNELS, HIDDEN_CHANNELS, kernel_size=3, padding=1),
            nn.GroupNorm(GROUPS, HIDDEN_CHANNELS),
            nn.ReLU(inplace=False),
        )
        self.readout = nn.Conv2d(HIDDEN_CHANNELS, 4 * 3 * self.grid_shape[2], kernel_size=1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, t0_tokens):
        bev_yx = native_tokens_to_conv_bev(t0_tokens, self.grid_shape)
        raw_yx = self.readout(self.trunk(bev_yx))
        return readout_to_source_xyz(raw_yx, self.grid_shape[2])

    def extra_repr(self):
        return 'grid_shape={}, horizons_seconds={}, output_units=metres'.format(self.grid_shape, self.horizons_seconds)
