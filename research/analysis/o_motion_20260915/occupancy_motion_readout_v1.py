"""Supervised diagnostic readout from five binary-occupancy count volumes.

Input counts come from fixed 4x4x4 blocks of native binary GMO: uint8,
[B,5,X,Y,Z], values 0..64. They are occupancy fractions after division by
64, NOT model probabilities. No labels, object queries, metadata, sensors,
latent features or existing motion fields enter this module.

The default (128,128,10) grid is a shared output-only representation. This
readout is neither O's native flow nor a claim of identifiable scene flow.
It predicts four future-keyframe XYZ displacements in metres, not velocity.
"""

import torch
from torch import nn
from torch.nn import functional as F


INPUT_FRAMES = 5
OUTPUT_HORIZONS = 4
BLOCK_VOLUME = 64
DEFAULT_GRID_SHAPE = (128, 128, 10)  # physical X,Y,Z
NATIVE_GRID_SHAPE = (200, 200, 16)   # physical X,Y,Z; same metric extent
WIDTHS = (32, 64, 96, 128)
GROUPS = 8
DEFAULT_PARAMETER_COUNT = 923576


def _grid_shape(shape):
    shape = tuple(shape)
    if (len(shape) != 3 or any(isinstance(v, bool) or not isinstance(v, int)
                               or v <= 0 for v in shape)):
        raise ValueError('grid shape must be three positive integers (X,Y,Z)')
    return shape


def counts_to_conv_yx(counts, grid_shape=DEFAULT_GRID_SHAPE):
    """uint8[B,5,X,Y,Z] -> float32[B,5*Z,Y,X], channel=t*Z+z.

    Counts are divided by 64 exactly once. The input is never modified.
    Non-square test grids use the same physical-axis contract as the default.
    """
    nx, ny, nz = _grid_shape(grid_shape)
    if not isinstance(counts, torch.Tensor) or counts.dtype != torch.uint8:
        raise TypeError('counts must be a torch.uint8 tensor, not logits/probabilities')
    if (counts.ndim != 5 or counts.shape[0] <= 0
            or tuple(counts.shape[1:]) != (INPUT_FRAMES, nx, ny, nz)):
        raise ValueError('counts layout must be [B,5,X,Y,Z], B>0, with the configured grid')
    if bool((counts > BLOCK_VOLUME).any()):
        raise ValueError('A 4x4x4 binary block cannot contain more than 64 foreground voxels')
    values = counts.to(dtype=torch.float32) / BLOCK_VOLUME
    return values.permute(0, 1, 4, 3, 2).reshape(
        counts.shape[0], INPUT_FRAMES * nz, ny, nx).contiguous()


def readout_to_xyz(raw_yx, height_bins):
    """[B,4*3*Z,Y,X] -> [B,4,3,X,Y,Z].

    Output channel=(h*3+axis)*Z+z; axis 0/1/2 denotes metric dx/dy/dz.
    The four displacements are independent outputs, not scaled velocities.
    """
    nz = _grid_shape((1, 1, height_bins))[2]
    if not isinstance(raw_yx, torch.Tensor) or raw_yx.ndim != 4:
        raise ValueError('raw readout must have layout [B,4*3*Z,Y,X]')
    b, c, ny, nx = raw_yx.shape
    if min(b, ny, nx) <= 0 or c != OUTPUT_HORIZONS * 3 * nz:
        raise ValueError('raw readout does not match four horizons, three axes and Z')
    return raw_yx.reshape(b, OUTPUT_HORIZONS, 3, nz, ny, nx).permute(
        0, 1, 2, 5, 4, 3).contiguous()


def to_native_field(field_xyz, target_shape=NATIVE_GRID_SHAPE):
    """Resample metre displacements on the same extent; do NOT scale vectors.

    [B,4,3,X,Y,Z] -> [B,4,3,Xt,Yt,Zt]. PyTorch interpolation receives
    [B,12,Z,Y,X] and size=(Zt,Yt,Xt), trilinear, align_corners=False.
    Thus voxel-centre coordinates use (j+.5)*n_in/n_out-.5, with edge-value
    extension outside the input centre range, as in F.interpolate.
    This is interpolation, not spatial transport or a coordinate transform.
    """
    nx, ny, nz = _grid_shape(target_shape)
    if (not isinstance(field_xyz, torch.Tensor) or field_xyz.ndim != 6
            or tuple(field_xyz.shape[1:3]) != (OUTPUT_HORIZONS, 3)
            or any(n <= 0 for n in field_xyz.shape)):
        raise ValueError('field layout must be [B,4,3,X,Y,Z] with nonempty dimensions')
    if not field_xyz.is_floating_point():
        raise TypeError('displacements must be floating-point metres')
    b, _, _, sx, sy, sz = field_xyz.shape
    if (sx, sy, sz) == (nx, ny, nz):
        return field_xyz
    volume_zyx = field_xyz.permute(0, 1, 2, 5, 4, 3).reshape(
        b, OUTPUT_HORIZONS * 3, sz, sy, sx)
    result = F.interpolate(volume_zyx, size=(nz, ny, nx), mode='trilinear',
                           align_corners=False)
    return result.reshape(b, OUTPUT_HORIZONS, 3, nz, ny, nx).permute(
        0, 1, 2, 5, 4, 3).contiguous()


def _double_conv(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(GROUPS, out_channels),
        nn.SiLU(inplace=False),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(GROUPS, out_channels),
        nn.SiLU(inplace=False),
    )


class OccupancyMotionReadout(nn.Module):
    """Small spatial U-Net; default 923,576 trainable parameters.

    Encoder widths 32/64/96/128, three stride-2 first convolutions. Every
    encoder/decoder block has two 3x3 Conv-GN8-SiLU operations. Decoder
    features are bilinearly resized to each skip's exact Y/X shape with
    align_corners=False, concatenated, then passed through a double block.
    The final 1x1 weights and biases are zero. At initialization the output
    and gradients into the trunk are zero; the final layer can receive
    gradients and subsequently enable the trunk. Construction changes no
    seed, device, training policy or external model.

    grid_shape is configurable only to permit small/non-square contract
    tests; the formal representation is DEFAULT_GRID_SHAPE.
    """
    def __init__(self, grid_shape=DEFAULT_GRID_SHAPE):
        super().__init__()
        self.grid_shape = _grid_shape(grid_shape)
        if min(self.grid_shape[:2]) < 8:
            raise ValueError('X and Y must be at least 8 for three downsampling stages')
        z = self.grid_shape[2]
        self.encoder = nn.ModuleList([
            _double_conv(INPUT_FRAMES * z, WIDTHS[0]),
            _double_conv(WIDTHS[0], WIDTHS[1], stride=2),
            _double_conv(WIDTHS[1], WIDTHS[2], stride=2),
            _double_conv(WIDTHS[2], WIDTHS[3], stride=2),
        ])
        self.decoder = nn.ModuleList([
            _double_conv(WIDTHS[3] + WIDTHS[2], WIDTHS[2]),
            _double_conv(WIDTHS[2] + WIDTHS[1], WIDTHS[1]),
            _double_conv(WIDTHS[1] + WIDTHS[0], WIDTHS[0]),
        ])
        self.readout = nn.Conv2d(WIDTHS[0], OUTPUT_HORIZONS * 3 * z, 1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, counts):
        x = counts_to_conv_yx(counts, self.grid_shape)
        skips = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)
        for block, skip in zip(self.decoder, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        return readout_to_xyz(self.readout(x), self.grid_shape[2])

    def extra_repr(self):
        return 'grid_shape={}, units=metres, input=binary_block_counts/64'.format(self.grid_shape)
