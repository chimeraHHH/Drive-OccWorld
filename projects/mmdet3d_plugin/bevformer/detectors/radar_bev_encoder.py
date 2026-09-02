import torch
import torch.nn as nn
import torch.nn.functional as F


class RadarBEVEncoder(nn.Module):
    """Encode a fixed radar raster into Drive-OccWorld BEV tokens."""

    def __init__(self,
                 in_channels=8,
                 hidden_channels=64,
                 out_channels=256,
                 num_groups=8,
                 zero_init_residual=True):
        super().__init__()
        if hidden_channels % num_groups != 0:
            raise ValueError('hidden_channels must be divisible by num_groups')
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.backbone = nn.Sequential(
            nn.Conv2d(self.in_channels, hidden_channels, 3, padding=1,
                      bias=False),
            nn.GroupNorm(num_groups, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1,
                      bias=False),
            nn.GroupNorm(num_groups, hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.projection = nn.Conv2d(
            hidden_channels, self.out_channels, 1, bias=True)
        if zero_init_residual:
            nn.init.zeros_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)

    def forward(self, radar_bev, output_size=None):
        if radar_bev.ndim != 4:
            raise ValueError(
                'radar_bev must have shape [B, C, H, W], got '
                f'{tuple(radar_bev.shape)}')
        if radar_bev.shape[1] != self.in_channels:
            raise ValueError(
                f'Expected {self.in_channels} radar channels, got '
                f'{radar_bev.shape[1]}')
        radar_feat = self.projection(self.backbone(radar_bev))
        if output_size is not None and radar_feat.shape[-2:] != tuple(output_size):
            radar_feat = F.interpolate(
                radar_feat, size=output_size, mode='bilinear',
                align_corners=False)
        return radar_feat
