import torch
import torch.nn as nn


class DopplerRadialFlowLoss(nn.Module):
    """Constrain only the radar-observable radial component of BEV flow.

    ``flow_preds`` follows the existing WorldHeadV1 convention and contains a
    backward displacement in coarse flow-grid cells.  It is converted to a
    forward metric velocity before projection onto the sensor-to-return unit
    vector.  No target is placed on the tangential component.
    """

    def __init__(self,
                 point_cloud_range,
                 bev_h,
                 bev_w,
                 radial_velocity_channel=8,
                 radial_direction_channels=(9, 10),
                 presence_channel=0,
                 height_channel=2,
                 time_lag_channel=7,
                 radar_velocity_norm=20.0,
                 radar_height_norm=5.0,
                 radar_time_lag_norm=0.5,
                 time_decay_tau=0.25,
                 flow_cell_size=(0.8, 0.8),
                 time_step=0.5,
                 backward_flow=True,
                 max_abs_radial_velocity=30.0,
                 loss_weight=0.1):
        super().__init__()
        if len(point_cloud_range) != 6:
            raise ValueError('point_cloud_range must contain 6 values')
        if len(radial_direction_channels) != 2:
            raise ValueError(
                'radial_direction_channels must contain x and y')
        if len(flow_cell_size) != 2:
            raise ValueError('flow_cell_size must contain x and y')
        if bev_h <= 0 or bev_w <= 0:
            raise ValueError('bev_h and bev_w must be positive')
        if radar_velocity_norm <= 0 or time_step <= 0:
            raise ValueError(
                'radar_velocity_norm and time_step must be positive')
        if min(flow_cell_size) <= 0:
            raise ValueError('flow_cell_size values must be positive')

        self.point_cloud_range = tuple(float(x) for x in point_cloud_range)
        self.bev_h = int(bev_h)
        self.bev_w = int(bev_w)
        self.radial_velocity_channel = int(radial_velocity_channel)
        self.radial_direction_channels = tuple(
            int(x) for x in radial_direction_channels)
        self.presence_channel = int(presence_channel)
        self.height_channel = (
            None if height_channel is None else int(height_channel))
        self.time_lag_channel = (
            None if time_lag_channel is None else int(time_lag_channel))
        self.radar_velocity_norm = float(radar_velocity_norm)
        self.radar_height_norm = float(radar_height_norm)
        self.radar_time_lag_norm = float(radar_time_lag_norm)
        self.time_decay_tau = float(time_decay_tau)
        self.flow_cell_size = tuple(float(x) for x in flow_cell_size)
        self.time_step = float(time_step)
        self.backward_flow = bool(backward_flow)
        self.max_abs_radial_velocity = float(max_abs_radial_velocity)
        self.loss_weight = float(loss_weight)

    def _validate_inputs(self, flow_preds, radar_bev):
        if flow_preds.ndim != 5 or flow_preds.shape[-1] < 2:
            raise ValueError(
                'flow_preds must have shape [I, B, H*W, D, C>=2], got '
                '{}'.format(tuple(flow_preds.shape)))
        inter_num, batch_size, num_cells = flow_preds.shape[:3]
        if inter_num <= 0 or num_cells != self.bev_h * self.bev_w:
            raise ValueError(
                'flow_preds has incompatible intermediate/cell dimensions: '
                '{}'.format(tuple(flow_preds.shape)))
        required_channels = max(
            self.presence_channel,
            self.radial_velocity_channel,
            *self.radial_direction_channels,
            -1 if self.height_channel is None else self.height_channel,
            -1 if self.time_lag_channel is None else self.time_lag_channel,
        ) + 1
        if (radar_bev.ndim != 4 or radar_bev.shape[0] != batch_size or
                radar_bev.shape[1] < required_channels):
            raise ValueError(
                'radar_bev must have shape [B, C, H, W] with at least {} '
                'channels, got {}'.format(
                    required_channels, tuple(radar_bev.shape)))

    def _sample_flow_at_radar_height(self, flow_xy, radar_bev):
        """Select the voxel-flow height bin intersected by each radar return."""
        inter_num, batch_size, _, depth, _ = flow_xy.shape
        flow_xy = flow_xy.view(
            inter_num, batch_size, self.bev_h, self.bev_w, depth, 2)
        if self.height_channel is None:
            return flow_xy.mean(dim=4)

        radar_height = (
            radar_bev[:, self.height_channel].float() *
            self.radar_height_norm)
        z_min, z_max = self.point_cloud_range[2], self.point_cloud_range[5]
        height_index = torch.floor(
            (radar_height - z_min) / (z_max - z_min) * depth)
        height_index = height_index.clamp(0, depth - 1).long()
        height_index = height_index.view(
            1, batch_size, self.bev_h, self.bev_w, 1, 1)
        height_index = height_index.expand(
            inter_num, -1, -1, -1, -1, 2)
        return torch.gather(flow_xy, 4, height_index).squeeze(4)

    def forward(self, flow_preds, radar_bev):
        self._validate_inputs(flow_preds, radar_bev)
        radar_float = radar_bev.float()
        if radar_float.shape[-2:] != (self.bev_h, self.bev_w):
            raise ValueError(
                'M2 requires radar and flow BEV grids to match; got radar {} '
                'and flow ({}, {})'.format(
                    tuple(radar_float.shape[-2:]), self.bev_h, self.bev_w))

        flow_xy = self._sample_flow_at_radar_height(
            flow_preds[..., :2].float(), radar_float)
        velocity_scale = flow_xy.new_tensor(self.flow_cell_size)
        velocity_scale = velocity_scale / self.time_step
        if self.backward_flow:
            velocity_scale = -velocity_scale
        predicted_velocity = flow_xy * velocity_scale

        radial_direction = radar_float[:, self.radial_direction_channels]
        direction_norm = torch.linalg.norm(
            radial_direction, dim=1, keepdim=True)
        unit_direction = radial_direction / direction_norm.clamp_min(1e-6)
        unit_direction = unit_direction.permute(0, 2, 3, 1)
        predicted_radial = torch.sum(
            predicted_velocity * unit_direction.unsqueeze(0), dim=-1)

        measured_radial = (
            radar_float[:, self.radial_velocity_channel] *
            self.radar_velocity_norm)
        measured_radial = measured_radial.unsqueeze(0)
        residual = torch.abs(predicted_radial - measured_radial)

        valid = radar_float[:, self.presence_channel] > 0
        valid = valid & (direction_norm[:, 0] > 1e-6)
        valid = valid & torch.isfinite(measured_radial[0])
        if self.max_abs_radial_velocity > 0:
            valid = valid & (
                measured_radial[0].abs() <= self.max_abs_radial_velocity)
        valid = valid.unsqueeze(0).expand_as(predicted_radial)

        # Avoid NaN * 0 at empty cells while still allowing a non-finite
        # prediction at a genuinely supervised return to surface as NaN.
        residual = torch.where(valid, residual, torch.zeros_like(residual))

        weights = valid.to(residual.dtype)
        if (self.time_lag_channel is not None and
                self.time_decay_tau > 0):
            time_lag = (
                radar_float[:, self.time_lag_channel].clamp_min(0) *
                self.radar_time_lag_norm)
            time_weight = torch.exp(-time_lag / self.time_decay_tau)
            weights = weights * time_weight.unsqueeze(0)

        # clamp_min makes an empty radar raster produce an exact, connected
        # zero without a device synchronization or a special-case branch.
        denominator = weights.sum().clamp_min(1.0)
        radial_mae = (residual * weights).sum() / denominator
        valid_returns = valid[0].sum().to(radial_mae.dtype)
        return {
            'loss_doppler_radial': radial_mae * self.loss_weight,
            'doppler_radial_mae': radial_mae.detach(),
            'doppler_valid_returns': valid_returns.detach(),
        }
