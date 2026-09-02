import torch
import torch.nn as nn
import torch.nn.functional as F


class DopplerBEVAdvection(nn.Module):
    """Build a measured-motion prior from a sparse radar BEV.

    Radar velocities live at source cells.  A direct backward warp would treat
    them as a target-defined flow and drop moving returns.  We therefore first
    bilinearly splat source displacements to their measured future locations,
    then use ``grid_sample`` to pull current BEV features into those locations.
    A second ``grid_sample`` applies Drive-OccWorld's future-to-reference ego
    transform so the returned tokens are expressed in the decoder frame.

    The output is multiplied by a learnable, channel-wise ``tanh`` gate that is
    initialized to exactly zero.  Enabling this module therefore preserves the
    baseline forward pass at initialization while allowing gradients to open
    the measured-motion path.
    """

    def __init__(self,
                 point_cloud_range,
                 bev_h,
                 bev_w,
                 embed_dims=256,
                 velocity_channels=(4, 5),
                 presence_channel=0,
                 radar_velocity_norm=20.0,
                 time_step=0.5,
                 min_dynamic_speed=0.5,
                 max_dynamic_speed=30.0,
                 padding_mode='zeros',
                 gate_init=0.0):
        super().__init__()
        if len(point_cloud_range) != 6:
            raise ValueError('point_cloud_range must contain 6 values')
        if len(velocity_channels) != 2:
            raise ValueError('velocity_channels must contain vx and vy')
        if bev_h <= 0 or bev_w <= 0:
            raise ValueError('bev_h and bev_w must be positive')
        if radar_velocity_norm <= 0 or time_step <= 0:
            raise ValueError('radar_velocity_norm and time_step must be positive')

        self.point_cloud_range = tuple(float(x) for x in point_cloud_range)
        self.bev_h = int(bev_h)
        self.bev_w = int(bev_w)
        self.embed_dims = int(embed_dims)
        self.velocity_channels = tuple(int(x) for x in velocity_channels)
        self.presence_channel = int(presence_channel)
        self.radar_velocity_norm = float(radar_velocity_norm)
        self.time_step = float(time_step)
        self.min_dynamic_speed = float(min_dynamic_speed)
        self.max_dynamic_speed = float(max_dynamic_speed)
        self.padding_mode = padding_mode

        self.prior_gate = nn.Parameter(
            torch.full((1, 1, self.embed_dims), float(gate_init)))

    @property
    def cell_size_x(self):
        return ((self.point_cloud_range[3] - self.point_cloud_range[0]) /
                self.bev_w)

    @property
    def cell_size_y(self):
        return ((self.point_cloud_range[4] - self.point_cloud_range[1]) /
                self.bev_h)

    def _prepare_motion(self, radar_bev, output_size):
        required_channels = max(
            self.presence_channel, *self.velocity_channels) + 1
        if radar_bev.ndim != 4 or radar_bev.shape[1] < required_channels:
            raise ValueError(
                'radar_bev must have shape [B, C, H, W] with at least {} '
                'channels, got {}'.format(
                    required_channels, tuple(radar_bev.shape)))

        radar_float = radar_bev.float()
        presence = radar_float[:, self.presence_channel:self.presence_channel + 1]
        velocity = radar_float[:, self.velocity_channels] * self.radar_velocity_norm
        if radar_float.shape[-2:] != tuple(output_size):
            # Nearest-neighbour resizing preserves sparse return/velocity pairs.
            presence = F.interpolate(presence, size=output_size, mode='nearest')
            velocity = F.interpolate(velocity, size=output_size, mode='nearest')

        speed = torch.linalg.norm(velocity, dim=1, keepdim=True)
        dynamic = ((presence > 0) &
                   (speed >= self.min_dynamic_speed) &
                   torch.isfinite(speed)).to(velocity.dtype)
        finite_velocity = torch.isfinite(velocity).all(dim=1, keepdim=True)
        dynamic = dynamic * finite_velocity.to(dynamic.dtype)

        if self.max_dynamic_speed > 0:
            scale = torch.clamp(
                self.max_dynamic_speed / speed.clamp_min(1e-6), max=1.0)
            velocity = velocity * scale
        velocity = torch.nan_to_num(velocity) * dynamic
        return velocity, dynamic

    @staticmethod
    def _scatter_add(flat_output, flat_index, flat_value):
        flat_output.scatter_add_(1, flat_index, flat_value)

    def _splat_target_flow(self, displacement, source_mask):
        """Splat source-defined pixel displacement onto destination cells."""
        batch_size, _, height, width = displacement.shape
        device = displacement.device
        dtype = displacement.dtype

        ys = torch.arange(height, device=device, dtype=dtype)
        xs = torch.arange(width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs)
        grid_x = grid_x.reshape(1, -1).expand(batch_size, -1)
        grid_y = grid_y.reshape(1, -1).expand(batch_size, -1)

        disp_x = displacement[:, 0].reshape(batch_size, -1)
        disp_y = displacement[:, 1].reshape(batch_size, -1)
        mask = source_mask[:, 0].reshape(batch_size, -1)
        target_x = grid_x + disp_x
        target_y = grid_y + disp_y
        x0 = torch.floor(target_x)
        y0 = torch.floor(target_y)

        normalizer = displacement.new_zeros((batch_size, height * width))
        flow_x_sum = displacement.new_zeros((batch_size, height * width))
        flow_y_sum = displacement.new_zeros((batch_size, height * width))

        for offset_y in (0, 1):
            for offset_x in (0, 1):
                dst_x = x0 + offset_x
                dst_y = y0 + offset_y
                weight_x = 1.0 - torch.abs(target_x - dst_x)
                weight_y = 1.0 - torch.abs(target_y - dst_y)
                weight = weight_x.clamp_min(0) * weight_y.clamp_min(0) * mask
                valid = ((dst_x >= 0) & (dst_x < width) &
                         (dst_y >= 0) & (dst_y < height))
                weight = weight * valid.to(weight.dtype)
                dst_x = dst_x.clamp(0, width - 1).long()
                dst_y = dst_y.clamp(0, height - 1).long()
                flat_index = dst_y * width + dst_x
                self._scatter_add(normalizer, flat_index, weight)
                self._scatter_add(flow_x_sum, flat_index, weight * disp_x)
                self._scatter_add(flow_y_sum, flat_index, weight * disp_y)

        denominator = normalizer.clamp_min(1e-6)
        target_flow = torch.stack(
            (flow_x_sum / denominator, flow_y_sum / denominator), dim=1)
        target_flow = target_flow.view(batch_size, 2, height, width)
        confidence = normalizer.view(batch_size, 1, height, width).clamp(0, 1)
        return target_flow, confidence

    @staticmethod
    def _identity_grid(batch_size, height, width, device, dtype):
        ys = torch.arange(height, device=device, dtype=dtype)
        xs = torch.arange(width, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs)
        grid_x = (2.0 * (grid_x + 0.5) / width) - 1.0
        grid_y = (2.0 * (grid_y + 0.5) / height) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1)
        return grid.unsqueeze(0).expand(batch_size, -1, -1, -1)

    def _advect_reference_features(self, ref_features, velocity, dynamic,
                                   future_step):
        batch_size, _, height, width = ref_features.shape
        delta_t = float(future_step) * self.time_step
        displacement = torch.stack(
            (velocity[:, 0] * delta_t / self.cell_size_x,
             velocity[:, 1] * delta_t / self.cell_size_y), dim=1)
        target_flow, target_confidence = self._splat_target_flow(
            displacement, dynamic)

        base_grid = self._identity_grid(
            batch_size, height, width, target_flow.device, target_flow.dtype)
        normalized_flow = torch.stack(
            (2.0 * target_flow[:, 0] / width,
             2.0 * target_flow[:, 1] / height), dim=-1)
        source_grid = (base_grid - normalized_flow).to(ref_features.dtype)
        advected = F.grid_sample(
            ref_features,
            source_grid,
            mode='bilinear',
            padding_mode=self.padding_mode,
            align_corners=False)
        return advected * target_confidence.to(ref_features.dtype)

    def forward(self, ref_bev_tokens, radar_bev, future_step,
                future_to_ref_grid):
        if ref_bev_tokens.ndim != 3:
            raise ValueError(
                'ref_bev_tokens must have shape [B, H*W, C], got {}'.format(
                    tuple(ref_bev_tokens.shape)))
        batch_size, num_cells, channels = ref_bev_tokens.shape
        if num_cells != self.bev_h * self.bev_w or channels != self.embed_dims:
            raise ValueError(
                'Expected ref_bev_tokens [B, {}, {}], got {}'.format(
                    self.bev_h * self.bev_w, self.embed_dims,
                    tuple(ref_bev_tokens.shape)))
        if future_to_ref_grid.shape != (batch_size, num_cells, 2):
            raise ValueError(
                'future_to_ref_grid must have shape [B, H*W, 2], got {}'.format(
                    tuple(future_to_ref_grid.shape)))

        velocity, dynamic = self._prepare_motion(
            radar_bev, (self.bev_h, self.bev_w))
        ref_features = ref_bev_tokens.transpose(1, 2).reshape(
            batch_size, channels, self.bev_h, self.bev_w)
        advected_ref = self._advect_reference_features(
            ref_features, velocity, dynamic, future_step)

        decoder_grid = future_to_ref_grid.reshape(
            batch_size, self.bev_h, self.bev_w, 2).to(ref_features.dtype)
        decoder_prior = F.grid_sample(
            advected_ref,
            decoder_grid,
            mode='bilinear',
            padding_mode=self.padding_mode,
            align_corners=False)
        decoder_prior = decoder_prior.flatten(2).transpose(1, 2).contiguous()
        gate = torch.tanh(self.prior_gate).to(decoder_prior.dtype)
        return decoder_prior * gate
