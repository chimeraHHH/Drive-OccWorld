"""P2 residuals on the complete M0 reference features.

``mean`` subtracts zero-object-motion transport from mean-motion transport;
both branches have identical source weighting, bilinear interpolation and ego
alignment. Source confidence is retained in a destination gate *after* splat
normalization, so isolated low-confidence returns cannot regain unit weight.

``capacity`` has exactly the same trainable tensors. It applies a pointwise,
posterior-modulated bottleneck to the same supported source features, followed
only by the common ego alignment. It never moves features by object velocity.
The constant term in its modulation makes this a functional learned residual
even for zero velocity, unlike W_zero - W_zero. It controls added parameter
capacity and the residual/gating form, not an identical transport operator.

The inherited Bayesian update is reused as a numerical utility. No M3 weights
or geometry-only radar encoding are used. Occupancy always conditions on all
valid current returns. Optional training NLL has a separate support-only state
computed from camera-only BEV; neither state is cached on the module.
"""
import math

import torch
import torch.nn.functional as F

from .doppler_posterior_transport import (
    DopplerPosteriorTransport, forward_bilinear_splat)


class CredibleMotionResidual(DopplerPosteriorTransport):
    def __init__(self, point_cloud_range, mode='mean', nll_weight=0.0,
                 max_speed_mps=30.0, max_displacement_m=20.0,
                 confidence_velocity_scale=5.0, **kwargs):
        if mode not in ('mean', 'capacity'):
            raise ValueError('mode must be mean or capacity')
        if nll_weight < 0 or (mode == 'capacity' and nll_weight != 0):
            raise ValueError('NLL is nonnegative and only enabled for mean mode')
        if min(max_speed_mps, max_displacement_m, confidence_velocity_scale) <= 0:
            raise ValueError('Speed, displacement and confidence bounds must be positive')
        forbidden = {'transport_mode', 'covariance_mode', 'use_doppler_update',
                     'loss_weight'}.intersection(kwargs)
        if forbidden:
            raise ValueError('P2 fixes these options: {}'.format(sorted(forbidden)))
        super().__init__(point_cloud_range=point_cloud_range,
                         transport_mode='mean', covariance_mode='full',
                         use_doppler_update=True, loss_weight=1.0, **kwargs)
        self.mode, self.nll_weight = mode, float(nll_weight)
        self.max_speed_mps = float(max_speed_mps)
        self.max_displacement_m = float(max_displacement_m)
        self.confidence_velocity_scale = float(confidence_velocity_scale)

    def prepare(self, camera_bev, observations, training=None,
                nll_support_mask=None):
        training = self.training if training is None else bool(training)
        # Explicit training=False is essential: setting the NLL coefficient to
        # zero alone would leave the inherited random holdout active.
        occupancy = super().prepare(camera_bev, observations, training=False)
        occupancy['path'] = 'p2_all_returns'
        occupancy['nll_state'] = None
        if training and self.nll_weight > 0:
            # The optional random partition must not advance the baseline's
            # CPU/CUDA RNG streams. Its draws start from the current step's
            # state, which can change with the common training path, and are
            # restored afterward. This remains random per-return holdout,
            # not independent sensor/time-grouped validation.
            devices = [observations.get_device()] if observations.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                nll_state = super().prepare(camera_bev, observations, training=True,
                                            support_mask=nll_support_mask)
            occupancy['nll_state'] = nll_state
            occupancy['loss_p2_doppler_nll'] = (
                self.nll_weight * nll_state['loss_doppler_nll'])
        return occupancy

    def forward(self, camera_bev, observations, training=None,
                nll_support_mask=None):
        return self.prepare(camera_bev, observations, training=training,
                            nll_support_mask=nll_support_mask)

    def _source_confidence(self, state):
        presence = state['geometry'][:, :1]
        information = state['information_eigenvalues'].sum(dim=1, keepdim=True)
        precision = state['prior_std'].square().reciprocal()
        observed_fraction = information / (information + precision).clamp_min(1e-8)
        # This discounts weakly constrained tangential motion rather than
        # treating a single radial observation as a fully observed 2-D vector.
        variance_trace = state['cov'][:, 0, 0] + state['cov'][:, 1, 1]
        velocity_confidence = 1.0 / (
            1.0 + variance_trace.clamp_min(1e-8).sqrt().unsqueeze(1) /
            self.confidence_velocity_scale)
        return presence * observed_fraction * velocity_confidence

    def _bounded_displacement(self, mean, future_index):
        velocity_norm = mean.square().sum(dim=1, keepdim=True).clamp_min(1e-12).sqrt()
        velocity = mean * (self.max_speed_mps / velocity_norm).clamp_max(1.0)
        displacement_m = velocity * (float(future_index) * self.time_step)
        distance = displacement_m.square().sum(dim=1, keepdim=True).clamp_min(1e-12).sqrt()
        displacement_m = displacement_m * (
            self.max_displacement_m / distance).clamp_max(1.0)
        return torch.cat((displacement_m[:, :1] / self.cell_size_x,
                          displacement_m[:, 1:] / self.cell_size_y), dim=1)

    def transport(self, ref_bev, state, future_index, future2ref_grid):
        features = self._features(ref_bev)
        batch, _, height, width = features.shape
        if future_index < 0:
            raise ValueError('future_index must be nonnegative')
        if future2ref_grid.shape != (batch, height * width, 2):
            raise ValueError('future2ref_grid must have shape [B,H*W,2]')
        reduced = self.transport_reduce(features.to(self.transport_reduce.weight.dtype)).float()
        confidence = self._source_confidence(state)
        displacement = self._bounded_displacement(state['mean'], future_index)
        zero_displacement = torch.zeros_like(displacement)
        zero_branch, zero_mass = forward_bilinear_splat(
            reduced, zero_displacement, source_weight=confidence)

        if self.mode == 'mean':
            active_branch, active_mass = forward_bilinear_splat(
                reduced, displacement, source_weight=confidence)
            residual = active_branch - zero_branch
            destination_confidence = torch.maximum(active_mass, zero_mass).clamp(0.0, 1.0)
        else:
            # Fixed channel basis adds no trainable capacity. Posterior mean
            # changes a local feature gain, never the location of a feature.
            channels = reduced.shape[1]
            angles = torch.arange(channels, device=reduced.device,
                                  dtype=reduced.dtype) * (2.0 * math.pi / channels)
            bounded_mean = torch.tanh(state['mean'] / self.max_speed_mps)
            modulation = 1.0 + 0.5 * (
                bounded_mean[:, :1] * angles.cos().view(1, -1, 1, 1) +
                bounded_mean[:, 1:] * angles.sin().view(1, -1, 1, 1))
            active_branch, active_mass = forward_bilinear_splat(
                reduced * modulation, zero_displacement, source_weight=confidence)
            residual = active_branch
            destination_confidence = torch.maximum(active_mass, zero_mass).clamp(0.0, 1.0)

        grid = future2ref_grid.reshape(batch, height, width, 2).float()
        aligned = F.grid_sample(residual, grid, mode='bilinear',
                                padding_mode='zeros', align_corners=False)
        target_gate = F.grid_sample(destination_confidence, grid, mode='bilinear',
                                   padding_mode='zeros', align_corners=False).clamp(0.0, 1.0)
        expanded = self.transport_expand(aligned.to(self.transport_expand.weight.dtype))
        expanded = expanded * target_gate.to(expanded.dtype)
        tokens = expanded.flatten(2).transpose(1, 2).contiguous().to(ref_bev.dtype)
        output = tokens * torch.tanh(self.prior_gate).to(tokens.dtype)
        # Scalars only, detached and kept with this explicit sample state.
        state['transport_diagnostics'] = dict(
            gate_abs_mean=torch.tanh(self.prior_gate).abs().mean().detach(),
            source_coverage=(confidence > 0).float().mean().detach(),
            source_confidence_mean=confidence.mean().detach(),
            target_confidence_mean=target_gate.mean().detach(),
            reference_rms=features.square().mean().sqrt().detach(),
            active_branch_rms=active_branch.square().mean().sqrt().detach(),
            zero_branch_rms=zero_branch.square().mean().sqrt().detach(),
            ungated_residual_rms=tokens.square().mean().sqrt().detach(),
            gated_residual_rms=output.square().mean().sqrt().detach(),
            displacement_max_pixels=displacement.square().sum(1).sqrt().max().detach())
        return output
