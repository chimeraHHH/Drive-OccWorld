"""Per-return Doppler information updates and probabilistic feature transport.

The Gaussian information update is standard Bayesian linear regression, not a
novel inference rule. Its role here is to expose radar's directional
observability and connect the *same* posterior to withheld-return prediction
and future occupancy features. Covariance denotes this model's conditional
uncertainty; calibration must be measured, not assumed.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def solve_isotropic_posterior(prior_mean, prior_std, information, information_vector):
    """Solve a 2-D Gaussian update on a grid, retaining the prior in nullspace.

    Args: mean [B,2,H,W], std [B,1,H,W], information [B,2,2,H,W],
        information_vector [B,2,H,W]. All calculations use float32.
    Returns: posterior mean [B,2,H,W], covariance [B,2,2,H,W].
    """
    mean = prior_mean.float()
    alpha = prior_std.float().square().reciprocal()[:, 0]
    info = information.float()
    a00, a01, a11 = info[:, 0, 0], info[:, 0, 1], info[:, 1, 1]
    p00, p11 = a00 + alpha, a11 + alpha
    # Written this way to avoid subtracting two large, nearly equal terms
    # after adding a small prior precision to a rank-one observation matrix.
    data_det = (a00 * a11 - a01.square()).clamp_min(0.0)
    determinant = data_det + alpha * (a00 + a11) + alpha.square()
    c00, c01, c11 = p11 / determinant, -a01 / determinant, p00 / determinant
    covariance = torch.stack((torch.stack((c00, c01), dim=1),
                              torch.stack((c01, c11), dim=1)), dim=1)
    residual = information_vector.float() - torch.stack(
        (a00 * mean[:, 0] + a01 * mean[:, 1],
         a01 * mean[:, 0] + a11 * mean[:, 1]), dim=1)
    correction = torch.stack(
        (c00 * residual[:, 0] + c01 * residual[:, 1],
         c01 * residual[:, 0] + c11 * residual[:, 1]), dim=1)
    return mean + correction, covariance


def positive_sigma_points(mean, covariance):
    """Five positive-weight points matching a 2-D Gaussian's first 2 moments.

    Returns points [B,5,2,H,W], weights [5]. The finite quadrature is only an
    approximation to the transported Gaussian distribution.
    """
    covariance = covariance.float()
    l00 = covariance[:, 0, 0].clamp_min(1e-10).sqrt()
    l10 = covariance[:, 1, 0] / l00
    l11 = (covariance[:, 1, 1] - l10.square()).clamp_min(1e-10).sqrt()
    first = math.sqrt(3.0) * torch.stack((l00, l10), dim=1)
    second = math.sqrt(3.0) * torch.stack((torch.zeros_like(l11), l11), dim=1)
    points = torch.stack((mean, mean + first, mean - first,
                          mean + second, mean - second), dim=1)
    weights = mean.new_tensor((1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0,
                              1.0 / 6.0, 1.0 / 6.0))
    return points, weights


def forward_bilinear_splat(features, displacement, source_weight=None,
                           normalize=True):
    """Push source features to source+displacement, with bilinear scattering.

    Features [B,C,H,W], displacement [B,2,H,W] in pixels, optional nonnegative
    source_weight [B,1,H,W]. Returns (features_or_numerator, mass), both on the
    source-sized grid. Out-of-grid contributions are dropped; colliding
    contributions are normalized only when ``normalize`` is True. This is
    feature resampling, not a claim of conserved physical occupancy mass.
    """
    if features.ndim != 4 or displacement.shape != (
            features.shape[0], 2, features.shape[2], features.shape[3]):
        raise ValueError('Expected features [B,C,H,W] and displacement [B,2,H,W]')
    batch, channels, height, width = features.shape
    values = features.float().reshape(batch, channels, -1)
    displacement = displacement.float()
    ys = torch.arange(height, device=features.device, dtype=torch.float32)
    xs = torch.arange(width, device=features.device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs)
    target_x = grid_x.reshape(1, -1) + displacement[:, 0].reshape(batch, -1)
    target_y = grid_y.reshape(1, -1) + displacement[:, 1].reshape(batch, -1)
    finite = torch.isfinite(target_x) & torch.isfinite(target_y)
    target_x = torch.where(finite, target_x, torch.zeros_like(target_x))
    target_y = torch.where(finite, target_y, torch.zeros_like(target_y))
    x0, y0 = target_x.floor(), target_y.floor()
    if source_weight is None:
        source_weight = torch.ones_like(displacement[:, :1])
    if source_weight.shape != (batch, 1, height, width):
        raise ValueError('source_weight must have shape [B,1,H,W]')
    source_weight = source_weight.float().reshape(batch, -1)
    finite = finite & torch.isfinite(source_weight) & (source_weight >= 0)
    source_weight = torch.where(finite, source_weight, torch.zeros_like(source_weight))
    numerator = values.new_zeros((batch, channels, height * width))
    mass = values.new_zeros((batch, 1, height * width))
    for dy in (0, 1):
        for dx in (0, 1):
            x, y = x0 + dx, y0 + dy
            valid = finite & (x >= 0) & (x < width) & (y >= 0) & (y < height)
            weight = ((1.0 - (target_x - x).abs()).clamp_min(0.0) *
                      (1.0 - (target_y - y).abs()).clamp_min(0.0) *
                      source_weight * valid.float())
            index = (y.clamp(0, height - 1).long() * width +
                     x.clamp(0, width - 1).long()).unsqueeze(1)
            numerator = numerator.scatter_add(
                2, index.expand(-1, channels, -1), values * weight.unsqueeze(1))
            mass = mass.scatter_add(2, index, weight.unsqueeze(1))
    numerator = numerator.reshape(batch, channels, height, width)
    mass = mass.reshape(batch, 1, height, width)
    output = numerator / mass.clamp_min(1e-8) if normalize else numerator
    return output, mass


class DopplerPosteriorTransport(nn.Module):
    """Camera motion prior updated by support returns and used during inference.

    Observations are padded [B,N,9]: x,y,z,rcs,ux,uy,vr,lag,valid. Radial
    velocities are physical m/s and are never clipped by the feature encoder.
    Only causal returns no older than max_time_lag are retained. Each cell uses
    a shared constant 2-D velocity and independent radial noise; mixed-object
    cells and correlated returns violate those approximations and need explicit
    robustness evaluation. Default short-lag, one-sweep inputs reduce temporal
    misregistration but do not solve it.
    """
    def __init__(self, point_cloud_range, bev_h=200, bev_w=200,
                 embed_dims=256, hidden_dims=64, transport_dims=32,
                 time_step=0.5, sensor_std=1.5, prior_std_bounds=(0.5, 15.0),
                 initial_prior_std=5.0, holdout_fraction=0.2,
                 max_time_lag=0.15, lag_acceleration_std=3.0,
                 loss_weight=1.0, gate_init=0.0, count_clip=32.0,
                 rcs_norm=50.0, transport_mode='sigma', use_doppler_update=True,
                 covariance_mode='full'):
        super().__init__()
        if len(point_cloud_range) != 6 or bev_h <= 0 or bev_w <= 0:
            raise ValueError('Expected a six-value range and positive BEV dimensions')
        if any(point_cloud_range[i + 3] <= point_cloud_range[i] for i in range(3)):
            raise ValueError('point_cloud_range maxima must exceed minima')
        lo, hi = (float(x) for x in prior_std_bounds)
        if not 0 < lo < initial_prior_std < hi:
            raise ValueError('Require 0 < min std < initial std < max std')
        if sensor_std <= 0 or time_step <= 0 or max_time_lag <= 0:
            raise ValueError('Noise, time step and maximum lag must be positive')
        if not 0 <= holdout_fraction < 1 or loss_weight < 0:
            raise ValueError('holdout_fraction must be in [0,1), loss_weight >= 0')
        if lag_acceleration_std < 0 or count_clip <= 0 or rcs_norm <= 0:
            raise ValueError('Require nonnegative lag noise and positive normalizers')
        if transport_mode not in ('sigma', 'mean', 'off'):
            raise ValueError('transport_mode must be sigma, mean, or off')
        if covariance_mode not in ('full', 'isotropic', 'rotated'):
            raise ValueError('covariance_mode must be full, isotropic, or rotated')
        self.point_cloud_range = tuple(float(x) for x in point_cloud_range)
        self.bev_h, self.bev_w = int(bev_h), int(bev_w)
        self.embed_dims = int(embed_dims)
        self.time_step, self.sensor_std = float(time_step), float(sensor_std)
        self.prior_std_bounds = (lo, hi)
        self.holdout_fraction, self.max_time_lag = float(holdout_fraction), float(max_time_lag)
        self.lag_acceleration_std, self.loss_weight = float(lag_acceleration_std), float(loss_weight)
        self.count_clip, self.rcs_norm = float(count_clip), float(rcs_norm)
        self.transport_mode = transport_mode
        self.use_doppler_update = bool(use_doppler_update)
        self.covariance_mode = covariance_mode
        self.prior_head = nn.Sequential(
            nn.Conv2d(embed_dims, hidden_dims, 3, padding=1), nn.ReLU(inplace=False),
            nn.Conv2d(hidden_dims, 3, 1))
        nn.init.zeros_(self.prior_head[-1].weight)
        nn.init.zeros_(self.prior_head[-1].bias)
        fraction = (initial_prior_std - lo) / (hi - lo)
        with torch.no_grad():
            self.prior_head[-1].bias[2] = math.log(fraction / (1.0 - fraction))
        self.transport_reduce = nn.Conv2d(embed_dims, transport_dims, 1, bias=False)
        self.transport_expand = nn.Conv2d(transport_dims, embed_dims, 1, bias=False)
        self.prior_gate = nn.Parameter(torch.full((1, 1, embed_dims), float(gate_init)))

    @property
    def cell_size_x(self):
        return (self.point_cloud_range[3] - self.point_cloud_range[0]) / self.bev_w

    @property
    def cell_size_y(self):
        return (self.point_cloud_range[4] - self.point_cloud_range[1]) / self.bev_h

    def _features(self, tokens):
        if tokens.ndim != 3 or tokens.shape[1:] != (self.bev_h * self.bev_w, self.embed_dims):
            raise ValueError('Expected BEV tokens [B,H*W,embed_dims]')
        return tokens.transpose(1, 2).reshape(tokens.shape[0], self.embed_dims,
                                              self.bev_h, self.bev_w)

    def _prepare_observations(self, observations):
        if observations.ndim != 3 or observations.shape[-1] != 9:
            raise ValueError('Expected padded observations [B,N,9]')
        observations = observations.float()
        finite = torch.isfinite(observations).all(dim=-1)
        obs = torch.where(finite.unsqueeze(-1), observations, torch.zeros_like(observations))
        x, y, z, lag = obs[..., 0], obs[..., 1], obs[..., 2], obs[..., 7]
        norm = obs[..., 4:6].square().sum(dim=-1).sqrt()
        xmin, ymin, zmin, xmax, ymax, zmax = self.point_cloud_range
        valid = (finite & (obs[..., 8] > 0.5) & (norm >= 0.95) & (norm <= 1.05) &
                 (x >= xmin) & (x < xmax) & (y >= ymin) & (y < ymax) &
                 (z >= zmin) & (z < zmax) & (lag >= 0) & (lag <= self.max_time_lag))
        direction = obs[..., 4:6] / norm.clamp_min(1e-8).unsqueeze(-1)
        # Sanitizing invalid indices also makes N=0 and all-padding batches safe.
        xi = ((x - xmin) / self.cell_size_x).floor().long().clamp(0, self.bev_w - 1)
        yi = ((y - ymin) / self.cell_size_y).floor().long().clamp(0, self.bev_h - 1)
        return obs, direction, valid, yi * self.bev_w + xi

    def _split(self, valid, training, support_mask=None):
        if support_mask is not None:
            if support_mask.shape != valid.shape:
                raise ValueError('support_mask must have shape [B,N]')
            support = support_mask.to(device=valid.device, dtype=torch.bool) & valid
        elif training and self.holdout_fraction > 0:
            support = valid & (torch.rand(valid.shape, device=valid.device) >= self.holdout_fraction)
            # Retain at least the first valid return in each nonempty sample.
            # With N=0 there is no argmax; do not synchronize CUDA with .item().
            if valid.shape[1] > 0:
                first = valid.long().argmax(dim=1, keepdim=True)
                rescue = torch.zeros_like(valid).scatter(1, first, True)
                support = support | (rescue & valid & ~support.any(dim=1, keepdim=True))
        else:
            support = valid
        return support, valid & ~support

    def prepare(self, camera_bev, observations, training=None, support_mask=None):
        """Return an explicit state; held-out radial targets never condition it.

        ``support_mask`` is an optional reproducible partition for tests and
        ablations. Omit it for normal training (random split) and inference
        (all valid returns). Empty held-out sets produce a differentiable zero
        loss, not NaN. The caller adds ``loss_doppler_nll`` to training losses.
        """
        features = self._features(camera_bev)
        if observations.shape[0] != features.shape[0] or observations.device != features.device:
            raise ValueError('Camera tokens and observations must share batch and device')
        prediction = self.prior_head(features.to(self.prior_head[0].weight.dtype)).float()
        prior_mean = prediction[:, :2]
        lo, hi = self.prior_std_bounds
        prior_std = lo + (hi - lo) * prediction[:, 2:3].sigmoid()
        obs, direction, valid, index = self._prepare_observations(observations)
        support, heldout = self._split(valid, self.training if training is None else training,
                                      support_mask)
        batch, cells = features.shape[0], self.bev_h * self.bev_w
        def raster(values):
            return values.new_zeros((batch, cells)).scatter_add(1, index, values)
        radial_noise_var = self.sensor_std ** 2 + (self.lag_acceleration_std * obs[..., 7]).square()
        weight = support.float() / radial_noise_var
        ux, uy, radial = direction[..., 0], direction[..., 1], obs[..., 6]
        # Mask before multiplication: excluded finite extreme targets must not
        # cause 0*inf overflow in support sufficient statistics.
        radial_support = torch.where(support, radial, torch.zeros_like(radial))
        a00, a01, a11 = (raster(weight * ux * ux), raster(weight * ux * uy),
                         raster(weight * uy * uy))
        information = torch.stack((torch.stack((a00, a01), dim=1),
                                   torch.stack((a01, a11), dim=1)), dim=1)
        information = information.reshape(batch, 2, 2, self.bev_h, self.bev_w)
        information_vector = torch.stack((raster(weight * ux * radial_support),
                                          raster(weight * uy * radial_support)), dim=1)
        information_vector = information_vector.reshape(batch, 2, self.bev_h, self.bev_w)
        update_information = information if self.use_doppler_update else information * 0.0
        update_vector = information_vector if self.use_doppler_update else information_vector * 0.0
        mean, covariance = solve_isotropic_posterior(prior_mean, prior_std, update_information,
                                                    update_vector)
        counts = raster(support.float())
        denominator = counts.clamp_min(1.0)
        def average(column):
            values = torch.where(support, obs[..., column], torch.zeros_like(obs[..., column]))
            return raster(values) / denominator
        z_norm = max(abs(self.point_cloud_range[2]), abs(self.point_cloud_range[5]), 1.0)
        zeros = torch.zeros_like(counts)
        geometry = torch.stack(((counts > 0).float(),
                                torch.log1p(counts.clamp_max(self.count_clip)) / math.log1p(self.count_clip),
                                (average(2) / z_norm).clamp(-1.0, 1.0),
                                (average(3) / self.rcs_norm).clamp(-1.0, 1.0),
                                zeros, zeros, zeros,
                                (average(7) / self.max_time_lag).clamp(0.0, 1.0)), dim=1)
        geometry = geometry.reshape(batch, 8, self.bev_h, self.bev_w)
        point_mean = mean.flatten(2).gather(2, index.unsqueeze(1).expand(-1, 2, -1)).transpose(1, 2)
        point_cov = covariance.reshape(batch, 4, cells).gather(
            2, index.unsqueeze(1).expand(-1, 4, -1)).transpose(1, 2).reshape(batch, -1, 2, 2)
        predicted_radial = (point_mean * direction).sum(dim=-1)
        predictive_variance = ((point_cov * direction.unsqueeze(-1) *
                                direction.unsqueeze(-2)).sum(dim=(-1, -2)) + radial_noise_var).clamp_min(1e-8)
        # Mask targets before squaring: non-target padding should never poison
        # the NLL, even if a padded record happens to contain extreme values.
        residual = torch.where(heldout, predicted_radial - radial, torch.zeros_like(radial))
        nll = 0.5 * (math.log(2.0 * math.pi) + predictive_variance.log() +
                     residual.square() / predictive_variance)
        loss = (torch.where(heldout, nll, torch.zeros_like(nll)).sum() /
                heldout.sum().clamp_min(1).float()) * self.loss_weight
        # A zero-return batch retains a connection to the prior for backward.
        loss = loss + prior_mean.sum() * 0.0
        difference = (a00 - a11).square() + 4.0 * a01.square()
        eig_gap = difference.clamp_min(0.0).sqrt()
        eigenvalues = torch.stack(((a00 + a11 - eig_gap).clamp_min(0.0) * 0.5,
                                   (a00 + a11 + eig_gap) * 0.5), dim=1)
        return dict(mean=mean, cov=covariance, prior_mean=prior_mean, prior_std=prior_std,
                    geometry=geometry, support_mask=support, holdout_mask=heldout,
                    loss_doppler_nll=loss,
                    information_eigenvalues=eigenvalues.reshape(batch, 2, self.bev_h, self.bev_w),
                    diagnostics=dict(valid_count=valid.sum().detach(),
                                     support_count=support.sum().detach(),
                                     heldout_count=heldout.sum().detach()))

    def forward(self, camera_bev, observations, training=None, support_mask=None):
        return self.prepare(camera_bev, observations, training=training,
                            support_mask=support_mask)

    def _transport_covariance(self, covariance):
        """Ablate orientation only in transport, retaining the NLL posterior."""
        if self.covariance_mode == 'full':
            return covariance
        c00, c01, c11 = covariance[:, 0, 0], covariance[:, 0, 1], covariance[:, 1, 1]
        if self.covariance_mode == 'isotropic':
            diagonal = (c00 + c11) * 0.5
            offdiag = torch.zeros_like(diagonal)
            return torch.stack((torch.stack((diagonal, offdiag), dim=1),
                                torch.stack((offdiag, diagonal), dim=1)), dim=1)
        # R Sigma R^T for a 90-degree rotation: preserves spectrum and trace.
        return torch.stack((torch.stack((c11, -c01), dim=1),
                            torch.stack((-c01, c00), dim=1)), dim=1)

    def transport(self, ref_bev, state, future_index, future2ref_grid):
        """Transport by a five-point posterior quadrature then change ego frame.

        The normalized ensemble push-forward handles collisions and empty
        destinations. It is a feature prior, not a calibrated occupancy
        probability. The exact zero gate preserves the baseline at init.
        """
        features = self._features(ref_bev)
        batch, _, height, width = features.shape
        if future2ref_grid.shape != (batch, height * width, 2):
            raise ValueError('future2ref_grid must have shape [B,H*W,2]')
        if future_index < 0:
            raise ValueError('future_index must be nonnegative')
        if self.transport_mode == 'off':
            # Keep the disabled transport parameters in the graph for DDP.
            graph_zero = (self.transport_reduce.weight.sum() +
                          self.transport_expand.weight.sum() + self.prior_gate.sum()) * 0.0
            return ref_bev * 0.0 + graph_zero.to(ref_bev.dtype)
        reduced = self.transport_reduce(features.to(self.transport_reduce.weight.dtype)).float()
        if self.transport_mode == 'sigma':
            points, weights = positive_sigma_points(
                state['mean'], self._transport_covariance(state['cov']))
        else:
            points, weights = state['mean'].unsqueeze(1), state['mean'].new_ones((1,))
        numerator = torch.zeros_like(reduced)
        mass = reduced.new_zeros((batch, 1, height, width))
        delta_t = float(future_index) * self.time_step
        for k in range(points.shape[1]):
            displacement = torch.stack((points[:, k, 0] * delta_t / self.cell_size_x,
                                         points[:, k, 1] * delta_t / self.cell_size_y), dim=1)
            contribution, point_mass = forward_bilinear_splat(reduced, displacement, normalize=False)
            numerator = numerator + weights[k] * contribution
            mass = mass + weights[k] * point_mass
        transported = numerator / mass.clamp_min(1e-8)
        grid = future2ref_grid.reshape(batch, height, width, 2).float()
        aligned = F.grid_sample(transported, grid, mode='bilinear',
                                padding_mode='zeros', align_corners=False)
        expanded = self.transport_expand(aligned.to(self.transport_expand.weight.dtype))
        tokens = expanded.flatten(2).transpose(1, 2).contiguous().to(ref_bev.dtype)
        return tokens * torch.tanh(self.prior_gate).to(tokens.dtype)
