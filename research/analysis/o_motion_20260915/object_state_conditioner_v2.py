"""Current states plus CV-hypothesized centres condition native future queries.

Inputs are current predicted states already expressed in t0 LiDAR R, with
geometric centres (the CRN origin adaptation belongs to the caller). No labels,
future poses, future timestamps or instance identities enter this module.
The caller keeps the original 27 current-state features. Each forward appends
a three-coordinate constant-velocity hypothesis p_R + h * v_R using only a
fixed nominal horizon. G clears velocity before deriving that hypothesis.
This is a token feature, never a flow-output shortcut or a spatial mask.
Construction uses ordinary PyTorch parameter initialization; forward and the
hook context neither seed/sample RNG nor change model modes.
"""
from contextlib import contextmanager
import inspect

import torch
from torch import nn
from torch.nn import functional as F


GMO_CLASSES = ('car', 'bus', 'truck', 'trailer', 'construction_vehicle',
               'motorcycle', 'bicycle', 'pedestrian')
HORIZONS_SECONDS = (.5, 1., 1.5, 2.)
STATE_DIM = 27
ENCODED_STATE_DIM = STATE_DIM + 3
GEOMETRY_DIM = 24
LATENT_DIM = 64
NUM_LATENTS = 32
NUM_HEADS = 4
NATIVE_DIM = 256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pack_object_states(center_R, size_wlh, rotation_R, class_index, score,
                       velocity_R, valid=None):
    """Pack [B,N,*] current predicted states into [B,N,27], preserving order.

    Feature order: centre/51.2 [3], wlh/51.2 [3], full rotation row-major [9],
    GMO one-hot [8], raw confidence [1], velocity_R/20 [3]. No clipping,
    confidence threshold, sorting or top-k is applied. N=0 is supported.
    Caller filters the fixed eight GMO classes and pads only with valid=False.
    """
    require(isinstance(center_R, torch.Tensor) and center_R.ndim == 3 and
            center_R.shape[-1] == 3 and center_R.shape[0] > 0,
            'centre must be [B,N,3]')
    batch, count = center_R.shape[:2]
    shape = (batch, count)
    require(center_R.dtype == torch.float32, 'State boundary must be float32')
    for name, value, tail in (
            ('size_wlh', size_wlh, (3,)), ('rotation_R', rotation_R, (3, 3)),
            ('score', score, ()), ('velocity_R', velocity_R, (3,))):
        require(isinstance(value, torch.Tensor) and tuple(value.shape) == shape + tail
                and value.dtype == center_R.dtype and value.device == center_R.device,
                name + ' shape/dtype/device differs')
    require(isinstance(class_index, torch.Tensor) and tuple(class_index.shape) == shape
            and class_index.dtype == torch.int64 and class_index.device == center_R.device,
            'class_index must be int64 [B,N] in fixed GMO order')
    if valid is None:
        valid = torch.ones(shape, dtype=torch.bool, device=center_R.device)
    require(isinstance(valid, torch.Tensor) and tuple(valid.shape) == shape
            and valid.dtype == torch.bool and valid.device == center_R.device,
            'valid must be bool [B,N]')
    for value in (center_R, size_wlh, rotation_R, score, velocity_R):
        require(bool(torch.isfinite(value[valid]).all()), 'Nonfinite valid predicted state')
    require(bool((size_wlh[valid] > 0).all()), 'Valid wlh must be positive')
    require(bool(((class_index[valid] >= 0) & (class_index[valid] < 8)).all()),
            'Unknown valid GMO class index')
    if bool(valid.any()):
        rotation = rotation_R[valid]
        eye = torch.eye(3, dtype=rotation.dtype, device=rotation.device)
        require(torch.allclose(rotation.transpose(-1, -2) @ rotation, eye.expand_as(rotation),
                               rtol=1e-4, atol=1e-4) and
                torch.allclose(torch.linalg.det(rotation), torch.ones_like(rotation[:, 0, 0]),
                               rtol=1e-4, atol=1e-4), 'rotation_R is not a proper rotation')
    classes = F.one_hot(class_index.clamp(0, 7), num_classes=8).to(center_R.dtype)
    packed = torch.cat((center_R / 51.2, size_wlh / 51.2,
                        rotation_R.reshape(batch, count, 9), classes,
                        score.unsqueeze(-1), velocity_R / 20.), dim=-1)
    # Padding never leaks nonfinite filler values into attention K/V.
    packed = torch.where(valid.unsqueeze(-1), packed, torch.zeros_like(packed))
    return packed, valid


class ObjectStateConditioner(nn.Module):
    """Identical V/G parameter layout; only the three velocity inputs differ.

    All supported objects enter cross-attention. Query count remains native
    HW; only the number of scene latents is fixed. Empty examples return zero
    priors rather than evaluating an all-masked attention operation.
    """
    def __init__(self):
        super().__init__()
        self.state_mlp = nn.Sequential(nn.Linear(ENCODED_STATE_DIM, LATENT_DIM), nn.SiLU(),
                                       nn.Linear(LATENT_DIM, LATENT_DIM))
        self.latents = nn.Parameter(torch.empty(NUM_LATENTS, LATENT_DIM))
        nn.init.normal_(self.latents, std=.02)
        self.object_attention = nn.MultiheadAttention(LATENT_DIM, NUM_HEADS, dropout=0., batch_first=True)
        self.object_norm = nn.LayerNorm(LATENT_DIM)
        self.self_attention = nn.MultiheadAttention(LATENT_DIM, NUM_HEADS, dropout=0., batch_first=True)
        self.self_norm = nn.LayerNorm(LATENT_DIM)
        self.time_mlp = nn.Sequential(nn.Linear(1, LATENT_DIM), nn.SiLU(), nn.Linear(LATENT_DIM, LATENT_DIM))
        self.query_projection = nn.Linear(NATIVE_DIM, LATENT_DIM)
        self.query_norm = nn.LayerNorm(LATENT_DIM)
        self.query_attention = nn.MultiheadAttention(LATENT_DIM, NUM_HEADS, dropout=0., batch_first=True)
        self.output_projection = nn.Linear(LATENT_DIM, NATIVE_DIM)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, states, valid, bev_embedding, prev_features,
                target_frame_index, *, use_velocity):
        require(type(use_velocity) is bool, 'Choose V/G via explicit boolean use_velocity')
        require(type(target_frame_index) is int and 1 <= target_frame_index <= 4,
                'Condition only native future frames 1..4')
        require(isinstance(prev_features, torch.Tensor) and prev_features.ndim == 4
                and prev_features.shape[0] > 0 and prev_features.shape[1] > 0
                and prev_features.shape[-1] == NATIVE_DIM, 'prev_features must be [B,T,HW,256]')
        batch, _, queries, _ = prev_features.shape
        require(isinstance(bev_embedding, torch.Tensor) and tuple(bev_embedding.shape) == (queries, NATIVE_DIM),
                'Expected original bev_embedding [HW,256]')
        require(isinstance(states, torch.Tensor) and states.ndim == 3 and
                states.shape[0] == batch and states.shape[-1] == STATE_DIM,
                'states must be [B,N,27]')
        require(isinstance(valid, torch.Tensor) and tuple(valid.shape) == tuple(states.shape[:2])
                and valid.dtype == torch.bool, 'valid must be bool [B,N]')
        for value in (states, bev_embedding, prev_features):
            require(value.dtype == torch.float32 and value.device == prev_features.device,
                    'Use one native FP32 device for state/query/features')
        require(valid.device == states.device, 'Validity mask device differs')
        require(self.latents.dtype == torch.float32 and self.latents.device == states.device,
                'Conditioner parameters must use the native FP32 device')
        require(bool(torch.isfinite(states[valid]).all()), 'Nonfinite valid state tokens')
        nonempty = valid.any(dim=1)
        result = prev_features.new_zeros((batch, queries, NATIVE_DIM))
        if not bool(nonempty.any()):
            return result
        indices = nonempty.nonzero(as_tuple=False).squeeze(-1)
        present = valid.index_select(0, indices)
        source = states.index_select(0, indices)
        source = torch.where(present.unsqueeze(-1), source, torch.zeros_like(source))
        if not use_velocity:
            source = torch.cat((source[..., :GEOMETRY_DIM],
                                torch.zeros_like(source[..., GEOMETRY_DIM:])), dim=-1)
        horizon = HORIZONS_SECONDS[target_frame_index - 1]
        # Current centre and all other 27 source slots remain available.
        # Undo only velocity normalization when constructing p_R + h*v_R,
        # then express the hypothesis in the same /51.2 centre units.
        # Derive from the ORIGINAL current centre on every call, not by
        # cumulatively advecting the previous horizon. Do not clip/filter.
        future_center = source[..., :3] + (horizon * 20. / 51.2) * source[..., GEOMETRY_DIM:]
        token = self.state_mlp(torch.cat((source, future_center), dim=-1))
        latent = self.latents.unsqueeze(0).expand(len(indices), -1, -1)
        observed, _ = self.object_attention(latent, token, token,
                                             key_padding_mask=~present, need_weights=False)
        latent = self.object_norm(latent + observed)
        propagated, _ = self.self_attention(latent, latent, latent, need_weights=False)
        latent = self.self_norm(latent + propagated)
        # A shared additive horizon bias is retained; it is not itself
        # object-specific dynamics. The explicit h*v interaction is above.
        time = states.new_full((len(indices), 1, 1), horizon / 2.)
        latent = latent + self.time_mlp(time)
        # The native column index remains y*X+x. No label-based spatial query.
        query = bev_embedding.unsqueeze(0) + prev_features.index_select(0, indices)[:, -1]
        query = self.query_norm(self.query_projection(query))
        update, _ = self.query_attention(query, latent, latent, need_weights=False)
        delta = self.output_projection(update)
        return result.index_copy(0, indices, delta)


@contextmanager
def condition_future_head(head, conditioner, states, valid, *, use_velocity):
    """Inject exactly frames 1..4 through the existing rollout_prior argument.

    The caller runs the entire original four-step native replay inside this
    context. On success the complete order is required; on any exception the
    hook is removed without masking the original error. No tensors/graphs are
    retained by the audit dictionary. Modes, RNG and original inputs are not
    changed. A pre-existing non-None prior is always rejected.
    """
    require(type(use_velocity) is bool, 'use_velocity must be a boolean')
    require(isinstance(head, nn.Module) and isinstance(conditioner, ObjectStateConditioner),
            'Expected native module and ObjectStateConditioner')
    signature = inspect.signature(head.forward)
    names = list(signature.parameters)
    for name in ('prev_feats', 'target_frame_index', 'rollout_prior', 'bev_h', 'bev_w'):
        require(name in names, 'Native forward argument missing: ' + name)
    positional = [p.name for p in signature.parameters.values()
                  if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    prior_position = positional.index('rollout_prior') if 'rollout_prior' in positional else None
    require(prior_position is None or positional[-1] == 'rollout_prior',
            'Native prior must remain the final positional argument')
    audit = dict(schema='object-state-conditioner-hook-v2', use_velocity=use_velocity,
                 frames=[], entries=[], hook_removed=False, completed=False)

    def inject(module, args, kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        require(bound.arguments['rollout_prior'] is None, 'Refuse to replace an existing rollout_prior')
        frame = bound.arguments['target_frame_index']
        require(type(frame) is int and frame == len(audit['frames']) + 1 and frame <= 4,
                'Expected exactly native future frames 1,2,3,4 in order; t0 is not conditioned')
        previous = bound.arguments['prev_feats']
        height, width = bound.arguments['bev_h'], bound.arguments['bev_w']
        require(type(height) is int and type(width) is int and height > 0 and width > 0
                and previous.shape[2] == height * width, 'Native query grid shape differs')
        require(isinstance(module.bev_embedding, nn.Embedding), 'Expected native bev_embedding module')
        prior = conditioner(states, valid, module.bev_embedding.weight, previous, frame,
                            use_velocity=use_velocity)
        changed_args = tuple(args)
        if prior_position is not None and len(args) > prior_position:
            require(len(args) == prior_position + 1, 'Unexpected argument after prior')
            changed_args = tuple(args[:prior_position])
        changed_kwargs = dict(kwargs)
        changed_kwargs['rollout_prior'] = prior
        audit['frames'].append(frame)
        audit['entries'].append(dict(frame=frame, horizon_seconds=HORIZONS_SECONDS[frame - 1],
                                     prior_shape=list(prior.shape), prior_dtype=str(prior.dtype),
                                     valid_objects_per_batch=valid.sum(1).detach().cpu().tolist()))
        return changed_args, changed_kwargs

    handle = head.register_forward_pre_hook(inject, with_kwargs=True)
    try:
        yield audit
        require(audit['frames'] == [1, 2, 3, 4], 'Native replay ended before four conditioned future frames')
        audit['completed'] = True
    finally:
        handle.remove()
        audit['hook_removed'] = True
