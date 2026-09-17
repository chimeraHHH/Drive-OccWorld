"""Single-slot reference-coordinate state experiment, without output warping.

Install into a freshly, strictly loaded native M0 BEFORE optimizer creation:
    receipt = install_frame_consistent_adapter(model, mode='reference')
    output = native_state_cache.replay(model, sample, training=True)
The matched control is mode='native1'. The caller owns train/eval mode, precision,
parameter freezing and the optimizer; installation changes none of these.

The frozen original future_pred AST is retained except for a shared input guard
and, for A/reference, two statement insertions: allocate the single R->R spatial
frame, then replace cross-attention ref_points after the original alignment.
The original ref_to_history_list remains a PHYSICAL-TIME POSE queue. It no longer
describes A's latent spatial basis. Its ref2future appends and future2history
return feed ego normalization unchanged at every step. The spatial queue is a
separate constant identity: all current/generated slots live in R. The slot's
time advances from 0 to 3, target_frame_index advances from 1 to 4. The native
single-slot prev_frame_embedding is preserved; it is not an absolute-time code.

The learned query/position tables, latent values, offsets, task decoder, all
intermediate loss paths, GT slicing and original valid_frames/no_grad branches
remain native. No detach is added. Original future_to_ref_grid is unused because
planning/motion/Doppler branches are forbidden. It is not an A spatial route.
The helper reproduces the original metric-coordinate/normalized-grid roundtrip
(including its [x,y,1,1] convention); no voxel mask, resampling of logits, or
change to the evaluation ROI is introduced. Under nonzero ego motion, changed
ref_points intentionally change the function despite identical parameters.

This controlled AST derivation inherits the original Drive-OccWorld Apache-2.0
source. Exact detector/grid-helper file hashes and both original method AST
hashes are required. No filesystem source or class-wide method is modified.
"""
import ast
import copy
import hashlib
import inspect
import textwrap
import types
from pathlib import Path

MODES = ('native1', 'reference')
SOURCE_SHA256 = '67a8f36e6e886f1f722a27efb4eaf83bb51685679f4c85c74ded3b1c6f2a56b5'
GRID_SOURCE_SHA256 = '5e73296356636d50474a55fb6a4360404d8d46d995cd9e196ac33ce83c2283be'
FUTURE_AST_SHA256 = '2593f6549742ec99c7c0b9877d63361c57f3126012626f51b7e96db7d494cbba'
ALIGN_AST_SHA256 = '8dfd862ebe234c60bf8f7d7efc268aa1a580f89428de800e7f1036f156f3ac6e'
M0_CHECKPOINT_SHA256 = '0348dc8f28d2959637dcd9ac7ccd29fea0da20c25c9856cdb4709c3e221dd9dc'
HEAD_PARAMETERS = 13274016


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_ast(node):
    """Stable Python 3.10/3.14 serialization; no broad source compatibility."""
    node = copy.deepcopy(node)
    for item in ast.walk(node):
        if 'type_params' in item._fields:
            require(not getattr(item, 'type_params', []), 'Unreviewed generic AST')
            item._fields = tuple(k for k in item._fields if k != 'type_params')
    options = dict(include_attributes=False)
    if 'show_empty' in inspect.signature(ast.dump).parameters:
        options['show_empty'] = True
    return ast.dump(node, **options)


def ast_sha(node):
    return hashlib.sha256(canonical_ast(node).encode()).hexdigest()


def method_node(source, name, expected_sha):
    tree = ast.parse(textwrap.dedent(source))
    require(len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef),
            'Expected one ordinary audited method')
    node = tree.body[0]
    require(node.name == name and not node.decorator_list, 'Wrong/decorated method')
    require(ast_sha(node) == expected_sha, 'Unreviewed ' + name + ' AST')
    return tree, node


def slot_contract(target_step, mode='reference'):
    """Semantic identity BEFORE a 1-based transition; no new time embedding."""
    require(mode in MODES and type(target_step) is int and 1 <= target_step <= 4,
            'Expected native1/reference and target step 1..4')
    return dict(target_time_index=target_step, memory_time_index=target_step-1,
                memory_spatial_frame='R' if mode == 'reference' else 'frame_'+str(target_step-1),
                query_spatial_frame='R' if mode == 'reference' else 'frame_'+str(target_step),
                original_action_index=target_step, original_slot_embedding_index=0)


def _validate_call(model, features, condition, valid_frames, occ_flow):
    require(occ_flow == 'occ', 'Only native occupancy is supported')
    require(features.ndim == 4 and features.shape[1] == 1 and
            features.shape[2] == model.bev_h * model.bev_w and
            features.shape[3] == model.future_pred_head.embed_dims,
            'Expected measured t0 [B,1,H*W,C], with native XY order')
    require('occ_gts' in condition and condition['occ_gts'] is None,
            'Future GT cannot enter the forward inputs')
    if model.training:
        require(list(valid_frames) == [1, 2, 3, 4],
                'Training requires all four valid_frames for native BPTT/task loss')


def _reference_memory(physical_pose_queue, tensor_ops):
    """Create a separate spatial queue; never overwrite physical ego metadata."""
    require(physical_pose_queue.ndim == 4 and physical_pose_queue.shape[1:] == (1, 4, 4),
            'Only one observed R-frame slot is supported')
    spatial = physical_pose_queue.new_tensor([[1., 0., 0., 0.], [0., 1., 0., 0.],
                                             [0., 0., 1., 0.], [0., 0., 0., 1.]])
    spatial = spatial.view(1, 1, 4, 4).repeat(physical_pose_queue.shape[0], 1, 1, 1)
    require(tensor_ops.allclose(physical_pose_queue, spatial, rtol=0., atol=1e-5),
            'Initial measured slot is not in R; do not relabel a history slot')
    return spatial


def _reference_grids(tgt_grids, spatial_pose_queue, bev_h, bev_w, pc_range,
                     grid_utils, tensor_ops):
    """Original identity geometry roundtrip, preserving rectangular H/W order.

    Private tensor_ops injection permits exact-helper NumPy/AST CPU tests.
    Production always passes the audited native torch module. The function
    never receives the physical pose queue or modifies action/ego dictionaries.
    """
    require(tgt_grids.ndim == 3 and tuple(tgt_grids.shape[1:]) == (bev_h * bev_w, 2),
            'Target grid must be [B,H*W,2]')
    require(tuple(spatial_pose_queue.shape) == (tgt_grids.shape[0], 1, 4, 4),
            'Spatial queue must be [B,1,4,4]')
    coords = grid_utils.bev_grids_to_coordinates(tgt_grids.unsqueeze(1), pc_range)
    homogeneous = tensor_ops.cat([coords, tensor_ops.ones_like(coords[..., :2])], -1)
    aligned = tensor_ops.matmul(homogeneous, spatial_pose_queue)[..., :2]
    grids, _unused_native_border_mask = grid_utils.bev_coords_to_grids(
        aligned, bev_h, bev_w, pc_range)
    return ((grids + 1.) / 2.).permute(0, 2, 1, 3).contiguous()


def derive_future_pred(source, globals_dict, mode):
    """Derive only the exact locked method; public for source-only CPU tests."""
    require(mode in MODES, 'Unknown coordinate mode')
    tree, node = method_node(source, 'future_pred', FUTURE_AST_SHA256)
    guard = ast.parse('_fc_validate_call(self, prev_bev_input, cond_norm_dict, valid_frames, occ_flow)').body[0]
    node.body.insert(0, guard)
    changes = dict(input_guard=1, separate_R_spatial_queue=0, replace_cross_attention_grid=0)
    if mode == 'reference':
        init = [i for i, n in enumerate(node.body) if isinstance(n, ast.Assign)
                and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == 'ref_to_history_list']
        require(len(init) == 1, 'Ambiguous initial physical pose queue')
        node.body.insert(init[0] + 1, ast.parse(
            'spatial_ref_to_history_list = _fc_reference_memory(ref_to_history_list, torch)').body[0])
        loops = [n for n in node.body if isinstance(n, ast.For) and
                 isinstance(n.target, ast.Name) and n.target.id == 'future_frame_index']
        require(len(loops) == 1, 'Ambiguous native rollout')
        body = loops[0].body
        align = [i for i, n in enumerate(body) if isinstance(n, ast.Assign) and
                 isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                 and n.value.func.attr == '_align_bev_coordnates']
        require(len(align) == 1, 'Ambiguous native alignment call')
        body.insert(align[0] + 1, ast.parse(
            'aligned_prev_grids = _fc_reference_grids(tgt_grids, spatial_ref_to_history_list, '
            'self.bev_h, self.bev_w, self.point_cloud_range, e2e_predictor_utils, torch)').body[0])
        changes.update(separate_R_spatial_queue=1, replace_cross_attention_grid=1)
    ast.fix_missing_locations(tree)
    helpers = dict(_fc_validate_call=_validate_call, _fc_reference_memory=_reference_memory,
                   _fc_reference_grids=_reference_grids)
    require(not set(helpers).intersection(globals_dict), 'Adapter helper namespace collision')
    namespace = dict(globals_dict, **helpers)
    exec(compile(tree, '<M0-frame-consistent:'+mode+'>', 'exec'), namespace)
    return namespace['future_pred'], dict(original_ast_sha256=FUTURE_AST_SHA256,
        derived_ast_sha256=ast_sha(node), structural_changes=changes)


def state_digest(module):
    """Exact state tensor values; same name/dtype/shape/bytes convention as M0."""
    h = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        value = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode()); h.update(value.numpy().tobytes())
    return h.hexdigest()


def install_frame_consistent_adapter(model, mode='reference'):
    """Instance-only install, weights/buffers/RNG/requires_grad unchanged.

    Returns a serializable receipt. Supports exactly native epoch24 M0 with one
    measured memory, four futures, seven-frame labels and no alternative prior.
    It does not modify old results/checkpoints or cache inputs. Use a fresh model
    for each mode; repeated/nested adapter installation fails closed.
    """
    require(mode in MODES, 'Expected native1 or reference')
    require(not hasattr(model, 'frame_consistent_receipt') and
            not hasattr(model, 'observation_memory_receipt'), 'Use a fresh unadapted native model')
    require(model.memory_queue_len == 1 and model.future_pred_frame_num == 4 and
            model.test_future_frame_num == 4, 'Only the four-future native single slot is supported')
    require(not model.turn_on_plan and not model.predict_flow, 'Planning/flow branches are unsupported')
    for name in ('motion_residual', 'doppler_posterior', 'doppler_advection'):
        require(getattr(model, name, None) is None, 'Unsupported alternate spatial route: '+name)
    head = model.future_pred_head
    require(head.memory_queue_len == 1 and head.history_queue_length == 2 and
            tuple(head.prev_frame_embedding.shape) == (1, head.embed_dims), 'Native head/label history mismatch')
    require(not head.use_plan_traj and not head.sem_norm, 'Unexpected query/GT-conditioned branch')
    neck = head.prev_render_neck
    require(neck is not None and not neck.sem_norm and not neck.sem_gt_train and neck.ego_motion_ln,
            'Expected audited ego-conditioned, non-GT native normalization')
    provenance = getattr(model, '_native_state_provenance', {})
    require(provenance.get('checkpoint_sha256') == M0_CHECKPOINT_SHA256 and
            provenance.get('radar_contract') == 'native_loader_B_without_common_source_override',
            'Build the strictly loaded native M0 with native_state_cache first')
    original = model.future_pred.__func__
    align = model._align_bev_coordnates.__func__
    source_path = inspect.getsourcefile(original)
    require(source_path is not None and file_sha(source_path) == SOURCE_SHA256,
            'Unreviewed original detector source')
    require(inspect.getsourcefile(align) == source_path, 'Alignment source differs from native detector')
    method_node(inspect.getsource(align), '_align_bev_coordnates', ALIGN_AST_SHA256)
    grid_utils = original.__globals__['e2e_predictor_utils']
    require(file_sha(grid_utils.__file__) == GRID_SOURCE_SHA256, 'Unreviewed original geometry helpers')
    derived, ast_receipt = derive_future_pred(inspect.getsource(original), original.__globals__, mode)
    parameters = list(head.named_parameters())
    count = sum(p.numel() for _, p in parameters)
    require(count == HEAD_PARAMETERS, 'Unexpected native future-head capacity')
    identities = [(name, id(p), tuple(p.shape), p.requires_grad) for name, p in parameters]
    weights_sha = state_digest(head)
    # Binding one Python method draws no RNG, allocates no parameters and changes
    # no tensor. All potentially failing source/shape checks above precede it.
    model.future_pred = types.MethodType(derived, model)
    require(identities == [(n, id(p), tuple(p.shape), p.requires_grad) for n, p in head.named_parameters()]
            and weights_sha == state_digest(head), 'Installation changed parameter identity or values')
    receipt = dict(schema='m0-frame-consistent-adapter-v1', mode=mode,
        adapter_sha256=file_sha(__file__), detector_source_sha256=SOURCE_SHA256,
        grid_source_sha256=GRID_SOURCE_SHA256, original_align_ast_sha256=ALIGN_AST_SHA256,
        future_pred_derivation=ast_receipt, parameters_before=count, parameters_after=count,
        state_sha256_before=weights_sha, state_sha256_after=weights_sha,
        parameter_objects_values_shapes_requires_grad_unchanged=True,
        added_parameters=0, changed_weight_tensors=0, added_detach_operations=0,
        input_memory_slots=1, generated_memory_slots=1, data_history_frames=2,
        future_steps=4, slot_contracts=[slot_contract(h, mode) for h in range(1, 5)],
        spatial_route='independent R-to-R grid' if mode == 'reference' else 'native future-to-physical-history grid',
        physical_pose_queue='original ref2future appends; original future2history -> ego normalization',
        action_condition='original values, basis, target_frame_index and planning-disabled GT trajectory path',
        task='original complete decoder, all intermediate layers and four future task losses; no output warp or ROI change',
        gradients='original autoregressive BPTT; training valid_frames must be [1,2,3,4]',
        intended_nonparity='reference cross-attention changes under nonzero ego; no logit parity required there',
        identity_pose_parity='expected up to native geometry roundtrip precision; real CUDA test remains required',
        verification='installation/source contracts only; not a GPU forward or effectiveness receipt')
    model.frame_consistent_receipt = receipt
    return receipt
