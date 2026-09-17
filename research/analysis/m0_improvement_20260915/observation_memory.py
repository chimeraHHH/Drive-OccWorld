"""Two-slot memory adaptation of the audited M0 future transition.

Install AFTER strict loading of the native M0 checkpoint and BEFORE constructing
an optimizer. This module changes runtime instances only, never frozen sources.
The detector still extracts one measured t0 state. Its real-data history length
and future_head.history_queue_length (GT slicing) are never changed.

The derived future_pred is the exact audited source AST except for: an input
guard, duplication immediately after the native reference-transform lookup,
and replacement of the four rolling-queue assignments with one paired update.
All action/ego logic, native SE(3) alignment, valid_frames/no_grad branches,
occupancy decoding, and return values stay in that original function.

Source provenance: Drive-OccWorld-sota-p2, Drive_OccWorld.future_pred, inspected
2026-09-15. v2 normalizes Python 3.14's omitted empty AST lists to Python 3.10
serialization. Empty type_params (introduced in 3.12) are omitted; nonempty
ones fail closed. Both exact source-file SHA and method-AST SHA are required.
The source inherits Apache-2.0 Drive-OccWorld code. Scientific scope is M0:
no planning/flow/motion branches, no GT-conditioned normalization. Their code
is retained by derivation but those untested configurations fail closed.
"""
import ast
import copy
import hashlib
import inspect
import textwrap
import types
from pathlib import Path


MODES = ("native1", "persistent2", "rolling2")
EXPECTED_FUTURE_PRED_AST_SHA256 = (
    "2593f6549742ec99c7c0b9877d63361c57f3126012626f51b7e96db7d494cbba")
EXPECTED_SOURCE_FILE_SHA256 = (
    "67a8f36e6e886f1f722a27efb4eaf83bb51685679f4c85c74ded3b1c6f2a56b5")


def memory_plan(mode, future_step):
    """Source frame indices visible BEFORE the requested 1-based future step."""
    if mode not in MODES or not isinstance(future_step, int) or future_step < 1:
        raise ValueError("Expected a known mode and a positive integer step")
    last = future_step - 1
    if mode == "native1":
        return (last,)
    return ((0, last) if mode == "persistent2" else (max(0, last - 1), last))


def projection_row_indices(num_heads, num_points, components):
    """Index map [head, old_level=1, point, component] -> two equal levels.

    Repeating the flattened weight matrix twice would incorrectly interleave
    heads. The explicit map is shared by weight and bias and draws no RNG.
    """
    if min(num_heads, num_points, components) < 1:
        raise ValueError("All projection dimensions must be positive")
    return [h * num_points * components + p * components + c
            for h in range(num_heads) for _level in range(2)
            for p in range(num_points) for c in range(components)]


def _validate_inputs(features, cond_norm_dict, occ_flow):
    if occ_flow != "occ":
        raise ValueError("Observation-memory adapter supports M0 occupancy only")
    if features.ndim != 4 or features.shape[1] != 1:
        raise ValueError("Expected the native single measured t0 state [B,1,HW,C]")
    if "occ_gts" not in cond_norm_dict or cond_norm_dict["occ_gts"] is not None:
        raise ValueError("GT occupancy cannot condition the future transition")


def initialize_memory(features, transforms):
    """Duplicate the measured state and its native reference transform.

    No detach or in-place update: gradient flow into the reference state is
    preserved, including summation of contributions from the two copies.
    """
    import torch
    if features.ndim != 4 or features.shape[1] != 1:
        raise ValueError("Initial memory must contain exactly t0")
    if tuple(transforms.shape) != (features.shape[0], 1, 4, 4):
        raise ValueError("Initial feature and coordinate memories disagree")
    return (torch.cat((features, features), dim=1).contiguous(),
            torch.cat((transforms, transforms), dim=1).contiguous())


def update_memory(features, transforms, next_feature, ref_to_next, mode):
    """Keep feature and coordinate slots paired; matrices use native convention.

    persistent2 preserves slot zero as measured t0. rolling2 preserves only
    the latest old slot. Both append the newly generated feature AND its own
    ref2future matrix. Never warp values here: native attention samples each
    slot through _align_bev_coordnates, so another warp would double-align it.
    """
    import torch
    if mode not in ("persistent2", "rolling2"):
        raise ValueError("Two-slot update requires persistent2 or rolling2")
    if features.ndim != 4 or features.shape[1] != 2:
        raise ValueError("Two slots are required before update")
    if tuple(transforms.shape) != (features.shape[0], 2, 4, 4):
        raise ValueError("Feature and transform slot counts disagree")
    if tuple(next_feature.shape) != tuple(features[:, 0].shape):
        raise ValueError("Generated feature shape mismatch")
    if tuple(ref_to_next.shape) != (features.shape[0], 4, 4):
        raise ValueError("Generated coordinate transform shape mismatch")
    keep = slice(0, 1) if mode == "persistent2" else slice(-1, None)
    return (torch.cat((features[:, keep], next_feature.unsqueeze(1)), 1).contiguous(),
            torch.cat((transforms[:, keep], ref_to_next.unsqueeze(1)), 1).contiguous())


def _canonical_ast_dump(node):
    """Stable Python 3.10-style dump without changing the compilable AST.

    Verified against the same 67a8... source file on 3.10 and 3.14: 3.14's
    default dump omits empty args/keywords/orelse/decorator_list. show_empty
    restores them. New empty generic type parameters have no counterpart in
    3.10, and are removed only from a private serialization copy.
    """
    tree = copy.deepcopy(node)
    for item in ast.walk(tree):
        if "type_params" in item._fields:
            if getattr(item, "type_params", []):
                raise ValueError("Generic type parameters are outside audited M0 source")
            item._fields = tuple(name for name in item._fields if name != "type_params")
    kwargs = {"include_attributes": False}
    if "show_empty" in inspect.signature(ast.dump).parameters:
        kwargs["show_empty"] = True
    return ast.dump(tree, **kwargs)


def _ast_digest(node):
    return hashlib.sha256(_canonical_ast_dump(node).encode()).hexdigest()


def derive_future_pred(source, globals_dict, mode, expected_ast_sha256=None):
    """Compile only the known transition with audited, structural edits.

    Public for source-only tests; production installation always supplies the
    frozen expected digest. No untrusted external source is loaded here.
    """
    if mode not in MODES:
        raise ValueError("Unknown mode")
    tree = ast.parse(textwrap.dedent(source))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Expected one ordinary future_pred function")
    fn = tree.body[0]
    if fn.name != "future_pred" or fn.decorator_list:
        raise ValueError("Unexpected method name/decorator")
    source_digest = _ast_digest(fn)
    expected = expected_ast_sha256 or EXPECTED_FUTURE_PRED_AST_SHA256
    if source_digest != expected:
        raise ValueError("Unreviewed future_pred AST: " + source_digest)
    original = copy.deepcopy(fn)
    guard = ast.parse(
        "_om_validate_inputs(prev_bev_input, cond_norm_dict, occ_flow)").body[0]
    fn.body.insert(0, guard)
    changes = {"input_guard": 1, "initial_duplication": 0, "paired_update": 0}
    if mode != "native1":
        found = []
        for i, node in enumerate(fn.body):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "ref_to_history_list"
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == "_get_history_ref_to_previous_transform"):
                found.append(i)
        if len(found) != 1:
            raise ValueError("Native initial coordinate lookup is ambiguous")
        duplicate = ast.parse(
            "prev_bev_input, ref_to_history_list = "
            "_om_initialize_memory(prev_bev_input, ref_to_history_list)").body[0]
        fn.body.insert(found[0] + 1, duplicate)
        changes["initial_duplication"] = 1
        old_text = """
prev_bev_input = torch.cat([prev_bev_input, pred_feat[-1].unsqueeze(1)], 1)
prev_bev_input = prev_bev_input[:, 1:, ...].contiguous()
ref_to_history_list = torch.cat([ref_to_history_list, ref2future.unsqueeze(1)], 1)
ref_to_history_list = ref_to_history_list[:, 1:].contiguous()
"""
        expected_nodes = [ast.dump(n, include_attributes=False)
                          for n in ast.parse(old_text).body]
        loops = [n for n in fn.body if isinstance(n, ast.For)
                 and isinstance(n.target, ast.Name)
                 and n.target.id == "future_frame_index"]
        if len(loops) != 1:
            raise ValueError("Future loop is ambiguous")
        body = loops[0].body
        starts = [i for i in range(len(body) - 3)
                  if [ast.dump(n, include_attributes=False)
                      for n in body[i:i + 4]] == expected_nodes]
        if len(starts) != 1:
            raise ValueError("Native feature/coordinate paired update changed")
        replacement = ast.parse(
            "prev_bev_input, ref_to_history_list = _om_update_memory("
            "prev_bev_input, ref_to_history_list, pred_feat[-1], ref2future, "
            + repr(mode) + ")").body[0]
        body[starts[0]:starts[0] + 4] = [replacement]
        changes["paired_update"] = 1
    ast.fix_missing_locations(tree)
    namespace = dict(globals_dict)
    if any(k in namespace for k in ("_om_validate_inputs", "_om_initialize_memory",
                                    "_om_update_memory")):
        raise ValueError("Unexpected helper collision in original namespace")
    namespace.update(_om_validate_inputs=_validate_inputs,
                     _om_initialize_memory=initialize_memory,
                     _om_update_memory=update_memory)
    compiled_filename = "<M0-observation-memory:" + mode + ">"
    exec(compile(tree, compiled_filename, "exec"), namespace)
    derived = namespace["future_pred"]
    return derived, dict(source_ast_sha256=_ast_digest(original),
                         derived_ast_sha256=_ast_digest(fn),
                         structural_changes=changes)


def _cross_attention_modules(head):
    """Select temporal cross attention by operation_order, never by fuzzy name."""
    result = []
    layers = head.transformer.decoder.layers
    for layer_index, layer in enumerate(layers):
        attn_index = 0
        for op in layer.operation_order:
            if op in ("self_attn", "cross_attn", "cross_attn_action"):
                if op == "cross_attn":
                    result.append(("transformer.decoder.layers.%d.attentions.%d" %
                                   (layer_index, attn_index), layer.attentions[attn_index]))
                attn_index += 1
            elif "attn" in op:
                raise ValueError("Unknown attention operation: " + op)
        if attn_index != len(layer.attentions):
            raise ValueError("Attention operation/module count mismatch")
    if not result or len({id(m) for _, m in result}) != len(result):
        raise ValueError("No distinct temporal attention modules")
    return result


def _expanded_linear(linear, heads, points, components):
    import torch
    if not isinstance(linear, torch.nn.Linear):
        raise TypeError("Expected a linear projection")
    old_rows = heads * points * components
    if linear.out_features != old_rows or linear.weight.shape[0] != old_rows:
        raise ValueError("Unexpected original projection dimensions")
    ids = torch.tensor(projection_row_indices(heads, points, components),
                       device=linear.weight.device, dtype=torch.long)
    expanded = copy.deepcopy(linear)
    expanded.out_features = old_rows * 2
    expanded.weight = torch.nn.Parameter(
        linear.weight.detach().index_select(0, ids).clone(),
        requires_grad=linear.weight.requires_grad)
    if linear.bias is not None:
        expanded.bias = torch.nn.Parameter(
            linear.bias.detach().index_select(0, ids).clone(),
            requires_grad=linear.bias.requires_grad)
    return expanded


def install_observation_memory(model, mode):
    """Return a JSON-serializable migration receipt; enable the entire future head.

    model is the unwrapped detector, with native weights already strictly
    loaded. native1 keeps all parameter values/shapes and original queue rule;
    all modes add the no-GT runtime guard. Install exactly once per instance.
    This changes parameters for two-slot modes; pre-existing optimizer states
    must NOT be reused. The caller controls all other train/freeze decisions.
    """
    import torch
    if mode not in MODES:
        raise ValueError("mode must be one of " + repr(MODES))
    if hasattr(model, "observation_memory_receipt"):
        raise ValueError("Adapter already installed; build a fresh native model")
    if getattr(model, "memory_queue_len", None) != 1:
        raise ValueError("Detector must retain one measured input state")
    if getattr(model, "turn_on_plan", False) or getattr(model, "predict_flow", False):
        raise ValueError("Only the audited M0 occupancy configuration is supported")
    for name in ("motion_residual", "doppler_posterior", "doppler_advection"):
        if getattr(model, name, None) is not None:
            raise ValueError("Not a native M0 model: active " + name)
    head = model.future_pred_head
    if head.memory_queue_len != 1 or tuple(head.prev_frame_embedding.shape) != (1, head.embed_dims):
        raise ValueError("Expected a native one-slot future head")
    neck = head.prev_render_neck
    if neck is not None and (getattr(neck, "sem_norm", False)
                             or getattr(neck, "sem_gt_train", False)):
        raise ValueError("Semantic/GT normalization is outside the M0 contract")
    original = model.future_pred.__func__
    source_path = inspect.getsourcefile(original)
    if source_path is None or not Path(source_path).is_file():
        raise ValueError("Original method must have an auditable source file")
    source_file_sha = hashlib.sha256(Path(source_path).read_bytes()).hexdigest()
    if source_file_sha != EXPECTED_SOURCE_FILE_SHA256:
        raise ValueError("Unreviewed detector source file SHA: " + source_file_sha)
    source = inspect.getsource(original)
    derived, provenance = derive_future_pred(source, original.__globals__, mode)
    provenance.update(source_file=str(Path(source_path).resolve()),
                      source_file_sha256=source_file_sha,
                      ast_serialization="python310-show-empty-without-empty-type-params-v1")
    modules = _cross_attention_modules(head)
    before = sum(p.numel() for p in head.parameters())
    pending = []
    mappings = []
    for name, module in modules:
        if module.num_levels != 1:
            raise ValueError("Temporal cross attention is not native one-level: " + name)
        offsets = _expanded_linear(module.sampling_offsets, module.num_heads,
                                   module.num_points, 2)
        weights = _expanded_linear(module.attention_weights, module.num_heads,
                                   module.num_points, 1)
        pending.append((module, offsets, weights))
        mappings.append(dict(module=name,heads=module.num_heads,points=module.num_points,
                             old_levels=1,new_levels=(1 if mode == "native1" else 2),
                             row_order="head,level,point,component",
                             unchanged=["value_proj", "output_proj"]))
    # Every validation/compilation above happens before changing model state.
    real_history_length = head.history_queue_length
    if mode != "native1":
        for module, offsets, weights in pending:
            module.sampling_offsets = offsets
            module.attention_weights = weights
            module.num_levels = 2
        head.prev_frame_embedding = torch.nn.Parameter(
            head.prev_frame_embedding.detach().repeat(2, 1).clone(),
            requires_grad=head.prev_frame_embedding.requires_grad)
        head.memory_queue_len = 2
    head.requires_grad_(True)
    model.future_pred = types.MethodType(derived, model)
    assert model.memory_queue_len == 1 and head.history_queue_length == real_history_length
    receipt = dict(schema="m0-observation-memory-adapter-v2",mode=mode,
                   detector_input_memory_queue_len=1,
                   transition_memory_queue_len=head.memory_queue_len,
                   data_history_queue_length=real_history_length,
                   head_parameters_before=before,
                   head_parameters_after=sum(p.numel() for p in head.parameters()),
                   trainable_future_head_parameters=sum(p.numel() for p in head.parameters() if p.requires_grad),
                   parameter_mappings=mappings,source_provenance=provenance,
                   gradients="No new detach on measured/generated states; original valid_frames/no_grad logic retained",
                   protocol="No future GT conditioning; native action and per-slot SE(3) alignment retained",
                   first_step_equivalence="Analytically expected for duplicate states; actual float/CUDA parity must be measured",
                   optimizer_contract="Install before optimizer construction; no native optimizer-state migration")
    model.observation_memory_receipt = receipt
    return receipt
