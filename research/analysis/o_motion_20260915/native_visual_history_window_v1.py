"""Scoped cold-start camera-window wrapper for a loaded Drive_OccWorld model.

This module does not load data, run a forward pass, or alter model parameters.
It exposes ``observation_window(model, camera_times, trace_dict)`` as a
context manager.  Inside the context, the yielded model can be called through
the normal MMDet entry point, for example::

    with observation_window(model, 2, trace) as wrapped:
        result = wrapped(return_loss=False, **data)

The wrapper keeps the original future transition and evaluator.  It only
shortens the already prepared camera/meta window and rebuilds the real
memory input from the history BEVs plus the current reference BEV.
"""

import copy
import hashlib
from contextlib import contextmanager
from types import MethodType


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _copy_cold_start_meta(meta):
    """Copy a meta and apply the official first-frame cold-start convention."""
    result = copy.deepcopy(meta)
    _require(isinstance(result, dict), "Each camera meta must be a dictionary")
    _require("can_bus" in result, "Cold-start meta must contain can_bus")
    can_bus = copy.deepcopy(result["can_bus"])
    _require(len(can_bus) >= 3, "can_bus must have at least three position values")
    can_bus[:3] = 0
    can_bus[-1] = 0
    result["can_bus"] = can_bus
    result["prev_bev_exists"] = False
    return result


def _meta_batches(img_metas, batch_size):
    """Validate the normal MMDet batched mapping and return its maps."""
    _require(isinstance(img_metas, (list, tuple)),
             "forward_test expects img_metas as a batch sequence")
    _require(len(img_metas) == batch_size,
             "img_metas batch size differs from img")
    result = []
    for batch_index, mapping in enumerate(img_metas):
        _require(isinstance(mapping, dict),
                 "img_metas[%d] must map camera indices to metadata" % batch_index)
        keys = list(mapping.keys())
        _require(set(keys) == {0, 1, 2},
                 "Expected exactly camera meta keys {0,1,2}")
        result.append(mapping)
    return result


def _select_window(img_metas, camera_times):
    selected = []
    original_indices = list(range(3 - camera_times, 3))
    for mapping in img_metas:
        selected_mapping = {
            new_index: copy.deepcopy(mapping[old_index])
            for new_index, old_index in enumerate(original_indices)
        }
        # A shortened history must begin from a genuine measured frame.  H3
        # is left byte-for-byte at the metadata level for native parity.
        if camera_times < 3:
            selected_mapping[0] = _copy_cold_start_meta(selected_mapping[0])
        selected.append(selected_mapping)
    return selected, original_indices


def _tensor_sha256(value):
    """Digest one tensor without retaining it in the trace."""
    import torch

    _require(torch.is_tensor(value), "Expected a tensor for image digest")
    cpu = value.detach().to(device="cpu").contiguous()
    raw = memoryview(cpu.numpy()).cast("B")
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode("ascii"))
    digest.update(repr(tuple(cpu.shape)).encode("ascii"))
    digest.update(raw)
    return digest.hexdigest()


def _identity(meta):
    return {
        key: str(meta[key]) for key in ("sample_token", "sample_idx", "lidar_token", "scene_token")
        if key in meta
    }


def _trace_call(trace_dict, selected_indices, selected_metas, selected_img):
    _require(isinstance(trace_dict, dict), "trace_dict must be a dictionary")
    current_metas = [mapping[len(selected_indices) - 1] for mapping in selected_metas]
    selected_timestamps = [
        [mapping[i].get("timestamp") for i in selected_indices]
        for mapping in selected_metas
    ]
    trace_dict.setdefault("calls", []).append({
        "selected_original_indices": list(selected_indices),
        "input_camera_count": int(selected_img.shape[1]),
        "input_camera_times": list(selected_indices),
        "selected_timestamps_us": selected_timestamps,
        "current_img_sha256": _tensor_sha256(selected_img[:, -1]),
        "current_meta_identity": [_identity(meta) for meta in current_metas],
        "cold_start_reset": bool(len(selected_indices) < 3),
        "reset_fields": (["prev_bev_exists", "can_bus[:3]", "can_bus[-1]"]
                          if len(selected_indices) < 3 else []),
    })


def _validate_model_boundary(model):
    _require(getattr(model, "memory_queue_len", None) == 1,
             "This wrapper requires model.memory_queue_len == 1")
    head = getattr(model, "future_pred_head", None)
    _require(head is not None and getattr(head, "memory_queue_len", None) == 1,
             "This wrapper requires future_head.memory_queue_len == 1")
    _require(getattr(model, "turn_on_plan", None) is False,
             "This wrapper requires turn_on_plan=False")
    _require(getattr(model, "turn_on_flow", None) is False,
             "This wrapper requires turn_on_flow=False")
    _require(getattr(model, "scientific_eval", None) is True,
             "This wrapper requires scientific_eval=True")
    _require(getattr(model, "only_generate_dataset", False) is False,
             "This wrapper requires normal forward_test evaluation")


def _window_forward_test(self, img_metas, img=None, **kwargs):
    import torch

    _validate_model_boundary(self)
    _require(torch.is_tensor(img) and img.ndim == 6,
             "Expected img with shape [B,3,6,C,H,W]")
    _require(img.shape[1] == 3 and img.shape[2] == 6,
             "Expected exactly three six-camera timestamps")

    camera_times = self._native_visual_history_window_camera_times
    trace_dict = self._native_visual_history_window_trace
    batches = _meta_batches(img_metas, img.shape[0])
    selected_metas, selected_indices = _select_window(batches, camera_times)
    selected_img = img[:, -camera_times:].contiguous()
    _trace_call(trace_dict, selected_indices, selected_metas, selected_img)

    # This is the original forward_test route, with only the selected camera
    # window substituted.  The future transition and evaluator remain native.
    self.eval()
    num_frames = selected_img.shape[1]
    prev_img = selected_img[:, :-1, ...]
    prev_img_metas = copy.deepcopy(selected_metas)
    if num_frames > 1:
        prev_bev, prev_bev_list = self.obtain_history_bev(
            prev_img, prev_img_metas)
    else:
        # H1 has no history BEV.  Do not fabricate a zero or duplicated slot.
        prev_bev, prev_bev_list = None, []

    current_img = selected_img[:, -1, ...]
    current_metas = [mapping[num_frames - 1] for mapping in selected_metas]
    ref_bev, motion_state = self.obtain_ref_bev(
        current_img, current_metas, prev_bev,
        radar_bev=kwargs.get("radar_bev"),
        radar_observations=kwargs.get("radar_observations"),
        return_motion_state=True,
        radar_nll_support_mask=kwargs.get("radar_nll_support_mask"))

    # ``prev_bev_list`` can be empty for H1/H2 when memory_queue_len == 1.
    # Stack actual states only, then keep the final memory slot exactly as the
    # native path does for H3.
    memory_states = list(prev_bev_list) + [ref_bev]
    _require(memory_states, "At least the current ref_bev must exist")
    prev_bev_input = torch.stack(memory_states, dim=1)
    prev_bev_input = prev_bev_input[:, -self.memory_queue_len:, ...].contiguous()

    valid_frames = []
    cond_norm_dict = {"occ_gts": None}
    action_condition_dict = {
        "command": kwargs.get("command"),
        "vel_steering": kwargs.get("vel_steering"),
    }
    plan_dict = {
        "sem_occupancy": None,
        "sample_traj": kwargs.get("sample_traj"),
        "gt_traj": kwargs.get("sdc_planning"),
        "ref_pose_pred": None,
    }
    next_bev_preds, _, next_pose_preds, _ = self.future_pred(
        prev_bev_input, action_condition_dict, cond_norm_dict, plan_dict,
        valid_frames, current_metas, prev_img_metas, num_frames,
        occ_flow="occ", radar_bev=kwargs.get("radar_bev"),
        motion_state=motion_state)
    del next_pose_preds

    return {"occ_records": self.evaluate_occ_records(
        next_bev_preds, kwargs.get("segmentation"), current_metas)}


@contextmanager
def observation_window(model, camera_times, trace_dict):
    """Temporarily run H1/H2/H3 cold-start camera windows.

    ``camera_times`` is an integer in ``{1, 2, 3}``; the last N timestamps of
    the prepared three-timestamp input are selected.  The yielded object is
    the same model, so its normal ``__call__`` accepts ``return_loss=False``.
    """
    _validate_model_boundary(model)
    _require(isinstance(camera_times, int) and camera_times in (1, 2, 3),
             "camera_times must be one of 1, 2, 3")
    _require(isinstance(trace_dict, dict), "trace_dict must be a dictionary")
    marker = object()
    old_times = getattr(model, "_native_visual_history_window_camera_times", marker)
    old_trace = getattr(model, "_native_visual_history_window_trace", marker)
    old_forward = getattr(model, "forward_test")
    model._native_visual_history_window_camera_times = camera_times
    model._native_visual_history_window_trace = trace_dict
    model.forward_test = MethodType(_window_forward_test, model)
    try:
        yield model
    finally:
        model.forward_test = old_forward
        if old_times is marker:
            del model._native_visual_history_window_camera_times
        else:
            model._native_visual_history_window_camera_times = old_times
        if old_trace is marker:
            del model._native_visual_history_window_trace
        else:
            model._native_visual_history_window_trace = old_trace
