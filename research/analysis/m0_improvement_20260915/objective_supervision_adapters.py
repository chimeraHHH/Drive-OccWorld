"""Independent supervision interventions on the audited native1 model.

F: original 12 losses on complete GT, via the frozen two-edit adapter.
O: original coarse-GT 12 losses are still calculated, but only CE/Lovasz
contribute to the returned objective. sem/geo forward computation is retained;
this is deliberately not a claim of computational savings or numerical repair.
No forward, targets, probability formula, class weight or evaluator is changed.
"""
import inspect
import math
from pathlib import Path
import types

FULL_SOURCE_SHA = '12698cb2bb071dd6daf79ccc64ba8f506d0a773a676dd57296f8410fe747d0f7'
DETECTOR_SOURCE_SHA = '67a8f36e6e886f1f722a27efb4eaf83bb51685679f4c85c74ded3b1c6f2a56b5'
HEAD_SOURCE_SHA = '4738c03a28e353efd4dec68adf6c9407aa73b9273bd13c2f67021795d6240a7a'
FAMILIES = ('ce', 'sem_scal', 'geo_scal', 'lovasz')
ALL_KEYS = frozenset('loss_voxel_'+family+'_inter_'+str(i) for family in FAMILIES for i in range(3))
O_KEYS = frozenset('loss_voxel_'+family+'_inter_'+str(i) for family in ('ce', 'lovasz') for i in range(3))


def _sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def select_objective(losses, arm):
    """Preserve original dictionary order and tensor identity; no rescaling."""
    _require(arm in ('F', 'O') and set(losses) == ALL_KEYS, 'Unexpected arm/original 12-loss schema')
    keys = ALL_KEYS if arm == 'F' else O_KEYS
    return {key: value for key, value in losses.items() if key in keys}


def install_objective_supervision(model, arm):
    _require(arm in ('F', 'O'), 'Only F and O are supported')
    _require(not hasattr(model, 'objective_supervision_receipt'), 'Objective adapter already installed')
    head = model.future_pred_head
    _require(getattr(model, 'observation_memory_receipt', {}).get('mode') == 'native1',
             'Must start from the source-validated native1 model')
    _require(model.memory_queue_len == head.memory_queue_len == 1 and head.history_queue_length == 2,
             'Memory/history contract differs')
    _require(not hasattr(model, 'frame_consistent_receipt') and
             not hasattr(head, '_full_resolution_supervision_receipt'), 'Do not combine interventions')
    original = model.compute_occ_loss
    _require(_sha(inspect.getsourcefile(original)) == DETECTOR_SOURCE_SHA,
             'compute_occ_loss must be the unchanged native source')
    for method in (head.loss_occ, head.loss_voxel):
        _require(_sha(inspect.getsourcefile(method)) == HEAD_SOURCE_SHA, 'Expected original loss head methods')
    # The native1 adapter already verifies the native future transition AST.
    # Bind its receipt here so this loss intervention cannot select two-memory.
    _require(model.observation_memory_receipt['data_history_queue_length'] == 2,
             'Native history receipt differs')
    full_file = Path(__file__).with_name('full_resolution_supervision_preflight.py')
    _require(_sha(full_file) == FULL_SOURCE_SHA, 'Frozen full-resolution source changed')
    full_receipt = None
    if arm == 'F':
        from full_resolution_supervision_preflight import install_full_resolution
        full_receipt = install_full_resolution(head)

    def compute_objective(self, occ_preds, occ_gts):
        losses = original(occ_preds, occ_gts)
        selected = select_objective(losses, arm)
        # Scalars only, never references to the discarded sem/geo autograd graph.
        all_scalars = {key: float(value.detach().item()) for key, value in losses.items()}
        _require(all(math.isfinite(value) for value in all_scalars.values()),
                 'Nonfinite original loss; diagnostic does not silently stabilize/drop invalid formulas')
        self._objective_last_loss_audit = dict(arm=arm, original_components=all_scalars,
            optimized_keys=list(selected), optimized_sum=sum(all_scalars[key] for key in selected),
            omitted_sum=sum(all_scalars[key] for key in losses if key not in selected),
            all_original_twelve_evaluated=True, probability_formulas_unchanged=True)
        return selected

    model.compute_occ_loss = types.MethodType(compute_objective, model)
    receipt = dict(schema='m0-objective-supervision-adapter-v1', arm=arm,
        adapter_sha256=_sha(__file__), full_resolution_source_sha256=FULL_SOURCE_SHA,
        detector_source_sha256=DETECTOR_SOURCE_SHA, head_source_sha256=HEAD_SOURCE_SHA,
        native_memory_mode='native1', history_queue_length=2,
        full_resolution_adapter=full_receipt,
        GT_resolution=[512,512,40] if arm == 'F' else [256,256,20],
        optimized_keys=sorted(ALL_KEYS if arm == 'F' else O_KEYS),
        all_original_twelve_evaluated=True,
        objective='original 12 losses on complete GT' if arm == 'F' else 'original coarse CE[1,5] plus Lovasz, three layers and five pooled horizons',
        omitted_family_forward_cost_retained=(arm == 'O'),
        forward_space_action_evaluator_unchanged=True,
        simultaneous_frame_or_memory_intervention=False)
    model.objective_supervision_receipt = receipt
    return receipt
