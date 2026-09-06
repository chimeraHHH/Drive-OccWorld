"""Sample-deduplicated occupancy metrics and scene-cluster confidence intervals.

Only NumPy is required. Confusion matrices have ground-truth rows and predicted
columns; IoU is unchanged if both axes are consistently transposed. Undefined
class IoUs are ``None`` and excluded from mIoU, including bootstrap replicates.
The time-weighted legacy score uses horizon *index* weights 1, 1/2, ... .
"""

import math
import numbers

import numpy as np


ABSENT_CLASS_POLICY = 'null_per_class_and_exclude_from_mIoU'


def _positive_integer(value, name, allow_zero=False):
    minimum = 0 if allow_zero else 1
    if (isinstance(value, bool) or not isinstance(value, numbers.Integral)
            or value < minimum):
        raise ValueError('{} must be an integer >= {}'.format(name, minimum))
    return int(value)


def _normalise_record(record):
    if not isinstance(record, dict):
        raise ValueError('each occupancy record must be a dictionary')
    required = ('sample_token', 'scene_token', 'horizon_seconds',
                'hist_by_horizon')
    if any(key not in record for key in required):
        raise ValueError('occupancy record requires {}'.format(required))
    for key in ('sample_token', 'scene_token'):
        if not isinstance(record[key], str) or not record[key].strip():
            raise ValueError('{} must be a nonempty string'.format(key))
    try:
        horizons = np.asarray(record['horizon_seconds'], dtype=np.float64)
        hist = np.asarray(record['hist_by_horizon'], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError('invalid numeric occupancy record') from exc
    if (horizons.ndim != 1 or horizons.size < 2
            or not np.isfinite(horizons).all() or horizons[0] != 0
            or np.any(np.diff(horizons) <= 0)):
        raise ValueError('horizons must start at 0 and strictly increase')
    if (hist.ndim != 3 or hist.shape[0] != horizons.size
            or hist.shape[1] == 0 or hist.shape[1] != hist.shape[2]):
        raise ValueError('hist_by_horizon must have shape [T, C, C]')
    if not np.isfinite(hist).all() or np.any(hist < 0):
        raise ValueError('confusion entries must be finite and nonnegative')
    return dict(sample_token=record['sample_token'],
                scene_token=record['scene_token'],
                horizon_seconds=horizons.tolist(),
                hist_by_horizon=hist.copy())


def _unique_records(records, expected_samples=None):
    if expected_samples is not None:
        expected_samples = _positive_integer(
            expected_samples, 'expected_samples', allow_zero=True)
    unique = {}
    reference_horizons = None
    reference_shape = None
    for raw_record in records:
        record = _normalise_record(raw_record)
        if reference_horizons is None:
            reference_horizons = record['horizon_seconds']
            reference_shape = record['hist_by_horizon'].shape
        if record['horizon_seconds'] != reference_horizons:
            raise ValueError('horizon_seconds differ across occupancy records')
        if record['hist_by_horizon'].shape != reference_shape:
            raise ValueError('confusion shapes differ across occupancy records')
        token = record['sample_token']
        if token in unique:
            previous = unique[token]
            if (previous['scene_token'] != record['scene_token']
                    or not np.array_equal(previous['hist_by_horizon'],
                                          record['hist_by_horizon'])):
                raise ValueError('conflicting duplicate sample: {}'.format(token))
        else:
            unique[token] = record
    if expected_samples is not None and len(unique) != expected_samples:
        raise ValueError('expected {} unique samples, received {}'.format(
            expected_samples, len(unique)))
    if not unique:
        raise ValueError('no occupancy records; metrics are undefined')
    # Stable ordering makes the bootstrap seed independent of rank collection.
    return [unique[token] for token in sorted(unique)]


def confusion_ious(hist):
    """Return IoUs in [0, 1], with absent classes explicitly represented."""
    hist = np.asarray(hist, dtype=np.float64)
    if (hist.ndim != 2 or hist.shape[0] != hist.shape[1]
            or hist.shape[0] == 0 or not np.isfinite(hist).all()
            or np.any(hist < 0)):
        raise ValueError('expected a finite nonnegative square confusion matrix')
    intersection = np.diag(hist)
    union = hist.sum(axis=0) + hist.sum(axis=1) - intersection
    present = union > 0
    ious = np.divide(intersection, union, out=np.zeros_like(union), where=present)
    return {
        'per_class_iou': [float(value) if valid else None
                          for value, valid in zip(ious, present)],
        'mIoU': float(ious[present].mean()) if np.any(present) else None,
        'defined_class_indices': np.flatnonzero(present).tolist(),
    }


def _scoped_confusions(hist, horizons):
    scopes = {'all': hist.sum(axis=0), 'current': hist[0],
              'future': hist[1:].sum(axis=0),
              'future_time_weighted': np.sum(
                  hist[1:] / np.arange(1, len(horizons))[:, None, None], axis=0)}
    for horizon, confusion in zip(horizons, hist):
        scopes['horizon/{}'.format(format(horizon, '.12g'))] = confusion
    return scopes


def aggregate_records(records, expected_samples=None):
    """Deduplicate distributed sampler padding before computing any metric.

    Duplicate records must agree exactly, including floating-point confusion
    entries. Unexpected sample counts or incompatible protocols fail loudly.
    The original unique records are retained for scene bootstrap and pairing.
    """
    unique = _unique_records(records, expected_samples)
    hist = np.stack([record['hist_by_horizon'] for record in unique]).sum(axis=0)
    horizons = unique[0]['horizon_seconds']
    scopes = _scoped_confusions(hist, horizons)
    return {
        'occ_records': unique,
        'num_unique_samples': len(unique),
        'num_scenes': len({record['scene_token'] for record in unique}),
        'horizon_seconds': horizons,
        'hist_for_iou': [scopes['all']],
        'hist_for_iou_current': [scopes['current']],
        'hist_for_iou_future': [scopes['future']],
        'hist_for_iou_future_time_weighting': [scopes['future_time_weighted']],
        'horizon_ious': {
            format(horizon, '.12g'): confusion_ious(confusion)
            for horizon, confusion in zip(horizons, hist)},
        'scope_ious': {name: confusion_ious(confusion)
                       for name, confusion in scopes.items()
                       if not name.startswith('horizon/')},
        'absent_class_policy': ABSENT_CLASS_POLICY,
    }


def _flat_metrics(hist, horizons):
    flat = {}
    for scope, confusion in _scoped_confusions(hist, horizons).items():
        metrics = confusion_ious(confusion)
        flat[scope + '/mIoU'] = metrics['mIoU']
        for index, value in enumerate(metrics['per_class_iou']):
            flat[scope + '/class_{}_iou'.format(index)] = value
    return flat


def _scene_totals(records, scene_tokens):
    grouped = {}
    for record in records:
        scene = record['scene_token']
        if scene not in grouped:
            grouped[scene] = np.zeros_like(record['hist_by_horizon'])
        grouped[scene] += record['hist_by_horizon']
    return np.stack([grouped[scene] for scene in scene_tokens])


def _validate_pair(first, second):
    first_map = {record['sample_token']: record for record in first}
    second_map = {record['sample_token']: record for record in second}
    if first_map.keys() != second_map.keys():
        raise ValueError('paired comparison requires identical sample tokens')
    for token, record in first_map.items():
        other = second_map[token]
        if (record['scene_token'] != other['scene_token']
                or record['horizon_seconds'] != other['horizon_seconds']
                or record['hist_by_horizon'].shape != other['hist_by_horizon'].shape):
            raise ValueError('paired protocol mismatch for sample {}'.format(token))


def _delta(first, second):
    return {name: (None if value is None or second[name] is None
                   else value - second[name]) for name, value in first.items()}


def _intervals(draws, point):
    result = {}
    for name, values in draws.items():
        finite = [value for value in values
                  if value is not None and math.isfinite(value)]
        bounds = (np.percentile(finite, [2.5, 97.5]).tolist()
                  if finite else [None, None])
        result[name] = {'estimate': point[name], 'ci95': bounds,
                        'defined_replicates': len(finite),
                        'total_replicates': len(values)}
    return result


def summarize_records(records, bootstrap=2000, seed=0, compare_records=None,
                      expected_samples=None):
    """Percentile scene-cluster bootstrap, optionally with paired differences.

    A bootstrap draw samples S complete scenes with replacement from S scenes.
    All frames of a selected scene stay together. Every IoU is recomputed from
    the summed confusion matrices, rather than averaging scene or frame IoUs.
    Paired differences are ``input - comparison`` on identical scene draws.
    These intervals quantify scene sampling variation, not training-seed noise.
    """
    bootstrap = _positive_integer(bootstrap, 'bootstrap')
    seed = _positive_integer(seed, 'seed', allow_zero=True)
    aggregate = aggregate_records(records, expected_samples=expected_samples)
    unique = aggregate['occ_records']
    scene_tokens = sorted({record['scene_token'] for record in unique})
    horizons = aggregate['horizon_seconds']
    scenes = _scene_totals(unique, scene_tokens)
    point = _flat_metrics(scenes.sum(axis=0), horizons)
    comparison = None
    compare_point = None
    if compare_records is not None:
        compare_aggregate = aggregate_records(
            compare_records, expected_samples=len(unique))
        _validate_pair(unique, compare_aggregate['occ_records'])
        comparison = _scene_totals(compare_aggregate['occ_records'], scene_tokens)
        compare_point = _flat_metrics(comparison.sum(axis=0), horizons)
    samples = {name: [] for name in point}
    comparison_samples = {name: [] for name in point}
    difference_samples = {name: [] for name in point}
    random = np.random.RandomState(seed)
    for _ in range(bootstrap):
        selected = random.randint(0, len(scene_tokens), size=len(scene_tokens))
        metrics = _flat_metrics(scenes[selected].sum(axis=0), horizons)
        for name, value in metrics.items():
            samples[name].append(value)
        if comparison is not None:
            other = _flat_metrics(comparison[selected].sum(axis=0), horizons)
            differences = _delta(metrics, other)
            for name in point:
                comparison_samples[name].append(other[name])
                difference_samples[name].append(differences[name])
    report = {
        'schema': 'm3-occupancy-scene-bootstrap-v1',
        'num_unique_samples': len(unique), 'num_scenes': len(scene_tokens),
        'scene_tokens': scene_tokens, 'horizon_seconds': horizons,
        'num_classes': int(scenes.shape[-1]),
        'absent_class_policy': ABSENT_CLASS_POLICY,
        'iou_units': 'fraction_0_to_1',
        'bootstrap': {'replicates': bootstrap, 'seed': seed,
                      'unit': 'scene', 'interval': 'percentile_95',
                      'missing_metric_replicates': 'excluded_and_counted',
                      'scope': 'scene_sampling_not_training_seed_variability'},
        'metrics': _intervals(samples, point),
    }
    if comparison is not None:
        report['comparison_metrics'] = _intervals(comparison_samples, compare_point)
        report['paired_difference'] = {
            'direction': 'input_minus_comparison',
            'metrics': _intervals(difference_samples, _delta(point, compare_point)),
        }
    return report


def json_safe(value):
    """Convert NumPy records and undefined floating metrics to strict JSON."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
