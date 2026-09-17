"""Pure adapter for a source-verified CRN bottom-centre export.

Apply only after the actual inference/export chain has established that
translation is the bottom-face centre. This module does not establish that
fact, inspect GT, infer an origin, or score predictions.

Output translation = input translation + R_export[:, 2] * (height / 2).
R_export is the full global box rotation, including ego pitch and roll.
Every input is left untouched; outputs are deep copies with only each box's
translation replaced. Size remains w,l,h; rotation and velocity are retained.
The functions are deliberately not idempotent: adapt raw export boxes once.
This corrects serialized geometry, not bitwise inversion of earlier FP32
subtraction or rotation roundoff.
"""
import argparse
import copy
import hashlib
import json
import math
from numbers import Real
from pathlib import Path


def _finite_vector(value, length, name):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(name + ' must have length ' + str(length))
    if not all(isinstance(x, Real) and not isinstance(x, bool) and math.isfinite(x) for x in value):
        raise ValueError(name + ' must contain finite numbers')
    return [float(x) for x in value]


def _geometric_center(box):
    if not isinstance(box, dict):
        raise ValueError('Expected an exported box dictionary')
    center = _finite_vector(box['translation'], 3, 'translation')
    size = _finite_vector(box['size'], 3, 'size wlh')
    if not all(x > 0 for x in size):
        raise ValueError('Box width, length and height must be positive')
    q = _finite_vector(box['rotation'], 4, 'rotation wxyz')
    scale = max(abs(x) for x in q)
    if scale == 0:
        raise ValueError('Quaternion must be nonzero')
    q = [x / scale for x in q]
    norm = math.sqrt(sum(x*x for x in q))
    w, x, y, z = [v / norm for v in q]
    # Third column of the active box-local-to-global rotation matrix.
    up_global = (2*(x*z + w*y), 2*(y*z - w*x), 1 - 2*(x*x + y*y))
    result = [c + a * (size[2] / 2) for c, a in zip(center, up_global)]
    if not all(math.isfinite(v) for v in result):
        raise ValueError('Nonfinite adapted centre')
    return result


def adapt_box(box):
    """Return a deep copy with bottom-face centre translated to box centre."""
    center = _geometric_center(box)
    result = copy.deepcopy(box)
    result['translation'] = center
    return result


def adapt_record(record):
    """Return a deep-copied extracted record; preserve box order/all other fields."""
    if not isinstance(record, dict) or not isinstance(record.get('boxes'), list):
        raise ValueError('Expected a record with its original boxes list')
    centers = [_geometric_center(box) for box in record['boxes']]
    result = copy.deepcopy(record)
    for box, center in zip(result['boxes'], centers):
        box['translation'] = center
    return result


def self_test():
    base = dict(sample_token='analytic', translation=[10., 20., 30.], size=[2., 8., 4.],
                rotation=[1., 0., 0., 0.], velocity=[3., 5.], detection_name='car',
                detection_score=.7, attribute_name='vehicle.moving', extra={'nested': [1, 2]})
    half = math.sqrt(.5)
    cases = (
        ('identity', [1., 0., 0., 0.], [10., 20., 32.]),
        ('yaw90', [half, 0., 0., half], [10., 20., 32.]),
        ('pitch90', [half, 0., half, 0.], [12., 20., 30.]),
        ('roll90', [half, half, 0., 0.], [10., 18., 30.]),
        ('yaw90_pitch90', [.5, -.5, .5, .5], [10., 22., 30.]),
        ('negative_scaled_pitch90', [-3*half, 0., -3*half, 0.], [12., 20., 30.]),
    )
    checks = []
    for name, rotation, expected in cases:
        raw = copy.deepcopy(base); raw['rotation'] = rotation
        before = copy.deepcopy(raw)
        adapted = adapt_box(raw)
        assert raw == before
        assert all(math.isclose(a, b, rel_tol=0., abs_tol=1e-12) for a, b in zip(adapted['translation'], expected)), name
        assert set(adapted) == set(raw)
        assert all(adapted[k] == raw[k] for k in raw if k != 'translation')
        checks.append(dict(name=name, expected=expected, actual=adapted['translation']))
    record = dict(ordinal=7, sample_token='analytic', metadata={'keep': [1, 2]},
                  boxes=[copy.deepcopy(base), dict(copy.deepcopy(base), rotation=[half, 0., half, 0.])])
    before = copy.deepcopy(record)
    result = adapt_record(record)
    assert record == before and list(result) == list(record)
    assert result['boxes'][0]['translation'] == [10., 20., 32.]
    assert result['metadata'] == record['metadata']
    for raw, adjusted in zip(record['boxes'], result['boxes']):
        assert all(adjusted[k] == raw[k] for k in raw if k != 'translation')
    result['boxes'][0]['velocity'][0] = -99
    result['boxes'][0]['extra']['nested'].append(3)
    result['metadata']['keep'].append(3)
    assert record == before, 'Nested output data aliases original export'
    empty = {'boxes': [], 'meta': {'keep': True}}
    assert adapt_record(empty) == empty and adapt_record(empty) is not empty
    return dict(status='PASS_ANALYTIC_ORIGIN_ADAPTER_CASES', checks=checks,
                input_and_non_translation_fields_preserved=True, nested_copies_independent=True,
                box_order_preserved=True, empty_record_preserved=True,
                analytic_comparison_abs_tol=1e-12, actual_export_origin_verified=False,
                actual_data_read=False, GT_read=False, real_scoring=False, GPU_used=False,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if not args.self_test:
        parser.error('This module only exposes a pure adapter and --self-test; no real-data scoring entry point')
    print(json.dumps(self_test(), indent=2, allow_nan=False))
