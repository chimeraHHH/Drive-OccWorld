#!/usr/bin/env python3
"""Summarize M3 evaluation records with scene bootstrap and optional pairing.

Inputs are JSON or locally produced tools/test.py pickle results containing
``occ_records``; a bare list of records is also accepted. No model is loaded.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import pickle
import tempfile


MODULE_PATH = (Path(__file__).resolve().parents[1] /
               'projects/mmdet3d_plugin/datasets/m3_metrics.py')
SPEC = importlib.util.spec_from_file_location('m3_metrics_standalone', str(MODULE_PATH))
METRICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(METRICS)


def load_records(path):
    path = Path(path)
    if path.suffix.lower() == '.json':
        with path.open('r', encoding='utf-8') as stream:
            value = json.load(stream)
    elif path.suffix.lower() in ('.pkl', '.pickle'):
        with path.open('rb') as stream:
            value = pickle.load(stream)
    else:
        raise ValueError('input must be .json, .pkl or .pickle')
    if isinstance(value, dict):
        if 'occ_records' not in value:
            raise ValueError('evaluation result has no occ_records')
        value = value['occ_records']
    if not isinstance(value, list):
        raise ValueError('occ_records must be a list')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--compare', help='paired baseline; delta is input minus baseline')
    parser.add_argument('--output', required=True)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--expected-samples', type=int)
    args = parser.parse_args()
    report = METRICS.summarize_records(
        load_records(args.input), bootstrap=args.bootstrap, seed=args.seed,
        compare_records=load_records(args.compare) if args.compare else None,
        expected_samples=args.expected_samples)
    report['source'] = {'input': str(Path(args.input).resolve()),
                        'compare': str(Path(args.compare).resolve())
                        if args.compare else None}
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', dir=str(output.parent),
                prefix='.' + output.name + '.', delete=False) as stream:
            temporary = stream.name
            json.dump(METRICS.json_safe(report), stream, indent=2,
                      ensure_ascii=False, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, str(output))
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    print('Saved {} unique samples from {} scenes to {}'.format(
        report['num_unique_samples'], report['num_scenes'], output))


if __name__ == '__main__':
    main()
