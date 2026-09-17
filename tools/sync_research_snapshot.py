"""Copy an allowlisted research checkpoint; does not commit, push, or run jobs.

Usage: python tools/sync_research_snapshot.py --workspace /path/to/dropple
Large/raw experiment assets and site-specific launch/connection files stay local.
The manifest records byte-exact source hashes. Existing unlisted files are kept.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    repo = Path(__file__).resolve().parents[1]
    research = repo / 'research'
    research.mkdir(exist_ok=True)
    selected = set()
    excluded = []
    for name in ('m0_improvement_20260915', 'o_motion_20260915'):
        root = workspace / 'analysis' / name
        for p in root.iterdir():
            if not p.is_file() or p.suffix not in ('.py', '.md', '.json'):
                continue
            if (p.stat().st_size > 250_000 or
                any(s in p.name for s in ('dispatch', 'remote_readonly', 'wait_gpu',
                                         'wait_dependency', 'watch_', 'current_research_state',
                                         'metadata_evidence', 'repository_snapshot',
                                         'launch_request', 'authorization'))):
                excluded.append(str(p.relative_to(workspace)))
                continue
            selected.add(p)
        selected.update((root / 'research_notes').glob('*.md'))
        # Aggregate evaluation results only, not raw points, targets, or samples.
        for directory in root.glob('*_evaluation_v1'):
            for filename in ('complete.json', 'summary.csv', 'rules.json',
                             'independent_score_audit.json'):
                if (directory / filename).is_file():
                    selected.add(directory / filename)
    p = workspace / 'analysis/m0_improvement_20260915/server_results'
    for campaign in ('campaign_objective_v1', 'campaign_objective_joint_full_v2'):
        for filename in ('complete.json', 'metrics.csv', 'contrasts.csv'):
            selected.add(p / campaign / 'summary_v1' / filename)
    p = workspace / 'analysis/o_motion_20260915/server_results/diagnostics/dense_history_inputs_train16_v1'
    for filename in ('complete.json', 'summary.json'):
        if (p / filename).is_file():
            selected.add(p / filename)
    # Explicit, aggregate-only learning-curve artifacts; no arrays or weights.
    for directory in ('dense_state_learning_curve_evaluation_v1', 'dense_material_learning_curve_evaluation_v1'):
        p = workspace / 'analysis/o_motion_20260915' / directory
        for filename in ('learning_curves.csv', 'aggregate_audit.json',
                         'learning_curves.png', 'learning_curves.pdf'):
            if (p / filename).is_file():
                selected.add(p / filename)
    # The actual P2 integration, including its imported module and focused test.
    overlays = (
        'projects/mmdet3d_plugin/bevformer/detectors/drive_occworld.py',
        'projects/mmdet3d_plugin/bevformer/detectors/credible_motion_residual.py',
        'projects/mmdet3d_plugin/datasets/nuscenes_world_dataset_template.py',
        'tests/test_credible_motion_residual.py',
    )
    pairs = [(p, research / p.relative_to(workspace)) for p in sorted(selected)]
    pairs += [(workspace / 'code/Drive-OccWorld-sota-p2' / f, repo / f) for f in overlays]
    secret = re.compile(r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|'
                        r'gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{40,}|'
                        r'sshpass\s+-p')
    records = []
    for src, dest in pairs:
        assert src.is_file() and not src.is_symlink(), src
        data = src.read_bytes()
        if src.suffix in ('.png', '.pdf'):
            assert src.name in ('learning_curves.png', 'learning_curves.pdf')
            assert len(data) < 2_000_000
            assert data.startswith(b'\x89PNG\r\n\x1a\n' if src.suffix == '.png' else b'%PDF-')
            text = data.decode('latin-1')
        else:
            text = data.decode('utf-8')
        assert not secret.search(text), f'Credential-like content: {src.name}'
        if src.suffix == '.py':
            ast.parse(text, filename=src.name)
        elif src.suffix == '.json':
            json.loads(text)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        assert dest.read_bytes() == data
        records.append(dict(source=str(src.relative_to(workspace)),
                            destination=str(dest.relative_to(repo)),
                            bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    manifest = dict(schema=1, source_copy='byte_exact', files=records,
                    excluded_root_files=sorted(excluded),
                    excluded_categories=['datasets', 'checkpoints', 'raw arrays',
                                         'external source mirrors', 'connection configuration',
                                         'job environments', 'site dispatchers', 'runtime logs'],
                    caveat='Historical versions and drafts are retained; README identifies current conclusions. '
                           'Hash references may identify local assets intentionally absent from this archive. '
                           'This is not a self-contained dataset or model release.')
    (research / 'snapshot_manifest.json').write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(files=len(records), bytes=sum(r['bytes'] for r in records))))


if __name__ == '__main__':
    main()
