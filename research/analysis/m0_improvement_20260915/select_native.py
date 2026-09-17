"""Freeze identities before candidate outcomes; no image/label/score selection."""
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose(rows, scenes, per_scene, salt):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['scene_token'], []).append(row)
    scene_order = sorted(grouped, key=lambda s: (hashlib.sha256((salt + s).encode()).hexdigest(), s))[:scenes]
    assert len(scene_order) == scenes
    result = []
    for scene in scene_order:
        rr = sorted(grouped[scene], key=lambda r: r['official_index'])
        assert len(rr) >= per_scene
        # Interior, uniformly spaced positions; avoid selecting only scene edges.
        positions = [int((i + 0.5) * len(rr) / per_scene) for i in range(per_scene)]
        assert len(set(positions)) == per_scene
        result.extend(rr[i] for i in positions)
    return result


def main():
    catalog_path = ROOT / 'server_results/catalog_v1/catalog.json'
    catalog = json.loads(catalog_path.read_text())
    train = choose(catalog['splits']['train']['records'], 256, 2, 'm0-native-state-v1:train:')
    dev_all = [r for r in catalog['splits']['validation']['records'] if r['split'] == 'development']
    dev = choose(dev_all, 100, 2, 'm0-native-state-v1:dev:')
    assert not {r['scene_token'] for r in train} & {r['scene_token'] for r in dev}
    selection = dict(schema='native-state-selection-v1', catalog_sha256=digest(catalog_path),
                     rule='SHA ordered scenes, two evenly spaced interior usable positions per scene; identities only',
                     records=train + dev, training_scenes=256, development_scenes=100,
                     historical_val_exposure=True, locked_scenes_used=False,
                     interpretation='Development architecture screen, not full validation or independent test evidence')
    path = ROOT / 'selection_v1.json'
    assert not path.exists(), 'Do not replace a frozen selection'
    path.write_text(json.dumps(selection, indent=2) + '\n')
    preflight = dict(selection, records=[train[0], train[2]],
                     interpretation='Two distinct real training scenes for engineering parity and resource profile only')
    (ROOT / 'preflight_selection_v1.json').write_text(json.dumps(preflight, indent=2) + '\n')
    print(json.dumps(dict(train_samples=len(train), development_samples=len(dev), selection_sha256=digest(path))))


if __name__ == '__main__':
    main()
