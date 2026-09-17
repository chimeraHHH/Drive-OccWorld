"""Convert the authenticated existing CRN dev export exactly once; no inference."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time

BASE = Path(__file__).resolve().parent
BOUND = {
    'crn_state_dev200_v1/complete.json': 'ddeff445a45608955c7b6b5b164d9b70786d95e042c05a1bfce87c39fd0ea226',
    'crn_state_dev200_v1/manifest.json': '152c86fb03ee3695358fc7ad4d5731f6b21a77db560af6815329535d7c44c275',
    'crn_state_dev200_v1/predictions.json': 'b673ebbce36d7fa101a0c085b7e0115fec11a5e4bd2874bb4d28d4448a2d647e',
    'crn_box_origin_adapter_v1.py': 'aa05e4cd8d2c9ffbf41e751ebc0bda7114040753e404c0560975ee134e23a101',
    'crn_origin_correction_contract_v1.json': 'c4b99aadab4bdd20485c3a544ba9cd735441101f59ec228230a6d79f5d46a51b',
    'receipts/crn_z_origin_runtime_v1.json': '2d9b711857e82ff1c3e88be16064a05d12b03d63f23f1d2545941c5b6fca0149',
}
SELECTION_SHA = '5a940471f03746892dac862054e05590fcf463359a6b16097a1106f4100dde4d'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        stream.write('\n')


def main():
    started = time.monotonic()
    for name, digest in BOUND.items(): assert sha(BASE/name) == digest, name
    selection_path = BASE.with_name('m0_improvement_20260915')/'selection_v1.json'
    assert sha(selection_path) == SELECTION_SHA
    selected = [r for r in read(selection_path)['records'] if r['split'] == 'development']
    source = BASE/'crn_state_dev200_v1'
    original = read(source/'predictions.json'); original_manifest = read(source/'manifest.json')
    assert original['schema'] == 'crn-state-dev200-predictions-v1'
    records = original['records']
    assert len(records) == len(selected) == 200 and len({r['scene_token'] for r in records}) == 100
    assert sum(len(r['boxes']) for r in records) == 94892
    for i, (record, identity) in enumerate(zip(records, selected)):
        assert record['ordinal'] == i and all(record[k] == identity[k] for k in identity)
        assert all(box['sample_token'] == identity['sample_token'] for box in record['boxes'])
    spec = importlib.util.spec_from_file_location('_verified_crn_center_adapter', BASE/'crn_box_origin_adapter_v1.py')
    adapter = importlib.util.module_from_spec(spec); spec.loader.exec_module(adapter)
    centered = [adapter.adapt_record(record) for record in records]
    for old, new in zip(records, centered):
        assert {k:v for k,v in old.items() if k != 'boxes'} == {k:v for k,v in new.items() if k != 'boxes'}
        assert len(old['boxes']) == len(new['boxes'])
        for a,b in zip(old['boxes'],new['boxes']):
            assert {k:v for k,v in a.items() if k != 'translation'} == {k:v for k,v in b.items() if k != 'translation'}
    out = BASE/'crn_state_dev200_centered_v1'; out.mkdir(exist_ok=False)
    write(out/'predictions.json', dict(schema='crn-state-dev200-centered-predictions-v1',
        meta=original['meta'], box_origin='global_geometric_center',
        raw_export_origin='global_transformed_ego_bottom_center',
        center_conversion='translation + R(export_quaternion) @ [0,0,height/2]', records=centered))
    manifest = dict(schema='crn-state-dev200-centering-manifest-v1', selection_sha256=SELECTION_SHA,
        raw_extraction={name+'_sha256':BOUND['crn_state_dev200_v1/'+name+'.json'] for name in ('complete','manifest','predictions')},
        checkpoint_sha256=original_manifest['checkpoint_sha256'],official_commit=original_manifest['commit'],
        origin_adapter_sha256=BOUND['crn_box_origin_adapter_v1.py'],
        origin_contract_sha256=BOUND['crn_origin_correction_contract_v1.json'],
        origin_proof_sha256=BOUND['receipts/crn_z_origin_runtime_v1.json'],
        selected_samples=200,selected_scenes=100,selected_boxes=94892,
        all_fields_except_translation_exact=True, original_order_preserved=True,
        origin_applied_once=True, no_class_or_score_filtering=True,
        new_detector_inference=False,optimizer_updates=0,GPU_used=False,
        source_sha256=sha(__file__),elapsed_seconds=time.monotonic()-started)
    write(out/'manifest.json',manifest)
    for name, digest in BOUND.items(): assert sha(BASE/name) == digest, 'Source changed: '+name
    complete = dict(schema='crn-state-dev200-centering-complete-v1',status='COMPLETE_CRN_STATE_DEV200_CENTERING',
        files_sha256={name:sha(out/name) for name in ('manifest.json','predictions.json')},
        source_sha256=sha(__file__),selected_samples=200,selected_boxes=94892,
        detector_inference_performed=False,optimizer_updates=0)
    write(out/'complete.json',complete)
    print(json.dumps(dict(complete_sha256=sha(out/'complete.json'),**complete)))


if __name__ == '__main__': main()
