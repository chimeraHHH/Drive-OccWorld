"""Read-only dependency gate for diagnostics after the fixed full evaluation.

Does not inspect performance during evaluation, choose models, launch jobs,
restart processes, or modify the dependency. Consumers must enforce their own
resource and GPU-ownership gates after this returns.
"""
import hashlib
import json
import math
from pathlib import Path
import time

EVALUATION_SHA = 'b1b1b046b5acf9dc1207a19e0b7b740d6db38a0982a3794f670fa06862189f28'
AUTHORIZATION_SHA = '81cd07a6155ff232f98698e5397243b84fac6e8c3d2a61681181e618f4c27980'
SELECTION_SHA = '60811c8a0847eaf38caeac489eb5dafc3f34cf90b402e640a182356a620eb391'
MERGER_SHA = '0394ff12140891f70f38527f290b7f71e3c6b52bfe76bde80002620c20ac003b'
WRAPPER_SHA = '837bafed3419c0c187ea4827a368ce7567c4e02cd7143e8a52550a84b7dd90be'
MODELS = ['M0', 'M0_fp32', 'native1', 'O']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path, expected=None):
    path = Path(path)
    data = path.read_bytes()
    require(expected is None or hashlib.sha256(data).hexdigest() == expected,
            'Dependency file hash differs: '+str(path))
    return json.loads(data)


def process_live(identity):
    path = Path('/proc')/str(identity['pid'])
    try:
        stat = (path/'stat').read_text()
        fields = stat[stat.rfind(')')+2:].split()
        return (path.stat().st_uid == identity['uid'] and
                int(fields[19]) == identity['startticks'] and fields[0] != 'Z')
    except FileNotFoundError:
        return False


def completed_receipt(campaign):
    """Called only after final campaign completion exists; all outputs are final."""
    root = Path(campaign)
    require(not (root/'failed.json').exists(), 'Full evaluation failed; no diagnostic continuation')
    done = read(root/'complete.json')
    require(done['status'] == 'COMPLETE_OBJECTIVE_JOINT_CAMPAIGN' and
            done['mode'] == 'full' and done['optimizer_steps'] == 0, 'Wrong full completion')
    manifest = read(root/'manifest.json', done['manifest_sha256'])
    require(manifest['schema'] == 'objective-joint-two-gpu-campaign-v2' and
            manifest['evaluation_protocol_sha256'] == EVALUATION_SHA and
            manifest['full_authorization_sha256'] == AUTHORIZATION_SHA and
            manifest['wrapper_sha256'] == WRAPPER_SHA and
            manifest['models'] == MODELS, 'Wrong full campaign binding')
    directory = root/'summary_v1'
    merged = read(directory/'complete.json', done['summary_complete_sha256'])
    require(merged['schema'] == 'objective-joint-merge-complete-v2' and
            merged['status'] == 'COMPLETE' and merged['mode'] == 'full' and
            merged['samples'] == 5119 and merged['scenes'] == 150 and
            merged['evaluation_protocol_sha256'] == EVALUATION_SHA, 'Full merge incomplete')
    files = merged['files_sha256']
    require(set(files) == {'summary.json', 'report.md', 'sample_hash_ledger.json',
            'development_disclosure.json', 'raw_confusions.npz', 'metrics.csv', 'contrasts.csv'},
            'Unexpected full merged artifacts')
    for filename, digest in files.items():
        require(Path(filename).name == filename and sha(directory/filename) == digest,
                'Full merged artifact changed: '+filename)
    require(merged['summary_sha256'] == files['summary.json'], 'Summary hash chain differs')
    summary = read(directory/'summary.json', merged['summary_sha256'])
    require(summary['schema'] == 'objective-joint-full-summary-v2' and
            summary['status'] == 'COMPLETE_FULL_NATIVE_COMPARISON' and
            summary['samples'] == 5119 and summary['scenes'] == 150 and
            summary['model_names'] == MODELS and summary['candidate_names'] == ['O'] and
            summary['merger_sha256'] == MERGER_SHA and summary['full_validation'] is True,
            'Wrong full summary scope')
    sources = summary['sources']
    require(sources['evaluation_protocol_sha256'] == EVALUATION_SHA and
            sources['full_authorization_sha256'] == AUTHORIZATION_SHA and
            sources['selection_sha256'] == SELECTION_SHA, 'Full summary protocol differs')
    identities = summary['ordered_identities']
    require(len(identities) == 5119 and len({x['sample_token'] for x in identities}) == 5119 and
            len({x['scene_token'] for x in identities}) == 150 and
            sorted(x['official_index'] for x in identities) == list(range(5119)),
            'Full identities incomplete')
    return dict(campaign=str(root), campaign_complete_sha256=sha(root/'complete.json'),
                merge_complete_sha256=sha(directory/'complete.json'),
                summary_sha256=merged['summary_sha256'], samples=5119, scenes=150,
                selected_from_partial_performance=False, performance_gate_applied=False,
                interpretation='Integrity/completion gate only; no success or improvement decision.')


def job_binding(campaign, job):
    root, job = Path(campaign).resolve(), Path(job).resolve()
    require(root.name == 'campaign_objective_joint_full_v2' and
            job == root.parent/'jobs/objective_joint_full_v2', 'Wrong dependency job path')
    launch, spec = read(job/'launch.json'), read(job/'spec.json')
    require(launch['name'] == 'objective_joint_full_v2' and launch['uid'] == 1009 and
            Path(launch['job']).resolve() == job and launch['command'] == spec['command'] and
            spec['seconds'] == 22800, 'Dependency launch/spec differs')
    package = job/'package'
    command = spec['command']
    require(command[:3] == ['/storage/data/metaiot_data/wangning/RadarFlowOcc_m3_h200/envs/hym_driveocc_m3/bin/python',
            '-B', str(package/'run_objective_joint_campaign_v2.py')], 'Wrong full command')
    expected_options = {'--root': str(root), '--mode': 'full',
        '--evaluation-protocol': str(package/'objective_joint_evaluation_protocol_v2.json'),
        '--objective-protocol': str(package/'objective_supervision_protocol_v1.json'),
        '--protocol': str(package/'protocol_v2.json'),
        '--full-authorization': str(package/'objective_joint_full_authorization_v2.json'),
        '--repo': '/home/wangning/Workspace/RadarFlowOcc-sota-p2',
        '--config': '/storage/data/metaiot_data/wangning/RadarFlowOcc_sota_p2_20260911/runtime/configs/S0.py',
        '--checkpoint': '/storage/data/metaiot_data/wangning/RadarFlowOcc_sota_p2_20260911/assets/m0_epoch24.pth'}
    require(len(command) == 3+2*len(expected_options), 'Unexpected full command arguments')
    for option, expected in expected_options.items():
        require(command.count(option) == 1 and command[command.index(option)+1] == expected,
                'Wrong dependency command option: '+option)
    required_sources = {
        'run_objective_joint_campaign_v2.py': WRAPPER_SHA,
        'objective_joint_evaluation_protocol_v2.json': EVALUATION_SHA,
        'objective_joint_full_authorization_v2.json': AUTHORIZATION_SHA,
        'objective_joint_merge_v2.py': MERGER_SHA,
        'objective_joint_evaluation_v2.py': 'f8e3da3de084f5cde3d35b9574124d9684b7ecf77387bd03cf5828c02f7d88ca',
        'objective_supervision_protocol_v1.json': '275b2551775b169471cb230a1416d0454c0560d229c4fe9b7ec5166cfc4c460d',
        'protocol_v2.json': '071547971498579a5f695697aacaa2dbf17eecfdd93c066c011227bcac08505a',
        'job_runner.py': '07c26a06723d97c218eb6d6243406bbb6512355588031a079f156e624818d8a8'}
    for filename, digest in required_sources.items():
        require(launch['script_sha256'][filename] == digest and sha(package/filename) == digest,
                'Dependency staged source changed: '+filename)
    state = read(job/'state.json')
    require(set(state['child']) == {'pid', 'uid', 'startticks'} and state['child']['uid'] == launch['uid'],
            'Wrong dependency child identity')
    return dict(job=str(job), launch_sha256=sha(job/'launch.json'), spec_sha256=sha(job/'spec.json'),
                runner_identity=dict(pid=launch['runner_pid'], uid=launch['uid'], startticks=launch['startticks']),
                child_identity=state['child'], command=command)


def wait_for_completed_full(campaign, job, deadline_unix, progress):
    """Wait through runner cleanup; final artifact marker alone cannot release GPU work."""
    require(math.isfinite(deadline_unix) and deadline_unix > time.time(), 'Finite future deadline required')
    root, job = Path(campaign).resolve(), Path(job).resolve()
    binding = job_binding(root, job)
    missing_child_since = None
    while True:
        require(time.time() < deadline_unix, 'Full dependency wait deadline exceeded; no retry')
        require(not (root/'failed.json').exists(), 'Full campaign failed; stop dependent diagnostic')
        state = read(job/'state.json')
        require(state.get('child') == binding['child_identity'], 'Dependency child identity changed; no restart')
        if state['status'] == 'EXITED_ZERO':
            require(state['returncode'] == 0 and (root/'complete.json').is_file(), 'Runner zero without full completion')
            require(sha(job/'launch.json') == binding['launch_sha256'] and
                    sha(job/'spec.json') == binding['spec_sha256'], 'Dependency launch/spec changed')
            receipt = completed_receipt(root)
            # Verify the bound terminal state again after hashing completed files.
            terminal = read(job/'state.json')
            require(terminal == state and not (root/'failed.json').exists(), 'Dependency terminal state changed')
            receipt.update(job_binding=binding, job_state_sha256=sha(job/'state.json'),
                           runner_status='EXITED_ZERO', runner_returncode=0)
            return receipt
        require(state['status'] == 'RUNNING', 'Full runner stopped or failed; no diagnostic continuation')
        child_live = process_live(binding['child_identity'])
        runner_live = process_live(binding['runner_identity'])
        if not child_live:
            missing_child_since = missing_child_since or time.monotonic()
            # A just-exited child may precede its runner's final cleanup receipt.
            newer = read(job/'state.json')
            if newer['status'] != 'RUNNING':
                continue
            require(runner_live and time.monotonic()-missing_child_since <= 45,
                    'Exact dependency child absent beyond finalization grace; no restart')
        else:
            missing_child_since = None
            require(runner_live, 'Dependency runner is absent; no diagnostic continuation')
        progress(dict(status='WAITING_FOR_FULL', dependency_job=str(job),
                      exact_child_live=child_live, exact_runner_live=runner_live,
                      finalization_grace=not child_live, dependency_child=binding['child_identity'],
                      partial_performance_read=False, deadline_unix=deadline_unix))
        time.sleep(min(20, max(.1, deadline_unix-time.time())))
