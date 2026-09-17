"""Read existing O and reproduce new CPU initial states; no final V/G claims."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
started = time.monotonic()
root = Path('/storage/data/metaiot_data/wangning/RadarFlowOcc_research_20260914/o_motion_20260915')
package = root / 'jobs/object_state_forecast_train_v1/package'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            h.update(block)
    return h.hexdigest()


def digest(state, historical=False):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update((str(tuple(value.shape)) if historical else json.dumps(list(value.shape))).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


assert sha(package/'object_state_forecast_training_protocol_v1.json') == '93bd80fa938739f40d46a13339a888a4bf1e377385e5cf269ce644f60982ccc9'
protocol = json.loads((package/'object_state_forecast_training_protocol_v1.json').read_text())
manifest = json.loads((root/'object_state_forecast_train_v1/manifest.json').read_text())
for name in ('future_state_motion_v1.py', 'object_state_conditioner_v2.py'):
    assert sha(package/name) == protocol['sources_sha256'][name]
sys.path.insert(0, str(package))
import torch
assert not torch.cuda.is_initialized()
torch.set_num_threads(2)
checkpoint = root.parent/'m0_improvement_20260915/campaign_objective_v1/runs/O/latest.pth'
assert sha(checkpoint) == 'ee9d91f58005b4732f5f68ffe90343a299016cc38582514f58b12aca77e0af70'
original = torch.load(str(checkpoint), map_location='cpu', weights_only=False)
head = original['future_pred_head']; metadata = original['metadata']
assert original['arm'] == 'O' and original['update'] == 512 and metadata['seed'] == 11
assert metadata['trainable_parameters'] == 13274016
old_names = metadata['trainable_parameter_names']; prefix = 'future_pred_head.'
assert len(old_names) == len(set(old_names)) == 130 and all(n.startswith(prefix) for n in old_names)
names = [n[len(prefix):] for n in old_names]
assert names == manifest['future_head_parameter_names'] and set(names) <= set(head)
assert all(t.device.type == 'cpu' and t.dtype == torch.float32 and bool(torch.isfinite(t).all()) for t in head.values())
assert digest(head) == manifest['initial_future_head_sha256']
assert digest(head, True) == '1816cc040b3068b6e5ed3591aec6b8b919d2b3bd8b54b2186e4311ed2fa284fb'
from future_state_motion_v1 import FutureStateMotionReadout
from object_state_conditioner_v2 import ObjectStateConditioner
with torch.random.fork_rng(devices=[]):
    torch.random.default_generator.manual_seed(11)
    readout = FutureStateMotionReadout()
    conditioner = ObjectStateConditioner()
assert digest(readout.state_dict()) == manifest['initial_readout_sha256']
assert digest(conditioner.state_dict()) == manifest['initial_conditioner_sha256']
assert ['readout.'+n for n,_ in readout.named_parameters()] + ['conditioner.'+n for n,_ in conditioner.named_parameters()] == manifest['readout_parameter_names']
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS_EXISTING_O_AND_FRESH_CPU_SCHEMA_ONLY', O_checkpoint_sha256=sha(checkpoint),
    original_head_tensors=len(head), original_head_parameters=len(names),
    old_name_prefix=prefix, name_mapping_exact=True, initial_head_sha256=digest(head),
    initial_readout_sha256=digest(readout.state_dict()), initial_conditioner_sha256=digest(conditioner.state_dict()),
    torch_version=str(torch.__version__), elapsed_seconds=time.monotonic()-started,
    final_VG_checkpoint_read=False, model_forward_calls=0, optimizer_updates=0, cuda_initialized=False)))
