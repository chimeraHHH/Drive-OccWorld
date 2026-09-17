import os,sys,json,time,hashlib
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch
import torch.nn.functional as F
sys.path.insert(0,sys.argv[1])
import native_state_cache as native
from dense_task_state_v2 import DenseTaskState
native._prepare_repo(sys.argv[2])
from projects.mmdet3d_plugin.bevformer.dense_heads import world_head_v1 as losses
torch.set_num_threads(2);torch.manual_seed(11);torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
free,total=torch.cuda.mem_get_info();assert free>=16*2**30
torch.cuda.set_per_process_memory_fraction(8*2**30/total)
m=DenseTaskState().cuda()
for mod in (m.dynamics,m.content_increment,m.velocity):mod.requires_grad_(False)
x=torch.randn(1,40000,256,device='cuda')
# Sparse synthetic current target exercises positive, empty and ignored blocks.
target=torch.zeros(1,512,512,40,dtype=torch.long);target[:,100:140,200:220,5:12]=1;target[:,0:2]=255
cpu=losses._downsample_occ_target(target.clone());gpu=losses._downsample_occ_target(target.cuda()).cpu();assert torch.equal(cpu,gpu)
records=[]
for rep in range(2):
 m.zero_grad(set_to_none=True);start=time.monotonic();z=m.current(x)
 r=F.interpolate(z.cpu(),size=(256,256,20),mode='trilinear',align_corners=False)
 loss=losses.CE_ssc_loss(r,cpu,torch.tensor([1.,5.]),ignore_index=255)+losses.lovasz_softmax(torch.softmax(r,1),cpu,ignore=255)
 loss.backward();gradient=torch.cat([p.grad.flatten() for p in m.parameters() if p.requires_grad]);assert torch.isfinite(gradient).all()
 records.append(dict(loss=float(loss),gradient_sha256=native._tensor_digest(gradient),gradient_norm=float(gradient.norm()),seconds=time.monotonic()-start))
assert records[0]['loss']==records[1]['loss'] and records[0]['gradient_sha256']==records[1]['gradient_sha256']
print(json.dumps(dict(status='PASS_DETERMINISTIC_CURRENT_CODEC_GRAPH',synthetic_operator_probe=True,optimizer_updates=0,coarse_cpu_gpu_exact=True,records=records,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)))
