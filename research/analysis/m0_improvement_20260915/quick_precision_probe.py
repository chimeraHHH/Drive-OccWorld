"""Engineering-only duplicate-memory precision probe. No optimizer or candidate scores."""
import argparse, copy, json, time
from pathlib import Path
import torch
from native_state_cache import build_native_model, load_sample, replay
from observation_memory import install_observation_memory


def main():
    p=argparse.ArgumentParser()
    for name in ['config','checkpoint','cache','out']:p.add_argument('--'+name,required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.manual_seed(11)
    m=build_native_model(a.config,a.checkpoint,'cuda')
    native=copy.deepcopy(m);double=copy.deepcopy(m);del m
    install_observation_memory(native,'native1');install_observation_memory(double,'persistent2')
    records=json.loads((Path(a.cache)/'index.json').read_text())['records']
    rows=[]
    for matmul,cudnn in [(True,True),(False,False),(True,False),(False,True)]:
        torch.backends.cuda.matmul.allow_tf32=matmul;torch.backends.cudnn.allow_tf32=cudnn
        for record in records[:2]:
            sample=load_sample(a.cache,record,'cuda');begin=time.monotonic()
            x=replay(native,sample,False)[0];y=replay(double,sample,False)[0]
            diff=(y[:2]-x[:2]).abs();tol=1e-4+1e-4*x[:2].abs()
            h0=native.evaluate_occ_records(x,sample['targets'],sample['inputs']['img_metas'])[0]['hist_by_horizon']
            h1=double.evaluate_occ_records(y,sample['targets'],sample['inputs']['img_metas'])[0]['hist_by_horizon']
            row=dict(matmul_tf32=matmul,cudnn_tf32=cudnn,sample_token=record['sample_token'],
                     max_abs=float(diff.max()),fraction_over_original_tolerance=float((diff>tol).float().mean()),
                     first_step_native_hist=h0[1].tolist(),first_step_double_hist=h1[1].tolist(),
                     first_step_hist_delta=(h1[1]-h0[1]).tolist(),seconds=time.monotonic()-begin)
            rows.append(row);print(json.dumps(row),flush=True)
            (out/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
            del x,y,sample,diff,tol
    (out/'complete.json').write_text(json.dumps(dict(status='PRECISION_DIAGNOSIS_ONLY',rows=rows),indent=2)+'\n')


if __name__=='__main__':main()
