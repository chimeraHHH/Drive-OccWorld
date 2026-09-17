"""Read-only streaming extraction from a completed official CRN JSON export.

No model, framework, GPU, score threshold, class filter, or GT matching. All
selected prediction dictionaries and their original list order are retained.
The entire source export is parsed and SHA-256 checked, not merely selected
tokens. Output records follow the independently frozen development selection.
"""
import argparse
import codecs
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time
import traceback

SELECTION_SHA='5a940471f03746892dac862054e05590fcf463359a6b16097a1106f4100dde4d'
EXPORT_SHA='fec6541461bf6eb0ad5f3e113e4fcb13c937896731203ef5de003a6ac0055653'
EXPORT_BYTES=1309171876
COMMIT='5e9d2fa2f91c714b297e75ca666d27fc4ad0d13d'
CHECKPOINT_SHA='f725aafc7f033f484f13704134178f2377b3e12bb7e85207b447f8a3ebfa0cb3'
SOURCE_FILES={
    'CRN/attempt01_full/manifest.json':'1cd1a86b3dbfe829c578385f88be4becc72fd40655797285bd66174671e1b58e',
    'CRN/attempt01_full/complete.json':'1a38d5fcdc69da0e28dce476d3d3a9c39717a547f5dc5ac254aafc02f4178102',
    'CRN/attempt01_full/completion_recovery_audit.json':'9493056d3a9c9abeb5273ae1809a00dbacf48cb7aa3c2800ad1bb3b4a431ee76',
    'official_assets/CRN_r50_256x704_128x128_4key.receipt.json':'31effaf066ad2acf05a9b106cc49cb2120e4ae5ae25eb95de6c70bb677eff83a',
}


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def snapshot(path):
    s=Path(path).stat()
    return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,ino=s.st_ino,dev=s.st_dev)


def write(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(value,f,ensure_ascii=False,allow_nan=False,separators=(',',':'))
        f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def reject_constant(value):raise ValueError('Nonfinite JSON constant '+value)


def unique_object(pairs):
    result={}
    for key,value in pairs:
        require(key not in result,'Duplicate nested JSON key');result[key]=value
    return result


def finite_tree(value):
    if isinstance(value,float):require(math.isfinite(value),'Nonfinite box value')
    elif isinstance(value,dict):
        for v in value.values():finite_tree(v)
    elif isinstance(value,list):
        for v in value:finite_tree(v)


class Stream:
    def __init__(self,path,check):
        self.handle=Path(path).open('rb');self.buffer='';self.digest=hashlib.sha256();self.bytes=0
        self.utf8=codecs.getincrementaldecoder('utf-8')();self.eof=False;self.check=check
        self.decoder=json.JSONDecoder(parse_constant=reject_constant,object_pairs_hook=unique_object)

    def more(self):
        self.check();chunk=self.handle.read(1024*1024)
        if not chunk:
            if not self.eof:self.buffer+=self.utf8.decode(b'',final=True)
            self.eof=True;return False
        self.bytes+=len(chunk);self.digest.update(chunk);self.buffer+=self.utf8.decode(chunk);return True

    def whitespace(self):
        self.buffer=self.buffer.lstrip()
        while not self.buffer and self.more():self.buffer=self.buffer.lstrip()

    def char(self,value):
        self.whitespace();require(self.buffer.startswith(value),'Unexpected JSON delimiter')
        self.buffer=self.buffer[len(value):]

    def value(self):
        self.whitespace()
        while True:
            try:
                value,end=self.decoder.raw_decode(self.buffer);self.buffer=self.buffer[end:];return value
            except json.JSONDecodeError:
                if not self.more():raise


def run(a,out,started):
    def check():require(time.monotonic()-started<a.max_seconds,'CPU extraction deadline')
    require(math.isfinite(a.max_seconds) and 0<a.max_seconds<=570,'Bounded CPU deadline required')
    require(sha(a.selection)==SELECTION_SHA,'Frozen selection changed')
    selection=json.loads(Path(a.selection).read_text())
    selected=[r for r in selection['records'] if r['split']=='development']
    require(len(selected)==200 and len({r['sample_token'] for r in selected})==200 and
        len({r['scene_token'] for r in selected})==100,'Require fixed 200 anchors/100 scenes')
    selected_tokens={r['sample_token'] for r in selected};root=Path(a.source_root).resolve()
    receipts={}
    for name,digest in SOURCE_FILES.items():
        path=root/name;require(sha(path)==digest,'Source provenance changed: '+name)
        receipts[name]=json.loads(path.read_text())
    source_manifest=receipts['CRN/attempt01_full/manifest.json']
    complete=receipts['CRN/attempt01_full/complete.json'];recovery=receipts['CRN/attempt01_full/completion_recovery_audit.json']
    asset=receipts['official_assets/CRN_r50_256x704_128x128_4key.receipt.json']
    expected=set(source_manifest['expected_tokens'])
    require(len(expected)==len(source_manifest['expected_tokens'])==source_manifest['samples']==6019 and selected_tokens<=expected,
        'Source expected-token contract differs')
    require(source_manifest['model']==complete['model']=='CRN' and source_manifest['limit'] is None
        and complete['status']=='COMPLETE' and complete['samples']==6019 and not source_manifest['training']
        and not complete['training'] and source_manifest['commit']==complete['commit']==asset['commit']==COMMIT,
        'Source completion/checkpoint commit differs')
    require(recovery['export_sha256']==EXPORT_SHA and recovery['samples']==6019 and recovery['boxes']==2826237
        and asset['local_sha256']==CHECKPOINT_SHA,'Recovered source/export asset differs')
    checkpoint=Path(source_manifest['checkpoint']);require(str(checkpoint)==complete['checkpoint'],'Checkpoint paths differ')
    checkpoint_before=snapshot(checkpoint);require(sha(checkpoint)==CHECKPOINT_SHA,'Current source checkpoint bytes differ')
    export=root/'CRN/attempt01_full/results_nusc.json';before=snapshot(export)
    require(before['size']==EXPORT_BYTES,'Source export size differs')
    stream=Stream(export,check);seen=set();found={};boxes=0;root_keys=set();extras={}
    stream.char('{')
    try:
        while True:
            key=stream.value();require(isinstance(key,str) and key not in root_keys,'Duplicate root key')
            root_keys.add(key);stream.char(':')
            if key!='results':extras[key]=stream.value()
            else:
                stream.char('{')
                while True:
                    token=stream.value();require(isinstance(token,str) and token in expected and token not in seen,'Unknown/duplicate sample token')
                    seen.add(token);stream.char(':');rows=stream.value();require(isinstance(rows,list),'Prediction list required')
                    for row in rows:
                        require(isinstance(row,dict) and row['sample_token']==token,'Box/token identity differs')
                        for field,size in (('translation',3),('size',3),('rotation',4),('velocity',2)):
                            require(isinstance(row[field],list) and len(row[field])==size and
                                all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in row[field]),
                                'Invalid finite prediction vector: '+field)
                        require(isinstance(row['detection_name'],str) and isinstance(row['detection_score'],(float,int))
                            and math.isfinite(row['detection_score']),'Invalid detection class/score')
                        finite_tree(row)
                    boxes+=len(rows)
                    if token in selected_tokens:found[token]=rows
                    if len(seen)%500==0:
                        write(out/'progress.json',dict(tokens=len(seen),boxes=boxes,selected_tokens=len(found),seconds=time.monotonic()-started))
                    stream.whitespace()
                    if stream.buffer.startswith('}'):stream.char('}');break
                    stream.char(',')
            stream.whitespace()
            if stream.buffer.startswith('}'):stream.char('}');break
            stream.char(',')
        stream.whitespace();require(not stream.buffer and not stream.more(),'Trailing JSON data')
    finally:stream.handle.close()
    require(root_keys=={'meta','results'} and seen==expected and boxes==2826237 and set(found)==selected_tokens,
        'Incomplete full export or selected sample coverage')
    require(stream.bytes==EXPORT_BYTES and stream.digest.hexdigest()==EXPORT_SHA,'Full source export SHA differs')
    require(snapshot(export)==before and snapshot(checkpoint)==checkpoint_before,'Source changed during streaming')
    for name,digest in SOURCE_FILES.items():require(sha(root/name)==digest,'Source receipt changed during extraction')
    records=[dict(ordinal=i,**record,boxes=found[record['sample_token']]) for i,record in enumerate(selected)]
    selected_boxes=sum(len(r['boxes']) for r in records)
    predictions=dict(schema='crn-state-dev200-predictions-v1',meta=extras['meta'],records=records)
    write(out/'predictions.json',predictions)
    manifest=dict(schema='crn-state-dev200-extraction-v1',status='COMPLETE',selection_sha256=SELECTION_SHA,
        export_sha256=EXPORT_SHA,export_path=str(export),export_bytes=EXPORT_BYTES,
        export_samples=len(seen),export_boxes=boxes,selected_samples=200,selected_scenes=100,selected_boxes=selected_boxes,
        checkpoint_sha256=CHECKPOINT_SHA,checkpoint_path=str(checkpoint),commit=COMMIT,
        source=dict(files_sha256=SOURCE_FILES,complete=complete,completion_recovery=recovery,checkpoint_receipt=asset,
            original_manifest_sha256=SOURCE_FILES['CRN/attempt01_full/manifest.json'],
            original_precision=source_manifest['precision'],original_seed=source_manifest['seed']),
        export_stat_before=before,export_stat_after=snapshot(export),checkpoint_stat_before=checkpoint_before,
        checkpoint_stat_after=snapshot(checkpoint),source_sha256=sha(__file__),
        selected_records=[dict(ordinal=i,**r,boxes=len(found[r['sample_token']])) for i,r in enumerate(selected)],
        prediction_filtering=False,prediction_order_changed=False,GT_matching_performed=False,
        detector_inference_performed=False,training_performed=False,GPU_used=False,
        full_export_unique_tokens_checked=True,all_export_boxes_finite_checked=True,
        original_numeric_values_and_all_fields_retained=True,host=socket.gethostname(),uid=os.getuid(),pid=os.getpid(),
        finished_utc=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-started)
    write(out/'manifest.json',manifest);check()
    write(out/'complete.json',dict(schema='crn-state-dev200-extraction-complete-v1',status='COMPLETE_CRN_STATE_DEV200_EXTRACTION',
        files_sha256={name:sha(out/name) for name in ('predictions.json','manifest.json')},
        selected_samples=200,selected_boxes=selected_boxes,selection_sha256=SELECTION_SHA,export_sha256=EXPORT_SHA,
        source_sha256=sha(__file__),seconds=time.monotonic()-started))
    print(json.dumps(dict(status='COMPLETE_CRN_STATE_DEV200_EXTRACTION',selected_samples=200,selected_boxes=selected_boxes,
        complete_sha256=sha(out/'complete.json'),seconds=time.monotonic()-started)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',required=True)
    p.add_argument('--selection',required=True);p.add_argument('--out',required=True);p.add_argument('--max-seconds',type=float,default=570.)
    a=p.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    try:run(a,out,started)
    except BaseException as exc:
        write(out/'failed.json',dict(status='FAILED_NO_RETRY',error=repr(exc),traceback=traceback.format_exc(),seconds=time.monotonic()-started));raise


if __name__=='__main__':main()
