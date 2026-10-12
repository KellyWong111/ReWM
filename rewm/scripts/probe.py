#!/usr/bin/env python3
"""Run frozen layerwise probes and native own-latent prediction diagnostics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',choices=['reacher','tworoom','pusht'],required=True)
    p.add_argument('--architecture',choices=['full','official'],required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--data-h5',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();out=a.output.resolve()
    model_code=ROOT/('src' if a.architecture=='full' else 'controlled/official')
    case=dict(id=f'{a.task}_{a.architecture}',task=a.task,architecture=a.architecture,
              checkpoint=str(a.checkpoint.resolve()),model_code=str(model_code),
              kernel_root=str(ROOT/'third_party/moda_triton'))
    if a.dry_run:
        print(json.dumps(dict(case=case,dataset=str(a.data_h5.resolve()),output=str(out),smoke=a.smoke),indent=2));return
    for f in [a.checkpoint,a.data_h5]:
        if not f.is_file():p.error(f'File not found: {f}')
    if out.exists() and any(out.iterdir()):p.error('Choose a fresh output directory.')
    out.mkdir(parents=True, exist_ok=True);case['checkpoint_sha256']=sha(a.checkpoint)
    (out/'case.json').write_text(json.dumps(case,indent=2)+'\n')
    original=ROOT/'diagnostics/banks'/a.task
    bank=out/'bank';bank.mkdir()
    shutil.copyfile(original/'bank.npz',bank/'bank.npz')
    meta=json.loads((original/'metadata.json').read_text());meta['spec']['h5']=str(a.data_h5.resolve())
    (bank/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    cmd=[sys.executable,str(ROOT/'diagnostics/evaluate_raw.py'),'--case',str(out/'case.json'),
         '--bank',str(bank),'--output',str(out/'results'),'--device',a.device]
    if a.smoke:cmd.append('--smoke')
    env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([str(model_code),str(ROOT/'src'),str(ROOT/'third_party/moda_triton')])
    env['FLA_LIGHT_IMPORT']='1'
    subprocess.run(cmd,check=True,env=env,cwd=ROOT)

if __name__=='__main__':main()
