"""Fetch pinned official EMA weights, normalize their container, and stage CPU DCP."""
import argparse,hashlib,json
from pathlib import Path
import torch
import torch.distributed.checkpoint as dcp
from huggingface_hub import HfApi,snapshot_download

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def stage(destination, *, download=True):
 r=Path(destination);r.mkdir(parents=True,exist_ok=True);lock=r/'prior_revision.json'
 if lock.exists():revision=json.loads(lock.read_text())['revision']
 else:
  revision='1b9b0872cef786e527ecc5f12b5ab2e6a02d55a4'
  lock.write_text(json.dumps(dict(repo='nvidia/PiD',revision=revision),indent=2))
 base=r/'checkpoints/PixelDiT_finetune_2kto4k';prior=base/'model'
 filename='checkpoints/PixelDiT_finetune_2kto4k/model_ema_bf16.pth'
 if download:snapshot_download('nvidia/PiD',revision=revision,local_dir=str(r),allow_patterns=[filename])
 source=r/filename
 if not source.is_file():raise FileNotFoundError(source)
 source_sha=digest(source)
 if (prior/'asset_manifest.json').exists():
  previous=json.loads((prior/'asset_manifest.json').read_text())
  if previous.get('source_sha256')!=source_sha:raise ValueError('Existing staged prior has a different source')
  return prior
 state=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
 if isinstance(state,dict) and 'state_dict' in state:state=state['state_dict']
 if not isinstance(state,dict) or not state or not all(isinstance(v,torch.Tensor) for v in state.values()):
  raise ValueError('Unexpected official EMA checkpoint structure; inspect explicitly')
 if any(k.startswith('net_ema.') for k in state):state={k:v for k,v in state.items() if k.startswith('net_ema.')}
 elif any(k.startswith('net.') for k in state):state={'net_ema.'+k[4:]:v for k,v in state.items() if k.startswith('net.')}
 else:state={'net_ema.'+k:v for k,v in state.items()}
 dcp.save(state,checkpoint_id=prior)
 manifest={f.name:dict(bytes=f.stat().st_size,sha256=digest(f)) for f in sorted(prior.iterdir()) if f.is_file() and f.name!='asset_manifest.json'}
 (prior/'asset_manifest.json').write_text(json.dumps(dict(revision=revision,source=str(source),source_sha256=source_sha,files=manifest),indent=2))
 return prior

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--destination',required=True);p.add_argument('--stage-only',action='store_true');a=p.parse_args()
 print(stage(a.destination,download=not a.stage_only))
