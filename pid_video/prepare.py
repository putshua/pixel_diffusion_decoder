"""Freeze an independent run from explicit data and model asset paths."""
import argparse,hashlib,json,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser()
 for key in ['run','sources','validation-json','vae','prior']:p.add_argument('--'+key,required=True)
 p.add_argument('--ffmpeg',default='ffmpeg');p.add_argument('--world-size',type=int,default=4,choices=[4,8]);p.add_argument('--global-batch',type=int,default=64);p.add_argument('--micro-batch',type=int,default=2)
 p.add_argument('--max-steps',type=int,default=30000);p.add_argument('--seed',type=int,default=20261116)
 a=p.parse_args();r=Path(a.run).resolve()
 if (r/'config.json').exists():raise ValueError('Existing run; use a new directory')
 if a.micro_batch<1 or a.global_batch<1 or a.global_batch%(a.world_size*a.micro_batch):raise ValueError('Invalid batch plan')
 if not 1<=a.max_steps<=30000:raise ValueError('max-steps must be 1..30000')
 ff=shutil.which(a.ffmpeg)
 if not ff:raise FileNotFoundError(a.ffmpeg)
 rows=[json.loads(l) for l in Path(a.sources).read_text().splitlines() if l.strip()]
 val=json.loads(Path(a.validation_json).read_text());val['path']=str(Path(val['path']).expanduser().resolve());val.setdefault('start_seconds',0.)
 for row in rows:
  if not isinstance(row.get('source_id'),str) or not isinstance(row.get('path'),str):raise ValueError('Each source requires source_id and path strings')
  row['path']=str(Path(row['path']).expanduser().resolve())
 if len({x['source_id'] for x in rows})!=len(rows):raise ValueError('Duplicate source_id')
 if any(x['source_id']==val['source_id'] or x['path']==val['path'] for x in rows):raise ValueError('Remove explicitly chosen validation video from training sources')
 if not rows or len(rows)%a.world_size:raise ValueError('Nonempty source count must divide world size; choose the manifest explicitly')
 r.mkdir(parents=True,exist_ok=True);sources=r/'sources.jsonl';sources.write_text(''.join(json.dumps(row)+'\n' for row in rows))
 files=list((ROOT/'pid_video').glob('*.py'))+list((ROOT/'scripts').glob('*.sh'))+list((ROOT/'third_party').rglob('*.py'))
 cfg=dict(bucket=[512,320,56],sr_scale=1,latent_noise=0,latent_kind='H3 untiled encoder posterior mean then post_quant_conv; matches current pixeldecoder',prediction='x0',loss='rf_velocity_mse_from_x0',pixel_normalization='imagenet',caption='learned null token; no text encoder',sources=str(sources),sources_sha256=sha(sources),source_count=len(rows),windows_per_source=3,validation=val,vae=str(Path(a.vae).resolve()),ffmpeg=ff,prior=str(Path(a.prior).resolve()),prior_prefix='net_ema.',world_size=a.world_size,max_steps=a.max_steps,callback_every=200,checkpoint_every=2000,sample_steps=8,lr=2e-6,learning_rates=dict(backbone=2e-6,adapter_temporal=2e-5,head=1e-4),betas=[.9,.95],weight_decay=.01,grad_clip=1.,warmup=200,seed=a.seed,p_mean=-.8,p_std=.8,global_batch=a.global_batch,micro_batch=a.micro_batch,code_hashes={str(f.relative_to(ROOT)):sha(f) for f in files})
 (r/'config.json').write_text(json.dumps(cfg,indent=2));(r/'preparation.json').write_text(json.dumps(dict(created_unix=time.time(),gpu_validation='pending',input_manifest_sha256=sha(Path(a.sources))),indent=2));print(r)
if __name__=='__main__':main()
