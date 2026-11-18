"""Fail explicitly before training if mounted data/assets/code are incomplete."""
import argparse,hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--audit-only',action='store_true');a=p.parse_args()
    r=Path(a.run);cfg=json.loads((r/'config.json').read_text())
    for name,expected in cfg['code_hashes'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise ValueError('Code changed: '+name)
    manifest=Path(cfg['sources'])
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=cfg['sources_sha256']:raise ValueError('Source manifest changed')
    rows=[json.loads(l) for l in manifest.read_text().splitlines()]
    for row in [cfg['validation']]+rows[:8]:
        if not Path(row['path']).is_file():raise FileNotFoundError(row['path'])
    vae=Path(cfg['vae']);vc=json.loads((vae/'config.json').read_text())
    weights=vae/ vc['source_path']/vc['source_safetensors_path']
    if not weights.is_file():raise FileNotFoundError(weights)
    subprocess.run([cfg['ffmpeg'],'-version'],check=True,stdout=subprocess.DEVNULL)
    if not a.audit_only:
        prior=Path(cfg['prior'])
        if not (prior/'.metadata').is_file():raise FileNotFoundError(f'PiD spatial prior not prepared: {prior}; run fetch_prior.py first')
        if not list(prior.glob('*.distcp')):raise FileNotFoundError('No prior DCP shards')
        assets=json.loads((prior/'asset_manifest.json').read_text())
        for name,record in assets['files'].items():
            f=prior/name
            if f.stat().st_size!=record['bytes']:raise ValueError('Prior asset size mismatch: '+name)
            h=hashlib.sha256()
            with f.open('rb') as stream:
                for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
            if h.hexdigest()!=record['sha256']:raise ValueError('Prior asset SHA mismatch: '+name)
        digest=hashlib.sha256((prior/'asset_manifest.json').read_bytes()).hexdigest()
        identity=r/'prior_identity.json'
        if identity.exists() and json.loads(identity.read_text())['sha256']!=digest:raise ValueError('Prior asset identity changed')
        identity.write_text(json.dumps(dict(sha256=digest,revision=assets['revision']),indent=2))
    print('CPU preflight passed; GPU shape/memory/pretrained-load test still required')
if __name__=='__main__':main()
