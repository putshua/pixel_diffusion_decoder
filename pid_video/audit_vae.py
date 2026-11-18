"""Same-source A/B audit: direct vs temporal vs spatially tiled H3 decoding."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import torch
from third_party.minimax_h3_video_vae.minimax_h3_video_vae import MiniMaxH3VideoVAE
from pid_video.source_data import decode_pixels
from pid_video.temporal import contract_pixels
from pid_video.video_io import write_comparison

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args()
    run=Path(a.run);cfg=json.loads((run/'config.json').read_text());out=run/'vae_audit';out.mkdir(exist_ok=True)
    x,sha=decode_pixels(cfg['validation'],cfg['bucket'],cfg['ffmpeg'],cfg['validation']['start_seconds']);x=x.cuda()
    vae=MiniMaxH3VideoVAE.from_pretrained(cfg['vae']).model.cuda().eval().requires_grad_(False)
    results={'GT':x.cpu()}
    with torch.autocast('cuda',dtype=torch.bfloat16):
        vae.encoder_tiling=False;vae.decoder_tiling=False
        raw=vae.encode_temporal(x).chunk(2,1)[0]
        results['A_current_direct']=contract_pixels(vae.decoder(vae.post_quant_conv(raw))).float().cpu()
        results['B_temporal_untiled']=vae.decode_base(raw,frame_num=56).float().cpu()
        vae.decoder_tiling=True
        results['C_temporal_tiled_same_latent']=vae.decode_base(raw,frame_num=56).float().cpu()
        vae.encoder_tiling=True
        raw_tiled=vae.encode_temporal(x).chunk(2,1)[0]
        results['D_tiled_encode_decode_mean']=vae.decode_base(raw_tiled,frame_num=56).float().cpu()
        # Official encode_base samples posterior. Include seeded sampling as separate factor.
        torch.manual_seed(1984)
        raw_sample=vae.encode_base(x)
        results['E_tiled_encode_decode_sample']=vae.decode_base(raw_sample,frame_num=56).float().cpu()
    stats={}
    for name,v in results.items():
        if v.shape!=results['GT'].shape or not torch.isfinite(v).all():raise ValueError(name)
        dx=(v[:,:,:,:,1:]-v[:,:,:,:,:-1]).abs();dy=(v[:,:,:,1:,:]-v[:,:,:,:-1,:]).abs()
        stats[name]=dict(mse=(v-results['GT']).square().mean().item(),boundary16_x=dx[:,:,:,:,15::16].mean().item(),boundary16_y=dy[:,:,:,15::16,:].mean().item(),all_dx=dx.mean().item(),all_dy=dy.mean().item())
        if name!='GT':
            pair=[results['GT'],v]
            pair=[torch.cat([q[:,:,:1].expand(-1,-1,3,-1,-1),q],2) for q in pair]
            write_comparison(out/f'{name}.mp4',pair,cfg['ffmpeg'])
    (out/'metrics.json').write_text(json.dumps(dict(pixel_sha256=sha,normalization='ImageNet; raw MSE',metrics=stats),indent=2))
    (out/'gallery.html').write_text('<h1>GT | candidate decode</h1>'+''.join(f'<p>{n}</p><video controls width="1024" src="{n}.mp4"></video>' for n in results if n!='GT'))
    print(json.dumps(stats),flush=True)
if __name__=='__main__':main()
