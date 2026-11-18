"""Single-node FSDP PiD video adaptation. No automatic platform submission."""
import argparse, hashlib, json, os, signal, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from pid_video.model import VideoPiD,load_spatial_prior
from pid_video.objective import mix,loss_x0,sample_x0,to_rgb
from pid_video.optimization import batch_plan,regroup_optimizer,set_learning_rates,next_batch
from pid_video.temporal import CheckpointBlock,contract_pixels
from pid_video.source_data import SourcePool,decode_pixels
from pid_video.checkpoint import ModelState,OptimizerState
from third_party.minimax_h3_video_vae.minimax_h3_video_vae import MiniMaxH3VideoVAE
STOP=False

def signal_stop(*_):
    global STOP
    STOP=True

def atomic(path,value):
    p=path.with_suffix('.tmp');p.write_text(json.dumps(value,indent=2));p.replace(path)

@torch.no_grad()
def encode(vae,x):
    with torch.autocast('cuda',dtype=torch.bfloat16):
        raw=vae.encode_temporal(x).chunk(2,1)[0]
        return vae.post_quant_conv(raw).detach()



def shard(model,rank):
    opts=dict(device_id=rank,use_orig_params=True,sharding_strategy=ShardingStrategy.FULL_SHARD,
              sync_module_states=True,limit_all_gathers=True)
    for name in ['patch_blocks','pixel_blocks']:
        blocks=getattr(model.net,name)
        setattr(model.net,name,torch.nn.ModuleList([CheckpointBlock(b) for b in blocks]))
    # Root-only FSDP preserves upstream shared AdaLN parameters across blocks.
    return FSDP(model,**opts)


def save(model,opt,pool,run,step,contract):
    rank=dist.get_rank();out=run/'checkpoints'/f'{step:08d}';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():raise ValueError('Refusing checkpoint overwrite')
    dcp.save(dict(model=ModelState(model),optimizer=OptimizerState(model,opt)),checkpoint_id=out/'state')
    torch.save(dict(step=step,contract=contract,cursor=pool.cursor,cpu_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state()),out/f'rank{rank}.pt')
    dist.barrier()
    if rank==0:
        atomic(out/'complete.json',dict(step=step,contract=contract,world_size=dist.get_world_size()))
        atomic(run/'latest.json',dict(path=str(out),step=step))
    dist.barrier()

@torch.no_grad()
def callback(model,z,target,baseline,cfg,run,step,rank):
    import subprocess
    model.eval();g=torch.Generator(device='cuda').manual_seed(90000)
    noise=torch.randn(target.shape,device='cuda',generator=g)
    fixed=[]
    with torch.autocast('cuda',dtype=torch.bfloat16):
        for tv in (.2,.5,.8):
            t=target.new_tensor([tv])
            fixed.append(loss_x0(model(z,mix(target,noise,t),t),target,t).item())
        x=sample_x0(model,z,noise,cfg['sample_steps'])
    if not torch.isfinite(x).all():raise ValueError('Nonfinite callback')
    if rank==0:
        out=run/'callbacks'/f'{step:06d}';out.mkdir(parents=True,exist_ok=True)
        pixels=to_rgb(torch.cat([target,baseline[0],baseline[1],x],dim=-1))
        frames=(pixels[0].permute(1,2,3,0)*255).byte().cpu().numpy()
        cmd=[cfg['ffmpeg'],'-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{frames.shape[2]}x{frames.shape[1]}','-r','24','-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(out/'comparison.mp4')]
        subprocess.run(cmd,input=frames.tobytes(),check=True,timeout=180)
        atomic(out/'metrics.json',dict(step=step,sample_mse=(x-target).square().mean().item(),fixed_rf_loss=sum(fixed)/3,loss_by_t=fixed,columns=['GT','native direct H3 (existing baseline)','native tiled/temporal H3 (same latent)','PiD x0 8-step'],normalization='ImageNet'))
        paths=sorted((run/'callbacks').glob('*/comparison.mp4'))
        (run/'gallery.html').write_text('<meta charset="utf-8"><h1>GT | Native direct H3 | Native tiled/temporal H3 | PiD x0 (8 step)</h1>'+''.join(f'<p>{p.parent.name}</p><video controls width="1200" src="{p.relative_to(run)}"></video>' for p in paths))
    dist.barrier();model.train()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int);a=p.parse_args()
    run=Path(a.run).resolve();raw=(run/'config.json').read_bytes();cfg=json.loads(raw);contract=hashlib.sha256(raw).hexdigest()
    if cfg['bucket']!=[512,320,56] or cfg['latent_noise']!=0:raise ValueError('Initial experiment is clean-latent 320p/56f decode only')
    for path,sha in cfg['code_hashes'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=sha:raise ValueError('Code contract changed: '+path)
    if hashlib.sha256(Path(cfg['sources']).read_bytes()).hexdigest()!=cfg['sources_sha256']:raise ValueError('Manifest changed')
    if cfg.get('prediction')!='x0' or cfg.get('loss')!='rf_velocity_mse_from_x0':raise ValueError('Direct x0 contract required')
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);dist.init_process_group('nccl',device_id=torch.device('cuda',rank))
    world=dist.get_world_size()
    if world!=cfg['world_size']:raise ValueError('World size changed')
    micro,accum=batch_plan(cfg,world)
    torch.manual_seed(cfg['seed']+rank)
    rows=[json.loads(l) for l in Path(cfg['sources']).read_text().splitlines()]
    pool=SourcePool(rows,rank,world,cfg['ffmpeg'],(run/f'rejections_rank{rank}.jsonl').open('a'),cfg['seed'],windows_per_source=cfg['windows_per_source'])
    if (run/'latest.json').exists() and not a.resume:raise ValueError('Use --resume')
    vae=MiniMaxH3VideoVAE.from_pretrained(cfg['vae']).model.eval().requires_grad_(False).cuda()
    vae.encoder_tiling=False;vae.decoder_tiling=True
    vx,sha=decode_pixels(cfg['validation'],cfg['bucket'],cfg['ffmpeg'],cfg['validation']['start_seconds']);vx=vx.cuda()
    vz=encode(vae,vx);target=vx
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        raw_val=vae.encode_temporal(vx).chunk(2,1)[0]
        baseline=(contract_pixels(vae.decoder(vz)).float(),vae.decode_base(raw_val,frame_num=56).float())
    del raw_val
    del vae.decoder,vx;torch.cuda.empty_cache()
    model=VideoPiD()
    # Every rank loads the same verified spatial prior before FSDP synchronization.
    load_report=load_spatial_prior(model,cfg['prior'],cfg['prior_prefix'])
    model.reset_x0_head();model.add_temporal();model=model.cuda()
    if rank==0:atomic(run/'initialization.json',dict(**load_report,output_head='Reinitialized direct x0 head; spatial backbone loaded from prior',parameters=sum(p.numel() for p in model.parameters()),trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),validation_sha256=sha))
    model=shard(model,rank);opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['lr'],betas=tuple(cfg['betas']),weight_decay=cfg['weight_decay'],foreach=False)
    first=0
    if a.resume and cfg.get('learning_rates'):regroup_optimizer(model,opt,cfg['learning_rates'])
    if a.resume:
        cp=Path(json.loads((run/'latest.json').read_text())['path']);complete=json.loads((cp/'complete.json').read_text())
        if complete['contract']!=contract or complete['world_size']!=world:raise ValueError('Checkpoint contract mismatch')
        rs=torch.load(cp/f'rank{rank}.pt',weights_only=False,map_location='cpu')
        if rs['contract']!=contract or rs['step']!=complete['step']:raise ValueError('Rank checkpoint mismatch')
        dcp.load(dict(model=ModelState(model),optimizer=OptimizerState(model,opt)),checkpoint_id=cp/'state')
        first=rs['step'];pool.cursor=rs['cursor'];torch.set_rng_state(rs['cpu_rng']);torch.cuda.set_rng_state(rs['cuda_rng'])
    elif cfg.get('fork_from'):
        cp=Path(cfg['fork_from']);source=Path(cfg['fork_config']).read_bytes()
        source_contract=hashlib.sha256(source).hexdigest();parent=json.loads(source)
        complete=json.loads((cp/'complete.json').read_text())
        if source_contract!=cfg['fork_config_sha256'] or complete['contract']!=source_contract or complete['world_size']!=world:
            raise ValueError('Fork source checkpoint contract mismatch')
        for key in ('bucket','prediction','loss','pixel_normalization','latent_kind','latent_noise','sources_sha256','seed','windows_per_source','prior','prior_prefix','betas','weight_decay'):
            if cfg[key]!=parent[key]:raise ValueError('Unsupported fork contract change: '+key)
        rs=torch.load(cp/f'rank{rank}.pt',weights_only=False,map_location='cpu')
        if rs['contract']!=source_contract or rs['step']!=complete['step']:raise ValueError('Fork rank checkpoint mismatch')
        dcp.load(dict(model=ModelState(model),optimizer=OptimizerState(model,opt)),checkpoint_id=cp/'state')
        first=rs['step'];pool.cursor=rs['cursor'];torch.set_rng_state(rs['cpu_rng']);torch.cuda.set_rng_state(rs['cuda_rng'])
        if cfg.get('learning_rates'):regroup_optimizer(model,opt,cfg['learning_rates'])
        if rank==0:atomic(run/'fork_loaded.json',dict(source=str(cp),step=first,optimizer='Adam moments retained',global_batch=cfg['global_batch'],micro_batch=micro,gradient_accumulation=accum))
    elif cfg.get('learning_rates'):regroup_optimizer(model,opt,cfg['learning_rates'])
    signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
    if first==0:callback(model,vz,target,baseline,cfg,run,0,rank)
    end=min(a.stop_after or cfg['max_steps'],cfg['max_steps'])
    if end<first:raise ValueError('stop-after precedes resumed step')
    logs=(run/f'rank{rank}.jsonl').open('a')
    for step in range(first,end):
        start=time.monotonic();set_learning_rates(opt,cfg,step)
        opt.zero_grad(set_to_none=True)
        metrics=torch.zeros(2,device='cuda');records=[]
        for micro_step in range(accum):
            x,z,batch_records=next_batch(pool,cfg,encode,vae,micro)
            records.extend(dict(micro_step=micro_step,**rec) for rec in batch_records)
            t=torch.sigmoid(torch.randn(micro,device='cuda')*cfg['p_std']+cfg['p_mean'])
            noise=torch.randn_like(x);xt=mix(x,noise,t)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred=model(z,xt,t);loss=loss_x0(pred,x,t)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            # Synchronize each micro-step to keep FSDP gradients sharded.
            # no_sync would retain full gradients and change memory requirements.
            (loss/accum).backward()
            metrics+=torch.stack([loss.detach(),(pred.detach().float()-x).square().mean()])/accum
            del x,z,noise,xt,pred,loss
        norms=torch.zeros(3,device='cuda')
        for name,param in model.named_parameters():
            if param.grad is not None:
                group=1 if 'lq_proj' in name else (2 if any(k in name for k in ['.down.','.conv.','.up.']) and 'patch_blocks' in name else 0)
                norms[group]+=param.grad.detach().float().square().sum()
        dist.all_reduce(norms);norms=norms.sqrt()
        grad=model.clip_grad_norm_(cfg['grad_clip'])
        if not torch.isfinite(grad):raise ValueError('Nonfinite gradient')
        opt.step();dist.all_reduce(metrics);metrics/=world
        torch.cuda.synchronize();row=dict(step=step+1,rf_loss=metrics[0].item(),x_mse=metrics[1].item(),grad_norm=grad.item(),spatial_grad_norm=norms[0].item(),latent_adapter_grad_norm=norms[1].item(),temporal_grad_norm=norms[2].item(),seconds=time.monotonic()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30,cursor=pool.cursor,global_batch=cfg['global_batch'],micro_batch=micro,gradient_accumulation=accum,learning_rates={g.get('name','all'):g['lr'] for g in opt.param_groups})
        logs.write(json.dumps(row)+'\n');logs.flush()
        with (run/f'samples_rank{rank}.jsonl').open('a') as f:
            for record in records:f.write(json.dumps(dict(step=step+1,**record))+'\n')
        if rank==0:print(json.dumps(row),flush=True);atomic(run/'status.json',dict(status='running',**row))
        stop=torch.tensor(int(STOP),device='cuda');dist.all_reduce(stop,op=dist.ReduceOp.MAX)
        stopping=bool(stop.item())
        if not stopping and ((step+1)%cfg['callback_every']==0 or step+1==end):callback(model,vz,target,baseline,cfg,run,step+1,rank)
        if stopping or (step+1)%cfg['checkpoint_every']==0 or step+1==end:save(model,opt,pool,run,step+1,contract)
        if stopping:
            if rank==0:atomic(run/'status.json',dict(status='interrupted_checkpoint_saved',step=step+1))
            dist.destroy_process_group();raise SystemExit(143)
    if rank==0:atomic(run/'status.json',dict(status='complete' if end==cfg['max_steps'] else 'preflight_complete',step=end))
    dist.destroy_process_group()
if __name__=='__main__':main()
