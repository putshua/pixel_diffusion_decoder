"""Real four-GPU batch capacity and matched fixed-batch LR comparison.

All branches reload the same complete model + Adam checkpoint. This is a
bounded diagnostic; it never overwrites a production checkpoint.
"""
import argparse, hashlib, json, os, time
from pathlib import Path
from pid_video.train import (
    torch, dist, dcp, VideoPiD, MiniMaxH3VideoVAE, SourcePool,
    ModelState, OptimizerState, shard, encode, atomic, mix, loss_x0,
)
from pid_video.optimization import regroup_optimizer, next_batch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-run',required=True);p.add_argument('--output',required=True)
    p.add_argument('--micro-batch',type=int,required=True,choices=[1,2,4])
    p.add_argument('--steps',type=int,default=2);p.add_argument('--compare',action='store_true')
    p.add_argument('--global-batch',type=int,default=64)
    a=p.parse_args();run=Path(a.output);run.mkdir(parents=True,exist_ok=True)
    if a.global_batch % (4*a.micro_batch):raise ValueError('Invalid global batch')
    local_clips=a.global_batch//4
    source=Path(a.source_run);raw=(source/'config.json').read_bytes();cfg=json.loads(raw)
    cp=Path(json.loads((source/'latest.json').read_text())['path'])
    complete=json.loads((cp/'complete.json').read_text())
    if complete['contract']!=hashlib.sha256(raw).hexdigest():raise ValueError('Source checkpoint contract mismatch')
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
    dist.init_process_group('nccl',device_id=torch.device('cuda',rank))
    if dist.get_world_size()!=4 or complete['world_size']!=4:raise ValueError('Four GPU probe required')
    torch.manual_seed(cfg['seed']+rank)
    rs=torch.load(cp/f'rank{rank}.pt',weights_only=False,map_location='cpu')
    if rs['contract']!=complete['contract'] or rs['step']!=complete['step']:raise ValueError('Rank checkpoint mismatch')
    rows=[json.loads(l) for l in Path(cfg['sources']).read_text().splitlines()]
    pool=SourcePool(rows,rank,4,cfg['ffmpeg'],(run/f'rejections_rank{rank}.jsonl').open('a'),cfg['seed'],windows_per_source=cfg['windows_per_source'])
    pool.cursor=rs['cursor']
    vae=MiniMaxH3VideoVAE.from_pretrained(cfg['vae']).model.eval().requires_grad_(False).cuda()
    vae.encoder_tiling=False;del vae.decoder
    fixed=[];records=[]
    # Identical distinct clips per rank for B=2 and B=4 and both LR arms.
    for _ in range(local_clips):
        x,z,recs=next_batch(pool,cfg,encode,vae,1)
        fixed.append((x.cpu(),z.cpu()));records+=recs
        del x,z
    atomic(run/f'samples_rank{rank}.json',records)
    del rows,pool
    # Retain frozen encoder during the probe, just as in the actual trainer.
    model=VideoPiD();model.add_temporal();model=shard(model.cuda(),rank)
    log=(run/f'rank{rank}.jsonl').open('a');results={}
    arms=['old','new'] if a.compare else ['new']
    for arm in arms:
        model.zero_grad(set_to_none=True)
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['lr'],betas=tuple(cfg['betas']),weight_decay=cfg['weight_decay'],foreach=False)
        dcp.load(dict(model=ModelState(model),optimizer=OptimizerState(model,opt)),checkpoint_id=cp/'state')
        rates=dict(backbone=2e-6,adapter_temporal=2e-5,head=1e-4) if arm=='new' else dict(backbone=2e-6,adapter_temporal=2e-6,head=2e-6)
        regroup_optimizer(model,opt,rates)
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        accum=local_clips//a.micro_batch
        def batch(j):
            items=fixed[j*a.micro_batch:(j+1)*a.micro_batch]
            x=torch.cat([v[0] for v in items]).cuda();z=torch.cat([v[1] for v in items]).cuda()
            # Seed per clip, independent of micro-batch grouping.
            noises=[];ts=[]
            for k in range(a.micro_batch):
                g=torch.Generator(device='cuda').manual_seed(71000+rank*local_clips+j*a.micro_batch+k)
                ts.append(torch.sigmoid(torch.randn(1,device='cuda',generator=g)*cfg['p_std']+cfg['p_mean']))
                noises.append(torch.randn(x[k:k+1].shape,device='cuda',generator=g))
            t=torch.cat(ts);noise=torch.cat(noises)
            return x,z,t,mix(x,noise,t)
        @torch.no_grad()
        def evaluate():
            metrics=torch.zeros(2,device='cuda')
            for j in range(accum):
                x,z,t,xt=batch(j)
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(z,xt,t)
                metrics+=torch.stack([loss_x0(pred,x,t),(pred.float()-x).square().mean()])/accum
            dist.all_reduce(metrics);return (metrics/4).tolist()
        initial=evaluate();steps=[]
        for i in range(a.steps):
            start=time.monotonic();opt.zero_grad(set_to_none=True);metrics=torch.zeros(2,device='cuda')
            for j in range(accum):
                x,z,t,xt=batch(j)
                with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(z,xt,t);loss=loss_x0(pred,x,t)
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                (loss/accum).backward()
                metrics+=torch.stack([loss.detach(),(pred.detach().float()-x).square().mean()])/accum
                del x,z,t,xt,pred,loss
            grad=model.clip_grad_norm_(cfg['grad_clip'])
            if not torch.isfinite(grad):raise ValueError('Nonfinite gradient')
            opt.step();dist.all_reduce(metrics);metrics/=4;torch.cuda.synchronize()
            memory=torch.tensor([torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()],device='cuda',dtype=torch.float64)
            dist.all_reduce(memory,op=dist.ReduceOp.MAX)
            row=dict(arm=arm,step=i+1,rf_loss=metrics[0].item(),x_mse=metrics[1].item(),grad_norm=grad.item(),seconds=time.monotonic()-start,peak_allocated_gib=memory[0].item()/2**30,peak_reserved_gib=memory[1].item()/2**30,micro_batch=a.micro_batch,gradient_accumulation=accum,global_batch=a.global_batch,learning_rates=rates)
            steps.append(row);log.write(json.dumps(row)+'\n');log.flush()
            if rank==0:print(json.dumps(row),flush=True);atomic(run/'status.json',dict(status='running',**row))
        final=evaluate();results[arm]=dict(initial=initial,final=final,steps=steps)
        del opt
    if rank==0:
        if a.compare and results['old']['initial']!=results['new']['initial']:raise ValueError('Comparison initial states differ')
        atomic(run/'result.json',dict(source_checkpoint=str(cp),global_batch=a.global_batch,micro_batch=a.micro_batch,gradient_accumulation=local_clips//a.micro_batch,results=results,scope='fixed training batch; not generalization',frozen_encoder_retained=True))
        atomic(run/'status.json',dict(status='complete'))
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
