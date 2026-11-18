"""Explicit synthetic infrastructure test, never a pretrained quality test."""
import os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party/diffusers-sol/src'))
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from pid_video.model import VideoPiD,BASE
from pid_video.train import shard
from pid_video.checkpoint import ModelState,OptimizerState
from pid_video.objective import loss_x0
rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank)
dist.init_process_group('nccl',device_id=torch.device('cuda',rank))
cfg=dict(BASE,hidden_size=64,num_groups=4,pixel_hidden_size=4,pixel_attn_hidden_size=64,pixel_num_groups=4,patch_depth=2,pixel_depth=1,txt_embed_dim=16,txt_max_length=4,lq_hidden_dim=16,lq_num_res_blocks=1)
def make():
 torch.manual_seed(44);m=VideoPiD(cfg,temporal_width=8);m.reset_x0_head();m.add_temporal();m=shard(m.cuda(),rank)
 return m,torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-3)
x=torch.randn(1,3,56,16,16,device='cuda');z=torch.randn(1,24,17,1,1,device='cuda');t=torch.tensor([.3],device='cuda')
def step(m,o):
 o.zero_grad(set_to_none=True)
 with torch.autocast('cuda',dtype=torch.bfloat16):l=loss_x0(m(z,x,t),torch.ones_like(x),t)
 l.backward();m.clip_grad_norm_(1.);o.step();return l.detach()
m,o=make();step(m,o);step(m,o)
path=Path(sys.argv[1]);path.mkdir(parents=True,exist_ok=True)
dcp.save(dict(model=ModelState(m),optimizer=OptimizerState(m,o)),checkpoint_id=path/'state')
expected=step(m,o);weights=[p.detach().clone() for p in m.parameters()]
n,q=make();dcp.load(dict(model=ModelState(n),optimizer=OptimizerState(n,q)),checkpoint_id=path/'state')
actual=step(n,q);torch.testing.assert_close(actual,expected,rtol=0,atol=0)
for p,w in zip(n.parameters(),weights):torch.testing.assert_close(p,w,rtol=0,atol=0)
(path/f'passed_rank{rank}.txt').write_text('actual PiD tiny FSDP + shared params + frozen branches + exact optimizer next-step resume passed')
dist.destroy_process_group()
