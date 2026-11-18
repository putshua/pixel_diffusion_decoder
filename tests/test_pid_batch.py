import pytest
import torch
from pid_video.optimization import batch_plan,regroup_optimizer,set_learning_rates
from pid_video.model import VideoPiD,BASE,TemporalBlock


def tiny():
    m=VideoPiD(dict(BASE,hidden_size=64,num_groups=4,pixel_hidden_size=4,
        pixel_attn_hidden_size=64,pixel_num_groups=4,patch_depth=2,pixel_depth=1,
        txt_embed_dim=16,txt_max_length=4,lq_hidden_dim=16,lq_num_res_blocks=1),temporal_width=8)
    m.add_temporal();return m


def test_batch_and_groups_preserve_moments():
    assert batch_plan(dict(micro_batch=4,global_batch=32),4)==(4,2)
    assert batch_plan(dict(micro_batch=2,global_batch=32),4)==(2,4)
    assert batch_plan(dict(micro_batch=2,global_batch=64),4)==(2,8)
    assert batch_plan(dict(micro_batch=4,global_batch=64),4)==(4,4)
    with pytest.raises(ValueError):batch_plan(dict(micro_batch=3,global_batch=32),4)
    m=tiny();opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=2e-6)
    p=m.net.final_layer.linear.weight
    p.grad=torch.ones_like(p);opt.step();moment=opt.state[p]['exp_avg'].clone()
    rates=dict(backbone=2e-6,adapter_temporal=2e-5,head=1e-4)
    regroup_optimizer(m,opt,rates)
    assert torch.equal(moment,opt.state[p]['exp_avg'])
    ids={g['name']:{id(p) for p in g['params']} for g in opt.param_groups}
    assert ids['head']=={id(p) for p in m.net.final_layer.linear.parameters()}
    for module in m.modules():
        if isinstance(module,TemporalBlock):
            assert all(id(p) in ids['adapter_temporal'] for p in module.norm.parameters())
            assert all(id(p) in ids['backbone'] for p in module.spatial.parameters() if p.requires_grad)
    set_learning_rates(opt,dict(warmup=200,lr=2e-6),497)
    assert {g['name']:g['lr'] for g in opt.param_groups}==rates


def test_video_micro_batch_and_accumulation_equivalence():
    from pid_video.objective import loss_x0
    torch.manual_seed(11);m=tiny()
    torch.nn.init.normal_(m.net.final_layer.linear.weight,std=.02)
    x=torch.randn(2,3,56,16,16);z=torch.randn(2,24,17,1,1);t=torch.tensor([.2,.7])
    both=m(z,x,t)
    single=torch.cat([m(z[i:i+1],x[i:i+1],t[i:i+1]) for i in range(2)])
    torch.testing.assert_close(both,single,rtol=1e-4,atol=1e-5)
    loss_x0(both,x,t).backward()
    grads={n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None}
    m.zero_grad(set_to_none=True)
    for i in range(2):(loss_x0(m(z[i:i+1],x[i:i+1],t[i:i+1]),x[i:i+1],t[i:i+1])/2).backward()
    for n,p in m.named_parameters():
        if n in grads:torch.testing.assert_close(grads[n],p.grad,rtol=2e-3,atol=2e-5)


def test_grouped_adam_checkpoint_next_update(tmp_path):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict,set_state_dict,StateDictOptions
    rates=dict(backbone=2e-6,adapter_temporal=2e-5,head=1e-4)
    def make():
        m=tiny();o=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=2e-6)
        regroup_optimizer(m,o,rates);return m,o
    def step(m,o):
        o.zero_grad(set_to_none=True)
        for p in m.parameters():
            if p.requires_grad:p.grad=torch.ones_like(p)*.01
        o.step()
    m,o=make();step(m,o)
    from pid_video.checkpoint import ModelState,OptimizerState
    dcp.save(dict(model=ModelState(m),optimizer=OptimizerState(m,o)),checkpoint_id=tmp_path/'state')
    step(m,o)
    n,q=make()
    dcp.load(dict(model=ModelState(n),optimizer=OptimizerState(n,q)),checkpoint_id=tmp_path/'state')
    step(n,q)
    for p,v in zip(m.parameters(),n.parameters()):torch.testing.assert_close(p,v,rtol=0,atol=0)
    assert {g['name']:g['lr'] for g in q.param_groups}==rates
