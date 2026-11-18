"""CPU contracts only; these tests do not validate pretrained video quality."""
import torch
from pid_video.model import VideoPiD,align_latents,TemporalBlock,BASE

def test_h3_alignment():
    z=torch.arange(17.).view(1,1,17,1,1).expand(1,24,17,1,1)
    out=align_latents(z)
    assert out.shape==(1,24,56,1,1)
    assert out[0,0,:,0,0].tolist()==[0]+sum(([i]*4 for i in range(1,5)),[])+[5]+sum(([i]*4 for i in range(6,10)),[])+[10]+sum(([i]*4 for i in range(11,15)),[])+[15]+[16]*4

def test_temporal_zero_start_and_neighbor_coupling():
    class Spatial(torch.nn.Module):
        def forward(self,x,y):return x,y
    block=TemporalBlock(Spatial(),8,frames=4,width=4)
    x=torch.randn(4,2,8,requires_grad=True);y=torch.randn(4,1,8)
    assert torch.equal(block(x,y)[0],x)
    torch.nn.init.normal_(block.up.weight,std=.1)
    grad=torch.autograd.grad(block(x,y)[0][1].sum(),x)[0]
    assert grad[0].abs().sum()>0 and grad[2].abs().sum()>0

def test_real_network_tiny_shape_and_gradients():
    cfg=dict(BASE,hidden_size=64,num_groups=4,pixel_hidden_size=4,pixel_attn_hidden_size=64,
             pixel_num_groups=4,patch_depth=2,pixel_depth=1,txt_embed_dim=16,txt_max_length=4,
             lq_hidden_dim=16,lq_num_res_blocks=1)
    m=VideoPiD(cfg,temporal_width=8)
    # Official fresh final head is zero; simulate nonzero pretrained spatial head
    # solely to test propagation into the adapter (not pretrained quality).
    torch.nn.init.normal_(m.net.final_layer.linear.weight,std=.02)
    m.add_temporal()
    x=torch.randn(1,3,56,16,16);z=torch.randn(1,24,17,1,1);t=torch.tensor([.5])
    out=m(z,x,t);assert out.shape==x.shape
    loss=(out-x).square().mean();loss.backward()
    assert torch.isfinite(loss)
    assert all(p.requires_grad for n,p in m.named_parameters() if not any(k in n for k in ['attn.proj_y','norm_y2','mlp_y']))
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in m.named_parameters() if 'lq_proj' in n)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in m.named_parameters() if 'spatial' in n)

def test_spatial_prior_strict_dcp_roundtrip(tmp_path):
    import torch.distributed.checkpoint as dcp
    from pid_video.model import load_spatial_prior
    cfg=dict(BASE,hidden_size=64,num_groups=4,pixel_hidden_size=4,pixel_attn_hidden_size=64,
             pixel_num_groups=4,patch_depth=2,pixel_depth=1,txt_embed_dim=16,txt_max_length=4,
             lq_hidden_dim=16,lq_num_res_blocks=1)
    m=VideoPiD(cfg)
    source={'net_ema.'+k:v.detach().clone() for k,v in m.net.state_dict().items() if not k.startswith('lq_proj.')}
    dcp.save(source,checkpoint_id=tmp_path/'prior')
    with torch.no_grad():m.net.s_embedder.proj.weight.add_(1)
    load_spatial_prior(m,tmp_path/'prior')
    for k,v in m.net.state_dict().items():
        if not k.startswith('lq_proj.'):assert torch.equal(v,source['net_ema.'+k])
    import pytest
    with pytest.raises(ValueError,match='Incomplete'):load_spatial_prior(m,tmp_path/'prior',prefix='missing.')

def test_x0_objective_matches_current_training_and_sampler():
    from pid_video.objective import mix,loss_x0,sample_x0
    from pid_video.temporal import masked_flow_loss
    torch.manual_seed(7)
    target=torch.randn(2,3,56,2,2);pred=torch.randn_like(target);noise=torch.randn_like(target)
    t=torch.tensor([.2,.99])
    assert torch.equal(loss_x0(pred,target,t),masked_flow_loss(pred,target,t,torch.ones(56,dtype=torch.bool)))
    assert torch.equal(mix(target,noise,torch.zeros(2)),noise)
    assert torch.equal(mix(target,noise,torch.ones(2)),target)
    def oracle(z,x,t):return target
    assert torch.allclose(sample_x0(oracle,None,noise,8),target,atol=1e-6)

def test_direct_x0_head_has_no_velocity_conversion():
    cfg=dict(BASE,hidden_size=64,num_groups=4,pixel_hidden_size=4,pixel_attn_hidden_size=64,
             pixel_num_groups=4,patch_depth=2,pixel_depth=1,txt_embed_dim=16,txt_max_length=4,
             lq_hidden_dim=16,lq_num_res_blocks=1)
    m=VideoPiD(cfg);m.reset_x0_head();m.add_temporal()
    assert all(p.ndim>0 for p in m.parameters())
    x=torch.randn(1,3,56,16,16);z=torch.randn(1,24,17,1,1)
    # A zero direct head yields zero x0, not the noisy input as a velocity wrapper would.
    out=m(z,x,torch.tensor([.3]));assert torch.count_nonzero(out)==0
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-4)
    from pid_video.objective import loss_x0
    loss_x0(out,torch.ones_like(x),torch.tensor([.3])).backward();opt.step();opt.zero_grad()
    loss_x0(m(z,x,torch.tensor([.3])),torch.ones_like(x),torch.tensor([.3])).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in m.named_parameters() if 'spatial' in n)
