"""Pixel/latent temporal alignment for H3 17n+5 clips, without production blending."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def expand_pixels(x):
    if x.ndim != 5 or x.shape[2] < 5 or (x.shape[2]-5)%17:
        raise ValueError('Expected a BCTHW video with 17n+5 frames')
    pieces=[]; masks=[]
    for start in range(0,x.shape[2],17):
        clip=x[:,:,start:start+17]
        pieces.append(torch.cat([clip[:,:,:1].expand(-1,-1,3,-1,-1),clip],dim=2))
        masks.extend([False]*3+[True]*clip.shape[2])
    return torch.cat(pieces,dim=2),torch.tensor(masks,device=x.device)


def contract_pixels(x):
    if x.shape[2]<8 or (x.shape[2]-8)%20:
        raise ValueError('Expected expanded temporal length 20n+8')
    return torch.cat([x[:,:,start+3:min(start+20,x.shape[2])] for start in range(0,x.shape[2],20)],dim=2)


class CheckpointBlock(nn.Module):
    def __init__(self,block):
        super().__init__();self.block=block
    def forward(self,*args):
        if torch.is_grad_enabled():
            return checkpoint(self.block,*args,use_reentrant=False)
        return self.block(*args)


def checkpoint_decoder(decoder):
    decoder.transformer_blocks=nn.ModuleList([CheckpointBlock(b) for b in decoder.transformer_blocks])
    return decoder


def masked_flow_loss(pred,target,t,mask):
    if mask.dtype != torch.bool or mask.shape != (target.shape[2],) or not mask.any():
        raise ValueError('Invalid temporal mask')
    den=(1-t.float()).clamp_min(.05).view(-1,1,1,1,1)
    return (((pred.float()-target.float())/den)[:,:,mask]**2).mean()
