"""PiD spatial backbone with zero-init temporal residuals; same-resolution H3 decode."""
from pathlib import Path
import sys
import torch
from torch import nn
import torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[1]
if not (ROOT/'third_party/pid_official/pid/_src/networks/pid_net.py').is_file():
    raise FileNotFoundError('Bundled PiD source missing; use an editable repository checkout')
sys.path.insert(0, str(ROOT / 'third_party/pid_official'))
from pid._src.networks.pid_net import PidNet

BASE = dict(in_channels=3, num_groups=24, hidden_size=1536, pixel_hidden_size=16,
            pixel_attn_hidden_size=1152, pixel_num_groups=16, patch_depth=14,
            pixel_depth=2, patch_size=16, txt_embed_dim=2304, txt_max_length=300,
            use_text_rope=True, rope_mode='ntk_aware', rope_ref_h=1024, rope_ref_w=1024,
            repa_encoder_index=-1, lq_in_channels=0, lq_latent_channels=24,
            lq_hidden_dim=512, lq_interval=2, zero_init_lq=True,
            train_lq_proj_only=False, sr_scale=1, latent_spatial_down_factor=16)


def align_latents(z):
    """17 latent frames -> 68 expanded positions -> 56 real H3 frames.

    Each 17-pixel encode block has five latent positions, with 3 synthetic prefix
    pixels in its 20-pixel decode. Use explicit native correspondence, not linear
    interpolation across encode-block boundaries.
    """
    if z.ndim != 5 or z.shape[1:3] != (24, 17):
        raise ValueError('Expected H3 posterior-mean post-quant latent [B,24,17,H/16,W/16]')
    from pid_video.temporal import contract_pixels
    return contract_pixels(z.repeat_interleave(4, dim=2))


class TemporalBlock(nn.Module):
    def __init__(self, spatial, dim, frames=56, width=128):
        super().__init__(); self.spatial=spatial; self.frames=frames
        self.norm=nn.LayerNorm(dim); self.down=nn.Linear(dim,width)
        self.conv=nn.Conv1d(width,width,3,padding=1)
        self.up=nn.Linear(width,dim)
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
    def forward(self, *args):
        x,y=self.spatial(*args)
        n,l,d=x.shape
        if n % self.frames: raise ValueError('Temporal block requires complete clips')
        v=F.silu(self.down(self.norm(x))).reshape(n//self.frames,self.frames,l,-1)
        v=v.permute(0,2,3,1).reshape(-1,v.shape[-1],self.frames)
        v=self.conv(v).reshape(n//self.frames,l,-1,self.frames).permute(0,3,1,2).reshape(n,l,-1)
        return x+self.up(F.silu(v)),y


class VideoPiD(nn.Module):
    def __init__(self, network=None, frames=56, temporal_width=128):
        super().__init__();self.frames=frames
        self.net=PidNet(**(network or BASE));self.net.init_weights()
        # FSDP1 cannot shard 0-D parameters. Length-one gate alpha is numerically
        # identical under broadcasting and belongs to the newly initialized adapter.
        for gate in self.net.lq_proj.gate_modules:
            if gate.log_alpha.ndim==0:
                gate.log_alpha=nn.Parameter(gate.log_alpha.detach().reshape(1))
        # Explicit no-caption ablation: learned null token, not H3 text features.
        self.null_text=nn.Parameter(torch.zeros(1,1,self.net.txt_embed_dim))
        self.temporal_width=temporal_width
    def add_temporal(self):
        for i,b in enumerate(self.net.patch_blocks):
            if i%2==1:self.net.patch_blocks[i]=TemporalBlock(b,self.net.hidden_size,self.frames,self.temporal_width)
    def reset_x0_head(self):
        # Spatial backbone retains prior weights; direct clean-pixel readout is new.
        nn.init.zeros_(self.net.final_layer.linear.weight)
        nn.init.zeros_(self.net.final_layer.linear.bias)

    def forward(self,z,noisy,t):
        b,c,frames,h,w=noisy.shape
        if frames!=self.frames or (h,w)!=(z.shape[-2]*16,z.shape[-1]*16):
            raise ValueError('Same-resolution video contract violated')
        latent=align_latents(z).permute(0,2,1,3,4).reshape(b*frames,24,h//16,w//16)
        x=noisy.permute(0,2,1,3,4).reshape(b*frames,3,h,w)
        sig=(1-t).repeat_interleave(frames)
        # Keep prior time-embedding coordinate (noise fraction * 1000), but the
        # NEW output head directly predicts clean ImageNet-normalized pixels.
        # No noisy-input skip and no velocity-to-x0 conversion is applied.
        v=self.net(x,sig*1000,self.null_text.expand(b*frames,-1,-1),lq_latent=latent,
                   degrade_sigma=torch.zeros_like(sig))
        return v.reshape(b,frames,3,h,w).permute(0,2,1,3,4)


def load_spatial_prior(model, checkpoint, prefix='net_ema.'):
    """Strictly load every spatial prior tensor from official DCP; only new LQ heads excluded."""
    import torch.distributed.checkpoint as dcp
    path=Path(checkpoint)
    if not (path/'.metadata').is_file():raise FileNotFoundError(f'Expected official DCP model directory: {path}')
    metadata=dcp.FileSystemReader(path).read_metadata().state_dict_metadata
    expected={k:v for k,v in model.net.state_dict().items() if not k.startswith('lq_proj.')}
    missing=[prefix+k for k in expected if prefix+k not in metadata]
    if missing:raise ValueError(f'Incomplete or incompatible spatial checkpoint: {missing[:8]}')
    for k,v in expected.items():
        if tuple(metadata[prefix+k].size)!=tuple(v.shape):raise ValueError(f'Shape mismatch: {k}')
    state={prefix+k:v for k,v in expected.items()}
    dcp.load(state,checkpoint_id=path)
    result=model.net.load_state_dict({k:state[prefix+k] for k in expected},strict=False)
    if result.unexpected_keys or any(not k.startswith('lq_proj.') for k in result.missing_keys):
        raise ValueError(str(result))
    return dict(loaded_tensors=len(expected),new_parameters=result.missing_keys,prefix=prefix)
