"""Same JiT-style direct x0 prediction / RF weighting as H3 pixeldecoder."""
import torch

def mix(target,noise,t):
    t=t.reshape(-1,1,1,1,1)
    return t*target+(1-t)*noise

def loss_x0(pred,target,t):
    den=(1-t.float()).clamp_min(.05).reshape(-1,1,1,1,1)
    return ((pred.float()-target.float())/den).square().mean()

@torch.no_grad()
def sample_x0(model,z,noise,steps=8):
    x=noise.clone()
    for j in range(steps):
        t=x.new_full((x.shape[0],),j/steps)
        pred=model(z,x,t)
        x=x+(pred.float()-x)/(1-j/steps)/steps
    return x

def to_rgb(x):
    mean=x.new_tensor([.485,.456,.406]).reshape(1,3,1,1,1)
    std=x.new_tensor([.229,.224,.225]).reshape(1,3,1,1,1)
    return (x*std+mean).clamp(0,1)
