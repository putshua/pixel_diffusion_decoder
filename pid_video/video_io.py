import torch
import subprocess

def write_comparison(path, videos, ffmpeg):
    mean = videos[0].new_tensor([.485,.456,.406]).view(1,3,1,1,1)
    std = videos[0].new_tensor([.229,.224,.225]).view(1,3,1,1,1)
    pixels = torch.cat([(v[:,:,3:].float()*std+mean).clamp(0,1) for v in videos],dim=-1)
    pixels = (pixels[0].permute(1,2,3,0)*255).round().byte().cpu().numpy()
    cmd = [ffmpeg,'-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',
           f'{pixels.shape[2]}x{pixels.shape[1]}','-r','24','-i','pipe:0',
           '-an','-c:v','libx264','-threads','1','-crf','18','-pix_fmt','yuv420p',str(path)]
    subprocess.run(cmd,input=pixels.tobytes(),check=True)
