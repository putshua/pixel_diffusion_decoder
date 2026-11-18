"""Deterministic, source-balanced re-extraction from a frozen original-video pool."""
import hashlib
import json
import random
import math
import re
import subprocess
from pathlib import Path
import torch


def probe_video(path,ffmpeg):
    stat=Path(path).stat()
    result=subprocess.run([ffmpeg,'-hide_banner','-i',path],stdout=subprocess.DEVNULL,
                          stderr=subprocess.PIPE,timeout=60)
    text=result.stderr.decode(errors='replace')
    duration=re.search(r'Duration: (\d+):(\d+):(\d+(?:\.\d+)?)',text)
    video=next((line for line in text.splitlines() if 'Video:' in line),'')
    size=re.search(r'[, ](\d{2,5})x(\d{2,5})[, ]',video)
    if not duration or not size:raise ValueError('Video metadata unavailable')
    hh,mm,ss=map(float,duration.groups())
    return dict(duration=hh*3600+mm*60+ss,width=int(size[1]),height=int(size[2]),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)


def decode_pixels(row,bucket,ffmpeg,start):
    w,h,f=bucket
    cmd=[ffmpeg,'-v','error','-threads','1','-ss',str(start),'-i',row['path'],'-vf',
         f'fps=24,scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}',
         '-frames:v',str(f),'-threads','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1']
    raw=subprocess.check_output(cmd,timeout=180)
    if len(raw)!=f*h*w*3:raise ValueError(f'Insufficient frames: {len(raw)} bytes, expected {f*h*w*3}')
    pixels=torch.frombuffer(bytearray(raw),dtype=torch.uint8).reshape(f,h,w,3).permute(3,0,1,2)[None].float()/255
    mean=pixels.new_tensor([.485,.456,.406]).view(1,3,1,1,1)
    std=pixels.new_tensor([.229,.224,.225]).view(1,3,1,1,1)
    return (pixels-mean)/std,hashlib.sha256(raw).hexdigest()


class SourcePool:
    def __init__(self,rows,rank,world,ffmpeg,rejections,seed,cursor=0,windows_per_source=1):
        if len(rows)%world:raise ValueError('Training source count must divide world size')
        self.rows=rows;self.rank=rank;self.world=world;self.ffmpeg=ffmpeg
        self.rejections=rejections;self.seed=seed;self.cursor=cursor
        self.windows=windows_per_source
        if self.windows not in (1,3):raise ValueError("Supported windows per source: 1 or 3")
        self.multiplier=104729
        while math.gcd(self.multiplier,len(rows)*self.windows)!=1:self.multiplier+=2
    def next(self,bucket):
        for _ in range(64):
            cursor=self.cursor;logical=(cursor*self.world+self.rank)%(len(self.rows)*self.windows)
            if self.windows>1:logical=(logical*self.multiplier+self.seed)%(len(self.rows)*self.windows)
            index,variant=divmod(logical,self.windows)
            row=self.rows[index];self.cursor+=1
            try:
                info=probe_video(row['path'],self.ffmpeg)
                w,h,f=bucket
                if info['duration']<f/24+.1 or info['width']<w or info['height']<h:
                    raise ValueError('Source is shorter/smaller than bucket (no upscaling)')
                seed=int(hashlib.sha256(f'{self.seed}:{cursor}:{self.rank}:{row["source_id"]}'.encode()).hexdigest()[:16],16)
                if self.windows==1:
                    start=random.Random(seed).uniform(0,max(0,info['duration']-f/24-.2))
                else:
                    count=math.floor((info['duration']-f/24-.2)*24)+1
                    if count<self.windows:raise ValueError('Insufficient temporal extent for distinct starts')
                    low=variant*count//self.windows;high=(variant+1)*count//self.windows-1
                    start=random.Random(seed).randint(low,high)/24
                pixels,sha=decode_pixels(row,bucket,self.ffmpeg,start)
                record=dict(**row,bucket=bucket,start_seconds=start,cursor=cursor,window_variant=variant,pixel_sha256=sha,source_stat=info)
                return pixels,record
            except (OSError,ValueError,subprocess.SubprocessError) as exc:
                self.rejections.write(json.dumps(dict(cursor=cursor,source=row,bucket=bucket,error=str(exc)))+'\n');self.rejections.flush()
        raise RuntimeError('64 consecutive ineligible or unreadable sources; refusing an endless retry loop')
