import json,sys
import pytest
from pid_video import prepare

def test_portable_prepare_freezes_inputs(tmp_path,monkeypatch):
 root=tmp_path/'repo';(root/'pid_video').mkdir(parents=True);(root/'pid_video/model.py').write_text('# test fixture')
 vendor=root/'third_party/diffusers-sol/src/diffusers';vendor.mkdir(parents=True);(vendor/'__init__.py').write_text('# fixture')
 monkeypatch.setattr(prepare,'ROOT',root)
 sources=tmp_path/'sources.jsonl';sources.write_text(''.join(json.dumps(dict(source_id=str(i),path=str(tmp_path/f'{i}.mp4')))+'\n' for i in range(4)))
 val=tmp_path/'validation.json';val.write_text(json.dumps(dict(source_id='val',path=str(tmp_path/'val.mp4'))))
 ff=tmp_path/'ffmpeg';ff.write_text('#!/bin/sh\nexit 0\n');ff.chmod(0o755)
 run=tmp_path/'run'
 args=['prepare','--run',str(run),'--sources',str(sources),'--validation-json',str(val),'--vae',str(tmp_path/'vae'),'--prior',str(tmp_path/'prior'),'--ffmpeg',str(ff)]
 monkeypatch.setattr(sys,'argv',args);prepare.main()
 cfg=json.loads((run/'config.json').read_text())
 assert cfg['global_batch']==64 and cfg['micro_batch']==2 and cfg['world_size']==4
 assert cfg['sources_sha256']==prepare.sha(run/'sources.jsonl')
 assert cfg['code_hashes']['pid_video/model.py']==prepare.sha(root/'pid_video/model.py')
 with pytest.raises(ValueError,match='Existing run'):prepare.main()
 monkeypatch.setattr(sys,'argv',args+['--micro-batch','3','--run',str(tmp_path/'bad')])
 with pytest.raises(ValueError,match='Invalid batch'):prepare.main()
