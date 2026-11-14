"""Joint causal endpoint codec study only, never Transformer inference."""
import hashlib
import io
import json
import time
import urllib.request
from pathlib import Path
from importlib.metadata import version

import numpy as np
import torch
import yaml
from diffusers import AutoencoderKLWan
from PIL import Image, ImageDraw

root=Path('/Volumes/WorkSpace/Factory/Gen_Image:Video')
out=root/'validation/diagnostics/wan_joint_endpoint_context_cpu_v1'
out.mkdir(exist_ok=False)
script=Path(__file__).read_bytes();(out/'diagnostic.py').write_bytes(script)
report={'classification':'ORIGINAL_CPU_VAE_JOINT_CAUSAL_ENDPOINT_COMPONENT_STUDY_ONLY',
    'status':'NOT_MET','script_sha256':hashlib.sha256(script).hexdigest(),
    'runtime_versions':{n:version(n) for n in ('torch','diffusers','numpy')},
    'input_sequence_scope':'Explicit photo-crop diagnostic sequence and held/interpolated conditioning sequences derived from two real photographs. None is recorded motion or Transformer output.',
    'native_gpu_inference_executed':False,'full_wan_model_inference_executed':False,
    'v3_certified':False,'production_ready':False,'full_factory_e2e_executed':False}

def tensor(im):
    return torch.from_numpy(np.asarray(im).copy()).permute(2,0,1)[None,:,None].float()/127.5-1

def sha(t):return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()

try:
    torch.set_num_threads(2)
    manifest=yaml.safe_load((root/'manifests/models/wan-2.2.yaml').read_text())
    model=Path('/tmp/factory-native-video/wan-vae-cpu-weights')
    for f in manifest['weights']['files'].values():
        if f['path'].startswith('vae/'):
            p=model/f['path'];h=hashlib.sha256()
            with p.open('rb') as stream:
                while chunk:=stream.read(8*1024*1024):h.update(chunk)
            assert h.hexdigest()==f['sha256'] and p.stat().st_size==f['size_bytes']
    suite=json.loads((root/'benchmarks/native/golden_wave_d.json').read_text())
    images=[]
    for name in ('astronaut_first','astronaut_last'):
        a=suite['assets'][name];raw=(root/a['fixture_path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==a['sha256']
        images.append(tensor(Image.open(io.BytesIO(raw)).convert('RGB')))
    first,last=images
    a=suite['assets']['astronaut_first']
    raw=urllib.request.urlopen(a['original_url'],timeout=60).read()
    assert hashlib.sha256(raw).hexdigest()==a['original_sha256']
    (out/'original_photo.png').write_bytes(raw)
    original=Image.open(io.BytesIO(raw)).convert('RGB')
    frames=[]
    for i in range(17):
        crop=2*i
        frames.append(tensor(original.crop((crop,crop,original.width-crop,original.height-crop)).resize((384,256),Image.Resampling.LANCZOS)))
    source=torch.cat(frames,dim=2)
    assert torch.equal(source[:,:,:1],first) and torch.equal(source[:,:,-1:],last)
    held=torch.cat([first.repeat(1,1,16,1,1),last],dim=2)
    blend=torch.cat([(1-i/16)*first+(i/16)*last for i in range(17)],dim=2)
    report['source_tensor_sha256']=sha(source)
    report['conditioning_tensor_sha256']={'held_first_then_last':sha(held),'linear_endpoint_blend':sha(blend)}
    vae,info=AutoencoderKLWan.from_pretrained(model,subfolder='vae',dtype=torch.float32,local_files_only=True,output_loading_info=True)
    assert all(not info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
    report['load_info']=info;vae.eval();start=time.monotonic()
    with torch.inference_mode():
        z=vae.encode(source).latent_dist.mode()
        candidates={'actual_photo_crop_sequence_roundtrip':z}
        for name,context in [('joint_held_first_then_last',held),('joint_linear_endpoint_blend',blend)]:
            condition=vae.encode(context).latent_dist.mode()
            mixed=z.clone();mixed[:,:,:1]=condition[:,:,:1];mixed[:,:,-1:]=condition[:,:,-1:]
            candidates[name]=mixed
        report['cases']=[]
        for name,latent in candidates.items():
            decoded=vae.decode(latent).sample
            assert decoded.shape==source.shape and torch.isfinite(decoded).all()
            error=(decoded.clamp(-1,1)-source).abs().mean(dim=(0,1,3,4))/2
            row={'name':name,'latent_sha256':sha(latent),'decoded_tensor_sha256':sha(decoded),
                'first_frame_pixel_mae':float(error[0]),'last_frame_pixel_mae':float(error[-1]),
                'per_frame_pixel_mae':error.tolist(),'decoded_frames':17}
            sheet=Image.new('RGB',(4*384,5*276),'white');draw=ImageDraw.Draw(sheet)
            for i in range(17):
                pixels=((decoded[0,:,i].clamp(-1,1).permute(1,2,0).numpy()+1)*127.5).round().astype(np.uint8)
                im=Image.fromarray(pixels);im.save(out/f'{name}-{i}.png')
                x,y=(i%4)*384,(i//4)*276;sheet.paste(im,(x,y));draw.text((x+2,y+256),str(i),fill='black')
            p=out/f'{name}-all-frames.png';sheet.save(p)
            row['all_frame_contact_sheet']={'file':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            report['cases'].append(row)
    report['elapsed_seconds']=time.monotonic()-start
    report['status']='JOINT_CONTEXT_COMPONENT_STUDY_COMPLETE_NOT_FL2V_CERTIFICATION'
except Exception as exc:report['error']={'type':type(exc).__name__,'message':str(exc)}
finally:
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'error':report.get('error'),'cases':report.get('cases'),'elapsed_seconds':report.get('elapsed_seconds')}),flush=True)
