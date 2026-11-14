"""Real original VAE with experimental Adapter preparation, no Transformer/GPU."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from PIL import Image
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

root=Path('/Volumes/WorkSpace/Factory/Gen_Image:Video')
out=root/'validation/diagnostics/wan_experimental_preparation_cpu_v1'
out.mkdir(exist_ok=False)
script=Path(__file__).read_bytes();(out/'diagnostic.py').write_bytes(script)
report={'classification':'EXACT_CPU_VAE_EXPERIMENTAL_ADAPTER_LATENT_PREPARATION_ONLY',
    'script_sha256':hashlib.sha256(script).hexdigest(),'status':'NOT_MET',
    'native_gpu_inference_executed':False,'full_wan_model_inference_executed':False,
    'v3_certified':False,'production_ready':False,'full_factory_e2e_executed':False}
try:
    torch.set_num_threads(2)
    path=root/'factory/adapters/wan/endpoint_conditioning.py'
    spec=importlib.util.spec_from_file_location('experimental_wan_endpoint_conditioning',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cls=module.endpoint_pipeline_class(WanImageToVideoPipeline)
    manifest=yaml.safe_load((root/'manifests/models/wan-2.2.yaml').read_text())
    model=Path('/tmp/factory-native-video/wan-vae-cpu-weights')
    for f in manifest['weights']['files'].values():
        if f['path'].startswith('vae/'):
            p=model/f['path'];h=hashlib.sha256()
            with p.open('rb') as stream:
                while chunk:=stream.read(8*1024*1024):h.update(chunk)
            assert h.hexdigest()==f['sha256'] and p.stat().st_size==f['size_bytes']
    suite=json.loads((root/'benchmarks/native/golden_wave_d.json').read_text());images={}
    for name in ('astronaut_first','astronaut_last'):
        a=suite['assets'][name];raw=(root/a['fixture_path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==a['sha256']
        im=Image.open(root/a['fixture_path']).convert('RGB').resize((64,64),Image.Resampling.LANCZOS)
        images[name]=torch.from_numpy(np.asarray(im).copy()).permute(2,0,1).unsqueeze(0).float()/127.5-1
    vae,info=AutoencoderKLWan.from_pretrained(model,subfolder='vae',dtype=torch.float32,local_files_only=True,output_loading_info=True)
    assert all(not info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
    pipe=cls(tokenizer=None,text_encoder=None,vae=vae.eval(),scheduler=None,expand_timesteps=True)
    report['runtime_identity']=module.runtime_identity();outputs=[];records=[]
    with torch.inference_mode():
        for last in (images['astronaut_first'],images['astronaut_last']):
            outputs.append(pipe.prepare_latents(images['astronaut_first'],batch_size=1,num_channels_latents=48,
                height=64,width=64,num_frames=9,dtype=torch.float32,device=torch.device('cpu'),
                generator=torch.Generator(device='cpu').manual_seed(314159),last_image=last))
            records.append(pipe._factory_endpoint_call_evidence)
    a,b=outputs
    report['checks']={'same_noise_for_same_seed':torch.equal(a[0],b[0]),
        'same_first_condition':torch.equal(a[1][:,:,:1],b[1][:,:,:1]),
        'different_real_last_image_changes_final_condition':not torch.equal(a[1][:,:,-1:],b[1][:,:,-1:]),
        'mask_exactly_two_endpoints_and_one_free_interior':bool(torch.all(a[2][:,:,0]==0) and torch.all(a[2][:,:,-1]==0) and torch.all(a[2][:,:,1:-1]==1)),
        'inactive_middle_condition_zero':bool(torch.all(a[1][:,:,1:-1]==0)),
        'all_finite':all(bool(torch.isfinite(t).all()) for group in outputs for t in group)}
    report['actual_call_evidence']=records
    assert all(report['checks'].values())
    report['status']='BOTH_ENDPOINT_CONSUMPTION_REPRODUCED_COMPONENT_ONLY'
except Exception as exc:report['error']={'type':type(exc).__name__,'message':str(exc)}
finally:
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
