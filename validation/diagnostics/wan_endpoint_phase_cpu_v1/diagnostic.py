"""Original VAE causal endpoint study, never full Wan video generation."""
import hashlib
import json
import time
from pathlib import Path
from importlib.metadata import version

import numpy as np
import torch
import yaml
from diffusers import AutoencoderKLWan
from PIL import Image, ImageDraw

root = Path('/Volumes/WorkSpace/Factory/Gen_Image:Video')
out = root / 'validation/diagnostics/wan_endpoint_phase_cpu_v1'
out.mkdir(exist_ok=False)
script = Path(__file__).read_bytes()
(out / 'diagnostic.py').write_bytes(script)
report = {'classification': 'ORIGINAL_CPU_VAE_CAUSAL_ENDPOINT_COMPONENT_STUDY_ONLY',
    'status': 'NOT_MET', 'script_sha256': hashlib.sha256(script).hexdigest(),
    'runtime_versions': {n: version(n) for n in ('torch', 'diffusers', 'numpy')},
    'input_sequence_scope': '17 explicitly held/changed frames derived from two actual photographed keyframes; not a recorded motion clip or Transformer-generated video.',
    'native_gpu_inference_executed': False, 'full_wan_model_inference_executed': False,
    'v3_certified': False, 'production_ready': False, 'full_factory_e2e_executed': False}

def digest(t):
    return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()

try:
    torch.set_num_threads(2)
    spec = yaml.safe_load((root / 'manifests/models/wan-2.2.yaml').read_text())
    model = Path('/tmp/factory-native-video/wan-vae-cpu-weights')
    for f in spec['weights']['files'].values():
        if f['path'].startswith('vae/'):
            p = model / f['path']; h = hashlib.sha256()
            with p.open('rb') as s:
                while chunk := s.read(8 * 1024 * 1024): h.update(chunk)
            assert h.hexdigest() == f['sha256'] and p.stat().st_size == f['size_bytes']
    suite = json.loads((root / 'benchmarks/native/golden_wave_d.json').read_text())
    images = {}
    report['inputs'] = {}
    for name in ('astronaut_first', 'astronaut_last'):
        a = suite['assets'][name]; raw = (root / a['fixture_path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == a['sha256']
        im = Image.open(root / a['fixture_path']).convert('RGB').resize((384, 256), Image.Resampling.LANCZOS)
        images[name] = torch.from_numpy(np.asarray(im).copy()).permute(2,0,1)[None,:,None].float()/127.5-1
        report['inputs'][name] = {'source_sha256': a['sha256'], 'derived_tensor_sha256': digest(images[name])}
    vae, info = AutoencoderKLWan.from_pretrained(model, subfolder='vae', dtype=torch.float32,
                                               local_files_only=True, output_loading_info=True)
    assert all(not info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs'))
    vae.eval()
    report['load_info'] = info
    report['original_vae_parameter_count'] = sum(p.numel() for p in vae.parameters())
    first, last = images['astronaut_first'], images['astronaut_last']
    source = torch.cat([first.repeat(1,1,16,1,1), last], dim=2)
    start = time.monotonic()
    with torch.inference_mode():
        baseline = vae.encode(source).latent_dist.mode()
        start_anchor = vae.encode(first).latent_dist.mode()
        last_single = vae.encode(last).latent_dist.mode()
        last_repeated = vae.encode(last.repeat(1,1,5,1,1)).latent_dist.mode()[:,:,-1:]
        cases = {'real_photo_sequence_roundtrip': baseline,
                 'independent_single_frame_last_anchor': baseline.clone(),
                 'independent_repeated_five_frame_last_anchor': baseline.clone()}
        cases['independent_single_frame_last_anchor'][:,:,0:1] = start_anchor
        cases['independent_single_frame_last_anchor'][:,:,-1:] = last_single
        cases['independent_repeated_five_frame_last_anchor'][:,:,0:1] = start_anchor
        cases['independent_repeated_five_frame_last_anchor'][:,:,-1:] = last_repeated
        report['anchor_latents'] = {n: {'sha256':digest(t),'shape':list(t.shape)} for n,t in
            [('first',start_anchor),('last_single',last_single),('last_repeated_final',last_repeated)]}
        report['cases'] = []
        for name, z in cases.items():
            decoded = vae.decode(z).sample
            assert decoded.shape == source.shape and torch.isfinite(decoded).all()
            error = (decoded.clamp(-1,1)-source).abs().mean(dim=(0,1,3,4))/2
            metrics = {'name':name, 'latent_sha256':digest(z), 'decoded_tensor_sha256':digest(decoded),
                'first_frame_pixel_mae':float(error[0]), 'last_frame_pixel_mae':float(error[-1]),
                'per_frame_pixel_mae':error.tolist(), 'decoded_frames':17}
            sheet = Image.new('RGB', (4*384,5*(256+20)), 'white'); draw=ImageDraw.Draw(sheet)
            for i in range(17):
                pixels=((decoded[0,:,i].clamp(-1,1).permute(1,2,0).numpy()+1)*127.5).round().astype(np.uint8)
                im=Image.fromarray(pixels); im.save(out/f'{name}-{i}.png')
                x,y=(i%4)*384,(i//4)*276;sheet.paste(im,(x,y));draw.text((x+2,y+256),str(i),fill='black')
            target=out/f'{name}-all-frames.png';sheet.save(target)
            metrics['all_frame_contact_sheet']={'file':target.name,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
            report['cases'].append(metrics)
    report['elapsed_seconds']=time.monotonic()-start
    report['status']='COMPONENT_STUDY_COMPLETE_NOT_FL2V_CERTIFICATION'
except Exception as exc:
    report['error']={'type':type(exc).__name__,'message':str(exc)}
finally:
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'error':report.get('error'),'cases':report.get('cases'),'elapsed_seconds':report.get('elapsed_seconds')}),flush=True)
