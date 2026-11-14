"""Exact CPU VAE/native conditioning diagnosis; no full Wan model or GPU inference."""
import hashlib
import inspect
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import yaml
from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
from PIL import Image

root = Path('/Volumes/WorkSpace/Factory/Gen_Image:Video')
out = root / 'validation/diagnostics/wan_final_anchor_ignored_cpu_v1'
out.mkdir(exist_ok=False)
script = Path(__file__).read_bytes()
(out / 'diagnostic.py').write_bytes(script)
report = {'classification': 'EXACT_CPU_VAE_AND_NATIVE_LATENT_PREPARATION_ONLY_NOT_FULL_WAN_OR_GPU_INFERENCE',
          'script_sha256': hashlib.sha256(script).hexdigest(), 'status': 'NOT_MET',
          'native_gpu_inference_executed': False, 'full_wan_model_inference_executed': False,
          'v3_certified': False, 'production_ready': False, 'full_factory_e2e_executed': False,
          'runtime_versions': {n: version(n) for n in ('torch', 'diffusers', 'transformers', 'numpy')},
          'verified_files': {}}
try:
    torch.set_num_threads(2)
    assert version('diffusers') == '0.43.0'
    index_raw = Path('/tmp/factory-wan-exact-model-index.json').read_bytes()
    manifest = yaml.safe_load((root / 'manifests/models/wan-2.2.yaml').read_text())
    index_spec = next(s for s in manifest['weights']['files'].values() if s['path'] == 'model_index.json')
    assert hashlib.sha256(index_raw).hexdigest() == index_spec['sha256']
    assert len(index_raw) == index_spec['size_bytes']
    (out / 'model_index.json').write_bytes(index_raw)
    report['exact_model_index_sha256'] = index_spec['sha256']
    assert json.loads(index_raw)['expand_timesteps'] is True
    model = Path('/tmp/factory-native-video/wan-vae-cpu-weights')
    for s in manifest['weights']['files'].values():
        if not s['path'].startswith('vae/'):
            continue
        p = model / s['path']
        h = hashlib.sha256()
        with p.open('rb') as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                h.update(chunk)
        assert h.hexdigest() == s['sha256'] and p.stat().st_size == s['size_bytes']
        report['verified_files'][s['path']] = {'sha256': h.hexdigest(), 'size_bytes': p.stat().st_size}
    suite = json.loads((root / 'benchmarks/native/golden_wave_d.json').read_text())
    inputs = {}
    for name in ('astronaut_first', 'astronaut_last'):
        spec = suite['assets'][name]
        raw = (root / spec['fixture_path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == spec['sha256'] and len(raw) == spec['size_bytes']
        image = Image.open(root / spec['fixture_path']).convert('RGB').resize((64, 64), Image.Resampling.LANCZOS)
        inputs[name] = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1
    assert not torch.equal(inputs['astronaut_first'], inputs['astronaut_last'])
    report['distinct_actual_photographed_inputs'] = {n: suite['assets'][n]['sha256'] for n in inputs}
    vae, info = AutoencoderKLWan.from_pretrained(model, subfolder='vae', dtype=torch.float32,
                                                local_files_only=True, output_loading_info=True)
    assert all(not info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'))
    report['load_info'] = info
    report['vae_parameters'] = {'count': sum(p.numel() for p in vae.parameters()),
                                'devices': sorted({str(p.device) for p in vae.parameters()}),
                                'dtypes': sorted({str(p.dtype) for p in vae.parameters()})}
    pipeline = WanImageToVideoPipeline(tokenizer=None, text_encoder=None, vae=vae.eval(), scheduler=None,
                                       expand_timesteps=True)
    source = inspect.getsource(WanImageToVideoPipeline.prepare_latents).encode()
    (out / 'native_prepare_latents.py.txt').write_bytes(source)
    report['native_prepare_latents_sha256'] = hashlib.sha256(source).hexdigest()
    start = time.monotonic()
    outputs = []
    with torch.inference_mode():
        for last in (inputs['astronaut_first'], inputs['astronaut_last']):
            outputs.append(pipeline.prepare_latents(inputs['astronaut_first'], batch_size=1,
                num_channels_latents=48, height=64, width=64, num_frames=5, dtype=torch.float32,
                device=torch.device('cpu'), generator=torch.Generator(device='cpu').manual_seed(314159),
                last_image=last))
    report['elapsed_seconds'] = time.monotonic() - start
    report['returned_tensors'] = []
    for name, a, b in zip(('noise', 'condition', 'first_frame_mask'), *outputs, strict=True):
        report['returned_tensors'].append({'name': name, 'shape': list(a.shape),
            'finite': bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
            'identical_when_last_image_changes': torch.equal(a, b),
            'sha256': [hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest() for t in (a, b)]})
    assert all(t['identical_when_last_image_changes'] for t in report['returned_tensors'])
    report['status'] = 'LAST_IMAGE_IGNORED_REPRODUCED_COMPONENT_ONLY'
except Exception as exc:
    report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
finally:
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
