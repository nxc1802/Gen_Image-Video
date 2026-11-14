"""Five recorded frames through exact Wan VAE on CPU; diagnostic only."""
import hashlib
import json
import subprocess
import time
import urllib.request
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLWan
from PIL import Image

root = Path('/Volumes/WorkSpace/Factory/Gen_Image:Video')
out = root / 'validation/diagnostics/wan_vae_cpu_temporal_roundtrip_v1'
out.mkdir(parents=True, exist_ok=False)
script = Path(__file__).read_bytes()
(out / 'diagnostic.py').write_bytes(script)
report = {'classification': 'LOCAL_CPU_EXACT_VAE_RECORDED_VIDEO_COMPONENT_ROUNDTRIP_ONLY_NOT_WAN_T2V_OR_T4_CERTIFICATION', 'status': 'NOT_MET', 'script_sha256': hashlib.sha256(script).hexdigest(), 'native_gpu_inference_executed': False, 'full_wan_model_inference_executed': False, 'v3_certified': False, 'full_factory_e2e_executed': False, 'production_ready': False, 'runtime_versions': {n: version(n) for n in ['torch', 'diffusers', 'transformers', 'numpy', 'pillow']}}
try:
    torch.set_num_threads(2)
    original = Path('/tmp/factory-native-video/vtest-original.avi')
    spec = json.loads((root / 'benchmarks/native/golden_wave_d.json').read_text())['assets']['pedestrians']
    if not original.exists():
        with urllib.request.urlopen(spec['original_url'], timeout=120) as response:
            original.write_bytes(response.read(10 * 1024 * 1024))
    raw = original.read_bytes()
    assert len(raw) == spec['original_size_bytes'] and hashlib.sha256(raw).hexdigest() == spec['original_sha256']
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-i', str(original), '-frames:v', '5', '-vf', 'scale=384:288', '-sws_flags', 'bitexact+accurate_rnd', '-pix_fmt', 'rgb24', '-f', 'rawvideo', 'pipe:1']
    rgb = subprocess.check_output(command, timeout=60)
    assert len(rgb) == 5 * 384 * 288 * 3
    report['input'] = {'original_url': spec['original_url'], 'original_sha256': spec['original_sha256'], 'original_size_bytes': len(raw), 'decoded_rgb_sha256': hashlib.sha256(rgb).hexdigest(), 'shape_thwc': [5, 288, 384, 3], 'decode_command': command}
    pixels = np.frombuffer(rgb, dtype=np.uint8).reshape(5, 288, 384, 3).copy()
    tensor = torch.from_numpy(pixels).permute(3, 0, 1, 2).unsqueeze(0).float() / 127.5 - 1
    checkpoint = Path('/tmp/factory-native-video/wan-vae-cpu-weights/vae/diffusion_pytorch_model.safetensors')
    sha = hashlib.sha256()
    with checkpoint.open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            sha.update(chunk)
    assert sha.hexdigest() == '62cd18f19438e35b32ac63020e2852f566e9b02f46b6cdbd87972a356e3c6f4b' and checkpoint.stat().st_size == 2818777808
    config = checkpoint.with_name('config.json').read_bytes()
    assert hashlib.sha256(config).hexdigest() == 'd996c340fe9a7df5d7371f76a7d8d6956f6c98256080074d8434fa5eeac11360'
    report['vae_weights_sha256'] = sha.hexdigest()
    report['vae_config_sha256'] = hashlib.sha256(config).hexdigest()
    vae, info = AutoencoderKLWan.from_pretrained(checkpoint.parent, dtype=torch.float32, local_files_only=True, output_loading_info=True)
    report['load_info'] = info
    assert all(not info.get(k) for k in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'])
    vae.eval()
    params = list(vae.parameters())
    report['parameters'] = {'count': sum(p.numel() for p in params), 'devices': sorted({str(p.device) for p in params}), 'dtypes': sorted({str(p.dtype) for p in params})}
    assert report['parameters'] == {'count': 704688668, 'devices': ['cpu'], 'dtypes': ['torch.float32']}
    start = time.monotonic()
    with torch.inference_mode():
        latent = vae.encode(tensor).latent_dist.mode()
        decoded = vae.decode(latent).sample
    report['roundtrip_seconds'] = time.monotonic() - start
    assert decoded.shape == tensor.shape and torch.isfinite(latent).all() and torch.isfinite(decoded).all()
    report['latent'] = {'shape': list(latent.shape), 'minimum': latent.min().item(), 'maximum': latent.max().item(), 'finite': True}
    decoded = decoded.clamp(-1, 1)
    report['decoded'] = {'shape': list(decoded.shape), 'minimum': decoded.min().item(), 'maximum': decoded.max().item(), 'finite': True, 'frame_mae_0_to_1': [(decoded[:, :, i] - tensor[:, :, i]).abs().mean().item() / 2 for i in range(5)]}
    frames = ((decoded[0].permute(1, 2, 3, 0).cpu().numpy() + 1) * 127.5).round().astype(np.uint8)
    report['frame_artifacts'] = []
    for i in range(5):
        for name, data in [('input', pixels[i]), ('reconstructed', frames[i])]:
            dest = out / f'{name}_{i}.png'
            Image.fromarray(data).save(dest)
            body = dest.read_bytes()
            report['frame_artifacts'].append({'file': dest.name, 'sha256': hashlib.sha256(body).hexdigest(), 'size_bytes': len(body)})
    report['status'] = 'COMPONENT_EXECUTION_COMPLETE_NOT_CERTIFICATION'
except Exception as exc:
    report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
finally:
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'roundtrip_seconds': report.get('roundtrip_seconds'), 'decoded': report.get('decoded'), 'error': report.get('error')}), flush=True)
