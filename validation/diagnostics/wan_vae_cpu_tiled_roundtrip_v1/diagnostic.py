"""Exact Wan VAE CPU component diagnosis only; no T4/T2V certification."""
import hashlib
import json
import time
import urllib.request
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import yaml
from diffusers import AutoencoderKLWan
from PIL import Image

root = Path('/Volumes/WorkSpace/Factory/Gen_Image:Video')
out = root / 'validation/diagnostics/wan_vae_cpu_tiled_roundtrip_v1'
out.mkdir(parents=True, exist_ok=False)
script = Path(__file__).read_bytes()
(out / 'diagnostic.py').write_bytes(script)
report = {
    'classification': 'LOCAL_CPU_EXACT_TILED_VAE_HIGH_RES_PHOTO_ROUNDTRIP_ONLY_NOT_WAN_T2V_OR_T4_CERTIFICATION',
    'status': 'NOT_MET',
    'script_sha256': hashlib.sha256(script).hexdigest(),
    'native_gpu_inference_executed': False,
    'full_wan_model_inference_executed': False,
    'v3_certified': False,
    'full_factory_e2e_executed': False,
    'production_ready': False,
    'runtime_versions': {n: version(n) for n in ['torch', 'diffusers', 'transformers', 'numpy', 'pillow']},
    'verified_files': {},
}
try:
    torch.set_num_threads(2)
    manifest = yaml.safe_load((root / 'manifests/models/wan-2.2.yaml').read_text())
    model = Path('/tmp/factory-native-video/wan-vae-cpu-weights')
    for spec in manifest['weights']['files'].values():
        if not spec['path'].startswith('vae/'):
            continue
        target = model / spec['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            url = f"https://huggingface.co/{spec['repository']}/resolve/{spec['revision']}/{spec['path']}"
            with urllib.request.urlopen(url, timeout=180) as response, target.open('wb') as sink:
                while chunk := response.read(8 * 1024 * 1024):
                    sink.write(chunk)
        sha = hashlib.sha256()
        with target.open('rb') as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                sha.update(chunk)
        assert target.stat().st_size == spec['size_bytes'] and sha.hexdigest() == spec['sha256']
        report['verified_files'][spec['path']] = {'repository': spec['repository'], 'revision': spec['revision'], 'sha256': sha.hexdigest(), 'size_bytes': target.stat().st_size, 'matched': True}
    raw = (root / 'tests/fixtures/native/astronaut_first.png').read_bytes()
    expected = json.loads((root / 'benchmarks/native/golden_wave_d.json').read_text())['assets']['astronaut_first']
    assert hashlib.sha256(raw).hexdigest() == expected['sha256'] and len(raw) == expected['size_bytes']
    (out / 'input.png').write_bytes(raw)
    image = Image.open(out / 'input.png').convert('RGB').resize((1280, 704), Image.Resampling.LANCZOS)
    image.save(out / 'input.png')
    resized = (out / 'input.png').read_bytes()
    report['input'] = {'source_sha256': expected['sha256'], 'source_size_bytes': len(raw), 'sha256': hashlib.sha256(resized).hexdigest(), 'size_bytes': len(resized), 'dimensions': list(image.size), 'conversion': 'Actual predeclared photograph resized to 1280x704 with Pillow Lanczos; no generated input pixels.'}
    vae, load_info = AutoencoderKLWan.from_pretrained(model, subfolder='vae', dtype=torch.float32, local_files_only=True, output_loading_info=True)
    report['load_info'] = load_info
    assert all(not load_info.get(key) for key in ['missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'])
    vae.eval()
    policy = manifest['runtime_profiles']['kaggle-dual-t4']['component_tiling']['vae']
    vae.enable_tiling(tile_sample_min_height=policy['tile_height'], tile_sample_min_width=policy['tile_width'], tile_sample_stride_height=policy['stride_height'], tile_sample_stride_width=policy['stride_width'])
    actual = {'tile_height': vae.tile_sample_min_height, 'tile_width': vae.tile_sample_min_width, 'stride_height': vae.tile_sample_stride_height, 'stride_width': vae.tile_sample_stride_width}
    assert vae.use_tiling and actual == policy
    report['component_tiling'] = {'enabled': vae.use_tiling, 'vae': actual}
    params = list(vae.parameters())
    report['parameters'] = {'count': sum(p.numel() for p in params), 'devices': sorted({str(p.device) for p in params}), 'dtypes': sorted({str(p.dtype) for p in params})}
    assert report['parameters'] == {'count': 704688668, 'devices': ['cpu'], 'dtypes': ['torch.float32']}
    tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0).unsqueeze(2).float() / 127.5 - 1
    start = time.monotonic()
    with torch.inference_mode():
        latent = vae.encode(tensor).latent_dist.mode()
        decoded = vae.decode(latent).sample
    report['roundtrip_seconds'] = time.monotonic() - start
    assert decoded.shape == tensor.shape and torch.isfinite(latent).all() and torch.isfinite(decoded).all()
    report['latent'] = {'shape': list(latent.shape), 'minimum': latent.min().item(), 'maximum': latent.max().item(), 'finite': True}
    report['decoded'] = {'shape': list(decoded.shape), 'minimum': decoded.min().item(), 'maximum': decoded.max().item(), 'finite': True, 'mean_absolute_error_0_to_1': (decoded.clamp(-1, 1) - tensor).abs().mean().item() / 2, 'root_mean_square_error_0_to_1': (decoded.clamp(-1, 1) - tensor).square().mean().sqrt().item() / 2}
    pixels = ((decoded[0, :, 0].clamp(-1, 1).permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().astype(np.uint8)
    Image.fromarray(pixels).save(out / 'reconstructed.png')
    result = (out / 'reconstructed.png').read_bytes()
    report['output'] = {'file': 'reconstructed.png', 'sha256': hashlib.sha256(result).hexdigest(), 'size_bytes': len(result)}
    report['status'] = 'COMPONENT_EXECUTION_COMPLETE_NOT_CERTIFICATION'
except Exception as exc:
    report['error'] = {'type': type(exc).__name__, 'message': str(exc)}
finally:
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'roundtrip_seconds': report.get('roundtrip_seconds'), 'decoded': report.get('decoded'), 'error': report.get('error')}), flush=True)
