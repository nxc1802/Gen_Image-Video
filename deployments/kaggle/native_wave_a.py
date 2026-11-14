"""Push/collect exact-checkpoint Phase 8 Wave A kernels, preserving every attempt."""

import argparse
import base64
import hashlib
import io
import json
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from deployments.kaggle.backend_lifecycle_probe import source_bundle
from deployments.kaggle.runner import KaggleAPIClient

ROOT = Path(__file__).resolve().parents[2]
IMAGE_BATCHES = {
    "image_t2i": ["image-standard-red-apple", "image-turbo-red-apple"],
    "image_i2i": ["image-i2i-pencil-v2"],
    "image_masked": ["image-inpaint-blue-patch", "image-outpaint-background"],
    "image_edit": ["image-edit-blue-suit", "image-reference-cat"],
    "image_multi": ["image-multi-cat-coffee"],
}
VIDEO_BATCHES = {
    "wan_t2v": ["wan-t2v-red-ball-screen-right-v3"],
    "minimax_t2v": ["minimax-t2v-red-ball-motion-v2"],
    "minimax_conditioned": [
        "minimax-i2v-astronaut-native-temporal-v2",
        "minimax-fl2v-astronaut-zoom-native-temporal-v2",
    ],
    "minimax_turbo": [
        "minimax-turbo-t2v-red-ball-motion",
        "minimax-turbo-i2v-astronaut",
        "minimax-turbo-fl2v-astronaut-zoom",
    ],
    "minimax_reference": ["minimax-reference-astronaut"],
    "minimax_audio": ["minimax-spoken-audio-native-dialogue-v2"],
    "minimax_reference_audio": ["minimax-reference-spoken-audio-native-dialogue-v2"],
    "wan_extension": ["wan-extension-pedestrians-native-grid-v2"],
    "wan_i2v": ["wan-i2v-astronaut-native-grid-v2"],
    "wan_fl2v": ["wan-fl2v-astronaut-zoom-native-grid-v2"],
    "wan_conditioned": [
        "wan-i2v-astronaut-native-grid-v2",
        "wan-fl2v-astronaut-zoom-native-grid-v2",
    ],
}


def execution_budget(kind):
    if kind not in VIDEO_BATCHES:
        return 4500, 7200
    from model_registry.deployment import ModelCatalog

    catalog = ModelCatalog(ROOT / "manifests")
    suite = json.loads((ROOT / "benchmarks/native/golden_wave_d.json").read_text())
    cases = [case for case in suite["cases"] if case["id"] in VIDEO_BATCHES[kind]]
    assert cases and len(cases) == len(VIDEO_BATCHES[kind])
    budget = sum(
        4 * catalog.get(case["model"]).runtime_profiles["kaggle-dual-t4"].inference_timeout_seconds
        for case in cases
    )
    driver = max(4500, budget + 2400)
    session = max(7200, driver + 3600)
    if session > 43200:
        raise ValueError("Selected native batch exceeds a bounded 12-hour session; split the batch")
    return driver, session


def bundle(kind=None):
    fixtures = (
        {ROOT / "tests/fixtures/native/vtest-first2s.mp4"} if kind in {None, "vlm"} else set()
    )
    if kind in VIDEO_BATCHES:
        suite = json.loads((ROOT / "benchmarks/native/golden_wave_d.json").read_text())
        for case in suite["cases"]:
            if case["id"] in VIDEO_BATCHES[kind]:
                fixtures.update(
                    ROOT / suite["assets"][name]["fixture_path"] for name in case["assets"]
                )
    stream = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(source_bundle())) as base,
        zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive,
    ):
        for name in base.namelist():
            archive.writestr(base.getinfo(name), base.read(name))
        for path in sorted((ROOT / "benchmarks/native").glob("*")):
            if path.suffix not in {".py", ".json"}:
                continue
            if path.name == "gate_audit_kernel.py":
                continue
            if path.name.startswith("minimax_vae_diagnostic") and kind not in {"minimax_vae_diag", "minimax_native_canvas_diag", "minimax_sampler_diag"}:
                continue
            info = zipfile.ZipInfo(str(path.relative_to(ROOT)), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
        for path in sorted(fixtures):
            info = zipfile.ZipInfo(str(path.relative_to(ROOT)), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    return stream.getvalue()


def render_kernel(kind):
    source = bundle(kind)
    driver_timeout, _ = execution_budget(kind)
    runtime = json.loads((ROOT / "deployments/kaggle/runtime_wave_a.json").read_text())
    if kind in {"asr", "tts"}:
        runtime.update(json.loads((ROOT / "deployments/kaggle/runtime_wave_b.json").read_text()))
    if kind.startswith("image"):
        runtime.update(json.loads((ROOT / "deployments/kaggle/runtime_wave_c.json").read_text()))
    if kind in VIDEO_BATCHES:
        runtime.update(json.loads((ROOT / "deployments/kaggle/runtime_wave_d.json").read_text()))
    image_cases = IMAGE_BATCHES.get(kind)
    suite = json.loads((ROOT / "benchmarks/native/golden_wave_c.json").read_text())
    image_models = list(
        dict.fromkeys(
            c["model"] for c in suite["cases"] if not image_cases or c["id"] in image_cases
        )
    )
    video_cases = VIDEO_BATCHES.get(kind)
    video_suite = json.loads((ROOT / "benchmarks/native/golden_wave_d.json").read_text())
    video_models = list(
        dict.fromkeys(c["model"] for c in video_suite["cases"] if c["id"] in (video_cases or []))
    )
    prefix = (
        "BUNDLE="
        + repr(base64.b64encode(source).decode())
        + "\nBUNDLE_SHA="
        + repr(hashlib.sha256(source).hexdigest())
        + "\nCORE_SHA="
        + repr(hashlib.sha256(source_bundle()).hexdigest())
        + "\nDRIVER_TIMEOUT="
        + repr(driver_timeout)
        + "\nRUNTIME="
        + repr(runtime)
        + "\nKIND="
        + repr(kind)
        + "\nIMAGE_CASES="
        + repr(image_cases)
        + "\nIMAGE_MODELS="
        + repr(image_models)
        + "\nVIDEO_CASES="
        + repr(video_cases)
        + "\nVIDEO_MODELS="
        + repr(video_models)
        + "\n"
    )
    return (
        prefix
        + r"""
import base64,hashlib,io,json,os,shutil,subprocess,sys,traceback,urllib.request,zipfile
from pathlib import Path
root=Path('/tmp/factory_phase8_a');root.mkdir(exist_ok=True)
wave='d' if VIDEO_CASES else 'c' if KIND.startswith('image') else 'b' if KIND in ('asr','vlm','tts') else 'a'
out=Path('/kaggle/working/native_wave_'+wave);out.mkdir(exist_ok=True)
report={'classification':'NATIVE_RUNTIME_SETUP_ONLY','source_bundle_sha256':CORE_SHA,'validation_bundle_sha256':BUNDLE_SHA,'runtime_pins':RUNTIME,'kind':KIND,'driver_timeout_seconds':DRIVER_TIMEOUT,'status':'NOT_MET','native_inference_executed':False,'production_ready':False,'full_factory_e2e_executed':False}
def logged(command,**kwargs):
    kwargs.pop('capture_output',None)
    with (out/'installation.log').open('ab') as sink:
        return subprocess.run(command,stdout=sink,stderr=subprocess.STDOUT,**kwargs)
try:
    zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))).extractall(root)
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,driver_version','--format=csv,noheader,nounits'],text=True)
    report['hardware_inventory']=inventory
    devices=[line for line in inventory.splitlines() if line.strip()]
    assert len(devices)==2 and all('T4' in line for line in devices),'Exact Dual-T4 unavailable before setup'
    if shutil.which('uv') is None:subprocess.run([sys.executable,'-m','pip','install','--quiet','uv'],check=True,timeout=120)
    python=root/'venv/bin/python'
    venv_args=['uv','venv','--python',sys.executable]
    subprocess.run(venv_args+[str(root/'venv')],check=True,timeout=60)
    if KIND in ('vllm','vlm'):
        spec=RUNTIME['vllm']['wheel'];target=root/spec['filename'];sha=hashlib.sha256()
        with urllib.request.urlopen(spec['url'],timeout=120) as response,target.open('wb') as sink:
            while chunk:=response.read(8*1024*1024):sha.update(chunk);sink.write(chunk)
        assert sha.hexdigest()==spec['sha256'] and target.stat().st_size==spec['size_bytes'],'Exact backend wheel integrity failed'
        report['backend_wheel_verified']=True
        logged(['uv','pip','install','--no-cache','--python',str(python),str(target),'wrapt>=1.16','psutil>=6','pyyaml','httpx','pydantic>=2.7'],check=True,timeout=2400)
        models=['qwen-vlm'] if KIND=='vlm' else ['qwen-decision','qwen-embedding','qwen-reranker']
    elif KIND=='asr':
        pin=RUNTIME['qwen_asr']
        logged(['uv','pip','install','--python',str(python),'--torch-backend','cu130','torch=='+pin['torch'],'torchvision=='+pin['torchvision'],'torchaudio=='+pin['torchaudio'],'qwen-asr=='+pin['version'],'transformers=='+pin['transformers'],'huggingface-hub=='+pin['huggingface_hub'],'fastapi==0.141.1','uvicorn==0.54.0','psutil==7.2.2','pyyaml','httpx==0.28.1','pydantic>=2.7','wrapt>=1.16'],check=True,timeout=1800)
        check="from importlib.metadata import version; import torch,torchvision,torchaudio; assert "+repr({name:pin[name] for name in ['torch','torchvision','torchaudio']})+" == {name:version(name) for name in ['torch','torchvision','torchaudio']}"
        logged([str(python),'-c',check],check=True,timeout=120)
        models=['qwen-asr']
    elif KIND=='tts':
        pin=RUNTIME['vllm_omni'];targets=[]
        for spec in [pin['vllm_wheel'],pin['wheel']]:
            target=root/spec['filename'];sha=hashlib.sha256()
            with urllib.request.urlopen(spec['url'],timeout=120) as response,target.open('wb') as sink:
                while chunk:=response.read(8*1024*1024):sha.update(chunk);sink.write(chunk)
            assert sha.hexdigest()==spec['sha256'] and target.stat().st_size==spec['size_bytes'],'Exact TTS backend wheel integrity failed'
            targets.append(str(target))
        report['backend_wheels_verified']=True
        logged(['uv','pip','install','--no-cache','--python',str(python),'--torch-backend','cu130',*targets,'torch=='+pin['torch'],'transformers=='+pin['transformers'],'openai-whisper=='+pin['whisper'],'psutil>=6','pyyaml','httpx','pydantic>=2.7','wrapt>=1.16'],check=True,timeout=2400)
        models=['qwen-tts']
    elif KIND.startswith('image') or KIND.startswith('minimax'):
        pin=RUNTIME['comfyui']
        def checkout(path,url,revision):
            subprocess.run(['git','init',str(path)],check=True,capture_output=True)
            subprocess.run(['git','-C',str(path),'remote','add','origin',url],check=True,capture_output=True)
            subprocess.run(['git','-C',str(path),'fetch','--depth','1','origin',revision],check=True,capture_output=True,timeout=180)
            subprocess.run(['git','-C',str(path),'checkout','FETCH_HEAD'],check=True,capture_output=True)
            assert subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()==revision
        source=root/'ComfyUI'
        checkout(source,pin['repository'],pin['source_revision'])
        checkout(source/'custom_nodes/ComfyUI-GGUF',pin['gguf_repository'],pin['gguf_revision'])
        logged(['uv','pip','install','--no-cache','--python',str(python),'--torch-backend','cu130','-r',str(source/'requirements.txt'),'torch=='+pin['torch'],'torchvision=='+pin['torchvision'],'torchaudio=='+pin['torchaudio'],'transformers=='+pin['transformers'],'gguf=='+pin['gguf'],'comfy-aimdo=='+pin['comfy-aimdo'],'comfy-kitchen=='+pin['comfy-kitchen'],*(['openai-whisper=='+RUNTIME['video_whisper']['version']] if VIDEO_CASES else []),'psutil>=6','pyyaml','httpx','pydantic>=2.7','wrapt>=1.16'],check=True,timeout=2400)
        check="from importlib.metadata import version; import torch,torchvision,torchaudio; assert "+repr({name:pin[name] for name in ['torch','torchvision','torchaudio','transformers','gguf']})+" == {name:version(name) for name in ['torch','torchvision','torchaudio','transformers','gguf']}"
        logged([str(python),'-c',check],check=True,timeout=120)
        # Runtime pin file is benchmark metadata, deliberately outside the production source bundle.
        (root/'deployments/kaggle').mkdir(parents=True,exist_ok=True)
        (root/'deployments/kaggle/runtime_wave_c.json').write_text(json.dumps({'comfyui':pin}))
        models=VIDEO_MODELS or IMAGE_MODELS
    elif VIDEO_CASES:
        pin=RUNTIME['diffusers'];spec=pin['wheel'];target=root/spec['filename'];sha=hashlib.sha256()
        with urllib.request.urlopen(spec['url'],timeout=120) as response,target.open('wb') as sink:
            while chunk:=response.read(8*1024*1024):sha.update(chunk);sink.write(chunk)
        assert sha.hexdigest()==spec['sha256'] and target.stat().st_size==spec['size_bytes'],'Exact Diffusers wheel integrity failed'
        report['backend_wheel_verified']=True
        packages=['torch','torchvision','torchaudio','transformers','accelerate','imageio','imageio-ffmpeg','sentencepiece','ftfy']
        logged(['uv','pip','install','--no-cache','--python',str(python),'--torch-backend','cu130',str(target),*[name+'=='+pin[name] for name in packages],'fastapi==0.141.1','uvicorn==0.54.0','psutil>=6','pyyaml','httpx','pydantic>=2.7','wrapt>=1.16'],check=True,timeout=2400)
        check="from importlib.metadata import version; import torch,torchvision,torchaudio; assert "+repr({name:pin[name] for name in packages})+" == {name:version(name) for name in "+repr(packages)+"}"
        logged([str(python),'-c',check],check=True,timeout=120)
        models=VIDEO_MODELS
    else:
        revision=RUNTIME['llama_cpp']['source_revision'];source=root/'llama.cpp'
        subprocess.run(['git','init',str(source)],check=True,capture_output=True)
        subprocess.run(['git','-C',str(source),'remote','add','origin',RUNTIME['llama_cpp']['repository']],check=True,capture_output=True)
        subprocess.run(['git','-C',str(source),'fetch','--depth','1','origin',revision],check=True,capture_output=True,timeout=180)
        subprocess.run(['git','-C',str(source),'checkout','FETCH_HEAD'],check=True,capture_output=True)
        driver=None
        for line in subprocess.check_output(['ldconfig','-p'],text=True).splitlines():
            if 'libcuda.so.1 ' in line and '=>' in line:
                candidate=Path(line.split('=>',1)[1].strip())
                if candidate.is_file():driver=candidate.resolve();break
        if driver is None:
            import ctypes
            ctypes.CDLL('libcuda.so.1')
            for line in Path('/proc/self/maps').read_text().splitlines():
                if '/libcuda.so' in line:
                    candidate=Path(line.split()[-1])
                    if candidate.is_file():driver=candidate.resolve();break
        if driver is None:raise RuntimeError('Real NVIDIA driver library unavailable; no CPU build fallback')
        report['cuda_driver_library']=str(driver)
        logged(['cmake','-S',str(source),'-B',str(source/'build'),'-DCUDA_cuda_driver_LIBRARY='+str(driver),'-DGGML_CUDA=ON','-DCMAKE_CUDA_ARCHITECTURES=75','-DLLAMA_CURL=OFF','-DBUILD_SHARED_LIBS=ON','-DCMAKE_BUILD_TYPE=Release'],check=True,timeout=120)
        logged(['cmake','--build',str(source/'build'),'--target','llama-server','--parallel','4'],check=True,timeout=2400)
        logged(['uv','pip','install','--python',str(python),'psutil>=6','pyyaml','httpx','pydantic>=2.7','wrapt>=1.16','huggingface-hub'],check=True,timeout=180)
        models=['qwen-text']
    for package in ['contracts','model-registry']:
        logged(['uv','pip','install','--no-deps','--python',str(python),'-e',str(root/'packages'/package)],check=True,timeout=120)
    inventory_code="import json; from importlib.metadata import distributions; print(json.dumps(dict(sorted((d.metadata['Name'],d.version) for d in distributions()))))"
    (out/'runtime_packages.json').write_bytes(subprocess.check_output([str(python),'-c',inventory_code]))
    paths=[str(root)]+[str(root/'packages'/name/'src') for name in ['contracts','model-registry']]
    env=os.environ.copy();env.update(PATH=str(python.parent)+os.pathsep+env.get('PATH',''),VIRTUAL_ENV=str(root/'venv'),PYTHONPATH=os.pathsep.join(paths),FACTORY_MANIFESTS_DIR=str(root/'manifests'),VLLM_WORKER_MULTIPROC_METHOD='spawn',TOKENIZERS_PARALLELISM='false')
    if VIDEO_CASES:
        (root/'deployments/kaggle').mkdir(parents=True,exist_ok=True)
        (root/'deployments/kaggle/runtime_wave_d.json').write_text(json.dumps(RUNTIME))
    log=out/'driver.log'
    with log.open('wb') as sink:
        chosen_cases=VIDEO_CASES or IMAGE_CASES
        extra=['--cases',*chosen_cases] if chosen_cases else []
        result=subprocess.run([str(python),'-m','benchmarks.native.wave_'+wave,'--root',str(root),'--output',str(out),'--models',*models,*extra],cwd=root,env=env,stdout=sink,stderr=subprocess.STDOUT,timeout=DRIVER_TIMEOUT)
    report['driver_exit_code']=result.returncode
    if (out/('wave_'+wave+'.json')).exists():
        native=json.loads((out/('wave_'+wave+'.json')).read_text());report.update(classification='REAL_NATIVE_VALIDATION_ATTEMPT',status=native['status'],native_inference_executed=any(c.get('native_inference_executed') for m in native['models'] for c in m['cases']))
except Exception as exc:report['error']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
finally:
    (out/'setup.json').write_text(json.dumps(report,indent=2)+'\n')
    with zipfile.ZipFile('/kaggle/working/native_wave_'+wave+'.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in out.rglob('*'):
            if path.is_file():archive.write(path,path.relative_to(out))
    print(json.dumps({'kind':KIND,'status':report['status'],'native_inference_executed':report['native_inference_executed'],'error':report.get('error',{}).get('message')}))
if report['status']!='PASS':raise SystemExit(1)
"""
    )


def collect(kind):
    client = KaggleAPIClient()
    record = ROOT / f"validation/native_{kind}_run.json"
    run = json.loads(record.read_text())
    version = run["response"]["versionNumber"]
    status = client.get_kernel_status(run["slug"])
    (ROOT / f"validation/native_{kind}_poll.json").write_text(
        json.dumps(
            {
                "observed_at_utc": datetime.now(UTC).isoformat(),
                "version": version,
                "kernel_url": run["response"]["url"],
                "run_record_sha256": hashlib.sha256(record.read_bytes()).hexdigest(),
                "kernel_status": status,
                "classification": "REMOTE_JOB_STATUS_ONLY_NOT_INFERENCE_EVIDENCE",
            },
            indent=2,
        )
        + "\n"
    )
    if status["status"] not in {"complete", "error", "failed"}:
        print(json.dumps({"kind": kind, "status": status["status"]}))
        return
    entries = client.get_kernel_output(run["slug"])
    folder = ROOT / f"validation/native_{kind}_v{version}"
    folder.mkdir(exist_ok=True)
    records = []
    for row in entries.get("files", []):
        if Path(row["fileName"]).name != (
            "native_wave_d.zip"
            if kind in VIDEO_BATCHES
            else "native_wave_c.zip"
            if kind.startswith("image")
            else "native_wave_b.zip"
            if kind in {"asr", "vlm", "tts"}
            else "native_wave_a.zip"
        ):
            continue
        with urllib.request.urlopen(row["url"], timeout=120) as response:
            raw = response.read(64 * 1024 * 1024 + 1)
        if len(raw) > 64 * 1024 * 1024:
            raise ValueError("Oversized native evidence archive")
        archive = zipfile.ZipFile(io.BytesIO(raw))
        total = sum(z.file_size for z in archive.infolist())
        if total > 128 * 1024 * 1024:
            raise ValueError("Oversized decoded native evidence")
        for item in archive.infolist():
            target = (folder / item.filename).resolve()
            if not target.is_relative_to(folder.resolve()) or item.is_dir():
                raise ValueError("Invalid native artifact path")
            data = archive.read(item)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.read_bytes() != data:
                raise ValueError("Historical native output changed")
            target.write_bytes(data)
            records.append(
                {"file": str(target.relative_to(ROOT)), "sha256": hashlib.sha256(data).hexdigest()}
            )
    receipt = {
        "kind": kind,
        "version": version,
        "kernel_status": status,
        "kernel_url": run["response"]["url"],
        "run_record_sha256": hashlib.sha256(record.read_bytes()).hexdigest(),
        "artifacts": records,
    }
    (ROOT / f"validation/native_{kind}_receipt_v{version}.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "kind": kind,
                "version": version,
                "status": status["status"],
                "artifacts": len(records),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind",
        choices=[
            "vllm",
            "text",
            "asr",
            "vlm",
            "tts",
            "image",
            "image_fp32",
            *IMAGE_BATCHES,
            *VIDEO_BATCHES,
        ],
        default="vllm",
    )
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if args.collect:
        return collect(args.kind)
    code = render_kernel(args.kind)
    if not args.run:
        print(json.dumps({"kind": args.kind, "kernel_bytes": len(code)}))
        return
    slug = "factory-phase-8-native-" + args.kind.replace("_", "-")
    client = KaggleAPIClient()
    response = client.push_kernel(
        slug=slug,
        title="Factory Phase 8 Native " + args.kind,
        code_content=code,
        enable_gpu=True,
        enable_internet=True,
        is_private=True,
        accelerator="NvidiaTeslaT4",
        session_timeout_seconds=execution_budget(args.kind)[1],
    )
    path = ROOT / f"validation/native_{args.kind}_run.json"
    if path.exists():
        old = json.loads(path.read_text())
        path.with_name(
            f"native_{args.kind}_run_v" + str(old["response"]["versionNumber"]) + ".json"
        ).write_text(path.read_text())
    run = {
        "observed_at_utc": datetime.now(UTC).isoformat(),
        "slug": slug,
        "response": response,
        "kernel_code_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "source_bundle_sha256": hashlib.sha256(source_bundle()).hexdigest(),
        "validation_bundle_sha256": hashlib.sha256(bundle(args.kind)).hexdigest(),
        "accelerator": "NvidiaTeslaT4",
        "driver_timeout_seconds": execution_budget(args.kind)[0],
        "session_timeout_seconds": execution_budget(args.kind)[1],
    }
    path.write_text(json.dumps(run, indent=2) + "\n")
    print(
        json.dumps(
            {"kind": args.kind, "url": response["url"], "version": response["versionNumber"]}
        )
    )


if __name__ == "__main__":
    main()
