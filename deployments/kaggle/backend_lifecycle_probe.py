"""Run the Phase 4 lifecycle witness on real Kaggle Dual-T4, without inference."""
from __future__ import annotations
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import urllib.parse
import urllib.request
import zipfile

from deployments.kaggle.runner import KaggleAPIClient

ROOT = Path(__file__).resolve().parents[2]


def source_bundle():
    paths = set()
    for relative in ["packages/contracts", "packages/model-registry", "packages/auth", "factory/apps/control-api", "factory/core", "factory/engines", "factory/adapters", "services/text/src", "services/vlm/src"]:
        for path in (ROOT / relative).rglob("*"):
            if path.is_file() and (path.suffix in {".py", ".toml", ".md", ".jinja"} or path.name == "LICENSE.vllm") and not {"tests", "__pycache__", ".venv"}.intersection(path.relative_to(ROOT).parts):
                paths.add(path)
    for relative in ["factory/__init__.py", "factory/apps/__init__.py", "manifests/models/qwen-asr.yaml"]:
        path = ROOT / relative
        if path.is_file():
            paths.add(path)
    for directory in ["manifests/models", "manifests/hardware", "manifests/services", "manifests/capabilities"]:
        paths.update((ROOT / directory).glob("*.yaml"))
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            info = zipfile.ZipInfo(str(path.relative_to(ROOT)), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    return stream.getvalue()


def render_kernel():
    bundle = source_bundle()
    prefix = "BUNDLE = " + repr(base64.b64encode(bundle).decode()) + "\nSOURCE_SHA256 = " + repr(hashlib.sha256(bundle).hexdigest()) + "\n"
    return prefix + r'''
import base64, hashlib, io, json, os, secrets, shutil, signal, subprocess, sys, time, traceback, urllib.request, zipfile
from pathlib import Path

out = Path('/kaggle/working/phase4_lifecycle.json')
root = Path('/tmp/factory_phase4')
report = {'classification':'REAL_BACKEND_LIFECYCLE_ONLY', 'source_bundle_sha256':SOURCE_SHA256,
          'phase4_exit_gate':'NOT_MET', 'native_inference_executed':False, 'semantic_validation_executed':False,
          'full_factory_e2e_executed':False, 'production_ready':False}
api = None
status = None
log = None
try:
    zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))).extractall(root)
    alias = root/'factory/apps/control_api'
    if not alias.exists(): alias.symlink_to('control-api', target_is_directory=True)
    subprocess.run([sys.executable,'-m','pip','install','--quiet','qwen-asr==0.0.6','fastapi>=0.111','uvicorn','python-multipart','psutil>=6.0','pyyaml'],check=True,timeout=1200)
    for package in ['contracts','model-registry','auth']:
        subprocess.run([sys.executable,'-m','pip','install','--quiet','--no-deps','-e',str(root/'packages'/package)],check=True,timeout=120)
    subprocess.run([sys.executable,'-m','pip','install','--quiet','--no-deps','-e',str(root/'factory/apps/control-api')],check=True,timeout=120)
    paths = [str(root)]
    paths += [str(root/'packages'/name/'src') for name in ['contracts','model-registry','auth']]
    sys.path[:0] = paths
    os.environ['FACTORY_MANIFESTS_DIR'] = str(root/'manifests')
    from model_registry.deployment import ModelCatalog
    from model_registry.deployment_validator import inspect_environment
    inventory = inspect_environment()
    report['hardware'] = inventory.gpus
    from importlib.metadata import version
    report['runtime_versions'] = {name:version(name) for name in ['qwen-asr','torch','transformers','huggingface-hub','psutil','fastapi','uvicorn']}
    assert set(inventory.gpus)=={0,1} and all('t4' in value['name'].lower() for value in inventory.gpus.values()), 'Exact Dual-T4 target unavailable'
    model = ModelCatalog(root/'manifests').get('qwen-asr')
    from huggingface_hub import hf_hub_download
    store = root/'models'/model.metadata.id
    report['checkpoint_files'] = []
    for key, spec in model.weights.files.items():
        assert spec.provider=='huggingface'
        source = Path(hf_hub_download(repo_id=spec.repository, filename=spec.path, revision=spec.revision, local_dir=str(root/'downloads'/spec.repository.replace('/','--'))))
        digest = hashlib.sha256()
        with source.open('rb') as stream:
            while chunk:=stream.read(8*1024*1024):digest.update(chunk)
        matched = digest.hexdigest()==spec.sha256 and source.stat().st_size==spec.size_bytes
        report['checkpoint_files'].append({'key':key,'repository':spec.repository,'revision':spec.revision,'file':spec.path,'sha256':digest.hexdigest(),'size_bytes':source.stat().st_size,'matched':matched})
        assert matched, 'Exact checkpoint integrity mismatch'
        target = store/(spec.destination or spec.path)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.symlink_to(source.resolve())
    import yaml
    config = {'apiVersion':'factory.ai/v1','kind':'DeploymentConfig','hardware':'kaggle-dual-t4',
              'models_dir':str(root/'models'),'runtime_dir':str(root/'runtime'),'api_host':'127.0.0.1','api_port':18000,
              'deployments':[{'model':'qwen-asr','backend':'qwen_asr','devices':[0],'port':19000,
                              'backend_options':{'startup_timeout_seconds':600,'stop_timeout_seconds':10}}]}
    config_path = root/'deployment.yaml'
    config_path.write_text(yaml.safe_dump(config))
    key = secrets.token_urlsafe(40)
    env = os.environ.copy()
    env.update(FACTORY_API_KEY=key,ENVIRONMENT='production',PYTHONPATH=os.pathsep.join(paths),FACTORY_MANIFESTS_DIR=str(root/'manifests'),FACTORY_DEPLOYMENT_FILE=str(config_path))
    log = (root/'factory_serve.log').open('wb')
    api = subprocess.Popen([sys.executable,'-m','model_registry.cli','--deployment',str(config_path),'serve'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    deadline = time.monotonic()+900
    diagnostic_at = 0
    def request(path):
        req=urllib.request.Request('http://127.0.0.1:18000'+path,headers={'Authorization':'Bearer '+key})
        with urllib.request.urlopen(req,timeout=5) as response:return json.load(response)
    while time.monotonic()<deadline:
        if api.poll() is not None:raise RuntimeError('factory serve exited before API/lifecycle ready: '+str(api.returncode))
        if time.monotonic() >= diagnostic_at:
            import psutil
            children = [{'pid':p.pid,'birth':p.create_time()} for p in psutil.Process(api.pid).children(recursive=True)]
            gpu = subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
            diagnostic = {'stage':'waiting_for_selected_backend','owned_children':children,'observed_gpu_processes':gpu.stdout.strip()}
            report.setdefault('startup_observations',[]).append(diagnostic)
            print(json.dumps(diagnostic),flush=True)
            diagnostic_at = time.monotonic()+30
        try:
            status=request('/v1/registry/deployment')
            if status.get('healthy_backends')==['qwen-asr']:break
        except OSError:pass
        time.sleep(1)
    else:raise RuntimeError('factory serve/lifecycle startup timeout')
    report['session'] = status
    report['control_plane'] = request('/v1/registry/runtime')
    public = request('/v1/models')
    report['public_discovery'] = public
    assert public=={'object':'list','data':[]}, 'Unvalidated model leaked into production discovery'
    for endpoint, payload in [('/v1/chat/completions',{'model':'qwen-asr','messages':[{'role':'user','content':'probe must be rejected'}]}),
                               ('/v1/responses',{'model':'qwen-asr','input':'probe must be rejected'})]:
        req=urllib.request.Request('http://127.0.0.1:18000'+endpoint,data=json.dumps(payload).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
        try:
            urllib.request.urlopen(req,timeout=5)
            raise AssertionError('Unvalidated inference was admitted')
        except urllib.error.HTTPError as exc:
            assert exc.code==503, 'Unvalidated inference did not fail closed'
    report['unvalidated_inference_rejected'] = True
    native = status['backend_lifecycle']['backends']['qwen-asr']
    assert native['backend']=='qwen_asr' and native['devices']==[0] and native['weights_verified'] and native['model_loaded']
    assert native['gpu_processes'] and {p['device'] for p in native['gpu_processes']}=={0}
    assert not native['inference_verified'] and not native['production_ready']
    report['launch_health_identity_gpu_checks'] = True
except Exception as exc:
    report['error'] = {'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
finally:
    if api is not None:
        if api.poll() is None:
            os.killpg(api.pid,signal.SIGTERM)
            try:api.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(api.pid,signal.SIGKILL);api.wait(timeout=10)
        report['api_exit_code'] = api.returncode
        receipts = list(root.glob('runtime/*/shutdown.json'))
        report['shutdown_receipt'] = json.loads(receipts[0].read_text()) if len(receipts)==1 else None
        import psutil
        leaked=[]
        if status:
            for model, native in status.get('backend_lifecycle',{}).get('backends',{}).items():
                pid=native.get('pid')
                if pid and psutil.pid_exists(pid):
                    process=psutil.Process(pid)
                    if process.status()!=psutil.STATUS_ZOMBIE and process.create_time()==native.get('pid_started_at'):
                        leaked.append(pid)
                        os.killpg(pid,signal.SIGKILL)
        report['backend_processes_leaked'] = leaked
        # Uvicorn may re-raise the handled SIGTERM after lifespan cleanup. A
        # completed owned-session receipt, not exit code alone, proves cleanup.
        receipt = report['shutdown_receipt']
        report['stop_verified'] = bool(not leaked and api.returncode in (0,-signal.SIGTERM) and receipt
            and receipt['stop_completed'] and not receipt['errors'] and status
            and receipt['session_id']==status['session_id'] and receipt['config_sha256']==status['config_sha256']
            and all(row['state']=='STOPPED' for row in receipt['backends'].values()))
    if log is not None:log.close()
    for path in root.glob('runtime/*/*/backend.log'):
        shutil.copyfile(path,Path('/kaggle/working')/('phase4_'+path.parent.name+'_backend.log'))
    if (root/'factory_serve.log').exists():shutil.copyfile(root/'factory_serve.log','/kaggle/working/phase4_factory_serve.log')
    if report.get('launch_health_identity_gpu_checks') and report.get('stop_verified') and 'error' not in report:
        report['phase4_exit_gate']='PASS'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'phase4_exit_gate':report['phase4_exit_gate'],'native_inference_executed':False,'error':report.get('error',{}).get('message')}))
if report['phase4_exit_gate']!='PASS':raise SystemExit(1)
'''


def collect(client, slug):
    """Retain bounded, named lifecycle artifacts, including unsuccessful runs."""
    run_path = ROOT / "validation/backend_lifecycle_run.json"
    run = json.loads(run_path.read_text())
    if run["requested_slug"] != slug:
        raise ValueError("Requested slug does not match the recorded push")
    status = client.get_kernel_status(slug)
    if status.get("status") not in {"complete", "error", "failed"}:
        print(json.dumps({"status":status.get("status"), "artifacts_collected":False}))
        return
    output = client.get_kernel_output(slug)
    version = run["response"]["versionNumber"]
    artifacts = []
    for item in output.get("files", []):
        name = Path(item.get("fileName", "")).name
        if name not in {"phase4_lifecycle.json", "phase4_factory_serve.log", "phase4_qwen-asr_backend.log"}:
            continue
        request = urllib.request.Request(item["url"])
        if urllib.parse.urlparse(item["url"]).hostname == "www.kaggle.com":
            request.add_header("Authorization", client.get_auth_header())
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read(8 * 1024 * 1024 + 1)
        if len(body) > 8 * 1024 * 1024:
            raise ValueError("Lifecycle artifact exceeds collection limit")
        path = ROOT / "validation" / (f"backend_lifecycle_observation_v{version}.json" if name == "phase4_lifecycle.json" else f"{Path(name).stem}_v{version}.log")
        if path.exists() and path.read_bytes() != body:
            raise ValueError("Refusing to overwrite a different historical observation")
        path.write_bytes(body)
        artifacts.append({"remote_name":name, "file":str(path.relative_to(ROOT)), "sha256":hashlib.sha256(body).hexdigest(), "size_bytes":len(body)})
    if not any(item["remote_name"] == "phase4_lifecycle.json" for item in artifacts):
        raise ValueError("Finished kernel has no lifecycle observation")
    receipt = {"collected_at_utc":datetime.now(timezone.utc).isoformat(), "kernel_url":run["response"]["url"],
        "version":version, "kernel_status":status, "run_record_sha256":hashlib.sha256(run_path.read_bytes()).hexdigest(), "artifacts":artifacts}
    (ROOT / f"validation/backend_lifecycle_receipt_v{version}.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({"status":status.get("status"), "version":version, "artifacts_collected":len(artifacts)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--collect", action="store_true")
    parser.add_argument("--slug", default="factory-phase-4-backend-lifecycle")
    args = parser.parse_args()
    if args.collect:
        collect(KaggleAPIClient(), args.slug)
        return
    code = render_kernel()
    if not args.run:
        print(json.dumps({"kernel_bytes":len(code.encode()), "purpose":"real backend lifecycle only; no inference", "execute":"add --run"}))
        return
    client = KaggleAPIClient()
    response = client.push_kernel(slug=args.slug, title="Factory Phase 4 Backend Lifecycle", code_content=code,
        enable_gpu=True, enable_internet=True, is_private=True, accelerator="NvidiaTeslaT4")
    record = {"observed_at_utc":datetime.now(timezone.utc).isoformat(),"response":response,
        "requested_slug":args.slug,"kernel_code_sha256":hashlib.sha256(code.encode()).hexdigest(),
        "accelerator":"NvidiaTeslaT4", "native_inference_executed":False}
    path = ROOT / "validation/backend_lifecycle_run.json"
    if path.is_file():
        previous = json.loads(path.read_text())
        version = previous.get("response", {}).get("versionNumber", "previous")
        path.with_name(f"backend_lifecycle_run_v{version}.json").write_text(path.read_text())
    path.write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps({"url":response.get("url"),"hasError":response.get("hasError"),"record":str(path)}))


if __name__ == "__main__":
    main()
