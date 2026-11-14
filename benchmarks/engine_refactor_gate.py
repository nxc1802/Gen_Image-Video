"""Phase 5 architecture exit gate with separately scoped real enhancement probe."""
from __future__ import annotations
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from benchmarks.backend_lifecycle_gate import evaluate as lifecycle_gate
from deployments.kaggle.enhancement_probe import render_kernel, source_bundle, COMFY_REVISION, INPUT_SHA, INPUT_URL
from factory.core.model_adapter import load_adapter
from model_registry.deployment import ModelCatalog, DeploymentRegistry
from factory_contracts.deployment import DeploymentConfig, DeploymentEntry

ENGINES={"vector":"VectorEngine","decision":"DecisionEngine","responses":"ResponsesEngine","speech":"SpeechEngine",
         "image":"ImageEngine","video":"VideoEngine","music":"MusicEngine","enhancement":"EnhancementEngine"}


def enhancement_evidence():
    checks=[]
    def check(name,passed):checks.append({"check":name,"passed":bool(passed)})
    result={"classification":"ISOLATED_DISABLED_MODEL_NATIVE_BACKEND_PROBE","checks":checks}
    try:
        run_path=ROOT/"validation/enhancement_probe_run.json"
        run=json.loads(run_path.read_text());version=run["response"]["versionNumber"]
        observation_path=ROOT/f"validation/enhancement_observation_v{version}.json"
        receipt_path=ROOT/f"validation/enhancement_receipt_v{version}.json"
        observation=json.loads(observation_path.read_text());receipt=json.loads(receipt_path.read_text())
        result.update(observation=str(observation_path.relative_to(ROOT)),receipt=str(receipt_path.relative_to(ROOT)))
        check("current_kernel_and_source",run["kernel_code_sha256"]==hashlib.sha256(render_kernel().encode()).hexdigest()
              and observation["source_bundle_sha256"]==hashlib.sha256(source_bundle()).hexdigest())
        check("completed_exact_run",receipt["version"]==version and receipt["kernel_status"]["status"]=="complete"
              and receipt["kernel_url"]==run["response"]["url"]
              and receipt["run_record_sha256"]==hashlib.sha256(run_path.read_bytes()).hexdigest())
        artifacts={row["remote_name"]:row for row in receipt["artifacts"]}
        for remote,path in [("phase5_enhancement.json",observation_path),("phase5_enhancement_output.png",ROOT/f"validation/phase5_enhancement_v{version}.png")]:
            row=artifacts[remote]
            check("artifact_integrity_"+remote,row["file"]==str(path.relative_to(ROOT)) and row["sha256"]==hashlib.sha256(path.read_bytes()).hexdigest())
        hardware=observation["hardware"]
        check("two_real_t4_devices",set(hardware)=={"0","1"} and all("t4" in row["name"].lower() and row["total_vram_gb"]>=14.2 for row in hardware.values()))
        catalog=ModelCatalog(ROOT/"manifests");model=catalog.get("enhancement-suite-v1")
        expected={key:(s.repository,s.revision,s.sha256,s.size_bytes) for key,s in model.weights.files.items()}
        actual={row["key"]:(row["repository"],row["revision"],row["sha256"],row["size_bytes"]) for row in observation["checkpoint_files"]}
        check("complete_exact_checkpoint_bundle",actual==expected and len(observation["checkpoint_files"])==len(expected))
        binding=DeploymentRegistry(DeploymentConfig(deployments=(DeploymentEntry(model=model.metadata.id,backend="comfyui",devices=(0,)),)),catalog).assignment(model.metadata.id)
        runtime=observation["runtime"]
        check("configured_backend_identity",runtime["model"]==model.metadata.id and runtime["adapter"]==model.backend.adapter and runtime["backend"]=="comfyui"
              and runtime["checkpoint_graph_sha256"]==binding.checkpoint_graph_sha256 and runtime["devices"]==[0]
              and runtime["source_revision"]==observation["comfy_revision"]==COMFY_REVISION and runtime["weights_verified"] is True
              and runtime["state"]=="HEALTHY" and runtime["pid"]>0 and runtime["executable_sha256"])
        samples=[p for sample in observation["gpu_samples_during_inference"] for p in sample]
        peak=max((p["used_memory_mb"] for p in samples),default=0)
        result["gpu_peak_memory_mb"]=peak
        check("owned_gpu_process_during_inference",bool(samples) and all(p["pid"]==runtime["pid"] and p["device"]==0 and p["used_memory_mb"]>0 for p in samples) and peak>1000)
        graph=observation["graph"]
        check("actual_checkpoint_graph",set(graph)=={"input","model","upscale","save"}
              and graph["input"]["class_type"]=="LoadImage" and graph["save"]["class_type"]=="SaveImage"
              and graph["model"]=={"class_type":"UpscaleModelLoader","inputs":{"model_name":Path(model.weights.files["primary"].destination or model.weights.files["primary"].path).name}}
              and graph["upscale"]=={"class_type":"ImageUpscaleWithModel","inputs":{"upscale_model":["model",0],"image":["input",0]}}
              and graph["save"]["inputs"]["images"]==["upscale",0]
              and observation["graph_sha256"]==hashlib.sha256(json.dumps(graph,sort_keys=True).encode()).hexdigest() and bool(observation["prompt_id"]))
        check("real_pinned_photo_input",observation["input"]=={"url":INPUT_URL,"sha256":INPUT_SHA,"dimensions":[512,512]})
        body=(ROOT/artifacts["phase5_enhancement_output.png"]["file"]).read_bytes()
        from PIL import Image
        with Image.open(io.BytesIO(body)) as image:
            image.load();dimensions=list(image.size)
        check("decoded_actual_x4_output",dimensions==observation["output"]["dimensions"]==[2048,2048]
              and len(body)==observation["output"]["size_bytes"] and hashlib.sha256(body).hexdigest()==observation["output"]["sha256"]!=INPUT_SHA)
        check("native_probe_success_and_cleanup",observation["exit_gate"]=="PASS" and observation["native_inference_executed"] is True
              and observation["stop_verified"] is True and "error" not in observation and "cleanup_error" not in observation)
        check("no_production_or_v3_v4_promotion",observation["classification"]==result["classification"] and observation["model_state"]==model.validation.state.value=="DISABLED"
              and observation["public_models"]==[] and not any(model.capabilities.values())
              and all(observation[k] is False for k in ("production_ready","full_factory_e2e_executed","v3_certified","semantic_quality_certified")))
    except (OSError,ValueError,KeyError,TypeError) as exc:
        result["error"]=f"{type(exc).__name__}: {exc}"
    result["exit_gate"]="PASS" if checks and all(row["passed"] for row in checks) and "error" not in result else "NOT_MET"
    return result


def extensibility_contracts():
    source=ROOT/"tests/contract/test_engine_extensibility.py"
    with tempfile.TemporaryDirectory(prefix="factory-engine-gate-") as temp:
        junit=Path(temp)/"results.xml"
        command=[sys.executable,"-m","pytest",str(source),"-q","-o","addopts=",f"--junitxml={junit}"]
        run=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=120)
        cases=[];digest=None
        if junit.is_file():
            digest=hashlib.sha256(junit.read_bytes()).hexdigest()
            for case in ET.parse(junit).iter("testcase"):
                if len(case)==0:cases.append(case.attrib["name"])
        expected={f"test_model_addition_without_core_change[{engine}]" for engine in ENGINES}
        passed=run.returncode==0 and set(cases)==expected and len(cases)==8
        return {"exit_gate":"PASS" if passed else "NOT_MET","classification":"SOFTWARE_ONLY_EXPLICIT_MOCKS",
                "command":command,"exit_code":run.returncode,"passed_cases":cases,"junit_sha256":digest,
                "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"diagnostics":None if passed else (run.stdout+run.stderr)[-5000:]}


def evaluate():
    findings,rows=[],[]
    prior=lifecycle_gate()
    if prior["exit_gate"]!="PASS":findings.append("Previous lifecycle/deployment/canonicalization/model truth gates failed: "+str(prior["findings"]))
    catalog=ModelCatalog(ROOT/"manifests")
    for engine,class_name in ENGINES.items():
        path=ROOT/f"factory/engines/{engine}/__init__.py";issues=[];models=[]
        if not path.is_file():issues.append("Canonical engine missing")
        else:
            tree=ast.parse(path.read_text())
            if class_name not in {n.name for n in ast.walk(tree) if isinstance(n,ast.ClassDef)}:issues.append("Generic engine class missing")
            for source in path.parent.rglob("*.py"):
                for node in ast.walk(ast.parse(source.read_text())):
                    if isinstance(node,ast.Constant) and isinstance(node.value,str) and any(name in node.value.lower() for name in ["qwen","minimax","llama","vllm","acestep","yue2","comfyui"]):
                        issues.append(f"Model/backend string in engine: {source.relative_to(ROOT)}:{node.lineno}")
            for model in catalog.models().values():
                if model.engine!=engine:continue
                try:
                    adapter=load_adapter(model.backend.adapter)
                    enabled={cid for cid,value in model.capabilities.items() if value}
                    if not enabled<=adapter.capabilities:issues.append(f"Missing declared capabilities: {model.metadata.id}")
                    if not set(model.backend.supported)<=adapter.supported_backends:issues.append(f"Unsupported backend: {model.metadata.id}")
                    if set(model.validation.implemented_capabilities)!=enabled:issues.append(f"Implementation truth mismatch: {model.metadata.id}")
                    models.append({"model":model.metadata.id,"adapter":model.backend.adapter,"implemented_capabilities":sorted(enabled),"production_ready":False})
                except Exception as exc:issues.append(f"{model.metadata.id}: {type(exc).__name__}: {exc}")
        if issues:findings.append(f"{engine}: "+"; ".join(issues))
        rows.append({"engine":engine,"class":class_name,"source":str(path.relative_to(ROOT)),"models":models,
                     "source_sha256":hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
                     "structural_checks":"PASS" if not issues else "NOT_MET","findings":issues})
    contracts=extensibility_contracts()
    if contracts["exit_gate"]!="PASS":findings.append("Cross-engine model addition contracts failed")
    native=enhancement_evidence()
    if native["exit_gate"]!="PASS":findings.append("Actual Enhancement backend evidence failed")
    return {"phase":5,"observed_at_utc":datetime.now(timezone.utc).isoformat(),"exit_gate":"PASS" if not findings else "NOT_MET",
        "classification":"ARCHITECTURE_AND_SOFTWARE_WITH_SCOPED_ENHANCEMENT_NATIVE_PROBE","engines":rows,"findings":findings,
        "prior_phase_gate":prior["exit_gate"],"extensibility_contracts":contracts,"enhancement_native_probe":native,
        "scope":"Eight engines support new Manifest/Adapter without core edits. Actual native inference is witnessed only for disabled RealESRGAN image x4; no other adapter is inference-certified.",
        "production_ready":False,"native_inference_executed":native["exit_gate"]=="PASS","semantic_validation_executed":False,"full_factory_e2e_executed":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"validation/engine_refactor_gate.json")
    args=parser.parse_args();report=evaluate()
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"exit_gate":report["exit_gate"],"findings":report["findings"],"engines":{r["engine"]:r["structural_checks"] for r in report["engines"]},"native_probe":report["enhancement_native_probe"]}))
    return 0 if report["exit_gate"]=="PASS" else 1


if __name__=="__main__":raise SystemExit(main())
