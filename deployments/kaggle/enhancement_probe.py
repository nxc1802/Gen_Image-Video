"""Isolated exact-checkpoint backend probe. Disabled model is never public."""
import argparse,base64,hashlib,json,urllib.request
from datetime import datetime,timezone
from pathlib import Path
from deployments.kaggle.backend_lifecycle_probe import source_bundle
from deployments.kaggle.runner import KaggleAPIClient
ROOT=Path(__file__).resolve().parents[2]
COMFY_REVISION="c5751c81abe131202eea9e4c66d000589013dd0b"
INPUT_URL="https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/astronaut.png"
INPUT_SHA="88431cd9653ccd539741b555fb0a46b61558b301d4110412b5bc28b5e3ea6cb5"


def render_kernel():
 bundle=source_bundle()
 prefix="BUNDLE="+repr(base64.b64encode(bundle).decode())+"\nSOURCE_SHA="+repr(hashlib.sha256(bundle).hexdigest())+"\nCOMFY_REVISION="+repr(COMFY_REVISION)+"\nINPUT_URL="+repr(INPUT_URL)+"\nINPUT_SHA="+repr(INPUT_SHA)+"\n"
 return prefix+r'''
import asyncio,base64,hashlib,io,json,os,shutil,subprocess,sys,traceback,urllib.request,zipfile
from pathlib import Path
root=Path('/tmp/factory_p5_enhance');root.mkdir(exist_ok=True)
report={'classification':'ISOLATED_DISABLED_MODEL_NATIVE_BACKEND_PROBE','source_bundle_sha256':SOURCE_SHA,'exit_gate':'NOT_MET',
 'production_ready':False,'full_factory_e2e_executed':False,'v3_certified':False,'semantic_quality_certified':False,'native_inference_executed':False}
session=None
try:
 zipfile.ZipFile(io.BytesIO(base64.b64decode(BUNDLE))).extractall(root)
 source=root/'ComfyUI'
 subprocess.run(['git','init',str(source)],check=True,capture_output=True)
 subprocess.run(['git','-C',str(source),'remote','add','origin','https://github.com/Comfy-Org/ComfyUI.git'],check=True,capture_output=True)
 subprocess.run(['git','-C',str(source),'fetch','--depth','1','origin',COMFY_REVISION],check=True,capture_output=True,timeout=180)
 subprocess.run(['git','-C',str(source),'checkout','FETCH_HEAD'],check=True,capture_output=True)
 subprocess.run([sys.executable,'-m','pip','install','--quiet','-r',str(source/'requirements.txt'),'psutil>=6','pydantic>=2.7','pyyaml','httpx'],check=True,timeout=1200)
 for name in ['contracts','model-registry']:
  subprocess.run([sys.executable,'-m','pip','install','--quiet','--no-deps','-e',str(root/'packages'/name)],check=True,timeout=120)
 paths=[str(root)]+[str(root/'packages'/name/'src') for name in ['contracts','model-registry']]
 sys.path[:0]=paths;os.environ['PYTHONPATH']=os.pathsep.join(paths)
 from model_registry.deployment import ModelCatalog,DeploymentRegistry
 from model_registry.deployment_validator import inspect_environment
 from factory_contracts.deployment import DeploymentConfig,DeploymentEntry,BackendOptions
 from model_registry.backends.session import BackendSession
 from factory.core.model_adapter import ExecutionContext,load_adapter
 from factory.core.media import MediaRequest
 inventory=inspect_environment();report['hardware']=inventory.gpus
 assert set(inventory.gpus)=={0,1} and all('t4' in v['name'].lower() for v in inventory.gpus.values())
 catalog=ModelCatalog(root/'manifests');model=catalog.get('enhancement-suite-v1')
 assert model.validation.state.value=='DISABLED' and not any(model.capabilities.values())
 from huggingface_hub import hf_hub_download
 report['checkpoint_files']=[]
 for key,spec in model.weights.files.items():
  downloaded=Path(hf_hub_download(spec.repository,filename=spec.path,revision=spec.revision))
  digest=hashlib.sha256(downloaded.read_bytes()).hexdigest()
  assert digest==spec.sha256 and downloaded.stat().st_size==spec.size_bytes
  target=root/'models'/model.metadata.id/(spec.destination or spec.path);target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(downloaded.resolve())
  report['checkpoint_files'].append({'key':key,'repository':spec.repository,'revision':spec.revision,'sha256':digest,'size_bytes':spec.size_bytes})
 with urllib.request.urlopen(INPUT_URL,timeout=60) as response: image=response.read()
 assert hashlib.sha256(image).hexdigest()==INPUT_SHA
 from PIL import Image
 im=Image.open(io.BytesIO(image));report['input']={'url':INPUT_URL,'sha256':INPUT_SHA,'dimensions':list(im.size)}
 config=DeploymentConfig(models_dir=root/'models',runtime_dir=root/'runtime',deployments=(DeploymentEntry(model=model.metadata.id,backend='comfyui',devices=(0,),port=18988,
  backend_options=BackendOptions(source_dir=source,source_revision=COMFY_REVISION,startup_timeout_seconds=300,stop_timeout_seconds=10)),))
 registry=DeploymentRegistry(config,catalog,config_path=root/'probe.yaml');session=BackendSession(registry)
 async def run():
  await session.start()
  assert session.healthy_models()==(model.metadata.id,),session.failures
  context=ExecutionContext(registry,session,registry.assignment(model.metadata.id),model)
  adapter=load_adapter(model.backend.adapter)
  request=MediaRequest(model=model.metadata.id,capability='enhancement.image_super_resolution',inputs={'image':image,'scale':4})
  adapter.validate_request(model,request,request.capability)
  prepared=await asyncio.to_thread(adapter.prepare,context,request,request.capability)
  report['graph']=prepared.payload['graph']
  report['graph_sha256']=hashlib.sha256(json.dumps(report['graph'],sort_keys=True).encode()).hexdigest()
  samples=[];sampling=True
  async def telemetry():
   while sampling:
    await asyncio.to_thread(context.backend._gpu_ownership,context.backend._owned_pids())
    samples.append(context.backend.runtime_info()['gpu_processes'])
    await asyncio.sleep(0.2)
  observer=asyncio.create_task(telemetry())
  try:output=await asyncio.to_thread(adapter.infer,context,prepared)
  finally:
   sampling=False;await observer
  report['gpu_samples_during_inference']=samples
  report['prompt_id']=output['prompt_id']
  result=await asyncio.to_thread(adapter.normalize_output,context,prepared,output)
  assert len(result['artifacts'])==1
  data=result['artifacts'][0]['data'];decoded=Image.open(io.BytesIO(data));decoded.load()
  assert decoded.size==(im.width*4,im.height*4) and hashlib.sha256(data).hexdigest()!=INPUT_SHA
  report['output']={'sha256':hashlib.sha256(data).hexdigest(),'size_bytes':len(data),'dimensions':list(decoded.size)}
  Path('/kaggle/working/phase5_enhancement_output.png').write_bytes(data)
  context.backend.health();report['runtime']=context.backend.runtime_info()
  assert report['runtime']['gpu_processes'] and {p['device'] for p in report['runtime']['gpu_processes']}=={0}
  assert registry.active_models()==() and model.validation.state.value=='DISABLED'
  report['public_models']=[];report['model_state']='DISABLED';report['native_inference_executed']=True
  await session.stop();report['stop_verified']=all(b.process.poll() is not None for b in session.backends.values())
 asyncio.run(run())
 report['comfy_revision']=COMFY_REVISION;report['exit_gate']='PASS'
except Exception as exc:
 report['error']={'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
finally:
 if session is not None:
  try:asyncio.run(session.stop())
  except Exception as exc:report['cleanup_error']=str(exc);report['exit_gate']='NOT_MET'
  for log in session.directory.glob('*/backend.log'):shutil.copyfile(log,'/kaggle/working/phase5_enhancement_backend.log')
 Path('/kaggle/working/phase5_enhancement.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps({'gate':report['exit_gate'],'error':report.get('error',{}).get('message'),'native_inference_executed':report['native_inference_executed']}))
if report['exit_gate']!='PASS':raise SystemExit(1)
'''


def collect():
 client=KaggleAPIClient();path=ROOT/'validation/enhancement_probe_run.json';run=json.loads(path.read_text());version=run['response']['versionNumber']
 status=client.get_kernel_status(run['slug'])
 if status['status'] not in {'complete','error','failed'}:print(json.dumps({'status':status['status']}));return
 output=client.get_kernel_output(run['slug']);records=[]
 for item in output.get('files',[]):
  name=Path(item['fileName']).name
  if name not in {'phase5_enhancement.json','phase5_enhancement_backend.log','phase5_enhancement_output.png'}:continue
  with urllib.request.urlopen(item['url'],timeout=60) as response:body=response.read(64*1024*1024+1)
  if len(body)>64*1024*1024:raise ValueError('Oversized native artifact')
  dest=ROOT/'validation'/(f'enhancement_observation_v{version}.json' if name.endswith('.json') else f'phase5_enhancement_v{version}'+Path(name).suffix)
  if dest.exists() and dest.read_bytes()!=body:raise ValueError('Historical observation differs')
  dest.write_bytes(body);records.append({'file':str(dest.relative_to(ROOT)),'sha256':hashlib.sha256(body).hexdigest(),'remote_name':name})
 receipt={'version':version,'kernel_url':run['response']['url'],'run_record_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'kernel_status':status,'artifacts':records}
 (ROOT/f'validation/enhancement_receipt_v{version}.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps({'status':status['status'],'version':version,'artifacts':len(records)}))


def main():
 parser=argparse.ArgumentParser();actions=parser.add_mutually_exclusive_group();actions.add_argument('--run',action='store_true');actions.add_argument('--collect',action='store_true');args=parser.parse_args()
 if args.collect:collect();return
 code=render_kernel()
 if not args.run:print(json.dumps({'kernel_bytes':len(code)}));return
 client=KaggleAPIClient();result=client.push_kernel(slug='factory-phase-5-enhancement-native',title='Factory Phase 5 Enhancement Native',code_content=code,enable_gpu=True,enable_internet=True,is_private=True,accelerator='NvidiaTeslaT4')
 path=ROOT/'validation/enhancement_probe_run.json'
 if path.exists():
  old=json.loads(path.read_text());path.with_name('enhancement_probe_run_v'+str(old['response']['versionNumber'])+'.json').write_text(path.read_text())
 path.write_text(json.dumps({'observed_at_utc':datetime.now(timezone.utc).isoformat(),'response':result,'slug':'factory-phase-5-enhancement-native','kernel_code_sha256':hashlib.sha256(code.encode()).hexdigest()},indent=2)+'\n')
 print(json.dumps({'url':result.get('url'),'version':result.get('versionNumber')}))

if __name__=='__main__':main()
