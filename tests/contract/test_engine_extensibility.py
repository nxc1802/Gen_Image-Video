"""Phase 5 architecture exit contracts, explicitly mocked and never inference evidence."""
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import yaml
from factory_contracts.deployment import DeploymentConfig,DeploymentEntry
from factory.core.model_adapter import ModelAdapter,PreparedCall
from factory.core.media import MediaRequest
from model_registry.deployment import ModelCatalog,DeploymentRegistry

pytestmark=pytest.mark.contract_mock
ROOT=Path(__file__).resolve().parents[2]
CASES=[("vector","VectorEngine","qwen-embedding","vector.text_embedding"),
 ("decision","DecisionEngine","qwen-decision","decision.predicate_evaluation"),
 ("responses","ResponsesEngine","qwen-text","responses.chat_completion"),
 ("speech","SpeechEngine","qwen-asr","speech.asr"),("image","ImageEngine","qwen-image-2.1","image.text_to_image"),
 ("video","VideoEngine","wan-2.2","video.text_to_video"),("music","MusicEngine","ace-step-1.5","music.text_to_music"),
 ("enhancement","EnhancementEngine","enhancement-suite-v1","enhancement.image_super_resolution")]


@pytest.mark.parametrize("engine,class_name,template,capability",CASES,ids=[row[0] for row in CASES])
def test_model_addition_without_core_change(tmp_path,monkeypatch,engine,class_name,template,capability):
    module=importlib.import_module("factory.engines."+engine)
    core=Path(module.__file__).read_bytes()
    data=yaml.safe_load((ROOT/f"manifests/models/{template}.yaml").read_text())
    model_id="contract-new-"+engine
    adapter_id="contract_plugin_"+engine
    data["metadata"]["id"]=model_id;data["aliases"]=[];data["backend"]["adapter"]=adapter_id
    data["capabilities"]={capability:True};data["validation"]={"state":"EXPERIMENTAL","reasons":["Software fixture only"],"implemented_capabilities":[]}
    directory=tmp_path/"catalog/models";directory.mkdir(parents=True)
    (directory/"new.yaml").write_text(yaml.safe_dump(data))
    calls=[]
    class Plugin(ModelAdapter):
        capabilities=frozenset({capability})
        supported_backends=frozenset(data["backend"]["supported"])
        def validate_request(self,model,request,selected):
            assert model.metadata.id==model_id and selected==capability
            calls.append("validate")
        def prepare(self,context,request,selected):
            calls.append("prepare");return PreparedCall(selected,{"contract_only":True})
        def infer(self,context,prepared):
            assert context.binding.devices==(0,) and context.binding.model==model_id
            calls.append("infer_contract_hook");return {"contract_hook":True}
        def normalize_output(self,context,prepared,output):
            assert output=={"contract_hook":True};calls.append("normalize")
            return {"contract_only":True,"model":context.binding.model,"capability":prepared.capability}
    monkeypatch.setitem(sys.modules,"factory.adapters."+adapter_id,SimpleNamespace(Adapter=Plugin))
    registry=DeploymentRegistry(DeploymentConfig(deployments=(DeploymentEntry(model=model_id,devices=(0,)),)),ModelCatalog(directory.parent))
    # No public config can grant this authorization: it is a local test double.
    monkeypatch.setattr(registry,"require_active",lambda *args:None)
    backend=SimpleNamespace(registry=registry,binding=registry.assignment(model_id),health=lambda:True)
    session=SimpleNamespace(registry=registry,backends={model_id:backend},healthy_models=lambda:(model_id,))
    instance=getattr(module,class_name)(registry,session)
    if engine=="vector":
        from factory.engines.vector.contracts import EmbeddingRequest
        result=instance.generate_embeddings(EmbeddingRequest(model=model_id,input="contract input"))
    elif engine=="decision":
        from factory.engines.decision.contracts import DecisionRequest
        result=instance.evaluate(DecisionRequest(model=model_id,task=capability,context="contract context",hypothesis="contract hypothesis"))
    elif engine=="responses":
        from factory.engines.responses.contracts import ChatCompletionRequest,ChatMessage
        result=instance.generate(ChatCompletionRequest(model=model_id,messages=[ChatMessage(role="user",content="contract input")]))
    else:result=instance.run(MediaRequest(model=model_id,capability=capability,inputs={"contract_only":True}))
    assert result=={"contract_only":True,"model":model_id,"capability":capability}
    assert calls[-3:]==["prepare","infer_contract_hook","normalize"]
    assert Path(module.__file__).read_bytes()==core
    assert not registry.active_models()
