"""Phase 5 vector software contracts. All backend outputs/GPU health are mocked."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import httpx
import pytest
import yaml

from factory_contracts.deployment import DeploymentConfig, DeploymentEntry
from factory_contracts.errors import ModelNotLoadedError, ModelUnavailableError, UnsupportedCapabilityError
from model_registry.deployment import DeploymentRegistry, ModelCatalog
from factory.core.model_adapter import ExecutionContext, load_adapter
from factory.engines.vector import VectorEngine
from factory.engines.vector.contracts import EmbeddingRequest, RerankRequest

pytestmark = pytest.mark.contract_mock
ROOT = Path(__file__).resolve().parents[2]


def mocked_session(registry):
    # Never a real process, model or GPU. No observation is written to validation/.
    backends = {binding.model:SimpleNamespace(registry=registry,binding=binding,token="contract-private-key",health=lambda:True) for binding in registry.bindings()}
    return SimpleNamespace(registry=registry,backends=backends,healthy_models=lambda:tuple(backends))


@pytest.fixture
def vector_runtime(routing_registry):
    registry = routing_registry([("qwen-embedding",[0]),("qwen-reranker",[1])])
    session = mocked_session(registry)
    return registry, session


def embedding_output(model="qwen-embedding", dimensions=1024):
    return {"object":"list","model":model,"data":[{"object":"embedding","index":0,"embedding":[0.25]*dimensions}],"usage":{"prompt_tokens":3,"total_tokens":3}}


def test_healthy_but_uncertified_model_never_reaches_inference(vector_runtime, monkeypatch):
    registry, session = vector_runtime
    def forbidden(*args, **kwargs):
        pytest.fail("Unvalidated model reached an inference transport")
    monkeypatch.setattr(ExecutionContext,"post_json",forbidden)
    with pytest.raises(ModelNotLoadedError):
        VectorEngine(registry,session).generate_embeddings(EmbeddingRequest(model="qwen-embedding",input="contract input"))


def test_embedding_adapter_uses_bound_local_backend_and_manifest_identity(vector_runtime, monkeypatch):
    registry, session = vector_runtime
    # Software-only authorization fixture. Production has no equivalent switch.
    monkeypatch.setattr(registry,"require_active",lambda *args:None)
    seen=[]
    actual_client = httpx.Client
    def handler(request):
        seen.append(request)
        return httpx.Response(200,json=embedding_output())
    options=[]
    def client(**kwargs):
        options.append(kwargs)
        return actual_client(transport=httpx.MockTransport(handler),**kwargs)
    monkeypatch.setattr("factory.core.model_adapter.httpx.Client",client)
    result = VectorEngine(registry,session).generate_embeddings(EmbeddingRequest(model="qwen-embed",input="contract input"))
    assert result.model == "qwen-embedding" and len(result.data[0].embedding)==1024
    assert str(seen[0].url)=="http://127.0.0.1:9200/v1/embeddings"
    assert json.loads(seen[0].content)=={"model":"qwen-embedding","input":["contract input"],"dimensions":1024}
    assert seen[0].headers["authorization"]=="Bearer contract-private-key"
    assert options[0]["follow_redirects"] is False and options[0]["trust_env"] is False


@pytest.mark.parametrize("change", ["wrong_model","missing_usage","duplicate_index","nan","wrong_dimensions","zero_vector","boolean_index"])
def test_embedding_rejects_invalid_backend_results(vector_runtime, change):
    registry,session=vector_runtime
    context=ExecutionContext(registry,session,registry.assignment("qwen-embedding"),registry.catalog.get("qwen-embedding"))
    adapter=load_adapter(context.binding.adapter)
    prepared=adapter.prepare(context,EmbeddingRequest(model="qwen-embedding",input="contract input"),"vector.text_embedding")
    body=embedding_output()
    if change=="wrong_model":body["model"]="other-checkpoint"
    elif change=="missing_usage":body.pop("usage")
    elif change=="duplicate_index":body["data"].append(copy.deepcopy(body["data"][0]))
    elif change=="nan":body["data"][0]["embedding"][0]=float("nan")
    elif change=="wrong_dimensions":body["data"][0]["embedding"].pop()
    elif change=="zero_vector":body["data"][0]["embedding"]=[0.0]*1024
    elif change=="boolean_index":body["data"][0]["index"]=False
    with pytest.raises(ModelUnavailableError):adapter.normalize_output(context,prepared,body)


def test_multimodal_request_cannot_be_downgraded_to_text(vector_runtime):
    registry,session=vector_runtime
    with pytest.raises(UnsupportedCapabilityError):
        VectorEngine(registry,session).generate_embeddings(EmbeddingRequest(model="qwen-embedding",input=[{"image_url":"https://example.com/photo.png"}]))


def test_reranker_preserves_documents_and_never_uses_an_alternate_endpoint(vector_runtime,monkeypatch):
    registry,session=vector_runtime
    monkeypatch.setattr(registry,"require_active",lambda *args:None)
    calls=[]
    def post(context,path,payload):
        calls.append((context.binding.model,context.binding.devices,path,payload))
        return {"model":"qwen-reranker","data":[{"index":0,"score":0.2},{"index":1,"score":0.9}]}
    monkeypatch.setattr(ExecutionContext,"post_json",post)
    request=RerankRequest(model="qwen-reranker",query="contract query",documents=["first",{"text":"second","id":"keep-me"}],top_n=1)
    result=VectorEngine(registry,session).rerank_documents(request)
    assert calls==[("qwen-reranker",(1,),"/score",{"model":"qwen-reranker","text_1":"contract query","text_2":["first","second"]})]
    assert result.results[0].index==1 and result.results[0].document==request.documents[1]


@pytest.mark.parametrize("rows", [[{"index":0,"score":float("nan")}],[{"index":0,"score":2.0}],[{"index":1,"score":0.2}],[{"index":0,"score":0.2},{"index":0,"score":0.7}]])
def test_reranker_rejects_nonfinite_out_of_range_and_wrong_indices(vector_runtime,rows):
    registry,session=vector_runtime
    context=ExecutionContext(registry,session,registry.assignment("qwen-reranker"),registry.catalog.get("qwen-reranker"))
    adapter=load_adapter(context.binding.adapter)
    prepared=adapter.prepare(context,RerankRequest(model="qwen-reranker",query="query",documents=["document"]),"vector.reranking")
    with pytest.raises(ModelUnavailableError):adapter.normalize_output(context,prepared,{"model":"qwen-reranker","data":rows})


def test_another_session_cannot_supply_an_inference_process(vector_runtime,monkeypatch):
    registry,session=vector_runtime
    monkeypatch.setattr(registry,"require_active",lambda *args:None)
    session.registry=object()
    with pytest.raises(ModelUnavailableError,match="exact deployment"):
        VectorEngine(registry,session).generate_embeddings(EmbeddingRequest(model="qwen-embedding",input="contract input"))


def test_new_manifest_adapter_requires_no_vector_engine_change(tmp_path,monkeypatch):
    data=yaml.safe_load((ROOT/"manifests/models/qwen-embedding.yaml").read_text())
    data["metadata"]["id"]="contract-new-vector-model"
    data["aliases"]=[]
    data["backend"]["adapter"]="contract_vector_adapter"
    data["parameters"]["dimensions"]["default"]=4
    data["parameters"]["dimensions"]["minimum"]=1
    directory=tmp_path/"catalog/models";directory.mkdir(parents=True)
    (directory/"model.yaml").write_text(yaml.safe_dump(data))
    from factory.adapters.qwen_embedding import Adapter
    monkeypatch.setitem(sys.modules,"factory.adapters.contract_vector_adapter",SimpleNamespace(Adapter=Adapter))
    registry=DeploymentRegistry(DeploymentConfig(deployments=(DeploymentEntry(model="contract-new-vector-model",devices=(0,)),)),ModelCatalog(directory.parent))
    monkeypatch.setattr(registry,"require_active",lambda *args:None)
    monkeypatch.setattr(ExecutionContext,"post_json",lambda context,path,payload:embedding_output(context.binding.model,payload["dimensions"]))
    result=VectorEngine(registry,mocked_session(registry)).generate_embeddings(EmbeddingRequest(model="contract-new-vector-model",input="software contract"))
    assert result.model=="contract-new-vector-model" and len(result.data[0].embedding)==4


def test_missing_adapter_does_not_select_another_provider():
    with pytest.raises(ModelUnavailableError,match="fallback is forbidden"):
        load_adapter("missing_contract_adapter")


def test_transport_rejects_adapter_model_substitution(vector_runtime):
    registry,session=vector_runtime
    context=ExecutionContext(registry,session,registry.assignment("qwen-embedding"),registry.catalog.get("qwen-embedding"))
    with pytest.raises(ModelUnavailableError,match="model identity"):
        context.post_json("/v1/embeddings",{"model":"other-checkpoint","input":["contract"]})


def test_reranker_launch_uses_pinned_score_template_with_original_weights(tmp_path):
    import hashlib
    from model_registry.adapters.launch import RerankerLaunchAdapter
    backend=SimpleNamespace(directory=tmp_path,adapter_assets={},_files={})
    args=RerankerLaunchAdapter().extra_args(backend)
    overrides=json.loads(args[args.index("--hf-overrides")+1])
    assert overrides["is_original_qwen3_reranker"] and overrides["classifier_from_token"]==["no","yes"]
    template=Path(args[args.index("--chat-template")+1])
    source=json.loads((ROOT/"validation/vector_adapter_asset_source.json").read_text())
    assert hashlib.sha256(template.read_bytes()).hexdigest()==source["bundled_sha256"]
    assert backend.adapter_assets[template.name]==source["bundled_sha256"]
    assert str(template) in backend._files


def test_retained_vector_api_defaults_to_canonical_engine_and_refuses_unvalidated_model(vector_runtime,tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from services.embedding.src.app import app
    registry,_=vector_runtime
    config=tmp_path/"deployment.yaml";config.write_text(yaml.safe_dump(registry.config.model_dump(mode="json")))
    monkeypatch.setenv("FACTORY_DEPLOYMENT_FILE",str(config))
    monkeypatch.setenv("FACTORY_MANIFESTS_DIR",str(ROOT/"manifests"))
    monkeypatch.setenv("ENVIRONMENT","test")
    monkeypatch.setenv("FACTORY_API_KEY","vector-contract-only-key")
    with TestClient(app) as client:
        assert isinstance(app.state.engine,VectorEngine)
        response=client.post("/v1/embeddings",json={"model":"qwen-embedding","input":"software contract"},headers={"Authorization":"Bearer vector-contract-only-key"})
    assert response.status_code==503 and "data" not in response.json()
