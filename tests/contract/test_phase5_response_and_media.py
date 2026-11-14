"""Protocol and failure contracts only. All SDK/HTTP outputs here are explicit mocks."""
import asyncio
import base64
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import wave
import httpx
import pytest
from factory.core.model_adapter import ExecutionContext,load_adapter
from factory.core.media import MediaRequest,native_artifacts
from factory.engines.responses import ResponsesEngine
from factory.engines.responses.contracts import ChatCompletionRequest,ChatMessage,ResponseApiRequest
from factory_contracts.errors import ModelUnavailableError,ModelNotLoadedError,InvalidRequestError,UnsupportedCapabilityError

pytestmark=pytest.mark.contract_mock


def context_for(registry,model):
    binding=registry.assignment(model)
    backend=SimpleNamespace(registry=registry,binding=binding,health=lambda:True,token="contract-private-key")
    session=SimpleNamespace(registry=registry,backends={model:backend},healthy_models=lambda:(model,))
    return ExecutionContext(registry,session,binding,registry.catalog.get(model))


def completion(model):
    return {"id":"contract-id","created":123,"object":"chat.completion","model":model,
        "choices":[{"index":0,"message":{"role":"assistant","content":"Contract-only backend response"},"finish_reason":"stop"}],
        "usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}}


def test_text_and_vision_alias_one_engine():
    from services.text.src.engine import TextEngine
    from services.vlm.src.engine import VLMEngine
    assert TextEngine is VLMEngine is ResponsesEngine


@pytest.mark.parametrize("model",["qwen-text","qwen-vlm"])
def test_healthy_backend_never_authorizes_responses(routing_registry,model):
    registry=routing_registry([(model,[0,1] if model=="qwen-text" else [0])])
    ctx=context_for(registry,model)
    with pytest.raises(ModelNotLoadedError):
        ResponsesEngine(registry,ctx.session).generate(ChatCompletionRequest(model=model,messages=[ChatMessage(role="user",content="contract")]))


@pytest.mark.parametrize("field",["id","created","usage","choices"])
def test_response_does_not_manufacture_missing_fields(routing_registry,field):
    registry=routing_registry([("qwen-text",[0,1])]);ctx=context_for(registry,"qwen-text");adapter=load_adapter(ctx.binding.adapter)
    req=ChatCompletionRequest(model="qwen-text",messages=[ChatMessage(role="user",content="contract")]);prepared=adapter.prepare(ctx,req,"responses.chat_completion")
    raw=completion("qwen-text");raw.pop(field)
    with pytest.raises(ModelUnavailableError):adapter.normalize_output(ctx,prepared,raw)


def test_native_response_shape_and_usage_preserved(routing_registry):
    registry=routing_registry([("qwen-text",[0,1])]);ctx=context_for(registry,"qwen-text");adapter=load_adapter(ctx.binding.adapter)
    req=ResponseApiRequest(model="qwen-text",input="contract");prepared=adapter.prepare(ctx,req,"responses.create")
    raw={"id":"native-contract","created_at":123,"object":"response","status":"completed","model":"qwen-text","output":[{"type":"message","content":[{"type":"output_text","text":"contract"}]}],"usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}
    assert adapter.normalize_output(ctx,prepared,raw) is raw


@pytest.mark.parametrize("records",[
    ['data: {"model":"wrong-model"}'],
    ['data: [DONE]'],
    ['data: not-json'],
    ['data: '+json.dumps({"id":"x","object":"chat.completion.chunk","model":"qwen-text","choices":[{"delta":{"content":"part"},"finish_reason":None}]})],
])
def test_corrupt_or_truncated_stream_has_no_fabricated_done(routing_registry,monkeypatch,records):
    registry=routing_registry([("qwen-text",[0,1])]);ctx=context_for(registry,"qwen-text");adapter=load_adapter(ctx.binding.adapter)
    req=ChatCompletionRequest(model="qwen-text",messages=[ChatMessage(role="user",content="contract")],stream=True);prepared=adapter.prepare(ctx,req,"responses.chat_completion")
    actual=httpx.AsyncClient
    transport=httpx.MockTransport(lambda request:httpx.Response(200,text="\n\n".join(records)+"\n\n"))
    monkeypatch.setattr("factory.adapters.response_protocol.httpx.AsyncClient",lambda **kwargs:actual(transport=transport,**kwargs))
    received=[]
    async def run():
        async for value in adapter.stream(ctx,prepared):received.append(value)
    with pytest.raises(ModelUnavailableError):asyncio.run(run())
    assert not any('[DONE]' in value for value in received)


def test_vision_remote_url_is_not_forwarded_to_backend(routing_registry):
    registry=routing_registry([("qwen-vlm",[0])]);ctx=context_for(registry,"qwen-vlm");adapter=load_adapter(ctx.binding.adapter)
    req=ChatCompletionRequest(model="qwen-vlm",messages=[ChatMessage(role="user",content=[{"type":"image_url","image_url":{"url":"http://127.0.0.1/secret"}}])])
    with pytest.raises(InvalidRequestError):adapter.validate_request(ctx.model,req,"responses.image_chat")


def test_native_artifact_path_cannot_escape_session(routing_registry,tmp_path):
    registry=routing_registry([("wan-2.2",[0])]);ctx=context_for(registry,"wan-2.2");ctx.backend.directory=tmp_path
    (tmp_path.parent/"outside-contract.mp4").write_bytes(b"not a model output")
    with pytest.raises(ModelUnavailableError):native_artifacts(ctx,{"model":"wan-2.2","artifacts":[{"filename":"../../outside-contract.mp4","type":"video","mime_type":"video/mp4"}]})


@pytest.mark.parametrize("model,capability,inputs",[
 ("qwen-asr","speech.asr",{"audio":"file:///etc/passwd"}),
 ("qwen-tts","speech.tts",{"text":"contract"}),
 ("qwen-tts","speech.voice_cloning",{"text":"contract","reference_audio":"https://external.invalid/audio.wav","reference_text":"contract"}),
 ("qwen-image-2.1","image.text_to_image",{"prompt":"contract","variant":"turbo"}),
 ("minimax-h3","video.first_last_frame_to_video",{"prompt":"contract","image":"data:test;base64,YQ=="}),
 ("ace-step-1.5","music.text_to_music",{"prompt":"contract","backend":"other-backend"}),
 ("yue2","music.song_variation",{"prompt":"contract"}),
])
def test_media_invalid_inputs_do_not_choose_another_path(routing_registry,model,capability,inputs):
    registry=routing_registry([(model,[0])]);ctx=context_for(registry,model);adapter=load_adapter(ctx.binding.adapter)
    with pytest.raises(InvalidRequestError):adapter.validate_request(ctx.model,MediaRequest(model=model,capability=capability,inputs=inputs),capability)


def test_turbo_graph_uses_only_its_manifest_lora(routing_registry,monkeypatch):
    registry=routing_registry([("minimax-h3-turbo",[0])]);ctx=context_for(registry,"minimax-h3-turbo");adapter=load_adapter(ctx.binding.adapter)
    req=MediaRequest(model=ctx.binding.model,capability="video.text_to_video",inputs={"prompt":"contract","num_frames":107})
    prepared=adapter.prepare(ctx,req,req.capability)
    assert prepared.payload["graph"]["lora"]["inputs"]["lora_name"]==Path(ctx.model.weights.files["turbo_lora"].path).name
    assert "audio" not in prepared.payload["graph"]


def test_shared_device_execution_is_serialized_across_engines(routing_registry):
    from factory.core.execution import execution_scope
    from concurrent.futures import ThreadPoolExecutor
    import threading,time
    registry=routing_registry([("qwen-embedding",[0]),("qwen-decision",[0])])
    contexts=[context_for(registry,model) for model in ["qwen-embedding","qwen-decision"]]
    state={"running":0,"peak":0};guard=threading.Lock();barrier=threading.Barrier(2)
    def work(context):
        barrier.wait()
        with execution_scope(context):
            with guard:state["running"]+=1;state["peak"]=max(state["peak"],state["running"])
            time.sleep(0.02)
            with guard:state["running"]-=1
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(work,contexts))
    assert state["peak"]==1 and all(ctx.binding.devices==(0,) for ctx in contexts)


def test_native_runtime_rejects_wrong_auth_model_and_capability(tmp_path,monkeypatch):
    from model_registry.backends import native_server
    from fastapi.testclient import TestClient
    token=tmp_path/"key";token.write_text("contract-private-key")
    # NumPy is supplied by the real ASR backend SDK, not the lightweight local
    # control-plane environment. This shape-only double is a software contract.
    import sys
    class ContractArray:
        def __init__(self,data):self.shape=(len(data)//2,)
        def astype(self,dtype):return self
        def __truediv__(self,value):return self
    monkeypatch.setitem(sys.modules,"numpy",SimpleNamespace(frombuffer=lambda data,**kwargs:ContractArray(data),int16="int16",float32="float32"))
    loaded=SimpleNamespace(transcribe=Mock(return_value=[SimpleNamespace(text="Mock transcript: software contract only",language="English")]))
    monkeypatch.setattr(native_server,"load_sdk",lambda config:loaded)
    config={"adapter":"qwen_asr","sdk":"qwen_asr","model":"qwen-asr","token_file":str(token),"boot_nonce":"contract-nonce",
            "checkpoint_graph_sha256":"contract-no-weights","capabilities":["speech.asr"]}
    wav=io.BytesIO()
    with wave.open(wav,"wb") as writer:writer.setnchannels(1);writer.setsampwidth(2);writer.setframerate(16000);writer.writeframes(b"\0\0"*160)
    payload={"model":"qwen-asr","capability":"speech.asr","inputs":{"audio":base64.b64encode(wav.getvalue()).decode()}}
    headers={"Authorization":"Bearer contract-private-key"}
    with TestClient(native_server.create_app(config)) as client:
        assert client.post('/_factory/infer',json=payload).status_code==401
        assert client.post('/_factory/infer',json={**payload,"model":"other"},headers=headers).status_code==400
        assert client.post('/_factory/infer',json={**payload,"capability":"speech.tts"},headers=headers).status_code==400
        response=client.post('/_factory/infer',json=payload,headers=headers)
        assert response.status_code==200 and response.json()['text'].startswith('Mock transcript')
        args=loaded.transcribe.call_args.kwargs
        assert args['audio'][0].shape==(160,) and args['audio'][1]==16000
        loaded.transcribe.side_effect=RuntimeError('contract SDK unavailable')
        failure=client.post('/_factory/infer',json=payload,headers=headers)
        assert failure.status_code==503 and 'text' not in failure.json()


def test_legacy_entrypoint_refuses_production(monkeypatch):
    from factory.core.legacy import forbid_legacy_production
    monkeypatch.setenv('ENVIRONMENT','production')
    with pytest.raises(ModelNotLoadedError):forbid_legacy_production()
