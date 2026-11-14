"""Request aliases normalize once; workflow/queue code receives canonical IDs."""
from unittest.mock import AsyncMock, MagicMock
import pytest
from factory_contracts.capabilities import CAPABILITIES, normalize_capability
from factory_contracts.tasks import CANONICAL_TASK_MAP, normalize_task
from factory_contracts.jobs import JobSubmissionRequest
from factory.apps.control_api.src.queue.manager import JobLifecycleManager
from workflows.image.registry import ImageWorkflowRegistry
from workflows.video.registry import VideoWorkflowRegistry


@pytest.mark.parametrize('cid', list(CAPABILITIES))
def test_every_registered_alias_resolves_with_its_engine(cid):
    item = CAPABILITIES[cid]
    for alias in (item.id, *item.aliases):
        assert normalize_task(alias, engine=item.engine) == cid
        assert normalize_task('  '+alias.upper()+'\t', engine=item.engine) == cid


def test_unambiguous_compatibility_map_is_read_only_and_derived():
    assert CANONICAL_TASK_MAP['image.t2i'] == 'image.text_to_image'
    with pytest.raises(TypeError):
        CANONICAL_TASK_MAP['t2i'] = 'another.id'


def test_ambiguous_short_alias_requires_context():
    for alias in ['reference', 'reference_conditioning', 'edit', 'score']:
        with pytest.raises(ValueError, match='Ambiguous'):
            normalize_task(alias)
    assert normalize_task('reference', engine='image') == 'image.reference_image'
    assert normalize_task('reference', engine='video') == 'video.reference_conditioning'
    assert normalize_task('reference', engine='music') == 'music.reference_audio'
    assert normalize_task('score', engine='decision') == 'decision.score_ranking'
    assert normalize_task('score', engine='music') == 'music.symbolic_score'


def test_unknown_empty_names_remain_unknown():
    assert normalize_task('  CUSTOM_INFERENCE  ') == 'custom_inference'
    assert normalize_task('') == normalize_task(None) == ''


def test_control_pose_and_depth_do_not_collapse():
    assert normalize_task('control_pose') == 'image.control_pose'
    assert normalize_task('control_depth') == 'image.control_depth'
    assert normalize_capability('image.controlnet', inputs={'control_type':'depth'}) == 'image.control_depth'


@pytest.mark.parametrize('alias', ['t2i', 'image.t2i', 'text2image'])
def test_request_boundary_normalizes_before_workflow_dispatch(alias):
    req = JobSubmissionRequest(service='image', model='qwen-image-2.1', task=alias, inputs={'prompt':'a cat'})
    assert req.task == 'image.text_to_image'
    graph = ImageWorkflowRegistry().get_workflow(req.task)
    assert '474' in graph
    with pytest.raises(ValueError, match='canonical'):
        ImageWorkflowRegistry().get_workflow(alias)


def test_video_registry_accepts_canonical_only_without_cross_model_fallback():
    registry = VideoWorkflowRegistry()
    assert '9' in registry.get_workflow('wan-2.2', 'video.text_to_video')
    with pytest.raises(ValueError, match='canonical'):
        registry.get_workflow('wan-2.2', 't2v')
    with pytest.raises(KeyError, match='fallback'):
        registry.get_workflow('wan-2.2', 'video.video_with_audio')
    assert registry.get_workflow('minimax-h3', 'video.video_with_audio')


@pytest.mark.asyncio
async def test_job_lifecycle_manager_stores_the_normalized_request(routing_registry):
    repository = MagicMock()
    repository.get_by_idempotency_key = AsyncMock(return_value=None)
    repository.create_job = AsyncMock()
    repository.count_active_and_queued = AsyncMock(return_value=(0, 1))
    broker = MagicMock()
    broker.enqueue = AsyncMock()
    stream = MagicMock()
    stream.publish = AsyncMock()
    manager = JobLifecycleManager(repository=repository, broker=broker, stream_manager=stream, deployment_registry=routing_registry([("qwen-image-2.1", [0])]))
    req = JobSubmissionRequest(service='image', model='qwen-image-2.1', task='image.t2i', inputs={'prompt':'a cat'})
    job, created = await manager.submit_job(req, request_id='req-12345678-1234-1234-1234-123456789abc')
    assert created and job.task == 'image.text_to_image'
    assert repository.create_job.call_args[0][0].task == req.task
