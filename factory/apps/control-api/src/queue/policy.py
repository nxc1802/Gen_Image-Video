"""Deployment-bound placement. No subsystem placement or default device."""
from factory_contracts.errors import ModelNotLoadedError


def binding_for_job(registry, job):
    binding = registry.assignment(job.model)
    if job.deployment_config_sha256 and job.deployment_config_sha256 != registry.config_sha256:
        raise ModelNotLoadedError(message="Queued job belongs to a different deployment configuration")
    if job.deployment_session_id and job.deployment_session_id != registry.session_id:
        raise ModelNotLoadedError(message="Queued job belongs to an earlier runtime session")
    if job.devices and tuple(job.devices) != binding.devices:
        raise ModelNotLoadedError(message="Queued job GPU binding differs from the configured deployment")
    return binding
