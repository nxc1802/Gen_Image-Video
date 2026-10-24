"""
factory/apps/control-api/src/queue/state_machine.py
Deterministic Job State Machine and transition validator.
"""
from typing import Dict, Set

from factory_contracts.jobs.job_contract import JobState
from factory_contracts.errors import FactoryError, ErrorCode


class InvalidStateTransitionError(FactoryError):
    """Raised when an illegal lifecycle transition is requested."""

    def __init__(self, current_state: JobState, target_state: JobState, job_id: str = ""):
        super().__init__(
            message=f"Cannot transition job {job_id} from {current_state.value} to {target_state.value}.",
            code=ErrorCode.INVALID_REQUEST.value,
            status_code=400,
            details={
                "current_state": current_state.value,
                "target_state": target_state.value,
                "job_id": job_id,
            },
        )


class JobStateMachine:
    """
    Deterministic Job State Machine enforcing valid transitions:
    CREATED -> QUEUED -> ADMITTED -> RUNNING -> STREAMING -> SUCCEEDED / FAILED / CANCELLED / RETRYING
    """

    # Allowed transitions map: Current State -> Set of allowed Target States
    ALLOWED_TRANSITIONS: Dict[JobState, Set[JobState]] = {
        JobState.CREATED: {JobState.QUEUED, JobState.CANCELLED, JobState.FAILED},
        JobState.QUEUED: {JobState.ADMITTED, JobState.RUNNING, JobState.CANCELLED, JobState.RETRYING, JobState.FAILED},
        JobState.ADMITTED: {JobState.RUNNING, JobState.CANCELLED, JobState.FAILED, JobState.QUEUED},
        JobState.RUNNING: {JobState.STREAMING, JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED, JobState.RETRYING},
        JobState.STREAMING: {JobState.RUNNING, JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED, JobState.RETRYING},
        JobState.SUCCEEDED: set(),  # Terminal: immutable
        JobState.FAILED: {JobState.RETRYING},  # Can be retried
        JobState.CANCELLED: {JobState.CANCELLED, JobState.RETRYING},  # Idempotent cancel, or retry
        JobState.RETRYING: {JobState.QUEUED, JobState.FAILED, JobState.CANCELLED},
    }

    TERMINAL_STATES: Set[JobState] = {
        JobState.SUCCEEDED,
        JobState.FAILED,
        JobState.CANCELLED,
    }

    @classmethod
    def can_transition(cls, current: JobState, target: JobState) -> bool:
        if current == target:
            # Idempotent same-state transitions (e.g. CANCELLED -> CANCELLED, RUNNING -> RUNNING)
            return True
        allowed = cls.ALLOWED_TRANSITIONS.get(current, set())
        return target in allowed

    @classmethod
    def validate_transition(cls, current: JobState, target: JobState, job_id: str = "") -> None:
        if current == target:
            return
        if not cls.can_transition(current, target):
            raise InvalidStateTransitionError(current, target, job_id)

    @classmethod
    def is_terminal(cls, state: JobState) -> bool:
        return state in cls.TERMINAL_STATES
