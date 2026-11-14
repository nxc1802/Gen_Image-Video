"""Per-session locks on the preconfigured physical devices. No routing decisions."""

from contextlib import contextmanager, asynccontextmanager
import asyncio
from threading import Lock
from weakref import WeakKeyDictionary

_guard = Lock()
_sessions = WeakKeyDictionary()


def device_locks(context):
    with _guard:
        locks = _sessions.setdefault(context.registry, {})
        return [locks.setdefault(device, Lock()) for device in sorted(context.binding.devices)]


@contextmanager
def execution_scope(context):
    locks = device_locks(context)
    acquired = []
    try:
        for lock in locks:
            lock.acquire()
            acquired.append(lock)
        yield
    finally:
        for lock in reversed(acquired):
            lock.release()


@asynccontextmanager
async def async_execution_scope(context):
    """Wait without occupying executor threads; cancellation releases partial locks."""
    acquired = []
    try:
        for lock in device_locks(context):
            while not lock.acquire(blocking=False):
                await asyncio.sleep(0.01)
            acquired.append(lock)
        yield
    finally:
        for lock in reversed(acquired):
            lock.release()


async def finish_native_call(function):
    """Keep the device reservation until the actual native call ends.

    Cancelling an asyncio wrapper cannot stop a Python executor thread or a
    remote SDK already running on the GPU. Drain it and discard its result;
    shutdown stops the owned backend to interrupt outstanding RPCs promptly.
    """
    worker = asyncio.create_task(asyncio.to_thread(function))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        if worker.done() and not worker.cancelled():
            worker.exception()
        raise
