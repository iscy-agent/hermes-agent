"""Turn-process ownership and reap for API-server agent lifecycles (topical sibling of
``api_server.py``).

API-server turns bypass ``TurnRunner``, so they record process ownership themselves
and reap the baseline diff on disconnect/stop (mirrors the gateway-turn cleanup in
``gateway/run.py``, #76115).
"""

import itertools
import threading
from contextvars import copy_context
from typing import Any, Optional

# Per-task-id run epochs for the reap gate: monotonic counter (never reused),
# pruned on clear while still current, so the dict is bounded to in-flight runs.
_TURN_PROCESS_EPOCHS: dict[str, int] = {}
_TURN_PROCESS_EPOCH_LOCK = threading.Lock()
_TURN_PROCESS_EPOCH_COUNTER = itertools.count(1)


def _reap_disconnected_agent_processes(
    agent: Any, *, source: str = "api_server_sse_disconnect") -> None:
    """Reap background processes an abandoned API-server turn created (these turns bypass
    ``TurnRunner``). Daemon-thread fire-and-forget; epoch-gated so a stale reaper never kills
    a newer run's process on a shared task_id."""
    process_task_id = getattr(agent, "_gateway_turn_process_task_id", "")
    process_baseline = getattr(agent, "_gateway_turn_process_baseline", None)
    if not process_task_id or process_baseline is None:
        return
    epoch = getattr(agent, "_gateway_turn_process_epoch", None)
    is_still_current: Optional[Any] = None
    if epoch is not None:
        def _epoch_still_current(_task_id=process_task_id, _epoch=epoch):
            # Skip only when a NEWER run claimed this task_id. A missing entry means
            # our own clear pruned it — no newer claimant, so the reap must proceed.
            with _TURN_PROCESS_EPOCH_LOCK:
                current = _TURN_PROCESS_EPOCHS.get(_task_id)
            return current is None or current == _epoch
        is_still_current = _epoch_still_current
    from gateway.run import _reap_gateway_turn_processes
    threading.Thread(
        target=copy_context().run, args=(_reap_gateway_turn_processes, process_task_id, process_baseline),
        kwargs={"source": source, "is_still_current": is_still_current},
        name=f"api-turn-reaper-{process_task_id[:12]}", daemon=True).start()


def _publish_turn_process_ownership(agent: Any, task_id: str) -> None:
    """Snapshot the process baseline and claim the task_id's epoch — the single place every
    API-server agent lifecycle records turn ownership (marker names cannot drift)."""
    from tools.process_registry import process_registry
    with _TURN_PROCESS_EPOCH_LOCK:
        epoch = next(_TURN_PROCESS_EPOCH_COUNTER)
        _TURN_PROCESS_EPOCHS[task_id] = epoch
    agent._gateway_turn_process_task_id = task_id
    agent._gateway_turn_process_baseline = process_registry.snapshot_running_ids(task_id)
    agent._gateway_turn_process_epoch = epoch


def _clear_turn_process_ownership(agent: Any) -> None:
    """Clear turn ownership as soon as the turn ends: a later disconnect/cancel must not reap
    background work the turn deliberately left running (same guard as gateway/run.py)."""
    task_id = getattr(agent, "_gateway_turn_process_task_id", "")
    epoch = getattr(agent, "_gateway_turn_process_epoch", None)
    if task_id and epoch is not None:
        with _TURN_PROCESS_EPOCH_LOCK:
            # Prune only when this run is still the current claimant; a
            # newer concurrent run owns the entry otherwise.
            if _TURN_PROCESS_EPOCHS.get(task_id) == epoch:
                del _TURN_PROCESS_EPOCHS[task_id]
    agent._gateway_turn_process_task_id = ""
    agent._gateway_turn_process_baseline = frozenset()
    agent._gateway_turn_process_epoch = None
