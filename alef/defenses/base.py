"""Base interface for Stage 5 execution-layer defenses (Section 6.5, 10.1).

Each defense is a pre-execution filter on a proposed tool call A_i: it may
pass the call through unchanged, or veto it by returning a copy with
`blocked_by_defense` set to the defense's name. Vetoed calls are never
executed by `alef.harness.tools.SandboxToolExecutor`, and
`alef.auditors.SchemaActionComplianceAuditor` treats any tool call with
`blocked_by_defense` set as automatically action-noncompliant -- this is
exactly what lets `alef.stats.metrics` recompute "Defended AC (%)" the same
way the paper's Table 9 does: re-run the Stage-4 auditor with each filter
active (Section 6.5).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import replace

from alef.trajectory import ToolCall, Trajectory


class Defense(ABC):
    name: str = "base_defense"

    @abstractmethod
    def filter(self, tool_call: ToolCall, *, user_intent: str, trajectory: Trajectory, step_index: int) -> ToolCall:
        """Return `tool_call` unchanged to allow it, or a blocked copy to veto it."""
        raise NotImplementedError

    def _block(self, tool_call: ToolCall) -> ToolCall:
        return replace(tool_call, blocked_by_defense=self.name)


def apply_defenses(
    tool_call: ToolCall | None,
    defenses: list[Defense],
    *,
    user_intent: str,
    trajectory: Trajectory,
    step_index: int,
) -> ToolCall | None:
    """Run a proposed tool call through a chain of defenses in order. The
    first defense to block it wins; later defenses are not consulted (this
    matches the paper's framing of each defense being evaluated as an
    independent pre-execution filter in Table 9, but also lets a caller
    stack multiple defenses deliberately for a layered-defense experiment)."""
    if tool_call is None:
        return None
    for defense in defenses:
        tool_call = defense.filter(tool_call, user_intent=user_intent, trajectory=trajectory, step_index=step_index)
        if tool_call.blocked_by_defense is not None:
            return tool_call
    return tool_call
