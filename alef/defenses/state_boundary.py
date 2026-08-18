"""State Boundary Isolation (Section 10.1).

"Introduce mandatory memory-flushing protocols when navigating across
distinct domain boundaries (e.g., clearing context history H_t when moving
from an unverified public URL to an authenticated private API boundary)."

Operationalized here as: a tool call is blocked if any of its string
parameter values do not appear anywhere in the *current* step's own
observation content. Intuitively, if a memory-flushing boundary were
correctly enforced, a tool call issued while processing page P_{k+n} could
only be justified by content actually present on P_{k+n} -- a parameter
value that only exists in earlier (flushed) context is, by construction,
evidence of exactly the cross-page contamination gamma_CS measures
(alef/contamination.py).
"""

from __future__ import annotations

from alef.defenses.base import Defense
from alef.trajectory import ToolCall, Trajectory


class StateBoundaryIsolation(Defense):
    name = "state_boundary_isolation"

    def __init__(self, min_value_length: int = 4):
        # Ignore trivially short parameter values (e.g. short flags) when
        # deciding whether a value was "carried over" from flushed context.
        self.min_value_length = min_value_length

    def filter(self, tool_call: ToolCall, *, user_intent: str, trajectory: Trajectory, step_index: int) -> ToolCall:
        current_step = next((s for s in trajectory.steps if s.step_index == step_index), None)
        if current_step is None:
            return tool_call
        current_content = current_step.observation.content

        for value in tool_call.params.values():
            value_str = str(value)
            if len(value_str) < self.min_value_length:
                continue
            if value_str not in current_content:
                return self._block(tool_call)
        return tool_call
