"""Core data model for agentic trajectories.

Implements the formalization of Section 5.1 of the paper:

    T = {(O_1, M_1, A_1), (O_2, M_2, A_2), ..., (O_N, M_N, A_N)}

where O_i is the environment observation at step i, M_i is the generated
natural-language output, and A_i is the executed tool invocation (or None).

Context history accumulation follows Eq. 2:

    H_t = H_{t-1} || Parse(O_t) || M_{t-1} || A_{t-1}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class PageRepresentation(str, Enum):
    """The five structural page representations of Section 6.2 / Table 5."""

    PLAIN_TEXT = "plain_text"
    HTML = "html"
    RAW_HTTP = "raw_http"
    RENDERED_SNAPSHOT = "rendered_snapshot"
    ACCESSIBILITY_TREE = "accessibility_tree"


@dataclass
class Observation:
    """An environment observation O_i: the (possibly injected) page content
    the agent perceives at step i, in a given structural representation."""

    page_id: str
    representation: PageRepresentation
    content: str
    injected_payload: Optional[str] = None

    @property
    def is_injected(self) -> bool:
        return self.injected_payload is not None


@dataclass
class ModelOutput:
    """The generated natural-language reasoning/response M_i (the text
    channel), separate from any tool call the model also emits."""

    text: str
    raw: Optional[Any] = None  # provider-native response object, if useful for debugging


@dataclass
class ToolCall:
    """An executed tool invocation A_i = Invoke(fn, theta), Eq. 1.

    `fn` is the tool name (e.g. "send_email"); `params` is the theta
    dictionary of key-value arguments captured at the sandbox API boundary
    (Section 7.5). ToolCall is None-able at the trajectory-step level to
    represent "no tool call issued at this step" (A_i = the empty symbol).
    """

    fn: str
    params: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    blocked_by_defense: Optional[str] = None  # name of the defense that vetoed this call, if any


@dataclass
class TrajectoryStep:
    """One (O_i, M_i, A_i) triple plus the page index it occurred on."""

    step_index: int
    page_id: str
    observation: Observation
    model_output: ModelOutput
    tool_call: Optional[ToolCall] = None


@dataclass
class Trajectory:
    """A full N-step agentic trajectory T (Eq. 3), plus bookkeeping needed
    for cross-layer auditing: which step carried the injection, and the
    user's original intent (used by defenses such as dual-channel sanity
    checking, Section 10.1)."""

    trajectory_id: str
    user_intent: str
    steps: list[TrajectoryStep] = field(default_factory=list)
    injected_step_index: Optional[int] = None
    injected_payload: Optional[str] = None
    model_name: str = ""
    representation: Optional[PageRepresentation] = None

    def add_step(self, step: TrajectoryStep) -> None:
        if step.observation.is_injected:
            self.injected_step_index = step.step_index
            self.injected_payload = step.observation.injected_payload
        self.steps.append(step)

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    def steps_after_injection(self) -> list[TrajectoryStep]:
        """Steps strictly after the injection point (P4, P5 in the paper's
        five-page design) -- the population over which context contamination
        (gamma_CS, Section 5.3) is measured."""
        if self.injected_step_index is None:
            return []
        return [s for s in self.steps if s.step_index > self.injected_step_index]

    def context_history_text(self, up_to_step: int) -> str:
        """Reconstruct a linear textual approximation of H_t (Eq. 2) by
        concatenating parsed observations and prior model outputs/tool calls
        up to (and including) `up_to_step`. This is a convenience view used
        by rule-based auditors and defenses; provider adapters may maintain
        their own richer native context instead."""
        parts: list[str] = []
        for s in self.steps:
            if s.step_index > up_to_step:
                break
            parts.append(f"[O{s.step_index}:{s.page_id}] {s.observation.content}")
            parts.append(f"[M{s.step_index}] {s.model_output.text}")
            if s.tool_call is not None:
                parts.append(f"[A{s.step_index}] {s.tool_call.fn}({s.tool_call.params})")
        return "\n".join(parts)
