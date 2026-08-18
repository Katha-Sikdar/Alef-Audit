"""Dual-Channel Sanity Checking (Section 10.1).

"Decouple conversational generation M_i from argument parsing A_i. A
secondary verification model computes the semantic similarity score
Sim(U_0, theta_i) between the initial user prompt and the generated
tool-parameter dictionary theta_i. Tool calls with Sim(U_0, theta_i) < tau
are blocked prior to execution."

This module implements Sim as a default, dependency-free Jaccard
token-overlap heuristic so the defense is testable offline; pass a custom
`similarity_fn` (e.g. backed by a real embedding model or an LLM judge) for
closer fidelity to the paper's "secondary verification model."
"""

from __future__ import annotations

import re
from typing import Callable

from alef.defenses.base import Defense
from alef.trajectory import ToolCall, Trajectory

SimilarityFn = Callable[[str, dict], float]

_WORD_RE = re.compile(r"[a-zA-Z0-9@._/\-]+")


def default_jaccard_similarity(user_intent: str, params: dict) -> float:
    """Sim(U_0, theta_i) proxy: Jaccard token overlap between the user's
    original intent string and the flattened tool-parameter values."""
    intent_tokens = set(t.lower() for t in _WORD_RE.findall(user_intent))
    param_text = " ".join(str(v) for v in params.values())
    param_tokens = set(t.lower() for t in _WORD_RE.findall(param_text))
    if not intent_tokens or not param_tokens:
        return 0.0
    intersection = intent_tokens & param_tokens
    union = intent_tokens | param_tokens
    return len(intersection) / len(union) if union else 0.0


class DualChannelSanityChecking(Defense):
    name = "dual_channel_sanity_checking"

    def __init__(self, tau: float = 0.08, similarity_fn: SimilarityFn = default_jaccard_similarity):
        self.tau = tau
        self.similarity_fn = similarity_fn

    def filter(self, tool_call: ToolCall, *, user_intent: str, trajectory: Trajectory, step_index: int) -> ToolCall:
        score = self.similarity_fn(user_intent, tool_call.params)
        if score < self.tau:
            return self._block(tool_call)
        return tool_call
