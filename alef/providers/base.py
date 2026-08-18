"""Abstract LLM provider interface.

Every provider (mock or real) implements two methods:

- `complete`: a plain system+user -> text completion, used by the
  LLM-as-judge auditor (alef/judge.py).
- `act`: one agentic trajectory step. Given the running context, the model
  either narrates text only, or narrates text *and* emits a tool call --
  mirroring the dual-channel split of Figure 1 in the paper (the natural
  -language channel M_i and the tool-calling parameter channel theta_i).

Real adapters (OpenAI, Anthropic, vLLM) never receive the ground-truth
IPIPayload -- they only ever see whatever text is embedded in the page
observation, exactly like the black-box agents the paper measures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from alef.trajectory import ModelOutput, Observation, ToolCall


@dataclass
class AgentStepResult:
    model_output: ModelOutput
    tool_call: Optional[ToolCall] = None


class LLMProvider(ABC):
    """Common interface implemented by every model backend."""

    name: str = "base"

    @abstractmethod
    def complete(self, *, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> ModelOutput:
        """Plain text completion, used for LLM-as-judge auditing."""
        raise NotImplementedError

    @abstractmethod
    def act(
        self,
        *,
        system_prompt: str,
        user_intent: str,
        observation: Observation,
        context_history: str,
        available_tools: list[str],
        temperature: float = 0.0,
    ) -> AgentStepResult:
        """One agentic trajectory step: produce narrated text and, optionally,
        a tool call, given the current observation and accumulated context."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{self.__class__.__name__} name={self.name!r}>"
