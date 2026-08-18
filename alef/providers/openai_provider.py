"""Real OpenAI adapter (used for the SOTA/Mid-Tier Closed tiers, e.g.
gpt-4o-2024-08-06, gpt-3.5-turbo-0125 -- Section 7.1).

Requires `pip install alef-audit[openai]` (or `pip install openai`) and the
OPENAI_API_KEY environment variable. Only imported/instantiated when
actually selected -- the rest of the framework, including the full test
suite, never needs the `openai` package installed.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from alef.harness.tools import openai_tool_schemas
from alef.providers.base import AgentStepResult, LLMProvider
from alef.trajectory import ModelOutput, Observation, ToolCall


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str, api_key: Optional[str] = None):
        try:
            import openai  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The 'openai' package is required for OpenAIProvider. "
                "Install with: pip install alef-audit[openai]"
            ) from e

        from openai import OpenAI

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it or pass api_key= explicitly."
            )
        self.model = model
        self.name = f"openai:{model}"
        self._client = OpenAI(api_key=key)

    def complete(self, *, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> ModelOutput:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = resp.choices[0].message.content or ""
        return ModelOutput(text=text, raw=resp)

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
        user_prompt = (
            f"User task: {user_intent}\n\n"
            f"Context so far:\n{context_history}\n\n"
            f"Current page ({observation.page_id}, {observation.representation.value}):\n"
            f"{observation.content}"
        )
        tools = openai_tool_schemas(available_tools) if available_tools else None

        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=tools,
        )
        message = resp.choices[0].message
        text = message.content or ""
        tool_call: Optional[ToolCall] = None

        if getattr(message, "tool_calls", None):
            call = message.tool_calls[0]
            try:
                params = json.loads(call.function.arguments)
            except (json.JSONDecodeError, TypeError):
                params = {}
            tool_call = ToolCall(fn=call.function.name, params=params)

        return AgentStepResult(model_output=ModelOutput(text=text, raw=resp), tool_call=tool_call)
