"""Real Anthropic adapter (e.g. claude-3-5-sonnet-20240620,
claude-3-haiku-20240307 -- Section 7.1).

Requires `pip install alef-audit[anthropic]` (or `pip install anthropic`)
and the ANTHROPIC_API_KEY environment variable. Only imported/instantiated
when actually selected.
"""

from __future__ import annotations

import os
from typing import Optional

from alef.harness.tools import TOOL_REGISTRY
from alef.providers.base import AgentStepResult, LLMProvider
from alef.trajectory import ModelOutput, Observation, ToolCall


def _anthropic_tool_schemas(tool_names: list[str]) -> list[dict]:
    schemas = []
    for name in tool_names:
        spec = TOOL_REGISTRY[name]
        schemas.append(
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": {
                    "type": "object",
                    "properties": spec.json_schema_properties,
                    "required": list(spec.required_params),
                },
            }
        )
    return schemas


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str, api_key: Optional[str] = None):
        try:
            import anthropic  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The 'anthropic' package is required for AnthropicProvider. "
                "Install with: pip install alef-audit[anthropic]"
            ) from e

        from anthropic import Anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Export it or pass api_key= explicitly."
            )
        self.model = model
        self.name = f"anthropic:{model}"
        self._client = Anthropic(api_key=key)

    def complete(self, *, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> ModelOutput:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = "".join(block.text for block in resp.content if getattr(block, "type", None) == "text")
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
        tools = _anthropic_tool_schemas(available_tools) if available_tools else []

        resp = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            tools=tools,
        )

        text_parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        tool_call: Optional[ToolCall] = None
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use":
                tool_call = ToolCall(fn=block.name, params=dict(block.input))
                break

        return AgentStepResult(
            model_output=ModelOutput(text="".join(text_parts), raw=resp), tool_call=tool_call
        )
