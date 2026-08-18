"""Local open-weight adapter via an OpenAI-compatible endpoint (e.g. vLLM's
`--api-key` OpenAI-compatible server, v0.5.4 as used in the paper's Section
7.1 for Llama-3-70B-Instruct, Qwen2-72B-Instruct, Llama-3-8B-Instruct, and
Mistral-7B-Instruct-v0.3).

This reuses the OpenAI wire protocol (vLLM's serving layer implements it),
just pointed at a local base_url instead of api.openai.com.
"""

from __future__ import annotations

import os
from typing import Optional

from alef.providers.openai_provider import OpenAIProvider


class VLLMProvider(OpenAIProvider):
    """Identical request/response shape to OpenAIProvider, but targets a
    local (or self-hosted) OpenAI-compatible base_url, e.g.
    http://localhost:8000/v1 for a vLLM server."""

    def __init__(self, model: str, base_url: Optional[str] = None, api_key: Optional[str] = None):
        try:
            import openai  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The 'openai' package is required for VLLMProvider (used for its "
                "OpenAI-compatible client). Install with: pip install openai"
            ) from e

        from openai import OpenAI

        resolved_base_url = base_url or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        resolved_key = api_key or os.environ.get("VLLM_API_KEY", "EMPTY")

        self.model = model
        self.name = f"vllm:{model}"
        self._client = OpenAI(api_key=resolved_key, base_url=resolved_base_url)
