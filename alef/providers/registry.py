"""Model capability tiers and roster (Section 7.1) plus a factory that
instantiates the right provider adapter for a given model spec.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from alef.providers.base import LLMProvider
from alef.providers.mock import MockProvider


class CapabilityTier(str, Enum):
    SOTA_CLOSED = "sota_closed"
    MID_TIER_CLOSED = "mid_tier_closed"
    OPEN_WEIGHT_HEAVY = "open_weight_heavy"
    OPEN_WEIGHT_LIGHT = "open_weight_light"


class ProviderType(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    VLLM = "vllm"
    MOCK = "mock"


@dataclass(frozen=True)
class ModelSpec:
    display_name: str
    tier: CapabilityTier
    provider_type: ProviderType
    model_id: str


# Section 7.1's eight-model roster. Update `model_id` values to whatever
# snapshot/deployment you actually have access to -- these mirror the
# paper's stated identifiers as of its August 2026 trial window.
MODEL_REGISTRY: list[ModelSpec] = [
    ModelSpec("GPT-4o", CapabilityTier.SOTA_CLOSED, ProviderType.OPENAI, "gpt-4o-2024-08-06"),
    ModelSpec("Claude 3.5 Sonnet", CapabilityTier.SOTA_CLOSED, ProviderType.ANTHROPIC, "claude-3-5-sonnet-20240620"),
    ModelSpec("GPT-3.5-Turbo", CapabilityTier.MID_TIER_CLOSED, ProviderType.OPENAI, "gpt-3.5-turbo-0125"),
    ModelSpec("Claude 3 Haiku", CapabilityTier.MID_TIER_CLOSED, ProviderType.ANTHROPIC, "claude-3-haiku-20240307"),
    ModelSpec("Llama-3-70B-Instruct", CapabilityTier.OPEN_WEIGHT_HEAVY, ProviderType.VLLM, "meta-llama/Meta-Llama-3-70B-Instruct"),
    ModelSpec("Qwen2-72B-Instruct", CapabilityTier.OPEN_WEIGHT_HEAVY, ProviderType.VLLM, "Qwen/Qwen2-72B-Instruct"),
    ModelSpec("Llama-3-8B-Instruct", CapabilityTier.OPEN_WEIGHT_LIGHT, ProviderType.VLLM, "meta-llama/Meta-Llama-3-8B-Instruct"),
    ModelSpec("Mistral-7B-Instruct-v0.3", CapabilityTier.OPEN_WEIGHT_LIGHT, ProviderType.VLLM, "mistralai/Mistral-7B-Instruct-v0.3"),
]


def get_model_spec(display_name: str) -> ModelSpec:
    for spec in MODEL_REGISTRY:
        if spec.display_name == display_name:
            return spec
    raise KeyError(f"Unknown model: {display_name!r}. Known models: {[s.display_name for s in MODEL_REGISTRY]}")


def get_provider(name: str, **kwargs) -> LLMProvider:
    """Factory: `name` is either 'mock' (mode=... optional kwarg), or a
    display_name from MODEL_REGISTRY (e.g. 'GPT-4o'), or a raw provider:model
    string (e.g. 'openai:gpt-4o-2024-08-06', 'vllm:Qwen/Qwen2-72B-Instruct').
    """
    if name == "mock":
        return MockProvider(mode=kwargs.get("mode", "action_silent"))

    if ":" in name:
        provider_type_str, model_id = name.split(":", 1)
        provider_type = ProviderType(provider_type_str)
    else:
        spec = get_model_spec(name)
        provider_type = spec.provider_type
        model_id = spec.model_id

    if provider_type == ProviderType.OPENAI:
        from alef.providers.openai_provider import OpenAIProvider
        return OpenAIProvider(model=model_id, api_key=kwargs.get("api_key"))
    if provider_type == ProviderType.ANTHROPIC:
        from alef.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(model=model_id, api_key=kwargs.get("api_key"))
    if provider_type == ProviderType.VLLM:
        from alef.providers.vllm_provider import VLLMProvider
        return VLLMProvider(model=model_id, base_url=kwargs.get("base_url"), api_key=kwargs.get("api_key"))

    raise ValueError(f"Unhandled provider type: {provider_type!r}")
