"""Sandboxed, real-world-shaped side-effect tools (Section 3.2, Section 7.5).

`send_email`, `save_file`, and `submit_form` mirror the paper's real-world
side-effect tools. Every call is intercepted at an API-hook boundary
*before* any network dispatch would occur: nothing here ever sends a real
email, writes outside an explicit sandbox directory, or hits a real HTTP
endpoint. This matches the paper's ethical design (Section 13): "simulated
exfiltration attempts were intercepted at the API boundary before any
network dispatch."

`navigate` and `click` are included as lightweight tools to support the
Reputation Manipulation objective category (Table 2), which the paper notes
is audited via destination-URL/selector matching rather than strict schema
parameter matching (Table 4 footnote).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from alef.trajectory import ToolCall


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    required_params: tuple[str, ...]
    json_schema_properties: dict[str, Any] = field(default_factory=dict)


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "send_email": ToolSpec(
        name="send_email",
        description="Send an email on the user's behalf.",
        required_params=("to", "body"),
        json_schema_properties={
            "to": {"type": "string", "description": "Recipient email address."},
            "subject": {"type": "string", "description": "Email subject."},
            "body": {"type": "string", "description": "Email body."},
        },
    ),
    "save_file": ToolSpec(
        name="save_file",
        description="Persist extracted data to a file/database path.",
        required_params=("path", "content"),
        json_schema_properties={
            "path": {"type": "string", "description": "Destination path."},
            "content": {"type": "string", "description": "Content to write."},
        },
    ),
    "submit_form": ToolSpec(
        name="submit_form",
        description="Submit a structured form to an endpoint.",
        required_params=("endpoint", "payload"),
        json_schema_properties={
            "endpoint": {"type": "string", "description": "Form submission endpoint."},
            "payload": {"type": "object", "description": "Form field values."},
        },
    ),
    "navigate": ToolSpec(
        name="navigate",
        description="Navigate the browser to a URL.",
        required_params=("url",),
        json_schema_properties={"url": {"type": "string", "description": "Destination URL."}},
    ),
    "click": ToolSpec(
        name="click",
        description="Click a page element by selector.",
        required_params=("selector",),
        json_schema_properties={"selector": {"type": "string", "description": "CSS/AX selector."}},
    ),
}


def openai_tool_schemas(tool_names: Optional[list[str]] = None) -> list[dict[str, Any]]:
    """Render TOOL_REGISTRY entries as OpenAI/Anthropic-style function-calling
    tool schemas, for use by the real provider adapters."""
    names = tool_names or list(TOOL_REGISTRY.keys())
    schemas = []
    for name in names:
        spec = TOOL_REGISTRY[name]
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": {
                        "type": "object",
                        "properties": spec.json_schema_properties,
                        "required": list(spec.required_params),
                    },
                },
            }
        )
    return schemas


class SchemaValidationError(ValueError):
    pass


@dataclass
class SandboxToolExecutor:
    """Intercepts tool calls at the API-hook boundary (Section 7.5): logs a
    structured, schema-validated ToolCall and never performs a real side
    effect. `on_call` is an optional callback invoked with every accepted
    ToolCall, useful for streaming logs during a run.
    """

    on_call: Optional[Callable[[ToolCall], None]] = None
    log: list[ToolCall] = field(default_factory=list)

    def execute(self, fn: str, params: dict[str, Any]) -> ToolCall:
        spec = TOOL_REGISTRY.get(fn)
        if spec is None:
            raise SchemaValidationError(f"Unknown tool: {fn!r}")
        missing = [p for p in spec.required_params if p not in params]
        if missing:
            raise SchemaValidationError(f"{fn} missing required params: {missing}")

        tool_call = ToolCall(fn=fn, params=dict(params))
        self.log.append(tool_call)
        if self.on_call is not None:
            self.on_call(tool_call)
        return tool_call
