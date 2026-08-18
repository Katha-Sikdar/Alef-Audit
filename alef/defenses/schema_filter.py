"""Schema-Level Argument Filtering (Section 10.1).

"Apply strict regex patterns and recipient-domain allowlists to outgoing
parameters (e.g., verifying that the `to` parameter in send_email strictly
matches pre-approved user domains)."

This is the strictest of the three defenses (matching the paper's Table 9
result: 87.2% relative AC reduction, 0% false-positive rate on benign
control tasks) because it is a pure allowlist -- anything not explicitly
approved is blocked, benign or not. Configure it with the allowlists your
benign control tasks actually use, or false positives will appear.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from alef.defenses.base import Defense
from alef.trajectory import ToolCall, Trajectory


@dataclass
class SchemaLevelArgumentFiltering(Defense):
    approved_email_domains: frozenset[str] = field(default_factory=frozenset)
    approved_path_prefixes: tuple[str, ...] = ()
    approved_endpoint_prefixes: tuple[str, ...] = ()
    approved_url_domains: frozenset[str] = field(default_factory=frozenset)

    name: str = field(default="schema_level_argument_filtering", init=False)

    def filter(self, tool_call: ToolCall, *, user_intent: str, trajectory: Trajectory, step_index: int) -> ToolCall:
        if tool_call.fn == "send_email":
            to = str(tool_call.params.get("to", ""))
            domain = to.split("@")[-1].lower() if "@" in to else ""
            if domain not in self.approved_email_domains:
                return self._block(tool_call)

        elif tool_call.fn == "save_file":
            path = str(tool_call.params.get("path", ""))
            if not any(path.startswith(prefix) for prefix in self.approved_path_prefixes):
                return self._block(tool_call)

        elif tool_call.fn == "submit_form":
            endpoint = str(tool_call.params.get("endpoint", ""))
            if not any(endpoint.startswith(prefix) for prefix in self.approved_endpoint_prefixes):
                return self._block(tool_call)

        elif tool_call.fn in ("navigate", "click"):
            target = str(tool_call.params.get("url") or tool_call.params.get("selector", ""))
            if self.approved_url_domains and not any(d in target for d in self.approved_url_domains):
                return self._block(tool_call)

        return tool_call
