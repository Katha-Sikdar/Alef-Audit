"""IPI payload corpus loading and ground-truth adversarial parameters.

The paper's 300-payload corpus is reconstructed from the restricted-access
Khodayari et al. dataset and is not redistributed here (see README.md and
the paper's Section 13/17). This module defines the payload schema and loads
whatever JSONL corpus you point it at -- either the small illustrative
`alef/data/sample_payloads.jsonl` shipped with this repo, or your own
restricted-access reconstruction, as long as it matches the schema below
(see `alef/data/payload_schema.json`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from alef.objectives import IPIObjective

_DATA_DIR = Path(__file__).parent / "data"
DEFAULT_PAYLOAD_PATH = _DATA_DIR / "sample_payloads.jsonl"


@dataclass
class IPIPayload:
    """One in-the-wild-style injection payload plus the ground-truth
    adversarial parameters an f_AC auditor checks tool calls against.

    `adversarial_params` maps a tool-call parameter name to the value (or a
    substring that must appear in the value) an attacker-controlled tool
    call would carry -- e.g. {"to": "attacker@malicious.com"} for a
    data-exfiltration payload targeting send_email's `to` field.
    """

    payload_id: str
    objective: IPIObjective
    text: str
    target_tool: str
    adversarial_params: dict[str, str] = field(default_factory=dict)
    text_compliance_markers: list[str] = field(default_factory=list)
    """Substrings that, if present in a model's visible text output, are
    treated as an explicit textual acknowledgment/compliance signal by the
    default rule-based f_TC auditor (alef/auditors.py). The LLM-as-judge
    auditor (alef/judge.py) does not use this field."""
    source: str = "synthetic-illustrative"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "IPIPayload":
        return cls(
            payload_id=d["payload_id"],
            objective=IPIObjective(d["objective"]),
            text=d["text"],
            target_tool=d["target_tool"],
            adversarial_params=dict(d.get("adversarial_params", {})),
            text_compliance_markers=list(d.get("text_compliance_markers", [])),
            source=d.get("source", "synthetic-illustrative"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "payload_id": self.payload_id,
            "objective": self.objective.value,
            "text": self.text,
            "target_tool": self.target_tool,
            "adversarial_params": self.adversarial_params,
            "text_compliance_markers": self.text_compliance_markers,
            "source": self.source,
        }


def load_payloads(path: Path | str | None = None) -> list[IPIPayload]:
    """Load a JSONL payload corpus. Defaults to the bundled sample set."""
    path = Path(path) if path is not None else DEFAULT_PAYLOAD_PATH
    payloads: list[IPIPayload] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payloads.append(IPIPayload.from_dict(json.loads(line)))
    return payloads


def iter_payloads(path: Path | str | None = None) -> Iterator[IPIPayload]:
    for payload in load_payloads(path):
        yield payload
