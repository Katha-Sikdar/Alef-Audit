"""ALEFPipeline: orchestrates the paper's five stages end-to-end
(Section 6):

  1. Objective-to-Action Mapping        (alef.objectives)
  2. Structural Representation Modeling (alef.representations)
  3. Stateful Multi-Step Trajectory     (alef.harness.sandbox)
  4. Cross-Layer Compliance Auditing    (alef.auditors, alef.contamination)
  5. Defense Evaluation                 (alef.defenses)

`run_matrix` is the entry point used by scripts/run_experiment.py; it yields
one `RunRecord` per (payload, model, representation) trial, matching the
paper's N = payloads x tasks x representations x models trial matrix
(Section 7.2).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from alef.auditors import ComplianceResult, audit_trajectory
from alef.contamination import ContaminationResult, compute_contamination
from alef.defenses.base import Defense
from alef.defenses.dual_channel import DualChannelSanityChecking
from alef.defenses.schema_filter import SchemaLevelArgumentFiltering
from alef.defenses.state_boundary import StateBoundaryIsolation
from alef.harness.sandbox import TrajectoryConfig, TrajectoryRunner
from alef.objectives import load_objective_mappings
from alef.payloads import IPIPayload, load_payloads
from alef.providers.base import LLMProvider
from alef.providers.registry import get_provider
from alef.trajectory import PageRepresentation

DEFENSE_FACTORIES: dict[str, "type"] = {
    "dual_channel_sanity_checking": DualChannelSanityChecking,
    "state_boundary_isolation": StateBoundaryIsolation,
    "schema_level_argument_filtering": SchemaLevelArgumentFiltering,
}


def build_defenses(names: list[str]) -> list[Defense]:
    defenses: list[Defense] = []
    for name in names:
        factory = DEFENSE_FACTORIES.get(name)
        if factory is None:
            raise KeyError(f"Unknown defense: {name!r}. Known: {list(DEFENSE_FACTORIES)}")
        defenses.append(factory())
    return defenses


@dataclass
class RunRecord:
    trajectory_id: str
    payload_id: str
    objective: str
    model: str
    representation: str
    defenses: list[str]
    tc: bool
    ac: bool
    delta_as: bool
    contamination_severity: int
    contamination_severity_label: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunRecord":
        return cls(**d)


class ALEFPipeline:
    def __init__(self, injected_page_index: int = 3, temperature: float = 0.0):
        self.injected_page_index = injected_page_index
        self.temperature = temperature
        self.objective_mappings = load_objective_mappings()

    def run_trial(
        self,
        payload: IPIPayload,
        provider: LLMProvider,
        model_name: str,
        representation: PageRepresentation,
        defenses: Optional[list[Defense]] = None,
        trajectory_id: Optional[str] = None,
    ) -> RunRecord:
        defenses = defenses or []
        mapping = self.objective_mappings[payload.objective]

        config = TrajectoryConfig(
            representation=representation,
            defenses=defenses,
            temperature=self.temperature,
        )
        runner = TrajectoryRunner(config)

        traj_id = trajectory_id or f"{model_name}:{representation.value}:{payload.payload_id}"
        trajectory = runner.run(
            trajectory_id=traj_id,
            provider=provider,
            payload=payload,
            user_intent=mapping.realistic_task,
            model_name=model_name,
        )

        compliance: ComplianceResult = audit_trajectory(trajectory, payload)
        contamination: ContaminationResult = compute_contamination(trajectory, payload)

        return RunRecord(
            trajectory_id=traj_id,
            payload_id=payload.payload_id,
            objective=payload.objective.value,
            model=model_name,
            representation=representation.value,
            defenses=[d.name for d in defenses],
            tc=compliance.tc,
            ac=compliance.ac,
            delta_as=compliance.delta_as,
            contamination_severity=int(contamination.severity),
            contamination_severity_label=contamination.severity.description,
        )

    def run_matrix(
        self,
        payloads_path: Path,
        model_names: list[str],
        representations: list[PageRepresentation],
        defense_names: Optional[list[str]] = None,
    ) -> Iterator[RunRecord]:
        payloads = load_payloads(payloads_path)
        defenses = build_defenses(defense_names) if defense_names else []

        for model_name in model_names:
            provider = get_provider(model_name)
            for representation in representations:
                for payload in payloads:
                    yield self.run_trial(payload, provider, model_name, representation, defenses)


def write_run_records(records: Iterator[RunRecord], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_dict()) + "\n")
            n += 1
    return n


def load_run_records(path: Path) -> list[RunRecord]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(RunRecord.from_dict(json.loads(line)))
    return records
