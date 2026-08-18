"""ALEF: Action-Level Evaluation Framework for indirect prompt injection
in multi-step LLM web agents.

See README.md for the mapping from paper sections to modules.
"""

from alef.trajectory import Observation, ModelOutput, ToolCall, TrajectoryStep, Trajectory
from alef.auditors import (
    TextComplianceAuditor,
    ActionComplianceAuditor,
    ComplianceResult,
    audit_trajectory,
)
from alef.contamination import ContaminationSeverity, compute_contamination

__all__ = [
    "Observation",
    "ModelOutput",
    "ToolCall",
    "TrajectoryStep",
    "Trajectory",
    "TextComplianceAuditor",
    "ActionComplianceAuditor",
    "ComplianceResult",
    "audit_trajectory",
    "ContaminationSeverity",
    "compute_contamination",
]

__version__ = "0.1.0"
