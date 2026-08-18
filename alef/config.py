"""Experiment configuration for a full ALEF trial matrix (Section 7.2):
payloads x page representations x models (x defenses, optionally).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from alef.trajectory import PageRepresentation

ALL_REPRESENTATIONS: list[PageRepresentation] = list(PageRepresentation)


@dataclass
class ExperimentConfig:
    payloads_path: Path
    models: list[str] = field(default_factory=lambda: ["mock"])
    representations: list[PageRepresentation] = field(default_factory=lambda: list(ALL_REPRESENTATIONS))
    defense_names: list[str] = field(default_factory=list)
    """Names of defenses to activate as a Stage-5 pre-execution filter chain,
    e.g. ["dual_channel_sanity_checking"]. Empty = undefended baseline."""
    temperature: float = 0.0
    injected_page_index: int = 3
    out_path: Path = Path("runs/run.jsonl")
