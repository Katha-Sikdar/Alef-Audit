#!/usr/bin/env python
"""CLI entry point: turn a JSONL run log (produced by run_experiment.py)
into paper-style summary tables -- TC%, AC%, Delta_AC-TC, raw p, adjusted q
grouped by representation and by model, plus the contamination severity
distribution (Table 6 / Figure 4 equivalents).

Writes both CSV and Markdown versions of each table to --out.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import click

from alef.auditors import ComplianceResult
from alef.contamination import ContaminationResult, ContaminationSeverity
from alef.pipeline import load_run_records
from alef.stats.metrics import GroupSummary, build_comparison_table, severity_distribution


def _record_to_compliance(record) -> ComplianceResult:
    return ComplianceResult(trajectory_id=record.trajectory_id, tc=record.tc, ac=record.ac, delta_as=record.delta_as)


def _record_to_contamination(record) -> ContaminationResult:
    return ContaminationResult(
        trajectory_id=record.trajectory_id,
        severity=ContaminationSeverity(record.contamination_severity),
        triggering_step_index=None,
    )


def _write_table(summaries: list[GroupSummary], out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    md_path = out_dir / f"{name}.md"

    header = ["group", "n", "TC(%)", "AC(%)", "Delta_AC-TC(%)", "Cohens_h", "raw_p", "adj_q"]
    rows = []
    for s in summaries:
        rows.append([
            s.group,
            s.n,
            f"{s.tc_rate * 100:.1f}",
            f"{s.ac_rate * 100:.1f}",
            f"{s.delta_ac_tc * 100:+.1f}",
            f"{s.cohens_h:.3f}" if s.cohens_h is not None else "",
            f"{s.raw_p:.4f}" if s.raw_p is not None else "",
            f"{s.adjusted_q:.4f}" if s.adjusted_q is not None else "",
        ])

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(c) for c in row) + "\n")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join("---" for _ in header) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(str(c) for c in row) + " |\n")

    click.echo(f"Wrote {csv_path} and {md_path}")


@click.command()
@click.option("--runs", "runs_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--out", "out_dir", type=click.Path(path_type=Path), default=Path("reports"))
def main(runs_path: Path, out_dir: Path):
    records = load_run_records(runs_path)
    if not records:
        click.echo("No run records found.")
        return

    by_representation: dict[str, list] = defaultdict(list)
    by_model: dict[str, list] = defaultdict(list)
    contamination_results = []

    for r in records:
        by_representation[r.representation].append(_record_to_compliance(r))
        by_model[r.model].append(_record_to_compliance(r))
        contamination_results.append(_record_to_contamination(r))

    _write_table(build_comparison_table(by_representation), out_dir, "compliance_by_representation")
    _write_table(build_comparison_table(by_model), out_dir, "compliance_by_model")

    severity = severity_distribution(contamination_results)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "contamination_severity.md", "w", encoding="utf-8") as f:
        f.write("| Severity | % of Trajectories |\n|---|---|\n")
        for label, pct in severity.items():
            f.write(f"| {label} | {pct:.1f}% |\n")
    click.echo(f"Wrote {out_dir / 'contamination_severity.md'}")


if __name__ == "__main__":
    main()
