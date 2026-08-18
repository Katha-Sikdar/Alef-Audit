#!/usr/bin/env python
"""CLI entry point: run an ALEF trial matrix and write a JSONL run log.

Examples
--------
Mock provider, no API keys needed (smoke test):

    python scripts/run_experiment.py --provider mock \\
        --payloads alef/data/sample_payloads.jsonl --out runs/mock_run.jsonl

Real provider:

    export OPENAI_API_KEY=sk-...
    python scripts/run_experiment.py --provider openai:gpt-4o-2024-08-06 \\
        --payloads alef/data/sample_payloads.jsonl --out runs/gpt4o_run.jsonl

With a defense active:

    python scripts/run_experiment.py --provider mock \\
        --defense schema_level_argument_filtering \\
        --payloads alef/data/sample_payloads.jsonl --out runs/mock_defended.jsonl
"""

from __future__ import annotations

from pathlib import Path

import click

from alef.pipeline import ALEFPipeline, write_run_records
from alef.trajectory import PageRepresentation


@click.command()
@click.option("--provider", "providers", multiple=True, default=("mock",),
              help="One or more provider specs: 'mock', a registry display name "
                   "(e.g. 'GPT-4o'), or 'provider:model_id' (e.g. 'openai:gpt-4o-2024-08-06').")
@click.option("--payloads", "payloads_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--representation", "representations", multiple=True,
              type=click.Choice([r.value for r in PageRepresentation]),
              help="Restrict to specific representations; default: all five.")
@click.option("--defense", "defense_names", multiple=True,
              type=click.Choice(["dual_channel_sanity_checking", "state_boundary_isolation",
                                  "schema_level_argument_filtering"]),
              help="Activate one or more Stage-5 defenses as a filter chain. Default: none (undefended baseline).")
@click.option("--out", "out_path", type=click.Path(path_type=Path), default=Path("runs/run.jsonl"))
@click.option("--temperature", type=float, default=0.0)
def main(providers, payloads_path, representations, defense_names, out_path, temperature):
    reps = [PageRepresentation(r) for r in representations] or list(PageRepresentation)

    pipeline = ALEFPipeline(temperature=temperature)
    records = pipeline.run_matrix(
        payloads_path=payloads_path,
        model_names=list(providers),
        representations=reps,
        defense_names=list(defense_names),
    )
    n = write_run_records(records, out_path)
    click.echo(f"Wrote {n} run records to {out_path}")


if __name__ == "__main__":
    main()
