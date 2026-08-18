# ALEF-Audit

**Action-Level Evaluation Framework (ALEF)** — a measurement harness for auditing
indirect prompt injection (IPI) against multi-step, tool-calling LLM web agents
at both the **text layer** (what the agent says) and the **tool-invocation layer**
(what the agent actually executes).

This repository implements the framework described in *"Beyond Text-Level
Compliance: An Action-Level Evaluation Framework for Indirect Prompt Injection
in Multi-Step LLM Web Agents"* (Katha & Prova, 2026). It reproduces the paper's
five-stage pipeline as runnable, testable code:

1. **Objective-to-Action Mapping** — map IPI objective categories onto realistic
   tool-invocation goals (`alef/objectives.py`).
2. **Structural Representation Modeling** — transform a page into five
   representations: plain text, HTML, raw HTTP response, rendered snapshot,
   accessibility tree (`alef/representations.py`).
3. **Stateful Multi-Step Trajectory Execution** — a five-page (P1–P5) sandboxed
   browsing trajectory with a persistent context window and real-shaped,
   side-effect tools (`send_email`, `save_file`, `submit_form`) intercepted at
   an API boundary (`alef/harness/`).
4. **Cross-Layer Compliance Auditing** — the dual-channel auditor computing
   textual compliance (`f_TC`), action compliance (`f_AC`), the Action-Silent
   Compliance Gap (`δ_AS`), and the raw gap (`Δ_AC-TC`) (`alef/auditors.py`).
5. **Defense Evaluation** — three execution-layer defenses (dual-channel sanity
   checking, state-boundary isolation, schema-level argument filtering) as
   pre-execution filters, with measured relative-reduction / false-positive
   metrics (`alef/defenses/`).

A statistics module (`alef/stats/`) implements two-proportion z-tests,
Benjamini–Hochberg FDR correction, and Cohen's h, matching the paper's
reporting methodology (Section 7.7).

## Status and scope of this scaffold

This is a **working, testable implementation of the ALEF pipeline's logic**,
not a re-run of the paper's original 47,812-trial study. In particular:

- **No live LLM calls are required to run the test suite.** `alef/providers/mock.py`
  is a deterministic mock provider that exercises every code path (including
  Action-Silent trajectories) without any API key.
- **Real provider adapters are included** for OpenAI, Anthropic, and a local
  vLLM / OpenAI-compatible endpoint (`alef/providers/`), matching the paper's
  four capability tiers (`alef/providers/registry.py`). They activate once you
  set the relevant environment variables (see below); nothing about them is
  mocked or faked.
- **The 300-payload in-the-wild IPI corpus from Khodayari et al. is not
  redistributed here.** That corpus is under restricted academic access (see
  the paper's Section 13/17). `alef/data/sample_payloads.jsonl` ships a small
  set of clearly-synthetic, illustrative example payloads (structurally similar
  to the ones already published in the paper's own case studies) so the
  pipeline and tests are exercisable end-to-end. Swap in the real corpus under
  your own restricted-access agreement to reproduce the paper's actual numbers.
- **Tool execution never leaves the sandbox.** `alef/harness/tools.py` never
  makes a real network call, sends a real email, or writes outside a temp
  directory — every "side effect" is captured as a structured, logged
  `ToolCall` object for auditing, exactly as described in the paper's Section 7.5
  and Section 13 (ethical design).

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Running the tests

```bash
pytest -q
```

All tests run fully offline against the mock provider.

## Running an experiment

```bash
# Mock provider, no API keys needed — good for smoke-testing the pipeline:
python scripts/run_experiment.py --provider mock --payloads alef/data/sample_payloads.jsonl --out runs/mock_run.jsonl

# Real provider, e.g. OpenAI (requires OPENAI_API_KEY):
export OPENAI_API_KEY=sk-...
python scripts/run_experiment.py --provider openai --model gpt-4o-2024-08-06 \
    --payloads alef/data/sample_payloads.jsonl --out runs/gpt4o_run.jsonl
```

Then compute the paper-style summary tables from a run log:

```bash
python scripts/analyze_results.py --runs runs/mock_run.jsonl --out reports/
```

This produces CSV/Markdown tables analogous to the paper's Tables 4, 5, 7, 8
(TC%, AC%, Δ_AC-TC, raw/adjusted p-values) and Table 6/Figure 4 (contamination
severity distribution).

## Repository layout

```
alef/
  trajectory.py        # Trajectory / Observation / ModelOutput / ToolCall data model (Eq. 1-3)
  objectives.py         # Stage 1: IPI objective -> tool-invocation task mapping (Table 2)
  representations.py    # Stage 2: five structural page representations
  auditors.py            # Stage 4: f_TC, f_AC, delta_AS (Algorithm 1)
  contamination.py       # gamma_CS cross-page contamination + severity scale (Table 6)
  judge.py                # LLM-as-judge f_TC implementation with calibration hooks
  pipeline.py             # ALEFPipeline: orchestrates Stages 1-5
  config.py               # experiment configuration dataclasses
  harness/
    tools.py              # sandboxed send_email/save_file/submit_form + API-boundary hook
    sandbox.py             # Stage 3: stateful five-page trajectory runner
  providers/
    base.py                # LLMProvider interface
    mock.py                 # deterministic offline provider used by tests
    openai_provider.py      # real OpenAI adapter
    anthropic_provider.py   # real Anthropic adapter
    vllm_provider.py        # local / OpenAI-compatible open-weight adapter
    registry.py              # capability-tier -> model roster mapping
  defenses/
    dual_channel.py          # Dual-Channel Sanity Checking
    state_boundary.py        # State Boundary Isolation
    schema_filter.py         # Schema-Level Argument Filtering
  stats/
    significance.py          # two-proportion z-test + Benjamini-Hochberg + Cohen's h
    metrics.py                # aggregate TC/AC/gap/carryover table builders
tests/                        # pytest suite, offline, no API keys required
scripts/                      # CLI entry points (run_experiment, analyze_results)
examples/                     # minimal single-trial usage example
docker/                       # sandbox container definition
```

## Ethical use

This code is a defensive measurement and evaluation tool. The bundled sample
payloads are simple, already-published illustrative strings (see the paper's
Section 9 case studies) intended solely to exercise the auditing pipeline —
they are not a novel attack corpus. Tool execution is fully sandboxed and never
produces a real-world side effect. Do not point the real provider adapters at
production systems with real credentials, real recipients, or real destinations.

## License

Apache License 2.0 — see `LICENSE`.

## Citation

If you use this framework, please cite the paper:

```bibtex
@article{katha2026alef,
  title   = {Beyond Text-Level Compliance: An Action-Level Evaluation Framework
             for Indirect Prompt Injection in Multi-Step LLM Web Agents},
  author  = {Katha, Jannatul Ferdous and Prova, Tasmia Tahmid},
  year    = {2026}
}
```
