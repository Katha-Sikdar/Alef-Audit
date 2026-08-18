"""Minimal usage example: run one trajectory against the offline mock
provider and print the audit result. No API keys required.

    python examples/run_single_trial.py
"""

from __future__ import annotations

from alef.payloads import load_payloads
from alef.pipeline import ALEFPipeline
from alef.providers.registry import get_provider
from alef.trajectory import PageRepresentation


def main() -> None:
    payloads = load_payloads()  # bundled sample corpus
    payload = next(p for p in payloads if p.payload_id == "exfil-0001")

    provider = get_provider("mock")  # action_silent mode by default
    pipeline = ALEFPipeline()

    record = pipeline.run_trial(
        payload=payload,
        provider=provider,
        model_name="mock",
        representation=PageRepresentation.PLAIN_TEXT,
    )

    print(f"Payload:        {payload.payload_id} ({payload.objective.value})")
    print(f"Text Compliance (TC):   {record.tc}")
    print(f"Action Compliance (AC): {record.ac}")
    print(f"Action-Silent gap:      {record.delta_as}")
    print(f"Contamination severity: {record.contamination_severity_label} "
          f"(level {record.contamination_severity})")


if __name__ == "__main__":
    main()
