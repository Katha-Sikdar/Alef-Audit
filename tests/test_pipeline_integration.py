"""End-to-end integration tests using the offline mock provider only --
no API keys or network access required. Exercises Stages 1-5 together."""

from alef.defenses.schema_filter import SchemaLevelArgumentFiltering
from alef.pipeline import ALEFPipeline
from alef.providers.registry import get_provider
from alef.trajectory import PageRepresentation


def test_action_silent_mode_reproduces_the_core_finding(exfil_payload):
    provider = get_provider("mock", mode="action_silent")
    pipeline = ALEFPipeline()

    record = pipeline.run_trial(
        payload=exfil_payload,
        provider=provider,
        model_name="mock-action-silent",
        representation=PageRepresentation.PLAIN_TEXT,
    )

    assert record.tc is False
    assert record.ac is True
    assert record.delta_as is True


def test_faithful_refusal_mode_has_no_gap(exfil_payload):
    provider = get_provider("mock", mode="faithful_refusal")
    pipeline = ALEFPipeline()

    record = pipeline.run_trial(
        payload=exfil_payload,
        provider=provider,
        model_name="mock-refusal",
        representation=PageRepresentation.PLAIN_TEXT,
    )

    assert record.tc is False
    assert record.ac is False
    assert record.delta_as is False


def test_schema_filter_defense_blocks_the_action_silent_exfiltration(exfil_payload):
    provider = get_provider("mock", mode="action_silent")
    pipeline = ALEFPipeline()
    defense = SchemaLevelArgumentFiltering(approved_email_domains=frozenset({"company.example"}))

    record = pipeline.run_trial(
        payload=exfil_payload,
        provider=provider,
        model_name="mock-action-silent",
        representation=PageRepresentation.PLAIN_TEXT,
        defenses=[defense],
    )

    # The defense should veto the hijacked send_email call, collapsing AC to
    # False -- this is exactly what Table 9's "Defended AC (%)" measures.
    assert record.ac is False
    assert record.delta_as is False


def test_run_matrix_over_sample_corpus_produces_one_record_per_cell(all_sample_payloads, tmp_path):
    from alef.payloads import DEFAULT_PAYLOAD_PATH

    pipeline = ALEFPipeline()
    representations = [PageRepresentation.PLAIN_TEXT, PageRepresentation.ACCESSIBILITY_TREE]
    records = list(
        pipeline.run_matrix(
            payloads_path=DEFAULT_PAYLOAD_PATH,
            model_names=["mock"],
            representations=representations,
        )
    )

    assert len(records) == len(all_sample_payloads) * len(representations)
    assert {r.representation for r in records} == {"plain_text", "accessibility_tree"}


def test_contamination_delayed_hijack_case_study(submit_payload):
    """Mirrors the paper's Case Study 3 (Section 9.3): payload injected on
    P3, dormant on clean P4, triggers a hijacked submit_form on P5."""
    provider = get_provider("mock", mode="action_silent")
    pipeline = ALEFPipeline()

    record = pipeline.run_trial(
        payload=submit_payload,
        provider=provider,
        model_name="mock-action-silent",
        representation=PageRepresentation.PLAIN_TEXT,
    )

    assert record.ac is True
    assert record.tc is False
