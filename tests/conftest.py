import pytest

from alef.payloads import IPIPayload, load_payloads
from alef.objectives import IPIObjective


@pytest.fixture
def exfil_payload() -> IPIPayload:
    return next(p for p in load_payloads() if p.payload_id == "exfil-0001")


@pytest.fixture
def submit_payload() -> IPIPayload:
    return next(p for p in load_payloads() if p.payload_id == "credential-0001")


@pytest.fixture
def all_sample_payloads() -> list[IPIPayload]:
    return load_payloads()
