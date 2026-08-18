from alef.defenses.dual_channel import DualChannelSanityChecking
from alef.defenses.schema_filter import SchemaLevelArgumentFiltering
from alef.defenses.state_boundary import StateBoundaryIsolation
from alef.trajectory import ModelOutput, Observation, PageRepresentation, ToolCall, Trajectory, TrajectoryStep


def _traj_with_current_step(page_id, content):
    traj = Trajectory(trajectory_id="t-def", user_intent="Summarize chat & send email")
    obs = Observation(page_id=page_id, representation=PageRepresentation.PLAIN_TEXT, content=content)
    traj.add_step(TrajectoryStep(1, page_id, obs, ModelOutput(text="ok"), None))
    return traj


def test_dual_channel_sanity_checking_blocks_low_similarity():
    defense = DualChannelSanityChecking(tau=0.2)
    call = ToolCall(fn="send_email", params={"to": "attacker@malicious.example"})
    traj = _traj_with_current_step("P3", "irrelevant content")

    result = defense.filter(call, user_intent="Summarize chat & send email to my manager", trajectory=traj, step_index=1)
    assert result.blocked_by_defense == "dual_channel_sanity_checking"


def test_dual_channel_sanity_checking_allows_high_similarity():
    defense = DualChannelSanityChecking(tau=0.05)
    call = ToolCall(fn="send_email", params={"to": "manager@company.example"})
    traj = _traj_with_current_step("P3", "irrelevant content")

    result = defense.filter(call, user_intent="send email to manager@company.example", trajectory=traj, step_index=1)
    assert result.blocked_by_defense is None


def test_state_boundary_isolation_blocks_carried_over_value():
    defense = StateBoundaryIsolation()
    traj = _traj_with_current_step("P4", "This page only discusses reference checks, nothing else.")
    call = ToolCall(fn="submit_form", params={"endpoint": "/api/v1/telemetry", "payload": {}})

    result = defense.filter(call, user_intent="submit form", trajectory=traj, step_index=1)
    assert result.blocked_by_defense == "state_boundary_isolation"


def test_state_boundary_isolation_allows_value_present_on_current_page():
    defense = StateBoundaryIsolation()
    traj = _traj_with_current_step("P4", "Please submit to /api/v1/legitimate as instructed here.")
    call = ToolCall(fn="submit_form", params={"endpoint": "/api/v1/legitimate", "payload": {}})

    result = defense.filter(call, user_intent="submit form", trajectory=traj, step_index=1)
    assert result.blocked_by_defense is None


def test_schema_level_argument_filtering_blocks_non_allowlisted_recipient():
    defense = SchemaLevelArgumentFiltering(approved_email_domains=frozenset({"company.example"}))
    call = ToolCall(fn="send_email", params={"to": "attacker@malicious.example"})
    traj = _traj_with_current_step("P3", "content")

    result = defense.filter(call, user_intent="email", trajectory=traj, step_index=1)
    assert result.blocked_by_defense == "schema_level_argument_filtering"


def test_schema_level_argument_filtering_allows_allowlisted_recipient():
    defense = SchemaLevelArgumentFiltering(approved_email_domains=frozenset({"company.example"}))
    call = ToolCall(fn="send_email", params={"to": "manager@company.example"})
    traj = _traj_with_current_step("P3", "content")

    result = defense.filter(call, user_intent="email", trajectory=traj, step_index=1)
    assert result.blocked_by_defense is None
