from alef.trajectory import (
    ModelOutput,
    Observation,
    PageRepresentation,
    ToolCall,
    Trajectory,
    TrajectoryStep,
)


def _make_step(index, page_id, injected=False, tool_call=None):
    obs = Observation(
        page_id=page_id,
        representation=PageRepresentation.PLAIN_TEXT,
        content=f"content of {page_id}",
        injected_payload="payload text" if injected else None,
    )
    return TrajectoryStep(
        step_index=index,
        page_id=page_id,
        observation=obs,
        model_output=ModelOutput(text=f"output for {page_id}"),
        tool_call=tool_call,
    )


def test_add_step_tracks_injection_point():
    traj = Trajectory(trajectory_id="t1", user_intent="do the thing")
    traj.add_step(_make_step(1, "P1"))
    traj.add_step(_make_step(2, "P2"))
    traj.add_step(_make_step(3, "P3", injected=True))
    traj.add_step(_make_step(4, "P4"))

    assert traj.injected_step_index == 3
    assert traj.injected_payload == "payload text"
    assert traj.n_steps == 4


def test_steps_after_injection():
    traj = Trajectory(trajectory_id="t2", user_intent="do the thing")
    for i, pid in enumerate(["P1", "P2", "P3", "P4", "P5"], start=1):
        traj.add_step(_make_step(i, pid, injected=(pid == "P3")))

    after = traj.steps_after_injection()
    assert [s.page_id for s in after] == ["P4", "P5"]


def test_steps_after_injection_empty_when_no_injection():
    traj = Trajectory(trajectory_id="t3", user_intent="do the thing")
    traj.add_step(_make_step(1, "P1"))
    assert traj.steps_after_injection() == []


def test_context_history_text_accumulates_and_truncates():
    traj = Trajectory(trajectory_id="t4", user_intent="do the thing")
    tool_call = ToolCall(fn="save_file", params={"path": "/x"})
    traj.add_step(_make_step(1, "P1"))
    traj.add_step(_make_step(2, "P2", tool_call=tool_call))
    traj.add_step(_make_step(3, "P3"))

    history_up_to_2 = traj.context_history_text(up_to_step=2)
    assert "P1" in history_up_to_2
    assert "P2" in history_up_to_2
    assert "P3" not in history_up_to_2
    assert "save_file" in history_up_to_2
