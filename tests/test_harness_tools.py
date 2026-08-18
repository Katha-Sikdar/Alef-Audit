import pytest

from alef.harness.tools import SandboxToolExecutor, SchemaValidationError, openai_tool_schemas


def test_executor_logs_valid_call():
    executor = SandboxToolExecutor()
    call = executor.execute("send_email", {"to": "a@b.example", "body": "hi"})
    assert call.fn == "send_email"
    assert executor.log == [call]


def test_executor_rejects_unknown_tool():
    executor = SandboxToolExecutor()
    with pytest.raises(SchemaValidationError):
        executor.execute("delete_everything", {})


def test_executor_rejects_missing_required_params():
    executor = SandboxToolExecutor()
    with pytest.raises(SchemaValidationError):
        executor.execute("send_email", {"to": "a@b.example"})  # missing 'body'


def test_executor_never_performs_real_side_effects(monkeypatch):
    # There is no network/filesystem call anywhere in SandboxToolExecutor.execute;
    # this test just documents/asserts that invariant by checking the call is a
    # pure, in-memory ToolCall record.
    executor = SandboxToolExecutor()
    call = executor.execute("save_file", {"path": "/tmp/whatever.txt", "content": "x"})
    import os
    assert not os.path.exists("/tmp/whatever.txt")
    assert call.params["path"] == "/tmp/whatever.txt"


def test_openai_tool_schemas_shape():
    schemas = openai_tool_schemas(["send_email"])
    assert len(schemas) == 1
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "send_email"
    assert "to" in schemas[0]["function"]["parameters"]["properties"]
