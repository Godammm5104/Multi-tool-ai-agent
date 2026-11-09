from types import SimpleNamespace

import pytest

from agent import MultiToolAgent
from tools import TOOL_SCHEMAS


def tool_call(call_id: str, name: str, arguments: str):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def message(content: str | None, tool_calls=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls)
            )
        ]
    )


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


def test_agent_calls_tool_then_returns_final_answer():
    client = FakeClient(
        [
            message(
                None,
                [
                    tool_call(
                        "call_1",
                        "word_count",
                        '{"text": "one two"}',
                    )
                ],
            ),
            message("The text has 2 words."),
        ]
    )
    agent = MultiToolAgent(client=client, model="test")

    answer = agent.run("count the words in: one two")

    assert answer == "The text has 2 words."
    assert len(agent.trace) == 1
    assert agent.trace[0].name == "word_count"
    assert "Words: 2" in agent.trace[0].result
    assert client.chat.completions.calls[0]["tools"] == TOOL_SCHEMAS


def test_agent_handles_multiple_tool_calls_in_one_step():
    client = FakeClient(
        [
            message(
                None,
                [
                    tool_call("call_1", "word_count", '{"text": "one two"}'),
                    tool_call(
                        "call_2",
                        "generate_ideas",
                        '{"topic": "testing", "count": 2}',
                    ),
                ],
            ),
            message("done"),
        ]
    )
    agent = MultiToolAgent(client=client, model="test")

    assert agent.run("do both") == "done"
    assert [item.name for item in agent.trace] == ["word_count", "generate_ideas"]


def test_agent_records_invalid_json_as_tool_error():
    client = FakeClient(
        [
            message(
                None,
                [tool_call("call_1", "word_count", "{bad json")],
            ),
            message("I could not run the tool."),
        ]
    )
    agent = MultiToolAgent(client=client, model="test")

    answer = agent.run("count this")

    assert answer == "I could not run the tool."
    assert agent.trace[0].error
    assert "invalid arguments" in agent.trace[0].result


def test_agent_records_unknown_tool_as_error():
    client = FakeClient(
        [
            message(None, [tool_call("call_1", "does_not_exist", "{}")]),
            message("That tool is unavailable."),
        ]
    )
    agent = MultiToolAgent(client=client, model="test")

    agent.run("use a fake tool")

    assert agent.trace[0].error == "unknown tool"


def test_agent_stops_at_step_limit():
    client = FakeClient(
        [
            message(
                None,
                [tool_call("call_1", "word_count", '{"text": "one two"}')],
            )
        ]
    )
    agent = MultiToolAgent(client=client, model="test", max_steps=1)

    answer = agent.run("count this")

    assert "step limit" in answer
    assert len(agent.trace) == 1


def test_agent_rejects_empty_message():
    agent = MultiToolAgent(client=FakeClient([]), model="test")

    with pytest.raises(ValueError, match="empty"):
        agent.run("   ")
