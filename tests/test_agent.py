# -*- coding: utf-8 -*-
"""Agent function-calling 循环测试：用假 LLM 客户端验证工具调用与回填机制。"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core import tasks


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(tasks, "DB_PATH", tmp_path / "test.db")
    tasks._local.conn = None
    tasks.init_db()


def _fake_tool_call(name, arguments, call_id="call_1"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
        model_dump=lambda: {
            "id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)},
        },
    )


class FakeLLM:
    """第一轮要求调 explain_knowledge，第二轮返回纯文本。"""
    def __init__(self):
        self.rounds = 0
        self.seen_messages = []

    def __call__(self, messages):
        self.rounds += 1
        self.seen_messages = messages
        if self.rounds == 1:
            return SimpleNamespace(
                content=None,
                tool_calls=[_fake_tool_call("explain_knowledge", {"query": "MFCC是什么"})],
            )
        return SimpleNamespace(content="MFCC 是一种声学特征……", tool_calls=None)


def test_agent_tool_loop(monkeypatch):
    import app.agent.agent as agent_mod

    fake = FakeLLM()
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    agent = agent_mod.ChatAgent.__new__(agent_mod.ChatAgent)
    agent.client = None
    agent.model = "fake-model"
    monkeypatch.setattr(agent, "_call_llm", fake)

    reply = agent.chat("sess-1", "MFCC 是什么？")

    assert reply == "MFCC 是一种声学特征……"
    assert fake.rounds == 2

    # 验证第二轮时工具结果已回填
    tool_msgs = [m for m in fake.seen_messages if isinstance(m, dict) and m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    payload = json.loads(tool_msgs[0]["content"])
    assert payload["found"] is True
    assert "MFCC" in payload["content"]

    # 验证会话历史已持久化
    history = tasks.get_chat_history("sess-1")
    assert history[-1]["role"] == "assistant"
    assert history[-1]["content"] == reply


def test_agent_plain_reply_without_tools(monkeypatch):
    import app.agent.agent as agent_mod

    monkeypatch.setenv("LLM_API_KEY", "test-key")
    agent = agent_mod.ChatAgent.__new__(agent_mod.ChatAgent)
    agent.client = None
    agent.model = "fake-model"
    monkeypatch.setattr(agent, "_call_llm",
                        lambda msgs: SimpleNamespace(content="你好呀", tool_calls=None))

    assert agent.chat("sess-2", "你好") == "你好呀"
