# -*- coding: utf-8 -*-
"""任务存储 CRUD 测试（SQLite，不依赖重型库）。"""
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core import tasks


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """每个测试用独立 SQLite 文件，并重置线程本地连接。"""
    monkeypatch.setattr(tasks, "DB_PATH", tmp_path / "test.db")
    tasks._local.conn = None
    tasks.init_db()
    yield


def test_task_lifecycle():
    task = tasks.create_task(filename="demo.flac", audio_path="/tmp/demo.flac")
    assert task["status"] == "pending"
    assert task["step"] == 0

    fetched = tasks.get_task(task["id"])
    assert fetched["filename"] == "demo.flac"
    assert "report_content" not in fetched  # 列表/详情默认不含报告全文

    tasks._update(task["id"], status="running", step=3, step_name="ASR+说话人分割")
    running = tasks.get_task(task["id"])
    assert running["status"] == "running"
    assert running["step"] == 3

    tasks._update(task["id"], status="done", step=5, report_content="# 报告")
    full = tasks.get_task_full(task["id"])
    assert full["report_content"] == "# 报告"


def test_list_and_find():
    tasks.create_task(filename="LA_E_1000147.flac", audio_path="/tmp/a.flac")
    tasks.create_task(filename="other.wav", audio_path="/tmp/b.wav")

    history = tasks.list_tasks()
    assert len(history) == 2

    hit = tasks.find_task_by_filename("1000147")
    assert hit and hit["filename"] == "LA_E_1000147.flac"

    miss = tasks.find_task_by_filename("不存在")
    assert miss is None


def test_chat_history_order():
    tasks.add_chat_message("s1", "user", "第一条")
    tasks.add_chat_message("s1", "assistant", "第二条")
    tasks.add_chat_message("s2", "user", "别的会话")

    history = tasks.get_chat_history("s1")
    assert [m["content"] for m in history] == ["第一条", "第二条"]
    assert [m["role"] for m in history] == ["user", "assistant"]
