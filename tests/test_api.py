# -*- coding: utf-8 -*-
"""API 层测试：FastAPI TestClient，检测接口打桩，不加载真实模型。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
os.environ["SKIP_WARMUP"] = "1"

from fastapi.testclient import TestClient

from app.core import tasks


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(tasks, "DB_PATH", tmp_path / "test.db")
    tasks._local.conn = None
    # 不真正跑流水线
    monkeypatch.setattr(tasks, "submit_detection", lambda task_id: None)

    from app.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_upload_creates_task(client):
    res = client.post(
        "/api/detect",
        files={"file": ("demo.wav", b"RIFF fake wav bytes", "audio/wav")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "pending"

    res = client.get(f"/api/tasks/{data['task_id']}")
    assert res.status_code == 200
    assert res.json()["filename"] == "demo.wav"


def test_upload_rejects_bad_extension(client):
    res = client.post(
        "/api/detect",
        files={"file": ("evil.exe", b"MZ", "application/octet-stream")},
    )
    assert res.status_code == 400


def test_get_missing_task(client):
    assert client.get("/api/tasks/nonexistent").status_code == 404


def test_list_tasks(client):
    client.post("/api/detect", files={"file": ("a.wav", b"x", "audio/wav")})
    client.post("/api/detect", files={"file": ("b.flac", b"x", "audio/flac")})
    res = client.get("/api/tasks")
    assert len(res.json()["tasks"]) == 2


def test_chat_requires_message(client):
    res = client.post("/api/chat", json={"message": "  "})
    assert res.status_code == 400
