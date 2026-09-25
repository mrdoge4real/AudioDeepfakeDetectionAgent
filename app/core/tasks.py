# -*- coding: utf-8 -*-
"""任务管理：SQLite 持久化 + 线程池后台执行检测流水线。"""
import json
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime

from app.config import DB_PATH
from app.core.pipeline import STEPS, run_pipeline

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="detect")
_local = threading.local()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    audio_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',   -- pending/running/done/failed
    step INTEGER NOT NULL DEFAULT 0,
    step_name TEXT DEFAULT '',
    message TEXT DEFAULT '',
    error TEXT DEFAULT '',
    report_path TEXT DEFAULT '',
    report_content TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

STEP_NAME_MAP = {no: name for no, name, _ in STEPS}


@contextmanager
def _db():
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
    try:
        yield _local.conn
        _local.conn.commit()
    except Exception:
        _local.conn.rollback()
        raise


def init_db():
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.executescript(_SCHEMA)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _row_to_dict(row) -> dict:
    d = dict(row)
    d.pop("report_content", None)
    return d


# ---------------------------------------------------------------------------
# 任务 CRUD
# ---------------------------------------------------------------------------

def create_task(filename: str, audio_path: str) -> dict:
    task_id = uuid.uuid4().hex[:12]
    now = _now()
    with _db() as conn:
        conn.execute(
            "INSERT INTO tasks (id, filename, audio_path, status, created_at, updated_at)"
            " VALUES (?, ?, ?, 'pending', ?, ?)",
            (task_id, filename, audio_path, now, now),
        )
    return get_task(task_id)


def get_task(task_id: str):
    with _db() as conn:
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return _row_to_dict(row) if row else None


def get_task_full(task_id: str):
    """含 report_content，仅内部/详情接口使用。"""
    with _db() as conn:
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return dict(row) if row else None


def list_tasks(limit: int = 20) -> list:
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def _update(task_id: str, **fields):
    fields["updated_at"] = _now()
    sql = ", ".join(f"{k} = ?" for k in fields)
    with _db() as conn:
        conn.execute(f"UPDATE tasks SET {sql} WHERE id = ?", (*fields.values(), task_id))


# ---------------------------------------------------------------------------
# 后台执行
# ---------------------------------------------------------------------------

def submit_detection(task_id: str):
    _executor.submit(_run, task_id)


def _run(task_id: str):
    task = get_task(task_id)
    if not task:
        return

    def on_progress(step, name, message):
        _update(task_id, status="running", step=step, step_name=name, message=message)

    try:
        on_progress(1, STEP_NAME_MAP[1], "流水线启动")
        result = run_pipeline(task["audio_path"], on_progress=on_progress)
        if result["success"]:
            _update(
                task_id, status="done", step=5, step_name="完成",
                message="检测完成", report_path=result["report_path"],
                report_content=result["report_content"],
            )
        else:
            _update(
                task_id, status="failed", step=result.get("failed_step", 0),
                step_name=STEP_NAME_MAP.get(result.get("failed_step", 0), ""),
                error=result["error"], message="检测失败",
            )
    except Exception as e:
        _update(task_id, status="failed", error=f"任务执行异常：{e}", message="检测失败")


# ---------------------------------------------------------------------------
# 会话历史（供 Agent 层使用）
# ---------------------------------------------------------------------------

def add_chat_message(session_id: str, role: str, content: str):
    with _db() as conn:
        conn.execute(
            "INSERT INTO chat_messages (session_id, role, content, created_at)"
            " VALUES (?, ?, ?, ?)",
            (session_id, role, content, _now()),
        )


def get_chat_history(session_id: str, limit: int = 20) -> list:
    with _db() as conn:
        rows = conn.execute(
            "SELECT role, content FROM chat_messages WHERE session_id = ?"
            " ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]


def find_recent_done_task() -> dict:
    with _db() as conn:
        row = conn.execute(
            "SELECT * FROM tasks WHERE status = 'done' ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


def find_task_by_filename(filename: str) -> dict:
    with _db() as conn:
        row = conn.execute(
            "SELECT * FROM tasks WHERE filename LIKE ? ORDER BY created_at DESC LIMIT 1",
            (f"%{filename}%",),
        ).fetchone()
    return dict(row) if row else None


# 供 JSON 序列化的安全副本
def task_to_json(task: dict) -> str:
    return json.dumps(task, ensure_ascii=False, default=str)
