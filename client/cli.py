# -*- coding: utf-8 -*-
"""命令行客户端：通过 HTTP 接口使用音频伪造检测服务。

用法示例：
    python -m client detect /path/to/audio.flac     # 上传并等待检测报告
    python -m client status <task_id>               # 查询任务进度
    python -m client report <task_id>               # 查看检测报告
    python -m client history                        # 最近任务列表
    python -m client chat                           # 进入对话模式
    python -m client chat -m "刚才的音频有问题吗"     # 单条提问

服务地址默认 http://localhost:8000，可用 --server 或环境变量 ADD_SERVER 覆盖。
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

DEFAULT_SERVER = os.getenv("ADD_SERVER", "http://localhost:8000")
SESSION_FILE = Path.home() / ".cache" / "add_client_session"
POLL_INTERVAL = 2.0
STEPS_TOTAL = 5

STATUS_LABEL = {
    "pending": "⏳ 排队中",
    "running": "🔄 检测中",
    "done": "✅ 已完成",
    "failed": "❌ 失败",
}


def _url(server: str, path: str) -> str:
    return server.rstrip("/") + path


def _check(resp: requests.Response) -> dict:
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        print(f"❌ 请求失败 [{resp.status_code}]：{detail}", file=sys.stderr)
        sys.exit(1)
    return resp.json()


def _print_progress(task: dict, last_len: int = 0) -> int:
    step = task.get("step", 0)
    bar = "█" * step + "░" * (STEPS_TOTAL - step)
    label = STATUS_LABEL.get(task["status"], task["status"])
    line = f"\r{label} [{bar}] {step}/{STEPS_TOTAL} {task.get('step_name','')} {task.get('message','')}"
    padded = line + " " * max(0, last_len - len(line))
    print(padded, end="", flush=True)
    return len(line)


# ---------------------------------------------------------------------------
# 子命令
# ---------------------------------------------------------------------------

def cmd_detect(args):
    audio = Path(args.file)
    if not audio.exists():
        print(f"❌ 文件不存在：{audio}", file=sys.stderr)
        sys.exit(1)

    print(f"📤 上传 {audio.name} …")
    with open(audio, "rb") as f:
        resp = requests.post(_url(args.server, "/api/detect"), files={"file": (audio.name, f)})
    data = _check(resp)
    task_id = data["task_id"]
    print(f"🆔 task_id = {task_id}")

    last_len = 0
    while True:
        task = _check(requests.get(_url(args.server, f"/api/tasks/{task_id}")))
        last_len = _print_progress(task, last_len)
        if task["status"] == "done":
            print()
            break
        if task["status"] == "failed":
            print(f"\n❌ 检测失败：{task.get('error','')}", file=sys.stderr)
            sys.exit(2)
        time.sleep(POLL_INTERVAL)

    full = _check(requests.get(
        _url(args.server, f"/api/tasks/{task_id}"), params={"include_report": "true"}
    ))
    print("\n" + "=" * 70)
    print(full.get("report_content", "（无报告内容）"))
    print("=" * 70)


def cmd_status(args):
    task = _check(requests.get(_url(args.server, f"/api/tasks/{args.task_id}")))
    label = STATUS_LABEL.get(task["status"], task["status"])
    print(f"{label} 步骤 {task['step']}/{STEPS_TOTAL} {task.get('step_name','')}")
    print(f"文件：{task['filename']}")
    if task.get("message"):
        print(f"信息：{task['message']}")
    if task.get("error"):
        print(f"错误：{task['error']}")


def cmd_report(args):
    task = _check(requests.get(
        _url(args.server, f"/api/tasks/{args.task_id}"), params={"include_report": "true"}
    ))
    if task["status"] != "done":
        print(f"⚠️ 任务未完成（{STATUS_LABEL.get(task['status'])}）")
        sys.exit(2)
    print(task.get("report_content", "（无报告内容）"))


def cmd_history(args):
    tasks = _check(requests.get(_url(args.server, "/api/tasks")))["tasks"]
    if not tasks:
        print("（暂无任务）")
        return
    print(f"{'task_id':<14}{'状态':<10}{'进度':<8}{'文件名'}")
    print("-" * 70)
    for t in tasks:
        label = STATUS_LABEL.get(t["status"], t["status"])
        print(f"{t['id']:<14}{label:<10}{t['step']}/{STEPS_TOTAL:<6}{t['filename']}")


def _load_session() -> str:
    try:
        return SESSION_FILE.read_text().strip()
    except Exception:
        return ""


def _save_session(session_id: str):
    SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    SESSION_FILE.write_text(session_id)


def _chat_once(server: str, message: str) -> str:
    session_id = _load_session()
    resp = requests.post(
        _url(server, "/api/chat"),
        json={"message": message, "session_id": session_id},
        timeout=120,
    )
    data = _check(resp)
    _save_session(data["session_id"])
    return data["reply"]


def cmd_chat(args):
    if args.message:
        print(f"🤖 {_chat_once(args.server, args.message)}")
        return
    print("💬 对话模式（输入 exit 退出，clear 开启新会话）")
    while True:
        try:
            msg = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not msg:
            continue
        if msg.lower() in ("exit", "quit", "退出"):
            break
        if msg.lower() in ("clear", "新会话"):
            SESSION_FILE.unlink(missing_ok=True)
            print("🆕 已开启新会话")
            continue
        try:
            print(f"🤖 {_chat_once(args.server, msg)}\n")
        except requests.ConnectionError:
            print("❌ 连不上服务，请确认服务已启动", file=sys.stderr)
            break


def main():
    parser = argparse.ArgumentParser(
        prog="add-client", description="音频伪造检测服务 CLI 客户端"
    )
    parser.add_argument("--server", default=DEFAULT_SERVER,
                        help=f"服务地址（默认 {DEFAULT_SERVER}）")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("detect", help="上传音频并等待检测报告")
    p.add_argument("file", help="音频文件路径")
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("status", help="查询任务状态")
    p.add_argument("task_id")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("report", help="查看检测报告")
    p.add_argument("task_id")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("history", help="最近任务列表")
    p.set_defaults(func=cmd_history)

    p = sub.add_parser("chat", help="与智能助手对话")
    p.add_argument("-m", "--message", help="单条消息（不传则进入交互模式）")
    p.set_defaults(func=cmd_chat)

    args = parser.parse_args()
    try:
        args.func(args)
    except requests.ConnectionError:
        print(f"❌ 连不上服务 {args.server}，请先启动：python -m app.main", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
