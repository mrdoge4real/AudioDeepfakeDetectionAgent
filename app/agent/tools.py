# -*- coding: utf-8 -*-
"""Agent 可用工具集：检测任务查询、报告读取、知识库检索。"""
import json

from app.agent.knowledge import search_knowledge
from app.core import tasks

# ---------------------------------------------------------------------------
# OpenAI tools schema
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_task_status",
            "description": "查询音频检测任务的状态和进度。可通过任务ID或音频文件名查询；都不传则返回最近一次完成的任务。",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "任务ID（可选）"},
                    "filename": {"type": "string", "description": "音频文件名关键词（可选）"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_report",
            "description": "获取已完成检测任务的完整 Markdown 分析报告内容。可通过任务ID或音频文件名查询；都不传则取最近一次完成的报告。",
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "任务ID（可选）"},
                    "filename": {"type": "string", "description": "音频文件名关键词（可选）"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_history",
            "description": "列出最近的音频检测任务历史（最多20条），含状态、进度和结果摘要。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_knowledge",
            "description": "检索音频伪造检测专业知识库，解释 MFCC、梅尔能量、异常值判定、风险等级、检测流程等概念。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "要查询的概念关键词"},
                },
                "required": ["query"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# 工具实现
# ---------------------------------------------------------------------------

def _resolve_task(task_id: str = "", filename: str = "") -> dict:
    if task_id:
        task = tasks.get_task_full(task_id)
    elif filename:
        task = tasks.find_task_by_filename(filename)
    else:
        task = tasks.find_recent_done_task()
    return task or {}


def tool_get_task_status(task_id: str = "", filename: str = "") -> str:
    task = _resolve_task(task_id, filename)
    if not task:
        return json.dumps({"found": False, "message": "未找到匹配的检测任务"}, ensure_ascii=False)
    task.pop("report_content", None)
    return tasks.task_to_json({"found": True, "task": task})


def tool_get_report(task_id: str = "", filename: str = "") -> str:
    task = _resolve_task(task_id, filename)
    if not task:
        return json.dumps({"found": False, "message": "未找到匹配的检测任务"}, ensure_ascii=False)
    if task["status"] != "done":
        return json.dumps({
            "found": True,
            "message": f"任务尚未完成（当前状态：{task['status']}，步骤：{task['step_name']}）",
        }, ensure_ascii=False)
    return json.dumps({
        "found": True,
        "filename": task["filename"],
        "report_content": task["report_content"],
    }, ensure_ascii=False)


def tool_list_history() -> str:
    history = tasks.list_tasks(limit=20)
    return tasks.task_to_json({"count": len(history), "tasks": history})


def tool_explain_knowledge(query: str) -> str:
    content = search_knowledge(query)
    if not content:
        return json.dumps({
            "found": False,
            "message": "知识库未收录该概念，可用关键词：MFCC、梅尔能量、异常值判定、风险等级、音频伪造检测流程",
        }, ensure_ascii=False)
    return json.dumps({"found": True, "content": content}, ensure_ascii=False)


TOOL_HANDLERS = {
    "get_task_status": tool_get_task_status,
    "get_report": tool_get_report,
    "list_history": tool_list_history,
    "explain_knowledge": tool_explain_knowledge,
}
