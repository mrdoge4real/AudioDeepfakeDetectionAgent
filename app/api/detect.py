# -*- coding: utf-8 -*-
"""检测相关接口：上传音频、查询任务进度/结果、历史列表。"""
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import UPLOAD_DIR
from app.core import tasks

router = APIRouter(prefix="/api", tags=["detect"])

ALLOWED_SUFFIXES = {".flac", ".wav", ".mp3", ".m4a", ".wma", ".ogg", ".aac"}


@router.post("/detect")
async def create_detection(file: UploadFile = File(...)):
    """multipart/form-data 上传音频，立即返回 task_id，后台异步执行检测。"""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"不支持的音频格式 {suffix}，支持：{sorted(ALLOWED_SUFFIXES)}")

    safe_name = f"{uuid.uuid4().hex[:8]}_{Path(file.filename).name}"
    save_path = UPLOAD_DIR / safe_name
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    task = tasks.create_task(filename=file.filename, audio_path=str(save_path))
    tasks.submit_detection(task["id"])
    return {"task_id": task["id"], "filename": file.filename, "status": "pending"}


@router.get("/tasks")
def list_tasks():
    return {"tasks": tasks.list_tasks(limit=20)}


@router.get("/tasks/{task_id}")
def get_task(task_id: str, include_report: bool = False):
    task = tasks.get_task(task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
    if include_report and task["status"] == "done":
        full = tasks.get_task_full(task_id)
        task["report_content"] = full.get("report_content", "")
    return task
