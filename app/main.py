# -*- coding: utf-8 -*-
"""FastAPI 入口：挂载 API 路由与静态页面，启动时预热模型。"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import BASE_DIR
from app.core import tasks

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

STATIC_DIR = Path(__file__).parent / "web" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    tasks.init_db()
    # 预热重型模型（失败不阻断，首次调用时重试）；置环境变量 SKIP_WARMUP=1 可跳过
    import os
    if os.getenv("SKIP_WARMUP") != "1":
        from app.core.model_registry import warmup
        warmup(include_diarization=False)
    yield


app = FastAPI(title="音频伪造检测服务", lifespan=lifespan)

from app.api import chat, detect  # noqa: E402
app.include_router(detect.router)
app.include_router(chat.router)


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
def health():
    return {"status": "ok", "base_dir": str(BASE_DIR)}


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
