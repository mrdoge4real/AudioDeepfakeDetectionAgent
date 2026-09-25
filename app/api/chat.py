# -*- coding: utf-8 -*-
"""对话接口：与 Agent 多轮对话。"""
import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    message: str
    session_id: str = ""


class ChatResponse(BaseModel):
    reply: str
    session_id: str


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "消息不能为空")
    session_id = req.session_id or uuid.uuid4().hex[:12]

    try:
        from app.agent.agent import get_agent
        reply = get_agent().chat(session_id, req.message.strip())
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(500, f"Agent 处理失败：{e}")
    return ChatResponse(reply=reply, session_id=session_id)
