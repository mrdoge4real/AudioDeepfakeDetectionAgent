# -*- coding: utf-8 -*-
"""对话 Agent：基于原生 function calling 的手写循环，LLM 自主决定调用哪些工具。"""
import json

from openai import OpenAI

from app.agent.knowledge import KNOWLEDGE_BASE
from app.agent.tools import TOOL_HANDLERS, TOOL_SCHEMAS
from app.config import AGENT_MAX_TOOL_ROUNDS, LLM_API_BASE, LLM_API_KEY, LLM_MODEL
from app.core import tasks

SYSTEM_PROMPT = f"""
你是一个智能音频伪造检测助手，服务于一个音频 Deepfake 检测系统。

【你的能力】
1. 通过工具查询检测任务的状态、进度和历史记录（get_task_status / list_history）
2. 通过工具获取已完成的检测报告并向用户解读（get_report）
3. 通过工具检索专业知识库，解释 MFCC、梅尔能量、异常判定、风险等级等概念（explain_knowledge）

【行为规则】
1. 用户询问检测结果、进度、历史时，**必须先调用工具获取真实数据**，禁止编造任何数值；
2. 解读报告时保留关键数值与阈值对比（如：梅尔能量均值(-43.2dB)偏高，正常≤-43.5002dB）；
3. 用户问专业概念时优先调用 explain_knowledge，再用自己的话通俗解释；
4. 用户想检测新音频时，引导其通过页面上传音频文件（你无法接收文件）；
5. 闲聊友好自然；与音频检测无关且超出闲聊范围的问题，礼貌说明你的主要职责；
6. 回答分点清晰、口语化，中文为主。

【知识库覆盖主题】{list(KNOWLEDGE_BASE.keys())}
"""


class ChatAgent:
    def __init__(self):
        if not LLM_API_KEY:
            raise RuntimeError("未配置 LLM_API_KEY，请在 .env 中填写")
        self.client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)
        self.model = LLM_MODEL

    def _call_llm(self, messages: list) -> dict:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.7,
        )
        return resp.choices[0].message

    def chat(self, session_id: str, user_message: str) -> str:
        tasks.add_chat_message(session_id, "user", user_message)

        history = tasks.get_chat_history(session_id, limit=20)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

        for _ in range(AGENT_MAX_TOOL_ROUNDS):
            msg = self._call_llm(messages)

            if not msg.tool_calls:
                reply = msg.content or "（无回复）"
                tasks.add_chat_message(session_id, "assistant", reply)
                return reply

            # 执行所有工具调用，回填结果后继续循环
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            })
            for tc in msg.tool_calls:
                handler = TOOL_HANDLERS.get(tc.function.name)
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    result = handler(**args) if handler else json.dumps(
                        {"error": f"未知工具：{tc.function.name}"}, ensure_ascii=False
                    )
                except Exception as e:
                    result = json.dumps({"error": f"工具执行异常：{e}"}, ensure_ascii=False)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        fallback = "抱歉，处理这个问题时步骤过多，请换个方式再问我一次～"
        tasks.add_chat_message(session_id, "assistant", fallback)
        return fallback


_agent = None


def get_agent() -> ChatAgent:
    global _agent
    if _agent is None:
        _agent = ChatAgent()
    return _agent
