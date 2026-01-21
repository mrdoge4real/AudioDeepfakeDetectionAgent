# -*- coding: utf-8 -*-
import os
import re
import json
import sys
import traceback
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import autogen
from autogen import AssistantAgent, UserProxyAgent
from autogen.agentchat import Agent, ConversableAgent
from typing import Dict, Any, List

# ========== 导入独立的reference工具 ==========
try:
    import reference_tool
    # 直接映射函数，保留参数传递能力
    reference_tool_main = reference_tool.generate_reference_report
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import reference_tool
    reference_tool_main = reference_tool.generate_reference_report

# ========== 基础配置 ==========
sys.setrecursionlimit(2000)
load_dotenv()
sys.stdout.reconfigure(encoding='utf-8')

# ========== 全局状态管理 ==========
global_tool_results: Dict[str, Any] = {}
current_step: int = 1
tool_executed: List[str] = []
STEP_TO_TOOL = {
    1: "tool_convert_audio",
    2: "tool_anti_spoof_detection",
    3: "tool_asr_speaker_diarization",
    4: "tool_extract_features",
    5: "tool_generate_reference_report"
}
TOOL_REQUIRED_PARAMS = {
    "tool_convert_audio": ["audio_path"],
    "tool_anti_spoof_detection": ["standard_audio_path"],
    "tool_asr_speaker_diarization": ["standard_audio_path"],
    "tool_extract_features": ["audio_filename"],
    "tool_generate_reference_report": ["audio_filename"]
}

# ========== 专业知识库（音频检测相关） ==========
PROFESSIONAL_KNOWLEDGE = {
    "mfcc": """
### MFCC（梅尔频率倒谱系数）是什么？
MFCC是音频处理中最常用的声学特征之一，核心作用是模拟人类听觉系统对声音的感知：
1. **原理**：将音频的频谱转换到梅尔刻度（更符合人耳对低频敏感、高频不敏感的特性），再提取倒谱系数；
2. **在伪造检测中的作用**：真实音频和AI伪造音频的MFCC特征分布有明显差异：
   - 真实音频：MFCC均值/标准差分布自然，无异常突变；
   - 伪造音频：MFCC均值绝对值常超过0.5，整体标准差超过35.0141（我们的异常阈值）。
""",
    "异常值判定": """
### 音频伪造的异常值判定规则（基于LibriSpeech 500样本统计）：
1. **MFCC相关**：
   - 均值绝对值 > 0.5 → 判定为异常；
   - 整体标准差 > 35.0141 → 判定为异常；
   - 维度内标准差 > 44.6855 → 判定为异常；
2. **梅尔能量相关**：
   - 能量值 > -43.5002 或 < -65.9447 → 判定为异常；
3. **判定逻辑**：
   - 单个片段满足任意2个异常条件 → 标记为可疑伪造片段；
   - 可疑片段占比 > 10% → 整体判定为“存在伪造嫌疑”。
""",
    "梅尔能量": """
### 梅尔能量（Mel Energy）
1. **定义**：梅尔频谱上各频带的能量值，反映音频在不同频率段的能量分布；
2. **伪造特征**：AI生成的音频常出现梅尔能量“断层”——某一频段能量突然飙升/骤降，偏离正常范围（-65.9447 ~ -43.5002）。
""",
    "音频伪造检测流程": """
### 我们的音频伪造检测完整流程：
1. 格式标准化：将任意音频转为WAV格式（16kHz、单声道）；
2. 反伪造初检：识别明显的AI伪造特征；
3. ASR+说话人分割：定位说话人片段，排除无声音频；
4. 特征提取：提取MFCC、梅尔能量等核心特征；
5. 异常判定：对比阈值，标记可疑片段；
6. 生成报告：综合判定伪造风险等级。
""",
    "风险等级": """
### 伪造风险等级判定：
1. **低风险**：无异常片段，所有特征均在正常阈值内；
2. **中等风险**：1-3个可疑片段，占比≤10%；
3. **高风险**：可疑片段≥3个，或占比>10%。
"""
}

# ========== 路径处理工具 ==========
def normalize_path(path: str) -> str:
    path = path.strip().strip('"\'')
    abs_path = os.path.abspath(path)
    return os.path.normpath(abs_path)

def extract_audio_path_from_text(text: str) -> str:
    path_pattern = r'([A-Za-z]:[\\/][^:;"\'<>|?*\n]+?\.(flac|wav|mp3|wma))'
    match = re.search(path_pattern, text, re.IGNORECASE)
    if match:
        return normalize_path(match.group(1))
    return ""

# ========== 增强版意图识别（分层处理） ==========
def recognize_user_intent(user_input: str, chat_history: List[str] = None) -> Dict[str, Any]:
    if chat_history is None:
        chat_history = []
    user_input = user_input.strip()
    lower_input = user_input.lower()

    # 第一层：强规则识别核心指令（检测/退出）
    # 1. 退出意图
    quit_patterns = [r'exit', r'quit', r'退出', r'结束', r'拜拜']
    for pattern in quit_patterns:
        if re.search(pattern, user_input, re.IGNORECASE):
            return {
                "intent": "quit",
                "audio_path": "",
                "reply": "👋 再见！如有音频检测需求，随时可以再来找我～"
            }

    # 2. 检测意图（关键修改：兼容"纯路径"和"检测+路径"两种输入）
    audio_path = extract_audio_path_from_text(user_input)
    if audio_path:  # 只要能提取到音频路径，就判定为检测意图
        return {
            "intent": "detection",
            "audio_path": audio_path,
            "reply": ""
        }
    # 原检测意图判定（保留，兼容"检测+路径"）
    elif re.search(r'检测', user_input):
        return {
            "intent": "invalid_detection",
            "audio_path": "",
            "reply": "⚠️ 未识别到有效音频路径！\n请按格式输入：检测 + 音频文件绝对路径\n示例：检测 E:/DeepfakedetectionAgent/audio_files/uploads/LA_E_1000147.flac"
        }

    # 第二层：专业问题识别（匹配知识库关键词）
    for keyword, content in PROFESSIONAL_KNOWLEDGE.items():
        if re.search(keyword, lower_input):
            return {
                "intent": "professional_question",
                "audio_path": "",
                "reply": content
            }

    # 第三层：问候意图
    greeting_patterns = [r'你好', r'哈喽', r'hi', r'hello', r'嗨', r'早上好', r'下午好', r'晚上好']
    for pattern in greeting_patterns:
        if re.search(pattern, user_input, re.IGNORECASE):
            return {
                "intent": "greeting",
                "audio_path": "",
                "reply": "你好😊！我是智能音频伪造检测助手～\n✅ 我能帮你检测音频是否被伪造，也能解答MFCC、异常值判定等专业问题\n📌 输入示例：\n- 检测 E:/xxx.flac\n- MFCC是什么？\n- 异常值怎么样就算伪造？"
            }

    # 第四层：LLM兜底处理（闲聊/其他问题）
    return {
        "intent": "chat",
        "audio_path": "",
        "reply": ""  # 空回复，交给LLM处理
    }

# ========== LLM闲聊/专业解答Agent ==========
def get_chat_agent():
    """创建专门处理闲聊和专业追问的Agent"""
    chat_agent = AssistantAgent(
        name="ChatAgent",
        system_message=f"""
你是一个懂音频伪造检测的智能助手，遵循以下规则：
1. **专业问题**：优先使用以下知识库回答（{json.dumps(list(PROFESSIONAL_KNOWLEDGE.keys()), ensure_ascii=False)}），回答要通俗易懂，避免太专业的术语；
2. **闲聊问题**：友好、自然地回应（比如天气、打招呼、日常问题）；
3. **边界问题**：如果问题和音频检测无关且超出闲聊范围，礼貌说明你主要负责音频检测；
4. **格式要求**：回答分点清晰，用口语化的语言，避免生硬。

专业知识库参考：
{json.dumps(PROFESSIONAL_KNOWLEDGE, ensure_ascii=False, indent=2)}
""",
        llm_config={
            "config_list": config_list,
            "temperature": 0.7,  # 闲聊更自然
            "max_tokens": 1000
        }
    )
    # ========== 关键修改 ==========
    chat_user_proxy = UserProxyAgent(
        name="ChatUserProxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,  # 改为0，禁止自动二次回复
        code_execution_config={"use_docker": False},
        # 新增终止规则：只要Agent回复了，就终止对话
        is_termination_msg=lambda msg: True if msg.get("content") else False
    )
    return chat_agent, chat_user_proxy

# ========== 目录与阈值配置 ==========
# 标准化路径函数
def normalize_path_config(path):
    """标准化路径，解决Windows分隔符问题"""
    if not path:
        return ""
    return os.path.normpath(os.path.abspath(path))

# 软编码读取BASE_DIR + 严格校验
BASE_DIR = os.getenv("BASE_DIR")
BASE_DIR = normalize_path_config(BASE_DIR)
if not BASE_DIR or not os.path.exists(BASE_DIR):
    raise ValueError(f"❌ BASE_DIR配置无效！请在.env文件中配置：BASE_DIR")

UPLOAD_DIR = os.path.join(BASE_DIR, "audio_files", "uploads")
STANDARD_AUDIO_DIR = os.path.join(BASE_DIR, "audio_files", "standard_audio")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
for dir_path in [UPLOAD_DIR, STANDARD_AUDIO_DIR, OUTPUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

ANOMALY_THRESHOLDS = {
    "mfcc_mean_abs": 0.5,
    "mfcc_std_upper": 35.0141,
    "mfcc_inner_std_upper": 44.6855,
    "mel_energy_upper": -43.5002,
    "mel_energy_lower": -65.9447
}

# ========== LLM 配置 ==========
config_list = [
    {
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-reasoner"),
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": os.getenv("DEEPSEEK_API_BASE"),
    }
]

# ========== 工具声明 ==========
tools_declaration = [
    {
        "name": "tool_convert_audio",
        "description": "【步骤1/必须先执行】音频格式标准化转换",
        "parameters": {
            "type": "object",
            "properties": {
                "audio_path": {"type": "string", "description": "原始音频完整绝对路径"}
            },
            "required": ["audio_path"]
        }
    },
    {
        "name": "tool_anti_spoof_detection",
        "description": "【步骤2/仅步骤1成功后执行】反伪造检测",
        "parameters": {
            "type": "object",
            "properties": {
                "standard_audio_path": {"type": "string", "description": "步骤1返回的标准化音频路径"}
            },
            "required": ["standard_audio_path"]
        }
    },
    {
        "name": "tool_asr_speaker_diarization",
        "description": "【步骤3/仅步骤1成功后执行】ASR语音识别+说话人分割",
        "parameters": {
            "type": "object",
            "properties": {
                "standard_audio_path": {"type": "string", "description": "步骤1返回的标准化音频路径"}
            },
            "required": ["standard_audio_path"]
        }
    },
    {
        "name": "tool_extract_features",
        "description": "【步骤4/仅步骤1成功后执行】可疑片段特征提取",
        "parameters": {
            "type": "object",
            "properties": {
                "audio_filename": {"type": "string", "description": "步骤1返回的音频文件名（去后缀）"}
            },
            "required": ["audio_filename"]
        }
    },
    {
        "name": "tool_generate_reference_report",
        "description": "【步骤5/仅步骤4成功后执行】基于可疑片段特征生成伪造检测分析报告（调用独立reference_tool工具）",
        "parameters": {
            "type": "object",
            "properties": {
                "audio_filename": {"type": "string", "description": "步骤1返回的音频文件名（去后缀）"}
            },
            "required": ["audio_filename"]
        }
    }
]

llm_config = {
    "config_list": config_list,
    "temperature": 0.0,
    "timeout": 60,
    "functions": tools_declaration,
    "max_tokens": 4096
}

# ========== 业务工具函数 ==========

try:
    from anti_spoof_detector import run_anti_spoof_detection
    from asr_diarization import extract_asr_with_speaker_diarization
    from audio_converter import convert_audio_to_standard
    from suspicious_feature_extractor import extract_suspicious_segments_features as real_suspicious_feature_extractor
except ImportError as e:
    # 抛出更明确的异常，提示问题原因
    raise ImportError(
        f"导入核心音频处理模块失败：{e}\n"
        "请确保 anti_spoof_detector、asr_diarization 等模块已存在，且依赖已安装。"
    ) from e

# 工具1：音频转换
def tool_convert_audio(audio_path: str) -> str:
    global global_tool_results, current_step, tool_executed
    try:
        audio_path = normalize_path(audio_path)
        convert_result_str = convert_audio_to_standard(audio_path)
        if isinstance(convert_result_str, dict):
            convert_result_str = json.dumps(convert_result_str)
        convert_result = json.loads(convert_result_str)
        result = {
            "success": convert_result.get("success", False),
            "error": convert_result.get("error", None),
            "audio_filename": convert_result.get("audio_filename", ""),
            "standard_audio_path": normalize_path(convert_result.get("audio_path", ""))
        }
        global_tool_results["tool_convert_audio"] = result
        if "tool_convert_audio" not in tool_executed:
            tool_executed.append("tool_convert_audio")
        current_step = 2 if result["success"] else 1
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        result = {"success": False, "error": f"转换失败: {str(e)}", "audio_filename": "", "standard_audio_path": ""}
        global_tool_results["tool_convert_audio"] = result
        current_step = 1
        return json.dumps(result, ensure_ascii=False)

# 工具2：反伪造检测
def tool_anti_spoof_detection(standard_audio_path: str) -> str:
    global global_tool_results, current_step, tool_executed
    try:
        standard_audio_path = normalize_path(standard_audio_path)
        spoof_result_str = run_anti_spoof_detection(standard_audio_path)
        if isinstance(spoof_result_str, dict):
            spoof_result_str = json.dumps(spoof_result_str)
        spoof_result = json.loads(spoof_result_str)
        suspicious_segments = spoof_result.get("data", {}).get("suspicious_segments", [])
        result = {
            "success": spoof_result.get("success", False),
            "error": spoof_result.get("error", ""),
            "suspicious_segments": suspicious_segments,  # 可疑片段列表
            "segment_count": len(suspicious_segments)
        }
        global_tool_results["tool_anti_spoof_detection"] = result
        # 【新增】额外保存可疑片段到全局，供reference_tool直接读取
        global_tool_results["anti_spoof_suspicious_segments"] = suspicious_segments
        if "tool_anti_spoof_detection" not in tool_executed:
            tool_executed.append("tool_anti_spoof_detection")
        current_step = 3 if result["success"] else 2
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        result = {"success": False, "error": f"检测失败: {str(e)}", "suspicious_segments": [], "segment_count": 0}
        global_tool_results["tool_anti_spoof_detection"] = result
        global_tool_results["anti_spoof_suspicious_segments"] = []
        current_step = 2
        return json.dumps(result, ensure_ascii=False)

# 工具3：ASR+说话人分割
def tool_asr_speaker_diarization(standard_audio_path: str) -> str:
    global global_tool_results, current_step, tool_executed
    try:
        standard_audio_path = normalize_path(standard_audio_path)
        asr_result_str = extract_asr_with_speaker_diarization(standard_audio_path)
        if isinstance(asr_result_str, dict):
            asr_result_str = json.dumps(asr_result_str)
        asr_result = json.loads(asr_result_str)
        result = {
            "success": asr_result.get("success", False),
            "error": asr_result.get("error", ""),
            "full_text": asr_result.get("full_text", ""),
            "speaker_count": asr_result.get("total_speakers", 0)
        }
        global_tool_results["tool_asr_speaker_diarization"] = result
        if "tool_asr_speaker_diarization" not in tool_executed:
            tool_executed.append("tool_asr_speaker_diarization")
        current_step = 4 if result["success"] else 3
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        result = {"success": False, "error": f"ASR失败: {str(e)}", "full_text": "", "speaker_count": 0}
        global_tool_results["tool_asr_speaker_diarization"] = result
        current_step = 3
        return json.dumps(result, ensure_ascii=False)

# 工具4：特征提取（修复lightweight_feature_data未定义问题）
def tool_extract_features(audio_filename: str) -> str:
    global global_tool_results, current_step, tool_executed
    try:
        feature_result_str = real_suspicious_feature_extractor(audio_filename)
        if isinstance(feature_result_str, dict):
            feature_result_str = json.dumps(feature_result_str)
        feature_result = json.loads(feature_result_str)
        
        # 直接使用原始结果，不再调用lightweight_feature_data
        result = {
            "success": feature_result.get("success", False),
            "error": feature_result.get("error", ""),
            "segments_features": feature_result.get("segments_features", []),
            "total_suspicious_segments": feature_result.get("total_suspicious_segments", 0)
        }
        global_tool_results["tool_extract_features"] = result
        if "tool_extract_features" not in tool_executed:
            tool_executed.append("tool_extract_features")
        current_step = 5 if result["success"] else 4
        
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        result = {"success": False, "error": f"特征提取失败: {str(e)}", "segments_features": [], "total_suspicious_segments": 0}
        global_tool_results["tool_extract_features"] = result
        current_step = 4
        return json.dumps(result, ensure_ascii=False)

def tool_generate_reference_report(audio_filename: str) -> str:
    global global_tool_results, current_step, tool_executed
    try:
        # 1. 调用生成函数（新增调试）
        print(f"📢 开始生成MD报告，音频名：{audio_filename}")
        ref_tool_result = reference_tool.generate_reference_report(audio_filename)
        
        # 2. 打印返回结果（关键调试）
        print(f"📢 generate_reference_report返回：{ref_tool_result}")
        
        reference_result = {
            "success": ref_tool_result.get("success", False),
            "audio_filename": audio_filename,
            "report_path": ref_tool_result.get("report_path", ""),
            "error": ref_tool_result.get("error", ""),
            "suspicious_segments": global_tool_results.get("tool_anti_spoof_detection", {}).get("suspicious_segments", []),
            "asr_text": global_tool_results.get("tool_asr_speaker_diarization", {}).get("full_text", "")
        }

        # 3. 保存到全局状态
        global_tool_results["tool_generate_reference_report"] = reference_result
        tool_executed.append("tool_generate_reference_report")
        current_step = 6

        # 4. 调试打印最终路径
        print(f"📢 最终存入全局的report_path：{reference_result['report_path']}")
        
        return json.dumps(reference_result, ensure_ascii=False, indent=2)
    except Exception as e:
        # 新增：打印完整异常栈
        print(f"❌ 调用reference_tool失败：{str(e)}")
        traceback.print_exc()
        error_result = {
            "success": False,
            "error": f"Reference工具调用异常：{str(e)}\n{traceback.format_exc()}",
            "report_path": "",
            "audio_filename": audio_filename,
            "suspicious_segments": [],
            "asr_text": ""
        }
        global_tool_results["tool_generate_reference_report"] = error_result
        tool_executed.append("tool_generate_reference_report")
        current_step = 6
        return json.dumps(error_result, ensure_ascii=False, indent=2)

# ========== 自定义FeedbackUserProxyAgent ==========
class FeedbackUserProxyAgent(UserProxyAgent):
    def _extract_function_call(self, message: str) -> Dict[str, Any]:
        try:
            json_match = re.search(r'\{[\s\S]*\}', message.strip())
            if json_match:
                func_call = json.loads(json_match.group())
                if "name" in func_call and "parameters" in func_call:
                    return func_call
        except json.JSONDecodeError:
            pass
        return {}

    def _check_params(self, tool_name: str, params: Dict[str, Any]) -> str:
        required_params = TOOL_REQUIRED_PARAMS.get(tool_name, [])
        missing_params = [p for p in required_params if p not in params]
        if missing_params:
            return f"参数错误：工具 {tool_name} 缺少必填参数 {missing_params}，正确参数为 {required_params}"
        return ""

    def generate_reply(self, messages: List[Dict[str, Any]], sender: Agent, **kwargs) -> str:
        global global_tool_results, current_step
        # ===== 新增终止判定：如果current_step=6 且 最后一条消息包含"流程结束"，直接返回None =====
        last_msg = messages[-1]["content"].strip() if messages else ""
        if current_step == 6 and "流程结束" in last_msg:
            return None  # 返回None会强制终止AutoGen对话循环
        
        # 以下原有逻辑保持不变
        if current_step == 6:
            return "所有工具执行完成，可生成最终报告"

        last_message = messages[-1]["content"].strip()
        if not last_message:
            return None

        func_call = self._extract_function_call(last_message)
        if not func_call:
            if current_step == 6:
                return None
            else:
                return f"错误：必须输出工具调用JSON，当前步骤 {current_step} 应调用工具 {STEP_TO_TOOL[current_step]}"

        # 以下原有逻辑不变...
        tool_name = func_call["name"]
        tool_params = func_call["parameters"]

        if tool_name not in STEP_TO_TOOL.values():
            error_msg = f"错误：未知工具 {tool_name}，仅允许调用 {list(STEP_TO_TOOL.values())}"
            return error_msg
        param_error = self._check_params(tool_name, tool_params)
        if param_error:
            return param_error

        expected_tool = STEP_TO_TOOL.get(current_step)
        if tool_name != expected_tool:
            return f"步骤错误：当前步骤 {current_step} 必须调用 {expected_tool}，不能调用 {tool_name}"

        tool_functions = {
            "tool_convert_audio": tool_convert_audio,
            "tool_anti_spoof_detection": tool_anti_spoof_detection,
            "tool_asr_speaker_diarization": tool_asr_speaker_diarization,
            "tool_extract_features": tool_extract_features,
            "tool_generate_reference_report": tool_generate_reference_report
        }
        try:
            print(f"\n🔧 执行工具：{tool_name} | 参数：{tool_params}")
            if tool_name == "tool_convert_audio":
                tool_result = tool_functions[tool_name](tool_params["audio_path"])
            elif tool_name in ["tool_anti_spoof_detection", "tool_asr_speaker_diarization"]:
                tool_result = tool_functions[tool_name](tool_params["standard_audio_path"])
            elif tool_name in ["tool_extract_features", "tool_generate_reference_report"]:
                tool_result = tool_functions[tool_name](tool_params["audio_filename"])
            else:
                tool_result = json.dumps({"success": False, "error": "未知工具"})
        except Exception as e:
            tool_result = json.dumps({"success": False, "error": f"工具执行异常: {str(e)}"})

        tool_result_dict = json.loads(tool_result)
        if current_step == 5 and tool_result_dict.get("success"):
            next_step = 6
        else:
            next_step = current_step + 1 if tool_result_dict.get("success") else current_step
        next_tool = STEP_TO_TOOL.get(next_step, "生成最终检测报告")

        if next_step == 6:
            feedback_msg = f"""
    【工具执行结果】{tool_name}：执行成功
    【全局状态更新】所有工具执行完成，即将生成最终检测报告
    """
            current_step = 6  # 立即标记为终止步骤
        else:
            feedback_msg = f"""
    【工具执行结果】{tool_name}：
    {json.dumps(tool_result_dict, ensure_ascii=False, indent=2)}

    【全局状态更新】
    - 当前步骤：{current_step} → {next_step}
    - 已执行工具：{tool_executed}
    - 下一步操作：{"调用工具 " + next_tool if next_step <=5 else "生成最终检测报告"}

    【强制规则】
    1. 若工具执行失败，请修复参数后重新调用同一工具
    2. 若工具执行成功，请按步骤调用下一个工具
    3. 所有工具执行完成后，生成报告时必须100%使用工具返回结果，禁止编造
    """
        return feedback_msg

# ========== 检测智能体 ==========
detection_agent = AssistantAgent(
    name="AudioDetectionAgent",
    system_message=f"""
你是严格遵守规则的音频伪造检测专家，你的回复必须遵守以下规则：

【核心规则】
1. **格式要求**：步骤1-5执行期间，回复**只能是纯工具调用JSON**，禁止任何其他文字、解释、标点！
   JSON结构：{{"name":"工具名","parameters":{{"参数名":"参数值"}}}}
2. **步骤要求**：必须按 步骤1→步骤2→步骤3→步骤4→步骤5 执行，不能跳过、乱序、重复
   - 步骤1：调用 tool_convert_audio，参数 audio_path = 用户指定的音频路径
   - 步骤2：仅步骤1成功后，调用 tool_anti_spoof_detection，参数 standard_audio_path = 步骤1返回值
   - 步骤3：仅步骤1成功后，调用 tool_asr_speaker_diarization，参数 standard_audio_path = 步骤1返回值
   - 步骤4：仅步骤1成功后，调用 tool_extract_features，参数 audio_filename = 步骤1返回值
   - 步骤5：仅步骤4成功后，调用 tool_generate_reference_report，参数 audio_filename = 步骤1返回值
3. **终止规则**：
   - 调用完 tool_generate_reference_report 后，再生成最终报告，且报告末尾必须添加关键词「流程结束」
   - 禁止在未调用 tool_generate_reference_report 的情况下直接生成报告
4. **参数要求**：所有参数必须来自上一步工具的返回结果，禁止编造、修改
""",
    llm_config=llm_config,
    function_map={
        "tool_convert_audio": tool_convert_audio,
        "tool_anti_spoof_detection": tool_anti_spoof_detection,
        "tool_asr_speaker_diarization": tool_asr_speaker_diarization,
        "tool_extract_features": tool_extract_features,
        "tool_generate_reference_report": tool_generate_reference_report
    }
)

# ========== 初始化反馈代理 ==========
user_proxy = FeedbackUserProxyAgent(
    name="FeedbackUserProxy",
    system_message="你是用户代理，负责执行工具并反馈结果",
    code_execution_config={"work_dir": "work_dir", "use_docker": False},
    human_input_mode="NEVER",
    max_consecutive_auto_reply=1,  # 仅允许1轮自动回复，避免循环
    # 增强终止判定：包含"完成"或"结束"关键词就终止
    is_termination_msg=lambda msg: current_step == 6 or any(word in msg.get("content", "").lower() for word in ["完成", "结束", "流程结束"])
)

# ========== 报告生成函数 ==========
def generate_detection_report(tool_results: Dict[str, Any]) -> str:
    # ========== 1. 优先提取全局工具的真实执行数据 ==========
    anti_spoof_result = tool_results.get("tool_anti_spoof_detection", {})
    suspicious_segments = anti_spoof_result.get("suspicious_segments", [])
    suspicious_count = len(suspicious_segments)
    
    asr_result = tool_results.get("tool_asr_speaker_diarization", {})
    asr_text = asr_result.get("full_text", "未识别到语音内容")
    
    convert_result = tool_results.get("tool_convert_audio", {})
    audio_filename = convert_result.get("audio_filename", "未知")

    # ========== 2. 核心修复：主动调用 reference_tool 生成 MD 文件 ==========
    ref_full_content = ""
    ref_report_path = ""
    if audio_filename != "未知":
        # 手动调用你验证过的 MD 生成函数
        md_result = reference_tool.generate_reference_report(audio_filename)
        if md_result.get("success"):
            ref_report_path = md_result.get("report_path")
            # 读取生成好的 MD 文件内容
            if os.path.exists(ref_report_path):
                with open(ref_report_path, "r", encoding="utf-8") as f:
                    ref_full_content = f.read()
                print(f"✅ 主动生成并读取MD报告：{ref_report_path}")
            else:
                ref_full_content = f"MD文件生成成功但读取失败：{ref_report_path}"
        else:
            ref_full_content = f"MD文件生成失败：{md_result.get('error')}"
    else:
        # 兜底信息（仅音频名未知时用）
        suspicious_time_list = []
        for idx, seg in enumerate(suspicious_segments):
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            suspicious_time_list.append(f"第{idx+1}段：{start}s - {end}s")
        suspicious_time_str = "\n    - ".join(suspicious_time_list) if suspicious_time_list else "无"
        
        ref_full_content = f"""
### 音频检测真实执行数据（必填）
1. 音频文件名称：{audio_filename}
2. 可疑片段总数量：{suspicious_count}个
3. 可疑片段时间段：
    - {suspicious_time_str}
4. ASR语音识别完整内容：{asr_text}
5. 异常特征阈值参考：
   - 梅尔能量正常范围：-65.9447dB ~ -43.5002dB
   - MFCC均值绝对值正常阈值：≤0.5
   - MFCC整体标准差正常阈值：≤35.0141
6. Reference报告状态：文件不存在（{ref_report_path}）
"""

    # ========== 3. 构造提示词（LLM 读取真实 MD 内容） ==========
    prompt = f"""
### 强制指令（必须严格遵守）
请基于以下完整的音频伪造检测MD报告内容，生成总结报告，**必须包含且明确标注以下4个核心字段**：
1. 【可疑片段数量】：明确写出具体数字（如：1个）；
2. 【可疑片段时间段】：列出所有片段的起止时间（如：第1段：0.0s - 2.7s）；
3. 【ASR语音内容】：完整输出识别到的语音文本；
4. 【风险等级+异常特征】：包含数值与阈值的精准对比。

### 格式要求
- 分点清晰，每个核心字段单独成项，标注明确的小标题；
- 异常特征描述示例：梅尔能量均值(-43.2dB)偏高（正常≤-43.5002dB）；
- 总字数不超过400字，语言专业简洁；
- 严格基于报告内容，禁止编造任何数据；
- 报告末尾必须添加「流程结束」关键词。

### 完整MD报告内容
{ref_full_content}
"""

    # ========== 4. LLM 生成最终报告 ==========
    report_agent = AssistantAgent(
        name="ReportAgent",
        system_message="""
你是严格遵守规则的音频检测报告总结专家，必须满足以下强制要求：
1. 总结报告中**必须明确包含**：可疑片段数量（带数字）、可疑片段时间段（带具体秒数）、ASR语音完整内容、风险等级+异常特征数值对比；
2. 缺失任何一个字段，直接判定回答失败；
3. 异常特征必须标注具体数值和正常阈值的对比；
4. 禁止遗漏、简写或模糊化任何核心字段；
5. 语言简洁专业，分点呈现。
        """,
        llm_config={
            "config_list": config_list,
            "temperature": 0.0,
            "max_tokens": 1500,
            "timeout": 30
        }
    )
    report_user_proxy = UserProxyAgent(
        name="ReportUserProxy",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config={"use_docker": False}
    )
    chat_result = report_user_proxy.initiate_chat(
        recipient=report_agent,
        message=prompt,
        clear_history=True
    )
    
    final_report = ""
    for msg in chat_result.chat_history:
        # 只取 ReportAgent 发送的消息
        if msg["name"] == "ReportAgent":
            final_report = msg["content"]
            break  # 取第一条有效回复后立即退出

    # 确保末尾有「流程结束」
    if "流程结束" not in final_report:
        final_report += "\n\n流程结束"

    return final_report

# ========== 主对话入口（带记忆和分层意图） ==========
def start_detection_chat():
    print("="*80)
    print("🎙️ 智能音频伪造检测助手（支持专业问答+闲聊）")
    print("="*80)
    print("✅ 我能做：")
    print("  1. 检测音频是否被伪造（格式：检测 + 音频绝对路径）")
    print("  2. 解答专业问题（比如：MFCC是什么？异常值怎么判定？）")
    print("  3. 日常闲聊（打招呼、简单问答）")
    print("🚪 退出指令：exit/退出/拜拜")
    print("="*80 + "\n")
    
    chat_history = []
    
    while True:
        user_input = input("请输入你的指令：").strip()
        if not user_input:
            print("⚠️  输入不能为空！")
            continue
        
        chat_history.append(f"用户：{user_input}")
        
        intent_result = recognize_user_intent(user_input, chat_history)
        intent_type = intent_result["intent"]
        audio_path = intent_result["audio_path"]
        reply_msg = intent_result["reply"]
        
        if intent_type == "quit":
            print(f"\n🤖 {reply_msg}\n")
            break
        elif intent_type == "greeting":
            print(f"\n🤖 {reply_msg}\n")
            chat_history.append(f"助手：{reply_msg}")
            continue
        elif intent_type == "invalid_detection":
            print(f"\n🤖 {reply_msg}\n")
            chat_history.append(f"助手：{reply_msg}")
            continue
        elif intent_type == "professional_question":
            print(f"\n🤖 {reply_msg}\n")
            chat_history.append(f"助手：{reply_msg}")
            continue
        elif intent_type == "detection":
            # 重置工具执行状态
            global global_tool_results, current_step, tool_executed
            global_tool_results = {}
            current_step = 1
            tool_executed = []
            
            print(f"\n🚀 开始处理指令：检测 {audio_path}")
            # 关键修复：设置 max_consecutive_auto_reply=1，只执行必要的工具调用，不重复生成报告
            chat_result = user_proxy.initiate_chat(
                recipient=detection_agent,
                message=f"检测 {audio_path}",
                clear_history=True,
                max_consecutive_auto_reply=1  # 仅1轮回复，执行完工具就停
            )
            
            # 手动提取并打印最终报告，不再依赖detection_agent重复输出
            if "tool_generate_reference_report" in tool_executed or current_step == 6:
                # 在 start_detection_chat 函数里，生成报告前加：
                print("===== 验证反伪造检测数据 =====")
                print("可疑片段数：", len(global_tool_results.get("tool_anti_spoof_detection", {}).get("suspicious_segments", [])))
                final_report = generate_detection_report(global_tool_results)

                print("\n" + "="*80)
                print("🎯 最终检测报告：")
                print("="*80)
                print(final_report)
                print("="*80 + "\n")
                
                chat_history.append(f"助手：已完成音频{audio_path}的检测，报告如下：{final_report}")
            else:
                print("\n❌ 工具执行未完成，无法生成报告！\n")
                chat_history.append(f"助手：检测失败，工具执行未完成")
            # 强制回到输入框
            continue
        elif intent_type == "chat":
            chat_agent, chat_user_proxy = get_chat_agent()
            context = "\n".join(chat_history[-5:])
            prompt = f"""
上下文：
{context}

用户当前问题：{user_input}

请友好、自然地回答用户问题，注意：
1. 如果是音频检测相关问题，优先用专业知识库回答；
2. 如果是闲聊问题，保持轻松自然；
3. 如果超出你的能力范围，礼貌说明你主要负责音频检测。
"""
            chat_result = chat_user_proxy.initiate_chat(
                recipient=chat_agent,
                message=prompt,
                clear_history=True
            )
            llm_reply = chat_result.chat_history[-1]['content'] if chat_result.chat_history else "我还在学习中，暂时回答不了这个问题😜"
            print(f"\n🤖 {llm_reply}\n")
            chat_history.append(f"助手：{llm_reply}")
            continue

# ========== 程序入口 ==========
if __name__ == "__main__":
    start_detection_chat()