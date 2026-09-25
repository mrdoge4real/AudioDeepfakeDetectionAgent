# -*- coding: utf-8 -*-
"""集中配置管理：所有环境变量、路径、阈值唯一定义于此。"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
def _resolve_base_dir() -> Path:
    raw = (os.getenv("BASE_DIR") or "").strip().strip('"\'')
    # 容忍 .env 中的行内注释（如 BASE_DIR=/path#注释）
    if "#" in raw:
        raw = raw.split("#", 1)[0].strip()
    candidate = Path(raw).expanduser() if raw else None
    if candidate and candidate.exists():
        return candidate.resolve()
    if raw:
        print(f"⚠️ .env 中的 BASE_DIR 无效（{raw}），回退到仓库根目录")
    return Path(__file__).resolve().parent.parent


BASE_DIR = _resolve_base_dir()
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
STANDARD_AUDIO_DIR = DATA_DIR / "standard_audio"
OUTPUT_ROOT = DATA_DIR / "outputs"
ANTI_SPOOF_DIR = OUTPUT_ROOT / "anti_spoof"
ASR_OUTPUT_DIR = OUTPUT_ROOT / "asr"
SUSPICIOUS_FEATURE_DIR = OUTPUT_ROOT / "suspicious_features"
REPORT_DIR = OUTPUT_ROOT / "reports"
DB_PATH = DATA_DIR / "app.db"

for _d in (UPLOAD_DIR, STANDARD_AUDIO_DIR, ANTI_SPOOF_DIR, ASR_OUTPUT_DIR,
           SUSPICIOUS_FEATURE_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# LLM（OpenAI 兼容端点，默认 DeepSeek；换 qwen 只改 .env）
# ---------------------------------------------------------------------------
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# HuggingFace（pyannote 说话人分割需要）
HF_TOKEN = os.getenv("HF_TOKEN", "")

# ---------------------------------------------------------------------------
# 音频处理参数
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
# 反伪造滑窗
WINDOW_SIZE = 0.5
HOP_SIZE = 0.1
FAKE_THRESHOLD = 0.7
# Whisper 模型规格
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
# Deepfake 检测模型
DEEPFAKE_MODEL_NAME = os.getenv(
    "DEEPFAKE_MODEL_NAME", "MelodyMachine/Deepfake-audio-detection-V2"
)
PYANNOTE_PIPELINE = os.getenv("PYANNOTE_PIPELINE", "pyannote/speaker-diarization")

# MFCC / 梅尔频谱参数
MFCC_PARAMS = {"n_mfcc": 13, "n_fft": 512, "hop_length": 160}
MEL_PARAMS = {"n_fft": 512, "hop_length": 160, "n_mels": 80}

# ---------------------------------------------------------------------------
# 异常判定阈值（基于 LibriSpeech dev-clean 500 条真人语音统计，3σ 原则）
# ---------------------------------------------------------------------------
ANOMALY_THRESHOLDS = {
    "mfcc_mean_abs": 0.5,
    "mfcc_std_upper": 35.0141,
    "mfcc_inner_std_upper": 44.6855,
    "mel_energy_upper": -43.5002,
    "mel_energy_lower": -65.9447,
}

# Agent 循环上限
AGENT_MAX_TOOL_ROUNDS = 8
