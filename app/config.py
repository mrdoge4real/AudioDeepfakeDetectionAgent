# -*- coding: utf-8 -*-
"""集中配置管理：所有环境变量、路径、阈值唯一定义于此。"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------
def _clean_env(key: str, default: str = "") -> str:
    """读取环境变量并剥离行内注释（.env 中 `KEY=value#注释` 的写法会污染值）。"""
    raw = (os.getenv(key) or "").strip().strip('"\'')
    if "#" in raw:
        raw = raw.split("#", 1)[0].strip()
    return raw or default


def _resolve_base_dir() -> Path:
    raw = _clean_env("BASE_DIR")
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
# 模型缓存：统一下载到项目 models/ 目录
# 必须在 transformers / huggingface_hub 导入前设置才生效，
# 本文件总是最先被导入（app 各模块均从 config 取路径）。
# ---------------------------------------------------------------------------
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(MODELS_DIR / "huggingface" / "hub"))

# ---------------------------------------------------------------------------
# LLM（OpenAI 兼容端点，默认 DeepSeek；换 qwen 只改 .env）
# ---------------------------------------------------------------------------
LLM_API_KEY = _clean_env("LLM_API_KEY")
LLM_API_BASE = _clean_env("LLM_API_BASE", "https://api.deepseek.com")
LLM_MODEL = _clean_env("LLM_MODEL", "deepseek-chat")
# 部分模型（如 kimi-for-coding）不允许自定义 temperature，置空则不传该参数
LLM_TEMPERATURE = _clean_env("LLM_TEMPERATURE", "0.7")

# HuggingFace（pyannote 说话人分割需要）
HF_TOKEN = _clean_env("HF_TOKEN")
# .env 中被注释污染的 HF_TOKEN 不能留在环境变量里，
# 否则 huggingface_hub 会把它当作 Authorization 头发送导致编码错误
if not HF_TOKEN and os.environ.get("HF_TOKEN"):
    del os.environ["HF_TOKEN"]
# HF 镜像端点（如 https://hf-mirror.com），需在 transformers/huggingface_hub 导入前设置
_hf_endpoint = _clean_env("HF_ENDPOINT")
if _hf_endpoint:
    os.environ.setdefault("HF_ENDPOINT", _hf_endpoint)

def _prefer_local(local_name: str, hub_id: str) -> str:
    """models/ 下已下载同名目录则用本地路径，否则回退 HF 仓库 id。"""
    local = MODELS_DIR / local_name
    return str(local) if local.exists() else hub_id


# ---------------------------------------------------------------------------
# 音频处理参数
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
# 反伪造滑窗（WavLM 检测模型输入定长 5s）
WINDOW_SIZE = 5.0
HOP_SIZE = 1.0
FAKE_THRESHOLD = 0.5
# Deepfake 检测模型（WavLM-large + AASIST，自定义结构，权重为 safetensors）
DEEPFAKE_MODEL_NAME = _clean_env("DEEPFAKE_MODEL_NAME") or _prefer_local(
    "forensics_0.3B_wavlm_oc_softmax_deepfake_classifier",
    "eliya/forensics_0.3B_wavlm_oc_softmax_deepfake_classifier",
)
# Paraformer ASR 模型（FunASR）
PARAFORMER_MODEL = _clean_env("PARAFORMER_MODEL") or _prefer_local(
    "paraformer-zh", "paraformer-zh"
)
# pyannote 说话人分割（community-1 或本地目录）
PYANNOTE_PIPELINE = _clean_env("PYANNOTE_PIPELINE") or _prefer_local(
    "speaker-diarization-community-1", "pyannote/speaker-diarization-community-1"
)

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
