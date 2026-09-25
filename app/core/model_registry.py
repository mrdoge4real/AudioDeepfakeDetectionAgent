# -*- coding: utf-8 -*-
"""模型注册表：所有重型模型全局只加载一次，启动时预热 / 首次使用时懒加载。"""
import threading

import torch

from app.config import (
    DEEPFAKE_MODEL_NAME,
    HF_TOKEN,
    PYANNOTE_PIPELINE,
    SAMPLE_RATE,
    WHISPER_MODEL_SIZE,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_lock = threading.Lock()
_deepfake = None   # (feature_extractor, model)
_whisper = None
_diarization = None


def get_deepfake_model():
    """返回 (feature_extractor, model)，全局单例。"""
    global _deepfake
    if _deepfake is None:
        with _lock:
            if _deepfake is None:
                from transformers import (
                    AutoFeatureExtractor,
                    AutoModelForAudioClassification,
                )
                fe = AutoFeatureExtractor.from_pretrained(DEEPFAKE_MODEL_NAME)
                model = AutoModelForAudioClassification.from_pretrained(DEEPFAKE_MODEL_NAME)
                model.to(DEVICE)
                model.eval()
                _deepfake = (fe, model)
    return _deepfake


def get_whisper_model():
    global _whisper
    if _whisper is None:
        with _lock:
            if _whisper is None:
                import whisper
                _whisper = whisper.load_model(WHISPER_MODEL_SIZE, device=str(DEVICE))
    return _whisper


def get_diarization_pipeline():
    """pyannote 说话人分割流水线，需要 HF_TOKEN，首次使用时加载。"""
    global _diarization
    if _diarization is None:
        with _lock:
            if _diarization is None:
                if not HF_TOKEN:
                    raise RuntimeError(
                        "说话人分割需要 HuggingFace Token，请在 .env 中配置 HF_TOKEN"
                    )
                from pyannote.audio import Pipeline
                _diarization = Pipeline.from_pretrained(
                    PYANNOTE_PIPELINE, use_auth_token=HF_TOKEN
                )
    return _diarization


def warmup(include_diarization: bool = False):
    """服务启动时预热模型，失败只告警不阻断（接口调用时再报错）。"""
    for name, loader in [("deepfake", get_deepfake_model), ("whisper", get_whisper_model)]:
        try:
            loader()
            print(f"✅ 模型预热完成: {name} (device={DEVICE})")
        except Exception as e:
            print(f"⚠️ 模型预热失败: {name} -> {e}（首次调用时会重试）")
    if include_diarization:
        try:
            get_diarization_pipeline()
            print("✅ 模型预热完成: pyannote diarization")
        except Exception as e:
            print(f"⚠️ pyannote 预热失败: {e}")
