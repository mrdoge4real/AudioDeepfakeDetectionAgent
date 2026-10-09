# -*- coding: utf-8 -*-
"""模型注册表：所有重型模型全局只加载一次，启动时预热 / 首次使用时懒加载。"""
import threading

import torch

from app.config import (
    DEEPFAKE_MODEL_NAME,
    HF_TOKEN,
    PARAFORMER_MODEL,
    PYANNOTE_PIPELINE,
    SAMPLE_RATE,
)

def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _pick_device()

_lock = threading.Lock()
_deepfake = None   # (feature_extractor, model)
_asr = None        # FunASR AutoModel
_diarization = None


def get_deepfake_model():
    """加载 WavLM+AASIST 伪造检测模型（自定义结构），全局单例。

    模型仓库自带 model.py / 权重，不走 transformers AutoModel：
    输出单个 logit，sigmoid(logit)=P(real)，伪造概率 = 1 - sigmoid(logit)。
    """
    global _deepfake
    if _deepfake is None:
        with _lock:
            if _deepfake is None:
                _deepfake = _load_forensics_detector()
    return _deepfake


def _load_forensics_detector():
    import importlib.util
    import sys
    from pathlib import Path

    repo = Path(DEEPFAKE_MODEL_NAME)
    if not repo.exists():
        # 在线仓库 id：先拉取快照到 HF 缓存
        from huggingface_hub import snapshot_download
        repo = Path(snapshot_download(DEEPFAKE_MODEL_NAME))

    # 动态加载仓库自带的 model.py（DeepfakeDetector 内部会加载 microsoft/wavlm-large）
    sys.path.insert(0, str(repo))
    try:
        spec = importlib.util.spec_from_file_location("forensics_model", repo / "model.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(repo))

    model = mod.DeepfakeDetector()

    # 优先 safetensors 权重，兜底 .pt
    from safetensors.torch import load_file
    weights = sorted(repo.glob("*.safetensors"))
    if weights:
        state = load_file(str(weights[0]))
    else:
        pts = sorted(repo.glob("*.pt"))
        if not pts:
            raise FileNotFoundError(f"模型仓库缺少权重文件：{repo}")
        state = torch.load(str(pts[0]), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)

    model.to(DEVICE)
    model.eval()
    return model


def get_asr_model():
    """FunASR Paraformer ASR 模型，全局单例。"""
    global _asr
    if _asr is None:
        with _lock:
            if _asr is None:
                from funasr import AutoModel
                device = "cuda:0" if DEVICE.type == "cuda" else "cpu"
                _asr = AutoModel(
                    model=PARAFORMER_MODEL, device=device, disable_update=True
                )
    return _asr


def get_diarization_pipeline():
    """pyannote 说话人分割流水线，需要 HF_TOKEN，首次使用时加载。"""
    global _diarization
    if _diarization is None:
        with _lock:
            if _diarization is None:
                from pathlib import Path
                # 本地目录直接加载，无需任何凭据；
                # 在线仓库才需要 HF_TOKEN 或 `hf auth login` 的凭据
                if not Path(PYANNOTE_PIPELINE).exists():
                    from huggingface_hub import get_token
                    if not (HF_TOKEN or get_token()):
                        raise RuntimeError(
                            "说话人分割需要 HuggingFace 凭据：在 .env 配置 HF_TOKEN，"
                            "或先执行 hf auth login，并在 HF 网站接受 pyannote 模型协议；"
                            "也可以用 models/download_models.sh 下载到本地目录"
                        )
                from pyannote.audio import Pipeline
                # token 为 None 时自动使用 `hf auth login` 保存的凭据；
                # PYANNOTE_PIPELINE 为本地目录时无需凭据
                _diarization = Pipeline.from_pretrained(
                    PYANNOTE_PIPELINE, token=HF_TOKEN or None,
                )
    return _diarization


def warmup(include_diarization: bool = False):
    """服务启动时预热模型，失败只告警不阻断（接口调用时再报错）。"""
    for name, loader in [("deepfake", get_deepfake_model), ("paraformer", get_asr_model)]:
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
