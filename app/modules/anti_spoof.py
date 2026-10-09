# -*- coding: utf-8 -*-
"""步骤2：反伪造初检 —— Deepfake 检测模型滑窗扫描，定位可疑片段。"""
import json
from pathlib import Path

import librosa
import torch

from app.config import (
    ANTI_SPOOF_DIR, FAKE_THRESHOLD, HOP_SIZE, SAMPLE_RATE, WINDOW_SIZE,
)
from app.core.model_registry import DEVICE, get_deepfake_model


def _prepare_input(audio_segment, target_len: int):
    """对齐官方 inference.py 的预处理：峰值归一化 + 中央裁剪/平铺填充到定长。"""
    import numpy as np
    x = np.asarray(audio_segment, dtype=np.float32)
    peak = np.abs(x).max()
    if peak > 0:
        x = x / peak
    if len(x) >= target_len:
        start = (len(x) - target_len) // 2
        return x[start:start + target_len]
    repeats = int(np.ceil(target_len / len(x)))
    return np.tile(x, repeats)[:target_len]


@torch.no_grad()
def _infer_fake_prob(audio_segment, model) -> float:
    """模型输出 logit，sigmoid(logit)=P(real)（训练标签 1=real, 0=fake）。"""
    x = _prepare_input(audio_segment, int(WINDOW_SIZE * SAMPLE_RATE))
    waveform = torch.from_numpy(x).unsqueeze(0).to(DEVICE)
    logit = model(waveform)
    real_prob = torch.sigmoid(logit).item()
    return 1.0 - real_prob


def _sliding_window_detection(audio_path: Path):
    audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    duration = len(audio) / sr

    model = get_deepfake_model()

    window_len = int(WINDOW_SIZE * sr)
    hop_len = int(HOP_SIZE * sr)

    fake_scores, time_stamps = [], []
    # 音频不足一个窗口时，用整段（模型内部会 pad 到定长）
    starts = list(range(0, max(len(audio) - window_len + 1, 1), hop_len))
    for start in starts:
        segment = audio[start:start + window_len]
        fake_scores.append(round(_infer_fake_prob(segment, model), 4))
        time_stamps.append(round(start / sr, 3))

    return fake_scores, time_stamps, duration


def _extract_suspicious_segments(fake_scores, time_stamps, threshold):
    segments, start_time, end_time = [], None, None

    for score, t in zip(fake_scores, time_stamps):
        if score >= threshold:
            if start_time is None:
                start_time = t
            end_time = t + WINDOW_SIZE
        elif start_time is not None:
            segments.append({"start": round(start_time, 3), "end": round(end_time, 3)})
            start_time = None

    if start_time is not None:
        segments.append({
            "start": round(start_time, 3),
            "end": round(time_stamps[-1] + WINDOW_SIZE, 3),
        })
    return segments


def run_anti_spoof_detection(audio_path: str) -> dict:
    audio_path = Path(audio_path).resolve()
    if not audio_path.exists():
        return {
            "agent": "Anti_Spoofing_Agent",
            "success": False,
            "error": f"音频文件不存在：{audio_path}",
            "data": {"suspicious_segments": []},
        }

    audio_filename = audio_path.stem
    fake_scores, time_stamps, duration = _sliding_window_detection(audio_path)
    suspicious_segments = _extract_suspicious_segments(fake_scores, time_stamps, FAKE_THRESHOLD)

    result = {
        "agent": "Anti_Spoofing_Agent",
        "success": True,
        "audio_filename": audio_filename,
        "audio_path": str(audio_path),
        "audio_duration": round(duration, 2),
        "sample_rate": SAMPLE_RATE,
        "window_size": WINDOW_SIZE,
        "hop_size": HOP_SIZE,
        "threshold": FAKE_THRESHOLD,
        "data": {
            "fake_scores": fake_scores,
            "time_stamps": time_stamps,
            "suspicious_segments": suspicious_segments,
            "num_suspicious_segments": len(suspicious_segments),
        },
    }

    json_path = ANTI_SPOOF_DIR / f"{audio_filename}_anti_spoof.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"✅ Anti-Spoof 检测结果已保存: {json_path}")

    return result
