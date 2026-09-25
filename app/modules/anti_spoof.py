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


@torch.no_grad()
def _infer_fake_prob(audio_segment, feature_extractor, model) -> float:
    inputs = feature_extractor(
        audio_segment, sampling_rate=SAMPLE_RATE,
        return_tensors="pt", padding=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    return probs[0, 1].item()


def _sliding_window_detection(audio_path: Path):
    audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    duration = len(audio) / sr

    feature_extractor, model = get_deepfake_model()

    window_len = int(WINDOW_SIZE * sr)
    hop_len = int(HOP_SIZE * sr)

    fake_scores, time_stamps = [], []
    for start in range(0, len(audio) - window_len + 1, hop_len):
        segment = audio[start:start + window_len]
        fake_scores.append(round(_infer_fake_prob(segment, feature_extractor, model), 4))
        time_stamps.append(round(start / sr, 3))

    return fake_scores, time_stamps, duration


def _extract_suspicious_segments(fake_scores, time_stamps, threshold):
    segments, start_time, end_time = [], None, None

    for score, t in zip(fake_scores, time_stamps):
        if score >= threshold:
            if start_time is None:
                start_time = t
            end_time = t + HOP_SIZE
        elif start_time is not None:
            segments.append({"start": round(start_time, 3), "end": round(end_time, 3)})
            start_time = None

    if start_time is not None:
        segments.append({
            "start": round(start_time, 3),
            "end": round(time_stamps[-1] + HOP_SIZE, 3),
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
