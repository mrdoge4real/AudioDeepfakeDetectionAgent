# -*- coding: utf-8 -*-
"""步骤4：可疑片段特征提取 —— 归一化 MFCC + log-Mel 频谱（含频谱图 PNG）。"""
import json
from pathlib import Path

import librosa
import numpy as np

from app.config import (
    ANTI_SPOOF_DIR, MFCC_PARAMS, MEL_PARAMS, SAMPLE_RATE, SUSPICIOUS_FEATURE_DIR,
)


def _load_anti_spoof_json(audio_filename: str) -> dict:
    json_path = ANTI_SPOOF_DIR / f"{audio_filename}_anti_spoof.json"
    if not json_path.exists():
        return {"success": False, "error": f"未找到反伪造检测结果：{json_path}"}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data.get("success"):
            return {"success": False, "error": "反伪造检测执行失败，无有效数据"}
        return {
            "success": True,
            "audio_filename": data.get("audio_filename") or audio_filename,
            "audio_path": data.get("audio_path"),
            "suspicious_segments": data.get("data", {}).get("suspicious_segments", []),
        }
    except Exception as e:
        return {"success": False, "error": f"解析反伪造 JSON 失败：{e}"}


def _extract_mfcc(segment_audio, audio_filename, segment_id) -> dict:
    output_dir = SUSPICIOUS_FEATURE_DIR / audio_filename / "mfcc" / f"mfcc_segment_{segment_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        mfcc = librosa.feature.mfcc(
            y=segment_audio, sr=SAMPLE_RATE,
            n_mfcc=MFCC_PARAMS["n_mfcc"], n_fft=MFCC_PARAMS["n_fft"],
            hop_length=MFCC_PARAMS["hop_length"], win_length=MFCC_PARAMS["n_fft"],
            window="hann",
        )
        mean = np.mean(mfcc, axis=1, keepdims=True)
        std = np.std(mfcc, axis=1, keepdims=True)
        mfcc_norm = (mfcc - mean) / (std + 1e-8)

        result = {
            "success": True,
            "segment_id": segment_id,
            "mfcc_shape": list(mfcc_norm.shape),
            "mfcc_stats": {
                "mean": float(np.mean(mfcc_norm)),
                "std": float(np.std(mfcc_norm)),
            },
            "save_path": str(output_dir),
        }
        with open(output_dir / "mfcc_feature.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return result
    except Exception as e:
        return {"success": False, "segment_id": segment_id, "error": f"提取 MFCC 失败：{e}"}


def _extract_mel(segment_audio, audio_filename, segment_id) -> dict:
    output_dir = SUSPICIOUS_FEATURE_DIR / audio_filename / "mel" / f"mel_segment_{segment_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        mel = librosa.feature.melspectrogram(
            y=segment_audio, sr=SAMPLE_RATE,
            n_fft=MEL_PARAMS["n_fft"], hop_length=MEL_PARAMS["hop_length"],
            n_mels=MEL_PARAMS["n_mels"], power=2.0,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(
            log_mel, sr=SAMPLE_RATE, hop_length=MEL_PARAMS["hop_length"],
            x_axis="time", y_axis="mel",
        )
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"Audio {audio_filename} - Suspicious Segment {segment_id} Log-Mel Spectrogram")
        plt.tight_layout()
        png_path = output_dir / "mel_spectrogram.png"
        plt.savefig(str(png_path), dpi=200)
        plt.close()

        result = {
            "success": True,
            "segment_id": segment_id,
            "mel_shape": list(log_mel.shape),
            "mel_energy_stats": {
                "mean": float(np.mean(log_mel)),
                "std": float(np.std(log_mel)),
            },
            "mel_png_path": str(png_path),
            "save_path": str(output_dir),
        }
        with open(output_dir / "mel_feature.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return result
    except Exception as e:
        return {"success": False, "segment_id": segment_id, "error": f"提取梅尔频谱失败：{e}"}


def extract_suspicious_segments_features(audio_filename: str) -> dict:
    """读取步骤2的可疑片段，逐段提取 MFCC / 梅尔特征，输出汇总 JSON。"""
    anti_spoof = _load_anti_spoof_json(audio_filename)
    if not anti_spoof["success"]:
        return anti_spoof

    audio_path = Path(anti_spoof["audio_path"])
    suspicious_segments = anti_spoof["suspicious_segments"]

    if not audio_path.exists():
        return {"success": False, "error": f"音频文件不存在：{audio_path}"}

    summary_dir = SUSPICIOUS_FEATURE_DIR / audio_filename
    summary_dir.mkdir(parents=True, exist_ok=True)

    if not suspicious_segments:
        summary = {
            "agent": "Suspicious_Feature_Agent",
            "success": True,
            "audio_filename": audio_filename,
            "audio_path": str(audio_path),
            "total_suspicious_segments": 0,
            "extracted_segments_count": 0,
            "segments_features": [],
        }
        with open(summary_dir / "suspicious_features_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return summary

    try:
        audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        return {"success": False, "error": f"加载音频失败：{e}"}

    all_features = []
    for idx, segment in enumerate(suspicious_segments):
        start_idx = max(0, int(segment["start"] * SAMPLE_RATE))
        end_idx = min(len(audio), int(segment["end"] * SAMPLE_RATE))
        segment_audio = audio[start_idx:end_idx]
        if len(segment_audio) == 0:
            print(f"⚠️ {audio_filename} 片段{idx} 无有效音频数据，跳过")
            continue

        all_features.append({
            "audio_filename": audio_filename,
            "segment_id": idx,
            "time_range": {"start": segment["start"], "end": segment["end"]},
            "mfcc_feature": _extract_mfcc(segment_audio, audio_filename, idx),
            "mel_feature": _extract_mel(segment_audio, audio_filename, idx),
        })

    summary = {
        "agent": "Suspicious_Feature_Agent",
        "success": True,
        "audio_filename": audio_filename,
        "audio_path": str(audio_path),
        "total_suspicious_segments": len(suspicious_segments),
        "extracted_segments_count": len(all_features),
        "segments_features": all_features,
    }
    with open(summary_dir / "suspicious_features_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"✅ {audio_filename} 可疑片段特征提取完成：{summary_dir / 'suspicious_features_summary.json'}")
    return summary
