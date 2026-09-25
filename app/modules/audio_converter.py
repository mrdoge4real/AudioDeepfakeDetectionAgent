# -*- coding: utf-8 -*-
"""步骤1：音频格式标准化 —— 任意格式 → 16kHz 单声道 WAV（ffmpeg）。"""
import subprocess
from pathlib import Path

import librosa

from app.config import SAMPLE_RATE, STANDARD_AUDIO_DIR


def convert_audio_to_standard(input_audio_path: str) -> dict:
    """返回 dict：success / audio_filename / audio_path / sr / duration / error"""
    input_path = Path(input_audio_path).resolve()
    if not input_path.exists():
        return {
            "success": False,
            "error": f"输入文件不存在：{input_path}",
            "audio_filename": None,
            "audio_path": None,
            "sr": SAMPLE_RATE,
            "duration": None,
        }

    audio_filename = input_path.stem
    output_path = STANDARD_AUDIO_DIR / f"{audio_filename}.wav"

    try:
        cmd = [
            "ffmpeg", "-i", str(input_path),
            "-ar", str(SAMPLE_RATE), "-ac", "1",
            "-f", "wav", "-y", str(output_path),
        ]
        proc = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            shell=False, encoding="utf-8",
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg 执行失败：{proc.stderr[:500]}")
        if not output_path.exists():
            raise RuntimeError("转换后的 WAV 文件未生成")

        audio, sr = librosa.load(str(output_path), sr=SAMPLE_RATE)
        duration = librosa.get_duration(y=audio, sr=sr)

        return {
            "success": True,
            "error": None,
            "audio_filename": audio_filename,
            "audio_path": str(output_path),
            "sr": sr,
            "duration": round(duration, 2),
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"转换失败：{e}",
            "audio_filename": audio_filename,
            "audio_path": None,
            "sr": SAMPLE_RATE,
            "duration": None,
        }
