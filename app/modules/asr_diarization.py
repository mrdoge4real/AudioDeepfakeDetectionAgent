# -*- coding: utf-8 -*-
"""步骤3：ASR 语音识别（Whisper）+ 说话人分割（pyannote），按词中点时间对齐。"""
import json
import traceback
from pathlib import Path

import librosa

from app.config import ASR_OUTPUT_DIR, SAMPLE_RATE
from app.core.model_registry import get_diarization_pipeline, get_whisper_model


def _save_result(result: dict, audio_filename: str):
    json_path = ASR_OUTPUT_DIR / f"{audio_filename}_asr_diarization.json"
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"📁 ASR 结果已保存：{json_path}")
    except Exception as e:
        print(f"❌ 保存 ASR JSON 失败：{e}")


def extract_asr_with_speaker_diarization(audio_path: str, save_json: bool = True) -> dict:
    audio_path = Path(audio_path).resolve()
    audio_filename = audio_path.stem

    if not audio_path.exists():
        result = {
            "success": False,
            "audio_filename": audio_filename,
            "error": f"音频文件不存在：{audio_path}",
            "segments": None,
        }
        if save_json:
            _save_result(result, audio_filename)
        return result

    try:
        audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
        duration = librosa.get_duration(y=audio, sr=sr)

        whisper_model = get_whisper_model()
        asr_result = whisper_model.transcribe(
            str(audio_path),
            language="en",
            task="transcribe",
            word_timestamps=True,
            verbose=False,
        )

        words = []
        for seg in asr_result.get("segments", []):
            for w in seg.get("words", []):
                words.append({
                    "word": w["word"].strip(),
                    "start": float(w["start"]),
                    "end": float(w["end"]),
                })

        diarization = get_diarization_pipeline()(audio_path)

        speaker_segments = [
            {"speaker_id": speaker, "start": float(turn.start), "end": float(turn.end)}
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]

        aligned_words = []
        for w in words:
            mid_time = (w["start"] + w["end"]) / 2.0
            speaker_id = "UNKNOWN"
            for seg in speaker_segments:
                if seg["start"] <= mid_time <= seg["end"]:
                    speaker_id = seg["speaker_id"]
                    break
            aligned_words.append({
                "speaker_id": speaker_id,
                "word": w["word"],
                "start": round(w["start"], 3),
                "end": round(w["end"], 3),
            })

        result = {
            "success": True,
            "audio_filename": audio_filename,
            "error": None,
            "language": "en",
            "full_text": asr_result.get("text", "").strip(),
            "segments": aligned_words,
            "total_words": len(aligned_words),
            "total_speakers": len({w["speaker_id"] for w in aligned_words}),
            "audio_path": str(audio_path),
            "audio_duration": round(duration, 2),
        }
        if save_json:
            _save_result(result, audio_filename)
        return result

    except Exception as e:
        result = {
            "success": False,
            "audio_filename": audio_filename,
            "error": f"ASR + 说话人分离失败：{e}\n详细错误：{traceback.format_exc()}",
            "segments": None,
            "audio_path": str(audio_path),
        }
        if save_json:
            _save_result(result, audio_filename)
        return result
