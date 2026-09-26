# -*- coding: utf-8 -*-
"""步骤3：ASR 语音识别（FunASR Paraformer）+ 说话人分割（pyannote），按词中点时间对齐。

Paraformer 输出 token 级时间戳（中文每字一个 token，英文按字母切分），
这里把时间戳合并成词级：连续英文字母合成一个词，每个中文字独立成词。
"""
import json
import traceback
from pathlib import Path

import librosa

from app.config import ASR_OUTPUT_DIR, SAMPLE_RATE
from app.core.model_registry import get_asr_model, get_diarization_pipeline


def _save_result(result: dict, audio_filename: str):
    json_path = ASR_OUTPUT_DIR / f"{audio_filename}_asr_diarization.json"
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"📁 ASR 结果已保存：{json_path}")
    except Exception as e:
        print(f"❌ 保存 ASR JSON 失败：{e}")


def _tokens_to_words(text: str, timestamp_ms: list) -> list:
    """把 token 级时间戳合并成词级 [{word, start, end}]（秒）。

    text 中空格仅用于分隔英文字母 token，不参与时间戳对齐。
    时间戳缺失/对不上时返回空列表，由调用方兜底。
    """
    chars = [c for c in text if c != " "]
    if not chars or not timestamp_ms or len(chars) != len(timestamp_ms):
        return []

    def is_latin(c: str) -> bool:
        return c.isascii() and (c.isalnum() or c in "'-")

    words = []
    cur_chars, cur_start, cur_end = [], None, None

    def flush():
        nonlocal cur_chars, cur_start, cur_end
        if cur_chars:
            words.append({
                "word": "".join(cur_chars),
                "start": cur_start / 1000.0,
                "end": cur_end / 1000.0,
            })
            cur_chars, cur_start, cur_end = [], None, None

    for ch, (s_ms, e_ms) in zip(chars, timestamp_ms):
        if is_latin(ch):
            if not cur_chars:
                cur_start = s_ms
            cur_chars.append(ch)
            cur_end = e_ms
        else:
            flush()
            words.append({"word": ch, "start": s_ms / 1000.0, "end": e_ms / 1000.0})
    flush()
    return words


def _run_asr(audio_path: Path, duration: float) -> tuple:
    """返回 (full_text, words)。words 为空时退化为整段一个词条。"""
    model = get_asr_model()
    res = model.generate(input=str(audio_path), batch_size_s=300)
    item = res[0] if res else {}
    full_text = (item.get("text") or "").strip()
    timestamp = item.get("timestamp") or []

    words = _tokens_to_words(full_text, timestamp)
    if not words and full_text:
        # 无词级时间戳时整段兜底
        words = [{"word": full_text, "start": 0.0, "end": float(duration)}]
    return full_text, words


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

        full_text, words = _run_asr(audio_path, duration)

        # 传 waveform 而不是文件路径：绕过 torchcodec 对 FFmpeg 共享库的依赖
        import torch
        waveform = torch.from_numpy(audio).unsqueeze(0).float()
        diarization = get_diarization_pipeline()(
            {"waveform": waveform, "sample_rate": SAMPLE_RATE}
        )
        # pyannote 4.x 返回 DiarizeOutput（标注在 .speaker_diarization），3.x 直接返回 Annotation
        annotation = getattr(diarization, "speaker_diarization", diarization)
        speaker_segments = [
            {"speaker_id": speaker, "start": float(turn.start), "end": float(turn.end)}
            for turn, _, speaker in annotation.itertracks(yield_label=True)
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
            "asr_engine": "paraformer",
            "full_text": full_text,
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
