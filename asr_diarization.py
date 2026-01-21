import os
import json
import librosa
import whisper
import torch
from pathlib import Path
from dotenv import load_dotenv
from pyannote.audio import Pipeline
# 新增：导入ListConfig用于安全列表配置
from omegaconf.listconfig import ListConfig

# ====================== 1. 加载.env配置（核心） ======================
# 加载.env文件（优先从脚本所在目录找，找不到则从项目根目录找）
load_dotenv()

# 从.env读取项目根目录，设置兜底值（动态推导）
BASE_DIR = os.getenv("BASE_DIR") or str(Path(__file__).resolve().parent.parent)
# 从.env读取HF_TOKEN（pyannote模型需要认证）
HF_TOKEN = os.getenv("HF_TOKEN", "")

# ====================== 全局配置（跨平台+对齐分布式架构） ======================
# ASR结果输出根目录（基于.env的BASE_DIR，跨平台兼容）
ASR_OUTPUT_ROOT = Path(BASE_DIR) / "outputs" / "asr"
ASR_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)  # 自动创建多级目录
ASR_OUTPUT_ROOT = str(ASR_OUTPUT_ROOT)  # 转为字符串兼容os模块

# ====================== 关键修复：PyTorch 2.6 兼容性配置 ======================
# 1. 将ListConfig加入PyTorch安全全局列表，允许加载该类型
torch.serialization.add_safe_globals([ListConfig])

# 2. 强制启用torch.load补丁（解决weights_only问题，必开）
def patch_torch_load():
    original_load = torch.load
    def patched_load(*args, **kwargs):
        kwargs["weights_only"] = False  # 强制关闭安全检查
        return original_load(*args, **kwargs)
    torch.load = patched_load

# 执行补丁（必须启用，否则仍会报错）
patch_torch_load()

# ====================== 工具函数：提取音频文件名（去后缀，跨平台） ======================
def extract_audio_filename(audio_path):
    """
    跨平台提取音频文件名（去后缀），兼容Linux/Windows路径
    示例：
    - /home/bowen/audio/standard_audio/LA_E_1000147.wav → LA_E_1000147
    - E:/audio/standard_audio/test.wav → test
    - ./my_audio.wav → my_audio
    """
    filename = Path(audio_path).stem
    return filename

# ====================== 核心函数：ASR + 说话人分割（对齐文件名+跨平台） ======================
def extract_asr_with_speaker_diarization(
    audio_path: str,
    whisper_model_size: str = "base",
    save_json: bool = True
):
    """
    Whisper ASR + pyannote Speaker Diarization
    核心优化：兼容PyTorch 2.6 + pyannote认证 + 跨平台路径
    """
    # 跨平台标准化输入路径
    audio_path = Path(audio_path).resolve()
    audio_filename = extract_audio_filename(audio_path)
    audio_path_str = str(audio_path)

    if not audio_path.exists():
        result = {
            "success": False,
            "audio_filename": audio_filename,
            "error": f"音频文件不存在：{audio_path_str}",
            "segments": None
        }
        if save_json:
            save_asr_result(result, audio_filename)
        return json.dumps(result, ensure_ascii=False, indent=2)

    try:
        # ========= 1. 音频校验 =========
        audio, sr = librosa.load(audio_path_str, sr=16000, mono=True)
        if sr != 16000:
            raise ValueError(f"采样率错误：{sr}Hz（必须为 16kHz）")
        duration = librosa.get_duration(y=audio, sr=sr)

        # ========= 2. Whisper ASR =========
        # 自动选择设备（CUDA可用则用GPU，否则CPU）
        device = "cuda" if torch.cuda.is_available() else "cpu"
        whisper_model = whisper.load_model(whisper_model_size, device=device)
        asr_result = whisper_model.transcribe(
            audio_path_str,
            language="en",
            task="transcribe",
            word_timestamps=True,
            verbose=False
        )

        # 提取单词级时间戳
        words = []
        for seg in asr_result.get("segments", []):
            for w in seg.get("words", []):
                words.append({
                    "word": w["word"].strip(),
                    "start": float(w["start"]),
                    "end": float(w["end"])
                })

        # ========= 3. Speaker Diarization（核心修复：移除device参数） =========
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization",
        )

        diarization = diarization_pipeline(audio_path)

        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                "speaker_id": speaker,
                "start": float(turn.start),
                "end": float(turn.end)
            })


        # ========= 4. 词 ↔ 说话人 对齐 =========
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
                "end": round(w["end"], 3)
            })

        # ========= 5. 构造结果 =========
        result = {
            "success": True,
            "audio_filename": audio_filename,
            "error": None,
            "language": "en",
            "full_text": asr_result.get("text", "").strip(),
            "segments": aligned_words,
            "total_words": len(aligned_words),
            "total_speakers": len(set(w["speaker_id"] for w in aligned_words)),
            "audio_path": audio_path_str,
            "audio_duration": round(duration, 2),
            "device_used": device  # 新增：记录使用的设备
        }

        # ========= 6. 保存结果 =========
        if save_json:
            save_asr_result(result, audio_filename)

        return json.dumps(result, ensure_ascii=False, indent=2)

    except Exception as e:
        # 详细打印错误栈，方便排查
        import traceback
        error_detail = traceback.format_exc()
        result = {
            "success": False,
            "audio_filename": audio_filename,
            "error": f"ASR + 说话人分离失败：{str(e)}\n详细错误：{error_detail}",
            "segments": None,
            "audio_path": audio_path_str
        }
        if save_json:
            save_asr_result(result, audio_filename)
        return json.dumps(result, ensure_ascii=False, indent=2)

# ====================== 工具函数：保存ASR结果 ======================
def save_asr_result(result, audio_filename):
    """按文件名保存ASR结果（跨平台兼容）"""
    json_filename = f"{audio_filename}_asr_diarization.json"
    json_path = str(Path(ASR_OUTPUT_ROOT) / json_filename)
    
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"📁 ASR结果已保存：{json_path}")
    except Exception as e:
        print(f"❌ 保存ASR JSON失败：{str(e)}")

# ====================== 批量处理函数 ======================
def batch_process_standard_audio(audio_dir: str = None):
    """批量处理standard_audio目录下的WAV文件"""
    if audio_dir is None:
        audio_dir = Path(BASE_DIR) / "audio_files" / "standard_audio"
    else:
        audio_dir = Path(audio_dir).resolve()
    
    print("=" * 80)
    print(f"🚀 开始批量处理ASR + 说话人分割")
    print(f"处理目录：{str(audio_dir)}")
    print("=" * 80)
    
    if not audio_dir.exists():
        print(f"❌ 目录不存在：{str(audio_dir)}")
        return
    
    wav_files = [f for f in audio_dir.iterdir() if f.is_file() and f.suffix.lower() == ".wav"]
    if not wav_files:
        print(f"ℹ️ 目录下无WAV文件：{str(audio_dir)}")
        return
    
    total = len(wav_files)
    success_count = 0
    for idx, wav_file in enumerate(wav_files):
        print(f"\n[{idx+1}/{total}] 处理文件：{wav_file.name}")
        result_json = extract_asr_with_speaker_diarization(str(wav_file))
        result = json.loads(result_json)
        if result["success"]:
            success_count += 1
            print(f"✅ 处理成功：{wav_file.name}")
        else:
            print(f"❌ 处理失败：{result['error'][:200]}...")  # 截断长错误信息
    
    print("\n" + "=" * 80)
    print(f"📊 批量处理完成 | 总数：{total} | 成功：{success_count} | 失败：{total - success_count}")
    print(f"📁 结果保存目录：{ASR_OUTPUT_ROOT}")
    print("=" * 80)

# ====================== 测试函数 ======================
def test_asr_diarization():
    """测试单个音频文件"""
    test_audio_path = Path(BASE_DIR) / "audio_files" / "standard_audio" / "LA_E_1000147.wav"
    test_audio_path_str = str(test_audio_path)

    print("=" * 80)
    print(f"🎧 开始 ASR + 说话人分离测试")
    print(f"项目根目录（来自.env）：{BASE_DIR}")
    print(f"音频路径：{test_audio_path_str}")
    print("=" * 80)

    result_json = extract_asr_with_speaker_diarization(
        audio_path=test_audio_path_str,
        whisper_model_size="base"
    )

    print("📄 ASR + Diarization 结果（JSON）：")
    print(result_json)
    print("=" * 80)

    result = json.loads(result_json)
    if not result["success"]:
        print(f"❌ 测试失败：{result['error'][:300]}...")
        return

    print("✅ 测试成功！关键信息如下：")
    print(f"   - 音频文件名：{result['audio_filename']}")
    print(f"   - 语言：{result['language']}")
    print(f"   - 音频时长：{result['audio_duration']} 秒")
    print(f"   - 总词数：{result['total_words']}")
    print(f"   - 说话人数量：{result['total_speakers']}")
    print(f"   - 使用设备：{result['device_used']}")  # 新增：显示使用的设备
    print("-" * 80)

    preview_n = min(20, len(result["segments"]))
    print(f"🧩 前 {preview_n} 个词（含 speaker 对齐）：")
    for i in range(preview_n):
        seg = result["segments"][i]
        print(f"[{seg['start']:>6.2f}s - {seg['end']:>6.2f}s] {seg['speaker_id']}: {seg['word']}")

    json_save_path = Path(ASR_OUTPUT_ROOT) / f"{result['audio_filename']}_asr_diarization.json"
    print("=" * 80)
    print(f"📁 结果已保存至：{str(json_save_path)}")
    print("🎉 ASR + Speaker Diarization 测试完成")

# ====================== 主入口 ======================
if __name__ == "__main__":
    print(f"📌 项目根目录（来自.env）: {BASE_DIR}")
    # 选择1：测试单个文件
    test_asr_diarization()
    # 选择2：批量处理所有标准化音频
    # batch_process_standard_audio()