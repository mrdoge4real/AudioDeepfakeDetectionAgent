import os
import json
import numpy as np
import librosa
import torch
from pathlib import Path
from dotenv import load_dotenv  # 导入dotenv加载.env文件
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


# ====================== 1. 加载.env配置（核心） ======================
# 加载.env文件（优先从脚本所在目录找，找不到则从项目根目录找）
load_dotenv()  # 自动读取当前目录/.env文件

# 从.env读取配置，同时设置兜底值（避免配置缺失）
# 项目根目录：优先用.env的BASE_DIR，否则动态推导
BASE_DIR = os.getenv("BASE_DIR") or str(Path(__file__).resolve().parent.parent)
# 本地模型路径（可选，这里暂时用不到，仅演示如何加载）
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "")

# ====================== 2. 全局配置（基于.env的BASE_DIR） ======================
# 跨平台输出目录：统一放在.env指定的BASE_DIR/outputs/anti_spoof
OUTPUT_ROOT = Path(BASE_DIR) / "outputs" / "anti_spoof"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)  # 自动创建多级目录

# 音频参数（保持不变）
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SIZE = 0.5   # 秒
HOP_SIZE = 0.1      # 秒
FAKE_THRESHOLD = 0.7


# ====================== 工具函数：提取音频文件名（去后缀） ======================
def extract_audio_filename(audio_path):
    """
    跨平台提取音频文件名（去后缀），兼容Linux/Windows路径
    示例1：/home/bowen/audio/LA_E_1000147.wav → LA_E_1000147
    示例2：E:/audio/LA_E_1000147.wav → LA_E_1000147
    """
    filename = Path(audio_path).stem
    return filename


# ====================== 1. 加载 HuggingFace 模型 ======================

def load_deepfake_model():
    # 可选：如果模型路径也想从.env加载，可这样写
    # model_name = os.getenv("ANTI_SPOOF_MODEL_NAME", "MelodyMachine/Deepfake-audio-detection-V2")
    model_name = "MelodyMachine/Deepfake-audio-detection-V2"
    
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_name,
        # 可选：从.env加载HF_TOKEN（私有模型需要）
        # use_auth_token=os.getenv("HF_TOKEN", "")
    )

    model = AutoModelForAudioClassification.from_pretrained(
        model_name,
        # use_auth_token=os.getenv("HF_TOKEN", "")
    )

    model.to(DEVICE)
    model.eval()
    return feature_extractor, model


# ====================== 2. 单窗口推理 ======================
@torch.no_grad()
def infer_fake_prob(audio_segment, feature_extractor, model):
    inputs = feature_extractor(
        audio_segment,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=True
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)
    return probs[0, 1].item()  # index 1 = fake


# ====================== 3. 滑窗检测 ======================
def sliding_window_detection(audio_path):
    audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    duration = len(audio) / sr

    feature_extractor, model = load_deepfake_model()

    window_len = int(WINDOW_SIZE * sr)
    hop_len = int(HOP_SIZE * sr)

    fake_scores = []
    time_stamps = []

    for start in range(0, len(audio) - window_len + 1, hop_len):
        end = start + window_len
        segment = audio[start:end]

        fake_prob = infer_fake_prob(segment, feature_extractor, model)

        fake_scores.append(round(fake_prob, 4))
        time_stamps.append(round(start / sr, 3))

    return fake_scores, time_stamps, duration


# ====================== 4. 聚合可疑片段 ======================
def extract_suspicious_segments(fake_scores, time_stamps, threshold):
    segments = []
    start_time = None

    for score, t in zip(fake_scores, time_stamps):
        if score >= threshold:
            if start_time is None:
                start_time = t
            end_time = t + HOP_SIZE
        else:
            if start_time is not None:
                segments.append({
                    "start": round(start_time, 3),
                    "end": round(end_time, 3)
                })
                start_time = None

    if start_time is not None:
        segments.append({
            "start": round(start_time, 3),
            "end": round(time_stamps[-1] + HOP_SIZE, 3)
        })

    return segments


# ====================== 5. Agent 主接口（基于.env路径保存） ======================
def run_anti_spoof_detection(audio_path):
    # 1. 跨平台标准化音频路径
    audio_path = Path(audio_path).resolve()
    if not audio_path.exists():
        return {
            "agent": "Anti_Spoofing_Agent",
            "success": False,
            "error": f"音频文件不存在：{str(audio_path)}",
            "data": {"suspicious_segments": []}
        }

    # 2. 提取音频文件名
    audio_filename = extract_audio_filename(audio_path)

    # 3. 执行检测
    fake_scores, time_stamps, duration = sliding_window_detection(audio_path)
    suspicious_segments = extract_suspicious_segments(
        fake_scores,
        time_stamps,
        FAKE_THRESHOLD
    )

    # 4. 构造结果
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
            "num_suspicious_segments": len(suspicious_segments)
        }
    }

    # 5. 保存JSON（路径基于.env的BASE_DIR）
    json_filename = f"{audio_filename}_anti_spoof.json"
    json_path = OUTPUT_ROOT / json_filename
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ Anti-Spoof 检测结果已保存: {str(json_path)}")

    return result


# ====================== 6. 测试入口（基于.env配置） ======================
if __name__ == "__main__":
    # 测试路径：基于.env的BASE_DIR拼接，不再硬编码
    test_audio = Path(BASE_DIR) / "audio_files" / "standard_audio" / "LA_E_1000147.wav"

    print("===== Anti-Spoof 检测开始 =====")
    print(f"📌 项目根目录（来自.env）: {BASE_DIR}")
    print(f"📌 测试音频路径: {test_audio}")
    
    output = run_anti_spoof_detection(test_audio)
    print(json.dumps(output, ensure_ascii=False, indent=2))