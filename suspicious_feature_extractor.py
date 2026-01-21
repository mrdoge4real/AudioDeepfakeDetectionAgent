import os
import json
import librosa
import numpy as np
import soundfile as sf
import sys
from pathlib import Path
from dotenv import load_dotenv  # 导入dotenv加载.env配置

# ====================== 1. 加载.env配置（核心） ======================
# 加载.env文件（优先从脚本所在目录找，找不到则从项目根目录找）
load_dotenv()

# 从.env读取项目根目录，设置兜底值（动态推导）
BASE_DIR = os.getenv("BASE_DIR") or str(Path(__file__).resolve().parent.parent)

# ====================== 全局配置（跨平台+可配置） ======================
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# 反伪造检测结果根目录（基于.env的BASE_DIR，跨平台兼容）
ANTI_SPOOF_ROOT = Path(BASE_DIR) / "outputs" / "anti_spoof"
ANTI_SPOOF_ROOT.mkdir(parents=True, exist_ok=True)  # 自动创建多级目录
ANTI_SPOOF_ROOT = str(ANTI_SPOOF_ROOT)

# 可疑片段特征输出根目录（基于.env的BASE_DIR，跨平台兼容）
SUSPICIOUS_FEATURE_ROOT = Path(BASE_DIR) / "outputs" / "suspicious_features"
SUSPICIOUS_FEATURE_ROOT.mkdir(parents=True, exist_ok=True)
SUSPICIOUS_FEATURE_ROOT = str(SUSPICIOUS_FEATURE_ROOT)

# 音频基础配置（与anti_spoof_agent保持一致）
SAMPLE_RATE = 16000

# MFCC提取参数（复用mfcc_extect_agent的配置）
MFCC_PARAMS = {
    "n_mfcc": 13,
    "n_fft": 512,
    "hop_length": 160
}

# 梅尔频谱提取参数（复用melspectral_Extract_agent的配置）
MEL_PARAMS = {
    "n_fft": 512,
    "hop_length": 160,
    "n_mels": 80
}

# ====================== 工具函数：提取音频文件名（去后缀，跨平台） ======================
def extract_audio_filename(audio_path):
    """
    跨平台提取音频文件名（去后缀），兼容Linux/Windows路径
    示例：
    - /home/bowen/audio/LA_E_1000147.wav → LA_E_1000147
    - E:/audio/LA_E_1000147.wav → LA_E_1000147
    - ./my_audio.wav → my_audio
    """
    # 用pathlib跨平台解析路径，避免os.path的系统差异
    filename = Path(audio_path).stem
    return filename

# ====================== 工具函数：查找反伪造检测JSON文件（跨平台） ======================
def find_anti_spoof_json(audio_filename=None):
    """
    查找反伪造检测生成的JSON文件（跨平台兼容）
    - 指定audio_filename时：找对应文件
    - 未指定时：找目录下第一个JSON文件（测试用）
    """
    anti_spoof_root_path = Path(ANTI_SPOOF_ROOT)
    if not anti_spoof_root_path.exists():
        return {"success": False, "error": f"反伪造检测目录不存在：{ANTI_SPOOF_ROOT}"}
    
    # 筛选所有_anti_spoof.json文件（跨平台兼容）
    json_files = []
    for f in anti_spoof_root_path.iterdir():
        if f.is_file() and f.name.endswith("_anti_spoof.json"):
            json_files.append(f.name)
    
    if not json_files:
        return {"success": False, "error": f"反伪造检测目录下无有效JSON文件：{ANTI_SPOOF_ROOT}"}
    
    # 指定文件名时精准匹配
    if audio_filename:
        target_file = f"{audio_filename}_anti_spoof.json"
        if target_file in json_files:
            json_path = str(anti_spoof_root_path / target_file)
            return {
                "success": True,
                "json_path": json_path
            }
        else:
            return {
                "success": False,
                "error": f"未找到{audio_filename}对应的反伪造检测JSON文件"
            }
    # 未指定时返回第一个文件（测试场景）
    else:
        json_path = str(anti_spoof_root_path / json_files[0])
        return {
            "success": True,
            "json_path": json_path
        }

# ====================== 1. 读取反伪造检测JSON文件（跨平台） ======================
def load_anti_spoof_json(audio_filename=None):
    """
    读取anti_spoof_agent生成的JSON文件（按语音文件名匹配）
    :param audio_filename: 语音文件名（去后缀），如LA_E_1000147
    :return: 解析结果字典（含音频路径、可疑片段列表、语音文件名）
    """
    # 第一步：找到对应的JSON文件
    json_find_result = find_anti_spoof_json(audio_filename)
    if not json_find_result["success"]:
        return json_find_result
    
    json_path = json_find_result["json_path"]

    # 第二步：读取并解析JSON
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            anti_spoof_data = json.load(f)

        # 校验JSON核心字段
        if not anti_spoof_data.get("success"):
            return {
                "success": False,
                "error": "反伪造检测执行失败，无有效数据"
            }

        # 提取核心信息（优先用JSON内的audio_filename，无则解析路径）
        audio_path = anti_spoof_data.get("audio_path")
        suspicious_segments = anti_spoof_data.get("data", {}).get("suspicious_segments", [])
        # 核心：用语音文件名作为标识（不再解析数字ID）
        audio_filename = anti_spoof_data.get("audio_filename") or extract_audio_filename(audio_path)

        return {
            "success": True,
            "audio_filename": audio_filename,  # 语音文件名（核心标识）
            "audio_path": audio_path,
            "suspicious_segments": suspicious_segments
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"解析JSON失败：{str(e)}"
        }

# ====================== 2. 提取单个可疑片段的MFCC特征（跨平台） ======================
def extract_mfcc_for_segment(segment_audio, audio_filename, segment_id):
    """
    对单个可疑音频片段提取MFCC特征（按语音文件名分层保存，跨平台）
    :param segment_audio: 可疑片段的音频数据（numpy数组）
    :param audio_filename: 语音文件名（核心标识）
    :param segment_id: 片段编号
    :return: MFCC特征结果字典
    """
    # 新目录结构：suspicious_features/语音文件名/mfcc/mfcc_segment_0/（跨平台拼接）
    output_dir = Path(SUSPICIOUS_FEATURE_ROOT) / audio_filename / "mfcc" / f"mfcc_segment_{segment_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_str = str(output_dir)

    try:
        # 提取MFCC并标准化（CMVN）
        mfcc = librosa.feature.mfcc(
            y=segment_audio,
            sr=SAMPLE_RATE,
            n_mfcc=MFCC_PARAMS["n_mfcc"],
            n_fft=MFCC_PARAMS["n_fft"],
            hop_length=MFCC_PARAMS["hop_length"],
            win_length=MFCC_PARAMS["n_fft"],
            window="hann"
        )
        # CMVN标准化
        mean = np.mean(mfcc, axis=1, keepdims=True)
        std = np.std(mfcc, axis=1, keepdims=True)
        mfcc_norm = (mfcc - mean) / (std + 1e-8)

        # 构造结果
        result = {
            "success": True,
            "audio_filename": audio_filename,
            "segment_id": segment_id,
            "mfcc_shape": list(mfcc_norm.shape),
            "mfcc_stats": {
                "mean": float(np.mean(mfcc_norm)),
                "std": float(np.std(mfcc_norm))
            },
            "mfcc_data": mfcc_norm.tolist(),
            "save_path": output_dir_str
        }

        # 保存MFCC结果到JSON（跨平台路径）
        json_path = str(output_dir / "mfcc_feature.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"✅ 语音{audio_filename} 片段{segment_id} MFCC特征已保存：{json_path}")
        return result
    except Exception as e:
        error_result = {
            "success": False,
            "audio_filename": audio_filename,
            "segment_id": segment_id,
            "error": f"提取MFCC失败：{str(e)}"
        }
        print(f"❌ 语音{audio_filename} 片段{segment_id} MFCC提取失败：{str(e)}")
        return error_result

# ====================== 3. 提取单个可疑片段的梅尔频谱特征（跨平台） ======================
def extract_mel_for_segment(segment_audio, audio_filename, segment_id):
    """
    对单个可疑音频片段提取梅尔频谱特征（按语音文件名分层保存，跨平台）
    :param segment_audio: 可疑片段的音频数据（numpy数组）
    :param audio_filename: 语音文件名（核心标识）
    :param segment_id: 片段编号
    :return: 梅尔频谱特征结果字典
    """
    # 新目录结构：suspicious_features/语音文件名/mel/mel_segment_0/（跨平台拼接）
    output_dir = Path(SUSPICIOUS_FEATURE_ROOT) / audio_filename / "mel" / f"mel_segment_{segment_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_str = str(output_dir)

    try:
        # 提取梅尔频谱并转对数刻度
        mel = librosa.feature.melspectrogram(
            y=segment_audio,
            sr=SAMPLE_RATE,
            n_fft=MEL_PARAMS["n_fft"],
            hop_length=MEL_PARAMS["hop_length"],
            n_mels=MEL_PARAMS["n_mels"],
            power=2.0
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)

        # 保存梅尔频谱可视化图（跨平台路径）
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(log_mel, sr=SAMPLE_RATE, hop_length=MEL_PARAMS["hop_length"], x_axis="time", y_axis="mel")
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"Audio {audio_filename} - Suspicious Segment {segment_id} Log-Mel Spectrogram")
        plt.tight_layout()
        png_path = str(output_dir / "mel_spectrogram.png")
        plt.savefig(png_path, dpi=200)
        plt.close()

        # 构造结果
        result = {
            "success": True,
            "audio_filename": audio_filename,
            "segment_id": segment_id,
            "mel_shape": list(log_mel.shape),
            "mel_energy_stats": {
                "mean": float(np.mean(log_mel)),
                "std": float(np.std(log_mel))
            },
            "log_mel_data": log_mel.tolist(),
            "mel_png_path": png_path,
            "save_path": output_dir_str
        }

        # 保存梅尔频谱结果到JSON（跨平台路径）
        json_path = str(output_dir / "mel_feature.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"✅ 语音{audio_filename} 片段{segment_id} 梅尔频谱特征已保存：{json_path}")
        return result
    except Exception as e:
        error_result = {
            "success": False,
            "audio_filename": audio_filename,
            "segment_id": segment_id,
            "error": f"提取梅尔频谱失败：{str(e)}"
        }
        print(f"❌ 语音{audio_filename} 片段{segment_id} 梅尔频谱提取失败：{str(e)}")
        return error_result

# ====================== 4. 核心：遍历可疑片段提取特征（跨平台） ======================
def extract_suspicious_segments_features(audio_filename=None):
    """
    主函数：读取JSON→解析可疑片段→逐个提取特征（按语音文件名保存，跨平台）
    :param audio_filename: 可选，指定要处理的语音文件名（如LA_E_1000147）
    """
    # 步骤1：读取反伪造检测结果（按语音文件名匹配）
    anti_spoof_result = load_anti_spoof_json(audio_filename)
    if not anti_spoof_result["success"]:
        print(f"❌ 读取反伪造检测结果失败：{anti_spoof_result['error']}")
        return anti_spoof_result

    audio_path = anti_spoof_result["audio_path"]
    audio_filename = anti_spoof_result["audio_filename"]  # 核心标识：语音文件名
    suspicious_segments = anti_spoof_result["suspicious_segments"]

    # 步骤2：校验音频文件和可疑片段（跨平台）
    audio_path_obj = Path(audio_path)
    if not audio_path_obj.exists():
        error_msg = f"音频文件不存在：{audio_path}"
        print(f"❌ {error_msg}")
        return {"success": False, "error": error_msg}

    if len(suspicious_segments) == 0:
        print(f"ℹ️ 语音{audio_filename} 无可疑片段，无需提取特征")
        return {
            "success": True,
            "audio_filename": audio_filename,
            "message": "无可疑片段，特征提取跳过",
            "suspicious_segments_count": 0
        }

    # 步骤3：加载完整音频（仅加载一次）
    try:
        audio, sr = librosa.load(str(audio_path_obj), sr=SAMPLE_RATE, mono=True)
        if sr != SAMPLE_RATE:
            raise RuntimeError(f"音频采样率错误，要求{SAMPLE_RATE}Hz，实际{sr}Hz")
    except Exception as e:
        error_msg = f"加载音频失败：{str(e)}"
        print(f"❌ {error_msg}")
        return {"success": False, "error": error_msg}

    # 步骤4：遍历每个可疑片段，提取特征
    all_segments_features = []
    for idx, segment in enumerate(suspicious_segments):
        start_time = segment["start"]
        end_time = segment["end"]
        print(f"\n🔍 处理语音{audio_filename} 可疑片段 {idx}：{start_time}s → {end_time}s")

        # 时间戳转采样点索引（防止越界）
        start_idx = int(start_time * SAMPLE_RATE)
        end_idx = int(end_time * SAMPLE_RATE)
        start_idx = max(0, start_idx)
        end_idx = min(len(audio), end_idx)

        # 切片获取可疑片段音频数据
        segment_audio = audio[start_idx:end_idx]
        if len(segment_audio) == 0:
            print(f"⚠️ 语音{audio_filename} 片段{idx} 无有效音频数据，跳过")
            continue

        # 提取MFCC特征
        mfcc_result = extract_mfcc_for_segment(segment_audio, audio_filename, idx)
        # 提取梅尔频谱特征
        mel_result = extract_mel_for_segment(segment_audio, audio_filename, idx)

        # 整合该片段的所有特征结果
        segment_feature = {
            "audio_filename": audio_filename,
            "segment_id": idx,
            "time_range": {"start": start_time, "end": end_time},
            "mfcc_feature": mfcc_result,
            "mel_feature": mel_result
        }
        all_segments_features.append(segment_feature)

    # 步骤5：保存汇总结果（按语音文件名保存，跨平台）
    summary_dir = Path(SUSPICIOUS_FEATURE_ROOT) / audio_filename
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_result = {
        "agent": "Suspicious_Feature_Agent",
        "success": True,
        "audio_filename": audio_filename,  # 核心标识：语音文件名
        "audio_path": str(audio_path_obj),
        "total_suspicious_segments": len(suspicious_segments),
        "extracted_segments_count": len(all_segments_features),
        "segments_features": all_segments_features
    }

    # 保存汇总JSON（跨平台路径）
    summary_json_path = str(summary_dir / "suspicious_features_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_result, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 语音{audio_filename} 所有可疑片段特征提取完成，汇总结果保存：{summary_json_path}")
    return summary_result

# ====================== 5. 测试入口（跨平台通用） ======================
if __name__ == "__main__":
    print(f"===== 可疑片段特征提取Agent 开始运行（对齐语音文件名版） =====")
    print(f"📌 项目根目录（来自.env）: {BASE_DIR}")
    print(f"📌 反伪造检测结果目录: {ANTI_SPOOF_ROOT}")
    print(f"📌 特征输出目录: {SUSPICIOUS_FEATURE_ROOT}")
    
    # 可指定语音文件名（如LA_E_1000147），不指定则处理目录下第一个JSON
    result = extract_suspicious_segments_features(audio_filename="LA_E_1000147")
    
    # 打印结果（可选）
    print("\n===== 特征提取结果汇总 =====")
    print(json.dumps(result, ensure_ascii=False, indent=2))