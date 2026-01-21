import os
import json
import librosa
import numpy as np
import soundfile as sf
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.getenv("BASE_DIR") or str(Path(__file__).resolve().parent.parent)

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

ANTI_SPOOF_ROOT = Path(BASE_DIR) / "outputs" / "anti_spoof"
ANTI_SPOOF_ROOT.mkdir(parents=True, exist_ok=True)
ANTI_SPOOF_ROOT = str(ANTI_SPOOF_ROOT)

SUSPICIOUS_FEATURE_ROOT = Path(BASE_DIR) / "outputs" / "suspicious_features"
SUSPICIOUS_FEATURE_ROOT.mkdir(parents=True, exist_ok=True)
SUSPICIOUS_FEATURE_ROOT = str(SUSPICIOUS_FEATURE_ROOT)

SAMPLE_RATE = 16000

MFCC_PARAMS = {
    "n_mfcc": 13,
    "n_fft": 512,
    "hop_length": 160
}

MEL_PARAMS = {
    "n_fft": 512,
    "hop_length": 160,
    "n_mels": 80
}

def extract_audio_filename(audio_path):
    filename = Path(audio_path).stem
    return filename

def find_anti_spoof_json(audio_filename=None):
    anti_spoof_root_path = Path(ANTI_SPOOF_ROOT)
    if not anti_spoof_root_path.exists():
        return {"success": False, "error": f"反伪造检测目录不存在：{ANTI_SPOOF_ROOT}"}
    
    json_files = []
    for f in anti_spoof_root_path.iterdir():
        if f.is_file() and f.name.endswith("_anti_spoof.json"):
            json_files.append(f.name)
    
    if not json_files:
        return {"success": False, "error": f"反伪造检测目录下无有效JSON文件：{ANTI_SPOOF_ROOT}"}
    
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
    else:
        json_path = str(anti_spoof_root_path / json_files[0])
        return {
            "success": True,
            "json_path": json_path
        }

def load_anti_spoof_json(audio_filename=None):
    json_find_result = find_anti_spoof_json(audio_filename)
    if not json_find_result["success"]:
        return json_find_result
    
    json_path = json_find_result["json_path"]

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            anti_spoof_data = json.load(f)

        if not anti_spoof_data.get("success"):
            return {
                "success": False,
                "error": "反伪造检测执行失败，无有效数据"
            }

        audio_path = anti_spoof_data.get("audio_path")
        suspicious_segments = anti_spoof_data.get("data", {}).get("suspicious_segments", [])
        audio_filename = anti_spoof_data.get("audio_filename") or extract_audio_filename(audio_path)

        return {
            "success": True,
            "audio_filename": audio_filename,
            "audio_path": audio_path,
            "suspicious_segments": suspicious_segments
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"解析JSON失败：{str(e)}"
        }

def extract_mfcc_for_segment(segment_audio, audio_filename, segment_id):
    output_dir = Path(SUSPICIOUS_FEATURE_ROOT) / audio_filename / "mfcc" / f"mfcc_segment_{segment_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_str = str(output_dir)

    try:
        mfcc = librosa.feature.mfcc(
            y=segment_audio,
            sr=SAMPLE_RATE,
            n_mfcc=MFCC_PARAMS["n_mfcc"],
            n_fft=MFCC_PARAMS["n_fft"],
            hop_length=MFCC_PARAMS["hop_length"],
            win_length=MFCC_PARAMS["n_fft"],
            window="hann"
        )
        mean = np.mean(mfcc, axis=1, keepdims=True)
        std = np.std(mfcc, axis=1, keepdims=True)
        mfcc_norm = (mfcc - mean) / (std + 1e-8)

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

def extract_mel_for_segment(segment_audio, audio_filename, segment_id):
    output_dir = Path(SUSPICIOUS_FEATURE_ROOT) / audio_filename / "mel" / f"mel_segment_{segment_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_str = str(output_dir)

    try:
        mel = librosa.feature.melspectrogram(
            y=segment_audio,
            sr=SAMPLE_RATE,
            n_fft=MEL_PARAMS["n_fft"],
            hop_length=MEL_PARAMS["hop_length"],
            n_mels=MEL_PARAMS["n_mels"],
            power=2.0
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(log_mel, sr=SAMPLE_RATE, hop_length=MEL_PARAMS["hop_length"], x_axis="time", y_axis="mel")
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"Audio {audio_filename} - Suspicious Segment {segment_id} Log-Mel Spectrogram")
        plt.tight_layout()
        png_path = str(output_dir / "mel_spectrogram.png")
        plt.savefig(png_path, dpi=200)
        plt.close()

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

def extract_suspicious_segments_features(audio_filename=None):
    anti_spoof_result = load_anti_spoof_json(audio_filename)
    if not anti_spoof_result["success"]:
        print(f"❌ 读取反伪造检测结果失败：{anti_spoof_result['error']}")
        return anti_spoof_result

    audio_path = anti_spoof_result["audio_path"]
    audio_filename = anti_spoof_result["audio_filename"]
    suspicious_segments = anti_spoof_result["suspicious_segments"]

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

    try:
        audio, sr = librosa.load(str(audio_path_obj), sr=SAMPLE_RATE, mono=True)
        if sr != SAMPLE_RATE:
            raise RuntimeError(f"音频采样率错误，要求{SAMPLE_RATE}Hz，实际{sr}Hz")
    except Exception as e:
        error_msg = f"加载音频失败：{str(e)}"
        print(f"❌ {error_msg}")
        return {"success": False, "error": error_msg}

    all_segments_features = []
    for idx, segment in enumerate(suspicious_segments):
        start_time = segment["start"]
        end_time = segment["end"]
        print(f"\n🔍 处理语音{audio_filename} 可疑片段 {idx}：{start_time}s → {end_time}s")

        start_idx = int(start_time * SAMPLE_RATE)
        end_idx = int(end_time * SAMPLE_RATE)
        start_idx = max(0, start_idx)
        end_idx = min(len(audio), end_idx)

        segment_audio = audio[start_idx:end_idx]
        if len(segment_audio) == 0:
            print(f"⚠️ 语音{audio_filename} 片段{idx} 无有效音频数据，跳过")
            continue

        mfcc_result = extract_mfcc_for_segment(segment_audio, audio_filename, idx)
        mel_result = extract_mel_for_segment(segment_audio, audio_filename, idx)

        segment_feature = {
            "audio_filename": audio_filename,
            "segment_id": idx,
            "time_range": {"start": start_time, "end": end_time},
            "mfcc_feature": mfcc_result,
            "mel_feature": mel_result
        }
        all_segments_features.append(segment_feature)

    summary_dir = Path(SUSPICIOUS_FEATURE_ROOT) / audio_filename
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_result = {
        "agent": "Suspicious_Feature_Agent",
        "success": True,
        "audio_filename": audio_filename,
        "audio_path": str(audio_path_obj),
        "total_suspicious_segments": len(suspicious_segments),
        "extracted_segments_count": len(all_segments_features),
        "segments_features": all_segments_features
    }

    summary_json_path = str(summary_dir / "suspicious_features_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_result, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 语音{audio_filename} 所有可疑片段特征提取完成，汇总结果保存：{summary_json_path}")
    return summary_result

if __name__ == "__main__":
    print(f"===== 可疑片段特征提取Agent 开始运行（对齐语音文件名版） =====")
    print(f"📌 项目根目录（来自.env）: {BASE_DIR}")
    print(f"📌 反伪造检测结果目录: {ANTI_SPOOF_ROOT}")
    print(f"📌 特征输出目录: {SUSPICIOUS_FEATURE_ROOT}")
    
    result = extract_suspicious_segments_features(audio_filename="LA_E_1000147")
    
    print("\n===== 特征提取结果汇总 =====")
    print(json.dumps(result, ensure_ascii=False, indent=2))
