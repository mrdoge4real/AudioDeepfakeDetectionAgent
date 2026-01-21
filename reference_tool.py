# -*- coding: utf-8 -*-
import os
import json
import sys
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.getenv("BASE_DIR")

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

SUSPICIOUS_FEATURE_ROOT = os.path.join(BASE_DIR, "outputs", "suspicious_features")
ASR_OUTPUT_ROOT = os.path.join(BASE_DIR, "outputs", "asr")
REFERENCE_OUTPUT_ROOT = os.path.join(BASE_DIR, "outputs", "reference_report")
os.makedirs(REFERENCE_OUTPUT_ROOT, exist_ok=True)

ANOMALY_THRESHOLDS = {
    "mfcc_mean_abs": 0.5,
    "mfcc_std_upper": 35.0141,
    "mfcc_inner_std_upper": 44.6855,
    "mel_energy_upper": -43.5002,
    "mel_energy_lower": -65.9447,
    "high_risk_anomaly_count": 2,
    "medium_risk_anomaly_count": 1
}

def load_asr_diarization_data(audio_filename):
    asr_json_path = os.path.join(ASR_OUTPUT_ROOT, f"{audio_filename}_asr_diarization.json")
    
    if not os.path.exists(asr_json_path):
        print(f"ℹ️ 未找到{audio_filename}的ASR+说话人数据：{asr_json_path}")
        return None
    
    try:
        with open(asr_json_path, "r", encoding="utf-8") as f:
            asr_data = json.load(f)
        
        if not asr_data.get("success"):
            print(f"ℹ️ {audio_filename}的ASR数据无效：{asr_data.get('error')}")
            return None
        
        return asr_data
    except Exception as e:
        print(f"ℹ️ 解析ASR数据失败：{str(e)}")
        return None

def match_suspicious_segment_with_text(suspicious_segment, asr_segments):
    seg_start = suspicious_segment["start"]
    seg_end = suspicious_segment["end"]
    matched_words = []
    
    for word_seg in asr_segments:
        word_start = word_seg["start"]
        word_end = word_seg["end"]
        
        if not (word_end < seg_start or word_start > seg_end):
            matched_words.append({
                "speaker_id": word_seg["speaker_id"],
                "word": word_seg["word"],
                "start": word_seg["start"],
                "end": word_seg["end"]
            })
    
    matched_text = " ".join([w["word"] for w in matched_words])
    return {
        "matched_words": matched_words,
        "matched_text": matched_text,
        "total_matched_words": len(matched_words),
        "speakers_in_segment": list(set([w["speaker_id"] for w in matched_words]))
    }

def load_suspicious_features(audio_filename):
    summary_path = os.path.join(SUSPICIOUS_FEATURE_ROOT, audio_filename, "suspicious_features_summary.json")
    
    if not os.path.exists(summary_path):
        return {
            "success": False,
            "error": f"可疑片段特征汇总文件不存在：{summary_path}"
        }
    
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            feature_data = json.load(f)
        
        if not feature_data.get("success"):
            return {
                "success": False,
                "error": f"特征提取失败，汇总文件标记为失败状态"
            }
        
        return {
            "success": True,
            "data": feature_data
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"解析特征文件失败：{str(e)}"
        }

def analyze_feature_anomaly(segment_feature, asr_data=None):
    analysis_lines = []
    segment_id = segment_feature["segment_id"]
    time_range = segment_feature["time_range"]
    time_str = f"{time_range['start']}s - {time_range['end']}s"

    analysis_lines.append(f"### 片段{segment_id}（时间范围：{time_str}）")

    if asr_data and asr_data.get("segments"):
        match_result = match_suspicious_segment_with_text(time_range, asr_data["segments"])
        analysis_lines.append(f"- **语音内容**：{match_result['matched_text'] or '无匹配内容'}")
        analysis_lines.append(f"- **说话人**：{', '.join(match_result['speakers_in_segment']) or 'UNKNOWN'}")
        analysis_lines.append(f"- **匹配词数**：{match_result['total_matched_words']}")
    else:
        analysis_lines.append(f"- **语音内容**：未获取到ASR数据")
        analysis_lines.append(f"- **说话人**：未获取到说话人数据")
    
    analysis_lines.append("")

    mfcc_feature = segment_feature["mfcc_feature"]
    if mfcc_feature["success"]:
        mfcc_stats = mfcc_feature["mfcc_stats"]
        mfcc_mean_abs = abs(mfcc_stats["mean"])
        mfcc_std = mfcc_stats["std"]
        
        mfcc_analysis = []
        if mfcc_mean_abs > ANOMALY_THRESHOLDS["mfcc_mean_abs"]:
            mfcc_analysis.append(f"MFCC均值绝对值({round(mfcc_mean_abs, 3)})超出正常范围（≤{ANOMALY_THRESHOLDS['mfcc_mean_abs']}）")
        if mfcc_std > ANOMALY_THRESHOLDS["mfcc_std_upper"]:
            mfcc_analysis.append(f"MFCC整体标准差({round(mfcc_std, 3)})超出真人语音基准（≤{ANOMALY_THRESHOLDS['mfcc_std_upper']}），频谱波动异常（合成音频典型特征）")
        
        if mfcc_analysis:
            analysis_lines.append(f"- **MFCC特征异常**：{'; '.join(mfcc_analysis)}；")
        else:
            analysis_lines.append(f"- **MFCC特征**：均值绝对值({round(mfcc_mean_abs, 3)})、整体标准差({round(mfcc_std, 3)})均符合真人语音基准；")
    else:
        analysis_lines.append(f"- **MFCC特征**：提取失败 → {mfcc_feature.get('error', '未知错误')}；")

    mel_feature = segment_feature["mel_feature"]
    if mel_feature["success"]:
        mel_stats = mel_feature["mel_energy_stats"]
        mel_mean = mel_stats["mean"]
        
        mel_analysis = []
        if mel_mean > ANOMALY_THRESHOLDS["mel_energy_upper"]:
            mel_analysis.append(f"梅尔能量均值({round(mel_mean, 1)}dB)偏高（正常≤{ANOMALY_THRESHOLDS['mel_energy_upper']}dB），频域能量分布异常")
        elif mel_mean < ANOMALY_THRESHOLDS["mel_energy_lower"]:
            mel_analysis.append(f"梅尔能量均值({round(mel_mean, 1)}dB)偏低（正常≥{ANOMALY_THRESHOLDS['mel_energy_lower']}dB），高频信息缺失（合成音频典型特征）")
        
        if mel_analysis:
            analysis_lines.append(f"- **梅尔频谱特征异常**：{'; '.join(mel_analysis)}；")
        else:
            analysis_lines.append(f"- **梅尔频谱特征**：能量均值({round(mel_mean, 1)}dB)符合真人语音基准（{ANOMALY_THRESHOLDS['mel_energy_lower']}~{ANOMALY_THRESHOLDS['mel_energy_upper']}dB）；")
    else:
        analysis_lines.append(f"- **梅尔频谱特征**：提取失败 → {mel_feature.get('error', '未知错误')}；")

    return "\n".join(analysis_lines)

def generate_reference_report(audio_filename):
    feature_result = load_suspicious_features(audio_filename)
    if not feature_result["success"]:
        print(f"❌ {feature_result['error']}")
        return feature_result
    
    feature_data = feature_result["data"]
    total_segments = feature_data["total_suspicious_segments"]
    extracted_segments = feature_data["extracted_segments_count"]
    audio_path = feature_data["audio_path"]

    asr_data = load_asr_diarization_data(audio_filename)

    report_content = []
    report_content.append(f"# 音频伪造检测分析报告")
    report_content.append(f"## 基础信息")
    report_content.append(f"- 语音文件标识：{audio_filename}")
    report_content.append(f"- 原始音频路径：{audio_path}")
    report_content.append(f"- 检测到的可疑片段总数：{total_segments}")
    report_content.append(f"- 成功提取特征的片段数：{extracted_segments}")
    report_content.append(f"- 异常判定基准：LibriSpeech dev-clean 500条真人语音统计（3σ原则）")

    if asr_data:
        report_content.append(f"- 语音识别语言：{asr_data.get('language', '未知')}")
        report_content.append(f"- 音频总时长：{asr_data.get('audio_duration', '未知')} 秒")
        report_content.append(f"- 识别总词数：{asr_data.get('total_words', 0)}")
        report_content.append(f"- 检测到的说话人数量：{asr_data.get('total_speakers', 0)}")
        report_content.append(f"- 完整语音内容：{asr_data.get('full_text', '无')}")
    else:
        report_content.append(f"- 语音识别状态：未获取到ASR+说话人数据")

    report_content.append(f"\n## 可疑片段特征+语音内容分析")

    if extracted_segments == 0:
        report_content.append(f"> 未检测到任何可疑片段，该音频无伪造风险。")
    else:
        for segment_feature in feature_data["segments_features"]:
            anomaly_analysis = analyze_feature_anomaly(segment_feature, asr_data)
            report_content.append(anomaly_analysis)
            report_content.append("")
        
        report_content.append(f"\n## 整体风险评估")
        has_anomaly = False
        anomaly_details = []
        
        for seg in feature_data["segments_features"]:
            mfcc = seg.get("mfcc_feature", {})
            if mfcc.get("success"):
                mfcc_mean_abs = abs(mfcc["mfcc_stats"]["mean"])
                mfcc_std = mfcc["mfcc_stats"]["std"]
                if mfcc_mean_abs > ANOMALY_THRESHOLDS["mfcc_mean_abs"]:
                    has_anomaly = True
                    anomaly_details.append(f"片段{seg['segment_id']}MFCC均值异常")
                if mfcc_std > ANOMALY_THRESHOLDS["mfcc_std_upper"]:
                    has_anomaly = True
                    anomaly_details.append(f"片段{seg['segment_id']}MFCC标准差异常")
            
            mel = seg.get("mel_feature", {})
            if mel.get("success"):
                mel_mean = mel["mel_energy_stats"]["mean"]
                if mel_mean > ANOMALY_THRESHOLDS["mel_energy_upper"] or mel_mean < ANOMALY_THRESHOLDS["mel_energy_lower"]:
                    has_anomaly = True
                    anomaly_details.append(f"片段{seg['segment_id']}梅尔能量异常")
        
        if has_anomaly:
            report_content.append(f"> ⚠️ 检测到以下异常：{'; '.join(anomaly_details)}；该音频**存在伪造风险**。")
            if asr_data:
                report_content.append(f"> 📢 异常片段对应的语音内容已标注，可结合语义进一步验证伪造风险。")
        else:
            report_content.append(f"> ✅ 所有片段特征均符合LibriSpeech真人语音基准，该音频**伪造风险较低**。")

    report_filename = f"{audio_filename}_fake_detection_report.md"
    if not os.path.exists(REFERENCE_OUTPUT_ROOT):
        try:
            os.makedirs(REFERENCE_OUTPUT_ROOT, mode=0o755)
            print(f"✅ 创建目录成功：{REFERENCE_OUTPUT_ROOT}")
        except Exception as e:
            return {
                "success": False,
                "error": f"创建报告目录失败：{str(e)}（权限不足？）"
            }
    report_path = os.path.abspath(os.path.join(REFERENCE_OUTPUT_ROOT, report_filename))
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_content))
        if os.path.exists(report_path):
            print(f"✅ MD文件生成成功：{report_path}")
            print(f"✅ 文件大小：{os.path.getsize(report_path)} 字节")
            return {
                "success": True,
                "audio_filename": audio_filename,
                "report_path": report_path,
                "message": f"分析报告已成功生成：{report_path}"
            }
        else:
            return {
                "success": False,
                "error": f"文件写入后不存在，可能是权限问题：{report_path}"
            }
    except PermissionError:
        return {
            "success": False,
            "error": f"无写入权限：{report_path}（请检查目录权限）"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"保存分析报告失败：{str(e)}"
        }

if __name__ == "__main__":
    test_audio_filename = "LA_E_1000147"
    result = generate_reference_report(test_audio_filename)
    
    if result["success"]:
        print(f"✅ {result['message']}")
    else:
        print(f"❌ {result['error']}")
