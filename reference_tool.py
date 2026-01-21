# -*- coding: utf-8 -*-
import os
import json
import sys
import numpy as np
from pathlib import Path
from dotenv import load_dotenv  # 导入dotenv加载.env配置

# ====================== 1. 加载.env配置（软编码核心） ======================
# 加载.env文件（优先从脚本所在目录找，找不到则从项目根目录找）
load_dotenv()

# 【修复】强制指定BASE_DIR为DeepfakedetectionAgent2（和主程序一致），避免路径推导错误
BASE_DIR = os.getenv("BASE_DIR")  # 兜底值改为主程序的根目录

# ====================== 全局配置（基于BASE_DIR软编码，跨平台兼容） ======================
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# 各Agent输出目录（基于BASE_DIR动态拼接，软编码）
SUSPICIOUS_FEATURE_ROOT = os.path.join(BASE_DIR, "outputs", "suspicious_features")
ASR_OUTPUT_ROOT = os.path.join(BASE_DIR, "outputs", "asr")
REFERENCE_OUTPUT_ROOT = os.path.join(BASE_DIR, "outputs", "reference_report")
os.makedirs(REFERENCE_OUTPUT_ROOT, exist_ok=True)

# ====================== 核心：基于LibriSpeech统计的异常阈值 ======================
ANOMALY_THRESHOLDS = {
    # MFCC阈值（来自LibriSpeech 500样本统计）
    "mfcc_mean_abs": 0.5,             # MFCC均值绝对值异常阈值
    "mfcc_std_upper": 35.0141,        # MFCC整体标准差异常上限
    "mfcc_inner_std_upper": 44.6855,  # MFCC维度内标准差异常上限
    # 梅尔频谱能量阈值（来自LibriSpeech 500样本统计）
    "mel_energy_upper": -43.5002,     # 梅尔能量偏高异常阈值
    "mel_energy_lower": -65.9447,     # 梅尔能量偏低异常阈值
    # 风险等级判定阈值
    "high_risk_anomaly_count": 2,     # ≥2个异常类型 → 高风险
    "medium_risk_anomaly_count": 1    # 1个异常类型 → 中风险
}

# ====================== 工具函数：读取ASR+说话人数据 ======================
def load_asr_diarization_data(audio_filename):
    """
    读取指定音频的ASR+说话人分割结果
    :param audio_filename: 语音文件名（去后缀），如LA_E_1000147
    :return: ASR数据字典 / None（无数据时）
    """
    asr_json_path = os.path.join(ASR_OUTPUT_ROOT, f"{audio_filename}_asr_diarization.json")
    
    # 检查ASR文件是否存在
    if not os.path.exists(asr_json_path):
        print(f"ℹ️ 未找到{audio_filename}的ASR+说话人数据：{asr_json_path}")
        return None
    
    try:
        with open(asr_json_path, "r", encoding="utf-8") as f:
            asr_data = json.load(f)
        
        # 校验ASR数据是否有效
        if not asr_data.get("success"):
            print(f"ℹ️ {audio_filename}的ASR数据无效：{asr_data.get('error')}")
            return None
        
        return asr_data
    except Exception as e:
        print(f"ℹ️ 解析ASR数据失败：{str(e)}")
        return None

# ====================== 工具函数：匹配可疑片段与语音内容 ======================
def match_suspicious_segment_with_text(suspicious_segment, asr_segments):
    """
    匹配可疑片段对应的语音内容
    :param suspicious_segment: 可疑片段（start/end）
    :param asr_segments: ASR词级分段数据
    :return: 该时间段内的语音内容列表
    """
    seg_start = suspicious_segment["start"]
    seg_end = suspicious_segment["end"]
    matched_words = []
    
    for word_seg in asr_segments:
        # 词的时间范围与可疑片段有交集即匹配
        word_start = word_seg["start"]
        word_end = word_seg["end"]
        
        if not (word_end < seg_start or word_start > seg_end):
            matched_words.append({
                "speaker_id": word_seg["speaker_id"],
                "word": word_seg["word"],
                "start": word_seg["start"],
                "end": word_seg["end"]
            })
    
    # 拼接成完整文本
    matched_text = " ".join([w["word"] for w in matched_words])
    return {
        "matched_words": matched_words,
        "matched_text": matched_text,
        "total_matched_words": len(matched_words),
        "speakers_in_segment": list(set([w["speaker_id"] for w in matched_words]))
    }

# ====================== 1. 读取可疑片段特征汇总文件 ======================
def load_suspicious_features(audio_filename):
    """
    读取指定语音文件的可疑片段特征汇总文件
    """
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

# ====================== 2. 特征异常分析核心逻辑（核心更新） ======================
def analyze_feature_anomaly(segment_feature, asr_data=None):
    """
    分析单个可疑片段的特征异常 + 关联语音内容/说话人
    核心更新：基于LibriSpeech统计阈值判定异常
    """
    analysis_lines = []
    segment_id = segment_feature["segment_id"]
    time_range = segment_feature["time_range"]
    time_str = f"{time_range['start']}s - {time_range['end']}s"

    # 基础信息
    analysis_lines.append(f"### 片段{segment_id}（时间范围：{time_str}）")

    # ===== ASR+说话人内容关联 =====
    if asr_data and asr_data.get("segments"):
        match_result = match_suspicious_segment_with_text(time_range, asr_data["segments"])
        analysis_lines.append(f"- **语音内容**：{match_result['matched_text'] or '无匹配内容'}")
        analysis_lines.append(f"- **说话人**：{', '.join(match_result['speakers_in_segment']) or 'UNKNOWN'}")
        analysis_lines.append(f"- **匹配词数**：{match_result['total_matched_words']}")
    else:
        analysis_lines.append(f"- **语音内容**：未获取到ASR数据")
        analysis_lines.append(f"- **说话人**：未获取到说话人数据")
    
    analysis_lines.append("")  # 空行分隔

    # ===== 核心更新：MFCC特征分析（基于LibriSpeech统计阈值） =====
    mfcc_feature = segment_feature["mfcc_feature"]
    if mfcc_feature["success"]:
        mfcc_stats = mfcc_feature["mfcc_stats"]
        mfcc_mean_abs = abs(mfcc_stats["mean"])  # 均值绝对值
        mfcc_std = mfcc_stats["std"]              # 对应mfcc_std_mean（整体标准差）
        
        mfcc_analysis = []
        # 1. MFCC均值异常判定
        if mfcc_mean_abs > ANOMALY_THRESHOLDS["mfcc_mean_abs"]:
            mfcc_analysis.append(f"MFCC均值绝对值({round(mfcc_mean_abs, 3)})超出正常范围（≤{ANOMALY_THRESHOLDS['mfcc_mean_abs']}）")
        # 2. MFCC整体标准差异常判定（核心）
        if mfcc_std > ANOMALY_THRESHOLDS["mfcc_std_upper"]:
            mfcc_analysis.append(f"MFCC整体标准差({round(mfcc_std, 3)})超出真人语音基准（≤{ANOMALY_THRESHOLDS['mfcc_std_upper']}），频谱波动异常（合成音频典型特征）")
        
        if mfcc_analysis:
            analysis_lines.append(f"- **MFCC特征异常**：{'; '.join(mfcc_analysis)}；")
        else:
            analysis_lines.append(f"- **MFCC特征**：均值绝对值({round(mfcc_mean_abs, 3)})、整体标准差({round(mfcc_std, 3)})均符合真人语音基准；")
    else:
        analysis_lines.append(f"- **MFCC特征**：提取失败 → {mfcc_feature.get('error', '未知错误')}；")

    # ===== 核心更新：梅尔频谱特征分析（基于LibriSpeech统计阈值） =====
    mel_feature = segment_feature["mel_feature"]
    if mel_feature["success"]:
        mel_stats = mel_feature["mel_energy_stats"]
        mel_mean = mel_stats["mean"]  # 梅尔能量均值（dB）
        
        mel_analysis = []
        # 1. 梅尔能量偏高异常
        if mel_mean > ANOMALY_THRESHOLDS["mel_energy_upper"]:
            mel_analysis.append(f"梅尔能量均值({round(mel_mean, 1)}dB)偏高（正常≤{ANOMALY_THRESHOLDS['mel_energy_upper']}dB），频域能量分布异常")
        # 2. 梅尔能量偏低异常
        elif mel_mean < ANOMALY_THRESHOLDS["mel_energy_lower"]:
            mel_analysis.append(f"梅尔能量均值({round(mel_mean, 1)}dB)偏低（正常≥{ANOMALY_THRESHOLDS['mel_energy_lower']}dB），高频信息缺失（合成音频典型特征）")
        
        if mel_analysis:
            analysis_lines.append(f"- **梅尔频谱特征异常**：{'; '.join(mel_analysis)}；")
        else:
            analysis_lines.append(f"- **梅尔频谱特征**：能量均值({round(mel_mean, 1)}dB)符合真人语音基准（{ANOMALY_THRESHOLDS['mel_energy_lower']}~{ANOMALY_THRESHOLDS['mel_energy_upper']}dB）；")
    else:
        analysis_lines.append(f"- **梅尔频谱特征**：提取失败 → {mel_feature.get('error', '未知错误')}；")

    return "\n".join(analysis_lines)

# ====================== 3. 生成最终分析报告（更新异常判定逻辑） ======================
def generate_reference_report(audio_filename):
    """
    生成完整的音频伪造检测分析报告（含ASR+说话人+特征异常）
    核心更新：基于LibriSpeech统计阈值做整体风险评估
    """
    # 步骤1：读取特征数据
    feature_result = load_suspicious_features(audio_filename)
    if not feature_result["success"]:
        print(f"❌ {feature_result['error']}")
        return feature_result
    
    feature_data = feature_result["data"]
    total_segments = feature_data["total_suspicious_segments"]
    extracted_segments = feature_data["extracted_segments_count"]
    audio_path = feature_data["audio_path"]

    # 步骤2：读取ASR+说话人数据
    asr_data = load_asr_diarization_data(audio_filename)

    # 步骤3：构建报告内容
    report_content = []
    # 报告标题
    report_content.append(f"# 音频伪造检测分析报告")
    report_content.append(f"## 基础信息")
    report_content.append(f"- 语音文件标识：{audio_filename}")
    report_content.append(f"- 原始音频路径：{audio_path}")
    report_content.append(f"- 检测到的可疑片段总数：{total_segments}")
    report_content.append(f"- 成功提取特征的片段数：{extracted_segments}")
    # 新增：标注阈值基准来源
    report_content.append(f"- 异常判定基准：LibriSpeech dev-clean 500条真人语音统计（3σ原则）")

    # ===== ASR+说话人基础信息 =====
    if asr_data:
        report_content.append(f"- 语音识别语言：{asr_data.get('language', '未知')}")
        report_content.append(f"- 音频总时长：{asr_data.get('audio_duration', '未知')} 秒")
        report_content.append(f"- 识别总词数：{asr_data.get('total_words', 0)}")
        report_content.append(f"- 检测到的说话人数量：{asr_data.get('total_speakers', 0)}")
        report_content.append(f"- 完整语音内容：{asr_data.get('full_text', '无')}")
    else:
        report_content.append(f"- 语音识别状态：未获取到ASR+说话人数据")

    report_content.append(f"\n## 可疑片段特征+语音内容分析")

    # 无可疑片段场景
    if extracted_segments == 0:
        report_content.append(f"> 未检测到任何可疑片段，该音频无伪造风险。")
    # 有可疑片段场景：逐个分析（关联ASR）
    else:
        for segment_feature in feature_data["segments_features"]:
            anomaly_analysis = analyze_feature_anomaly(segment_feature, asr_data)
            report_content.append(anomaly_analysis)
            report_content.append("")  # 分段空行
        
        # ===== 核心更新：整体风险评估（基于LibriSpeech阈值） =====
        report_content.append(f"\n## 整体风险评估")
        has_anomaly = False
        anomaly_details = []
        
        for seg in feature_data["segments_features"]:
            # MFCC异常判定
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
            
            # 梅尔频谱异常判定
            mel = seg.get("mel_feature", {})
            if mel.get("success"):
                mel_mean = mel["mel_energy_stats"]["mean"]
                if mel_mean > ANOMALY_THRESHOLDS["mel_energy_upper"] or mel_mean < ANOMALY_THRESHOLDS["mel_energy_lower"]:
                    has_anomaly = True
                    anomaly_details.append(f"片段{seg['segment_id']}梅尔能量异常")
        
        # 风险评估结论
        if has_anomaly:
            report_content.append(f"> ⚠️ 检测到以下异常：{'; '.join(anomaly_details)}；该音频**存在伪造风险**。")
            if asr_data:
                report_content.append(f"> 📢 异常片段对应的语音内容已标注，可结合语义进一步验证伪造风险。")
        else:
            report_content.append(f"> ✅ 所有片段特征均符合LibriSpeech真人语音基准，该音频**伪造风险较低**。")

    # 步骤4：保存报告文件（修改这部分）
    report_filename = f"{audio_filename}_fake_detection_report.md"
    # 1. 验证目录是否可写
    if not os.path.exists(REFERENCE_OUTPUT_ROOT):
        try:
            os.makedirs(REFERENCE_OUTPUT_ROOT, mode=0o755)  # 显式指定权限
            print(f"✅ 创建目录成功：{REFERENCE_OUTPUT_ROOT}")
        except Exception as e:
            return {
                "success": False,
                "error": f"创建报告目录失败：{str(e)}（权限不足？）"
            }
    # 2. 生成绝对路径
    report_path = os.path.abspath(os.path.join(REFERENCE_OUTPUT_ROOT, report_filename))
    # 3. 写入文件（添加错误捕获）
    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_content))
        # 验证文件是否真的生成
        if os.path.exists(report_path):
            print(f"✅ MD文件生成成功：{report_path}")
            print(f"✅ 文件大小：{os.path.getsize(report_path)} 字节")
            return {
                "success": True,
                "audio_filename": audio_filename,
                "report_path": report_path,  # 返回绝对路径
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
# ====================== 主程序入口（测试用） ======================
if __name__ == "__main__":
    test_audio_filename = "LA_E_1000147"  # 示例音频文件名（去后缀）
    result = generate_reference_report(test_audio_filename)
    
    if result["success"]:
        print(f"✅ {result['message']}")
    else:
        print(f"❌ {result['error']}")