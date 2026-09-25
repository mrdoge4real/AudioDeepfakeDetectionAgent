# -*- coding: utf-8 -*-
"""步骤5：报告生成 —— 汇总可疑片段特征 + ASR 内容，比对阈值，输出 Markdown 报告。"""
import json
from pathlib import Path

from app.config import (
    ANOMALY_THRESHOLDS as T, ASR_OUTPUT_DIR, REPORT_DIR, SUSPICIOUS_FEATURE_DIR,
)


def _load_asr_data(audio_filename: str):
    path = ASR_OUTPUT_DIR / f"{audio_filename}_asr_diarization.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if data.get("success") else None
    except Exception:
        return None


def _load_suspicious_features(audio_filename: str) -> dict:
    path = SUSPICIOUS_FEATURE_DIR / audio_filename / "suspicious_features_summary.json"
    if not path.exists():
        return {"success": False, "error": f"可疑片段特征汇总文件不存在：{path}"}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not data.get("success"):
            return {"success": False, "error": "特征提取失败，汇总文件标记为失败状态"}
        return {"success": True, "data": data}
    except Exception as e:
        return {"success": False, "error": f"解析特征文件失败：{e}"}


def _match_segment_text(time_range: dict, asr_segments: list) -> dict:
    matched = [
        w for w in asr_segments
        if not (w["end"] < time_range["start"] or w["start"] > time_range["end"])
    ]
    return {
        "matched_text": " ".join(w["word"] for w in matched),
        "total_matched_words": len(matched),
        "speakers_in_segment": sorted({w["speaker_id"] for w in matched}),
    }


def _segment_anomalies(seg: dict) -> list:
    """返回该片段触发的异常描述列表（空列表 = 无异常）。"""
    anomalies = []
    mfcc = seg.get("mfcc_feature", {})
    if mfcc.get("success"):
        stats = mfcc["mfcc_stats"]
        if abs(stats["mean"]) > T["mfcc_mean_abs"]:
            anomalies.append(
                f"MFCC均值绝对值({round(abs(stats['mean']), 3)})超出正常范围（≤{T['mfcc_mean_abs']}）"
            )
        if stats["std"] > T["mfcc_std_upper"]:
            anomalies.append(
                f"MFCC整体标准差({round(stats['std'], 3)})超出真人语音基准（≤{T['mfcc_std_upper']}），频谱波动异常"
            )
    mel = seg.get("mel_feature", {})
    if mel.get("success"):
        mean = mel["mel_energy_stats"]["mean"]
        if mean > T["mel_energy_upper"]:
            anomalies.append(
                f"梅尔能量均值({round(mean, 1)}dB)偏高（正常≤{T['mel_energy_upper']}dB），频域能量分布异常"
            )
        elif mean < T["mel_energy_lower"]:
            anomalies.append(
                f"梅尔能量均值({round(mean, 1)}dB)偏低（正常≥{T['mel_energy_lower']}dB），高频信息缺失"
            )
    return anomalies


def _assess_risk(suspicious_count: int, anomaly_count: int, duration: float) -> str:
    if suspicious_count == 0 or anomaly_count == 0:
        return "低风险"
    suspicious_ratio = 0.0
    if duration > 0:
        # 粗略以可疑片段数量估计占比（每段平均 2.5s 仅为兜底，真实占比在流水线中可细化）
        suspicious_ratio = min(1.0, suspicious_count * 2.5 / duration)
    if suspicious_count >= 3 or suspicious_ratio > 0.1:
        return "高风险"
    return "中等风险"


def generate_report(audio_filename: str) -> dict:
    feature_result = _load_suspicious_features(audio_filename)
    if not feature_result["success"]:
        return feature_result

    feature_data = feature_result["data"]
    asr_data = _load_asr_data(audio_filename)
    duration = feature_data.get("audio_duration") or (asr_data or {}).get("audio_duration", 0) or 0

    lines = [
        "# 音频伪造检测分析报告",
        "## 基础信息",
        f"- 语音文件标识：{audio_filename}",
        f"- 原始音频路径：{feature_data['audio_path']}",
        f"- 检测到的可疑片段总数：{feature_data['total_suspicious_segments']}",
        f"- 成功提取特征的片段数：{feature_data['extracted_segments_count']}",
        "- 异常判定基准：LibriSpeech dev-clean 500条真人语音统计（3σ原则）",
    ]
    if asr_data:
        lines += [
            f"- 语音识别语言：{asr_data.get('language', '未知')}",
            f"- 音频总时长：{asr_data.get('audio_duration', '未知')} 秒",
            f"- 识别总词数：{asr_data.get('total_words', 0)}",
            f"- 检测到的说话人数量：{asr_data.get('total_speakers', 0)}",
            f"- 完整语音内容：{asr_data.get('full_text', '无')}",
        ]
    else:
        lines.append("- 语音识别状态：未获取到ASR+说话人数据")

    lines.append("\n## 可疑片段特征+语音内容分析")

    total_anomalies = 0
    if feature_data["extracted_segments_count"] == 0:
        lines.append("> 未检测到任何可疑片段，该音频无伪造风险。")
    else:
        for seg in feature_data["segments_features"]:
            tr = seg["time_range"]
            lines.append(f"### 片段{seg['segment_id']}（时间范围：{tr['start']}s - {tr['end']}s）")
            if asr_data and asr_data.get("segments"):
                m = _match_segment_text(tr, asr_data["segments"])
                lines.append(f"- **语音内容**：{m['matched_text'] or '无匹配内容'}")
                lines.append(f"- **说话人**：{', '.join(m['speakers_in_segment']) or 'UNKNOWN'}")
            anomalies = _segment_anomalies(seg)
            total_anomalies += len(anomalies)
            if anomalies:
                lines.append(f"- **异常特征**：{'; '.join(anomalies)}；")
            else:
                lines.append("- **特征状态**：MFCC 与梅尔能量均符合真人语音基准；")
            lines.append("")

        risk = _assess_risk(
            feature_data["total_suspicious_segments"], total_anomalies, duration
        )
        lines.append("\n## 整体风险评估")
        if total_anomalies:
            lines.append(
                f"> ⚠️ 共检测到 {total_anomalies} 项异常，风险等级：**{risk}**，该音频存在伪造风险。"
            )
            if asr_data:
                lines.append("> 📢 异常片段对应的语音内容已标注，可结合语义进一步验证。")
        else:
            lines.append("> ✅ 所有片段特征均符合 LibriSpeech 真人语音基准，风险等级：**低风险**。")

    report_path = REPORT_DIR / f"{audio_filename}_fake_detection_report.md"
    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        content = "\n".join(lines)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ 检测报告已生成：{report_path}")
        return {
            "success": True,
            "audio_filename": audio_filename,
            "report_path": str(report_path),
            "report_content": content,
        }
    except Exception as e:
        return {"success": False, "error": f"保存分析报告失败：{e}"}
