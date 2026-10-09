# -*- coding: utf-8 -*-
"""步骤5：报告生成 —— 以 Deepfake 检测模型结论为主判定，MFCC/梅尔特征仅作辅助解释。"""
import json

from app.config import (
    ANOMALY_THRESHOLDS as T, ANTI_SPOOF_DIR, ASR_OUTPUT_DIR, REPORT_DIR,
    SUSPICIOUS_FEATURE_DIR,
)


def _load_json(path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_asr_data(audio_filename: str):
    data = _load_json(ASR_OUTPUT_DIR / f"{audio_filename}_asr_diarization.json")
    return data if data and data.get("success") else None


def _load_anti_spoof(audio_filename: str):
    data = _load_json(ANTI_SPOOF_DIR / f"{audio_filename}_anti_spoof.json")
    return data if data and data.get("success") else None


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
        "speakers_in_segment": sorted({w["speaker_id"] for w in matched}),
    }


def _segment_max_fake_prob(time_range: dict, anti_spoof: dict) -> float:
    """模型在该片段时间范围内给出的最高伪造概率。"""
    scores = anti_spoof["data"].get("fake_scores", [])
    stamps = anti_spoof["data"].get("time_stamps", [])
    in_range = [s for s, t in zip(scores, stamps) if time_range["start"] <= t <= time_range["end"]]
    return max(in_range) if in_range else 0.0


def _segment_feature_cues(seg: dict) -> list:
    """MFCC/梅尔特征与真人语音统计基准的偏离描述，仅作辅助解释，不作判定依据。"""
    cues = []
    mfcc = seg.get("mfcc_feature", {})
    if mfcc.get("success"):
        stats = mfcc["mfcc_stats"]
        if abs(stats["mean"]) > T["mfcc_mean_abs"]:
            cues.append(f"MFCC均值绝对值 {round(abs(stats['mean']), 3)}（基准 ≤{T['mfcc_mean_abs']}）")
        if stats["std"] > T["mfcc_std_upper"]:
            cues.append(f"MFCC整体标准差 {round(stats['std'], 3)}（基准 ≤{T['mfcc_std_upper']}），频谱波动偏大")
    mel = seg.get("mel_feature", {})
    if mel.get("success"):
        mean = mel["mel_energy_stats"]["mean"]
        if mean > T["mel_energy_upper"]:
            cues.append(f"梅尔能量均值 {round(mean, 1)}dB 偏高（基准 ≤{T['mel_energy_upper']}dB）")
        elif mean < T["mel_energy_lower"]:
            cues.append(f"梅尔能量均值 {round(mean, 1)}dB 偏低（基准 ≥{T['mel_energy_lower']}dB），高频信息偏少")
    return cues


def _assess_risk(suspicious_count: int, suspicious_duration: float, duration: float) -> str:
    """基于模型判定结果评估风险：可疑片段数量 + 占音频时长比例。"""
    if suspicious_count == 0:
        return "低风险"
    ratio = suspicious_duration / duration if duration > 0 else 0.0
    if suspicious_count >= 3 or ratio > 0.1:
        return "高风险"
    return "中等风险"


def generate_report(audio_filename: str) -> dict:
    anti_spoof = _load_anti_spoof(audio_filename)
    if not anti_spoof:
        return {"success": False, "error": "反伪造初检结果不存在，无法生成报告"}
    feature_result = _load_suspicious_features(audio_filename)
    if not feature_result["success"]:
        return feature_result

    feature_data = feature_result["data"]
    asr_data = _load_asr_data(audio_filename)
    duration = anti_spoof.get("audio_duration") or (asr_data or {}).get("audio_duration", 0) or 0
    suspicious_segments = anti_spoof["data"]["suspicious_segments"]
    suspicious_duration = sum(s["end"] - s["start"] for s in suspicious_segments)
    threshold = anti_spoof.get("threshold", 0.7)

    lines = [
        "# 音频伪造检测分析报告",
        "## 基础信息",
        f"- 语音文件标识：{audio_filename}",
        f"- 原始音频路径：{anti_spoof['audio_path']}",
        f"- 检测模型：Deepfake 音频检测模型（滑窗 {anti_spoof.get('window_size')}s / 步长 {anti_spoof.get('hop_size')}s，伪造概率阈值 {threshold}）",
        f"- 模型判定的可疑片段数：{len(suspicious_segments)}",
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

    lines.append("\n## 可疑片段分析（模型判定 + 特征解释）")

    seg_features = feature_data.get("segments_features", [])
    if not suspicious_segments:
        lines.append("> 检测模型未发现伪造概率超过阈值的片段，该音频判定为真实人声。")
    else:
        for i, tr in enumerate(suspicious_segments, 1):
            lines.append(f"### 片段{i}（时间范围：{tr['start']}s - {tr['end']}s）")
            max_prob = _segment_max_fake_prob(tr, anti_spoof)
            lines.append(f"- **模型伪造概率**：{round(max_prob * 100, 1)}%（≥{round(threshold * 100)}% 判定为可疑）")
            if asr_data and asr_data.get("segments"):
                m = _match_segment_text(tr, asr_data["segments"])
                lines.append(f"- **语音内容**：{m['matched_text'] or '无匹配内容'}")
                lines.append(f"- **说话人**：{', '.join(m['speakers_in_segment']) or 'UNKNOWN'}")
            # MFCC/梅尔特征：仅供人工参考的物理解释，不参与判定
            seg_feat = next(
                (s for s in seg_features if s.get("time_range") == tr), None
            )
            if seg_feat:
                cues = _segment_feature_cues(seg_feat)
                if cues:
                    lines.append(f"- **声学特征参考**：{'；'.join(cues)}（与 LibriSpeech 真人语音统计基准的偏离，仅供解释）")
                else:
                    lines.append("- **声学特征参考**：MFCC 与梅尔能量均在真人语音统计基准范围内")
            lines.append("")

        risk = _assess_risk(len(suspicious_segments), suspicious_duration, duration)
        ratio_pct = round(suspicious_duration / duration * 100, 1) if duration > 0 else 0
        lines.append("\n## 整体风险评估")
        lines.append(
            f"> ⚠️ 检测模型判定 {len(suspicious_segments)} 个片段为伪造"
            f"（合计约 {round(suspicious_duration, 1)}s，占音频 {ratio_pct}%），"
            f"风险等级：**{risk}**，该音频存在伪造风险。"
        )
        if asr_data:
            lines.append("> 📢 可疑片段对应的语音内容已标注，可结合语义进一步人工核验。")

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
