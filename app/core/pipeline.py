# -*- coding: utf-8 -*-
"""检测流水线：确定性 5 步编排，纯代码执行，通过回调上报进度。

注意：app.modules 各模块依赖 librosa/torch 等重型库，这里全部延迟到
run_pipeline 内部导入，保证 API 层在无音频依赖的环境下也能启动。
"""
import traceback
from typing import Callable, Optional

STEPS = [
    (1, "格式转换", "正在转换为 16kHz 单声道 WAV…"),
    (2, "反伪造初检", "正在用 Deepfake 模型滑窗扫描可疑片段…"),
    (3, "ASR+说话人分割", "正在识别语音内容并对齐说话人…"),
    (4, "特征提取", "正在提取可疑片段的 MFCC / 梅尔频谱特征…"),
    (5, "报告生成", "正在比对阈值并生成分析报告…"),
]

ProgressCallback = Optional[Callable[[int, str, str], None]]  # (step, step_name, message)


def _noop(step, name, msg):
    pass


def run_pipeline(audio_path: str, on_progress: ProgressCallback = None) -> dict:
    """执行完整检测流程。返回 {success, audio_filename, report_path, report_content, error, steps}"""
    cb = on_progress or _noop
    steps_done = []

    try:
        # 延迟导入重型模块（librosa/torch/funasr/pyannote）
        from app.modules import (
            anti_spoof, asr_diarization, audio_converter, feature_extractor,
            report_generator,
        )
    except ImportError as e:
        return {
            "success": False,
            "error": f"音频处理依赖缺失：{e}（请先 pip install -r requirements.txt）",
            "failed_step": 0,
            "steps": [],
        }

    def fail(step_no, name, error):
        return {
            "success": False,
            "error": f"步骤{step_no}（{name}）失败：{error}",
            "failed_step": step_no,
            "steps": steps_done,
        }

    try:
        # 步骤1：格式转换
        cb(1, "格式转换", STEPS[0][2])
        conv = audio_converter.convert_audio_to_standard(audio_path)
        if not conv["success"]:
            return fail(1, "格式转换", conv["error"])
        steps_done.append({"step": 1, "name": "格式转换", "result": conv})
        filename = conv["audio_filename"]
        std_path = conv["audio_path"]

        # 步骤2：反伪造初检
        cb(2, "反伪造初检", STEPS[1][2])
        spoof = anti_spoof.run_anti_spoof_detection(std_path)
        if not spoof["success"]:
            return fail(2, "反伪造初检", spoof["error"])
        steps_done.append({"step": 2, "name": "反伪造初检", "result": {
            "suspicious_segments": spoof["data"]["suspicious_segments"],
        }})

        # 步骤3：ASR + 说话人分割
        cb(3, "ASR+说话人分割", STEPS[2][2])
        asr = asr_diarization.extract_asr_with_speaker_diarization(std_path)
        if not asr["success"]:
            return fail(3, "ASR+说话人分割", asr["error"])
        steps_done.append({"step": 3, "name": "ASR+说话人分割", "result": {
            "full_text": asr.get("full_text", ""),
            "total_speakers": asr.get("total_speakers", 0),
        }})

        # 步骤4：特征提取
        cb(4, "特征提取", STEPS[3][2])
        feats = feature_extractor.extract_suspicious_segments_features(filename)
        if not feats["success"]:
            return fail(4, "特征提取", feats["error"])
        steps_done.append({"step": 4, "name": "特征提取", "result": {
            "extracted_segments_count": feats["extracted_segments_count"],
        }})

        # 步骤5：报告生成
        cb(5, "报告生成", STEPS[4][2])
        report = report_generator.generate_report(filename)
        if not report["success"]:
            return fail(5, "报告生成", report["error"])
        steps_done.append({"step": 5, "name": "报告生成", "result": {
            "report_path": report["report_path"],
        }})

        return {
            "success": True,
            "error": None,
            "audio_filename": filename,
            "report_path": report["report_path"],
            "report_content": report["report_content"],
            "steps": steps_done,
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"流水线异常：{e}\n{traceback.format_exc()}",
            "steps": steps_done,
        }
