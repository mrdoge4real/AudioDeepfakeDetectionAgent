# -*- coding: utf-8 -*-
"""流水线编排测试：用 stub 模块替换真实的音频处理模块，验证 5 步顺序与失败短路。"""
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _stub_modules(monkeypatch, behavior):
    """向 sys.modules 注入 app.modules.* 的 stub。behavior 控制每步的返回值。"""
    pkg = types.ModuleType("app.modules")
    for name in ["audio_converter", "anti_spoof", "asr_diarization",
                 "feature_extractor", "report_generator"]:
        mod = types.ModuleType(f"app.modules.{name}")
        setattr(pkg, name, mod)
        monkeypatch.setitem(sys.modules, f"app.modules.{name}", mod)
    monkeypatch.setitem(sys.modules, "app.modules", pkg)

    sys.modules["app.modules.audio_converter"].convert_audio_to_standard = \
        behavior["convert"]
    sys.modules["app.modules.anti_spoof"].run_anti_spoof_detection = \
        behavior["spoof"]
    sys.modules["app.modules.asr_diarization"].extract_asr_with_speaker_diarization = \
        behavior["asr"]
    sys.modules["app.modules.feature_extractor"].extract_suspicious_segments_features = \
        behavior["features"]
    sys.modules["app.modules.report_generator"].generate_report = \
        behavior["report"]


def _ok_behaviors(calls):
    return {
        "convert": lambda p: calls.append(1) or {
            "success": True, "audio_filename": "demo",
            "audio_path": "/tmp/demo.wav", "error": None,
        },
        "spoof": lambda p: calls.append(2) or {
            "success": True, "data": {"suspicious_segments": [{"start": 0.0, "end": 2.7}]},
        },
        "asr": lambda p: calls.append(3) or {
            "success": True, "full_text": "hello", "total_speakers": 1,
        },
        "features": lambda f: calls.append(4) or {
            "success": True, "extracted_segments_count": 1,
        },
        "report": lambda f: calls.append(5) or {
            "success": True, "report_path": "/tmp/report.md", "report_content": "# 报告",
        },
    }


def test_pipeline_runs_five_steps_in_order(monkeypatch):
    calls = []
    _stub_modules(monkeypatch, _ok_behaviors(calls))
    from app.core.pipeline import run_pipeline

    progress = []
    result = run_pipeline("/tmp/demo.flac",
                          on_progress=lambda step, name, msg: progress.append(step))

    assert result["success"] is True
    assert calls == [1, 2, 3, 4, 5]
    assert progress == [1, 2, 3, 4, 5]
    assert result["audio_filename"] == "demo"
    assert result["report_content"] == "# 报告"


def test_pipeline_short_circuits_on_failure(monkeypatch):
    calls = []
    behavior = _ok_behaviors(calls)
    behavior["spoof"] = lambda p: calls.append(2) or {
        "success": False, "error": "模型推断失败",
    }
    _stub_modules(monkeypatch, behavior)
    from app.core.pipeline import run_pipeline

    result = run_pipeline("/tmp/demo.flac")

    assert result["success"] is False
    assert result["failed_step"] == 2
    assert "模型推断失败" in result["error"]
    assert calls == [1, 2]  # 步骤3-5 未执行
