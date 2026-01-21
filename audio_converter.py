import autogen
import soundfile as sf
import librosa
import numpy as np
import os
import json
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.getenv("BASE_DIR") or str(Path(__file__).resolve().parent.parent)

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

STANDARD_AUDIO_ROOT = Path(BASE_DIR) / "audio_files" / "standard_audio"
STANDARD_AUDIO_ROOT.mkdir(parents=True, exist_ok=True)
STANDARD_AUDIO_ROOT = str(STANDARD_AUDIO_ROOT)

def extract_audio_filename(audio_path):
    filename = Path(audio_path).stem
    return filename

def convert_audio_to_standard(input_audio_path):
    input_audio_path = Path(input_audio_path).resolve()
    if not input_audio_path.exists():
        result = {
            "success": False,
            "error": f"输入文件不存在：{str(input_audio_path)}",
            "audio_filename": None,
            "audio_path": None,
            "sr": 16000,
            "duration": None
        }
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    audio_filename = extract_audio_filename(input_audio_path)
    output_filename = f"{audio_filename}.wav"
    output_path = str(Path(STANDARD_AUDIO_ROOT) / output_filename)
    
    try:
        cmd = [
            "ffmpeg",
            "-i", str(input_audio_path),
            "-ar", "16000",
            "-ac", "1",
            "-f", "wav",
            "-y",
            output_path
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            encoding='utf-8'
        )
        
        if result.returncode != 0:
            raise Exception(f"ffmpeg执行失败：{result.stderr[:500]}")
        
        if not Path(output_path).exists():
            raise Exception("转换后的WAV文件未生成")
        
        audio, sr = librosa.load(output_path, sr=16000)
        duration = librosa.get_duration(y=audio, sr=sr)
        
        final_result = {
            "success": True,
            "error": None,
            "audio_filename": audio_filename,
            "audio_path": output_path,
            "sr": sr,
            "duration": round(duration, 2)
        }
        return json.dumps(final_result, ensure_ascii=False, indent=2)
    
    except Exception as e:
        final_result = {
            "success": False,
            "error": f"转换失败：{str(e)}",
            "audio_filename": audio_filename if 'audio_filename' in locals() else None,
            "audio_path": None,
            "sr": 16000,
            "duration": None
        }
        return json.dumps(final_result, ensure_ascii=False, indent=2)

format_convert_agent = autogen.UserProxyAgent(
    name="Format_Convert_Agent",
    system_message="""你是音频伪造检测系统的格式转换智能体，核心职责：
    1. 接收任意格式音频文件路径（FLAC/MP3/M4A/WAV等）；
    2. 提取原始音频文件名（去掉后缀），将音频转为16kHz单声道WAV，输出文件名为「原始文件名.wav」；
    3. 返回JSON格式的转换结果（包含success、audio_filename、audio_path、sr、duration等字段）；
    4. 仅处理音频格式转换，不参与其他逻辑。""",
    human_input_mode="NEVER",
    code_execution_config={
        "work_dir": STANDARD_AUDIO_ROOT,
        "use_docker": False,
        "timeout": 60,
    },
)

def test_format_convert_agent(input_audio_path):
    print(f"===== 开始转换音频：{input_audio_path} =====")
    conversion_result = convert_audio_to_standard(input_audio_path)
    print("转换结果（JSON格式）：")
    print(conversion_result)
    print("="*60)
    return conversion_result

if __name__ == "__main__":
    test_audio_path = Path(BASE_DIR) /"audio_files"/ "uploads"/ "LA_E_1000147.flac"

    print(f"📌 项目根目录（来自.env）: {BASE_DIR}")
    print(f"📌 测试音频路径: {test_audio_path}")
    
    test_format_convert_agent(test_audio_path)
