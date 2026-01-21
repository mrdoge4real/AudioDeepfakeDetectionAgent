import autogen
import soundfile as sf
import librosa
import numpy as np
import os
import json
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv  # 导入dotenv加载.env配置

# ====================== 1. 加载.env配置（核心） ======================
# 加载.env文件（优先从脚本所在目录找，找不到则从项目根目录找）
load_dotenv()

# 从.env读取项目根目录，设置兜底值（动态推导）
BASE_DIR = os.getenv("BASE_DIR") or str(Path(__file__).resolve().parent.parent)

# ====================== 全局配置（解决编码问题+跨平台路径） ======================
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# 标准化音频输出根目录（基于.env的BASE_DIR，跨平台兼容）
STANDARD_AUDIO_ROOT = Path(BASE_DIR) / "audio_files" / "standard_audio"
STANDARD_AUDIO_ROOT.mkdir(parents=True, exist_ok=True)  # 自动创建多级目录
STANDARD_AUDIO_ROOT = str(STANDARD_AUDIO_ROOT)  # 转为字符串兼容os模块

# ====================== 工具函数：提取原始文件名（去后缀，跨平台） ======================
def extract_audio_filename(audio_path):
    """
    跨平台提取音频文件名（去后缀），兼容Linux/Windows路径
    示例：
    - /home/bowen/audio/LA_E_1000147.flac → LA_E_1000147
    - E:/audio/test_audio.mp3 → test_audio
    - ./my_audio.wav → my_audio
    """
    # 用pathlib跨平台解析路径，避免os.path的系统差异
    filename = Path(audio_path).stem
    return filename

# ====================== 音频格式转换核心函数（适配环境变量+跨平台） ======================
def convert_audio_to_standard(input_audio_path):
    """
    将任意格式音频（FLAC/MP3/M4A等）转换为16kHz单声道WAV
    输出文件名：原始文件名（去后缀）.wav（保存在BASE_DIR/audio_files/standard_audio/目录下）
    依赖：ffmpeg已配置到系统环境变量（终端输入ffmpeg -version可验证）
    """
    # 1. 跨平台标准化输入路径（转为绝对路径，兼容所有系统）
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
    
    # 2. 提取原始文件名（去后缀），动态生成输出路径（跨平台）
    audio_filename = extract_audio_filename(input_audio_path)
    output_filename = f"{audio_filename}.wav"  # 原始文件名.wav
    output_path = str(Path(STANDARD_AUDIO_ROOT) / output_filename)  # pathlib拼接
    
    try:
        # 3. 核心：直接调用ffmpeg命令（环境变量已配置，无需指定路径）
        cmd = [
            "ffmpeg",          # 系统从环境变量找ffmpeg，跨平台兼容
            "-i", str(input_audio_path),  # 转为字符串，兼容subprocess
            "-ar", "16000",    # 采样率16kHz（声纹/频谱特征提取标准）
            "-ac", "1",        # 单声道
            "-f", "wav",       # 输出格式WAV
            "-y",              # 覆盖已有文件（无需确认）
            output_path        # 输出路径：BASE_DIR/audio_files/standard_audio/原始文件名.wav
        ]
        
        # 执行ffmpeg命令（捕获输出，便于排查）
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,       # 跨平台建议False，避免路径解析问题
            encoding='utf-8'
        )
        
        # 检查ffmpeg执行结果（返回码0=成功）
        if result.returncode != 0:
            raise Exception(f"ffmpeg执行失败：{result.stderr[:500]}")  # 只显示前500字符
        
        # 4. 验证转换后的WAV文件
        if not Path(output_path).exists():
            raise Exception("转换后的WAV文件未生成")
        
        # 5. 获取音频基础信息（采样率/时长）
        audio, sr = librosa.load(output_path, sr=16000)
        duration = librosa.get_duration(y=audio, sr=sr)
        
        # 6. 构造成功结果（JSON格式，包含原始文件名）
        final_result = {
            "success": True,
            "error": None,
            "audio_filename": audio_filename,  # 原始文件名（去后缀）
            "audio_path": output_path,         # 输出路径（跨平台字符串）
            "sr": sr,
            "duration": round(duration, 2)
        }
        return json.dumps(final_result, ensure_ascii=False, indent=2)
    
    except Exception as e:
        # 构造失败结果
        final_result = {
            "success": False,
            "error": f"转换失败：{str(e)}",
            "audio_filename": audio_filename if 'audio_filename' in locals() else None,
            "audio_path": None,
            "sr": 16000,
            "duration": None
        }
        return json.dumps(final_result, ensure_ascii=False, indent=2)

# ====================== AutoGen格式转换智能体定义（跨平台） ======================
format_convert_agent = autogen.UserProxyAgent(
    name="Format_Convert_Agent",
    system_message="""你是音频伪造检测系统的格式转换智能体，核心职责：
    1. 接收任意格式音频文件路径（FLAC/MP3/M4A/WAV等）；
    2. 提取原始音频文件名（去掉后缀），将音频转为16kHz单声道WAV，输出文件名为「原始文件名.wav」；
    3. 返回JSON格式的转换结果（包含success、audio_filename、audio_path、sr、duration等字段）；
    4. 仅处理音频格式转换，不参与其他逻辑。""",
    human_input_mode="NEVER",  # 自动执行，无需人工干预
    code_execution_config={
        "work_dir": STANDARD_AUDIO_ROOT,  # 基于.env的工作目录
        "use_docker": False,              # 本地运行，无需Docker
        "timeout": 60,                    # 转换超时时间60秒
    },
)

# ====================== 测试函数（基于.env配置，跨平台） ======================
def test_format_convert_agent(input_audio_path):
    """测试格式转换智能体（跨平台通用）"""
    print(f"===== 开始转换音频：{input_audio_path} =====")
    # 执行转换
    conversion_result = convert_audio_to_standard(input_audio_path)
    # 打印结果
    print("转换结果（JSON格式）：")
    print(conversion_result)
    print("="*60)
    return conversion_result

# 执行测试（基于.env的BASE_DIR，无硬编码路径）
if __name__ == "__main__":
    # 测试路径：基于.env的BASE_DIR拼接，跨平台兼容
    test_audio_path = Path(BASE_DIR) /"audio_files"/ "uploads"/ "LA_E_1000147.flac"

    print(f"📌 项目根目录（来自.env）: {BASE_DIR}")
    print(f"📌 测试音频路径: {test_audio_path}")
    
    test_format_convert_agent(test_audio_path)