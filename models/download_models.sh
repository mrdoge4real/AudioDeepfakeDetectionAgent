#!/bin/bash
# 下载本项目全部模型，每个模型放在 models/ 下同名文件夹中
#
# 前置：
#   1. source ~/miniforge3/bin/activate antiagent311
#   2. hf auth login   （在线下载 pyannote 需授权+网页接受协议；下到本地后运行不再依赖网络）
# 用法：
#   bash models/download_models.sh
set -e
cd "$(dirname "$0")"

# 与服务端 app/config.py 保持一致：HF 缓存放 models/huggingface
export HF_HOME="$(pwd)/huggingface"

# Deepfake 反伪造检测模型（WavLM-large + AASIST，仓库自带 model.py + 权重）
hf download eliya/forensics_0.3B_wavlm_oc_softmax_deepfake_classifier --local-dir forensics_0.3B_wavlm_oc_softmax_deepfake_classifier

# 上面模型的骨干网络（model.py 内部按 HF 缓存加载，不放本地目录）
hf download microsoft/wavlm-large

# Paraformer 中文语音识别（FunASR）
hf download funasr/paraformer-zh --local-dir paraformer-zh

# pyannote 说话人分割（受限，需先 hf auth login 并接受协议）
hf download pyannote/speaker-diarization-community-1 --local-dir speaker-diarization-community-1

echo ""
echo "🎉 全部下载完成："
du -sh ./*/ 2>/dev/null
