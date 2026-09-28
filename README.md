# 智能音频伪造检测智能体

一款基于 **FastAPI + 原生 Function Calling Agent** 的**全流程自动化音频伪造（Deepfake）检测服务**，专为识别 AI 生成 / TTS / VC 音频设计。上传音频即可自动完成「格式标准化 → 反伪造初检 → ASR+说话人分割 → 特征提取 → 报告生成」五步流水线，并通过真·智能体提供对话式结果解读与专业知识问答。

---

## 架构

```
┌─────────────────────────────────────────┐
│  FastAPI 服务层                          │
│  POST /api/detect (multipart 上传)       │
│  GET  /api/tasks/{id}  (状态/进度/报告)   │
│  POST /api/chat    (对话式 Agent)        │
├─────────────────────────────────────────┤
│  真·Agent 层（对话大脑）                  │
│  LLM (原生 function calling) 自主决策：   │
│  get_task_status / get_report /          │
│  list_history / explain_knowledge        │
├─────────────────────────────────────────┤
│  确定性流水线层（纯代码编排，零 LLM）      │
│  convert → anti_spoof → asr →           │
│  features → report                       │
├─────────────────────────────────────────┤
│  能力模块层（模型单例，启动预热）          │
│  Deepfake 检测模型 / Paraformer / pyannote│
└─────────────────────────────────────────┘
```

### 与旧版（main 分支）的区别

| | 旧版 | 新版 |
|---|---|---|
| 交互方式 | 命令行 + 正则解析 Windows 路径 | Web 页面上传（multipart）+ REST API |
| "Agent" | LLM 被强制输出固定 JSON，实为状态机 | 检测流水线纯代码执行；对话层 LLM 自主 function calling |
| 模型加载 | 每次检测重新加载 3 个模型 | 启动预热 + 全局单例 |
| 任务执行 | 同步阻塞 | 异步任务 + 进度轮询，SQLite 持久化 |
| LLM | 锁死 deepseek-reasoner | 任意 OpenAI 兼容端点可配置（默认 deepseek-chat） |

---

## 部署

### 1. 环境准备（uv）

```bash
# 安装 uv（任选其一）
curl -LsSf https://astral.sh/uv/install.sh | sh
# brew install uv

git clone https://github.com/mrdoge4real/AudioDeepfakeDetectionAgent.git
cd AudioDeepfakeDetectionAgent

# 一键复现环境：自动下载 Python 3.11、创建 .venv、安装全部依赖
uv sync
# 国内网络可加镜像：UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple uv sync

# 激活环境
source .venv/bin/activate
```

> Linux/Windows 需要 CUDA 版 PyTorch 时：`uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121`

### 2. 配置 `.env`

```bash
cp .env.example .env
```

```ini
# LLM：需要支持 function calling 的模型（OpenAI 兼容端点皆可）
# Kimi 示例：
LLM_API_KEY=sk-kimi-xxx
LLM_API_BASE=https://api.kimi.com/coding/v1
LLM_MODEL=kimi-for-coding
LLM_TEMPERATURE=1
# DeepSeek 示例：
# LLM_API_KEY=sk-xxx
# LLM_API_BASE=https://api.deepseek.com
# LLM_MODEL=deepseek-chat
```

模型用脚本统一下载到 `models/` 目录（每个模型一个文件夹，下完即离线可用）：

```bash
bash models/download_models.sh
# pyannote 是受限模型：先 hf auth login 并在 HF 网页接受协议；
# 也可以找已下载的同事直接拷贝 models/ 目录
```

### 3. 启动服务

```bash
python -m app.main
# 或 uvicorn app.main:app --host 0.0.0.0 --port 8000
```

打开 http://localhost:8000 ，上传音频即可检测，右侧可与智能助手对话（详见下文「使用方式」）。

模型用脚本统一下载到 `models/` 目录（每个模型一个文件夹，下完即离线可用）：

```bash
bash models/download_models.sh
```

---

## 使用方式

服务启动后有两种客户端，都是纯接口调用方，可分开使用：

### Web 仪表盘

浏览器打开 http://localhost:8000 ，三栏布局：

| 左栏 | 中栏 | 右栏 |
|---|---|---|
| 📤 上传音频（拖拽/点击） | 📄 检测报告（Markdown 渲染，风险等级彩色徽章） | 💬 智能助手对话 |
| 🗂 检测历史（3s 自动刷新进度，点击看报告） | | 可查结果、问专业问题 |

### CLI 客户端

```bash
python -m client detect /path/to/audio.flac   # 上传并等待检测报告（带进度条）
python -m client history                      # 任务列表
python -m client status <task_id>             # 查询任务进度
python -m client report <task_id>             # 查看检测报告
python -m client chat                         # 对话模式（记住会话，支持多轮）
python -m client chat -m "刚才的音频有问题吗"   # 单条提问
```

连接远程服务（服务部署在 GPU 服务器、本地只跑客户端）：

```bash
python -m client --server http://<服务器IP>:8000 detect /path/to/audio.flac
# 或设环境变量：export ADD_SERVER=http://<服务器IP>:8000
```

---

## API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/detect` | multipart 上传音频（字段名 `file`），返回 `task_id` |
| GET | `/api/tasks/{id}` | 查询任务状态/进度，`?include_report=true` 附带报告全文 |
| GET | `/api/tasks` | 最近 20 条任务历史 |
| POST | `/api/chat` | 对话 `{"message": "...", "session_id": "可选"}` |
| GET | `/health` | 健康检查 |

```bash
# 示例
curl -F "file=@LA_E_1000147.flac" http://localhost:8000/api/detect
curl "http://localhost:8000/api/tasks/<task_id>?include_report=true"
curl -X POST http://localhost:8000/api/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "刚才检测的音频有问题吗？"}'
```

## 测试

```bash
pip install pytest httpx
python -m pytest tests/   # 接口层与 Agent 循环均打桩，不需要下载模型
```

## 检测原理

- **反伪造初检**：`MelodyMachine/Deepfake-audio-detection-V2` 模型，0.5s 窗口 / 0.1s 步长滑窗，伪造概率 ≥ 0.7 的连续区间标记为可疑片段
- **ASR + 说话人分割**：FunASR Paraformer 词级时间戳 + pyannote diarization，按词中点时间对齐说话人
- **异常判定**：可疑片段的 MFCC / 梅尔能量与 LibriSpeech dev-clean 500 条真人语音统计阈值（3σ 原则）比对：
  - MFCC 均值绝对值 > 0.5，或整体标准差 > 35.0141 → 异常
  - 梅尔能量均值超出 -65.9447 ~ -43.5002 dB → 异常
- **风险等级**：低风险（无异常）/ 中等风险（1-3 个可疑片段）/ 高风险（≥3 个或占比 >10%）

## 性能（旧版流水线，ASVspoof2019 LA-dev）

| 准确率 | 精确率 | 召回率 | F1 |
|---|---|---|---|
| 0.7587 | 0.8811 | 0.8452 | 0.8628 |
