# PDF_GalaxAI

本项目是一个本地运行的“论文星系”可视化 + RAG 问答系统：

- 前端：Vite + React + Three.js（默认 `http://localhost:3000`）
- 后端：FastAPI（默认 `http://127.0.0.1:8000`）
- 向量库：Chroma（本地持久化）
- 问答模型：支持 **本地 Ollama**、**云端 Gemini**，以及“OpenAI 兼容接口”（可接 DeepSeek / Kimi / 智谱 / Groq / LMStudio / 自建网关等）

本 README 同时作为**用户手册**，包含安装、启动、核心功能使用方式与排错指南。

## 0. 本文结构（按需跳转）

| 章节 | 内容 |
| --- | --- |
| 第 1 节 | 环境要求、快速上手、**新手建议流程** |
| 第 2～3 节 | 首次安装、启动与健康检查 |
| 第 4～6 节 | 模型切换、Ollama、Gemini 与 `.env` |
| 第 7 节 | 上传/删除、**提问模式**、**向量索引与重建** |
| 第 8 节 | OCR、Tesseract、关键词提取 |
| 第 9 节 | 深度阅读、**截图提问** |
| 第 10 节 | **常见问题与故障排查**（含现象速查表） |
| 第 11 节 | 安全建议 |
| 第 12 节 | **MinerU**（可选） |

界面里还可通过 **模型管理** 添加 OpenAI 兼容 API、配置 **网络模式（直连/代理）**；具体字段说明见应用内提示与 `backend/.env.example`。

---

## 1. 环境要求

- Node.js 18+
- Python 3.10+
- （可选）Ollama（如果使用本地模型）
- （可选）Gemini API Key（如果使用云端模型）
- （可选）Tesseract OCR（如果希望对扫描版 / 乱码 PDF 做 OCR 识别，见第 8 节）
- （可选）MinerU（如果希望更好地解析理科论文的公式/表格/版面，见第 12 节）

> 首次运行后端时，`sentence-transformers` 可能下载模型，耗时取决于网络。

---

## 1.1 一分钟快速上手（已安装好依赖的情况下）

需要两个终端：

```bash
# 终端 A（后端）
npm run backend          # Windows 常用
npm run backend:py3      # macOS/Linux 常用

# 终端 B（前端）
npm run dev
```

打开 `http://localhost:3000`，上传一篇 PDF 后即可开始问答。

---

## 1.2 新手建议流程（第一次使用推荐顺序）

按下面顺序走一遍，能最快验证「上传 → 索引 → 问答 → 深度阅读」整条链路是否正常。

1. **准备配置**：复制 `backend/.env.example` 为 `backend/.env`，至少确认 `GEMINI_API_KEY`（若要用云端）或本地 Ollama 已安装并拉好模型。
2. **启动服务**：终端 A 跑后端、终端 B 跑前端（见第 1.1 / 第 3 节）。
3. **健康检查**：浏览器打开 `http://127.0.0.1:8000/api/health`，确认后端在线。
4. **上传 1～2 篇 PDF**：用界面上的上传按钮即可；等待处理完成（星图出现节点）。
5. **向量索引（重要）**：若右侧出现 **黄色 embedding 提示**，说明换过 embedding 模型或库内向量与当前模型不一致，点一次 **「重建全库向量」** 后再测 RAG（见第 7.2 节）。日常只上传新文献时**不需要**每次重建。
6. **主对话试跑**：
   - 模式选 **自适应** 或 **文献问答**，问一个与你库里内容强相关的问题，看是否有合理回答与引用。
   - 再问一个与文献无关的闲聊，切到 **通用对话**，确认不会乱挂文献引用。
7. **深度阅读**：打开某篇文献的深度阅读，在右侧 **阅读对话** 里问与当前论文相关的问题；需要讲公式/图表时，用 **截图 + 粘贴**（见第 9 节），模型需选 **Gemini / GPT-4o** 等多模态。
8. **（可选）理科 PDF 加强**：对单篇论文在详情里点 **重新识别 → MinerU**，再重复第 6～7 步对比公式与检索效果（见第 12 节）。

---

## 2. 首次安装

### 2.1 Windows

在项目根目录（例如 `D:\destop\PDF_GalaxAI`）执行：

```powershell
cd D:\destop\PDF_GalaxAI
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -U pip
py -m pip install -r .\backend\requirements.txt
npm install
```

### 2.2 macOS

前置要求：Node.js 18+、Python 3.10+（推荐用 Homebrew 装）。

```bash
# 若尚未安装 Python / Node：
brew install python@3.11 node

# 克隆/进入项目目录，然后：
cd /path/to/PDF_GalaxAI
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r backend/requirements.txt
npm install

# OCR（可选，但强烈推荐）
brew install tesseract tesseract-lang
which tesseract   # 查路径，用 brew 一般在 PATH 里不用填 .env
```

> M 系列 Mac（Apple Silicon）完全可用。`sentence-transformers` / `pymupdf` /
> `chromadb` 都有原生 arm64 wheel，不用 Rosetta。

### 2.3 Linux

和 macOS 几乎一致，把 brew 换成你发行版的包管理器即可：

```bash
# Debian/Ubuntu：
sudo apt install python3.11 python3.11-venv nodejs npm tesseract-ocr tesseract-ocr-chi-sim
# Arch：
sudo pacman -S python nodejs npm tesseract tesseract-data-chi_sim
```

余下步骤同 macOS 2.2 节。

---

## 3. 启动项目

需要两个终端。

### 终端 A：启动后端

Windows（PowerShell）：

```powershell
cd D:\destop\PDF_GalaxAI
.\.venv\Scripts\Activate.ps1
npm run backend
```

macOS / Linux（bash/zsh）：

```bash
cd /path/to/PDF_GalaxAI
source .venv/bin/activate
npm run backend:py3     # 注意：Mac/Linux 用 python3，要走这个别名
```

> `npm run backend` 调的是 `python` 命令，Windows 下一般能跑，但 macOS 上通常只有
> `python3`。所以 Mac/Linux 用 `npm run backend:py3`，两者只有可执行文件名不同。

健康检查：

- `http://127.0.0.1:8000/api/health`

### 终端 B：启动前端

```bash
npm run dev
```

打开：

- `http://localhost:3000`

---

## 4. 本地模型与云端模型切换

前端右上角有切换按钮：

- `本地 Ollama`
- `云端 Gemini`
- `自定义模型`（OpenAI 兼容：DeepSeek / Kimi / 智谱 / Groq / LMStudio / 自建网关等）

后端 `POST /api/query` 也支持传入 `provider`：

```json
{
  "question": "MLP-2602 这篇文章讲了什么？",
  "provider": "gemini"
}
```

响应会包含 `provider_used`，用于确认实际走了哪个提供方。

---

## 5. Ollama（本地模型）准备

### 安装并拉取模型

```powershell
ollama pull qwen2.5:3b
ollama run qwen2.5:3b
```

如果 `ollama run` 能正常对话，说明本地模型可用。

可通过环境变量改默认模型名：

- `SCHOLAR_OLLAMA_MODEL=qwen2.5:3b`

---

## 6. Gemini（云端模型）配置

### 推荐方式：使用 `backend/.env`

- 复制模板：

Windows（PowerShell）：

```powershell
copy .\backend\.env.example .\backend\.env
```

macOS / Linux：

```bash
cp backend/.env.example backend/.env
```

- 编辑 `backend/.env`，填入你的 key：

```env
SCHOLAR_LLM_PROVIDER=local
SCHOLAR_OLLAMA_MODEL=qwen2.5:3b
SCHOLAR_GEMINI_MODEL=gemini-2.5-flash
GEMINI_API_KEY=your_real_key
```

- 重启后端。

### 可选方式：当前终端临时设置

```powershell
$env:GEMINI_API_KEY="your_real_key"
$env:SCHOLAR_GEMINI_MODEL="gemini-2.5-flash"
npm run backend
```

> 临时环境变量只对当前终端有效，开新终端后需要重新设置。

### 检查 Gemini 是否生效

- `GET /api/llm/status`
- 关注字段：`gemini_key_configured: true`

---

## 7. 上传 / 删除文献

### 上传

- UI 上传按钮（推荐）
- 或放入 `backend/data/inbox/` 后重启后端
- 或调接口：`POST /api/upload` / `POST /api/papers/upload`
- `inbox` 导入成功后，源文件会自动移动到 `backend/data/inbox/_processed/`，避免重启重复导入

### 删除

- 在文献详情面板点击“删除文献”，可选两种模式：
  - **彻底删除（推荐）**：删除知识库记录 + `inbox`/`inbox/_processed` 同名源文件（不会回流）
  - **仅从知识库移除**：删除知识库记录，但保留 `inbox` 源文件（可重新导入）
- 或调接口：`DELETE /api/papers/{paper_id}`

删除会同步清理：

- 文献元数据
- 对应 PDF 文件
- 向量库中的该文献记录
- `inbox` / `inbox/_processed` 里同名源文件（仅“彻底删除”模式）

---

## 7.1 提问模式说明（很重要）

你可以在右侧对话面板顶部切换三种模式：

- **自适应（auto）**：默认。系统会根据“检索是否命中/命中质量”决定是走纯聊天还是走 RAG。
- **文献问答（rag）**：强制基于文献库回答，尽力给出溯源引用。
- **通用对话（chat）**：强制纯聊天，不检索不引用。适合问与文献无关的问题。

当你把某篇论文“加入对话”或在阅读器里开启“锁定本篇”时，会更倾向于文献问答。

---

## 7.2 向量索引（Embedding）与“重建全库向量”

### 7.2.1 什么是 embedding / 向量索引？

系统会把每篇论文的摘要/关键词，以及每篇论文切分出来的文本片段（chunk）编码成向量，写入本地向量库（Chroma），用于语义检索（RAG）。

### 7.2.2 什么时候需要“重建全库向量”？

**不需要每次上传都重建。**

- **上传新论文**：只会对新论文做增量索引（论文级向量 + chunk 向量）。
- **需要重建全库**：通常只有在你更换 embedding 模型（例如从 `all-MiniLM-L6-v2` 换到 `BAAI/bge-m3`）时。

原因：不同 embedding 模型的向量维度/语义空间不兼容。换模型后老向量会导致检索明显变差，需要重建一次统一到新模型。

### 7.2.3 在 UI 里怎么做？

右侧对话面板顶部有一个显示当前 embedding 模型的小按钮：

- 黄色：提示“模型已变更，需要重建”
- 灰色：表示一致

点击后按提示执行“重建全库向量”即可（这是偶发的运维动作，不是日常操作）。

---

## 8. PDF 识别（OCR）与关键词提取

### 8.1 OCR 模式

后端支持三种 OCR 策略，通过 `backend/.env` 里的 `SCHOLAR_OCR_ENABLED` 切换：

| 模式 | 行为 | 适用场景 |
| --- | --- | --- |
| `off` | 只用 PDF 原生文本层 | 纯文本版、全部是正常文字 PDF |
| `auto` | 逐页判断：native 质量差才 OCR | 大多数场景，速度/质量平衡（推荐） |
| `force` | 每一页都跑 OCR | 大量扫描版、或 native 文本乱码严重 |

相关环境变量（见 `backend/.env.example`）：

```env
SCHOLAR_OCR_ENABLED=force
SCHOLAR_OCR_LANG=chi_sim+eng
SCHOLAR_OCR_ZOOM=2.0
SCHOLAR_OCR_MIN_QUALITY=0.55
# 如果 tesseract 不在 PATH，可以显式指定路径：
# SCHOLAR_TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

> `force` 模式比 `auto` 慢很多（每页都渲染成图像再跑 OCR）。大文档建议默认 `auto`。

### 8.2 安装 Tesseract（OCR 引擎）

我们项目只依赖 `pytesseract`（Python 绑定）；**真正的 OCR 引擎要单独装**。没装会自动降级为 native 提取（不会报错）。

#### Windows

1. 下载安装包（选带中文简体 `chi_sim` 的构建，如 UB-Mannheim 版本）：
   <https://github.com/UB-Mannheim/tesseract/wiki>
2. 安装时勾选 **Chinese (Simplified)** / 需要的其他语种。
3. 安装完成后：
   - 把安装目录（默认 `C:\Program Files\Tesseract-OCR\`）加入系统环境变量 `PATH`；
   - 或在 `backend/.env` 里显式设置：

     ```env
     SCHOLAR_TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
     ```

4. 验证：新开一个终端输入 `tesseract --version`，能输出版本号即可。

#### macOS

```bash
brew install tesseract tesseract-lang
```

`tesseract-lang` 会一次性装上中文等常用语种。

验证：

```bash
tesseract --version
tesseract --list-langs    # 看 chi_sim、eng 是否都在
```

#### Linux（Debian/Ubuntu 示例）

```bash
sudo apt update
sudo apt install -y tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-eng
```

### 8.3 关键词提取

- 如果 PDF 自带 `Keywords:` 或 `关键词:` 段落，会**原样使用**（标记为 `declared`）。
- 否则用**全库 TF-IDF + jieba + MMR** 自动抽取（标记为 `auto`）：
  - 跨所有已上传文献统计 IDF，比单文档 TF-IDF 更能反映区分度；
  - 过滤中英文停用词、数字、乱码 token；
  - 标题 / 摘要命中会加权；
  - MMR 多样化，避免近义词堆叠。
- 每次上传 / 删除后会自动重算所有 `auto` 关键词；`declared` 关键词不会被覆盖。

---

## 9. 阅读器（深度阅读）与截图提问（公式/图表推荐用法）

### 9.1 阅读器对话（Reader Chat）

在深度阅读模式中，右侧会出现独立的阅读对话面板：

- 可以在面板顶部切换模型（和主对话一致）
- **锁定本篇**：开启后会优先基于当前论文回答（适合细读该论文）

### 9.2 截图提问（Vision → 描述 → RAG → 回答）

当你要问“这条公式怎么理解/推导”“这张图表在讲什么”时，纯文本抽取常常不可靠；截图提问能显著提高准确率。

使用方式（Windows / macOS 通用）：

1. 用系统截图工具截一块公式/图表区域（Windows：Win+Shift+S，macOS：Shift+Cmd+4）
2. 点击阅读器对话输入框
3. 直接粘贴（Ctrl+V / Cmd+V），或点击输入框旁的 📷 按钮上传图片
4. 输入问题并发送

注意：

- 截图提问要求模型支持看图（例如 Gemini / GPT-4o 等）。纯文本模型无法处理图片，这是模型能力限制。

---

## 10. 常见问题与故障排查

下面按「现象 → 可能原因 → 建议处理」列出；仍解决不了时，把**后端终端最后 30 行日志**与**浏览器控制台报错**一并保存，便于定位。

### 10.1 现象速查表

| 现象 | 可能原因 | 建议处理 |
| --- | --- | --- |
| 前端提示连不上后端 | 后端未启动或端口不是 8000 | 确认 `http://127.0.0.1:8000/api/health` 可访问；检查防火墙 |
| RAG 回答很空、乱引用 | 未重建向量 / embedding 模型与库不一致 | 点右侧 **embedding 状态按钮**，按提示 **重建全库向量**（见第 7.2 节） |
| 首次启动后端很慢 | 正在下载 `BAAI/bge-m3` 等模型 | 等待完成；国内网络可设 `HF_ENDPOINT=https://hf-mirror.com` 后重试 |
| 截图发送后报错「不支持看图」 | 当前选的是纯文本模型 | 切换到 **Gemini / GPT-4o** 等多模态模型 |
| MinerU 菜单显示「未安装」 | 系统找不到 `mineru` 或 `SCHOLAR_MINERU_CMD` 填错 | 按第 12 节安装并配置路径，重启后端 |
| MinerU 很慢或超时 | CPU 解析大 PDF | 只对关键论文使用；或增大 `SCHOLAR_MINERU_TIMEOUT` |
| Gemini 连不上 / 超时 | 网络、代理、地区限制 | 检查 VPN；在模型配置里试 **直连 / 系统代理 / 自定义代理** |
| 自定义 OpenAI 兼容模型 401/403 | Key 或 base_url 错误 | 在「模型管理」里核对；用「测试连接」验证 |
| DeepSeek 返回 402 | 账户余额不足 | 在 DeepSeek 控制台充值或换模型名 |
| 安装依赖装错到全局 Python | 未激活 `.venv` | 先 `.\.venv\Scripts\Activate.ps1`（Windows）或 `source .venv/bin/activate`（macOS/Linux），再 `pip install` |

### Q1：前端提示未连接后端

- 检查后端是否在 `127.0.0.1:8000` 正常运行。

### Q2：云端 Gemini 提示未配置 `GEMINI_API_KEY`

- 先看 `GET /api/llm/status` 的 `gemini_key_configured` 是否为 `true`。
- 推荐改用 `backend/.env`，并重启后端。

### Q3：本地模型 502

- 确认 Ollama 在运行；
- 确认模型已完整下载（不是只有几 KB manifest）；
- `ollama run qwen2.5:3b` 能对话再回项目测试。

### Q4：预览 / 问答看到乱码

- 多半是 PDF 字体没有提供正确的 Unicode 映射（中文 PDF 尤其常见）。
- 方案 A（推荐）：在 `backend/.env` 里把 `SCHOLAR_OCR_ENABLED=auto`，低质量页面会自动用 OCR 重抽。
- 方案 B：`SCHOLAR_OCR_ENABLED=force` 全 OCR（最稳但较慢）。
- 若日志里看到 `[ocr] Tesseract unavailable`，说明 OCR 引擎没装，照第 8.2 节装好即可。

### Q5：关键词看起来和文献无关

- 上传阶段若 PDF 文本乱码严重，会连带污染关键词；先按 Q4 打开 OCR。
- 之后可以调用 `POST /api/analyze` 或重启一次后端，触发全库关键词刷新。
- 仅对 `_keywords_source=auto` 的论文生效；论文自带 `Keywords:` 的关键词不会被改。

---

## 11. 安全建议

- **不要把 API Key 写进代码并提交仓库**。
- `backend/.env` 应加入 `.gitignore`（如未加入请手动添加）。
- 若 key 曾暴露，请立即在平台控制台轮换。

---

## 12. MinerU（可选，理科论文公式/表格强烈推荐）

### 12.1 MinerU 是什么？

MinerU 是一个面向学术 PDF 的解析器，擅长处理：

- 公式（更容易还原成 LaTeX/可读文本）
- 表格（结构化输出）
- 多栏版式/标题层级（减少阅读顺序错乱）

在本项目中，它作为“可选解析器”出现在论文详情的“重新识别”菜单中，你可以只对某一篇论文启用它（无需全库启用）。

### 12.2 安装方式（建议单独环境，避免依赖冲突）

先决条件（Windows）：

- 需要先安装 **Miniconda** 或 **Anaconda**，这样才会有 `conda` 命令和
  `Anaconda Prompt / Miniconda Prompt`。
- 若未安装，可先装 Miniconda（更轻量，推荐）：
  <https://docs.conda.io/en/latest/miniconda.html>

安装完成后，请在 **Anaconda Prompt / Miniconda Prompt** 里执行以下命令。

Windows（conda 示例）：

```powershell
conda create -n mineru python=3.10 -y
conda activate mineru
pip install -U "mineru[core]"
mineru --help
where mineru
```

把 `where mineru` 打印出来的 `mineru.exe` 路径填到 `backend/.env`：

```env
SCHOLAR_MINERU_CMD=C:\ProgramData\miniconda3\envs\mineru\Scripts\mineru.exe
```

macOS/Linux（conda 示例）：

```bash
conda create -n mineru python=3.10 -y
conda activate mineru
pip install -U "mineru[core]"
mineru --help
which mineru
```

把 `which mineru` 的路径填到 `backend/.env` 的 `SCHOLAR_MINERU_CMD`（或不填，让系统 PATH 自行找到）。

不使用 conda 也可以，但不推荐（更容易与项目主环境依赖冲突）：

- 方案 A：使用系统 Python + `venv` 单独建 `mineru-venv`，再安装 `mineru[core]`；
- 方案 B：直接装到项目 `.venv`（仅在你非常确认依赖兼容时使用）。

### 12.3 在 UI 里如何使用？

打开某篇论文详情 → `重新识别` → 选择 `MinerU（公式/表格最优）`。

性能提示：

- 有 NVIDIA CUDA：通常每页 1~3 秒
- 只有 CPU：可能 20~60 秒/页（建议只对关键论文使用）
