# PDF_GalaxAI

本项目是一个本地运行的“论文星系”可视化 + RAG 问答系统：

- 前端：Vite + React + Three.js（默认 `http://localhost:3000`）
- 后端：FastAPI（默认 `http://127.0.0.1:8000`）
- 向量库：Chroma（本地持久化）
- 问答模型：支持 **本地 Ollama** 与 **云端 Gemini** 切换

---

## 1. 环境要求
1234
- Node.js 18+
- Python 3.10+
- （可选）Ollama（如果使用本地模型）
- （可选）Gemini API Key（如果使用云端模型）

> 首次运行后端时，`sentence-transformers` 可能下载模型，耗时取决于网络。

---

## 2. 首次安装（Windows）

在项目根目录（例如 `D:\destop\PDF_GalaxAI`）执行：

```powershell
cd D:\destop\PDF_GalaxAI
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -U pip
py -m pip install -r .\backend\requirements.txt
npm install
```

---

## 3. 启动项目

需要两个终端：

### 终端 A：启动后端

```powershell
cd D:\destop\PDF_GalaxAI
.\.venv\Scripts\Activate.ps1
npm run backend
```

健康检查：

- `http://127.0.0.1:8000/api/health`

### 终端 B：启动前端

```powershell
cd D:\destop\PDF_GalaxAI
npm run dev
```

打开：

- `http://localhost:3000`

---

## 4. 本地模型与云端模型切换

前端右上角有切换按钮：

- `本地 Ollama`
- `云端 Gemini`

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

```powershell
copy .\backend\.env.example .\backend\.env
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

### 删除

- 在文献详情面板点击“删除该文献”
- 或调接口：`DELETE /api/papers/{paper_id}`

删除会同步清理：

- 文献元数据
- 对应 PDF 文件
- 向量库中的该文献记录

---

## 8. 常见问题

### Q1：前端提示未连接后端

- 检查后端是否在 `127.0.0.1:8000` 正常运行。

### Q2：云端 Gemini 提示未配置 `GEMINI_API_KEY`

- 先看 `GET /api/llm/status` 的 `gemini_key_configured` 是否为 `true`。
- 推荐改用 `backend/.env`，并重启后端。

### Q3：本地模型 502

- 确认 Ollama 在运行；
- 确认模型已完整下载（不是只有几 KB manifest）；
- `ollama run qwen2.5:3b` 能对话再回项目测试。

---

## 9. 安全建议

- **不要把 API Key 写进代码并提交仓库**。
- `backend/.env` 应加入 `.gitignore`（如未加入请手动添加）。
- 若 key 曾暴露，请立即在平台控制台轮换。
