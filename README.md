# PDF_GalaxAI

一个本地运行的“论文星系”可视化 Demo：
- 前端：Vite + React + Three.js（默认 http://localhost:3000）
- 后端：FastAPI（默认 http://127.0.0.1:8000），负责 PDF 入库、抽取文本、生成向量与聚类、提供可视化数据接口

## 目录结构

```text
PDF_GalaxAI/
├─ backend/                         # FastAPI 后端
│  ├─ api.py                        # API 路由与 CORS
│  ├─ main.py                       # uvicorn 入口：app
│  ├─ store.py                      # PDF 入库/向量化/聚类/可视化数据
│  ├─ config.py                     # 数据目录与环境变量
│  ├─ text_processing.py            # PDF 文本抽取与关键词处理
│  ├─ clustering.py                 # 降维/颜色/布局等
│  ├─ server.py
│  └─ requirements.txt
├─ frontend/                        # React 源码
│  └─ src/
│     ├─ api/
│     │  └─ client.ts
│     ├─ rendering/
│     │  └─ GalaxyRenderer.tsx
│     ├─ types/
│     │  └─ scholar.ts
│     ├─ ui/
│     │  └─ App.tsx
│     └─ main.tsx
├─ index.html
├─ index.tsx                        # Vite 入口（转到 frontend/src/main）
├─ index.css
├─ package.json
├─ tsconfig.json
├─ vite.config.ts                   # dev 端口/代理（/api, /files -> 8000）
├─ tailwind.config.cjs
├─ postcss.config.cjs
├─ package-lock.json
├─ .gitignore
└─ README.md
```

## 环境要求

- Node.js：建议 18+（用于 Vite/React）
- Python：建议 3.10+（用于 FastAPI 与数据处理）

首次运行后端时，`sentence-transformers` 可能会下载默认模型（取决于网络环境）。

## 快速开始（Windows）

在项目根目录 `d:\PDF_GalaxAI` 打开两个终端分别启动后端与前端。

### 1) 启动后端（FastAPI）

```powershell
cd d:\PDF_GalaxAI
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -U pip
py -m pip install -r .\backend\requirements.txt

# 启动后端（等价于：cd backend && py -m uvicorn main:app --reload --host 127.0.0.1 --port 8000）
npm run backend
```
kkkk
健康检查：
- http://127.0.0.1:8000/api/health

### 2) 启动前端（Vite）

```powershell
cd d:\PDF_GalaxAI
npm install
npm run dev
```

打开页面：
- http://localhost:3000

前端已配置代理：
- `/api/*` 与 `/files/*` -> `http://127.0.0.1:8000`

## 快速开始（macOS）

### macOS（Homebrew）

```bash
# 首次安装依赖（已安装可跳过）
brew --version >/dev/null 2>&1 || /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python node

cd /path/to/PDF_GalaxAI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r ./backend/requirements.txt

cd backend
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

另开终端启动前端：

```bash
cd /path/to/PDF_GalaxAI
npm install
npm run dev
```

## 数据与导入 PDF

后端默认数据目录：
- `backend/data/`
  - `backend/data/inbox/`：放入待导入的 `.pdf`
  - `backend/data/files/`：后端保存的 PDF（按 id 命名）

导入方式：
- 把 PDF 放到 `backend/data/inbox/` 后，启动后端会自动扫描一次
- 或者在后端运行期间调用扫描接口：`POST /api/scan`
- 或者通过上传接口：`POST /api/upload`（form-data，字段名 `file`）

## 常用环境变量（可选）

在启动后端前设置：
- `SCHOLAR_DATA_DIR`：自定义数据目录（默认 `backend/data`）
- `SCHOLAR_INBOX_DIR`：自定义 inbox 目录（默认 `SCHOLAR_DATA_DIR/inbox`）
- `SCHOLAR_ST_MODEL`：Sentence-Transformers 模型名（默认 `all-MiniLM-L6-v2`）
- `SCHOLAR_OFFLINE=1`：强制离线加载模型（不会下载；需要本地已缓存模型）
