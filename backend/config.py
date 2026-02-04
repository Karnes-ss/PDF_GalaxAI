from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("SCHOLAR_DATA_DIR") or (ROOT / "data")).resolve()
FILES_DIR = DATA_DIR / "files"
PAPERS_JSON = DATA_DIR / "papers.json"
INBOX_DIR = Path(os.getenv("SCHOLAR_INBOX_DIR") or (DATA_DIR / "inbox")).resolve()

DATA_DIR.mkdir(parents=True, exist_ok=True)
FILES_DIR.mkdir(parents=True, exist_ok=True)
INBOX_DIR.mkdir(parents=True, exist_ok=True)

# --- 讯飞星火 WebSocket 配置 (X1.5 深度推理) ---
# 鉴权信息 (来自你的图片)
SPARK_APP_ID = os.getenv("SPARK_APP_ID", "85daaba4")
SPARK_API_SECRET = os.getenv("SPARK_API_SECRET", "Yzk0YWQzM2NmYzJlNjczNjNhYTBkN2lz")
SPARK_API_KEY = os.getenv("SPARK_API_KEY", "bb7100fbef7dc3b46ce64ee3ca4da562")

# 接口地址 (WebSocket)
SPARK_WS_URL = "wss://spark-api.xf-yun.com/v1/x1"

# 模型 Domain (必须是 spark-x)
SPARK_DOMAIN = "spark-x"