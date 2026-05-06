from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _load_local_env_file() -> None:
    """
    轻量读取 backend/.env（不依赖 python-dotenv）。
    规则：
    - 支持 KEY=VALUE
    - 跳过空行与 # 注释
    - 仅在当前进程里该变量不存在时才写入（不覆盖系统环境变量）
    - 支持去掉首尾引号 "..." / '...'
    """
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            if not key:
                continue
            if (val.startswith('"') and val.endswith('"')) or (
                val.startswith("'") and val.endswith("'")
            ):
                val = val[1:-1]
            os.environ.setdefault(key, val)
    except Exception as e:
        print(f"[config] Warning: failed to load .env: {e}")


_load_local_env_file()

DATA_DIR = Path(os.getenv("SCHOLAR_DATA_DIR") or (ROOT / "data")).resolve()
FILES_DIR = DATA_DIR / "files"
PAPERS_JSON = DATA_DIR / "papers.json"
INBOX_DIR = Path(os.getenv("SCHOLAR_INBOX_DIR") or (DATA_DIR / "inbox")).resolve()

CHROMA_DIR = DATA_DIR / "chroma"
CUSTOM_MODELS_JSON = DATA_DIR / "custom_models.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FILES_DIR.mkdir(parents=True, exist_ok=True)
INBOX_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# --- 讯飞星火 WebSocket 配置 (X1.5 深度推理) ---
# 鉴权信息 (来自你的图片)
SPARK_APP_ID = os.getenv("SPARK_APP_ID", "85daaba4")
SPARK_API_SECRET = os.getenv("SPARK_API_SECRET", "Yzk0YWQzM2NmYzJlNjczNjNhYTBkN2lz")
SPARK_API_KEY = os.getenv("SPARK_API_KEY", "bb7100fbef7dc3b46ce64ee3ca4da562")

# 接口地址 (WebSocket)
SPARK_WS_URL = "wss://spark-api.xf-yun.com/v1/x1"

# 模型 Domain (必须是 spark-x)
SPARK_DOMAIN = "spark-x"