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
