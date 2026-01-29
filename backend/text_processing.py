from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import fitz
except ImportError:
    fitz = None


def safe_stem(filename: str) -> str:
    name = filename.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    if name.lower().endswith(".pdf"):
        name = name[:-4]
    return name.strip() or "Untitled"


def read_pdf_text(pdf_path: Path, max_pages: int = 5) -> str:
    if pdfplumber is not None:
        parts: list[str] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            pages = pdf.pages[:max_pages] if max_pages else pdf.pages
            for page in pages:
                parts.append(page.extract_text() or "")
        return "\n".join(parts)
    if fitz is not None:
        doc = fitz.open(str(pdf_path))
        parts = []
        page_count = min(doc.page_count, max_pages or doc.page_count)
        for i in range(page_count):
            parts.append(doc.load_page(i).get_text("text") or "")
        doc.close()
        return "\n".join(parts)
    raise RuntimeError("No PDF parser available")


def clean_text(text: str) -> str:
    t = text or ""
    t = t.replace("\x00", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[^\S\r\n]{2,}", " ", t)
    return t.strip()


def extract_title_from_text(text: str, fallback: str) -> str:
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    candidates = [l for l in lines[:20] if 8 <= len(l) <= 200]
    if candidates:
        return max(candidates, key=len)
    return fallback


def extract_abstract_block(text: str, limit: int = 1200) -> str:
    t = text or ""
    m = re.search(r"(abstract|摘要)[:：]?\s*", t, re.IGNORECASE)
    if not m:
        return t[:limit].strip()
    tail = t[m.end() :]
    stop = re.search(r"\n\s*(introduction|1\s+introduction|关键词|keywords)[:：]?\s*", tail, re.IGNORECASE)
    if stop:
        tail = tail[: stop.start()]
    return tail.strip()[:limit]


def extract_keywords_block(text: str, top_k: int = 8) -> list[str]:
    t = text or ""
    m = re.search(r"(keywords|关键词)[:：]\s*(.+)", t, re.IGNORECASE)
    if m:
        raw = m.group(2)
        parts = re.split(r"[;,，、\n]", raw)
        kws = [p.strip() for p in parts if p.strip()]
        return kws[:top_k]
    return []


def tfidf_keywords_block(text: str, top_k: int = 8) -> list[str]:
    if not text.strip():
        return []
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    X = vectorizer.fit_transform([text])
    row = X.toarray().ravel()
    feats = vectorizer.get_feature_names_out()
    idx = np.argsort(row)[::-1][:top_k]
    return [str(feats[i]) for i in idx if row[i] > 0]
