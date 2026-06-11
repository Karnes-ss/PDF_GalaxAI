from __future__ import annotations

import re
from dataclasses import dataclass

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    _LC_AVAILABLE = True
except ImportError:
    _LC_AVAILABLE = False


@dataclass
class Chunk:
    paper_id: str
    chunk_id: str   # "{paper_id}_{index}"
    text: str
    index: int      # 在该论文中的顺序编号


# 中英文混合的递归分隔符优先级：先按段落，再按句子，再按词/字
_SEPARATORS = ["。", "！", "？", ".", "!", "?", "\n\n", "\n", " ", ""]


def chunk_text(
    paper_id: str,
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[Chunk]:
    """
    将文档全文切分为有重叠的 Chunk 列表。
    优先使用 LangChain RecursiveCharacterTextSplitter（中英文混合友好）；
    若未安装 langchain-text-splitters，则退回内置滑动窗口实现。
    """
    if not text or not text.strip():
        return []

    if _LC_AVAILABLE:
        return _chunk_langchain(paper_id, text.strip(), chunk_size, overlap)
    return _chunk_builtin(paper_id, text.strip(), chunk_size, overlap)


def _chunk_langchain(
    paper_id: str, text: str, chunk_size: int, overlap: int
) -> list[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=_SEPARATORS,
    )
    raw = splitter.split_text(text)
    return [
        Chunk(paper_id=paper_id, chunk_id=f"{paper_id}_{i}", text=t.strip(), index=i)
        for i, t in enumerate(raw)
        if t.strip()
    ]


def _chunk_builtin(
    paper_id: str, text: str, chunk_size: int, overlap: int
) -> list[Chunk]:
    """内置滑动窗口 + 句子边界回溯（langchain 未安装时的兜底）。"""
    raw_chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                raw_chunks.append(chunk)
            break

        search_from = max(start, end - 150)
        boundary = -1
        for m in re.finditer(r"[.!?。！？]\s", text[search_from:end]):
            boundary = search_from + m.end()

        if boundary == -1:
            space = text.rfind(" ", start, end)
            boundary = (space + 1) if space > start else end

        chunk = text[start:boundary].strip()
        if chunk:
            raw_chunks.append(chunk)

        next_start = boundary - overlap
        start = max(start + 1, next_start)

    return [
        Chunk(paper_id=paper_id, chunk_id=f"{paper_id}_{i}", text=c, index=i)
        for i, c in enumerate(raw_chunks)
        if c
    ]
