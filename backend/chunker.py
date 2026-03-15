from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    paper_id: str
    chunk_id: str   # "{paper_id}_{index}"
    text: str
    index: int      # 在该论文中的顺序编号


def chunk_text(
    paper_id: str,
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[Chunk]:
    """
    将文档全文切分为有重叠的 Chunk 列表。

    策略：
    1. 滑动窗口，窗口大小 chunk_size 字符；
    2. 在窗口末尾附近优先找句子边界（.!?。！？后空格），找不到则退化到空格；
    3. 下一个窗口起点后退 overlap 字符，保留上下文衔接。
    """
    if not text or not text.strip():
        return []

    text = text.strip()
    raw_chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                raw_chunks.append(chunk)
            break

        # 在 [end-150, end] 范围内找句子边界
        search_from = max(start, end - 150)
        boundary = -1
        for m in re.finditer(r"[.!?。！？]\s", text[search_from:end]):
            boundary = search_from + m.end()

        if boundary == -1:
            # 退化：找最近的空格
            space = text.rfind(" ", start, end)
            boundary = (space + 1) if space > start else end

        chunk = text[start:boundary].strip()
        if chunk:
            raw_chunks.append(chunk)

        # 下一个起点后退 overlap，但必须比 start 前进至少 1
        next_start = boundary - overlap
        start = max(start + 1, next_start)

    return [
        Chunk(
            paper_id=paper_id,
            chunk_id=f"{paper_id}_{i}",
            text=c,
            index=i,
        )
        for i, c in enumerate(raw_chunks)
        if c
    ]
