"""
抽取式摘要（TextRank + MMR）。

用途：当一篇 PDF 没有显式 "Abstract:" / "摘要:" 段时，
自动从正文里挑几句"最能代表全文主题"的句子，拼成一段摘要。

特点：
- 使用已加载的 sentence-transformers 模型生成句向量，零新增依赖
- 用 numpy 手写 TextRank（幂迭代），避免引入 networkx
- MMR 去冗余，防止选出的几句话意思重复
- 多层兜底：句子过少 / encode 失败 / 文本过短时都能返回合理结果
"""

from __future__ import annotations

import re

import numpy as np

from text_processing import _is_formula_heavy, _looks_like_title_or_meta


_SENT_END = set(".。!?！？")


def _split_sentences(text: str) -> list[str]:
    """中英混合分句。保留句末标点。"""
    if not text:
        return []
    # 先把多余空白压成单空格（但保留换行之间的隔断通过 end punctuation 判断）
    t = re.sub(r"[ \t]+", " ", text)
    t = re.sub(r"\n{2,}", "\n\n", t)

    sents: list[str] = []
    buf: list[str] = []
    for ch in t:
        buf.append(ch)
        if ch in _SENT_END:
            s = "".join(buf).strip()
            if s:
                sents.append(s)
            buf = []
        elif ch == "\n":
            # 连续两个换行当作段落界，强制切句
            if buf[-2:] == ["\n", "\n"]:
                s = "".join(buf).strip()
                if s:
                    sents.append(s)
                buf = []
    tail = "".join(buf).strip()
    if tail:
        sents.append(tail)
    return [re.sub(r"\s+", " ", s).strip() for s in sents if s.strip()]


def _sentence_ok(s: str) -> bool:
    """过滤不适合作摘要候选的句子。"""
    if not s:
        return False
    n = len(s)
    if n < 15 or n > 300:
        return False
    if _looks_like_title_or_meta(s):
        return False
    if _is_formula_heavy(s):
        return False
    readable = len(re.findall(r"[A-Za-z\u4e00-\u9fff]", s))
    if readable / max(1, n) < 0.5:
        return False
    return True


def _textrank(sim: np.ndarray, damping: float = 0.85,
              max_iter: int = 60, tol: float = 1e-4) -> np.ndarray:
    """在句间相似度矩阵上跑 TextRank，返回每句中心度得分。"""
    n = sim.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    if n == 1:
        return np.ones(1, dtype=np.float32)

    m = np.clip(sim.astype(np.float64, copy=True), 0.0, 1.0)
    np.fill_diagonal(m, 0.0)

    row_sum = m.sum(axis=1, keepdims=True)
    zero_rows = (row_sum == 0).flatten()
    row_sum[row_sum == 0] = 1.0
    transition = m / row_sum   # 行归一化

    scores = np.ones(n, dtype=np.float64) / n
    for _ in range(max_iter):
        new_scores = (1.0 - damping) / n + damping * (transition.T @ scores)
        if zero_rows.any():
            # 孤立节点均匀分配，避免质量漏出
            new_scores += damping * scores[zero_rows].sum() / n
        diff = float(np.abs(new_scores - scores).sum())
        scores = new_scores
        if diff < tol:
            break
    # 返回为 float32
    return scores.astype(np.float32)


def _mmr_select(scores: np.ndarray, sim: np.ndarray,
                k: int, lam: float = 0.7) -> list[int]:
    """
    MMR 选 k 句：relevance = TextRank 得分；redundancy = 与已选的最大相似度。
    返回句子索引列表（未排序）。
    """
    n = int(len(scores))
    if n == 0 or k <= 0:
        return []
    if k >= n:
        return list(range(n))

    selected: list[int] = []
    remaining = set(range(n))
    while remaining and len(selected) < k:
        best_idx, best_val = -1, -1e18
        for i in remaining:
            rel = float(scores[i])
            if selected:
                redundancy = max(float(sim[i, j]) for j in selected)
            else:
                redundancy = 0.0
            val = lam * rel - (1.0 - lam) * redundancy
            if val > best_val:
                best_val = val
                best_idx = i
        if best_idx < 0:
            break
        selected.append(best_idx)
        remaining.discard(best_idx)
    return selected


def extractive_summary(
    text: str,
    model,
    max_sentences: int = 4,
    max_chars: int = 420,
) -> str:
    """
    从 `text` 中抽取 max_sentences 句组成摘要，总长度不超过 max_chars。
    `model` 必须提供 `.encode(list[str], normalize_embeddings=True)` 接口
    （sentence-transformers 默认就是）。若 model 为 None 或 encode 失败，
    会退化为"前几句合格文本"。
    """
    if not text:
        return ""

    # 只从前 60% 抽摘要（防止命中参考文献/致谢）
    head_limit = max(1500, int(len(text) * 0.6))
    head = text[: head_limit]
    sents = [s for s in _split_sentences(head) if _sentence_ok(s)]
    if not sents:
        return ""
    if len(sents) == 1:
        return sents[0][:max_chars]

    # 句子过多时截断（控制 encode 成本）
    if len(sents) > 80:
        sents = sents[:80]

    embs = None
    if model is not None:
        try:
            embs = model.encode(sents, normalize_embeddings=True)
            embs = np.asarray(embs, dtype=np.float32)
        except Exception as e:
            print(f"[summarizer] encode failed, fall back to head: {e}")
            embs = None

    if embs is None or embs.size == 0:
        # 退化：前几句合格文本拼接
        picked, total = [], 0
        for s in sents:
            if total + len(s) > max_chars and picked:
                break
            picked.append(s)
            total += len(s) + 1
            if len(picked) >= max_sentences:
                break
        return " ".join(picked)[:max_chars].strip()

    sim = embs @ embs.T
    # 数值稳定：裁到 [0, 1]
    sim = np.clip(sim, 0.0, 1.0)

    scores = _textrank(sim)
    # 句子位置先验：越靠前给一点小加成（摘要句常在开头），避免纯 PageRank
    # 把"开篇优先"融进来让总结更自然
    n = len(sents)
    position_bonus = np.linspace(0.05, 0.0, n, dtype=np.float32)
    final_scores = scores + position_bonus

    k = min(max_sentences, n)
    selected = _mmr_select(final_scores, sim, k=k, lam=0.72)
    selected.sort()   # 保留原文顺序

    picked: list[str] = []
    total = 0
    for i in selected:
        s = sents[i]
        if total + len(s) > max_chars and picked:
            break
        picked.append(s)
        total += len(s) + 1
    if not picked:
        # 极端兜底
        top_idx = int(np.argmax(final_scores))
        return sents[top_idx][:max_chars]
    return " ".join(picked)[:max_chars].strip()
