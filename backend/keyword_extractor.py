from __future__ import annotations

"""
全库 TF-IDF + MMR 关键词提取。

与旧版 tfidf_keywords_block 的区别：
  - 在整个文献库上拟合 TF-IDF（单文档 TF-IDF 的 IDF 没有意义）
  - 中英文混合 tokenize（中文优先 jieba；没装则回退字符 n-gram）
  - 过滤停用词 / 数字 / 过短 token / 乱码占比高的 token
  - 用 MMR 多样化，避免 "neural network / neural networks / network" 同义堆叠
  - 文献标题和摘要里的 token 获得加权
"""

import re
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from stopwords import STOPWORDS_EN, STOPWORDS_ZH

try:
    import jieba
    jieba.setLogLevel(20)  # 禁掉 jieba 的 INFO 日志
except ImportError:
    jieba = None


_ZH_ONLY = re.compile(r"^[\u4e00-\u9fff]+$")
_EN_WORD = re.compile(r"[A-Za-z][A-Za-z\-]{2,}")
_ZH_RUN = re.compile(r"[\u4e00-\u9fff]+")
_NUMERIC_LIKE = re.compile(r"^[\d\W_]+$")


def _looks_noisy(token: str) -> bool:
    """过滤乱码 / 疑似 PDF 编码错位 token。"""
    if not token:
        return True
    if "?" in token or "�" in token:
        return True
    if _NUMERIC_LIKE.match(token):
        return True
    return False


def _mixed_tokenize(text: str) -> list[str]:
    """中英文混合分词。中文有 jieba 用 jieba，否则退化到字符 bi/tri-gram。"""
    if not text:
        return []

    tokens: list[str] = []
    text = text.replace("\n", " ")

    if jieba is not None:
        for seg in jieba.cut(text):
            seg = seg.strip()
            if not seg or _looks_noisy(seg):
                continue
            if _ZH_ONLY.match(seg):
                if 2 <= len(seg) <= 6 and seg not in STOPWORDS_ZH:
                    tokens.append(seg)
                continue
            # 英文分支：jieba 切出来的英文保留原形
            for m in _EN_WORD.findall(seg):
                w = m.lower()
                if len(w) >= 3 and w not in STOPWORDS_EN:
                    tokens.append(w)
    else:
        for m in _ZH_RUN.findall(text):
            # 退化策略：字符 2-gram / 3-gram
            for n in (2, 3):
                for i in range(len(m) - n + 1):
                    w = m[i: i + n]
                    if w not in STOPWORDS_ZH and not _looks_noisy(w):
                        tokens.append(w)
        for m in _EN_WORD.findall(text):
            w = m.lower()
            if len(w) >= 3 and w not in STOPWORDS_EN:
                tokens.append(w)

    return tokens


def _jaccard_chars(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.75
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


class GlobalKeywordExtractor:
    """
    全库 TF-IDF 关键词提取器。

    使用方式：
        kw = GlobalKeywordExtractor()
        kw.add("paper1", cleaned_text1)
        kw.add("paper2", cleaned_text2)
        keywords = kw.extract("paper1", top_k=8, title=..., abstract=...)

    语料发生变化时，自动标记脏状态，在下一次 extract 时懒重建 TF-IDF。
    """

    def __init__(self) -> None:
        self._corpus: dict[str, str] = {}
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None
        self._doc_ids: list[str] = []
        self._dirty: bool = True

    # ------------------------- 语料维护 ------------------------- #

    def add(self, doc_id: str, text: str) -> None:
        self._corpus[doc_id] = text or ""
        self._dirty = True

    def remove(self, doc_id: str) -> None:
        if self._corpus.pop(doc_id, None) is not None:
            self._dirty = True

    def size(self) -> int:
        return len(self._corpus)

    # ------------------------- TF-IDF 拟合 ----------------------- #

    def _fit(self) -> None:
        ids = list(self._corpus.keys())
        docs = [self._corpus[i] for i in ids]
        if not docs:
            self._vectorizer = None
            self._matrix = None
            self._doc_ids = []
            self._dirty = False
            return

        # 语料非常小时放宽 max_df，避免所有词都被过滤
        max_df = 0.95 if len(docs) >= 5 else 1.0

        self._vectorizer = TfidfVectorizer(
            tokenizer=_mixed_tokenize,
            preprocessor=lambda x: x,
            token_pattern=None,
            lowercase=False,
            min_df=1,
            max_df=max_df,
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform(docs)
        self._doc_ids = ids
        self._dirty = False

    # ------------------------- 关键词提取 ----------------------- #

    def extract(
        self,
        doc_id: str,
        top_k: int = 8,
        title: str = "",
        abstract: str = "",
    ) -> list[str]:
        if self._dirty or self._vectorizer is None:
            self._fit()
        if self._vectorizer is None or not self._doc_ids:
            return []
        if doc_id not in self._doc_ids:
            return []

        idx = self._doc_ids.index(doc_id)
        row = self._matrix[idx].toarray().ravel()
        feats = self._vectorizer.get_feature_names_out()

        # 标题 / 摘要命中加成
        title_l = (title or "").lower()
        abstract_l = (abstract or "").lower()
        boosted = row.copy()
        for i, term in enumerate(feats):
            if boosted[i] <= 0:
                continue
            if term in title_l:
                boosted[i] *= 1.8
            if term in abstract_l:
                boosted[i] *= 1.2

        candidate_k = min(max(top_k * 5, 30), len(feats))
        cand_order = np.argsort(boosted)[::-1][:candidate_k]
        candidates = [int(i) for i in cand_order if boosted[int(i)] > 0]
        if not candidates:
            return []

        # MMR 多样化，减小近义/子串词堆叠
        lamb = 0.7
        selected: list[int] = []
        pool = list(candidates)
        while len(selected) < top_k and pool:
            best_i, best_score = None, -1e18
            for i in pool:
                diversity = 0.0
                for j in selected:
                    diversity = max(diversity, _jaccard_chars(feats[i], feats[j]))
                score = lamb * float(boosted[i]) - (1 - lamb) * diversity
                if score > best_score:
                    best_i, best_score = i, score
            if best_i is None:
                break
            selected.append(best_i)
            pool.remove(best_i)

        return [str(feats[i]) for i in selected]
