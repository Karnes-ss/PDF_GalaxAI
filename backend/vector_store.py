from __future__ import annotations

"""
VectorStore：基于 Chroma 的向量存储层。

两个 Chroma Collection：
  - scholar_papers  : 论文级向量（每篇一条），供聚类/可视化使用
  - scholar_chunks  : Chunk 级向量（每个 chunk 一条），供 RAG 检索使用

本层只负责存取，不负责编码（embedding 由 store.py 完成后传入）。
"""

from typing import Any

import numpy as np

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    _CHROMA_OK = True
except ImportError:
    _CHROMA_OK = False

from chunker import Chunk
from config import CHROMA_DIR


class VectorStore:
    _PAPERS_COL = "scholar_papers"
    _CHUNKS_COL = "scholar_chunks"

    def __init__(self) -> None:
        if not _CHROMA_OK:
            raise RuntimeError(
                "chromadb 未安装，请在虚拟环境中执行: pip install chromadb"
            )

        self._client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # cosine 空间：distance = 1 - similarity
        self._papers_col = self._client.get_or_create_collection(
            name=self._PAPERS_COL,
            metadata={"hnsw:space": "cosine"},
        )
        self._chunks_col = self._client.get_or_create_collection(
            name=self._CHUNKS_COL,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------ #
    # 论文级接口
    # ------------------------------------------------------------------ #

    def add_paper(self, paper_id: str, embedding: np.ndarray) -> None:
        """写入/更新论文级向量（upsert）。"""
        self._papers_col.upsert(
            ids=[paper_id],
            embeddings=[embedding.tolist()],
            metadatas=[{"paper_id": paper_id}],
        )

    def get_paper_embeddings(self, paper_ids: list[str]) -> dict[str, np.ndarray]:
        """
        批量获取论文级向量。
        返回 {paper_id: np.ndarray}，不存在的 paper_id 不出现在结果中。
        """
        if not paper_ids:
            return {}
        res = self._papers_col.get(ids=paper_ids, include=["embeddings"])
        return {
            pid: np.array(emb, dtype=np.float32)
            for pid, emb in zip(res["ids"], res["embeddings"])
        }

    def has_paper(self, paper_id: str) -> bool:
        """检查论文是否已有向量记录（以 papers 集合为准）。"""
        res = self._papers_col.get(ids=[paper_id], include=[])
        return len(res["ids"]) > 0

    def delete_paper(self, paper_id: str) -> None:
        """删除某篇论文的全部数据（论文向量 + 所有 chunks）。"""
        try:
            self._papers_col.delete(ids=[paper_id])
        except Exception:
            pass
        try:
            self._chunks_col.delete(where={"paper_id": paper_id})
        except Exception:
            pass

    def papers_count(self) -> int:
        return self._papers_col.count()

    # ------------------------------------------------------------------ #
    # Chunk 级接口
    # ------------------------------------------------------------------ #

    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        """
        写入 chunk 向量（upsert）。
        embeddings shape: (len(chunks), dim)
        """
        if not chunks:
            return
        self._chunks_col.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings.tolist(),
            documents=[c.text for c in chunks],
            metadatas=[
                {"paper_id": c.paper_id, "index": c.index} for c in chunks
            ],
        )

    def search_chunks(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        paper_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        检索最相关的 chunks。
        返回 list of:
          {chunk_id, paper_id, score (0~1), snippet (≤400字), full_text}
        """
        total = self._chunks_col.count()
        if total == 0:
            return []

        n = min(top_k, total)
        kwargs: dict[str, Any] = dict(
            query_embeddings=[query_embedding.tolist()],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        if paper_id:
            kwargs["where"] = {"paper_id": paper_id}

        res = self._chunks_col.query(**kwargs)

        results: list[dict[str, Any]] = []
        if res and res["ids"] and res["ids"][0]:
            for i, cid in enumerate(res["ids"][0]):
                distance = float(res["distances"][0][i])
                score = max(0.0, 1.0 - distance)
                meta = res["metadatas"][0][i]
                text = res["documents"][0][i]
                results.append(
                    {
                        "chunk_id": cid,
                        "paper_id": meta["paper_id"],
                        "score": score,
                        "snippet": text[:400],
                        "full_text": text,
                    }
                )
        return results

    def chunks_count(self) -> int:
        return self._chunks_col.count()

    # ------------------------------------------------------------------ #
    # 维护接口
    # ------------------------------------------------------------------ #

    def reset_all(self) -> None:
        """清空两张向量表（切换 embedding 模型时必须做，否则维度冲突）。"""
        try:
            self._client.delete_collection(self._PAPERS_COL)
        except Exception:
            pass
        try:
            self._client.delete_collection(self._CHUNKS_COL)
        except Exception:
            pass
        self._papers_col = self._client.get_or_create_collection(
            name=self._PAPERS_COL,
            metadata={"hnsw:space": "cosine"},
        )
        self._chunks_col = self._client.get_or_create_collection(
            name=self._CHUNKS_COL,
            metadata={"hnsw:space": "cosine"},
        )
