from __future__ import annotations

import json
import os
import re
import uuid
from threading import Lock
from typing import Any

# 设置 HF 镜像以解决国内连接问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from chunker import chunk_text
from clustering import cluster_palette, reduce_to_3d
from config import FILES_DIR, INBOX_DIR, PAPERS_JSON
from text_processing import (
    clean_text,
    extract_abstract_block,
    extract_keywords_block,
    extract_title_from_text,
    read_pdf_text,
    safe_stem,
    tfidf_keywords_block,
)
from vector_store import VectorStore


class ScholarStore:
    """
    核心数据层。职责划分：
      - PAPERS_JSON  : 论文元信息 + UI 状态（pos/cluster/color），重启不丢
      - VectorStore  : 论文级向量（聚类用）+ Chunk 向量（RAG 检索用）
      - 内存           : self._vectors 运行时缓存，供 api_papers 计算边权
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._papers: list[dict[str, Any]] = []
        self._vectors: np.ndarray | None = None   # 运行时缓存，不持久化
        self._model = None
        self._model_name = os.getenv("SCHOLAR_ST_MODEL") or "all-MiniLM-L6-v2"
        self._offline = (
            (os.getenv("SCHOLAR_OFFLINE") or "").strip().lower()
            in {"1", "true", "yes"}
        )
        self._vstore: VectorStore | None = None

        self._load_db()

    # ------------------------------------------------------------------ #
    # 懒加载：模型 & 向量库
    # ------------------------------------------------------------------ #

    def _ensure_model(self) -> SentenceTransformer:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers 未安装")
        if self._model is not None:
            return self._model
        try:
            if self._offline:
                self._model = SentenceTransformer(
                    self._model_name, local_files_only=True
                )
            else:
                self._model = SentenceTransformer(self._model_name)
        except Exception as e:
            if not self._offline:
                try:
                    self._model = SentenceTransformer(
                        self._model_name, local_files_only=True
                    )
                except Exception:
                    pass
            if self._model is None:
                raise RuntimeError(f"Failed to load embedding model: {e}") from e
        return self._model

    def _ensure_vstore(self) -> VectorStore:
        if self._vstore is None:
            self._vstore = VectorStore()
        return self._vstore

    # ------------------------------------------------------------------ #
    # 持久化：JSON 只存元信息 + UI 状态，不存向量
    # ------------------------------------------------------------------ #

    def _load_db(self) -> None:
        """从 JSON 加载论文元信息与 UI 状态（不含向量）。"""
        if not PAPERS_JSON.exists():
            return
        try:
            data = json.loads(PAPERS_JSON.read_text(encoding="utf-8"))
            with self._lock:
                papers = data.get("papers", [])
                # 兼容旧格式：去掉可能残留的 vectors 字段（节省内存）
                for p in papers:
                    p.pop("vectors", None)
                self._papers = papers
            print(f"[store] Loaded {len(self._papers)} papers from JSON")
        except Exception as e:
            print(f"[store] Failed to load DB: {e}")

    def _save_db(self) -> None:
        """将论文元信息与 UI 状态持久化到 JSON（不含向量）。"""
        try:
            # 保存前确保不含向量字段
            papers_clean = [
                {k: v for k, v in p.items() if k != "vectors"}
                for p in self._papers
            ]
            PAPERS_JSON.write_text(
                json.dumps({"papers": papers_clean}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            print(f"[store] Failed to save DB: {e}")

    # ------------------------------------------------------------------ #
    # 向量化辅助
    # ------------------------------------------------------------------ #

    @staticmethod
    def _paper_embed_text(paper: dict[str, Any]) -> str:
        """生成论文级向量的文本（摘要 + 关键词）。"""
        return (
            f"{paper.get('abstract', '')}\n"
            f"{' '.join(str(k) for k in (paper.get('keywords') or []))}"
        ).strip()

    @staticmethod
    def _tokenize_query(text: str) -> list[str]:
        """
        将查询词拆为关键词 token（中英文混合）。
        """
        t = (text or "").lower()
        toks = re.findall(r"[a-z0-9]{2,}|[\u4e00-\u9fff]{1,6}", t)
        return list(dict.fromkeys(toks))

    @staticmethod
    def _text_quality_score(text: str) -> float:
        """
        估计文本质量（0~1），用于降低乱码 chunk 的排名。
        经验规则：
        - 出现 replacement char '�'、'??'、'a?��' 等异常模式会扣分
        - 可读字符占比越高分越高
        """
        s = text or ""
        if not s.strip():
            return 0.0
        total = max(1, len(s))

        replacement_cnt = s.count("�")
        qmark_cnt = s.count("?")
        double_q_cnt = s.count("??")
        mojibake_hint_cnt = len(re.findall(r"[a-zA-Z]\?��|\?��", s))
        token_with_q_cnt = len(re.findall(r"\b\S*\?\S*\b", s))

        # '?' 在正常学术文本中很少大量出现；出现密集通常是编码映射损坏
        bad = (
            replacement_cnt * 3
            + double_q_cnt
            + mojibake_hint_cnt * 2
            + qmark_cnt * 0.8
            + token_with_q_cnt * 0.6
        )
        bad_ratio = min(1.0, bad / total)

        readable_chars = len(re.findall(r"[a-zA-Z0-9\u4e00-\u9fff\s,.;:!?()\-_/]", s))
        readable_ratio = readable_chars / total

        score = max(0.0, min(1.0, 0.65 * (1.0 - bad_ratio) + 0.35 * readable_ratio))
        return score

    def _index_paper(self, paper: dict[str, Any], full_text: str) -> None:
        """
        对单篇论文做向量化并写入 VectorStore（论文级 + Chunk 级）。
        允许失败（仅打印警告），不影响主流程。
        """
        try:
            model = self._ensure_model()
            vstore = self._ensure_vstore()

            # 论文级向量（用摘要+关键词）
            paper_emb = model.encode(
                [self._paper_embed_text(paper)], normalize_embeddings=True
            )[0].astype(np.float32)
            vstore.add_paper(paper["id"], paper_emb)

            # Chunk 级向量（用全文）
            chunks = chunk_text(paper["id"], full_text)
            # 过滤明显乱码/过短 chunk，避免污染向量检索
            chunks = [
                c for c in chunks
                if len(c.text.strip()) >= 40 and self._text_quality_score(c.text) >= 0.30
            ]
            if chunks:
                chunk_embs = model.encode(
                    [c.text for c in chunks], normalize_embeddings=True
                ).astype(np.float32)
                vstore.add_chunks(chunks, chunk_embs)
                print(
                    f"[store] Indexed {len(chunks)} chunks for '{paper.get('filename', paper['id'])}'"
                )
        except Exception as e:
            print(f"[store] Warning: vector indexing failed for {paper.get('filename', '')}: {e}")

    def ensure_all_indexed(self) -> None:
        """
        启动时调用：检查 JSON 中已有论文是否在 Chroma 里有向量；
        若缺失则从对应 PDF 文件重新索引（兼容旧数据迁移）。
        """
        with self._lock:
            papers = list(self._papers)

        if not papers:
            return

        try:
            vstore = self._ensure_vstore()
        except Exception as e:
            print(f"[store] VectorStore unavailable, skipping index check: {e}")
            return

        missing = [p for p in papers if not vstore.has_paper(p["id"])]
        if not missing:
            return

        print(f"[store] Re-indexing {len(missing)} papers missing from Chroma ...")
        for paper in missing:
            pdf_path = FILES_DIR / f"{paper['id']}.pdf"
            if not pdf_path.exists():
                print(f"[store] PDF missing for {paper['id']}, skipping")
                continue
            try:
                full_text = clean_text(read_pdf_text(pdf_path, max_pages=0))
                self._index_paper(paper, full_text)
            except Exception as e:
                print(f"[store] Re-index failed for {paper['id']}: {e}")

    # ------------------------------------------------------------------ #
    # 添加 PDF
    # ------------------------------------------------------------------ #

    def add_pdf(self, filename: str, raw: bytes, recompute: bool = True) -> str:
        paper_id = uuid.uuid4().hex[:10]
        pdf_path = FILES_DIR / f"{paper_id}.pdf"
        try:
            pdf_path.write_bytes(raw)
        except Exception as e:
            raise ValueError(f"PDF 文件保存失败: {e}")

        try:
            full_text = read_pdf_text(pdf_path, max_pages=0)   # 全文
            cleaned = clean_text(full_text)
        except Exception as e:
            if pdf_path.exists():
                pdf_path.unlink()
            raise ValueError(f"PDF 解析失败: {e}")

        if not cleaned or len(cleaned) < 50:
            if pdf_path.exists():
                pdf_path.unlink()
            raise ValueError(f"PDF 无可提取文本（长度: {len(cleaned) if cleaned else 0}）")

        display_title = safe_stem(filename)
        try:
            title = extract_title_from_text(cleaned, display_title)
            abstract = extract_abstract_block(cleaned)
        except Exception as e:
            if pdf_path.exists():
                pdf_path.unlink()
            raise ValueError(f"元数据提取失败: {e}")

        first_sentence = ""
        try:
            m = re.search(r"[^.!?。！？]+[.!?。！？]", cleaned)
            first_sentence = m.group(0).strip() if m else cleaned[:100].strip() + "..."
        except Exception:
            first_sentence = cleaned[:100].strip() + "..."

        try:
            keywords = extract_keywords_block(cleaned)
            if not keywords:
                keywords = tfidf_keywords_block(f"{title}\n{abstract}")
        except Exception as e:
            print(f"[store] Warning: keyword extraction failed: {e}")
            keywords = [title[:20]] if title else []

        paper: dict[str, Any] = {
            "id": paper_id,
            "title": title,
            "display_title": display_title,
            "abstract": abstract,
            "first_sentence": first_sentence,
            "keywords": keywords,
            "filename": filename,
            "field": "Processing...",
            "confidence": 0.0,
            "cluster": 0,
            "pos": [0.0, 0.0, 0.0],
            "size": 3.0,
        }

        # 向量化写入 Chroma
        try:
            self._index_paper(paper, cleaned)
        except Exception as e:
            print(f"[store] Warning: indexing failed: {e}")

        with self._lock:
            self._papers.append(paper)
            try:
                if recompute:
                    self._recompute_locked()
                else:
                    self._save_db()
            except Exception as e:
                self._papers.pop()
                if pdf_path.exists():
                    pdf_path.unlink()
                raise ValueError(f"图表计算失败: {e}")

        return paper_id

    # ------------------------------------------------------------------ #
    # Inbox 批量导入
    # ------------------------------------------------------------------ #

    def ingest_from_inbox(self) -> int:
        if not INBOX_DIR.exists():
            return 0

        with self._lock:
            existing = {p.get("filename") for p in self._papers}

        count = 0
        for pdf_path in INBOX_DIR.glob("*.pdf"):
            if pdf_path.name in existing:
                continue
            try:
                self.add_pdf(pdf_path.name, pdf_path.read_bytes(), recompute=False)
                count += 1
                print(f"[store] Ingested: {pdf_path.name}")
            except Exception as e:
                print(f"[store] Error ingesting {pdf_path.name}: {e}")

        if count > 0:
            with self._lock:
                self._recompute_locked()

        return count

    # ------------------------------------------------------------------ #
    # 聚类 + 可视化坐标重算
    # ------------------------------------------------------------------ #

    def _recompute_locked(self) -> None:
        """
        用 Chroma 中的论文级向量重新做聚类和 3D 降维，
        结果写回 self._papers 并持久化到 JSON。
        同时更新 self._vectors 运行时缓存（供 api_papers 计算边权）。
        """
        if not self._papers:
            self._vectors = None
            self._save_db()
            return

        vectors = self._get_paper_vectors_locked()
        if vectors is None or len(vectors) != len(self._papers):
            self._save_db()
            return

        self._vectors = vectors   # 更新运行时缓存
        n = len(self._papers)

        # --- KMeans 聚类 ---
        if n == 1:
            clusters = np.array([0])
            centers = vectors.copy()
        else:
            best_km, best_labels, best_score = None, None, None
            for k in range(2, min(8, n) + 1):
                try:
                    km = KMeans(n_clusters=k, n_init="auto", random_state=42)
                    labels = km.fit_predict(vectors)
                except Exception:
                    continue
                if np.unique(labels).size < 2:
                    continue
                try:
                    s = float(silhouette_score(vectors, labels, metric="cosine"))
                except Exception:
                    continue
                if best_score is None or s > best_score:
                    best_score, best_km, best_labels = s, km, labels

            if best_km is None:
                k = min(5, max(2, int(round(np.sqrt(n)))), n)
                best_km = KMeans(n_clusters=k, n_init="auto", random_state=42)
                best_labels = best_km.fit_predict(vectors)

            clusters = best_labels
            centers = best_km.cluster_centers_

        palette = cluster_palette()
        for i, p in enumerate(self._papers):
            cid = int(clusters[i])
            p["cluster"] = cid
            p["field"] = f"Topic {cid + 1}"
            p["color"] = palette[cid % len(palette)]
            sim = float(
                cosine_similarity(
                    vectors[i].reshape(1, -1), centers[cid].reshape(1, -1)
                )[0, 0]
            )
            p["confidence"] = max(0.0, min(1.0, sim))

        # --- 3D 降维 + 簇间分离 ---
        coords = reduce_to_3d(vectors)
        k_count = int(np.max(clusters)) + 1 if clusters.size else 1

        # For small numbers of papers, ensure good separation
        if n <= 3:
            # Use fixed positions for better separation
            if n == 2:
                coords[0] = np.array([-3.0, 0.0, 0.0], dtype=np.float32)
                coords[1] = np.array([3.0, 0.0, 0.0], dtype=np.float32)
            elif n == 3:
                coords[0] = np.array([0.0, 3.0, 0.0], dtype=np.float32)
                coords[1] = np.array([-3.0, -1.5, 0.0], dtype=np.float32)
                coords[2] = np.array([3.0, -1.5, 0.0], dtype=np.float32)
        elif k_count > 1:
            tightened = coords.astype(np.float32, copy=True)
            for cid in range(k_count):
                idx = np.where(clusters == cid)[0]
                if idx.size == 0:
                    continue
                cm = tightened[idx].mean(axis=0, keepdims=True)
                tightened[idx] = (tightened[idx] - cm) * 0.6
                angle = float(2.0 * np.pi * cid / k_count)
                offset = np.array(
                    [np.cos(angle) * 5.0, 0.0, np.sin(angle) * 5.0],
                    dtype=np.float32,
                )
                tightened[idx] += offset
            coords = tightened

        for i, p in enumerate(self._papers):
            p["pos"] = [float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2])]
            p["size"] = float(3.0 + p.get("confidence", 0.0) * 5.0)

        self._save_db()

    def _get_paper_vectors_locked(self) -> np.ndarray | None:
        """
        从 Chroma 获取所有论文的向量（顺序与 self._papers 一致）。
        若某篇论文缺失，则重新编码并补写 Chroma。
        返回 shape (n, dim) 的 numpy array；失败返回 None。
        """
        paper_ids = [p["id"] for p in self._papers]
        try:
            vstore = self._ensure_vstore()
            model = self._ensure_model()
        except Exception as e:
            print(f"[store] Cannot get vectors: {e}")
            return None

        cached = vstore.get_paper_embeddings(paper_ids)

        missing = [p for p in self._papers if p["id"] not in cached]
        if missing:
            texts = [self._paper_embed_text(p) for p in missing]
            embs = model.encode(texts, normalize_embeddings=True)
            for p, emb in zip(missing, embs):
                arr = emb.astype(np.float32)
                vstore.add_paper(p["id"], arr)
                cached[p["id"]] = arr

        rows = [cached.get(pid) for pid in paper_ids]
        if any(r is None for r in rows):
            return None
        return np.stack(rows, axis=0)

    # ------------------------------------------------------------------ #
    # 列表 / 可视化
    # ------------------------------------------------------------------ #

    def list_pdfs(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._papers)

    def delete_paper(self, paper_id: str) -> bool:
        """
        删除指定论文（元数据 + PDF 文件 + 向量库记录），并重算可视化状态。
        返回是否成功删除到论文记录。
        """
        with self._lock:
            idx = next((i for i, p in enumerate(self._papers) if str(p.get("id") or "") == paper_id), -1)
            if idx < 0:
                return False
            self._papers.pop(idx)

            # 删除向量库记录（失败不阻断）
            try:
                self._ensure_vstore().delete_paper(paper_id)
            except Exception as e:
                print(f"[store] Warning: failed to delete vectors for {paper_id}: {e}")

            # 删除 PDF 文件（失败不阻断）
            try:
                pdf_path = FILES_DIR / f"{paper_id}.pdf"
                if pdf_path.exists():
                    pdf_path.unlink()
            except Exception as e:
                print(f"[store] Warning: failed to delete pdf for {paper_id}: {e}")

            try:
                self._recompute_locked()
            except Exception as e:
                print(f"[store] Warning: failed to recompute after delete: {e}")
                self._save_db()
        return True

    def analyze(self) -> dict[str, Any]:
        with self._lock:
            self._recompute_locked()
            return self._visualization_locked()

    def visualization(self) -> dict[str, Any]:
        with self._lock:
            return self._visualization_locked()

    def _visualization_locked(self) -> dict[str, Any]:
        nodes, fields_map = [], {}
        for p in self._papers:
            cid = int(p.get("cluster", 0))
            field = p.get("field", f"Topic {cid + 1}")
            nodes.append(
                {
                    "id": p["id"],
                    "x": p["pos"][0],
                    "y": p["pos"][1],
                    "z": p["pos"][2],
                    "field": field,
                }
            )
            if cid not in fields_map:
                fields_map[cid] = {"name": field, "count": 0}
            fields_map[cid]["count"] += 1
        return {"nodes": nodes, "fields": list(fields_map.values())}

    # ------------------------------------------------------------------ #
    # 语义检索（供 API 层调用）
    # ------------------------------------------------------------------ #

    def search_chunks(
        self,
        query: str,
        top_k: int = 5,
        paper_id: str | None = None,
        min_score: float = 0.15,
    ) -> tuple[list[dict[str, Any]], str]:
        """
        对 query 做 Chunk 级语义检索。

        返回 (results, status)
        status 枚举：
          "ok"          - 成功，results 非空
          "no_match"    - 未找到相关度 >= min_score 的 chunk
          "empty_db"    - 向量库无任何 chunk（尚未上传文档）
          "model_error" - 模型或向量库初始化失败
        """
        try:
            model = self._ensure_model()
            vstore = self._ensure_vstore()
        except Exception as e:
            print(f"[store] search_chunks model/vstore error: {e}")
            return [], "model_error"

        if vstore.chunks_count() == 0:
            return [], "empty_db"

        q_emb = model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
        # 先多取一些候选，再做重排
        candidate_k = max(top_k * 5, 20)
        candidates = vstore.search_chunks(q_emb, top_k=candidate_k, paper_id=paper_id)

        q_tokens = self._tokenize_query(query)
        reranked: list[dict[str, Any]] = []
        for r in candidates:
            text = str(r.get("full_text") or r.get("snippet") or "")
            text_l = text.lower()

            lexical_hits = 0
            for tok in q_tokens:
                if tok in text_l:
                    lexical_hits += 1
            lexical_bonus = 0.0
            if q_tokens:
                lexical_bonus = lexical_hits / len(q_tokens)

            quality = self._text_quality_score(text)
            # 联合分：语义相似度 + 词面命中 + 文本质量
            final_score = (
                0.72 * float(r["score"]) +
                0.18 * float(lexical_bonus) +
                0.10 * float(quality)
            )
            rr = dict(r)
            rr["raw_score"] = float(r["score"])
            rr["quality"] = float(quality)
            rr["score"] = float(final_score)
            reranked.append(rr)

        reranked.sort(key=lambda x: x["score"], reverse=True)
        results = [r for r in reranked if r["score"] >= min_score][:top_k]

        if not results:
            return [], "no_match"
        return results, "ok"
