from __future__ import annotations

import os
# 设置 HF 镜像以解决国内连接问题
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import uuid
from threading import Lock
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from config import FILES_DIR, INBOX_DIR
from clustering import cluster_palette, reduce_to_3d
from text_processing import (
    clean_text,
    extract_abstract_block,
    extract_keywords_block,
    extract_title_from_text,
    read_pdf_text,
    safe_stem,
    tfidf_keywords_block,
)


class ScholarStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._papers: list[dict[str, Any]] = []
        self._vectors: np.ndarray | None = None
        self._model = None
        self._model_name = os.getenv("SCHOLAR_ST_MODEL") or "all-MiniLM-L6-v2"
        self._offline = (os.getenv("SCHOLAR_OFFLINE") or "").strip().lower() in {"1", "true", "yes"}

    def _ensure_model(self):
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed")
        if self._model is not None:
            return self._model

        try:
            if self._offline:
                self._model = SentenceTransformer(self._model_name, local_files_only=True)
            else:
                self._model = SentenceTransformer(self._model_name)
        except Exception as e:
            if not self._offline:
                try:
                    self._model = SentenceTransformer(self._model_name, local_files_only=True)
                except Exception:
                    pass
            if self._model is None:
                raise RuntimeError(f"Failed to load embedding model: {e}") from e
        return self._model

    def add_pdf(self, filename: str, raw: bytes, recompute: bool = True) -> str:
        paper_id = uuid.uuid4().hex[:10]
        pdf_path = FILES_DIR / f"{paper_id}.pdf"
        pdf_path.write_bytes(raw)
        raw_text = read_pdf_text(pdf_path, max_pages=5)
        cleaned = clean_text(raw_text)
        display_title = safe_stem(filename)
        title = extract_title_from_text(cleaned, display_title)
        abstract = extract_abstract_block(cleaned)
        
        # 提取第一句话
        first_sentence = ""
        if cleaned:
            # 简单的分句逻辑：查找第一个句号、问号或感叹号
            import re
            match = re.search(r'[^.!?。！？]+[.!?。！？]', cleaned)
            if match:
                first_sentence = match.group(0).strip()
            else:
                first_sentence = cleaned[:100].strip() + "..."
        
        keywords = extract_keywords_block(cleaned)
        if not keywords:
            keywords = tfidf_keywords_block(f"{title}\n{abstract}")
        paper = {
            "id": paper_id,
            "title": title,
            "display_title": display_title,
            "abstract": abstract,
            "first_sentence": first_sentence,
            "keywords": keywords,
            "filename": filename,
        }
        with self._lock:
            self._papers.append(paper)
            if recompute:
                self._recompute_locked()
        return paper_id

    def ingest_from_inbox(self) -> int:
        if not INBOX_DIR.exists():
            return 0

        count = 0
        with self._lock:
            existing_filenames = {p.get("filename") for p in self._papers}

        for pdf_path in INBOX_DIR.glob("*.pdf"):
            if pdf_path.name in existing_filenames:
                continue
            try:
                raw = pdf_path.read_bytes()
                self.add_pdf(pdf_path.name, raw, recompute=False)
                count += 1
                print(f"Ingested: {pdf_path.name}")
            except Exception as e:
                print(f"Error ingesting {pdf_path.name}: {e}")

        if count > 0:
            with self._lock:
                self._recompute_locked()
        return count

    def list_pdfs(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        with self._lock:
            for p in self._papers:
                out.append(
                    {
                        "id": p["id"],
                        "title": p.get("title", ""),
                        "abstract": p.get("abstract", ""),
                        "keywords": p.get("keywords", []),
                        "field": p.get("field", ""),
                        "confidence": float(p.get("confidence", 0.0)),
                    }
                )
        return out

    def analyze(self) -> dict[str, Any]:
        with self._lock:
            self._recompute_locked()
            return self._visualization_locked()

    def visualization(self) -> dict[str, Any]:
        with self._lock:
            return self._visualization_locked()

    def _recompute_locked(self) -> None:
        if not self._papers:
            self._vectors = None
            return
        model = self._ensure_model()
        texts = [
            f"{p.get('abstract','')}\n{' '.join([str(k) for k in (p.get('keywords') or [])])}".strip()
            for p in self._papers
        ]
        vectors = model.encode(texts, normalize_embeddings=True)
        self._vectors = np.array(vectors, dtype=np.float32)
        n = len(self._papers)
        if n == 1:
            clusters = np.array([0])
            centers = self._vectors.copy()
        else:
            max_k = min(8, n)
            candidate_ks = list(range(2, max_k + 1))

            best_kmeans = None
            best_clusters = None
            best_key = None

            for k in candidate_ks:
                try:
                    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
                    labels = kmeans.fit_predict(self._vectors)
                except Exception:
                    continue

                uniq = np.unique(labels)
                if uniq.size < 2 or uniq.size >= n:
                    continue

                sil = None
                ch = None
                try:
                    sil = float(silhouette_score(self._vectors, labels, metric="cosine"))
                except Exception:
                    sil = None

                try:
                    ch = float(calinski_harabasz_score(self._vectors, labels))
                except Exception:
                    ch = None

                if sil is None and ch is None:
                    continue

                key = (
                    1 if sil is not None else 0,
                    sil if sil is not None else float("-inf"),
                    ch if ch is not None else float("-inf"),
                    -k,
                )

                if best_key is None or key > best_key:
                    best_key = key
                    best_kmeans = kmeans
                    best_clusters = labels

            if best_kmeans is None or best_clusters is None:
                k = min(5, max(2, int(round(np.sqrt(n)))), n)
                kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
                clusters = kmeans.fit_predict(self._vectors)
                centers = kmeans.cluster_centers_
            else:
                clusters = best_clusters
                centers = best_kmeans.cluster_centers_
        palette = cluster_palette()
        for i, p in enumerate(self._papers):
            cid = int(clusters[i])
            p["cluster"] = cid
            p["field"] = f"领域{cid + 1}"
            p["color"] = palette[cid % len(palette)]
            center = centers[cid] if centers is not None else self._vectors[i]
            v = np.asarray(self._vectors[i], dtype=np.float32).reshape(1, -1)
            c = np.asarray(center, dtype=np.float32).reshape(1, -1)
            sim = float(cosine_similarity(v, c)[0, 0])
            p["confidence"] = max(0.0, min(1.0, sim))
        coords = reduce_to_3d(self._vectors)

        k = int(np.max(clusters)) + 1 if clusters.size else 1
        if k > 1 and coords.shape[0] == clusters.shape[0]:
            tightened = coords.astype(np.float32, copy=True)
            radius = 5.0
            for cid in range(k):
                idx = np.where(clusters == cid)[0]
                if idx.size == 0:
                    continue
                center = tightened[idx].mean(axis=0, keepdims=True)
                tightened[idx] = (tightened[idx] - center) * 0.55
                angle = float(2.0 * np.pi * (cid / k))
                offset = np.array([np.cos(angle) * radius, 0.0, np.sin(angle) * radius], dtype=np.float32)
                tightened[idx] = tightened[idx] + offset
            coords = tightened

        for i, p in enumerate(self._papers):
            p["pos"] = [float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2])]
            p["size"] = float(3.0 + (p.get("confidence", 0.0) * 4.0))

    def _visualization_locked(self) -> dict[str, Any]:
        nodes: list[dict[str, Any]] = []
        fields_map: dict[int, dict[str, Any]] = {}
        for p in self._papers:
            pos = p.get("pos", [0.0, 0.0, 0.0])
            cid = int(p.get("cluster", 0))
            color = p.get("color", cluster_palette()[0])
            field_name = p.get("field", f"领域{cid + 1}")
            nodes.append(
                {
                    "id": p["id"],
                    "title": p.get("title", ""),
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                    "z": float(pos[2]),
                    "color": color,
                    "field": field_name,
                    "size": float(p.get("size", 4.0)),
                }
            )
            if cid not in fields_map:
                fields_map[cid] = {"name": field_name, "color": color, "count": 0}
            fields_map[cid]["count"] += 1
        fields = list(fields_map.values())
        return {"nodes": nodes, "fields": fields}
