from __future__ import annotations

import json
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

from config import FILES_DIR, INBOX_DIR, PAPERS_JSON
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
        
        # 初始化时尝试加载本地数据
        self._load_db()

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

    def _load_db(self) -> None:
        """从本地 JSON 加载数据，恢复 papers 和 vectors"""
        if not PAPERS_JSON.exists():
            return
        
        try:
            data = json.loads(PAPERS_JSON.read_text(encoding="utf-8"))
            with self._lock:
                self._papers = data.get("papers", [])
                vectors_list = data.get("vectors", [])
                
                # 如果保存了向量，恢复为 numpy array
                if vectors_list and len(vectors_list) == len(self._papers):
                    self._vectors = np.array(vectors_list, dtype=np.float32)
                else:
                    self._vectors = None
            print(f"Loaded {len(self._papers)} papers from {PAPERS_JSON}")
        except Exception as e:
            print(f"Failed to load DB: {e}")

    def _save_db(self) -> None:
        """将内存数据持久化到本地 JSON"""
        try:
            vectors_list = []
            if self._vectors is not None:
                # Numpy array 无法直接序列化，需转为 list
                vectors_list = self._vectors.tolist()
            
            data = {
                "papers": self._papers,
                "vectors": vectors_list
            }
            PAPERS_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"Failed to save DB: {e}")

    def add_pdf(self, filename: str, raw: bytes, recompute: bool = True) -> str:
        paper_id = uuid.uuid4().hex[:10]
        pdf_path = FILES_DIR / f"{paper_id}.pdf"
        pdf_path.write_bytes(raw)
        
        # 提取文本
        try:
            raw_text = read_pdf_text(pdf_path, max_pages=5)
            cleaned = clean_text(raw_text)
        except Exception as e:
            # 删除损坏文件
            if pdf_path.exists():
                pdf_path.unlink()
            raise ValueError(f"PDF parsing failed: {str(e)}")

        # 校验：如果提取内容为空或太短，视为无效文件
        if not cleaned or len(cleaned) < 50:
            if pdf_path.exists():
                pdf_path.unlink()
            raise ValueError("No text extracted from PDF (file might be image-only or encrypted).")

        display_title = safe_stem(filename)
        title = extract_title_from_text(cleaned, display_title)
        abstract = extract_abstract_block(cleaned)
        
        # 提取第一句话
        first_sentence = ""
        if cleaned:
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
            # 初始化默认字段，防止缺少键值
            "field": "Processing...",
            "confidence": 0.0,
            "cluster": 0,
            "pos": [0.0, 0.0, 0.0],
            "size": 3.0
        }
        
        with self._lock:
            self._papers.append(paper)
            if recompute:
                self._recompute_locked()
            else:
                # 如果不立即重算，也需要保存 papers 列表
                self._save_db()
                
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
                # 注意：这里会抛出 ValueError，需要捕获
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
        # 这里的 list_pdfs 主要是简单的列表返回，
        # 详细逻辑现在主要由 API 层面的 api_papers 负责
        out: list[dict[str, Any]] = []
        with self._lock:
            for p in self._papers:
                out.append(p)
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
            self._save_db()
            return
            
        model = self._ensure_model()
        texts = [
            f"{p.get('abstract','')}\n{' '.join([str(k) for k in (p.get('keywords') or [])])}".strip()
            for p in self._papers
        ]
        
        # 计算向量
        vectors = model.encode(texts, normalize_embeddings=True)
        self._vectors = np.array(vectors, dtype=np.float32)
        n = len(self._papers)

        # --- 聚类逻辑 (KMeans) ---
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
                try:
                    sil = float(silhouette_score(self._vectors, labels, metric="cosine"))
                except Exception:
                    pass

                if sil is None:
                    continue
                
                # 简化评分逻辑，主要看 Silhouette
                key = sil
                if best_key is None or key > best_key:
                    best_key = key
                    best_kmeans = kmeans
                    best_clusters = labels

            if best_kmeans is None or best_clusters is None:
                # fallback
                k = min(5, max(2, int(round(np.sqrt(n)))), n)
                kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
                clusters = kmeans.fit_predict(self._vectors)
                centers = kmeans.cluster_centers_
            else:
                clusters = best_clusters
                centers = best_kmeans.cluster_centers_

        # --- 更新论文属性 ---
        palette = cluster_palette()
        for i, p in enumerate(self._papers):
            cid = int(clusters[i])
            p["cluster"] = cid
            p["field"] = f"Topic {cid + 1}" # 简化命名
            p["color"] = palette[cid % len(palette)]
            
            center = centers[cid] if centers is not None else self._vectors[i]
            v = np.asarray(self._vectors[i], dtype=np.float32).reshape(1, -1)
            c = np.asarray(center, dtype=np.float32).reshape(1, -1)
            sim = float(cosine_similarity(v, c)[0, 0])
            p["confidence"] = max(0.0, min(1.0, sim))

        # --- 降维 (3D 坐标) ---
        coords = reduce_to_3d(self._vectors)
        
        # 简单的径向分离 (使聚类在空间上更开)
        k_count = int(np.max(clusters)) + 1 if clusters.size else 1
        if k_count > 1 and coords.shape[0] == clusters.shape[0]:
            tightened = coords.astype(np.float32, copy=True)
            radius = 5.0
            for cid in range(k_count):
                idx = np.where(clusters == cid)[0]
                if idx.size == 0:
                    continue
                # 将该簇向中心收缩
                center_mass = tightened[idx].mean(axis=0, keepdims=True)
                tightened[idx] = (tightened[idx] - center_mass) * 0.6
                # 将该簇整体移向圆周
                angle = float(2.0 * np.pi * (cid / k_count))
                offset = np.array([np.cos(angle) * radius, 0.0, np.sin(angle) * radius], dtype=np.float32)
                tightened[idx] = tightened[idx] + offset
            coords = tightened

        for i, p in enumerate(self._papers):
            p["pos"] = [float(coords[i, 0]), float(coords[i, 1]), float(coords[i, 2])]
            p["size"] = float(3.0 + (p.get("confidence", 0.0) * 5.0))

        # 计算完成后保存到磁盘
        self._save_db()

    def _visualization_locked(self) -> dict[str, Any]:
        # 该方法可以复用 api.py 中的逻辑，或者保持现状
        # 为了避免 api.py 和 store.py 逻辑重复，这里仅返回基础结构，主要由 api 组装
        nodes = []
        fields_map = {}
        for p in self._papers:
            cid = int(p.get("cluster", 0))
            field_name = p.get("field", f"Topic {cid + 1}")
            
            nodes.append({
                "id": p["id"],
                "x": p["pos"][0],
                "y": p["pos"][1],
                "z": p["pos"][2],
                "field": field_name,
            })
            if cid not in fields_map:
                fields_map[cid] = {"name": field_name, "count": 0}
            fields_map[cid]["count"] += 1
            
        return {"nodes": nodes, "fields": list(fields_map.values())}