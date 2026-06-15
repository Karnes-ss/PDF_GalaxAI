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
from keyword_extractor import GlobalKeywordExtractor
from summarizer import extractive_summary
from text_processing import (
    clean_text,
    extract_abstract_block,
    extract_keywords_block,
    extract_title_from_text,
    read_pdf_pages,
    read_pdf_text,
    safe_stem,
)
from vector_store import VectorStore


_ABSTRACT_ANCHOR_RE = re.compile(r"(abstract|摘要)[:：]?\s", re.IGNORECASE)


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
        # 默认换成 BAAI/bge-m3：多语言 + 学术术语友好，1024 维，比 MiniLM(384) 强非常多。
        # 首次加载需要下载 ~2.3GB 模型，之后会走 HF cache。
        # 若要切换，改环境变量 SCHOLAR_ST_MODEL（比如用纯中文的 BAAI/bge-large-zh-v1.5）。
        self._model_name = os.getenv("SCHOLAR_ST_MODEL") or "BAAI/bge-m3"
        self._offline = (
            (os.getenv("SCHOLAR_OFFLINE") or "").strip().lower()
            in {"1", "true", "yes"}
        )
        self._vstore: VectorStore | None = None
        # 记录当前库里向量来自哪个模型；换模型后检测到不一致时会警告
        self._last_emb_model: str | None = None

        # 全库关键词提取器：
        # _kw_corpus 保留每篇论文的清洗后全文，用于跨文献 TF-IDF。
        # 上传 / 删除时同步更新，下次 extract 时懒重建 TF-IDF。
        self._kw_extractor = GlobalKeywordExtractor()
        self._kw_corpus: dict[str, str] = {}

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
                for p in papers:
                    p.pop("vectors", None)
                self._papers = papers
                self._last_emb_model = data.get("emb_model") or None
            print(f"[store] Loaded {len(self._papers)} papers from JSON")
            if self._last_emb_model and self._last_emb_model != self._model_name:
                print(
                    f"[store] ⚠ Embedding model changed: "
                    f"was '{self._last_emb_model}', now '{self._model_name}'. "
                    f"向量维度/语义都不兼容，请在 UI 里点『重建全库索引』。"
                )
        except Exception as e:
            print(f"[store] Failed to load DB: {e}")

    def _save_db(self) -> None:
        """将论文元信息与 UI 状态持久化到 JSON（不含向量）。"""
        try:
            papers_clean = [
                {k: v for k, v in p.items() if k != "vectors"}
                for p in self._papers
            ]
            payload = {
                "papers": papers_clean,
                "emb_model": self._last_emb_model or self._model_name,
            }
            PAPERS_JSON.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
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

    # ------------------------------------------------------------------ #
    # 关键词语料维护
    # ------------------------------------------------------------------ #

    def _register_corpus(self, paper_id: str, cleaned_text: str) -> None:
        """把一篇论文的清洗文本加入全库语料（用于跨文献 TF-IDF）。"""
        self._kw_corpus[paper_id] = cleaned_text or ""
        self._kw_extractor.add(paper_id, cleaned_text or "")

    def _unregister_corpus(self, paper_id: str) -> None:
        self._kw_corpus.pop(paper_id, None)
        self._kw_extractor.remove(paper_id)

    def _auto_keywords(
        self,
        paper_id: str,
        top_k: int = 8,
        title: str = "",
        abstract: str = "",
    ) -> list[str]:
        """基于当前全库 TF-IDF 给指定论文生成关键词。"""
        try:
            return self._kw_extractor.extract(
                paper_id, top_k=top_k, title=title, abstract=abstract
            )
        except Exception as e:
            print(f"[store] keyword extraction failed for {paper_id}: {e}")
            return []

    def _build_best_abstract(self, cleaned: str) -> str:
        """
        统一的"最佳摘要"策略：
          1. 原文含 Abstract / 摘要 段 → 直接使用（最权威）
          2. 否则 → TextRank + MMR 抽取式摘要
          3. 两者都失败 → extract_abstract_block 的段落兜底
        """
        if not cleaned:
            return ""
        if _ABSTRACT_ANCHOR_RE.search(cleaned):
            return extract_abstract_block(cleaned)
        try:
            model = self._ensure_model()
        except Exception:
            model = None
        summary = extractive_summary(cleaned, model, max_sentences=4, max_chars=420)
        if summary:
            return summary
        return extract_abstract_block(cleaned)

    def rebuild_index_for_all(self) -> dict[str, int]:
        """
        对所有已入库文献"清向量 + 重新切块 + 重新向量化"。
        不改 pos/cluster/color/size（等所有重建完一次性 _recompute_locked）。
        适用于：修改过 PDF 抽取策略（如 OCR 模式切换）后，让 RAG 召回立即生效。
        """
        with self._lock:
            papers = list(self._papers)

        if not papers:
            return {"total": 0, "rebuilt": 0, "skipped_missing_pdf": 0, "failed": 0}

        try:
            vstore = self._ensure_vstore()
        except Exception as e:
            print(f"[store] rebuild_index: vstore unavailable: {e}")
            return {"total": len(papers), "rebuilt": 0, "skipped_missing_pdf": 0, "failed": len(papers)}

        rebuilt, skipped_missing_pdf, failed = 0, 0, 0
        for p in papers:
            pid = str(p.get("id") or "")
            if not pid:
                continue
            pdf_path = FILES_DIR / f"{pid}.pdf"
            if not pdf_path.exists():
                skipped_missing_pdf += 1
                continue
            try:
                cleaned = clean_text(read_pdf_text(pdf_path, max_pages=0))
                if not cleaned or len(cleaned) < 30:
                    failed += 1
                    continue

                # 清掉旧的 paper 向量 + 全部 chunks
                try:
                    vstore.delete_paper(pid)
                except Exception as e:
                    print(f"[store] rebuild_index: delete_paper({pid}) warn: {e}")

                # 刷新全库语料（关键词也会更新）
                self._register_corpus(pid, cleaned)

                # 读取当前 paper 的元信息（外层 papers list 是副本，重新定位）
                with self._lock:
                    target = next(
                        (x for x in self._papers if str(x.get("id") or "") == pid),
                        None,
                    )
                    paper_snapshot = dict(target) if target else None

                if paper_snapshot is None:
                    failed += 1
                    continue

                self._index_paper(paper_snapshot, cleaned)
                rebuilt += 1
            except Exception as e:
                print(f"[store] rebuild_index failed for {pid}: {e}")
                failed += 1

        # 最后一次性重算聚类/坐标
        try:
            with self._lock:
                self._recompute_locked()
        except Exception as e:
            print(f"[store] rebuild_index: _recompute_locked warn: {e}")

        # 语料也变了，刷一次自动关键词
        try:
            kw_updated = self.refresh_keywords_for_all()
        except Exception as e:
            print(f"[store] rebuild_index: refresh_keywords warn: {e}")
            kw_updated = 0

        return {
            "total": len(papers),
            "rebuilt": rebuilt,
            "skipped_missing_pdf": skipped_missing_pdf,
            "failed": failed,
            "keyword_updated": kw_updated,
        }

    def refresh_previews_for_all(self) -> dict[str, int]:
        """
        重算所有已存在论文的「预览字段」：title / abstract / first_sentence /
        自动关键词。不触碰 Chroma 向量、不改 pos/cluster/color/size，仅刷新
        文本层面展示内容。返回统计信息。

        适用场景：升级 clean_text / extract_abstract_block 后，让已入库文献
        的预览框立即生效，而不需要用户重新上传。
        """
        with self._lock:
            papers = list(self._papers)

        updated, skipped_missing_pdf, failed = 0, 0, 0
        for p in papers:
            pid = str(p.get("id") or "")
            if not pid:
                continue
            pdf_path = FILES_DIR / f"{pid}.pdf"
            if not pdf_path.exists():
                skipped_missing_pdf += 1
                continue
            try:
                full_text = read_pdf_text(pdf_path, max_pages=0)
                cleaned = clean_text(full_text)
                if not cleaned or len(cleaned) < 30:
                    failed += 1
                    continue

                display_title = str(
                    p.get("display_title")
                    or safe_stem(str(p.get("filename") or "")) or pid
                )
                new_title = extract_title_from_text(cleaned, display_title)
                new_abstract = self._build_best_abstract(cleaned)

                first_sentence = ""
                try:
                    m = re.search(r"[^.!?。！？]+[.!?。！？]", cleaned)
                    first_sentence = (
                        m.group(0).strip() if m else cleaned[:100].strip() + "..."
                    )
                except Exception:
                    first_sentence = cleaned[:100].strip() + "..."

                # 刷新该论文在全库语料中的内容（影响后续关键词提取）
                self._register_corpus(pid, cleaned)

                with self._lock:
                    # 用 id 重新定位（避免期间有并发变更）
                    target = next(
                        (x for x in self._papers if str(x.get("id") or "") == pid),
                        None,
                    )
                    if target is None:
                        continue
                    target["title"] = new_title
                    target["abstract"] = new_abstract
                    target["first_sentence"] = first_sentence
                    target["display_title"] = display_title

                updated += 1
            except Exception as e:
                print(f"[store] refresh_previews failed for {pid}: {e}")
                failed += 1

        # 语料整体更新完之后再刷一次自动关键词（declared 的会被保留）
        try:
            kw_updated = self.refresh_keywords_for_all()
        except Exception as e:
            print(f"[store] refresh_keywords_for_all after previews failed: {e}")
            kw_updated = 0

        with self._lock:
            self._save_db()

        return {
            "total": len(papers),
            "updated": updated,
            "keyword_updated": kw_updated,
            "skipped_missing_pdf": skipped_missing_pdf,
            "failed": failed,
        }

    def refresh_keywords_for_all(self, top_k: int = 8) -> int:
        """
        对全库每篇论文重算关键词（基于最新语料）。
        仅覆盖"论文自带 Keywords 段"之外的情形（保留人工/官方关键词）。
        返回被更新的篇数。
        """
        with self._lock:
            updated = 0
            for p in self._papers:
                pid = str(p.get("id") or "")
                if not pid:
                    continue
                # 如果论文本身带有 Keywords 段（_keywords_source == "declared"），保留
                if p.get("_keywords_source") == "declared":
                    continue
                new_kws = self._auto_keywords(
                    pid,
                    top_k=top_k,
                    title=str(p.get("title") or ""),
                    abstract=str(p.get("abstract") or ""),
                )
                if new_kws:
                    p["keywords"] = new_kws
                    p["_keywords_source"] = "auto"
                    updated += 1
            if updated > 0:
                self._save_db()
            return updated

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
        启动时调用：
          1. 检查 JSON 中已有论文是否在 Chroma 里有向量；缺失则重新索引
          2. 顺便把每篇论文的清洗全文注册进全库关键词语料
          3. 完成后若有新增注册，全局重算一次关键词
        """
        with self._lock:
            papers = list(self._papers)

        if not papers:
            return

        try:
            vstore = self._ensure_vstore()
        except Exception as e:
            print(f"[store] VectorStore unavailable, skipping index check: {e}")
            vstore = None

        missing_vec_ids: set[str] = set()
        if vstore is not None:
            missing_vec_ids = {p["id"] for p in papers if not vstore.has_paper(p["id"])}

        # 缺语料的：只要在语料里没有就补（保证关键词始终能算）
        missing_corpus_ids = {p["id"] for p in papers if p["id"] not in self._kw_corpus}

        need_ids = missing_vec_ids | missing_corpus_ids
        if not need_ids:
            return

        if missing_vec_ids:
            print(f"[store] Re-indexing {len(missing_vec_ids)} papers missing from Chroma ...")
        if missing_corpus_ids:
            print(f"[store] Registering {len(missing_corpus_ids)} papers into keyword corpus ...")

        for paper in papers:
            if paper["id"] not in need_ids:
                continue
            pdf_path = FILES_DIR / f"{paper['id']}.pdf"
            if not pdf_path.exists():
                print(f"[store] PDF missing for {paper['id']}, skipping")
                continue
            try:
                cleaned = clean_text(read_pdf_text(pdf_path, max_pages=0))
                # 语料（无论是否缺向量都补一次）
                self._register_corpus(paper["id"], cleaned)
                # 向量（只补缺失的）
                if paper["id"] in missing_vec_ids and vstore is not None:
                    self._index_paper(paper, cleaned)
            except Exception as e:
                print(f"[store] Re-index failed for {paper['id']}: {e}")

        # 刚刚大量注册了语料，全局刷新一次关键词（自动关键词才会被覆盖）
        try:
            n = self.refresh_keywords_for_all()
            if n > 0:
                print(f"[store] Auto keywords refreshed for {n} papers")
        except Exception as e:
            print(f"[store] refresh_keywords_for_all failed: {e}")

    # ------------------------------------------------------------------ #
    # 添加 PDF
    # ------------------------------------------------------------------ #

    def add_pdf(
        self,
        filename: str,
        raw: bytes,
        recompute: bool = True,
        ocr_mode: str | None = None,
        parser: str | None = None,
    ) -> str:
        paper_id = uuid.uuid4().hex[:10]
        pdf_path = FILES_DIR / f"{paper_id}.pdf"
        try:
            pdf_path.write_bytes(raw)
        except Exception as e:
            raise ValueError(f"PDF 文件保存失败: {e}")

        try:
            full_text = read_pdf_text(
                pdf_path, max_pages=0, mode_override=ocr_mode, parser=parser
            )
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
            abstract = self._build_best_abstract(cleaned)
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

        declared_kws = extract_keywords_block(cleaned)
        keywords_source = "declared" if declared_kws else "auto"

        paper: dict[str, Any] = {
            "id": paper_id,
            "title": title,
            "display_title": display_title,
            "abstract": abstract,
            "first_sentence": first_sentence,
            "keywords": declared_kws,   # 先占位，下面如有 auto 再覆盖
            "_keywords_source": keywords_source,
            "_ocr_mode": (ocr_mode or "").strip().lower() or "auto",
            "_parser": (parser or "").strip().lower() or "default",
            "filename": filename,
            "field": "Processing...",
            "confidence": 0.0,
            "cluster": 0,
            "pos": [0.0, 0.0, 0.0],
            "size": 3.0,
        }

        # 注册语料（先注册再抽关键词，保证新论文自身也进入 TF-IDF）
        self._register_corpus(paper_id, cleaned)

        # 论文没有自带 Keywords 段时，用全库 TF-IDF + MMR 提取
        if not declared_kws:
            try:
                kws = self._auto_keywords(
                    paper_id,
                    top_k=8,
                    title=title,
                    abstract=abstract,
                )
                if not kws and title:
                    kws = [title[:20]]
                paper["keywords"] = kws
            except Exception as e:
                print(f"[store] Warning: keyword extraction failed: {e}")
                paper["keywords"] = [title[:20]] if title else []

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
                self._unregister_corpus(paper_id)
                if pdf_path.exists():
                    pdf_path.unlink()
                raise ValueError(f"图表计算失败: {e}")

        # 新增一篇论文后语料扩大，给其它自动关键词的论文也刷新一下
        # （对老论文中显式声明的 Keywords 不会动）
        if recompute:
            try:
                self.refresh_keywords_for_all()
            except Exception as e:
                print(f"[store] Warning: refresh_keywords_for_all after add failed: {e}")

        return paper_id

    # ------------------------------------------------------------------ #
    # Inbox 批量导入
    # ------------------------------------------------------------------ #

    def ingest_from_inbox(self) -> int:
        if not INBOX_DIR.exists():
            return 0

        processed_dir = INBOX_DIR / "_processed"
        try:
            processed_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # 不阻断主流程；后面如果 move 失败会退化为保留原文件
            pass

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
                # 导入后把 inbox 原文件移到 _processed，避免重启时重复导入
                try:
                    target = processed_dir / pdf_path.name
                    if target.exists():
                        stem = pdf_path.stem
                        suffix = pdf_path.suffix or ".pdf"
                        i = 1
                        while True:
                            cand = processed_dir / f"{stem}__dup{i}{suffix}"
                            if not cand.exists():
                                target = cand
                                break
                            i += 1
                    pdf_path.replace(target)
                except Exception as move_err:
                    print(f"[store] Warning: failed to archive inbox file {pdf_path.name}: {move_err}")
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

    def get_paper(self, paper_id: str) -> dict[str, Any] | None:
        """线程安全地取一份论文元数据副本。不存在返回 None。"""
        with self._lock:
            for p in self._papers:
                if str(p.get("id") or "") == paper_id:
                    return dict(p)
        return None

    def get_chunk_context(
        self, chunk_id: str, window: int = 1
    ) -> dict[str, Any] | None:
        """
        取某个 chunk 及其前后相邻 chunk 的文本（按 index 邻近），
        用于防止检索片段被切断导致断章取义。
        返回 {paper_id, center_index, chunks:[{chunk_id,index,text}]} 或 None。
        """
        try:
            vstore = self._ensure_vstore()
        except Exception as e:
            print(f"[store] get_chunk_context vstore error: {e}")
            return None

        info = vstore.get_chunk(chunk_id)
        if not info or not info.get("paper_id"):
            return None

        paper_id = info["paper_id"]
        center = int(info.get("index", 0))
        ordered = vstore.get_paper_chunks(paper_id)
        sel = [c for c in ordered if abs(int(c["index"]) - center) <= max(0, window)]
        if not sel:
            sel = [info]
        return {"paper_id": paper_id, "center_index": center, "chunks": sel}

    def get_cleaned_fulltext(self, paper_id: str) -> str:
        """读取并清洗某篇论文的全文（供 LLM 摘要润色等上下文使用）。"""
        pdf_path = FILES_DIR / f"{paper_id}.pdf"
        if not pdf_path.exists():
            return ""
        try:
            return clean_text(read_pdf_text(pdf_path, max_pages=0))
        except Exception as e:
            print(f"[store] get_cleaned_fulltext failed for {paper_id}: {e}")
            return ""

    def get_pages_text(self, paper_id: str, pages: list[int]) -> list[tuple[int, str]]:
        """按页码（1-based）读取指定页的清洗文本。越界或文件缺失则返回空串。

        返回 [(page_num, cleaned_text), ...]，按输入顺序去重并保留。
        """
        pdf_path = FILES_DIR / f"{paper_id}.pdf"
        if not pdf_path.exists() or not pages:
            return []
        try:
            raw_pages = read_pdf_pages(pdf_path)
        except Exception as e:
            print(f"[store] read_pdf_pages failed for {paper_id}: {e}")
            return []

        total = len(raw_pages)
        if total == 0:
            return []

        out: list[tuple[int, str]] = []
        seen: set[int] = set()
        for p in pages:
            if p in seen:
                continue
            seen.add(p)
            if 1 <= p <= total:
                cleaned = clean_text(raw_pages[p - 1])
                out.append((p, cleaned))
            else:
                out.append((p, ""))
        return out

    def get_page_count(self, paper_id: str) -> int:
        """返回 PDF 页数（失败返回 0）。"""
        pdf_path = FILES_DIR / f"{paper_id}.pdf"
        if not pdf_path.exists():
            return 0
        try:
            return len(read_pdf_pages(pdf_path))
        except Exception:
            return 0

    @staticmethod
    def _norm_match_text(text: str) -> str:
        s = clean_text(text or "").lower()
        # 仅保留中英文与数字，去掉空白/标点，便于跨行、断词匹配
        s = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", s)
        return s

    def locate_snippet_page(self, paper_id: str, snippet: str) -> int | None:
        """
        粗定位 snippet 所在页（1-based）。
        优先做规范化后的子串命中；命中不到返回 None。
        """
        pdf_path = FILES_DIR / f"{paper_id}.pdf"
        if not pdf_path.exists():
            return None
        target = self._norm_match_text(snippet)
        if len(target) < 12:
            return None
        # 过长片段只取前中后三段做匹配，降低误差与开销
        probes = [target]
        if len(target) > 80:
            probes = [
                target[:60],
                target[max(0, len(target) // 2 - 30): max(0, len(target) // 2 + 30)],
                target[-60:],
            ]
        try:
            pages = read_pdf_pages(pdf_path)
        except Exception:
            return None
        for i, page_text in enumerate(pages, 1):
            pn = self._norm_match_text(page_text)
            if not pn:
                continue
            hit_count = sum(1 for p in probes if p and p in pn)
            if hit_count >= max(1, len(probes) // 2):
                return i

        # 二级兜底：按关键词命中数打分，返回最可能页
        raw = clean_text(snippet or "")
        zh_terms = re.findall(r"[\u4e00-\u9fff]{3,}", raw)
        en_terms = re.findall(r"[A-Za-z][A-Za-z0-9_-]{4,}", raw)
        terms = [*zh_terms, *en_terms]
        # 去噪、去重、长度优先
        terms = [t.lower() for t in terms if t.lower() not in {"paper_id", "chunk_id"}]
        terms = sorted(list(dict.fromkeys(terms)), key=len, reverse=True)[:10]
        if not terms:
            return None
        best_page, best_score = None, 0
        for i, page_text in enumerate(pages, 1):
            pt = clean_text(page_text or "").lower()
            if not pt:
                continue
            score = sum(1 for t in terms if t in pt)
            if score > best_score:
                best_score = score
                best_page = i
        # 至少命中 2 个关键词才认定，避免误跳
        return best_page if best_score >= 2 else None

    def set_llm_summary(self, paper_id: str, summary: str) -> bool:
        """写入 / 清空某篇论文的 LLM 润色摘要。返回是否命中论文。"""
        with self._lock:
            for p in self._papers:
                if str(p.get("id") or "") == paper_id:
                    if summary:
                        p["llm_summary"] = summary
                    else:
                        p.pop("llm_summary", None)
                    self._save_db()
                    return True
        return False

    def reindex_all_papers(self) -> dict[str, Any]:
        """
        对全库所有论文重跑 embedding 与 chunk 向量。
        换 embedding 模型后必须调一次（维度/语义都变了）。

        流程：
          1. 清空 Chroma 两张表（维度可能已变）
          2. 对每篇论文：重读 PDF → clean_text → 重新 _index_paper
          3. 更新 emb_model 字段

        注意：只重跑向量，不重跑文本解析（想连文本一起重跑请用 reprocess_paper）。
        """
        with self._lock:
            papers = list(self._papers)
        if not papers:
            return {"total": 0, "success": 0, "failed": 0, "model": self._model_name}

        # 清空向量表
        try:
            self._ensure_vstore().reset_all()
        except Exception as e:
            print(f"[store] Warning: reset vectors failed: {e}")

        # 让 model 真正触发一次加载（可能下载几 GB）
        try:
            self._ensure_model()
        except Exception as e:
            raise RuntimeError(f"加载 embedding 模型失败: {e}")

        success, failed = 0, 0
        for i, paper in enumerate(papers, 1):
            pdf_path = FILES_DIR / f"{paper['id']}.pdf"
            if not pdf_path.exists():
                print(f"[reindex] [{i}/{len(papers)}] PDF 缺失，跳过: {paper['id']}")
                failed += 1
                continue
            try:
                print(f"[reindex] [{i}/{len(papers)}] {paper.get('filename', paper['id'])}")
                cleaned = clean_text(read_pdf_text(pdf_path, max_pages=0))
                if not cleaned or len(cleaned) < 50:
                    failed += 1
                    continue
                self._index_paper(paper, cleaned)
                success += 1
            except Exception as e:
                print(f"[reindex] failed for {paper.get('filename', paper['id'])}: {e}")
                failed += 1

        # 记录此次使用的 embedding 模型
        with self._lock:
            self._last_emb_model = self._model_name
            self._save_db()

        return {
            "total": len(papers),
            "success": success,
            "failed": failed,
            "model": self._model_name,
        }

    def reprocess_paper(
        self,
        paper_id: str,
        ocr_mode: str | None = None,
        parser: str | None = None,
        recompute_geometry: bool = False,
    ) -> dict[str, Any]:
        """
        按 ocr_mode（off/auto/force）重新识别指定文档的文本，并刷新：
          - 清洗后全文 / abstract / first_sentence
          - 关键词（仅当原本是 auto 推断的才覆盖；显式 Keywords 保留）
          - 向量库 chunks（先删再建）
          - 关键词语料
        保留：id/filename/display_title/title（标题不回退，除非解析出更好的）、
             人工 polish_summary 等用户数据。

        recompute_geometry 默认 False：只刷文本/向量，不重跑聚类与坐标，
        避免整张星系图因为一篇文献刷新而抖动。
        """
        with self._lock:
            idx = next(
                (i for i, p in enumerate(self._papers) if str(p.get("id") or "") == paper_id),
                -1,
            )
            if idx < 0:
                raise ValueError(f"paper {paper_id} not found")
            paper = self._papers[idx]

        pdf_path = FILES_DIR / f"{paper_id}.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF missing for {paper_id}")

        full_text = read_pdf_text(
            pdf_path, max_pages=0, mode_override=ocr_mode, parser=parser
        )
        cleaned = clean_text(full_text)
        if not cleaned or len(cleaned) < 50:
            raise ValueError(
                f"重新识别失败：提取文本过短（长度 {len(cleaned) if cleaned else 0}），"
                "可能是扫描件且 OCR 未正确安装或语言包缺失。"
            )

        display_title = paper.get("display_title") or safe_stem(paper.get("filename", ""))
        new_title_candidate = extract_title_from_text(cleaned, display_title)
        # 新标题明显比 display_title 长才采用，避免把标题回退成文件名
        new_title = paper.get("title") or display_title
        if new_title_candidate and len(new_title_candidate) > max(len(display_title), 8):
            new_title = new_title_candidate

        abstract = self._build_best_abstract(cleaned)
        first_sentence = ""
        try:
            m = re.search(r"[^.!?。！？]+[.!?。！？]", cleaned)
            first_sentence = m.group(0).strip() if m else cleaned[:100].strip() + "..."
        except Exception:
            first_sentence = cleaned[:100].strip() + "..."

        declared_kws = extract_keywords_block(cleaned)
        prev_kw_source = str(paper.get("_keywords_source") or "auto")
        if declared_kws:
            paper["keywords"] = declared_kws
            paper["_keywords_source"] = "declared"
        elif prev_kw_source != "declared":
            # 原本就是自动推断的才刷新；用户若是 declared（论文自带）保留
            try:
                kws = self._auto_keywords(
                    paper_id,
                    top_k=8,
                    title=new_title,
                    abstract=abstract,
                )
                if not kws and new_title:
                    kws = [new_title[:20]]
                paper["keywords"] = kws
                paper["_keywords_source"] = "auto"
            except Exception as e:
                print(f"[store] Warning: auto keywords refresh failed: {e}")

        paper["title"] = new_title
        paper["display_title"] = display_title
        paper["abstract"] = abstract
        paper["first_sentence"] = first_sentence
        paper["_ocr_mode"] = (ocr_mode or "").lower().strip() or paper.get("_ocr_mode") or "auto"
        paper["_parser"] = (parser or "").lower().strip() or paper.get("_parser") or "default"

        # 向量库：先删旧，再建新
        try:
            vstore = self._ensure_vstore()
            vstore.delete_paper(paper_id)
        except Exception as e:
            print(f"[store] Warning: failed to purge old vectors for {paper_id}: {e}")

        # 关键词语料：先注销再注册，保证 TF-IDF 用最新文本
        self._unregister_corpus(paper_id)
        self._register_corpus(paper_id, cleaned)

        try:
            self._index_paper(paper, cleaned)
        except Exception as e:
            print(f"[store] Warning: reindex failed for {paper_id}: {e}")

        with self._lock:
            self._papers[idx] = paper
            try:
                if recompute_geometry:
                    self._recompute_locked()
                else:
                    self._save_db()
            except Exception as e:
                print(f"[store] Warning: save after reprocess failed: {e}")

        return {
            "paper_id": paper_id,
            "ocr_mode": paper["_ocr_mode"],
            "parser": paper["_parser"],
            "text_len": len(cleaned),
            "abstract_len": len(abstract or ""),
            "keywords": list(paper.get("keywords") or []),
        }

    def delete_paper(self, paper_id: str, delete_source_files: bool = True) -> bool:
        """
        删除指定论文（元数据 + PDF 文件 + 向量库记录），并重算可视化状态。
        delete_source_files=True 时会额外删除 inbox/_processed 里的同名源文件，
        彻底阻断“重启后从 inbox 回流”。
        返回是否成功删除到论文记录。
        """
        with self._lock:
            idx = next((i for i, p in enumerate(self._papers) if str(p.get("id") or "") == paper_id), -1)
            if idx < 0:
                return False
            paper = self._papers.pop(idx)
            filename = str(paper.get("filename") or "").strip()

            # 删除向量库记录（失败不阻断）
            try:
                self._ensure_vstore().delete_paper(paper_id)
            except Exception as e:
                print(f"[store] Warning: failed to delete vectors for {paper_id}: {e}")

            # 从全库关键词语料中移除
            self._unregister_corpus(paper_id)

            # 删除 PDF 文件（失败不阻断）
            try:
                pdf_path = FILES_DIR / f"{paper_id}.pdf"
                if pdf_path.exists():
                    pdf_path.unlink()
            except Exception as e:
                print(f"[store] Warning: failed to delete pdf for {paper_id}: {e}")

            # 可选：同步清理 inbox 里的同名源文件，避免重启时回流
            if delete_source_files and filename:
                for p in [INBOX_DIR / filename, INBOX_DIR / "_processed" / filename]:
                    try:
                        if p.exists():
                            p.unlink()
                    except Exception as e:
                        print(f"[store] Warning: failed to delete source file {p}: {e}")

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
