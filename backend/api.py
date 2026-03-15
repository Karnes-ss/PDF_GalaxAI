from __future__ import annotations

"""
API 路由层。

错误状态码（status 字段）约定：
  "ok"          - 成功
  "no_match"    - 检索无命中（相关度不足）
  "empty_db"    - 向量库为空（未上传文档）
  "model_error" - 向量模型加载/初始化失败
  "llm_error"   - LLM 调用失败
"""

import re
import os
from typing import Any

import httpx
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

from clustering import fallback_pos
from config import FILES_DIR
from text_processing import safe_stem

# ------------------------------------------------------------------ #
# 请求体模型
# ------------------------------------------------------------------ #

class AnalyzeBody(BaseModel):
    vectors: list[list[float]] | None = None


class ChatBody(BaseModel):
    question: str
    provider: str | None = None  # local | gemini


class SearchBody(BaseModel):
    query: str
    top_k: int = 5
    paper_id: str | None = None   # 可选：限定在某篇论文内检索


# ------------------------------------------------------------------ #
# 错误提示文案（前端可直接展示）
# ------------------------------------------------------------------ #

_STATUS_MSG: dict[str, str] = {
    "no_match":    "未在本地文献库中找到相关内容，请尝试换一种提问方式或上传更多 PDF。",
    "empty_db":    "文献库为空，请先上传 PDF 文档，再来提问。",
    "model_error": "向量模型初始化失败，请检查后端日志。",
    "llm_error":   "AI 响应失败，请检查本地模型或云端模型配置。",
}

_DEFAULT_PROVIDER = (os.getenv("SCHOLAR_LLM_PROVIDER") or "local").strip().lower()
_OLLAMA_MODEL = (os.getenv("SCHOLAR_OLLAMA_MODEL") or "qwen2.5:3b").strip()
_GEMINI_MODEL = (os.getenv("SCHOLAR_GEMINI_MODEL") or "gemini-2.5-flash").strip()


def _load_env_file_var(key: str) -> str:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return ""
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                if k.strip() == key:
                    return v.strip().strip('"').strip("'")
    except Exception:
        return ""
    return ""


def _current_gemini_api_key() -> str:
    # 优先读进程环境；再兜底读 backend/.env（便于本地开发）
    key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or _load_env_file_var("GEMINI_API_KEY")
    )
    return (key or "").strip()


async def _call_ollama(prompt: str, question: str) -> str:
    answer = None
    payload_with_ctx = {
        "model": _OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "你是一个严谨的学术助手。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
    }
    payload_no_ctx = {
        "model": _OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": "你是一个严谨的学术助手。"},
            {"role": "user", "content": f"请简洁回答这个问题：{question[:500]}"},
        ],
        "temperature": 0.3,
    }
    timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=30.0)
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        for payload in (payload_with_ctx, payload_no_ctx):
            for attempt in range(2):
                resp = await client.post(
                    "http://127.0.0.1:11434/v1/chat/completions",
                    json=payload,
                )
                if resp.status_code == 502 and attempt == 0:
                    continue
                if resp.status_code == 502:
                    break
                resp.raise_for_status()
                data = resp.json()
                answer = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
                if answer:
                    break
            if answer:
                break
    if not answer:
        raise RuntimeError("Ollama 返回空内容")
    return answer


async def _call_gemini(prompt: str) -> str:
    api_key = _current_gemini_api_key()
    if not api_key:
        raise RuntimeError("未配置 GEMINI_API_KEY")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{_GEMINI_MODEL}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3},
    }
    timeout = httpx.Timeout(connect=15.0, read=120.0, write=30.0, pool=30.0)
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    parts = (
        data.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [])
    )
    answer = "".join([str(p.get("text") or "") for p in parts]).strip()
    if not answer:
        raise RuntimeError("Gemini 返回空内容")
    return answer


# ------------------------------------------------------------------ #
# 应用工厂
# ------------------------------------------------------------------ #

def create_app(store) -> FastAPI:
    app = FastAPI()

    @app.on_event("startup")
    def _startup() -> None:
        # 1. 扫描 inbox，导入新 PDF
        store.ingest_from_inbox()
        # 2. 对 JSON 中已有论文补全 Chroma 索引（迁移兼容）
        store.ensure_all_indexed()
        # 3. 若已有论文但运行时向量缓存为空，重建聚类
        if store._papers and store._vectors is None:
            with store._lock:
                store._recompute_locked()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---------------------------------------------------------------- #
    # 健康检查
    # ---------------------------------------------------------------- #

    @app.get("/api/health")
    def api_health() -> dict[str, Any]:
        return {"ok": True}

    @app.get("/api/llm/status")
    def api_llm_status() -> dict[str, Any]:
        return {
            "default_provider": _DEFAULT_PROVIDER,
            "ollama_model": _OLLAMA_MODEL,
            "gemini_model": _GEMINI_MODEL,
            "gemini_key_configured": bool(_current_gemini_api_key()),
        }

    # ---------------------------------------------------------------- #
    # 论文列表（可视化用）
    # ---------------------------------------------------------------- #

    @app.get("/api/papers")
    def api_papers() -> dict[str, Any]:
        with store._lock:
            # 若论文缺少 pos/cluster，触发重算
            if store._papers and any(
                ("pos" not in p) or ("cluster" not in p) for p in store._papers
            ):
                store._recompute_locked()

            papers: list[dict[str, Any]] = []
            for p in store._papers:
                paper_id = str(p.get("id") or "")
                pos = p.get("pos")
                if not isinstance(pos, (list, tuple)) or len(pos) < 3:
                    pos = fallback_pos(paper_id)
                papers.append(
                    {
                        "id": paper_id,
                        "title": str(p.get("title", "Untitled")),
                        "displayTitle": str(
                            p.get("display_title") or safe_stem(str(p.get("filename") or ""))
                        ),
                        "firstSentence": str(
                            p.get("first_sentence")
                            or p.get("abstract", "")[:120]
                            or "No content available."
                        ),
                        "abstract": str(p.get("abstract", "")),
                        "filename": str(p.get("filename", "")),
                        "field": str(p.get("field", "Uncategorized")),
                        "confidence": float(p.get("confidence", 0.0)),
                        "size": float(p.get("size", 3.0)),
                        "pos": (float(pos[0]), float(pos[1]), float(pos[2])),
                        "color": str(p.get("color", "#60a5fa")),
                        "keywords": p.get("keywords", []),
                        "cluster": int(p.get("cluster", 0)),
                    }
                )

            # 计算论文间相似度边（用运行时缓存 _vectors）
            edges: list[dict[str, Any]] = []
            if store._vectors is not None and len(store._papers) >= 2:
                try:
                    sims = cosine_similarity(store._vectors)
                    np.fill_diagonal(sims, 0.0)
                    used: set[tuple[str, str]] = set()
                    topk = min(4, len(store._papers) - 1)
                    for i, pi in enumerate(store._papers):
                        src = str(pi.get("id") or "")
                        for j in np.argsort(sims[i])[::-1][:topk]:
                            w = float(sims[i, j])
                            if w < 0.20:
                                continue
                            dst = str(store._papers[int(j)].get("id") or "")
                            a, b = (src, dst) if src <= dst else (dst, src)
                            if (a, b) in used:
                                continue
                            used.add((a, b))
                            t = (
                                "intra"
                                if int(pi.get("cluster", 0))
                                == int(store._papers[int(j)].get("cluster", 0))
                                else "bridge"
                            )
                            edges.append(
                                {"source": src, "target": dst, "weight": w, "type": t}
                            )
                except Exception as e:
                    print(f"[api] Edge computation error: {e}")

        return {"papers": papers, "edges": edges}

    # ---------------------------------------------------------------- #
    # 其他列表 / 触发接口
    # ---------------------------------------------------------------- #

    @app.get("/api/pdfs")
    def api_pdfs() -> dict[str, Any]:
        return {"pdfs": store.list_pdfs()}

    @app.post("/api/scan")
    def api_scan() -> dict[str, Any]:
        count = store.ingest_from_inbox()
        return {"added": count, "total": len(store.list_pdfs())}

    @app.post("/api/upload")
    async def api_upload(file: UploadFile = File(...)) -> dict[str, Any]:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="只支持 .pdf 文件")
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="文件为空")
        try:
            pdf_id = store.add_pdf(file.filename, raw)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
            )
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            print(f"[api] Upload failed: {e}")
            raise HTTPException(status_code=500, detail="内部错误")
        return {"success": True, "pdf_id": pdf_id}

    @app.post("/api/papers/upload")
    async def api_papers_upload(file: UploadFile = File(...)) -> dict[str, Any]:
        return await api_upload(file)

    @app.delete("/api/papers/{paper_id}")
    def api_delete_paper(paper_id: str) -> dict[str, Any]:
        ok = store.delete_paper(paper_id)
        if not ok:
            raise HTTPException(status_code=404, detail="paper not found")
        return {"success": True, "paper_id": paper_id}

    @app.post("/api/analyze")
    def api_analyze(body: AnalyzeBody) -> dict[str, Any]:
        return store.analyze()

    # ---------------------------------------------------------------- #
    # /api/search — Chunk 级语义检索
    # ---------------------------------------------------------------- #

    @app.post("/api/search")
    def api_search(body: SearchBody) -> dict[str, Any]:
        """
        输入 query（和可选的 paper_id）→ 返回 topK 最相关 chunks。

        响应字段：
          results : list of {chunk_id, paper_id, score, snippet}
          status  : "ok" | "no_match" | "empty_db" | "model_error"
          message : 状态对应的提示文案（no_match / error 时非空）
        """
        q = (body.query or "").strip()
        if not q:
            raise HTTPException(status_code=400, detail="query 不能为空")

        results, stat = store.search_chunks(
            q, top_k=body.top_k, paper_id=body.paper_id
        )
        return {
            "results": results,
            "status": stat,
            "message": _STATUS_MSG.get(stat, "") if stat != "ok" else "",
        }

    # ---------------------------------------------------------------- #
    # /api/query — RAG 对话（检索 + LLM 生成）
    # ---------------------------------------------------------------- #

    @app.post("/api/query")
    async def api_query(body: ChatBody) -> dict[str, Any]:
        """
        RAG 全链路：
          1. Chunk 检索 → 找到最相关段落
          2. Prompt 组装 → 将 chunks 作为上下文
          3. LLM 生成   → 调用本地 Ollama qwen2.5:3b
          4. 返回答案 + 引用（paperId + chunkId + snippet）

        响应字段：
          answer  : LLM 生成的回答（或错误提示）
          cites   : list of {paper_id, chunk_id, snippet}
          status  : "ok" | "no_match" | "empty_db" | "model_error" | "llm_error"
          message : 非 ok 时的用户提示文案
        """
        question = (body.question or "").strip()
        if not question:
            return {"answer": "请输入您的问题。", "cites": [], "status": "ok", "message": ""}

        # ── 1. Chunk 检索（优先命中文献名）──────────────────────────────
        def _norm(s: str) -> str:
            s = (s or "").lower().strip()
            if s.endswith(".pdf"):
                s = s[:-4]
            return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", s)

        q_norm = _norm(question)
        target_paper_id: str | None = None
        if q_norm:
            with store._lock:
                candidates = list(store._papers)
            best_len = 0
            for p in candidates:
                keys = [
                    str(p.get("display_title") or ""),
                    str(p.get("title") or ""),
                    str(p.get("filename") or ""),
                ]
                for k in keys:
                    nk = _norm(k)
                    if nk and nk in q_norm and len(nk) > best_len:
                        best_len = len(nk)
                        target_paper_id = str(p.get("id") or "")

        # 如果问题里点名了文献，则先在该文献内检索；无命中再回退全库
        if target_paper_id:
            chunks, stat = store.search_chunks(
                question, top_k=6, paper_id=target_paper_id, min_score=0.05
            )
            if stat != "ok":
                chunks, stat = store.search_chunks(question, top_k=5)
        else:
            chunks, stat = store.search_chunks(question, top_k=5)

        if stat != "ok":
            return {
                "answer": _STATUS_MSG.get(stat, "检索失败"),
                "cites": [],
                "status": stat,
                "message": _STATUS_MSG.get(stat, ""),
            }

        # ── 2. Prompt 组装 ────────────────────────────────────────────
        def _sanitize_for_llm(text: str, max_len: int) -> str:
            # 去除控制字符，避免某些 PDF 脏文本触发 Ollama 502
            s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", text or "")
            s = s.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
            s = re.sub(r"\s{2,}", " ", s).strip()
            return s[:max_len]

        def _is_noisy_text(text: str) -> bool:
            s = (text or "")
            if not s:
                return True
            q_ratio = s.count("?") / max(1, len(s))
            if q_ratio >= 0.03:
                return True
            if "a?��" in s or "e???" in s or "�" in s:
                return True
            return False

        context_parts = []
        high_quality_chunks = [
            c for c in chunks
            if float(c.get("quality", 1.0)) >= 0.75
            and not _is_noisy_text(str(c.get("snippet", "")))
        ]
        if high_quality_chunks:
            source_chunks = high_quality_chunks
        else:
            # 对“点名文献”问题，如果 chunk 全是噪声，则不要再把噪声喂给模型
            source_chunks = [] if target_paper_id else chunks
        for c in source_chunks[:3]:
            snippet = _sanitize_for_llm(str(c.get("snippet", "")), max_len=220)
            context_parts.append(
                f"【来源 paper_id={c['paper_id']} chunk_id={c['chunk_id']}】\n{snippet}"
            )

        # 若目标文献命中但 chunk 质量偏低，补充使用论文元信息（title/abstract/keywords）作为上下文
        if target_paper_id and not high_quality_chunks:
            target_meta = None
            with store._lock:
                for p in store._papers:
                    if str(p.get("id") or "") == target_paper_id:
                        target_meta = p
                        break
            if target_meta:
                title = _sanitize_for_llm(str(target_meta.get("title") or ""), 300)
                abstract = _sanitize_for_llm(str(target_meta.get("abstract") or ""), 1200)
                keywords = ", ".join([str(k) for k in (target_meta.get("keywords") or [])[:12]])
                context_parts.append(
                    "【来源 元信息】\n"
                    f"标题: {title}\n"
                    f"关键词: {keywords}\n"
                    f"摘要: {abstract}"
                )
        context = "\n\n".join(context_parts)

        prompt = (
            "你是一个专业的论文分析助手。请基于以下参考资料回答用户问题，"
            "引用时注明来源的 paper_id。若资料中无相关信息，请直接说明。\n\n"
            f"参考资料：\n{context}\n\n"
            f"用户问题：\n{question}"
        )
        if target_paper_id:
            prompt += f"\n\n补充要求：用户已指定目标文献 paper_id={target_paper_id}，请优先围绕该文献回答。"

        # ── 3. LLM 调用（本地 / 云端可切换）───────────────────────────────
        provider = "local"
        try:
            requested_provider = (body.provider or _DEFAULT_PROVIDER or "local").strip().lower()
            provider = requested_provider if requested_provider in {"local", "gemini"} else "local"
            if provider == "gemini":
                answer = await _call_gemini(prompt)
            else:
                answer = await _call_ollama(prompt, _sanitize_for_llm(question, 500))

            # ── 4. 返回答案 + 引用 ─────────────────────────────────────
            cite_details = [
                {
                    "paper_id": c["paper_id"],
                    "chunk_id": c["chunk_id"],
                    "snippet": c["snippet"],
                }
                for c in chunks
            ]
            cites = list(dict.fromkeys([c["paper_id"] for c in cite_details]))
            return {
                "answer": answer,
                "cites": cites,
                "cite_details": cite_details,
                "provider_used": provider,
                "status": "ok",
                "message": "",
            }

        except Exception as e:
            print(f"[api] LLM Error: {e}")
            return {
                "answer": _STATUS_MSG["llm_error"] + f"\n错误详情: {e}",
                "cites": [],
                "provider_used": provider,
                "status": "llm_error",
                "message": _STATUS_MSG["llm_error"],
            }

    # ---------------------------------------------------------------- #
    # 文件服务
    # ---------------------------------------------------------------- #

    @app.get("/files/{paper_id}.pdf")
    def api_files(paper_id: str) -> FileResponse:
        pdf_path = FILES_DIR / f"{paper_id}.pdf"
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail="PDF not found")
        return FileResponse(
            str(pdf_path),
            media_type="application/pdf",
            headers={"Content-Disposition": f'inline; filename="{paper_id}.pdf"'},
        )

    @app.get("/api/visualization")
    def api_visualization() -> dict[str, Any]:
        return store.visualization()

    return app
