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

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

from agent import build_agent, classify_intent, should_use_agent
from clustering import fallback_pos
from config import FILES_DIR
from llm_client import (
    call_llm,
    call_llm_with_tools,
    encode_image_to_base64,
    is_vision_capable,
    test_connection,
)
from models_registry import ModelsRegistry
from text_processing import safe_stem

# 全局模型注册表（内置 + 自定义）
_registry = ModelsRegistry()

# ------------------------------------------------------------------ #
# 请求体模型
# ------------------------------------------------------------------ #

class AnalyzeBody(BaseModel):
    vectors: list[list[float]] | None = None


class ChatTurn(BaseModel):
    role: str  # 'user' | 'ai' | 'assistant'
    text: str


class ChatBody(BaseModel):
    question: str
    provider: str | None = None  # 模型 id：local | gemini | cm_xxx
    history: list[ChatTurn] | None = None   # 最近几轮对话（前端维护）
    # 模式：'auto'（默认，按检索质量自适应） | 'chat'（强制通用对话，不检索） | 'rag'（强制基于文献，尽力引用）
    mode: str | None = None


class SearchBody(BaseModel):
    query: str
    top_k: int = 5
    paper_id: str | None = None   # 可选：限定在某篇论文内检索


class PolishBody(BaseModel):
    provider: str | None = None   # 模型 id；缺省走 _DEFAULT_PROVIDER


class ModelCreateBody(BaseModel):
    label: str
    protocol: str             # openai | gemini
    base_url: str | None = None
    model: str
    api_key: str | None = None
    # 网络模式：''/'auto' | 'direct' | 'http(s)://...' | 'socks5(h)://...'
    proxy: str | None = None


class ModelUpdateBody(BaseModel):
    label: str | None = None
    protocol: str | None = None
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None   # 空串表示不修改
    proxy: str | None = None     # 显式传入（包括空串）才更新


class ModelTestBody(BaseModel):
    protocol: str | None = None
    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None
    proxy: str | None = None


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


def _resolve_provider(requested: str | None) -> tuple[str, dict[str, Any]]:
    """
    将前端传入的 provider（模型 id）解析为 (model_id, config)。
    如果不存在，回退到默认 provider，再回退到 local。
    返回的 config 包含 api_key 明文（供 call_llm 使用）。
    """
    candidates: list[str] = []
    req = (requested or "").strip()
    if req:
        candidates.append(req)
    if _DEFAULT_PROVIDER and _DEFAULT_PROVIDER not in candidates:
        candidates.append(_DEFAULT_PROVIDER)
    if "local" not in candidates:
        candidates.append("local")

    for mid in candidates:
        cfg = _registry.get_config(mid)
        if cfg:
            return mid, cfg
    # 最坏兜底：local 都没有时，强制返回 local 默认
    cfg = _registry.get_config("local")
    return "local", cfg or {}


# ------------------------------------------------------------------ #
# Meta 问题识别：关于"文献库本身"的问题，直接用真实数据作答
# ------------------------------------------------------------------ #

_META_COUNT_PATTERNS = [
    # "多少篇 / 几篇 / 多少份 / 总共多少 / 一共多少"
    re.compile(r"(多少|几|总共|一共)[\s\S]{0,6}(篇|份|个|文献|论文|文章|pdf|ppt)", re.IGNORECASE),
    re.compile(r"(篇|份|个|文献|论文|文章|pdf)[\s\S]{0,6}(多少|几|总共|一共)", re.IGNORECASE),
    re.compile(r"how\s+many\s+(papers?|files?|documents?|pdfs?)", re.IGNORECASE),
    # "提供给你十篇 / 上传了.*篇 / 给了你.*篇"（用户在强调他上传的数量）
    re.compile(r"(提供|上传|给|传|放).{0,8}(\d+|一|二|三|四|五|六|七|八|九|十|几|多少|所有).{0,4}(篇|份|个)"),
]

_META_LIST_PATTERNS = [
    re.compile(r"(都有|哪些|列出|列表|清单|有什么|是什么)[\s\S]{0,6}(文献|论文|文章|文件|pdf)", re.IGNORECASE),
    re.compile(r"(文献|论文|文章|文件|pdf|库)[\s\S]{0,6}(都有|哪些|列表|清单|目录)", re.IGNORECASE),
    re.compile(r"(所有|全部|每一?(篇|份|个))[\s\S]{0,4}(文献|论文|文章)", re.IGNORECASE),
    re.compile(r"list\s+(all\s+)?(papers?|files?|documents?)", re.IGNORECASE),
    re.compile(r"(what|which)\s+(papers?|files?|documents?)", re.IGNORECASE),
]

_META_GREETING_PATTERNS = [
    re.compile(r"^\s*(你好|您好|在吗|hi|hello|hey)[!！。.\s]*$", re.IGNORECASE),
]

# 身份 / 自我介绍 / 能力类闲聊 → 必须跳过 RAG，避免误引用文献
_META_IDENTITY_PATTERNS = [
    re.compile(r"你\s*(是谁|叫(什么|啥)|是(什么|啥))", re.IGNORECASE),
    re.compile(r"(介绍|说说)[\s\S]{0,4}(你自己|一下你)", re.IGNORECASE),
    re.compile(r"你(能|可以|会)(做|干|帮)(什么|啥|些什么)", re.IGNORECASE),
    re.compile(r"你有(什么|啥)(功能|用处|本事|能力)", re.IGNORECASE),
    re.compile(r"(怎么|如何)(使用|用)你", re.IGNORECASE),
    re.compile(r"^\s*(who\s+are\s+you|what\s+are\s+you|what\s+can\s+you\s+do|help)\b", re.IGNORECASE),
]


_CN_DIGIT = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _cn_num_to_int(s: str) -> int | None:
    """把小范围中文数字串（不超过 99）转成 int。失败返回 None。"""
    if not s:
        return None
    if s.isdigit():
        try:
            return int(s)
        except Exception:
            return None
    # 简化：一/两/三/.../九 / 十 / 十一 / 二十 / 二十一 / 九十九
    if s == "十":
        return 10
    if s.startswith("十") and len(s) == 2 and s[1] in _CN_DIGIT:
        return 10 + _CN_DIGIT[s[1]]
    if len(s) == 1 and s in _CN_DIGIT:
        return _CN_DIGIT[s]
    if len(s) == 3 and s[0] in _CN_DIGIT and s[1] == "十" and s[2] in _CN_DIGIT:
        return _CN_DIGIT[s[0]] * 10 + _CN_DIGIT[s[2]]
    if len(s) == 2 and s[0] in _CN_DIGIT and s[1] == "十":
        return _CN_DIGIT[s[0]] * 10
    return None


_PAGE_PATTERNS = [
    re.compile(r"第\s*([0-9零一二两三四五六七八九十]+)\s*(?:到|至|-|~|—)\s*([0-9零一二两三四五六七八九十]+)\s*页", re.IGNORECASE),
    re.compile(r"第\s*([0-9零一二两三四五六七八九十]+)\s*页", re.IGNORECASE),
    re.compile(r"(?:page|p\.|pg\.)\s*([0-9]+)\s*(?:to|-|~|—)\s*([0-9]+)", re.IGNORECASE),
    re.compile(r"(?:page|p\.|pg\.)\s*([0-9]+)", re.IGNORECASE),
    re.compile(r"\b([0-9]{1,3})\s*(?:st|nd|rd|th)\s+page\b", re.IGNORECASE),
    # 纯数字：只在出现"页"字时才触发，避免误伤
    re.compile(r"([0-9]{1,3})\s*页", re.IGNORECASE),
]


def _extract_pages_from_question(q: str) -> list[int]:
    """从用户问题里抽出用户关心的页码列表（1-based）。"""
    if not q:
        return []
    found: list[int] = []
    seen: set[int] = set()
    for pat in _PAGE_PATTERNS:
        for m in pat.finditer(q):
            groups = [g for g in m.groups() if g]
            if len(groups) == 2:
                a = _cn_num_to_int(groups[0])
                b = _cn_num_to_int(groups[1])
                if a and b and 0 < a <= b <= 999 and (b - a) <= 20:
                    for i in range(a, b + 1):
                        if i not in seen:
                            seen.add(i)
                            found.append(i)
            elif len(groups) == 1:
                n = _cn_num_to_int(groups[0])
                if n and 0 < n <= 999 and n not in seen:
                    seen.add(n)
                    found.append(n)
    return found[:6]   # 最多 6 页，避免上下文爆炸


def _try_answer_meta_question(store, question: str) -> str | None:
    """
    识别并直接回答关于"文献库本身"的元问题。
    返回答案字符串；不是元问题则返回 None，由上层继续走 RAG。
    """
    q = (question or "").strip()
    if not q or len(q) > 200:
        return None

    with store._lock:
        papers = list(store._papers)
    n = len(papers)

    # 打招呼
    if any(p.search(q) for p in _META_GREETING_PATTERNS):
        if n == 0:
            return "你好！当前文献库为空，欢迎上传 PDF 文档开启探索。"
        return (
            f"你好！当前文献库共有 {n} 篇文献。你可以问我任意一篇的内容，"
            f"例如：「XXX 讲了什么」「XXX 的主要方法是什么」。"
        )

    # 身份 / 自我介绍 / 能力介绍
    if any(p.search(q) for p in _META_IDENTITY_PATTERNS):
        lib_part = (
            f"目前你的文献库里共有 **{n} 篇**文献。"
            if n > 0
            else "当前文献库还是空的，你可以先上传一些 PDF。"
        )
        return (
            "我是 **PDF GalaxAI** 的学术助手。我的能力包括：\n"
            "1. **文献问答**：针对你库里的任意一篇 PDF，回答其中的细节、方法、结论等；\n"
            "2. **跨文献检索**：语义检索 + 重排，帮你在多篇文献里找相关段落并给出证据溯源；\n"
            "3. **AI 润色摘要**：对已入库文献生成可读性更高的中文摘要；\n"
            "4. **多模型调用**：支持本地 Ollama、云端 Gemini，以及任意 OpenAI 兼容服务（GPT / DeepSeek / 智谱 / Kimi 等）。\n\n"
            f"{lib_part}你可以问「XXX 这篇讲了什么」「对比 A 和 B 的方法」这类问题。"
        )

    # 数量类
    is_count = any(p.search(q) for p in _META_COUNT_PATTERNS)
    # 列表类
    is_list = any(p.search(q) for p in _META_LIST_PATTERNS)

    if not (is_count or is_list):
        return None

    if n == 0:
        return "当前文献库为空，请先上传一些 PDF 文献。"

    if is_list:
        lines = [f"当前文献库共有 **{n} 篇**文献："]
        for i, p in enumerate(papers, 1):
            title = (
                str(p.get("display_title") or p.get("title") or p.get("filename") or "")
                .strip()
                or f"#{p.get('id', '')[:8]}"
            )
            lines.append(f"{i}. {title}")
        lines.append("")
        lines.append("你可以针对其中任意一篇提问，比如「XXX 的核心方法是什么」。")
        return "\n".join(lines)

    # 纯数量问题
    sample_titles = []
    for p in papers[:3]:
        t = (
            str(p.get("display_title") or p.get("title") or p.get("filename") or "")
            .strip()
        )
        if t:
            sample_titles.append(t)
    preview = "、".join(sample_titles) if sample_titles else ""
    tail = (
        f"包括 {preview} 等。"
        if preview and n > len(sample_titles)
        else (f"分别是：{preview}。" if preview else "")
    )
    return (
        f"当前文献库共有 **{n} 篇**文献。{tail}"
        f"如果想看完整列表，可以问我「列出所有文献」。"
    )


async def _rewrite_query(
    cfg: dict[str, Any],
    question: str,
    history: list[Any] | None,
) -> str:
    """
    检索前的查询改写：多轮指代消解 + 轻量术语补全。

    只在「有对话历史」或「查询过短」时触发（这两种情况原始 query 向量化效果最差）。
    任何异常或可疑输出都退回原始 question，保证不劣化。
    返回用于检索/文献名匹配的查询（最终回答仍用原始 question）。
    """
    hist_lines: list[str] = []
    for turn in (history or [])[-4:]:
        role = (getattr(turn, "role", "") or "").lower()
        text = (getattr(turn, "text", "") or "").strip()
        if not text:
            continue
        who = "用户" if role in ("user", "human") else "助手"
        hist_lines.append(f"{who}：{text[:300]}")
    hist_block = "\n".join(hist_lines) if hist_lines else "（无历史）"

    sys_prompt = (
        "你是检索查询改写器。根据对话历史，把用户的最新问题改写成一个可以"
        "独立用于文献向量检索的查询：\n"
        "1) 消解指代：把『它/这个/上面那篇/前面说的』替换成历史里对应的具体对象；\n"
        "2) 适当补全学术术语，让查询语义更完整，但严禁编造历史中不存在的信息；\n"
        "3) 保持简洁，只输出改写后的查询本身，不要任何解释、前缀或引号。"
    )
    user_prompt = (
        f"对话历史：\n{hist_block}\n\n"
        f"用户最新问题：{question}\n\n"
        f"改写后的检索查询："
    )
    try:
        rewritten = await call_llm(
            cfg,
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
    except Exception as e:
        print(f"[api] query rewrite failed: {type(e).__name__}: {e}")
        return question

    rewritten = (rewritten or "").strip().strip('"""\u300c\u300d\'')
    # 安全护栏：空 / 过长 / 模型答非所问 → 退回原问题
    if not rewritten or len(rewritten) > 200:
        return question
    return rewritten


async def _summarize_history(
    cfg: dict[str, Any],
    history: list[Any] | None,
    keep_recent: int = 4,
    trigger: int = 8,
) -> tuple[str | None, list[dict[str, str]]]:
    """
    长对话摘要记忆（SummaryMemory）。

    - 对话消息数 <= trigger：不压缩，原样返回（标准化为 {role,content} dict）。
    - 超过 trigger：把较早的轮次用一次 LLM 调用压成 ≤200 字要点摘要，
      与最近 keep_recent 条逐字保留，一起返回。
    任何异常/可疑输出都退回「最近 6 条」滑动窗口，绝不破坏对话。
    返回 (summary_text|None, recent_turns)。
    """
    norm: list[dict[str, str]] = []
    for turn in (history or []):
        role = (getattr(turn, "role", "") or "").lower().strip()
        text = (getattr(turn, "text", "") or "").strip()
        if not text:
            continue
        r = "assistant" if role in ("ai", "assistant", "model") else "user"
        norm.append({"role": r, "content": text})

    if len(norm) <= trigger:
        return None, norm

    older = norm[:-keep_recent]
    recent = norm[-keep_recent:]
    convo = "\n".join(
        f"{'用户' if m['role'] == 'user' else '助手'}：{m['content'][:400]}"
        for m in older
    )
    sys_prompt = (
        "你是对话记忆压缩器。把下面这段较早的对话压缩成简洁的中文要点摘要，"
        "保留关键事实、已确认的结论、用户偏好、提到过的论文/概念；省略寒暄与重复。"
        "控制在 200 字内，只输出摘要本身。"
    )
    try:
        summary = await call_llm(
            cfg,
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": convo},
            ],
            temperature=0.2,
        )
    except Exception as e:
        print(f"[api] history summarize failed: {type(e).__name__}: {e}")
        return None, norm[-6:]

    summary = (summary or "").strip()
    if not summary or len(summary) > 600:
        return None, norm[-6:]
    return summary, recent


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
        gemini_cfg = _registry.get_config("gemini") or {}
        local_cfg = _registry.get_config("local") or {}
        return {
            "default_provider": _DEFAULT_PROVIDER,
            "ollama_model": str(local_cfg.get("model") or ""),
            "gemini_model": str(gemini_cfg.get("model") or ""),
            "gemini_key_configured": bool(gemini_cfg.get("api_key")),
        }

    # ---------------------------------------------------------------- #
    # 模型注册表：内置 + 自定义 CRUD
    # ---------------------------------------------------------------- #

    @app.get("/api/models")
    def api_list_models() -> dict[str, Any]:
        return {
            "models": _registry.list_public(),
            "default_provider": _DEFAULT_PROVIDER,
        }

    @app.post("/api/models")
    def api_create_model(body: ModelCreateBody) -> dict[str, Any]:
        try:
            record = _registry.add(body.model_dump())
            return {"success": True, "model": record}
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    @app.put("/api/models/{model_id}")
    def api_update_model(model_id: str, body: ModelUpdateBody) -> dict[str, Any]:
        try:
            record = _registry.update(
                model_id, {k: v for k, v in body.model_dump().items() if v is not None}
            )
            return {"success": True, "model": record}
        except KeyError:
            raise HTTPException(status_code=404, detail="model not found")
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    @app.delete("/api/models/{model_id}")
    def api_delete_model(model_id: str) -> dict[str, Any]:
        try:
            removed = _registry.delete(model_id)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))
        if not removed:
            raise HTTPException(status_code=404, detail="model not found")
        return {"success": True, "model_id": model_id}

    @app.post("/api/models/test")
    async def api_test_model_adhoc(body: ModelTestBody) -> dict[str, Any]:
        """前端"添加/编辑模型"弹窗里的即时测试。直接用 body 里的参数探测，不保存。"""
        cfg = {
            "protocol": (body.protocol or "openai").strip().lower(),
            "base_url": (body.base_url or "").strip().rstrip("/"),
            "model": (body.model or "").strip(),
            "api_key": (body.api_key or "").strip(),
            "proxy": (body.proxy or "").strip(),
        }
        if cfg["protocol"] not in {"openai", "gemini"}:
            raise HTTPException(status_code=422, detail="protocol 仅支持 openai / gemini")
        if not cfg["model"]:
            raise HTTPException(status_code=422, detail="model 不能为空")
        if cfg["protocol"] == "openai" and not cfg["base_url"]:
            raise HTTPException(status_code=422, detail="openai 协议必须填 base_url")
        if cfg["protocol"] == "gemini" and not cfg["base_url"]:
            cfg["base_url"] = "https://generativelanguage.googleapis.com/v1beta"
        return await test_connection(cfg)

    @app.post("/api/models/{model_id}/test")
    async def api_test_model(model_id: str) -> dict[str, Any]:
        """测试一个已保存模型（包括内置）的连通性。"""
        cfg = _registry.get_config(model_id)
        if not cfg:
            raise HTTPException(status_code=404, detail="model not found")
        return await test_connection(cfg)

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
                        "llmSummary": str(p.get("llm_summary") or ""),
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
    async def api_upload(
        file: UploadFile = File(...),
        ocr_mode: str | None = Form(default=None),
    ) -> dict[str, Any]:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="只支持 .pdf 文件")
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="文件为空")
        mode = (ocr_mode or "").strip().lower()
        if mode and mode not in {"auto", "force", "off"}:
            raise HTTPException(status_code=400, detail=f"未知的 ocr_mode: {ocr_mode}")
        try:
            print(
                f"[api] Starting upload: {file.filename} ({len(raw)} bytes)"
                f"{f' ocr={mode}' if mode else ''}"
            )
            pdf_id = store.add_pdf(file.filename, raw, ocr_mode=mode or None)
            print(f"[api] Upload successful: {file.filename} -> {pdf_id}")
        except ValueError as e:
            print(f"[api] Validation error during upload: {e}")
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)
            )
        except RuntimeError as e:
            print(f"[api] Runtime error during upload: {e}")
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            print(f"[api] Unexpected error during upload: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
        return {"success": True, "pdf_id": pdf_id}

    @app.post("/api/papers/upload")
    async def api_papers_upload(
        file: UploadFile = File(...),
        ocr_mode: str | None = Form(default=None),
    ) -> dict[str, Any]:
        return await api_upload(file, ocr_mode=ocr_mode)

    class ReprocessBody(BaseModel):
        ocr_mode: str | None = None  # 'auto' | 'force' | 'off'
        parser: str | None = None    # 'default' | 'fitz' | 'mineru'

    @app.get("/api/mineru/status")
    def api_mineru_status() -> dict[str, Any]:
        """检查 MinerU 是否可用，供前端判断按钮 enable/disable。"""
        try:
            from mineru_parser import is_mineru_available
            available = bool(is_mineru_available())
        except Exception as e:
            return {"available": False, "error": str(e)}
        return {
            "available": available,
            "cmd": os.getenv("SCHOLAR_MINERU_CMD") or "mineru",
            "backend": os.getenv("SCHOLAR_MINERU_BACKEND") or "pipeline",
            "lang": os.getenv("SCHOLAR_MINERU_LANG") or "ch",
        }

    @app.get("/api/admin/index-status")
    def api_index_status() -> dict[str, Any]:
        return {
            "current_model": store._model_name,
            "last_indexed_model": store._last_emb_model,
            "needs_reindex": bool(
                store._last_emb_model and store._last_emb_model != store._model_name
            ),
            "total_papers": len(store._papers),
        }

    @app.post("/api/admin/reindex")
    def api_reindex_all() -> dict[str, Any]:
        try:
            print(f"[api] Starting full reindex with model={store._model_name}")
            result = store.reindex_all_papers()
            print(f"[api] Reindex done: {result}")
            return {"success": True, **result}
        except Exception as e:
            print(f"[api] Reindex failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/papers/{paper_id}/reprocess")
    def api_reprocess_paper(paper_id: str, body: ReprocessBody | None = None) -> dict[str, Any]:
        mode = ((body.ocr_mode if body else None) or "").strip().lower()
        parser = ((body.parser if body else None) or "").strip().lower()
        if mode and mode not in {"auto", "force", "off"}:
            raise HTTPException(status_code=400, detail=f"未知的 ocr_mode: {mode}")
        if parser and parser not in {"default", "fitz", "mineru"}:
            raise HTTPException(status_code=400, detail=f"未知的 parser: {parser}")
        try:
            print(
                f"[api] Reprocess paper {paper_id} "
                f"ocr={mode or 'default'} parser={parser or 'default'}"
            )
            result = store.reprocess_paper(
                paper_id,
                ocr_mode=mode or None,
                parser=parser or None,
            )
            return {"success": True, **result}
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
        except Exception as e:
            print(f"[api] Reprocess failed for {paper_id}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/papers/{paper_id}")
    def api_delete_paper(
        paper_id: str,
        purge_source: bool = Query(
            default=True,
            description="是否同时删除 inbox/_processed 中同名源文件（默认 true）",
        ),
    ) -> dict[str, Any]:
        try:
            print(f"[api] Starting delete: {paper_id}, purge_source={purge_source}")
            ok = store.delete_paper(paper_id, delete_source_files=purge_source)
            if not ok:
                print(f"[api] Delete failed: paper {paper_id} not found")
                raise HTTPException(status_code=404, detail="paper not found")
            print(f"[api] Delete successful: {paper_id}")
            return {"success": True, "paper_id": paper_id, "purge_source": purge_source}
        except HTTPException:
            raise
        except Exception as e:
            print(f"[api] Unexpected error during delete: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/analyze")
    def api_analyze(body: AnalyzeBody) -> dict[str, Any]:
        return store.analyze()

    @app.post("/api/papers/{paper_id}/polish-summary")
    async def api_polish_summary(paper_id: str, body: PolishBody) -> dict[str, Any]:
        """
        用 LLM 为单篇论文生成一段"润色摘要"，持久化到 paper.llm_summary。
        前端预览优先显示 llm_summary（如有），否则显示 abstract。
        """
        paper = store.get_paper(paper_id)
        if not paper:
            raise HTTPException(status_code=404, detail="paper not found")

        full_text = store.get_cleaned_fulltext(paper_id)
        if not full_text or len(full_text) < 30:
            raise HTTPException(status_code=422, detail="文献正文为空或无法提取")

        title = str(paper.get("title") or paper.get("display_title") or "")
        existing_abstract = str(paper.get("abstract") or "")[:800]
        body_excerpt = full_text[:3000]

        prompt = (
            "你是学术摘要专家。请基于以下文献信息，用简洁流畅的中文生成一段 2-4 句、"
            "不超过 200 字的摘要，概括其主题、方法和主要结论或价值。"
            "不要加'摘要：''Summary:'等前缀，不要虚构内容，不要重复标题，不要使用 markdown。\n\n"
            f"标题：{title}\n\n"
            f"已提取摘要（可能不完整）：\n{existing_abstract}\n\n"
            f"正文节选：\n{body_excerpt}\n\n"
            "请直接输出纯文本摘要："
        )

        provider, cfg = _resolve_provider(body.provider)

        polish_messages = [
            {"role": "system", "content": "你是学术摘要专家，用简洁自然的中文撰写摘要，不要使用 markdown。"},
            {"role": "user", "content": prompt},
        ]
        try:
            raw = await call_llm(cfg, polish_messages, temperature=0.3)
        except Exception as e:
            print(f"[api] polish-summary LLM failed: {type(e).__name__}: {e}")
            raise HTTPException(status_code=503, detail=f"AI 润色失败：{e}")

        summary = (raw or "").strip()
        # 清理常见前缀 & markdown
        summary = re.sub(
            r"^\s*(摘要|Abstract|Summary|概要|简介)\s*[:：]?\s*",
            "",
            summary,
            flags=re.IGNORECASE,
        )
        summary = re.sub(r"^```[\w]*\s*|\s*```$", "", summary).strip()
        summary = summary[:800]

        if not summary:
            raise HTTPException(status_code=503, detail="AI 返回空摘要")

        store.set_llm_summary(paper_id, summary)
        return {
            "success": True,
            "paper_id": paper_id,
            "summary": summary,
            "provider_used": provider,
        }

    @app.delete("/api/papers/{paper_id}/polish-summary")
    def api_delete_polish_summary(paper_id: str) -> dict[str, Any]:
        """清空某篇论文的 LLM 润色摘要，回退到自动抽取的 abstract。"""
        ok = store.set_llm_summary(paper_id, "")
        if not ok:
            raise HTTPException(status_code=404, detail="paper not found")
        return {"success": True, "paper_id": paper_id}

    @app.post("/api/rebuild-index")
    def api_rebuild_index() -> dict[str, Any]:
        """
        对全库文献重建 chunk 向量索引（清向量 + 重新切块 + 重新 embedding）。
        用于：修改了 OCR / 清洗策略后，让 RAG 召回立即用上新数据。
        注意：耗时较长（每篇 PDF ~1-5 秒），建议放在设置入口或一次性脚本里。
        """
        try:
            stats = store.rebuild_index_for_all()
            return {"success": True, **stats}
        except Exception as e:
            print(f"[api] rebuild-index failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/refresh-previews")
    def api_refresh_previews() -> dict[str, Any]:
        """
        一键刷新所有已入库文献的预览字段（标题/摘要/首句/自动关键词），
        基于升级后的 clean_text + extract_abstract_block 规则重算。
        不触碰向量、不改可视化布局。
        """
        try:
            stats = store.refresh_previews_for_all()
            return {"success": True, **stats}
        except Exception as e:
            print(f"[api] refresh-previews failed: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

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

        # 用户强制指定的模式：'chat' / 'rag' / 'auto'（默认）
        forced_mode = (body.mode or "auto").strip().lower()
        if forced_mode not in {"auto", "chat", "rag"}:
            forced_mode = "auto"

        # ── 0. Meta 问题直通车（关于"系统/库"本身，不走 RAG）──────────────
        #   强制 chat 模式时也跳过 meta，让 LLM 自己回答，保持"纯 AI 助手"体验
        if forced_mode != "chat":
            meta_answer = _try_answer_meta_question(store, question)
            if meta_answer is not None:
                return {
                    "answer": meta_answer,
                    "cites": [],
                    "cite_details": [],
                    "provider_used": "system",
                    "status": "ok",
                    "message": "",
                }

        # ── 1. Chunk 检索（优先命中文献名）──────────────────────────────
        # 策略：双通道命名识别
        #   A) 全名子串匹配（问题里完整出现文件名/标题，强信号）
        #   B) Token 重叠匹配（问题里出现文献名的关键词，如 "MLP"→"MLP-2602"）
        # 前者优先，没命中才用后者。

        _QUERY_STOPWORDS = {
            # 中文问法停用词（和文献内容无关的修饰）
            "文献", "论文", "文章", "文件", "研究", "工作", "报告",
            "讲了", "说了", "介绍", "告诉", "解释", "总结", "概括", "分析",
            "是什么", "干嘛", "什么", "怎么", "如何", "为何", "为什么",
            "关于", "有关", "内容", "主要", "一下", "它的", "他的",
            "以及", "和", "或", "与", "的", "了", "啊", "呀", "吗",
            # 英文问法停用词
            "paper", "papers", "article", "document", "file", "files",
            "what", "how", "why", "the", "is", "are", "of", "about",
            "tell", "me", "explain", "summary", "summarize",
        }

        def _norm(s: str) -> str:
            s = (s or "").lower().strip()
            if s.endswith(".pdf"):
                s = s[:-4]
            return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", s)

        def _tokens(s: str) -> list[str]:
            """把字符串拆成关键词 token（英数连续块 / 中文 2-6 字）。"""
            t = (s or "").lower()
            toks = re.findall(r"[a-z0-9]{2,}|[\u4e00-\u9fff]{2,6}", t)
            return [x for x in toks if x not in _QUERY_STOPWORDS]

        def _tokens_with_single(s: str) -> list[str]:
            """论文名 token（允许单字英文缩写如 'A', 'C'）。"""
            t = (s or "").lower()
            toks = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,6}", t)
            return [x for x in toks if x not in _QUERY_STOPWORDS]

        # ── 1.0 检索查询改写（多轮指代消解 + 短查询补全）─────────────────
        # search_query 仅用于检索 / 文献名匹配；最终回答仍基于原始 question。
        provider, cfg = _resolve_provider(body.provider)
        search_query = question
        if forced_mode != "chat" and (body.history or len(question) <= 8):
            search_query = await _rewrite_query(cfg, question, body.history)

        q_norm = _norm(search_query)
        q_tokens = set(_tokens(search_query))
        orig_q_tokens = set(_tokens(question))
        target_paper_id: str | None = None

        with store._lock:
            candidates = list(store._papers)

        # A) 全名子串
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

        # B) Token 重叠（A 没命中才走）
        if target_paper_id is None and q_tokens:
            best_score = 0
            for p in candidates:
                joined = " ".join(
                    [
                        str(p.get("display_title") or ""),
                        str(p.get("title") or ""),
                        str(p.get("filename") or ""),
                    ]
                )
                p_tokens = set(_tokens_with_single(joined))
                if not p_tokens:
                    continue
                hits = q_tokens & p_tokens
                if not hits:
                    continue
                # 打分：命中的 token 总长度（长 token 权重高，"mlp"=3，"tomcat"=6）
                score = sum(len(t) for t in hits)
                # 至少命中一个 ≥3 字符的 token 才算数，避免"的"/"是"之类误伤
                if score >= 3 and max(len(t) for t in hits) >= 3 and score > best_score:
                    best_score = score
                    target_paper_id = str(p.get("id") or "")

        # ── 2. 检索：命名文献优先 → 全库兜底 ────────────────────────────
        chunks: list[dict[str, Any]] = []
        stat = "empty_db"
        # 强制 chat 模式：完全跳过向量检索，纯 LLM 聊天
        if forced_mode == "chat":
            pass
        elif target_paper_id:
            chunks, stat = store.search_chunks(
                search_query, top_k=6, paper_id=target_paper_id, min_score=0.05
            )
            if stat != "ok":
                chunks, stat = store.search_chunks(search_query, top_k=5)
        else:
            chunks, stat = store.search_chunks(search_query, top_k=5)

        # ── 3. 自适应路由：看看检索质量如何 ──────────────────────────────
        def _sanitize_for_llm(text: str, max_len: int) -> str:
            s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", text or "")
            s = s.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
            s = re.sub(r"\s{2,}", " ", s).strip()
            return s[:max_len]

        def _is_noisy_text(text: str) -> bool:
            s = (text or "")
            if not s:
                return True
            if s.count("?") / max(1, len(s)) >= 0.03:
                return True
            if "a?\ufffd\ufffd" in s or "e???" in s or "\ufffd" in s:
                return True
            return False

        high_quality_chunks = [
            c for c in chunks
            if float(c.get("quality", 1.0)) >= 0.75
            and not _is_noisy_text(str(c.get("snippet", "")))
        ]

        # 重排后的 final_score ≈ 0.72 语义 + 0.18 词面 + 0.10 质量。
        # 用 raw_score（语义裸分）做模式判定更可靠。
        top_raw = 0.0
        if chunks:
            top_raw = float(chunks[0].get("raw_score") or chunks[0].get("score") or 0.0)

        # 模式：
        #   rag   - 有高质量命中 或 用户点名了文献 → 必须基于文献回答
        #   rag_soft - 命中但质量/相关度一般 → 把文献当参考，允许通用知识补充
        #   chat  - 完全没命中或极弱 → 通用对话，知道文献库里有哪些
        # 短问句且没有任何实质关键词（都被停用词过滤掉了）→ 视为闲聊
        #   - 避免"你是谁""谢谢""好的"等被误当成 RAG
        is_chitchat_only = (not orig_q_tokens) and len(question) <= 20

        # rag_soft 需要：语义分 ≥ 0.22 且至少有一个 chunk 质量 ≥ 0.5。
        # 避免"你了解 XXX 吗"这种和文献词面沾边、实际无关的问题被拖进 RAG。
        has_decent_chunk = any(
            float(c.get("quality", 1.0)) >= 0.5 for c in chunks
        ) if chunks else False

        if forced_mode == "chat":
            # 用户显式要求通用对话：即使检索有命中也忽略
            mode = "chat"
        elif forced_mode == "rag":
            # 用户显式要求基于文献：只要有命中就走 rag
            mode = "rag" if chunks else "rag_soft"
        elif is_chitchat_only:
            mode = "chat"
        elif target_paper_id and (high_quality_chunks or chunks):
            mode = "rag"
        elif high_quality_chunks and top_raw >= 0.30:
            mode = "rag"
        elif chunks and top_raw >= 0.22 and has_decent_chunk:
            mode = "rag_soft"
        else:
            mode = "chat"

        # ── 4. 组装 System Prompt（随模式变化）────────────────────────────
        with store._lock:
            all_papers_snapshot = [
                {
                    "id": str(p.get("id") or ""),
                    "title": str(p.get("display_title") or p.get("title") or p.get("filename") or ""),
                }
                for p in store._papers
            ]

        library_overview = ""
        if all_papers_snapshot:
            lib_lines = [f"- {p['title']}" for p in all_papers_snapshot[:20] if p["title"]]
            if lib_lines:
                library_overview = (
                    f"【当前文献库共 {len(all_papers_snapshot)} 篇】\n"
                    + "\n".join(lib_lines)
                )

        context_parts: list[str] = []

        # (A) 锚定某篇文献时，总是先注入"文献总览"（标题/关键词/摘要/润色摘要），
        #     给弱模型一个稳定锚点，不再只靠运气检索到的小片段。
        target_meta = None
        if target_paper_id:
            with store._lock:
                for p in store._papers:
                    if str(p.get("id") or "") == target_paper_id:
                        target_meta = dict(p)
                        break
        if target_meta:
            title = _sanitize_for_llm(str(target_meta.get("title") or ""), 300)
            summary_text = _sanitize_for_llm(
                str(target_meta.get("llm_summary") or target_meta.get("abstract") or ""),
                1400,
            )
            keywords = ", ".join(
                [str(k) for k in (target_meta.get("keywords") or [])[:12]]
            )
            context_parts.append(
                f"【文献总览 paper_id={target_paper_id}】\n"
                f"标题: {title}\n"
                f"关键词: {keywords}\n"
                f"摘要: {summary_text}"
            )

        # (B) 页码精确命中：用户问"第 N 页/page N"时，直接从 PDF 抽出那几页原文。
        #     仅在锚定了某篇文献时启用，避免跨文献歧义。
        requested_pages: list[int] = []
        if target_paper_id and forced_mode != "chat":
            requested_pages = _extract_pages_from_question(question)
        if requested_pages:
            try:
                pages_text = store.get_pages_text(target_paper_id, requested_pages)
            except Exception as _e:
                print(f"[api] get_pages_text failed: {_e}")
                pages_text = []
            total_pages = store.get_page_count(target_paper_id) if target_paper_id else 0
            for pn, txt in pages_text:
                if 1 <= pn <= (total_pages or pn):
                    snippet = _sanitize_for_llm(txt, max_len=1500)
                    if snippet:
                        context_parts.append(
                            f"【原文 paper_id={target_paper_id} 第 {pn} 页】\n{snippet}"
                        )
                    else:
                        context_parts.append(
                            f"【原文 paper_id={target_paper_id} 第 {pn} 页】\n（该页无可抽取的文本；可能是纯图片页或扫描件。）"
                        )
                else:
                    context_parts.append(
                        f"【提示】用户问到第 {pn} 页，但该文献共 {total_pages} 页，请据此提醒用户。"
                    )

        # (C) 语义检索到的片段（不够高质的也给一些，让 LLM 有更多线索）
        source_chunks = high_quality_chunks if high_quality_chunks else (
            [] if (target_paper_id and mode == "rag" and not chunks) else chunks
        )
        for c in source_chunks[:4]:
            snippet = _sanitize_for_llm(str(c.get("snippet", "")), max_len=360)
            context_parts.append(
                f"【片段 paper_id={c['paper_id']} chunk_id={c['chunk_id']}】\n{snippet}"
            )

        context = "\n\n".join(context_parts)

        if mode == "rag":
            system_prompt = (
                "你是一位博学、乐于助人的学术助手。用户有自己的文献库，"
                "接下来的消息里可能附带检索到的相关段落。\n"
                "回答原则：\n"
                "1) 有相关段落时，基于段落回答，但不要在正文输出任何 paper_id / chunk_id 等技术字段；\n"
                "2) 段落不够完整时，可以自然地用你自己的通用知识补充，但要说清哪些是来自文献、哪些是补充；\n"
                "3) 语气自然、直接、一段话式，不要机械地说『根据参考资料…』；\n"
                "4) 如果段落与问题无关，坦率说'文献里没直接写到这个'，然后再给出一般性回答。"
            )
        elif mode == "rag_soft":
            system_prompt = (
                "你是一位博学、乐于助人的学术助手。用户的文献库里可能和问题相关度一般，"
                "下方附带的片段仅供参考。请自然地回答，可以用你自己的知识补充，"
                "若引用了片段请自然说明来源，但不要输出任何 paper_id / chunk_id 技术字段。"
            )
        else:
            system_prompt = (
                "你是一位博学、乐于助人的 AI 助手，既能闲聊也能回答学术问题。"
                "用户有一个文献库（列表见下），但当前问题和文献关联不大，"
                "你可以像普通大模型一样自由、自然、有帮助地回答。"
                "如果问题其实涉及某一篇文献的细节，你可以善意提醒用户"
                "『可以直接问 XXX 这篇文献的具体内容』。"
            )

        if library_overview:
            system_prompt += "\n\n" + library_overview

        # 统一的输出风格规范（所有 mode 都适用）——让前端 Markdown/KaTeX 能正确渲染。
        system_prompt += (
            "\n\n输出格式要求："
            "\n- 使用 Markdown：重点用 **加粗**、列表用 - 或 1. 开头、标题用 ##。"
            "\n- 数学公式一律用 LaTeX：行内用 $...$，独立成行用 $$...$$。"
            "\n- 不要输出孤立的反斜杠、星号或乱码字符；若原文里有 OCR 残留符号，请自行整理成人类可读的形式。"
            "\n- 不要对公式字符做任何转义（不要写 \\_、\\*），直接给合法的 LaTeX。"
            "\n- 不要在最终回答中出现 `paper_id=`、`chunk_id=`、`[CITE:...]` 等内部标识。"
            "\n- 所有回答用简体中文。"
        )

        user_parts: list[str] = []
        if context:
            user_parts.append("参考文献片段：\n" + context)
        user_parts.append("用户问题：\n" + question)
        if mode == "rag" and target_paper_id:
            user_parts.append(
                f"补充要求：用户已指定目标文献 paper_id={target_paper_id}，请优先围绕该文献作答。"
            )
        if requested_pages:
            pages_str = ", ".join(f"第 {p} 页" for p in requested_pages)
            user_parts.append(
                f"补充要求：用户问到的是本文献的 {pages_str}，"
                f"上方已附带这些页的原文。请直接基于这些原文逐页讲解内容、图表、要点，"
                f"不要泛泛而谈，也不要说『文献里没写到』这种话。"
            )
        user_content = "\n\n".join(user_parts)

        # ── 5. 加入历史消息（多轮对话 + 长对话摘要记忆）──────────────────
        # 短对话逐字带最近轮次；长对话把较早内容压成摘要，省 token 又不丢上下文。
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        summary, recent_turns = await _summarize_history(cfg, body.history)
        if summary:
            messages.append({
                "role": "system",
                "content": "【早前对话摘要（已压缩）】\n" + summary,
            })
        for m in recent_turns:
            text = _sanitize_for_llm(m["content"], 1500)
            if not text:
                continue
            messages.append({"role": m["role"], "content": text})
        messages.append({"role": "user", "content": user_content})

        # ── 6. 调用 LLM ─────────────────────────────────────────────────
        # provider/cfg 已在步骤 1.0 解析（供查询改写复用），此处直接使用。
        try:
            temperature = 0.4 if mode == "rag" else 0.6
            answer = await call_llm(cfg, messages, temperature=temperature)

            # ── 7. 返回答案 + 引用 ─────────────────────────────────────
            if mode == "chat":
                # 纯聊天模式不返回引用
                cite_details: list[dict[str, Any]] = []
                cites: list[str] = []
            else:
                cite_details = []
                for c in (high_quality_chunks or chunks):
                    pid = c["paper_id"]
                    snip = c["snippet"]
                    page_no = store.locate_snippet_page(pid, snip)
                    cite_details.append(
                        {
                            "paper_id": pid,
                            "chunk_id": c["chunk_id"],
                            "snippet": snip,
                            "page": page_no,
                        }
                    )
                cites = list(dict.fromkeys([c["paper_id"] for c in cite_details]))

            return {
                "answer": answer,
                "cites": cites,
                "cite_details": cite_details,
                "provider_used": provider,
                "mode": mode,
                "status": "ok",
                "message": "",
            }

        except Exception as e:
            print(f"[api] LLM Error: {e}")
            return {
                "answer": _STATUS_MSG["llm_error"] + f"\n错误详情: {e}",
                "cites": [],
                "provider_used": provider,
                "mode": mode,
                "status": "llm_error",
                "message": _STATUS_MSG["llm_error"],
            }

    # ---------------------------------------------------------------- #
    # /api/agent-query — ReAct 多步 Agent（跨文献对比/综合类问题）
    # ---------------------------------------------------------------- #

    class AgentBody(BaseModel):
        question: str
        provider: str | None = None
        history: list[ChatTurn] | None = None
        # force=True 跳过启发式检测，直接走 agent
        force: bool = False

    @app.post("/api/agent-query")
    async def api_agent_query(body: AgentBody) -> dict[str, Any]:
        """
        ReAct Agent 端点。
        - 自动检测跨文献/对比类问题时路由到此；也可前端显式调用。
        - 返回与 /api/query 相同结构，额外附带 steps 字段供调试。
        """
        question = (body.question or "").strip()
        if not question:
            return {"answer": "请输入问题。", "cites": [], "steps": [], "status": "ok", "message": ""}

        provider, cfg = _resolve_provider(body.provider)

        async def llm_fn(messages: list[dict], temperature: float = 0.3) -> str:
            return await call_llm(cfg, messages, temperature=temperature)

        # 路由：正则快筛先挡掉明显单篇问题，再用 LLM 意图分类兜底
        if not body.force:
            if should_use_agent(question):
                intent = "multi"
            else:
                intent = await classify_intent(question, llm_fn)
            if intent != "multi":
                return {
                    "answer": "该问题适合普通 RAG，请使用 /api/query 接口。",
                    "cites": [],
                    "steps": [],
                    "status": "ok",
                    "message": "",
                }

        # 把历史转成 messages 格式（含长对话摘要记忆）
        history_msgs: list[dict[str, str]] = []
        summary, recent_turns = await _summarize_history(cfg, body.history)
        if summary:
            history_msgs.append({
                "role": "system",
                "content": "【早前对话摘要（已压缩）】\n" + summary,
            })
        history_msgs.extend(recent_turns)

        # 原生函数调用仅 OpenAI 兼容协议支持；Gemini 走文本 ReAct 回退
        proto = str(cfg.get("protocol") or "openai").strip().lower()
        llm_tool_fn = None
        if proto != "gemini":
            async def llm_tool_fn(messages: list[dict], tools: list[dict], temperature: float = 0.3) -> dict:
                return await call_llm_with_tools(cfg, messages, tools, temperature=temperature)

        agent = build_agent(store, llm_fn, llm_tool_fn=llm_tool_fn)

        try:
            result = await agent.run(question, history=history_msgs)
        except Exception as e:
            print(f"[api] agent error: {type(e).__name__}: {e}")
            return {
                "answer": f"Agent 执行失败：{e}",
                "cites": [],
                "steps": [],
                "provider_used": provider,
                "status": "llm_error",
                "message": _STATUS_MSG["llm_error"],
            }

        # 从 Agent 收集的引用回填 cite_details（带页码，供前端跳转 PDF）
        cite_details: list[dict[str, Any]] = []
        for c in result.get("cites", []):
            pid = c.get("paper_id")
            cid = c.get("chunk_id")
            if not pid:
                continue
            snippet, page_no = "", None
            ctx = store.get_chunk_context(str(cid), window=0) if cid else None
            if ctx and ctx.get("chunks"):
                snippet = str(ctx["chunks"][0].get("text") or "")[:400]
                page_no = store.locate_snippet_page(pid, snippet) if snippet else None
            cite_details.append(
                {"paper_id": pid, "chunk_id": cid, "snippet": snippet, "page": page_no}
            )
        cites = list(dict.fromkeys([c["paper_id"] for c in cite_details]))

        return {
            "answer": result["answer"],
            "cites": cites,
            "cite_details": cite_details,
            "steps": result.get("steps", []),
            "provider_used": provider,
            "mode": "agent",
            "status": "ok",
            "message": "",
        }

    # ---------------------------------------------------------------- #
    # 截图提问（多模态 RAG）
    # ---------------------------------------------------------------- #

    _VISION_MIME_MAP = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
        "gif": "image/gif",
    }

    def _normalize_image_mime(upload: UploadFile) -> str:
        mime = (upload.content_type or "").lower().strip()
        if mime.startswith("image/"):
            return mime
        name = (upload.filename or "").lower()
        ext = name.rsplit(".", 1)[-1] if "." in name else ""
        return _VISION_MIME_MAP.get(ext, "image/png")

    @app.post("/api/query_vision")
    async def api_query_vision(
        image: UploadFile = File(...),
        question: str = Form(...),
        provider: str | None = Form(default=None),
        paper_id: str | None = Form(default=None),
        mode: str | None = Form(default=None),
    ) -> dict[str, Any]:
        """
        截图 + 文字提问。流程：
          Phase A: 让多模态 LLM 看图，给出一段简洁的视觉描述
          Phase B: 用 (question + description) 去做 RAG 检索
          Phase C: 把 [图片 + 问题 + 检索片段] 一起发给多模态 LLM，产出最终答案

        必须使用支持图片的模型（Gemini / GPT-4o 等），否则直接报错。
        """
        q = (question or "").strip()
        if not q:
            raise HTTPException(status_code=400, detail="缺少 question")

        raw = await image.read()
        if not raw:
            raise HTTPException(status_code=400, detail="截图为空")
        if len(raw) > 8 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="截图过大（上限 8MB）")
        mime = _normalize_image_mime(image)

        provider_id, cfg = _resolve_provider(provider)
        if not is_vision_capable(cfg):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"当前模型 '{provider_id}' 不支持看图。"
                    "请切换到 Gemini / GPT-4o / Claude 3.5+ 这类多模态模型。"
                ),
            )

        img_part = encode_image_to_base64(raw, mime=mime)
        forced_mode = (mode or "").strip().lower()

        # ── Phase A: 视觉描述 ──────────────────────────────────────────
        describe_messages = [
            {
                "role": "system",
                "content": (
                    "你是图文理解专家。用户会给你一张论文截图，你的任务是：\n"
                    "用一段中文精准描述图中的内容，特别是：\n"
                    "- 任何公式（请用 LaTeX 写出，行内 $...$，独立 $$...$$）\n"
                    "- 变量/符号含义（x、y、θ 等分别代表什么）\n"
                    "- 图表标题、坐标轴、关键标注\n"
                    "- 表格结构（表头 + 前几行示例）\n"
                    "要求：只描述，不要展开回答，不要对公式做反斜杠转义，控制在 300 字内。"
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"用户的问题是：{q}\n请先描述这张图。"},
                    img_part,
                ],
            },
        ]

        try:
            description = await call_llm(cfg, describe_messages, temperature=0.2)
        except Exception as e:
            print(f"[vision] describe failed: {type(e).__name__}: {e}")
            raise HTTPException(status_code=503, detail=f"视觉描述失败：{e}")

        description = (description or "").strip()

        # ── Phase B: RAG 检索 ─────────────────────────────────────────
        chunks: list[dict[str, Any]] = []
        stat = "empty_db"
        if forced_mode != "chat":
            rag_query = f"{q}\n\n[图片描述]\n{description}"[:1500]
            try:
                if paper_id:
                    chunks, stat = store.search_chunks(
                        rag_query, top_k=6, paper_id=paper_id, min_score=0.05
                    )
                    if stat != "ok":
                        chunks, stat = store.search_chunks(rag_query, top_k=5)
                else:
                    chunks, stat = store.search_chunks(rag_query, top_k=5)
            except Exception as e:
                print(f"[vision] rag search failed: {e}")
                chunks, stat = [], "empty_db"

        context_parts: list[str] = []
        if paper_id:
            target = next(
                (p for p in store._papers if str(p.get("id") or "") == paper_id), None
            )
            if target:
                context_parts.append(
                    "【目标文献】\n"
                    f"- 标题：{target.get('title') or target.get('display_title') or ''}\n"
                    f"- 摘要：{(target.get('abstract') or '')[:400]}\n"
                    f"- 关键词：{', '.join(str(k) for k in (target.get('keywords') or []))}"
                )

        for c in chunks[:4]:
            snippet = str(c.get("snippet") or "")[:360]
            if snippet:
                context_parts.append(
                    f"【片段 paper_id={c.get('paper_id')} chunk_id={c.get('chunk_id')}】\n{snippet}"
                )
        context = "\n\n".join(context_parts)

        # ── Phase C: 结合图 + 文献 → 最终回答 ────────────────────────
        system_prompt = (
            "你是一位博学、乐于助人的学术助手。用户会给你一张论文截图 + 一段文字提问，"
            "同时下方可能附带从用户文献库检索到的相关段落。回答原则：\n"
            "1) 直接基于图中的可见内容 + 检索段落作答；若有公式请逐符号解释变量含义。\n"
            "2) 若引用检索段落，在关键论点末尾标注 (paper_id=XXX)。\n"
            "3) 图里看不清的部分坦诚说明，不要瞎猜。\n"
            "4) 用简体中文，Markdown 结构化输出，公式一律 LaTeX（行内 $...$，独立 $$...$$）。"
        )

        user_parts: list[dict[str, Any]] = []
        if context:
            user_parts.append({"type": "text", "text": "参考文献片段：\n" + context})
        if description:
            user_parts.append({"type": "text", "text": "图片视觉描述（供参考，可能不完整）：\n" + description})
        user_parts.append({"type": "text", "text": "用户问题：\n" + q})
        user_parts.append(img_part)

        final_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_parts},
        ]

        try:
            answer = await call_llm(cfg, final_messages, temperature=0.4)
        except Exception as e:
            print(f"[vision] final call failed: {type(e).__name__}: {e}")
            raise HTTPException(status_code=503, detail=f"LLM 回答失败：{e}")

        cite_details = []
        for c in chunks:
            pid = c["paper_id"]
            snip = c["snippet"]
            page_no = store.locate_snippet_page(pid, snip)
            cite_details.append(
                {
                    "paper_id": pid,
                    "chunk_id": c["chunk_id"],
                    "snippet": snip,
                    "page": page_no,
                }
            )
        cites = list(dict.fromkeys([c["paper_id"] for c in cite_details]))

        return {
            "answer": (answer or "").strip(),
            "description": description,
            "cites": cites,
            "cite_details": cite_details,
            "provider_used": provider_id,
            "status": "ok",
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
