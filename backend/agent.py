from __future__ import annotations

"""
ReAct Agent — 多步工具调用的学术问答。

两种执行模式（按 LLM 能力自动选择）：
  - 原生函数调用（OpenAI 兼容协议）：模型直接返回结构化 tool_calls，
    无需正则解析，格式不会崩 —— 生产首选。
  - 文本 ReAct（Gemini 等无原生工具调用时回退）：
    LLM → Thought + Action → Observation 循环，正则解析。

无论哪种模式都遵守同一套规则：检索失败要重试/明确告知、严禁编造、
最终回答必须挂引用（paper_id + chunk）。
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable


# ------------------------------------------------------------------ #
# 工具定义
# ------------------------------------------------------------------ #

@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., str]
    # JSON Schema，用于原生函数调用；文本模式仅用 description
    parameters: dict[str, Any] = field(default_factory=lambda: {
        "type": "object", "properties": {},
    })


def _to_openai_tool(t: Tool) -> dict[str, Any]:
    """转成 OpenAI function-calling 工具 schema。"""
    return {
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
        },
    }


# ------------------------------------------------------------------ #
# 行为规则（两种模式共用）
# ------------------------------------------------------------------ #

_RULES = """行为规则（必须严格遵守）：
1. 失败恢复：若某次检索返回「未检索到」，先换同义词/上位词或限定具体某篇文献重试一次；
   仍为空则在回答中明确告知用户「文献库中没有相关内容」，绝不编造。
2. 严禁编造：所有结论必须来自工具返回的 Observation；不得使用文献库以外的知识臆测。
3. 强制引用：最终回答里每一个关键结论后面都要标注来源，格式 [paper_id=xxx, chunk=yyy]，
   来源取自工具返回片段里的 paper_id / chunk 标记。
4. 所有回答用简体中文，行内公式用 $...$，独立公式用 $$...$$。"""

_SYSTEM_TEXT_TMPL = """你是一个学术文献研究助手，可以调用以下工具来查阅用户的文献库：

{tool_descriptions}

每次回复**严格**遵守以下两种格式之一：

【需要调用工具时】
Thought: 分析当前状态，决定调用哪个工具
Action: 工具名称
Action Input: {{"参数名": "参数值"}}

【信息足够，直接回答时】
Final Answer: 完整回答（支持 Markdown 与 LaTeX）

- 每次只能调用一个工具
- 收到 Observation 后继续输出 Thought/Action，直到信息足够
- 最多调用 {max_steps} 次工具，超出后直接输出 Final Answer

{rules}
"""

_SYSTEM_TOOL_TMPL = """你是一个学术文献研究助手，可以调用工具查阅用户的文献库。
按需调用工具收集证据，证据足够后直接用自然语言给出最终回答（不要再调用工具）。
最多调用 {max_steps} 次工具。

{rules}
"""


# ------------------------------------------------------------------ #
# ReAct Agent
# ------------------------------------------------------------------ #

class ReActAgent:
    def __init__(
        self,
        tools: list[Tool],
        llm_call_fn: Callable,          # async fn(messages, temperature) -> str
        llm_tool_fn: Callable | None = None,  # async fn(messages, tools, temperature) -> {content, tool_calls}
        cite_sink: list[dict[str, Any]] | None = None,
        max_steps: int = 5,
    ) -> None:
        self.tools = {t.name: t for t in tools}
        self._llm = llm_call_fn
        self._llm_tool = llm_tool_fn
        self._cite_sink = cite_sink if cite_sink is not None else []
        self.max_steps = max_steps

    # -------------------------------------------------------------- #
    # 工具执行（两种模式共用）
    # -------------------------------------------------------------- #

    def _exec_tool(self, name: str, args: dict[str, Any]) -> str:
        if name not in self.tools:
            return (
                f"错误：工具 '{name}' 不存在。可用工具：{list(self.tools.keys())}"
            )
        try:
            return self.tools[name].fn(**(args or {}))
        except TypeError as e:
            return f"参数错误：{e}"
        except Exception as e:
            return f"工具执行失败：{e}"

    def _collect_cites(self) -> list[dict[str, Any]]:
        seen: set[tuple] = set()
        out: list[dict[str, Any]] = []
        for c in self._cite_sink:
            key = (c.get("paper_id"), c.get("chunk_id"))
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
        return out

    # -------------------------------------------------------------- #
    # 文本 ReAct 解析
    # -------------------------------------------------------------- #

    def _parse_action(self, text: str) -> tuple[str, dict[str, Any]] | None:
        m = re.search(
            r"Action:\s*(\w+)\s*\nAction Input:\s*(\{[\s\S]*?\})",
            text,
        )
        if not m:
            return None
        try:
            return m.group(1).strip(), json.loads(m.group(2))
        except (json.JSONDecodeError, ValueError):
            return None

    def _parse_final(self, text: str) -> str | None:
        if "Final Answer:" in text:
            return text.split("Final Answer:", 1)[-1].strip()
        return None

    # -------------------------------------------------------------- #
    # 主入口：优先原生函数调用，否则回退文本 ReAct
    # -------------------------------------------------------------- #

    async def run(
        self,
        user_query: str,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        if self._llm_tool is not None:
            try:
                return await self._run_tool_calling(user_query, history)
            except Exception as e:
                # 原生工具调用异常 → 退回文本模式，保证可用性
                print(f"[agent] tool-calling failed, fallback to text ReAct: "
                      f"{type(e).__name__}: {e}")
        return await self._run_text_react(user_query, history)

    # -------------------------------------------------------------- #
    # 模式 A：原生函数调用
    # -------------------------------------------------------------- #

    async def _run_tool_calling(
        self,
        user_query: str,
        history: list[dict[str, str]] | None,
    ) -> dict[str, Any]:
        system = _SYSTEM_TOOL_TMPL.format(max_steps=self.max_steps, rules=_RULES)
        tools_schema = [_to_openai_tool(t) for t in self.tools.values()]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            *(history or []),
            {"role": "user", "content": user_query},
        ]
        steps: list[dict[str, Any]] = []

        for step_n in range(self.max_steps):
            resp = await self._llm_tool(messages, tools_schema, temperature=0.3)
            tool_calls = resp.get("tool_calls") or []

            if not tool_calls:
                answer = (resp.get("content") or "").strip()
                return {
                    "answer": answer,
                    "steps": steps,
                    "cites": self._collect_cites(),
                    "mode": "agent",
                }

            messages.append({
                "role": "assistant",
                "content": resp.get("content") or "",
                "tool_calls": tool_calls,
            })

            for tc in tool_calls:
                name = tc.get("name") or ""
                args = tc.get("args") or {}
                observation = self._exec_tool(name, args)
                steps.append({
                    "step": step_n + 1,
                    "action": name,
                    "input": args,
                    "observation": observation[:600],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id") or "",
                    "content": observation,
                })

        # 达上限：禁用工具，强制收尾
        messages.append({
            "role": "user",
            "content": "已达到最大工具调用次数，请基于以上 Observation 直接给出最终回答，不要再调用工具。",
        })
        resp = await self._llm_tool(messages, [], temperature=0.4)
        return {
            "answer": (resp.get("content") or "").strip(),
            "steps": steps,
            "cites": self._collect_cites(),
            "mode": "agent",
        }

    # -------------------------------------------------------------- #
    # 模式 B：文本 ReAct（回退）
    # -------------------------------------------------------------- #

    async def _run_text_react(
        self,
        user_query: str,
        history: list[dict[str, str]] | None,
    ) -> dict[str, Any]:
        tool_descs = "\n".join(
            f"- **{t.name}**: {t.description}" for t in self.tools.values()
        )
        system = _SYSTEM_TEXT_TMPL.format(
            tool_descriptions=tool_descs,
            max_steps=self.max_steps,
            rules=_RULES,
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            *(history or []),
            {"role": "user", "content": user_query},
        ]
        steps: list[dict[str, Any]] = []

        for step_n in range(self.max_steps):
            response: str = await self._llm(messages, temperature=0.3)
            messages.append({"role": "assistant", "content": response})

            final = self._parse_final(response)
            if final:
                return {
                    "answer": final, "steps": steps,
                    "cites": self._collect_cites(), "mode": "agent",
                }

            parsed = self._parse_action(response)
            if not parsed:
                return {
                    "answer": response.strip(), "steps": steps,
                    "cites": self._collect_cites(), "mode": "agent",
                }

            tool_name, tool_input = parsed
            observation = self._exec_tool(tool_name, tool_input)
            steps.append({
                "step": step_n + 1,
                "action": tool_name,
                "input": tool_input,
                "observation": observation[:600],
            })
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}",
            })

        messages.append({
            "role": "user",
            "content": "已达到最大工具调用次数，请基于以上所有 Observation 输出 Final Answer。",
        })
        summary = await self._llm(messages, temperature=0.4)
        final = self._parse_final(summary) or summary.strip()
        return {
            "answer": final, "steps": steps,
            "cites": self._collect_cites(), "mode": "agent",
        }


# ------------------------------------------------------------------ #
# 工厂函数：绑定 store + llm，返回配好工具的 Agent 实例
# ------------------------------------------------------------------ #

def _coerce_id_list(value: Any) -> list[str]:
    """paper_ids 可能是 list、逗号分隔字符串或单个字符串，统一成 list[str]。"""
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [s.strip() for s in re.split(r"[,，\s]+", value) if s.strip()]
    return []


def build_agent(
    store: Any,
    llm_call_fn: Callable,
    llm_tool_fn: Callable | None = None,
) -> ReActAgent:
    """
    llm_call_fn: async (messages, temperature) -> str
    llm_tool_fn: async (messages, tools, temperature) -> {content, tool_calls}；
                 为 None 时走文本 ReAct 回退。
    store: ScholarStore 实例
    """
    cite_sink: list[dict[str, Any]] = []

    def _add_cite(paper_id: Any, chunk_id: Any) -> None:
        if paper_id:
            cite_sink.append({"paper_id": str(paper_id), "chunk_id": chunk_id})

    def search(query: str, paper_id: str | None = None, top_k: int = 4) -> str:
        """语义检索文献片段。"""
        results, status = store.search_chunks(
            query, top_k=int(top_k), paper_id=paper_id or None
        )
        if status != "ok" or not results:
            return (
                "未在文献库中检索到相关内容。"
                "建议：换同义词/上位词重试，或先用 list_papers 确认库里有哪些文献。"
            )
        parts = []
        for r in results:
            _add_cite(r.get("paper_id"), r.get("chunk_id"))
            parts.append(
                f"[paper_id={r['paper_id']} chunk={r.get('chunk_id')}]\n"
                f"{str(r.get('full_text') or r.get('snippet', ''))[:500]}"
            )
        return "\n\n---\n\n".join(parts)

    def get_paper_meta(paper_id: str) -> str:
        """获取某篇论文的元数据。"""
        paper = store.get_paper(paper_id)
        if not paper:
            return f"未找到 paper_id={paper_id!r} 的论文。请先用 list_papers 确认 ID。"
        title = paper.get("display_title") or paper.get("title") or ""
        abstract = (paper.get("llm_summary") or paper.get("abstract") or "")[:900]
        kws = ", ".join(str(k) for k in (paper.get("keywords") or [])[:12])
        return f"标题: {title}\n关键词: {kws}\n摘要: {abstract}"

    def list_papers() -> str:
        """列出文献库所有论文及其 paper_id。"""
        with store._lock:
            papers = list(store._papers)
        if not papers:
            return "文献库为空。"
        lines = [f"共 {len(papers)} 篇文献："]
        for i, p in enumerate(papers, 1):
            title = (
                p.get("display_title") or p.get("title")
                or p.get("filename") or "未知"
            )
            lines.append(f"{i}. paper_id={p['id']}  标题: {title}")
        return "\n".join(lines)

    def compare_papers(paper_ids: Any, dimension: str, top_k: int = 2) -> str:
        """在同一维度上并行检索多篇文献，返回对齐的片段，便于横向对比。"""
        ids = _coerce_id_list(paper_ids)
        if not ids:
            return "参数错误：paper_ids 不能为空（用 list_papers 获取 ID）。"
        if not str(dimension or "").strip():
            return "参数错误：dimension 不能为空（如：方法 / 数据集 / 结论）。"
        blocks = []
        for pid in ids:
            meta = store.get_paper(pid)
            title = (meta or {}).get("display_title") or (meta or {}).get("title") or pid
            results, status = store.search_chunks(dimension, top_k=int(top_k), paper_id=pid)
            if status != "ok" or not results:
                blocks.append(f"### {title} (paper_id={pid})\n该维度下未检索到相关内容。")
                continue
            seg = []
            for r in results:
                _add_cite(pid, r.get("chunk_id"))
                seg.append(
                    f"[chunk={r.get('chunk_id')}] "
                    f"{str(r.get('full_text') or r.get('snippet') or '')[:400]}"
                )
            blocks.append(f"### {title} (paper_id={pid})\n" + "\n".join(seg))
        return f"维度「{dimension}」跨文献对比：\n\n" + "\n\n".join(blocks)

    def get_chunk_context(chunk_id: str, window: int = 1) -> str:
        """取某个检索片段的前后相邻文本，避免断章取义。"""
        ctx = store.get_chunk_context(chunk_id, window=int(window))
        if not ctx:
            return f"未找到 chunk_id={chunk_id!r} 或其上下文。"
        _add_cite(ctx["paper_id"], chunk_id)
        parts = [f"paper_id={ctx['paper_id']} 中心块 index={ctx['center_index']} 的上下文："]
        for c in ctx["chunks"]:
            marker = "→" if int(c["index"]) == ctx["center_index"] else "  "
            parts.append(
                f"{marker} [index={c['index']} chunk={c['chunk_id']}] "
                f"{str(c['text'])[:300]}"
            )
        return "\n".join(parts)

    tools = [
        Tool(
            name="search",
            description=(
                "在文献库中做语义检索，返回最相关的段落片段（含 paper_id/chunk 来源）。"
                "参数: query(必填)检索词或问题句；paper_id(可选)限定某篇；top_k(可选,默认4)条数。"
            ),
            fn=search,
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索词或问题句"},
                    "paper_id": {"type": "string", "description": "可选，限定某篇文献"},
                    "top_k": {"type": "integer", "description": "返回条数，默认4"},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_paper_meta",
            description="获取某篇论文的标题、关键词、摘要。参数: paper_id(必填)。",
            fn=get_paper_meta,
            parameters={
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string", "description": "论文 ID"},
                },
                "required": ["paper_id"],
            },
        ),
        Tool(
            name="list_papers",
            description="列出文献库中所有论文的 paper_id 和标题。无参数。",
            fn=list_papers,
            parameters={"type": "object", "properties": {}},
        ),
        Tool(
            name="compare_papers",
            description=(
                "在同一维度上并行检索多篇文献并对齐返回，专用于横向对比。"
                "参数: paper_ids(必填,列表或逗号分隔)；dimension(必填,对比维度，如方法/数据集/结论)；"
                "top_k(可选,默认2)每篇取几段。"
            ),
            fn=compare_papers,
            parameters={
                "type": "object",
                "properties": {
                    "paper_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要对比的论文 ID 列表",
                    },
                    "dimension": {"type": "string", "description": "对比维度，如 方法/数据集/结论"},
                    "top_k": {"type": "integer", "description": "每篇取几段，默认2"},
                },
                "required": ["paper_ids", "dimension"],
            },
        ),
        Tool(
            name="get_chunk_context",
            description=(
                "取某个检索片段的前后相邻文本，避免断章取义。"
                "参数: chunk_id(必填,来自 search 返回)；window(可选,默认1)前后各取几块。"
            ),
            fn=get_chunk_context,
            parameters={
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string", "description": "search 返回的 chunk 标识"},
                    "window": {"type": "integer", "description": "前后各取几块，默认1"},
                },
                "required": ["chunk_id"],
            },
        ),
    ]

    return ReActAgent(
        tools,
        llm_call_fn,
        llm_tool_fn=llm_tool_fn,
        cite_sink=cite_sink,
        max_steps=5,
    )


# ------------------------------------------------------------------ #
# 复杂问题路由：正则快筛 + LLM 意图分类
# ------------------------------------------------------------------ #

_AGENT_TRIGGERS = re.compile(
    r"(比较|对比|异同|区别|综合|汇总|总结.*所有|哪篇|哪些论文|"
    r"跨.*文献|多篇|所有.*论文|论文.*对比|compare|versus|vs\.)",
    re.IGNORECASE,
)


def should_use_agent(question: str) -> bool:
    """启发式快筛：正则命中跨文献/对比类关键词。"""
    return bool(_AGENT_TRIGGERS.search(question))


async def classify_intent(question: str, llm_call_fn: Callable) -> str:
    """
    用一次轻量 LLM 调用判断问题该走 Agent（multi）还是单篇 RAG（single）。
    失败时回退到正则快筛。返回 "multi" 或 "single"。
    """
    msgs = [
        {
            "role": "system",
            "content": (
                "你是问题路由器。判断用户问题属于哪类，只回一个英文单词：\n"
                "- multi：需要跨多篇文献检索、对比、综合、汇总，或要先找出是哪篇\n"
                "- single：针对单篇或单一事实的定向问答\n"
                "只输出 single 或 multi，不要其他任何字符。"
            ),
        },
        {"role": "user", "content": question},
    ]
    try:
        raw = (await llm_call_fn(msgs, temperature=0.0)).strip().lower()
    except Exception:
        return "multi" if should_use_agent(question) else "single"
    if "multi" in raw:
        return "multi"
    if "single" in raw:
        return "single"
    return "multi" if should_use_agent(question) else "single"
