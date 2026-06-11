from __future__ import annotations

"""
ReAct Agent — 多步工具调用的学术问答。

流程：
  LLM → Thought + Action → 工具执行 → Observation → 循环 → Final Answer

只在用户提出跨文献/对比/综合类问题时触发，单篇 QA 仍走原有 RAG 路径。
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Callable


# ------------------------------------------------------------------ #
# 工具定义
# ------------------------------------------------------------------ #

@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., str]


# ------------------------------------------------------------------ #
# ReAct Agent
# ------------------------------------------------------------------ #

_SYSTEM_TMPL = """你是一个学术文献研究助手，可以调用以下工具来查阅用户的文献库：

{tool_descriptions}

每次回复**严格**遵守以下两种格式之一：

【需要调用工具时】
Thought: 分析当前状态，决定调用哪个工具
Action: 工具名称
Action Input: {{"参数名": "参数值"}}

【信息足够，直接回答时】
Final Answer: 完整回答（支持 Markdown 与 LaTeX 公式 $...$）

规则：
- 每次只能调用一个工具
- 收到 Observation 后继续输出 Thought/Action，直到有足够信息
- 最多调用 {max_steps} 次工具，超出后直接输出 Final Answer
- 所有回答用简体中文，行内公式用 $...$，独立公式用 $$...$$
"""


class ReActAgent:
    def __init__(
        self,
        tools: list[Tool],
        llm_call_fn: Callable,  # async fn(messages, temperature) -> str
        max_steps: int = 5,
    ) -> None:
        self.tools = {t.name: t for t in tools}
        self._llm = llm_call_fn
        self.max_steps = max_steps

    # -------------------------------------------------------------- #
    # 解析
    # -------------------------------------------------------------- #

    def _parse_action(self, text: str) -> tuple[str, dict[str, Any]] | None:
        """从 LLM 回复里提取 Action 和 Action Input。"""
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
    # 主循环
    # -------------------------------------------------------------- #

    async def run(
        self,
        user_query: str,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        tool_descs = "\n".join(
            f"- **{t.name}**: {t.description}" for t in self.tools.values()
        )
        system = _SYSTEM_TMPL.format(
            tool_descriptions=tool_descs,
            max_steps=self.max_steps,
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

            # 优先检测 Final Answer
            final = self._parse_final(response)
            if final:
                return {"answer": final, "steps": steps, "mode": "agent"}

            # 解析工具调用
            parsed = self._parse_action(response)
            if not parsed:
                # LLM 没遵守格式 → 把当前回复当作最终答案
                return {
                    "answer": response.strip(),
                    "steps": steps,
                    "mode": "agent",
                }

            tool_name, tool_input = parsed

            if tool_name not in self.tools:
                observation = (
                    f"错误：工具 '{tool_name}' 不存在。"
                    f"可用工具：{list(self.tools.keys())}"
                )
            else:
                try:
                    observation = self.tools[tool_name].fn(**tool_input)
                except TypeError as e:
                    observation = f"参数错误：{e}"
                except Exception as e:
                    observation = f"工具执行失败：{e}"

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

        # 超过 max_steps：让 LLM 汇总
        messages.append({
            "role": "user",
            "content": "已达到最大工具调用次数，请基于以上所有 Observation 输出 Final Answer。",
        })
        summary = await self._llm(messages, temperature=0.4)
        final = self._parse_final(summary) or summary.strip()
        return {"answer": final, "steps": steps, "mode": "agent"}


# ------------------------------------------------------------------ #
# 工厂函数：绑定 store + llm，返回配好工具的 Agent 实例
# ------------------------------------------------------------------ #

def build_agent(store: Any, llm_call_fn: Callable) -> ReActAgent:
    """
    llm_call_fn: async (messages: list[dict], temperature: float) -> str
    store: ScholarStore 实例
    """

    def search(query: str, paper_id: str | None = None, top_k: int = 4) -> str:
        """语义检索文献片段。"""
        results, status = store.search_chunks(
            query, top_k=int(top_k), paper_id=paper_id or None
        )
        if status != "ok" or not results:
            return "未在文献库中检索到相关内容。"
        parts = [
            f"[paper_id={r['paper_id']} chunk={r['chunk_id']}]\n"
            f"{str(r.get('full_text') or r.get('snippet', ''))[:500]}"
            for r in results
        ]
        return "\n\n---\n\n".join(parts)

    def get_paper_meta(paper_id: str) -> str:
        """获取某篇论文的元数据。"""
        paper = store.get_paper(paper_id)
        if not paper:
            return f"未找到 paper_id={paper_id!r} 的论文。"
        title = paper.get("display_title") or paper.get("title") or ""
        abstract = (
            paper.get("llm_summary") or paper.get("abstract") or ""
        )[:900]
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
                p.get("display_title")
                or p.get("title")
                or p.get("filename")
                or "未知"
            )
            lines.append(f"{i}. paper_id={p['id']}  标题: {title}")
        return "\n".join(lines)

    tools = [
        Tool(
            name="search",
            description=(
                "在文献库中做语义检索，返回最相关的段落片段。"
                "参数: query(str,必填)—检索词或问题句；"
                "paper_id(str,可选)—限定某篇文献；"
                "top_k(int,可选,默认4)—返回条数。"
            ),
            fn=search,
        ),
        Tool(
            name="get_paper_meta",
            description=(
                "获取某篇论文的标题、关键词、摘要。"
                "参数: paper_id(str,必填)—论文 ID（从 list_papers 获取）。"
            ),
            fn=get_paper_meta,
        ),
        Tool(
            name="list_papers",
            description="列出文献库中所有论文的 paper_id 和标题。无参数。",
            fn=list_papers,
        ),
    ]

    return ReActAgent(tools, llm_call_fn, max_steps=5)


# ------------------------------------------------------------------ #
# 复杂问题检测（供 api.py 用于自动路由）
# ------------------------------------------------------------------ #

_AGENT_TRIGGERS = re.compile(
    r"(比较|对比|异同|区别|综合|汇总|总结.*所有|哪篇|哪些论文|"
    r"跨.*文献|多篇|所有.*论文|论文.*对比|compare|versus|vs\.)",
    re.IGNORECASE,
)


def should_use_agent(question: str) -> bool:
    """启发式判断：是否应走 Agent 多步推理而非单次 RAG。"""
    return bool(_AGENT_TRIGGERS.search(question))
