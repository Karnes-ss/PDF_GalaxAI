"""
LLM 统一调用层。

按模型配置里的 `protocol` 派发：
- openai : OpenAI 兼容接口（OpenAI / DeepSeek / Kimi / 智谱 / Groq / LMStudio / Ollama）
- gemini : Google Gemini generateContent

网络模式（proxy 字段）：
- ''/'auto'          ：遵从系统环境变量 HTTP(S)_PROXY（同时兼容 TUN/虚拟网卡模式，
                       因为 TUN 模式根本不依赖 env var，直接透明接管）
- 'direct'           ：绕过所有代理与环境变量，直连 base_url
                       - 适合 TUN 模式同时 env 里留有旧代理的场景
                       - 本机服务（ollama/lmstudio）永远走这个
- 'http://host:port' / 'socks5://host:port'：显式走指定代理

调用方式：
    answer = await call_llm(cfg, messages, temperature=0.5)
"""

from __future__ import annotations

import base64
from typing import Any
from urllib.parse import urlparse

import httpx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI


# ---------------------------------------------------------------- #
# 多模态辅助
# ---------------------------------------------------------------- #

def _is_vision_capable(cfg: dict[str, Any]) -> bool:
    """
    粗略判断一个模型是否支持图片输入。
    Gemini 全系列都支持；OpenAI 需要看具体 model 名字。
    """
    proto = str(cfg.get("protocol") or "openai").strip().lower()
    if proto == "gemini":
        return True
    model = str(cfg.get("model") or "").lower()
    vision_hints = (
        "gpt-4o", "gpt-4.1", "gpt-5", "gpt-4-turbo",
        "claude-3.5", "claude-3-7", "claude-4",
        "qwen-vl", "qwen2-vl", "qwen2.5-vl",
        "llava", "yi-vl",
        "gemini",  # 兼容自定义兼容层
        "internvl",
    )
    return any(h in model for h in vision_hints)


def _content_parts_text_only(content: Any) -> str:
    """把 content（字符串或分段列表）展平成纯文本用于老协议 fallback。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        buf: list[str] = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                buf.append(str(p.get("text") or ""))
        return "\n".join(buf)
    return str(content or "")


def _is_localhost(url: str) -> bool:
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        host = ""
    return host in {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def _resolve_proxy_mode(cfg: dict[str, Any]) -> tuple[bool, str | None]:
    """
    返回 (trust_env, explicit_proxy_url)：
    - trust_env=True  → 让 httpx 读 HTTP(S)_PROXY 环境变量
    - explicit_proxy_url 非空 → 直接用指定代理，忽略环境变量
    localhost 一律 direct。
    """
    base_url = str(cfg.get("base_url") or "")
    if _is_localhost(base_url):
        return False, None

    raw = str(cfg.get("proxy") or "").strip().lower()

    if raw == "direct":
        return False, None
    if raw == "" or raw == "auto":
        return True, None
    # 显式代理 URL
    # （保留原大小写/路径，上面只做了判等用的 lower）
    explicit = str(cfg.get("proxy") or "").strip()
    return False, explicit


def _build_client(cfg: dict[str, Any]) -> httpx.AsyncClient:
    timeout = httpx.Timeout(connect=15.0, read=180.0, write=30.0, pool=30.0)
    trust_env, proxy = _resolve_proxy_mode(cfg)
    kwargs: dict[str, Any] = {"timeout": timeout, "trust_env": trust_env}
    if proxy:
        # httpx 0.27+ 用 `proxy=`，老版本是 `proxies=`；这里两者都兼容
        try:
            return httpx.AsyncClient(proxy=proxy, **kwargs)
        except TypeError:
            return httpx.AsyncClient(proxies=proxy, **kwargs)
    return httpx.AsyncClient(**kwargs)


def _openai_translate_messages(
    messages: list[dict[str, Any]],
    vision: bool,
) -> list[dict[str, Any]]:
    """
    将内部统一格式（content 可以是 str 或 [{type:text/image,...}]）
    转换成 OpenAI Chat Completions 的格式：
      - vision=True: 保留分段，image → {type:image_url,image_url:{url:'data:<mime>;base64,<data>'}}
      - vision=False: 拍扁成纯文本，丢弃图片
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content")
        if isinstance(content, list):
            if vision:
                parts: list[dict[str, Any]] = []
                for p in content:
                    if not isinstance(p, dict):
                        continue
                    ptype = p.get("type")
                    if ptype == "text":
                        parts.append({"type": "text", "text": str(p.get("text") or "")})
                    elif ptype == "image":
                        data = str(p.get("data") or "")
                        mime = str(p.get("mime") or "image/png")
                        url = f"data:{mime};base64,{data}"
                        parts.append({"type": "image_url", "image_url": {"url": url}})
                out.append({"role": role, "content": parts})
            else:
                out.append({"role": role, "content": _content_parts_text_only(content)})
        else:
            out.append({"role": role, "content": str(content or "")})
    return out


def _to_lc_messages(
    messages: list[dict[str, Any]],
    vision: bool,
) -> list[BaseMessage]:
    """将内部统一格式转换为 LangChain BaseMessage 列表。"""
    translated = _openai_translate_messages(messages, vision=vision)
    out: list[BaseMessage] = []
    for m in translated:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            out.append(SystemMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
        else:
            out.append(HumanMessage(content=content))
    return out


def _lc_content_to_text(content: Any) -> str:
    """把 LangChain 响应的 content（可能是 str 或分段列表）展平成纯文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        buf: list[str] = []
        for p in content:
            if isinstance(p, dict):
                buf.append(str(p.get("text") or ""))
            else:
                buf.append(str(p))
        return "".join(buf)
    return str(content or "")


async def _call_openai_compatible(
    cfg: dict[str, Any],
    messages: list[dict[str, Any]],
    temperature: float,
) -> str:
    base_url = str(cfg.get("base_url") or "").rstrip("/")
    if not base_url:
        raise RuntimeError("模型配置缺少 base_url")
    model = str(cfg.get("model") or "")
    if not model:
        raise RuntimeError("模型配置缺少 model")

    # ChatOpenAI 强制要求非空 api_key；本地服务（Ollama/LMStudio）会忽略它。
    api_key = str(cfg.get("api_key") or "").strip() or "no-key"

    lc_messages = _to_lc_messages(messages, vision=_is_vision_capable(cfg))

    # 复用既有代理逻辑：把配好代理的 httpx 客户端注入 ChatOpenAI。
    http_client = _build_client(cfg)
    try:
        llm = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_retries=2,
            http_async_client=http_client,
        )
        resp = await llm.ainvoke(lc_messages)
    finally:
        await http_client.aclose()

    answer = _lc_content_to_text(resp.content).strip()
    if not answer:
        raise RuntimeError("LLM 返回空内容")
    return answer


async def _call_gemini(
    cfg: dict[str, Any],
    messages: list[dict[str, Any]],
    temperature: float,
) -> str:
    api_key = str(cfg.get("api_key") or "").strip()
    if not api_key:
        raise RuntimeError("未配置 Gemini API Key")
    base_url = str(cfg.get("base_url") or "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
    model = str(cfg.get("model") or "gemini-2.5-flash")
    url = f"{base_url}/models/{model}:generateContent?key={api_key}"

    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content")

        if role == "system":
            # system 只接纯文本（Gemini systemInstruction 不支持图片）
            txt = _content_parts_text_only(content)
            if txt:
                system_parts.append(txt)
            continue

        gemini_role = "model" if role == "assistant" else "user"
        parts_out: list[dict[str, Any]] = []
        if isinstance(content, list):
            for p in content:
                if not isinstance(p, dict):
                    continue
                ptype = p.get("type")
                if ptype == "text":
                    txt = str(p.get("text") or "")
                    if txt:
                        parts_out.append({"text": txt})
                elif ptype == "image":
                    data = str(p.get("data") or "")
                    mime = str(p.get("mime") or "image/png")
                    if data:
                        parts_out.append({"inline_data": {"mime_type": mime, "data": data}})
        else:
            txt = str(content or "")
            if txt:
                parts_out.append({"text": txt})

        if parts_out:
            contents.append({"role": gemini_role, "parts": parts_out})

    if not contents:
        contents = [{"role": "user", "parts": [{"text": "你好"}]}]

    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": {"temperature": temperature},
    }
    if system_parts:
        payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}

    async with _build_client(cfg) as client:
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


async def call_llm(
    cfg: dict[str, Any],
    messages: list[dict[str, Any]],
    temperature: float = 0.5,
) -> str:
    proto = str(cfg.get("protocol") or "openai").strip().lower()
    if proto == "gemini":
        return await _call_gemini(cfg, messages, temperature)
    return await _call_openai_compatible(cfg, messages, temperature)


def is_vision_capable(cfg: dict[str, Any]) -> bool:
    """供外部路由层判断一个模型是否能处理图片。"""
    return _is_vision_capable(cfg)


def encode_image_to_base64(raw: bytes, mime: str = "image/png") -> dict[str, Any]:
    """构造一个图片 content part，方便上层拼 messages。"""
    return {
        "type": "image",
        "mime": mime,
        "data": base64.b64encode(raw).decode("ascii"),
    }


async def test_connection(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    发一条极短的探测消息，验证配置是否能正常连通。
    返回里会把实际使用的网络模式回写到 `proxy_mode`，便于前端诊断。
    """
    trust_env, proxy = _resolve_proxy_mode(cfg)
    if _is_localhost(str(cfg.get("base_url") or "")):
        proxy_mode = "direct (localhost)"
    elif proxy:
        proxy_mode = f"proxy: {proxy}"
    elif trust_env:
        proxy_mode = "auto (trust system env)"
    else:
        proxy_mode = "direct"

    test_messages = [
        {"role": "system", "content": "只回复两个字符：OK"},
        {"role": "user", "content": "ping"},
    ]
    try:
        sample = await call_llm(cfg, test_messages, temperature=0.0)
        return {"ok": True, "message": "连接成功", "sample": sample[:60], "proxy_mode": proxy_mode}
    except Exception as e:
        return {
            "ok": False,
            "message": f"{type(e).__name__}: {e}",
            "proxy_mode": proxy_mode,
        }
