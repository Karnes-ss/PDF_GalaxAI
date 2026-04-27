"""
LLM 模型注册表。

职责：
- 暴露一份"可用模型"清单给前端：内置（local Ollama / Gemini）+ 用户自定义
- 自定义模型配置持久化到 data/custom_models.json
- 提供 CRUD + 按 id 取完整配置（供 llm_client 调度）

两种协议（足以覆盖主流云厂商 + 本地推理）：
- openai    : OpenAI 兼容 /v1/chat/completions（OpenAI、DeepSeek、Kimi、智谱 GLM、
              Groq、SiliconFlow、LMStudio、Ollama…）
- gemini    : Google Gemini generateContent（只要协议格式一致的镜像也可）
"""

from __future__ import annotations

import json
import os
import re
import uuid
from threading import Lock
from typing import Any

from config import CUSTOM_MODELS_JSON


_BUILTIN_IDS = {"local", "gemini"}


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
    key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or _load_env_file_var("GEMINI_API_KEY")
    )
    return (key or "").strip()


def _ollama_model() -> str:
    return (os.getenv("SCHOLAR_OLLAMA_MODEL") or "qwen2.5:3b").strip()


def _gemini_model() -> str:
    return (os.getenv("SCHOLAR_GEMINI_MODEL") or "gemini-2.5-flash").strip()


def _default_builtin_proxy() -> str:
    """
    为内置 Gemini（云端）推断一个合理的 proxy 默认值。
    用户可以通过 .env 显式声明：
        SCHOLAR_LLM_PROXY=auto            # 默认。遵从系统环境变量 HTTP(S)_PROXY
        SCHOLAR_LLM_PROXY=direct          # 绕过所有代理（适合 TUN/虚拟网卡模式）
        SCHOLAR_LLM_PROXY=http://127.0.0.1:7890  # 显式指定代理
    """
    v = (
        os.getenv("SCHOLAR_LLM_PROXY")
        or _load_env_file_var("SCHOLAR_LLM_PROXY")
        or ""
    ).strip()
    return v


def _mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "*" * len(key)
    return key[:4] + "*" * (len(key) - 8) + key[-4:]


class ModelsRegistry:
    """
    线程安全。
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._customs: list[dict[str, Any]] = []
        self._load()

    # ---------------------------------------------------------------- #
    # 内置模型
    # ---------------------------------------------------------------- #

    def _builtin(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "local",
                "label": "本地 Ollama",
                "protocol": "openai",
                "base_url": "http://127.0.0.1:11434/v1",
                "model": _ollama_model(),
                "api_key": "",
                # 本机端口永远不走代理
                "proxy": "direct",
                "builtin": True,
                "requires_key": False,
            },
            {
                "id": "gemini",
                "label": "云端 Gemini",
                "protocol": "gemini",
                "base_url": "https://generativelanguage.googleapis.com/v1beta",
                "model": _gemini_model(),
                "api_key": _current_gemini_api_key(),
                # 默认 auto：遵从系统 HTTP(S)_PROXY；可通过 SCHOLAR_LLM_PROXY 覆盖
                "proxy": _default_builtin_proxy(),
                "builtin": True,
                "requires_key": True,
            },
        ]

    # ---------------------------------------------------------------- #
    # 持久化
    # ---------------------------------------------------------------- #

    def _load(self) -> None:
        if not CUSTOM_MODELS_JSON.exists():
            return
        try:
            data = json.loads(CUSTOM_MODELS_JSON.read_text(encoding="utf-8"))
            raw = data.get("models", [])
            cleaned: list[dict[str, Any]] = []
            for m in raw:
                if not isinstance(m, dict):
                    continue
                if not m.get("id") or not m.get("label"):
                    continue
                if m.get("id") in _BUILTIN_IDS:
                    continue
                cleaned.append(self._normalize(m))
            self._customs = cleaned
            print(f"[models] Loaded {len(self._customs)} custom models")
        except Exception as e:
            print(f"[models] Failed to load custom_models.json: {e}")

    def _save(self) -> None:
        try:
            CUSTOM_MODELS_JSON.write_text(
                json.dumps({"models": self._customs}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            print(f"[models] Failed to save custom_models.json: {e}")

    # ---------------------------------------------------------------- #
    # 规范化 / 校验
    # ---------------------------------------------------------------- #

    @staticmethod
    def _normalize(raw: dict[str, Any]) -> dict[str, Any]:
        proto = str(raw.get("protocol") or "openai").strip().lower()
        if proto not in {"openai", "gemini"}:
            proto = "openai"
        base_url = str(raw.get("base_url") or "").strip().rstrip("/")
        proxy = str(raw.get("proxy") or "").strip()
        return {
            "id": str(raw.get("id") or "").strip(),
            "label": str(raw.get("label") or "").strip()[:64],
            "protocol": proto,
            "base_url": base_url,
            "model": str(raw.get("model") or "").strip()[:128],
            "api_key": str(raw.get("api_key") or "").strip(),
            "proxy": proxy[:256],
            "builtin": False,
            "requires_key": True,
        }

    @staticmethod
    def _validate_proxy(v: str) -> str | None:
        """合法值：'' / 'auto' / 'direct' / http(s)://... / socks5(h)://..."""
        if not v or v in {"auto", "direct"}:
            return None
        if not re.match(r"^(https?|socks5h?)://", v):
            return "proxy 仅支持 auto / direct / http(s)://... / socks5(h)://..."
        return None

    @staticmethod
    def _validate_for_create(payload: dict[str, Any]) -> str | None:
        """返回错误消息，或 None 表示通过。"""
        if not payload.get("label"):
            return "label 不能为空"
        proto = payload.get("protocol")
        if proto not in {"openai", "gemini"}:
            return "protocol 仅支持 openai / gemini"
        if proto == "openai":
            if not payload.get("base_url"):
                return "OpenAI 兼容协议必须填 base_url（例如 https://api.deepseek.com/v1）"
            if not re.match(r"^https?://", payload["base_url"]):
                return "base_url 必须是 http(s):// 开头的完整 URL"
        if not payload.get("model"):
            return "必须填写 model（如 gpt-4o-mini / deepseek-chat / moonshot-v1-8k）"
        proxy_err = ModelsRegistry._validate_proxy(str(payload.get("proxy") or ""))
        if proxy_err:
            return proxy_err
        return None

    # ---------------------------------------------------------------- #
    # 公开 API
    # ---------------------------------------------------------------- #

    def list_public(self) -> list[dict[str, Any]]:
        """给前端的列表：api_key 以掩码形式返回，不泄露明文。"""
        with self._lock:
            customs = list(self._customs)
        items: list[dict[str, Any]] = []
        for m in self._builtin() + customs:
            item = dict(m)
            key = str(item.get("api_key") or "")
            item["api_key"] = ""         # 绝不返回明文
            item["api_key_mask"] = _mask_key(key)
            item["has_api_key"] = bool(key)
            # proxy 字段始终存在，方便前端直接读
            item["proxy"] = str(item.get("proxy") or "")
            items.append(item)
        return items

    def get_config(self, model_id: str) -> dict[str, Any] | None:
        """内部使用：取完整配置（含 api_key 明文），用于实际调用 LLM。"""
        if not model_id:
            return None
        for m in self._builtin():
            if m["id"] == model_id:
                return dict(m)
        with self._lock:
            for m in self._customs:
                if m["id"] == model_id:
                    return dict(m)
        return None

    def add(self, payload: dict[str, Any]) -> dict[str, Any]:
        proto = str(payload.get("protocol") or "openai").strip().lower()
        record = {
            "id": "cm_" + uuid.uuid4().hex[:10],
            "label": str(payload.get("label") or "").strip(),
            "protocol": proto,
            "base_url": str(payload.get("base_url") or "").strip().rstrip("/"),
            "model": str(payload.get("model") or "").strip(),
            "api_key": str(payload.get("api_key") or "").strip(),
            "proxy": str(payload.get("proxy") or "").strip(),
        }
        err = self._validate_for_create(record)
        if err:
            raise ValueError(err)
        record = self._normalize(record)
        with self._lock:
            self._customs.append(record)
            self._save()
        out = dict(record)
        out["api_key"] = ""
        out["api_key_mask"] = _mask_key(record["api_key"])
        out["has_api_key"] = bool(record["api_key"])
        return out

    def update(self, model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        if model_id in _BUILTIN_IDS:
            raise ValueError("内置模型不可修改")
        with self._lock:
            idx = next((i for i, m in enumerate(self._customs) if m["id"] == model_id), -1)
            if idx < 0:
                raise KeyError("model not found")
            old = dict(self._customs[idx])

        merged = {
            "id": model_id,
            "label": payload.get("label", old["label"]),
            "protocol": payload.get("protocol", old["protocol"]),
            "base_url": payload.get("base_url", old["base_url"]),
            "model": payload.get("model", old["model"]),
            # api_key：空字符串或未提供都表示"不修改"，保留旧值
            "api_key": (
                str(payload.get("api_key") or "").strip()
                or old["api_key"]
            ),
            # proxy：显式传入才更新（哪怕传空串表示想清空）
            "proxy": (
                payload["proxy"]
                if "proxy" in payload and payload["proxy"] is not None
                else old.get("proxy", "")
            ),
        }
        err = self._validate_for_create(merged)
        if err:
            raise ValueError(err)
        record = self._normalize(merged)
        with self._lock:
            self._customs[idx] = record
            self._save()
        out = dict(record)
        out["api_key"] = ""
        out["api_key_mask"] = _mask_key(record["api_key"])
        out["has_api_key"] = bool(record["api_key"])
        return out

    def delete(self, model_id: str) -> bool:
        if model_id in _BUILTIN_IDS:
            raise ValueError("内置模型不可删除")
        with self._lock:
            before = len(self._customs)
            self._customs = [m for m in self._customs if m["id"] != model_id]
            removed = len(self._customs) < before
            if removed:
                self._save()
        return removed
