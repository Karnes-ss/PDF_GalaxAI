"""
MinerU PDF 解析封装层。

为什么用 subprocess 而不是 import？
  MinerU 依赖 PyTorch + Paddle + 一大堆模型权重（~3GB），如果和当前后端共享
  Python env，容易出现 numpy/torch 版本冲突。用 CLI subprocess 隔离，用户可以
  把 MinerU 装在独立 conda env 里，只要把可执行文件路径写进 SCHOLAR_MINERU_CMD。

用法：
    text = parse_pdf_with_mineru(Path("x.pdf"))  # 返回合并后的 markdown，含 LaTeX

环境变量：
  SCHOLAR_MINERU_CMD   可执行文件路径，默认 "mineru"（假设在 PATH 里）
                       Windows 示例：C:\\ProgramData\\miniconda3\\envs\\mineru\\Scripts\\mineru.exe
  SCHOLAR_MINERU_LANG  识别语种，默认 "ch"（可选 en / ch / korean / japan ...）
  SCHOLAR_MINERU_BACKEND  "pipeline" | "vlm-transformers" | "vlm-sglang-client"
                          默认 pipeline（最稳，不要额外起 VLM 服务）
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


class MinerUNotAvailable(RuntimeError):
    """MinerU 不存在 / 不可调用。"""


class MinerUTimeout(RuntimeError):
    """MinerU 跑得太久超时。"""


def _mineru_cmd() -> str:
    v = (os.getenv("SCHOLAR_MINERU_CMD") or "").strip()
    return v or "mineru"


def _mineru_lang() -> str:
    v = (os.getenv("SCHOLAR_MINERU_LANG") or "").strip()
    return v or "ch"


def _mineru_backend() -> str:
    v = (os.getenv("SCHOLAR_MINERU_BACKEND") or "").strip().lower()
    return v or "pipeline"


def _mineru_timeout_sec() -> int:
    try:
        return max(60, int(os.getenv("SCHOLAR_MINERU_TIMEOUT") or "900"))
    except ValueError:
        return 900


def is_mineru_available() -> bool:
    """检查 MinerU CLI 能否调用（带 --help 探测一次）。"""
    cmd = _mineru_cmd()
    try:
        r = subprocess.run(
            [cmd, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
            check=False,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def parse_pdf_with_mineru(pdf_path: Path, lang: str | None = None) -> str:
    """
    调用 MinerU 解析单个 PDF，返回合并后的 markdown 文本（含 LaTeX）。

    输出结构（MinerU 默认）：
      <output>/<pdf_stem>/auto/<pdf_stem>.md       markdown（首选）
      <output>/<pdf_stem>/<pdf_stem>.md            部分老版本
      <output>/<pdf_stem>.md                       极简 fallback

    失败时抛 MinerUNotAvailable / MinerUTimeout / RuntimeError。
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise RuntimeError(f"PDF 不存在: {pdf_path}")

    cmd = _mineru_cmd()
    lang = (lang or _mineru_lang()).strip() or "ch"
    backend = _mineru_backend()
    timeout = _mineru_timeout_sec()

    # 用独立临时目录，避免把输出留在项目里
    with tempfile.TemporaryDirectory(prefix="mineru_") as tmp_dir:
        out_dir = Path(tmp_dir)
        argv = [
            cmd,
            "-p", str(pdf_path),
            "-o", str(out_dir),
            "-b", backend,
            "-l", lang,
        ]
        try:
            proc = subprocess.run(
                argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
        except FileNotFoundError as e:
            raise MinerUNotAvailable(
                f"找不到 MinerU 可执行文件（SCHOLAR_MINERU_CMD={cmd}）。"
                f"请先 pip install -U mineru，或设置 SCHOLAR_MINERU_CMD 指向它。原始错误：{e}"
            )
        except subprocess.TimeoutExpired:
            raise MinerUTimeout(
                f"MinerU 解析 {pdf_path.name} 超时（{timeout}s）。"
                "可能是文档很长或没用 GPU；可增大 SCHOLAR_MINERU_TIMEOUT。"
            )

        if proc.returncode != 0:
            stderr = (proc.stderr or b"").decode("utf-8", errors="ignore")[-1200:]
            raise RuntimeError(
                f"MinerU 调用失败 (exit={proc.returncode}):\n{stderr}"
            )

        md_text = _collect_markdown(out_dir, pdf_path.stem)
        if not md_text or len(md_text.strip()) < 20:
            stderr = (proc.stderr or b"").decode("utf-8", errors="ignore")[-600:]
            raise RuntimeError(
                f"MinerU 未产出 markdown 文件（可能是 GPU 内存不足或权重缺失）。stderr 尾部：\n{stderr}"
            )
        return _normalize_mineru_markdown(md_text)


def _collect_markdown(out_dir: Path, stem: str) -> str:
    """在 MinerU 输出目录里找到 markdown 并返回文本。"""
    candidates = [
        out_dir / stem / "auto" / f"{stem}.md",
        out_dir / stem / f"{stem}.md",
        out_dir / f"{stem}.md",
    ]
    for c in candidates:
        if c.exists():
            try:
                return c.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
    # 最后兜底：递归找第一个 .md
    for md in out_dir.rglob("*.md"):
        try:
            return md.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
    return ""


def _normalize_mineru_markdown(md: str) -> str:
    """
    轻微规整：
      - 删除 MinerU 的图片占位（![](images/xxx.jpg)）——我们用不上
      - 保留公式定界符 $...$ / $$...$$
      - 压缩过多空行
    """
    t = md or ""
    # 图片占位一律删掉
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)
    # 连续空行收敛为最多 2 个
    t = re.sub(r"\n{4,}", "\n\n\n", t)
    return t.strip()
