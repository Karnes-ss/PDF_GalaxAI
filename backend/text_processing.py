from __future__ import annotations

import io
import os
import re
import unicodedata
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None


# ---------------------------------------------------------------- #
# 文件名辅助
# ---------------------------------------------------------------- #

def safe_stem(filename: str) -> str:
    name = filename.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    if name.lower().endswith(".pdf"):
        name = name[:-4]
    return name.strip() or "Untitled"


# ---------------------------------------------------------------- #
# OCR 配置与自检
# ---------------------------------------------------------------- #

_VALID_OCR_MODES = {"auto", "force", "off"}


def _load_env_file_var(key: str) -> str:
    """
    兜底读取 backend/.env。
    说明：OCR 代码运行在 store 启动阶段，此时不一定已把 .env 注入进程环境变量。
    """
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return ""
    try:
        with env_path.open("r", encoding="utf-8") as f:
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


def _env_or_file(key: str, default: str = "") -> str:
    return (os.getenv(key) or _load_env_file_var(key) or default).strip()


def _ocr_mode() -> str:
    """OCR 模式：auto | force | off。缺省 auto。"""
    m = _env_or_file("SCHOLAR_OCR_ENABLED", "auto").lower()
    return m if m in _VALID_OCR_MODES else "auto"


def _ocr_lang() -> str:
    """OCR 语种，遵循 Tesseract 语言代码，多语种用 + 连接。"""
    return _env_or_file("SCHOLAR_OCR_LANG", "chi_sim+eng")


def _ocr_zoom() -> float:
    try:
        v = float(_env_or_file("SCHOLAR_OCR_ZOOM", "2.0"))
    except ValueError:
        v = 2.0
    return max(1.0, min(4.0, v))


def _ocr_min_quality() -> float:
    """auto 模式下，native 文本低于该质量才触发 OCR。"""
    try:
        v = float(_env_or_file("SCHOLAR_OCR_MIN_QUALITY", "0.55"))
    except ValueError:
        v = 0.55
    return max(0.0, min(1.0, v))


_tesseract_ready_cache: bool | None = None


def _tesseract_ready() -> bool:
    """
    检查本机是否安装可执行的 Tesseract。
    第一次调用会真正探测，并把结果缓存，避免每页都探测。
    """
    global _tesseract_ready_cache
    if _tesseract_ready_cache is not None:
        return _tesseract_ready_cache
    if pytesseract is None or Image is None:
        _tesseract_ready_cache = False
        return False
    # 允许通过环境变量显式指定 Tesseract 路径
    custom = _env_or_file("SCHOLAR_TESSERACT_CMD", "")
    if custom:
        pytesseract.pytesseract.tesseract_cmd = custom
    try:
        pytesseract.get_tesseract_version()
        _tesseract_ready_cache = True
    except Exception as e:
        print(f"[ocr] Tesseract unavailable: {e}")
        _tesseract_ready_cache = False
    return _tesseract_ready_cache


# ---------------------------------------------------------------- #
# 文本质量评分（与 store._text_quality_score 等价的独立实现）
# ---------------------------------------------------------------- #

def _page_quality(text: str) -> float:
    s = text or ""
    if not s.strip():
        return 0.0
    total = max(1, len(s))
    bad = s.count("?") * 0.8 + s.count("�") * 3 + s.count("a?��") * 2
    readable = len(re.findall(r"[A-Za-z0-9\u4e00-\u9fff]", s))
    bad_ratio = min(1.0, bad / total)
    readable_ratio = readable / total
    return max(0.0, min(1.0, 0.65 * (1 - bad_ratio) + 0.35 * readable_ratio))


# ---------------------------------------------------------------- #
# PDF 文本抽取（native + OCR 混合）
# ---------------------------------------------------------------- #

def _render_page_to_image(page, zoom: float):
    """把 fitz.Page 渲染成 PIL Image。"""
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.open(io.BytesIO(pix.tobytes("png")))


def _ocr_page(page, lang: str, zoom: float) -> str:
    if not _tesseract_ready():
        return ""
    try:
        img = _render_page_to_image(page, zoom)
        txt = pytesseract.image_to_string(img, lang=lang)
        return txt or ""
    except Exception as e:
        print(f"[ocr] page OCR failed: {e}")
        return ""


def _read_with_plumber(pdf_path: Path, max_pages: int) -> str:
    """pdfplumber 兜底（无 fitz 时）。"""
    parts: list[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        pages = pdf.pages[:max_pages] if max_pages else pdf.pages
        for page in pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


def read_pdf_pages(pdf_path: Path) -> list[str]:
    """按页返回 PDF 文本，列表长度 = 页数；第 i 项对应第 i+1 页。

    仅用 native 文本层，保持快速。页面为空时返回空串。
    """
    pages: list[str] = []
    if fitz is not None:
        try:
            doc = fitz.open(str(pdf_path))
            try:
                for i in range(doc.page_count):
                    try:
                        pages.append(doc.load_page(i).get_text("text") or "")
                    except Exception:
                        pages.append("")
            finally:
                doc.close()
            return pages
        except Exception as e:
            print(f"[pdf] fitz pages read failed ({e}); fallback to pdfplumber")

    if pdfplumber is None:
        return pages

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for p in pdf.pages:
                try:
                    pages.append(p.extract_text() or "")
                except Exception:
                    pages.append("")
    except Exception as e:
        print(f"[pdf] pdfplumber pages read failed: {e}")
    return pages


def read_pdf_text(
    pdf_path: Path,
    max_pages: int = 0,
    mode_override: str | None = None,
    parser: str | None = None,
) -> str:
    """
    抽取 PDF 文本。支持三种 OCR 模式（通过 SCHOLAR_OCR_ENABLED 配置）：

      - off   : 只用 native 文本层（最快）
      - auto  : native 优先；若某页文本质量不足再对该页 OCR（推荐）
      - force : 对每一页都跑 OCR（最稳，但很慢）

    ``mode_override`` 允许在单次调用里临时改写 OCR 模式。

    ``parser`` 允许选择底层解析器：
      - None / "default" / "fitz" : 默认路径（fitz + 可选 OCR）
      - "mineru"                   : 使用 MinerU 做版面/公式/表格识别，产出 markdown + LaTeX

    MinerU 分支需要单独装 mineru CLI，详见 mineru_parser.py 与 README。

    环境变量：
      SCHOLAR_OCR_ENABLED       auto|force|off
      SCHOLAR_OCR_LANG          默认 chi_sim+eng
      SCHOLAR_OCR_ZOOM          默认 2.0
      SCHOLAR_OCR_MIN_QUALITY   默认 0.55
      SCHOLAR_TESSERACT_CMD     自定义 tesseract.exe 路径（可选）
      SCHOLAR_MINERU_CMD        mineru 可执行文件路径（parser=mineru 时用到）
    """
    parser_kind = (parser or "").strip().lower()
    if parser_kind in {"mineru"}:
        from mineru_parser import parse_pdf_with_mineru  # 延迟 import
        return parse_pdf_with_mineru(pdf_path)

    mode = (mode_override or "").lower().strip()
    if mode not in _VALID_OCR_MODES:
        mode = _ocr_mode()
    lang = _ocr_lang()
    zoom = _ocr_zoom()
    min_q = _ocr_min_quality()

    # 无 fitz 时退回 pdfplumber，仅支持 native
    if fitz is None:
        if pdfplumber is None:
            raise RuntimeError("No PDF parser available")
        if mode != "off":
            print("[ocr] fitz not available; OCR disabled")
        return _read_with_plumber(pdf_path, max_pages)

    use_ocr = mode in {"auto", "force"}
    if use_ocr and not _tesseract_ready():
        print(
            "[ocr] Tesseract 未安装或不可用，已自动降级为 native 文本提取。"
            "如需 OCR 请参考 README 安装 Tesseract。"
        )
        use_ocr = False

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        if pdfplumber is None:
            raise RuntimeError(f"PDF 打开失败: {e}")
        print(f"[pdf] fitz open failed ({e}); fallback to pdfplumber")
        return _read_with_plumber(pdf_path, max_pages)

    parts: list[str] = []
    try:
        total = doc.page_count
        limit = total if not max_pages else min(total, max_pages)
        for i in range(limit):
            page = doc.load_page(i)
            native_text = page.get_text("text") or ""

            if mode == "off" or not use_ocr:
                parts.append(native_text)
                continue

            if mode == "force":
                ocr_text = _ocr_page(page, lang, zoom)
                # OCR 为空或质量更差时退回 native
                if ocr_text.strip() and _page_quality(ocr_text) >= _page_quality(native_text):
                    parts.append(ocr_text)
                else:
                    parts.append(native_text or ocr_text)
                continue

            # auto
            if native_text.strip() and _page_quality(native_text) >= min_q:
                parts.append(native_text)
            else:
                ocr_text = _ocr_page(page, lang, zoom)
                parts.append(ocr_text or native_text)
    finally:
        doc.close()

    return "\n".join(parts)


# ---------------------------------------------------------------- #
# 文本清洗
# ---------------------------------------------------------------- #

_PUNCT_ONLY_LINE = re.compile(r"^[\s\W_]+$", re.UNICODE)
_STRAY_TRAIL_BANG = re.compile(r"(?<=[\u4e00-\u9fff。，；：？！、）】」』])!+\s*$", re.MULTILINE)


def clean_text(text: str) -> str:
    t = text or ""
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("\x00", " ")
    t = re.sub(r"[\x01-\x08\x0B\x0C\x0E-\x1F]", " ", t)
    # PDF 换行导致的英文单词粘连
    t = re.sub(r"-\n([A-Za-z])", r"\1", t)
    t = re.sub(r"(?<=[A-Za-z])\n(?=[A-Za-z])", " ", t)
    # 只清理"离谱的连续下划线"（PDF 分隔线），保留单个下划线以免毁掉 LaTeX 下标（x_i、\theta_{ij}）
    t = re.sub(r"_{3,}", " ", t)
    t = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", t)
    t = re.sub(r"(?<=[A-Za-z])(?=[0-9])", " ", t)
    t = re.sub(r"(?<=[0-9])(?=[A-Za-z])", " ", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[^\S\r\n]{2,}", " ", t)
    t = re.sub(r"\?{3,}", "??", t)

    # Markdown / Typora 导出常见脏字符（行尾孤立的 ! ¶ · • ¤）
    t = _STRAY_TRAIL_BANG.sub("", t)
    # 删掉"只有标点 / 只有 1 个字符"这种没有信息量的行
    cleaned_lines: list[str] = []
    for line in t.splitlines():
        s = line.strip()
        if not s:
            cleaned_lines.append("")
            continue
        # 纯标点 / 符号行
        if _PUNCT_ONLY_LINE.match(s) and len(s) <= 4:
            continue
        # 单字符行（公式逐行破碎、散落标记）
        if len(s) == 1 and not s.isalnum():
            continue
        cleaned_lines.append(s)

    # 合并多余空行
    collapsed: list[str] = []
    blank = 0
    for s in cleaned_lines:
        if s:
            collapsed.append(s)
            blank = 0
        else:
            blank += 1
            if blank <= 1:
                collapsed.append("")
    t = "\n".join(collapsed)
    return t.strip()


# ---------------------------------------------------------------- #
# 标题 / 摘要 / 关键词抽取
# ---------------------------------------------------------------- #

def extract_title_from_text(text: str, fallback: str) -> str:
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    candidates = [l for l in lines[:20] if 8 <= len(l) <= 200]
    if candidates:
        return max(candidates, key=len)
    return fallback


_FORMULA_CHARS = set("∑∏∫∞≈≠≤≥→←↔⇒⇔∇∂∈∉∀∃∧∨¬±×÷·∘⊕⊗⊙σΣΠΩμαβγδεθλπφψω")


def _looks_like_title_or_meta(line: str) -> bool:
    """标题 / 作者 / 机构 / 邮件 / URL / 页眉 这类 meta 行。"""
    s = line.strip()
    if not s:
        return True
    if len(s) <= 4:
        return True
    if re.search(r"@[\w.-]+\.\w+", s):       # email
        return True
    if re.search(r"https?://|www\.", s):     # url
        return True
    if re.match(r"^(page\s+)?\d+\s*(/\s*\d+)?$", s, re.IGNORECASE):
        return True
    return False


def _is_formula_heavy(line: str) -> bool:
    """公式行：非文字字符占比过高。"""
    s = line.strip()
    if not s:
        return True
    total = len(s)
    syms = sum(1 for ch in s if ch in _FORMULA_CHARS)
    digits_or_ops = sum(1 for ch in s if ch in "0123456789+-*/=<>()[]{}^_|\\")
    readable = len(re.findall(r"[A-Za-z\u4e00-\u9fff]", s))
    if readable == 0 and total > 0:
        return True
    # 符号/数字/操作符比例过高
    return (syms + digits_or_ops) / max(1, total) > 0.45


def _first_real_paragraph(text: str, limit: int) -> str:
    """
    从清洗文本中挑出第一段"真正的自然语言"：
    - 跳过 meta / 公式 / 极短行
    - 连续正文行聚合成一段，长度 >= 60 时采用
    """
    buffer: list[str] = []
    chosen: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            if buffer:
                para = " ".join(buffer).strip()
                if len(para) >= 60 and re.search(r"[.。!?！？]", para):
                    chosen.append(para)
                    if sum(len(c) for c in chosen) >= limit:
                        break
                buffer = []
            continue
        if _looks_like_title_or_meta(s) or _is_formula_heavy(s):
            continue
        buffer.append(s)
    if buffer and not chosen:
        para = " ".join(buffer).strip()
        if len(para) >= 40:
            chosen.append(para)
    if not chosen:
        return ""
    return " ".join(chosen)[:limit].strip()


def extract_abstract_block(text: str, limit: int = 1200) -> str:
    t = text or ""
    m = re.search(r"(abstract|摘要)[:：]?\s*", t, re.IGNORECASE)
    if m:
        tail = t[m.end():]
        stop = re.search(
            r"\n\s*(introduction|1\s+introduction|关键词|keywords)[:：]?\s*",
            tail,
            re.IGNORECASE,
        )
        if stop:
            tail = tail[: stop.start()]
        return tail.strip()[:limit]

    # 没有显式 abstract / 摘要段：挑第一段真正的段落
    picked = _first_real_paragraph(t, limit)
    if picked:
        return picked
    # 兜底：原文前 limit 字，但已被 clean_text 压缩过
    return t[:limit].strip()


def extract_keywords_block(text: str, top_k: int = 8) -> list[str]:
    """优先抽取论文自带 Keywords/关键词 段落。抓不到则返回空。"""
    t = text or ""
    m = re.search(r"(keywords|关键词)[:：]\s*(.+)", t, re.IGNORECASE)
    if m:
        raw = m.group(2)
        parts = re.split(r"[;,，、\n]", raw)
        kws = [p.strip() for p in parts if p.strip()]
        return kws[:top_k]
    return []


def tfidf_keywords_block(text: str, top_k: int = 8) -> list[str]:
    """
    单文档 TF-IDF 回退（仅用于全库语料不足时的保守兜底）。
    推荐使用 keyword_extractor.GlobalKeywordExtractor 做全库 TF-IDF + MMR。
    """
    if not text or not text.strip():
        return []
    try:
        vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
        X = vectorizer.fit_transform([text])
    except ValueError:
        return []
    row = X.toarray().ravel()
    feats = vectorizer.get_feature_names_out()
    idx = np.argsort(row)[::-1][:top_k]
    return [str(feats[i]) for i in idx if row[i] > 0]
