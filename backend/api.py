from __future__ import annotations

from typing import Any
import json
import time
import hmac
import hashlib
import base64
import urllib.parse
from datetime import datetime
from wsgiref.handlers import format_date_time

# 引入 WebSocket 库
import websockets
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

from clustering import cluster_palette, fallback_pos
# 引入新配置
from config import FILES_DIR, SPARK_APP_ID, SPARK_API_SECRET, SPARK_API_KEY, SPARK_WS_URL, SPARK_DOMAIN
from text_processing import safe_stem


class AnalyzeBody(BaseModel):
    vectors: list[list[float]] | None = None

class ChatBody(BaseModel):
    question: str


# --- 辅助函数：生成 WebSocket 鉴权 URL ---
def get_auth_url(host_url, api_key, api_secret):
    ul = urllib.parse.urlparse(host_url)
    hostname = ul.hostname
    path = ul.path

    # 生成 RFC1123 格式的时间戳
    date = format_date_time(time.mktime(datetime.now().timetuple()))

    # 拼接签名字符串
    signature_origin = f"host: {hostname}\ndate: {date}\nGET {path} HTTP/1.1"
    
    # HMAC-SHA256 加密
    signature_sha = hmac.new(
        api_secret.encode('utf-8'),
        signature_origin.encode('utf-8'),
        digestmod=hashlib.sha256
    ).digest()
    
    signature_sha_base64 = base64.b64encode(signature_sha).decode(encoding='utf-8')
    
    authorization_origin = f'api_key="{api_key}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha_base64}"'
    authorization = base64.b64encode(authorization_origin.encode('utf-8')).decode(encoding='utf-8')
    
    # 组合最终 URL
    v = {
        "authorization": authorization,
        "date": date,
        "host": hostname
    }
    return host_url + '?' + urllib.parse.urlencode(v)


def create_app(store) -> FastAPI:
    app = FastAPI()

    @app.on_event("startup")
    def _startup_ingest() -> None:
        store.ingest_from_inbox()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], 
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def api_health() -> dict[str, Any]:
        return {"ok": True}

    @app.get("/api/papers")
    def api_papers() -> dict[str, Any]:
        with store._lock:
            if store._papers and any(("pos" not in p) or ("cluster" not in p) for p in store._papers):
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
                        "displayTitle": str(p.get("display_title") or safe_stem(str(p.get("filename") or ""))),
                        "firstSentence": str(p.get("first_sentence") or p.get("abstract", "")[:120] or "No content available."),
                        "abstract": str(p.get("abstract", "")),
                        "filename": str(p.get("filename", "")),
                        "field": str(p.get("field", "Uncategorized")),
                        "confidence": float(p.get("confidence", 0.0)),
                        "size": float(p.get("size", 3.0)),
                        "pos": (float(pos[0]), float(pos[1]), float(pos[2])),
                        "color": str(p.get("color", cluster_palette()[0])),
                        "keywords": p.get("keywords", []),
                        "cluster": int(p.get("cluster", 0)),
                    }
                )

            edges: list[dict[str, Any]] = []
            if store._vectors is not None and len(store._papers) >= 2:
                try:
                    sims = cosine_similarity(store._vectors)
                    np.fill_diagonal(sims, 0.0)
                    used: set[tuple[str, str]] = set()
                    topk = min(4, len(store._papers) - 1)
                    for i, pi in enumerate(store._papers):
                        src = str(pi.get("id") or "")
                        order = np.argsort(sims[i])[::-1][:topk]
                        for j in order:
                            w = float(sims[i, j])
                            if w < 0.20: continue
                            dst = str(store._papers[int(j)].get("id") or "")
                            a, b = (src, dst) if src <= dst else (dst, src)
                            if (a, b) in used: continue
                            used.add((a, b))
                            t = "intra" if int(pi.get("cluster", 0)) == int(store._papers[int(j)].get("cluster", 0)) else "bridge"
                            edges.append({"source": src, "target": dst, "weight": w, "type": t})
                except Exception as e:
                    print(f"Error computing edges: {e}")
                    edges = []

        return {"papers": papers, "edges": edges}

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
            raise HTTPException(status_code=400, detail="Only .pdf files are supported")
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty file")
        try:
            pdf_id = store.add_pdf(file.filename, raw)
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except Exception as e:
            print(f"Upload failed: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
        return {"success": True, "pdf_id": pdf_id}

    @app.post("/api/papers/upload")
    async def api_papers_upload(file: UploadFile = File(...)) -> dict[str, Any]:
        return await api_upload(file)

    @app.post("/api/analyze")
    def api_analyze(body: AnalyzeBody) -> dict[str, Any]:
        return store.analyze()

    # --- 核心修改：使用 WebSocket 连接星火 X1.5 ---
    @app.post("/api/query")
    async def api_query(body: ChatBody) -> dict[str, Any]:
        question = body.question.strip()
        if not question:
            return {"answer": "请输入有效的问题。", "cites": []}

        # 1. 生成鉴权 URL
        auth_url = get_auth_url(SPARK_WS_URL, SPARK_API_KEY, SPARK_API_SECRET)

        # 2. 构造请求参数 (符合 X1.5 文档)
        data = {
            "header": {
                "app_id": SPARK_APP_ID,
                "uid": "user_default"
            },
            "parameter": {
                "chat": {
                    "domain": SPARK_DOMAIN, # spark-x
                    "temperature": 0.5,
                    "max_tokens": 4096,
                    "thinking": { "type": "enabled" } # 开启深度思考
                }
            },
            "payload": {
                "message": {
                    "text": [
                        {"role": "user", "content": question}
                    ]
                }
            }
        }

        answer_content = ""
        # reasoning_content = "" # 如果你想展示思考过程，可以收集这个变量

        try:
            async with websockets.connect(auth_url) as ws:
                # 发送请求
                await ws.send(json.dumps(data))

                # 循环接收流式响应
                async for message in ws:
                    response = json.loads(message)
                    header = response.get('header', {})
                    code = header.get('code')

                    if code != 0:
                        err_msg = header.get('message') or "Unknown Error"
                        print(f"Spark Error Code {code}: {err_msg}")
                        return {"answer": f"AI Error: {err_msg} (Code {code})", "cites": []}

                    payload = response.get('payload', {})
                    choices = payload.get('choices', {})
                    text_list = choices.get('text', [])

                    for text in text_list:
                        # 收集最终结果
                        if 'content' in text:
                            answer_content += text['content']
                        # 收集思考过程 (暂不返回给前端，因为前端暂无展示位)
                        # if 'reasoning_content' in text:
                        #     reasoning_content += text['reasoning_content']

                    status_code = header.get('status')
                    if status_code == 2:
                        # 会话结束
                        break
            
            return {"answer": answer_content, "cites": []}

        except Exception as e:
            print(f"WebSocket Exception: {e}")
            return {"answer": "连接 AI 服务超时或失败，请检查网络。", "cites": []}

    @app.get("/files/{paper_id}.pdf")
    def api_files(paper_id: str) -> FileResponse:
        pdf_path = FILES_DIR / f"{paper_id}.pdf"
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail="PDF not found")
        return FileResponse(
            str(pdf_path),
            media_type="application/pdf",
            headers={"Content-Disposition": f"inline; filename=\"{paper_id}.pdf\""},
        )

    @app.get("/api/visualization")
    def api_visualization() -> dict[str, Any]:
        return store.visualization()

    return app