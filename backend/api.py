from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

from clustering import cluster_palette, fallback_pos
from config import FILES_DIR
from text_processing import safe_stem


class AnalyzeBody(BaseModel):
    vectors: list[list[float]] | None = None


def create_app(store) -> FastAPI:
    app = FastAPI()

    @app.on_event("startup")
    def _startup_ingest() -> None:
        store.ingest_from_inbox()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "null"],
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
                        "title": p.get("title", ""),
                        "displayTitle": p.get("display_title") or safe_stem(str(p.get("filename") or "")),
                        "pos": (float(pos[0]), float(pos[1]), float(pos[2])),
                        "color": p.get("color", cluster_palette()[0]),
                        "category": p.get("category") or p.get("field") or "User Upload",
                        "keywords": p.get("keywords", []),
                        "cluster": int(p.get("cluster", 0)),
                    }
                )

            edges: list[dict[str, Any]] = []
            if store._vectors is not None and len(store._papers) >= 2:
                sims = cosine_similarity(store._vectors)
                np.fill_diagonal(sims, 0.0)
                used: set[tuple[str, str]] = set()
                topk = min(4, len(store._papers) - 1)
                for i, pi in enumerate(store._papers):
                    src = str(pi.get("id") or "")
                    order = np.argsort(sims[i])[::-1][:topk]
                    for j in order:
                        w = float(sims[i, j])
                        if w < 0.28:
                            continue
                        dst = str(store._papers[int(j)].get("id") or "")
                        a, b = (src, dst) if src <= dst else (dst, src)
                        if (a, b) in used:
                            continue
                        used.add((a, b))
                        t = (
                            "intra"
                            if int(pi.get("cluster", 0)) == int(store._papers[int(j)].get("cluster", 0))
                            else "bridge"
                        )
                        edges.append({"source": src, "target": dst, "weight": w, "type": t})

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
            raise HTTPException(status_code=400, detail="Only .pdf is supported")
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Empty file")
        try:
            pdf_id = store.add_pdf(file.filename, raw)
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
        return {"success": True, "pdf_id": pdf_id}

    @app.post("/api/papers/upload")
    async def api_papers_upload(file: UploadFile = File(...)) -> dict[str, Any]:
        return await api_upload(file)

    @app.post("/api/analyze")
    def api_analyze(body: AnalyzeBody) -> dict[str, Any]:
        return store.analyze()

    @app.post("/api/query")
    def api_query(body: dict[str, Any]) -> dict[str, Any]:
        return {"answer": "当前 MVP 后端未实现 /api/query（本地检索/对话）。", "cites": []}

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
