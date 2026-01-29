from __future__ import annotations

import hashlib

import numpy as np
from sklearn.decomposition import PCA

try:
    import umap
except ImportError:
    umap = None


def cluster_palette() -> list[str]:
    return [
        "#60a5fa",
        "#f59e0b",
        "#10b981",
        "#a78bfa",
        "#f472b6",
        "#22c55e",
        "#38bdf8",
        "#fb7185",
        "#eab308",
        "#14b8a6",
    ]


def fallback_pos(paper_id: str) -> list[float]:
    h = hashlib.sha1((paper_id or "").encode("utf-8")).digest()
    a = int.from_bytes(h[0:4], "little", signed=False) / 2**32
    b = int.from_bytes(h[4:8], "little", signed=False) / 2**32
    c = int.from_bytes(h[8:12], "little", signed=False) / 2**32
    x = (a - 0.5) * 14.0
    y = (b - 0.5) * 14.0
    z = (c - 0.5) * 14.0
    return [float(x), float(y), float(z)]


def reduce_to_3d(vectors: np.ndarray) -> np.ndarray:
    n = vectors.shape[0]
    if n == 1:
        return np.zeros((1, 3), dtype=np.float32)

    coords: np.ndarray | None = None
    if umap is not None and n >= 5:
        try:
            reducer = umap.UMAP(
                n_components=3,
                n_neighbors=min(10, n - 1),
                min_dist=0.12,
                init="random" if n < 10 else "spectral",
                random_state=42,
            )
            coords = reducer.fit_transform(vectors)
        except Exception:
            coords = None

    if coords is None:
        pca = PCA(n_components=3, random_state=42)
        coords = pca.fit_transform(vectors)

    coords = coords - coords.mean(axis=0, keepdims=True)
    max_abs = float(np.max(np.abs(coords))) if coords.size else 1.0
    if max_abs < 1e-6:
        max_abs = 1.0
    return (coords / max_abs * 5.5).astype(np.float32)
