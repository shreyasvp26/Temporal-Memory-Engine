"""Cached embedder wrapping an Embedder port with SHA-256 in-memory/SQLite cache."""

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from memory_engine.ports.embed import Embedder

__all__ = ['CachedEmbedder']


class CachedEmbedder(Embedder):
    """Embedder wrapper caching embeddings by SHA-256(model_id:kind:text)."""

    def __init__(
        self,
        inner: Embedder,
        cache_db_path: Path | str | None = None,
    ) -> None:
        self.model_id = inner.model_id
        self.dim = inner.dim
        self._inner = inner
        self._cache_db_path = Path(cache_db_path) if cache_db_path else None
        self._mem_cache: dict[str, list[float]] = {}
        self.hits: int = 0
        self.misses: int = 0

        if self._cache_db_path is not None:
            self._init_sqlite()

    def _init_sqlite(self) -> None:
        if self._cache_db_path is None:
            return
        self._cache_db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._cache_db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    cache_key TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    vector TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def _make_key(self, kind: str, text: str) -> str:
        content = f'{self.model_id}:{kind}:{text}'.encode()
        return hashlib.sha256(content).hexdigest()

    def _lookup(self, key: str) -> list[float] | None:
        if key in self._mem_cache:
            return self._mem_cache[key]

        if self._cache_db_path is not None:
            with sqlite3.connect(self._cache_db_path) as conn:
                cur = conn.execute('SELECT vector FROM embedding_cache WHERE cache_key = ?', (key,))
                row = cur.fetchone()
                if row is not None:
                    raw_vec = json.loads(row[0])
                    vec: list[float] = [float(x) for x in raw_vec]
                    self._mem_cache[key] = vec
                    return vec

        return None

    def _store(self, key: str, kind: str, vector: list[float]) -> None:
        float_vec = [float(x) for x in vector]
        self._mem_cache[key] = float_vec
        if self._cache_db_path is not None:
            now_iso = datetime.now(UTC).isoformat()
            vec_json = json.dumps(float_vec)
            with sqlite3.connect(self._cache_db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO embedding_cache (cache_key, model_id, kind, vector, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO NOTHING
                    """,
                    (key, self.model_id, kind, vec_json, now_iso),
                )
                conn.commit()

    async def _embed_with_cache(self, kind: str, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []
        keys: list[str] = []

        for i, t in enumerate(texts):
            key = self._make_key(kind, t)
            keys.append(key)
            cached_vec = self._lookup(key)
            if cached_vec is not None:
                results[i] = cached_vec
                self.hits += 1
            else:
                self.misses += 1
                missing_indices.append(i)
                missing_texts.append(t)

        if missing_texts:
            if kind == 'document':
                new_vecs = await self._inner.embed_documents(missing_texts)
            else:
                new_vecs = await self._inner.embed_queries(missing_texts)

            for idx, _text, vec in zip(missing_indices, missing_texts, new_vecs, strict=True):
                key = keys[idx]
                self._store(key, kind, vec)
                results[idx] = vec

        # All results are populated
        return [r for r in results if r is not None]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed documents with caching under kind='document'."""
        return await self._embed_with_cache('document', texts)

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed queries with caching under kind='query'."""
        return await self._embed_with_cache('query', texts)

    def clear(self) -> None:
        """Clear in-memory and SQLite cache."""
        self._mem_cache.clear()
        self.hits = 0
        self.misses = 0
        if self._cache_db_path is not None and self._cache_db_path.exists():
            with sqlite3.connect(self._cache_db_path) as conn:
                conn.execute('DELETE FROM embedding_cache')
                conn.commit()
