"""Embedder port."""

from collections.abc import Sequence
from typing import Protocol


class Embedder(Protocol):
    model_id: str
    dim: int

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...
