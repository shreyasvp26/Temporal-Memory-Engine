"""Reranking interface and baseline implementations per LLD Section 9.4."""

from collections.abc import Sequence
from typing import Protocol

from memory_engine.retrieve.fusion import RankedItem

__all__ = [
    'NoopReranker',
    'Reranker',
]


class Reranker(Protocol):
    """Protocol for neural or heuristic post-fusion rerankers [FROZEN]."""

    async def rerank(
        self,
        question: str,
        items: Sequence[RankedItem],
        top_n: int | None = None,
    ) -> list[RankedItem]:
        """Rerank items given the question, optionally returning the top N items."""
        ...


class NoopReranker:
    """Default pass-through reranker preserving fused and decayed order."""

    async def rerank(
        self,
        question: str,
        items: Sequence[RankedItem],
        top_n: int | None = None,
    ) -> list[RankedItem]:
        """Return the items unchanged, sliced to top_n if positive."""
        if top_n is not None and top_n >= 0:
            return list(items[:top_n])
        return list(items)
