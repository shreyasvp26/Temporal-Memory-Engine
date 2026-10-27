"""Reciprocal Rank Fusion (RRF) and hit ranking per LLD Section 9.4 [FROZEN]."""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID

from memory_engine.domain.errors import InvariantViolation
from memory_engine.domain.models import Fact, Source

__all__ = [
    'DEFAULT_LANE_WEIGHTS',
    'Hit',
    'RankedItem',
    'RrfFuser',
]

DEFAULT_LANE_WEIGHTS: dict[str, float] = {
    'fact_vector': 1.0,
    'vector_facts': 1.0,
    'fact_keyword': 0.8,
    'fts_facts': 0.8,
    'structured': 1.5,
    'temporal_facts': 1.5,
    'source_vector': 0.6,
    'vector_sources': 0.6,
    'source_keyword': 0.5,
    'fts_sources': 0.5,
    'graph_expansion': 0.7,
    'graph_expand': 0.7,
}


@dataclass(frozen=True, slots=True)
class Hit:
    """Raw retrieval hit from a single search lane [FROZEN]."""

    kind: Literal['fact', 'source']
    id: UUID
    lane: str
    rank: int
    raw_score: float
    fact: Fact | None = None
    source: Source | None = None

    def __post_init__(self) -> None:
        if self.kind not in ('fact', 'source'):
            raise InvariantViolation(f"kind must be 'fact' or 'source', got {self.kind!r}")
        if self.kind == 'fact' and self.fact is None:
            raise InvariantViolation('fact must be provided when kind is fact')
        if self.kind == 'source' and self.source is None:
            raise InvariantViolation('source must be provided when kind is source')


@dataclass(frozen=True, slots=True)
class RankedItem:
    """Unified retrieval item after multi-lane reciprocal rank fusion [FROZEN]."""

    kind: Literal['fact', 'source']
    id: UUID
    score: float
    fact: Fact | None = None
    source: Source | None = None
    lanes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in ('fact', 'source'):
            raise InvariantViolation(f"kind must be 'fact' or 'source', got {self.kind!r}")
        if self.kind == 'fact' and self.fact is None:
            raise InvariantViolation('fact must be provided when kind is fact')
        if self.kind == 'source' and self.source is None:
            raise InvariantViolation('source must be provided when kind is source')

    @property
    def timestamp(self) -> datetime:
        """Timestamp used for temporal tie-breaking (newer valid_from or source_ts)."""
        if self.fact is not None:
            return self.fact.valid_from
        if self.source is not None:
            return self.source.source_ts
        raise InvariantViolation('RankedItem has neither fact nor source')


def _sort_ranked_items(items: list[RankedItem]) -> list[RankedItem]:
    """Sort ranked items deterministically: score desc, newer timestamp desc, string id asc."""
    items.sort(key=lambda item: str(item.id))
    items.sort(key=lambda item: item.timestamp, reverse=True)
    items.sort(key=lambda item: item.score, reverse=True)
    return items


class RrfFuser:
    """Reciprocal Rank Fusion coordinator with deterministic tie-breaking [FROZEN].

    For each lane list and each item at 1-based rank r:
      score[item] += weight / (rrf_k + r)
    Ties break by newer valid_from (facts) or source_ts (sources), then by id.
    """

    def __init__(
        self,
        rrf_k: int = 60,
        lane_weights: Mapping[str, float] | None = None,
    ) -> None:
        self.rrf_k = rrf_k
        self.lane_weights = (
            dict(lane_weights) if lane_weights is not None else dict(DEFAULT_LANE_WEIGHTS)
        )

    def fuse(
        self,
        lane_hits: (
            Sequence[Sequence[Hit]]
            | Mapping[str, Sequence[Hit]]
            | Sequence[tuple[Sequence[Hit], float]]
        ),
        weights: Mapping[str, float] | None = None,
    ) -> list[RankedItem]:
        """Fuse results from multiple retrieval lanes using Reciprocal Rank Fusion."""
        effective_weights = dict(self.lane_weights)
        if weights is not None:
            effective_weights.update(weights)

        # Standardize lane lists into sequence of (hits, weight)
        lane_pairs: list[tuple[Sequence[Hit], float]] = []
        if isinstance(lane_hits, Mapping):
            for lane_name, hits in lane_hits.items():
                w = effective_weights.get(lane_name, 1.0)
                lane_pairs.append((hits, w))
        else:
            for item in lane_hits:
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], (int, float)):
                    raw_hits, raw_w = item
                    lane_pairs.append((raw_hits, float(raw_w)))
                else:
                    seq_item: Sequence[Hit] = item
                    if seq_item:
                        first_lane = seq_item[0].lane
                        w = effective_weights.get(first_lane, 1.0)
                    else:
                        w = 1.0
                    lane_pairs.append((seq_item, w))

        scores: dict[tuple[Literal['fact', 'source'], UUID], float] = defaultdict(float)
        facts: dict[tuple[Literal['fact', 'source'], UUID], Fact] = {}
        sources: dict[tuple[Literal['fact', 'source'], UUID], Source] = {}
        lanes_by_key: dict[tuple[Literal['fact', 'source'], UUID], list[str]] = defaultdict(list)

        for hits, weight in lane_pairs:
            for idx, hit in enumerate(hits, start=1):
                r = hit.rank if hit.rank > 0 else idx
                key: tuple[Literal['fact', 'source'], UUID] = (hit.kind, hit.id)
                scores[key] += weight / (self.rrf_k + r)
                lanes_by_key[key].append(hit.lane)
                if hit.fact is not None:
                    facts[key] = hit.fact
                if hit.source is not None:
                    sources[key] = hit.source

        result: list[RankedItem] = []
        for (kind, item_id), total_score in scores.items():
            key_tuple: tuple[Literal['fact', 'source'], UUID] = (kind, item_id)
            unique_lanes = tuple(dict.fromkeys(lanes_by_key[key_tuple]))
            result.append(
                RankedItem(
                    kind=kind,
                    id=item_id,
                    score=total_score,
                    fact=facts.get(key_tuple),
                    source=sources.get(key_tuple),
                    lanes=unique_lanes,
                )
            )

        return _sort_ranked_items(result)
