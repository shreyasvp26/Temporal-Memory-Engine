"""Temporal decay scoring per LLD Section 9.4 [FROZEN]."""

from collections.abc import Mapping, Sequence
from typing import Any

from memory_engine.domain.enums import DecayClass, Intent
from memory_engine.ports.repositories import PredicateRegistry
from memory_engine.retrieve.fusion import RankedItem, _sort_ranked_items
from memory_engine.retrieve.planner import QueryPlan

__all__ = [
    'DecayScorer',
]


class DecayScorer:
    """Multiplicative half-life decay scorer anchored at plan.question_ts [FROZEN].

    Applies only if:
      plan.intent in (Intent.OPEN, Intent.CURRENT_STATE)
      and plan.time_range is None
      and plan.as_of is None

    Formula for fact items:
      factor = floor + (1 - floor) * 0.5 ** (age_days / half_life[decay_class])
      age_days = max(0, (question_ts - max(valid_from, last_accessed_at or valid_from)) in days)
      floor = 0.3, half_life: slow 365, normal 90, fast 14 days, permanent: factor 1.0
      final_score = fused_score * factor

    Source items are never decayed.
    """

    def __init__(
        self,
        cfg: Any = None,
        registry: PredicateRegistry | None = None,
        *,
        floor: float | None = None,
        halflife_slow: int | None = None,
        halflife_normal: int | None = None,
        halflife_fast: int | None = None,
        decay_enabled: bool | None = None,
    ) -> None:
        self.registry = registry

        if floor is not None:
            self.floor = floor
        elif cfg is not None and hasattr(cfg, 'decay_floor'):
            self.floor = float(cfg.decay_floor)
        else:
            self.floor = 0.3

        if halflife_slow is not None:
            self.halflife_slow = halflife_slow
        elif cfg is not None and hasattr(cfg, 'halflife_slow'):
            self.halflife_slow = int(cfg.halflife_slow)
        else:
            self.halflife_slow = 365

        if halflife_normal is not None:
            self.halflife_normal = halflife_normal
        elif cfg is not None and hasattr(cfg, 'halflife_normal'):
            self.halflife_normal = int(cfg.halflife_normal)
        else:
            self.halflife_normal = 90

        if halflife_fast is not None:
            self.halflife_fast = halflife_fast
        elif cfg is not None and hasattr(cfg, 'halflife_fast'):
            self.halflife_fast = int(cfg.halflife_fast)
        else:
            self.halflife_fast = 14

        if decay_enabled is not None:
            self.decay_enabled = decay_enabled
        elif cfg is not None and hasattr(cfg, 'decay_enabled'):
            self.decay_enabled = bool(cfg.decay_enabled)
        else:
            self.decay_enabled = True

        self.half_lives: Mapping[DecayClass, int] = {
            DecayClass.SLOW: self.halflife_slow,
            DecayClass.NORMAL: self.halflife_normal,
            DecayClass.FAST: self.halflife_fast,
        }

    def score(self, plan: QueryPlan, items: Sequence[RankedItem]) -> list[RankedItem]:
        """Apply temporal decay to ranked items if the query plan allows it."""
        if not self.decay_enabled:
            return list(items)

        # Bypassed for non-open / non-current-state intents or time-scoped queries (T-DECAY)
        if plan.intent not in (Intent.OPEN, Intent.CURRENT_STATE):
            return list(items)
        if plan.time_range is not None or plan.as_of is not None:
            return list(items)

        decayed: list[RankedItem] = []
        for item in items:
            if item.kind != 'fact' or item.fact is None:
                # Source items are not decayed
                decayed.append(item)
                continue

            fact = item.fact
            decay_class = DecayClass.NORMAL
            if self.registry is not None:
                try:
                    pred = self.registry.get(fact.predicate)
                    decay_class = pred.decay_class
                except Exception:
                    decay_class = DecayClass.NORMAL

            if decay_class == DecayClass.PERMANENT:
                factor = 1.0
            else:
                hl = self.half_lives.get(decay_class, self.halflife_normal)
                last_anchor = max(fact.valid_from, fact.last_accessed_at or fact.valid_from)
                diff_seconds = (plan.question_ts - last_anchor).total_seconds()
                age_days = max(0.0, diff_seconds / 86400.0)
                factor = self.floor + (1.0 - self.floor) * (0.5 ** (age_days / hl))

            final_score = item.score * factor
            decayed.append(
                RankedItem(
                    kind=item.kind,
                    id=item.id,
                    score=final_score,
                    fact=item.fact,
                    source=item.source,
                    lanes=item.lanes,
                )
            )

        return _sort_ranked_items(decayed)

    def apply(self, plan: QueryPlan, items: Sequence[RankedItem]) -> list[RankedItem]:
        """Alias for score."""
        return self.score(plan, items)

    def __call__(self, plan: QueryPlan, items: Sequence[RankedItem]) -> list[RankedItem]:
        """Allow calling instance directly as a function."""
        return self.score(plan, items)
