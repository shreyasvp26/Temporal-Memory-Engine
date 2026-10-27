"""Strategy registry, ConflictResolver coordinator, and default registry factory."""

from collections.abc import Sequence

from memory_engine.domain.enums import Cardinality
from memory_engine.domain.errors import InvariantViolation
from memory_engine.domain.models import Candidate, Fact, JudgeHint
from memory_engine.ingest.conflict.event import EventStrategy
from memory_engine.ingest.conflict.multi import MultiValuedStrategy
from memory_engine.ingest.conflict.plan import (
    ResolutionContext,
    ResolutionPlan,
    ResolutionStrategy,
)
from memory_engine.ingest.conflict.single import SingleValuedStrategy


class StrategyRegistry:
    """Open/Closed registry mapping predicate cardinality to a ResolutionStrategy."""

    def __init__(self) -> None:
        self._strategies: dict[Cardinality, ResolutionStrategy] = {}

    def register(self, s: ResolutionStrategy) -> None:
        """Register a ResolutionStrategy instance for its declared cardinality."""
        self._strategies[s.cardinality] = s

    def for_cardinality(self, c: Cardinality) -> ResolutionStrategy:
        """Retrieve the registered ResolutionStrategy for the requested cardinality."""
        if c not in self._strategies:
            raise InvariantViolation(f'No resolution strategy registered for cardinality {c!r}')
        return self._strategies[c]


class ConflictResolver:
    """Coordinator dispatching candidate conflict resolution to cardinality strategies."""

    def __init__(self, registry: StrategyRegistry) -> None:
        self._registry = registry

    def plan(
        self,
        c: Candidate,
        timeline: Sequence[Fact],
        ctx: ResolutionContext,
        hint: JudgeHint | None = None,
    ) -> ResolutionPlan:
        """Compute a resolution plan for the candidate against the existing fact timeline."""
        effective_hint = hint if hint is not None else c.hint
        strategy = self._registry.for_cardinality(c.cardinality)
        return strategy.plan(c, timeline, ctx, hint=effective_hint)


def default_registry() -> StrategyRegistry:
    """Construct and return the default StrategyRegistry with all frozen baseline strategies."""
    registry = StrategyRegistry()
    registry.register(SingleValuedStrategy())
    registry.register(MultiValuedStrategy())
    registry.register(EventStrategy())
    return registry
