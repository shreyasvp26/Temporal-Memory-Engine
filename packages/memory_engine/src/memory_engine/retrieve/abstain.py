"""Abstention policy and decision models per LLD Section 9.6 [FROZEN]."""

from dataclasses import dataclass
from typing import Any, Literal

from memory_engine.retrieve.assembler import AssembledContext
from memory_engine.retrieve.planner import QueryPlan

__all__ = [
    'AbstentionDecision',
    'AbstentionPolicy',
]

DEFAULT_ABSTAIN_REPLY = "I don't have anything in your memory about that yet."


@dataclass(frozen=True, slots=True)
class AbstentionDecision:
    """Decision indicating whether retrieval has sufficient evidence to synthesize an answer."""

    should_abstain: bool
    reason: Literal['none', 'no_evidence', 'low_evidence'] = 'none'
    reply: str | None = None

    def __bool__(self) -> bool:
        return self.should_abstain

    @property
    def abstained(self) -> bool:
        return self.should_abstain

    @classmethod
    def proceed(cls) -> 'AbstentionDecision':
        """Construct a decision to proceed with synthesis (do not abstain)."""
        return cls(should_abstain=False, reason='none', reply=None)

    @classmethod
    def abstain(
        cls,
        reason: Literal['no_evidence', 'low_evidence'],
        reply: str = DEFAULT_ABSTAIN_REPLY,
    ) -> 'AbstentionDecision':
        """Construct an abstention decision with the canned response."""
        return cls(should_abstain=True, reason=reason, reply=reply)


class AbstentionPolicy:
    """Evaluates assembled context and retrieval scores to decide whether to abstain [FROZEN].

    Rules:
      1. Statement-only never abstains.
      2. No facts and no sources in the context gives abstain(no_evidence) with canned reply:
         "I don't have anything in your memory about that yet."
      3. If min_score > 0, a best fused score below it with no StructuredLane hit
         gives abstain(low_evidence).
    """

    def __init__(
        self,
        cfg: Any = None,
        *,
        min_score: float | None = None,
    ) -> None:
        if min_score is not None:
            self.min_score = min_score
        elif cfg is not None and hasattr(cfg, 'abstain_min_score'):
            self.min_score = float(cfg.abstain_min_score)
        else:
            self.min_score = 0.0

    def decide(
        self,
        plan: QueryPlan,
        result: Any | None,
        ctx: AssembledContext,
        statement_only: bool = False,
    ) -> AbstentionDecision:
        """Decide whether to abstain from LLM synthesis based on context and retrieval scores."""
        # Rule 1: Statement-only never abstains
        if statement_only:
            return AbstentionDecision.proceed()

        # Rule 2: No facts and no sources in the context gives abstain(no_evidence)
        has_facts = bool(ctx.fact_labels)
        has_sources = any(t.kind == 'source' for t in ctx.traces)
        if not has_facts and not has_sources:
            return AbstentionDecision.abstain(
                reason='no_evidence',
                reply=DEFAULT_ABSTAIN_REPLY,
            )

        # Rule 3: If min_score > 0, low best score with no StructuredLane hit
        # gives abstain(low_evidence)
        if self.min_score > 0.0 and result is not None:
            fallback = result if isinstance(result, (list, tuple)) else []
            raw_items = getattr(result, 'items', fallback)
            items = list(raw_items)
            best_score = max((item.score for item in items), default=0.0)
            has_structured = any(
                'structured' in getattr(item, 'lanes', ())
                or 'temporal_facts' in getattr(item, 'lanes', ())
                for item in items
            ) or getattr(result, 'has_structured_hit', False)

            if best_score < self.min_score and not has_structured:
                return AbstentionDecision.abstain(
                    reason='low_evidence',
                    reply=DEFAULT_ABSTAIN_REPLY,
                )

        return AbstentionDecision.proceed()

    def evaluate(
        self,
        plan: QueryPlan,
        result: Any | None = None,
        ctx: AssembledContext | None = None,
        statement_only: bool = False,
    ) -> AbstentionDecision:
        """Evaluate retrieval result and assembled context for abstention."""
        assert ctx is not None, 'ctx must not be None'
        return self.decide(
            plan=plan,
            result=result,
            ctx=ctx,
            statement_only=statement_only,
        )
