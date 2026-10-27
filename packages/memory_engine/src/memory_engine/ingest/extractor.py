"""Fact extraction pipeline per docs/LLD.md Section 8.3 and 8.4."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from memory_engine.domain.models import ExtractedFact, ExtractionResult, Source
from memory_engine.ports.llm import ChatMessage, LLMGateway, LLMRequest
from memory_engine.ports.repositories import PredicateRegistry, SourceWindow

_logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You extract durable personal facts from a conversation window.
Only extract from lines marked TARGET.
Lines marked CONTEXT exist only to resolve pronouns and references.
Rules:
1. Extract only what the speaker explicitly states about themselves or others.
   Never infer or guess.
2. Write the first person as the single word I in the subject or object field.
3. Use one of the allowed predicate names. If none fits, use other.
4. Copy time expressions exactly as written into valid_from_text or valid_to_text
   (for example: last March, 1 Oct 2026, two years ago, since 2019).
   Never convert them to dates. Leave null if no time is stated.
5. Set is_retraction true when the speaker says a state ended
   (no longer, quit, moved out of, stopped).
   Use the same predicate and object as the state being ended.
6. Questions, hypotheticals, jokes, and other people reporting rumors are not facts.
7. evidence_span must be copied verbatim from the TARGET line and be at most 200 characters.
8. source_turn is the turn number shown on the TARGET line.
9. Intentions about the future use plans_to, never lives_in or works_at.
Return only JSON matching the provided schema. Return an empty facts list when nothing qualifies."""


@dataclass(frozen=True, slots=True)
class ValidatedExtraction:
    """Pairing of an extracted fact with its resolved target Source."""

    fact: ExtractedFact
    source: Source
    span_start: int | None = None
    span_end: int | None = None


def norm_span_text(text: str) -> str:
    """Collapse internal whitespace and lower-case for robust matching."""
    return ' '.join(text.casefold().split())


def format_window(window: SourceWindow, targets: Sequence[Source]) -> str:
    """Render conversation window with labeled CONTEXT and TARGET lines."""
    lines: list[str] = []
    for s in window.context:
        if s.deleted_at is None:
            role_val = s.role.value if hasattr(s.role, 'value') else str(s.role)
            ts = s.source_ts.isoformat()
            text = s.content[:4000]
            lines.append(f'[CONTEXT] [turn {s.turn_index} | {ts} | {role_val}]: {text}')
    for s in targets:
        role_val = s.role.value if hasattr(s.role, 'value') else str(s.role)
        ts = s.source_ts.isoformat()
        text = s.content[:4000]
        lines.append(f'[TARGET] [turn {s.turn_index} | {ts} | {role_val}]: {text}')
    return '\n'.join(lines)


class FactExtractor:
    """Fact extractor leveraging LLMGateway and post-validation per LLD Section 8.4."""

    def __init__(
        self,
        llm: LLMGateway,
        registry: PredicateRegistry | None = None,
        cfg: Any | None = None,
        *,
        min_confidence: float = 0.5,
        extract_assistant: bool = False,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.min_confidence = (
            getattr(cfg, 'min_confidence', min_confidence) if cfg is not None else min_confidence
        )
        self.extract_assistant = (
            getattr(cfg, 'extract_assistant', extract_assistant)
            if cfg is not None
            else extract_assistant
        )
        self.span_mismatches: int = 0

    def _is_extractable(self, source: Source) -> bool:
        if source.deleted_at is not None:
            return False
        role = source.role.value if hasattr(source.role, 'value') else str(source.role)
        if role == 'user':
            return True
        return bool(role == 'assistant' and self.extract_assistant)

    async def extract(self, window: SourceWindow) -> list[ValidatedExtraction]:
        """Extract durable facts from target turns in the given conversation window."""
        targets = [s for s in window.targets if self._is_extractable(s)]
        if not targets:
            return []

        # Build window text and allowed predicates block
        window_text = format_window(window, targets)
        if self.registry is not None:
            preds = self.registry.all()
            predicates_str = ', '.join(p.name for p in preds)
        else:
            predicates_str = 'any allowed predicate or other'

        user_content = (
            f'ALLOWED PREDICATES:\n{predicates_str}\n\n'
            f'CONVERSATION WINDOW:\n{window_text}\n\n'
            'Extract durable facts conforming to the rules. Return JSON only.'
        )

        req = LLMRequest(
            purpose='extract',
            tier='fast',
            messages=[
                ChatMessage(role='system', content=EXTRACTION_SYSTEM_PROMPT),
                ChatMessage(role='user', content=user_content),
            ],
            prompt_version='extract.v1',
            temperature=0.0,
            max_tokens=1500,
            json_mode=True,
        )

        raw_result = await self.llm.complete_json(req, ExtractionResult)
        if isinstance(raw_result, ExtractionResult):
            facts = raw_result.facts
        elif isinstance(raw_result, dict):
            validated_model = ExtractionResult.model_validate(raw_result)
            facts = validated_model.facts
        else:
            facts = []

        target_by_turn: dict[int, Source] = {s.turn_index: s for s in targets}
        context_turn_indices: set[int] = {s.turn_index for s in window.context}

        validated: list[ValidatedExtraction] = []
        seen_keys: set[tuple[str, str, str, str]] = set()

        for fact in facts:
            # 1. Reject empty subject, predicate, or object
            if not fact.subject.strip() or not fact.predicate.strip() or not fact.object.strip():
                continue

            # 2. Check minimum confidence
            if fact.confidence < self.min_confidence:
                continue

            # 3. Reject source_turn pointing to read-only context turn
            if fact.source_turn in context_turn_indices:
                continue

            # 4. Must map to an extractable target turn
            target = target_by_turn.get(fact.source_turn)
            if target is None:
                continue

            # 5. Whitespace-normalized case-insensitive evidence_span check
            span = fact.evidence_span.strip()
            if not span:
                self.span_mismatches += 1
                _logger.warning('span_mismatch: empty_span in turn %d', fact.source_turn)
                continue

            norm_span = norm_span_text(span)
            norm_content = norm_span_text(target.content)
            if norm_span not in norm_content:
                self.span_mismatches += 1
                _logger.warning(
                    'span_mismatch: %r not in target turn %d content',
                    fact.evidence_span,
                    fact.source_turn,
                )
                continue

            # 6. Drop exact duplicates by (subject, predicate, object, valid_from_text)
            dedup_key = (
                norm_span_text(fact.subject),
                norm_span_text(fact.predicate),
                norm_span_text(fact.object),
                (fact.valid_from_text or '').strip().lower(),
            )
            if dedup_key in seen_keys:
                continue
            seen_keys.add(dedup_key)

            # Compute span offsets if exact or casefold substring
            span_start: int | None = None
            span_end: int | None = None
            idx = target.content.find(fact.evidence_span)
            if idx != -1:
                span_start = idx
                span_end = idx + len(fact.evidence_span)
            else:
                idx_lower = target.content.casefold().find(fact.evidence_span.casefold())
                if idx_lower != -1:
                    span_start = idx_lower
                    span_end = idx_lower + len(fact.evidence_span)

            validated.append(
                ValidatedExtraction(
                    fact=fact,
                    source=target,
                    span_start=span_start,
                    span_end=span_end,
                )
            )

            # Cap at 30 facts per window
            if len(validated) >= 30:
                break

        return validated

    async def extract_result(self, window: SourceWindow) -> ExtractionResult:
        """Convenience method returning standard ExtractionResult."""
        extractions = await self.extract(window)
        return ExtractionResult(facts=[e.fact for e in extractions])


LLMFactExtractor = FactExtractor
