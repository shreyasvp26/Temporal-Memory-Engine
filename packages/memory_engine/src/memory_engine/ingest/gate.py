"""Ingest gate implementation per LLD Section 8.1."""

import re
from dataclasses import dataclass

__all__ = [
    'GateDecision',
    'IngestGate',
]

_INTERROGATIVE_STARTERS: frozenset[str] = frozenset(
    {
        'what',
        'when',
        'where',
        'who',
        'whom',
        'whose',
        'why',
        'how',
        'which',
        'do',
        'does',
        'did',
        'can',
        'could',
        'will',
        'would',
        'should',
        'is',
        'are',
        'am',
        'was',
        'were',
        'have',
        'has',
        'had',
    }
)

_IMPERATIVE_STARTERS: frozenset[str] = frozenset(
    {
        'tell',
        'show',
        'list',
        'give',
        'find',
        'remind',
        'summarize',
        'explain',
        'write',
        'help',
        'search',
        'recall',
        'check',
    }
)


@dataclass(frozen=True, slots=True)
class GateDecision:
    """Decision produced by IngestGate."""

    ingest: bool
    reason: str


class IngestGate:
    """Gate deciding whether incoming chat message text should be ingested."""

    def decide(self, text: str) -> GateDecision:
        """Evaluate text against the 6 ingestion gate rules.

        Rule 1: Fewer than 4 non-space characters: skip (too_short).
        Rule 2: Split into sentences on terminal punctuation and newlines.
        Rule 3: A sentence is interrogative if it ends with '?' or begins
                (case-insensitive) with an interrogative starter word.
        Rule 4: A sentence is imperative if its first word is an imperative starter word.
        Rule 5: Ingest if >= 1 sentence is neither interrogative nor imperative and
                has >= 3 words (has_statement). Otherwise skip (question_only or imperative_only).
        Rule 6: Notes and documents bypass the gate (handled by caller).
        """
        non_space_count = sum(1 for c in text if not c.isspace())
        if non_space_count < 4:
            return GateDecision(ingest=False, reason='too_short')

        sentences = self._split_sentences(text)
        if not sentences:
            return GateDecision(ingest=False, reason='too_short')

        has_statement = False
        has_question = False
        has_imperative = False

        for sentence in sentences:
            is_q = self._is_interrogative(sentence)
            is_imp = self._is_imperative(sentence)

            if is_q:
                has_question = True
            if is_imp:
                has_imperative = True

            words = self._get_words(sentence)
            if not is_q and not is_imp and len(words) >= 3:
                has_statement = True
                break

        if has_statement:
            return GateDecision(ingest=True, reason='has_statement')

        if has_question:
            return GateDecision(ingest=False, reason='question_only')

        if has_imperative:
            return GateDecision(ingest=False, reason='imperative_only')

        return GateDecision(ingest=False, reason='too_short')

    def is_statement_only(self, text: str) -> bool:
        """True when decide(text).ingest is True and NO sentence in text is interrogative."""
        decision = self.decide(text)
        if not decision.ingest:
            return False
        sentences = self._split_sentences(text)
        return not any(self._is_interrogative(s) for s in sentences)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split into sentences on terminal punctuation (.?!) and newlines."""
        parts = re.split(r'(\r?\n+|[.?!]+)', text)
        sentences: list[str] = []
        current = ''
        for part in parts:
            if not part:
                continue
            if re.fullmatch(r'\r?\n+|[.?!]+', part):
                if any(c in '.?!' for c in part):
                    current += part
                    if current.strip():
                        sentences.append(current.strip())
                    current = ''
                else:
                    if current.strip():
                        sentences.append(current.strip())
                    current = ''
            else:
                current += part
        if current.strip():
            sentences.append(current.strip())
        return sentences

    @classmethod
    def _get_words(cls, sentence: str) -> list[str]:
        """Extract words from a sentence, stripping common punctuation."""
        return [w for token in sentence.split() if (w := token.strip('.,?!:;"\'()[]{}<>-'))]

    @classmethod
    def _is_interrogative(cls, sentence: str) -> bool:
        """Check if a sentence is interrogative."""
        stripped = sentence.rstrip()
        if stripped.endswith('?') or stripped.rstrip('"\'').endswith('?'):
            return True
        words = cls._get_words(sentence)
        return bool(words and words[0].lower() in _INTERROGATIVE_STARTERS)

    @classmethod
    def _is_imperative(cls, sentence: str) -> bool:
        """Check if a sentence is imperative."""
        words = cls._get_words(sentence)
        return bool(words and words[0].lower() in _IMPERATIVE_STARTERS)
