"""Unit tests for IngestGate and NoteChunker (LLD Section 8.1, T-ING)."""

from dataclasses import FrozenInstanceError

import pytest
from memory_engine.ingest.gate import GateDecision, IngestGate
from memory_engine.ingest.windowing import NoteChunker

# -----------------------------------------------------------------------------
# IngestGate Tests (Rules 1-6 & is_statement_only)
# -----------------------------------------------------------------------------


def test_gate_decision_dataclass_invariants() -> None:
    decision = GateDecision(ingest=True, reason='has_statement')
    assert decision.ingest is True
    assert decision.reason == 'has_statement'
    with pytest.raises(FrozenInstanceError):
        decision.ingest = False  # type: ignore[misc]


def test_gate_rule_1_too_short() -> None:
    gate = IngestGate()

    assert gate.decide('').reason == 'too_short'
    assert gate.decide('   ').reason == 'too_short'
    assert gate.decide('a').reason == 'too_short'
    assert gate.decide('Hi').reason == 'too_short'
    assert gate.decide('Hey').reason == 'too_short'
    assert gate.decide('  a b  ').reason == 'too_short'

    dec = gate.decide('Hey')
    assert dec.ingest is False
    assert dec.reason == 'too_short'


def test_gate_rule_3_interrogative_examples() -> None:
    gate = IngestGate()

    # Spec example
    dec = gate.decide('Where do I live?')
    assert dec.ingest is False
    assert dec.reason == 'question_only'

    # Question ending with ? without question word
    dec = gate.decide('You live in Pune?')
    assert dec.ingest is False
    assert dec.reason == 'question_only'

    # Multiple questions
    dec = gate.decide('Where are you? What time is it?')
    assert dec.ingest is False
    assert dec.reason == 'question_only'


def test_gate_rule_3_all_interrogative_starter_words() -> None:
    gate = IngestGate()
    starters = [
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
    ]
    for starter in starters:
        text = f'{starter.capitalize()} you know the answer'
        dec = gate.decide(text)
        assert dec.ingest is False, f'Failed for starter: {starter}'
        assert dec.reason == 'question_only', f'Failed for starter: {starter}'


def test_gate_rule_4_imperative_examples() -> None:
    gate = IngestGate()

    # Spec example
    dec = gate.decide('Tell me what I decided')
    assert dec.ingest is False
    assert dec.reason == 'imperative_only'

    # With punctuation
    dec = gate.decide('Remind me to call Mom tomorrow.')
    assert dec.ingest is False
    assert dec.reason == 'imperative_only'


def test_gate_rule_4_all_imperative_starter_words() -> None:
    gate = IngestGate()
    starters = [
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
    ]
    for starter in starters:
        text = f'{starter.capitalize()} the latest status update'
        dec = gate.decide(text)
        assert dec.ingest is False, f'Failed for imperative starter: {starter}'
        assert dec.reason == 'imperative_only', f'Failed for imperative starter: {starter}'


def test_gate_rule_5_has_statement_examples() -> None:
    gate = IngestGate()

    # Spec example: mixed statement and question
    dec = gate.decide('I just moved to Pune. Where is a good cafe nearby?')
    assert dec.ingest is True
    assert dec.reason == 'has_statement'

    # Pure statement
    dec = gate.decide('I am a software engineer working on memory engines.')
    assert dec.ingest is True
    assert dec.reason == 'has_statement'

    # Statement split across newlines
    dec = gate.decide('I moved to Pune\nWhere is a good cafe?')
    assert dec.ingest is True
    assert dec.reason == 'has_statement'

    # Statement with exclamation mark
    dec = gate.decide('I love this new city!')
    assert dec.ingest is True
    assert dec.reason == 'has_statement'


def test_gate_short_sentences_under_three_words_not_statement() -> None:
    gate = IngestGate()

    # Two words - not enough for a statement
    dec = gate.decide('I moved.')
    assert dec.ingest is False
    assert dec.reason == 'too_short'

    # Three words - counts as statement
    dec = gate.decide('I moved away.')
    assert dec.ingest is True
    assert dec.reason == 'has_statement'


def test_gate_mixed_imperative_and_question_without_statement() -> None:
    gate = IngestGate()

    # Imperative followed by question: question takes priority as reason
    dec = gate.decide('Tell me please. Where are you going?')
    assert dec.ingest is False
    assert dec.reason == 'question_only'


def test_gate_is_statement_only() -> None:
    gate = IngestGate()

    # Statement with question -> False
    assert gate.is_statement_only('I just moved to Pune. Where is a good cafe nearby?') is False

    # Statement only -> True
    assert gate.is_statement_only('I just moved to Pune. I love the new apartment.') is True
    assert gate.is_statement_only('I am living in Pune now.') is True

    # Questions -> False
    assert gate.is_statement_only('Where do I live?') is False

    # Imperatives -> False
    assert gate.is_statement_only('Tell me what I decided') is False

    # Too short -> False
    assert gate.is_statement_only('Hi') is False

    # Statement with imperative (no question) -> True
    assert gate.is_statement_only('I moved to Pune. Tell me you heard that.') is True


# -----------------------------------------------------------------------------
# NoteChunker Tests (Paragraph merging, sentence splitting, hard split, discard)
# -----------------------------------------------------------------------------


def test_chunker_argument_validation() -> None:
    chunker = NoteChunker()

    with pytest.raises(ValueError, match='max_chars must be greater than 0'):
        chunker.split('Some text here.', max_chars=0)

    with pytest.raises(ValueError, match='max_chars must be greater than 0'):
        chunker.split('Some text here.', max_chars=-10)

    with pytest.raises(ValueError, match='overlap must be non-negative and less than max_chars'):
        chunker.split('Some text here.', max_chars=100, overlap=-1)

    with pytest.raises(ValueError, match='overlap must be non-negative and less than max_chars'):
        chunker.split('Some text here.', max_chars=100, overlap=100)

    with pytest.raises(ValueError, match='overlap must be non-negative and less than max_chars'):
        chunker.split('Some text here.', max_chars=100, overlap=150)


def test_chunker_discard_under_20_non_space_chars() -> None:
    chunker = NoteChunker()

    assert chunker.split('') == []
    assert chunker.split('   \n\n   ') == []
    # 19 non-space characters: "1234567890123456789"
    assert chunker.split('1234567890123456789') == []
    # 20 non-space characters
    result = chunker.split('12345678901234567890')
    assert result == ['12345678901234567890']


def test_chunker_paragraph_merging() -> None:
    chunker = NoteChunker()
    para1 = 'This is the first paragraph of the note.'
    para2 = 'This is the second paragraph of the note.'
    para3 = 'This is the third paragraph of the note.'

    text = f'{para1}\n\n{para2}\n\n{para3}'

    # When max_chars is large enough, all paragraphs merge into 1 chunk
    chunks = chunker.split(text, max_chars=1500)
    assert len(chunks) == 1
    assert chunks[0] == f'{para1}\n\n{para2}\n\n{para3}'

    # When max_chars fits only 2 paragraphs
    limit = len(para1) + 2 + len(para2) + 5
    chunks_split = chunker.split(text, max_chars=limit, overlap=10)
    assert len(chunks_split) == 2
    assert chunks_split[0] == f'{para1}\n\n{para2}'
    assert chunks_split[1] == para3


def test_chunker_oversize_paragraph_sentence_splitting() -> None:
    chunker = NoteChunker()

    s1 = 'Sentence number one is about twenty-five chars.'
    s2 = 'Sentence number two is also about thirty characters.'
    s3 = 'Sentence number three concludes the paragraph test.'
    para = f'{s1} {s2} {s3}'

    # Set max_chars so s1 and s2 fit, but s3 must go to a second chunk
    max_chars = len(s1) + 1 + len(s2) + 5
    chunks = chunker.split(para, max_chars=max_chars, overlap=10)

    assert len(chunks) == 2
    assert chunks[0] == f'{s1} {s2}'
    assert chunks[1] == s3


def test_chunker_hard_split_with_overlap_when_no_sentence_boundary() -> None:
    chunker = NoteChunker()

    # Long text with NO sentence boundaries (no terminal punctuation or newlines)
    # 250 characters of repeating alphabet
    raw = 'abcdefghijklmnopqrstuvwxyz0123456789' * 7  # 252 chars
    max_chars = 100
    overlap = 20

    chunks = chunker.split(raw, max_chars=max_chars, overlap=overlap)

    assert len(chunks) == 3
    # First chunk: 0..100
    assert chunks[0] == raw[0:100]
    # Second chunk: 80..180 (overlap of 20 with first chunk)
    assert chunks[1] == raw[80:180]
    assert chunks[0][-20:] == chunks[1][:20]
    # Third chunk: 160..252 (overlap of 20 with second chunk)
    assert chunks[2] == raw[160:252]
    assert chunks[1][-20:] == chunks[2][:20]


def test_chunker_overlap_only_applies_to_hard_splits() -> None:
    chunker = NoteChunker()

    # Sentences with sentence boundaries
    s1 = 'Sentence one has ending.'
    s2 = 'Sentence two has ending.'
    para = f'{s1} {s2}'

    # Force split across sentence boundary
    chunks = chunker.split(para, max_chars=30, overlap=10)
    assert len(chunks) == 2
    assert chunks[0] == s1
    assert chunks[1] == s2
    # Ensure no overlap was injected between s1 and s2
    assert not chunks[1].startswith(chunks[0][-10:])


def test_chunker_mixed_document_flow() -> None:
    chunker = NoteChunker()

    p1 = 'Introduction to the system architecture and overview.'
    # Oversize paragraph with sentences
    s_a = 'This is section A of the long detailed documentation.'
    s_b = 'This is section B containing further technical specifics.'
    p2 = f'{s_a} {s_b}'
    p3 = 'Conclusion and summary of findings.'

    full_text = f'{p1}\n\n{p2}\n\n{p3}'

    # With max_chars=65:
    # p1 fits in 65
    # p2 is ~110 chars, split into s_a (~53 chars) and s_b (~57 chars)
    # p3 fits in 65
    chunks = chunker.split(full_text, max_chars=65, overlap=15)
    assert len(chunks) == 4
    assert chunks[0] == p1
    assert chunks[1] == s_a
    assert chunks[2] == s_b
    assert chunks[3] == p3
