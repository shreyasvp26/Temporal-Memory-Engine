"""Note chunker implementation per LLD Section 8.1."""

import re

__all__ = [
    'NoteChunker',
]


class NoteChunker:
    """Chunker for notes and documents implementing LLD Section 8.1 rules."""

    def split(self, text: str, max_chars: int = 1500, overlap: int = 200) -> list[str]:
        """Split text into chunks per Section 8.1 frozen rules.

        Rules:
        - Split on blank lines.
        - Merge adjacent paragraphs up to max_chars.
        - Split oversize paragraphs at sentence boundaries.
        - Hard-split at max_chars only when no boundary exists (overlap applies
          only to hard splits).
        - Discard chunks under 20 non-space characters.
        """
        if max_chars <= 0:
            raise ValueError('max_chars must be greater than 0')
        if overlap < 0 or overlap >= max_chars:
            raise ValueError('overlap must be non-negative and less than max_chars')

        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        if not paragraphs:
            return []

        chunks: list[str] = []
        current_para_merge: list[str] = []
        current_len = 0

        def flush_merged_paragraphs() -> None:
            nonlocal current_para_merge, current_len
            if current_para_merge:
                chunks.append('\n\n'.join(current_para_merge))
                current_para_merge = []
                current_len = 0

        for para in paragraphs:
            if len(para) > max_chars:
                flush_merged_paragraphs()
                oversize_pieces = self._split_oversize(para, max_chars, overlap)
                chunks.extend(oversize_pieces)
            else:
                needed = len(para) if not current_para_merge else current_len + 2 + len(para)
                if needed <= max_chars:
                    current_para_merge.append(para)
                    current_len = needed
                else:
                    flush_merged_paragraphs()
                    current_para_merge.append(para)
                    current_len = len(para)

        flush_merged_paragraphs()

        # Discard chunks under 20 non-space characters
        return [c.strip() for c in chunks if sum(1 for ch in c if not ch.isspace()) >= 20]

    def _split_oversize(self, para: str, max_chars: int, overlap: int) -> list[str]:
        """Split an oversize paragraph at sentence boundaries or hard-split if no boundary."""
        # Split on sentence boundaries: terminal punctuation (.?!) followed by space, or newlines
        sentences = [s.strip() for s in re.split(r'(?<=[.?!])\s+|\n+', para) if s.strip()]
        if len(sentences) <= 1:
            return self._hard_split(para, max_chars, overlap)

        pieces: list[str] = []
        current_sentence_group: list[str] = []
        current_s_len = 0

        def flush_sentences() -> None:
            nonlocal current_sentence_group, current_s_len
            if current_sentence_group:
                pieces.append(' '.join(current_sentence_group))
                current_sentence_group = []
                current_s_len = 0

        for s in sentences:
            if len(s) > max_chars:
                flush_sentences()
                pieces.extend(self._hard_split(s, max_chars, overlap))
            else:
                needed = len(s) if not current_sentence_group else current_s_len + 1 + len(s)
                if needed <= max_chars:
                    current_sentence_group.append(s)
                    current_s_len = needed
                else:
                    flush_sentences()
                    current_sentence_group.append(s)
                    current_s_len = len(s)

        flush_sentences()
        return pieces

    @staticmethod
    def _hard_split(text: str, max_chars: int, overlap: int) -> list[str]:
        """Hard-split text at max_chars with overlap."""
        if len(text) <= max_chars:
            return [text]
        step = max_chars - overlap
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start += step
        return chunks
