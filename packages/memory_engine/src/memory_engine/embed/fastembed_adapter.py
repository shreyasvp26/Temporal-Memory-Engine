"""FastEmbed adapter implementing the Embedder port with Nomic v1.5 embeddings."""

import asyncio
import math
from collections.abc import Iterable, Sequence
from typing import Any

from memory_engine.domain.errors import EmbeddingError
from memory_engine.ports.embed import Embedder

__all__ = ['FastEmbedAdapter']


def _l2_normalize(vec: Sequence[float]) -> list[float]:
    """L2-normalize a vector, raising EmbeddingError on zero norm or NaN."""
    sq_sum = 0.0
    for x in vec:
        if math.isnan(x) or math.isinf(x):
            raise EmbeddingError('Vector contains NaN or Inf components', code='E-EMB-01')
        sq_sum += x * x

    norm = math.sqrt(sq_sum)
    if norm == 0.0:
        raise EmbeddingError('Cannot normalize zero vector', code='E-EMB-01')

    return [float(x / norm) for x in vec]


class FastEmbedAdapter(Embedder):
    """Local Embedder adapter using fastembed nomic-ai/nomic-embed-text-v1.5."""

    def __init__(
        self,
        model_id: str = 'nomic-ai/nomic-embed-text-v1.5',
        dim: int = 768,
        model: Any = None,
        max_batch_size: int = 32,
    ) -> None:
        self.model_id = model_id
        self.dim = dim
        self._max_batch_size = max_batch_size
        self._semaphore = asyncio.Semaphore(2)
        self._model = model

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.model_id)
            return self._model
        except ImportError as err:
            raise EmbeddingError(
                'fastembed is not installed. Please install fastembed or provide an initialized '
                'model.',
                code='E-EMB-01',
            ) from err

    def _encode_sync(self, prefixed_texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        results: list[list[float]] = []

        for i in range(0, len(prefixed_texts), self._max_batch_size):
            batch = prefixed_texts[i : i + self._max_batch_size]
            if hasattr(model, 'embed'):
                raw_out = model.embed(batch)
            elif callable(model):
                raw_out = model(batch)
            else:
                raise EmbeddingError(
                    f'Unsupported model object type: {type(model)!r}', code='E-EMB-01'
                )

            # raw_out can be an iterable of numpy arrays or lists
            if isinstance(raw_out, Iterable):
                for item in raw_out:
                    vec = list(item)
                    if len(vec) != self.dim:
                        raise EmbeddingError(
                            f'Expected vector dimension {self.dim}, got {len(vec)}',
                            code='E-EMB-01',
                        )
                    norm_vec = _l2_normalize(vec)
                    results.append(norm_vec)
            else:
                raise EmbeddingError('Model output is not iterable', code='E-EMB-01')

        return results

    async def _encode_prefixed(self, prefix: str, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        # Validate max text length per LLD Section 7.4 (reject > 2000 chars)
        for t in texts:
            if len(t) > 2000:
                raise EmbeddingError(
                    f'Text length {len(t)} exceeds max 2000 characters limit',
                    code='E-EMB-01',
                )

        prefixed_texts = [f'{prefix}{t}' for t in texts]

        async with self._semaphore:
            return await asyncio.to_thread(self._encode_sync, prefixed_texts)

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed documents with prefix 'search_document: '."""
        return await self._encode_prefixed('search_document: ', texts)

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed queries with prefix 'search_query: '."""
        return await self._encode_prefixed('search_query: ', texts)
