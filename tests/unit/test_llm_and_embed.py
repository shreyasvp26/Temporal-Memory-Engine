"""Unit tests for LLM gateway, provider adapters, schemas, and embedders.

All tests run socket-free using httpx.MockTransport and in-memory/sqlite fixtures.
"""

import json
import math
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from memory_engine.config import Settings
from memory_engine.domain.enums import Intent
from memory_engine.domain.errors import (
    EmbeddingError,
    LLMContentBlocked,
    LLMRateLimited,
    LLMTimeout,
    LLMUnavailable,
    SchemaViolation,
    StreamInterrupted,
)
from memory_engine.embed.cached import CachedEmbedder
from memory_engine.embed.fastembed_adapter import FastEmbedAdapter
from memory_engine.llm.gateway import DefaultLLMGateway
from memory_engine.llm.gemini import GeminiClient
from memory_engine.llm.groq import GroqClient
from memory_engine.llm.ollama import OllamaClient
from memory_engine.llm.prompts import (
    build_conflict_judge_user_prompt,
    build_extraction_user_prompt,
    build_grounded_chat_user_prompt,
    build_planner_user_prompt,
    build_repair_user_prompt,
)
from memory_engine.llm.schemas import (
    JudgeLLMOutput,
    PlannerLLMOutput,
    RepairPromptBuilder,
)
from memory_engine.ports.clock import FixedClock
from memory_engine.ports.llm import ChatMessage, LLMClient, LLMRequest, LLMResponse
from pydantic import BaseModel, ConfigDict, ValidationError


class SampleModel(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    name: str
    count: int


# ==============================================================================
# 1. Schemas & Prompts Tests
# ==============================================================================


def test_planner_llm_output_schema() -> None:
    output = PlannerLLMOutput(
        intent=Intent.CURRENT_STATE,
        time_phrases=['now', 'today'],
        entity_names=['User', 'Paris'],
        predicates=['lives_in'],
        multi_hop=False,
    )
    assert output.intent == Intent.CURRENT_STATE
    assert output.time_phrases == ['now', 'today']
    assert output.multi_hop is False


def test_judge_llm_output_schema() -> None:
    output = JudgeLLMOutput(
        verdict='supersedes',
        rationale='User moved to a new apartment',
        confidence=0.95,
    )
    assert output.verdict == 'supersedes'
    assert output.confidence == 0.95

    with pytest.raises(ValidationError):
        JudgeLLMOutput.model_validate({'verdict': 'invalid_verdict'})


def test_repair_prompt_builder() -> None:
    prompt = RepairPromptBuilder.build(
        invalid_text='{bad json',
        error_message='Expecting property name',
        schema_def='{"type": "object"}',
    )
    assert 'The previous JSON response was invalid' in prompt
    assert '{bad json' in prompt
    assert 'Expecting property name' in prompt
    assert '{"type": "object"}' in prompt


def test_prompt_builders() -> None:
    ext_prompt = build_extraction_user_prompt(
        allowed_predicates=['lives_in', 'works_at'],
        window_text='[TARGET 1] I live in Berlin.',
    )
    assert 'ALLOWED PREDICATES:' in ext_prompt
    assert 'lives_in, works_at' in ext_prompt

    plan_prompt = build_planner_user_prompt('Where did I live in 2020?')
    assert 'Where did I live in 2020?' in plan_prompt

    judge_prompt = build_conflict_judge_user_prompt(
        candidate_fact='User lives in Paris',
        existing_fact='User lives in London',
        evidence='I moved to Paris last week.',
    )
    assert 'User lives in Paris' in judge_prompt
    assert 'User lives in London' in judge_prompt

    chat_prompt = build_grounded_chat_user_prompt(
        today='2026-10-27',
        memory_block='[F1] User lives in Berlin.',
        question='Where do I live?',
    )
    assert 'Today: 2026-10-27' in chat_prompt
    assert '[F1] User lives in Berlin.' in chat_prompt
    assert 'QUESTION: Where do I live?' in chat_prompt

    repair_prompt = build_repair_user_prompt(
        invalid_text='{"name":}',
        error_message='SyntaxError',
    )
    assert 'SyntaxError' in repair_prompt


# ==============================================================================
# 2. Provider Clients Tests (httpx.MockTransport)
# ==============================================================================


@pytest.mark.asyncio
async def test_groq_client_success_and_errors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        if 'error_429' in str(body):
            return httpx.Response(429, headers={'retry-after': '15.0'}, text='Rate limited')
        if 'error_408' in str(body):
            return httpx.Response(408, text='Request timeout')
        if 'error_blocked' in str(body):
            return httpx.Response(400, text='Blocked by safety moderation')
        if 'error_500' in str(body):
            return httpx.Response(500, text='Internal Server Error')

        resp_data = {
            'choices': [{'message': {'content': 'Hello from Groq'}}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5},
            'model': 'llama-3.1-8b-instant',
        }
        return httpx.Response(200, json=resp_data)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        groq = GroqClient(api_key='test-key', client=http_client)

        # 1. Success
        req = LLMRequest(
            purpose='extract',
            tier='fast',
            messages=[ChatMessage(role='user', content='Hello')],
            prompt_version='v1',
        )
        resp = await groq.complete(req)
        assert resp.text == 'Hello from Groq'
        assert resp.provider == 'groq'
        assert resp.prompt_tokens == 10

        # 2. 429 Rate limited
        req_429 = LLMRequest(
            purpose='extract',
            tier='fast',
            messages=[ChatMessage(role='user', content='error_429')],
            prompt_version='v1',
        )
        with pytest.raises(LLMRateLimited) as exc_info:
            await groq.complete(req_429)
        assert exc_info.value.retry_after == 15.0

        # 3. 408 Timeout
        req_408 = LLMRequest(
            purpose='extract',
            tier='fast',
            messages=[ChatMessage(role='user', content='error_408')],
            prompt_version='v1',
        )
        with pytest.raises(LLMTimeout):
            await groq.complete(req_408)

        # 4. Content blocked
        req_blocked = LLMRequest(
            purpose='extract',
            tier='fast',
            messages=[ChatMessage(role='user', content='error_blocked')],
            prompt_version='v1',
        )
        with pytest.raises(LLMContentBlocked):
            await groq.complete(req_blocked)

        # 5. 500 Unavailable
        req_500 = LLMRequest(
            purpose='extract',
            tier='fast',
            messages=[ChatMessage(role='user', content='error_500')],
            prompt_version='v1',
        )
        with pytest.raises(LLMUnavailable):
            await groq.complete(req_500)


@pytest.mark.asyncio
async def test_groq_streaming() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        sse_lines = [
            'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n',
            'data: {"choices": [{"delta": {"content": " world"}}]}\n\n',
            'data: [DONE]\n\n',
        ]
        return httpx.Response(200, text=''.join(sse_lines))

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        groq = GroqClient(api_key='test-key', client=http_client)
        req = LLMRequest(
            purpose='synth',
            tier='fast',
            messages=[ChatMessage(role='user', content='stream test')],
            prompt_version='v1',
        )
        chunks = [c async for c in groq.stream(req)]
        assert ''.join(chunks) == 'Hello world'


@pytest.mark.asyncio
async def test_gemini_client_success_and_blocked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        if 'error_blocked' in str(body):
            resp_blocked: dict[str, Any] = {'candidates': [{'finishReason': 'SAFETY'}]}
            return httpx.Response(200, json=resp_blocked)
        if 'error_429' in str(body):
            return httpx.Response(429, headers={'retry-after': '20.0'}, text='Quota exceeded')

        resp_data: dict[str, Any] = {
            'candidates': [{'content': {'parts': [{'text': 'Gemini answer'}]}}],
            'usageMetadata': {'promptTokenCount': 12, 'candidatesTokenCount': 6},
        }
        return httpx.Response(200, json=resp_data)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        gemini = GeminiClient(api_key='test-key', client=http_client)

        req = LLMRequest(
            purpose='synth',
            tier='synth',
            messages=[ChatMessage(role='user', content='Hi Gemini')],
            prompt_version='v1',
        )
        resp = await gemini.complete(req)
        assert resp.text == 'Gemini answer'
        assert resp.provider == 'gemini'

        req_blocked = LLMRequest(
            purpose='synth',
            tier='synth',
            messages=[ChatMessage(role='user', content='error_blocked')],
            prompt_version='v1',
        )
        with pytest.raises(LLMContentBlocked):
            await gemini.complete(req_blocked)

        req_429 = LLMRequest(
            purpose='synth',
            tier='synth',
            messages=[ChatMessage(role='user', content='error_429')],
            prompt_version='v1',
        )
        with pytest.raises(LLMRateLimited) as exc_info:
            await gemini.complete(req_429)
        assert exc_info.value.retry_after == 20.0


@pytest.mark.asyncio
async def test_ollama_client() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        resp_data = {
            'message': {'content': '{"name": "test", "count": 1}'},
            'prompt_eval_count': 8,
            'eval_count': 4,
            'model': 'llama3.1:8b',
        }
        return httpx.Response(200, json=resp_data)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        ollama = OllamaClient(client=http_client)
        req = LLMRequest(
            purpose='extract',
            tier='fast',
            messages=[ChatMessage(role='user', content='test')],
            prompt_version='v1',
            json_mode=True,
        )
        resp = await ollama.complete(req)
        assert '{"name": "test"' in resp.text
        assert resp.provider == 'ollama'


# ==============================================================================
# 3. DefaultLLMGateway Tests (Failover, JSON Validation & Repair)
# ==============================================================================


class FakeClient(LLMClient):
    def __init__(self, name: str, responses: list[Any]) -> None:
        self.name = name
        self._responses = list(responses)
        self.call_count: int = 0
        self.last_req: LLMRequest | None = None

    async def complete(self, req: LLMRequest) -> LLMResponse:
        self.call_count += 1
        self.last_req = req
        if not self._responses:
            raise LLMUnavailable(f'{self.name} out of responses')
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        if isinstance(item, LLMResponse):
            return item
        return LLMResponse(
            text=str(item),
            model=f'{self.name}-model',
            provider=self.name,
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=10.0,
        )

    def stream(self, req: LLMRequest) -> AsyncIterator[str]:
        self.call_count += 1
        self.last_req = req
        if not self._responses:
            raise LLMUnavailable(f'{self.name} out of responses')
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item

        async def _generator() -> AsyncIterator[str]:
            if isinstance(item, list):
                for chunk in item:
                    if isinstance(chunk, Exception):
                        raise chunk
                    yield str(chunk)
            else:
                yield str(item)

        return _generator()


@pytest.mark.asyncio
async def test_gateway_complete_text_failover() -> None:
    # Primary groq fails with 429 -> falls back to gemini
    groq = FakeClient('groq', [LLMRateLimited('rate limited', retry_after=12.0)])
    gemini = FakeClient('gemini', ['Response from gemini'])
    clock = FixedClock(fixed_now=datetime(2026, 10, 27, 12, 0, tzinfo=UTC))
    settings = Settings(llm_fast_chain='groq,gemini')

    gateway = DefaultLLMGateway(
        providers={'groq': groq, 'gemini': gemini},
        settings=settings,
        clock=clock,
    )

    req = LLMRequest(
        purpose='plan',
        tier='fast',
        messages=[ChatMessage(role='user', content='plan this')],
        prompt_version='v1',
    )
    resp = await gateway.complete_text(req)
    assert resp.text == 'Response from gemini'
    assert resp.provider == 'gemini'
    assert groq.call_count == 1
    assert gemini.call_count == 1


@pytest.mark.asyncio
async def test_gateway_complete_text_all_fail_carries_retry_after() -> None:
    groq = FakeClient('groq', [LLMRateLimited('groq limited', retry_after=10.0)])
    gemini = FakeClient('gemini', [LLMRateLimited('gemini limited', retry_after=25.0)])
    settings = Settings(llm_fast_chain='groq,gemini')

    gateway = DefaultLLMGateway(
        providers={'groq': groq, 'gemini': gemini},
        settings=settings,
    )

    req = LLMRequest(
        purpose='plan',
        tier='fast',
        messages=[ChatMessage(role='user', content='plan this')],
        prompt_version='v1',
    )
    with pytest.raises(LLMUnavailable) as exc_info:
        await gateway.complete_text(req)
    assert exc_info.value.retry_after == 25.0


@pytest.mark.asyncio
async def test_gateway_complete_text_content_blocked_no_fallback() -> None:
    groq = FakeClient('groq', [LLMContentBlocked('unsafe content')])
    gemini = FakeClient('gemini', ['Should not be reached'])
    settings = Settings(llm_fast_chain='groq,gemini')

    gateway = DefaultLLMGateway(
        providers={'groq': groq, 'gemini': gemini},
        settings=settings,
    )

    req = LLMRequest(
        purpose='extract',
        tier='fast',
        messages=[ChatMessage(role='user', content='unsafe')],
        prompt_version='v1',
    )
    with pytest.raises(LLMContentBlocked):
        await gateway.complete_text(req)
    assert gemini.call_count == 0


@pytest.mark.asyncio
async def test_gateway_json_strips_code_fences() -> None:
    fenced_json = '```json\n{"name": "Alice", "count": 3}\n```'
    groq = FakeClient('groq', [fenced_json])
    settings = Settings(llm_fast_chain='groq')

    gateway = DefaultLLMGateway(
        providers={'groq': groq},
        settings=settings,
    )

    req = LLMRequest(
        purpose='extract',
        tier='fast',
        messages=[ChatMessage(role='user', content='extract')],
        prompt_version='v1',
    )
    result = await gateway.complete_json(req, SampleModel)
    assert result.name == 'Alice'
    assert result.count == 3


@pytest.mark.asyncio
async def test_gateway_json_repair_retry_success() -> None:
    # 1st response: invalid JSON
    # 2nd response (repair call): valid JSON
    groq = FakeClient('groq', ['invalid { json', '{"name": "Bob", "count": 7}'])
    settings = Settings(llm_fast_chain='groq')

    gateway = DefaultLLMGateway(
        providers={'groq': groq},
        settings=settings,
    )

    req = LLMRequest(
        purpose='extract',
        tier='fast',
        messages=[ChatMessage(role='user', content='extract')],
        prompt_version='v1',
    )
    result = await gateway.complete_json(req, SampleModel)
    assert result.name == 'Bob'
    assert result.count == 7
    assert groq.call_count == 2


@pytest.mark.asyncio
async def test_gateway_json_repair_fails_raises_schema_violation() -> None:
    # 1st response: invalid JSON
    # 2nd response (repair): still invalid
    groq = FakeClient('groq', ['invalid json 1', 'still invalid json 2'])
    settings = Settings(llm_fast_chain='groq')

    gateway = DefaultLLMGateway(
        providers={'groq': groq},
        settings=settings,
    )

    req = LLMRequest(
        purpose='extract',
        tier='fast',
        messages=[ChatMessage(role='user', content='extract')],
        prompt_version='v1',
    )
    with pytest.raises(SchemaViolation):
        await gateway.complete_json(req, SampleModel)


@pytest.mark.asyncio
async def test_gateway_stream_text_fallback_before_first_chunk() -> None:
    groq = FakeClient('groq', [LLMTimeout('timeout on start')])
    gemini = FakeClient('gemini', [['Chunk 1', ' Chunk 2']])
    settings = Settings(llm_synth_chain='groq,gemini')

    gateway = DefaultLLMGateway(
        providers={'groq': groq, 'gemini': gemini},
        settings=settings,
    )

    req = LLMRequest(
        purpose='synth',
        tier='synth',
        messages=[ChatMessage(role='user', content='synth')],
        prompt_version='v1',
    )
    stream = gateway.stream_text(req)
    chunks = [c async for c in stream]
    assert chunks == ['Chunk 1', ' Chunk 2']


@pytest.mark.asyncio
async def test_gateway_stream_text_fails_after_first_chunk_raises_stream_interrupted() -> None:
    # First chunk yielded, then RuntimeError raised during stream
    groq = FakeClient('groq', [['First chunk', RuntimeError('Socket broken')]])
    gemini = FakeClient('gemini', [['Fallback should not be called']])
    settings = Settings(llm_synth_chain='groq,gemini')

    gateway = DefaultLLMGateway(
        providers={'groq': groq, 'gemini': gemini},
        settings=settings,
    )

    req = LLMRequest(
        purpose='synth',
        tier='synth',
        messages=[ChatMessage(role='user', content='synth')],
        prompt_version='v1',
    )
    stream = gateway.stream_text(req)
    with pytest.raises(StreamInterrupted):
        _ = [c async for c in stream]
    assert gemini.call_count == 0


# ==============================================================================
# 4. Embedder Tests (FastEmbedAdapter & CachedEmbedder)
# ==============================================================================


class FakeEmbeddingModel:
    """Mock fastembed model recording texts passed to embed()."""

    def __init__(self, dim: int = 768) -> None:
        self.dim = dim
        self.recorded_calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.recorded_calls.append(list(texts))
        out: list[list[float]] = []
        for i, _ in enumerate(texts):
            # Create a non-normalized vector with known pattern
            vec = [float(i + 1)] * self.dim
            out.append(vec)
        return out


@pytest.mark.asyncio
async def test_fastembed_adapter_prefixes_and_normalizes() -> None:
    fake_model = FakeEmbeddingModel(dim=768)
    adapter = FastEmbedAdapter(
        model_id='nomic-ai/nomic-embed-text-v1.5',
        dim=768,
        model=fake_model,
    )

    # 1. Documents prefix
    doc_vecs = await adapter.embed_documents(['text one', 'text two'])
    assert len(doc_vecs) == 2
    assert len(doc_vecs[0]) == 768
    # Verify prefix
    assert fake_model.recorded_calls[0] == [
        'search_document: text one',
        'search_document: text two',
    ]
    # Verify L2 normalization: sum of squares == 1.0
    norm_sq = sum(x * x for x in doc_vecs[0])
    assert math.isclose(norm_sq, 1.0, rel_tol=1e-5)

    # 2. Queries prefix
    query_vecs = await adapter.embed_queries(['query one'])
    assert len(query_vecs) == 1
    assert fake_model.recorded_calls[1] == ['search_query: query one']
    norm_sq_q = sum(x * x for x in query_vecs[0])
    assert math.isclose(norm_sq_q, 1.0, rel_tol=1e-5)


@pytest.mark.asyncio
async def test_fastembed_adapter_rejects_oversized_text() -> None:
    fake_model = FakeEmbeddingModel(dim=768)
    adapter = FastEmbedAdapter(model=fake_model)

    oversized = 'a' * 2001
    with pytest.raises(EmbeddingError) as exc_info:
        await adapter.embed_documents([oversized])
    assert 'exceeds max 2000' in str(exc_info.value)


@pytest.mark.asyncio
async def test_fastembed_adapter_rejects_dimension_mismatch() -> None:
    # Model returns dim=100 when adapter expects 768
    bad_model = FakeEmbeddingModel(dim=100)
    adapter = FastEmbedAdapter(dim=768, model=bad_model)

    with pytest.raises(EmbeddingError) as exc_info:
        await adapter.embed_documents(['hello'])
    assert 'dimension' in str(exc_info.value).lower()


class CountingEmbedder:
    """Mock Embedder tracking number of calls and texts embedded."""

    def __init__(self, dim: int = 4) -> None:
        self.model_id = 'test-embedder'
        self.dim = dim
        self.doc_call_count = 0
        self.query_call_count = 0

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.doc_call_count += 1
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        self.query_call_count += 1
        return [[0.0, 1.0, 0.0, 0.0] for _ in texts]


@pytest.mark.asyncio
async def test_cached_embedder_in_memory() -> None:
    inner = CountingEmbedder()
    cached = CachedEmbedder(inner=inner)

    assert cached.model_id == 'test-embedder'
    assert cached.dim == 4

    # First call: cache miss
    vecs1 = await cached.embed_documents(['doc A', 'doc B'])
    assert len(vecs1) == 2
    assert cached.misses == 2
    assert cached.hits == 0
    assert inner.doc_call_count == 1

    # Second call with same docs: cache hit
    vecs2 = await cached.embed_documents(['doc A', 'doc B'])
    assert vecs2 == vecs1
    assert cached.hits == 2
    assert inner.doc_call_count == 1

    # Partial cache hit
    vecs3 = await cached.embed_documents(['doc B', 'doc C'])
    assert len(vecs3) == 2
    assert cached.hits == 3
    assert cached.misses == 3
    assert inner.doc_call_count == 2

    # Query caching
    q_vecs = await cached.embed_queries(['query 1'])
    assert len(q_vecs) == 1
    assert inner.query_call_count == 1
    q_vecs2 = await cached.embed_queries(['query 1'])
    assert q_vecs2 == q_vecs
    assert inner.query_call_count == 1


@pytest.mark.asyncio
async def test_cached_embedder_sqlite(tmp_path: Path) -> None:
    db_file = tmp_path / 'cache' / 'embed_cache.db'
    inner = CountingEmbedder()

    cached1 = CachedEmbedder(inner=inner, cache_db_path=db_file)
    await cached1.embed_documents(['persistent text'])
    assert inner.doc_call_count == 1

    # New instance using the same SQLite db file
    cached2 = CachedEmbedder(inner=inner, cache_db_path=db_file)
    vecs = await cached2.embed_documents(['persistent text'])
    assert len(vecs) == 1
    assert inner.doc_call_count == 1  # Loaded from SQLite without calling inner!
    assert cached2.hits == 1

    # Clear cache
    cached2.clear()
    assert cached2.hits == 0
    await cached2.embed_documents(['persistent text'])
    assert inner.doc_call_count == 2  # Re-embedded after clear
