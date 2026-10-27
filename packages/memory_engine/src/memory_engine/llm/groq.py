"""Groq LLM provider adapter using OpenAI-compatible HTTP chat completions."""

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from memory_engine.domain.errors import (
    LLMContentBlocked,
    LLMRateLimited,
    LLMTimeout,
    LLMUnavailable,
)
from memory_engine.ports.llm import LLMClient, LLMRequest, LLMResponse

__all__ = ['GroqClient']


def _parse_retry_after(headers: httpx.Headers) -> float | None:
    val = headers.get('retry-after')
    if not val:
        val = headers.get('x-ratelimit-reset-requests') or headers.get('x-ratelimit-reset-tokens')
    if val:
        try:
            return float(val)
        except ValueError:
            return None
    return None


class GroqClient(LLMClient):
    """Groq LLM client communicating over OpenAI-compatible chat completions."""

    name: str = 'groq'

    def __init__(
        self,
        api_key: str | None = None,
        model_fast: str = 'llama-3.1-8b-instant',
        base_url: str = 'https://api.groq.com/openai/v1',
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key or ''
        self._model_fast = model_fast
        self._base_url = base_url.rstrip('/')
        self._client = client

    def _get_client(self, timeout_s: float) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        return httpx.AsyncClient(timeout=timeout_s)

    def _build_payload(self, req: LLMRequest, stream: bool = False) -> dict[str, Any]:
        messages = [{'role': m.role, 'content': m.content} for m in req.messages]
        payload: dict[str, Any] = {
            'model': self._model_fast,
            'messages': messages,
            'temperature': req.temperature,
            'max_tokens': req.max_tokens,
            'stream': stream,
        }
        if req.json_mode:
            payload['response_format'] = {'type': 'json_object'}
        return payload

    def _build_headers(self) -> dict[str, str]:
        headers = {'Content-Type': 'application/json'}
        if self._api_key:
            headers['Authorization'] = f'Bearer {self._api_key}'
        return headers

    async def complete(self, req: LLMRequest) -> LLMResponse:
        """Execute chat completion request against Groq."""
        client = self._get_client(req.timeout_s)
        payload = self._build_payload(req, stream=False)
        headers = self._build_headers()
        url = f'{self._base_url}/chat/completions'

        start_t = time.perf_counter()
        try:
            response = await client.post(url, json=payload, headers=headers, timeout=req.timeout_s)
        except httpx.TimeoutException as err:
            raise LLMTimeout(f'Groq request timed out after {req.timeout_s}s: {err}') from err
        except httpx.RequestError as err:
            raise LLMUnavailable(f'Groq request failed: {err}') from err

        latency_ms = (time.perf_counter() - start_t) * 1000.0

        if response.status_code == 429:
            retry_after = _parse_retry_after(response.headers)
            raise LLMRateLimited(
                f'Groq rate limited: {response.text}',
                retry_after=retry_after,
            )
        if response.status_code == 408:
            raise LLMTimeout(f'Groq returned 408: {response.text}')
        if response.status_code == 400:
            text_lower = response.text.lower()
            if any(k in text_lower for k in ('content', 'moderation', 'safety', 'blocked')):
                raise LLMContentBlocked(f'Groq content blocked: {response.text}')
            raise LLMUnavailable(f'Groq client error: {response.status_code} {response.text}')
        if response.status_code >= 500:
            raise LLMUnavailable(
                f'Groq service unavailable: {response.status_code} {response.text}'
            )
        if response.status_code != 200:
            raise LLMUnavailable(f'Groq error: {response.status_code} {response.text}')

        try:
            data = response.json()
            content = data['choices'][0]['message']['content']
            usage = data.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', 0)
            completion_tokens = usage.get('completion_tokens', 0)
            model_name = data.get('model', self._model_fast)
        except Exception as err:
            raise LLMUnavailable(f'Failed to parse Groq response: {err}') from err

        return LLMResponse(
            text=content,
            model=model_name,
            provider=self.name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )

    async def stream(self, req: LLMRequest) -> AsyncIterator[str]:
        """Stream completion chunks from Groq."""
        client = self._get_client(req.timeout_s)
        payload = self._build_payload(req, stream=True)
        headers = self._build_headers()
        url = f'{self._base_url}/chat/completions'

        try:
            async with client.stream(
                'POST', url, json=payload, headers=headers, timeout=req.timeout_s
            ) as response:
                if response.status_code == 429:
                    retry_after = _parse_retry_after(response.headers)
                    raise LLMRateLimited(
                        f'Groq rate limited: {response.status_code}',
                        retry_after=retry_after,
                    )
                if response.status_code == 408:
                    raise LLMTimeout('Groq stream request timed out (408)')
                if response.status_code >= 500:
                    raise LLMUnavailable(f'Groq service unavailable: {response.status_code}')
                if response.status_code != 200:
                    raise LLMUnavailable(f'Groq stream error: {response.status_code}')

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith('data: '):
                        data_str = line[len('data: ') :].strip()
                        if data_str == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk.get('choices', [{}])[0].get('delta', {})
                            text = delta.get('content')
                            if text:
                                yield text
                        except json.JSONDecodeError:
                            continue
        except httpx.TimeoutException as err:
            raise LLMTimeout(f'Groq stream timed out: {err}') from err
        except httpx.RequestError as err:
            raise LLMUnavailable(f'Groq stream network error: {err}') from err
