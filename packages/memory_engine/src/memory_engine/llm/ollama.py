"""Ollama local LLM provider adapter using /api/chat HTTP endpoint."""

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

from memory_engine.domain.errors import (
    LLMTimeout,
    LLMUnavailable,
)
from memory_engine.ports.llm import LLMClient, LLMRequest, LLMResponse

__all__ = ['OllamaClient']


class OllamaClient(LLMClient):
    """Ollama local chat client adapter."""

    name: str = 'ollama'

    def __init__(
        self,
        url: str = 'http://localhost:11434',
        model: str = 'llama3.1:8b',
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._url = url.rstrip('/')
        self._model = model
        self._client = client

    def _get_client(self, timeout_s: float) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        return httpx.AsyncClient(timeout=timeout_s)

    def _build_payload(self, req: LLMRequest, stream: bool = False) -> dict[str, Any]:
        messages = [{'role': m.role, 'content': m.content} for m in req.messages]
        payload: dict[str, Any] = {
            'model': self._model,
            'messages': messages,
            'stream': stream,
            'options': {
                'temperature': req.temperature,
                'num_predict': req.max_tokens,
            },
        }
        if req.json_mode:
            payload['format'] = 'json'
        return payload

    async def complete(self, req: LLMRequest) -> LLMResponse:
        """Execute chat completion request against local Ollama."""
        client = self._get_client(req.timeout_s)
        payload = self._build_payload(req, stream=False)
        url = f'{self._url}/api/chat'

        start_t = time.perf_counter()
        try:
            response = await client.post(url, json=payload, timeout=req.timeout_s)
        except httpx.TimeoutException as err:
            raise LLMTimeout(f'Ollama request timed out after {req.timeout_s}s: {err}') from err
        except httpx.RequestError as err:
            raise LLMUnavailable(f'Ollama request failed: {err}') from err

        latency_ms = (time.perf_counter() - start_t) * 1000.0

        if response.status_code == 408:
            raise LLMTimeout(f'Ollama returned 408: {response.text}')
        if response.status_code >= 500:
            raise LLMUnavailable(
                f'Ollama service unavailable: {response.status_code} {response.text}'
            )
        if response.status_code != 200:
            raise LLMUnavailable(f'Ollama error: {response.status_code} {response.text}')

        try:
            data = response.json()
            content = data['message']['content']
            prompt_tokens = data.get('prompt_eval_count', 0)
            completion_tokens = data.get('eval_count', 0)
            model_name = data.get('model', self._model)
        except Exception as err:
            raise LLMUnavailable(f'Failed to parse Ollama response: {err}') from err

        return LLMResponse(
            text=content,
            model=model_name,
            provider=self.name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )

    async def stream(self, req: LLMRequest) -> AsyncIterator[str]:
        """Stream chat completion chunks from Ollama."""
        client = self._get_client(req.timeout_s)
        payload = self._build_payload(req, stream=True)
        url = f'{self._url}/api/chat'

        try:
            async with client.stream('POST', url, json=payload, timeout=req.timeout_s) as response:
                if response.status_code == 408:
                    raise LLMTimeout('Ollama stream timed out')
                if response.status_code >= 500:
                    raise LLMUnavailable(f'Ollama stream unavailable: {response.status_code}')
                if response.status_code != 200:
                    raise LLMUnavailable(f'Ollama stream error: {response.status_code}')

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        content = chunk.get('message', {}).get('content')
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue
        except httpx.TimeoutException as err:
            raise LLMTimeout(f'Ollama stream timed out: {err}') from err
        except httpx.RequestError as err:
            raise LLMUnavailable(f'Ollama stream network error: {err}') from err
