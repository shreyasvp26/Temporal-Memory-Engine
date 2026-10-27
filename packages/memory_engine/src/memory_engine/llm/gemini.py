"""Google Gemini LLM provider adapter using REST generateContent."""

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

__all__ = ['GeminiClient']


def _parse_retry_after(headers: httpx.Headers) -> float | None:
    val = headers.get('retry-after')
    if val:
        try:
            return float(val)
        except ValueError:
            return None
    return None


class GeminiClient(LLMClient):
    """Google Gemini LLM adapter communicating over generateContent REST API."""

    name: str = 'gemini'

    def __init__(
        self,
        api_key: str | None = None,
        model_fast: str = 'gemini-1.5-flash',
        model_synth: str = 'gemini-1.5-flash',
        base_url: str = 'https://generativelanguage.googleapis.com/v1beta',
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key or ''
        self._model_fast = model_fast
        self._model_synth = model_synth
        self._base_url = base_url.rstrip('/')
        self._client = client

    def _get_client(self, timeout_s: float) -> httpx.AsyncClient:
        if self._client is not None:
            return self._client
        return httpx.AsyncClient(timeout=timeout_s)

    def _select_model(self, req: LLMRequest) -> str:
        if req.tier == 'synth':
            return self._model_synth
        return self._model_fast

    def _build_payload(self, req: LLMRequest) -> dict[str, Any]:
        system_parts: list[str] = []
        contents: list[dict[str, Any]] = []

        for m in req.messages:
            if m.role == 'system':
                system_parts.append(m.content)
            else:
                role = 'user' if m.role == 'user' else 'model'
                contents.append({'role': role, 'parts': [{'text': m.content}]})

        body: dict[str, Any] = {'contents': contents}
        if system_parts:
            body['system_instruction'] = {'parts': [{'text': '\n\n'.join(system_parts)}]}

        gen_config: dict[str, Any] = {
            'temperature': req.temperature,
            'maxOutputTokens': req.max_tokens,
        }
        if req.json_mode:
            gen_config['responseMimeType'] = 'application/json'
        body['generationConfig'] = gen_config
        return body

    def _build_headers(self) -> dict[str, str]:
        headers = {'Content-Type': 'application/json'}
        if self._api_key:
            headers['x-goog-api-key'] = self._api_key
        return headers

    async def complete(self, req: LLMRequest) -> LLMResponse:
        """Execute generateContent against Gemini."""
        client = self._get_client(req.timeout_s)
        model = self._select_model(req)
        payload = self._build_payload(req)
        headers = self._build_headers()
        url = f'{self._base_url}/models/{model}:generateContent'

        start_t = time.perf_counter()
        try:
            response = await client.post(url, json=payload, headers=headers, timeout=req.timeout_s)
        except httpx.TimeoutException as err:
            raise LLMTimeout(f'Gemini request timed out after {req.timeout_s}s: {err}') from err
        except httpx.RequestError as err:
            raise LLMUnavailable(f'Gemini request failed: {err}') from err

        latency_ms = (time.perf_counter() - start_t) * 1000.0

        if response.status_code == 429:
            retry_after = _parse_retry_after(response.headers)
            raise LLMRateLimited(
                f'Gemini rate limited: {response.text}',
                retry_after=retry_after,
            )
        if response.status_code == 408:
            raise LLMTimeout(f'Gemini returned 408: {response.text}')
        if response.status_code == 400:
            text_lower = response.text.lower()
            if any(k in text_lower for k in ('safety', 'blocked', 'prohibited', 'content')):
                raise LLMContentBlocked(f'Gemini content blocked: {response.text}')
            raise LLMUnavailable(f'Gemini client error: {response.status_code} {response.text}')
        if response.status_code >= 500:
            raise LLMUnavailable(
                f'Gemini service unavailable: {response.status_code} {response.text}'
            )
        if response.status_code != 200:
            raise LLMUnavailable(f'Gemini error: {response.status_code} {response.text}')

        try:
            data = response.json()
            candidates = data.get('candidates', [])
            if not candidates:
                raise LLMUnavailable('Gemini response contained no candidates')

            cand = candidates[0]
            finish_reason = cand.get('finishReason', '')
            if finish_reason in ('SAFETY', 'BLOCKLIST', 'PROHIBITED_CONTENT'):
                raise LLMContentBlocked(f'Gemini generation blocked: {finish_reason}')

            parts = cand.get('content', {}).get('parts', [])
            content = ''.join(p.get('text', '') for p in parts)
            usage = data.get('usageMetadata', {})
            prompt_tokens = usage.get('promptTokenCount', 0)
            completion_tokens = usage.get('candidatesTokenCount', 0)
        except (LLMContentBlocked, LLMUnavailable):
            raise
        except Exception as err:
            raise LLMUnavailable(f'Failed to parse Gemini response: {err}') from err

        return LLMResponse(
            text=content,
            model=model,
            provider=self.name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )

    async def stream(self, req: LLMRequest) -> AsyncIterator[str]:
        """Stream chunks from Gemini via streamGenerateContent."""
        client = self._get_client(req.timeout_s)
        model = self._select_model(req)
        payload = self._build_payload(req)
        headers = self._build_headers()
        url = f'{self._base_url}/models/{model}:streamGenerateContent?alt=sse'

        try:
            async with client.stream(
                'POST', url, json=payload, headers=headers, timeout=req.timeout_s
            ) as response:
                if response.status_code == 429:
                    retry_after = _parse_retry_after(response.headers)
                    raise LLMRateLimited(
                        f'Gemini rate limited: {response.status_code}',
                        retry_after=retry_after,
                    )
                if response.status_code == 408:
                    raise LLMTimeout('Gemini stream request timed out')
                if response.status_code >= 500:
                    raise LLMUnavailable(f'Gemini stream unavailable: {response.status_code}')
                if response.status_code != 200:
                    raise LLMUnavailable(f'Gemini stream error: {response.status_code}')

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
                            cands = chunk.get('candidates', [])
                            if cands:
                                parts = cands[0].get('content', {}).get('parts', [])
                                for p in parts:
                                    txt = p.get('text')
                                    if txt:
                                        yield txt
                        except json.JSONDecodeError:
                            continue
        except httpx.TimeoutException as err:
            raise LLMTimeout(f'Gemini stream timed out: {err}') from err
        except httpx.RequestError as err:
            raise LLMUnavailable(f'Gemini stream network error: {err}') from err
