"""Default LLM Gateway implementing routing, rate limiting, circuit breaker and repair."""

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, TypeVar, cast

from memory_engine.config import Settings
from memory_engine.domain.errors import (
    LLMContentBlocked,
    LLMRateLimited,
    LLMTimeout,
    LLMUnavailable,
    SchemaViolation,
    StreamInterrupted,
)
from memory_engine.llm.prompts import (
    PROMPT_VERSION_REPAIR,
    REPAIR_SYSTEM_PROMPT,
    build_repair_user_prompt,
)
from memory_engine.obs.logging import get_logger
from memory_engine.ports.clock import Clock, SystemClock
from memory_engine.ports.llm import (
    ChatMessage,
    LLMClient,
    LLMGateway,
    LLMRequest,
    LLMResponse,
)

logger = get_logger(__name__)

T = TypeVar('T')


def _strip_code_fences(text: str) -> str:
    """Strip markdown code block fences and whitespace from model response."""
    text = text.strip()
    if text.startswith('```'):
        lines = text.splitlines()
        if lines and lines[0].startswith('```'):
            lines = lines[1:]
        if lines and lines[-1].strip() == '```':
            lines = lines[:-1]
        text = '\n'.join(lines).strip()
    return text


@dataclass
class _CircuitBreaker:
    failure_timestamps: list[float]
    state: str  # 'CLOSED', 'OPEN', 'HALF_OPEN'
    opened_at: float = 0.0

    def can_attempt(self, now: float) -> bool:
        if self.state == 'CLOSED':
            return True
        if self.state == 'OPEN':
            if now - self.opened_at >= 30.0:
                self.state = 'HALF_OPEN'
                return True
            return False
        # HALF_OPEN allows one probe
        return True

    def record_success(self) -> None:
        self.state = 'CLOSED'
        self.failure_timestamps.clear()

    def record_failure(self, now: float) -> None:
        self.failure_timestamps = [t for t in self.failure_timestamps if now - t <= 60.0]
        self.failure_timestamps.append(now)
        if self.state == 'HALF_OPEN' or len(self.failure_timestamps) >= 5:
            self.state = 'OPEN'
            self.opened_at = now


class _TokenBucket:
    def __init__(self, rpm: int, tpm: int, rpd: int) -> None:
        self.rpm = rpm
        self.tpm = tpm
        self.rpd = rpd
        self.blocked_until: float = 0.0
        self._req_times: list[float] = []
        self._tok_records: list[tuple[float, int]] = []

    async def acquire(
        self, estimated_tokens: int, priority: str, max_wait_s: float, now: float
    ) -> None:
        # Ollama / unbounded providers
        if self.rpm <= 0 and self.tpm <= 0 and self.rpd <= 0:
            return

        wait_s = 0.0
        if self.blocked_until > now:
            wait_s = max(wait_s, self.blocked_until - now)

        # Cleanup records older than 86400s
        self._req_times = [t for t in self._req_times if now - t < 86400.0]
        self._tok_records = [rec for rec in self._tok_records if now - rec[0] < 60.0]

        factor = 0.7 if priority == 'background' else 1.0
        rpm_limit = max(1, int(self.rpm * factor)) if self.rpm > 0 else 999999
        tpm_limit = max(1, int(self.tpm * factor)) if self.tpm > 0 else 999999999
        rpd_limit = max(1, int(self.rpd * factor)) if self.rpd > 0 else 999999999

        # Minute requests
        recent_reqs = [t for t in self._req_times if now - t < 60.0]
        if len(recent_reqs) >= rpm_limit:
            wait_s = max(wait_s, 60.0 - (now - recent_reqs[0]))

        # Daily requests
        if len(self._req_times) >= rpd_limit:
            wait_s = max(wait_s, 86400.0 - (now - self._req_times[0]))

        # Minute tokens
        recent_tokens = sum(rec[1] for rec in self._tok_records)
        if recent_tokens + estimated_tokens > tpm_limit and self._tok_records:
            wait_s = max(wait_s, 60.0 - (now - self._tok_records[0][0]))

        if wait_s > max_wait_s:
            raise LLMRateLimited(
                f'Token bucket wait of {wait_s:.1f}s exceeds max_wait_s={max_wait_s}s',
                retry_after=wait_s,
            )

        if wait_s > 0:
            await asyncio.sleep(wait_s)
            now += wait_s

        self._req_times.append(now)
        self._tok_records.append((now, estimated_tokens))

    def set_blocked(self, blocked_until: float) -> None:
        self.blocked_until = max(self.blocked_until, blocked_until)


def _sanitize_facts_data(data: Any) -> Any:
    """Sanitize extraction facts data to handle LLM quirks like speaker vs subject."""
    if isinstance(data, list):
        data = {'facts': data}
    if not isinstance(data, dict) or 'facts' not in data or not isinstance(data['facts'], list):
        return data

    from memory_engine.domain.models import ExtractedFact

    allowed_fields = set(ExtractedFact.model_fields.keys())
    cleaned_facts: list[dict[str, Any]] = []

    for item in data['facts']:
        if not isinstance(item, dict):
            continue

        item_dict = dict(item)

        # If subject is missing or empty and speaker is present, rename speaker to subject
        subj = item_dict.get('subject')
        if (subj is None or not str(subj).strip()) and 'speaker' in item_dict:
            item_dict['subject'] = str(item_dict.pop('speaker')).strip()
        else:
            item_dict.pop('speaker', None)

        # Pop any unknown keys not in ExtractedFact
        for k in list(item_dict.keys()):
            if k not in allowed_fields:
                item_dict.pop(k, None)

        # Filter out if subject, predicate, object, or evidence_span is None or empty after
        # stripping
        is_valid = True
        for field in ('subject', 'predicate', 'object', 'evidence_span'):
            val = item_dict.get(field)
            if val is None or not str(val).strip():
                is_valid = False
                break
        if not is_valid:
            continue

        # Standard string clean and defaults
        item_dict['subject'] = str(item_dict['subject']).strip()
        item_dict['predicate'] = str(item_dict['predicate']).strip()
        item_dict['object'] = str(item_dict['object']).strip()
        item_dict['evidence_span'] = str(item_dict['evidence_span']).strip()

        item_dict.setdefault('confidence', 1.0)
        item_dict.setdefault('subject_type', 'person')
        item_dict.setdefault('object_type', 'literal')
        item_dict.setdefault('is_retraction', False)
        item_dict.setdefault('valid_from_text', None)
        item_dict.setdefault('valid_to_text', None)
        item_dict.setdefault('source_turn', 0)

        cleaned_facts.append(item_dict)

    data['facts'] = cleaned_facts
    return data


class DefaultLLMGateway(LLMGateway):
    """Default multi-provider LLM gateway with fallback, rate limiting, and repair."""

    def __init__(
        self,
        providers: Mapping[str, LLMClient],
        settings: Settings | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._providers = dict(providers)
        self._settings = settings or Settings()
        self._clock = clock or SystemClock()

        # Build chains
        self._fast_chain = [
            p.strip() for p in self._settings.llm_fast_chain.split(',') if p.strip()
        ]
        self._synth_chain = [
            p.strip() for p in self._settings.llm_synth_chain.split(',') if p.strip()
        ]

        # Rate buckets and circuit breakers per provider
        self._breakers: dict[str, _CircuitBreaker] = {
            name: _CircuitBreaker(failure_timestamps=[], state='CLOSED') for name in self._providers
        }
        self._buckets: dict[str, _TokenBucket] = {}
        for name in self._providers:
            if name == 'groq':
                self._buckets[name] = _TokenBucket(
                    rpm=self._settings.rate_groq_rpm,
                    tpm=self._settings.rate_groq_tpm,
                    rpd=self._settings.rate_groq_rpd,
                )
            elif name == 'gemini':
                self._buckets[name] = _TokenBucket(
                    rpm=self._settings.rate_gemini_rpm,
                    tpm=self._settings.rate_gemini_tpm,
                    rpd=self._settings.rate_gemini_rpd,
                )
            else:
                self._buckets[name] = _TokenBucket(rpm=0, tpm=0, rpd=0)

    def _now_ts(self) -> float:
        return self._clock.now().timestamp()

    def _get_chain(self, tier: str) -> list[str]:
        chain = self._synth_chain if tier == 'synth' else self._fast_chain
        return [p for p in chain if p in self._providers]

    def _estimate_tokens(self, req: LLMRequest) -> int:
        total_chars = sum(len(m.content) for m in req.messages)
        return max(1, total_chars // 4)

    async def _before_call(self, provider_name: str, req: LLMRequest) -> None:
        now = self._now_ts()
        breaker = self._breakers.get(provider_name)
        if breaker and not breaker.can_attempt(now):
            raise LLMUnavailable(f'Circuit open for provider {provider_name}')

        bucket = self._buckets.get(provider_name)
        if bucket:
            max_wait_s = 2.0 if req.priority == 'interactive' else 120.0
            est_tokens = self._estimate_tokens(req)
            await bucket.acquire(est_tokens, req.priority, max_wait_s, now)

    def _record_success(self, provider_name: str) -> None:
        breaker = self._breakers.get(provider_name)
        if breaker:
            breaker.record_success()

    def _record_failure(self, provider_name: str, err: Exception) -> None:
        now = self._now_ts()
        if isinstance(err, LLMRateLimited):
            bucket = self._buckets.get(provider_name)
            if bucket:
                retry_after = err.retry_after if err.retry_after is not None else 1.0
                bucket.set_blocked(now + retry_after)
            return

        if isinstance(err, (LLMTimeout, LLMUnavailable)):
            breaker = self._breakers.get(provider_name)
            if breaker:
                breaker.record_failure(now)

    async def complete_text(self, req: LLMRequest) -> LLMResponse:
        """Complete text with fallback across configured provider chain."""
        chain = self._get_chain(req.tier)
        if not chain:
            raise LLMUnavailable('No available LLM providers configured in chain')

        max_retry_after: float | None = None
        last_error: Exception | None = None

        for provider_name in chain:
            client = self._providers[provider_name]
            try:
                await self._before_call(provider_name, req)
                resp = await client.complete(req)
                self._record_success(provider_name)
                return resp
            except LLMContentBlocked:
                # Content blocked errors are terminal; never fall back
                raise
            except (LLMRateLimited, LLMTimeout, LLMUnavailable) as err:
                last_error = err
                self._record_failure(provider_name, err)
                if (
                    isinstance(err, (LLMRateLimited, LLMUnavailable))
                    and err.retry_after
                    and (max_retry_after is None or err.retry_after > max_retry_after)
                ):
                    max_retry_after = err.retry_after
                logger.warning(
                    'llm_provider_fallback',
                    provider=provider_name,
                    error=str(err),
                    tier=req.tier,
                )
                continue

        raise LLMUnavailable(
            f'All providers in chain {chain} failed: {last_error}',
            retry_after=max_retry_after,
        )

    async def complete_json(self, req: LLMRequest, schema: type[T]) -> T:
        """Complete JSON output with schema validation and 1 repair retry per provider."""
        chain = self._get_chain(req.tier)
        if not chain:
            raise LLMUnavailable('No available LLM providers configured in chain')

        # Ensure json mode is set
        req_json = LLMRequest(
            purpose=req.purpose,
            tier=req.tier,
            messages=req.messages,
            prompt_version=req.prompt_version,
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            json_mode=True,
            priority=req.priority,
            timeout_s=req.timeout_s,
        )

        max_retry_after: float | None = None
        last_validation_error: str | None = None
        all_failed_transient = True

        schema_json_str: str | None = None
        schema_obj: Any = schema
        if hasattr(schema_obj, 'model_json_schema'):
            try:
                schema_json_str = json.dumps(schema_obj.model_json_schema(), indent=2)
            except Exception:
                schema_json_str = None

        for provider_name in chain:
            client = self._providers[provider_name]
            try:
                await self._before_call(provider_name, req_json)
                resp = await client.complete(req_json)
                self._record_success(provider_name)
                all_failed_transient = False
            except LLMContentBlocked:
                raise
            except (LLMRateLimited, LLMTimeout, LLMUnavailable) as err:
                self._record_failure(provider_name, err)
                if (
                    isinstance(err, (LLMRateLimited, LLMUnavailable))
                    and err.retry_after
                    and (max_retry_after is None or err.retry_after > max_retry_after)
                ):
                    max_retry_after = err.retry_after
                logger.warning(
                    'llm_json_provider_failed',
                    provider=provider_name,
                    error=str(err),
                )
                continue

            # Try parsing initial response
            raw_text = resp.text
            clean_text = _strip_code_fences(raw_text)
            parsed_obj, val_err = self._validate_json(clean_text, schema)
            if parsed_obj is not None:
                return parsed_obj

            last_validation_error = val_err or 'Invalid JSON structure'
            logger.warning(
                'llm_json_validation_failed_attempting_repair',
                provider=provider_name,
                error=last_validation_error,
            )

            # Attempt 1 repair call on the same provider
            repair_prompt = build_repair_user_prompt(
                invalid_text=clean_text,
                error_message=last_validation_error,
                schema_json=schema_json_str,
            )
            repair_messages = [
                ChatMessage(role='system', content=REPAIR_SYSTEM_PROMPT),
                ChatMessage(role='user', content=repair_prompt),
            ]
            repair_req = LLMRequest(
                purpose=req.purpose,
                tier=req.tier,
                messages=repair_messages,
                prompt_version=PROMPT_VERSION_REPAIR,
                temperature=0.0,
                max_tokens=req.max_tokens,
                json_mode=True,
                priority=req.priority,
                timeout_s=req.timeout_s,
            )

            try:
                await self._before_call(provider_name, repair_req)
                repair_resp = await client.complete(repair_req)
                self._record_success(provider_name)
                clean_repair = _strip_code_fences(repair_resp.text)
                repaired_obj, repair_val_err = self._validate_json(clean_repair, schema)
                if repaired_obj is not None:
                    return repaired_obj
                last_validation_error = repair_val_err or 'Repair output violated schema'
            except LLMContentBlocked:
                raise
            except (LLMRateLimited, LLMTimeout, LLMUnavailable) as err:
                self._record_failure(provider_name, err)
                logger.warning(
                    'llm_json_repair_call_failed',
                    provider=provider_name,
                    error=str(err),
                )

        if all_failed_transient and max_retry_after is not None:
            raise LLMUnavailable(
                'All LLM providers unavailable for JSON completion',
                retry_after=max_retry_after,
            )

        raise SchemaViolation(
            f'LLM output violated schema after repair and fallbacks: {last_validation_error}'
        )

    def _validate_json(self, text: str, schema: type[T]) -> tuple[T | None, str | None]:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as err:
            return None, f'JSON parse error: {err}'

        data = _sanitize_facts_data(data)

        schema_obj: Any = schema
        if hasattr(schema_obj, 'model_fields') and isinstance(data, dict):
            allowed_fields = set(schema_obj.model_fields.keys())
            data = {k: v for k, v in data.items() if k in allowed_fields}

        if hasattr(schema_obj, 'model_validate'):
            try:
                validated = schema_obj.model_validate(data)
                return cast('T', validated), None
            except Exception as err:
                return None, f'Pydantic validation error: {err}'

        return cast('T', data), None

    _parse_and_validate = _validate_json

    async def stream_text(self, req: LLMRequest) -> AsyncIterator[str]:
        """Stream completion text with fallback before the first yielded chunk."""
        chain = self._get_chain(req.tier)
        if not chain:
            raise LLMUnavailable('No available LLM providers configured in chain')

        max_retry_after: float | None = None
        last_error: Exception | None = None

        for provider_name in chain:
            client = self._providers[provider_name]
            first_chunk_yielded = False
            try:
                await self._before_call(provider_name, req)
                stream_iter = client.stream(req)
                self._record_success(provider_name)

                async for chunk in stream_iter:
                    first_chunk_yielded = True
                    yield chunk
                return
            except LLMContentBlocked:
                raise
            except (LLMRateLimited, LLMTimeout, LLMUnavailable, Exception) as err:
                if first_chunk_yielded:
                    # After first chunk has been yielded, fallback is forbidden
                    raise StreamInterrupted(f'LLM stream interrupted: {err}') from err

                last_error = err
                self._record_failure(provider_name, err)
                if (
                    isinstance(err, (LLMRateLimited, LLMUnavailable))
                    and err.retry_after
                    and (max_retry_after is None or err.retry_after > max_retry_after)
                ):
                    max_retry_after = err.retry_after
                logger.warning(
                    'llm_stream_provider_fallback',
                    provider=provider_name,
                    error=str(err),
                )
                continue

        raise LLMUnavailable(
            f'All providers in chain failed before stream started: {last_error}',
            retry_after=max_retry_after,
        )


__all__ = ['DefaultLLMGateway']
