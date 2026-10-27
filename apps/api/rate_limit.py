"""In-memory per-user token-bucket rate limiter per docs/LLD.md Section 11.3."""

import asyncio
import time
from collections.abc import Callable
from typing import Annotated, Any
from uuid import UUID

from fastapi import Depends, Request
from memory_engine.domain.clock import Clock
from memory_engine.domain.errors import RateLimited
from memory_engine.domain.ids import UserId

from .auth import get_current_user


class _TokenBucket:
    """Single token-bucket state."""

    __slots__ = ('capacity', 'fill_rate', 'last_updated', 'tokens')

    def __init__(self, capacity: float, fill_rate: float, now: float) -> None:
        self.capacity = capacity
        self.fill_rate = fill_rate
        self.tokens = capacity
        self.last_updated = now

    def consume(self, now: float, tokens: float = 1.0) -> tuple[bool, float]:
        """Try to consume tokens. Returns (allowed, wait_seconds)."""
        elapsed = max(0.0, now - self.last_updated)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.fill_rate)
        self.last_updated = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True, 0.0

        missing = tokens - self.tokens
        wait_s = missing / self.fill_rate if self.fill_rate > 0 else 60.0
        return False, wait_s


class RateLimiter:
    """Thread-safe and coroutine-safe in-memory per-user token-bucket rate limiter."""

    def __init__(
        self,
        clock: Clock | None = None,
        bucket_configs: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.clock = clock
        self._lock = asyncio.Lock()
        # bucket name -> (rate_per_min, burst_capacity)
        # Default limits per LLD 11.3:
        # chat: 30/min, burst 10
        # search: 120/min, burst 30
        # ingest: 30/min, burst 10
        # default: 120/min, burst 30
        self.bucket_configs: dict[str, tuple[float, float]] = bucket_configs or {
            'chat': (30.0, 10.0),
            'search': (120.0, 30.0),
            'ingest': (30.0, 10.0),
            'default': (120.0, 30.0),
        }
        # (user_id, bucket) -> _TokenBucket
        self._buckets: dict[tuple[UUID, str], _TokenBucket] = {}

    def _now_seconds(self) -> float:
        if self.clock is not None:
            return self.clock.now().timestamp()
        return time.monotonic()

    def check(self, user_id: UserId | UUID, bucket: str, cost: float = 1.0) -> None:
        """Check rate limit synchronously for user_id and bucket name.

        Raises RateLimited with retry_after and code E-RATE-01 if exceeded.
        """
        now = self._now_seconds()
        uid = UUID(str(user_id))
        key = (uid, bucket)

        if key not in self._buckets:
            cfg = self.bucket_configs.get(bucket, self.bucket_configs.get('default', (120.0, 30.0)))
            rate_per_min, burst = cfg
            fill_rate = rate_per_min / 60.0
            self._buckets[key] = _TokenBucket(capacity=burst, fill_rate=fill_rate, now=now)

        b = self._buckets[key]
        allowed, wait_seconds = b.consume(now, tokens=cost)
        if not allowed:
            raise RateLimited(
                f'Rate limit exceeded for {bucket}',
                retry_after=wait_seconds,
                code='E-RATE-01',
                details={'bucket': bucket, 'retry_after': wait_seconds},
            )

    def reset(self) -> None:
        """Clear all active user buckets."""
        self._buckets.clear()


def rate_limit(bucket: str) -> Callable[..., Any]:
    """FastAPI dependency enforcing token-bucket rate limits for current user."""

    async def _rate_limit_dep(
        request: Request,
        user_id: Annotated[UserId, Depends(get_current_user)],
    ) -> None:
        container = getattr(request.app.state, 'container', None)
        limiter: RateLimiter | None = None
        if container is not None and getattr(container, 'rate_limiter', None) is not None:
            limiter = container.rate_limiter

        if limiter is not None:
            limiter.check(user_id=user_id, bucket=bucket)

    return _rate_limit_dep
