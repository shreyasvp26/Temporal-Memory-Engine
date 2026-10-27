"""LLM client and gateway ports."""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar

T = TypeVar('T')


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: Literal['system', 'user', 'assistant']
    content: str


@dataclass(frozen=True, slots=True)
class LLMRequest:
    purpose: Literal['extract', 'plan', 'judge', 'entity_match', 'synth']
    tier: Literal['fast', 'synth']
    messages: Sequence[ChatMessage]
    prompt_version: str
    temperature: float = 0.0
    max_tokens: int = 1024
    json_mode: bool = False
    priority: Literal['interactive', 'background'] = 'background'
    timeout_s: float = 30.0


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class LLMClient(Protocol):
    name: str

    async def complete(self, req: LLMRequest) -> LLMResponse: ...

    def stream(self, req: LLMRequest) -> AsyncIterator[str]: ...


class LLMGateway(Protocol):
    async def complete_text(self, req: LLMRequest) -> LLMResponse: ...

    async def complete_json(self, req: LLMRequest, schema: type[T]) -> T: ...

    def stream_text(self, req: LLMRequest) -> AsyncIterator[str]: ...
