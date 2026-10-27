"""LLM client and gateway ports."""

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from memory_engine.domain.enums import Intent

T = TypeVar('T')


class PlannerLLMOutput(BaseModel):
    """Structured LLM output for query planning fallback per LLD Section 9.2."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    intent: Intent = Intent.OPEN
    time_phrases: list[str] = Field(default_factory=list)
    entity_names: list[str] = Field(default_factory=list)
    predicates: list[str] = Field(default_factory=list)
    multi_hop: bool = False
    rationale: str | None = None


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
