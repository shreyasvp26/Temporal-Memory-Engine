"""Pydantic schemas for structured LLM outputs and repair builder."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from memory_engine.domain.models import ExtractedFact, ExtractionResult
from memory_engine.ports.llm import PlannerLLMOutput

__all__ = [
    'ExtractedFact',
    'ExtractionResult',
    'JudgeLLMOutput',
    'PlannerLLMOutput',
    'RepairPromptBuilder',
]


class JudgeLLMOutput(BaseModel):
    """Structured LLM output for ambiguity/conflict judge per LLD Section 8.8."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    verdict: Literal['same', 'supersedes', 'coexists']
    rationale: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class RepairPromptBuilder(BaseModel):
    """Schema and builder for JSON repair prompts per LLD Section 7.2."""

    model_config = ConfigDict(extra='forbid', frozen=True)

    invalid_text: str
    error_message: str
    schema_def: str | None = None

    def build_prompt(self) -> str:
        """Render repair user prompt text."""
        schema_part = f'\nRequired JSON Schema:\n{self.schema_def}\n' if self.schema_def else ''
        return (
            'The previous JSON response was invalid and failed schema validation.\n'
            f'{schema_part}\n'
            'Invalid output received:\n'
            f'{self.invalid_text}\n\n'
            'Validation error:\n'
            f'{self.error_message}\n\n'
            'Please correct the output so that it strictly adheres to the schema. '
            'Output ONLY the corrected JSON object without markdown fences, explanation, '
            'or commentary.'
        )

    @classmethod
    def build(cls, invalid_text: str, error_message: str, schema_def: str | None = None) -> str:
        """Convenience factory returning rendered repair prompt."""
        return cls(
            invalid_text=invalid_text,
            error_message=error_message,
            schema_def=schema_def,
        ).build_prompt()
