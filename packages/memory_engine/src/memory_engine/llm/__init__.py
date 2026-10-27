"""LLM subsystem adapters, gateway, schemas and frozen prompts."""

from memory_engine.llm.gateway import DefaultLLMGateway
from memory_engine.llm.gemini import GeminiClient
from memory_engine.llm.groq import GroqClient
from memory_engine.llm.ollama import OllamaClient
from memory_engine.llm.prompts import (
    CONFLICT_JUDGE_SYSTEM_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    GROUNDED_CHAT_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    PROMPT_VERSION_EXTRACT,
    PROMPT_VERSION_JUDGE,
    PROMPT_VERSION_PLAN,
    PROMPT_VERSION_REPAIR,
    PROMPT_VERSION_SYNTH,
    REPAIR_SYSTEM_PROMPT,
    build_conflict_judge_user_prompt,
    build_extraction_user_prompt,
    build_grounded_chat_user_prompt,
    build_planner_user_prompt,
    build_repair_user_prompt,
)
from memory_engine.llm.schemas import (
    ExtractedFact,
    ExtractionResult,
    JudgeLLMOutput,
    PlannerLLMOutput,
    RepairPromptBuilder,
)

__all__ = [
    'CONFLICT_JUDGE_SYSTEM_PROMPT',
    'DefaultLLMGateway',
    'EXTRACTION_SYSTEM_PROMPT',
    'ExtractedFact',
    'ExtractionResult',
    'GROUNDED_CHAT_SYSTEM_PROMPT',
    'GeminiClient',
    'GroqClient',
    'JudgeLLMOutput',
    'OllamaClient',
    'PLANNER_SYSTEM_PROMPT',
    'PROMPT_VERSION_EXTRACT',
    'PROMPT_VERSION_JUDGE',
    'PROMPT_VERSION_PLAN',
    'PROMPT_VERSION_REPAIR',
    'PROMPT_VERSION_SYNTH',
    'PlannerLLMOutput',
    'REPAIR_SYSTEM_PROMPT',
    'RepairPromptBuilder',
    'build_conflict_judge_user_prompt',
    'build_extraction_user_prompt',
    'build_grounded_chat_user_prompt',
    'build_planner_user_prompt',
    'build_repair_user_prompt',
]
