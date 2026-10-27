"""Ingestion subsystem for Temporal Memory Engine."""

from memory_engine.ingest.canonicalizer import (
    FALLBACK_PREDICATE,
    Canonicalizer,
    CanonicalObject,
)
from memory_engine.ingest.conflict import (
    ConflictResolver,
    Outcome,
    ResolutionContext,
    ResolutionPlan,
    default_registry,
)
from memory_engine.ingest.entity_linker import (
    EntityLinker,
    is_compatible_type,
    is_self_mention,
    norm_name,
    precompute_mention_embeddings,
)
from memory_engine.ingest.extractor import (
    FactExtractor,
    LLMFactExtractor,
    ValidatedExtraction,
)
from memory_engine.ingest.gate import GateDecision, IngestGate
from memory_engine.ingest.plan_executor import PlanExecutor
from memory_engine.ingest.service import (
    EnqueueResult,
    IngestionService,
    IngestService,
    NoteAccepted,
    NoteIngestResult,
    TurnAccepted,
)
from memory_engine.ingest.time_resolver import ResolvedCandidateTime, TimeResolver
from memory_engine.ingest.windowing import NoteChunker

__all__ = [
    'CanonicalObject',
    'Canonicalizer',
    'ConflictResolver',
    'EnqueueResult',
    'EntityLinker',
    'FALLBACK_PREDICATE',
    'FactExtractor',
    'GateDecision',
    'IngestGate',
    'IngestService',
    'IngestionService',
    'LLMFactExtractor',
    'NoteAccepted',
    'NoteChunker',
    'NoteIngestResult',
    'Outcome',
    'PlanExecutor',
    'ResolutionContext',
    'ResolutionPlan',
    'ResolvedCandidateTime',
    'TimeResolver',
    'TurnAccepted',
    'ValidatedExtraction',
    'default_registry',
    'is_compatible_type',
    'is_self_mention',
    'norm_name',
    'precompute_mention_embeddings',
]
