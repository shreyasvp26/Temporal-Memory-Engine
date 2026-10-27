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
from memory_engine.ingest.gate import GateDecision, IngestGate
from memory_engine.ingest.time_resolver import ResolvedCandidateTime, TimeResolver
from memory_engine.ingest.windowing import NoteChunker

__all__ = [
    'CanonicalObject',
    'Canonicalizer',
    'ConflictResolver',
    'EntityLinker',
    'FALLBACK_PREDICATE',
    'GateDecision',
    'IngestGate',
    'NoteChunker',
    'Outcome',
    'ResolutionContext',
    'ResolutionPlan',
    'ResolvedCandidateTime',
    'TimeResolver',
    'default_registry',
    'is_compatible_type',
    'is_self_mention',
    'norm_name',
    'precompute_mention_embeddings',
]
