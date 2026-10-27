"""Ingestion subsystem for Temporal Memory Engine."""

from memory_engine.ingest.conflict import (
    ConflictResolver,
    Outcome,
    ResolutionContext,
    ResolutionPlan,
    default_registry,
)
from memory_engine.ingest.gate import GateDecision, IngestGate
from memory_engine.ingest.time_resolver import ResolvedCandidateTime, TimeResolver
from memory_engine.ingest.windowing import NoteChunker

__all__ = [
    'ConflictResolver',
    'GateDecision',
    'IngestGate',
    'NoteChunker',
    'Outcome',
    'ResolutionContext',
    'ResolutionPlan',
    'ResolvedCandidateTime',
    'TimeResolver',
    'default_registry',
]
