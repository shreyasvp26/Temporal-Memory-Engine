"""Conflict resolution subsystem for fact ingestion."""

from memory_engine.ingest.conflict.event import EventStrategy
from memory_engine.ingest.conflict.multi import MultiValuedStrategy
from memory_engine.ingest.conflict.plan import (
    AddEvidence,
    AddRelation,
    BumpConfidence,
    CloseFact,
    ExtendValidFrom,
    FactRef,
    InsertFact,
    NewRef,
    Op,
    Outcome,
    ResolutionContext,
    ResolutionPlan,
    ResolutionStrategy,
    SetStatus,
)
from memory_engine.ingest.conflict.registry import (
    ConflictResolver,
    StrategyRegistry,
    default_registry,
)
from memory_engine.ingest.conflict.single import SingleValuedStrategy

__all__ = [
    'AddEvidence',
    'AddRelation',
    'BumpConfidence',
    'CloseFact',
    'ConflictResolver',
    'EventStrategy',
    'ExtendValidFrom',
    'FactRef',
    'InsertFact',
    'MultiValuedStrategy',
    'NewRef',
    'Op',
    'Outcome',
    'ResolutionContext',
    'ResolutionPlan',
    'ResolutionStrategy',
    'SetStatus',
    'SingleValuedStrategy',
    'StrategyRegistry',
    'default_registry',
]
