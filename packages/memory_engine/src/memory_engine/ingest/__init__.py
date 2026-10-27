"""Ingestion subsystem for Temporal Memory Engine."""

from memory_engine.ingest.gate import GateDecision, IngestGate
from memory_engine.ingest.windowing import NoteChunker

__all__ = [
    'GateDecision',
    'IngestGate',
    'NoteChunker',
]
