"""Domain identifier types."""

from typing import NewType
from uuid import UUID

FactId = NewType('FactId', UUID)
SourceId = NewType('SourceId', UUID)
EntityId = NewType('EntityId', UUID)
JobId = NewType('JobId', UUID)
UserId = NewType('UserId', UUID)

__all__ = [
    'EntityId',
    'FactId',
    'JobId',
    'SourceId',
    'UserId',
]
