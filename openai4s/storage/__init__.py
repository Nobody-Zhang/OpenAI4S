"""Domain repositories behind the compatibility ``Store`` facade."""

from openai4s.storage.agents import AgentProfileRepository
from openai4s.storage.annotations import AnnotationRepository
from openai4s.storage.connectors import ConnectorRepository
from openai4s.storage.memories import MemoryRepository
from openai4s.storage.metadata import (
    CompactionRepository,
    EndpointRepository,
    FolderRepository,
    HostCallRepository,
    NotesRepository,
)
from openai4s.storage.permissions import PermissionRuleRepository
from openai4s.storage.plans import PlanRepository
from openai4s.storage.settings import SettingsRepository

__all__ = [
    "AgentProfileRepository",
    "AnnotationRepository",
    "ConnectorRepository",
    "MemoryRepository",
    "CompactionRepository",
    "EndpointRepository",
    "FolderRepository",
    "HostCallRepository",
    "NotesRepository",
    "PermissionRuleRepository",
    "PlanRepository",
    "SettingsRepository",
]
