"""Host-side services used by the kernel RPC dispatcher."""

from openai4s.host.files import WorkspaceFileService, is_secret_path

__all__ = ["WorkspaceFileService", "is_secret_path"]
