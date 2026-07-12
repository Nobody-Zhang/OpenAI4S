"""Host-side skill lifecycle behaviour.

The dispatcher remains the policy boundary (permissions, audit, UI activity,
soft failures).  This service owns the skill domain itself so retrieval and
filesystem mutation are visible in one class instead of being scattered across
``HostDispatcher``.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from openai4s.config import Config
from openai4s.skills_loader import SkillLoader, SkillVersionService


class SkillService:
    """Retrieve and manage Code-as-Action skill directories."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.loader = SkillLoader(cfg=cfg)
        self.versions = SkillVersionService(cfg)
        self.project_id: str | None = None
        self.session_id: str | None = None

    def set_scope(
        self,
        *,
        project_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        """Retarget prompt/search/read/bootstrap to one capability snapshot."""

        self.loader = self.loader.scoped(
            project_id=project_id,
            session_id=session_id,
        )
        self.project_id = str(project_id or "").strip() or None
        self.session_id = str(session_id or "").strip() or None

    def _writable_scope(self, skill: Any) -> tuple[str, str | None]:
        """Return the (scope, project_id) that owns a discovered skill on disk.

        ``discover()`` gives project skills precedence over personal ones, so a
        project-scoped edit/delete must target the project scope.  Hardcoding
        ``personal`` here would write to (or look up) a shadowed personal copy
        the loader never surfaces, leaving the real project skill untouched.
        """

        if getattr(skill, "source", "") == "project" and self.project_id:
            return "project", self.project_id
        return "personal", None

    def load(self, name: str | dict) -> dict:
        """Load full guidance, with the historical fuzzy-name fallback."""
        if isinstance(name, dict):
            name = name.get("name", "")
        self.loader.discover()
        skill = self.loader.get(name)
        if skill is None:
            hits = self.loader.search(name, limit=1)
            if hits:
                skill = self.loader.get(hits[0]["name"])
        if skill is None:
            return {"error": f"no such skill: {name!r}"}
        try:
            content = (skill.root / "SKILL.md").read_text("utf-8")
        except Exception:  # noqa: BLE001 - loader doc is the compatibility fallback
            content = getattr(skill, "doc", "") or ""
        return {
            "name": skill.name,
            "origin": skill.origin,
            "description": skill.description,
            "content": content,
        }

    def search(self, spec: dict) -> list:
        self.loader.discover()
        return self.loader.search(
            spec.get("query", ""), limit=int(spec.get("limit", 5))
        )

    def list(self) -> list:
        self.loader.discover()
        return self.loader.catalog()

    def get(self, name: str) -> dict:
        self.loader.discover()
        skill = self.loader.get(name)
        if skill is None:
            raise KeyError(f"no such skill: {name!r}")
        return {
            "name": skill.name,
            "origin": skill.origin,
            "description": skill.description,
            "has_kernel": skill.has_kernel,
            "read_only": skill.read_only,
            "sidecar_gate": skill.sidecar_gate(),
        }

    def read(self, spec: dict) -> str:
        self.loader.discover()
        return self.loader.read(spec["name"], spec.get("path", "SKILL.md"))

    def edit(self, spec: dict) -> dict:
        name = spec["name"]
        relative = spec.get("path", "SKILL.md")
        content = spec.get("content", "")
        old_string = spec.get("old_string")
        self.loader.discover()
        existing = self.loader.get(name, include_disabled=True)
        if existing is not None and existing.read_only:
            raise PermissionError(
                f"skill {name!r} origin={existing.origin} is read-only"
            )

        if relative == "SKILL.md" and old_string is None:
            self._reject_bundled_name_collision(content, fallback=name)

        if existing is not None:
            root = existing.root
        else:
            user_directory = self.loader.user_skills_dir()
            if user_directory.is_symlink():
                raise ValueError("unsafe user skill directory")
            user_directory.mkdir(parents=True, exist_ok=True)
            user_directory = user_directory.resolve()
            candidate = user_directory / self.versions.slug(name)
            if candidate.is_symlink():
                raise ValueError(f"unsafe skill path: {name!r}")
            root = candidate.resolve()
            if root == user_directory or not root.is_relative_to(user_directory):
                raise ValueError(f"unsafe skill name: {name!r}")

        target = self._safe_path(root, relative)
        files = self.versions.read_package(root) if (root / "SKILL.md").exists() else {}
        if "SKILL.md" not in files and relative != "SKILL.md":
            files["SKILL.md"] = (
                f"---\nname: {name}\ndescription: (draft)\norigin: draft\n---\n"
                f"# Skill: {name}\n"
            ).encode("utf-8")
        if old_string is None:
            updated_content = content
            mode = "overwrite"
        else:
            if relative not in files:
                raise FileNotFoundError(f"{relative} does not exist for str_replace")
            current = files[relative].decode("utf-8")
            if old_string not in current:
                raise ValueError("old_string not found in file")
            updated_content = current.replace(old_string, content, 1)
            mode = "str_replace"

        if relative == "SKILL.md":
            self._reject_bundled_name_collision(
                updated_content,
                fallback=existing.name if existing is not None else name,
            )
        files[relative] = updated_content.encode("utf-8")
        scope, project_id = self._writable_scope(existing)
        self.versions.install(
            existing.name if existing is not None else name,
            files,
            event="upgraded" if existing is not None else "installed",
            slug=root.name,
            scope=scope,
            project_id=project_id,
            require_sidecar_gate=False,
            metadata={"source": "host_skills_edit", "path": relative},
        )

        result: dict[str, Any] = {
            "ok": True,
            "mode": mode,
            "path": str(target),
        }
        if target.name == "kernel.py":
            self.loader.discover()
            skill = self.loader.get(name, include_disabled=True)
            result["sidecar_gate"] = (
                skill.sidecar_gate() if skill else {"ok": True, "error": None}
            )
        return result

    def publish(self, name: str) -> dict:
        self.loader.discover()
        skill = self.loader.get(name, include_disabled=True)
        if skill is None:
            raise KeyError(f"no such skill: {name!r}")
        if skill.read_only:
            raise PermissionError(f"skill {name!r} is read-only")
        self.versions.publish(name, slug=skill.root.name)
        return {"ok": True, "origin": "personal"}

    def delete(self, name: str) -> dict:
        self.loader.discover()
        skill = self.loader.get(name, include_disabled=True)
        if skill is None:
            raise KeyError(f"no such skill: {name!r}")
        if skill.read_only:
            raise PermissionError(f"skill {name!r} is read-only")
        scope, project_id = self._writable_scope(skill)
        installation = self.versions.repository.get_installation(
            skill.name,
            scope=scope,
            scope_id=str(project_id or "") if scope == "project" else "",
        )
        if installation is not None and installation.get("active_version_id"):
            self.versions.delete(skill.name, scope=scope, project_id=project_id)
        else:
            shutil.rmtree(skill.root)
        return {"ok": True, "deleted": name}

    def _version_request(self, spec: str | dict) -> tuple[str, str, str | None, int]:
        if isinstance(spec, dict):
            name = str(spec.get("name") or "").strip()
            scope = str(spec.get("scope") or "personal").strip().lower()
            requested_project = str(spec.get("project_id") or "").strip() or None
            limit = int(spec.get("limit") or 200)
        else:
            name = str(spec or "").strip()
            scope = "personal"
            requested_project = None
            limit = 200
        if not name:
            raise ValueError("skill name is required")
        if scope not in {"personal", "project"}:
            raise ValueError("skill scope must be 'personal' or 'project'")
        if scope == "personal":
            if requested_project:
                raise ValueError("personal Skill scope cannot have project_id")
            project_id = None
        else:
            project_id = requested_project or self.project_id
            if not project_id:
                raise PermissionError("project Skill scope requires an active project")
            if self.project_id and project_id != self.project_id:
                raise PermissionError("project Skill scope cannot cross projects")
        return name, scope, project_id, max(1, min(limit, 200))

    def status(self, spec: str | dict) -> dict:
        """Return safe active-version metadata for one exact writable scope."""

        name, scope, project_id, _limit = self._version_request(spec)
        bundled = self.loader.bundled_name_collision(name)
        if bundled is not None:
            return {
                "name": bundled.name,
                "scope": "bundled",
                "scope_id": "",
                "installed": True,
                "active": True,
                "active_version_id": None,
                "manifest": {
                    "origin": bundled.origin,
                    "document_sha256": bundled.document_sha256,
                    "sidecar": {
                        "present": bundled.has_kernel,
                        "sha256": bundled.sidecar_sha256,
                        "gate": bundled.sidecar_gate(),
                    },
                },
                "read_only": True,
                "rollback_available": False,
            }
        return {
            **self.versions.status(
                name,
                scope=scope,
                project_id=project_id,
            ),
            "read_only": False,
        }

    def history(self, name: str | dict, *, limit: int = 200) -> dict:
        """Expose immutable lifecycle records to trusted orchestration code."""

        spec = name if isinstance(name, dict) else {"name": name, "limit": limit}
        skill_name, scope, project_id, resolved_limit = self._version_request(spec)
        if self.loader.bundled_name_collision(skill_name) is not None:
            raise PermissionError(f"skill {skill_name!r} is bundled and read-only")
        return self.versions.history(
            skill_name,
            scope=scope,
            project_id=project_id,
            limit=resolved_limit,
        )

    def rollback(self, name: str | dict, version_id: str | None = None) -> dict:
        spec = (
            name
            if isinstance(name, dict)
            else {"name": name, "version_id": version_id, "scope": "personal"}
        )
        skill_name, scope, project_id, _limit = self._version_request(spec)
        target_version = str(spec.get("version_id") or "").strip()
        if not target_version:
            raise ValueError("skill version_id is required")
        if self.loader.bundled_name_collision(skill_name) is not None:
            raise PermissionError(f"skill {skill_name!r} is bundled and read-only")
        self.loader.discover()
        result = self.versions.rollback(
            skill_name,
            target_version,
            scope=scope,
            project_id=project_id,
        )
        self.loader.discover()
        return result

    def _reject_bundled_name_collision(self, content: str, *, fallback: str) -> None:
        metadata, _body = self.loader.parse_document(content)
        declared_name = metadata.get("name") or fallback
        collision = self.loader.bundled_name_collision(declared_name)
        if collision is not None:
            raise PermissionError(
                f"skill name {declared_name!r} collides with read-only bundled "
                f"skill {collision.name!r}"
            )

    @staticmethod
    def _safe_path(root: Path, relative: str) -> Path:
        root = root.resolve()
        target = (root / relative).resolve()
        if root != target and root not in target.parents:
            raise ValueError(f"path escapes skill dir: {relative!r}")
        return target


__all__ = ["SkillService"]
