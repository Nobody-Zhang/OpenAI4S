"""Skill discovery, progressive disclosure, and sidecar structure gate.

Mirrors openai4s's skill model at three levels:
  1. Discovery      — scan skills_dir for <name>/SKILL.md (+ optional kernel.py).
  2. Progressive    — the system prompt only lists skill name + one-line summary;
     disclosure       full docs are pulled on demand via host.search_skills().
  3. Sidecar gate   — kernel.py sidecars are compile-checked before use, returning
                      {ok, error?} (openai4s's `sidecar_gate` structure).

SKILL.md may start with a YAML-ish frontmatter block:

    ---
    name: stats
    description: descriptive-statistics helpers (mean/std/quantile/zscore)
    origin: personal
    ---

`description` becomes the one-line summary shown in the prompt. `origin` is one
of openai4s|organization|personal|draft|unknown and drives the permission gate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from openai4s.config import Config, get_config

_VALID_ORIGINS = ("openai4s", "organization", "personal", "draft", "unknown")
# origins whose sidecar/doc is read-only (cannot be edited/deleted via CRUD)
_READONLY_ORIGINS = ("openai4s",)
_WORD = re.compile(r"[a-z0-9]+")


def _strip_scalar(v: str) -> str:
    """Normalize an inline YAML scalar: drop inline comments and surrounding
    quotes. Only strips a `#` comment on *unquoted* values so a `#` inside a
    quoted description survives."""
    v = v.strip()
    if len(v) >= 2 and v[0] in "\"'" and v[-1] == v[0]:
        return v[1:-1]
    # unquoted: a ` #` starts a trailing comment
    return v.split(" #", 1)[0].strip()


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split an optional leading `--- ... ---` frontmatter block off the body.

    Understands a deliberately small YAML subset — enough for skill
    frontmatter, not a general parser:

      * top-level `key: scalar` (quoted or unquoted, with inline comments);
      * top-level `key: >` / `key: |` **block scalars** (folded/literal), whose
        value is the following more-indented lines. Folded (`>`) joins lines
        with spaces; literal (`|`) preserves newlines. Chomping indicators
        (`-`/`+`) are accepted and ignored — descriptions are collapsed anyway.

    Indented lines that are NOT a block-scalar continuation belong to a nested
    mapping/sequence (e.g. metadata.third_party[].name) and are ignored so they
    cannot clobber a top-level key of the same name.
    """
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    raw = text[3:end].strip("\n")
    body = text[end + 4 :].lstrip("\n")
    meta: dict = {}
    lines = raw.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        # Only TOP-LEVEL keys start at column 0. Skip blanks, comments, list
        # items, and any indented (nested) lines.
        if not line or line[0] in (" ", "\t", "#", "-"):
            i += 1
            continue
        if ":" not in line:
            i += 1
            continue
        k, _, v = line.partition(":")
        key = k.strip().lower()
        marker = v.strip()
        # strip an optional chomping indicator to detect a block scalar
        if marker and marker[0] in "|>" and marker[1:] in ("", "-", "+"):
            folded = marker[0] == ">"
            block: list[str] = []
            i += 1
            while i < n and (lines[i] == "" or lines[i][0] in (" ", "\t")):
                block.append(lines[i])
                i += 1
            # dedent by the minimum indent of the non-blank block lines
            indents = [len(ln) - len(ln.lstrip(" \t")) for ln in block if ln.strip()]
            pad = min(indents) if indents else 0
            dedented = [ln[pad:] if ln.strip() else "" for ln in block]
            sep = " " if folded else "\n"
            meta[key] = sep.join(x.strip() if folded else x for x in dedented).strip()
            continue
        meta[key] = _strip_scalar(v)
        i += 1
    return meta, body


def _first_paragraph(body: str) -> str:
    for block in body.split("\n\n"):
        cleaned = " ".join(
            ln.strip().lstrip("#").strip()
            for ln in block.splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ).strip()
        if cleaned:
            return cleaned
    # fall back to first non-heading line
    for ln in body.splitlines():
        s = ln.strip().lstrip("#").strip()
        if s:
            return s
    return ""


def _tokenize(*texts: str) -> set[str]:
    toks: set[str] = set()
    for t in texts:
        toks.update(_WORD.findall(t.lower()))
    return toks


@dataclass
class Skill:
    name: str
    root: Path
    doc: str  # SKILL.md body (frontmatter stripped)
    has_kernel: bool  # kernel.py sidecar present?
    description: str = ""  # one-line summary for progressive disclosure
    origin: str = "unknown"
    keywords: set[str] = field(default_factory=set)

    @property
    def read_only(self) -> bool:
        return self.origin in _READONLY_ORIGINS

    @property
    def import_hint(self) -> str | None:
        """How the agent imports this skill's sidecar inside a kernel cell.

        The sidecar lives on disk under the *directory* name (which is what
        `bootstrap_code` puts on `sys.path`), so imports must use the dir name,
        not the declared frontmatter `name`. Directory names may contain
        hyphens (e.g. `pdf-explore`), which are not valid Python identifiers —
        `from pdf-explore.kernel import *` is a SyntaxError. For those, emit an
        `importlib.import_module(...)` hint, which resolves the sidecar as a
        namespace submodule and works with hyphenated dir names.
        """
        if not self.has_kernel:
            return None
        mod = self.root.name
        if mod.isidentifier():
            return f"from {mod}.kernel import * # or: import {mod}.kernel as k"
        return (
            f'import importlib; k = importlib.import_module("{mod}.kernel") '
            f"# '{mod}' isn't a valid identifier; import * won't work"
        )

    def summary_line(self) -> str:
        return f"- {self.name}: {self.description or '(no description)'}"

    def sidecar_gate(self) -> dict:
        """Compile-check the kernel.py sidecar (openai4s's structure gate).

        Returns {"ok": bool, "error": str|None}. A skill with no sidecar is
        trivially ok. This catches syntax errors BEFORE the agent tries to
        import the sidecar mid-task.
        """
        if not self.has_kernel:
            return {"ok": True, "error": None}
        path = self.root / "kernel.py"
        try:
            src = path.read_text("utf-8")
            compile(src, str(path), "exec")
            return {"ok": True, "error": None}
        except SyntaxError as e:
            return {"ok": False, "error": f"{e.__class__.__name__}: {e}"}
        except OSError as e:
            return {"ok": False, "error": f"cannot read sidecar: {e}"}


class SkillLoader:
    def __init__(self, skills_dir: Path | None = None, cfg: Config | None = None):
        self.cfg = cfg or get_config()
        self.skills_dir = Path(skills_dir) if skills_dir else self.cfg.skills_dir
        self._skills: dict[str, Skill] = {}

    def user_skills_dir(self) -> Path:
        """Writable dir for user-authored skills (kept separate from the bundled
        read-only skills). Discovered alongside the bundled ones."""
        return self.cfg.data_dir / "user-skills"

    def discover(self) -> dict[str, Skill]:
        self._skills = {}
        # bundled skills first, then user-authored ones. A user skill must NOT
        # silently shadow a trusted BUNDLED skill of the same dir-name — bundled
        # wins on collision (else agent loads untrusted content under a trusted name).
        for base in (self.skills_dir, self.user_skills_dir()):
            if not base or not base.exists():
                continue
            is_user = base.resolve() == self.user_skills_dir().resolve()
            for child in sorted(base.iterdir()):
                if not child.is_dir():
                    continue
                md = child / "SKILL.md"
                if not md.exists():
                    continue
                if is_user and child.name in self._skills:
                    continue  # bundled skill already claimed this name — keep it
                raw = md.read_text("utf-8")
                meta, body = _parse_frontmatter(raw)
                origin = (meta.get("origin") or "unknown").lower()
                if is_user:
                    origin = "user"
                elif origin not in _VALID_ORIGINS:
                    origin = "unknown"
                description = meta.get("description") or _first_paragraph(body)
                description = " ".join(description.split())  # collapse whitespace
                if len(description) > 200:
                    description = description[:197] + "..."
                name = meta.get("name") or child.name
                self._skills[child.name] = Skill(
                    name=name,
                    root=child,
                    doc=body,
                    has_kernel=(child / "kernel.py").exists(),
                    description=description,
                    origin=origin,
                    keywords=_tokenize(name, description, body),
                )
        return self._skills

    def skills(self) -> dict[str, Skill]:
        if not self._skills:
            self.discover()
        return self._skills

    def get(self, name: str) -> Skill | None:
        skills = self.skills()
        if name in skills:
            return skills[name]
        # allow lookup by declared skill.name too
        for s in skills.values():
            if s.name == name:
                return s
        return None

    def bootstrap_code(self) -> str:
        """Code the kernel runs at startup so skill sidecars import cleanly."""
        return (
            "import sys as _sys\n"
            f"_sd = {str(self.skills_dir)!r}\n"
            "if _sd not in _sys.path:\n"
            "    _sys.path.insert(0, _sd)\n"
        )

    def search(self, query: str, *, limit: int = 5) -> list[dict]:
        """Keyword-overlap skill retrieval (openai4s's search_skills route).

        Scores each skill by literal token overlap between the query and the
        skill's name/description/body. Purely lexical — no synonym expansion —
        matching the documented limitation of the skill-retrieval prompt.
        Returns the full doc of the top matches so the agent can then use them.
        """
        q_tokens = _tokenize(query)
        scored: list[tuple[float, Skill]] = []
        for s in self.skills().values():
            if not q_tokens:
                score = 0.0
            else:
                overlap = len(q_tokens & s.keywords)
                # bias toward name/description hits
                name_hit = len(q_tokens & _tokenize(s.name, s.description))
                score = overlap + 1.5 * name_hit
            if score > 0:
                scored.append((score, s))
        scored.sort(key=lambda t: t[0], reverse=True)
        results = []
        for score, s in scored[:limit]:
            gate = s.sidecar_gate()
            results.append(
                {
                    "name": s.name,
                    "origin": s.origin,
                    "description": s.description,
                    "import": s.import_hint,
                    "score": round(score, 2),
                    "doc": s.doc.strip(),
                    "sidecar_gate": gate,
                }
            )
        return results

    def catalog(self) -> list[dict]:
        """Lightweight listing (name/description/origin) — no full docs."""
        return [
            {
                "name": s.name,
                "description": s.description,
                "origin": s.origin,
                "has_kernel": s.has_kernel,
            }
            for s in self.skills().values()
        ]

    def system_context(self) -> str:
        """Progressive-disclosure block for the system prompt.

        Only skill NAMES + one-line summaries go here — NOT the full docs.
        The agent calls host.search_skills(query) to pull a skill's full recipe
        on demand: analytic tasks retrieve skills lazily instead of
        front-loading every doc into context.
        """
        skills = self.skills()
        if not skills:
            return ""
        lines = [
            "# Available skills (progressive disclosure)",
            "These skills exist but their full instructions are NOT loaded yet. "
            "When a task looks relevant to one, call "
            '`host.search_skills("<keywords>")` in a code cell to retrieve its '
            "full recipe, then import its sidecar and use it. Do NOT invent "
            "skills or APIs you have not retrieved.",
            "",
        ]
        for s in skills.values():
            lines.append(s.summary_line())
        return "\n".join(lines)


def discover_skills(
    skills_dir: Path | None = None, cfg: Config | None = None
) -> dict[str, Skill]:
    return SkillLoader(skills_dir, cfg).discover()
