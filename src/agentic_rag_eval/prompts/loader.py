from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from agentic_rag_eval.logging_setup import get_logger

logger = get_logger(__name__)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_PROMPT_DIR = _REPO_ROOT / "prompts"


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    description: str
    template: str

    def render(self, **kwargs: Any) -> str:
        """Render the template by substituting `{var}` placeholders."""
        result = self.template
        for key, value in kwargs.items():
            result = result.replace("{" + key + "}", str(value))
        return result


class PromptRegistry:
    """Loads prompt templates from a directory and serves them by name."""

    def __init__(self, prompt_dir: Path | None = None) -> None:
        self._dir = prompt_dir or _DEFAULT_PROMPT_DIR
        self._prompts: dict[str, Prompt] = {}
        self._load()

    def _load(self) -> None:
        if not self._dir.exists():
            logger.warning("prompt_dir_missing", extra={"path": str(self._dir)})
            return

        for yaml_file in sorted(self._dir.glob("*.yaml")):
            try:
                data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
            except yaml.YAMLError as e:
                logger.error(
                    "prompt_yaml_invalid",
                    extra={"file": str(yaml_file), "error": str(e)},
                )
                continue

            if not isinstance(data, dict):
                logger.error("prompt_yaml_not_dict", extra={"file": str(yaml_file)})
                continue

            required = {"name", "version", "template"}
            missing = required - data.keys()
            if missing:
                logger.error(
                    "prompt_missing_fields",
                    extra={"file": str(yaml_file), "missing": sorted(missing)},
                )
                continue

            prompt = Prompt(
                name=str(data["name"]),
                version=str(data["version"]),
                description=str(data.get("description", "")),
                template=str(data["template"]),
            )
            if prompt.name in self._prompts:
                logger.warning(
                    "prompt_duplicate_name",
                    extra={"name": prompt.name, "file": str(yaml_file)},
                )
            self._prompts[prompt.name] = prompt

        logger.info(
            "prompts_loaded",
            extra={"count": len(self._prompts), "dir": str(self._dir)},
        )

    def get(self, name: str) -> Prompt:
        try:
            return self._prompts[name]
        except KeyError as e:
            raise KeyError(f"Prompt {name!r} not found. Available: {sorted(self._prompts)}") from e

    def names(self) -> list[str]:
        return sorted(self._prompts)

    def versions(self) -> dict[str, str]:
        """Return a mapping of prompt name to version string."""
        return {name: p.version for name, p in sorted(self._prompts.items())}

    def version_hash(self) -> str:
        """Return a deterministic 16-char hash of all loaded prompt versions."""
        canonical = "|".join(f"{k}={v}" for k, v in self.versions().items())
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@lru_cache(maxsize=1)
def get_prompt_registry() -> PromptRegistry:
    """Return the cached PromptRegistry singleton."""
    return PromptRegistry()
