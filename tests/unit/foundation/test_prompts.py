from __future__ import annotations

import logging
from pathlib import Path

import pytest

from agentic_rag_eval.prompts.loader import Prompt, PromptRegistry


def _write_prompt(
    dir_: Path, name: str, *, version: str = "v1", template: str = "Hello {name}"
) -> Path:
    """Write a valid prompt YAML file into ``dir_`` and return the path."""
    path = dir_ / f"{name}.yaml"
    path.write_text(
        f"name: {name}\nversion: {version}\ndescription: test prompt\ntemplate: |\n  {template}\n",
        encoding="utf-8",
    )
    return path


def test_registry_loads_yaml_files(tmp_path: Path) -> None:
    """All valid YAML files in the prompt dir must be registered."""
    _write_prompt(tmp_path, "decompose", template="Decompose {question}")
    _write_prompt(tmp_path, "judge", template="Judge {answer}")

    registry = PromptRegistry(prompt_dir=tmp_path)

    assert set(registry.names()) == {"decompose", "judge"}
    assert isinstance(registry.get("decompose"), Prompt)


def test_registry_missing_dir_is_nonfatal(tmp_path: Path) -> None:
    """A non-existent prompt dir must not raise — registry should be empty."""
    missing = tmp_path / "does_not_exist"
    registry = PromptRegistry(prompt_dir=missing)
    assert registry.names() == []


def test_get_raises_for_unknown_prompt(tmp_path: Path) -> None:
    """Looking up an unregistered prompt must raise KeyError."""
    _write_prompt(tmp_path, "decompose")
    registry = PromptRegistry(prompt_dir=tmp_path)

    with pytest.raises(KeyError, match="not found"):
        registry.get("nonexistent")


def test_render_substitutes_placeholders(tmp_path: Path) -> None:
    """Placeholders of the form `{var}` must be replaced by kwargs."""
    _write_prompt(tmp_path, "greet", template="Hello {name}, you are {age}")
    registry = PromptRegistry(prompt_dir=tmp_path)

    rendered = registry.get("greet").render(name="Ada", age=36)

    assert "Hello Ada" in rendered
    assert "you are 36" in rendered
    assert "{name}" not in rendered


def test_render_preserves_literal_braces(tmp_path: Path) -> None:
    """Literal JSON-style braces that aren't placeholders must stay intact."""
    template = 'Return {"ok": true} and {value}'
    _write_prompt(tmp_path, "json_like", template=template)
    registry = PromptRegistry(prompt_dir=tmp_path)

    rendered = registry.get("json_like").render(value="yes")

    assert '{"ok": true}' in rendered
    assert "yes" in rendered


def test_version_hash_is_deterministic(tmp_path: Path) -> None:
    """Two registries loaded from the same content must have identical hashes."""
    _write_prompt(tmp_path, "a", version="v1")
    _write_prompt(tmp_path, "b", version="v2")

    h1 = PromptRegistry(prompt_dir=tmp_path).version_hash()
    h2 = PromptRegistry(prompt_dir=tmp_path).version_hash()
    assert h1 == h2
    assert len(h1) == 16


def test_version_hash_changes_when_version_bumps(tmp_path: Path) -> None:
    """Bumping any prompt version must change the aggregate hash."""
    _write_prompt(tmp_path, "a", version="v1")
    h1 = PromptRegistry(prompt_dir=tmp_path).version_hash()

    _write_prompt(tmp_path, "a", version="v2")
    h2 = PromptRegistry(prompt_dir=tmp_path).version_hash()

    assert h1 != h2


def test_malformed_yaml_is_logged_and_skipped(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Broken YAML must be logged, not crash loading of remaining prompts."""
    (tmp_path / "broken.yaml").write_text("::: this is not : valid :yaml:::", encoding="utf-8")
    _write_prompt(tmp_path, "good")

    with caplog.at_level(logging.ERROR):
        registry = PromptRegistry(prompt_dir=tmp_path)

    assert "good" in registry.names()
    assert "broken" not in registry.names()

    assert any("prompt_yaml_invalid" in rec.message for rec in caplog.records)


def test_missing_required_fields_skipped(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """YAML files missing required fields must be logged and skipped."""
    (tmp_path / "incomplete.yaml").write_text("name: just_a_name\n", encoding="utf-8")
    _write_prompt(tmp_path, "complete")

    with caplog.at_level(logging.ERROR):
        registry = PromptRegistry(prompt_dir=tmp_path)

    assert registry.names() == ["complete"]
    assert any("prompt_missing_fields" in rec.message for rec in caplog.records)
