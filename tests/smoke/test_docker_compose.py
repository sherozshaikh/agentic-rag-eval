from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.smoke

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = PROJECT_ROOT / "docker" / "docker-compose.yml"


def _docker_available() -> bool:
    """Return True iff the ``docker`` binary is installed AND the daemon responds."""
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _compose_plugin_available() -> bool:
    """Return True iff ``docker compose`` (the plugin) is available."""
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def test_compose_file_exists() -> None:
    """The docker-compose file must exist at the documented location."""
    assert COMPOSE_FILE.is_file(), f"docker-compose file not found: {COMPOSE_FILE}"


def test_docker_compose_config_is_valid() -> None:
    """``docker compose config --quiet`` must exit 0 for our compose file."""
    if not _docker_available():
        pytest.skip("docker binary not installed or daemon not reachable")
    if not _compose_plugin_available():
        pytest.skip("'docker compose' plugin not installed")

    result = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "config", "--quiet"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, (
        f"'docker compose config' failed with exit code {result.returncode}\n"
        f"stderr:\n{result.stderr}\n"
        f"stdout:\n{result.stdout}"
    )


def test_docker_compose_defines_expected_services() -> None:
    """The rendered config must include the three documented services."""
    if not _docker_available():
        pytest.skip("docker binary not installed or daemon not reachable")
    if not _compose_plugin_available():
        pytest.skip("'docker compose' plugin not installed")

    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(COMPOSE_FILE),
            "--profile",
            "full",
            "--profile",
            "gpu",
            "config",
            "--services",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, f"'docker compose config --services' failed: {result.stderr}"

    services = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    expected = {"qdrant", "agentic-rag-eval", "ollama"}
    missing = expected - services
    assert not missing, f"Compose file is missing expected services: {missing} (got {services})"
