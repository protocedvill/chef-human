"""Preflight checks for `chef-human doctor`.

Every external dependency (Python version, package metadata, filesystem,
Ollama reachability, optional-capability imports) is reached through an
injectable probe so tests can exercise every branch without depending on
real machine state. Callers that don't pass a probe get the real
implementation.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

Status = Literal["ok", "warn", "fail"]

DOCTOR_SCHEMA_VERSION = 1

_SUPPORTED_PYTHON = ((3, 12), (3, 13))

# (check name, module to probe, extra to install it)
_OPTIONAL_CAPABILITIES: tuple[tuple[str, str, str], ...] = (
    ("tui", "textual", "textual"),
    ("indexing", "tree_sitter", "indexing"),
    ("rag_embeddings", "sentence_transformers", "embeddings"),
    ("rag_vector_store", "faiss", "rag"),
    ("llamacpp", "llama_cpp", "llamacpp"),
)


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    message: str
    remedy: str | None = None


@dataclass(frozen=True)
class DoctorReport:
    checks: list[CheckResult]
    next_command: str
    invalid_config: bool = False

    @property
    def ready(self) -> bool:
        return not any(c.status == "fail" for c in self.checks)

    @property
    def exit_code(self) -> int:
        if self.invalid_config:
            return 2
        return 0 if self.ready else 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DOCTOR_SCHEMA_VERSION,
            "ready": self.ready,
            "checks": [
                {
                    "name": c.name,
                    "status": c.status,
                    "message": c.message,
                    "remedy": c.remedy,
                }
                for c in self.checks
            ],
            "next_command": self.next_command,
        }


def _default_package_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("chef-human")
    except Exception:
        return None


def _default_workspace_writable(path: Path) -> bool:
    target = path if path.exists() else path.parent
    return os.access(target, os.W_OK)


def _default_git_present(path: Path) -> bool:
    return (path / ".git").exists()


def _default_module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError):
        return False


def _default_ollama_list(host: str) -> list[str]:
    import ollama

    client = ollama.Client(host=host)
    response = client.list()
    return [m.model for m in response.models if m.model is not None]


@dataclass
class DoctorProbes:
    """Injectable seams for every real-world check `run_doctor` performs."""

    python_version_info: tuple[int, int, int] | None = None
    package_version: Callable[[], str | None] = field(default=_default_package_version)
    workspace_writable: Callable[[Path], bool] = field(default=_default_workspace_writable)
    git_present: Callable[[Path], bool] = field(default=_default_git_present)
    module_available: Callable[[str], bool] = field(default=_default_module_available)
    ollama_list: Callable[[str], list[str]] = field(default=_default_ollama_list)


def run_doctor(
    *,
    config_path: str | None = None,
    workspace: str | None = None,
    probes: DoctorProbes | None = None,
) -> DoctorReport:
    """Run every preflight check and return a report.

    Configuration is loaded (not passed in) so a broken config.toml or an
    invalid CHEF_ environment variable surfaces as its own check rather than
    raising out of the CLI command.
    """
    probes = probes or DoctorProbes()
    checks: list[CheckResult] = []

    from chef_human.config import load_settings

    try:
        settings = load_settings(config_path) if config_path else load_settings()
    except Exception as exc:
        checks.append(
            CheckResult(
                "configuration",
                "fail",
                f"Could not parse configuration: {exc}",
                remedy="Fix config.toml or the CHEF_-prefixed environment variables, "
                "then re-run `chef-human doctor`.",
            )
        )
        return DoctorReport(
            checks=checks,
            next_command="chef-human doctor",
            invalid_config=True,
        )

    checks.append(
        CheckResult(
            "configuration",
            "ok",
            f"Configuration loaded (backend={settings.llm_backend}, model={settings.ollama_model}).",
        )
    )

    pv = probes.python_version_info or sys.version_info[:3]
    if (pv[0], pv[1]) in _SUPPORTED_PYTHON:
        checks.append(
            CheckResult("python_version", "ok", f"Python {pv[0]}.{pv[1]}.{pv[2]} is supported.")
        )
    else:
        supported = ", ".join(f"{major}.{minor}" for major, minor in _SUPPORTED_PYTHON)
        checks.append(
            CheckResult(
                "python_version",
                "fail",
                f"Python {pv[0]}.{pv[1]}.{pv[2]} is not a supported version.",
                remedy=f"Install Python {supported} and recreate the virtual environment.",
            )
        )

    pkg_version = probes.package_version()
    if pkg_version:
        checks.append(
            CheckResult("chef_human_version", "ok", f"chef-human {pkg_version} installed.")
        )
    else:
        checks.append(
            CheckResult(
                "chef_human_version",
                "warn",
                "Could not determine the installed chef-human version.",
                remedy='Reinstall with `pip install -e ".[dev]"`.',
            )
        )

    ws_path = Path(workspace or settings.workspace or ".").resolve()
    if probes.workspace_writable(ws_path):
        checks.append(CheckResult("workspace", "ok", f"Workspace {ws_path} is writable."))
    else:
        checks.append(
            CheckResult(
                "workspace",
                "fail",
                f"Workspace {ws_path} is not writable.",
                remedy=f"Choose a writable directory with --workspace, "
                f"or fix permissions on {ws_path}.",
            )
        )

    if probes.git_present(ws_path):
        checks.append(CheckResult("git", "ok", "Workspace is a Git repository."))
    else:
        checks.append(
            CheckResult(
                "git",
                "warn",
                "Workspace is not a Git repository.",
                remedy="Run `git init` so changes are tracked and easy to review or undo.",
            )
        )

    if settings.llm_backend == "ollama":
        try:
            models = probes.ollama_list(settings.ollama_host)
        except Exception as exc:
            checks.append(
                CheckResult(
                    "ollama_server",
                    "fail",
                    f"Cannot reach Ollama at {settings.ollama_host}: {exc}",
                    remedy="Start Ollama (`ollama serve`) and re-run `chef-human doctor`.",
                )
            )
            checks.append(
                CheckResult(
                    "ollama_model",
                    "fail",
                    "Could not check model availability: server unreachable.",
                    remedy=f"Once Ollama is running, run `ollama pull {settings.ollama_model}`.",
                )
            )
        else:
            checks.append(
                CheckResult("ollama_server", "ok", f"Ollama reachable at {settings.ollama_host}.")
            )
            if settings.ollama_model in models:
                checks.append(
                    CheckResult(
                        "ollama_model", "ok", f"Model {settings.ollama_model} is available."
                    )
                )
            else:
                checks.append(
                    CheckResult(
                        "ollama_model",
                        "fail",
                        f"Model {settings.ollama_model} is not pulled.",
                        remedy=f"Run `ollama pull {settings.ollama_model}`.",
                    )
                )
    else:
        if settings.llamacpp_model_path:
            checks.append(
                CheckResult(
                    "llamacpp_model",
                    "ok",
                    f"llama.cpp model path set to {settings.llamacpp_model_path}.",
                )
            )
        else:
            checks.append(
                CheckResult(
                    "llamacpp_model",
                    "fail",
                    "No llama.cpp model path configured.",
                    remedy="Set llamacpp_model_path in config.toml.",
                )
            )

    for cap_name, module_name, extra in _OPTIONAL_CAPABILITIES:
        if probes.module_available(module_name):
            checks.append(CheckResult(cap_name, "ok", f"{module_name} is installed."))
        else:
            checks.append(
                CheckResult(
                    cap_name,
                    "warn",
                    f"{module_name} is not installed; {cap_name} features are unavailable.",
                    remedy=f'Install it with `pip install -e ".[{extra}]"`.',
                )
            )

    ready = not any(c.status == "fail" for c in checks)
    next_command = (
        'chef-human run "add validation and tests"'
        if ready
        else "chef-human doctor --json   # see failing checks and remedies"
    )
    return DoctorReport(checks=checks, next_command=next_command)
