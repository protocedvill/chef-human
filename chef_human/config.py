from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast


@dataclass(frozen=True)
class Settings:
    llm_backend: Literal["ollama", "llamacpp"] = "ollama"
    ollama_model: str = "qwen3.6:35b-a3b"
    ollama_host: str = "http://localhost:11434"
    ollama_think: bool | Literal["low", "medium", "high"] = False
    llamacpp_model_path: str | None = None
    llamacpp_n_gpu_layers: int = 0
    llamacpp_n_threads: int | None = None
    planner_ollama_model: str | None = None
    planner_llamacpp_model_path: str | None = None
    max_context_tokens: int = 32768
    max_response_tokens: int = 4096
    embed_model: str = "BAAI/bge-small-en-v1.5"
    temperature: float = 0.0
    workspace: str = ""
    max_index_files: int = 500
    rag_chunk_tokens: int = 512
    rag_chunk_overlap: int = 64
    rag_max_results: int = 5
    rag_index_dir: str = ".chef-human"
    rag_enabled: bool = False
    fuzzy_edit: bool = True
    fuzzy_threshold: float = 0.75
    show_diff_in_context: bool = True
    max_tool_retries: int = 3
    max_agent_steps: int = 25
    persist_index: bool = True
    watch_files: bool = False
    watch_interval: float = 2.0
    tool_timeout: float = 60.0

    @property
    def effective_max_response_tokens(self) -> int:
        """max_response_tokens, scaled up under think mode.

        Every consumer of a response-token budget (ReActConfig's per-turn
        completion cap, and ContextConfig's own reserved headroom when
        deciding how much conversation history it can pack into the prompt)
        must derive this the same way. They used to compute the think-mode
        multiplier independently -- create_agent() applied it to
        ReActConfig, but create_context_assembler() built ContextConfig
        straight from the raw max_response_tokens, so under think mode the
        context assembler kept trimming to leave only the *unscaled* amount
        of headroom while the real request asked Ollama for up to 4x that
        many response tokens. Prompt + response could then together exceed
        the model's actual context window, truncating a native tool_calls
        response mid-JSON -- which Ollama's own template parser doesn't
        handle gracefully (hard 500, "tool call parsing failed error=EOF"),
        rather than max_tokens simply capping generation early. Observed
        live: prompt_tokens climbed to 31917 of a 32768-token window,
        leaving only 851 tokens of real headroom for a response the context
        budget thought it had reserved 4096 for.
        """
        return self.max_response_tokens * (4 if self.ollama_think else 1)


_ENV_PREFIX = "CHEF_"


def _parse_value(value: str) -> str | int | float | bool:
    v = value.strip()
    if v.lower() in ("true", "1"):
        return True
    if v.lower() in ("false", "0"):
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _load_toml(path: str = "config.toml") -> dict[str, object]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("rb") as f:
        data = tomllib.load(f)
    result: dict[str, object] = data.get("chef_human", {})
    return result


def _find_project_config(start_dir: str | Path = ".") -> Path | None:
    current = Path(start_dir).resolve()
    for parent in [current] + list(current.parents):
        candidate = parent / ".chef-human" / "config.toml"
        if candidate.exists():
            return candidate
    return None


def _load_env() -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in os.environ.items():
        if key.startswith(_ENV_PREFIX):
            setting_name = key[len(_ENV_PREFIX) :].lower()
            result[setting_name] = _parse_value(value)
    return result


def load_settings(
    config_path: str = "config.toml",
    project_start: str | Path = ".",
) -> Settings:
    merged: dict[str, object] = {}
    merged.update(_load_toml(config_path))
    project_cfg = _find_project_config(project_start)
    if project_cfg is not None:
        merged.update(_load_toml(str(project_cfg)))
    merged.update(_load_env())
    # Configuration enters through TOML and environment variables, so the
    # mapping cannot retain per-key static types. Settings remains the typed
    # validation boundary for every consumer.
    return Settings(**cast(dict[str, Any], merged))


settings = load_settings()
