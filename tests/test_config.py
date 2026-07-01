from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from chef_human.config import Settings, load_settings
from chef_human.llm import create_backend


class TestSettings:
    def test_defaults(self):
        s = Settings()
        assert s.llm_backend == "ollama"
        assert s.ollama_model == "qwen2.5-coder:7b"
        assert s.ollama_host == "http://localhost:11434"
        assert s.llamacpp_model_path is None
        assert s.llamacpp_n_gpu_layers == 0
        assert s.llamacpp_n_threads is None
        assert s.max_context_tokens == 32768
        assert s.max_response_tokens == 4096
        assert s.embed_model == "BAAI/bge-small-en-v1.5"
        assert s.temperature == 0.0
        assert s.workspace == ""
        assert s.max_tool_retries == 3
        assert s.max_agent_steps == 25

    def test_frozen(self):
        s = Settings()
        with pytest.raises(AttributeError):
            s.llm_backend = "llamacpp"  # type: ignore[misc]

    def test_custom_values(self):
        s = Settings(llm_backend="llamacpp", temperature=0.5)
        assert s.llm_backend == "llamacpp"
        assert s.temperature == 0.5

    def test_llamapath_str(self):
        s = Settings(llamacpp_model_path="/models/model.gguf")
        assert s.llamacpp_model_path == "/models/model.gguf"


class TestLoadSettings:
    def test_defaults_when_no_config(self):
        with patch("chef_human.config._load_toml", return_value={}), patch(
            "chef_human.config._load_env", return_value={}
        ):
            s = load_settings()
            assert s.llm_backend == "ollama"

    def test_toml_overrides_defaults(self):
        toml_data = {"llm_backend": "llamacpp", "temperature": 0.7}
        with patch("chef_human.config._load_toml", return_value=toml_data), patch(
            "chef_human.config._load_env", return_value={}
        ):
            s = load_settings()
            assert s.llm_backend == "llamacpp"
            assert s.temperature == 0.7

    def test_env_overrides_toml(self):
        toml_data = {"llm_backend": "llamacpp", "temperature": 0.7}
        env_data = {"llm_backend": "ollama"}
        with patch("chef_human.config._load_toml", return_value=toml_data), patch(
            "chef_human.config._load_env", return_value=env_data
        ):
            s = load_settings()
            assert s.llm_backend == "ollama"
            assert s.temperature == 0.7

    def test_workspace_from_str(self):
        toml_data = {"workspace": "/home/user/project"}
        with patch("chef_human.config._load_toml", return_value=toml_data), patch(
            "chef_human.config._load_env", return_value={}
        ):
            s = load_settings()
            assert s.workspace == "/home/user/project"


class TestLoadToml:
    def test_returns_empty_when_file_missing(self):
        with patch("pathlib.Path.exists", return_value=False):
            from chef_human.config import _load_toml

            assert _load_toml("nonexistent.toml") == {}

    def test_parses_chef_human_section(self, tmp_path: Path):
        toml_file = tmp_path / "config.toml"
        toml_file.write_text(
            "[chef_human]\nllm_backend = 'llamacpp'\ntemperature = 0.5\n"
        )
        from chef_human.config import _load_toml

        data = _load_toml(str(toml_file))
        assert data["llm_backend"] == "llamacpp"
        assert data["temperature"] == 0.5

    def test_ignores_other_sections(self, tmp_path: Path):
        toml_file = tmp_path / "config.toml"
        toml_file.write_text("[other]\nfoo = 'bar'\n[chef_human]\nx = 1\n")
        from chef_human.config import _load_toml

        data = _load_toml(str(toml_file))
        assert "foo" not in data
        assert data["x"] == 1

    def test_empty_chef_human_section(self, tmp_path: Path):
        toml_file = tmp_path / "config.toml"
        toml_file.write_text("[chef_human]\n")
        from chef_human.config import _load_toml

        data = _load_toml(str(toml_file))
        assert data == {}


class TestLoadEnv:
    def test_reads_chef_prefixed_vars(self):
        with patch.dict(os.environ, {"CHEF_TEMPERATURE": "0.9", "CHEF_LLM_BACKEND": "llamacpp", "OTHER_VAR": "x"}):
            from chef_human.config import _load_env

            data = _load_env()
            assert data["temperature"] == 0.9
            assert data["llm_backend"] == "llamacpp"
            assert "other_var" not in data

    def test_empty_when_no_chef_vars(self):
        with patch.dict(os.environ, {"PATH": "/usr/bin"}):
            from chef_human.config import _load_env

            data = _load_env()
            assert data == {}


class TestParseValue:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("true", True),
            ("True", True),
            ("FALSE", False),
            ("1", True),
            ("0", False),
            ("42", 42),
            ("3.14", 3.14),
            ("hello", "hello"),
            ("http://localhost", "http://localhost"),
        ],
    )
    def test_parse_various_types(self, raw: str, expected: str | int | float | bool):
        from chef_human.config import _parse_value

        assert _parse_value(raw) == expected


class TestCreateBackend:
    def test_returns_ollama_backend(self):
        with patch(
            "chef_human.llm.ollama_backend.OllamaBackend"
        ) as MockOllamaBackend:
            with patch("chef_human.llm.settings") as mock_settings:
                mock_settings.llm_backend = "ollama"
                mock_settings.ollama_model = "test-model"
                mock_settings.ollama_host = "http://localhost:11434"
                _ = create_backend()
                MockOllamaBackend.assert_called_once_with(
                    model="test-model", host="http://localhost:11434"
                )

    def test_raises_for_unknown_backend(self):
        with patch("chef_human.llm.settings") as mock_settings:
            mock_settings.llm_backend = "invalid"
            with pytest.raises(ValueError, match="Unknown backend: invalid"):
                create_backend()

    def test_raises_when_llamacpp_model_path_none(self):
        with patch("chef_human.llm.settings") as mock_settings:
            mock_settings.llm_backend = "llamacpp"
            mock_settings.llamacpp_model_path = None
            with pytest.raises(
                ValueError, match="llamacpp_model_path must be set"
            ):
                create_backend()

    def test_returns_llamacpp_backend(self):
        with patch(
            "chef_human.llm.llamacpp_backend.LlamaCppBackend"
        ) as MockLlamaCppBackend:
            with patch("chef_human.llm.settings") as mock_settings:
                mock_settings.llm_backend = "llamacpp"
                mock_settings.llamacpp_model_path = "/models/test.gguf"
                mock_settings.llamacpp_n_gpu_layers = 24
                mock_settings.llamacpp_n_threads = 4
                _ = create_backend()
                MockLlamaCppBackend.assert_called_once_with(
                    model_path="/models/test.gguf",
                    n_gpu_layers=24,
                    n_threads=4,
                )


class TestProjectConfig:
    def test_find_project_config_not_found(self, tmp_path):
        from chef_human.config import _find_project_config
        assert _find_project_config(tmp_path) is None

    def test_find_project_config_found(self, tmp_path):
        from chef_human.config import _find_project_config
        cfg_dir = tmp_path / ".chef-human"
        cfg_dir.mkdir()
        cfg_file = cfg_dir / "config.toml"
        cfg_file.write_text("[chef_human]\nllm_backend = \"llamacpp\"\n")
        found = _find_project_config(tmp_path)
        assert found is not None
        assert found == cfg_file

    def test_find_project_config_walks_up(self, tmp_path):
        from chef_human.config import _find_project_config
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True)
        cfg_dir = tmp_path / ".chef-human"
        cfg_dir.mkdir()
        (cfg_dir / "config.toml").write_text("[chef_human]\n")
        found = _find_project_config(nested)
        assert found is not None

    def test_project_config_merges_before_env(self):
        from chef_human.config import load_settings
        toml_data = {"llm_backend": "ollama", "temperature": 0.5}
        with (
            patch("chef_human.config._load_toml", return_value=toml_data),
            patch("chef_human.config._load_env", return_value={}),
            patch("chef_human.config._find_project_config", return_value=None),
        ):
            s = load_settings()
            assert s.temperature == 0.5

    def test_env_still_overrides_project_config(self):
        from chef_human.config import load_settings
        toml_data = {"llm_backend": "ollama"}
        env_data = {"llm_backend": "llamacpp"}
        project_data = {"llm_backend": "ollama"}
        with (
            patch("chef_human.config._load_toml", side_effect=[toml_data, project_data]),
            patch("chef_human.config._load_env", return_value=env_data),
            patch("chef_human.config._find_project_config", return_value=Path("proj.toml")),
        ):
            s = load_settings()
            assert s.llm_backend == "llamacpp"
