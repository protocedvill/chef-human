from __future__ import annotations

from pathlib import Path

from chef_human.agent.doctor import DoctorProbes, run_doctor


def _ok_probes(**overrides) -> DoctorProbes:
    defaults = dict(
        python_version_info=(3, 12, 5),
        package_version=lambda: "0.1.0",
        workspace_writable=lambda path: True,
        git_present=lambda path: True,
        module_available=lambda name: True,
        ollama_list=lambda host: ["qwen3.6:35b-a3b"],
    )
    defaults.update(overrides)
    return DoctorProbes(**defaults)


def test_all_checks_pass_reports_ready() -> None:
    report = run_doctor(probes=_ok_probes())

    assert report.ready is True
    assert report.exit_code == 0
    assert report.invalid_config is False
    assert not any(c.status == "fail" for c in report.checks)
    assert "run" in report.next_command


def test_unsupported_python_version_fails() -> None:
    report = run_doctor(probes=_ok_probes(python_version_info=(3, 10, 0)))

    check = next(c for c in report.checks if c.name == "python_version")
    assert check.status == "fail"
    assert report.ready is False
    assert report.exit_code == 1


def test_unreachable_ollama_server_fails_with_remedy() -> None:
    def _raise(host: str) -> list[str]:
        raise ConnectionError("refused")

    report = run_doctor(probes=_ok_probes(ollama_list=_raise))

    server_check = next(c for c in report.checks if c.name == "ollama_server")
    model_check = next(c for c in report.checks if c.name == "ollama_model")
    assert server_check.status == "fail"
    assert "refused" in server_check.message
    assert model_check.status == "fail"
    assert report.exit_code == 1


def test_model_not_pulled_fails() -> None:
    report = run_doctor(probes=_ok_probes(ollama_list=lambda host: ["some-other-model"]))

    check = next(c for c in report.checks if c.name == "ollama_model")
    assert check.status == "fail"
    assert "ollama pull" in (check.remedy or "")


def test_unwritable_workspace_fails() -> None:
    report = run_doctor(probes=_ok_probes(workspace_writable=lambda path: False))

    check = next(c for c in report.checks if c.name == "workspace")
    assert check.status == "fail"
    assert report.exit_code == 1


def test_missing_git_is_a_warning_not_a_failure() -> None:
    report = run_doctor(probes=_ok_probes(git_present=lambda path: False))

    check = next(c for c in report.checks if c.name == "git")
    assert check.status == "warn"
    assert report.ready is True


def test_missing_optional_capability_is_a_warning() -> None:
    report = run_doctor(probes=_ok_probes(module_available=lambda name: name != "faiss"))

    check = next(c for c in report.checks if c.name == "rag_vector_store")
    assert check.status == "warn"
    assert "rag" in (check.remedy or "")
    assert report.ready is True


def test_invalid_config_contents_return_exit_code_two(tmp_path: Path) -> None:
    bad_config = tmp_path / "config.toml"
    bad_config.write_text("this is not valid toml [[[")

    report = run_doctor(config_path=str(bad_config), probes=_ok_probes())

    assert report.invalid_config is True
    assert report.exit_code == 2
    assert report.checks[0].status == "fail"


def test_unknown_config_key_returns_exit_code_two(tmp_path: Path) -> None:
    bad_config = tmp_path / "config.toml"
    bad_config.write_text('[chef_human]\nnot_a_real_setting = "x"\n')

    report = run_doctor(config_path=str(bad_config), probes=_ok_probes())

    assert report.invalid_config is True
    assert report.exit_code == 2


def test_to_dict_schema() -> None:
    report = run_doctor(probes=_ok_probes())
    payload = report.to_dict()

    assert payload["schema_version"] == 1
    assert isinstance(payload["ready"], bool)
    assert isinstance(payload["checks"], list)
    assert all({"name", "status", "message", "remedy"} <= c.keys() for c in payload["checks"])


def test_workspace_override_is_resolved(tmp_path: Path) -> None:
    seen: list[Path] = []
    probes = _ok_probes(workspace_writable=lambda path: (seen.append(path), True)[1])

    run_doctor(workspace=str(tmp_path), probes=probes)

    assert seen[0] == tmp_path.resolve()
