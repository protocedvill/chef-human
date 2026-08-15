from __future__ import annotations

from chef_human.agent.hardware import HardwareInfo
from chef_human.agent.model_advisor import (
    MODEL_CATALOG,
    find_model_spec,
    recommend_model,
)


class TestFindModelSpec:
    def test_finds_known_model(self):
        spec = find_model_spec("qwen2.5-coder:7b")
        assert spec is not None
        assert spec.display_name == "Qwen2.5-Coder-7B-Instruct"

    def test_unknown_model_returns_none(self):
        assert find_model_spec("some-random-model:1b") is None

    def test_catalog_is_nonempty(self):
        assert len(MODEL_CATALOG) > 0

    def test_catalog_tags_are_unique(self):
        tags = [m.ollama_tag for m in MODEL_CATALOG]
        assert len(tags) == len(set(tags))


class TestRecommendModel:
    def test_no_hardware_detected_returns_none(self):
        hw = HardwareInfo(ram_gb=None, vram_gb=None)
        assert recommend_model("qwen2.5-coder:7b", hardware=hw) is None

    def test_low_ram_no_upgrade_available(self):
        """With very little memory, nothing in the catalog fits -- no
        recommendation should be made (not even a downgrade)."""
        hw = HardwareInfo(ram_gb=4.0, vram_gb=None)
        assert recommend_model("qwen2.5-coder:7b", hardware=hw) is None

    def test_recommends_better_model_when_headroom_available(self):
        hw = HardwareInfo(ram_gb=64.0, vram_gb=None)
        rec = recommend_model("qwen2.5-coder:7b", hardware=hw)
        assert rec is not None
        assert rec.suggested.quality >= 5
        assert rec.current_model == "qwen2.5-coder:7b"

    def test_already_on_best_fit_returns_none(self):
        """Plenty of RAM, but already running the best-quality model that
        fits within budget -- nothing better to suggest."""
        hw = HardwareInfo(ram_gb=26.0, vram_gb=None)  # budget ~20.8 GB
        rec = recommend_model("qwen2.5-coder:14b", hardware=hw)
        assert rec is None

    def test_does_not_recommend_model_requiring_more_than_budget(self):
        hw = HardwareInfo(ram_gb=10.0, vram_gb=None)  # budget = 8 GB
        rec = recommend_model("qwen2.5-coder:7b", hardware=hw)
        # 7b (8GB) is already at budget; nothing bigger fits under 8GB budget.
        assert rec is None

    def test_uses_vram_when_larger_than_ram(self):
        hw = HardwareInfo(ram_gb=16.0, vram_gb=40.0)
        rec = recommend_model("qwen2.5-coder:7b", hardware=hw)
        assert rec is not None
        assert "40" in rec.reason or rec.hardware.vram_gb == 40.0

    def test_unknown_current_model_still_gets_recommendation(self):
        hw = HardwareInfo(ram_gb=64.0, vram_gb=None)
        rec = recommend_model("some-obscure-model:1b", hardware=hw)
        assert rec is not None

    def test_never_recommends_the_current_model_itself(self):
        hw = HardwareInfo(ram_gb=64.0, vram_gb=None)
        rec = recommend_model("qwen2.5-coder:32b", hardware=hw)
        if rec is not None:
            assert rec.suggested.ollama_tag != "qwen2.5-coder:32b"

    def test_reason_mentions_detected_capacity(self):
        hw = HardwareInfo(ram_gb=64.0, vram_gb=None)
        rec = recommend_model("qwen2.5-coder:7b", hardware=hw)
        assert rec is not None
        assert "64" in rec.reason
