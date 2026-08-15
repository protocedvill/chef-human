from __future__ import annotations

from dataclasses import dataclass

from chef_human.agent.hardware import HardwareInfo, detect_hardware

# Fraction of detected capacity we're willing to recommend using -- leaves
# headroom for the OS, the embedding model, and other running applications.
_BUDGET_FRACTION = 0.8


@dataclass(frozen=True)
class ModelSpec:
    ollama_tag: str
    display_name: str
    size_b: float
    """Parameter count in billions."""
    min_ram_gb: float
    """Comfortable RAM/VRAM budget to run this model well (quantized)."""
    quality: int
    """Relative coding quality, 1-5. See PLAN.md's Model Selection Guide."""
    license: str


# Mirrors PLAN.md's "Model Selection Guide" table.
MODEL_CATALOG: tuple[ModelSpec, ...] = (
    ModelSpec("qwen2.5-coder:7b", "Qwen2.5-Coder-7B-Instruct", 7, 8, 4, "Apache 2.0"),
    ModelSpec("codellama:13b", "CodeLlama-13B-Instruct", 13, 10, 3, "Llama 2"),
    ModelSpec("deepseek-coder-v2:16b", "DeepSeek-Coder-V2-Lite", 16, 12, 4, "MIT"),
    ModelSpec("starcoder2:15b", "StarCoder2-15B", 15, 12, 3, "StarCoder"),
    ModelSpec("qwen2.5-coder:14b", "Qwen2.5-Coder-14B", 14, 12, 5, "Apache 2.0"),
    ModelSpec("qwen2.5-coder:32b", "Qwen2.5-Coder-32B", 32, 24, 5, "Apache 2.0"),
    ModelSpec("deepseek-coder:33b", "DeepSeek-Coder-33B", 33, 24, 5, "MIT"),
    ModelSpec("mixtral:8x7b", "Mixtral-8x7B", 47, 32, 4, "Apache 2.0"),
)


def find_model_spec(tag: str) -> ModelSpec | None:
    return next((m for m in MODEL_CATALOG if m.ollama_tag == tag), None)


@dataclass(frozen=True)
class ModelRecommendation:
    current_model: str
    suggested: ModelSpec
    hardware: HardwareInfo
    reason: str


def recommend_model(
    current_model: str,
    hardware: HardwareInfo | None = None,
) -> ModelRecommendation | None:
    """Suggest a higher-quality model than `current_model` if this machine
    has enough RAM/VRAM headroom to comfortably run it.

    Returns None if hardware couldn't be detected, or `current_model` is
    already the best fit this machine can comfortably run.
    """
    hw = hardware if hardware is not None else detect_hardware()
    capacity = hw.capacity_gb()
    if capacity is None:
        return None

    current_spec = find_model_spec(current_model)
    current_quality = current_spec.quality if current_spec else 0
    current_size = current_spec.size_b if current_spec else 0.0

    budget = capacity * _BUDGET_FRACTION

    candidates = [
        m
        for m in MODEL_CATALOG
        if m.ollama_tag != current_model
        and m.min_ram_gb <= budget
        and (
            m.quality > current_quality
            or (m.quality == current_quality and m.size_b > current_size)
        )
    ]
    if not candidates:
        return None

    best = max(candidates, key=lambda m: (m.quality, m.size_b))

    vram_note = f", {hw.vram_gb:.0f} GB VRAM" if hw.vram_gb else ""
    reason = (
        f"Detected ~{capacity:.0f} GB usable memory ({hw.ram_gb or 0:.0f} GB RAM{vram_note}). "
        f"{best.display_name} ({best.size_b:.0f}B, needs ~{best.min_ram_gb:.0f} GB) fits "
        "comfortably and rates higher quality than your current model."
    )
    return ModelRecommendation(
        current_model=current_model, suggested=best, hardware=hw, reason=reason
    )
