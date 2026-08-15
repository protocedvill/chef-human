from chef_human import config
from chef_human.config import Settings
from chef_human.llm.backend import LLMBackend


def create_backend(
    model_override: str | None = None,
    *,
    settings: Settings | None = None,
) -> LLMBackend:
    cfg = settings or config.settings
    if cfg.llm_backend == "ollama":
        from chef_human.llm.ollama_backend import OllamaBackend

        return OllamaBackend(
            model=model_override or cfg.ollama_model,
            host=cfg.ollama_host,
        )
    elif cfg.llm_backend == "llamacpp":
        from chef_human.llm.llamacpp_backend import LlamaCppBackend

        model_path = model_override or cfg.llamacpp_model_path
        if model_path is None:
            raise ValueError(
                "llamacpp_model_path must be set when backend is 'llamacpp'"
            )
        return LlamaCppBackend(
            model_path=model_path,
            n_gpu_layers=cfg.llamacpp_n_gpu_layers,
            n_threads=cfg.llamacpp_n_threads,
        )
    else:
        raise ValueError(f"Unknown backend: {cfg.llm_backend}")


def create_planner_backend(
    main_backend: LLMBackend,
    *,
    settings: Settings | None = None,
) -> LLMBackend:
    """Backend for the Planner's judge calls (generate_plan, verify_step,
    update_plan). These only need a short structured answer (verify_step
    caps at max_tokens=100) and run on nearly every ReAct turn, so sharing
    the main reasoning model wastes a lot of per-turn latency. Builds a
    dedicated backend if a separate planner model is configured via
    settings.planner_ollama_model / settings.planner_llamacpp_model_path;
    otherwise reuses main_backend unchanged -- no extra connectivity check,
    no behavior change from today. Note: for llama.cpp, a configured
    planner model means a second GGUF fully loaded into memory/VRAM
    (unlike Ollama's cheap second client to the same server) -- real extra
    resource cost, not just a config toggle."""
    cfg = settings or config.settings
    if cfg.llm_backend == "ollama" and cfg.planner_ollama_model:
        return create_backend(model_override=cfg.planner_ollama_model, settings=cfg)
    if cfg.llm_backend == "llamacpp" and cfg.planner_llamacpp_model_path:
        return create_backend(model_override=cfg.planner_llamacpp_model_path, settings=cfg)
    return main_backend
