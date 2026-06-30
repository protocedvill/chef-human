from chef_human.config import settings
from chef_human.llm.backend import LLMBackend


def create_backend() -> LLMBackend:
    if settings.llm_backend == "ollama":
        from chef_human.llm.ollama_backend import OllamaBackend

        return OllamaBackend(
            model=settings.ollama_model,
            host=settings.ollama_host,
        )
    elif settings.llm_backend == "llamacpp":
        from chef_human.llm.llamacpp_backend import LlamaCppBackend

        if settings.llamacpp_model_path is None:
            raise ValueError(
                "llamacpp_model_path must be set when backend is 'llamacpp'"
            )
        return LlamaCppBackend(
            model_path=settings.llamacpp_model_path,
            n_gpu_layers=settings.llamacpp_n_gpu_layers,
            n_threads=settings.llamacpp_n_threads,
        )
    else:
        raise ValueError(f"Unknown backend: {settings.llm_backend}")
