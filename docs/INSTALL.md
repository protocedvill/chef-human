# Installation

## Requirements

- **Python** 3.12+
- **Ollama** (recommended backend) or llama.cpp

---

## Quick Start (Ollama)

The simplest way to get started is with the Ollama backend.

### 1. Install Ollama

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS / Windows: download from https://ollama.com
```

### 2. Pull a model

```bash
ollama pull qwen2.5-coder:7b
```

### 3. Install chef-human

```bash
pip install -e ".[dev]"
```

### 4. Verify

```bash
python -c "from chef_human.llm import create_backend; b = create_backend(); print(b.model_name)"
```

---

## Alternative Backends

### llama.cpp (no Ollama dependency)

Additional install step:

```bash
pip install -e ".[dev,llamacpp]"
```

Then download a GGUF model, e.g. from Hugging Face:

```bash
# Example: Qwen2.5-Coder-7B-Instruct Q4_K_M
wget https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf
```

Set `llamacpp_model_path` in `config.toml` to point to the downloaded file.

### Embeddings (for RAG / Phase 3)

```bash
pip install -e ".[embeddings]"
```

---

## Setup Script

An automated setup script is available:

```bash
bash scripts/setup.sh
```

This will:
1. Check Python version
2. Install chef-human with dev dependencies
3. Install Ollama if not present (prompts for confirmation)
4. Pull the recommended model
5. Verify installation

---

## Configuration

Configuration is managed via `config.toml` in the project root:

```toml
[chef_human]
llm_backend = "ollama"
ollama_model = "qwen2.5-coder:7b"
ollama_host = "http://localhost:11434"
max_context_tokens = 32768
temperature = 0.0
```

All settings can also be overridden via environment variables with a `CHEF_` prefix:

```bash
export CHEF_LLM_BACKEND=llamacpp
export CHEF_LLAMACPP_MODEL_PATH=/path/to/model.gguf
```

---

## Troubleshooting

### pydantic-core build fails

If you see an error building `pydantic-core` during install, your environment likely lacks a Rust compiler or a pre-built wheel for your Python version. Workarounds:

- Install pydantic-core from a pre-built wheel (if available for your platform)
- Install Rust via `rustup`
- Use a standard Python version (3.12 or 3.13) rather than a beta/RC
- Install with `--no-deps` and manually install only the packages you need:

```bash
pip install -e ".[dev]" --no-deps
pip install --no-deps ollama rich click python-dotenv tomli pytest ruff pyright pytest-asyncio
```

### Ollama not found

Ensure Ollama is running:

```bash
ollama list     # should list models or show "No models"
```

If the command is not found, install Ollama first (see step 1 above).

### Model not responding

Verify the model is pulled:

```bash
ollama pull qwen2.5-coder:7b
```
