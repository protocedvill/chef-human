# Installation

## Requirements

- **Linux** (supported prototype platform)
- **Python** 3.12 or 3.13
- **Ollama** (supported backend)

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
ollama pull qwen3.8:27b
```

### 3. Create an environment and install Chef Human

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

### 4. Verify

```bash
python -m chef_human --help
```

---

## Experimental capabilities

The base install uses regex symbol extraction when tree-sitter is absent. Experimental features are
installed explicitly:

```bash
python -m pip install -e '.[indexing]'  # tree-sitter symbol extraction/refactoring
python -m pip install -e '.[rag]'       # embeddings + NumPy/FAISS retrieval
```

RAG is also opt-in at runtime:

```toml
[chef_human]
rag_enabled = true
```

### llama.cpp backend

### llama.cpp (no Ollama dependency)

Additional install step:

```bash
python -m pip install -e '.[llamacpp]'
```

Then download a GGUF model, e.g. from Hugging Face:

```bash
# Example: Qwen2.5-Coder-7B-Instruct Q4_K_M
wget https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/qwen2.5-coder-7b-instruct-q4_k_m.gguf
```

Set `llamacpp_model_path` in `config.toml` to point to the downloaded file.

---

## Setup Script

An automated setup script is available:

```bash
bash scripts/setup.sh
```

This creates `.venv`, installs the required package dependencies, and verifies the command. It does
not execute a remote installer or download a multi-gigabyte model. Those actions remain explicit.

---

## Configuration

Configuration is managed via `config.toml` in the project root:

```toml
[chef_human]
llm_backend = "ollama"
ollama_model = "qwen3.8:27b"
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

### Unsupported Python version

Use Python 3.12 or 3.13. Pre-release and newer Python versions are intentionally excluded until the
project's binary optional dependencies publish compatible wheels.

### Ollama not found

Ensure Ollama is running:

```bash
ollama list     # should list models or show "No models"
```

If the command is not found, install Ollama first (see step 1 above).

### Model not responding

Verify the model is pulled:

```bash
ollama pull qwen3.8:27b
```
