#!/usr/bin/env bash
set -euo pipefail

echo "=== chef-human setup ==="

# 1. Check Python version
PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
    echo "Error: Python not found. Install Python 3.12+ first."
    exit 1
fi

PYVER=$($PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
MAJOR=$(echo "$PYVER" | cut -d. -f1)
MINOR=$(echo "$PYVER" | cut -d. -f2)

if [ "$MAJOR" -lt 3 ] || { [ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 12 ]; }; then
    echo "Error: Python 3.12+ required (found $PYVER)"
    exit 1
fi
echo "Python $PYVER found."

# 2. Install package
echo ""
echo "Installing chef-human (dev mode)..."
$PYTHON -m pip install -e ".[dev]" 2>&1 || {
    echo ""
    echo "WARNING: Full install failed. This is likely due to missing"
    echo "pre-built wheels for your Python version (e.g., pydantic-core"
    echo "for Python beta/RC releases)."
    echo ""
    echo "Falling back to --no-deps install..."
    $PYTHON -m pip install -e ".[dev]" --no-deps
    echo "Installing core dependencies individually..."
    $PYTHON -m pip install --no-deps ollama rich click python-dotenv tomli pytest ruff pyright pytest-asyncio 2>/dev/null || true
    echo "Core dependencies installed (some optional deps may be missing)."
    echo "To install optional extras later:"
    echo "  pip install -e \".[llamacpp]\"   # llama.cpp backend"
    echo "  pip install -e \".[embeddings]\"  # sentence-transformers for RAG"
}

# 3. Check/install Ollama
echo ""
if command -v ollama &>/dev/null; then
    echo "Ollama found."
else
    echo "Ollama not found."
    echo "Install it manually:"
    echo "  curl -fsSL https://ollama.com/install.sh | sh"
    echo "  (or download from https://ollama.com)"
    echo ""
    read -rp "Install Ollama now? [y/N] " yn
    if [[ "$yn" =~ ^[yY] ]]; then
        curl -fsSL https://ollama.com/install.sh | sh
    else
        echo "Skipping Ollama install. You'll need it before using chef-human."
    fi
fi

# 4. Pull model
if command -v ollama &>/dev/null; then
    MODEL="${CHEF_OLLAMA_MODEL:-qwen2.5-coder:7b}"
    echo ""
    echo "Pulling model: $MODEL (this may take a while)..."
    ollama pull "$MODEL"
    echo "Model $MODEL ready."
fi

# 5. Verify
echo ""
echo "=== Setup complete ==="
echo ""
echo "Verify with:"
echo "  python -c 'from chef_human.llm import create_backend; b = create_backend(); print(b.model_name)'"
echo ""
echo "Next steps:"
echo "  - Read docs/INSTALL.md  for detailed setup guide"
echo "  - Read docs/USAGE.md    for usage examples"
