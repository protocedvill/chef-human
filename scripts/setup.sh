#!/usr/bin/env bash
set -euo pipefail

echo "=== Chef Human setup ==="

PYTHON_BIN="${CH_INSTALL_PYTHON:-$(command -v python3 || command -v python || true)}"
if [[ -z "$PYTHON_BIN" ]]; then
    echo "Error: Python was not found. Install Python 3.12 or 3.13 first." >&2
    exit 1
fi

PYTHON_VERSION="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PYTHON_VERSION" != "3.12" && "$PYTHON_VERSION" != "3.13" ]]; then
    echo "Error: Chef Human currently supports Python 3.12 and 3.13 (found $PYTHON_VERSION)." >&2
    echo "Set CH_INSTALL_PYTHON to a supported interpreter and run this script again." >&2
    exit 1
fi

VENV_DIR="${CH_INSTALL_VENV:-.venv}"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "Creating virtual environment at $VENV_DIR with Python $PYTHON_VERSION..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    VENV_VERSION="$($VENV_DIR/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "$VENV_VERSION" != "$PYTHON_VERSION" ]]; then
        echo "Error: Existing environment at $VENV_DIR uses Python $VENV_VERSION," >&2
        echo "but CH_INSTALL_PYTHON selected Python $PYTHON_VERSION." >&2
        echo "Choose a different CH_INSTALL_VENV or remove/recreate that environment explicitly." >&2
        exit 1
    fi
fi

VENV_PYTHON="$VENV_DIR/bin/python"
echo "Installing Chef Human and required dependencies..."
"$VENV_PYTHON" -m pip install --upgrade pip
"$VENV_PYTHON" -m pip install -e .

echo "Verifying the installed command..."
"$VENV_PYTHON" -m chef_human --help >/dev/null

echo
echo "Chef Human is installed in $VENV_DIR."
echo "Activate it with: source $VENV_DIR/bin/activate"

if command -v ollama >/dev/null 2>&1; then
    echo "Ollama is installed. Check the recommended model with: ollama list"
else
    echo "Ollama is not installed yet. Follow https://ollama.com/download before running an agent task."
fi

echo "Then pull the default model explicitly: ollama pull qwen2.5-coder:7b"
echo "Experimental extras: pip install -e '.[indexing]', '.[rag]', or '.[llamacpp]'"
