# AGENTS.md

## Environment

- Python venv: `/home/louis/chef-human/.env` (not `.venv` — already has the project installed
  editable via `pip install -e ".[dev,indexing]"`). Use `.env/bin/python`, `.env/bin/pytest`,
  `.env/bin/ruff`, etc., or `source .env/bin/activate`. It does NOT have the `rag`/`embeddings`
  extras (`faiss-cpu`, `sentence-transformers`) installed — add with
  `.env/bin/pip install -e ".[rag,embeddings]"` if RAG-path tests/work are needed.
- Ollama: a systemd-managed `ollama serve` is already running on `http://localhost:11434` — no
  need to start one. Check loaded/available models with `ollama list` / `ollama ps`.
