#!/usr/bin/env bash
# Launch llama.cpp server for EXAONE 3.5 7.8B on RTX 4080 (16GB).
# Prereqs:
#   - llama.cpp built with CUDA (see https://github.com/ggml-org/llama.cpp).
#   - `llama-server` on PATH, or set LLAMA_SERVER_BIN.
#   - GGUF model present under models/.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-$ROOT/models/EXAONE-3.5-7.8B-Instruct-Q6_K.gguf}"
PORT="${PORT:-9070}"
HOST="${HOST:-127.0.0.1}"
CTX="${CTX:-16384}"
ALIAS="${ALIAS:-exaone-3.5-7.8b-instruct}"
API_KEY="${API_KEY:-sk-local}"

# Default to the locally built binary if one exists, else fall back to PATH.
DEFAULT_BIN="$HOME/llama.cpp/build/bin/llama-server"
if [[ -z "${LLAMA_SERVER_BIN:-}" && -x "$DEFAULT_BIN" ]]; then
  BIN="$DEFAULT_BIN"
else
  BIN="${LLAMA_SERVER_BIN:-llama-server}"
fi

if [[ ! -f "$MODEL" ]]; then
  echo "Model not found: $MODEL" >&2
  echo "Set MODEL=/path/to/model.gguf or place the default EXAONE GGUF under models/." >&2
  exit 1
fi

exec "$BIN" \
  -m "$MODEL" \
  --alias "$ALIAS" \
  --host "$HOST" --port "$PORT" \
  --api-key "$API_KEY" \
  --n-gpu-layers 999 \
  --ctx-size "$CTX" \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --parallel 1 \
  --no-mmap \
  --metrics
