#!/usr/bin/env bash
# Launch llama.cpp server for EXAONE 4.0.1 32B on a 16GB GPU budget.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-$ROOT/models/LGAI-EXAONE_EXAONE-4.0.1-32B-IQ3_XS.gguf}"
PORT="${PORT:-9081}"
HOST="${HOST:-127.0.0.1}"
CTX="${CTX:-4096}"
ALIAS="${ALIAS:-exaone-4.0.1-32b}"
API_KEY="${API_KEY:-sk-local}"

DEFAULT_BIN="$HOME/llama.cpp/build/bin/llama-server"
if [[ -z "${LLAMA_SERVER_BIN:-}" && -x "$DEFAULT_BIN" ]]; then
  BIN="$DEFAULT_BIN"
else
  BIN="${LLAMA_SERVER_BIN:-llama-server}"
fi

if [[ ! -f "$MODEL" ]]; then
  echo "Model not found: $MODEL" >&2
  echo "Set MODEL=/path/to/exaone4.gguf. For 16GB GPUs, start with IQ3_XS or IQ3_M plus a low CTX." >&2
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
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --chat-template-kwargs '{"enable_thinking": false}' \
  --parallel 1 \
  --no-mmap \
  --metrics
