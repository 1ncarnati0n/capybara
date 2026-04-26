#!/usr/bin/env bash
# Launch llama.cpp server with EXAONE 3.5 7.8B target + 2.4B draft speculative decoding.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${TARGET:-$ROOT/models/EXAONE-3.5-7.8B-Instruct-Q6_K.gguf}"
DRAFT="${DRAFT:-$ROOT/models/EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf}"
PORT="${PORT:-9072}"
HOST="${HOST:-127.0.0.1}"
CTX="${CTX:-16384}"
ALIAS="${ALIAS:-exaone-3.5-7.8b-spec}"
API_KEY="${API_KEY:-sk-local}"

DEFAULT_BIN="$HOME/llama.cpp/build/bin/llama-server"
if [[ -z "${LLAMA_SERVER_BIN:-}" && -x "$DEFAULT_BIN" ]]; then
  BIN="$DEFAULT_BIN"
else
  BIN="${LLAMA_SERVER_BIN:-llama-server}"
fi

if [[ ! -f "$TARGET" ]]; then
  echo "Target model not found: $TARGET" >&2
  exit 1
fi

if [[ ! -f "$DRAFT" ]]; then
  echo "Draft model not found: $DRAFT" >&2
  echo "Download EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf into models/ or set DRAFT=/path/to/draft.gguf." >&2
  exit 1
fi

exec "$BIN" \
  -m "$TARGET" \
  --model-draft "$DRAFT" \
  --alias "$ALIAS" \
  --host "$HOST" --port "$PORT" \
  --api-key "$API_KEY" \
  --n-gpu-layers 999 \
  --gpu-layers-draft 999 \
  --ctx-size "$CTX" \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --draft-max 16 \
  --draft-min 4 \
  --parallel 1 \
  --no-mmap \
  --metrics
