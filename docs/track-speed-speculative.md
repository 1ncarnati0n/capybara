# 속도 트랙 — Speculative Decoding (EXAONE 3.5 7.8B + 2.4B draft)

## 개요 / 언제 쓰는가

현행 EXAONE 3.5 7.8B Q6_K의 **출력 품질을 그대로 유지**하면서 **토큰/초를 1.5–2배로** 끌어올린다. 일반 소설·에세이 등 책 한 권을 빠르게 끝내고 싶을 때 적합. Rejection sampling 방식이므로 draft 모델이 어떤 토큰을 예측하든 최종 출력 분포는 7.8B target 모델과 **수학적으로 동일**.

**원리 요약**: 작고 빠른 draft(2.4B)가 N개 토큰을 미리 추측하면, 큰 target(7.8B)이 한 번의 forward로 검증한다. acceptance rate가 높을수록 가속이 커지며, 같은 EXAONE 3.5 패밀리라 한국어 출력 분포가 유사 → 통상 acceptance 60–75%대.

## 요구 모델 파일

| 역할 | Hugging Face 리포 | 파일 | 디스크 |
|---|---|---|---|
| target | `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-GGUF` (또는 현행) | `EXAONE-3.5-7.8B-Instruct-Q6_K.gguf` | 6.4GB |
| draft | `bartowski/EXAONE-3.5-2.4B-Instruct-GGUF` | `EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf` | ~1.5GB |

target은 이미 `models/EXAONE-3.5-7.8B-Instruct-Q6_K.gguf`에 있을 것.

## 다운로드 명령

```bash
# draft 모델만 추가로 받는다
uv run --extra download hf download bartowski/EXAONE-3.5-2.4B-Instruct-GGUF \
  --include "EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf" \
  --local-dir models
```

다운로드 후:

```bash
ls -lh models/EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf
```

## llama.cpp 기동 옵션

`scripts/lm-spec.sh` 형태로 새 스크립트를 만든다 (port 9072 권장 — 기존 EXAONE 7.8B 단독 9070, Yanolja 9090, EXAONE 4.0.1 9081과 충돌 회피).

```bash
#!/usr/bin/env bash
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
BIN="${LLAMA_SERVER_BIN:-${DEFAULT_BIN:-llama-server}}"

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
```

핵심 플래그:
- `--model-draft`: speculative draft GGUF
- `--gpu-layers-draft 999`: draft도 GPU에 전부 올려 latency 최소화 (CPU에 둘 경우 가속 효과 사라짐)
- `--draft-max 16 --draft-min 4`: 한 번에 추측할 토큰 길이 범위. 한국어 EUC/KS 토큰화 특성상 16이 안전한 상한
- `--cache-type-k/v q8_0`: KV 캐시 양자화 (현행 lm-server.sh와 동일)

llama.cpp의 speculative 옵션 이름은 빌드에 따라 `--model-draft` ↔ `--draft` ↔ `--speculative-draft`로 변동된 적 있음 — `~/llama.cpp/build/bin/llama-server --help | grep -i draft`로 실제 빌드의 표기 확인 권장.

## 권장 생성 파라미터

현행 7.8B 단독과 동일하게:

| 항목 | 값 |
|---|---|
| temperature | 0.25 |
| top_p | 0.9 |
| timeout (s) | 300 |
| max_group_tokens | 2200 |
| profile 라벨 | "속도" |

품질이 target과 동일하므로 파라미터를 바꿀 이유가 없다.

## VRAM 예산

| 항목 | 메모리 |
|---|---|
| target Q6_K | ~6.4GB |
| draft Q4_K_M | ~1.5GB |
| KV 캐시 (16K ctx, q8_0) | ~1.5GB |
| llama.cpp 오버헤드 | ~0.5GB |
| **합계** | **~10GB** |

16GB에 약 6GB 여유 → 다른 GPU 작업과 병행 가능.

## 앱 프리셋 추가 예시

`app.py`의 `MODEL_PRESETS` 딕셔너리에 다음 항목을 추가:

```python
"EXAONE 3.5 7.8B + 2.4B draft (속도)": {
    "base_url": "http://127.0.0.1:9072/v1",
    "model": "exaone-3.5-7.8b-spec",
    "temperature": 0.25,
    "top_p": 0.9,
    "timeout": 300,
    "max_group_tokens": 2200,
    "profile": "속도",
},
```

`_server_command()`에 분기 추가:

```python
if "속도" in preset_name or "spec" in preset_name.lower():
    return "scripts/lm-spec.sh"
```

## 검증 절차

1. **부팅 확인** — `scripts/lm-spec.sh`를 띄우고 stderr에 `loaded draft model` / `n_gpu_layers_draft = 999` 류 로그가 보이는지
2. **VRAM** — 다른 터미널에서 `nvidia-smi` → ~10GB 사용
3. **엔드포인트** — `curl -H 'Authorization: Bearer sk-local' http://127.0.0.1:9072/v1/models`가 JSON으로 응답
4. **속도 비교** — 같은 단편 EPUB을 (a) `cache/llm` 비운 상태로 7.8B 단독 9070, (b) speculative 9072로 각각 번역 → `core/progress.py` 로그의 청크 시간 비교. 1.5× 미만이면 acceptance rate 점검
5. **품질 비교** — 같은 챕터를 두 서버로 따로 번역해 한국어 자연스러움·고유명사 일관성 비교 (이론상 동일)
6. **acceptance rate 관찰** — `--metrics` 켜둔 상태에서 `curl http://127.0.0.1:9072/metrics | grep -i draft` 또는 llama-server stderr의 `accepted` 카운터 (빌드별 표기 다름)

## 알려진 함정

- **draft가 CPU로 빠지면 가속 사라짐**: `--gpu-layers-draft` 누락 시 draft가 CPU에서 돌아 오히려 느려진다. 부팅 로그 확인 필수
- **draft 양자화가 너무 낮으면 acceptance↓**: 2.4B Q4_K_M이 권장 하한. Q3로 더 줄이면 acceptance가 떨어져 가속이 무의미해질 수 있음
- **draft와 target의 토크나이저가 달라서는 안 됨**: 같은 EXAONE 3.5 패밀리이므로 OK. 다른 패밀리(예: Qwen) draft는 사용 불가
- **장문 컨텍스트에서 acceptance↓**: 한국어 긴 출력에서 draft 예측이 빗나가기 시작하면 가속이 1.2× 수준까지 떨어질 수 있음. 챕터별 로그로 모니터
- **llama.cpp 빌드별 옵션명 차이**: `--model-draft`, `--draft-max`, `--draft-min` 모두 비교적 최근에 정착된 이름. `--help`로 실제 빌드 확인

## 레퍼런스

- llama.cpp speculative decoding: <https://github.com/ggml-org/llama.cpp/tree/master/examples/speculative>
- bartowski 2.4B GGUF: <https://huggingface.co/bartowski/EXAONE-3.5-2.4B-Instruct-GGUF>
- 공식 7.8B GGUF: <https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-GGUF>
- EXAONE 3.5 기술 보고서: <https://arxiv.org/abs/2412.04862>
