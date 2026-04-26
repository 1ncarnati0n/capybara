# 품질 트랙 — EXAONE 4.0.1 32B IQ3_M

## 개요 / 언제 쓰는가

EXAONE 4.0은 14T 토큰으로 사전학습되어(EXAONE 3.5 32B의 약 2배), 한국어 처리·문맥 일관성·고유명사 표기가 7.8B 대비 큰 폭으로 향상. 짧은 기술서·논문체·매뉴얼처럼 **정확도가 속도보다 중요한 책**에 적합. 단점은 명확함 — RTX 4080 SUPER 16GB에서는 IQ3_M 이하 양자화가 강제되며, 토큰/초가 현행의 1/3~1/4로 떨어진다.

EXAONE 4.0은 hybrid reasoning 모델이지만 **번역 작업에서는 reasoning을 반드시 끈다** (추론 토큰이 번역 출력에 섞이거나 길이만 늘어 손해).

## 요구 모델 파일

| 역할 | Hugging Face 리포 | 파일 | 디스크 |
|---|---|---|---|
| 단독 모델 | `bartowski/LGAI-EXAONE_EXAONE-4.0.1-32B-GGUF` | `LGAI-EXAONE_EXAONE-4.0.1-32B-IQ3_M.gguf` | 14.4GB |

대안 양자화 (16GB 적합 한도):

| 양자화 | 크기 | 비고 |
|---|---|---|
| IQ3_XS | 13.3GB | 더 안전한 KV 여유, 품질 살짝↓ |
| **IQ3_M** | **14.4GB** | **권장** — 품질/메모리 균형 |
| Q3_K_M | 15.5GB | KV 매우 빠듯, ctx 4K 한정 |
| IQ4_XS | 17.2GB | ❌ 16GB 단독 부적합 |

## 다운로드 명령

```bash
uv run --extra download hf download bartowski/LGAI-EXAONE_EXAONE-4.0.1-32B-GGUF \
  --include "LGAI-EXAONE_EXAONE-4.0.1-32B-IQ3_M.gguf" \
  --local-dir models
```

다운로드 후:

```bash
ls -lh models/LGAI-EXAONE_EXAONE-4.0.1-32B-IQ3_M.gguf
```

## llama.cpp 기동 옵션

`scripts/lm-exaone4.sh` (port 9081 권장 — 기존 9070/9072/9080/9090과 충돌 회피).

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-$ROOT/models/LGAI-EXAONE_EXAONE-4.0.1-32B-IQ3_M.gguf}"
PORT="${PORT:-9081}"
HOST="${HOST:-127.0.0.1}"
CTX="${CTX:-8192}"
ALIAS="${ALIAS:-exaone-4.0.1-32b}"
API_KEY="${API_KEY:-sk-local}"

DEFAULT_BIN="$HOME/llama.cpp/build/bin/llama-server"
BIN="${LLAMA_SERVER_BIN:-${DEFAULT_BIN:-llama-server}}"

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
```

핵심 플래그:
- `--ctx-size 8192`: 32B + IQ3_M에서는 ctx를 16K까지 늘리면 KV 폭증으로 OOM 위험. 8K가 안전 상한
- `--cache-type-k/v q4_0`: KV 양자화를 q4까지 더 낮춤 (현행 7.8B의 q8_0보다 공격적). 8K ctx 기준 ~0.8GB 절감
- `--chat-template-kwargs '{"enable_thinking": false}'`: **번역에 reasoning 끔 (필수)**. 4.0은 기본이 enable_thinking=False지만 명시 권장. 4.5는 반대로 기본 True이므로 4.5 이전이라도 명시가 안전
- `--n-gpu-layers 999`: 32B 전체를 GPU에 적재

llama.cpp 요구 빌드: **b5932 이상** ([PR #14630](https://github.com/ggml-org/llama.cpp/pull/14630)). 현재 빌드 확인:

```bash
~/llama.cpp/build/bin/llama-server --version
```

`b5932` 미만이면 `git pull && cmake --build build --config Release` 후 재시도. EXAONE 4.0의 chat template은 일부 빌드에서 워크어라운드가 필요할 수 있음 — 부팅 시 `chat template not supported` 류 stderr가 나오면 `--chat-template exaone4` 명시 또는 [공식 4.0 리포 안내](https://github.com/LG-AI-EXAONE/EXAONE-4.0)의 simplified template 적용.

## 권장 생성 파라미터

| 항목 | 값 | 근거 |
|---|---|---|
| temperature | 0.5 | LG 권고 "non-reasoning 모드는 < 0.6" |
| top_p | 0.9 | 일반 권장값 |
| timeout (s) | 900 | 32B IQ3는 청크당 길어질 수 있음 |
| max_group_tokens | 1500 | 7.8B의 2200보다 보수적 (32B 처리 시간 보정) |
| profile 라벨 | "고품질·신모델" | UI 식별용 |

## VRAM 예산

| 항목 | 메모리 |
|---|---|
| 모델 IQ3_M | ~14.4GB |
| KV 캐시 (8K ctx, q4_0 K+V) | ~0.8GB |
| llama.cpp 오버헤드 | ~0.4GB |
| **합계** | **~15.6GB** |

16GB 중 ~0.4GB 여유 — 다른 GPU 작업과 동시 실행 불가. 부팅 후 `nvidia-smi`로 free memory가 음수에 근접하지 않는지 확인.

ctx를 16K로 늘리고 싶다면 IQ3_XS(13.3GB)로 내리고 `--cache-type q4_0` 유지.

## 앱 프리셋 추가 예시

`app.py`의 `MODEL_PRESETS`에 다음 항목 추가:

```python
"EXAONE 4.0.1 32B IQ3_M (고품질)": {
    "base_url": "http://127.0.0.1:9081/v1",
    "model": "exaone-4.0.1-32b",
    "temperature": 0.5,
    "top_p": 0.9,
    "timeout": 900,
    "max_group_tokens": 1500,
    "profile": "고품질·신모델",
},
```

`_server_command()`에 분기:

```python
if "4.0" in preset_name or "고품질" in preset_name:
    return "scripts/lm-exaone4.sh"
```

(앞서 EXAONE 3.5 32B Q4_K_M 슬롯이 남아 있다면 같은 PR에서 제거)

## 검증 절차

1. **빌드 확인** — `~/llama.cpp/build/bin/llama-server --version`이 b5932 이상
2. **부팅** — `scripts/lm-exaone4.sh`. stderr에 모델 layer 적재 로그가 끝까지 정상 출력되는지
3. **VRAM** — `nvidia-smi`로 free 가 100MiB 이상 남는지. 음수 근접하면 ctx를 4096으로 낮춤
4. **엔드포인트** — `curl -H 'Authorization: Bearer sk-local' http://127.0.0.1:9081/v1/models`
5. **reasoning off 확인** — 짧은 한국어 입력으로 `/v1/chat/completions` 호출, 응답에 `<think>...</think>` 류 토큰이 없어야 함:

   ```bash
   curl -s http://127.0.0.1:9081/v1/chat/completions \
     -H 'Authorization: Bearer sk-local' \
     -H 'Content-Type: application/json' \
     -d '{"model":"exaone-4.0.1-32b","messages":[{"role":"user","content":"Translate to Korean: Hello world"}],"max_tokens":50}'
   ```

6. **속도 측정** — `cache/llm`을 비운 상태로 같은 단편 EPUB을 7.8B 9070 vs 32B 9081로 번역 → 청크당 시간 비교. 0.25–0.3× 범위면 정상
7. **품질 비교** — 같은 챕터를 두 서버로 번역해 (a) 한국어 자연스러움 (b) 고유명사 첫 등장 표기 일관성 (c) 누락/요약 여부 확인. core/prompts.py의 KOREAN_TRANSLATION_RULES가 그대로 주입되므로 규칙 준수도 비교 가능
8. **장시간 안정성** — 한 권 끝까지 돌려보기. KV 오버플로 / OOM 없이 완주하는지

## 알려진 함정

- **OOM 위험**: 모델 14.4GB + KV로 16GB 한계 근접. 다른 프로세스가 GPU를 쓰고 있으면 부팅 실패. `nvidia-smi`로 사전 정리
- **reasoning 누출**: `--chat-template-kwargs` 누락 시 일부 입력에서 `<think>` 블록이 출력에 섞일 수 있음. 위 검증 5번 필수
- **chat template 호환성**: llama.cpp 빌드가 EXAONE 4.0 template를 미지원하면 부팅 시 경고. 공식 [EXAONE 4.0 README](https://github.com/LG-AI-EXAONE/EXAONE-4.0)의 simplified Jinja template를 `--chat-template-file`로 외부 주입 가능
- **속도 기대치 관리**: 책 한 권이 현행 1시간 → 약 3–4시간으로 늘어남. UI 진행률·로그가 멈춘 것처럼 보일 수 있으나 실제로는 진행 중
- **첫 청크 워밍업**: 32B는 첫 청크에서 GPU prefill에 수십 초 걸릴 수 있음. timeout을 900초로 잡은 이유
- **캐시 분리**: epub-translator는 모델 이름이 다르면 자동으로 캐시 키가 분리됨. 7.8B로 번역한 캐시를 32B가 재사용하지 않으니 같은 책을 두 모델로 번역해도 안전
- **EXAONE Deep와 혼동 주의**: bartowski 리포에 EXAONE-Deep-32B GGUF가 따로 있음. 추론 모델이므로 번역엔 받지 말 것

## 레퍼런스

- bartowski 4.0.1 32B GGUF: <https://huggingface.co/bartowski/LGAI-EXAONE_EXAONE-4.0.1-32B-GGUF>
- 공식 4.0 32B GGUF: <https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B-GGUF>
- EXAONE 4.0 공식 리포 (chat template 안내 포함): <https://github.com/LG-AI-EXAONE/EXAONE-4.0>
- EXAONE 4.0 기술 보고서: <https://arxiv.org/abs/2507.11407>
- llama.cpp EXAONE 4.0 PR: <https://github.com/ggml-org/llama.cpp/pull/14630>
- HF transformers EXAONE 4 doc: <https://huggingface.co/docs/transformers/main/en/model_doc/exaone4>
