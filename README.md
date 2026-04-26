# en2ko — 영→한 EPUB 번역기

영어 EPUB을 로컬 LLM(EXAONE 3.5 7.8B, llama.cpp)으로 한국어 번역본으로 변환합니다. 번역 코어는 [`oomol-lab/epub-translator`](https://github.com/oomol-lab/epub-translator), UI는 Streamlit입니다.

## 아키텍처

```
┌──────────────────────────────────┐
│  Streamlit UI  (app.py)          │  ← 업로드, 진행률, 다운로드
└────────────┬─────────────────────┘
             │ 백그라운드 스레드 + Queue
┌────────────▼─────────────────────┐
│  core/translator.py              │  ← epub-translator 래퍼
│  · KOREAN user_prompt 주입       │
│  · on_progress / on_fill_failed  │
└────────────┬─────────────────────┘
             │
┌────────────▼─────────────────────┐
│  epub_translator.translate(...)  │  ← 청크 분할, 캐시, XML 보존
│  · cache_path 로 자동 재개       │
└────────────┬─────────────────────┘
             │ OpenAI 호환 HTTP
┌────────────▼─────────────────────┐
│  llama-server (CUDA)             │  ← scripts/lm-server.sh
│  EXAONE-3.5-7.8B-Instruct Q6_K   │  기본
└──────────────────────────────────┘
```

핵심 설계 결정:

- **재개 가능성**은 epub-translator 내장 디스크 캐시(`cache/llm`)로 처리. 별도 잡 큐 없음.
- **듀얼 LLM**(`translation_llm` + `fill_llm`) 옵션은 16GB VRAM 한 대에 두 모델을 동시 로드하기 어려워 MVP에서는 단일 LLM 사용.
- **토크나이저 불일치**(tiktoken `o200k_base` vs EXAONE 자체 토크나이저)를 고려해 `max_group_tokens=2200`으로 보수적으로 설정. EXAONE은 한국어를 더 짧게 인코딩하므로 컨텍스트 초과는 발생하지 않음.
- **청크 동시성 1**: GPU 1대 환경에 맞춤(`concurrency=1`).

## 디렉터리

| 경로                        | 용도                                         |
| --------------------------- | -------------------------------------------- |
| `app.py`                    | Streamlit 앱 진입점                          |
| `core/llm_factory.py`       | `LLM(...)` 인스턴스 빌더                     |
| `core/translator.py`        | `translate(...)` 호출, 콜백→이벤트 큐        |
| `core/prompts.py`           | 한국어 번역 규칙 (user_prompt)               |
| `core/progress.py`          | 워커→UI 이벤트 채널                          |
| `scripts/lm-server.sh`      | EXAONE 3.5 7.8B Q6_K llama-server 기동       |
| `models/`                   | GGUF 파일 (gitignore)                        |
| `cache/`                    | epub-translator 응답 캐시 (gitignore)        |
| `uploads/`                  | 사용자 업로드 EPUB (gitignore)               |
| `output/`                   | 번역 결과 EPUB (gitignore)                   |
| `vendor/epub-translator/`   | 분석/참조용 클론 (gitignore)                 |

## 사전 요구

- **GPU**: NVIDIA RTX 4080 16GB (또는 동급)
- **llama.cpp**: CUDA 빌드. `llama-server`가 PATH에 있거나, `~/llama.cpp/build/bin/llama-server`에 있으면 자동 인식.  
  빌드: <https://github.com/ggml-org/llama.cpp>
- **Python 관리**: [uv](https://docs.astral.sh/uv/) (Python 3.11–3.13 자동 설치)
- **디스크**: 모델 ~6.5GB + 캐시 여유분

## 설치

```bash
# uv가 없다면 먼저 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 의존성 설치 (자동으로 .venv/ 생성)
uv sync

# EXAONE 3.5 7.8B Q6_K GGUF를 models/ 아래에 둡니다.
# 기본 경로: models/EXAONE-3.5-7.8B-Instruct-Q6_K.gguf
```

## 실행

터미널 두 개를 사용합니다.

**터미널 1 — llama.cpp 서버**

```bash
scripts/lm-server.sh
```

환경변수로 조정 가능: `MODEL`, `PORT`, `CTX`, `API_KEY`, `LLAMA_SERVER_BIN`.

**터미널 2 — Streamlit UI**

```bash
uv run app.py
```

브라우저에서 EPUB을 업로드하고 옵션을 골라 시작합니다.

Streamlit UI는 운영 대시보드 형태입니다.

- 사이드바에서 모델 프리셋을 고르고 `연결 확인`으로 `/v1/models` 응답을 확인합니다.
- `작업` 탭에서 EPUB을 업로드하고 번역을 시작합니다.
- `진행`, `결과`, `로그`, `설정 요약` 탭에서 상태와 결과 파일을 확인합니다.
- 서버가 꺼져 있으면 선택한 프리셋에 맞는 실행 명령이 UI에 표시됩니다.

## 출력 모드

| 모드                 | 결과                                             |
| -------------------- | ------------------------------------------------ |
| `APPEND_TEXT` (기본) | 원문 다음 줄에 아주 엷은 노란 배경의 번역문 추가 |
| `REPLACE`            | 원문을 한국어로 교체 → 한국어 단독본             |

## 파일명 규칙

- 업로드 EPUB은 `uploads/{제목}__{id}.epub` 형식으로 저장됩니다.
- 완료된 번역본은 실제 소요 시간을 붙여 `output/{제목}-{분}m{초}s_ko.epub` 형식으로 저장됩니다.
- 같은 이름의 결과가 이미 있으면 기존 파일을 덮어쓰지 않고 `-2`, `-3` suffix를 붙입니다.

## 한국어 번역 규칙

[`core/prompts.py`](core/prompts.py)의 `KOREAN_TRANSLATION_RULES`가 epub-translator의 `<rules>` 슬롯에 주입됩니다. 평서체 기본, 인명/지명/기술 용어 표기, 누락·요약 금지 등을 포함합니다. 책 성격에 맞게 자유롭게 수정하세요.

## 캐시와 재개

`cache/llm/`에 메시지 해시 → 응답 텍스트가 저장됩니다. 같은 입력으로 재실행하면 LLM 호출 없이 즉시 결과를 재사용합니다. 책 한 권 번역이 중간에 중단되어도 다시 실행하면 끝낸 챕터는 캐시에서 읽어 빠르게 진행합니다.

캐시 무효화는 `target_language` 또는 epub-translator 버전이 바뀔 때 자동으로 일어납니다. 강제로 재번역하려면 `cache/llm/`을 비우면 됩니다.

## 알려진 제약

- llama.cpp가 `stream_options.include_usage`를 지원해야 합니다(최근 빌드는 OK).
- EXAONE GGUF의 chat template이 자동 적용되지 않으면 `--chat-template exaone3` 명시 필요.
- EXAONE 라이선스는 비상업적 연구·개인 사용 위주입니다. 상업적 용도는 LG AI 라이선스를 확인하세요.
