# Capybara (카피바라) 🦫

영어 ↔ 한국어 번역을 지원하는 워크플로우입니다. UI에서는 LLM(iris-7b)과 Seq2Seq(NLLB) 모델을 선택할 수 있으며, 프로젝트 내부에는 LLM 전용 프롬프트/하이퍼파라미터 튜너가 포함되어 있습니다.

본 프로젝트는 "[dodari](https://github.com/vEduardovich/dodari)" 프로젝트에 영감을 받았습니다.

---

## 핵심 특징

- 두 모델 탑재: 고속 LLM(iris-7b)과 경량 Seq2Seq(NLLB)
- TXT·EPUB·SRT 입력 지원, 원문+번역본 동시 저장

---

## 요구사항 요약

- **OS**: Linux / WSL2 (Windows) / macOS(제한적)
- **Python**: 3.11
- **GPU**: CUDA 11.8+ 권장
  - LLM: VRAM 12–16GB 권장 (`gpu_memory_utilization=0.91`, `max_model_len=1024` 기본)
  - Seq2Seq: VRAM 3–4GB
- **RAM / Storage**: 16GB RAM, 모델 캐시 포함 20GB 이상 여유

---

## 빠른 시작

```bash
# 1. 환경 준비
conda create -n capybara python=3.11 -y
conda activate capybara

# 2. 의존성 설치
cd capybara
pip install -r requirements.txt

# 3. 실행 (자동 브라우저)
bash start.sh
# 또는
python capybara.py
```

최초 실행 시 모델이 `capybara/models/`에 캐시됩니다. `nvidia-smi`로 VRAM 여유를 확인하세요.

---

## 사용 방법

1. 브라우저(기본 포트 7860)에서 파일 업로드
2. 언어 자동 감지 결과 확인 (영↔한 모두 지원)
3. 모델 선택
   - `LLM (iris-7b)` : 빠른 배치 처리, 자연스러운 표현
   - `Seq2Seq (NLLB)` : 낮은 VRAM, 보수적인 번역
4. 번역 실행 → 완료 후 즉시 다음 파일 업로드 가능

### 출력

`outputs/` 폴더에 두 가지 파일이 생성됩니다.
- `파일명_ko(en)_llm.txt` / `_s2s.txt` : 번역문 + 원문
- `파일명_ko_llm.txt` / `_s2s.txt`     : 번역문만

---

## 튜너(LLM 프롬프트/하이퍼파라미터)

LLM 전용 프롬프트/샘플링 파라미터를 자동으로 스윕하고 전략별 점수를 비교합니다.

```bash
cd capybara
python tuner.py
```

기본 설정은 파일 상단의 상수로 정의됩니다.

- `TESTING_LLM_MODEL` (기본: `davidkim205/iris-7b`)
- 도메인/샘플 수/배치: `TESTING_DOMAINS`, `TESTING_PROMPT_LIMIT`, `TESTING_BATCH_SIZE`
- 샘플링: `TESTING_PROMPT_TEMPERATURE`, `TESTING_TOP_P`, `TESTING_MAX_TOKENS`, `TESTING_REPETITION_PENALTY`

스윕 결과와 각 실행의 상세 로그는 `capybara/분석결과/`에 저장됩니다.
최고 조합은 자동으로 `capybara/hyperparams.py`에 반영되어, 다음 앱 실행부터 기본값으로 적용됩니다.

튜너 특징:
- 5개 도메인(tech, literary, slang, casual, academic) 내장 샘플
- 5가지 프롬프트 전략 비교: domain_fewshot, instruct_only, minimal, cot_style, role_based
- 출력 규칙 강제(한 줄, 라벨/따옴표/설명 금지) + 후처리 + 언어 불일치 폴백
- 상세 설정 섹션과 샘플 출력/점수/패널티를 로그에 기록

### 자동 적용 흐름

1) `python tuner.py` 실행 → 각 설정/전략을 평가 → 최고 구성 선별
2) 최고 구성(temperature/top_p/max_tokens/repetition_penalty)이 `capybara/hyperparams.py`에 저장
3) `python capybara.py` 또는 `bash start.sh`로 앱 실행 시, `hyperparams.py`의 기본값을 사용해 LLM을 로드/생성

---

## 문제 해결 가이드

- **CUDA OOM**: 다른 GPU 작업 종료 → `gpu_memory_utilization`을 0.88 이하로 조정 → 필요 시 `max_model_len`을 768/512로 축소
- **vLLM 설치 실패**: Python 3.11·CUDA 11.8 이상 확인 후 `pip install --upgrade pip && pip install vllm>=0.8.2`
- **포트 충돌**: `capybara.py` 실행부의 `server_port` 값을 7861 등으로 변경
- **번역 속도 저하**: `nvidia-smi` 로드 확인 후 여유 VRAM에서 `max_num_seqs` 증가

---

## 프로젝트 구조

```
capybara/
├── capybara.py          # 메인 앱 (Gradio UI + 번역 로직)
├── tuner.py             # LLM 전용 프롬프트/하이퍼파라미터 튜너
├── start.sh             # 앱 실행 스크립트
├── requirements.txt
├── TUNER_GUIDE.md
├── CHANGELOG.md
├── outputs/             # 번역 결과 (자동 생성)
└── 분석결과/           # 튜너 분석 로그 (자동 생성)
```

---

## 라이선스 & 기여

- **License**: MIT
- **Contributing**: 이슈/PR 환영합니다. 버그나 요청 사항은 GitHub 이슈로 남겨주세요.

---

Happy Translating! 🦫
