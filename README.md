# Capybara (카피바라) 🦫

영어 ↔ 한국어 번역을 지원하는 듀얼 모델 워크플로우입니다. `davidkim205/iris-7b`를 vLLM으로 구동해 빠른 배치 번역을 제공하며, VRAM이 제한된 환경에서는 Seq2Seq(NLLB) 모델을 선택할 수 있습니다.

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

## 테스트 & 점검

테스트 구성은 이후 개편 예정이지만, 현재는 아래 스크립트로 핵심 동작을 확인할 수 있습니다.

```bash
# LLM 품질 및 샘플링 파라미터 확인
python test_llm.py

# LLM vs Seq2Seq 속도/출력 비교
python test_models.py

# 프롬프트 전략 탐색 (실험적)
python test_prompt_tuner.py --file 분석결과/test_prompt.txt --limit 100 --direction auto
```

모든 스크립트는 `capybara/분석결과/`에 타임스탬프 로그를 남깁니다.

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
├── capybara.py            # 메인 앱 (Gradio UI + 번역 로직)
├── start.sh               # 실행 스크립트
├── requirements.txt
├── test_llm.py            # LLM 품질 테스트
├── test_models.py         # 듀얼 모델 비교
├── test_prompt_tuner.py   # 프롬프트 전략 실험
├── CHANGELOG.md           # 변경 기록
└── outputs/               # 번역 결과 (자동 생성)
```

---

## 라이선스 & 기여

- **License**: MIT
- **Contributing**: 이슈/PR 환영합니다. 버그나 요청 사항은 GitHub 이슈로 남겨주세요.

---

Happy Translating! 🦫
