# TUNER 가이드 (LLM 전용)

## 개요

프롬프트 튜너 v2.0은 iris-7b 모델의 번역 품질을 최적화하기 위한 도구입니다. 다양한 도메인(기술, 문학, 신조어, 일상, 학술)의 테스트 문장을 사용하여 여러 프롬프트 전략을 자동으로 비교 평가합니다.

## 주요 특징

- ✅ **파일 내장형**: 별도의 테스트 파일 없이 실행 가능
- ✅ **5개 도메인**: 기술(tech), 문학(literary), 신조어/속어(slang), 일상(casual), 학술(academic)
- ✅ **5가지 전략**: domain_fewshot, instruct_only, minimal, cot_style, role_based
- ✅ **도메인별 평가**: 전문용어 보존, 문체 일관성 등 특화 지표
- ✅ **상세 리포트**: 마크다운 형식의 분석 결과 자동 생성

---

## 빠른 시작

### 기본 사용법

```bash
cd capybara
python tuner.py
```

### 고급 옵션

튜너 실행 시 파일 상단에 정의된 상수를 수정하여 설정을 조정할 수 있습니다.

- 모델: `TESTING_LLM_MODEL`
- 도메인/문장/배치: `TESTING_DOMAINS`, `TESTING_PROMPT_LIMIT`, `TESTING_BATCH_SIZE`
- 샘플링: `TESTING_PROMPT_TEMPERATURE`, `TESTING_TOP_P`, `TESTING_MAX_TOKENS`, `TESTING_REPETITION_PENALTY`

또는 `main()` 내부의 스윕 리스트(temps/reps/top_ps/max_toks)를 변경하여 그리드 스윕 범위를 확장하세요.

### 전체 옵션 목록

```bash
python test_prompt_tuner.py --help
```

---

## 도메인별 테스트 문장 예시

### 1. 기술 (tech)
- "The microservice architecture enables horizontal scaling through containerization."
- "We implemented a zero-trust security model using JWT tokens and OAuth2 flows."
- "The GPU utilizes CUDA cores for parallel processing of tensor operations."

### 2. 문학 (literary)
- "The autumn leaves danced gracefully in the gentle breeze."
- "Her laughter echoed through the empty corridors like a forgotten melody."
- "Time stood still as he gazed into her eyes, lost in their depths."

### 3. 신조어/속어 (slang)
- "That new feature is totally fire, no cap!"
- "He's ghosting me again, ugh, so annoying."
- "This bug is sus, we need to investigate ASAP."

### 4. 일상 (casual)
- "How's your day going so far?"
- "I'll grab some coffee on my way to the office."
- "Thanks for reaching out, I really appreciate it."

### 5. 학술 (academic)
- "The empirical results demonstrate a statistically significant correlation."
- "We propose a novel framework for evaluating neural network robustness."
- "The methodology employed in this study builds upon prior research."

---

## 프롬프트 전략 설명

### 1. domain_fewshot
- **설명**: 도메인별 few-shot 예시 포함
- **장점**: 일관된 번역 스타일, 도메인 용어 보존
- **추천**: 기술 문서, 전문 분야

### 2. instruct_only
- **설명**: 명시적 지침만 제공 (예시 없음)
- **장점**: 프롬프트 길이 절약
- **추천**: 일반적인 텍스트

### 3. minimal
- **설명**: 최소한의 프롬프트
- **장점**: 가장 빠른 처리 속도
- **단점**: 불안정한 출력 가능

### 4. cot_style
- **설명**: Chain-of-Thought 스타일 (단계별 지시)
- **장점**: 복잡한 문장 처리 우수
- **추천**: 문학 작품, 복잡한 표현

### 5. role_based
- **설명**: 역할 기반 프롬프트
- **장점**: 안정적이고 일관된 출력
- **추천**: 범용 번역

---

## 평가 지표

### 기본 지표
- **언어 감지**: 올바른 목표 언어로 번역되었는지 확인
- **길이 비율**: 과도하게 긴 번역 감지
- **노이즈 마커**: "English:", "Korean:" 등 불필요한 레이블 감지
- **따옴표 사용**: 불필요한 큰따옴표 감지

### 도메인별 특화 지표

#### 기술 (tech)
- **전문용어 보존**: API, GPU, JSON 등 기술 용어 유지 여부

#### 문학 (literary)
- **문체 보존**: 비유적 표현의 풍부함 유지

#### 신조어/속어 (slang)
- **격식 수준**: 과도한 존댓말 사용 방지

#### 학술 (academic)
- **학술적 격식**: 구어체 표현 방지

---

## 결과 해석

### 리포트 구조

1. **도메인별 분석**
   - 각 전략별 평균 점수
   - 샘플 번역 예시 (3개)
   - 패널티 세부사항

2. **전체 요약**
   - 도메인별 최고 전략
   - 전략별 전체 평균 점수

3. **최종 추천**
   - 전체 최고 전략
   - 사용 권장사항

### 점수 해석

- **1.0**: 완벽한 번역 (패널티 없음)
- **0.9~0.99**: 우수 (경미한 문제)
- **0.7~0.89**: 양호 (일부 개선 필요)
- **0.5~0.69**: 보통 (여러 문제 발견)
- **0.5 미만**: 불량 (주요 문제 있음)

---

## 실전 활용 팁

### 1. 도메인 선택
튜너는 기본적으로 5개 도메인을 모두 테스트합니다. 특정 도메인만 실험하려면 `TESTING_DOMAINS`를 수정하세요.

### 2. 결과 적용
리포트에서 추천된 전략을 [capybara.py](capybara.py)의 프롬프트/샘플링 로직에 반영:

```python
# 예: role_based 전략이 최고 점수를 받은 경우
template = f"""You are an expert translator specialized in {domain} texts.
Your task: Translate English to natural Korean.
Requirements: {domain_inst}
Rules: Output a single line containing only the Korean translation. No quotes, no labels, no explanations.

---

## 자동 적용 흐름

1) `python tuner.py` 실행 → 설정/전략을 평가 → 최고 구성 선별
2) 최고 구성(temperature/top_p/max_tokens/repetition_penalty)이 `capybara/hyperparams.py`에 저장(자동)
3) `python capybara.py` 또는 `bash start.sh` 실행 시, `hyperparams.py`의 기본값으로 LLM 로드/생성

---

## 향후 개선 제안

- 성능/구조 최적화
  - 스윕 중 LLM 인스턴스 1회만 로드하고 SamplingParams만 교체(현재는 조합별 재로딩 가능성 → 속도/VRAM 효율 개선).
  - 프롬프트 빌더/평가/스윕 제어를 모듈로 분리해 유지보수성을 향상.

- 평가 지표 보강
  - 정량 지표 추가: chrF / sentence-BLEU(참조가 있을 경우) 등으로 미세 품질 차이를 반영.
  - 언어 감지 보조: 한글 비율(가-힣 범위) 기반 간단 필터로 `langdetect` 오탐 보완.
  - 도메인 가중치 집계(가중 평균/최소값 기반 등)로 선택 기준 다변화.

- 스윕 확장/제어
  - `top_p`(예: 0.90/0.92/0.95), `max_tokens`(128/256/384), `temperature`(0.2/0.3/0.4) 등 매트릭스 확장.
  - 빠른 모드(전략 서브셋/도메인 서브셋) vs 정밀 모드(전체) 분리.
  - seed/고정 옵션과 재현성 메타데이터(실행 환경/파라미터) 로그화.

- 적용 자동화 고도화
  - 현재 구현: 최고 구성은 자동으로 `hyperparams.py`에 반영됨.
  - 추가: 최고 전략(전략명)을 JSON으로 저장하고 앱 프롬프트 템플릿에 자동 반영하는 옵션.
  - UI에서 "튜너 권장 설정 사용" 토글 제공.

- 운영/테스트
  - start.sh에 튜너 진입 옵션 추가(`bash start.sh --tuner`).
  - 회귀 테스트: 샘플 셋 고정 후 점수 하락 감지 시 알림/롤백.


[English Text]
{text}

[Korean Translation]
"""
```

### 3. 반복 테스트
- 프롬프트 수정 후 재테스트
- 다양한 temperature 값 실험
- 배치 크기 최적화로 속도 개선

---

## 문제 해결

### CUDA OOM 에러
```bash
# GPU 메모리 사용률 낮추기
python test_prompt_tuner.py --domains tech --limit 5 --gpu_mem 0.85

# max_model_len 줄이기
python test_prompt_tuner.py --domains tech --limit 5 --max_model_len 768
```

### 느린 속도
```bash
# 배치 크기 증가 (VRAM 여유 있는 경우)
python test_prompt_tuner.py --domains all --limit 10 --batch_size 32

# 테스트 문장 수 줄이기
python test_prompt_tuner.py --domains all --limit 5
```

### 불안정한 번역
```bash
# Temperature 낮추기 (더 일관된 출력)
python test_prompt_tuner.py --domains all --limit 10 --temperature 0.1
```

---

## 결과 파일 위치

모든 분석 결과는 다음 위치에 저장됩니다:
```
capybara/분석결과/prompt_tuner_v2_YYYYMMDD_HHMMSS.md
```

---

## 추가 개선 아이디어

- [ ] 한영(ko2en) 번역 테스트 추가
- [ ] 사용자 정의 테스트 문장 추가 기능
- [ ] BLEU/METEOR 등 자동 평가 지표 추가
- [ ] A/B 테스트 기능 (두 프롬프트 직접 비교)
- [ ] 웹 UI를 통한 인터랙티브 테스트

---

**Happy Testing!** 🦫
