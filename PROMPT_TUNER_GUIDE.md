# 프롬프트 튜너 v2.0 사용 가이드

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

# 모든 도메인 테스트 (각 도메인당 10개 문장)
python test_prompt_tuner.py --domains all --limit 10

# 특정 도메인만 테스트
python test_prompt_tuner.py --domains tech literary --limit 5

# 신조어/속어 집중 테스트
python test_prompt_tuner.py --domains slang --limit 15
```

### 고급 옵션

```bash
# GPU 메모리 사용률 조정
python test_prompt_tuner.py --domains all --limit 10 --gpu_mem 0.85

# 배치 크기 조정 (속도 최적화)
python test_prompt_tuner.py --domains all --limit 10 --batch_size 32

# Temperature 조정 (창의성 vs 일관성)
python test_prompt_tuner.py --domains all --limit 10 --temperature 0.1
```

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
```bash
# 기술 문서 번역 최적화
python test_prompt_tuner.py --domains tech --limit 15

# 소설/에세이 번역 최적화
python test_prompt_tuner.py --domains literary casual --limit 10

# 소셜 미디어/블로그 번역 최적화
python test_prompt_tuner.py --domains slang casual --limit 15
```

### 2. 결과 적용
리포트에서 추천된 전략을 [capybara.py](capybara.py)의 프롬프트에 반영:

```python
# 예: role_based 전략이 최고 점수를 받은 경우
template = f"""You are an expert translator specialized in {domain} texts.
Your task: Translate English to natural Korean.
Requirements: {domain_inst}
Output: Only the Korean translation, nothing else.

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
