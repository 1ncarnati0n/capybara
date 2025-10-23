#!/usr/bin/env python3
"""
LLM (iris-7b) 번역 품질 테스트
콘솔 출력과 동일한 내용을 capybara/분석결과 에 로그로 저장합니다.
"""

import os
import sys
import atexit
import time
from datetime import datetime
from vllm import LLM, SamplingParams

# ------------------------------------------------------------
# 결과 로그: 콘솔 출력과 파일 동시 기록 (분석결과/)
# ------------------------------------------------------------
def _setup_result_logging(prefix: str = "test_llm") -> str:
    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, "분석결과")
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(results_dir, f"{prefix}_{ts}.log")

    fh = open(log_path, "w", encoding="utf-8")

    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
            for s in self.streams:
                try:
                    s.flush()
                except Exception:
                    pass
        def flush(self):
            for s in self.streams:
                try:
                    s.flush()
                except Exception:
                    pass

    sys.stdout = Tee(sys.stdout, fh)
    sys.stderr = Tee(sys.stderr, fh)
    atexit.register(lambda: fh.close())
    print(f"[LOG] 결과 파일: {log_path}")
    return log_path

_setup_result_logging()

print("=" * 70)
print("LLM (iris-7b) 번역 품질 테스트")
print("개선된 프롬프트 + 최적화된 샘플링 파라미터")
print("=" * 70)
print()

# vLLM 모델 로딩
print("🔄 vLLM 모델 로딩 중...")
start_load = time.time()

# 공용 모델 캐시 디렉토리 (프로젝트 내부)
models_dir = os.path.join(os.getcwd(), "models")
os.makedirs(models_dir, exist_ok=True)
print(f"📁 모델 저장 위치: {models_dir}")

llm = LLM(
    model="davidkim205/iris-7b",
    download_dir=models_dir,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.91,
    max_model_len=1024,
    dtype="auto",
    kv_cache_dtype="fp8",
    enforce_eager=True,
    trust_remote_code=True
)

# 개선된 샘플링 파라미터
sampling_params = SamplingParams(
    temperature=0.3,           # 개선: 0.1 → 0.3
    top_p=0.95,                # 개선: 0.9 → 0.95
    max_tokens=128,            # 개선: 30 → 128
    repetition_penalty=1.1,    # 새로 추가
    skip_special_tokens=True,
    stop=["\n", "English:", "Korean:", "---", "###"]
)

load_time = time.time() - start_load
print(f"✅ 모델 로딩 완료 (소요 시간: {load_time:.2f}초)")
print()

# 테스트 케이스
test_cases_en_ko = [
    # 기본 인사
    "Hello, how are you today?",
    "Good morning. Have a nice day!",

    # 감사 표현
    "Thank you very much for your help.",
    "I really appreciate your support.",

    # 긴 문장 (max_tokens 테스트)
    "The quick brown fox jumps over the lazy dog, and then runs through the beautiful forest filled with colorful flowers and singing birds.",

    # 문맥 이해 (톤 보존)
    "I hope you have a wonderful day ahead of you.",
    "It would be great if we could meet tomorrow.",

    # 복잡한 문장
    "Machine learning is transforming the way we approach complex problems in various fields.",
    "The company announced that it will launch a new product next month.",

    # 일상 대화
    "What time does the meeting start?",
    "Could you please send me the report by tomorrow?"
]

test_cases_ko_en = [
    # 기본 인사
    "안녕하세요, 오늘 어떻게 지내세요?",
    "좋은 아침입니다. 좋은 하루 보내세요!",

    # 감사 표현
    "도움을 주셔서 정말 감사합니다.",
    "당신의 지원에 진심으로 감사드립니다.",

    # 긴 문장
    "빠른 갈색 여우가 게으른 개를 뛰어넘고, 그런 다음 아름다운 꽃과 노래하는 새들로 가득 찬 숲을 달려갑니다.",

    # 문맥 이해
    "앞으로 멋진 하루가 되시길 바랍니다.",
    "내일 만날 수 있다면 좋을 것 같아요.",

    # 복잡한 문장
    "기계 학습은 다양한 분야에서 복잡한 문제를 해결하는 방식을 변화시키고 있습니다.",
    "회사는 다음 달에 새로운 제품을 출시할 것이라고 발표했습니다.",

    # 일상 대화
    "회의는 몇 시에 시작하나요?",
    "내일까지 보고서를 보내주시겠어요?"
]

# ===== 영어 → 한국어 번역 테스트 =====
print("=" * 70)
print("테스트 1: 영어 → 한국어 번역 (개선된 프롬프트)")
print("=" * 70)
print()

prompts_en_ko = [f"""You are a professional translator. Translate the following English text to natural Korean. Preserve the tone and nuances.

Examples:
English: "Good morning."
Korean: "좋은 아침입니다."

English: "Thank you very much for your help."
Korean: "도움을 주셔서 정말 감사합니다."

English: "I hope you have a wonderful day."
Korean: "멋진 하루 보내시길 바랍니다."

Now translate this:
English: "{text}"
Korean:""" for text in test_cases_en_ko]

start_translate = time.time()
outputs = llm.generate(prompts_en_ko, sampling_params)
translate_time_en = time.time() - start_translate

for i, (original, output) in enumerate(zip(test_cases_en_ko, outputs), 1):
    translated = output.outputs[0].text.strip().strip('"').strip("'").strip()
    print(f"{i}. EN: {original}")
    print(f"   KO: {translated}")
    print()

print(f"⏱️  번역 시간: {translate_time_en:.2f}초 ({len(test_cases_en_ko)}개 문장)")
print(f"⚡ 평균 속도: {translate_time_en/len(test_cases_en_ko):.2f}초/문장")
print()

# ===== 한국어 → 영어 번역 테스트 =====
print("=" * 70)
print("테스트 2: 한국어 → 영어 번역 (개선된 프롬프트)")
print("=" * 70)
print()

prompts_ko_en = [f"""You are a professional translator. Translate the following Korean text to natural English. Preserve the tone and nuances.

Examples:
Korean: "좋은 아침입니다."
English: "Good morning."

Korean: "도움을 주셔서 정말 감사합니다."
English: "Thank you very much for your help."

Korean: "멋진 하루 보내시길 바랍니다."
English: "I hope you have a wonderful day."

Now translate this:
Korean: "{text}"
English:""" for text in test_cases_ko_en]

start_translate = time.time()
outputs = llm.generate(prompts_ko_en, sampling_params)
translate_time_ko = time.time() - start_translate

for i, (original, output) in enumerate(zip(test_cases_ko_en, outputs), 1):
    translated = output.outputs[0].text.strip().strip('"').strip("'").strip()
    print(f"{i}. KO: {original}")
    print(f"   EN: {translated}")
    print()

print(f"⏱️  번역 시간: {translate_time_ko:.2f}초 ({len(test_cases_ko_en)}개 문장)")
print(f"⚡ 평균 속도: {translate_time_ko/len(test_cases_ko_en):.2f}초/문장")
print()

# ===== 개선사항 비교 =====
print("=" * 70)
print("개선사항 요약")
print("=" * 70)
print()
print("📊 샘플링 파라미터:")
print("  - temperature: 0.1 → 0.3 (더 자연스러운 표현)")
print("  - top_p: 0.9 → 0.95 (어휘 다양성)")
print("  - max_tokens: 30 → 128 (긴 문장 완전 번역)")
print("  - repetition_penalty: 추가 (반복 방지)")
print()
print("📝 프롬프트:")
print("  - Instruction-based 접근")
print("  - 명확한 역할 정의 (professional translator)")
print("  - 품질 지침 (Preserve tone and nuances)")
print("  - 풍부한 예시")
print()
print("🎯 후처리:")
print("  - 불필요한 따옴표 제거")
print("  - 공백 정리")
print()

# ===== 성능 통계 =====
total_time = translate_time_en + translate_time_ko
total_sentences = len(test_cases_en_ko) + len(test_cases_ko_en)

print("=" * 70)
print("성능 통계")
print("=" * 70)
print()
print(f"총 번역 시간: {total_time:.2f}초")
print(f"총 문장 수: {total_sentences}개")
print(f"평균 속도: {total_time/total_sentences:.2f}초/문장")
print()

print("=" * 70)
print("✅ 모든 테스트 완료!")
print("=" * 70)
print()
print("다음 단계:")
print("  1. 번역 품질 확인 (자연스러움, 정확성)")
print("  2. Seq2Seq와 비교 (python test_models.py)")
print("  3. 실제 파일로 테스트 (python capybara.py)")
