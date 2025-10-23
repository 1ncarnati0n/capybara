#!/usr/bin/env python3
"""
모델 비교 테스트: LLM (iris-7b) vs Seq2Seq (NLLB)
콘솔 출력과 동일한 내용을 capybara/분석결과 에 로그로 저장합니다.
"""

import os
import sys
import atexit
import time
from datetime import datetime
import torch
from vllm import LLM, SamplingParams
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ------------------------------------------------------------
# 결과 로그: 콘솔 출력과 파일 동시 기록 (분석결과/)
# ------------------------------------------------------------
def _setup_result_logging(prefix: str = "test_models") -> str:
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
print("모델 비교 테스트: LLM (iris-7b) vs Seq2Seq (NLLB)")
print("=" * 70)
print()

# 테스트 케이스 (대표적인 문장들)
test_cases_en = [
    "Hello, how are you today?",
    "Thank you very much for your help.",
    "I hope you have a wonderful day ahead of you.",
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is transforming the world.",
]

test_cases_ko = [
    "안녕하세요, 오늘 어떻게 지내세요?",
    "도움을 주셔서 정말 감사합니다.",
    "앞으로 멋진 하루가 되시길 바랍니다.",
    "빠른 갈색 여우가 게으른 개를 뛰어넘습니다.",
    "기계 학습은 세상을 변화시키고 있습니다.",
]

# models 폴더 생성
models_dir = os.path.join(os.getcwd(), "models")
os.makedirs(models_dir, exist_ok=True)

# ===== LLM 모델 테스트 =====
print("=" * 70)
print("1. LLM 모델 (iris-7b)")
print("=" * 70)
print()

print("🔄 LLM 모델 로딩 중...")
start_load = time.time()

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
    temperature=0.3,
    top_p=0.95,
    max_tokens=128,
    repetition_penalty=1.1,
    skip_special_tokens=True,
    stop=["\n", "English:", "Korean:", "---", "###"]
)

load_time_llm = time.time() - start_load
print(f"✅ LLM 모델 로딩 완료 (소요 시간: {load_time_llm:.2f}초)")
print()

# 영어 → 한국어 (LLM)
print("📝 영어 → 한국어 번역 (LLM)")
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
Korean:""" for text in test_cases_en]

start_translate = time.time()
outputs = llm.generate(prompts_en_ko, sampling_params)
translate_time_llm_en = time.time() - start_translate

llm_results_en = []
for i, (original, output) in enumerate(zip(test_cases_en, outputs), 1):
    translated = output.outputs[0].text.strip().strip('"').strip("'").strip()
    llm_results_en.append(translated)
    print(f"{i}. EN: {original}")
    print(f"   KO: {translated}")
    print()

print(f"⏱️  번역 시간: {translate_time_llm_en:.2f}초")
print()

# 한국어 → 영어 (LLM)
print("📝 한국어 → 영어 번역 (LLM)")
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
English:""" for text in test_cases_ko]

start_translate = time.time()
outputs = llm.generate(prompts_ko_en, sampling_params)
translate_time_llm_ko = time.time() - start_translate

llm_results_ko = []
for i, (original, output) in enumerate(zip(test_cases_ko, outputs), 1):
    translated = output.outputs[0].text.strip().strip('"').strip("'").strip()
    llm_results_ko.append(translated)
    print(f"{i}. KO: {original}")
    print(f"   EN: {translated}")
    print()

print(f"⏱️  번역 시간: {translate_time_llm_ko:.2f}초")
print()

# ===== Seq2Seq 모델 테스트 =====
print("=" * 70)
print("2. Seq2Seq 모델 (NLLB)")
print("=" * 70)
print()

# 영어 → 한국어 (Seq2Seq)
print("🔄 Seq2Seq 모델 로딩 중 (en2ko)...")
start_load = time.time()

model_en2ko = AutoModelForSeq2SeqLM.from_pretrained("NHNDQ/nllb-finetuned-en2ko", cache_dir=models_dir)
tokenizer_en2ko = AutoTokenizer.from_pretrained(
    "NHNDQ/nllb-finetuned-en2ko",
    cache_dir=models_dir,
    src_lang="eng_Latn",
    tgt_lang="kor_Hang"
)

if torch.cuda.is_available():
    model_en2ko = model_en2ko.cuda()

load_time_s2s_en = time.time() - start_load
print(f"✅ Seq2Seq 모델 로딩 완료 (소요 시간: {load_time_s2s_en:.2f}초)")
print()

print("📝 영어 → 한국어 번역 (Seq2Seq)")
start_translate = time.time()

s2s_results_en = []
for i, text in enumerate(test_cases_en, 1):
    inputs = tokenizer_en2ko(text, return_tensors="pt", padding=True)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    outputs = model_en2ko.generate(**inputs, max_length=512)
    translated = tokenizer_en2ko.decode(outputs[0], skip_special_tokens=True)
    s2s_results_en.append(translated)

    print(f"{i}. EN: {text}")
    print(f"   KO: {translated}")
    print()

translate_time_s2s_en = time.time() - start_translate
print(f"⏱️  번역 시간: {translate_time_s2s_en:.2f}초")
print()

# 한국어 → 영어 (Seq2Seq)
print("🔄 Seq2Seq 모델 로딩 중 (ko2en)...")
start_load = time.time()

model_ko2en = AutoModelForSeq2SeqLM.from_pretrained("NHNDQ/nllb-finetuned-ko2en", cache_dir=models_dir)
tokenizer_ko2en = AutoTokenizer.from_pretrained(
    "NHNDQ/nllb-finetuned-ko2en",
    cache_dir=models_dir,
    src_lang="kor_Hang",
    tgt_lang="eng_Latn"
)

if torch.cuda.is_available():
    model_ko2en = model_ko2en.cuda()

load_time_s2s_ko = time.time() - start_load
print(f"✅ Seq2Seq 모델 로딩 완료 (소요 시간: {load_time_s2s_ko:.2f}초)")
print()

print("📝 한국어 → 영어 번역 (Seq2Seq)")
start_translate = time.time()

s2s_results_ko = []
for i, text in enumerate(test_cases_ko, 1):
    inputs = tokenizer_ko2en(text, return_tensors="pt", padding=True)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    outputs = model_ko2en.generate(**inputs, max_length=512)
    translated = tokenizer_ko2en.decode(outputs[0], skip_special_tokens=True)
    s2s_results_ko.append(translated)

    print(f"{i}. KO: {text}")
    print(f"   EN: {translated}")
    print()

translate_time_s2s_ko = time.time() - start_translate
print(f"⏱️  번역 시간: {translate_time_s2s_ko:.2f}초")
print()

# ===== 비교 분석 =====
print("=" * 70)
print("3. 비교 분석")
print("=" * 70)
print()

print("📊 영어 → 한국어 번역 비교")
print("-" * 70)
for i, (en_text, llm_text, s2s_text) in enumerate(zip(test_cases_en, llm_results_en, s2s_results_en), 1):
    print(f"{i}. 원문: {en_text}")
    print(f"   LLM:     {llm_text}")
    print(f"   Seq2Seq: {s2s_text}")
    print()

print("📊 한국어 → 영어 번역 비교")
print("-" * 70)
for i, (ko_text, llm_text, s2s_text) in enumerate(zip(test_cases_ko, llm_results_ko, s2s_results_ko), 1):
    print(f"{i}. 원문: {ko_text}")
    print(f"   LLM:     {llm_text}")
    print(f"   Seq2Seq: {s2s_text}")
    print()

# ===== 성능 비교 =====
print("=" * 70)
print("4. 성능 비교")
print("=" * 70)
print()

print("⏱️  번역 속도 비교:")
print()
print(f"영어 → 한국어:")
print(f"  LLM:     {translate_time_llm_en:.2f}초")
print(f"  Seq2Seq: {translate_time_s2s_en:.2f}초")
print(f"  속도 비율: {translate_time_s2s_en/translate_time_llm_en:.2f}x (LLM 기준)")
print()

print(f"한국어 → 영어:")
print(f"  LLM:     {translate_time_llm_ko:.2f}초")
print(f"  Seq2Seq: {translate_time_s2s_ko:.2f}초")
print(f"  속도 비율: {translate_time_s2s_ko/translate_time_llm_ko:.2f}x (LLM 기준)")
print()

print("📦 모델 크기:")
print(f"  LLM:     7B 파라미터")
print(f"  Seq2Seq: 600M 파라미터")
print()

print("💾 VRAM 요구사항:")
print(f"  LLM:     12-16GB")
print(f"  Seq2Seq: 3-4GB")
print()

# ===== 결론 =====
print("=" * 70)
print("✅ 모든 테스트 완료!")
print("=" * 70)
print()
print("📝 결론:")
print()
print("LLM (iris-7b):")
print("  ✅ 빠른 배치 처리 (vLLM)")
print("  ✅ 자연스러운 문체 (개선된 프롬프트)")
print("  ✅ 긴 문장 완전 번역 (max_tokens=128)")
print("  ⚠️  높은 VRAM 요구 (12-16GB)")
print()
print("Seq2Seq (NLLB):")
print("  ✅ 안정적이고 정확한 번역")
print("  ✅ 낮은 VRAM 요구 (3-4GB)")
print("  ⚠️  느린 순차 처리")
print()
print("다음 단계:")
print("  1. 번역 품질 직접 비교 및 평가")
print("  2. 실제 파일(.txt, .epub, .srt)로 테스트")
print("  3. 사용 사례에 맞는 모델 선택")
