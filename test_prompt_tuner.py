#!/usr/bin/env python3
"""
프롬프트 튜너 (Prompt Tuner)

분석결과/test_prompt.txt 파일의 문장들을 사용해 여러 프롬프트 전략을 비교/평가하고,
가장 안정적인 프롬프트를 추천합니다. 결과는 capybara/분석결과/*.md 로 저장됩니다.

사용 예시:
  conda activate capybara && cd capybara
  python test_prompt_tuner.py --file 분석결과/test_prompt.txt --limit 100 --direction auto

파일 형식:
  - 각 줄이 하나의 테스트 문장입니다. 공백/주석(#) 라인은 무시합니다.
  - 언어 방향은 자동 감지(auto) 또는 --direction en2ko|ko2en 지정 가능.
"""

import os
import sys
import time
import math
import argparse
from datetime import datetime
from typing import List, Tuple, Dict

from vllm import LLM, SamplingParams
from langdetect import detect


def setup_logging(prefix: str = "prompt_tuner") -> str:
    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, "분석결과")
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"{prefix}_{ts}.md")
    return path


def read_samples(path: str, limit: int) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"테스트 파일을 찾을 수 없습니다: {path}")
    samples: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                continue
            samples.append(s)
            if 0 < limit <= len(samples):
                break
    if not samples:
        raise ValueError("테스트 문장이 비어있습니다.")
    return samples


def chunk(lst: List[str], size: int) -> List[List[str]]:
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def build_prompts(texts: List[str], direction: str, strategy: str) -> List[str]:
    prompts: List[str] = []
    if direction not in ("en2ko", "ko2en"):
        raise ValueError("direction must be en2ko or ko2en")

    if strategy == "instruct_fewshot":
        if direction == "en2ko":
            template = (
                "You are a professional translator. Translate the following English text to natural Korean. "
                "Preserve tone and nuances. Output only the translation without quotes or labels.\n\n"
                "Examples:\n"
                "English: \"Good morning.\"\nKorean: \"좋은 아침입니다.\"\n\n"
                "English: \"Thank you very much for your help.\"\nKorean: \"도움을 주셔서 정말 감사합니다.\"\n\n"
                "English: \"I hope you have a wonderful day.\"\nKorean: \"멋진 하루 보내시길 바랍니다.\"\n\n"
                "Now translate this:\nEnglish: \"{text}\"\nKorean:"
            )
        else:
            template = (
                "You are a professional translator. Translate the following Korean text to natural English. "
                "Preserve tone and nuances. Output only the translation without quotes or labels.\n\n"
                "Examples:\n"
                "Korean: \"좋은 아침입니다.\"\nEnglish: \"Good morning.\"\n\n"
                "Korean: \"도움을 주셔서 정말 감사합니다.\"\nEnglish: \"Thank you very much for your help.\"\n\n"
                "Korean: \"멋진 하루 보내시길 바랍니다.\"\nEnglish: \"I hope you have a wonderful day.\"\n\n"
                "Now translate this:\nKorean: \"{text}\"\nEnglish:"
            )
        prompts = [template.format(text=t) for t in texts]

    elif strategy == "instruct_strict":
        if direction == "en2ko":
            template = (
                "You are a professional English-to-Korean translator.\n"
                "Rules: Output only the Korean translation. No quotes. No labels. No explanations.\n\n"
                "Text: {text}\n"
                "Output:"
            )
        else:
            template = (
                "You are a professional Korean-to-English translator.\n"
                "Rules: Output only the English translation. No quotes. No labels. No explanations.\n\n"
                "Text: {text}\n"
                "Output:"
            )
        prompts = [template.format(text=t) for t in texts]

    elif strategy == "translate_minimal":
        if direction == "en2ko":
            template = "Translate to Korean:\n{text}\n"
        else:
            template = "Translate to English:\n{text}\n"
        prompts = [template.format(text=t) for t in texts]

    elif strategy == "label_minimal":
        template = ("Korean: {text}" if direction == "en2ko" else "English: {text}")
        prompts = [template.format(text=t) for t in texts]

    else:
        raise ValueError(f"unknown strategy: {strategy}")

    return prompts


def detect_direction(text: str) -> str:
    lang = "en"
    try:
        lang = detect(text[:200])
    except Exception:
        pass
    return "en2ko" if lang.startswith("en") else "ko2en"


def score_output(inp: str, out: str, direction: str) -> Tuple[float, Dict[str, float]]:
    # 기본 점수 1.0에서 감점
    score = 1.0
    penalties: Dict[str, float] = {}

    # 언어 감지
    target = "ko" if direction == "en2ko" else "en"
    try:
        out_lang = detect(out[:200]) if out.strip() else ""
    except Exception:
        out_lang = ""
    if target not in (out_lang or ""):
        score -= 0.4
        penalties["lang_mismatch"] = 0.4

    # 과도한 길이
    if len(inp) > 0:
        ratio = len(out) / max(1, len(inp))
        if ratio > 3.0:
            score -= 0.3
            penalties["too_long_x3"] = 0.3
        elif ratio > 2.0:
            score -= 0.15
            penalties["too_long_x2"] = 0.15

    # 불필요한 토큰/레이블
    noisy_markers = ["English:", "Korean:", "번역:", "Translation:", "Here is", "This is"]
    if any(m in out for m in noisy_markers):
        score -= 0.2
        penalties["noisy_markers"] = 0.2

    # 큰따옴표. 유저가 선호하지 않으면 감점
    if out.strip().startswith(('"', "'")) or out.strip().endswith(('"', "'")):
        score -= 0.05
        penalties["quotes"] = 0.05

    # 줄바꿈 과다 (한 줄 출력 선호)
    if out.count("\n") >= 1:
        score -= 0.05
        penalties["newline"] = 0.05

    return max(0.0, score), penalties


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", default=os.path.join(os.path.dirname(__file__), "분석결과", "test_prompt.txt"))
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--direction", choices=["auto", "en2ko", "ko2en"], default="auto")
    p.add_argument("--model", default="davidkim205/iris-7b")
    p.add_argument("--max_model_len", type=int, default=1024)
    p.add_argument("--gpu_mem", type=float, default=0.91)
    p.add_argument("--temperature", type=float, default=0.2)
    args = p.parse_args()

    # 로깅 파일 준비
    out_path = setup_logging()

    # 입력 데이터
    samples = read_samples(args.file, args.limit)
    if args.direction == "auto":
        # 첫 샘플을 기준으로 방향 결정
        direction = detect_direction(samples[0])
    else:
        direction = args.direction

    # 모델/샘플링 파라미터
    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)

    llm = LLM(
        model=args.model,
        download_dir=models_dir,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        dtype="auto",
        kv_cache_dtype="fp8",
        enforce_eager=True,
        trust_remote_code=True,
    )

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=0.95,
        max_tokens=128,
        repetition_penalty=1.1,
        skip_special_tokens=True,
        stop=["\n", "English:", "Korean:", "---", "###"],
    )

    strategies = [
        "instruct_fewshot",
        "instruct_strict",
        "translate_minimal",
        "label_minimal",
    ]

    per_strategy_results = {}

    # 결과 파일 작성 시작
    with open(out_path, "w", encoding="utf-8") as wf:
        wf.write("# 프롬프트 튜너 리포트\n\n")
        wf.write(f"- 파일: {args.file}\n")
        wf.write(f"- 샘플 수: {len(samples)}\n")
        wf.write(f"- 방향: {direction}\n")
        wf.write(f"- 모델: {args.model}\n")
        wf.write(f"- 파라미터: gpu_mem={args.gpu_mem}, max_model_len={args.max_model_len}, temperature={args.temperature}\n")
        wf.write("\n---\n\n")

        for strat in strategies:
            total_score = 0.0
            n = 0
            detail_rows = []

            # 배치 추론
            for batch in chunk(samples, 32):
                prompts = build_prompts(batch, direction, strat)
                outputs = llm.generate(prompts, sampling)
                for inp, out in zip(batch, outputs):
                    text = out.outputs[0].text.strip().strip('"').strip("'").strip()
                    s, penalties = score_output(inp, text, direction)
                    total_score += s
                    n += 1
                    detail_rows.append((inp, text, s, penalties))

            avg = total_score / max(1, n)
            per_strategy_results[strat] = dict(avg_score=avg, details=detail_rows)

            wf.write(f"## 전략: {strat}\n")
            wf.write(f"- 평균 점수: {avg:.3f}\n")
            wf.write("- 샘플 3개 예시:\n")
            for i, (inp, out, s, pen) in enumerate(detail_rows[:3], 1):
                wf.write(f"  {i}. 원문: {inp}\n")
                wf.write(f"     번역: {out}\n")
                wf.write(f"     점수: {s:.3f}, 패널티: {pen}\n")
            wf.write("\n")

        # 최종 추천 전략
        best = max(per_strategy_results.items(), key=lambda x: x[1]["avg_score"]) if per_strategy_results else None
        if best:
            name, data = best
            wf.write("---\n\n")
            wf.write("## 최종 추천 프롬프트\n")
            wf.write(f"- 전략: {name}\n")
            wf.write(f"- 평균 점수: {data['avg_score']:.3f}\n\n")

            # 실제 템플릿 예시를 제공
            example = build_prompts(["{TEXT}"], direction, name)[0]
            wf.write("### 템플릿 미리보기\n")
            wf.write("````\n")
            wf.write(example)
            wf.write("\n````\n")

    print("=" * 70)
    print("프롬프트 튜너 실행 완료")
    print(f"결과 파일: {out_path}")


if __name__ == "__main__":
    main()

