#!/usr/bin/env python3
"""
프롬프트 튜너 v2.0 (Advanced Prompt Tuner)

다양한 영어 표현(기술 전문 분야, 문학 표현, 신조어, 일상, 학술 등)을
테스트하여 iris-7b 모델의 최적 프롬프트 전략을 찾습니다.

특징:
- 파일 내부에 도메인별 테스트 문장 내장
- 5개 도메인 지원: 기술, 문학, 신조어/속어, 일상, 학술
- 도메인별 평가 지표 (전문용어 보존, 문체 일관성 등)
- 다양한 프롬프트 전략 테스트
- 결과는 capybara/분석결과/*.md 로 저장

사용 예시:
  conda activate capybara && cd capybara
  python test_prompt_tuner.py --domains all --limit 10
  python test_prompt_tuner.py --domains tech literary --limit 5
"""

import os
import sys
import time
import argparse
from datetime import datetime
from typing import List, Tuple, Dict
from collections import defaultdict

from vllm import LLM, SamplingParams
from langdetect import detect


# =============================================================================
# 도메인별 테스트 문장 데이터셋
# =============================================================================

TEST_SENTENCES = {
    "tech": [
        # 기술 전문 용어 및 개발 표현
        "The microservice architecture enables horizontal scaling through containerization.",
        "We implemented a zero-trust security model using JWT tokens and OAuth2 flows.",
        "The GPU utilizes CUDA cores for parallel processing of tensor operations.",
        "Kubernetes orchestrates containerized applications across distributed clusters.",
        "The API gateway handles rate limiting, authentication, and request routing.",
        "We optimized database queries using indexing and query plan analysis.",
        "The CI/CD pipeline automates testing, building, and deployment processes.",
        "TensorFlow provides automatic differentiation for gradient-based optimization.",
        "The blockchain uses Merkle trees to ensure data integrity and immutability.",
        "WebAssembly enables near-native performance for web applications.",
        "The neural network employs dropout regularization to prevent overfitting.",
        "Redis implements an in-memory key-value store with persistence options.",
        "The distributed system achieves consensus through the Raft algorithm.",
        "GraphQL allows clients to request exactly the data they need.",
        "We leverage edge computing to reduce latency for real-time applications.",
    ],

    "literary": [
        # 문학적 표현 및 비유적 언어
        "The autumn leaves danced gracefully in the gentle breeze.",
        "Her laughter echoed through the empty corridors like a forgotten melody.",
        "Time stood still as he gazed into her eyes, lost in their depths.",
        "The old house creaked and groaned, whispering secrets of bygone eras.",
        "His heart was a fortress, impenetrable and cold.",
        "The city slept beneath a blanket of stars, peaceful and serene.",
        "She carried the weight of the world on her delicate shoulders.",
        "The river of time flows ceaselessly, washing away all sorrows.",
        "His words were daggers, sharp and unforgiving.",
        "The moon hung low, a silver lantern in the velvet night.",
        "Memory is a treacherous companion, beautiful yet unreliable.",
        "The silence between them spoke volumes that words never could.",
        "Hope flickered like a candle flame in the darkness of despair.",
        "The ocean's whisper carried tales of distant lands and forgotten dreams.",
        "She was a wildflower, thriving in the most unexpected places.",
    ],

    "slang": [
        # 신조어, 속어, 인터넷 밈
        "That new feature is totally fire, no cap!",
        "He's ghosting me again, ugh, so annoying.",
        "Let's vibe check this project before we ship it.",
        "She's lowkey the GOAT at coding.",
        "This bug is sus, we need to investigate ASAP.",
        "I'm deadass tired of debugging this legacy code.",
        "That's a big yikes from me, chief.",
        "This API is chef's kiss, absolutely perfect.",
        "He really said that? The audacity!",
        "I'm gonna touch grass after this sprint.",
        "That solution is so galaxy brain, I love it.",
        "Stop the cap, you know that's not true.",
        "This meeting could've been an email, not gonna lie.",
        "She's giving main character energy today.",
        "Let's circle back on this after lunch, bet.",
    ],

    "casual": [
        # 일상 회화 및 자연스러운 표현
        "How's your day going so far?",
        "I'll grab some coffee on my way to the office.",
        "Let me know if you need any help with that.",
        "Thanks for reaching out, I really appreciate it.",
        "Sorry I'm running a bit late, traffic is crazy today.",
        "Would you like to grab lunch together sometime this week?",
        "I'm not sure I understand what you mean, could you explain?",
        "That sounds like a great idea, let's do it!",
        "I'm feeling a bit under the weather today.",
        "It's been a while since we last caught up.",
        "I'm swamped with work right now, can we talk later?",
        "Don't worry about it, these things happen.",
        "I'm looking forward to seeing you soon.",
        "Could you do me a favor and send that file over?",
        "I completely forgot about our meeting, my bad!",
    ],

    "academic": [
        # 학술 논문 스타일 및 formal writing
        "The empirical results demonstrate a statistically significant correlation.",
        "We propose a novel framework for evaluating neural network robustness.",
        "The methodology employed in this study builds upon prior research.",
        "Subsequent analysis revealed several noteworthy limitations.",
        "This phenomenon can be attributed to underlying systemic factors.",
        "The findings suggest a paradigm shift in our understanding of cognition.",
        "We hypothesize that increased exposure leads to enhanced performance.",
        "The data was collected through a double-blind randomized control trial.",
        "These results are consistent with previously established theories.",
        "Further investigation is warranted to elucidate the causal mechanisms.",
        "The study's implications extend beyond the immediate scope of inquiry.",
        "We observed a pronounced effect in the treatment group relative to controls.",
        "This approach mitigates potential confounding variables effectively.",
        "The theoretical framework provides a comprehensive lens for analysis.",
        "Our conclusions are subject to the inherent limitations of cross-sectional data.",
    ],
}


# =============================================================================
# 프롬프트 전략 정의
# =============================================================================

def build_prompts(texts: List[str], direction: str, strategy: str, domain: str = "general") -> List[str]:
    """
    다양한 프롬프트 전략 생성

    Args:
        texts: 번역할 텍스트 리스트
        direction: "en2ko" 또는 "ko2en"
        strategy: 프롬프트 전략명
        domain: 도메인 타입 (tech, literary, slang, casual, academic)
    """
    prompts: List[str] = []

    if direction not in ("en2ko", "ko2en"):
        raise ValueError("direction must be en2ko or ko2en")

    # 도메인별 지침
    domain_instructions = {
        "tech": "Preserve all technical terms and acronyms. Maintain precise technical meaning.",
        "literary": "Preserve literary tone, metaphors, and poetic expressions. Maintain emotional nuance.",
        "slang": "Translate slang naturally to equivalent Korean expressions. Preserve casual tone.",
        "casual": "Use natural, conversational Korean. Maintain friendly and approachable tone.",
        "academic": "Use formal academic Korean. Preserve scholarly precision and objectivity.",
        "general": "Preserve tone and nuances.",
    }

    domain_inst = domain_instructions.get(domain, domain_instructions["general"])

    # Strategy 1: Domain-Aware Few-Shot
    if strategy == "domain_fewshot":
        if direction == "en2ko":
            # 도메인별 예시 선택
            if domain == "tech":
                examples = """Examples:
English: "The API returns a JSON response."
Korean: "API는 JSON 응답을 반환합니다."

English: "We use Docker for containerization."
Korean: "컨테이너화를 위해 Docker를 사용합니다."
"""
            elif domain == "literary":
                examples = """Examples:
English: "Her eyes sparkled like stars."
Korean: "그녀의 눈은 별처럼 빛났다."

English: "Time heals all wounds."
Korean: "시간은 모든 상처를 치유한다."
"""
            else:
                examples = """Examples:
English: "Good morning."
Korean: "좋은 아침입니다."

English: "Thank you very much."
Korean: "정말 감사합니다."
"""

            template = f"""You are a professional translator specializing in {domain} content.
{domain_inst}
Output only the Korean translation without quotes or labels.

{examples}
Now translate this:
English: "{{text}}"
Korean:"""

        else:  # ko2en
            template = f"""You are a professional translator specializing in {domain} content.
{domain_inst}
Output only the English translation without quotes or labels.

Now translate this:
Korean: "{{text}}"
English:"""

        prompts = [template.format(text=t) for t in texts]

    # Strategy 2: Instruction-Only (No Examples)
    elif strategy == "instruct_only":
        if direction == "en2ko":
            template = f"""Translate English to Korean. {domain_inst}
Output only the translation, no quotes or labels.

Text: {{text}}
Translation:"""
        else:
            template = f"""Translate Korean to English. {domain_inst}
Output only the translation, no quotes or labels.

Text: {{text}}
Translation:"""

        prompts = [template.format(text=t) for t in texts]

    # Strategy 3: Minimal Prompt
    elif strategy == "minimal":
        if direction == "en2ko":
            template = "Translate to Korean:\n{text}\n"
        else:
            template = "Translate to English:\n{text}\n"

        prompts = [template.format(text=t) for t in texts]

    # Strategy 4: Chain-of-Thought Style
    elif strategy == "cot_style":
        if direction == "en2ko":
            template = f"""You are translating {domain} content from English to Korean.
{domain_inst}

Step 1: Understand the meaning and context
Step 2: Translate naturally to Korean
Step 3: Output ONLY the final Korean translation

English: {{text}}
Korean translation:"""
        else:
            template = f"""You are translating {domain} content from Korean to English.
{domain_inst}

Step 1: Understand the meaning and context
Step 2: Translate naturally to English
Step 3: Output ONLY the final English translation

Korean: {{text}}
English translation:"""

        prompts = [template.format(text=t) for t in texts]

    # Strategy 5: Role-Based Prompt
    elif strategy == "role_based":
        if direction == "en2ko":
            template = f"""You are an expert translator specialized in {domain} texts.
Your task: Translate English to natural Korean.
Requirements: {domain_inst}
Output: Only the Korean translation, nothing else.

[English Text]
{{text}}

[Korean Translation]
"""
        else:
            template = f"""You are an expert translator specialized in {domain} texts.
Your task: Translate Korean to natural English.
Requirements: {domain_inst}
Output: Only the English translation, nothing else.

[Korean Text]
{{text}}

[English Translation]
"""

        prompts = [template.format(text=t) for t in texts]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return prompts


# =============================================================================
# 평가 함수
# =============================================================================

def score_output(inp: str, out: str, direction: str, domain: str) -> Tuple[float, Dict[str, float]]:
    """
    번역 출력 평가 (도메인별 특화 지표 포함)

    Returns:
        (점수, 패널티 세부사항) 튜플
    """
    score = 1.0
    penalties: Dict[str, float] = {}

    # 1. 언어 감지
    target = "ko" if direction == "en2ko" else "en"
    try:
        out_lang = detect(out[:200]) if out.strip() else ""
    except Exception:
        out_lang = ""

    if target not in (out_lang or ""):
        score -= 0.5
        penalties["lang_mismatch"] = 0.5

    # 2. 과도한 길이
    if len(inp) > 0:
        ratio = len(out) / max(1, len(inp))
        if ratio > 3.5:
            score -= 0.3
            penalties["too_long_x3.5"] = 0.3
        elif ratio > 2.5:
            score -= 0.2
            penalties["too_long_x2.5"] = 0.2

    # 3. 불필요한 레이블/마커
    noisy_markers = [
        "English:", "Korean:", "번역:", "Translation:",
        "Here is", "This is", "[", "]", "Output:", "Result:"
    ]
    if any(m in out for m in noisy_markers):
        score -= 0.25
        penalties["noisy_markers"] = 0.25

    # 4. 큰따옴표 과다 사용
    if out.strip().startswith(('"', "'")) and out.strip().endswith(('"', "'")):
        score -= 0.1
        penalties["quotes"] = 0.1

    # 5. 줄바꿈 과다
    if out.count("\n") >= 2:
        score -= 0.1
        penalties["excess_newlines"] = 0.1

    # 6. 도메인별 특화 평가
    if domain == "tech":
        # 기술 용어: 대문자 약어 보존 확인
        tech_terms = ["API", "GPU", "CPU", "JSON", "HTTP", "SQL", "URL", "JWT", "OAuth", "CI", "CD"]
        input_terms = [t for t in tech_terms if t in inp]
        # 한글 번역에서 원어 보존 또는 적절한 번역 여부 (간접 체크)
        if direction == "en2ko" and input_terms:
            # 출력에 기술용어가 일부라도 보존되었는지 확인
            preserved = any(term in out for term in input_terms)
            if not preserved and len(input_terms) > 0:
                score -= 0.15
                penalties["tech_term_loss"] = 0.15

    elif domain == "literary":
        # 문학 표현: 과도하게 직역되지 않았는지 (길이가 너무 짧으면 감점)
        if direction == "en2ko" and len(out) < len(inp) * 0.5:
            score -= 0.1
            penalties["literary_too_short"] = 0.1

    elif domain == "slang":
        # 신조어/속어: 너무 형식적이면 안 됨 (존댓말 과다 사용 체크)
        if direction == "en2ko":
            formal_markers = ["입니다", "습니다", "십시오"]
            if sum(out.count(m) for m in formal_markers) > 2:
                score -= 0.1
                penalties["slang_too_formal"] = 0.1

    elif domain == "academic":
        # 학술 표현: 너무 구어체면 안 됨
        if direction == "en2ko":
            casual_markers = ["ㅋ", "ㅎ", "!", "~", "요~"]
            if any(m in out for m in casual_markers):
                score -= 0.15
                penalties["academic_too_casual"] = 0.15

    return max(0.0, score), penalties


# =============================================================================
# 유틸리티 함수
# =============================================================================

def setup_logging(prefix: str = "prompt_tuner_v2") -> str:
    """로깅 파일 경로 생성"""
    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, "분석결과")
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(results_dir, f"{prefix}_{ts}.log")
    return path


def chunk(lst: List[str], size: int) -> List[List[str]]:
    """리스트를 배치 크기로 분할"""
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def detect_direction(text: str) -> str:
    """텍스트 언어 감지"""
    lang = "en"
    try:
        lang = detect(text[:200])
    except Exception:
        pass
    return "en2ko" if lang.startswith("en") else "ko2en"


# =============================================================================
# 메인 함수
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="프롬프트 튜너 v2.0 - 도메인별 번역 품질 테스트"
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        choices=["tech", "literary", "slang", "casual", "academic", "all"],
        default=["all"],
        help="테스트할 도메인 선택 (기본값: all)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="각 도메인당 테스트할 문장 수 (기본값: 10)"
    )
    parser.add_argument(
        "--model",
        default="davidkim205/iris-7b",
        help="사용할 모델명"
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=1024,
        help="최대 모델 길이"
    )
    parser.add_argument(
        "--gpu_mem",
        type=float,
        default=0.91,
        help="GPU 메모리 사용률 (0.0~1.0)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="샘플링 temperature (기본값: 0.2)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="배치 크기 (기본값: 16)"
    )

    args = parser.parse_args()

    # 도메인 선택
    if "all" in args.domains:
        selected_domains = list(TEST_SENTENCES.keys())
    else:
        selected_domains = args.domains

    # 로깅 파일 준비
    out_path = setup_logging()

    # 모델 로딩
    print("=" * 70)
    print(f"모델 로딩 중: {args.model}")
    print("=" * 70)

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
        max_tokens=256,
        repetition_penalty=1.1,
        skip_special_tokens=True,
        stop=["\n\n", "English:", "Korean:", "---", "###", "[END]"],
    )

    # 테스트할 프롬프트 전략들
    strategies = [
        "domain_fewshot",
        "instruct_only",
        "minimal",
        "cot_style",
        "role_based",
    ]

    # 결과 저장용
    all_results = defaultdict(lambda: defaultdict(dict))

    # 로그 파일 작성
    with open(out_path, "w", encoding="utf-8") as wf:
        # 헤더
        wf.write("[LOG] 결과 파일: {}\n".format(out_path))
        wf.write("="*70 + "\n")
        wf.write("프롬프트 튜너 v2.0 - 도메인별 번역 품질 테스트\n")
        wf.write("="*70 + "\n\n")
        wf.write(f"🔧 모델: {args.model}\n")
        wf.write(f"📁 테스트 도메인: {', '.join(selected_domains)}\n")
        wf.write(f"📝 각 도메인당 문장 수: {args.limit}\n")
        wf.write(f"⚙️  파라미터: gpu_mem={args.gpu_mem}, max_model_len={args.max_model_len}, temperature={args.temperature}\n")
        wf.write(f"🕐 생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 도메인별 테스트
        for domain in selected_domains:
            print(f"\n{'='*70}")
            print(f"도메인: {domain.upper()}")
            print(f"{'='*70}")

            wf.write("="*70 + "\n")
            wf.write(f"도메인: {domain.upper()}\n")
            wf.write("="*70 + "\n\n")

            samples = TEST_SENTENCES[domain][:args.limit]
            direction = "en2ko"  # 영한 번역 고정 (테스트 데이터가 영어이므로)

            wf.write(f"📋 테스트 문장 {len(samples)}개:\n")
            for i, s in enumerate(samples, 1):
                wf.write(f"  {i}. {s}\n")
            wf.write("\n")

            # 각 전략별 테스트
            for strat in strategies:
                print(f"\n전략: {strat}")

                total_score = 0.0
                n = 0
                detail_rows = []

                # 배치 추론
                for batch in chunk(samples, args.batch_size):
                    prompts = build_prompts(batch, direction, strat, domain)
                    outputs = llm.generate(prompts, sampling)

                    for inp, out in zip(batch, outputs):
                        text = out.outputs[0].text.strip().strip('"').strip("'").strip()

                        # 도메인별 평가
                        s, penalties = score_output(inp, text, direction, domain)
                        total_score += s
                        n += 1
                        detail_rows.append((inp, text, s, penalties))

                avg = total_score / max(1, n)
                all_results[domain][strat] = {
                    "avg_score": avg,
                    "details": detail_rows
                }

                print(f"  평균 점수: {avg:.3f}")

                wf.write(f"🔹 전략: {strat}\n")
                wf.write(f"   평균 점수: {avg:.3f}\n")
                wf.write(f"   샘플 3개 예시:\n\n")
                for i, (inp, out, s, pen) in enumerate(detail_rows[:3], 1):
                    wf.write(f"   [예시 {i}]\n")
                    wf.write(f"   원문: {inp}\n")
                    wf.write(f"   번역: {out}\n")
                    wf.write(f"   점수: {s:.3f}\n")
                    if pen:
                        wf.write(f"   패널티: {pen}\n")
                    wf.write("\n")

            wf.write("\n")

        # 전체 요약
        wf.write("="*70 + "\n")
        wf.write("전체 요약\n")
        wf.write("="*70 + "\n\n")

        wf.write("📊 도메인별 최고 전략\n")
        wf.write("-"*70 + "\n\n")

        for domain in selected_domains:
            best_strat = max(
                all_results[domain].items(),
                key=lambda x: x[1]["avg_score"]
            )
            strat_name, strat_data = best_strat

            wf.write(f"🏆 {domain.upper()}\n")
            wf.write(f"   최고 전략: {strat_name}\n")
            wf.write(f"   평균 점수: {strat_data['avg_score']:.3f}\n\n")

        # 전략별 평균 점수
        wf.write("📈 전략별 전체 평균 점수\n")
        wf.write("-"*70 + "\n\n")

        strategy_totals = defaultdict(list)
        for domain in selected_domains:
            for strat in strategies:
                strategy_totals[strat].append(all_results[domain][strat]["avg_score"])

        strategy_avgs = {
            strat: sum(scores) / len(scores)
            for strat, scores in strategy_totals.items()
        }

        sorted_strategies = sorted(
            strategy_avgs.items(),
            key=lambda x: x[1],
            reverse=True
        )

        for rank, (strat, avg) in enumerate(sorted_strategies, 1):
            wf.write(f"  {rank}. {strat}: {avg:.3f}\n")

        # 최종 추천
        wf.write("\n" + "="*70 + "\n")
        wf.write("최종 추천\n")
        wf.write("="*70 + "\n\n")

        best_overall = sorted_strategies[0]
        wf.write(f"✅ 전체 최고 전략: {best_overall[0]}\n")
        wf.write(f"   평균 점수: {best_overall[1]:.3f}\n\n")

        wf.write("💡 권장사항\n")
        wf.write("-"*70 + "\n")
        wf.write("  1. 범용 번역에는 '{}' 전략을 사용하세요.\n".format(best_overall[0]))
        wf.write("  2. 특정 도메인에 특화된 번역이 필요한 경우, 해당 도메인의 최고 전략을 참고하세요.\n")
        wf.write("  3. 기술 문서는 few-shot 예시를 포함한 프롬프트가 효과적입니다.\n")
        wf.write("  4. 문학 작품은 문체 보존을 명시한 instruction이 중요합니다.\n")
        wf.write("  5. 구어체/신조어는 형식적 표현을 피하도록 지시해야 합니다.\n")

        wf.write("\n")

    print("\n" + "=" * 70)
    print("프롬프트 튜너 v2.0 실행 완료!")
    print(f"결과 파일: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
