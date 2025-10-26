#!/usr/bin/env python3
"""
Capybara 통합 튜너 & 테스트 실행기

LLM 모델만 대상으로 프롬프트/하이퍼파라미터 튜닝을 수행하고
분석 결과 로그에 상세 설정을 기록합니다. (Seq2Seq 관련 코드는 제거됨)
"""

import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from langdetect import detect
from vllm import LLM, SamplingParams

# =============================================================================
# 공용 데이터셋 & 프롬프트 빌더
# =============================================================================

TEST_SENTENCES: Dict[str, List[str]] = {
    "tech": [
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

# 통합 실행 시 사용할 기본 테스트 설정
TESTING_DOMAINS: List[str] = ["tech", "literary", "slang", "casual", "academic"]
TESTING_LLM_MODEL: str = "davidkim205/iris-7b"
TESTING_PROMPT_LIMIT: int = 10
TESTING_BATCH_SIZE: int = 16
TESTING_PROMPT_TEMPERATURE: float = 0.3
TESTING_GPU_MEM: float = 0.91
TESTING_MAX_MODEL_LEN: int = 1024
TESTING_TOP_P: float = 0.95
TESTING_MAX_TOKENS: int = 256
TESTING_REPETITION_PENALTY: float = 1.15
TESTING_PROMPT_LOG_PREFIX: str = "prompt_tuner_v2"


def build_prompts(texts: List[str], direction: str, strategy: str, domain: str = "general") -> List[str]:
    """도메인/전략별 프롬프트 생성"""
    prompts: List[str] = []

    if direction not in ("en2ko", "ko2en"):
        raise ValueError("direction must be en2ko or ko2en")

    domain_instructions = {
        "tech": "Preserve all technical terms and acronyms. Maintain precise technical meaning.",
        "literary": "Preserve literary tone, metaphors, and poetic expressions. Maintain emotional nuance.",
        "slang": "Translate slang naturally to equivalent Korean expressions. Preserve casual tone.",
        "casual": "Use natural, conversational Korean. Maintain friendly and approachable tone.",
        "academic": "Use formal academic Korean. Preserve scholarly precision and objectivity.",
        "general": "Preserve tone and nuances.",
    }
    domain_inst = domain_instructions.get(domain, domain_instructions["general"])

    if strategy == "domain_fewshot":
        if direction == "en2ko":
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
            elif domain == "slang":
                examples = """Examples:
English: "That new feature is totally fire, no cap!"
Korean: "그 새로운 기능, 진짜 미쳤다 — 뻥 아님!"

English: "He's ghosting me again."
Korean: "또 잠수 탔어."

English: "Let's vibe check this project."
Korean: "분위기 한번 체크해보자."

English: "That's sus."
Korean: "수상하다."
"""
            else:
                examples = """Examples:
English: "Good morning."
Korean: "좋은 아침입니다."

English: "Thank you very much."
Korean: "정말 감사합니다."
"""
            rules = (
                "Rules: Output a single line containing only the Korean translation. "
                "Do NOT include quotes or any labels (e.g., 'Translation:', 'Output:', 'English:', 'Korean:', 'Slang:', 'Text:', 'Translate'). "
                "No explanations."
            )
            template = f"""You are a professional translator specializing in {domain} content.
{domain_inst}
{rules}

{examples}
Now translate this:
English: "{{text}}"
Korean:"""
        else:
            rules = (
                "Rules: Output a single line containing only the English translation. "
                "Do NOT include quotes or any labels (e.g., 'Translation:', 'Output:', 'English:', 'Korean:', 'Slang:', 'Text:', 'Translate'). "
                "No explanations."
            )
            template = f"""You are a professional translator specializing in {domain} content.
{domain_inst}
{rules}

Now translate this:
Korean: "{{text}}"
English:"""

        prompts = [template.format(text=t) for t in texts]

    elif strategy == "instruct_only":
        if direction == "en2ko":
            template = f"""Translate English to Korean. {domain_inst}
Rules: Output a single line containing only the Korean translation. No quotes, no labels, no explanations.

Text: {{text}}
Korean:"""
        else:
            template = f"""Translate Korean to English. {domain_inst}
Rules: Output a single line containing only the English translation. No quotes, no labels, no explanations.

Text: {{text}}
English:"""
        prompts = [template.format(text=t) for t in texts]

    elif strategy == "minimal":
        template = "Translate to Korean:\n{text}\n" if direction == "en2ko" else "Translate to English:\n{text}\n"
        # 최소 전략에도 단일 라인 출력 규칙을 명시해 라벨/반복을 방지
        if direction == "en2ko":
            template = "Translate to Korean. Output only the translation in one line.\n{text}\n"
        else:
            template = "Translate to English. Output only the translation in one line.\n{text}\n"
        prompts = [template.format(text=t) for t in texts]

    elif strategy == "cot_style":
        if direction == "en2ko":
            template = f"""You are translating {domain} content from English to Korean.
{domain_inst}

Step 1: Understand the meaning and context
Step 2: Translate naturally to Korean
Step 3: Output ONLY the final Korean translation as a single line (no quotes, no labels)

English: {{text}}
Korean translation:"""
        else:
            template = f"""You are translating {domain} content from Korean to English.
{domain_inst}

Step 1: Understand the meaning and context
Step 2: Translate naturally to English
Step 3: Output ONLY the final English translation as a single line (no quotes, no labels)

Korean: {{text}}
English translation:"""
        prompts = [template.format(text=t) for t in texts]

    elif strategy == "role_based":
        if direction == "en2ko":
            template = f"""You are an expert translator specialized in {domain} texts.
Your task: Translate English to natural Korean.
Requirements: {domain_inst}
Rules: Output a single line containing only the Korean translation. No quotes, no labels, no explanations.

[English Text]
{{text}}

[Korean Translation]
"""
        else:
            template = f"""You are an expert translator specialized in {domain} texts.
Your task: Translate Korean to natural English.
Requirements: {domain_inst}
Rules: Output a single line containing only the English translation. No quotes, no labels, no explanations.

[Korean Text]
{{text}}

[English Translation]
"""
        prompts = [template.format(text=t) for t in texts]

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return prompts


def score_output(inp: str, out: str, direction: str, domain: str) -> Tuple[float, Dict[str, float]]:
    """번역 결과 점수화 (언어 감지 및 도메인별 페널티 포함)"""
    score = 1.0
    penalties: Dict[str, float] = {}

    target = "ko" if direction == "en2ko" else "en"
    try:
        out_lang = detect(out[:200]) if out.strip() else ""
    except Exception:
        out_lang = ""

    if target not in (out_lang or ""):
        score -= 0.5
        penalties["lang_mismatch"] = 0.5

    if len(inp) > 0:
        ratio = len(out) / max(1, len(inp))
        if ratio > 3.5:
            score -= 0.3
            penalties["too_long_x3.5"] = 0.3
        elif ratio > 2.5:
            score -= 0.2
            penalties["too_long_x2.5"] = 0.2

    noisy_markers = [
        "English:", "Korean:", "번역:", "Translation:",
        "Here is", "This is", "[", "]", "Output:", "Result:"
    ]
    if any(m in out for m in noisy_markers):
        score -= 0.25
        penalties["noisy_markers"] = 0.25

    if out.strip().startswith(('"', "'")) and out.strip().endswith(('"', "'")):
        score -= 0.1
        penalties["quotes"] = 0.1

    if out.count("\n") >= 2:
        score -= 0.1
        penalties["excess_newlines"] = 0.1

    if domain == "tech":
        tech_terms = ["API", "GPU", "CPU", "JSON", "HTTP", "SQL", "URL", "JWT", "OAuth", "CI", "CD"]
        input_terms = [t for t in tech_terms if t in inp]
        if direction == "en2ko" and input_terms:
            preserved = any(term in out for term in input_terms)
            if not preserved:
                score -= 0.15
                penalties["tech_term_loss"] = 0.15

    elif domain == "literary":
        if direction == "en2ko" and len(out) < len(inp) * 0.5:
            score -= 0.1
            penalties["literary_too_short"] = 0.1

    elif domain == "slang":
        if direction == "en2ko":
            formal_markers = ["입니다", "습니다", "십시오"]
            if sum(out.count(m) for m in formal_markers) > 2:
                score -= 0.1
                penalties["slang_too_formal"] = 0.1

    elif domain == "academic":
        if direction == "en2ko":
            casual_markers = ["ㅋ", "ㅎ", "!", "~", "요~"]
            if any(m in out for m in casual_markers):
                score -= 0.15
                penalties["academic_too_casual"] = 0.15

    return max(0.0, score), penalties


def chunk(lst: List[str], size: int) -> List[List[str]]:
    """리스트를 지정된 배치 크기로 분할"""
    return [lst[i:i + size] for i in range(0, len(lst), size)]


def clean_output(text: str, target_lang: Optional[str] = None) -> str:
    """출력 후처리: 라벨/설명 제거, 따옴표/공백 제거, 언어 필터링.

    Args:
        text: 원 출력 문자열
        target_lang: 'ko' 또는 'en' 중 선택 시 해당 언어의 줄을 우선 선택
    """
    label_markers = [
        "English:", "Korean:", "Translation:", "Output:",
        "Slang:", "Text:", "Translate", "번역:", "출력:", "Rules:",
    ]
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    # 라벨 라인 제거
    filtered = [ln for ln in lines if not any(ln.startswith(m) for m in label_markers)]
    candidates = filtered or lines or [""]

    # 언어 매칭 우선
    if target_lang in ("ko", "en"):
        for ln in candidates:
            try:
                lang = detect(ln[:200]) if ln else ""
            except Exception:
                lang = ""
            if target_lang in (lang or ""):
                candidates = [ln]
                break

    # 첫 줄 선택 후 따옴표 제거
    out = candidates[0].strip()
    if (out.startswith('"') and out.endswith('"')) or (out.startswith("'") and out.endswith("'")):
        out = out[1:-1].strip()
    return out


def setup_logging(prefix: str = "prompt_tuner_v2") -> str:
    """프롬프트 튜너용 로그 파일 경로 생성"""
    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, "분석결과")
    os.makedirs(results_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(results_dir, f"{prefix}_{ts}.log")


def run_prompt_tuner(
    domains: List[str],
    limit: int = 10,
    model: str = "davidkim205/iris-7b",
    max_model_len: int = 1024,
    gpu_mem: float = 0.91,
    temperature: float = 0.3,
    top_p: float = 0.95,
    max_tokens: int = 256,
    repetition_penalty: float = 1.15,
    stop_tokens: Optional[List[str]] = None,
    batch_size: int = 16,
    log_prefix: str = "prompt_tuner_v2",
    strategies: Optional[List[str]] = None,
) -> Dict[str, object]:
    """프롬프트 전략별 번역 품질을 비교하고 로그 파일을 생성"""
    if "all" in domains:
        selected_domains = list(TEST_SENTENCES.keys())
    else:
        selected_domains = domains

    out_path = setup_logging(prefix=log_prefix)

    print("=" * 70)
    print(f"모델 로딩 중: {model}")
    print("=" * 70)

    models_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(models_dir, exist_ok=True)

    llm = LLM(
        model=model,
        download_dir=models_dir,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        dtype="auto",
        kv_cache_dtype="fp8",
        enforce_eager=True,
        trust_remote_code=True,
    )

    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
        skip_special_tokens=True,
        stop=(stop_tokens or DEFAULT_STOP_TOKENS),
    )

    if strategies is None:
        strategies = [
            "domain_fewshot",
            "instruct_only",
            "minimal",
            "cot_style",
            "role_based",
        ]

    all_results: Dict[str, Dict[str, Dict[str, object]]] = defaultdict(lambda: defaultdict(dict))
    strategy_totals: Dict[str, List[float]] = defaultdict(list)
    sorted_strategy_list: List[Tuple[str, float]] = []

    with open(out_path, "w", encoding="utf-8") as wf:
        wf.write("[LOG] 결과 파일: {}\n".format(out_path))
        wf.write("=" * 70 + "\n")
        wf.write("LLM 프롬프트 & 하이퍼파라미터 튜너\n")
        wf.write("=" * 70 + "\n\n")

        # 설정 상세
        wf.write("[설정]\n")
        wf.write(f"  모델: {model}\n")
        wf.write(f"  도메인: {', '.join(selected_domains)}\n")
        wf.write(f"  각 도메인당 문장 수: {limit}\n")
        wf.write(f"  배치 크기: {batch_size}\n")
        wf.write(f"  방향: en→ko (고정)\n")
        wf.write(f"  생성 파라미터:\n")
        wf.write(f"    - temperature: {temperature}\n")
        wf.write(f"    - top_p: {top_p}\n")
        wf.write(f"    - max_tokens: {max_tokens}\n")
        wf.write(f"    - repetition_penalty: {repetition_penalty}\n")
        wf.write(f"    - stop: {', '.join((stop_tokens or DEFAULT_STOP_TOKENS))}\n")
        wf.write(f"  시스템 자원:\n")
        wf.write(f"    - gpu_mem: {gpu_mem}\n")
        wf.write(f"    - max_model_len: {max_model_len}\n")
        wf.write(f"  전략 목록: {', '.join(strategies)}\n")
        wf.write(f"  출력 규칙: 단 한 줄, 따옴표/라벨/설명 금지 (clean_output 적용)\n")
        wf.write(f"  폴백: 언어 불일치 시 role_based 1회 재시도\n")
        wf.write(f"  생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for domain in selected_domains:
            print(f"\n{'=' * 70}")
            print(f"도메인: {domain.upper()}")
            print(f"{'=' * 70}")

            wf.write("=" * 70 + "\n")
            wf.write(f"도메인: {domain.upper()}\n")
            wf.write("=" * 70 + "\n\n")

            samples = TEST_SENTENCES[domain][:limit]
            direction = "en2ko"

            wf.write(f"📋 테스트 문장 {len(samples)}개:\n")
            for i, s in enumerate(samples, 1):
                wf.write(f"  {i}. {s}\n")
            wf.write("\n")

            for strat in strategies:
                print(f"\n전략: {strat}")

                total_score = 0.0
                n = 0
                detail_rows = []

                for batch in chunk(samples, batch_size):
                    prompts = build_prompts(batch, direction, strat, domain)
                    outputs = llm.generate(prompts, sampling)

                    for inp, out in zip(batch, outputs):
                        raw = out.outputs[0].text
                        text = clean_output(raw, target_lang="ko" if direction == "en2ko" else "en")

                        s, penalties = score_output(inp, text, direction, domain)
                        # 언어 불일치 시 role_based 전략으로 1회 폴백 시도
                        if penalties.get("lang_mismatch"):
                            fb_prompt = build_prompts([inp], direction, "role_based", domain)[0]
                            fb_out = llm.generate([fb_prompt], sampling)[0]
                            fb_text = clean_output(
                                fb_out.outputs[0].text,
                                target_lang="ko" if direction == "en2ko" else "en"
                            )
                            s2, penalties2 = score_output(inp, fb_text, direction, domain)
                            if s2 > s:
                                text = fb_text
                                s = s2
                                penalties = {**penalties2, "fallback": "role_based"}
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

        wf.write("=" * 70 + "\n")
        wf.write("전체 요약\n")
        wf.write("=" * 70 + "\n\n")

        wf.write("📊 도메인별 최고 전략\n")
        wf.write("-" * 70 + "\n\n")

        for domain in selected_domains:
            best_strat = max(
                all_results[domain].items(),
                key=lambda x: x[1]["avg_score"]
            )
            strat_name, strat_data = best_strat

            wf.write(f"🏆 {domain.upper()}\n")
            wf.write(f"   최고 전략: {strat_name}\n")
            wf.write(f"   평균 점수: {strat_data['avg_score']:.3f}\n\n")

        wf.write("📈 전략별 전체 평균 점수\n")
        wf.write("-" * 70 + "\n\n")

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

        wf.write("\n" + "=" * 70 + "\n")
        wf.write("최종 추천\n")
        wf.write("=" * 70 + "\n\n")

        if sorted_strategies:
            best_overall = sorted_strategies[0]
            wf.write(f"✅ 전체 최고 전략: {best_overall[0]}\n")
            wf.write(f"   평균 점수: {best_overall[1]:.3f}\n\n")
            wf.write("💡 권장사항\n")
            wf.write("-" * 70 + "\n")
            wf.write("  1. 범용 번역에는 '{}' 전략을 사용하세요.\n".format(best_overall[0]))
            wf.write("  2. 특정 도메인에 특화된 번역이 필요한 경우, 해당 도메인의 최고 전략을 참고하세요.\n")
            wf.write("  3. 기술 문서는 few-shot 예시를 포함한 프롬프트가 효과적입니다.\n")
            wf.write("  4. 문학 작품은 문체 보존을 명시한 instruction이 중요합니다.\n")
            wf.write("  5. 구어체/신조어는 형식적 표현을 피하도록 지시해야 합니다.\n")

        sorted_strategy_list = sorted_strategies

    print("\n" + "=" * 70)
    print("프롬프트 튜너 v2.0 실행 완료!")
    print(f"결과 파일: {out_path}")
    print("=" * 70)

    return {
        "log_path": out_path,
        "selected_domains": selected_domains,
        "strategy_rankings": sorted_strategy_list,
        "results": all_results,
    }


def _apply_best_to_hyperparams(model: str, cfg: Dict[str, object], stop_tokens: List[str]) -> None:
    """최고 구성값을 hyperparams.py에 기록하여 앱이 자동 적용하도록 함."""
    hp_path = os.path.join(os.path.dirname(__file__), "hyperparams.py")
    content = [
        '"""',
        'Capybara LLM 하이퍼파라미터 기본값 (튜너에 의해 자동 갱신됨)',
        '',
        '튜너(tuner.py)가 최적 조합을 찾으면 이 파일을 업데이트하여',
        '앱(capybara.py)이 다음 실행부터 자동으로 적용합니다.',
        '"""',
        '',
        'from typing import Dict, List',
        '',
        '# 공용 Stop 토큰: 한 줄 출력 유도 + 라벨/명령 토큰에서 즉시 중지',
        'DEFAULT_STOP_TOKENS: List[str] = [',
        "    " + ", ".join(repr(s) for s in stop_tokens),
        ']',
        '',
        f'LLM_MODEL: str = {repr(model)}',
        '',
        'VLLM_OPTS: Dict[str, object] = {',
        '    "tensor_parallel_size": 1,',
        '    "gpu_memory_utilization": 0.91,',
        '    "max_model_len": 1024,',
        '    "dtype": "auto",',
        '    "kv_cache_dtype": "fp8",',
        '    "enforce_eager": True,',
        '    "trust_remote_code": True,',
        '}',
        '',
        'SAMPLING_DEFAULTS: Dict[str, object] = {',
        f'    "temperature": {cfg.get("temperature", 0.3)},',
        f'    "top_p": {cfg.get("top_p", 0.95)},',
        f'    "max_tokens": {cfg.get("max_tokens", 256)},',
        f'    "repetition_penalty": {cfg.get("repetition_penalty", 1.15)},',
        '    "skip_special_tokens": True,',
        '    "stop": DEFAULT_STOP_TOKENS,',
        '}',
        '',
    ]
    with open(hp_path, "w", encoding="utf-8") as f:
        f.write("\n".join(content))


# =============================================================================
# 공용 Stop 토큰 정의
# =============================================================================

# 단 한 줄 출력 유도 + 라벨/명령 토큰에서 즉시 중지
DEFAULT_STOP_TOKENS = [
    "\n", "English:", "Korean:", "Translation:", "Output:",
    "Slang:", "Text:", "Translate", "---", "###", "[END]"
]

def main() -> None:
    """사전 정의된 그리드 스윕으로 LLM 프롬프트/하이퍼파라미터 튜닝 실행"""
    print("=" * 70)
    print("🚀 Capybara LLM 프롬프트/하이퍼파라미터 튜너 실행")
    print("=" * 70)
    print(f"🧪 Testing LLM model: {TESTING_LLM_MODEL}")
    print("=" * 70)

    # 작은 범위의 스윕 (필요 시 확장)
    temps = sorted(set([0.2, TESTING_PROMPT_TEMPERATURE]))
    reps = sorted(set([1.1, TESTING_REPETITION_PENALTY]))
    top_ps = [TESTING_TOP_P]
    max_toks = [TESTING_MAX_TOKENS]

    best_overall = None  # (score, cfg, result)
    sweep_log_path = os.path.join(
        os.path.dirname(__file__), "분석결과",
        f"prompt_tuner_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    os.makedirs(os.path.dirname(sweep_log_path), exist_ok=True)
    with open(sweep_log_path, "w", encoding="utf-8") as sw:
        sw.write("스윕 요약\n")
        sw.write("=" * 70 + "\n")
        sw.write(f"모델: {TESTING_LLM_MODEL}\n")
        sw.write(f"도메인: {', '.join(TESTING_DOMAINS)}\n")
        sw.write(f"배치 크기: {TESTING_BATCH_SIZE}, 문장 수: {TESTING_PROMPT_LIMIT}\n")
        sw.write(f"고정 stop: {', '.join(DEFAULT_STOP_TOKENS)}\n\n")

        for t in temps:
            for rp in reps:
                for tp in top_ps:
                    for mx in max_toks:
                        cfg = {
                            "temperature": t,
                            "repetition_penalty": rp,
                            "top_p": tp,
                            "max_tokens": mx,
                        }
                        print(f"[SWEEP] cfg={cfg}")
                        result = run_prompt_tuner(
                            domains=TESTING_DOMAINS,
                            limit=TESTING_PROMPT_LIMIT,
                            model=TESTING_LLM_MODEL,
                            max_model_len=TESTING_MAX_MODEL_LEN,
                            gpu_mem=TESTING_GPU_MEM,
                            temperature=t,
                            top_p=tp,
                            max_tokens=mx,
                            repetition_penalty=rp,
                            stop_tokens=DEFAULT_STOP_TOKENS,
                            batch_size=TESTING_BATCH_SIZE,
                            log_prefix=TESTING_PROMPT_LOG_PREFIX,
                            strategies=None,
                        )
                        top_strat = result["strategy_rankings"][0] if result["strategy_rankings"] else ("n/a", 0.0)
                        score = top_strat[1]
                        sw.write(f"cfg={cfg} | best_strategy={top_strat[0]} | score={score:.3f} | log={result['log_path']}\n")
                        if (best_overall is None) or (score > best_overall[0]):
                            best_overall = (score, cfg, result)

        sw.write("\n최고 구성\n")
        sw.write("-" * 70 + "\n")
        if best_overall:
            sw.write(f"score={best_overall[0]:.3f}, cfg={best_overall[1]}, log={best_overall[2]['log_path']}\n")

    if best_overall:
        print("-" * 70)
        print(f"[BEST] score={best_overall[0]:.3f}, cfg={best_overall[1]}")
        print(f"[BEST] log={best_overall[2]['log_path']}")
        print(f"[SWEEP] 요약 로그: {sweep_log_path}")
        # hyperparams.py 업데이트
        try:
            _apply_best_to_hyperparams(
                model=TESTING_LLM_MODEL,
                cfg=best_overall[1],
                stop_tokens=DEFAULT_STOP_TOKENS,
            )
            print("[APPLY] hyperparams.py 업데이트 완료")
        except Exception as e:
            print(f"[APPLY] hyperparams.py 업데이트 실패: {e}")
    print("=" * 70)
    print("✅ 튜닝 실행 완료")
    print("=" * 70)


if __name__ == "__main__":
    main()
