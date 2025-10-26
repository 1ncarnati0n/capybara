"""
Capybara LLM 하이퍼파라미터 기본값 (튜너에 의해 자동 갱신됨)

튜너(tuner.py)가 최적 조합을 찾으면 이 파일을 업데이트하여
앱(capybara.py)이 다음 실행부터 자동으로 적용합니다.
"""

from typing import Dict, List

# 공용 Stop 토큰: 한 줄 출력 유도 + 라벨/명령 토큰에서 즉시 중지
DEFAULT_STOP_TOKENS: List[str] = [
    '\n', 'English:', 'Korean:', 'Translation:', 'Output:', 'Slang:', 'Text:', 'Translate', '---', '###', '[END]'
]

LLM_MODEL: str = 'davidkim205/iris-7b'

VLLM_OPTS: Dict[str, object] = {
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.91,
    "max_model_len": 1024,
    "dtype": "auto",
    "kv_cache_dtype": "fp8",
    "enforce_eager": True,
    "trust_remote_code": True,
}

SAMPLING_DEFAULTS: Dict[str, object] = {
    "temperature": 0.3,
    "top_p": 0.95,
    "max_tokens": 256,
    "repetition_penalty": 1.1,
    "skip_special_tokens": True,
    "stop": DEFAULT_STOP_TOKENS,
}
