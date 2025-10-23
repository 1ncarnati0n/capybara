# Changelog

## 2025-10-22 — v2.2.0
- iris-7b를 기본 LLM으로 전환하고 vLLM 권장값(`gpu_memory_utilization=0.91`, `max_model_len=1024`, `kv_cache_dtype=fp8`)을 정리
- README를 단일 진입점으로 재구성하고 IRIS-7B/Usage/Testing 문서를 통합 정리
- 테스트 스크립트 출력 위치와 모델 캐시 경로를 `capybara/models/`로 통일

## 2025-10-22 — v2.1.0
- 번역 완료 후 자동 초기화 및 번역 히스토리 로그 추가
- Gradio UI에 실시간 로그 패널 제공, 연속 번역 워크플로우 정비

## 2025-10-22 — v2.0.0
- LLM(EEVE-2.8B) + Seq2Seq(NLLB) 듀얼 모델 구조 도입
- 출력 파일명에 모델 접미사(`_llm`, `_s2s`) 적용
- 모델 비교용 테스트 스크립트 추가

## 2025-10-20 — v1.0.0
- 초기 릴리스: vLLM 기반 번역 파이프라인, TXT/EPUB/SRT 지원, Gradio UI 제공
