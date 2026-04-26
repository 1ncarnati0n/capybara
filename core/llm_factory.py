from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from epub_translator import LLM

from .prompts import STRICT_XML_FILL_RULES


@dataclass
class LlamaServerConfig:
    base_url: str = "http://127.0.0.1:9070/v1"
    api_key: str = "sk-local"
    model: str = "exaone-3.5-7.8b-instruct"
    token_encoding: str = "o200k_base"
    timeout: float = 300.0
    temperature: float = 0.3
    top_p: float = 0.9
    retry_times: int = 5
    retry_interval_seconds: float = 6.0


def build_llm(cfg: LlamaServerConfig, cache_root: Path) -> LLM:
    cache_root.mkdir(parents=True, exist_ok=True)
    llm = LLM(
        key=cfg.api_key,
        url=cfg.base_url,
        model=cfg.model,
        token_encoding=cfg.token_encoding,
        timeout=cfg.timeout,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        retry_times=cfg.retry_times,
        retry_interval_seconds=cfg.retry_interval_seconds,
        cache_path=str(cache_root / "llm"),
        log_dir_path=str(cache_root / "log"),
    )
    llm._templates["fill"] = llm._env.from_string(STRICT_XML_FILL_RULES)
    return llm
