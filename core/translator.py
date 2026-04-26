from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path

from epub_translator import SubmitKind, language, translate

from . import patches  # noqa: F401  ← monkey-patches Zip for IRI-encoded hrefs
from .llm_factory import LlamaServerConfig, build_llm
from .progress import ProgressBus
from .prompts import KOREAN_TRANSLATION_RULES


@dataclass
class TranslationOptions:
    submit_mode: SubmitKind = SubmitKind.APPEND_TEXT
    max_group_tokens: int = 2200
    max_retries: int = 5
    concurrency: int = 1
    user_prompt: str = KOREAN_TRANSLATION_RULES


def run_translation(
    source_path: Path,
    target_path: Path,
    cache_root: Path,
    llm_cfg: LlamaServerConfig,
    options: TranslationOptions,
    bus: ProgressBus,
) -> None:
    """Blocking call. Run on a worker thread; emit events via `bus`."""
    try:
        bus.emit("log", f"Building LLM client at {llm_cfg.base_url} (model={llm_cfg.model})")
        llm = build_llm(llm_cfg, cache_root)

        bus.emit("log", f"Translating: {source_path.name} → {target_path.name}")
        bus.emit("log", f"Mode={options.submit_mode.name}, max_group_tokens={options.max_group_tokens}")

        def on_progress(p: float) -> None:
            bus.emit("progress", float(p))

        def on_fill_failed(event: object) -> None:
            bus.emit("fill_fail", str(event))

        translate(
            source_path=str(source_path),
            target_path=str(target_path),
            target_language=language.KOREAN,
            submit=options.submit_mode,
            user_prompt=options.user_prompt,
            max_retries=options.max_retries,
            max_group_tokens=options.max_group_tokens,
            concurrency=options.concurrency,
            llm=llm,
            on_progress=on_progress,
            on_fill_failed=on_fill_failed,
        )

        bus.emit("progress", 1.0)
        bus.emit("done", str(target_path))
    except Exception as exc:
        bus.emit("error", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
