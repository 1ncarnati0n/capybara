from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

from epub_translator import LLM, SubmitKind, language, translate

from . import patches  # noqa: F401  # monkey-patches EPUB/XML handling
from .control import TranslationCancelled, TranslationControl
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
    control: TranslationControl | None = None,
) -> None:
    """Blocking call. Run on a worker thread; emit events via `bus`."""
    stop_stats = threading.Event()
    control = control or TranslationControl()

    def emit_stats(llm: LLM, started_at: float) -> None:
        elapsed = max(time.monotonic() - started_at, 0.001)
        total_tokens = int(llm.total_tokens)
        bus.emit(
            "stats",
            {
                "total_tokens": total_tokens,
                "input_tokens": int(llm.input_tokens),
                "input_cache_tokens": int(llm.input_cache_tokens),
                "output_tokens": int(llm.output_tokens),
                "tokens_per_sec": total_tokens / elapsed,
            },
        )

    try:
        bus.emit("log", f"Building LLM client at {llm_cfg.base_url} (model={llm_cfg.model})")
        llm = build_llm(llm_cfg, cache_root)
        llm.deterministic_xml_fill = options.submit_mode == SubmitKind.APPEND_TEXT
        llm.translation_control = control
        stats_started_at = time.monotonic()

        def stats_monitor() -> None:
            while not stop_stats.wait(1.0):
                emit_stats(llm, stats_started_at)

        stats_thread = threading.Thread(target=stats_monitor, daemon=True)
        stats_thread.start()

        bus.emit("log", f"Translating: {source_path.name} → {target_path.name}")
        bus.emit("log", f"Mode={options.submit_mode.name}, max_group_tokens={options.max_group_tokens}")

        def on_progress(p: float) -> None:
            control.checkpoint()
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

        control.checkpoint()
        bus.emit("progress", 1.0)
        bus.emit("done", str(target_path))
    except TranslationCancelled as exc:
        bus.emit("cancelled", str(exc))
    except Exception as exc:
        bus.emit("error", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
    finally:
        stop_stats.set()
        if "stats_thread" in locals():
            stats_thread.join(timeout=0.2)
        if "llm" in locals() and "stats_started_at" in locals():
            emit_stats(llm, stats_started_at)
