from __future__ import annotations

import sys
import threading
import time
import uuid
from pathlib import Path

import streamlit as st
import streamlit.runtime as st_runtime
from epub_translator import SubmitKind

from core.llm_factory import LlamaServerConfig
from core.progress import ProgressBus
from core.translator import TranslationOptions, run_translation

ROOT = Path(__file__).parent
UPLOADS = ROOT / "uploads"
OUTPUT = ROOT / "output"
CACHE = ROOT / "cache"
for d in (UPLOADS, OUTPUT, CACHE):
    d.mkdir(parents=True, exist_ok=True)

MODEL_PRESETS = {
    "EXAONE 3.5 7.8B Q6_K (현재)": {
        "base_url": "http://127.0.0.1:9070/v1",
        "model": "exaone-3.5-7.8b-instruct",
        "temperature": 0.25,
        "top_p": 0.9,
        "timeout": 300,
        "max_group_tokens": 2200,
        "profile": "안정",
    },
    "Yanolja Rosetta 12B 2510 Q5_K_M": {
        "base_url": "http://127.0.0.1:9090/v1",
        "model": "yanolja/YanoljaNEXT-Rosetta-12B-2510",
        "temperature": 0.1,
        "top_p": 0.9,
        "timeout": 420,
        "max_group_tokens": 1800,
        "profile": "번역 특화",
    },
    "EXAONE 3.5 32B Q4_K_M": {
        "base_url": "http://127.0.0.1:9080/v1",
        "model": "exaone-3.5-32b-instruct",
        "temperature": 0.2,
        "top_p": 0.9,
        "timeout": 600,
        "max_group_tokens": 1800,
        "profile": "고품질",
    },
    "Custom": {
        "base_url": "http://127.0.0.1:9070/v1",
        "model": "custom-model",
        "temperature": 0.3,
        "top_p": 0.9,
        "timeout": 300,
        "max_group_tokens": 2200,
        "profile": "수동",
    },
}


def _init_state() -> None:
    st.session_state.setdefault("job", None)  # dict | None
    st.session_state.setdefault("logs", [])
    st.session_state.setdefault("progress", 0.0)
    st.session_state.setdefault("result_path", None)
    st.session_state.setdefault("error", None)


def _start_job(src: Path, dst: Path, llm_cfg: LlamaServerConfig, opts: TranslationOptions) -> None:
    bus = ProgressBus()
    thread = threading.Thread(
        target=run_translation,
        args=(src, dst, CACHE, llm_cfg, opts, bus),
        daemon=True,
    )
    thread.start()
    st.session_state.job = {"thread": thread, "bus": bus, "src": str(src), "dst": str(dst)}
    st.session_state.logs = []
    st.session_state.progress = 0.0
    st.session_state.result_path = None
    st.session_state.error = None


def _drain_events() -> bool:
    """Pull events from the worker into session state. Return True if job is still active."""
    job = st.session_state.job
    if job is None:
        return False
    bus: ProgressBus = job["bus"]
    for ev in bus.drain():
        if ev.kind == "progress":
            st.session_state.progress = float(ev.payload)
        elif ev.kind == "log":
            st.session_state.logs.append(f"[log] {ev.payload}")
        elif ev.kind == "fill_fail":
            st.session_state.logs.append(f"[fill-retry] {ev.payload}")
        elif ev.kind == "done":
            st.session_state.result_path = ev.payload
            st.session_state.logs.append(f"[done] {ev.payload}")
        elif ev.kind == "error":
            st.session_state.error = ev.payload
            st.session_state.logs.append(f"[error] {ev.payload}")
    return job["thread"].is_alive()


def main() -> None:
    st.set_page_config(page_title="EN→KO EPUB Translator", page_icon="📘", layout="wide")
    _init_state()

    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; max-width: 1180px; }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { margin-top: 0.6rem; }
        .status-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.75rem 0 1.25rem;
        }
        .status-card {
            border: 1px solid rgba(49, 51, 63, 0.16);
            border-radius: 8px;
            padding: 0.75rem 0.9rem;
            background: rgba(250, 250, 250, 0.65);
        }
        .status-label {
            color: rgba(49, 51, 63, 0.64);
            font-size: 0.78rem;
            margin-bottom: 0.15rem;
        }
        .status-value {
            font-weight: 650;
            font-size: 0.98rem;
            overflow-wrap: anywhere;
        }
        @media (max-width: 760px) {
            .status-strip { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("영→한 EPUB 번역기")
    st.caption("로컬 llama.cpp 서버와 epub-translator 기반")

    with st.sidebar:
        st.header("모델")
        preset_name = st.selectbox("프리셋", options=list(MODEL_PRESETS), index=0)
        preset = MODEL_PRESETS[preset_name]
        st.caption(f"프로파일: {preset['profile']}")

        base_url = st.text_input("Base URL", value=str(preset["base_url"]))
        model = st.text_input("Model name", value=str(preset["model"]))
        api_key = st.text_input("API key", value="sk-local", type="password")

        with st.expander("생성 파라미터", expanded=False):
            temperature = st.slider("Temperature", 0.0, 1.0, float(preset["temperature"]), 0.05)
            top_p = st.slider("Top-p", 0.1, 1.0, float(preset["top_p"]), 0.05)
            timeout = st.number_input(
                "Timeout (s)",
                min_value=30,
                max_value=1800,
                value=int(preset["timeout"]),
                step=30,
            )

        st.header("번역")
        mode_label = st.selectbox(
            "출력 모드",
            options=["APPEND_TEXT (인라인)", "REPLACE (한국어만)"],
            index=0,
        )
        submit_mode = {
            "APPEND_TEXT (인라인)": SubmitKind.APPEND_TEXT,
            "REPLACE (한국어만)": SubmitKind.REPLACE,
        }[mode_label]
        max_group_tokens = st.number_input(
            "max_group_tokens (tiktoken o200k_base)",
            min_value=512, max_value=8000, value=int(preset["max_group_tokens"]), step=100,
            help="청크당 토큰 상한. EXAONE 토크나이저는 o200k_base보다 한국어를 짧게 인코딩하므로 보수적으로 잡으면 안전합니다.",
        )
        max_retries = st.number_input("XML fill 최대 재시도", min_value=1, max_value=20, value=5)

    job_active = _drain_events()

    active_state = "번역 중" if job_active else "대기"
    if st.session_state.error:
        active_state = "오류"
    elif st.session_state.result_path:
        active_state = "완료"
    st.markdown(
        f"""
        <div class="status-strip">
          <div class="status-card">
            <div class="status-label">상태</div>
            <div class="status-value">{active_state}</div>
          </div>
          <div class="status-card">
            <div class="status-label">모델</div>
            <div class="status-value">{model}</div>
          </div>
          <div class="status-card">
            <div class="status-label">출력</div>
            <div class="status-value">{OUTPUT.name}/</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([2, 3])

    with col_left:
        st.subheader("EPUB")
        uploaded = st.file_uploader("English EPUB", type=["epub"], disabled=job_active)

        start_disabled = job_active or uploaded is None
        if st.button("번역 시작", type="primary", disabled=start_disabled, use_container_width=True):
            job_id = uuid.uuid4().hex[:8]
            src_path = UPLOADS / f"{job_id}__{uploaded.name}"
            dst_path = OUTPUT / f"{job_id}__{src_path.stem}.ko.epub"
            src_path.write_bytes(uploaded.getvalue())

            llm_cfg = LlamaServerConfig(
                base_url=base_url,
                api_key=api_key,
                model=model,
                temperature=float(temperature),
                top_p=float(top_p),
                timeout=float(timeout),
            )
            opts = TranslationOptions(
                submit_mode=submit_mode,
                max_group_tokens=int(max_group_tokens),
                max_retries=int(max_retries),
            )
            _start_job(src_path, dst_path, llm_cfg, opts)
            st.rerun()

    with col_right:
        st.subheader("진행 상황")
        st.progress(min(max(st.session_state.progress, 0.0), 1.0))
        if job_active:
            st.info("번역 진행 중… 진행률은 자동 갱신됩니다.")
        elif st.session_state.error:
            st.error("오류로 종료되었습니다. 아래 로그를 확인하세요.")
        elif st.session_state.result_path:
            result = Path(st.session_state.result_path)
            if result.exists():
                st.success(f"완료: {result.name}")
                st.download_button(
                    "번역본 EPUB 다운로드",
                    data=result.read_bytes(),
                    file_name=result.name,
                    mime="application/epub+zip",
                    use_container_width=True,
                )

        with st.expander("로그", expanded=True):
            st.code("\n".join(st.session_state.logs[-200:]) or "(no logs yet)", language="text")

    if job_active:
        time.sleep(0.8)
        st.rerun()


def cli() -> None:
    if not st_runtime.exists():
        from streamlit.web.cli import main as streamlit_main

        sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
        raise SystemExit(streamlit_main())
    main()


if __name__ == "__main__":
    cli()
