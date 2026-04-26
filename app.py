from __future__ import annotations

import json
import html
import re
import sys
import threading
import time
import uuid
import urllib.error
import urllib.request
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

_UNSAFE_FILENAME_CHARS_RE = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')

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
    st.session_state.setdefault("server_check", None)


def _safe_book_title(filename: str) -> str:
    title = Path(filename).stem.strip()
    title = _UNSAFE_FILENAME_CHARS_RE.sub("_", title)
    title = re.sub(r"\s+", " ", title).strip(" ._")
    return title or "book"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    for index in range(2, 1000):
        candidate = path.with_name(f"{path.stem}-{index}{path.suffix}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"Cannot find available output filename for {path.name}")


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    minutes, seconds_part = divmod(total_seconds, 60)
    return f"{minutes}m{seconds_part}s"


def _final_output_path(title: str, started_at: float) -> Path:
    duration = _format_duration(time.monotonic() - started_at)
    return _unique_path(OUTPUT / f"{title}-{duration}_ko.epub")


def _start_job(src: Path, dst: Path, title: str, llm_cfg: LlamaServerConfig, opts: TranslationOptions) -> None:
    bus = ProgressBus()
    started_at = time.monotonic()
    thread = threading.Thread(
        target=run_translation,
        args=(src, dst, CACHE, llm_cfg, opts, bus),
        daemon=True,
    )
    thread.start()
    st.session_state.job = {
        "thread": thread,
        "bus": bus,
        "src": str(src),
        "dst": str(dst),
        "title": title,
        "started_at": started_at,
    }
    st.session_state.logs = []
    st.session_state.progress = 0.0
    st.session_state.result_path = None
    st.session_state.error = None


def _models_endpoint(base_url: str) -> str:
    return base_url.rstrip("/") + "/models"


def _server_command(preset_name: str) -> str:
    if "EXAONE" in preset_name:
        return "scripts/lm-server.sh"
    return "MODEL=/path/to/model.gguf PORT=9090 ALIAS=my-model scripts/lm-server.sh"


def _check_server(base_url: str, api_key: str) -> dict[str, object]:
    endpoint = _models_endpoint(base_url)
    request = urllib.request.Request(endpoint)
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")

    started_at = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=1.5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return {
            "ok": False,
            "endpoint": endpoint,
            "latency_ms": None,
            "models": [],
            "error": str(exc),
        }

    latency_ms = int((time.monotonic() - started_at) * 1000)
    model_ids: list[str] = []
    for key in ("data", "models"):
        for item in payload.get(key, []) if isinstance(payload, dict) else []:
            if isinstance(item, dict):
                model_id = item.get("id") or item.get("model") or item.get("name")
                if model_id and str(model_id) not in model_ids:
                    model_ids.append(str(model_id))

    return {
        "ok": True,
        "endpoint": endpoint,
        "latency_ms": latency_ms,
        "models": model_ids,
        "error": None,
    }


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
            result_path = Path(ev.payload)
            try:
                title = str(job.get("title") or _safe_book_title(result_path.stem))
                started_at = float(job.get("started_at") or time.monotonic())
                final_path = _final_output_path(title, started_at)
                if result_path.exists():
                    result_path.rename(final_path)
                    result_path = final_path
                st.session_state.result_path = str(result_path)
                st.session_state.logs.append(f"[done] {result_path}")
            except Exception as exc:
                st.session_state.error = f"{type(exc).__name__}: {exc}"
                st.session_state.logs.append(f"[error] {st.session_state.error}")
        elif ev.kind == "error":
            st.session_state.error = ev.payload
            st.session_state.logs.append(f"[error] {ev.payload}")
    return job["thread"].is_alive()


def _active_state(job_active: bool) -> str:
    if st.session_state.error:
        return "오류"
    if st.session_state.result_path:
        return "완료"
    if job_active:
        return "번역 중"
    return "대기"


def _recent_logs(limit: int = 4) -> str:
    logs = st.session_state.logs[-limit:]
    return "\n".join(logs) if logs else "(최근 이벤트 없음)"


def _render_status_cards(active_state: str, model: str, server_check: dict[str, object] | None) -> None:
    server_label = "미확인"
    if server_check is not None:
        server_label = "연결됨" if server_check.get("ok") else "연결 안 됨"
    active_state = html.escape(active_state)
    server_label = html.escape(server_label)
    model = html.escape(model)
    output_name = html.escape(OUTPUT.name)

    st.markdown(
        f"""
        <div class="status-strip">
          <div class="status-card">
            <div class="status-label">작업 상태</div>
            <div class="status-value">{active_state}</div>
          </div>
          <div class="status-card">
            <div class="status-label">서버</div>
            <div class="status-value">{server_label}</div>
          </div>
          <div class="status-card">
            <div class="status-label">모델</div>
            <div class="status-value">{model}</div>
          </div>
          <div class="status-card">
            <div class="status-label">출력 폴더</div>
            <div class="status-value">{output_name}/</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.6rem; max-width: 1220px; }
        [data-testid="stSidebar"] { background: #f7f8fb; }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { margin-top: 0.55rem; }
        .status-strip {
            display: grid;
            grid-template-columns: 1fr 1fr 1.7fr 1fr;
            gap: 0.75rem;
            margin: 0.9rem 0 1.1rem;
        }
        .status-card {
            min-height: 76px;
            border: 1px solid rgba(49, 51, 63, 0.14);
            border-radius: 8px;
            padding: 0.72rem 0.85rem;
            background: #ffffff;
        }
        .status-label {
            color: rgba(49, 51, 63, 0.62);
            font-size: 0.76rem;
            margin-bottom: 0.22rem;
        }
        .status-value {
            font-weight: 650;
            font-size: 0.96rem;
            overflow-wrap: anywhere;
        }
        .command-box {
            border: 1px solid rgba(49, 51, 63, 0.14);
            border-radius: 8px;
            background: #fafafa;
            padding: 0.7rem 0.8rem;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            font-size: 0.84rem;
            overflow-wrap: anywhere;
        }
        .section-note {
            color: rgba(49, 51, 63, 0.68);
            font-size: 0.9rem;
        }
        @media (max-width: 860px) {
            .status-strip { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="EN→KO EPUB Translator", page_icon="📘", layout="wide")
    _init_state()
    _render_css()

    with st.sidebar:
        st.header("모델 서버")
        preset_name = st.selectbox("프리셋", options=list(MODEL_PRESETS), index=0)
        preset = MODEL_PRESETS[preset_name]
        preset_key = re.sub(r"[^a-zA-Z0-9_]+", "_", preset_name)
        st.caption(f"프로파일: {preset['profile']}")

        base_url = st.text_input("Base URL", value=str(preset["base_url"]), key=f"base_url_{preset_key}")
        model = st.text_input("Model name", value=str(preset["model"]), key=f"model_{preset_key}")
        api_key = st.text_input("API key", value="sk-local", type="password")
        if st.session_state.server_check is not None:
            if st.session_state.server_check.get("endpoint") != _models_endpoint(base_url):
                st.session_state.server_check = None

        col_check, col_hint = st.columns([1, 1])
        with col_check:
            if st.button("연결 확인", use_container_width=True):
                st.session_state.server_check = _check_server(base_url, api_key)
        with col_hint:
            st.caption(_server_command(preset_name))

        server_check = st.session_state.server_check
        if server_check is None:
            st.info("선택한 서버가 켜져 있는지 확인하세요.")
        elif server_check.get("ok"):
            st.success(f"연결됨 · {server_check.get('latency_ms')}ms")
            served_models = server_check.get("models") or []
            if served_models:
                st.caption("서버 모델: " + ", ".join(str(m) for m in served_models[:2]))
        else:
            st.warning("서버 연결 안 됨")
            st.markdown(
                f'<div class="command-box">{_server_command(preset_name)}</div>',
                unsafe_allow_html=True,
            )

        with st.expander("생성 파라미터", expanded=False):
            temperature = st.slider(
                "Temperature",
                0.0,
                1.0,
                float(preset["temperature"]),
                0.05,
                key=f"temperature_{preset_key}",
            )
            top_p = st.slider(
                "Top-p",
                0.1,
                1.0,
                float(preset["top_p"]),
                0.05,
                key=f"top_p_{preset_key}",
            )
            timeout = st.number_input(
                "Timeout (s)",
                min_value=30,
                max_value=1800,
                value=int(preset["timeout"]),
                step=30,
                key=f"timeout_{preset_key}",
            )

        st.header("번역 설정")
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
            key=f"max_group_tokens_{preset_key}",
        )
        max_retries = st.number_input("XML fill 최대 재시도", min_value=1, max_value=20, value=5)

    job_active = _drain_events()
    active_state = _active_state(job_active)
    server_check = st.session_state.server_check

    st.title("영→한 EPUB 번역 콘솔")
    st.caption("로컬 llama.cpp 서버와 epub-translator 기반")
    _render_status_cards(active_state, model, server_check)

    upload_tab, progress_tab, result_tab, log_tab, settings_tab = st.tabs(
        ["작업", "진행", "결과", "로그", "설정 요약"]
    )

    with upload_tab:
        col_upload, col_ready = st.columns([1.2, 1])
        with col_upload:
            st.subheader("EPUB 업로드")
            uploaded = st.file_uploader("English EPUB", type=["epub"], disabled=job_active)
            if uploaded is not None:
                title = _safe_book_title(uploaded.name)
                st.caption(f"저장될 업로드명: `{title}__<id>.epub`")
            else:
                title = "book"

            start_disabled = job_active or uploaded is None
            if st.button("번역 시작", type="primary", disabled=start_disabled, use_container_width=True):
                job_id = uuid.uuid4().hex[:8]
                title = _safe_book_title(uploaded.name)
                src_path = UPLOADS / f"{title}__{job_id}.epub"
                dst_path = OUTPUT / f"{title}__{job_id}.tmp.epub"
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
                _start_job(src_path, dst_path, title, llm_cfg, opts)
                st.rerun()

        with col_ready:
            st.subheader("실행 준비")
            st.metric("출력 모드", mode_label.split(" ")[0])
            st.metric("청크 토큰", int(max_group_tokens))
            if server_check is not None and server_check.get("ok"):
                st.success("서버 연결 확인 완료")
            else:
                st.warning("서버 연결을 먼저 확인하세요.")
                st.markdown(
                    f'<div class="command-box">{_server_command(preset_name)}</div>',
                    unsafe_allow_html=True,
                )

    with progress_tab:
        st.subheader("진행 상황")
        progress_value = min(max(st.session_state.progress, 0.0), 1.0)
        st.progress(progress_value)
        col_state, col_percent, col_recent = st.columns([1, 1, 2])
        col_state.metric("상태", active_state)
        col_percent.metric("진행률", f"{progress_value * 100:.0f}%")
        col_recent.text_area("최근 이벤트", value=_recent_logs(), height=130, disabled=True)
        if job_active:
            st.info("번역 진행 중입니다. 진행률은 자동 갱신됩니다.")
        elif st.session_state.error:
            st.error("오류로 종료되었습니다. 로그 탭에서 상세 내용을 확인하세요.")

    with result_tab:
        st.subheader("결과")
        result_path = st.session_state.result_path
        if result_path:
            result = Path(result_path)
            if result.exists():
                st.success(f"완료: {result.name}")
                st.caption(f"저장 경로: `{result}`")
                st.download_button(
                    "번역본 EPUB 다운로드",
                    data=result.read_bytes(),
                    file_name=result.name,
                    mime="application/epub+zip",
                    use_container_width=True,
                )
            else:
                st.warning("결과 경로가 기록되어 있지만 파일을 찾을 수 없습니다.")
        else:
            st.info("완료된 번역본이 아직 없습니다.")

    with log_tab:
        st.subheader("로그")
        st.code("\n".join(st.session_state.logs[-300:]) or "(no logs yet)", language="text")

    with settings_tab:
        st.subheader("현재 설정")
        st.json(
            {
                "preset": preset_name,
                "base_url": base_url,
                "model": model,
                "profile": preset["profile"],
                "submit_mode": submit_mode.name,
                "temperature": float(temperature),
                "top_p": float(top_p),
                "timeout": int(timeout),
                "max_group_tokens": int(max_group_tokens),
                "max_retries": int(max_retries),
                "server_command": _server_command(preset_name),
            }
        )

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
