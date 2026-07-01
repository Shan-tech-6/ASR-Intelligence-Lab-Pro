"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                        ASR INTELLIGENCE LAB PRO                                     ║
║                    Production-Ready · Streamlit Cloud Safe                          ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║  Engines  : Whisper · Faster-Whisper · Sarvam AI · Gemini · Shrutam               ║
║  Mic      : streamlit_mic_recorder  (no sounddevice / no ALSA)                     ║
║  TTS      : gTTS                                                                    ║
║  Metrics  : Latency · RTF · WER · CER · Accuracy                                  ║
║  Extras   : Live Captions · TTS Playback · Excel Log · Download Transcripts        ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────────────
# STDLIB
# ─────────────────────────────────────────────────────────────────────────────────────
import io
import os
import time
import wave
import tempfile
import datetime as dt
import traceback
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List

# ─────────────────────────────────────────────────────────────────────────────────────
# THIRD-PARTY (always available on Streamlit Cloud)
# ─────────────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────────────
# OPTIONAL SOFT IMPORTS — every block is isolated; missing package = graceful skip
# ─────────────────────────────────────────────────────────────────────────────────────

# Mic recorder (NO sounddevice / NO ALSA)
try:
    from streamlit_mic_recorder import mic_recorder
    MIC_RECORDER_AVAILABLE = True
except Exception:
    MIC_RECORDER_AVAILABLE = False

# Whisper (local)
try:
    import whisper as openai_whisper
    WHISPER_AVAILABLE = True
except Exception:
    WHISPER_AVAILABLE = False

# Faster-Whisper (local, CTranslate2)
try:
    from faster_whisper import WhisperModel as FWModel
    FASTER_WHISPER_AVAILABLE = True
except Exception:
    FASTER_WHISPER_AVAILABLE = False

# HTTP client for Sarvam AI + Shrutam
try:
    import requests as _requests
    REQUESTS_AVAILABLE = True
except Exception:
    REQUESTS_AVAILABLE = False

# Gemini (Google Generative AI)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

# WER / CER
try:
    import jiwer
    JIWER_AVAILABLE = True
except Exception:
    JIWER_AVAILABLE = False

# TTS — gTTS
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception:
    GTTS_AVAILABLE = False

# Plotly charts
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

# Excel export
try:
    import openpyxl  # noqa: F401 – just checking availability
    OPENPYXL_AVAILABLE = True
except Exception:
    OPENPYXL_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════════════
SARVAM_API_URL        = "https://api.sarvam.ai/speech-to-text"
SHRUTAM_API_URL       = "https://api.shrutam.ai/v1/transcribe"
GEMINI_MODEL          = "gemini-2.0-flash"
SARVAM_MAX_SECS       = 30.0
LOG_FILE              = "asr_lab_pro_logs.xlsx"
LIVE_CAPTION_LIMIT    = 5          # auto-clear after N recordings in live mode
SAMPLE_RATE           = 16_000

LANGUAGES = [
    ("auto", "🌐 Auto Detect"),
    ("en",   "🇬🇧 English"),
    ("hi",   "🇮🇳 Hindi"),
    ("ta",   "🇮🇳 Tamil"),
    ("te",   "🇮🇳 Telugu"),
    ("kn",   "🇮🇳 Kannada"),
    ("ml",   "🇮🇳 Malayalam"),
    ("mr",   "🇮🇳 Marathi"),
    ("bn",   "🇮🇳 Bengali"),
    ("gu",   "🇮🇳 Gujarati"),
    ("pa",   "🇮🇳 Punjabi"),
    ("ur",   "🇵🇰 Urdu"),
    ("or",   "🇮🇳 Odia"),
    ("as",   "🇮🇳 Assamese"),
    ("es",   "🇪🇸 Spanish"),
    ("fr",   "🇫🇷 French"),
    ("de",   "🇩🇪 German"),
    ("zh",   "🇨🇳 Chinese"),
    ("ja",   "🇯🇵 Japanese"),
    ("ko",   "🇰🇷 Korean"),
    ("ar",   "🇸🇦 Arabic"),
    ("pt",   "🇧🇷 Portuguese"),
    ("ru",   "🇷🇺 Russian"),
]
LANG_CODE_TO_LABEL = {c: l for c, l in LANGUAGES}

ENGINE_COLORS = {
    "Whisper":        "#4F46E5",
    "Faster-Whisper": "#06B6D4",
    "Sarvam AI":      "#F59E0B",
    "Shrutam":        "#EC4899",
    "Gemini":         "#10B981",
}

SARVAM_LANG_MAP = {
    "hi": "hi-IN", "ta": "ta-IN", "te": "te-IN", "kn": "kn-IN",
    "ml": "ml-IN", "mr": "mr-IN", "bn": "bn-IN", "gu": "gu-IN",
    "pa": "pa-IN", "or": "od-IN", "en": "en-IN",
}


# ══════════════════════════════════════════════════════════════════════════════════════
# DATA MODEL
# ══════════════════════════════════════════════════════════════════════════════════════
@dataclass
class EngineResult:
    engine:            str
    transcript:        str  = ""
    detected_language: str  = ""
    latency_sec:       float = 0.0
    rtf:               float = 0.0
    wer:               Optional[float] = None
    cer:               Optional[float] = None
    accuracy:          Optional[float] = None
    status:            str  = "pending"   # success | skipped | error
    error_message:     str  = ""

    def to_row(self) -> Dict[str, Any]:
        return asdict(self)


# ══════════════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════════════

def bytes_to_wav_file(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    ext = os.path.splitext(filename)[-1].lower() or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(audio_bytes)
    tmp.flush()
    tmp.close()

    if ext == ".mp3":
        wav_path = tmp.name.replace(".mp3", ".wav")
        os.system(f'ffmpeg -y -i "{tmp.name}" -ar 16000 -ac 1 "{wav_path}" -loglevel quiet')
        return wav_path
    return tmp.name


def get_wav_duration(path: str) -> float:
    """Return duration of a WAV file in seconds."""
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return 0.0


def compute_metrics(reference: str, hypothesis: str) -> Dict[str, Optional[float]]:
    """Compute WER / CER / Accuracy; returns Nones if jiwer unavailable or no ref."""
    if not reference or not JIWER_AVAILABLE:
        return {"wer": None, "cer": None, "accuracy": None}
    try:
        w = jiwer.wer(reference, hypothesis)
        c = jiwer.cer(reference, hypothesis)
        return {"wer": round(w, 4), "cer": round(c, 4),
                "accuracy": round(max(0.0, min(1.0, 1.0 - w)), 4)}
    except Exception:
        return {"wer": None, "cer": None, "accuracy": None}


def log_to_excel(rows: List[Dict], path: str = LOG_FILE):
    """Append rows to Excel workbook; create if missing."""
    if not OPENPYXL_AVAILABLE:
        return False, "openpyxl not installed"
    try:
        new_df = pd.DataFrame(rows)
        if os.path.exists(path):
            df = pd.concat([pd.read_excel(path), new_df], ignore_index=True)
        else:
            df = new_df
        df.to_excel(path, index=False)
        return True, os.path.abspath(path)
    except Exception as e:
        return False, str(e)


def make_tts_audio(text: str, lang_code: str = "en") -> Optional[bytes]:
    """Convert text → MP3 bytes using gTTS; returns None on failure."""
    if not GTTS_AVAILABLE or not text.strip():
        return None
    try:
        # gTTS needs a 2-letter ISO code; fall back to "en"
        tts_lang = lang_code if lang_code != "auto" and len(lang_code) == 2 else "en"
        tts = gTTS(text=text, lang=tts_lang, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════════════
# MODEL CACHES  (never reload across reruns)
# ══════════════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="⏳ Loading Whisper model…")
def get_whisper_model(size: str):
    return openai_whisper.load_model(size)


@st.cache_resource(show_spinner="⏳ Loading Faster-Whisper model…")
def get_faster_whisper_model(size: str):
    return FWModel(size, device="cpu", compute_type="int8")


# ══════════════════════════════════════════════════════════════════════════════════════
# ASR ENGINE WRAPPERS
# ══════════════════════════════════════════════════════════════════════════════════════

def run_whisper(path: str, size: str, lang: str) -> EngineResult:
    r = EngineResult(engine="Whisper")
    if not WHISPER_AVAILABLE:
        r.status = "error"; r.error_message = "openai-whisper not installed"; return r
    try:
        model = get_whisper_model(size)
        kwargs = {"task": "transcribe"}
        if lang != "auto":
           kwargs["language"] = lang
        t0 = time.perf_counter()
        out = model.transcribe(path, **kwargs)
        r.latency_sec      = round(time.perf_counter() - t0, 3)
        r.transcript       = out.get("text", "").strip()
        r.detected_language = out.get("language", lang)
        r.status           = "success"
    except Exception as e:
        r.status = "error"; r.error_message = str(e)
    return r


def run_faster_whisper(path: str, size: str, lang: str) -> EngineResult:
    r = EngineResult(engine="Faster-Whisper")
    if not FASTER_WHISPER_AVAILABLE:
        r.status = "error"; r.error_message = "faster-whisper not installed"; return r
    try:
        model = get_faster_whisper_model(size)
        kwargs = {"task": "transcribe"}
        if lang != "auto":
           kwargs["language"] = lang
        t0 = time.perf_counter()
        segs, info = model.transcribe(path, **kwargs)
        text = " ".join(s.text.strip() for s in segs)
        r.latency_sec       = round(time.perf_counter() - t0, 3)
        r.transcript        = text.strip()
        r.detected_language = getattr(info, "language", lang) or lang
        r.status            = "success"
    except Exception as e:
        r.status = "error"; r.error_message = str(e)
    return r


def run_sarvam(path: str, duration: float, api_key: str, lang: str) -> EngineResult:
    r = EngineResult(engine="Sarvam AI")
    if duration > SARVAM_MAX_SECS:
        r.status = "skipped"
        r.error_message = f"Audio {duration:.1f}s > {SARVAM_MAX_SECS:.0f}s Sarvam limit"
        return r
    if not REQUESTS_AVAILABLE:
        r.status = "error"; r.error_message = "requests not installed"; return r
    if not api_key:
        r.status = "skipped"; r.error_message = "No Sarvam API key"; return r
    try:
        sarvam_lang = SARVAM_LANG_MAP.get(lang, "unknown")
        t0 = time.perf_counter()
        with open(path, "rb") as f:
            resp = _requests.post(
                SARVAM_API_URL,
                headers={"api-subscription-key": api_key},
                files={"file": ("audio.wav", f, "audio/wav")},
                data={"model": "saarika:v2.5", "language_code": sarvam_lang},
                timeout=60,
            )
        r.latency_sec = round(time.perf_counter() - t0, 3)
        if resp.status_code == 200:
            payload = resp.json()
            r.transcript        = payload.get("transcript", "").strip()
            r.detected_language = payload.get("language_code", sarvam_lang)
            r.status            = "success"
        else:
            r.status = "error"; r.error_message = f"HTTP {resp.status_code}"
    except Exception as e:
        r.status = "error"; r.error_message = str(e)
    return r


def run_shrutam(path: str, api_key: str, lang: str) -> EngineResult:
    """Always safe-mode: skips rather than crashes on any failure."""
    r = EngineResult(engine="Shrutam")
    if not REQUESTS_AVAILABLE:
        r.status = "skipped"; r.error_message = "Safe mode: requests unavailable"; return r
    if not api_key:
        r.status = "skipped"; r.error_message = "Safe mode: no API key"; return r
    try:
        t0 = time.perf_counter()
        with open(path, "rb") as f:
            data = {} if lang == "auto" else {"language": lang}
            resp = _requests.post(
                SHRUTAM_API_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": ("audio.wav", f, "audio/wav")},
                data=data,
                timeout=60,
            )
        r.latency_sec = round(time.perf_counter() - t0, 3)
        if resp.status_code == 200:
            payload = resp.json()
            r.transcript        = payload.get("text", payload.get("transcript", "")).strip()
            r.detected_language = payload.get("language", lang)
            r.status            = "success"
        else:
            r.status = "skipped"; r.error_message = f"Safe mode: HTTP {resp.status_code}"
    except Exception as e:
        r.status = "skipped"; r.error_message = f"Safe mode: {e}"
    return r


def run_gemini(path: str, api_key: str, lang: str) -> EngineResult:
    """Always safe-mode: skips rather than crashes on any failure."""
    r = EngineResult(engine="Gemini")
    if not GEMINI_AVAILABLE:
        r.status = "error"; r.error_message = "google-generativeai not installed"; return r
    if not api_key:
        r.status = "skipped"; r.error_message = "Safe mode: no Gemini API key"; return r
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        lang_label = LANG_CODE_TO_LABEL.get(lang, "the spoken language")
        if lang == "auto":
            prompt = "Transcribe the audio verbatim. Output only the transcript text, nothing else."
        else:
            prompt = (f"Transcribe the audio verbatim in {lang_label}. "
                      "Output only the transcript text, nothing else.")
        t0 = time.perf_counter()
        uploaded = genai.upload_file(path=path, mime_type="audio/wav")
        resp = model.generate_content([prompt, uploaded])
        r.latency_sec       = round(time.perf_counter() - t0, 3)
        text                = (getattr(resp, "text", "") or "").strip()
        r.transcript        = text
        r.detected_language = lang
        r.status            = "success" if text else "error"
        if not text:
            r.error_message = "Gemini returned empty response"
    except Exception as e:
        r.status = "skipped"; r.error_message = f"Safe mode: {e}"
    return r


# ══════════════════════════════════════════════════════════════════════════════════════
# BENCHMARK ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════════════

def run_benchmark(path: str, duration: float, cfg: Dict[str, Any]) -> List[EngineResult]:
    """Run all enabled engines sequentially; compute RTF + metrics for each."""
    enabled = cfg["engines_enabled"]
    results: List[EngineResult] = []
    steps = [k for k, v in enabled.items() if v]
    total = max(len(steps), 1)
    bar = st.progress(0.0, text="Initialising benchmark…")
    done = 0

    def _tick(label: str):
        nonlocal done
        bar.progress(done / total, text=label)

    if enabled.get("Whisper"):
        _tick("⚙️ Running Whisper…")
        results.append(run_whisper(path, cfg["model_size"], cfg["lang"]))
        done += 1

    if enabled.get("Faster-Whisper"):
        _tick("⚙️ Running Faster-Whisper…")
        results.append(run_faster_whisper(path, cfg["model_size"], cfg["lang"]))
        done += 1

    if enabled.get("Sarvam AI"):
        _tick("☁️ Running Sarvam AI…")
        results.append(run_sarvam(path, duration, cfg["sarvam_key"], cfg["lang"]))
        done += 1

    if enabled.get("Shrutam"):
        _tick("☁️ Running Shrutam…")
        results.append(run_shrutam(path, cfg["shrutam_key"], cfg["lang"]))
        done += 1

    if enabled.get("Gemini"):
        _tick("🤖 Running Gemini…")
        results.append(run_gemini(path, cfg["gemini_key"], cfg["lang"]))
        done += 1

    bar.progress(1.0, text="✅ Benchmark complete!")
    time.sleep(0.4)
    bar.empty()

    # Post-process
    ref = cfg.get("reference_text", "")
    for r in results:
        if r.status == "success" and duration > 0:
            r.rtf = round(r.latency_sec / duration, 4)
        if r.status == "success":
            m = compute_metrics(ref, r.transcript)
            r.wer      = m["wer"]
            r.cer      = m["cer"]
            r.accuracy = m["accuracy"]

    return results


# ══════════════════════════════════════════════════════════════════════════════════════
# CHART HELPER
# ══════════════════════════════════════════════════════════════════════════════════════

def bar_chart(df: pd.DataFrame, col: str, title: str, y_label: str):
    if not PLOTLY_AVAILABLE:
        return None
    plot = df.dropna(subset=[col])
    if plot.empty:
        return None
    colors = [ENGINE_COLORS.get(e, "#6B7280") for e in plot["engine"]]
    fig = go.Figure(go.Bar(
        x=plot["engine"], y=plot[col],
        marker_color=colors,
        text=[f"{v:.3f}" for v in plot[col]],
        textposition="outside",
    ))
    fig.update_layout(
        title=title, yaxis_title=y_label, xaxis_title="Engine",
        template="plotly_white", height=340,
        margin=dict(t=55, b=30, l=30, r=10),
        showlegend=False,
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════════════

def page_setup():
    st.set_page_config(
        page_title="ASR Intelligence Lab Pro",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("""
    <style>
    /* ── Dark background ── */
    .stApp { background: #F8FAFC; color: #111827; }

    /* ── Gradient title ── */
    .lab-title {
        font-size: 2.6rem; font-weight: 900; line-height: 1.15;
        background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 50%, #10B981 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
    }
    .lab-subtitle { color: #6B7280; font-size: .95rem; margin-top: 0; margin-bottom: 16px; }

    /* ── Metric cards ── */
    .card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB; box-shadow: 0 2px 8px rgba(0,0,0,.6);
        padding: 18px 22px; box-shadow: 0 4px 15px rgba(0,0,0,.4);
    }
    .card-title { color: #6B7280; font-size: .8rem; text-transform: uppercase;
                  letter-spacing: .08em; font-weight: 600; margin-bottom: 4px; }
    .card-value { font-size: 1.5rem; font-weight: 800; color: #111827; }
    .card-sub   { font-size: .82rem; color: #9CA3AF; margin-top: 2px; }

    /* ── Engine pill badges ── */
    .pill {
        display: inline-block; border-radius: 999px; padding: 2px 12px;
        font-size: .75rem; font-weight: 700; margin-right: 6px;
    }

    /* ── Status ── */
    .ok  { color: #059669; font-weight: 700; }
    .sk  { color: #D97706; font-weight: 700; }
    .er  { color: #DC2626; font-weight: 700; }

    /* ── Live caption box ── */
    .caption-box {
        background: #F0F9FF; border: 1px solid #BAE6FD; border-radius: 12px;
        padding: 16px; min-height: 80px; font-size: 1.1rem;
        color: #0C4A6E; line-height: 1.6;
    }

    /* ── Availability strip ── */
    .avail-strip { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
    .avail-ok  { background:#ECFDF5; color:#065F46; border: 1px solid #A7F3D0; border-radius:8px;
                 padding:3px 10px; font-size:.75rem; font-weight:600; }
    .avail-err { background:#FEF2F2; color:#991B1B; border: 1px solid #FECACA; border-radius:8px;
                 padding:3px 10px; font-size:.75rem; font-weight:600; }

    /* ── Sidebar styling ── */
    section[data-testid="stSidebar"] { background: #FFFFFF !important; border-right: 1px solid #E5E7EB; }

    /* ── Buttons full width ── */
    .stButton > button { width: 100%; border-radius: 10px; font-weight: 700; }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════════════
# AVAILABILITY STRIP
# ══════════════════════════════════════════════════════════════════════════════════════

def render_availability():
    deps = [
        ("🎤 Mic Recorder", MIC_RECORDER_AVAILABLE),
        ("🧠 Whisper",      WHISPER_AVAILABLE),
        ("⚡ Faster-Whisper", FASTER_WHISPER_AVAILABLE),
        ("☁️ Sarvam AI",   REQUESTS_AVAILABLE),
        ("💗 Shrutam",     REQUESTS_AVAILABLE),
        ("🤖 Gemini",      GEMINI_AVAILABLE),
        ("📊 WER/CER",     JIWER_AVAILABLE),
        ("🔊 TTS",         GTTS_AVAILABLE),
        ("📒 Excel Log",   OPENPYXL_AVAILABLE),
    ]
    items = "".join(
        '<span class="{cls}">{icon} {name}</span>'.format(
            cls="avail-ok" if ok else "avail-err",
            icon="✅" if ok else "❌",
            name=n,
        )
        for n, ok in deps
    )
    st.markdown(f'<div class="avail-strip">{items}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════════════

def build_sidebar() -> Dict[str, Any]:
    st.sidebar.markdown("## ⚙️ Configuration")

    # Language
    st.sidebar.markdown("### 🌐 Language")
    lang_label = st.sidebar.selectbox(
        "Spoken language",
        [l for _, l in LANGUAGES],
        index=0,
        label_visibility="collapsed",
    )
    lang_code = next(c for c, l in LANGUAGES if l == lang_label)

    # Model
    st.sidebar.markdown("### 🧠 Local Model Size")
    model_size = st.sidebar.select_slider(
        "Whisper / Faster-Whisper",
        options=["tiny", "base", "small", "medium"],
        value="tiny",
        label_visibility="collapsed",
    )

    # API keys
    st.sidebar.markdown("### ☁️ API Keys")
    sarvam_key = st.sidebar.text_input("Sarvam AI Key", type="password",
                                        value=os.environ.get("SARVAM_API_KEY", ""))
    shrutam_key = st.sidebar.text_input("Shrutam Key", type="password",
                                         value=os.environ.get("SHRUTAM_API_KEY", ""),
                                         help="Safe mode if blank or unreachable")
    gemini_key = st.sidebar.text_input("Gemini API Key", type="password",
                                        value=os.environ.get("GEMINI_API_KEY",
                                               os.environ.get("GOOGLE_API_KEY", "")),
                                        help="Safe mode if blank or quota exceeded")

    # Engine toggles
    st.sidebar.markdown("### ✅ Engines")
    col_a, col_b = st.sidebar.columns(2)
    engines = {
        "Whisper":        col_a.checkbox("Whisper",         value=True),
        "Faster-Whisper": col_b.checkbox("Faster-Whisper",  value=True),
        "Sarvam AI":      col_a.checkbox("Sarvam AI",       value=True),
        "Shrutam":        col_b.checkbox("Shrutam",         value=True),
        "Gemini":         col_a.checkbox("Gemini",          value=True),
    }

    # Reference
    st.sidebar.markdown("### 📝 Reference Transcript")
    ref_text = st.sidebar.text_area("(optional — enables WER/CER)", height=90,
                                     label_visibility="collapsed")

    st.sidebar.markdown("---")
    st.sidebar.caption("ASR Intelligence Lab Pro · Built with Streamlit")

    return {
        "lang":            lang_code,
        "lang_label":      lang_label,
        "model_size":      model_size,
        "sarvam_key":      sarvam_key.strip(),
        "shrutam_key":     shrutam_key.strip(),
        "gemini_key":      gemini_key.strip(),
        "engines_enabled": engines,
        "reference_text":  ref_text.strip(),
    }


# ══════════════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE TRANSCRIPTION
# ══════════════════════════════════════════════════════════════════════════════════════

def tab_live(cfg: Dict[str, Any]):
    st.markdown("### 🎙️ Record & Transcribe")

    # Live mode toggle
    live_mode = st.toggle("⚡ Live Captions Mode", value=False,
                           help=f"Auto-clears after {LIVE_CAPTION_LIMIT} recordings")

    # Session state for live captions
    if "live_captions" not in st.session_state:
        st.session_state.live_captions = []
    if "recording_count" not in st.session_state:
        st.session_state.recording_count = 0

    # ── Mic recorder widget ──────────────────────────────────────────────────────────
    if not MIC_RECORDER_AVAILABLE:
        st.error("❌ `streamlit-mic-recorder` not installed.\n"
                 "Run: `pip install streamlit-mic-recorder`")
        return

    audio = mic_recorder(
        start_prompt="🎤  Start Recording",
        stop_prompt="⏹️  Stop Recording",
        format="wav",
        key="mic",
    )

    if not audio:
        st.info("👆 Press **Start Recording**, speak, then press **Stop Recording**.")
        if live_mode and st.session_state.live_captions:
            st.markdown("#### 📡 Live Captions")
            st.markdown(
                '<div class="caption-box">' +
                "<br>".join(f"[{i+1}] {t}" for i, t in enumerate(st.session_state.live_captions)) +
                '</div>', unsafe_allow_html=True)
        return

    # ── Audio received ───────────────────────────────────────────────────────────────
    audio_bytes: bytes = audio["bytes"]
    st.audio(audio_bytes, format="audio/wav")

    # Save to temp file
    path = bytes_to_wav_file(audio_bytes)
    duration = get_wav_duration(path)
    st.caption(f"Duration: **{duration:.2f}s**  |  Language hint: **{cfg['lang_label']}**")

    if duration < 0.3:
        st.warning("Recording too short. Please record at least 0.5 seconds.")
        return

    # ── Instant Whisper preview (live feel) ─────────────────────────────────────────
    if WHISPER_AVAILABLE and cfg["engines_enabled"].get("Whisper"):
        with st.spinner("⚡ Instant transcription…"):
            quick = run_whisper(path, cfg["model_size"], cfg["lang"])
        if quick.status == "success":
            st.success(f"**Whisper:** {quick.transcript}")
            if live_mode:
                st.session_state.recording_count += 1
                st.session_state.live_captions.append(quick.transcript)
                # Auto-clear after limit
                if st.session_state.recording_count >= LIVE_CAPTION_LIMIT:
                    st.session_state.live_captions = []
                    st.session_state.recording_count = 0
                    st.toast("🧹 Live captions cleared (limit reached)", icon="🔄")

    # ── Full benchmark ───────────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🚀 Run Full Benchmark (all engines)", use_container_width=True, type="primary"):
        with st.spinner("Running all ASR engines…"):
            results = run_benchmark(path, duration, cfg)

        st.session_state["last_results"] = results
        st.session_state["last_duration"] = duration
        st.session_state["last_path"]     = path

        # Excel log
        rows = []
        ts = dt.datetime.now().isoformat(timespec="seconds")
        for r in results:
            row = r.to_row()
            row.update({"timestamp": ts, "audio_duration_sec": round(duration, 3),
                        "language": cfg["lang_label"],
                        "reference_provided": bool(cfg["reference_text"])})
            rows.append(row)
        ok, info = log_to_excel(rows)
        if ok:
            st.toast(f"📒 Logged → {info}", icon="✅")
        else:
            st.warning(f"Excel log failed: {info}")

        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════════════
# TAB 2 — UPLOAD & BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════════════

def tab_upload(cfg: Dict[str, Any]):
    st.markdown("### 📁 Upload WAV File")
    uploaded = st.file_uploader("Choose a WAV or MP3 file", type=["wav", "mp3"], label_visibility="collapsed")

    if not uploaded:
        st.info("Upload a WAV or MP3 file to benchmark all engines.")
        return

    audio_bytes = uploaded.getbuffer()
    st.audio(uploaded, format="audio/wav")

    path = bytes_to_wav_file(bytes(audio_bytes), uploaded.name)
    duration = get_wav_duration(path)

    col1, col2 = st.columns(2)
    col1.metric("Duration", f"{duration:.2f}s")
    col2.metric("Language hint", cfg["lang_label"])

    if duration > SARVAM_MAX_SECS and cfg["engines_enabled"].get("Sarvam AI"):
        st.warning(f"⚠️ Audio {duration:.1f}s > {SARVAM_MAX_SECS:.0f}s — Sarvam AI will be skipped.")

    st.markdown("---")
    if st.button("🚀 Run Full Benchmark", use_container_width=True, type="primary"):
        with st.spinner("Running all ASR engines…"):
            results = run_benchmark(path, duration, cfg)

        st.session_state["last_results"] = results
        st.session_state["last_duration"] = duration
        st.session_state["last_path"]     = path

        rows = []
        ts = dt.datetime.now().isoformat(timespec="seconds")
        for r in results:
            row = r.to_row()
            row.update({"timestamp": ts, "audio_duration_sec": round(duration, 3),
                        "language": cfg["lang_label"],
                        "reference_provided": bool(cfg["reference_text"])})
            rows.append(row)
        ok, info = log_to_excel(rows)
        if ok:
            st.toast(f"📒 Logged → {info}", icon="✅")
        else:
            st.warning(f"Excel log failed: {info}")

        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════════════
# TAB 3 — RESULTS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════════════

def tab_results(cfg: Dict[str, Any]):
    results: Optional[List[EngineResult]] = st.session_state.get("last_results")
    duration = st.session_state.get("last_duration", 0.0)

    if not results:
        st.info("No benchmark results yet. Record or upload audio and run a benchmark first.")
        return

    df = pd.DataFrame([r.to_row() for r in results])
    successful = df[df["status"] == "success"].copy()

    # ── Summary cards ────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if not successful.empty and successful["accuracy"].notna().any():
            best = successful.loc[successful["accuracy"].idxmax()]
            st.markdown(
                f'<div class="card"><div class="card-title">🏆 Best Accuracy</div>'
                f'<div class="card-value">{best["engine"]}</div>'
                f'<div class="card-sub">Accuracy {best["accuracy"]:.2%}</div></div>',
                unsafe_allow_html=True)
        else:
            st.markdown('<div class="card"><div class="card-title">🏆 Best Accuracy</div>'
                        '<div class="card-sub">Add reference transcript</div></div>',
                        unsafe_allow_html=True)

    with c2:
        if not successful.empty:
            fast = successful.loc[successful["latency_sec"].idxmin()]
            st.markdown(
                f'<div class="card"><div class="card-title">⚡ Fastest Engine</div>'
                f'<div class="card-value">{fast["engine"]}</div>'
                f'<div class="card-sub">Latency {fast["latency_sec"]:.2f}s</div></div>',
                unsafe_allow_html=True)
        else:
            st.markdown('<div class="card"><div class="card-title">⚡ Fastest</div>'
                        '<div class="card-sub">No successful runs</div></div>',
                        unsafe_allow_html=True)

    with c3:
        n_ok  = (df["status"] == "success").sum()
        n_sk  = (df["status"] == "skipped").sum()
        n_er  = (df["status"] == "error").sum()
        st.markdown(
            f'<div class="card"><div class="card-title">📊 Run Summary</div>'
            f'<div class="card-value">{n_ok}/{len(df)}</div>'
            f'<div class="card-sub">✅ {n_ok} · ⏭️ {n_sk} · ❌ {n_er}</div></div>',
            unsafe_allow_html=True)

    with c4:
        st.markdown(
            f'<div class="card"><div class="card-title">⏱️ Audio Duration</div>'
            f'<div class="card-value">{duration:.2f}s</div>'
            f'<div class="card-sub">Language: {cfg["lang_label"]}</div></div>',
            unsafe_allow_html=True)

    st.markdown("")

    # ── Data table ───────────────────────────────────────────────────────────────────
    st.markdown("#### 📋 Detailed Results")
    display = df.copy()
    display["status"] = display["status"].map(
        {"success": "✅ Success", "skipped": "⏭️ Skipped", "error": "❌ Error"})
    st.dataframe(
        display[["engine", "status", "detected_language",
                 "latency_sec", "rtf", "wer", "cer", "accuracy", "error_message"]]
        .rename(columns={
            "engine": "Engine", "status": "Status",
            "detected_language": "Lang Detected",
            "latency_sec": "Latency(s)", "rtf": "RTF",
            "wer": "WER", "cer": "CER", "accuracy": "Accuracy",
            "error_message": "Notes"}),
        use_container_width=True, hide_index=True,
    )

    # ── Charts ───────────────────────────────────────────────────────────────────────
    if not PLOTLY_AVAILABLE:
        st.warning("Install plotly for charts: `pip install plotly`")
    elif successful.empty:
        st.info("No successful runs to chart.")
    else:
        st.markdown("#### 📉 Comparison Charts")
        cc1, cc2 = st.columns(2)
        with cc1:
            fig = bar_chart(successful, "latency_sec", "Latency by Engine", "Seconds")
            if fig: st.plotly_chart(fig, use_container_width=True)
        with cc2:
            fig = bar_chart(successful, "accuracy", "Accuracy by Engine", "Accuracy")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Add reference transcript for accuracy chart.")

        cc3, cc4 = st.columns(2)
        with cc3:
            fig = bar_chart(successful, "wer", "WER by Engine", "Word Error Rate")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Add reference transcript for WER chart.")
        with cc4:
            fig = bar_chart(successful, "cer", "CER by Engine", "Char Error Rate")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Add reference transcript for CER chart.")

    # ── Transcript comparison ─────────────────────────────────────────────────────────
    st.markdown("#### 📝 Transcripts")
    ok_results = [r for r in results if r.status == "success"]
    if not ok_results:
        st.warning("No successful transcripts to show.")
        return

    cols = st.columns(len(ok_results))
    for col, r in zip(cols, ok_results):
        with col:
            color = ENGINE_COLORS.get(r.engine, "#6B7280")
            st.markdown(
                f'<span class="pill" style="background:{color}22;color:{color};">'
                f'{r.engine}</span>', unsafe_allow_html=True)
            st.text_area("", r.transcript, height=150, key=f"tx_{r.engine}",
                         label_visibility="collapsed")
            st.download_button(
                "⬇️ Download",
                data=r.transcript,
                file_name=f"{r.engine.replace(' ','_').lower()}_transcript.txt",
                mime="text/plain",
                key=f"dl_{r.engine}",
                use_container_width=True,
            )

    # ── Download Excel log ────────────────────────────────────────────────────────────
    if os.path.exists(LOG_FILE):
        st.markdown("---")
        with open(LOG_FILE, "rb") as f:
            st.download_button(
                "📥 Download Full Excel Log",
                data=f.read(),
                file_name=LOG_FILE,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        st.caption(f"Log path: `{os.path.abspath(LOG_FILE)}`")


# ══════════════════════════════════════════════════════════════════════════════════════
# TAB 4 — TEXT-TO-SPEECH
# ══════════════════════════════════════════════════════════════════════════════════════

def tab_tts(cfg: Dict[str, Any]):
    st.markdown("### 🔊 Text-to-Speech")
    st.markdown("Convert any transcript (or custom text) into speech.")

    if not GTTS_AVAILABLE:
        st.error("❌ gTTS not installed. Run: `pip install gtts`")
        return

    # Pre-fill with last transcript if available
    last_results = st.session_state.get("last_results", [])
    best_text = ""
    if last_results:
        ok = [r for r in last_results if r.status == "success"]
        if ok:
            best_text = ok[0].transcript  # pick first successful

    tts_text = st.text_area(
        "Text to speak",
        value=best_text,
        height=160,
        placeholder="Type or paste text here…",
    )

    tts_lang = cfg["lang"] if cfg["lang"] != "auto" else "en"
    st.caption(f"TTS language: **{cfg['lang_label']}** (falls back to English for unsupported languages)")

    col1, col2 = st.columns(2)
    with col1:
        speed_slow = st.checkbox("🐢 Slow speed", value=False)
    with col2:
        st.markdown(f"Characters: **{len(tts_text)}**")

    if st.button("🎵 Generate Speech", type="primary", use_container_width=True,
                 disabled=not tts_text.strip()):
        with st.spinner("Generating speech…"):
            try:
                gtts_lang = tts_lang if len(tts_lang) == 2 else "en"
                tts = gTTS(text=tts_text, lang=gtts_lang, slow=speed_slow)
                buf = io.BytesIO()
                tts.write_to_fp(buf)
                buf.seek(0)
                audio_bytes = buf.read()
            except Exception as e:
                st.error(f"TTS failed: {e}")
                audio_bytes = None

        if audio_bytes:
            st.success("✅ Speech generated!")
            st.audio(audio_bytes, format="audio/mp3")
            st.download_button(
                "⬇️ Download MP3",
                data=audio_bytes,
                file_name="tts_output.mp3",
                mime="audio/mp3",
                use_container_width=True,
            )

    # ── Quick TTS from each engine transcript ────────────────────────────────────────
    if last_results:
        ok_results = [r for r in last_results if r.status == "success" and r.transcript]
        if ok_results:
            st.markdown("---")
            st.markdown("#### 🔁 Speak individual engine transcripts")
            for r in ok_results:
                with st.expander(f"🔊 {r.engine}"):
                    st.write(r.transcript)
                    if st.button(f"Generate speech for {r.engine}",
                                  key=f"tts_{r.engine}", use_container_width=True):
                        with st.spinner("Generating…"):
                            mp3 = make_tts_audio(r.transcript, tts_lang)
                        if mp3:
                            st.audio(mp3, format="audio/mp3")
                        else:
                            st.warning("TTS failed for this transcript.")


# ══════════════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════════════

def main():
    page_setup()

    # ── Header ───────────────────────────────────────────────────────────────────────
    st.markdown('<div class="lab-title">🎙️ ASR Intelligence Lab Pro</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="lab-subtitle">Benchmark Whisper · Faster-Whisper · Sarvam AI · '
        'Shrutam · Gemini across any language — with live captions & TTS</div>',
        unsafe_allow_html=True)

    render_availability()
    st.markdown("---")

    # ── Sidebar ───────────────────────────────────────────────────────────────────────
    cfg = build_sidebar()

    # ── Tabs ─────────────────────────────────────────────────────────────────────────
    t1, t2, t3, t4 = st.tabs([
        "🎤 Live Record",
        "📁 Upload WAV",
        "📊 Results Dashboard",
        "🔊 Text-to-Speech",
    ])

    with t1:
        tab_live(cfg)
    with t2:
        tab_upload(cfg)
    with t3:
        tab_results(cfg)
    with t4:
        tab_tts(cfg)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        st.error("An unexpected error occurred.")
        st.code(traceback.format_exc())
