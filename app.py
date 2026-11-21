import html
import re
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

# ─── Page config (sabse pehla Streamlit command) ───────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Light CSS (dark aur light dono theme mein chalta hai) ─────────────────────
st.markdown(
    """
<style>
.hero-title {
    font-size: clamp(1.8rem, 4vw, 2.9rem);
    font-weight: 800;
    line-height: 1.15;
    margin: 0 0 .2rem 0;
    background: linear-gradient(135deg, #7c3aed 0%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    opacity: .65;
    font-size: .75rem;
    letter-spacing: .18em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.brand {
    font-size: 1.55rem;
    font-weight: 800;
    line-height: 1.1;
    background: linear-gradient(135deg, #7c3aed 0%, #06b6d4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.session-title {
    font-size: 1.5rem;
    font-weight: 700;
    padding: .9rem 1.1rem;
    margin: .3rem 0 1rem 0;
    border-radius: 12px;
    border: 1px solid rgba(124, 58, 237, .35);
    border-left: 5px solid #7c3aed;
    background: rgba(124, 58, 237, .07);
}
[data-testid="stMetric"] {
    border: 1px solid rgba(128, 128, 128, .25);
    border-radius: 12px;
    padding: .7rem 1rem;
    background: rgba(124, 58, 237, .05);
}
.stButton > button, .stDownloadButton > button {
    border-radius: 10px;
    font-weight: 600;
}
.feature-icon { font-size: 2rem; margin-bottom: .3rem; }
.feature-title { font-weight: 700; margin-bottom: .2rem; }
.feature-text { opacity: .7; font-size: .85rem; line-height: 1.5; }
</style>
""",
    unsafe_allow_html=True,
)

# ─── Constants / session state ──────────────────────────────────────────────────
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

SUGGESTED_QUESTIONS = [
    "What is the main message of this video?",
    "Summarize the key points in 3 bullets.",
    "What advice does the speaker give?",
    "Who is speaking and what is the topic?",
]

for key, default in {
    "result": None,
    "chat_history": [],
    "pending_q": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─── Helpers ────────────────────────────────────────────────────────────────────
def wide_button(label: str, **kwargs):
    """Full-width button jo naye aur purane Streamlit dono par chale."""
    try:
        return st.button(label, width="stretch", **kwargs)
    except TypeError:
        return st.button(label, use_container_width=True, **kwargs)


def set_pending(question: str):
    st.session_state.pending_q = question


def reset_all():
    st.session_state.result = None
    st.session_state.chat_history = []
    st.session_state.pending_q = None


def safe_filename(text: str) -> str:
    cleaned = re.sub(r"[^\w\- ]", "", text).strip().replace(" ", "_")
    return cleaned[:50] or "video_report"


def build_report(r: dict) -> str:
    return (
        f"# {r['title']}\n\n"
        f"## Summary\n{r['summary']}\n\n"
        f"## Action Items\n{r['action_items']}\n\n"
        f"## Key Decisions\n{r['key_decisions']}\n\n"
        f"## Open Questions\n{r['open_questions']}\n\n"
        f"## Transcript\n{r['transcript']}\n"
    )


def run_analysis(source: str, language: str):
    """Pura pipeline chalao aur live progress dikhao."""
    reset_all()
    timings = {}
    started = time.time()

    with st.status("Analysing your video…", expanded=True) as status:
        try:
            def step(label, fn, *args):
                st.write(label)
                t0 = time.time()
                out = fn(*args)
                timings[label] = time.time() - t0
                return out

            chunks = step("🔊 Downloading and preparing audio", process_input, source)
            transcript = step("📝 Transcribing speech", transcribe_all, chunks, language)

            if not transcript or not transcript.strip():
                raise ValueError(
                    "No speech was detected in this video. "
                    "Try another video, or switch the language setting."
                )

            title = step("🏷️ Generating title", generate_title, transcript)
            summary = step("📋 Writing summary", summarize, transcript)
            actions = step("✅ Extracting action items", extract_action_items, transcript)
            decisions = step("🔑 Extracting key decisions", extract_key_decisions, transcript)
            questions = step("❓ Finding open questions", extract_questions, transcript)
            rag_chain = step("🧠 Building chat index", build_rag_chain, transcript)

        except Exception as e:
            status.update(label="Something went wrong", state="error", expanded=True)
            st.error(f"❌ {type(e).__name__}: {e}")
            return

        total = time.time() - started
        status.update(label=f"Done in {total:.0f}s", state="complete", expanded=False)

    st.session_state.result = {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": actions,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
        "n_chunks": len(chunks),
        "elapsed": total,
        "language": language,
    }
    st.rerun()


# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="brand">🎬 AI Video<br>Assistant</div>', unsafe_allow_html=True)
    st.caption("Transcribe · Summarise · Chat")
    st.divider()

    mode = st.radio("Input type", ["YouTube URL", "Upload file"], horizontal=True)

    source_url = ""
    uploaded = None
    if mode == "YouTube URL":
        source_url = st.text_input(
            "YouTube URL", placeholder="https://www.youtube.com/shorts/..."
        )
    else:
        uploaded = st.file_uploader(
            "Audio or video file",
            type=["mp3", "wav", "m4a", "aac", "ogg", "flac", "mp4", "mkv", "webm", "mov"],
        )

    language = st.selectbox(
        "Audio language",
        ["english", "hinglish"],
        index=0,
        help="english: Whisper (local). hinglish: Sarvam AI, use it for Hindi or Hindi+English audio.",
    )
    st.caption("🇬🇧 english → Whisper   |   🇮🇳 hinglish → Sarvam")

    run_btn = wide_button("⚡ Analyse video", type="primary")

    if st.session_state.result:
        st.divider()
        wide_button("🧹 New analysis", on_click=reset_all, key="reset_btn")

    st.divider()
    st.caption("Whisper · Sarvam · Groq · ChromaDB")

# ─── Run pipeline ───────────────────────────────────────────────────────────────
if run_btn:
    path = None
    if mode == "YouTube URL":
        if not source_url.strip():
            st.error("Please paste a YouTube URL first.")
        else:
            path = source_url.strip()
    else:
        if uploaded is None:
            st.error("Please upload a file first.")
        else:
            dest = UPLOAD_DIR / re.sub(r"[^\w.\-]", "_", uploaded.name)
            dest.write_bytes(uploaded.getbuffer())
            path = str(dest)

    if path:
        run_analysis(path, language)

# ─── Main area ──────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">AI Video Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Transcribe · Summarise · Chat with your videos</div>',
    unsafe_allow_html=True,
)

r = st.session_state.result

if r is None:
    # ── Empty state ──
    c1, c2, c3 = st.columns(3, gap="medium")
    features = [
        ("📝", "Transcribe", "Speech to text with Whisper, or Sarvam AI for Hindi and Hinglish."),
        ("📋", "Summarise", "Get a title, summary, action items, decisions and open questions."),
        ("💬", "Chat", "Ask questions. Answers come only from the video transcript."),
    ]
    for col, (icon, title, text) in zip((c1, c2, c3), features):
        with col:
            with st.container(border=True):
                st.markdown(
                    f'<div class="feature-icon">{icon}</div>'
                    f'<div class="feature-title">{title}</div>'
                    f'<div class="feature-text">{text}</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("#### How to use")
    st.markdown(
        "1. Paste a YouTube URL (or upload a file) in the sidebar.\n"
        "2. Pick the language: **english** or **hinglish**.\n"
        "3. Click **Analyse video** and watch the live progress.\n"
        "4. Read the results, then ask questions in the chat at the bottom."
    )

else:
    # ── Title ──
    st.markdown(
        f'<div class="session-title">📌 {html.escape(str(r["title"]))}</div>',
        unsafe_allow_html=True,
    )

    # ── Metrics ──
    words = len(r["transcript"].split())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Words", f"{words:,}")
    m2.metric("Reading time", f"{max(1, round(words / 200))} min")
    m3.metric("Audio chunks", r["n_chunks"])
    m4.metric("Processed in", f"{r['elapsed']:.0f}s")

    # ── Tabs ──
    tab_summary, tab_insights, tab_transcript = st.tabs(
        ["📋 Summary", "🔍 Insights", "📝 Transcript"]
    )

    with tab_summary:
        with st.container(border=True):
            st.markdown(r["summary"])

    with tab_insights:
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            with st.container(border=True):
                st.markdown("##### ✅ Action Items")
                st.markdown(r["action_items"])
        with c2:
            with st.container(border=True):
                st.markdown("##### 🔑 Key Decisions")
                st.markdown(r["key_decisions"])
        with c3:
            with st.container(border=True):
                st.markdown("##### ❓ Open Questions")
                st.markdown(r["open_questions"])

    with tab_transcript:
        st.text_area(
            "Full transcript",
            r["transcript"],
            height=320,
            disabled=True,
            key="transcript_view",
            label_visibility="collapsed",
        )

    # ── Downloads ──
    base = safe_filename(str(r["title"]))
    d1, d2, _ = st.columns([1, 1, 3])
    with d1:
        st.download_button(
            "⬇️ Download report (.md)",
            data=build_report(r),
            file_name=f"{base}.md",
            mime="text/markdown",
        )
    with d2:
        st.download_button(
            "⬇️ Download transcript (.txt)",
            data=r["transcript"],
            file_name=f"{base}_transcript.txt",
            mime="text/plain",
        )

    st.divider()

    # ── Chat ──
    st.markdown("### 💬 Chat with this video")
    history = st.session_state.chat_history

    if not history:
        st.caption("Try one of these, or type your own question in the box at the bottom:")
        q1, q2 = st.columns(2)
        for i, q in enumerate(SUGGESTED_QUESTIONS):
            with (q1 if i % 2 == 0 else q2):
                st.button(q, key=f"sugg_{i}", on_click=set_pending, args=(q,))

    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    typed = st.chat_input("Ask anything about this video…")
    question = typed or st.session_state.pending_q
    st.session_state.pending_q = None

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer = ask_question(r["rag_chain"], question)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
        st.rerun()

    if history:
        if st.button("🗑️ Clear chat", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()