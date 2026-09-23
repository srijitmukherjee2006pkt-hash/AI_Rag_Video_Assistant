import streamlit as st
import time
from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500;600&family=Inter:wght@400;500;600&display=swap');

/* ── Root Variables ── */
:root {
    --bg: #07070c;
    --bg-radial-1: rgba(124, 58, 237, 0.10);
    --bg-radial-2: rgba(6, 182, 212, 0.07);
    --surface: #0f0f17;
    --surface-2: #171722;
    --surface-3: #1e1e2c;
    --border: #26263a;
    --border-soft: #1c1c2a;
    --accent: #7c3aed;
    --accent-glow: #a970ff;
    --accent-2: #06b6d4;
    --text: #edEDF5;
    --text-muted: #8a8ab0;
    --text-faint: #55557a;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #f87171;
    --radius: 16px;
    --radius-sm: 10px;
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', 'JetBrains Mono', monospace;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp {
    background:
        radial-gradient(circle at 15% 0%, var(--bg-radial-1) 0%, transparent 45%),
        radial-gradient(circle at 85% 15%, var(--bg-radial-2) 0%, transparent 45%),
        var(--bg) !important;
}

/* Animated grid background */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(124, 58, 237, 0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(124, 58, 237, 0.035) 1px, transparent 1px);
    background-size: 44px 44px;
    mask-image: radial-gradient(circle at 50% 0%, black 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

.block-container {
    padding-top: 2.5rem !important;
    max-width: 1200px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--surface) 0%, #0b0b12 100%) !important;
    border-right: 1px solid var(--border-soft) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 2rem !important;
}

/* ── Headings ── */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}

/* ── Hero Title ── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.2rem, 5vw, 3.8rem);
    font-weight: 800;
    line-height: 1.08;
    margin: 0;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent-glow) 55%, var(--accent-2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: var(--text-muted);
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.hero-sub::before {
    content: '';
    width: 22px;
    height: 1px;
    background: linear-gradient(90deg, var(--accent-glow), transparent);
    display: inline-block;
}

.eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--accent-glow);
    background: rgba(124,58,237,0.10);
    border: 1px solid rgba(124,58,237,0.28);
    padding: 0.3rem 0.75rem;
    border-radius: 100px;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    margin-bottom: 1rem;
}

.eyebrow .pulse-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent-glow);
    box-shadow: 0 0 8px var(--accent-glow);
    animation: pulse 1.6s infinite;
}

/* ── Cards ── */
.card {
    background: linear-gradient(180deg, var(--surface-2) 0%, var(--surface) 100%);
    border: 1px solid var(--border-soft);
    border-radius: var(--radius);
    padding: 1.6rem 1.7rem;
    margin-bottom: 1.1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.25s ease, transform 0.25s ease, box-shadow 0.25s ease;
    box-shadow: 0 1px 0 rgba(255,255,255,0.02) inset, 0 12px 30px -18px rgba(0,0,0,0.6);
}

.card:hover {
    border-color: rgba(124,58,237,0.45);
    transform: translateY(-2px);
    box-shadow: 0 1px 0 rgba(255,255,255,0.03) inset, 0 20px 40px -18px rgba(124,58,237,0.25);
}

.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--accent), var(--accent-2));
    opacity: 0.85;
}

.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.9rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.card-content {
    font-size: 0.9rem;
    line-height: 1.75;
    color: var(--text);
}

/* ── Accent Badge ── */
.badge {
    display: inline-block;
    padding: 0.25rem 0.7rem;
    border-radius: 100px;
    font-size: 0.66rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}

.badge-purple { background: rgba(124,58,237,0.16); color: var(--accent-glow); border: 1px solid rgba(124,58,237,0.32); }
.badge-cyan   { background: rgba(6,182,212,0.13);  color: var(--accent-2);    border: 1px solid rgba(6,182,212,0.30); }
.badge-green  { background: rgba(16,185,129,0.13); color: var(--success);     border: 1px solid rgba(16,185,129,0.30); }

/* ── Input & Buttons ── */
.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: var(--surface-3) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    padding: 0.65rem 0.9rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.18) !important;
}

.stTextInput > div > div > input::placeholder {
    color: var(--text-faint) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, #5b21b6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.06em !important;
    padding: 0.7rem 1.5rem !important;
    transition: all 0.25s cubic-bezier(.2,.8,.2,1) !important;
    text-transform: uppercase !important;
    box-shadow: 0 8px 20px -8px rgba(124,58,237,0.55) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 14px 30px -8px rgba(124,58,237,0.65) !important;
    filter: brightness(1.08) !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
}

/* Secondary button */
.stButton > button[kind="secondary"],
.stButton > button[data-testid="baseButton-secondary"] {
    background: var(--surface-3) !important;
    border: 1px solid var(--border) !important;
    box-shadow: none !important;
}

.stButton > button[kind="secondary"]:hover,
.stButton > button[data-testid="baseButton-secondary"]:hover {
    border-color: var(--danger) !important;
    color: var(--danger) !important;
}

/* ── Progress / Status ── */
.status-bar {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.7rem 1rem;
    background: var(--surface-2);
    border-radius: var(--radius-sm);
    margin: 0.35rem 0;
    border: 1px solid var(--border-soft);
    font-size: 0.78rem;
    transition: border-color 0.3s, background 0.3s;
}

.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

.dot-active   { background: var(--accent-glow); box-shadow: 0 0 10px var(--accent-glow); animation: pulse 1.4s infinite; }
.dot-done     { background: var(--success); box-shadow: 0 0 8px rgba(16,185,129,0.6); }
.dot-pending  { background: var(--border); }

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.4; transform: scale(0.85); }
}

/* ── Chat ── */
.chat-container {
    background: linear-gradient(180deg, var(--surface-2), var(--surface));
    border: 1px solid var(--border-soft);
    border-radius: var(--radius);
    padding: 1.4rem;
    max-height: 440px;
    overflow-y: auto;
    margin-bottom: 1rem;
}

.chat-msg {
    margin-bottom: 1.1rem;
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
}

.chat-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
}

.chat-bubble {
    display: inline-block;
    padding: 0.7rem 1.05rem;
    border-radius: 12px;
    font-size: 0.87rem;
    line-height: 1.65;
    max-width: 92%;
    color: #ffffff !important;
}

.user-label  { color: var(--accent-glow); }
.bot-label   { color: var(--accent-2); }

.user-bubble {
    background: rgba(124,58,237,0.14);
    border: 1px solid rgba(124,58,237,0.28);
    align-self: flex-end;
    border-bottom-right-radius: 3px;
    color: #ffffff !important;
}

.bot-bubble {
    background: rgba(6,182,212,0.09);
    border: 1px solid rgba(6,182,212,0.22);
    align-self: flex-start;
    border-bottom-left-radius: 3px;
    color: #ffffff !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border-soft) !important;
    margin: 1.8rem 0 !important;
}

/* ── Transcript box ── */
.transcript-box {
    background: var(--surface-3);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 1.3rem;
    font-size: 0.83rem;
    line-height: 1.85;
    max-height: 320px;
    overflow-y: auto;
    color: var(--text-muted);
    white-space: pre-wrap;
    word-break: break-word;
    font-family: 'JetBrains Mono', monospace;
}

/* ── Section label ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    margin-bottom: 1.1rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    color: var(--text);
}

/* ── Empty state ── */
.empty-wrap {
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    padding: 5.5rem 2rem; text-align:center;
    border: 1px dashed var(--border);
    border-radius: var(--radius);
    background: radial-gradient(circle at 50% 0%, rgba(124,58,237,0.06), transparent 60%);
}

.empty-icon-wrap {
    width: 84px; height: 84px;
    border-radius: 22px;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, rgba(124,58,237,0.18), rgba(6,182,212,0.10));
    border: 1px solid rgba(124,58,237,0.3);
    font-size: 2.4rem;
    margin-bottom: 1.4rem;
    box-shadow: 0 0 40px rgba(124,58,237,0.18);
}

/* ── Streamlit elements ── */
.stProgress > div > div > div { background: linear-gradient(90deg, var(--accent), var(--accent-2)) !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text) !important; }
label { color: var(--text-muted) !important; font-size: 0.78rem !important; letter-spacing: 0.03em; }

[data-testid="stExpander"] {
    background: var(--surface-2) !important;
    border: 1px solid var(--border-soft) !important;
    border-radius: var(--radius-sm) !important;
    overflow: hidden;
}

[data-testid="stAlert"] {
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
}

/* scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* fade-in for content */
.fade-in { animation: fadeIn 0.5s ease both; }
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ──────────────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "pipeline_done": False,
    "pipeline_steps": {},
    "chat_input_val": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Helpers ────────────────────────────────────────────────────────────────────
PIPELINE_STAGES = [
    ("audio",      "🔊", "Audio Processing"),
    ("transcript", "📝", "Transcription"),
    ("title",      "🏷️", "Title Generation"),
    ("summary",    "📋", "Summarisation"),
    ("extract",    "🔍", "Extraction"),
    ("rag",        "🧠", "RAG Engine"),
]

def step_status(steps: dict, key: str) -> str:
    s = steps.get(key, "pending")
    if s == "active":  return "dot-active"
    if s == "done":    return "dot-done"
    return "dot-pending"

def render_step_bar(container, label: str, key: str, icon: str):
    css = step_status(st.session_state.pipeline_steps, key)
    container.markdown(f"""
    <div class="status-bar">
        <div class="status-dot {css}"></div>
        <span>{icon} {label}</span>
    </div>""", unsafe_allow_html=True)

def send_chat_message():
    user_text = st.session_state.get("chat_input_val", "").strip()
    if user_text and st.session_state.result:
        st.session_state.chat_history.append({"role": "user", "content": user_text})
        answer = ask_question(st.session_state.result["rag_chain"], user_text)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.session_state.chat_input_val = ""

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:0.7rem;margin-bottom:0.3rem">
        <div style="width:38px;height:38px;border-radius:11px;background:linear-gradient(135deg,#7c3aed,#06b6d4);
                    display:flex;align-items:center;justify-content:center;font-size:1.2rem;
                    box-shadow:0 6px 18px -6px rgba(124,58,237,0.6)">🎬</div>
        <div>
            <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.05rem;line-height:1.1;color:var(--text)">AI Video<br>Assistant</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="hero-sub" style="margin-top:0.6rem">Meeting Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<span class="badge badge-purple">📥 Input Source</span>', unsafe_allow_html=True)
    st.write("")
    source = st.text_input("YouTube URL or File Path", placeholder="https://youtube.com/watch?v=... or /path/to/file.mp4")

    language = st.selectbox("Language", ["english", "hinglish"], index=0)

    st.write("")
    run_btn = st.button("⚡  Analyse", use_container_width=True)

    sidebar_status_placeholder = st.empty()

    def refresh_sidebar_status():
        with sidebar_status_placeholder.container():
            badge_type = "badge-green" if st.session_state.pipeline_done else "badge-purple"
            badge_text = "✓ Pipeline Complete" if st.session_state.pipeline_done else "⚙️ Pipeline Status"
            st.markdown("---")
            st.markdown(f'<span class="badge {badge_type}">{badge_text}</span>', unsafe_allow_html=True)
            st.write("")
            for step, icon, label in PIPELINE_STAGES:
                render_step_bar(st, label, step, icon)

    if st.session_state.pipeline_steps:
        refresh_sidebar_status()

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.68rem;color:var(--text-faint);line-height:1.6;font-family:'JetBrains Mono',monospace">
        NEURAL TRANSCRIPTION<br>AUTOMATED EXTRACTION<br>CONVERSATIONAL RAG
    </div>
    """, unsafe_allow_html=True)

# ─── Main Area ──────────────────────────────────────────────────────────────────
st.markdown('<div class="eyebrow"><span class="pulse-dot"></span>AUDIO &amp; VIDEO INTELLIGENCE</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Video &amp; Audio<br>Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Transcribe · Summarise · Chat with your meetings</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Run Pipeline ────────────────────────────────────────────────────────────────
if run_btn:
    if not source.strip():
        st.error("Please enter a YouTube URL or file path.")
    else:
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {k: "pending" for k, _, _ in PIPELINE_STAGES}

        def update_step(key, state):
            st.session_state.pipeline_steps[key] = state
            refresh_sidebar_status()

        try:
            with st.status("🚀 Processing media pipeline...", expanded=True) as status_box:
                update_step("audio", "active")
                st.write("🔊 Extracting and segmenting audio...")
                chunks = process_input(source)
                update_step("audio", "done")

                update_step("transcript", "active")
                st.write("📝 Running neural transcription...")
                transcript = transcribe_all(chunks, language)
                update_step("transcript", "done")

                update_step("title", "active")
                st.write("🏷️ Generating title...")
                title = generate_title(transcript)
                update_step("title", "done")

                update_step("summary", "active")
                st.write("📋 Generating executive summary...")
                summary = summarize(transcript)
                update_step("summary", "done")

                update_step("extract", "active")
                st.write("🔍 Extracting action items and key decisions...")
                action_items = extract_action_items(transcript)
                decisions = extract_key_decisions(transcript)
                questions = extract_questions(transcript)
                update_step("extract", "done")

                update_step("rag", "active")
                st.write("🧠 Indexing transcript for RAG chat...")
                rag_chain = build_rag_chain(transcript)
                update_step("rag", "done")

                status_box.update(label="✅ Analysis complete!", state="complete", expanded=False)

            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
            }
            st.session_state.pipeline_done = True
            time.sleep(0.4)
            st.rerun()

        except Exception as e:
            for k, _, _ in PIPELINE_STAGES:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "pending"
            refresh_sidebar_status()
            st.error(f"❌ Error: {e}")

# ── Results ──────────────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result

    # Title banner
    st.markdown(f"""
    <div class="card fade-in">
        <div class="card-title">📌 Session Title</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.55rem;font-weight:700;color:var(--text);letter-spacing:-0.01em">
            {r['title']}
        </div>
    </div>""", unsafe_allow_html=True)

    # Top row: summary + transcript
    col1, col2 = st.columns([3, 2], gap="medium")

    with col1:
        st.markdown(f"""
        <div class="card fade-in">
            <div class="card-title">📋 Executive Summary</div>
            <div class="card-content">{r['summary']}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-label">📝 Transcript</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="transcript-box">{r["transcript"]}</div>', unsafe_allow_html=True)

    # Key extractions row
    col_a, col_b, col_c = st.columns(3, gap="medium")
    with col_a:
        st.markdown(f"""
        <div class="card fade-in">
            <div class="card-title">✅ Action Items</div>
            <div class="card-content">{r['action_items']}</div>
        </div>""", unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div class="card fade-in">
            <div class="card-title">🎯 Key Decisions</div>
            <div class="card-content">{r['key_decisions']}</div>
        </div>""", unsafe_allow_html=True)

    with col_c:
        st.markdown(f"""
        <div class="card fade-in">
            <div class="card-title">❓ Open Questions</div>
            <div class="card-content">{r['open_questions']}</div>
        </div>""", unsafe_allow_html=True)

    # ── Chat Section ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-label">💬 Chat with your Meeting</div>', unsafe_allow_html=True)

    # Render history
    if st.session_state.chat_history:
        chat_html = '<div class="chat-container">'
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                chat_html += f"""
                <div class="chat-msg" style="align-items: flex-end;">
                    <span class="chat-label user-label">You</span>
                    <div class="chat-bubble user-bubble">{msg['content']}</div>
                </div>"""
            else:
                chat_html += f"""
                <div class="chat-msg" style="align-items: flex-start;">
                    <span class="chat-label bot-label">🤖 Assistant</span>
                    <div class="chat-bubble bot-bubble">{msg['content']}</div>
                </div>"""
        chat_html += "</div>"
        st.markdown(chat_html, unsafe_allow_html=True)

    # Chat input & buttons
    c_in, c_send = st.columns([5, 1])
    with c_in:
        st.text_input(
            "Ask something about the meeting",
            placeholder="Type your question here and hit Enter...",
            label_visibility="collapsed",
            key="chat_input_val",
            on_change=send_chat_message
        )
    with c_send:
        st.button("Send ➔", on_click=send_chat_message, use_container_width=True)

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

else:
    st.markdown("""
    <div class="empty-wrap">
        <div class="empty-icon-wrap">🎙️</div>
        <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1.3rem;margin-bottom:0.5rem;color:var(--text)">
            No Meeting Loaded
        </div>
        <div style="color:var(--text-muted);font-size:0.88rem;max-width:380px;line-height:1.6">
            Enter a YouTube link or audio/video file path in the sidebar and click <b>Analyse</b> to begin.
        </div>
    </div>
    """, unsafe_allow_html=True)