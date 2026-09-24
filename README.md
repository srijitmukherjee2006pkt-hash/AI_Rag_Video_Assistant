# 🎬 AI Video Assistant

**Meeting & Video Intelligence — Transcribe · Summarise · Extract · Chat**

An end-to-end AI pipeline that takes any YouTube video or local audio/video file, transcribes it, generates a structured summary, extracts action items/decisions/open questions, and lets you have a conversational RAG chat with the content — all through a single Streamlit interface.

🔗 **Live Demo:** [srijit-ai-video-assistant.streamlit.app](https://srijit-ai-video-assistant.streamlit.app)

> ⚠️ **Note on the live demo:** YouTube blocks download requests from cloud/datacenter IPs (including Streamlit Cloud) as an anti-bot measure. This app routes requests through a proxy to work around that — if the demo ever times out or errors on a specific video, it's usually this upstream restriction, not the app logic. Running locally avoids this entirely.

---

## 📸 Screenshots

### Home & Input
![Home screen](assets/screenshot1.jpeg)

### Pipeline Progress & Extraction Output
![Pipeline status](assets/screenshot2.jpeg)

### Summary, Action Items, Key Decisions & Open Questions
![Results](assets/screenshot3.jpeg)

### Chat with your Meeting (RAG)
![RAG Chat](assets/screenshot4.jpeg)

---

## ✨ Features

- **Flexible input** — paste a YouTube URL or point to a local audio/video file
- **Multilingual transcription** — English audio via local **Whisper**, Hinglish audio via **Sarvam AI**'s speech-to-text-translate API
- **Auto-generated title** and **structured executive summary** (map-reduce style: chunked summarisation + synthesis)
- **Automated extraction** of:
  - ✅ Action items
  - 🎯 Key decisions
  - ❓ Open questions
- **Conversational RAG chat** — ask follow-up questions grounded strictly in the transcript, powered by a vector store + retriever
- **Live pipeline status** in the sidebar (Audio Processing → Transcription → Title → Summarisation → Extraction → RAG Engine)

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit (custom dark-themed CSS) |
| Orchestration | LangChain (`RunnablePassthrough`, `RunnableLambda`, prompt chains) |
| LLM | Google Gemini (via `langchain-google-genai`) |
| Transcription | OpenAI Whisper (local) + Sarvam AI STT-Translate API (Hinglish) |
| Vector Store | ChromaDB + Sentence Transformers |
| YouTube/Audio Handling | `yt-dlp`, `pydub`, `ffmpeg` |
| Environment | `python-dotenv` |

---

## 🏗️ Architecture

```
YouTube URL / Local File
        │
        ▼
  audio_processor.py  → download, normalize, chunk audio
        │
        ▼
  transcriber.py       → Whisper (English) / Sarvam (Hinglish)
        │
        ▼
  summarize.py          → title generation + map-reduce summary
        │
        ▼
  extractor.py            → action items, key decisions, open questions
        │
        ▼
  vector_store.py + rag_engine.py → chunk, embed, index → RAG chat
```

---

## 🚀 Local Setup

### Prerequisites
- Python 3.10+
- `ffmpeg` installed and on PATH

### Installation

```bash
git clone https://github.com/srijitmukherjee2006pkt-hash/AI_Rag_Video_Assistant.git
cd AI_Rag_Video_Assistant
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
WHISPER_MODEL=small
SARVAM_API_KEY=your_sarvam_api_key
SARVAM_STT_MODEL=saaras:v2.5

# Optional: only needed if downloading from a cloud/datacenter host
YTDLP_PROXY=http://username:password@proxy-host:port
```

### Run

```bash
streamlit run app.py
```

---

## ⚠️ Known Limitation: YouTube Downloads on Cloud Hosts

YouTube actively blocks download requests originating from datacenter IP ranges (used by most cloud platforms, including Streamlit Cloud), as a bot-prevention measure. This can surface as an `HTTP 403 Forbidden` error even with valid cookies.

**Workaround used in this project:** routing `yt-dlp` traffic through a proxy (`YTDLP_PROXY` env var) so requests originate from a non-datacenter IP. Free proxy tiers work for light/testing use; for production-grade reliability, a paid residential/rotating proxy is recommended.

This limitation does **not** affect the "local file upload" path, which works identically everywhere.

---

## 📄 License

This project is open for educational and portfolio use. Feel free to fork and adapt.

---

## 🙋 Author

**Srijit Mukherjee**
Pre-Final-Year B.Tech IT student, A.K. Choudhury School of IT, Calcutta University