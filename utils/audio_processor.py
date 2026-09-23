import os
import re
from urllib.parse import parse_qs, urlparse
from pydub import AudioSegment
import yt_dlp

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def sanitize_youtube_url(url: str) -> str:
    """Extract only the single video URL and strip playlist/mix parameters."""
    parsed = urlparse(url)
    if "youtube.com" in parsed.netloc:
        query_params = parse_qs(parsed.query)
        video_id = query_params.get("v")
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id[0]}"
    elif "youtu.be" in parsed.netloc:
        video_id = parsed.path.strip("/")
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
    return url


def download_youtube_audio(url: str) -> str:
    """Download audio from YouTube and ensure a valid 16kHz mono WAV file is returned."""
    clean_url = sanitize_youtube_url(url)
    output_template = os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "ios", "mweb"]
            }
        },
        "quiet": False,
        "no_warnings": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(clean_url, download=True)
        if "entries" in info and info["entries"]:
            info = info["entries"][0]
        video_id = info.get("id")

    wav_path = os.path.join(DOWNLOAD_DIR, f"{video_id}.wav")

    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Expected converted audio at {wav_path}, but file was not found.")

    return normalize_audio(wav_path)


def normalize_audio(input_path: str) -> str:
    """Convert audio to 16kHz mono WAV suitable for Whisper transcription."""
    output_path = os.path.splitext(input_path)[0] + "_norm.wav"
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    """Split audio into manageable chunks for speech-to-text models."""
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000

    # Return single file if under the chunk threshold
    if len(audio) <= chunk_ms:
        return [wav_path]

    chunks = []
    base_name = os.path.splitext(wav_path)[0]
    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start: start + chunk_ms]
        chunk_path = f"{base_name}_part_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks


def process_input(source: str) -> list:
    """Unified entry point for local audio/video files and YouTube URLs."""
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading and processing audio...")
        wav_path = download_youtube_audio(source)
    else:
        print("Detected local file. Converting to 16kHz mono WAV...")
        wav_path = normalize_audio(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready - {len(chunks)} chunk(s) created.")
    return chunks