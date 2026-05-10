# api_server_xtts_fixed.py
# FastAPI TTS API with XTTS-v2 (zero-shot), proper WAV bytes handling, simple cache, and home page.
# Endpoints:
#   POST /synthesize {text, speaker, language} -> WAV bytes
#   GET  /stream?q=...&speaker=...&language=tr -> WAV bytes (single chunk)
#   GET  /health
#   GET  /           -> serves a minimal HTML UI if web_demo.html exists
#
# Requirements:
#   pip install fastapi uvicorn TTS numpy

import io
import os
from typing import Dict, Tuple

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from TTS.api import TTS

app = FastAPI(title="XTTS API (Fixed)", version="0.3.0")

# Optional: serve local files (so / loads web_demo.html if present)
if os.path.exists("web_demo.html"):
    app.mount("/static", StaticFiles(directory=".", html=True), name="static")

    @app.get("/")
    def home():
        return FileResponse("web_demo.html")

# Lazy model load
_tts = None
def get_tts():
    global _tts
    if _tts is None:
        # GPU True -> RTX 4060 kullanır; isterseniz gpu=False yapabilirsiniz.
        _tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True, progress_bar=False)
    return _tts

# Simple memory cache: key=(text, speaker_abs, lang) -> wav_bytes
CACHE: Dict[Tuple[str,str,str], bytes] = {}

def _to_abs(path: str) -> str:
    # Resolve to absolute path relative to current working dir
    return os.path.abspath(path)

def wav_bytes_from_np(y: np.ndarray, sr: int) -> bytes:
    """Convert float32 [-1,1] numpy array to 16-bit PCM WAV bytes."""
    y = np.asarray(y, dtype=np.float32)
    y_clipped = np.clip(y, -1.0, 1.0)
    y_int16 = (y_clipped * 32767.0).astype(np.int16)

    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1 if y_int16.ndim == 1 else y_int16.shape[0])
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        # Ensure data is shape (n,) or (n,channels) in bytes
        if y_int16.ndim > 1:
            wf.writeframes(y_int16.tobytes(order="C"))
        else:
            wf.writeframes(y_int16.tobytes())
    return buf.getvalue()

def synth_bytes(text: str, speaker_wav: str, language: str="tr") -> bytes:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty text")
    speaker_abs = _to_abs(speaker_wav)
    if not os.path.exists(speaker_abs):
        raise FileNotFoundError(f"speaker not found: {speaker_abs}")

    key = (text, speaker_abs, language)
    if key in CACHE:
        return CACHE[key]

    tts = get_tts()
    out = tts.tts(text=text, speaker_wav=speaker_abs, language=language)

    if isinstance(out, tuple) and len(out) == 2:
        wav, sr = out
    else:
        wav = out
        sr = getattr(tts, "output_sample_rate", None) \
             or getattr(getattr(tts, "synthesizer", None), "output_sample_rate", None) \
             or 24000

    wav_bytes = wav_bytes_from_np(wav, sr)
    CACHE[key] = wav_bytes
    return wav_bytes


@app.get("/health")
def health():
    return JSONResponse({"ok": True})

@app.post("/synthesize")
def synth_endpoint(payload: dict = Body(...)):
    text = payload.get("text", "")
    speaker = payload.get("speaker", "data/wavs/trk_0001.wav")
    language = payload.get("language", "tr")
    try:
        wav = synth_bytes(text, speaker, language)
        return Response(content=wav, media_type="audio/wav")
    except FileNotFoundError as e:
        raise HTTPException(404, f"{e}")
    except Exception as e:
        raise HTTPException(500, f"synthesis error: {e}")

@app.get("/stream")
def stream_endpoint(
    q: str = Query(..., description="Text to speak"),
    speaker: str = Query("data/wavs/trk_0001.wav"),
    language: str = Query("tr")
):
    def gen():
        wav = synth_bytes(q, speaker, language)
        # Single-chunk stream (for simplicity). For true streaming, chunk split required.
        yield wav
    return StreamingResponse(gen(), media_type="audio/wav")
