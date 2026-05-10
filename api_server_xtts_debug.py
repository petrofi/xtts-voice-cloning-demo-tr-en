# api_server_xtts_debug.py
# Same API as fixed version, but with verbose diagnostics and safer waveform handling.

import io
import os
import traceback
from typing import Any, Dict, Tuple

import numpy as np
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False

from TTS.api import TTS

app = FastAPI(title="XTTS API (Debug)", version="0.3.1")

if os.path.exists("web_demo.html"):
    app.mount("/static", StaticFiles(directory=".", html=True), name="static")

    @app.get("/")
    def home():
        return FileResponse("web_demo.html")

def log(msg: str):
    print(f"[DEBUG] {msg}")

_tts = None
def get_tts():
    global _tts
    if _tts is None:
        log("Loading XTTS model...")
        _tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", gpu=True, progress_bar=False)
        log("Model loaded.")
    return _tts

def _to_abs(path: str) -> str:
    return os.path.abspath(path)

def to_wav_bytes(y: Any, sr: int) -> bytes:
    # Accept torch.Tensor, list, or np.ndarray
    if TORCH_OK and isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    y = np.array(y, dtype=np.float32).flatten()
    y = np.clip(y, -1.0, 1.0)
    y16 = (y * 32767.0).astype(np.int16)

    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(y16.tobytes())
    return buf.getvalue()

@app.get("/health")
def health():
    info = {
        "cwd": os.getcwd(),
        "torch": (torch.__version__ if TORCH_OK else "not_imported"),
        "cuda": (torch.cuda.is_available() if TORCH_OK else None),
        "speaker_exists": os.path.exists("data/wavs/trk_0001.wav")
    }
    return JSONResponse({"ok": True, "info": info})

@app.post("/synthesize")
async def synth_endpoint(payload: dict = Body(...), request: Request = None):
    try:
        text = payload.get("text", "")
        speaker = payload.get("speaker", "data/wavs/trk_0001.wav")
        language = payload.get("language", "tr")
        log(f"/synthesize payload={payload}")
        speaker_abs = _to_abs(speaker)
        if not os.path.exists(speaker_abs):
            log(f"Speaker not found: {speaker_abs}")
            raise HTTPException(404, f"speaker not found: {speaker_abs}")

        tts = get_tts()
        log("Calling tts.tts(...)")
        out = tts.tts(text=text, speaker_wav=speaker_abs, language=language)

        # Hem (wav, sr) hem de sadece wav dönebilir
        if isinstance(out, tuple) and len(out) == 2:
            wav, sr = out
        else:
            wav = out
            sr = getattr(tts, "output_sample_rate", None) \
                 or getattr(getattr(tts, "synthesizer", None), "output_sample_rate", None) \
                 or 24000

        wav_bytes = to_wav_bytes(wav, sr)
        log(f"Returning WAV {len(wav_bytes)} bytes.")
        return Response(content=wav_bytes, media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        log(f"ERROR: {e}\n{tb}")
        return PlainTextResponse(f"synthesis error: {e}\n\n{tb}", status_code=500)


@app.get("/stream")
def stream_endpoint(q: str, speaker: str="data/wavs/trk_0001.wav", language: str="tr"):
    try:
        speaker_abs = _to_abs(speaker)
        if not os.path.exists(speaker_abs):
            raise HTTPException(404, f"speaker not found: {speaker_abs}")

        tts = get_tts()
        out = tts.tts(text=q, speaker_wav=speaker_abs, language=language)

        if isinstance(out, tuple) and len(out) == 2:
            wav, sr = out
        else:
            wav = out
            sr = getattr(tts, "output_sample_rate", None) \
                 or getattr(getattr(tts, "synthesizer", None), "output_sample_rate", None) \
                 or 24000

        wav_bytes = to_wav_bytes(wav, sr)

        def gen():
            yield wav_bytes  # tek parça stream

        return StreamingResponse(gen(), media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as e:
        tb = traceback.format_exc()
        return PlainTextResponse(f"synthesis error: {e}\n\n{tb}", status_code=500)
