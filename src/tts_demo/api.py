from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .wav import to_wav_bytes
from .xtts import XttsConfig, synth_np


def _cache_limit() -> int:
    try:
        return int(os.getenv("TTS_CACHE_SIZE", "32"))
    except Exception:
        return 32


class LruBytesCache:
    def __init__(self, max_items: int):
        self.max_items = max(0, int(max_items))
        self._d: "OrderedDict[Tuple[str, str, str], bytes]" = OrderedDict()

    def get(self, key: Tuple[str, str, str]) -> Optional[bytes]:
        if self.max_items == 0:
            return None
        v = self._d.get(key)
        if v is None:
            return None
        self._d.move_to_end(key)
        return v

    def set(self, key: Tuple[str, str, str], value: bytes) -> None:
        if self.max_items == 0:
            return
        self._d[key] = value
        self._d.move_to_end(key)
        while len(self._d) > self.max_items:
            self._d.popitem(last=False)


cfg = XttsConfig()
cache = LruBytesCache(_cache_limit())
ROOT_DIR = Path.cwd().resolve()
SPEAKER_DIR = (ROOT_DIR / os.getenv("TTS_SPEAKER_DIR", "data/wavs")).resolve()

app = FastAPI(title="XTTS API", version="0.1.0")

if os.path.exists("web_demo.html"):
    if os.path.isdir("assets"):
        app.mount("/assets", StaticFiles(directory="assets"), name="assets")

    @app.get("/")
    def home():
        return FileResponse("web_demo.html")


@app.get("/health")
def health():
    return JSONResponse(
        {
            "ok": True,
            "cwd": os.getcwd(),
            "model_name": cfg.model_name,
            "gpu": cfg.gpu,
            "default_language": cfg.default_language,
            "default_speaker_wav": cfg.default_speaker_wav,
            "speaker_dir": str(SPEAKER_DIR),
            "cache_size": cache.max_items,
        }
    )


def _resolve_under(base: Path, value: str) -> Path:
    requested = Path(value or ".")
    if not requested.is_absolute():
        requested = ROOT_DIR / requested
    resolved = requested.resolve()
    if resolved != base and base not in resolved.parents:
        raise HTTPException(400, f"path must stay under {base}")
    return resolved


def _resolve_speaker(value: str) -> Path:
    speaker = _resolve_under(SPEAKER_DIR, value)
    if speaker.suffix.lower() != ".wav":
        raise HTTPException(400, "speaker must be a .wav file")
    if not speaker.is_file():
        raise HTTPException(404, f"speaker not found: {speaker}")
    return speaker


@app.get("/speakers")
def speakers(dir: str = Query("data/wavs", description="Directory to scan for speaker WAVs")):
    base = _resolve_under(SPEAKER_DIR, dir)
    if not base.is_dir():
        raise HTTPException(404, f"not a directory: {base}")
    out: List[str] = []
    for item in sorted(base.iterdir()):
        if item.is_file() and item.suffix.lower() == ".wav":
            try:
                out.append(item.relative_to(ROOT_DIR).as_posix())
            except ValueError:
                out.append(item.as_posix())
    return JSONResponse({"speakers": out})


def _synth_bytes(text: str, speaker: str, language: str) -> bytes:
    speaker_path = _resolve_speaker(speaker)
    speaker_abs = str(speaker_path)
    key = ((text or "").strip(), speaker_abs, (language or "").strip() or cfg.default_language)
    if not key[0]:
        raise ValueError("empty text")

    hit = cache.get(key)
    if hit is not None:
        return hit

    wav, sr = synth_np(text=key[0], speaker_wav=speaker_abs, language=key[2], cfg=cfg)
    wav_bytes = to_wav_bytes(wav, sr)
    cache.set(key, wav_bytes)
    return wav_bytes


@app.post("/synthesize")
def synthesize(payload: Dict = Body(...)):
    text = payload.get("text", "")
    speaker = payload.get("speaker", cfg.default_speaker_wav)
    language = payload.get("language", cfg.default_language)
    try:
        wav = _synth_bytes(text, speaker, language)
        return Response(content=wav, media_type="audio/wav")
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"synthesis error: {e}")


@app.get("/stream")
def stream(
    q: str = Query(..., description="Text to speak"),
    speaker: str = Query(cfg.default_speaker_wav),
    language: str = Query(cfg.default_language),
):
    def gen():
        yield _synth_bytes(q, speaker, language)

    return StreamingResponse(gen(), media_type="audio/wav")

