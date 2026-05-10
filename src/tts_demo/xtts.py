from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Any, Optional, Tuple


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class XttsConfig:
    model_name: str = os.getenv("TTS_MODEL_NAME", "tts_models/multilingual/multi-dataset/xtts_v2")
    gpu: bool = _env_bool("TTS_GPU", True)
    progress_bar: bool = _env_bool("TTS_PROGRESS_BAR", False)
    default_language: str = os.getenv("TTS_DEFAULT_LANGUAGE", "tr")
    default_speaker_wav: str = os.getenv("TTS_DEFAULT_SPEAKER_WAV", "data/wavs/trk_0001.wav")


_lock = threading.Lock()
_tts: Any = None


def get_tts(cfg: Optional[XttsConfig] = None):
    """Lazy-load XTTS model once per process."""
    global _tts
    if _tts is not None:
        return _tts
    with _lock:
        if _tts is not None:
            return _tts
        cfg = cfg or XttsConfig()
        try:
            from TTS.api import TTS
        except Exception as e:
            raise RuntimeError(
                "TTS library is not installed/importable. Install deps first:\n"
                "  pip install -r requirements.txt\n"
                "or\n"
                "  pip install -e .\n"
                f"\nImport error: {e}"
            ) from e

        _tts = TTS(model_name=cfg.model_name, gpu=cfg.gpu, progress_bar=cfg.progress_bar)
        return _tts


def synth_np(
    *,
    text: str,
    speaker_wav: str,
    language: str,
    cfg: Optional[XttsConfig] = None,
) -> Tuple[Any, int]:
    """Run XTTS and return (audio, sample_rate)."""
    text = (text or "").strip()
    if not text:
        raise ValueError("empty text")

    speaker_abs = os.path.abspath(speaker_wav)
    if not os.path.exists(speaker_abs):
        raise FileNotFoundError(f"speaker not found: {speaker_abs}")

    tts = get_tts(cfg)
    out = tts.tts(text=text, speaker_wav=speaker_abs, language=language)

    if isinstance(out, tuple) and len(out) == 2:
        wav, sr = out
    else:
        wav = out
        sr = (
            getattr(tts, "output_sample_rate", None)
            or getattr(getattr(tts, "synthesizer", None), "output_sample_rate", None)
            or 24000
        )
    return wav, int(sr)

