from __future__ import annotations

import io
from typing import Any

import numpy as np


def to_wav_bytes(audio: Any, sample_rate: int) -> bytes:
    """
    Convert audio (torch.Tensor / list / np.ndarray) into 16-bit PCM WAV bytes.

    - Accepts 1D mono audio (or anything flattenable).
    - Clips to [-1, 1] and writes little-endian PCM16.
    """
    try:
        import torch  # optional

        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
    except Exception:
        pass

    y = np.asarray(audio, dtype=np.float32).flatten()
    y = np.clip(y, -1.0, 1.0)
    y16 = (y * 32767.0).astype(np.int16)

    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(y16.tobytes())
    return buf.getvalue()

