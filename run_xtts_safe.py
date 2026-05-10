#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_xtts_safe.py
- Loads XTTS-v2 and synthesizes speech with robust checks.
- Adapts to installed TTS version (only passes supported kwargs).
- Prints environment diagnostics for easier debugging.
"""

import argparse
import sys
from pathlib import Path
import inspect

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text", type=str, default=None, help="Text to synthesize.")
    p.add_argument("--text_file", type=str, default=None, help="Read text from file (UTF-8).")
    p.add_argument("--speaker_wav", type=str, required=True, help="Reference speaker WAV path.")
    p.add_argument("--language", type=str, default="tr", help="Language code, e.g., 'tr'.")
    p.add_argument("--out_path", type=str, default="out_custom.wav", help="Output WAV path.")
    # Optional advanced knobs (will be passed only if supported)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--gpt_cond_len", type=int, default=None)
    p.add_argument("--gpt_cond_chunk", type=int, default=None)
    p.add_argument("--gpu", action="store_true", help="Use GPU if available.")
    args = p.parse_args()

    # Read text
    if args.text_file:
        tpath = Path(args.text_file)
        if not tpath.exists():
            print(f"[ERROR] text_file not found: {tpath}", file=sys.stderr)
            sys.exit(2)
        text = tpath.read_text(encoding="utf-8").strip()
    else:
        if not args.text:
            print("[ERROR] Provide --text or --text_file.", file=sys.stderr)
            sys.exit(2)
        text = args.text

    # Diagnostics
    try:
        import torch
        print(f"[INFO] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[INFO] cuda_device={torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"[WARN] torch diagnostics failed: {e}")

    try:
        import transformers
        print(f"[INFO] transformers={transformers.__version__}")
    except Exception as e:
        print(f"[WARN] transformers not importable: {e}")

    try:
        import TTS
        print(f"[INFO] TTS={TTS.__version__}")
    except Exception as e:
        print(f"[WARN] TTS version not detectable: {e}")

    from TTS.api import TTS as TTSApi

    print("[INFO] Loading model (xtts_v2)...")
    tts = TTSApi(model_name="tts_models/multilingual/multi-dataset/xtts_v2",
                 progress_bar=False, gpu=args.gpu)

    spk = Path(args.speaker_wav)
    if not spk.exists():
        print(f"[ERROR] speaker_wav not found: {spk}", file=sys.stderr)
        sys.exit(3)

    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Build kwargs based on supported signature
    sig = inspect.signature(tts.tts_to_file)
    supported = set(sig.parameters.keys())

    kw = dict(text=text, speaker_wav=str(spk), language=args.language, file_path=str(out))
    if args.temperature is not None and "temperature" in supported:
        kw["temperature"] = args.temperature
    if args.gpt_cond_len is not None and "gpt_cond_len" in supported:
        kw["gpt_cond_len"] = args.gpt_cond_len
    if args.gpt_cond_chunk is not None and "gpt_cond_chunk" in supported:
        kw["gpt_cond_chunk"] = args.gpt_cond_chunk

    print("[INFO] Supported tts_to_file params:", sorted(supported))
    print("[INFO] Synthesis starting...")
    try:
        tts.tts_to_file(**kw)
    except TypeError as te:
        # Reprint with the kwargs for clarity
        print("[ERROR] TypeError during tts_to_file call:", te, file=sys.stderr)
        print("[ERROR] Tried kwargs:", kw, file=sys.stderr)
        sys.exit(4)
    except Exception as e:
        print("[ERROR] Exception during synthesis:", repr(e), file=sys.stderr)
        sys.exit(5)

    print(f"[DONE] Saved: {out.resolve()}")

if __name__ == "__main__":
    main()
