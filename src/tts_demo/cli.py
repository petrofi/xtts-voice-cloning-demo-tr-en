from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="tts-demo", description="XTTS v2 command line synthesizer")
    p.add_argument("--text", type=str, default=None, help="Text to synthesize.")
    p.add_argument("--text_file", type=str, default=None, help="Read text from file (UTF-8).")
    p.add_argument("--speaker_wav", type=str, required=True, help="Reference speaker WAV path.")
    p.add_argument("--language", type=str, default="tr", help="Language code, e.g., tr.")
    p.add_argument("--out_path", type=str, default="out_custom.wav", help="Output WAV path.")
    p.add_argument("--gpu", action="store_true", help="Use GPU if available (depends on env/model).")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--gpt_cond_len", type=int, default=None)
    p.add_argument("--gpt_cond_chunk", type=int, default=None)
    args = p.parse_args(argv)

    if args.text_file:
        tpath = Path(args.text_file)
        if not tpath.exists():
            print(f"[ERROR] text_file not found: {tpath}", file=sys.stderr)
            return 2
        text = tpath.read_text(encoding="utf-8").strip()
    else:
        if not args.text:
            print("[ERROR] Provide --text or --text_file.", file=sys.stderr)
            return 2
        text = args.text

    spk = Path(args.speaker_wav)
    if not spk.exists():
        print(f"[ERROR] speaker_wav not found: {spk}", file=sys.stderr)
        return 3

    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        from TTS.api import TTS as TTSApi
    except Exception as e:
        print(f"[ERROR] Could not import TTS: {e}", file=sys.stderr)
        return 10

    print("[INFO] Loading model (xtts_v2)...")
    tts = TTSApi(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False, gpu=args.gpu)

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
        print("[ERROR] TypeError during tts_to_file call:", te, file=sys.stderr)
        print("[ERROR] Tried kwargs:", kw, file=sys.stderr)
        return 4
    except Exception as e:
        print("[ERROR] Exception during synthesis:", repr(e), file=sys.stderr)
        return 5

    print(f"[DONE] Saved: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

