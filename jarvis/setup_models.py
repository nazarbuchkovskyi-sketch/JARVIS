"""Download the local Piper voices (run once: `python -m jarvis.setup_models`)."""
from __future__ import annotations

import sys
import urllib.request

from .config import MODELS_DIR

BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
# (language folder, locale, speaker, quality)
VOICES = [
    ("ru", "ru_RU", "denis", "medium"),
]


def main() -> None:
    target = MODELS_DIR / "piper"
    target.mkdir(parents=True, exist_ok=True)
    for lang, locale, voice, quality in VOICES:
        for ext in ("onnx", "onnx.json"):
            name = f"{locale}-{voice}-{quality}.{ext}"
            dest = target / name
            if dest.exists() and dest.stat().st_size > 0:
                continue
            url = f"{BASE}/{lang}/{locale}/{voice}/{quality}/{name}"
            print(f"downloading {name} ...", flush=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(dest)
    print("voices ready:", ", ".join(p.name for p in sorted(target.glob("*.onnx"))))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"model download failed: {exc}", file=sys.stderr)
        sys.exit(1)
