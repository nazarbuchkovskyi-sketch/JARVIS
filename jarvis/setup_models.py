"""Download the local models once (`python -m jarvis.setup_models`):
the Piper voice and the speech-recognition model from config.json."""
from __future__ import annotations

import sys
import urllib.request

from .config import MODELS_DIR, Config

BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
# (language folder, locale, speaker, quality)
VOICES = [
    ("ru", "ru_RU", "denis", "medium"),
]
READY = MODELS_DIR / ".ready"


def download_voices() -> None:
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
    print("voice ready:", ", ".join(p.name for p in sorted(target.glob("*.onnx"))), flush=True)


def download_stt(name: str) -> None:
    """GigaAM comes from Hugging Face (~850 MB, cached in ~/.cache/huggingface)."""
    if name.startswith("gigaam"):
        import onnx_asr

        print(f"downloading speech recognition {name} (~850 MB, once) ...", flush=True)
        onnx_asr.load_model(name)
    else:
        from faster_whisper import WhisperModel

        print(f"downloading Whisper {name} ...", flush=True)
        WhisperModel(name, device="cpu", compute_type="int8")
    print("speech recognition ready:", name, flush=True)


def main() -> None:
    download_voices()
    download_stt(str(Config()["stt_model"]))
    READY.write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"model download failed: {exc}", file=sys.stderr)
        sys.exit(1)
