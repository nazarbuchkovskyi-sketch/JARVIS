"""Voice: local Piper TTS with sentence streaming and playback level metering."""
from __future__ import annotations

import logging
import queue
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Optional

import numpy as np

log = logging.getLogger("jarvis.voice")

_URL = re.compile(r"https?://\S+")
_MD = re.compile(r"[*_`#>|~\[\]]+")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️]")
_SENTENCE_END = re.compile(r"([.!?…]+[\"»”)]?)(\s+)")


def speakable(text: str) -> str:
    """Strip things that a speech synthesiser should not read aloud."""
    text = _URL.sub("ссылка", text)
    text = _EMOJI.sub("", text)
    text = _MD.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class SentenceSplitter:
    """Accumulates streamed text and yields complete sentences as soon as they are ready."""

    def __init__(self, soft_limit: int = 160):
        self.buf = ""
        self.soft_limit = soft_limit

    def feed(self, delta: str) -> list[str]:
        self.buf += delta
        out: list[str] = []
        while True:
            m = _SENTENCE_END.search(self.buf)
            nl = self.buf.find("\n")
            cut = None
            if m and (nl == -1 or m.end(1) <= nl):
                cut = m.end(1)
            elif nl != -1:
                cut = nl
            if cut is None:
                break
            sentence, self.buf = self.buf[:cut], self.buf[cut:].lstrip()
            if sentence.strip():
                out.append(sentence.strip())
        # Very long clause without punctuation: break at the last comma.
        if len(self.buf) > self.soft_limit:
            idx = self.buf.rfind(",", 0, self.soft_limit)
            if idx > 40:
                out.append(self.buf[: idx + 1].strip())
                self.buf = self.buf[idx + 1:].lstrip()
        return out

    def flush(self) -> list[str]:
        rest, self.buf = self.buf.strip(), ""
        return [rest] if rest else []


class Voice:
    def __init__(
        self,
        *,
        models_dir: Path,
        voice_name: str,
        length_scale: float,
        device: Optional[int | str],
        on_level: Callable[[float], None] = lambda _l: None,
    ):
        self.models_dir = models_dir
        self.voice_name = voice_name
        self.length_scale = length_scale
        self.device = device
        self.on_level = on_level
        self._piper = None
        self._sample_rate = 22050
        self._text_q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._audio_q: "queue.Queue[Optional[tuple[np.ndarray, int, str]]]" = queue.Queue(maxsize=8)
        self._stop_flag = threading.Event()
        self._generation = 0
        self._pending = 0
        self._pending_lock = threading.Lock()
        self._idle = threading.Event()
        self._idle.set()
        # What was said lately and when the speaker last made a sound: the mic hears the
        # speaker too, and this is how that echo is told apart from the user barging in.
        self.recent: deque[str] = deque(maxlen=8)
        self.last_sound_at = 0.0
        self.ready = False

    # ---- setup -------------------------------------------------------------
    def available_voices(self) -> list[str]:
        return sorted(p.name[: -len(".onnx")] for p in self.models_dir.glob("*.onnx"))

    def load(self, voice_name: Optional[str] = None) -> None:
        from piper import PiperVoice

        name = voice_name or self.voice_name
        path = self.models_dir / f"{name}.onnx"
        if not path.exists():
            raise FileNotFoundError(f"Голосовая модель не найдена: {path}")
        t = time.time()
        piper = PiperVoice.load(str(path))
        # Warm up so the first real sentence is fast.
        list(piper.synthesize("Готово."))
        self._piper = piper
        self._sample_rate = piper.config.sample_rate
        self.voice_name = name
        log.info("voice %s loaded in %.1fs", name, time.time() - t)
        self.ready = True

    def start(self) -> None:
        threading.Thread(target=self._synth_loop, daemon=True, name="voice-synth").start()
        threading.Thread(target=self._play_loop, daemon=True, name="voice-play").start()

    # ---- public API --------------------------------------------------------
    def say(self, text: str) -> None:
        text = speakable(text)
        if not text:
            return
        with self._pending_lock:
            self._pending += 1
            self._idle.clear()
        self._text_q.put(text)

    def stop(self) -> None:
        """Cut speech immediately and drop everything queued."""
        self._stop_flag.set()
        for q in (self._text_q, self._audio_q):
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
        with self._pending_lock:
            self._pending = 0
            self._idle.set()
        self.on_level(0.0)

    def wait_idle(self, timeout: Optional[float] = None) -> bool:
        return self._idle.wait(timeout)

    @property
    def speaking(self) -> bool:
        return not self._idle.is_set()

    # ---- threads -----------------------------------------------------------
    def _done_one(self) -> None:
        with self._pending_lock:
            self._pending = max(0, self._pending - 1)
            if self._pending == 0:
                self._idle.set()

    def _synth_loop(self) -> None:
        from piper import SynthesisConfig

        while True:
            text = self._text_q.get()
            if text is None:
                return
            self._stop_flag.clear()
            try:
                piper = self._piper  # a voice switch may replace it mid-way
                rate = piper.config.sample_rate
                cfg = SynthesisConfig(length_scale=self.length_scale)
                chunks = [c.audio_float_array for c in piper.synthesize(text, syn_config=cfg)]
                audio = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)
            except Exception:
                log.exception("synthesis failed for %r", text[:80])
                self._done_one()
                continue
            if self._stop_flag.is_set():
                continue
            self._audio_q.put((audio, rate, text))

    def _play_loop(self) -> None:
        import sounddevice as sd

        frame = 1024
        out = None  # kept open between sentences so Bluetooth outputs do not clip the start
        out_rate = 0
        while True:
            try:
                item = self._audio_q.get(timeout=2.0)
            except queue.Empty:
                if out is not None:
                    out.close()
                    out = None
                continue
            if item is None:
                return
            audio, rate, text = item
            try:
                if not self._stop_flag.is_set() and audio.size:
                    self.recent.append(text)  # only what really comes out of the speaker
                    if out is not None and out_rate != rate:
                        out.close()
                        out = None
                    if out is None:
                        out = sd.OutputStream(samplerate=rate, channels=1, dtype="float32",
                                              device=self.device, blocksize=frame)
                        out.start()
                        out_rate = rate
                        out.write(np.zeros((int(rate * 0.15), 1), np.float32))  # wake the device
                    tail = np.zeros(int(rate * 0.12), np.float32)  # breath between sentences
                    audio = np.concatenate([audio, tail])
                    for i in range(0, len(audio), frame):
                        if self._stop_flag.is_set():
                            break
                        block = audio[i:i + frame]
                        out.write(block.reshape(-1, 1))
                        self.last_sound_at = time.time()
                        self.on_level(float(np.sqrt(np.mean(block * block))))
                    if self._stop_flag.is_set() and out is not None:
                        out.abort()
                        out.close()
                        out = None
            except Exception:
                log.exception("playback failed")
                if out is not None:
                    try:
                        out.close()
                    except Exception:
                        pass
                    out = None
            finally:
                self.on_level(0.0)
                if not self._stop_flag.is_set():
                    self._done_one()
