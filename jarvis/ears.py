"""Ears: microphone capture, Silero VAD segmentation and faster-whisper transcription.

Everything here runs locally on the CPU.
"""
from __future__ import annotations

import collections
import logging
import queue
import re
import threading
import time
from typing import Callable, Optional

import numpy as np

log = logging.getLogger("jarvis.ears")

SAMPLE_RATE = 16000
BLOCK = 512  # Silero VAD expects 512-sample frames at 16 kHz (32 ms)

# Phrases Whisper tends to hallucinate on noise/silence.
_HALLUCINATIONS = re.compile(
    r"(субтитр|дякую за перегляд|спасибо за просмотр|продолжение следует|редактор субтитров|корректор|"
    r"підписуйтес|подписывайтесь|amara\.org|thanks for watching|ставте лайк)",
    re.IGNORECASE,
)


class Segmenter:
    """Turns a stream of VAD probabilities into utterances. Pure logic, easy to test."""

    def __init__(self, threshold: float, min_silence_ms: int, max_seconds: float,
                 preroll_ms: int = 300, min_speech_ms: int = 250):
        self.threshold = threshold
        self.neg_threshold = max(0.05, threshold - 0.15)
        self.silence_blocks = max(1, int(min_silence_ms / 1000 * SAMPLE_RATE / BLOCK))
        self.max_blocks = int(max_seconds * SAMPLE_RATE / BLOCK)
        self.min_speech_blocks = max(1, int(min_speech_ms / 1000 * SAMPLE_RATE / BLOCK))
        self.preroll: collections.deque = collections.deque(maxlen=max(1, int(preroll_ms / 1000 * SAMPLE_RATE / BLOCK)))
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.buf: list[np.ndarray] = []
        self.silence = 0
        self.speech = 0
        self.preroll.clear()

    def push(self, block: np.ndarray, prob: float) -> Optional[np.ndarray]:
        """Feed one block and its speech probability; return audio when an utterance ends."""
        if not self.active:
            self.preroll.append(block)
            if prob >= self.threshold:
                self.active = True
                self.buf = list(self.preroll)
                self.silence = 0
                self.speech = 1
            return None
        self.buf.append(block)
        if prob >= self.threshold:
            self.speech += 1
            self.silence = 0
        elif prob < self.neg_threshold:
            self.silence += 1
        if self.silence >= self.silence_blocks or len(self.buf) >= self.max_blocks:
            audio = np.concatenate(self.buf)
            enough = self.speech >= self.min_speech_blocks
            self.reset()
            return audio if enough else None
        return None


def clean_transcript(text: str) -> str:
    text = text.strip()
    if not text or _HALLUCINATIONS.search(text):
        return ""
    # Lone punctuation or a single filler syllable.
    if len(re.sub(r"[\W_]+", "", text)) < 2:
        return ""
    return text


class Ears:
    def __init__(
        self,
        *,
        stt_model: str,
        language: str,
        beam_size: int,
        device: Optional[int | str],
        vad_threshold: float,
        min_silence_ms: int,
        max_utterance_seconds: float,
        initial_prompt: str,
        on_utterance: Callable[[str, float, bool], None],
        on_level: Callable[[float], None] = lambda _l: None,
        on_speech_start: Callable[[], None] = lambda: None,
        on_speech_end: Callable[[], None] = lambda: None,
    ):
        self.language = language
        self.beam_size = beam_size
        self.device = device
        self.initial_prompt = initial_prompt
        self.on_utterance = on_utterance
        self.on_level = on_level
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self.segmenter = Segmenter(vad_threshold, min_silence_ms, max_utterance_seconds)
        self._stt_model_name = stt_model
        self._blocks: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=400)
        self._utterances: "queue.Queue[tuple[np.ndarray, bool]]" = queue.Queue(maxsize=3)
        self._paused = threading.Event()
        # Closed while no Jarvis window is open: the mic is then ignored completely.
        self._gate = threading.Event()
        self._ptt = threading.Event()
        self._ptt_buf: list[np.ndarray] = []
        self._ptt_lock = threading.Lock()
        self._stop = threading.Event()
        self._stream = None
        self.whisper = None
        self.vad = None
        self.ready = False

    # ---- setup -------------------------------------------------------------
    def load(self) -> None:
        from faster_whisper import WhisperModel
        from silero_vad import load_silero_vad

        t = time.time()
        self.vad = load_silero_vad(onnx=True)
        self.whisper = WhisperModel(self._stt_model_name, device="cpu", compute_type="int8")
        log.info("ears loaded in %.1fs (whisper=%s)", time.time() - t, self._stt_model_name)
        self.ready = True

    def start(self) -> None:
        import sounddevice as sd

        threading.Thread(target=self._vad_loop, daemon=True, name="ears-vad").start()
        threading.Thread(target=self._stt_loop, daemon=True, name="ears-stt").start()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=BLOCK,
            device=self.device, callback=self._callback,
        )
        self._stream.start()
        log.info("microphone started: %s", self._stream.device)

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    # ---- pause while Jarvis talks (half-duplex, avoids hearing itself) ------
    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        # Drop anything captured while paused.
        while not self._blocks.empty():
            try:
                self._blocks.get_nowait()
            except queue.Empty:
                break
        self.segmenter.reset()
        if self.vad is not None:
            self.vad.reset_states()
        self._paused.clear()

    @property
    def paused(self) -> bool:
        return self._paused.is_set() or not self._gate.is_set()

    def set_gate(self, open_: bool) -> None:
        """Listen only while the Jarvis window is open."""
        if open_:
            self._gate.set()
        else:
            self._gate.clear()
            self.ptt_cancel()

    # ---- push-to-talk: record everything while the key is held -------------
    def ptt_begin(self) -> None:
        with self._ptt_lock:
            self._ptt_buf = []
            self._ptt.set()

    def ptt_end(self) -> bool:
        """Stop recording; return True if the audio was queued for transcription."""
        with self._ptt_lock:
            if not self._ptt.is_set():
                return False
            self._ptt.clear()
            buf, self._ptt_buf = self._ptt_buf, []
        if len(buf) * BLOCK < SAMPLE_RATE * 0.3:
            return False  # too short to be speech
        audio = np.concatenate(buf)
        while True:  # push-to-talk wins over queued ambient speech
            try:
                self._utterances.put_nowait((audio, True))
                return True
            except queue.Full:
                try:
                    self._utterances.get_nowait()
                except queue.Empty:
                    pass

    def ptt_cancel(self) -> None:
        with self._ptt_lock:
            self._ptt.clear()
            self._ptt_buf = []

    @property
    def ptt_active(self) -> bool:
        return self._ptt.is_set()

    # ---- threads -----------------------------------------------------------
    def _callback(self, indata, frames, time_info, status) -> None:  # audio thread
        if status:
            log.debug("mic status: %s", status)
        try:
            self._blocks.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    def _vad_loop(self) -> None:
        import torch

        last_level = 0.0
        while not self._stop.is_set():
            try:
                block = self._blocks.get(timeout=0.5)
            except queue.Empty:
                continue
            level = float(np.sqrt(np.mean(block * block)))
            now = time.time()
            ptt = self._ptt.is_set()
            if now - last_level > 0.05:
                self.on_level(level if (ptt or not self.paused) else 0.0)
                last_level = now
            if not self._gate.is_set():
                if now - last_level > 0.05:
                    self.on_level(0.0)
                continue
            if ptt:
                with self._ptt_lock:
                    if self._ptt.is_set() and len(self._ptt_buf) * BLOCK < SAMPLE_RATE * 60:
                        self._ptt_buf.append(block)
                continue
            if self.paused or len(block) != BLOCK:
                continue
            try:
                prob = float(self.vad(torch.from_numpy(block), SAMPLE_RATE).item())
                was_active = self.segmenter.active
                audio = self.segmenter.push(block, prob)
                if not was_active and self.segmenter.active:
                    self.on_speech_start()
                if was_active and not self.segmenter.active and audio is None:
                    self.on_speech_end()  # too short: a cough or a click
                if audio is not None:
                    try:
                        self._utterances.put_nowait((audio, False))
                    except queue.Full:
                        log.warning("ears busy, dropping ambient utterance")
                        self.on_speech_end()
            except Exception:
                log.exception("vad step failed")
                self.segmenter.reset()

    def _stt_loop(self) -> None:
        while not self._stop.is_set():
            try:
                audio, forced = self._utterances.get(timeout=0.5)
            except queue.Empty:
                continue
            if self.paused and not forced:
                continue
            t = time.time()
            try:
                text = self.transcribe(audio)
            except Exception:
                log.exception("transcription failed")
                text = ""
            log.info("stt %.1fs audio -> %.2fs: %r", len(audio) / SAMPLE_RATE, time.time() - t, text)
            if forced or not self.paused:
                try:
                    self.on_utterance(text, len(audio) / SAMPLE_RATE, forced)
                except Exception:
                    log.exception("utterance handler failed")

    def transcribe(self, audio: np.ndarray) -> str:
        assert self.whisper is not None
        segments, _info = self.whisper.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt=self.initial_prompt,
            no_speech_threshold=0.6,
        )
        kept = [s.text for s in segments if not (s.no_speech_prob > 0.6 and s.avg_logprob < -1.0)]
        return clean_transcript(" ".join(kept))
