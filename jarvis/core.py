"""The orchestrator: wires ears -> brain -> voice and publishes events for the UI.

Every spoken or typed request becomes a *turn* on one worker thread, so "busy",
"mic paused" and "speaking" always have a single owner.
"""
from __future__ import annotations

import difflib
import logging
import queue
import random
import re
import secrets
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from . import approval, persona
from .config import MODELS_DIR, ROOT, VAULT_DIR, VISION_DIR, WORKSPACE_DIR, Config

log = logging.getLogger("jarvis.core")

_WORD = re.compile(r"[\w'’ʼ-]+", re.UNICODE)
# Words that may come before an address: "Слушай, Джарвис, ...", "Эй, Джарвис".
_INTERJECTIONS = {"слушай", "эй", "ей", "ну", "привет", "хей", "hey", "окей", "алло", "слухай", "гей"}


def _is_name(word: str, target: str, cutoff: float) -> bool:
    """Fuzzy match for Whisper's spellings, but not inflected forms ("Джарвиса", "Джарвисом")."""
    if len(word) > len(target) and word.startswith(target[: len(target) - 1]):
        return False  # the name plus a case ending: someone talking *about* Jarvis
    if abs(len(word) - len(target)) > 1:
        return False
    return difflib.SequenceMatcher(None, word, target).ratio() >= cutoff


def match_wake_word(text: str, wake_words: list[str], cutoff: float = 0.8) -> tuple[bool, str]:
    """Return (addressed, remainder). Fuzzy, because Whisper spells "Джарвис" many ways.

    Only an *address* counts: the name as the first word ("Джарвис, ..."), right after
    an interjection ("Слушай, Джарвис, ..."), or as the last word ("..., Джарвис?").
    Talking *about* him ("я говорил с Джарвисом", "мне Джарвис сказал") does not wake him:
    inflected forms are rejected because only the bare name is compared.
    """
    targets = [w.lower() for w in wake_words]
    tokens = list(_WORD.finditer(text))
    for idx, tok in enumerate(tokens):
        first = idx == 0 or (idx == 1 and tokens[0].group(0).lower() in _INTERJECTIONS)
        # A trailing name is an address only when set off by a comma/dash: "Который час, Джарвис?"
        # ("сказал Джарвис" / "это Джарвис" are talk *about* him).
        last = idx == len(tokens) - 1 and idx > 0 and bool(re.search(r"[,—–-]\s*$", text[: tok.start()]))
        if not first and not last:
            continue
        word = tok.group(0).lower().strip("-'’ʼ")
        inflected = any(len(word) > len(t) and word.startswith(t[:-1]) for t in targets)
        if not inflected and any(_is_name(word, t, cutoff) for t in targets):
            end = tok.end()
            while end < len(text) and text[end] in ",.!:;—- ":
                end += 1  # swallow the punctuation that belonged to the address
            start = tok.start()
            if idx == len(tokens) - 1 and idx > 0:  # trailing address "..., Джарвис?": drop the comma before it
                while start > 0 and text[start - 1] in ", —-":
                    start -= 1
                tail = text[tok.end():].strip(" ,")
                remainder = (text[:start] + tail).strip()
            else:
                remainder = (text[:start] + " " + text[end:]).strip()
            remainder = re.sub(r"^[\s,.!?…:;—-]+", "", remainder)
            remainder = re.sub(r"\s+([,.!?…])", r"\1", remainder)
            remainder = re.sub(r"\s{2,}", " ", remainder)
            return True, remainder.strip()
    return False, text.strip()


_STOP_WORDS = {"стоп", "хватит", "заткнись", "тихо", "погоди", "подожди", "харэ", "хорош", "замолчи",
               "помолчи", "стой", "тсс", "цыц", "stop", "замолкни", "умолкни"}
_STOP_FILLER = {"ну", "эй", "всё", "все", "да", "ладно", "братан", "бро", "чувак", "слушай", "блин", "бля",
                "сэр", "уже", "пожалуйста", "секунду", "минуту", "так"}


def _words(text: str) -> list[str]:
    return [w.lower().replace("ё", "е").replace("э", "е").strip("-'’ʼ") for w in _WORD.findall(text)]


def is_stop_command(text: str, wake_words: list[str]) -> bool:
    """ "Стоп", "Джарвис, хватит", "погоди, братан": shut up, no new question."""
    names = {w.lower() for w in wake_words}
    words = [w for w in _words(text) if w and w not in names]
    if not words or len(words) > 4 or not any(w in _STOP_WORDS for w in words):
        return False
    return all(w in _STOP_WORDS or w in _STOP_FILLER for w in words)


# Short function words say nothing about who spoke: "а", "в", "что" are in every sentence.
_FUNCTION_WORDS = {"а", "и", "в", "во", "на", "что", "как", "это", "не", "ну", "да", "нет", "с", "со", "к", "ко", "по",
                   "у", "за", "то", "так", "же", "бы", "ли", "ты", "я", "он", "она", "мы", "вы", "они", "тоже", "из",
                   "от", "до", "о", "об", "но", "или", "там", "тут", "уже", "еще", "вот", "мне", "тебе", "его", "ее"}


def _content(words: list[str]) -> list[str]:
    # Digits are left out too: STT writes "8" where Jarvis said "восемь".
    return [w for w in words if len(w) > 2 and w not in _FUNCTION_WORDS and not w.isdigit()]


def is_echo(heard: str, spoken: list[str], threshold: float = 0.6) -> bool:
    """True when the mic most likely picked up Jarvis's own voice from the speaker.

    The user cutting in on the same topic ("завтра тоже дождь?") shares a word or two with
    what Jarvis said, so short phrases also need a two-word sequence he actually spoke.
    """
    heard_words = _words(heard)
    spoken_words = _words(" ".join(spoken))
    words = _content(heard_words)
    pool = set(spoken_words)
    if not pool:
        return False
    if not words:  # only "да", "нет", "а": echo only if he really said each of them
        return all(w in pool for w in heard_words if not w.isdigit())
    matched = sum(1 for w in words if w in pool or difflib.get_close_matches(w, pool, 1, 0.75))
    if matched / len(words) < threshold:
        return False
    if len(words) > 3:
        return True
    if len(heard_words) < 2:  # a single word ("Слушаю."): matched above, no pair to check
        return True
    pairs = set(zip(spoken_words, spoken_words[1:]))
    return any(pair in pairs for pair in zip(heard_words, heard_words[1:]))


def proactive_delay(base_min: float, base_max: float, streak: int, rng=random) -> float:
    """Seconds of quiet before Jarvis speaks up on his own; backs off while he is ignored."""
    delay = rng.uniform(base_min, max(base_min, base_max))
    return min(600.0, delay * (2 ** max(0, streak - 2)))


def is_skip(text: str) -> bool:
    """Claude's "nothing worth saying" answer to a proactive look at the camera
    (also "**SKIP**" or "Гляну. SKIP" — the token anywhere means silence)."""
    t = re.sub(r"[^\wА-Яа-яЁё]+", " ", text).upper().split()
    return "SKIP" in t or (bool(t) and t[0] == "ПРОПУСК")


class EventBus:
    """Thread-safe fan-out of JSON events to any number of async subscribers."""

    def __init__(self):
        self._subs: list[Callable[[dict], None]] = []
        self._lock = threading.Lock()

    def subscribe(self, fn: Callable[[dict], None]) -> Callable[[], None]:
        with self._lock:
            self._subs.append(fn)

        def unsubscribe() -> None:
            with self._lock:
                if fn in self._subs:
                    self._subs.remove(fn)

        return unsubscribe

    def publish(self, event: dict) -> None:
        with self._lock:
            subs = list(self._subs)
        for fn in subs:
            try:
                fn(event)
            except Exception:
                log.exception("subscriber failed")


@dataclass
class Turn:
    kind: str  # "query" (ask Claude) or "local" (a canned spoken reply)
    text: str
    shown: str
    gen: int = 0  # stop() bumps the generation: turns queued before it are dropped


class Jarvis:
    def __init__(self, config: Config):
        self.config = config
        self.bus = EventBus()
        self.state = "booting"
        self.status: dict[str, str] = {"brain": "загрузка", "ears": "загрузка", "voice": "загрузка", "eyes": "выключено",
                                       "planner": "не подключён"}
        self.history: deque[dict] = deque(maxlen=60)
        self.mic_level = 0.0
        self.out_level = 0.0
        self.follow_until = 0.0
        self.approvals = approval.Approvals()
        self.approval_token = secrets.token_urlsafe(24)
        self._turns: "queue.Queue[Turn]" = queue.Queue()
        self._cancel = threading.Event()
        self._busy = threading.Event()
        self._booted = threading.Event()
        self._gen = 0
        self._gate_lock = threading.Lock()
        self._ptt_lock = threading.Lock()  # push-to-talk events are applied strictly in order
        self.window_open = False
        # What the turn worker is doing right now (for barge-in decisions).
        self._current: Optional[dict] = None
        self._last_activity = time.time()  # last user speech or end of a turn
        self._proactive_streak = 0  # proactive remarks in a row without the user speaking
        self._proactive_due = 0.0
        self._barge: Optional[dict] = None  # an interruption waiting for the rest of the sentence
        self._barge_lock = threading.Lock()
        self._barge_wait = 0.9  # seconds to wait for the rest of an interrupting sentence
        self.brain = None
        self.ears = None
        self.voice = None
        self.eyes = None
        self.planner = None  # PlannerService when planner.env has PLANNER_DATABASE_URL
        self._reminded: set = set()  # reminder keys already spoken
        self._remind_wait: list = []  # due reminders waiting for a quiet moment

    # ---- events ------------------------------------------------------------
    def emit(self, event: dict) -> None:
        self.bus.publish(event)

    def _log(self, role: str, text: str, **extra: Any) -> None:
        item = {"role": role, "text": text, "ts": time.time(), **extra}
        self.history.append(item)
        self.emit({"type": "message", **item})

    def set_state(self, state: str) -> None:
        if state != self.state:
            self.state = state
            self.emit({"type": "state", "state": state})

    def set_status(self, part: str, value: str) -> None:
        if self.status.get(part) != value:
            self.status[part] = value
            self.emit({"type": "status", "status": dict(self.status)})

    def snapshot(self) -> dict:
        cfg = self.config.snapshot()
        p = self.approvals.pending
        return {
            "type": "hello",
            "state": self.state,
            "status": dict(self.status),
            "history": list(self.history),
            "config": {k: cfg[k] for k in ("humor", "profanity", "wake_word_required", "proactive", "tts_voice", "camera_enabled", "assistant_name", "claude_model")},
            "approval": None if p is None else {"id": p.id, "tool": p.tool, "command": p.command},
        }

    # ---- startup -----------------------------------------------------------
    def start(self) -> None:
        threading.Thread(target=self._boot, daemon=True, name="boot").start()
        threading.Thread(target=self._turn_loop, daemon=True, name="turns").start()
        threading.Thread(target=self._ticker, daemon=True, name="ticker").start()
        threading.Thread(target=self._proactive_loop, daemon=True, name="proactive").start()
        threading.Thread(target=self._planner_loop, daemon=True, name="planner").start()

    def _brain_settings(self, frame_path: Path):
        from .brain import BrainSettings

        cfg = self.config
        home = Path.home().as_posix()
        hook = f'"{Path(sys.executable).as_posix()}" -I "{(ROOT / "jarvis" / "approve_hook.py").as_posix()}"'
        planner = self.planner is not None
        allowed = list(cfg["claude_allowed_tools"])
        mcp_servers: dict = {}
        if planner:
            # Read-only planner tools; the server reads planner.env itself, so no secret is on a command line.
            mcp_servers["planner"] = {
                "type": "stdio",
                "command": sys.executable,
                "args": ["-I", str(ROOT / "jarvis" / "planner_mcp.py")],
            }
            allowed.append("mcp__planner")
        return BrainSettings(
            model=cfg["claude_model"],
            effort=cfg.get("claude_effort"),
            system_prompt=persona.system_prompt(cfg["assistant_name"], cfg["user_title"], frame_path, VAULT_DIR,
                                                planner=planner),
            cwd=WORKSPACE_DIR,
            mcp_servers=mcp_servers,
            allowed_tools=allowed,
            disallowed_tools=list(cfg["claude_disallowed_tools"]),
            # Jarvis must not inherit the user's global coding instructions.
            # Nor the repo's own CLAUDE.md/AGENTS.md (setup instructions for Claude Code, not for him).
            exclude_claude_md=[f"{home}/.claude/CLAUDE.md", f"{home}/.claude/**",
                               f"{ROOT.as_posix()}/CLAUDE.md", f"{ROOT.as_posix()}/AGENTS.md"],
            # Every shell command and web fetch goes through the approval gate.
            extra_settings={"hooks": {"PreToolUse": [{
                "matcher": "Bash|PowerShell|WebFetch",
                "hooks": [{"type": "command", "command": hook, "timeout": 130}],
            }]}},
            extra_env={
                "JARVIS_APPROVE_URL": f"http://127.0.0.1:{int(cfg['port'])}/api/approve",
                "JARVIS_TOKEN": self.approval_token,
            },
        )

    def _boot(self) -> None:
        cfg = self.config
        from .brain import ClaudeBrain
        from .ears import Ears
        from .eyes import Eyes
        from .voice import Voice

        frame_path = VISION_DIR / "latest.jpg"
        WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
        persona.ensure_vault(VAULT_DIR)
        self._init_planner()  # before the brain: it decides whether Claude gets the planner tools

        self.eyes = Eyes(int(cfg["camera_index"]), frame_path)
        if cfg["camera_enabled"]:
            self.eyes.set_enabled(True)
            self.set_status("eyes", "включено")

        try:
            self.brain = ClaudeBrain(self._brain_settings(frame_path))
            self.brain.warm_up()
            self.set_status("brain", f"Claude · {cfg['claude_model']}")
        except Exception as exc:
            log.exception("brain failed")
            self.brain = None
            self.set_status("brain", "ошибка")
            self._log("system", f"Мозг недоступен: {exc}")

        try:
            self.voice = Voice(
                models_dir=MODELS_DIR / "piper",
                voice_name=cfg["tts_voice"],
                length_scale=float(cfg["tts_length_scale"]),
                device=cfg.get("output_device"),
                on_level=self._on_out_level,
            )
            try:
                self.voice.load()
            except FileNotFoundError:
                fallback = (self.voice.available_voices() or [None])[0]
                if fallback is None:
                    raise
                log.warning("voice %s missing, using %s", cfg["tts_voice"], fallback)
                self.voice.load(fallback)
            self.voice.start()
            self.set_status("voice", f"Piper · {self.voice.voice_name}")
        except Exception as exc:
            log.exception("voice failed")
            self.voice = None
            self.set_status("voice", "ошибка")
            self._log("system", f"Голос недоступен: {exc}")

        try:
            self.ears = Ears(
                stt_model=cfg["stt_model"],
                language=cfg["language"],
                beam_size=int(cfg["stt_beam_size"]),
                device=cfg.get("mic_device"),
                vad_threshold=float(cfg["vad_threshold"]),
                min_silence_ms=int(cfg["vad_min_silence_ms"]),
                max_utterance_seconds=float(cfg["max_utterance_seconds"]),
                initial_prompt=f"{cfg['assistant_name']}.",
                on_utterance=self._on_utterance,
                on_level=self._on_mic_level,
                on_speech_start=self._on_speech_start,
                on_speech_end=self._on_speech_end,
                is_talking=lambda: bool(self.voice and (self.voice.speaking or time.time() - self.voice.last_sound_at < 0.4)),
            )
            self.ears.load()
            with self._gate_lock:  # a window opening right now must not be missed
                self.ears.set_gate(self.window_open)
                self.ears.start()
            self.set_status("ears", f"{self.ears.engine} · {cfg['stt_model']}")
            self._start_hotkey()
        except Exception as exc:
            log.exception("ears failed")
            self.ears = None
            self.set_status("ears", "ошибка")
            self._log("system", f"Микрофон недоступен: {exc}")

        self.set_state("idle")
        self._booted.set()
        self._log("system", "Системы запущены." + (f" Скажите «{cfg['assistant_name']}» и команду." if cfg["wake_word_required"] else ""))

    def shutdown(self) -> None:
        self.approvals.resolve(None, False, "Джарвис выключается")
        for part in (self.ears, self.voice):
            try:
                if part is not None and hasattr(part, "stop"):
                    part.stop()
            except Exception:
                pass
        if self.eyes is not None:
            self.eyes.set_enabled(False)
        if self.brain is not None:
            self.brain.close(kill=True)

    # ---- levels and periodic stats ----------------------------------------
    def _on_mic_level(self, level: float) -> None:
        self.mic_level = level

    def _on_out_level(self, level: float) -> None:
        self.out_level = level

    def _ticker(self) -> None:
        import psutil

        psutil.cpu_percent(None)
        last_stats = 0.0
        while True:
            time.sleep(0.05)
            try:
                self.emit({"type": "levels", "mic": round(self.mic_level, 4), "out": round(self.out_level, 4)})
                now = time.time()
                if now - last_stats < 2.0:
                    continue
                last_stats = now
                if self.eyes is not None and self.eyes.enabled:
                    self.set_status("eyes", f"ошибка: {self.eyes.error}" if self.eyes.error else "включено")
                vm = psutil.virtual_memory()
                bat = psutil.sensors_battery()
                self.emit({
                    "type": "stats",
                    "cpu": psutil.cpu_percent(None),
                    "ram": vm.percent,
                    "battery": None if bat is None else round(bat.percent),
                    "plugged": None if bat is None else bat.power_plugged,
                    "follow": max(0.0, round(self.follow_until - now, 1)) if not self._busy.is_set() else 0.0,
                })
            except Exception:
                log.exception("ticker failed")
                time.sleep(1)

    # ---- push-to-talk ------------------------------------------------------
    def ptt_event(self, active: bool) -> None:
        """Apply press/release in arrival order (callers may be on different threads)."""
        with self._ptt_lock:
            if active:
                self._ptt_down()
            else:
                self._ptt_up()

    def ptt_down(self) -> None:
        self.ptt_event(True)

    def ptt_up(self) -> None:
        self.ptt_event(False)

    def _ptt_down(self) -> None:
        if self.ears is None or self.ears.ptt_active or not self.window_open:
            return
        if self._busy.is_set() and self.approvals.pending is None:
            self.stop()  # talking over Jarvis cuts him off
        self.ears.ptt_begin()
        self.emit({"type": "ptt", "active": True})
        if self.approvals.pending is None:
            self.set_state("listening")

    def _ptt_up(self) -> None:
        if self.ears is None or not self.ears.ptt_active:
            return
        queued = self.ears.ptt_end()
        self.emit({"type": "ptt", "active": False})
        if self.approvals.pending is not None:
            return
        if queued:
            self.set_state("transcribing")
        elif not self._busy.is_set():
            self.set_state("idle")

    def _start_hotkey(self) -> None:
        key_name = self.config.get("ptt_key")
        if not key_name:
            return
        try:
            from pynput import keyboard
        except ImportError:
            log.warning("pynput not installed: global push-to-talk disabled")
            return
        target = getattr(keyboard.Key, str(key_name), None) or keyboard.KeyCode.from_char(str(key_name))

        # The hook thread must never block, or Windows' keyboard input stalls: hand events
        # to one worker queue (keeps press/release order; auto-repeat presses are dropped).
        events: "queue.Queue[bool]" = queue.Queue()
        held = {"on": False}

        def worker() -> None:
            while True:
                self.ptt_event(events.get())

        threading.Thread(target=worker, daemon=True, name="ptt").start()

        def on_press(key) -> None:
            if key == target and not held["on"]:
                held["on"] = True
                events.put(True)

        def on_release(key) -> None:
            if key == target and held["on"]:
                held["on"] = False
                events.put(False)

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.daemon = True
        listener.start()
        log.info("push-to-talk hotkey: %s", key_name)

    # ---- what the ears hear ------------------------------------------------
    def _on_speech_start(self) -> None:
        if not self._busy.is_set():
            self.set_state("listening")

    def _on_speech_end(self) -> None:
        """A sound was too short to be speech (cough, click): leave "listening"."""
        self._settle_state()

    def _settle_state(self) -> None:
        """When no turn runs, show "listening" while someone talks, otherwise "idle"."""
        if self._busy.is_set():
            return
        ears = self.ears
        if self.window_open and (self._barge is not None or (ears and (ears.ptt_active or ears.segmenter.active))):
            self.set_state("listening")
        elif self._turns.empty():
            self.set_state("idle")

    def _heard_echo(self, text: str, started_at: float) -> bool:
        """The mic stays open while Jarvis talks, so it also hears the speaker."""
        v = self.voice
        if v is None or not text:
            return False
        if not v.speaking and started_at > v.last_sound_at + 0.6:
            return False  # began after Jarvis went quiet: that is the user
        return is_echo(text, list(v.recent))

    def _on_utterance(self, text: str, seconds: float, forced: bool = False, started_at: Optional[float] = None) -> None:
        if not self.window_open:
            return  # the window was closed while this was being transcribed
        if started_at is None:
            started_at = time.time() - seconds
        # 1) An open approval question takes a short yes/no, no wake word needed. Only speech
        # that began while Jarvis was still clearly talking can be his own "Да или нет?".
        if self.approvals.pending is not None:
            v = self.voice
            if text and not forced and v is not None and started_at < v.last_sound_at - 0.2 and is_echo(text, list(v.recent)):
                log.info("echo ignored during approval: %r", text)
                return
            if text:
                answer = approval.parse_answer(text)
                self._log("user", text)
                if answer is not None:
                    self.approvals.resolve(None, answer, "Разрешено голосом." if answer else "Пользователь ответил «нет».")
            return
        if not forced and self._heard_echo(text, started_at):
            log.info("echo ignored: %r", text)
            self._settle_state()
            return
        if text:
            self._last_activity = time.time()
            self._proactive_streak = 0
        # 2) Push-to-talk: always addressed to Jarvis; runs after any turn in progress.
        if forced:
            if text:
                addressed, remainder = match_wake_word(text, self.config["wake_words"])
                self._turns.put(Turn("query", remainder if addressed and remainder else text, text, self._gen))
            else:
                self._settle_state()
            return
        if not text:
            self._settle_state()
            return
        # 3) The rest of a sentence that interrupted Jarvis: ask it as one question.
        if self._continue_barge(text):
            return
        # 4) Speech while Jarvis is thinking or talking: the user cuts in.
        if self._busy.is_set():
            self._barge_in(text, started_at)
            return
        # 5) Ambient speech while idle.
        self._ambient(text, started_at)

    def _ambient(self, text: str, started_at: float) -> None:
        wake = self.config["wake_words"]
        # Measured from when the user *started* talking: transcription time does not count.
        in_follow_up = started_at < self.follow_until
        addressed, remainder = match_wake_word(text, wake)
        if self.config["wake_word_required"] and not addressed and not in_follow_up:
            self._log("heard", text)  # shown faded: heard, but not addressed to Jarvis
            self._settle_state()
            return
        if is_stop_command(text, wake):  # a late "стоп" after he already finished
            self._log("user", text)
            self._settle_state()
            return
        if addressed and not remainder:
            self._turns.put(Turn("local", f"Слушаю, {self.config['user_title']}.", text, self._gen))
            return
        self.submit(remainder if addressed else text, shown=text)

    def _barge_in(self, text: str, started_at: float) -> None:
        """The user talked over Jarvis: shut up at once, then answer the new words."""
        wake = self.config["wake_words"]
        cur = self._current or {}
        kind, spoke = cur.get("kind"), bool(cur.get("spoke"))
        addressed, remainder = match_wake_word(text, wake)
        if kind == "proactive" and not spoke and not addressed:
            # He was only looking at the camera; people talking ("Тихо!" to someone else
            # included) is not for him. Drop the look and treat it like any room speech.
            log.info("proactive look dropped, someone talks: %r", text)
            self.stop()
            self._ambient(text, started_at)
            return
        if is_stop_command(text, wake):
            log.info("barge-in stop: %r", text)
            self.stop()
            self._log("user", text)
            self.follow_until = time.time() + float(self.config["follow_up_seconds"])
            return
        # After "Слушаю, сэр." any phrase is the question, even one word ("Время?").
        answering_listen = kind == "local"
        # Jarvis has not said a word yet: this is the rest of the same question
        # ("Что думаешь..." pause "...о войне?"), so ask both parts together.
        same_question = kind == "query" and not spoke and time.time() - cur.get("started", 0) < 15
        if not addressed and not answering_listen and len(_words(text)) < 2:
            log.info("barge-in ignored (too short): %r", text)
            return  # a lone "ага" or a cough is not an interruption
        self.stop()
        if addressed and not remainder:
            log.info("barge-in by name: %r", text)
            self._turns.put(Turn("local", f"Слушаю, {self.config['user_title']}.", text, self._gen))
            return
        query = remainder if addressed else text
        if same_question:
            query = f"{cur['text']} {query}"
        log.info("barge-in: %r -> %r", text, query)
        # While Jarvis talked, the mic was cut into short pieces: the user may still be
        # mid-sentence. Wait for the rest (see _continue_barge) before asking Claude.
        with self._barge_lock:
            self._barge = {"query": query, "shown": text, "at": time.time(), "gen": self._gen}
        self._settle_state()
        self._later(self._barge_wait, self._flush_barge)

    @staticmethod
    def _later(delay: float, fn) -> None:
        t = threading.Timer(delay, fn)
        t.daemon = True
        t.start()

    def _continue_barge(self, text: str) -> bool:
        with self._barge_lock:
            pending, self._barge = self._barge, None
        if pending is None:
            return False
        if time.time() - pending["at"] > 8:
            self._turns.put(Turn("query", pending["query"], pending["shown"], pending["gen"]))
            return False
        addressed, remainder = match_wake_word(text, self.config["wake_words"])
        rest = remainder if addressed else text
        self._turns.put(Turn("query", f"{pending['query']} {rest}".strip(), f"{pending['shown']} {text}", pending["gen"]))
        return True

    def _flush_barge(self) -> None:
        """No continuation came: ask what we have."""
        with self._barge_lock:
            pending = self._barge
            if pending is None:
                return
            if self.ears is not None and self.ears.segmenter.active and time.time() - pending["at"] < 8:
                self._later(0.4, self._flush_barge)  # still talking: wait for the rest
                return
            self._barge = None
        # Stamped with the generation it was made in: a Stop in between drops it.
        self._turns.put(Turn("query", pending["query"], pending["shown"], pending["gen"]))

    def submit(self, text: str, shown: Optional[str] = None) -> None:
        text = text.strip()
        if text:
            self._turns.put(Turn("query", text, shown or text, self._gen))

    def set_window_open(self, is_open: bool) -> None:
        """Jarvis hears and acts only while his window is open."""
        with self._gate_lock:
            self.window_open = is_open
            if self.ears is not None:
                self.ears.set_gate(is_open)
        if not is_open:
            if self.ears is not None:
                self.ears.ptt_cancel()
            self.stop()

    def stop(self) -> None:
        """User pressed stop: cut speech, deny a pending approval, abort and drop queued turns."""
        self._gen += 1
        self._cancel.set()
        self.follow_until = 0.0
        with self._barge_lock:
            self._barge = None
        while True:
            try:
                self._turns.get_nowait()
            except queue.Empty:
                break
        self.approvals.resolve(None, False, "Пользователь нажал «Стоп».")
        if self.brain is not None:
            self.brain.interrupt()
        if self.voice is not None:
            self.voice.stop()
        self._settle_state()

    # ---- approval gate (called from the hook via the server) ---------------
    def request_approval(self, tool: str, tool_input: dict) -> tuple[bool, str]:
        if approval.is_safe_tool(tool, tool_input):
            return True, "Безопасная команда без изменений в системе."
        what = str(tool_input.get("command") or tool_input.get("url") or tool_input)[:2000]
        if self._cancel.is_set() or not self._busy.is_set() or not self.window_open:
            return False, "Нет активного запроса пользователя."
        deadline = time.time() + 100  # always answer before the hook's own 125 s timeout
        pending = self.approvals.open(tool, what)
        self.emit({"type": "approval", "id": pending.id, "tool": tool, "command": what})
        self._log("system", f"Нужно разрешение ({tool}): {what}")
        self.set_state("approval")
        if self.voice is not None:
            self._wait_voice(limit=max(0.0, deadline - time.time() - 60))
            if not pending.event.is_set():
                self.voice.say(f"{self.config['user_title'].capitalize()}, нужно ваше разрешение. Да или нет?")
                self._wait_voice(limit=max(0.0, deadline - time.time() - 60))
        if not pending.event.wait(max(1.0, deadline - time.time())):
            self.approvals.resolve(pending.id, False, "Время на ответ вышло.")
        self.emit({"type": "approval_done", "id": pending.id, "allowed": pending.allowed})
        self._log("system", "Команда разрешена." if pending.allowed else f"Команда отклонена: {pending.reason}")
        if not self._cancel.is_set():
            self.set_state("thinking")
        return pending.allowed, pending.reason

    def resolve_approval(self, pid: str, allowed: bool) -> None:
        self.approvals.resolve(pid, allowed, "Разрешено в окне." if allowed else "Отклонено в окне.")

    # ---- the turn worker ---------------------------------------------------
    def _wait_voice(self, limit: float = 120.0) -> None:
        deadline = time.time() + limit
        while self.voice is not None and self.voice.speaking and not self._cancel.is_set() and time.time() < deadline:
            time.sleep(0.05)

    def _turn_loop(self) -> None:
        self._booted.wait()
        while True:
            turn = self._turns.get()
            if turn.gen != self._gen or not self.window_open:
                continue  # queued before a Stop, or the window is gone
            self._busy.set()
            self._cancel.clear()
            if turn.gen != self._gen:  # Stop landed between the check and clear()
                self._cancel.set()
            # The mic stays open during the turn so the user can cut in (see _barge_in).
            self._current = {"kind": turn.kind, "text": turn.text, "started": time.time(), "spoke": False}
            try:
                if turn.kind == "remind":  # a planner reminder: said by Jarvis on his own, no user line
                    self._log("assistant", turn.text, reminder=True)
                    if self.voice is not None:
                        self.set_state("speaking")
                        self._current["spoke"] = True
                        self.voice.say(turn.text)
                elif turn.kind == "local":
                    self._log("user", turn.shown)
                    if self.voice is not None:
                        self.set_state("speaking")
                        self._log("assistant", turn.text)
                        self._current["spoke"] = True
                        self.voice.say(turn.text)
                elif turn.kind == "proactive":
                    self._handle(turn.text, turn.shown, proactive=True)
                else:
                    self._handle(turn.text, turn.shown)
                self._wait_voice()
            except Exception as exc:
                log.exception("turn failed")
                self._log("system", f"Ошибка: {exc}")
            finally:
                # A reminder opens no follow-up window: room talk after it must not reach Claude unaddressed.
                spoke = bool(self._current and self._current.get("spoke")) and turn.kind != "remind"
                self._current = None
                self._finish_turn(cancelled=self._cancel.is_set(), spoke=spoke)

    def _finish_turn(self, cancelled: bool = False, spoke: bool = True) -> None:
        # After Stop, stop() already decided the follow-up window (a spoken "стоп" keeps one open).
        if not cancelled:
            # No follow-up window after a silent (skipped) proactive look.
            self.follow_until = time.time() + float(self.config["follow_up_seconds"]) if spoke else 0.0
        self._last_activity = time.time()
        self._busy.clear()
        self._settle_state()

    def _handle(self, text: str, shown: str, proactive: bool = False) -> None:
        from .voice import SentenceSplitter

        if not proactive:
            self._log("user", shown)
        if self.brain is None:
            self._log("system", "Мозг не подключён.")
            return
        if self._cancel.is_set():
            return

        if not proactive:
            self.set_state("thinking")
        splitter = SentenceSplitter()
        turn_id = f"t{int(time.time() * 1000)}"
        t0 = time.time()
        st = {"started": False, "first": None}

        def speak(delta: str) -> None:
            if self._cancel.is_set():
                return
            if not st["started"]:
                st["started"] = True
                self.emit({"type": "assistant_start", "id": turn_id})
            self.emit({"type": "assistant_delta", "id": turn_id, "text": delta})
            if self.voice:
                for sentence in splitter.feed(delta):
                    if st["first"] is None:
                        st["first"] = time.time() - t0
                    if self._current is not None:
                        self._current["spoke"] = True
                    self.voice.say(sentence)
                    if self.approvals.pending is None:
                        self.set_state("speaking")

        if not proactive:  # a proactive remark is shown only once we know it is not "SKIP"
            st["started"] = True
            self.emit({"type": "assistant_start", "id": turn_id})

        def on_delta(delta: str) -> None:
            if self._cancel.is_set():
                return
            if proactive:
                return  # decided at the end: it may turn out to be "SKIP"
            speak(delta)

        def on_tool(summary: str) -> None:
            if proactive:
                return  # a silent look at the camera: no "thinking" flicker in the window
            self.emit({"type": "tool", "id": turn_id, "text": summary})
            if self.approvals.pending is None:
                self.set_state("thinking")

        header = persona.turn_header(int(self.config["humor"]), bool(self.config["profanity"]),
                                     bool(self.eyes and self.eyes.enabled), plan=self._plan_line(),
                                     now=self._planner_now())
        result = self.brain.ask(
            f"{header}\n{text}",
            on_delta=on_delta,
            on_tool=on_tool,
            # A new Claude session (first turn or after a restart) gets the memory file.
            fresh_prefix=lambda: persona.memory_block(VAULT_DIR / "memory.md"),
            cancel=self._cancel,
        )

        skipped = proactive and (not result.ok or self._cancel.is_set() or is_skip(result.text or "")
                                 or not (result.text or "").strip())
        if proactive and not skipped:
            speak(result.text)
        if not self._cancel.is_set() and self.voice and not skipped:
            for sentence in splitter.flush():
                if self._current is not None:
                    self._current["spoke"] = True
                self.voice.say(sentence)
                self.set_state("speaking")
        log.info("turn %s: first sentence %.2fs, total %.2fs%s | %r", "proactive" if proactive else "query",
                 st["first"] or -1, time.time() - t0, " (skipped)" if skipped else "", (result.text or "")[:300])
        if proactive and skipped:
            if not result.ok:
                log.warning("proactive turn failed: %s", result.error)
            return  # nothing worth saying: nothing was shown or spoken
        if result.ok:
            self._log("assistant", result.text or "…", id=turn_id, interrupted=result.interrupted or self._cancel.is_set(),
                      **({"proactive": True} if proactive else {}))
        else:
            self._log("system", f"Ошибка мозга: {result.error}", id=turn_id)
            if self.voice and not self._cancel.is_set():
                if self._current is not None:
                    self._current["spoke"] = True
                self.voice.say(f"Простите, {self.config['user_title']}, связь с мозгом барахлит.")
        if st["started"]:
            self.emit({"type": "assistant_end", "id": turn_id})
        if self.voice and self.voice.speaking and not self._cancel.is_set():
            self.set_state("speaking")

    # ---- runtime settings --------------------------------------------------
    def apply_setting(self, key: str, value: Any) -> None:
        try:
            value = self.config.set_runtime(key, value)
        except OSError as exc:
            raise ValueError(f"не удалось сохранить config.json: {exc}") from exc
        if key == "camera_enabled" and self.eyes is not None:
            self.eyes.set_enabled(bool(value))
            self.set_status("eyes", "включено" if value else "выключено")
        self.emit({"type": "config", "key": key, "value": value})

    # ---- proactive mode: Jarvis looks at the camera and speaks up on his own ---
    def _proactive_ready(self) -> bool:
        ears = self.ears
        return bool(
            self.config.get("proactive") and self.window_open and self.brain is not None
            and self.eyes is not None and self.eyes.enabled and not self.eyes.error
            and not self._busy.is_set() and self._turns.empty() and self.approvals.pending is None
            and ears is not None and not ears.ptt_active and not ears.segmenter.active
        )

    def _proactive_loop(self) -> None:
        self._booted.wait()
        seen_activity = None
        while True:
            time.sleep(1.0)
            try:
                if seen_activity != self._last_activity:  # something happened: restart the countdown
                    seen_activity = self._last_activity
                    self._proactive_due = self._last_activity + proactive_delay(
                        float(self.config.get("proactive_min_seconds", 35)),
                        float(self.config.get("proactive_max_seconds", 70)),
                        self._proactive_streak)
                if time.time() < self._proactive_due or not self._proactive_ready():
                    continue
                frame = VISION_DIR / "latest.jpg"
                if not frame.exists() or time.time() - frame.stat().st_mtime > 5:
                    continue
                self._proactive_streak += 1
                self._last_activity = time.time()  # the next countdown starts after this turn
                self._turns.put(Turn("proactive", persona.proactive_prompt(frame), "", self._gen))
            except Exception:
                log.exception("proactive loop failed")

    # ---- planner (Planning System, read-only): context and reminders ---------
    def _init_planner(self) -> None:
        from .planner import PlannerDB, PlannerError, PlannerService, load_settings

        try:
            settings = load_settings()
        except PlannerError as exc:
            self.set_status("planner", "ошибка")
            self._log("system", f"Планер: {exc}")
            return
        if settings is None:
            return  # not configured: no tools, no reminders
        try:
            import psycopg  # noqa: F401
        except ImportError:
            self.set_status("planner", "ошибка")
            self._log("system", "Планер: не установлен пакет psycopg. Запустите setup.bat.")
            return
        db = PlannerDB(settings)
        self.planner = PlannerService(db, poll_seconds=float(self.config.get("planner_poll_seconds", 60)))
        self.set_status("planner", "подключение")
        log.info("planner configured: %r", settings)

    def _plan_line(self) -> str:
        """'план: сейчас … дальше …' for the turn header, from the cached calendar (no DB call here)."""
        from .planner import header_line

        p = self.planner
        if p is None or p.error or not p.fresh():
            return ""
        try:
            return header_line(p.snapshot(), p.db.now())
        except Exception:
            log.exception("plan line failed")
            return ""

    def _planner_now(self):
        """The planner's wall clock (its time zone) once the user is known, else None (= this PC)."""
        p = self.planner
        if p is None or p.db._user is None:
            return None
        try:
            return p.db.now()
        except Exception:
            return None

    def _reminder_ready(self) -> bool:
        ears = self.ears
        return bool(
            self.window_open and self.voice is not None and not self._busy.is_set() and self._turns.empty()
            and self.approvals.pending is None and self._barge is None
            and not (ears is not None and (ears.ptt_active or ears.segmenter.active))
        )

    def _planner_loop(self) -> None:
        self._booted.wait()
        p = self.planner
        if p is None:
            return
        next_refresh = 0.0
        while True:
            time.sleep(1.0)
            try:
                if time.monotonic() >= next_refresh:
                    ok = p.refresh()
                    self.set_status("planner", p.status_text())
                    next_refresh = time.monotonic() + (p.poll_seconds if ok else min(300.0, p.poll_seconds * 2))
                self._planner_tick()
            except Exception:
                log.exception("planner loop failed")
                time.sleep(5)

    def _planner_tick(self) -> Optional[str]:
        """Queue due reminders when the room is quiet; returns the text queued (for tests)."""
        from .planner import due_reminders, pick_address, reminder_text, still_due

        p = self.planner
        leads = self.config.get("planner_reminders") or []
        if p is None or not leads or not p.fresh():  # a failed poll keeps the fresh cache usable
            return None
        now = p.db.now()
        queued = {r.key for r in self._remind_wait}
        self._remind_wait += [r for r in due_reminders(p.snapshot(), now, leads, self._reminded) if r.key not in queued]
        live = {(e.id, e.start.isoformat()) for e in p.snapshot() if not e.done}
        # Missed while busy, or done / moved / deleted meanwhile: drop.
        self._remind_wait = [r for r in self._remind_wait if still_due(r, now) and r.key[:2] in live]
        if not self._remind_wait or not self._reminder_ready():
            return None
        text = reminder_text(self._remind_wait, now, pick_address(self.config["user_title"]))
        self._reminded.update(r.key for r in self._remind_wait)
        self._remind_wait = []
        log.info("planner reminder: %s", text)
        self._turns.put(Turn("remind", text, "", self._gen))
        return text
