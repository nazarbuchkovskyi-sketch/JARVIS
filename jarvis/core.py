"""The orchestrator: wires ears -> brain -> voice and publishes events for the UI.

Every spoken or typed request becomes a *turn* on one worker thread, so "busy",
"mic paused" and "speaking" always have a single owner.
"""
from __future__ import annotations

import difflib
import logging
import queue
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
        self.status: dict[str, str] = {"brain": "загрузка", "ears": "загрузка", "voice": "загрузка", "eyes": "выключено"}
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
        self._voice_lock = threading.Lock()
        self._gen = 0
        self._gate_lock = threading.Lock()
        self._ptt_lock = threading.Lock()  # push-to-talk events are applied strictly in order
        self.window_open = False
        self.brain = None
        self.ears = None
        self.voice = None
        self.eyes = None

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
            "config": {k: cfg[k] for k in ("humor", "profanity", "wake_word_required", "tts_voice", "camera_enabled", "assistant_name", "claude_model")},
            "voices": self.voice.available_voices() if self.voice else [],
            "approval": None if p is None else {"id": p.id, "tool": p.tool, "command": p.command},
        }

    # ---- startup -----------------------------------------------------------
    def start(self) -> None:
        threading.Thread(target=self._boot, daemon=True, name="boot").start()
        threading.Thread(target=self._turn_loop, daemon=True, name="turns").start()
        threading.Thread(target=self._ticker, daemon=True, name="ticker").start()

    def _brain_settings(self, frame_path: Path):
        from .brain import BrainSettings

        cfg = self.config
        home = Path.home().as_posix()
        hook = f'"{Path(sys.executable).as_posix()}" -I "{(ROOT / "jarvis" / "approve_hook.py").as_posix()}"'
        return BrainSettings(
            model=cfg["claude_model"],
            effort=cfg.get("claude_effort"),
            system_prompt=persona.system_prompt(cfg["assistant_name"], cfg["user_title"], frame_path, VAULT_DIR),
            cwd=WORKSPACE_DIR,
            allowed_tools=list(cfg["claude_allowed_tools"]),
            disallowed_tools=list(cfg["claude_disallowed_tools"]),
            # Jarvis must not inherit the user's global coding instructions.
            exclude_claude_md=[f"{home}/.claude/CLAUDE.md", f"{home}/.claude/**"],
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
            )
            self.ears.load()
            with self._gate_lock:  # a window opening right now must not be missed
                self.ears.set_gate(self.window_open)
                self.ears.start()
            self.set_status("ears", f"Whisper · {cfg['stt_model']}")
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
        if not self._busy.is_set() and self.state == "listening" and not (self.ears and self.ears.ptt_active):
            self.set_state("idle")

    def _on_utterance(self, text: str, seconds: float, forced: bool = False) -> None:
        if not self.window_open:
            return  # the window was closed while this was being transcribed
        # 1) An open approval question takes a short yes/no, no wake word needed.
        if self.approvals.pending is not None:
            if text:
                answer = approval.parse_answer(text)
                self._log("user", text)
                if answer is not None:
                    self.approvals.resolve(None, answer, "Разрешено голосом." if answer else "Пользователь ответил «нет».")
            return
        # 2) Push-to-talk: always addressed to Jarvis; runs after any turn in progress.
        if forced:
            if text:
                addressed, remainder = match_wake_word(text, self.config["wake_words"])
                self._turns.put(Turn("query", remainder if addressed and remainder else text, text, self._gen))
            elif not self._busy.is_set():
                self.set_state("idle")
            return
        # 3) Ambient speech while idle.
        if self._busy.is_set():
            return
        if not text:
            self.set_state("idle")
            return
        in_follow_up = time.time() < self.follow_until
        addressed, remainder = match_wake_word(text, self.config["wake_words"])
        if self.config["wake_word_required"] and not addressed and not in_follow_up:
            self._log("heard", text)  # shown faded: heard, but not addressed to Jarvis
            self.set_state("idle")
            return
        if addressed and not remainder:
            self._turns.put(Turn("local", f"Слушаю, {self.config['user_title']}.", text, self._gen))
            return
        self.submit(remainder if addressed else text, shown=text)

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
        time.sleep(0.3)  # let the speaker echo die down before listening for the answer
        if self.ears is not None and not pending.event.is_set():
            self.ears.resume()  # listen for "да" / "нет"
        if not pending.event.wait(max(1.0, deadline - time.time())):
            self.approvals.resolve(pending.id, False, "Время на ответ вышло.")
        if self.ears is not None:
            self.ears.pause()
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
            if self.ears:
                self.ears.pause()
            try:
                if turn.kind == "local":
                    self._log("user", turn.shown)
                    if self.voice is not None:
                        self.set_state("speaking")
                        self._log("assistant", turn.text)
                        self.voice.say(turn.text)
                else:
                    self._handle(turn.text, turn.shown)
                self._wait_voice()
            except Exception as exc:
                log.exception("turn failed")
                self._log("system", f"Ошибка: {exc}")
            finally:
                self._finish_turn(cancelled=self._cancel.is_set())

    def _finish_turn(self, cancelled: bool = False) -> None:
        time.sleep(0.25)  # let the room echo die down
        # No follow-up window after Stop: whatever is said next is not for Jarvis.
        self.follow_until = 0.0 if cancelled else time.time() + float(self.config["follow_up_seconds"])
        if self.ears:
            self.ears.resume()
        self._busy.clear()
        self.set_state("idle")

    def _handle(self, text: str, shown: str) -> None:
        from .voice import SentenceSplitter

        self._log("user", shown)
        if self.brain is None:
            self._log("system", "Мозг не подключён.")
            return
        if self._cancel.is_set():
            return

        self.set_state("thinking")
        splitter = SentenceSplitter()
        turn_id = f"t{int(time.time() * 1000)}"
        self.emit({"type": "assistant_start", "id": turn_id})

        def on_delta(delta: str) -> None:
            if self._cancel.is_set():
                return
            self.emit({"type": "assistant_delta", "id": turn_id, "text": delta})
            if self.voice:
                for sentence in splitter.feed(delta):
                    self.voice.say(sentence)
                    if self.approvals.pending is None:
                        self.set_state("speaking")

        def on_tool(summary: str) -> None:
            self.emit({"type": "tool", "id": turn_id, "text": summary})
            if self.approvals.pending is None:
                self.set_state("thinking")

        header = persona.turn_header(int(self.config["humor"]), bool(self.config["profanity"]), bool(self.eyes and self.eyes.enabled))
        result = self.brain.ask(
            f"{header}\n{text}",
            on_delta=on_delta,
            on_tool=on_tool,
            # A new Claude session (first turn or after a restart) gets the memory file.
            fresh_prefix=lambda: persona.memory_block(VAULT_DIR / "memory.md"),
            cancel=self._cancel,
        )

        if not self._cancel.is_set() and self.voice:
            for sentence in splitter.flush():
                self.voice.say(sentence)
                self.set_state("speaking")
        if result.ok:
            self._log("assistant", result.text or "…", id=turn_id, interrupted=result.interrupted or self._cancel.is_set())
        else:
            self._log("system", f"Ошибка мозга: {result.error}", id=turn_id)
            if self.voice and not self._cancel.is_set():
                self.voice.say(f"Простите, {self.config['user_title']}, связь с мозгом барахлит.")
        self.emit({"type": "assistant_end", "id": turn_id})
        if self.voice and self.voice.speaking and not self._cancel.is_set():
            self.set_state("speaking")

    # ---- runtime settings --------------------------------------------------
    def apply_setting(self, key: str, value: Any) -> None:
        if key == "tts_voice":
            self._switch_voice(str(value))
            return
        try:
            value = self.config.set_runtime(key, value)
        except OSError as exc:
            raise ValueError(f"не удалось сохранить config.json: {exc}") from exc
        if key == "camera_enabled" and self.eyes is not None:
            self.eyes.set_enabled(bool(value))
            self.set_status("eyes", "включено" if value else "выключено")
        self.emit({"type": "config", "key": key, "value": value})

    def _switch_voice(self, name: str) -> None:
        if self.voice is None:
            raise ValueError("голос не загружен")
        if name not in self.voice.available_voices():
            raise ValueError(f"неизвестный голос {name!r}")
        with self._voice_lock:  # one switch at a time; save only after it loaded
            self.voice.load(name)
            self.config.set_runtime("tts_voice", name)
        self.set_status("voice", f"Piper · {self.voice.voice_name}")
        self.emit({"type": "config", "key": "tts_voice", "value": name})
