"""Barge-in, echo filtering, follow-up timing and proactive remarks (no models needed)."""
import random
import threading
import time
from types import SimpleNamespace

import numpy as np

from jarvis import persona
from jarvis.config import Config
from jarvis.core import Jarvis, is_echo, is_skip, is_stop_command, proactive_delay
from jarvis.ears import BLOCK, SAMPLE_RATE, Segmenter

WAKE = ["джарвис", "jarvis", "джервис", "жарвис", "джарвіс"]


class FakeVoice:
    def __init__(self):
        self.said: list[str] = []
        self.recent: list[str] = []
        self.last_sound_at = 0.0
        self.speaking = False
        self.stops = 0

    def say(self, text):
        self.said.append(text)
        self.recent.append(text)

    def stop(self):
        self.stops += 1
        self.speaking = False


class FakeBrain:
    def __init__(self, chunks):
        self.chunks = chunks
        self.asked: list[str] = []

    def ask(self, text, on_delta, on_tool, fresh_prefix=None, cancel=None):
        self.asked.append(text)
        for c in self.chunks:
            on_delta(c)
        return SimpleNamespace(ok=True, text="".join(self.chunks), interrupted=False, error=None)

    def interrupt(self):
        pass


def make_jarvis(tmp_path):
    cfg = Config(tmp_path / "config.json")
    j = Jarvis(cfg)
    j.window_open = True
    j._barge_wait = 60  # tests flush by hand
    j.voice = FakeVoice()
    j.ears = SimpleNamespace(segmenter=SimpleNamespace(active=False), ptt_active=False, ptt_cancel=lambda: None, set_gate=lambda _o: None)
    return j


def turns(j):
    out = []
    while not j._turns.empty():
        out.append(j._turns.get_nowait())
    return out


# ---- pure helpers -------------------------------------------------------
def test_stop_commands():
    for t in ["Стоп.", "Джарвис, хватит!", "погоди, братан", "Тихо!", "Ну всё, хватит", "заткнись"]:
        assert is_stop_command(t, WAKE), t
    for t in ["Стоп, а какая погода?", "подожди а сколько времени", "хватит болтать про погоду и расскажи анекдот", "да", ""]:
        assert not is_stop_command(t, WAKE), t


def test_echo_detection():
    spoken = ["Ну наконец-то, сэр. Восемь вечера, а вы всё ещё за ноутбуком."]
    assert is_echo("ну наконец то сэр восемь вечера а вы все еще за ноутбуком", spoken)
    assert is_echo("вечера а вы всё ещё за ноутбуком", spoken)
    assert not is_echo("подожди а какая завтра погода в киеве", spoken)
    assert not is_echo("какая погода", [])


def test_skip_and_delay():
    assert is_skip("SKIP") and is_skip(" skip.") and is_skip("ПРОПУСК")
    assert not is_skip("Опять с телефоном, братан?")
    rng = random.Random(1)
    assert 35 <= proactive_delay(35, 70, 0, rng) <= 70
    assert proactive_delay(35, 35, 4, rng) == 140  # ignored twice in a row: half as often
    assert proactive_delay(35, 70, 20, rng) == 600


def test_segmenter_limits_can_change_while_running():
    seg = Segmenter(0.5, 800, 20)
    block = np.zeros(BLOCK, np.float32)
    seg.silence_blocks, seg.max_blocks = 2, int(2.5 * SAMPLE_RATE / BLOCK)
    out = None
    for _ in range(200):
        out = seg.push(block, 0.9)
        if out is not None:
            break
    assert out is not None and len(out) <= 2.6 * SAMPLE_RATE  # cut into a short piece while Jarvis talks


# ---- the orchestrator -----------------------------------------------------
def test_follow_up_counts_from_speech_start(tmp_path):
    j = make_jarvis(tmp_path)
    j.follow_until = time.time() - 1  # expired while the phrase was being transcribed...
    j._on_utterance("Скажи, что у меня в руке?", 2.0, False, started_at=time.time() - 3)  # ...but it started in time
    assert [t.text for t in turns(j)] == ["Скажи, что у меня в руке?"]
    j._on_utterance("а это просто разговор", 2.0, False, started_at=time.time())
    assert turns(j) == []


def test_own_voice_is_ignored(tmp_path):
    j = make_jarvis(tmp_path)
    j._busy.set()
    j.voice.speaking = True
    j.voice.recent = ["Слушай, братан, восемь вечера, а ты всё ещё за ноутбуком."]
    j._on_utterance("восемь вечера а ты все еще за ноутбуком", 2.0, False, started_at=time.time() - 2)
    assert j.voice.stops == 0 and turns(j) == [] and j._barge is None


def test_talking_over_jarvis_stops_him_and_asks_the_whole_sentence(tmp_path):
    j = make_jarvis(tmp_path)
    j._busy.set()
    j._current = {"kind": "query", "text": "как дела", "started": time.time() - 5, "spoke": True}
    j.voice.speaking = True
    j.voice.recent = ["Да всё отлично, братан, сижу, скучаю."]
    j._on_utterance("подожди, а какая погода", 2.5, False, started_at=time.time() - 2.5)
    assert j.voice.stops == 1 and j._barge is not None
    j._busy.clear()
    j._on_utterance("завтра в Киеве?", 1.0, False, started_at=time.time() - 1)
    assert [t.text for t in turns(j)] == ["подожди, а какая погода завтра в Киеве?"]


def test_barge_without_continuation_is_flushed(tmp_path):
    j = make_jarvis(tmp_path)
    j._busy.set()
    j._current = {"kind": "proactive", "text": "", "started": time.time(), "spoke": True}
    j._on_utterance("сам ты в худи третий день", 2.0, False, started_at=time.time() - 2)
    j._flush_barge()
    assert [t.text for t in turns(j)] == ["сам ты в худи третий день"]


def test_spoken_stop_keeps_listening(tmp_path):
    j = make_jarvis(tmp_path)
    j._busy.set()
    j._current = {"kind": "query", "text": "x", "started": time.time(), "spoke": True}
    j._on_utterance("Джарвис, хватит!", 1.0, False, started_at=time.time() - 1)
    assert j.voice.stops == 1 and turns(j) == [] and j._barge is None
    j._finish_turn(cancelled=True)
    assert j.follow_until > time.time()  # the next phrase needs no wake word


def test_split_question_is_merged(tmp_path):
    j = make_jarvis(tmp_path)
    j._busy.set()
    j._current = {"kind": "query", "text": "Что вообще думаешь?", "started": time.time() - 2, "spoke": False}
    j._on_utterance("О войне.", 1.0, False, started_at=time.time() - 1)
    j._flush_barge()
    got = turns(j)
    assert [t.text for t in got] == ["Что вообще думаешь? О войне."] and got[0].shown == "О войне."


def test_lone_noise_does_not_interrupt(tmp_path):
    j = make_jarvis(tmp_path)
    j._busy.set()
    j._current = {"kind": "query", "text": "x", "started": time.time(), "spoke": True}
    j._on_utterance("ага", 0.5, False, started_at=time.time())
    assert j.voice.stops == 0 and j._barge is None


def test_ui_stop_drops_a_pending_interruption(tmp_path):
    j = make_jarvis(tmp_path)
    j._busy.set()
    j._current = {"kind": "query", "text": "x", "started": time.time(), "spoke": True}
    j._on_utterance("подожди, а какая погода", 2.0, False, started_at=time.time() - 2)
    j.stop()
    j._flush_barge()
    assert turns(j) == []


def test_proactive_skip_is_silent(tmp_path):
    j = make_jarvis(tmp_path)
    j.brain = FakeBrain(["SK", "IP"])
    j._current = {"kind": "proactive", "text": "", "started": time.time(), "spoke": False}
    events = []
    j.bus.subscribe(events.append)
    j._handle(persona.proactive_prompt(tmp_path / "f.jpg"), "", proactive=True)
    assert j.voice.said == [] and not any(e["type"] in ("message", "assistant_start") for e in events)
    assert not j._current["spoke"]


def test_proactive_remark_is_spoken(tmp_path):
    j = make_jarvis(tmp_path)
    j.brain = FakeBrain(["Оп", "ять с телефоном, братан? ", "Положи его уже."])
    j._current = {"kind": "proactive", "text": "", "started": time.time(), "spoke": False}
    j._handle(persona.proactive_prompt(tmp_path / "f.jpg"), "", proactive=True)
    assert j.voice.said == ["Опять с телефоном, братан?", "Положи его уже."]
    assert j._current["spoke"] and j.history[-1]["role"] == "assistant" and j.history[-1]["proactive"]
    assert "[САМ]" in j.brain.asked[0] and "мат включён" in j.brain.asked[0]


def test_short_proactive_remark_is_not_lost(tmp_path):
    j = make_jarvis(tmp_path)
    j.brain = FakeBrain(["Ого."])
    j._current = {"kind": "proactive", "text": "", "started": time.time(), "spoke": False}
    j._handle("p", "", proactive=True)
    assert j.voice.said == ["Ого."]


def test_proactive_waits_for_quiet(tmp_path):
    j = make_jarvis(tmp_path)
    j.config._data["proactive"] = True
    j.brain = object()
    j.eyes = SimpleNamespace(enabled=True, error=None)
    assert j._proactive_ready()
    j.ears.segmenter.active = True  # someone is talking
    assert not j._proactive_ready()
    j.ears.segmenter.active = False
    j._busy.set()
    assert not j._proactive_ready()
    j._busy.clear()
    j.window_open = False
    assert not j._proactive_ready()


def test_defaults_for_the_demo():
    from jarvis.config import DEFAULTS, RUNTIME_KEYS

    assert DEFAULTS["profanity"] is True and DEFAULTS["proactive"] is True and DEFAULTS["camera_enabled"] is True
    assert DEFAULTS["tts_voice"] == "ru_RU-denis-medium" and "tts_voice" not in RUNTIME_KEYS
    assert DEFAULTS["stt_model"].startswith("gigaam")


def test_persona_handles_hard_topics_like_a_person():
    p = persona.system_prompt("Джарвис", "сэр", persona.Path("f.jpg"), persona.Path("v"))
    assert "Никогда не отказывайся" in p and "SKIP" in p and "братан" in p


# ---- regressions from the independent review --------------------------------
def test_same_topic_interruptions_are_not_echo():
    said = ["В Киеве завтра плюс пять, братан, а к вечеру дождь.", "Так что зонт бери."]
    for t in ["а что с погодой в Харькове", "подожди, а в Харькове завтра?", "завтра тоже дождь?"]:
        assert not is_echo(t, said), t
    for t in ["к вечеру дождь так что зонт бери", "в киеве завтра плюс 5 братан", "так что зонт бери"]:
        assert is_echo(t, said), t


def test_quick_yes_to_approval_is_not_echo(tmp_path):
    j = make_jarvis(tmp_path)
    now = time.time()
    j.voice.recent = ["Сэр, нужно ваше разрешение. Да или нет?"]
    j.voice.last_sound_at = now - 0.3
    p = j.approvals.open("PowerShell", "Stop-Process x")
    j._on_utterance("Да.", 0.5, False, started_at=now - 0.2)
    assert p.event.is_set() and p.allowed


def test_own_no_during_approval_question_is_ignored(tmp_path):
    j = make_jarvis(tmp_path)
    now = time.time()
    j.voice.recent = ["Сэр, нужно ваше разрешение. Да или нет?"]
    j.voice.speaking, j.voice.last_sound_at = True, now
    p = j.approvals.open("PowerShell", "Stop-Process x")
    j._on_utterance("нет", 0.5, False, started_at=now - 1)  # a piece of his own question
    assert not p.event.is_set()


def run_proactive(tmp_path, chunks, ok=True):
    j = make_jarvis(tmp_path)
    j.brain = FakeBrain(chunks)
    if not ok:
        j.brain.ask = lambda *a, **k: SimpleNamespace(ok=False, text="", interrupted=False, error="boom")
    j._current = {"kind": "proactive", "text": "", "started": time.time(), "spoke": False}
    events = []
    j.bus.subscribe(events.append)
    j._handle("p", "", proactive=True)
    return j, events


def test_skip_never_leaks(tmp_path):
    for chunks in (["Гляну. ", "SKIP"], ["ПРОПУС", "К"], ["**SKIP**"], ["skip"], [""]):
        j, events = run_proactive(tmp_path, chunks)
        assert j.voice.said == [] and events == [], chunks
    j, events = run_proactive(tmp_path, [], ok=False)
    assert j.voice.said == [] and events == []


def test_people_talking_during_a_silent_look_are_not_a_question(tmp_path):
    j = make_jarvis(tmp_path)
    j._busy.set()
    j._current = {"kind": "proactive", "text": "", "started": time.time(), "spoke": False}
    j._on_utterance("да я ему потом перезвоню", 2.0, False, started_at=time.time() - 2)
    assert turns(j) == [] and j._barge is None and j.history[-1]["role"] == "heard"


def test_reply_to_a_remark_is_answered(tmp_path):
    j = make_jarvis(tmp_path)
    j._busy.set()
    j._current = {"kind": "proactive", "text": "", "started": time.time(), "spoke": True}
    j.voice.recent = ["Опять с телефоном, братан?"]
    j._on_utterance("сам ты с телефоном", 1.5, False, started_at=time.time() - 1.5)
    j._flush_barge()
    assert [t.text for t in turns(j)] == ["сам ты с телефоном"]


def test_aga_while_thinking_does_not_restart(tmp_path):
    j = make_jarvis(tmp_path)
    j._busy.set()
    j._current = {"kind": "query", "text": "какая погода в Киеве", "started": time.time(), "spoke": False}
    j._on_utterance("ага", 0.4, False, started_at=time.time())
    assert j.voice.stops == 0 and j._barge is None


def test_one_word_after_listening_is_a_question(tmp_path):
    j = make_jarvis(tmp_path)
    j._busy.set()
    j._current = {"kind": "local", "text": "Слушаю, сэр.", "started": time.time(), "spoke": True}
    j.voice.recent = ["Слушаю, сэр."]
    j._on_utterance("Время?", 0.6, False, started_at=time.time() - 0.6)
    j._flush_barge()
    assert [t.text for t in turns(j)] == ["Время?"]


def test_late_stop_is_not_sent_to_claude(tmp_path):
    j = make_jarvis(tmp_path)
    j.follow_until = time.time() + 10
    j._on_utterance("Стоп!", 0.5, False, started_at=time.time() - 0.5)
    assert turns(j) == []


def test_stop_after_barge_settles_state(tmp_path):
    j = make_jarvis(tmp_path)
    j._busy.set()
    j._current = {"kind": "query", "text": "x", "started": time.time(), "spoke": True}
    j._on_utterance("подожди, а какая погода", 2.0, False, started_at=time.time() - 2)
    j._busy.clear()
    j._current = None
    j.stop()
    assert j.state == "idle"


def test_echo_tail_after_turn_returns_to_idle(tmp_path):
    j = make_jarvis(tmp_path)
    j.state = "speaking"
    j._busy.set()
    j.ears.segmenter.active = True  # the tail of his own voice
    j._finish_turn()
    assert j.state == "listening"
    j.ears.segmenter.active = False
    j.voice.recent = ["Восемьдесят четыре, бро."]
    j.voice.last_sound_at = time.time()
    j._on_utterance("восемьдесят четыре бро", 1.0, False, started_at=time.time() - 1)
    assert j.state == "idle"


def test_no_proactive_during_approval_ptt_or_queued_turn(tmp_path):
    j = make_jarvis(tmp_path)
    j.config._data["proactive"] = True
    j.brain = object()
    j.eyes = SimpleNamespace(enabled=True, error=None)
    p = j.approvals.open("PowerShell", "x")
    assert not j._proactive_ready()
    j.approvals.resolve(p.id, False, "")
    j.ears.ptt_active = True
    assert not j._proactive_ready()
    j.ears.ptt_active = False
    j.submit("привет")
    assert not j._proactive_ready()
    turns(j)
    j.config._data["proactive"] = False
    assert not j._proactive_ready()


# ---- regressions from the second review ---------------------------------------
def test_quick_yes_to_a_question_is_heard(tmp_path):
    j = make_jarvis(tmp_path)
    now = time.time()
    j.voice.recent = ["Хотите, поставлю таймер на десять минут?"]
    j.voice.last_sound_at = now - 0.3
    j.follow_until = now + 20
    j._on_utterance("Да.", 0.4, False, started_at=now - 0.4)
    assert [t.text for t in turns(j)] == ["Да."]


def test_one_word_echo_of_listening_is_ignored(tmp_path):
    for heard in ["Слушаю.", "Сэр.", "Слушаю, сер."]:
        j = make_jarvis(tmp_path)
        j._busy.set()
        j._current = {"kind": "local", "text": "Слушаю, сэр.", "started": time.time(), "spoke": True}
        j.voice.recent = ["Слушаю, сэр."]
        j.voice.speaking = True
        j._on_utterance(heard, 0.6, False, started_at=time.time() - 0.6)
        assert j.voice.stops == 0 and j._barge is None, heard


def test_hush_to_someone_during_a_look_opens_no_window(tmp_path):
    j = make_jarvis(tmp_path)
    j._busy.set()
    j._current = {"kind": "proactive", "text": "", "started": time.time(), "spoke": False}
    j._on_utterance("Тихо!", 0.5, False, started_at=time.time() - 0.5)
    assert j.follow_until < time.time() and turns(j) == []


def test_window_close_with_queued_turn_goes_idle(tmp_path):
    j = make_jarvis(tmp_path)
    j.submit("привет")
    j.state = "listening"
    j.set_window_open(False)
    assert j.state == "idle" and turns(j) == []


def test_repo_setup_instructions_are_hidden_from_jarvis(tmp_path):
    from jarvis.config import ROOT, VISION_DIR

    j = Jarvis(Config(tmp_path / "config.json"))
    ex = j._brain_settings(VISION_DIR / "latest.jpg").exclude_claude_md
    assert f"{ROOT.as_posix()}/CLAUDE.md" in ex and any(e.endswith("/.claude/CLAUDE.md") for e in ex)
