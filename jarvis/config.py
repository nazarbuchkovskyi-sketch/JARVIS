"""Loading and saving the Jarvis configuration (config.json in the project root)."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"
MODELS_DIR = ROOT / "models"
WORKSPACE_DIR = ROOT / "workspace"
VISION_DIR = WORKSPACE_DIR / "vision"
VAULT_DIR = WORKSPACE_DIR / "vault"
UI_DIST = ROOT / "ui" / "dist"

DEFAULTS: dict[str, Any] = {
    "language": "ru",
    "assistant_name": "Джарвис",
    "user_title": "сэр",
    "humor": 60,
    "profanity": False,
    "wake_word_required": True,
    "wake_words": ["джарвис", "jarvis", "джервис", "жарвис", "джарвіс"],
    "follow_up_seconds": 10,
    "ptt_key": "f8",
    "claude_model": "sonnet",
    "claude_effort": "low",
    "claude_allowed_tools": ["Read", "Glob", "Grep", "WebSearch", "WebFetch", "Bash", "PowerShell", "Edit(vault/**)", "Write(vault/**)"],
    "claude_disallowed_tools": ["NotebookEdit", "Agent", "Task"],
    "stt_model": "small",
    "stt_beam_size": 1,
    "mic_device": None,
    "vad_threshold": 0.5,
    "vad_min_silence_ms": 700,
    "max_utterance_seconds": 20,
    "tts_voice": "ru_RU-ruslan-medium",
    "tts_length_scale": 1.0,
    "output_device": None,
    "camera_enabled": False,
    "camera_index": 0,
    "host": "127.0.0.1",
    "port": 8765,
    "open_window": True,
}

# Always denied, whatever config.json says: Jarvis must never read the user's secrets.
SECURITY_DENY = [
    f"{tool}({path})"
    for tool in ("Read", "Grep", "Glob")
    for path in ("~/.claude/**", "~/.claude.json", "~/.ssh/**", "~/.aws/**", "~/.azure/**", "~/.config/**",
                 "~/.gnupg/**", "~/.git-credentials", "~/.netrc", "~/AppData/**", "**/.env", "**/.env.*")
]

# Keys the UI is allowed to change at runtime (and their validators).
RUNTIME_KEYS = {
    "humor": lambda v: max(0, min(100, int(v))),
    "profanity": bool,
    "wake_word_required": bool,
    "tts_voice": str,
    "camera_enabled": bool,
}


class Config:
    def __init__(self, path: Path = CONFIG_PATH):
        self.path = path
        self._lock = threading.Lock()
        data = dict(DEFAULTS)
        if path.exists():
            data.update(json.loads(path.read_text(encoding="utf-8")))
        deny = list(data.get("claude_disallowed_tools") or [])
        data["claude_disallowed_tools"] = deny + [d for d in SECURITY_DENY if d not in deny]
        self._data = data

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self._data.get(key, default)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._data)

    def set_runtime(self, key: str, value: Any) -> Any:
        """Change a UI-editable key, persist it, and return the normalised value."""
        if key not in RUNTIME_KEYS:
            raise KeyError(f"key {key!r} cannot be changed at runtime")
        value = RUNTIME_KEYS[key](value)
        with self._lock:
            self._data[key] = value
            self._save_locked()
        return value

    def _save_locked(self) -> None:
        # Only write keys that differ from what is on disk, preserving user edits.
        on_disk: dict[str, Any] = {}
        if self.path.exists():
            on_disk = json.loads(self.path.read_text(encoding="utf-8"))
        for key in RUNTIME_KEYS:
            if key in self._data:
                on_disk[key] = self._data[key]
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(on_disk, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)
