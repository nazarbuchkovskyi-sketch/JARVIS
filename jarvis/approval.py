"""Approval gate for shell commands and web fetches Claude wants to run.

An *allowlist with a fixed grammar*: only a few read-only commands with plain
arguments pass automatically; anything else (or anything we cannot parse with
certainty) needs the user's "да" (voice) or a click in the window.
"""
from __future__ import annotations

import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlsplit

# Read-only commands and the EXACT flags each may take (no abbreviations, no values).
# PowerShell accepts any unambiguous prefix of a parameter ("-Comp" = -ComputerName),
# so only listed spellings pass; GNU tools take --opt=value, which never matches.
NO_ARGS: dict[str, set[str]] = {
    "get-date": set(), "date": set(), "hostname": set(), "whoami": set(), "systeminfo": set(),
    "tasklist": set(), "get-uptime": set(), "uptime": set(), "get-culture": set(), "get-timezone": set(),
    "get-location": set(), "pwd": set(), "uname": {"-a"}, "free": {"-h", "-m"}, "df": {"-h"},
    "get-computerinfo": set(), "get-volume": set(), "get-psdrive": set(), "get-netadapter": set(),
    "get-netipaddress": set(), "get-netipconfiguration": set(), "get-hotfix": set(),
}
# Read-only commands whose arguments are plain names (process, service, CIM class).
NAME_ARGS: dict[str, set[str]] = {
    "get-process": {"-name", "-id"}, "gps": {"-name", "-id"},
    "get-service": {"-name", "-displayname"}, "gsv": {"-name", "-displayname"},
    "get-ciminstance": {"-classname"}, "get-wmiobject": {"-class"},
}
# Listing directories / checking existence: local path arguments only.
PATH_ARGS: dict[str, set[str]] = {
    "get-childitem": {"-recurse", "-force", "-name", "-directory", "-file", "-filter", "-depth"},
    "gci": {"-recurse", "-force", "-name", "-directory", "-file", "-filter", "-depth"},
    "dir": {"-recurse", "-force", "-name", "-directory", "-file", "-filter", "-depth"},
    "ls": {"-l", "-a", "-la", "-al", "-lh", "-lah", "-1", "-r", "-t", "-lt", "-ltr"},
    "test-path": set(),
}
# Printing literal text.
TEXT_ARGS = {"echo", "write-output", "write-host"}
EXACT = {"ipconfig": {(), ("/all",)}}

# PowerShell output shapers allowed after a pipe (never GNU sort/grep: those read files).
PIPE_FILTERS: dict[str, set[str]] = {
    "select-object": {"-first", "-last", "-skip", "-unique", "-property"},
    "sort-object": {"-descending", "-unique", "-property"},
    "format-table": {"-autosize", "-wrap", "-property"}, "ft": {"-autosize", "-wrap", "-property"},
    "format-list": {"-property"}, "fl": {"-property"},
    "measure-object": {"-sum", "-average", "-maximum", "-minimum", "-property"},
    "group-object": {"-noelement", "-property"},
    "convertto-json": {"-depth", "-compress"}, "out-string": {"-width"},
}

_PLAIN = re.compile(r"[A-Za-z0-9_.*,\-]+")                    # names, numbers, property lists
_LOCAL_PATH = re.compile(r"(?:[A-Za-z]:)?[A-Za-z0-9_.*~/\\-]*")  # C:/x, ~/x, .\x — no other drive


def _local_path(token: str) -> bool:
    if not _LOCAL_PATH.fullmatch(token):
        return False
    norm = token.replace("/", "\\")
    return not norm.startswith("\\\\") and "\\\\" not in norm  # no UNC in any slash mix


def _args_ok(args: list[str], flags: set[str], value: "re.Pattern[str] | None" = None, path: bool = False) -> bool:
    for a in args:
        if a.startswith("-"):
            if a.lower() not in flags:
                return False
        elif path:
            if not _local_path(a):
                return False
        elif value is None or not value.fullmatch(a):
            return False
    return True


SAFE_APPS = {
    "notepad", "calc", "mspaint", "chrome", "msedge", "firefox", "spotify", "code",
    "telegram", "discord", "steam", "obsidian", "winword", "excel", "powerpnt", "taskmgr",
}
OPENERS = {"start", "start-process", "saps"}
# Sites that may be opened with a query string (the data then goes to them, not to an attacker).
QUERY_HOSTS = {"google.com", "www.google.com", "youtube.com", "www.youtube.com", "music.youtube.com",
               "uk.wikipedia.org", "en.wikipedia.org", "maps.google.com", "translate.google.com"}

# Anything that can nest, chain, expand or redirect is never auto-approved.
_META = re.compile(r"[(){}\[\]$@&<>;`^%!\r\n\"]|\|\||//|\\\\")  # \\ = UNC path (leaks NTLM)


def _strip_quotes(token: str) -> str:
    if len(token) >= 2 and token[0] == token[-1] == "'":
        return token[1:-1]
    return token


def _safe_url(url: str, quoted: bool) -> bool:
    query_chars = r"[A-Za-z0-9._~\-=+&%]*" if quoted else r"[A-Za-z0-9._~\-=+%]*"  # bare & would chain a command
    if not re.fullmatch(r"https?://[A-Za-z0-9.\-]+(:\d+)?(/[A-Za-z0-9._~\-/]*)?(\?" + query_chars + ")?", url):
        return False
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if parts.query:
        return host in QUERY_HOSTS
    return len(parts.path) <= 40


def _safe_open(args: list[str]) -> bool:
    """`Start-Process https://site` or `start notepad` — one target, no extra arguments."""
    args = [a for a in args if a.lower() not in ("-filepath",)]
    if len(args) != 1:
        return False
    target = _strip_quotes(args[0])
    if target.lower().startswith("http"):
        return _safe_url(target, quoted=target != args[0])
    if _META.search(target) or "'" in target:
        return False
    return target.lower().removesuffix(".exe") in SAFE_APPS


def _safe_segment(segment: str, piped: bool) -> bool:
    tokens = segment.split()
    if not tokens:
        return False
    word, args = tokens[0].lower(), tokens[1:]
    if word in OPENERS:
        return not piped and _safe_open(args)
    # Outside an opener's URL, no metacharacters and no quotes at all.
    if _META.search(segment) or "'" in segment:
        return False
    if piped:
        return word in PIPE_FILTERS and _args_ok(args, PIPE_FILTERS[word], _PLAIN)
    if word in EXACT:
        return tuple(a.lower() for a in args) in EXACT[word]
    if word in NO_ARGS:
        return _args_ok(args, NO_ARGS[word])
    if word in NAME_ARGS:
        return _args_ok(args, NAME_ARGS[word], _PLAIN)
    if word in PATH_ARGS:
        return _args_ok(args, PATH_ARGS[word], path=True)
    if word in TEXT_ARGS:
        return all(re.fullmatch(r"[\w.,:?*]+", a, re.UNICODE) for a in args)
    return False


def is_safe(command: str) -> bool:
    command = command.strip()
    if not command:
        return False
    # Openers keep their URL intact (it may contain & or %); they must be the whole command.
    first = command.split(None, 1)[0].lower()
    if first in OPENERS:
        return _safe_segment(command, piped=False)
    segments = command.split("|")
    return all(_safe_segment(seg.strip(), piped=i > 0) for i, seg in enumerate(segments))


def is_safe_tool(tool: str, tool_input: dict) -> bool:
    """WebFetch can carry data out in its URL, so it always asks."""
    if tool in ("Bash", "PowerShell"):
        return is_safe(str(tool_input.get("command", "")))
    return False


YES_WORDS = {"да", "yes", "давай", "выполняй", "выполни", "разрешаю", "подтверждаю", "конечно", "разрешаю"}
NO_WORDS = {"нет", "no", "отмена", "отмени", "отклони", "запрещаю", "стоп", "ні"}
# Words that may surround a clear answer without changing it.
FILLER = {"сэр", "джарвис", "пожалуйста", "можно", "можешь", "ладно", "ок", "окей"}
NEGATION_PHRASE = {"не", "надо", "делай", "нужно", "разрешаю"}


def parse_answer(text: str) -> Optional[bool]:
    """A short, unambiguous yes/no only.

    Every word must belong to the answer: "так себе", "ага щас", "да, но не сейчас" or
    a phone call in the room never approve a command.
    """
    words = [w.lower() for w in re.findall(r"\w+", text, re.UNICODE)]
    if not words or len(words) > 4:
        return None
    has_yes = any(w in YES_WORDS for w in words)
    has_no = any(w in NO_WORDS for w in words) or "не" in words
    if has_yes and not has_no and all(w in YES_WORDS or w in FILLER for w in words):
        return True
    if has_no and not has_yes and all(w in NO_WORDS or w in FILLER or w in NEGATION_PHRASE for w in words):
        return False
    if has_no and "не" in words and all(w in NO_WORDS or w in FILLER or w in NEGATION_PHRASE for w in words):
        return False  # "не разрешаю"
    return None  # unclear: keep waiting


@dataclass
class Pending:
    id: str
    tool: str
    command: str
    created: float = field(default_factory=time.time)
    event: threading.Event = field(default_factory=threading.Event)
    allowed: bool = False
    reason: str = ""


class Approvals:
    def __init__(self):
        self._lock = threading.Lock()
        self.pending: Optional[Pending] = None

    def open(self, tool: str, command: str) -> Pending:
        p = Pending(id=uuid.uuid4().hex[:10], tool=tool, command=command)
        with self._lock:
            if self.pending is not None:  # only one at a time: the older one is denied
                self._resolve_locked(self.pending, False, "заменён новым запросом")
            self.pending = p
        return p

    def resolve(self, pid: Optional[str], allowed: bool, reason: str) -> bool:
        with self._lock:
            p = self.pending
            if p is None or (pid is not None and p.id != pid):
                return False
            self._resolve_locked(p, allowed, reason)
            return True

    def _resolve_locked(self, p: Pending, allowed: bool, reason: str) -> None:
        p.allowed, p.reason = allowed, reason
        p.event.set()
        if self.pending is p:
            self.pending = None
