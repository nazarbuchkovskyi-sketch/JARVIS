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

# Read-only commands that take no arguments other than flags.
NO_ARGS = {
    "get-date", "date", "hostname", "whoami", "systeminfo", "tasklist", "get-uptime", "uptime",
    "get-culture", "get-timezone", "get-location", "pwd", "uname", "free", "df", "get-computerinfo",
    "get-volume", "get-psdrive", "get-netadapter", "get-netipaddress", "get-netipconfiguration", "get-hotfix",
}
# Read-only commands whose arguments are plain names (process, service, CIM class).
NAME_ARGS = {"get-process", "gps", "get-service", "gsv", "get-ciminstance", "get-wmiobject"}
# Listing directories / checking existence: path arguments allowed (never UNC).
PATH_ARGS = {"get-childitem", "gci", "ls", "dir", "test-path"}
# Printing literal text.
TEXT_ARGS = {"echo", "write-output", "write-host"}
EXACT = {"ipconfig": {(), ("/all",)}}

# Output shapers allowed after a pipe. None of them takes a file path.
PIPE_FILTERS = {
    "select-object", "select", "sort-object", "sort", "format-table", "ft", "format-list", "fl",
    "format-wide", "measure-object", "measure", "group-object", "convertto-json", "out-string",
}
# Parameters that would make an otherwise harmless command read/write files or run code.
BAD_PARAMS = re.compile(r"^-(path|literalpath|filepath|inputobject|outfile|outvariable|ov|ev|errorvariable|"
                        r"computername|cimsession|session|credential|scriptblock|"
                        r"argumentlist|verb|encodedcommand|e|ec|c|command|exec)\b", re.IGNORECASE)

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
    if any(BAD_PARAMS.match(a) for a in args):
        return False
    if piped:
        return word in PIPE_FILTERS
    if word in EXACT:
        return tuple(a.lower() for a in args) in EXACT[word]
    if word in NO_ARGS:
        return all(a.startswith("-") for a in args)
    if word in NAME_ARGS:
        return all(re.fullmatch(r"-?[A-Za-z0-9_.*\-]+", a) for a in args)
    if word in PATH_ARGS:
        return all(re.fullmatch(r"-?[A-Za-z0-9_.*:~/\\-]+", a) for a in args)
    if word in TEXT_ARGS:
        return all(re.fullmatch(r"[\w.,:!?*\-]+", a, re.UNICODE) for a in args)
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


YES = re.compile(r"^\W*(да|так|yes|ага|угу|давай|выполняй|выполни|разрешаю|подтверждаю|конечно|окей|ок|ok)\b", re.IGNORECASE)
NO = re.compile(r"\b(нет|не надо|не делай|не нужно|отмена|отмени|стоп|отклони|запрещаю|не разрешаю|no|ні)\b", re.IGNORECASE)


def parse_answer(text: str) -> Optional[bool]:
    """A short, unambiguous yes/no only: a phone call in the room must not approve a command."""
    words = re.findall(r"\w+", text, re.UNICODE)
    if not words or len(words) > 4:
        return None
    no, yes = NO.search(text), YES.search(text)
    if no and not yes:
        return False
    if yes and not no:
        return True
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
