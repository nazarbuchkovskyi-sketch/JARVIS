"""Entry point: `python -m jarvis` starts the core and opens the Jarvis window."""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from logging.handlers import RotatingFileHandler
from pathlib import Path

import uvicorn

from .config import ROOT, Config
from .server import create_app


def setup_logging() -> None:
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = RotatingFileHandler(logs / "jarvis.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(fmt)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(console)
    for noisy in ("uvicorn.access", "faster_whisper", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def find_edge() -> str | None:
    for candidate in (
        shutil.which("msedge"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ):
        if candidate and Path(candidate).exists():
            return candidate
    return None


def open_window(url: str) -> None:
    for _ in range(100):  # wait until the server answers
        try:
            urllib.request.urlopen(url, timeout=1)
            break
        except Exception:
            time.sleep(0.2)
    edge = find_edge()
    if edge:
        profile = ROOT / ".window-profile"
        subprocess.Popen([edge, f"--app={url}", "--window-size=1440,900", f"--user-data-dir={profile}",
                          "--autoplay-policy=no-user-gesture-required"])
    else:
        webbrowser.open(url)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    setup_logging()
    config = Config()
    host, port = config["host"], int(config["port"])
    if host not in ("127.0.0.1", "localhost", "::1"):
        logging.warning("host %s is not local; forcing 127.0.0.1", host)
        host = "127.0.0.1"
    url = f"http://127.0.0.1:{port}/"
    with_window = bool(config["open_window"]) and "--no-window" not in sys.argv
    if with_window:
        threading.Thread(target=open_window, args=(url,), daemon=True).start()
    server: uvicorn.Server | None = None

    def request_exit() -> None:  # the Jarvis window was closed: stop listening and exit
        if server is not None:
            server.should_exit = True

    app = create_app(config, request_exit=request_exit, expect_window=with_window)
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning"))
    server.run()


if __name__ == "__main__":
    main()
