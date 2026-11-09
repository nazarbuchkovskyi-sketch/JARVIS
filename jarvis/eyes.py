"""Eyes: webcam capture. Keeps the latest frame for the UI and a JPEG on disk that
Claude reads with its Read tool when a question needs vision."""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("jarvis.eyes")


class Eyes:
    def __init__(self, index: int, frame_path: Path, fps: float = 12.0, disk_interval: float = 1.0):
        self.index = index
        self.frame_path = frame_path
        self.fps = fps
        self.disk_interval = disk_interval
        self._jpeg: Optional[bytes] = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._seq = 0
        self._enabled = threading.Event()
        self._ctl_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.error: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return self._enabled.is_set()

    def set_enabled(self, on: bool) -> None:
        with self._ctl_lock:
            alive = self._thread is not None and self._thread.is_alive()
            if on:
                if self._enabled.is_set() and alive:
                    return
                if alive:  # a previous loop is still shutting down: let it finish first
                    self._thread.join(timeout=5)
                self._enabled.set()
                self._thread = threading.Thread(target=self._loop, daemon=True, name="eyes")
                self._thread.start()
            else:
                self._enabled.clear()

    def latest(self, after_seq: int = -1, timeout: float = 1.0) -> tuple[int, Optional[bytes]]:
        with self._cond:
            if self._seq <= after_seq:
                self._cond.wait(timeout)
            return self._seq, self._jpeg

    def _loop(self) -> None:
        import cv2

        self.frame_path.parent.mkdir(parents=True, exist_ok=True)
        cap = None
        last_disk = 0.0
        try:
            while self._enabled.is_set():
                if cap is None:
                    cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
                    if not cap.isOpened():
                        self.error = f"Камера {self.index} недоступна"
                        log.warning(self.error)
                        cap.release()
                        cap = None
                        time.sleep(3)
                        continue
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.error = None
                ok, frame = cap.read()
                if not ok:
                    cap.release()
                    cap = None
                    time.sleep(0.5)
                    continue
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    data = buf.tobytes()
                    with self._cond:
                        self._jpeg = data
                        self._seq += 1
                        self._cond.notify_all()
                    now = time.time()
                    if now - last_disk >= self.disk_interval:
                        last_disk = now
                        try:
                            tmp = self.frame_path.with_suffix(".tmp")
                            tmp.write_bytes(data)
                            os.replace(tmp, self.frame_path)
                        except OSError as exc:  # file briefly locked (antivirus, a reader): skip a frame
                            log.debug("frame write skipped: %s", exc)
                time.sleep(1.0 / self.fps)
        finally:
            if cap is not None:
                cap.release()
            with self._cond:
                self._jpeg = None
                self._seq += 1
                self._cond.notify_all()
            try:
                self.frame_path.unlink(missing_ok=True)
            except OSError:
                pass
            log.info("camera stopped")
