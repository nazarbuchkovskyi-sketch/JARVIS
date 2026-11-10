# Instructions for Claude Code: install and launch Jarvis

This repository is a ready-made local voice assistant ("Jarvis") for **Windows 10/11**.
When the user gives you this repo and says "запусти" / "run it" / "install it", do the
steps below end to end without asking for confirmation at each step. Talk to the user in
their language. Do **not** modify the project code, config or tests unless the user asks.

The user's own data lives in `workspace/` (memory notes, camera frames) and
`.window-profile/` — never read, print or upload it.

## 0. Platform

Windows only (uses Edge app windows, Windows audio, PowerShell). On macOS/Linux, tell the
user it is not supported and stop.

If the repo is not cloned yet, clone it into a folder whose path is **ASCII only**
(e.g. `%USERPROFILE%\JARVIS`); non-ASCII folder names break `cmd` working directories:

```powershell
git clone https://github.com/nazarbuchkovskyi-sketch/JARVIS.git "$env:USERPROFILE\JARVIS"
```

All commands below run in the repo root.

## 1. Prerequisites (check, install only what is missing)

| Need | Check | Install if missing |
|---|---|---|
| Python 3.13 (3.11–3.13 ok) | `python --version` | `winget install -e --id Python.Python.3.13` |
| Node.js 18+ | `node --version` | `winget install -e --id OpenJS.NodeJS.LTS` |
| Claude Code, logged in with a Claude subscription | `claude --version` (you are running in it) | — |
| Microsoft Edge | present on Windows by default | — |

After a winget install, open a new shell (PATH changes) before continuing.
Claude Code must be logged in with a subscription (Pro/Max) — Jarvis's brain runs `claude -p`
with the user's own login; no API key is used or needed.

## 2. Setup (one time, 5–15 min, downloads ~900 MB)

```powershell
cmd /c ".\setup.bat < nul"
```

It installs the pinned Python packages (`requirements.txt`), builds the UI (`ui/`, `npm ci`),
downloads the Piper voice `ru_RU-denis-medium` into `models/piper/` and the GigaAM v3
speech-recognition model into the Hugging Face cache, then writes `models/.ready`.
Run it with a long timeout (or in the background) and wait for `[Jarvis] Done.`
If it fails, read the error, fix the cause (missing tool, network), and re-run — it is idempotent.

Verify:

```powershell
python -m pytest -q
```

All tests must pass (they never touch the mic or camera).

## 3. Launch

Jarvis must keep running after your command returns. Start it with the hidden launcher `Jarvis.vbs`:
no console window, only the Jarvis window appears (do not run `start.bat` inside your shell tool — it would
be killed when the call ends). On the very first run (no `ui/dist` or `models/.ready`) the launcher opens
the visible `start.bat` instead, because downloads need a console.

```powershell
Start-Process wscript.exe -ArgumentList ('"' + (Join-Path (Get-Location) 'Jarvis.vbs') + '"')
```

Run this from PowerShell (the Windows default shell).

First check it is not already running — only one instance can use port 8765:

```powershell
try { Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 http://127.0.0.1:8765/ | Out-Null; "already running" } catch { "free" }
```

Start-up takes ~20–40 s. It is ready when `http://127.0.0.1:8765/` answers 200 and an Edge
window titled **Jarvis** is open. Poll for that; don't just sleep.

## 4. Tell the user how to use it

- Say «Джарвис, привет» (he speaks and understands **Russian**). Hold **F8** to talk without the name.
- Talk over him to interrupt; «стоп» / «хватит» makes him shut up. **Esc** or the «Стоп» button too.
- When it is quiet he looks at the camera and teases on his own (switch «Сам комментирует»).
- The mic and camera work **only while the Jarvis window is open**; closing it shuts Jarvis down
  after 5 s. Start again with the command from step 3 (or double-click `Jarvis.vbs`; `start.bat` = same with a console).
- Each reply uses the user's Claude subscription limits, like a normal Claude Code message.

## Troubleshooting

- `'setup.bat' is not recognized` → call it with `.\` (`.\setup.bat`); some shells don't run
  programs from the current folder.
- Python packages missing → `start.bat` says "Run setup.bat first": run step 2.
- He doesn't hear anything → Windows Settings → Privacy & security → Microphone → allow desktop
  apps; check the default input device; or set `mic_device` in `config.json`
  (list: `python -c "import sounddevice; print(sounddevice.query_devices())"`).
- Camera black / error → Privacy & security → Camera → allow desktop apps; close other apps
  using the camera; `camera_index` in `config.json`.
- «Мозг недоступен» → run `claude` once in a terminal and `/login`.
- Logs: `logs\jarvis.log` (`stt ...`, `turn query: first sentence ...`); `start.bat` shows the same live in a console.

Settings live in `config.json` (documented in `README.md`).
