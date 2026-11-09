import { useCallback, useEffect, useRef, useState } from "react";

export type JarvisState = "booting" | "idle" | "listening" | "transcribing" | "thinking" | "speaking" | "approval" | "offline";

export interface Approval {
  id: string;
  tool: string;
  command: string;
}

export interface LogItem {
  key: string;
  role: "user" | "assistant" | "system" | "heard";
  text: string;
  ts: number;
  tools?: string[];
  streaming?: boolean;
  interrupted?: boolean;
}

export interface Settings {
  humor: number;
  profanity: boolean;
  wake_word_required: boolean;
  tts_voice: string;
  proactive: boolean;
  camera_enabled: boolean;
  assistant_name: string;
  claude_model: string;
}

export interface Stats {
  cpu: number;
  ram: number;
  battery: number | null;
  plugged: boolean | null;
  follow: number;
}

export interface Levels {
  mic: number;
  out: number;
}

let keySeq = 0;
const nextKey = () => `k${++keySeq}`;

export function useJarvis() {
  const [state, setState] = useState<JarvisState>("offline");
  const [status, setStatus] = useState<Record<string, string>>({});
  const [log, setLog] = useState<LogItem[]>([]);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [ptt, setPtt] = useState(false);
  const [approval, setApproval] = useState<Approval | null>(null);
  const [hello, setHello] = useState(0); // bumps on every (re)connect
  const levels = useRef<Levels>({ mic: 0, out: 0 }); // high-frequency: kept out of React state
  const ws = useRef<WebSocket | null>(null);

  useEffect(() => {
    let closed = false;
    let retry: number | undefined;

    const connect = () => {
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const sock = new WebSocket(`${proto}://${location.host}/ws`);
      ws.current = sock;

      sock.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        switch (msg.type) {
          case "hello":
            setState(msg.state);
            setStatus(msg.status);
            setSettings(msg.config);
            setLog(
              (msg.history ?? []).map((h: LogItem) => ({ ...h, key: nextKey() })),
            );
            setApproval(msg.approval ?? null);
            setHello((n) => n + 1);
            break;
          case "approval":
            setApproval({ id: msg.id, tool: msg.tool, command: msg.command });
            break;
          case "approval_done":
            setApproval((a) => (a && a.id === msg.id ? null : a));
            break;
          case "state":
            setState(msg.state);
            break;
          case "status":
            setStatus(msg.status);
            break;
          case "levels":
            levels.current = { mic: msg.mic, out: msg.out };
            break;
          case "stats":
            setStats(msg);
            break;
          case "ptt":
            setPtt(Boolean(msg.active));
            break;
          case "config":
            setSettings((s) => (s ? { ...s, [msg.key]: msg.value } : s));
            break;
          case "message":
            setLog((items) => {
              // The final assistant message replaces its streaming draft.
              if (msg.role === "assistant" && msg.id) {
                const idx = items.findIndex((i) => i.key === msg.id);
                if (idx !== -1) {
                  const copy = items.slice();
                  copy[idx] = { ...copy[idx], text: msg.text, streaming: false, interrupted: msg.interrupted };
                  return copy;
                }
              }
              return [...items, { key: msg.id && msg.role === "assistant" ? msg.id : nextKey(), role: msg.role, text: msg.text, ts: msg.ts ?? Date.now() / 1000 }].slice(-200);
            });
            break;
          case "assistant_start":
            setLog((items) => [...items, { key: msg.id, role: "assistant" as const, text: "", ts: Date.now() / 1000, tools: [], streaming: true }].slice(-200));
            break;
          case "assistant_delta":
            setLog((items) => items.map((i) => (i.key === msg.id ? { ...i, text: i.text + msg.text } : i)));
            break;
          case "tool":
            setLog((items) => items.map((i) => (i.key === msg.id ? { ...i, tools: [...(i.tools ?? []), msg.text] } : i)));
            break;
          case "assistant_end":
            setLog((items) =>
              items
                .map((i) => (i.key === msg.id ? { ...i, streaming: false } : i))
                // Drop an empty draft (e.g. the turn errored before any text).
                .filter((i) => !(i.key === msg.id && !i.text && !(i.tools && i.tools.length))),
            );
            break;
        }
      };
      sock.onclose = () => {
        setState("offline");
        setApproval(null);
        if (!closed) retry = window.setTimeout(connect, 1500);
      };
      sock.onerror = () => sock.close();
    };

    connect();
    return () => {
      closed = true;
      window.clearTimeout(retry);
      ws.current?.close();
    };
  }, []);

  const send = useCallback((obj: object) => {
    if (ws.current?.readyState === WebSocket.OPEN) ws.current.send(JSON.stringify(obj));
  }, []);

  return {
    state,
    status,
    log,
    settings,
    stats,
    ptt,
    approval,
    hello,
    levels,
    sendText: (text: string) => send({ type: "text", text }),
    stop: () => send({ type: "stop" }),
    set: (key: keyof Settings, value: unknown) => send({ type: "set", key, value }),
    pttDown: () => send({ type: "ptt", active: true }),
    pttUp: () => send({ type: "ptt", active: false }),
    approve: (id: string, allow: boolean) => send({ type: "approve", id, allow }),
  };
}
