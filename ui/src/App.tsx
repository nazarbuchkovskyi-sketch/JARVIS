import { useEffect, useMemo, useRef, useState } from "react";
import { Orb } from "./Orb";
import { useJarvis, type JarvisState, type LogItem } from "./useJarvis";

const STATE_WORD: Record<JarvisState, string> = {
  offline: "ОФЛАЙН",
  booting: "ЗАПУСК",
  idle: "ЖДУ",
  listening: "СЛУШАЮ",
  transcribing: "РАЗБИРАЮ",
  thinking: "ДУМАЮ",
  speaking: "ГОВОРЮ",
  approval: "РАЗРЕШИТЬ?",
};

const STATE_KICKER: Record<JarvisState, string> = {
  offline: "Нет связи с ядром, переподключаюсь",
  booting: "Прогреваю слух, голос и мозг",
  idle: "",
  listening: "Говорите, я на связи",
  transcribing: "Превращаю голос в текст",
  thinking: "Claude обдумывает ответ",
  speaking: "Esc или «Стоп», чтобы перебить",
  approval: "Скажите «да» или «нет»",
};

function humorLabel(v: number) {
  if (v <= 10) return "Сухой протокол";
  if (v <= 35) return "Лёгкая ирония";
  if (v <= 65) return "Британский дворецкий";
  if (v <= 85) return "Сарказм и подколки";
  return "Без тормозов";
}

function voiceLabel(id: string) {
  const name = id.replace(/^[a-z]{2}_[A-Z]{2}-/, "").replace(/-(high|medium|low|x_low)$/, "");
  return name.charAt(0).toUpperCase() + name.slice(1);
}

const pad = (n: number) => String(n).padStart(2, "0");

function useClock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const t = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(t);
  }, []);
  return now;
}

function Arrow() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M7 17 17 7M8 7h9v9" />
    </svg>
  );
}

function Mark() {
  return (
    <svg className="mark" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" aria-hidden="true">
      <circle cx="12" cy="12" r="8.5" />
      <path d="M12 1.5v5M12 17.5v5M1.5 12h5M17.5 12h5" />
    </svg>
  );
}

function Switch({ checked, onChange, disabled, label }: { checked: boolean; onChange: (v: boolean) => void; disabled?: boolean; label: string }) {
  return (
    <label className="sw">
      <span>{label}</span>
      <input type="checkbox" checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
      <i aria-hidden="true" />
    </label>
  );
}

function Bar({ label, value, warn = 85 }: { label: string; value: number | null; warn?: number }) {
  const v = value ?? 0;
  return (
    <div className="bar">
      <div className="bar-head">
        <span>{label}</span>
        <span>{value === null ? "—" : `${Math.round(v)}%`}</span>
      </div>
      <div className="bar-track">
        <div className={`bar-fill ${v >= warn ? "hot" : ""}`} style={{ width: `${Math.min(100, v)}%` }} />
      </div>
    </div>
  );
}

const WHO: Record<LogItem["role"], string> = { user: "ВЫ", assistant: "ДЖАРВИС", system: "СИСТЕМА", heard: "УСЛЫШАНО" };

function LogEntry({ item, n }: { item: LogItem; n: number }) {
  const t = new Date(item.ts * 1000);
  return (
    <li className={`entry ${item.role}`}>
      <span className="cross" aria-hidden="true">+</span>
      <div className="entry-head">
        <span>[ {pad(n)} ]</span>
        <span>{WHO[item.role]}</span>
        <span className="entry-time">
          {pad(t.getHours())}:{pad(t.getMinutes())}
        </span>
      </div>
      {item.tools?.map((tool, i) => (
        <div key={i} className="entry-tool">
          → {tool}
        </div>
      ))}
      {(item.text || item.streaming) && (
        <p className="entry-text">
          {item.text}
          {item.streaming && <span className="caret" />}
          {item.interrupted && <span className="cutoff"> · прервано</span>}
        </p>
      )}
    </li>
  );
}

export default function App() {
  const j = useJarvis();
  const now = useClock();
  const [draft, setDraft] = useState("");
  const [humor, setHumor] = useState<number | null>(null);
  const [camKey, setCamKey] = useState(0);
  const logEnd = useRef<HTMLDivElement>(null);
  const pttHeld = useRef(false);
  const humorTimer = useRef<number | undefined>(undefined);

  useEffect(() => {
    // Braces matter: newer Chromium returns a Promise from scrollIntoView,
    // and React would treat a returned value as the effect's cleanup.
    logEnd.current?.scrollIntoView({ block: "end" });
  }, [j.log]);
  useEffect(() => {
    if (j.settings) setHumor(j.settings.humor);
  }, [j.settings?.humor]);
  useEffect(() => {
    setCamKey((k) => k + 1); // reconnect the camera stream after a toggle or a reconnect
  }, [j.settings?.camera_enabled, j.hello]);

  // Hold Space (outside the text box) to talk, like a walkie-talkie. F8 works globally.
  useEffect(() => {
    const typing = (e: KeyboardEvent) => (e.target as HTMLElement)?.closest("input, textarea, select");
    const down = (e: KeyboardEvent) => {
      if (e.code === "Space" && !typing(e) && !e.repeat) {
        e.preventDefault();
        pttHeld.current = true;
        j.pttDown();
      } else if (e.code === "Escape") {
        j.stop();
      }
    };
    const up = (e: KeyboardEvent) => {
      if (e.code === "Space" && pttHeld.current) {
        pttHeld.current = false;
        j.pttUp();
      }
    };
    const blur = () => {
      if (pttHeld.current) {
        pttHeld.current = false;
        j.pttUp();
      }
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    window.addEventListener("blur", blur);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
      window.removeEventListener("blur", blur);
    };
  }, [j.pttDown, j.pttUp, j.stop]);

  const changeHumor = (v: number) => {
    setHumor(v);
    window.clearTimeout(humorTimer.current);
    humorTimer.current = window.setTimeout(() => j.set("humor", v), 250);
  };

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const text = draft.trim();
    if (!text) return;
    j.sendText(text);
    setDraft("");
  };

  const s = j.settings;
  const name = s?.assistant_name ?? "Джарвис";
  const follow = j.stats?.follow ?? 0;
  const live = j.ptt || (j.state === "idle" && (follow > 0 || s?.wake_word_required === false));
  const word = j.ptt ? "ГОВОРИТЕ" : live ? "СЛУШАЮ" : STATE_WORD[j.state];
  const kicker = j.ptt
    ? "Отпустите F8 или пробел, чтобы отправить"
    : j.state === "idle"
      ? live
        ? follow > 0
          ? `Можно продолжать без «${name}» · ${Math.ceil(follow)} с`
          : "Слушаю без слова-активатора"
        : `Скажите «${name}» или держите F8`
      : STATE_KICKER[j.state];
  const busy = j.state === "thinking" || j.state === "speaking" || j.state === "approval";

  // The live caption: the last thing said to Jarvis and his (streaming) answer.
  const caption = useMemo(() => {
    let you: LogItem | undefined;
    let him: LogItem | undefined;
    for (let i = j.log.length - 1; i >= 0; i--) {
      const it = j.log[i];
      if (!him && it.role === "assistant" && it.text) him = it;
      if (it.role === "user") {
        you = it;
        break;
      }
    }
    if (him && you && j.log.indexOf(him) < j.log.indexOf(you)) him = undefined;
    return { you: you?.text ?? "", him: him?.text ?? "" };
  }, [j.log]);

  const systems: [string, string][] = [
    ["Мозг", j.status.brain ?? "—"],
    ["Слух", j.status.ears ?? "—"],
    ["Голос", j.status.voice ?? "—"],
    ["Зрение", j.status.eyes ?? "—"],
  ];
  const tone = (v: string) => (/ошибка|недоступ/i.test(v) ? "bad" : /загруз|запуск/i.test(v) ? "wait" : "ok");
  const turns = j.log.filter((i) => i.role === "user").length;

  return (
    <div className={`app st-${j.state} ${j.ptt ? "st-ptt" : ""}`}>
      <div className="smoke s1" aria-hidden="true" />
      <div className="smoke s2" aria-hidden="true" />
      <div className="film" aria-hidden="true" />

      <header className="nav">
        <div className="logo">
          <Mark />
          <span>Jarvis</span>
        </div>
        <div className="nav-mid">[ ЛОКАЛЬНЫЙ АССИСТЕНТ · CLAUDE ]</div>
        <div className="nav-right">
          <span className="nav-date">{now.toLocaleDateString("ru-RU", { weekday: "short", day: "numeric", month: "long" })}</span>
          <span className="nav-clock">
            {pad(now.getHours())}:{pad(now.getMinutes())}
            <em>:{pad(now.getSeconds())}</em>
          </span>
        </div>
      </header>

      <main className="main">
        <section className="stage" aria-live="polite">
          <div className="stage-meta">
            <span>[ 01 — СОСТОЯНИЕ ]</span>
            <span>
              {(j.status.ears ?? "STT").split(" ")[0]} → CLAUDE {(s?.claude_model ?? "").toUpperCase()} → {s ? voiceLabel(s.tts_voice).toUpperCase() : "TTS"}
            </span>
          </div>
          <Orb state={j.state} levels={j.levels} />
          <span className="cross c-tl" aria-hidden="true">+</span>
          <span className="cross c-br" aria-hidden="true">+</span>
          <div className="stage-title">
            <p className="kicker">
              {(live || j.state === "listening") && <span className="rec" aria-hidden="true" />}
              {kicker}
            </p>
            <h1 className="display">{word}</h1>
          </div>
          <p className="caption">
            {caption.you ? (
              <>
                <span className="you">{caption.you}</span> <span className="him">{caption.him}</span>
              </>
            ) : (
              <span className="him">Скажите «{name}, который час?» или напишите ниже.</span>
            )}
          </p>

          {j.approval && (
            <div className="approval" role="alertdialog" aria-label="Нужно разрешение">
              <div className="approval-top">
                <span>[ НУЖНО РАЗРЕШЕНИЕ ]</span>
                <span>{j.approval.tool.toUpperCase()}</span>
              </div>
              <code className="approval-cmd">{j.approval.command}</code>
              <p className="approval-hint">Скажите «да» или «нет», или нажмите кнопку.</p>
              <div className="approval-actions">
                <button type="button" className="btn btn-ink" onClick={() => j.approve(j.approval!.id, true)}>
                  Разрешить
                  <i>
                    <Arrow />
                  </i>
                </button>
                <button type="button" className="btn-line" onClick={() => j.approve(j.approval!.id, false)}>
                  Отклонить
                </button>
              </div>
            </div>
          )}
        </section>

        <aside className="rail">
          <div className="rail-head">
            <span>[ ЖУРНАЛ ]</span>
            <span className="outline-num">{pad(turns)}</span>
          </div>
          <ul className="log">
            {j.log.length === 0 && <li className="empty">Здесь появится разговор.</li>}
            {j.log.map((item, i) => (
              <LogEntry key={item.key} item={item} n={i + 1} />
            ))}
            <div ref={logEnd} />
          </ul>
        </aside>
      </main>

      <form className="composer" onSubmit={submit}>
        <span className="composer-label">[ ВВОД ]</span>
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder={`Напишите ${name}у…`}
          aria-label="Сообщение Джарвису"
          disabled={j.state === "offline"}
        />
        <button
          type="button"
          className={`btn-ghost ${j.ptt ? "held" : ""}`}
          onPointerDown={(e) => {
            e.currentTarget.setPointerCapture(e.pointerId);
            j.pttDown();
          }}
          onPointerUp={() => j.pttUp()}
          onPointerCancel={() => j.pttUp()}
          disabled={j.state === "offline" || j.state === "booting"}
          title="Удерживайте, чтобы говорить (или пробел / F8)"
        >
          <i className="mic" aria-hidden="true" />
          {j.ptt ? "Говорите…" : "Удерживайте"}
        </button>
        <button type="button" className="btn-ghost" onClick={j.stop} disabled={!busy} title="Остановить (Esc)">
          <i className="sq" aria-hidden="true" />
          Стоп
        </button>
        <button type="submit" className="btn" disabled={!draft.trim() || j.state === "offline"}>
          Отправить
          <i>
            <Arrow />
          </i>
        </button>
      </form>

      <section className="feat">
        <article className="fc fc-glow">
          <div className="fc-top">
            <span>[ 02 — ХАРАКТЕР ]</span>
            <span>{humor ?? "—"}%</span>
          </div>
          <h3>{humor === null ? "Юмор" : humorLabel(humor)}</h3>
          <input
            className="range"
            type="range"
            min={0}
            max={100}
            step={5}
            value={humor ?? 0}
            disabled={humor === null}
            onChange={(e) => changeHumor(Number(e.target.value))}
            aria-label="Уровень юмора"
            style={{ ["--v" as string]: `${humor ?? 0}%` }}
          />
          <div className="fc-row">
            <Switch label="Мат" checked={s?.profanity ?? false} disabled={!s} onChange={(v) => j.set("profanity", v)} />
            <Switch label={`Только на «${name}»`} checked={s?.wake_word_required ?? true} disabled={!s} onChange={(v) => j.set("wake_word_required", v)} />
          </div>
        </article>

        <article className="fc fc-light">
          <div className="fc-top">
            <span>[ 03 — ГОЛОС ]</span>
            <span>ЛОКАЛЬНО</span>
          </div>
          <h3>{s ? voiceLabel(s.tts_voice) : "—"}</h3>
          <p className="fc-text">Синтез речи прямо на ноутбуке, без интернета.</p>
          <select className="select" value={s?.tts_voice ?? ""} onChange={(e) => j.set("tts_voice", e.target.value)} disabled={!s} aria-label="Голос">
            {j.voices.map((v) => (
              <option key={v} value={v}>
                {voiceLabel(v)}
              </option>
            ))}
          </select>
        </article>

        <article className="fc fc-dark">
          <div className="fc-top">
            <span>[ 04 — ЗРЕНИЕ ]</span>
            <Switch label="" checked={s?.camera_enabled ?? false} disabled={!s} onChange={(v) => j.set("camera_enabled", v)} />
          </div>
          <div className="cam">
            {s?.camera_enabled ? (
              <img
                key={camKey}
                src={`./camera.mjpg?k=${camKey}`}
                alt="Изображение с веб-камеры"
                onError={() => window.setTimeout(() => setCamKey((k) => k + 1), 2000)}
              />
            ) : (
              <span>Камера выключена</span>
            )}
          </div>
        </article>

        <article className="fc fc-dark">
          <div className="fc-top">
            <span>[ 05 — СИСТЕМЫ ]</span>
            <span>{j.state === "offline" ? "OFF" : "ON"}</span>
          </div>
          <dl className="sys">
            {systems.map(([k, v]) => (
              <div key={k} className={tone(v)}>
                <dt>{k}</dt>
                <dd title={v}>{v}</dd>
              </div>
            ))}
          </dl>
          <Bar label="CPU" value={j.stats?.cpu ?? null} />
          <Bar label="RAM" value={j.stats?.ram ?? null} />
        </article>
      </section>

      <footer className="foot">
        <span>EST 2026 · РАБОТАЕТ ЛОКАЛЬНО</span>
        <span className="foot-mid">
          F8 / ПРОБЕЛ — ГОВОРИТЬ <b aria-hidden="true">+</b>
        </span>
        <span>ESC — СТОП</span>
      </footer>
    </div>
  );
}
