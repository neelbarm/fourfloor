"""Generate a self-contained ``preview.html`` for a finished remix.

No CDN, no build step, no external assets: one file with the session JSON and a
downsampled waveform inlined, next to the audio it plays.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np

from .audio import decode

WAVE_POINTS = 1400


def waveform(path: str | Path, points: int = WAVE_POINTS) -> list[float]:
    """Peak-per-bucket waveform of a file, normalised to [0, 1]."""
    return waveform_of(decode(path).mono, points)


def waveform_of(mono: np.ndarray, points: int = WAVE_POINTS) -> list[float]:
    """Peak-per-bucket waveform of a buffer already in memory.

    The web app has the rendered audio in hand and the decoded source in hand,
    so it draws both without paying for a second decode.
    """
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    if not len(mono):
        return [0.0] * points
    edges = np.linspace(0, len(mono), points + 1).astype(int)
    peaks = np.array([
        float(np.max(np.abs(mono[a:b]))) if b > a else 0.0
        for a, b in zip(edges[:-1], edges[1:])
    ])
    m = peaks.max()
    return [round(float(v / m), 4) for v in peaks] if m > 0 else peaks.tolist()


_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ — fourfloor</title>
<style>
  :root {
    --bg: #000000;
    --panel: rgba(255,255,255,0.055);
    --panel-line: rgba(255,255,255,0.10);
    --fg: #f5f5f7;
    --muted: #86868b;
    --accent: #ff4d9d;
    --accent-2: #2ee6d6;
    --drop: #ff4d9d;
    --build: #ffb340;
    --intro: #2ee6d6;
    --breakdown: #7d8cff;
    --outro: #6e6e73;
    --radius: 18px;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg); }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI",
                 Roboto, Helvetica, Arial, sans-serif;
    color: var(--fg);
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
    min-height: 100vh;
    overflow-x: hidden;
  }
  .glow {
    position: fixed; inset: -30% -10% auto -10%; height: 70vh; pointer-events: none;
    background: radial-gradient(60% 55% at 30% 0%, rgba(255,77,157,0.22), transparent 70%),
                radial-gradient(50% 50% at 75% 10%, rgba(46,230,214,0.16), transparent 70%);
    filter: blur(30px); z-index: 0;
  }
  .wrap { position: relative; z-index: 1; max-width: 1080px; margin: 0 auto; padding: 0 24px 96px; }

  header { padding: 96px 0 40px; }
  .eyebrow {
    font-size: 13px; letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--muted); font-weight: 590; margin: 0 0 18px;
    display: flex; align-items: center; gap: 10px;
  }
  .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--accent);
         box-shadow: 0 0 14px var(--accent); }
  h1 {
    font-size: clamp(42px, 8.2vw, 88px); line-height: 0.98; margin: 0 0 20px;
    font-weight: 700; letter-spacing: -0.038em;
    background: linear-gradient(180deg, #ffffff 30%, #a9a9b2 130%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }
  .sub { font-size: clamp(17px, 2.2vw, 21px); color: var(--muted); margin: 0; max-width: 62ch;
         line-height: 1.5; }
  .sub b { color: var(--fg); font-weight: 590; }

  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(152px, 1fr));
           gap: 12px; margin: 40px 0 0; }
  .stat {
    background: var(--panel); border: 1px solid var(--panel-line);
    border-radius: var(--radius); padding: 20px 20px 18px;
    backdrop-filter: blur(24px) saturate(160%);
    -webkit-backdrop-filter: blur(24px) saturate(160%);
  }
  .stat .v { font-size: 30px; font-weight: 640; letter-spacing: -0.03em;
             font-variant-numeric: tabular-nums; }
  .stat .k { font-size: 12px; color: var(--muted); letter-spacing: 0.07em;
             text-transform: uppercase; margin-top: 7px; font-weight: 560; }
  .stat .v.accent { color: var(--accent); }
  .stat .v.teal { color: var(--accent-2); }

  .panel {
    background: var(--panel); border: 1px solid var(--panel-line);
    border-radius: 24px; padding: 26px; margin-top: 20px;
    backdrop-filter: blur(24px) saturate(160%);
    -webkit-backdrop-filter: blur(24px) saturate(160%);
  }
  .panel h2 { font-size: 13px; letter-spacing: 0.12em; text-transform: uppercase;
              color: var(--muted); margin: 0 0 18px; font-weight: 590; }

  audio { width: 100%; height: 38px; margin-bottom: 20px; filter: invert(0.92) hue-rotate(180deg); }

  .canvas-wrap { position: relative; }
  canvas { width: 100%; height: 190px; display: block; border-radius: 12px; cursor: pointer; }
  #head { position: absolute; top: 0; bottom: 0; width: 1.5px; background: #fff;
          box-shadow: 0 0 12px rgba(255,255,255,0.85); pointer-events: none;
          transform: translateX(0); left: 0; }

  .legend { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 18px; }
  .chip {
    display: inline-flex; align-items: center; gap: 7px; padding: 7px 13px;
    border-radius: 999px; font-size: 12.5px; font-weight: 560;
    border: 1px solid var(--panel-line); background: rgba(255,255,255,0.045);
    cursor: pointer; transition: background .18s ease, transform .18s ease;
    color: var(--fg);
  }
  .chip:hover { background: rgba(255,255,255,0.11); transform: translateY(-1px); }
  .chip i { width: 8px; height: 8px; border-radius: 2px; display: inline-block; }
  .chip small { color: var(--muted); font-variant-numeric: tabular-nums; }

  #tip {
    position: fixed; pointer-events: none; opacity: 0; transform: translate(-50%, -118%);
    background: rgba(28,28,30,0.92); border: 1px solid rgba(255,255,255,0.14);
    border-radius: 12px; padding: 11px 14px; font-size: 12.5px; line-height: 1.5;
    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
    transition: opacity .14s ease; max-width: 300px; z-index: 10;
    box-shadow: 0 12px 40px rgba(0,0,0,0.55);
  }
  #tip b { display: block; font-size: 13.5px; margin-bottom: 3px; letter-spacing: -0.01em; }
  #tip span { color: var(--muted); }

  table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
  th { text-align: left; font-weight: 560; color: var(--muted); font-size: 11.5px;
       letter-spacing: 0.08em; text-transform: uppercase; padding: 0 12px 12px 0; }
  td { padding: 11px 12px 11px 0; border-top: 1px solid rgba(255,255,255,0.07);
       font-variant-numeric: tabular-nums; vertical-align: top; }
  td.note { color: var(--muted); font-variant-numeric: normal; }
  .tag { display: inline-flex; align-items: center; gap: 6px; font-weight: 590; }
  .tag i { width: 8px; height: 8px; border-radius: 2px; }

  footer { margin-top: 48px; color: var(--muted); font-size: 13px; line-height: 1.7; }
  footer code { background: rgba(255,255,255,0.08); padding: 2px 7px; border-radius: 6px;
                font-size: 12px; }
  a { color: var(--accent-2); text-decoration: none; }
  @media (max-width: 640px) {
    header { padding: 56px 0 28px; }
    .wrap { padding: 0 16px 64px; }
    .panel { padding: 18px; }
  }
  @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
</head>
<body>
<div class="glow"></div>
<div class="wrap">
  <header>
    <p class="eyebrow"><span class="dot"></span> fourfloor house remix</p>
    <h1>__TITLE__</h1>
    <p class="sub">__BPM__ BPM &middot; <b>__KEY__</b> &middot; Camelot <b>__CAMELOT__</b>
       &middot; __LENGTH__ &middot; built from <b>__SOURCE__</b>__SHIFT__</p>
    <div class="stats" id="stats"></div>
  </header>

  <section class="panel">
    <h2>Player</h2>
    <audio id="audio" controls preload="metadata" src="__AUDIO__"></audio>
    <div class="canvas-wrap">
      <canvas id="wave"></canvas>
      <div id="head"></div>
    </div>
    <div class="legend" id="legend"></div>
  </section>

  <section class="panel">
    <h2>Arrangement</h2>
    <table>
      <thead><tr><th>Slot</th><th>Time</th><th>Bars</th><th>From source</th><th>Decision</th></tr></thead>
      <tbody id="rows"></tbody>
    </table>
  </section>

  <footer>
    <p>Session file: <code>__SESSION__</code> — exact BPM, first-downbeat offset, key,
       cue points and a per-bar energy curve, so a DJ app can beat-match and cue this
       remix without re-analysing it.</p>
    <p>Generated by <a href="https://github.com/neelbarm/fourfloor">fourfloor</a>.</p>
  </footer>
</div>

<div id="tip"></div>

<script>
const SESSION = __SESSION_JSON__;
const WAVE = __WAVE_JSON__;
const COLORS = { intro:'#2ee6d6', build:'#ffb340', drop:'#ff4d9d',
                 breakdown:'#7d8cff', outro:'#6e6e73', end:'#6e6e73' };

const fmt = s => { const m = Math.floor(s/60), r = Math.round(s%60);
                   return m + ':' + String(r).padStart(2,'0'); };

/* ---------- count-up stats ---------- */
const STATS = [
  { k:'BPM',    v:SESSION.bpm,                         d:2, cls:'accent' },
  { k:'Key',    t:SESSION.key + ' / ' + SESSION.camelot,     cls:'teal' },
  { k:'Bars',   v:SESSION.bars,                        d:0 },
  { k:'Drops',  v:SESSION.sections.filter(s=>s.kind==='drop').length, d:0 },
  { k:'Peak',   t:SESSION.loudness.peak_db.toFixed(1) + ' dB' },
  { k:'RMS',    t:SESSION.loudness.rms_db.toFixed(1) + ' dB' },
];
const statsEl = document.getElementById('stats');
STATS.forEach((s, i) => {
  const el = document.createElement('div');
  el.className = 'stat';
  el.innerHTML = `<div class="v ${s.cls||''}">${s.t !== undefined ? s.t : '0'}</div>
                  <div class="k">${s.k}</div>`;
  statsEl.appendChild(el);
  if (s.v === undefined) return;
  const out = el.querySelector('.v');
  const dur = 900, t0 = performance.now() + i * 70;
  const tick = now => {
    const p = Math.min(1, Math.max(0, (now - t0) / dur));
    const e = 1 - Math.pow(1 - p, 3);
    out.textContent = (s.v * e).toFixed(s.d);
    if (p < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
});

/* ---------- waveform + section map ---------- */
const cv = document.getElementById('wave');
const ctx = cv.getContext('2d');
const total = SESSION.duration;
let W = 0, H = 0;

function resize() {
  const dpr = window.devicePixelRatio || 1;
  W = cv.clientWidth; H = cv.clientHeight;
  cv.width = W * dpr; cv.height = H * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

function draw() {
  ctx.clearRect(0, 0, W, H);
  const mid = H * 0.56, half = H * 0.42;

  /* section bands behind the waveform */
  SESSION.sections.forEach(s => {
    const x0 = s.start / total * W, x1 = s.end / total * W;
    const c = COLORS[s.kind] || '#6e6e73';
    const g = ctx.createLinearGradient(0, 0, 0, H);
    g.addColorStop(0, c + '30'); g.addColorStop(1, c + '06');
    ctx.fillStyle = g;
    ctx.fillRect(x0, 0, x1 - x0, H);
    ctx.fillStyle = c + 'cc';
    ctx.fillRect(x0, H - 3, Math.max(1, x1 - x0 - 1), 3);
  });

  /* waveform */
  const n = WAVE.length;
  for (let i = 0; i < n; i++) {
    const x = i / n * W;
    const sec = i / n * total;
    const s = SESSION.sections.find(v => sec >= v.start && sec < v.end);
    ctx.fillStyle = (COLORS[s ? s.kind : 'outro'] || '#6e6e73') + 'e0';
    const h = Math.max(1, WAVE[i] * half);
    ctx.fillRect(x, mid - h, Math.max(1, W / n - 0.35), h * 1.35);
  }

  /* cue markers */
  ctx.font = '500 10px -apple-system, system-ui, sans-serif';
  SESSION.cues.forEach(c => {
    if (c.kind === 'end') return;
    const x = c.time / total * W;
    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.fillRect(x, 0, 1, H - 3);
    ctx.fillStyle = 'rgba(255,255,255,0.72)';
    ctx.fillText(c.name, x + 5, 13);
  });
}

/* ---------- playhead ---------- */
const audio = document.getElementById('audio');
const head = document.getElementById('head');
function frame() {
  const p = (audio.currentTime || 0) / total;
  head.style.transform = 'translateX(' + (p * W) + 'px)';
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

cv.addEventListener('click', e => {
  const r = cv.getBoundingClientRect();
  audio.currentTime = Math.max(0, Math.min(total, (e.clientX - r.left) / r.width * total));
});

/* ---------- hover tooltip ---------- */
const tip = document.getElementById('tip');
cv.addEventListener('mousemove', e => {
  const r = cv.getBoundingClientRect();
  const sec = (e.clientX - r.left) / r.width * total;
  const s = SESSION.sections.find(v => sec >= v.start && sec < v.end);
  if (!s) { tip.style.opacity = 0; return; }
  tip.innerHTML = `<b style="color:${COLORS[s.kind]||'#fff'}">${s.kind} &middot; ${s.bars} bars</b>
    <span>${fmt(s.start)}–${fmt(s.end)} &middot; from the source <b style="color:#f5f5f7">
    ${s.source_label}</b> at ${fmt(s.source_start)}<br>${s.note}</span>`;
  tip.style.left = e.clientX + 'px';
  tip.style.top = r.top + 'px';
  tip.style.opacity = 1;
});
cv.addEventListener('mouseleave', () => { tip.style.opacity = 0; });

/* ---------- legend + table ---------- */
const legend = document.getElementById('legend');
SESSION.cues.filter(c => c.kind !== 'end').forEach(c => {
  const b = document.createElement('button');
  b.className = 'chip';
  b.innerHTML = `<i style="background:${COLORS[c.kind]||'#6e6e73'}"></i>${c.name}
                 <small>${fmt(c.time)}</small>`;
  b.addEventListener('click', () => { audio.currentTime = c.time; audio.play(); });
  legend.appendChild(b);
});

const rows = document.getElementById('rows');
SESSION.sections.forEach(s => {
  const tr = document.createElement('tr');
  tr.innerHTML = `<td><span class="tag"><i style="background:${COLORS[s.kind]||'#6e6e73'}"></i>
                  ${s.kind}</span></td>
                  <td>${fmt(s.start)}</td><td>${s.bars}</td>
                  <td>${s.source_label} @ ${fmt(s.source_start)}</td>
                  <td class="note">${s.note}</td>`;
  rows.appendChild(tr);
});

window.addEventListener('resize', resize);
resize();
</script>
</body>
</html>
"""


def _audio_src(audio_path: Path, out_path: Path) -> str:
    """URL for the ``<audio src>``, relative to wherever the page is written.

    The page is normally written beside the mp3, but ``preview -o`` can put it
    anywhere, and a bare filename would then point at nothing. Percent-encoding
    matters too: a ``#`` in a song title truncates the URL at the fragment and
    the player silently loads nothing.
    """
    try:
        rel = os.path.relpath(audio_path.resolve(), out_path.resolve().parent)
    except (OSError, ValueError):
        rel = audio_path.name
    return quote(Path(rel).as_posix())


def write_preview(audio_path: str | Path, session: dict, out: str | Path | None = None) -> Path:
    """Write ``preview.html`` beside ``audio_path`` and return its path."""
    audio_path = Path(audio_path)
    out_path = Path(out) if out else audio_path.parent / "preview.html"
    wave = waveform(audio_path)

    title = audio_path.stem.replace(".house", "").replace("_", " ").replace("-", " ")
    title = title.strip() or "House remix"
    shift = session.get("semitone_shift", 0)
    shift_txt = f" &middot; shifted <b>{shift:+d}</b> semitones" if shift else ""
    sess_name = f"{audio_path.stem}.session.json"

    html = (_HTML
            .replace("__TITLE__", _esc(title))
            .replace("__BPM__", f"{session['bpm']:.2f}")
            .replace("__KEY__", _esc(str(session["key"])))
            .replace("__CAMELOT__", _esc(str(session["camelot"])))
            .replace("__LENGTH__", _fmt(session["duration"]))
            .replace("__SOURCE__", _esc(str(session.get("source", {}).get("file", "the source"))))
            .replace("__SHIFT__", shift_txt)
            .replace("__AUDIO__", _esc(_audio_src(audio_path, out_path)))
            .replace("__SESSION_JSON__", json.dumps(session))
            .replace("__WAVE_JSON__", json.dumps(wave))
            .replace("__SESSION__", _esc(sess_name)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf8")
    return out_path


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def _fmt(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    return f"{m}:{s:02d}"
