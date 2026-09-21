/* fourfloor — the local app.
 *
 * Five screens as one state machine (drop → analysing → controls → remixing →
 * result), the source card as a single node that moves between them so the
 * View Transitions API can morph it, and one EventSource per job.
 *
 * No framework, no bundle, no network beyond this origin.
 */
(() => {
'use strict';

const $ = (sel, root = document) => root.querySelector(sel);
const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const COLORS = {
  intro: '#2ee6d6', build: '#ffb340', drop: '#ff4d9d', breakdown: '#7d8cff',
  outro: '#6e6e73', end: '#6e6e73',
  hook: '#ff4d9d', verse: '#7d8cff', section: '#8b8b93',
};
const PHASE_LABELS = {
  analyse: 'Analyse', warp: 'Warp to the grid', separate: 'Separate',
  arrange: 'Arrange', render: 'Render', write: 'Write the files',
};

/* Camelot: pitch class → wheel number, as fourfloor's key module has it. */
const MAJOR = { 0: 8, 1: 3, 2: 10, 3: 5, 4: 12, 5: 7, 6: 2, 7: 9, 8: 4, 9: 11, 10: 6, 11: 1 };
const MINOR = { 9: 8, 10: 3, 11: 10, 0: 5, 1: 12, 2: 7, 3: 2, 4: 9, 5: 4, 6: 11, 7: 6, 8: 1 };
const NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const CODE_TO_KEY = {};
for (const [pc, n] of Object.entries(MAJOR)) CODE_TO_KEY[n + 'B'] = { pc: +pc, minor: false };
for (const [pc, n] of Object.entries(MINOR)) CODE_TO_KEY[n + 'A'] = { pc: +pc, minor: true };
const keyName = code => {
  const k = CODE_TO_KEY[code];
  return k ? NAMES[k.pc] + (k.minor ? 'm' : '') : code;
};
const neighbours = code => {
  const n = parseInt(code, 10), l = code.slice(-1), other = l === 'A' ? 'B' : 'A';
  return [code, ((n % 12) + 1) + l, (((n - 2 + 12) % 12) + 1) + l, n + other];
};

/* One colour per feedback category, so a pin on the waveform, a row in the
 * list and a chip in the popover are all obviously the same thing. */
const CAT_COLORS = {
  'off-beat': '#ffb340', 'vocal-buried': '#7d8cff', 'drums-fake': '#ff4d9d',
  'clash': '#ff6b6b', 'transition': '#c78bff', 'too-loud': '#ff8a3d',
  'too-quiet': '#5ac8fa', 'boring': '#8b8b93', 'good': '#2ee6d6',
};

const state = {
  config: null, source: null, match: null, screen: 'drop', detail: null,
  job: null, es: null, key: 'keep', keyMode: 'keep', swingDirty: false,
  raf: 0, playerRaf: 0, revealT0: 0, reveal: 1,
  feedback: null, pending: null,
  ab: { a: null, b: null, side: 'a', raf: 0, fixedAt: 0, drift: 0, voted: '',
        ctx: null, gain: null, noCtx: false, drawnW: 0 },
};

/* ── small helpers ───────────────────────────────────────────────────────── */

const fmt = s => {
  if (!isFinite(s)) return '0:00';
  const m = Math.floor(s / 60), r = Math.floor(s % 60);
  return m + ':' + String(r).padStart(2, '0');
};
const esc = s => String(s == null ? '' : s)
  .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
  .replace(/"/g, '&quot;');

async function api(path, opts) {
  const res = await fetch(path, opts);
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch (e) { data = null; }
  if (!res.ok) throw new Error((data && data.error) || `${res.status} ${res.statusText}`);
  return data;
}

function countUp(el, to, decimals = 0, suffix = '') {
  const from = 0, dur = REDUCED ? 0 : 900, t0 = performance.now();
  const set = v => { el.textContent = v.toFixed(decimals) + suffix; };
  if (!dur) return set(to);
  const tick = now => {
    const p = Math.min(1, (now - t0) / dur);
    set(from + (to - from) * (1 - Math.pow(1 - p, 3)));
    if (p < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function tweenInput(input, to, decimals = 1) {
  const from = parseFloat(input.value);
  if (!isFinite(from) || REDUCED || Math.abs(to - from) < 0.01) {
    input.value = String(to); return;
  }
  const t0 = performance.now(), dur = 420;
  const tick = now => {
    const p = Math.min(1, (now - t0) / dur);
    const v = from + (to - from) * (1 - Math.pow(1 - p, 3));
    input.value = p < 1 ? v.toFixed(decimals) : String(to);
    if (p < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function showAlert(el, message) {
  if (!message) { el.hidden = true; el.textContent = ''; return; }
  el.hidden = false;
  el.textContent = message;
}

/* ── screens ─────────────────────────────────────────────────────────────── */

const SLOTS = {
  analysing: ['#slot-analysing', false], controls: ['#slot-controls', false],
  remixing: ['#slot-remixing', true], result: ['#slot-result', true],
  compare: [null, true],
};

function placeCard(screen) {
  const card = $('#sourceCard');
  const spec = SLOTS[screen];
  if (!spec || !spec[0] || !state.source) { card.hidden = true; return; }
  const host = $(spec[0]);
  card.hidden = false;
  card.classList.toggle('compact', spec[1]);
  if (card.parentElement !== host) host.appendChild(card);
}

function applyScreen(name) {
  state.screen = name;
  document.documentElement.dataset.screen = name;
  document.querySelectorAll('.screen').forEach(s => {
    s.classList.toggle('on', s.dataset.screen === name);
  });
  placeCard(name);
  refreshSegments();
  if (name === 'controls' || name === 'result') drawSourceWave(1);
  // the canvas has no width until its screen is displayed, so the player loop
  // can only start once this screen is the one on show
  if (name === 'result') { startPlayer(); renderPins(); }
  else closeMarkPop();
  if (name === 'compare') startAb(); else stopAb();
  window.scrollTo({ top: 0, behavior: REDUCED ? 'auto' : 'smooth' });
}

function go(name) {
  if (name === state.screen) { placeCard(name); return Promise.resolve(); }
  if (!REDUCED && document.startViewTransition) {
    try {
      const vt = document.startViewTransition(() => applyScreen(name));
      // starting a screen change while one is still running aborts it, which
      // rejects `ready`; nobody awaits that, so it would surface as an error
      if (vt.ready) vt.ready.catch(() => {});
      return vt.finished.catch(() => {});
    } catch (e) { /* fall through */ }
  }
  const stage = $('#stage');
  stage.classList.add('swap-out');
  return new Promise(resolve => {
    setTimeout(() => {
      stage.classList.remove('swap-out');
      applyScreen(name);
      stage.classList.add('swap-in');
      setTimeout(() => stage.classList.remove('swap-in'), 440);
      resolve();
    }, REDUCED ? 0 : 190);
  });
}

/* ── segmented controls ──────────────────────────────────────────────────── */

const segments = [];

function segment(root, items, value, onPick) {
  root.querySelectorAll('button').forEach(b => b.remove());
  items.forEach(item => {
    const b = document.createElement('button');
    b.type = 'button';
    b.dataset.value = String(item.value);
    b.textContent = item.label;
    if (item.disabled) { b.disabled = true; }
    if (item.title) b.title = item.title;
    b.addEventListener('click', () => { setSegment(root, item.value); onPick(item.value); });
    root.appendChild(b);
  });
  if (!segments.includes(root)) segments.push(root);
  setSegment(root, value);
}

function setSegment(root, value) {
  const thumb = $('.thumb', root);
  let target = null;
  root.querySelectorAll('button').forEach(b => {
    const on = b.dataset.value === String(value);
    b.setAttribute('aria-pressed', String(on));
    if (on) target = b;
  });
  if (!thumb) return;
  if (target && target.offsetWidth) {
    thumb.style.width = target.offsetWidth + 'px';
    thumb.style.transform = `translate3d(${target.offsetLeft}px,0,0)`;
    thumb.style.opacity = '1';
  } else {
    thumb.style.opacity = '0';
  }
}

function refreshSegments() {
  requestAnimationFrame(() => {
    segments.forEach(root => {
      const on = root.querySelector('button[aria-pressed="true"]');
      if (on) setSegment(root, on.dataset.value);
    });
  });
}
window.addEventListener('resize', refreshSegments);

/* ── canvas: the source structure timeline ───────────────────────────────── */

function sizeCanvas(cv) {
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (!w || !h) return null;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
  }
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function drawSourceWave(reveal) {
  const cv = $('#srcWave');
  const c = sizeCanvas(cv);
  if (!c) return;
  const { ctx, w, h } = c;
  const src = state.source;
  if (!src) { drawScanning(ctx, w, h); return; }
  const total = src.analysis.duration || 1;
  const sections = src.analysis.sections || [];
  const wave = src.wave || [];
  const cut = w * Math.max(0, Math.min(1, reveal));

  sections.forEach(s => {
    const x0 = s.start / total * w, x1 = s.end / total * w;
    if (x0 > cut) return;
    const col = COLORS[s.label] || '#8b8b93';
    const g = ctx.createLinearGradient(0, 0, 0, h);
    g.addColorStop(0, col + '2e'); g.addColorStop(1, col + '06');
    ctx.fillStyle = g;
    ctx.fillRect(x0, 0, Math.min(x1, cut) - x0, h);
    ctx.fillStyle = col + 'cc';
    ctx.fillRect(x0, h - 2.5, Math.max(1, Math.min(x1, cut) - x0 - 1), 2.5);
  });

  const n = wave.length || 1;
  const mid = h * 0.52, half = h * 0.36;
  for (let i = 0; i < wave.length; i++) {
    const x = i / n * w;
    if (x > cut) break;
    const sec = i / n * total;
    const s = sections.find(v => sec >= v.start && sec < v.end);
    ctx.fillStyle = (COLORS[s ? s.label : 'section'] || '#8b8b93') + 'd8';
    const bar = Math.max(1, wave[i] * half);
    ctx.fillRect(x, mid - bar, Math.max(1, w / n - 0.4), bar * 2);
  }
}

let scanPhase = 0;
function drawScanning(ctx, w, h) {
  const mid = h * 0.54;
  ctx.strokeStyle = 'rgba(255,255,255,0.14)';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, mid); ctx.lineTo(w, mid); ctx.stroke();
  const bars = 120;
  for (let i = 0; i < bars; i++) {
    const x = i / bars * w;
    const d = Math.abs(((scanPhase + i / bars) % 1) - 0.5);
    const a = Math.max(0, 1 - d * 5);
    const amp = (0.12 + 0.8 * a) * h * 0.32 * (0.5 + 0.5 * Math.sin(i * 1.7));
    ctx.fillStyle = `rgba(46,230,214,${0.12 + 0.5 * a})`;
    ctx.fillRect(x, mid - amp, Math.max(1, w / bars - 1), amp * 2);
  }
}

function startScanning() {
  if (REDUCED) { const c = sizeCanvas($('#srcWave')); if (c) drawScanning(c.ctx, c.w, c.h); return; }
  cancelAnimationFrame(state.raf);
  const loop = () => {
    if (state.source || state.screen !== 'analysing') return;
    scanPhase = (scanPhase + 0.008) % 1;
    const c = sizeCanvas($('#srcWave'));
    if (c) drawScanning(c.ctx, c.w, c.h);
    state.raf = requestAnimationFrame(loop);
  };
  state.raf = requestAnimationFrame(loop);
}

function revealSourceWave() {
  if (REDUCED) { drawSourceWave(1); return; }
  const t0 = performance.now(), dur = 900;
  const tick = now => {
    const p = Math.min(1, (now - t0) / dur);
    drawSourceWave(1 - Math.pow(1 - p, 3));
    if (p < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

/* ── the source card ─────────────────────────────────────────────────────── */

function renderSourcePlaceholder(name) {
  $('#srcName').textContent = name;
  $('#srcSub').textContent = 'reading…';
  $('#srcTargetWrap').hidden = true;
  ['#srcBpm', '#srcKey', '#srcCamelot', '#srcLen'].forEach(s => { $(s).textContent = '—'; });
  $('#srcLegend').innerHTML = '';
  $('#sourceCard').hidden = false;
}

function renderSource(src) {
  const a = src.analysis;
  $('#srcName').textContent = src.name;
  $('#srcSub').textContent = `${a.sections.length} sections · ${a.tempo.beat_count} beats tracked`;
  $('#srcTargetWrap').hidden = false;
  $('#srcTarget').textContent = `${src.suggested_bpm.toFixed(0)} BPM`;
  countUp($('#srcBpm'), a.tempo.bpm, 2);
  $('#srcKey').textContent = a.key.key;
  $('#srcCamelot').textContent = a.key.camelot;
  $('#srcLen').textContent = fmt(a.duration);

  const seen = [];
  a.sections.forEach(s => { if (!seen.includes(s.label)) seen.push(s.label); });
  $('#srcLegend').innerHTML = seen.map(l =>
    `<span><i style="background:${COLORS[l] || '#8b8b93'}"></i>${esc(l)}</span>`).join('');
  revealSourceWave();
}

/* ── the Camelot wheel ───────────────────────────────────────────────────── */

const RAD = a => (a - 90) * Math.PI / 180;
function arcPath(cx, cy, r0, r1, a0, a1) {
  const p = (r, a) => [cx + r * Math.cos(RAD(a)), cy + r * Math.sin(RAD(a))];
  const [x0, y0] = p(r1, a0), [x1, y1] = p(r1, a1);
  const [x2, y2] = p(r0, a1), [x3, y3] = p(r0, a0);
  return `M${x0} ${y0}A${r1} ${r1} 0 0 1 ${x1} ${y1}L${x2} ${y2}A${r0} ${r0} 0 0 0 ${x3} ${y3}Z`;
}

function buildWheel() {
  const svg = $('#wheel');
  const ns = 'http://www.w3.org/2000/svg';
  svg.innerHTML = '';
  const rings = [[150, 114, 'B'], [110, 74, 'A']];
  rings.forEach(([r1, r0, letter]) => {
    for (let n = 1; n <= 12; n++) {
      const a0 = (n - 1) * 30 - 15, a1 = a0 + 30;
      const code = n + letter;
      const g = document.createElementNS(ns, 'g');
      g.setAttribute('class', 'seg-arc');
      g.dataset.code = code;
      const path = document.createElementNS(ns, 'path');
      path.setAttribute('d', arcPath(160, 160, r0, r1, a0 + 1.2, a1 - 1.2));
      path.setAttribute('stroke', 'rgba(255,255,255,0.10)');
      path.setAttribute('stroke-width', '1');
      g.appendChild(path);
      const mid = (a0 + a1) / 2, rm = (r0 + r1) / 2;
      const t = document.createElementNS(ns, 'text');
      t.setAttribute('x', 160 + rm * Math.cos(RAD(mid)));
      t.setAttribute('y', 160 + rm * Math.sin(RAD(mid)) + 3.6);
      t.setAttribute('text-anchor', 'middle');
      t.textContent = code;
      g.appendChild(t);
      g.addEventListener('click', () => pickKey(code));
      svg.appendChild(g);
    }
  });
  const ring = document.createElementNS(ns, 'path');
  ring.setAttribute('class', 'ring-sel');
  svg.appendChild(ring);

  const hub = document.createElementNS(ns, 'text');
  hub.setAttribute('class', 'hub'); hub.setAttribute('x', '160');
  hub.setAttribute('y', '158'); hub.setAttribute('text-anchor', 'middle');
  hub.id = 'wheelHub';
  svg.appendChild(hub);
  const sub = document.createElementNS(ns, 'text');
  sub.setAttribute('class', 'hub-sub'); sub.setAttribute('x', '160');
  sub.setAttribute('y', '175'); sub.setAttribute('text-anchor', 'middle');
  sub.id = 'wheelSub';
  svg.appendChild(sub);
}

function paintWheel() {
  const svg = $('#wheel');
  if (!svg || !state.source) return;
  const srcCode = state.source.analysis.key.camelot;
  const ok = neighbours(srcCode);
  const selected = state.keyMode === 'pick' ? state.key : null;

  svg.querySelectorAll('.seg-arc').forEach(g => {
    const code = g.dataset.code;
    const n = parseInt(code, 10);
    const compatible = ok.includes(code);
    const isSel = code === selected;
    g.classList.toggle('compatible', compatible && !isSel);
    g.classList.toggle('dimmed', !compatible && !isSel);
    const hue = (n - 1) * 30;
    const path = g.querySelector('path');
    path.setAttribute('fill', isSel
      ? `hsl(${hue} 85% 62% / 0.92)`
      : compatible ? `hsl(${hue} 72% 56% / 0.44)` : `hsl(${hue} 34% 50% / 0.24)`);
    path.setAttribute('stroke', isSel ? '#fff' : 'rgba(255,255,255,0.10)');
    if (isSel) {
      const ring = svg.querySelector('.ring-sel');
      const letter = code.slice(-1);
      const [r1, r0] = letter === 'B' ? [154, 110] : [114, 70];
      const a0 = (n - 1) * 30 - 15;
      ring.setAttribute('d', arcPath(160, 160, r0, r1, a0 + 0.6, a0 + 29.4));
      ring.classList.add('on');
    }
  });
  if (!selected) svg.querySelector('.ring-sel').classList.remove('on');

  const hub = $('#wheelHub'), sub = $('#wheelSub');
  if (selected) {
    const a = CODE_TO_KEY[srcCode], b = CODE_TO_KEY[selected];
    let shift = ((b.pc - a.pc + 18) % 12) - 6;
    hub.textContent = keyName(selected);
    sub.textContent = shift === 0 ? 'no shift' : `${shift > 0 ? '+' : ''}${shift} semitones`;
  } else {
    hub.textContent = keyName(srcCode);
    sub.textContent = 'source key';
  }
}

function pickKey(code) {
  state.key = code;
  state.keyMode = 'pick';
  state.match = null;
  paintWheel();
  updateKeyOut();
}

function updateKeyOut() {
  const out = $('#keyOut');
  if (state.keyMode === 'match' && state.match) {
    out.textContent = `match ${state.match.analysis.key.camelot}`;
  } else if (state.keyMode === 'match') {
    out.textContent = 'pick a track';
  } else if (state.keyMode === 'pick' && state.key && state.key !== 'keep') {
    out.textContent = `${keyName(state.key)} · ${state.key}`;
  } else {
    out.textContent = state.source
      ? `keep ${state.source.analysis.key.key}` : 'keep';
  }
}

/* ── upload ──────────────────────────────────────────────────────────────── */

function checkFile(file) {
  const cfg = state.config;
  const dot = file.name.lastIndexOf('.');
  const ext = dot < 0 ? '' : file.name.slice(dot).toLowerCase();
  if (!cfg.upload_exts.includes(ext)) {
    return `fourfloor reads mp3, m4a, wav, flac and aiff — ${
      ext ? esc(ext) + ' files' : 'that file'} it does not.`;
  }
  if (file.size > cfg.max_upload_mb * 1024 * 1024) {
    return `That file is ${(file.size / 1048576).toFixed(0)} MB. The limit is ${
      cfg.max_upload_mb} MB — trim it first.`;
  }
  if (!file.size) return 'That file is empty.';
  return null;
}

function upload(file, onProgress) {
  return new Promise((resolve, reject) => {
    const body = new FormData();
    body.append('file', file, file.name);
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/upload');
    xhr.upload.addEventListener('progress', e => {
      if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
    });
    xhr.addEventListener('load', () => {
      let data = null;
      try { data = JSON.parse(xhr.responseText); } catch (e) { data = null; }
      if (xhr.status >= 200 && xhr.status < 300 && data) resolve(data);
      else reject(new Error((data && data.error) || `upload failed (${xhr.status})`));
    });
    xhr.addEventListener('error', () => reject(new Error('the upload was interrupted')));
    xhr.send(body);
  });
}

async function takeFile(file) {
  const problem = checkFile(file);
  if (problem) { showAlert($('#dropErr'), problem); return; }
  showAlert($('#dropErr'), '');
  state.source = null;
  renderSourcePlaceholder(file.name);
  await go('analysing');
  startScanning();
  const lede = $('#s-analysing .lede');
  try {
    const meta = await upload(file, p => {
      lede.textContent = p < 1
        ? `Reading the file — ${Math.round(p * 100)}%`
        : 'Beat grid, key, chords and structure — a few seconds.';
    });
    state.source = meta;
    state.match = null;
    state.keyMode = 'keep';
    state.key = 'keep';
    renderSource(meta);
    prepareControls(meta);
    await go('controls');
  } catch (err) {
    showAlert($('#dropErr'), err.message);
    state.source = null;
    await go('drop');
  } finally {
    lede.textContent = 'Beat grid, key, chords and structure — a few seconds.';
  }
}

/* ── paste a link ────────────────────────────────────────────────────────── */

/* A fetch is an ordinary job on the same queue as the remixes, so it arrives on
 * the same SSE stream. This resolves with the source the server made, which is
 * exactly what /api/upload returns -- so a link and a drop rejoin here and the
 * rest of the app cannot tell them apart. */
function followJob(jobId, onEvent) {
  return new Promise((resolve, reject) => {
    const es = new EventSource(`/api/jobs/${jobId}/events`);
    let settled = false;
    const finish = (fn, value) => {
      if (settled) return;
      settled = true;
      es.close();
      fn(value);
    };
    es.onmessage = e => {
      let ev = null;
      try { ev = JSON.parse(e.data); } catch (err) { return; }   /* keep-alive */
      if (onEvent) onEvent(ev);
      if (ev.type === 'done') finish(resolve, ev.result);
      else if (ev.type === 'error') finish(reject, new Error(ev.message));
    };
    es.addEventListener('end', () =>
      finish(reject, new Error('the fetch stopped before it finished')));
    es.onerror = () => {
      // EventSource reconnects on its own; ask the job whether it is over.
      if (settled) return;
      fetch(`/api/jobs/${jobId}`).then(r => r.json()).then(j => {
        if (!j || !j.state) return;
        if (j.state === 'failed') finish(reject, new Error(j.error || 'the fetch failed'));
        else if (j.state === 'done') {
          const done = (j.events || []).find(e => e.type === 'done');
          if (done) finish(resolve, done.result);
        }
      }).catch(() => {});
    };
  });
}

function checkLink(url) {
  const raw = (url || '').trim();
  if (!raw) return 'Paste a link first.';
  if (!/^https?:\/\//i.test(raw)) return 'Links start with http:// or https://.';
  return null;
}

function hostOf(url) {
  try { return new URL(url).hostname.replace(/^www\./, ''); } catch (e) { return 'the link'; }
}

/* Start a fetch job and report it through `onStep(percent, note)`. */
async function fetchLink(url, onStep) {
  const started = await api('/api/fetch', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ url }),
  });
  return followJob(started.job, ev => {
    if (!onStep) return;
    if (ev.type === 'queued' && ev.position > 0) onStep(0, `waiting — ${ev.position} ahead`);
    else if (ev.type === 'phase' && ev.name === 'fetch') onStep(0, ev.detail || 'reading the link');
    else if (ev.type === 'phase' && ev.name === 'analyse') onStep(100, 'reading the track');
    else if (ev.type === 'progress') onStep(ev.percent, ev.detail || '');
  });
}

function setFetchBar(pct, note) {
  $('#fetchProgress').hidden = false;
  $('#fetchBar').style.width = Math.max(0, Math.min(100, pct)) + '%';
  $('#fetchNote').textContent = pct > 0 && pct < 100
    ? `${Math.round(pct)}% — ${note}` : note;
}

async function takeLink(url) {
  const problem = checkLink(url);
  if (problem) { showAlert($('#dropErr'), problem); return; }
  const btn = $('#linkGo');
  showAlert($('#dropErr'), '');
  btn.disabled = true;
  state.source = null;
  renderSourcePlaceholder(hostOf(url));
  $('#srcSub').textContent = 'fetching…';
  await go('analysing');
  startScanning();
  const lede = $('#s-analysing .lede');
  lede.textContent = 'Downloading the audio, then reading it.';
  setFetchBar(0, 'reading the link');
  try {
    const meta = await fetchLink(url, (pct, note) => {
      setFetchBar(pct, note);
      if (note) $('#srcName').textContent = note.length > 48 ? note.slice(0, 47) + '…' : note;
    });
    state.source = meta;
    state.match = null;
    state.keyMode = 'keep';
    state.key = 'keep';
    $('#linkInput').value = '';
    renderSource(meta);
    prepareControls(meta);
    await go('controls');
  } catch (err) {
    showAlert($('#dropErr'), err.message);
    state.source = null;
    await go('drop');
  } finally {
    btn.disabled = false;
    $('#fetchProgress').hidden = true;
    $('#fetchBar').style.width = '0%';
    lede.textContent = 'Beat grid, key, chords and structure — a few seconds.';
  }
}

/* ── controls ────────────────────────────────────────────────────────────── */

function prepareControls(src) {
  const cfg = state.config;
  const suggested = src.suggested_bpm;

  segment($('#bpmSeg'), cfg.bpm_presets.map(v => ({ value: v, label: String(v) })),
    suggested, v => { tweenInput($('#bpmInput'), v); countUp($('#bpmOut'), v, 1, ''); syncBpmOut(v); });
  $('#bpmInput').value = String(suggested);
  syncBpmOut(suggested);

  segment($('#keyModeSeg'), [
    { value: 'keep', label: 'Keep the key' },
    { value: 'pick', label: 'Pick a key' },
    { value: 'match', label: 'Match a track' },
  ], 'keep', mode => {
    state.keyMode = mode;
    $('#keyWheelBody').hidden = mode !== 'pick';
    $('#keyMatchBody').hidden = mode !== 'match';
    if (mode === 'pick' && (!state.key || state.key === 'keep')) {
      state.key = src.analysis.key.camelot;
    }
    paintWheel();
    updateKeyOut();
    refreshSegments();
  });

  segment($('#stemsSeg'), [
    { value: 'hpss', label: 'HPSS' },
    {
      value: 'demucs', label: 'Demucs', disabled: !cfg.demucs,
      title: cfg.demucs ? 'four-way neural separation'
        : "demucs is not installed — pip install 'fourfloor[stems]'",
    },
  ], 'hpss', () => {});

  segment($('#formSeg'), cfg.forms.map(f => ({
    value: f, label: f[0].toUpperCase() + f.slice(1),
  })), 'club', () => {});

  segment($('#lenSeg'), [
    { value: '3:00', label: '3:00' }, { value: '4:30', label: '4:30' },
    { value: '6:00', label: '6:00' },
  ], '4:30', v => { $('#lengthInput').value = v; $('#lenOut').textContent = v; });

  const sel = $('#styleSelect');
  sel.innerHTML = '<option value="">fourfloor default</option>';
  cfg.styles.forEach(s => {
    const o = document.createElement('option');
    o.value = s.name;
    o.textContent = `${s.label}${s.bpm ? ` — ${Number(s.bpm).toFixed(0)} BPM` : ''}`;
    sel.appendChild(o);
  });

  buildWheel();
  paintWheel();
  updateKeyOut();
  $('#ctaNote').textContent =
    `${src.analysis.tempo.bpm.toFixed(2)} BPM → ${suggested.toFixed(0)} · ` +
    `${src.analysis.key.key} ${src.analysis.key.camelot} · takes about a minute`;
  showAlert($('#controlsErr'), '');
}

function syncBpmOut(v) {
  $('#bpmOut').innerHTML = `${Number(v).toFixed(1)}<i>BPM</i>`;
}

function readOptions() {
  const payload = {
    source: state.source.id,
    bpm: $('#bpmInput').value.trim(),
    stems: $('#stemsSeg button[aria-pressed="true"]').dataset.value,
    form: $('#formSeg button[aria-pressed="true"]').dataset.value,
    length: $('#lengthInput').value.trim(),
    style: $('#styleSelect').value || null,
  };
  if (state.keyMode === 'pick' && state.key && state.key !== 'keep') payload.key = state.key;
  if (state.keyMode === 'match' && state.match) payload.compatible_with = state.match.id;
  if (state.swingDirty) payload.swing = $('#swing').value;
  return payload;
}

/* ── the job ─────────────────────────────────────────────────────────────── */

function renderPhases() {
  const list = $('#phases');
  list.innerHTML = '';
  state.config.phases.forEach(name => {
    const li = document.createElement('li');
    li.className = 'phase';
    li.dataset.phase = name;
    li.innerHTML = `
      <span class="mark">
        <svg viewBox="0 0 20 20" aria-hidden="true">
          <circle class="ring" cx="10" cy="10" r="8"></circle>
          <circle class="arc" cx="10" cy="10" r="8"></circle>
          <path class="tick" d="M6 10.4l2.8 2.8L14.4 7.6"></path>
        </svg>
      </span>
      <span class="name">${esc(PHASE_LABELS[name] || name)}<span class="detail"></span></span>
      <span class="el"></span>`;
    list.appendChild(li);
  });
  $('#bar').style.width = '0%';
  document.documentElement.style.setProperty('--hue', '0deg');
}

function onJobEvent(ev) {
  const phases = state.config.phases;
  if (ev.type === 'queued') {
    $('#jobState').textContent = ev.position > 0
      ? `waiting — ${ev.position} ahead` : 'queued';
  } else if (ev.type === 'start') {
    $('#jobState').textContent = 'building';
  } else if (ev.type === 'phase') {
    const li = $(`.phase[data-phase="${ev.name}"]`);
    if (li) {
      li.classList.add('active');
      $('.detail', li).textContent = ev.detail || '';
    }
  } else if (ev.type === 'phase_done') {
    const li = $(`.phase[data-phase="${ev.name}"]`);
    if (li) {
      li.classList.remove('active');
      li.classList.add('done');
      $('.el', li).textContent = `${ev.elapsed.toFixed(1)}s`;
    }
    const done = document.querySelectorAll('.phase.done').length;
    $('#bar').style.width = Math.round(done / phases.length * 100) + '%';
    document.documentElement.style.setProperty('--hue', (done * 12) + 'deg');
  } else if (ev.type === 'error') {
    closeStream();
    $('#jobState').textContent = 'failed';
    showAlert($('#jobErr'), ev.message);
    $('#backFromJob').hidden = false;
    document.querySelectorAll('.phase.active').forEach(li => li.classList.remove('active'));
  } else if (ev.type === 'done') {
    closeStream();
    $('#bar').style.width = '100%';
    $('#jobState').textContent = 'done';
    openRemix(ev.result.id, ev.elapsed);
  }
}

function closeStream() {
  if (state.es) { state.es.close(); state.es = null; }
}

async function startRemix() {
  const btn = $('#remixBtn');
  btn.disabled = true;
  showAlert($('#controlsErr'), '');
  try {
    const res = await api('/api/remix', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(readOptions()),
    });
    state.job = res.job;
    $('#remixTitle').textContent = `Building ${res.title}.`;
    $('#jobErr').hidden = true;
    $('#backFromJob').hidden = true;
    renderPhases();
    await go('remixing');
    follow(res.job);
  } catch (err) {
    showAlert($('#controlsErr'), err.message);
  } finally {
    btn.disabled = false;
  }
}

function follow(jobId) {
  closeStream();
  const es = new EventSource(`/api/jobs/${jobId}/events`);
  state.es = es;
  es.onmessage = e => {
    try { onJobEvent(JSON.parse(e.data)); } catch (err) { /* keep-alive */ }
  };
  es.addEventListener('end', () => closeStream());
  es.onerror = () => {
    // EventSource retries on its own; only give up once the job is finished.
    if (!state.es) return;
    fetch(`/api/jobs/${jobId}`).then(r => r.json()).then(j => {
      if (j && (j.state === 'done' || j.state === 'failed')) closeStream();
    }).catch(() => {});
  };
}

/* ── result ──────────────────────────────────────────────────────────────── */

async function openRemix(id, elapsed) {
  const detail = await api(`/api/remixes/${id}`);
  state.detail = detail;
  if (detail.source && detail.source.analysis) {
    state.source = detail.source;
    renderSource(detail.source);
  }
  renderResult(detail, elapsed);
  await go('result');
  loadLibrary();
}

function renderResult(detail, elapsed) {
  const { meta, session } = detail;
  $('#resultTitle').textContent = meta.title + '.';
  $('#resultEyebrow').textContent = elapsed
    ? `done in ${Number(elapsed).toFixed(0)}s` : 'from the library';

  const stats = [
    { k: 'BPM', v: session.bpm, d: 2, cls: 'pink' },
    { k: 'Key', t: `${session.key} / ${session.camelot}`, cls: 'teal' },
    { k: 'Length', t: fmt(session.duration) },
    { k: 'Bars', v: session.bars, d: 0 },
    { k: 'Drops', v: session.sections.filter(s => s.kind === 'drop').length, d: 0 },
    { k: 'Peak', t: session.loudness.peak_db.toFixed(1) + ' dB' },
  ];
  const host = $('#stats');
  host.innerHTML = '';
  stats.forEach((s, i) => {
    const el = document.createElement('div');
    el.className = 'stat';
    el.style.animationDelay = (i * 55) + 'ms';
    el.innerHTML = `<div class="v ${s.cls || ''}">${s.t !== undefined ? esc(s.t) : '0'}</div>
                    <div class="k">${esc(s.k)}</div>`;
    host.appendChild(el);
    if (s.v !== undefined) countUp($('.v', el), s.v, s.d);
  });

  $('#playerMeta').innerHTML =
    `${esc(meta.form)} form · ${esc(meta.stems)} separation<br>` +
    `from ${esc(meta.source_name || '')}` +
    (meta.semitone_shift ? ` · ${meta.semitone_shift > 0 ? '+' : ''}${meta.semitone_shift} st` : '');

  const audio = $('#audio');
  audio.src = `/api/remixes/${meta.id}/remix.mp3`;
  document.body.classList.remove('playing');
  $('#tNow').textContent = '0:00';
  $('#tTotal').textContent = fmt(session.duration);

  const chips = $('#chips');
  chips.innerHTML = '';
  session.cues.filter(c => c.kind !== 'end').forEach((c, i) => {
    const b = document.createElement('button');
    b.className = 'chip';
    b.type = 'button';
    b.style.animationDelay = (i * 45) + 'ms';
    b.innerHTML = `<i style="background:${COLORS[c.kind] || '#6e6e73'}"></i>${esc(c.name)}
                   <small>${fmt(c.time)}</small>`;
    b.addEventListener('click', () => { audio.currentTime = c.time; audio.play(); });
    chips.appendChild(b);
  });

  const files = [
    ['remix.mp3', 'MP3', '320 kbps'],
    ['remix.wav', 'WAV', '44.1 kHz float'],
    ['remix.session.json', 'Session JSON', 'for your DJ app'],
    ['remix.plan.json', 'Plan JSON', 'every decision'],
  ].filter(([name]) => (meta.files || []).includes(name));
  $('#downloads').innerHTML = files.map(([name, label, sub]) =>
    `<a href="/api/remixes/${meta.id}/${name}?download=1" download>${esc(label)}
       <small>${esc(sub)}</small></a>`).join('');
  $('#sessionNote').textContent =
    `Exact tempo ${session.bpm} BPM, first downbeat at 0.000 s, ${session.cues.length} cues ` +
    `and a per-bar energy curve — everything a DJ app needs to sync into this without ` +
    `re-analysing it.`;

  $('#rows').innerHTML = session.sections.map(s => `
    <tr>
      <td><span class="tag"><i style="background:${COLORS[s.kind] || '#6e6e73'}"></i>${esc(s.kind)}</span></td>
      <td>${fmt(s.start)}</td><td>${s.bars}</td>
      <td>${esc(s.source_label)} @ ${fmt(s.source_start)}</td>
      <td class="note-cell">${esc(s.note)}</td>
    </tr>`).join('');

  const warn = (meta.warnings || []).concat(meta.note ? [meta.note] : []);
  $('#warnings').innerHTML = warn.map(w => `<div class="warn">${esc(w)}</div>`).join('');

  renderFeedback(detail);
}

function drawRemixWave(reveal) {
  const cv = $('#wave');
  const c = sizeCanvas(cv);
  if (!c || !state.detail) return;
  const { ctx, w, h } = c;
  const session = state.detail.session;
  const wave = state.detail.wave || [];
  const total = session.duration || 1;
  const cut = w * Math.max(0, Math.min(1, reveal));
  const mid = h * 0.54, half = h * 0.35;

  session.sections.forEach(s => {
    const x0 = s.start / total * w, x1 = Math.min(s.end / total * w, cut);
    if (x1 <= x0) return;
    const col = COLORS[s.kind] || '#6e6e73';
    const g = ctx.createLinearGradient(0, 0, 0, h);
    g.addColorStop(0, col + '30'); g.addColorStop(1, col + '06');
    ctx.fillStyle = g;
    ctx.fillRect(x0, 0, x1 - x0, h);
    ctx.fillStyle = col + 'cc';
    ctx.fillRect(x0, h - 3, Math.max(1, x1 - x0 - 1), 3);
  });

  const n = wave.length || 1;
  for (let i = 0; i < wave.length; i++) {
    const x = i / n * w;
    if (x > cut) break;
    const sec = i / n * total;
    const s = session.sections.find(v => sec >= v.start && sec < v.end);
    ctx.fillStyle = (COLORS[s ? s.kind : 'outro'] || '#6e6e73') + 'e0';
    const bar = Math.max(1, wave[i] * half);
    ctx.fillRect(x, mid - bar, Math.max(1, w / n - 0.35), bar * 2);
  }

  ctx.font = '500 10px -apple-system, system-ui, sans-serif';
  session.cues.forEach(c2 => {
    if (c2.kind === 'end') return;
    const x = c2.time / total * w;
    if (x > cut) return;
    ctx.fillStyle = 'rgba(255,255,255,0.45)';
    ctx.fillRect(x, 0, 1, h - 3);
    ctx.fillStyle = 'rgba(255,255,255,0.72)';
    ctx.fillText(c2.name, x + 5, 13);
  });
}

function startPlayer() {
  cancelAnimationFrame(state.playerRaf);
  state.reveal = 0;
  state.revealT0 = performance.now();
  state.playerRaf = requestAnimationFrame(playerFrame);
}

function playerFrame(now) {
  if (state.screen !== 'result' || !state.detail) { state.playerRaf = 0; return; }
  if (state.reveal < 1) {
    const p = REDUCED ? 1 : Math.min(1, (now - state.revealT0) / 1100);
    state.reveal = 1 - Math.pow(1 - p, 3);
    if (p >= 1) state.reveal = 1;
    drawRemixWave(state.reveal);
  }
  const audio = $('#audio');
  const total = state.detail ? state.detail.session.duration : 0;
  const w = $('#wave').clientWidth;
  const p = total ? (audio.currentTime || 0) / total : 0;
  $('#head').style.transform = `translate3d(${p * w}px,0,0)`;
  $('#tNow').textContent = fmt(audio.currentTime || 0);
  state.playerRaf = requestAnimationFrame(playerFrame);
}

/* ── listening notes ─────────────────────────────────────────────────────── *
 *
 * One person on this project can hear, so the point of all of this is to cost
 * him as little as possible: a key, a chip, a sentence. Nothing is asked twice
 * and nothing waits for a Save button it does not need — a star posts itself.
 */

/* A category off a chip has a label; a category another tool invented has only
 * its own slug, so tidy it rather than showing "drums-fake" next to "artifact"
 * in two different house styles. */
const catLabel = code => {
  const cats = (state.config && state.config.feedback_categories) || [];
  const hit = cats.find(c => c.code === code);
  if (hit) return hit.label;
  const words = String(code || 'unknown').replace(/[-_]+/g, ' ').trim();
  return words.charAt(0).toUpperCase() + words.slice(1);
};
const catColor = code => CAT_COLORS[code] || '#8b8b93';

/* The ears are not the only thing writing into this file any more: the critic
 * appends what Gemini heard, with author "gemini". A machine's opinion is
 * worth showing next to a person's and worth never being mistaken for one, so
 * anything that is not Neel gets a dashed pin and its name on the row. */
const ME = 'neel';
const authorLabel = slug => {
  const known = { neel: 'Neel', gemini: 'Gemini' };
  const s = slug || ME;
  return known[s] || s.replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
};

async function postFeedback(body) {
  const id = state.detail && state.detail.meta.id;
  if (!id) throw new Error('no remix is open');
  const data = await api(`/api/remixes/${id}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  state.feedback = data;
  return data;
}

function markers() {
  return (state.feedback && state.feedback.markers) || [];
}

/* -- pins on the waveform ------------------------------------------------- */

function renderPins() {
  const host = $('#pins');
  if (!host) return;
  host.innerHTML = '';
  if (!state.detail) return;
  const total = state.detail.session.duration || 1;
  /* The tooltip is absolutely positioned, which means it still counts towards
   * the page's scroll width even while it is invisible -- on a phone a pin
   * past halfway was widening the whole document by its own width. So each
   * one is measured and pinned inside the waveform rather than hung off its
   * marker and hoped for. */
  const wrapW = $('#waveWrap').clientWidth || 0;
  const tipW = wrapW ? Math.min(210, Math.max(120, wrapW - 24)) : 210;

  markers().forEach(m => {
    const p = Math.max(0, Math.min(1, m.time / total));
    const who = m.author || ME;
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'pin' + (who === ME ? '' : ' other');
    b.style.left = (p * 100) + '%';
    b.style.setProperty('--pin', catColor(m.category));
    if (wrapW) {
      const x = p * wrapW;                       // where the pin's stem lands
      const want = Math.max(4, Math.min(x + 12, wrapW - tipW - 4));
      b.style.setProperty('--tip-w', tipW + 'px');
      b.style.setProperty('--tip-left', (want - (x - 7)) + 'px');
    }
    const where = m.bar ? `bar ${m.bar}` : fmt(m.time);
    const by = who === ME ? '' : `${authorLabel(who)}: `;
    b.setAttribute('aria-label',
      `${by}${catLabel(m.category)} at ${fmt(m.time)}, ${where}. ` +
      `${m.note || 'no note'}. Play from here.`);
    b.innerHTML = `<i></i><span class="pin-tip">
      <b>${esc(catLabel(m.category))}</b>
      <small>${esc(where)} · ${esc(fmt(m.time))}${m.slot ? ' · ' + esc(m.slot) : ''}</small>
      ${m.note ? '<div>' + esc(m.note) + '</div>' : ''}
      ${who === ME ? '' : '<em class="pin-by">heard by ' + esc(authorLabel(who)) + '</em>'}
      </span>`;
    b.addEventListener('click', e => {
      e.stopPropagation();
      const audio = $('#audio');
      audio.currentTime = m.time;
      audio.play().catch(() => {});
    });
    host.appendChild(b);
  });
}

function renderMarkList() {
  const host = $('#marks');
  const rows = markers().slice().sort((a, b) => a.time - b.time);
  host.innerHTML = '';

  // say whose the dashed ones are, but only once somebody else has listened
  const others = [...new Set(rows.map(m => m.author || ME))].filter(a => a !== ME);
  const note = $('#earsOther');
  note.hidden = others.length === 0;
  note.textContent = others.length
    ? ` The dashed pins are not yours — ${others.map(authorLabel).join(' and ')} ` +
      `listened too, and the critic files what it heard in the same place.`
    : '';

  rows.forEach((m, i) => {
    const who = m.author || ME;
    const li = document.createElement('li');
    li.className = who === ME ? '' : 'other';
    li.style.animationDelay = Math.min(i, 8) * 40 + 'ms';
    li.innerHTML = `<i style="${who === ME
        ? 'background:' + catColor(m.category)
        : 'border:1.5px dashed ' + catColor(m.category)}"></i>
      <span class="m-cat">${esc(catLabel(m.category))}${who === ME ? ''
        : '<em class="m-by">' + esc(authorLabel(who)) + '</em>'}</span>
      <span class="m-where">${m.bar ? 'bar ' + m.bar : fmt(m.time)}${
        m.slot ? ' · ' + esc(m.slot) : ''}</span>
      <span class="m-note">${esc(m.note || '')}</span>`;
    const go = document.createElement('button');
    go.type = 'button';
    go.className = 'm-go';
    go.textContent = fmt(m.time);
    go.setAttribute('aria-label', `Play from ${fmt(m.time)}`);
    go.addEventListener('click', () => {
      const audio = $('#audio');
      audio.currentTime = m.time;
      audio.play().catch(() => {});
    });
    li.appendChild(go);
    host.appendChild(li);
  });
}

/* -- the popover ---------------------------------------------------------- */

function buildChips() {
  const host = $('#popChips');
  if (host.children.length) return;
  (state.config.feedback_categories || []).forEach(c => {
    const b = document.createElement('button');
    b.type = 'button';
    b.dataset.code = c.code;
    b.textContent = c.label;
    b.setAttribute('role', 'radio');
    b.setAttribute('aria-checked', 'false');
    b.style.setProperty('--cat', catColor(c.code));
    b.addEventListener('click', () => pickCategory(c.code));
    host.appendChild(b);
  });
}

function pickCategory(code) {
  if (!state.pending) return;
  state.pending.category = code;
  $('#popChips').querySelectorAll('button').forEach(b => {
    b.setAttribute('aria-checked', String(b.dataset.code === code));
  });
}

function openMarkPop(time) {
  if (!state.detail) return;
  buildChips();
  const total = state.detail.session.duration || 1;
  const t = Math.max(0, Math.min(total, time));
  state.pending = { time: t, category: null };
  const pop = $('#markPop');
  const wrap = $('#waveWrap');
  const w = wrap.clientWidth || 1;
  const x = t / total * w;
  // hang it under the playhead, but never off the edge of the panel
  pop.style.left = Math.max(0, Math.min(w - Math.min(304, w), x - 140)) + 'px';
  $('#popAt').textContent = fmt(t);
  $('#popNote').value = '';
  pickCategory('off-beat');
  pop.hidden = false;
  $('#markBtn').classList.add('armed');
  // it hangs below the waveform, which on a short window puts it under the
  // fold -- and a note you have to go looking for is a note that never gets
  // written
  const box = pop.getBoundingClientRect();
  if (box.bottom > window.innerHeight - 12 || box.top < 0) {
    pop.scrollIntoView({ behavior: REDUCED ? 'auto' : 'smooth', block: 'center' });
  }
  const first = $('#popChips').querySelector('button');
  if (first) first.focus({ preventScroll: true });
}

function closeMarkPop() {
  const pop = $('#markPop');
  if (!pop || pop.hidden) return;
  pop.hidden = true;
  state.pending = null;
  $('#markBtn').classList.remove('armed');
}

async function saveMark() {
  if (!state.pending) return;
  const body = {
    marker: {
      time: state.pending.time,
      category: state.pending.category || 'off-beat',
      note: $('#popNote').value,
    },
  };
  const btn = $('#popSave');
  btn.disabled = true;
  try {
    await postFeedback(body);
    closeMarkPop();
    renderPins();
    renderMarkList();
    flashEars('marker saved');
  } catch (err) {
    $('#popNote').value = err.message;
  } finally {
    btn.disabled = false;
  }
}

/* -- the rating ----------------------------------------------------------- */

const STAR = 'M12 2.6l2.9 5.9 6.5.9-4.7 4.6 1.1 6.5L12 17.4 6.2 20.5l1.1-6.5' +
             'L2.6 9.4l6.5-.9z';

function buildStars() {
  const host = $('#stars');
  if (host.children.length) return;
  for (let n = 1; n <= 5; n++) {
    const b = document.createElement('button');
    b.type = 'button';
    b.dataset.stars = String(n);
    b.setAttribute('role', 'radio');
    b.setAttribute('aria-checked', 'false');
    b.setAttribute('aria-label', `${n} out of 5`);
    b.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="${STAR}" fill="currentColor"/></svg>`;
    b.addEventListener('click', () => saveRating(n));
    host.appendChild(b);
  }
}

function paintRating() {
  buildStars();
  const rating = (state.feedback && state.feedback.rating) || {};
  const stars = rating.stars || 0;
  $('#stars').querySelectorAll('button').forEach(b => {
    const on = Number(b.dataset.stars) <= stars;
    b.classList.toggle('lit', on);
    b.setAttribute('aria-checked', String(Number(b.dataset.stars) === stars));
  });
  $('#verdict').value = rating.verdict || '';
  const state_el = $('#ratingState');
  state_el.classList.remove('saved');
  state_el.textContent = stars ? `${stars} of 5` : 'not rated yet';
}

function flashEars(message) {
  const el = $('#ratingState');
  el.textContent = message;
  el.classList.add('saved');
  clearTimeout(el._t);
  el._t = setTimeout(paintRating, 1900);
}

async function saveRating(stars) {
  try {
    await postFeedback({ stars });
    paintRating();
    flashEars('saved');
  } catch (err) {
    flashEars(err.message);
  }
}

async function saveVerdict() {
  const value = $('#verdict').value.trim();
  const had = ((state.feedback && state.feedback.rating) || {}).verdict || '';
  if (!value || value === had) return;
  try {
    await postFeedback({ verdict: value });
    paintRating();
    flashEars('verdict saved');
  } catch (err) {
    flashEars(err.message);
  }
}

function renderFeedback(detail) {
  state.feedback = detail.feedback || { markers: [], votes: [], rating: {} };
  paintRating();
  renderMarkList();
  renderPins();
  closeMarkPop();

  const pick = $('#abPick');
  const partners = detail.partners || [];
  pick.innerHTML = '';
  partners.forEach(p => {
    const o = document.createElement('option');
    o.value = p.kind + ':' + p.id;
    o.textContent = `${p.label}${p.sub ? ' — ' + p.sub : ''}`;
    pick.appendChild(o);
  });
  $('.ab-open').hidden = partners.length === 0;
}

/* ── A/B ─────────────────────────────────────────────────────────────────── *
 *
 * Two <audio> elements play at once and one of them is muted, so flipping is a
 * property change rather than a seek: no gap, no restart, and the ear is still
 * in the same bar when the other take arrives. The muted one is nudged back
 * onto the audible one's clock a few times a second -- decoders drift, and a
 * comparison you cannot trust to the bar is not a comparison.
 */

/* Two decoders started a moment apart do not stay together, and a seek to
 * pull one back is itself slow enough to overshoot -- correcting that way
 * oscillates around ±50 ms, which is a sixteenth note at 128 and plainly
 * audible on a flip. So the muted side is nudged instead: a fraction of a
 * percent of playback rate, inaudible because it is muted, closing the gap
 * over about a second. A hard seek is kept for the case a rate cannot fix. */
const DRIFT_NUDGE = 0.005;         // 5 ms: below this, leave it alone
const DRIFT_SEEK = 0.25;           // 250 ms: too far to walk back
const MAX_NUDGE = 0.012;           // ±1.2% of rate, time-stretched not pitched
const FIX_EVERY = 250;             // ms between corrections
/* currentTime is read off the audio clock and is jittery by a render quantum
 * either way, so the raw difference is noisy. Correcting on the raw number
 * makes the loop chase its own noise and ring; correcting on a smoothed one
 * settles. */
const DRIFT_SMOOTH = 0.12;

/* Two renders of the same track have the same title, which makes "A · lofi 7"
 * against "B · lofi 7" useless. When the names collide, name each side by the
 * first thing that actually differs. */
function abTag(side, other) {
  if (!other || side.label !== other.label) return side.label;
  return (side.sub || '').split(' · ')[0] || side.label;
}

function abSideOf(which) { return which === 'a' ? state.ab.a : state.ab.b; }
function abEl(which) { return which === 'a' ? $('#audioA') : $('#audioB'); }

function selfSide() {
  const { meta, session, wave } = state.detail;
  return {
    kind: 'remix', id: meta.id, label: meta.title,
    sub: `${Number(session.bpm).toFixed(0)} BPM · ${session.camelot} · ${fmt(session.duration)}`,
    url: `/api/remixes/${meta.id}/remix.mp3`,
    wave: wave || [], duration: session.duration,
    sections: (session.sections || []).map(s => ({ kind: s.kind, start: s.start, end: s.end })),
  };
}

async function resolveSide(p) {
  const side = {
    kind: p.kind, id: p.id, label: p.label, sub: p.sub, url: p.url,
    wave: [], duration: 0, sections: [],
  };
  if (p.kind === 'remix') {
    const d = await api(`/api/remixes/${p.id}`);
    side.wave = d.wave || [];
    side.duration = d.session.duration;
    side.sections = (d.session.sections || [])
      .map(s => ({ kind: s.kind, start: s.start, end: s.end }));
    side.sub = `${Number(d.session.bpm).toFixed(0)} BPM · ${d.session.camelot} · ` +
               fmt(d.session.duration);
  } else if (p.kind === 'source') {
    const src = state.detail && state.detail.source;
    if (src && src.analysis) {
      side.wave = src.wave || [];
      side.duration = src.analysis.duration;
      side.sections = (src.analysis.sections || [])
        .map(s => ({ kind: s.label, start: s.start, end: s.end }));
      side.sub = `${src.analysis.tempo.bpm.toFixed(1)} BPM · ` +
                 `${src.analysis.key.camelot} · the track you dropped in`;
    }
  }
  return side;                    // a reference file has no waveform to draw
}

function drawLane(cv, side, headEl) {
  const c = sizeCanvas(cv);
  if (!c || !side) return;
  const { ctx, w, h } = c;
  const total = side.duration || 1;
  const mid = h * 0.54, half = h * 0.35;

  (side.sections || []).forEach(s => {
    const x0 = s.start / total * w, x1 = s.end / total * w;
    const col = COLORS[s.kind] || '#6e6e73';
    const g = ctx.createLinearGradient(0, 0, 0, h);
    g.addColorStop(0, col + '2c'); g.addColorStop(1, col + '06');
    ctx.fillStyle = g;
    ctx.fillRect(x0, 0, Math.max(1, x1 - x0), h);
    ctx.fillStyle = col + 'bb';
    ctx.fillRect(x0, h - 2.5, Math.max(1, x1 - x0 - 1), 2.5);
  });

  const wave = side.wave || [];
  if (!wave.length) {
    ctx.fillStyle = 'rgba(255,255,255,0.10)';
    ctx.fillRect(0, mid - 1, w, 2);
    ctx.fillStyle = 'rgba(255,255,255,0.40)';
    ctx.font = '500 11px -apple-system, system-ui, sans-serif';
    ctx.fillText('played straight from disk — no waveform for this one', 10, mid - 12);
    if (headEl) headEl.hidden = false;
    return;
  }
  const n = wave.length;
  for (let i = 0; i < n; i++) {
    const x = i / n * w;
    const sec = i / n * total;
    const s = (side.sections || []).find(v => sec >= v.start && sec < v.end);
    ctx.fillStyle = (COLORS[s ? s.kind : 'outro'] || '#6e6e73') + 'e0';
    const bar = Math.max(1, wave[i] * half);
    ctx.fillRect(x, mid - bar, Math.max(1, w / n - 0.35), bar * 2);
  }
}

function drawLanes() {
  drawLane($('#waveA'), state.ab.a, $('#headA'));
  drawLane($('#waveB'), state.ab.b, $('#headB'));
}

/* Both elements feed one AudioContext and the flip is a gain change inside it.
 *
 * `muted`, and `volume = 0` too, silence an element by changing what its
 * output path is doing, and Chrome then reports its position off a different
 * clock -- the pair appear to step 50-80 ms apart at every flip although the
 * audio has not moved. A sync loop that believes that reading will pull the
 * real audio apart chasing it. Through one graph both elements always render,
 * both report against the same clock, and the flip is four milliseconds of
 * gain ramp: no gap, and no click either.
 */
function abGraph() {
  if (state.ab.gain || state.ab.noCtx) return state.ab.gain;
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!Ctx) { state.ab.noCtx = true; return null; }
  try {
    const ctx = new Ctx();
    const tap = el => {
      const g = ctx.createGain();
      ctx.createMediaElementSource(el).connect(g);
      g.connect(ctx.destination);
      return g;
    };
    state.ab.ctx = ctx;
    state.ab.gain = { a: tap($('#audioA')), b: tap($('#audioB')) };
  } catch (e) {
    state.ab.noCtx = true;                 // no graph: fall back to volume
  }
  return state.ab.gain;
}

const FLIP_RAMP = 0.004;                   // 4 ms, short enough to feel instant

/* An AudioContext may only start from a gesture, and openCompare awaits before
 * it plays -- so the click handlers call this while the gesture is still on
 * the stack. */
function abArm() {
  abGraph();
  if (state.ab.ctx && state.ab.ctx.state === 'suspended') {
    state.ab.ctx.resume().catch(() => {});
  }
}

function abFlip(which) {
  state.ab.side = which;
  const gain = abGraph();
  if (gain) {
    const t = state.ab.ctx.currentTime;
    gain.a.gain.setTargetAtTime(which === 'a' ? 1 : 0, t, FLIP_RAMP);
    gain.b.gain.setTargetAtTime(which === 'b' ? 1 : 0, t, FLIP_RAMP);
  } else {
    $('#audioA').volume = which === 'a' ? 1 : 0;
    $('#audioB').volume = which === 'b' ? 1 : 0;
  }
  $('#laneA').classList.toggle('live', which === 'a');
  $('#laneB').classList.toggle('live', which === 'b');
  setSegment($('#abSeg'), which);
}

function abPlay() {
  const A = $('#audioA'), B = $('#audioB');
  abArm();
  document.body.classList.add('ab-playing');
  [A, B].forEach(el => {
    const p = el.play();
    if (p && p.catch) p.catch(() => document.body.classList.remove('ab-playing'));
  });
}

function abPause() {
  $('#audioA').pause();
  $('#audioB').pause();
  document.body.classList.remove('ab-playing');
}

function abToggle() {
  if ($('#audioA').paused) abPlay(); else abPause();
}

function abSeek(t) {
  [$('#audioA'), $('#audioB')].forEach(el => {
    const d = isFinite(el.duration) ? el.duration : t;
    el.currentTime = Math.max(0, Math.min(d, t));
    el.playbackRate = 1;
  });
  state.ab.fixedAt = performance.now();
  state.ab.drift = 0;
}

function abFrame(now) {
  if (state.screen !== 'compare') { state.ab.raf = 0; return; }
  /* A is the clock and B chases it, whichever one you happen to be hearing.
   * Correcting "the muted one" instead would mean the roles -- and the sign
   * of the error -- swap on every flip, and a controller whose feedback term
   * inverts under it is a controller that kicks the pair apart each time. */
  const lead = $('#audioA'), follow = $('#audioB');
  const raw = (follow.currentTime || 0) - (lead.currentTime || 0);
  const drift = state.ab.drift + (raw - state.ab.drift) * DRIFT_SMOOTH;
  state.ab.drift = drift;

  if (!lead.paused && now - state.ab.fixedAt > FIX_EVERY &&
      isFinite(follow.duration)) {
    state.ab.fixedAt = now;
    if (Math.abs(drift) > DRIFT_SEEK) {
      follow.currentTime = Math.min(lead.currentTime, follow.duration);
      follow.playbackRate = 1;
      state.ab.drift = 0;
    } else if (Math.abs(drift) > DRIFT_NUDGE) {
      const rate = 1 - Math.max(-MAX_NUDGE, Math.min(MAX_NUDGE, drift * 1.2));
      if (Math.abs(follow.playbackRate - rate) > 0.0015) follow.playbackRate = rate;
    } else if (follow.playbackRate !== 1) {
      follow.playbackRate = 1;
    }
  }

  const ms = Math.abs(raw) * 1000;
  const el = $('#abDrift');
  const text = ms < 8 ? 'in sync' : `${ms.toFixed(0)} ms apart`;
  if (el.textContent !== text) el.textContent = text;
  el.classList.toggle('off', ms > 50);

  // the lanes are first drawn inside a view transition, where the canvas may
  // not have a width yet; redraw once it does, and whenever it changes
  const w = $('#waveA').clientWidth;
  if (w && w !== state.ab.drawnW) { state.ab.drawnW = w; drawLanes(); }

  ['a', 'b'].forEach(which => {
    const side = abSideOf(which);
    const audio = abEl(which);
    const head = which === 'a' ? $('#headA') : $('#headB');
    const cv = which === 'a' ? $('#waveA') : $('#waveB');
    const total = (side && side.duration) || audio.duration || 0;
    const p = total ? (audio.currentTime || 0) / total : 0;
    head.style.transform = `translate3d(${p * cv.clientWidth}px,0,0)`;
  });
  $('#abNow').textContent = fmt(lead.currentTime || 0);
  state.ab.raf = requestAnimationFrame(abFrame);
}

function startAb() {
  cancelAnimationFrame(state.ab.raf);
  state.ab.drawnW = 0;
  drawLanes();
  state.ab.raf = requestAnimationFrame(abFrame);
}

function stopAb() {
  cancelAnimationFrame(state.ab.raf);
  state.ab.raf = 0;
  if ($('#audioA').src || $('#audioB').src) abPause();
}

async function openCompare(partner) {
  if (!state.detail) return;
  const a = selfSide();
  let b;
  try {
    b = await resolveSide(partner);
  } catch (err) {
    showAlert($('#controlsErr'), err.message);
    return;
  }
  $('#audio').pause();
  state.ab.a = a;
  state.ab.b = b;
  state.ab.voted = '';

  $('#aName').textContent = a.label;
  $('#aSub').textContent = a.sub || '';
  $('#bName').textContent = b.label;
  $('#bSub').textContent = b.sub || '';
  $('#cmpTitle').textContent = b.kind === 'source'
    ? 'Your remix against the original.'
    : b.kind === 'reference' ? 'Your remix against a real one.'
      : 'Two takes, one transport.';
  $('#abTotal').textContent = fmt(Math.max(a.duration || 0, b.duration || 0));
  $('#abNow').textContent = '0:00';
  $('#voteReason').value = '';
  $('#voteState').textContent = '';
  $('#voteA').classList.remove('won');
  $('#voteB').classList.remove('won');
  $('#voteA').textContent = 'Prefer A';
  $('#voteB').textContent = 'Prefer B';
  showAlert($('#cmpErr'), '');

  segment($('#abSeg'), [{ value: 'a', label: 'A · ' + abTag(a, b) },
                        { value: 'b', label: 'B · ' + abTag(b, a) }], 'a', abFlip);

  const A = $('#audioA'), B = $('#audioB');
  [A, B].forEach(el => {
    // the sync nudges playback rate; time-stretch it rather than detune it,
    // because a flip can land on an element mid-correction
    el.preservesPitch = true;
    el.mozPreservesPitch = true;
    el.webkitPreservesPitch = true;
    el.playbackRate = 1;
  });
  A.src = a.url;
  B.src = b.url;
  A.currentTime = 0;
  B.currentTime = 0;
  state.ab.fixedAt = 0;
  state.ab.drift = 0;
  abFlip('a');
  await go('compare');
  abPlay();
}

async function saveVote(prefer) {
  const b = state.ab.b;
  if (!b || !state.detail) return;
  const btn = prefer === 'a' ? $('#voteA') : $('#voteB');
  btn.disabled = true;
  try {
    await postFeedback({
      vote: {
        other: b.id, prefer, reason: $('#voteReason').value,
        label: state.ab.a.label, other_label: b.label,
      },
    });
    state.ab.voted = prefer;
    $('#voteA').classList.toggle('won', prefer === 'a');
    $('#voteB').classList.toggle('won', prefer === 'b');
    $('#voteState').textContent = 'saved — you prefer ' +
      (prefer === 'a' ? 'A · ' + abTag(state.ab.a, b)
                      : 'B · ' + abTag(b, state.ab.a));
    renderMarkList();
  } catch (err) {
    showAlert($('#cmpErr'), err.message);
  } finally {
    btn.disabled = false;
  }
}

/* ── library ─────────────────────────────────────────────────────────────── */

async function loadLibrary() {
  let rows = [];
  try { rows = (await api('/api/remixes')).remixes || []; } catch (e) { return; }
  $('#libCount').textContent = String(rows.length);
  $('#libEmpty').hidden = rows.length > 0;
  const list = $('#libList');
  list.innerHTML = '';
  rows.forEach((r, i) => {
    const li = document.createElement('li');
    li.className = 'lib-row';
    li.style.animationDelay = Math.min(i, 8) * 45 + 'ms';
    const item = document.createElement('button');
    item.type = 'button';
    item.className = 'lib-item' +
      (state.detail && state.detail.meta.id === r.id ? ' current' : '');
    item.innerHTML = `<b>${esc(r.title)}</b>
      <small>${Number(r.bpm).toFixed(0)} BPM · ${esc(r.camelot)} ${esc(r.key)} · ${esc(r.length)}</small>`;
    item.addEventListener('click', () => openRemix(r.id));

    // straight into A/B against whatever is open, which is the comparison
    // you actually want: this take against the one you just built
    const ab = document.createElement('button');
    ab.type = 'button';
    ab.className = 'lib-ab';
    ab.textContent = '⇄';
    const open = state.detail && state.detail.meta.id;
    ab.title = open && open !== r.id
      ? `Compare ${r.title} against the remix you have open`
      : 'Open this remix, then compare it';
    ab.setAttribute('aria-label', ab.title);
    ab.addEventListener('click', () => {
      const current = state.detail && state.detail.meta.id;
      if (!current || current === r.id) { openRemix(r.id); return; }
      abArm();
      openCompare({
        kind: 'remix', id: r.id, label: r.title,
        sub: `${Number(r.bpm).toFixed(0)} BPM · ${r.camelot} · ${r.length}`,
        url: `/api/remixes/${r.id}/remix.mp3`,
      });
    });

    const del = document.createElement('button');
    del.type = 'button';
    del.className = 'lib-del';
    del.title = 'Delete this remix';
    del.setAttribute('aria-label', `Delete ${r.title}`);
    del.textContent = '×';
    let armed = false, timer = 0;
    del.addEventListener('click', async () => {
      if (!armed) {
        armed = true;
        del.textContent = '?';
        del.style.color = '#ffd9e6';
        timer = setTimeout(() => { armed = false; del.textContent = '×'; del.style.color = ''; }, 2600);
        return;
      }
      clearTimeout(timer);
      li.classList.add('going');
      try { await api(`/api/remixes/${r.id}`, { method: 'DELETE' }); } catch (e) { /* gone */ }
      setTimeout(loadLibrary, 340);
      if (state.detail && state.detail.meta.id === r.id) {
        state.detail = null;
        $('#audio').pause();
      }
    });
    li.appendChild(item);
    li.appendChild(ab);
    li.appendChild(del);
    list.appendChild(li);
  });
}

/* ── wiring ──────────────────────────────────────────────────────────────── */

function wireDropZone() {
  const input = $('#file');
  $('#orb').addEventListener('click', () => input.click());
  input.addEventListener('change', () => {
    if (input.files && input.files[0]) takeFile(input.files[0]);
    input.value = '';
  });

  const link = $('#linkInput');
  const go = () => takeLink(link.value);
  $('#linkGo').addEventListener('click', go);
  link.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); go(); } });
  link.addEventListener('input', () => showAlert($('#dropErr'), ''));

  let depth = 0;
  const over = e => {
    if (!e.dataTransfer || !Array.from(e.dataTransfer.types || []).includes('Files')) return;
    e.preventDefault();
    depth++;
    document.body.classList.add('dragging');
  };
  window.addEventListener('dragenter', over);
  window.addEventListener('dragover', e => {
    if (document.body.classList.contains('dragging')) e.preventDefault();
  });
  window.addEventListener('dragleave', () => {
    depth = Math.max(0, depth - 1);
    if (!depth) document.body.classList.remove('dragging');
  });
  window.addEventListener('drop', e => {
    if (!e.dataTransfer || !e.dataTransfer.files.length) return;
    e.preventDefault();
    depth = 0;
    document.body.classList.remove('dragging');
    const orb = $('#orb');
    orb.classList.remove('rippling');
    void orb.offsetWidth;
    orb.classList.add('rippling');
    takeFile(e.dataTransfer.files[0]);
  });
}

function wireControls() {
  const bpm = $('#bpmInput');
  bpm.addEventListener('input', () => {
    const v = parseFloat(bpm.value);
    if (isFinite(v)) { syncBpmOut(v); setSegment($('#bpmSeg'), v); }
  });
  $('#bpmSuggest').addEventListener('click', () => {
    if (!state.source) return;
    const v = state.source.suggested_bpm;
    tweenInput(bpm, v);
    syncBpmOut(v);
    setSegment($('#bpmSeg'), v);
  });
  $('#lengthInput').addEventListener('input', () => {
    const v = $('#lengthInput').value.trim();
    $('#lenOut').textContent = v || '—';
    setSegment($('#lenSeg'), v);
  });
  $('#swing').addEventListener('input', () => {
    state.swingDirty = true;
    $('#swingOut').textContent = 'swing ' + Number($('#swing').value).toFixed(2);
  });

  const matchInput = $('#matchFile');
  $('#matchPick').addEventListener('click', () => matchInput.click());
  matchInput.addEventListener('change', async () => {
    const file = matchInput.files && matchInput.files[0];
    matchInput.value = '';
    if (!file) return;
    const problem = checkFile(file);
    if (problem) { showAlert($('#controlsErr'), problem); return; }
    $('#matchLabel').textContent = 'reading ' + file.name + '…';
    try {
      const meta = await upload(file);
      state.match = meta;
      $('#matchLabel').textContent = meta.name;
      $('#matchMeta').textContent =
        `${meta.analysis.key.key} ${meta.analysis.key.camelot} · ` +
        `${meta.analysis.tempo.bpm.toFixed(1)} BPM — the remix is shifted to mix with it`;
      updateKeyOut();
    } catch (err) {
      showAlert($('#controlsErr'), err.message);
      $('#matchLabel').textContent = 'Choose a track to mix with…';
    }
  });

  const matchLink = $('#matchLinkInput'), matchGo = $('#matchLinkGo');
  const takeMatchLink = async () => {
    const url = matchLink.value;
    const problem = checkLink(url);
    if (problem) { showAlert($('#controlsErr'), problem); return; }
    showAlert($('#controlsErr'), '');
    matchGo.disabled = true;
    $('#matchLabel').textContent = 'fetching ' + hostOf(url) + '…';
    try {
      const meta = await fetchLink(url, (pct, note) => {
        $('#matchMeta').textContent = pct > 0 && pct < 100
          ? `${Math.round(pct)}% — ${note}` : (note || 'reading it');
      });
      state.match = meta;
      matchLink.value = '';
      $('#matchLabel').textContent = meta.name;
      $('#matchMeta').textContent =
        `${meta.analysis.key.key} ${meta.analysis.key.camelot} · ` +
        `${meta.analysis.tempo.bpm.toFixed(1)} BPM — the remix is shifted to mix with it`;
      updateKeyOut();
    } catch (err) {
      showAlert($('#controlsErr'), err.message);
      $('#matchLabel').textContent = 'Choose a track to mix with…';
      $('#matchMeta').textContent = 'its key is read the same way';
    } finally {
      matchGo.disabled = false;
    }
  };
  matchGo.addEventListener('click', takeMatchLink);
  matchLink.addEventListener('keydown', e => {
    if (e.key === 'Enter') { e.preventDefault(); takeMatchLink(); }
  });

  $('#remixBtn').addEventListener('click', startRemix);
  $('#backFromJob').addEventListener('click', () => { closeStream(); go('controls'); });
  $('#againBtn').addEventListener('click', () => {
    $('#audio').pause();
    if (!state.source) return go('drop');
    go('controls');
  });
  $('#newTrackBtn').addEventListener('click', () => {
    $('#audio').pause();
    state.source = null;
    state.detail = null;
    go('drop');
  });
}

function wirePlayer() {
  const audio = $('#audio');
  const play = $('#play');
  play.addEventListener('click', () => {
    if (audio.paused) audio.play(); else audio.pause();
  });
  audio.addEventListener('play', () => document.body.classList.add('playing'));
  audio.addEventListener('pause', () => document.body.classList.remove('playing'));
  audio.addEventListener('ended', () => document.body.classList.remove('playing'));
  const seek = e => {
    if (!state.detail) return;
    const r = $('#wave').getBoundingClientRect();
    const total = state.detail.session.duration;
    audio.currentTime = Math.max(0, Math.min(total, (e.clientX - r.left) / r.width * total));
  };
  $('#wave').addEventListener('click', seek);
  window.addEventListener('keydown', e => {
    if (e.code !== 'Space' || state.screen !== 'result') return;
    const tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'select' || tag === 'textarea' || tag === 'button') return;
    e.preventDefault();
    if (audio.paused) audio.play(); else audio.pause();
  });
  window.addEventListener('resize', () => {
    drawRemixWave(state.reveal);
    drawSourceWave(1);
    renderPins();            // the tooltips are clamped to a measured width
  });
}

const TYPING = el => {
  const tag = ((el && el.tagName) || '').toLowerCase();
  return tag === 'input' || tag === 'select' || tag === 'textarea';
};

function wireFeedback() {
  $('#markBtn').addEventListener('click', () => {
    if (!$('#markPop').hidden) { closeMarkPop(); return; }
    openMarkPop($('#audio').currentTime || 0);
  });
  $('#popClose').addEventListener('click', closeMarkPop);
  $('#popSave').addEventListener('click', saveMark);
  $('#popNote').addEventListener('keydown', e => {
    if (e.key === 'Enter') { e.preventDefault(); saveMark(); }
  });

  const verdict = $('#verdict');
  verdict.addEventListener('change', saveVerdict);
  verdict.addEventListener('keydown', e => {
    if (e.key === 'Enter') { e.preventDefault(); verdict.blur(); }
  });

  window.addEventListener('keydown', e => {
    if (e.key === 'Escape' && !$('#markPop').hidden) { closeMarkPop(); return; }
    if (state.screen !== 'result' || e.metaKey || e.ctrlKey || e.altKey) return;
    if (TYPING(e.target)) return;
    if (e.key === 'm' || e.key === 'M') {
      e.preventDefault();
      if ($('#markPop').hidden) openMarkPop($('#audio').currentTime || 0);
      else closeMarkPop();
    }
  });

  $('#abGo').addEventListener('click', () => {
    abArm();
    const value = $('#abPick').value;
    const partner = (state.detail.partners || [])
      .find(p => p.kind + ':' + p.id === value);
    if (partner) openCompare(partner);
  });
}

function wireCompare() {
  $('#abPlay').addEventListener('click', abToggle);
  $('#cmpBack').addEventListener('click', () => { abPause(); go('result'); });
  $('#voteA').addEventListener('click', () => saveVote('a'));
  $('#voteB').addEventListener('click', () => saveVote('b'));

  [['#waveA', 'a'], ['#waveB', 'b']].forEach(([sel, which]) => {
    $(sel).addEventListener('click', e => {
      const side = abSideOf(which);
      const total = (side && side.duration) || abEl(which).duration || 0;
      if (!total) return;
      const r = $(sel).getBoundingClientRect();
      abSeek((e.clientX - r.left) / r.width * total);
    });
  });

  ['#audioA', '#audioB'].forEach(sel => {
    $(sel).addEventListener('ended', () => {
      if (abEl(state.ab.side) === $(sel)) abPause();
    });
  });

  window.addEventListener('keydown', e => {
    if (state.screen !== 'compare' || e.metaKey || e.ctrlKey || e.altKey) return;
    if (TYPING(e.target)) return;
    if (e.key === 'Tab') {
      // Tab is the flip, but only from outside the controls: once you have
      // tabbed onto a button, Tab has to keep moving focus or the panel is a trap
      const tag = ((e.target && e.target.tagName) || '').toLowerCase();
      if (tag === 'button' || tag === 'a') return;
      e.preventDefault();
      abFlip(state.ab.side === 'a' ? 'b' : 'a');
    } else if (e.code === 'Space') {
      e.preventDefault();
      abToggle();
    }
  });

  window.addEventListener('resize', () => {
    if (state.screen === 'compare') drawLanes();
  });
}

function wireLibrary() {
  const lib = $('#library'), btn = $('#libToggle');
  btn.addEventListener('click', () => {
    const open = lib.classList.toggle('open');
    btn.setAttribute('aria-expanded', String(open));
    if (open) lib.scrollIntoView({ behavior: REDUCED ? 'auto' : 'smooth', block: 'start' });
  });
}

async function init() {
  wireDropZone();
  wireControls();
  wirePlayer();
  wireFeedback();
  wireCompare();
  wireLibrary();
  applyScreen('drop');
  try {
    state.config = await api('/api/config');
  } catch (err) {
    showAlert($('#dropErr'), 'the fourfloor server is not answering — is it still running?');
    return;
  }
  $('#maxMb').textContent = String(state.config.max_upload_mb);
  $('#libHome').textContent = state.config.home;
  if (state.config.fetch === false) {
    // no yt-dlp on this machine: say so once, rather than failing on submit
    $('#linkInput').disabled = true;
    $('#linkGo').disabled = true;
    $('#matchLinkRow').hidden = true;
    $('#linkHelp').textContent =
      'links need yt-dlp on this machine — install it with `brew install yt-dlp`';
  }
  loadLibrary();
}

init();
})();
