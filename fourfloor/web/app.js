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

const state = {
  config: null, source: null, match: null, screen: 'drop', detail: null,
  job: null, es: null, key: 'keep', keyMode: 'keep', swingDirty: false,
  raf: 0, playerRaf: 0, revealT0: 0, reveal: 1,
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
};

function placeCard(screen) {
  const card = $('#sourceCard');
  const spec = SLOTS[screen];
  if (!spec || !state.source) { card.hidden = true; return; }
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
  if (name === 'result') startPlayer();
  window.scrollTo({ top: 0, behavior: REDUCED ? 'auto' : 'smooth' });
}

function go(name) {
  if (name === state.screen) { placeCard(name); return Promise.resolve(); }
  if (!REDUCED && document.startViewTransition) {
    try {
      return document.startViewTransition(() => applyScreen(name)).finished
        .catch(() => {});
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
  loadLibrary();
}

init();
})();
