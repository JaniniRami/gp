/* ProactMAD live-replay UI: PSG traces + fire_now + MAD controller + ESP. */

const HOLD = 0, ADVANCE = 1, RETRACT = 2;
const RETRACTED = 0, ADVANCING = 1, ADVANCED = 2, RETRACTING = 3;
const POS_NAME = ["RETRACTED", "ADVANCING", "ADVANCED", "RETRACTING"];

class MadController {
  constructor(opt = {}) {
    this.advanceSec = opt.advanceSec ?? 10;
    this.retractSec = opt.retractSec ?? 10;
    this.refractorySec = opt.refractorySec ?? 60;
    this.quietRetractSec = opt.quietRetractSec ?? 90;
    this.threshold = opt.threshold ?? 0.55;
    this.reset();
  }
  reset() {
    this.position = RETRACTED;
    this.posTimer = 0;
    this.lastAdvanceT = -1e9;
    this.quietTimer = 0;
    this.t = 0;
    this.nAdvances = 0;
    this.advancedMask = [];
    this.actions = [];
  }
  step(t, prob, wake = false) {
    const dt = this.advancedMask.length ? Math.max(0, t - this.t) : 1;
    this.t = t;
    let action = HOLD;
    if (this.position === ADVANCING) {
      this.posTimer += dt;
      if (this.posTimer >= this.advanceSec) {
        this.position = ADVANCED;
        this.posTimer = 0;
      }
    } else if (this.position === RETRACTING) {
      this.posTimer += dt;
      if (this.posTimer >= this.retractSec) {
        this.position = RETRACTED;
        this.posTimer = 0;
      }
    }
    const highRisk = !wake && prob >= this.threshold;
    if (highRisk) this.quietTimer = 0;
    else this.quietTimer += dt;
    const canAdvance =
      this.position === RETRACTED &&
      t - this.lastAdvanceT >= this.refractorySec &&
      !wake;
    if (canAdvance && highRisk) {
      this.position = ADVANCING;
      this.posTimer = 0;
      this.lastAdvanceT = t;
      this.nAdvances += 1;
      action = ADVANCE;
    } else if (this.position === ADVANCED && this.quietTimer >= this.quietRetractSec) {
      this.position = RETRACTING;
      this.posTimer = 0;
      action = RETRACT;
    }
    const isAdv = this.position === ADVANCED || this.position === ADVANCING;
    this.advancedMask.push(isAdv);
    this.actions.push(action);
    return action;
  }
  simulate(probs, wake) {
    this.reset();
    for (let t = 0; t < probs.length; t++) this.step(t, probs[t], !!wake[t]);
    return {
      advanced: this.advancedMask.slice(),
      actions: this.actions.slice(),
      nAdvances: this.nAdvances,
    };
  }
}

const $ = (id) => document.getElementById(id);

function fmt(t) {
  t = Math.max(0, t);
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function eventCovered(ev, advanced, A = 10, lead = 30) {
  if (!ev.arousal_linked) return false;
  const deadlineRef = ev.arousal_start != null ? ev.arousal_start : ev.end;
  const deadline = deadlineRef - A;
  const earliest = ev.start - lead;
  const i0 = Math.max(0, Math.floor(earliest));
  const i1 = Math.min(advanced.length, Math.floor(deadline) + 1);
  for (let i = i0; i < i1; i++) if (advanced[i]) return true;
  return false;
}

function setupCanvas(c) {
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  const r = c.getBoundingClientRect();
  c.width = Math.max(1, Math.floor(r.width * dpr));
  c.height = Math.max(1, Math.floor(r.height * dpr));
  const ctx = c.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w: r.width, h: r.height };
}

function drawGrid(ctx, w, h, color = "#182033") {
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x < w; x += 40) {
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
  }
  for (let y = 0; y < h; y += 28) {
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
  }
  ctx.stroke();
}

let pack = null;
let ctrl = new MadController();
let sim = { advanced: [], actions: [], nAdvances: 0 };
let controllerProbs = [];
let tNow = 0;
let playing = false;
let lastTs = 0;
let winSec = 60;
let speed = 8;
let lastActionSent = -1;
let layoutMode = "stack";

const PRE_ROLL_SEC = 60; // seek this far before the event we jump to

// ESP link state. "open" = serial port open, "live" = the board answered us.
const esp = {
  mode: "none", // none | webserial | server
  state: "none", // none | open | live | err
  port: null,
  writer: null,
  lastTx: null,
  lastRx: null,
  lastRxAtMs: 0,
  rxCount: 0,
  note: "",
};

// Robust per-signal display ranges, computed once from the whole clip so the
// vertical scale never jumps while the trace scrolls.
const NORM = { pres: { lo: -1, hi: 1 }, spo2: { lo: 85, hi: 100 } };

function quantile(sorted, q) {
  if (!sorted.length) return 0;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

function robustRange(values, qLo = 0.005, qHi = 0.995, padFrac = 0.08) {
  const sorted = Float64Array.from(values).sort();
  let lo = quantile(sorted, qLo);
  let hi = quantile(sorted, qHi);
  if (!(hi > lo)) {
    const c = Number.isFinite(lo) ? lo : 0;
    lo = c - 1;
    hi = c + 1;
  }
  const pad = (hi - lo) * padFrac;
  return { lo: lo - pad, hi: hi + pad };
}

function unitOf(value, range) {
  const u = (value - range.lo) / (range.hi - range.lo);
  return Math.min(1, Math.max(0, u));
}

function yOfUnit(u, h) {
  return h - u * (h - 10) - 5;
}

function drawScaleHint(ctx, h, text) {
  ctx.fillStyle = "rgba(139,151,171,0.85)";
  ctx.font = '10px "IBM Plex Mono", monospace';
  ctx.textAlign = "left";
  ctx.fillText(text, 10, h - 6);
}

function drawUnitSeries(ctx, opts) {
  const { values, fs, range, color, width, tLeft, tRight, w, h } = opts;
  // Live monitor: never draw past the cursor, the future is not known yet.
  const tEnd = Math.min(tRight, opts.tMax ?? tRight);
  const i0 = Math.max(0, Math.floor(tLeft * fs));
  const i1 = Math.min(values.length, Math.floor(tEnd * fs) + 1);
  if (i1 <= i0) return;
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  for (let i = i0; i < i1; i++) {
    const x = xOf(i / fs, tLeft, tRight, w);
    const y = yOfUnit(unitOf(values[i], range), h);
    if (i === i0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function numberParam(id, fallback) {
  const value = Number($(id).value);
  return Number.isFinite(value) ? value : fallback;
}

function selectedKinds() {
  const kinds = new Set();
  if ($("target-oa").checked) kinds.add("obstructive");
  if ($("target-hyp").checked) kinds.add("hypopnea");
  if ($("target-unsure").checked) kinds.add("unsure");
  return kinds;
}

function oracleFireNow() {
  const probs = new Array(pack.duration_sec).fill(0);
  const kinds = selectedKinds();
  const lag = numberParam("p-lag", 10);
  const lead = numberParam("p-lead", 30);
  for (const event of pack.events) {
    if (!event.arousal_linked || !kinds.has(event.kind)) continue;
    const deadlineRef = event.arousal_start != null ? event.arousal_start : event.end;
    const first = Math.max(0, Math.ceil(event.start - lead));
    const last = Math.min(probs.length - 1, Math.floor(deadlineRef - lag));
    for (let t = first; t <= last; t++) probs[t] = 1;
  }
  return probs;
}

function policyEvents(events) {
  if ($("policy-source").value === "model") return events;
  const kinds = selectedKinds();
  return events.filter((event) => kinds.has(event.kind));
}

function setEspState(state, note = "") {
  esp.state = state;
  esp.note = note;
  paintEspLink();
}

function noteEspRx(line) {
  esp.lastRx = line;
  esp.lastRxAtMs = Date.now();
  esp.rxCount += 1;
  if (esp.state !== "live") setEspState("live");
  else paintEspLink();
}

function paintEspLink() {
  const chip = $("esp-chip");
  const dotText = $("esp-text");
  const rxText = $("esp-rx");
  chip.classList.remove("live", "open", "err");
  const via = esp.mode === "webserial" ? "Web Serial" : esp.mode === "server" ? "server" : "";
  if (esp.state === "live") {
    chip.classList.add("live");
    dotText.textContent = `ESP linked (${via})`;
  } else if (esp.state === "open") {
    chip.classList.add("open");
    dotText.textContent = `ESP port open (${via})`;
  } else if (esp.state === "err") {
    chip.classList.add("err");
    dotText.textContent = "ESP link error";
  } else {
    dotText.textContent = "ESP not connected";
  }
  let detail;
  if (esp.state === "none") {
    detail = "click Connect ESP";
  } else if (esp.lastRx) {
    const age = Math.max(0, Math.round((Date.now() - esp.lastRxAtMs) / 1000));
    detail = `rx "${esp.lastRx}" ${age}s ago`;
  } else if (esp.state === "err") {
    detail = esp.note || "no reply";
  } else {
    detail = "waiting for reply";
  }
  rxText.textContent = detail;
  const txPart = esp.lastTx ? ` | tx ${esp.lastTx}` : "";
  $("esp-last").textContent =
    esp.state === "none" ? "ESP: no link" : `ESP: ${detail}${txPart}`;
  const noLink = esp.mode === "none";
  for (const id of ["btn-adv", "btn-ret", "btn-stop", "btn-esp-test"]) {
    $(id).disabled = noLink;
  }
}

async function sendEsp(cmd) {
  if (esp.mode === "webserial" && esp.writer) {
    try {
      esp.lastTx = cmd;
      await esp.writer.write(new TextEncoder().encode(cmd + "\n"));
      paintEspLink();
    } catch (e) {
      setEspState("err", String(e.message || e));
    }
    return;
  }
  if (esp.mode === "server") {
    try {
      esp.lastTx = cmd;
      const r = await fetch("/api/esp/cmd", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cmd }),
      });
      if (!r.ok) setEspState("err", `server ${r.status}`);
      else paintEspLink();
    } catch (e) {
      setEspState("err", String(e.message || e));
    }
  }
}

async function readEspLoop(port) {
  try {
    const decoder = new TextDecoderStream();
    port.readable.pipeTo(decoder.writable).catch(() => {});
    const reader = decoder.readable.getReader();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += value;
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (line) noteEspRx(line);
      }
      if (buf.length > 400) buf = buf.slice(-200);
    }
    if (esp.mode === "webserial") setEspState("err", "serial stream closed");
  } catch (e) {
    setEspState("err", String(e.message || e));
  }
}

async function pollEspServer() {
  if (esp.mode !== "server") return;
  try {
    const st = await fetch("/api/esp/status").then((r) => r.json());
    if (!st.linked) {
      setEspState("err", "serial closed on server");
      return;
    }
    if (st.last_rx && st.last_rx !== esp.lastRx) {
      esp.lastRx = st.last_rx;
      esp.rxCount = st.rx_count || esp.rxCount + 1;
      esp.lastRxAtMs = Date.now() - Math.round((st.last_rx_age_sec || 0) * 1000);
      setEspState("live");
    } else {
      paintEspLink();
    }
  } catch {
    setEspState("err", "server unreachable");
  }
}

function resim() {
  const thr = Number($("thr").value) / 100;
  const oracle = $("policy-source").value === "oracle";
  ctrl.threshold = thr;
  ctrl.advanceSec = numberParam("p-advance", 10);
  ctrl.refractorySec = numberParam("p-refractory", 60);
  ctrl.quietRetractSec = numberParam("p-quiet", 90);
  controllerProbs = oracle ? oracleFireNow() : pack.fire_now;
  sim = ctrl.simulate(controllerProbs, pack.wake);
  $("m-thr").textContent = thr.toFixed(2);
  $("thr-lab").textContent = thr.toFixed(2);
  $("model-tag").textContent = oracle
    ? "Controller input - annotation oracle - active model"
    : "Model - fire_now - active";
  $("pill-geo").textContent = oracle
    ? `ORACLE | A=${numberParam("p-lag", 10)}s | lead=${numberParam("p-lead", 30)}s`
    : "MODEL | A=10s | lead=30s | no wake";
  $("pill-geo").classList.toggle("adv", oracle);
  lastActionSent = -1;
  updateHud();
}

function coverageNow() {
  const n = Math.max(1, Math.floor(tNow) + 1);
  const adv = sim.advanced.slice(0, n);
  const evs = policyEvents(pack.events).filter((e) => e.start <= tNow + 1);
  const linked = evs.filter((e) => e.arousal_linked);
  const lag = $("policy-source").value === "oracle" ? numberParam("p-lag", 10) : 10;
  const lead = $("policy-source").value === "oracle" ? numberParam("p-lead", 30) : 30;
  const cov = linked.filter((e) => eventCovered(e, sim.advanced, lag, lead)).length;
  return { linked: linked.length, cov };
}

function updateHud() {
  const i = Math.min(sim.advanced.length - 1, Math.max(0, Math.floor(tNow)));
  const pos = (() => {
    // reconstruct coarse position from mask + last action
    if (i < 0) return RETRACTED;
    const adv = sim.advanced[i];
    const act = sim.actions[i];
    if (act === ADVANCE) return ADVANCING;
    if (act === RETRACT) return RETRACTING;
    return adv ? ADVANCED : RETRACTED;
  })();
  $("mad-state").textContent = POS_NAME[pos];
  $("pill-mad").textContent = advLabel(pos);
  $("pill-mad").classList.toggle("adv", pos === ADVANCED || pos === ADVANCING);
  const p = controllerProbs[i] ?? 0;
  $("m-p").textContent = p.toFixed(2);
  $("m-adv").textContent = String(sim.nAdvances);
  const frac = sim.advanced.length
    ? sim.advanced.slice(0, i + 1).reduce((a, b) => a + b, 0) / (i + 1)
    : 0;
  const fracClip = sim.advanced.length
    ? sim.advanced.reduce((a, b) => a + b, 0) / sim.advanced.length
    : 0;
  $("m-frac").textContent =
    `${Math.round(frac * 100)}% (clip ${Math.round(fracClip * 100)}%)`;
  const c = coverageNow();
  $("m-cov").textContent = `${c.cov} / ${c.linked}`;
  $("mad-sub").textContent =
    pos === ADVANCED
      ? "Hold through burst -- new fires ignored"
      : pos === ADVANCING
        ? "Motor advancing (A = 10 s)"
        : pos === RETRACTING
          ? "Motor retracting"
          : p >= ctrl.threshold
            ? "fire_now above threshold"
            : "Monitoring nasal pressure + SpO2";
  const jaw = $("jaw-fill");
  const x = pos === RETRACTED ? 14 : pos === ADVANCED ? 186 : 100;
  jaw.setAttribute("x", String(x));
  jaw.setAttribute("fill", pos === RETRACTED ? "#64748b" : "#c4f25a");
  $("clock").textContent = `${fmt(tNow)} / ${fmt(pack.duration_sec)}`;
  $("now-air").textContent = `t = ${fmt(tNow)}`;
  $("now-overlay").textContent = `t = ${fmt(tNow)}`;
  updateSeekHint();
}

function advLabel(pos) {
  if (pos === ADVANCED) return "MAD advanced";
  if (pos === ADVANCING) return "MAD advancing";
  if (pos === RETRACTING) return "MAD retracting";
  return "MAD retracted";
}

function xOf(t, tLeft, tRight, w) {
  return ((t - tLeft) / (tRight - tLeft)) * w;
}

function drawEvents(ctx, h, tLeft, tRight, w, now) {
  // Scored annotations are revealed as the cursor passes them, never ahead of it.
  for (const e of pack.events) {
    if (e.end < tLeft || e.start > Math.min(tRight, now)) continue;
    const x0 = xOf(e.start, tLeft, tRight, w);
    const x1 = xOf(Math.min(e.end, now), tLeft, tRight, w);
    ctx.fillStyle = e.kind === "obstructive" ? "rgba(255,176,32,0.28)" : "rgba(192,132,252,0.28)";
    ctx.fillRect(x0, 0, Math.max(2, x1 - x0), h);
    ctx.fillStyle = e.kind === "obstructive" ? "#ffb020" : "#c084fc";
    ctx.font = "11px IBM Plex Sans";
    ctx.fillText(e.kind === "obstructive" ? "OA" : "HYP", x0 + 4, 16);
    if (e.is_cluster_first) {
      ctx.fillStyle = "#38bdf8";
      ctx.beginPath();
      ctx.moveTo(x0 - 5, h);
      ctx.lineTo(x0 + 5, h);
      ctx.lineTo(x0, h - 9);
      ctx.closePath();
      ctx.fill();
    }
  }
  ctx.strokeStyle = "#ffe08a";
  ctx.lineWidth = 2;
  for (const a of pack.arousals) {
    if (a.start < tLeft || a.start > Math.min(tRight, now)) continue;
    const x = xOf(a.start, tLeft, tRight, w);
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
}

function drawAdvanced(ctx, h, tLeft, tRight, w, now, veilNoteId = null) {
  ctx.fillStyle = "rgba(100,116,139,0.28)";
  let run = null;
  const n = sim.advanced.length;
  for (let t = Math.max(0, Math.floor(tLeft)); t <= Math.min(n - 1, Math.ceil(tRight)); t++) {
    if (sim.advanced[t] && t <= now) {
      if (run == null) run = t;
    } else if (run != null) {
      const x0 = xOf(run, tLeft, tRight, w);
      const x1 = xOf(t, tLeft, tRight, w);
      ctx.fillRect(x0, 0, x1 - x0, h);
      run = null;
    }
  }
  if (run != null) {
    const x0 = xOf(run, tLeft, tRight, w);
    const x1 = xOf(Math.min(now, tRight), tLeft, tRight, w);
    ctx.fillRect(x0, 0, x1 - x0, h);
  }
  if (!veilNoteId) return;
  const i = Math.min(n - 1, Math.max(0, Math.floor(now)));
  const note = $(veilNoteId);
  if (sim.advanced[i]) {
    const xNow = xOf(now, tLeft, tRight, w);
    const grd = ctx.createLinearGradient(xNow, 0, w, 0);
    grd.addColorStop(0, "rgba(15,23,42,0.15)");
    grd.addColorStop(1, "rgba(15,23,42,0.55)");
    ctx.fillStyle = grd;
    ctx.fillRect(xNow, 0, Math.max(0, w - xNow), h);
    note.style.display = "block";
    note.textContent = "HOLD - not looking ahead";
  } else {
    note.style.display = "none";
  }
}

function drawNow(ctx, h, tLeft, tRight, w, now) {
  const x = xOf(now, tLeft, tRight, w);
  ctx.strokeStyle = "#f8fafc";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(x, 0);
  ctx.lineTo(x, h);
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawAir(tLeft, tRight, now) {
  const { ctx, w, h } = setupCanvas($("c-air"));
  ctx.clearRect(0, 0, w, h);
  drawGrid(ctx, w, h);
  drawEvents(ctx, h, tLeft, tRight, w, now);
  drawAdvanced(ctx, h, tLeft, tRight, w, now, "veil-note");
  drawUnitSeries(ctx, {
    values: pack.pres,
    fs: pack.fs_pres,
    range: NORM.pres,
    color: "#5ec8ff",
    width: 1.4,
    tLeft,
    tRight,
    tMax: now,
    w,
    h,
  });
  drawScaleHint(ctx, h, `scale ${NORM.pres.lo.toFixed(2)} .. ${NORM.pres.hi.toFixed(2)}`);
  drawNow(ctx, h, tLeft, tRight, w, now);
}

function drawSpo2(tLeft, tRight, now) {
  const { ctx, w, h } = setupCanvas($("c-spo2"));
  ctx.clearRect(0, 0, w, h);
  drawGrid(ctx, w, h);
  drawEvents(ctx, h, tLeft, tRight, w, now);
  drawAdvanced(ctx, h, tLeft, tRight, w, now);
  drawUnitSeries(ctx, {
    values: pack.spo2,
    fs: pack.fs_decision,
    range: NORM.spo2,
    color: "#ff6b8a",
    width: 2,
    tLeft,
    tRight,
    tMax: now,
    w,
    h,
  });
  drawScaleHint(ctx, h, `${NORM.spo2.lo.toFixed(1)} .. ${NORM.spo2.hi.toFixed(1)} %`);
  drawNow(ctx, h, tLeft, tRight, w, now);
}

function drawOverlay(tLeft, tRight, now) {
  const { ctx, w, h } = setupCanvas($("c-overlay"));
  ctx.clearRect(0, 0, w, h);
  drawGrid(ctx, w, h);
  drawEvents(ctx, h, tLeft, tRight, w, now);
  drawAdvanced(ctx, h, tLeft, tRight, w, now, "veil-note-overlay");

  const yThr = yOfUnit(ctrl.threshold, h);
  ctx.strokeStyle = "#64748b";
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(0, yThr);
  ctx.lineTo(w, yThr);
  ctx.stroke();
  ctx.setLineDash([]);

  const unitRange = { lo: 0, hi: 1 };
  const p0 = Math.max(0, Math.floor(tLeft));
  const p1 = Math.min(controllerProbs.length, Math.floor(Math.min(tRight, now)) + 1);
  if (p1 > p0) {
    ctx.beginPath();
    ctx.fillStyle = "rgba(196,242,90,0.1)";
    ctx.moveTo(xOf(p0, tLeft, tRight, w), h);
    for (let i = p0; i < p1; i++) {
      ctx.lineTo(xOf(i, tLeft, tRight, w), yOfUnit(controllerProbs[i], h));
    }
    ctx.lineTo(xOf(p1 - 1, tLeft, tRight, w), h);
    ctx.closePath();
    ctx.fill();
  }
  drawUnitSeries(ctx, {
    values: pack.pres,
    fs: pack.fs_pres,
    range: NORM.pres,
    color: "rgba(94,200,255,0.7)",
    width: 1.2,
    tLeft,
    tRight,
    tMax: now,
    w,
    h,
  });
  drawUnitSeries(ctx, {
    values: pack.spo2,
    fs: pack.fs_decision,
    range: NORM.spo2,
    color: "#ff6b8a",
    width: 2,
    tLeft,
    tRight,
    tMax: now,
    w,
    h,
  });
  drawUnitSeries(ctx, {
    values: pack.active,
    fs: pack.fs_decision,
    range: unitRange,
    color: "rgba(148,163,184,0.75)",
    width: 1.1,
    tLeft,
    tRight,
    tMax: now,
    w,
    h,
  });
  drawUnitSeries(ctx, {
    values: controllerProbs,
    fs: pack.fs_decision,
    range: unitRange,
    color: "#c4f25a",
    width: 2.2,
    tLeft,
    tRight,
    tMax: now,
    w,
    h,
  });
  drawScaleHint(
    ctx,
    h,
    `Pres ${NORM.pres.lo.toFixed(2)}..${NORM.pres.hi.toFixed(2)}   ` +
      `SpO2 ${NORM.spo2.lo.toFixed(1)}..${NORM.spo2.hi.toFixed(1)}%   prob 0..1`,
  );
  drawNow(ctx, h, tLeft, tRight, w, now);
}

function drawModel(tLeft, tRight, now) {
  const { ctx, w, h } = setupCanvas($("c-model"));
  ctx.clearRect(0, 0, w, h);
  drawGrid(ctx, w, h, "#152033");
  drawAdvanced(ctx, h, tLeft, tRight, w, now);
  const yThr = h - ctrl.threshold * (h - 8) - 4;
  ctx.strokeStyle = "#64748b";
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(0, yThr);
  ctx.lineTo(w, yThr);
  ctx.stroke();
  ctx.setLineDash([]);
  const i0 = Math.max(0, Math.floor(tLeft));
  const i1 = Math.min(controllerProbs.length, Math.floor(Math.min(tRight, now)) + 1);
  if (i1 <= i0) {
    drawNow(ctx, h, tLeft, tRight, w, now);
    return;
  }
  ctx.beginPath();
  ctx.fillStyle = "rgba(196,242,90,0.18)";
  ctx.moveTo(xOf(i0, tLeft, tRight, w), h);
  for (let i = i0; i < i1; i++) {
    const y = h - controllerProbs[i] * (h - 8) - 4;
    ctx.lineTo(xOf(i, tLeft, tRight, w), y);
  }
  ctx.lineTo(xOf(i1 - 1, tLeft, tRight, w), h);
  ctx.closePath();
  ctx.fill();
  ctx.beginPath();
  ctx.strokeStyle = "#c4f25a";
  ctx.lineWidth = 1.8;
  for (let i = i0; i < i1; i++) {
    const y = h - controllerProbs[i] * (h - 8) - 4;
    const x = xOf(i, tLeft, tRight, w);
    if (i === i0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.beginPath();
  ctx.strokeStyle = "rgba(94,200,255,0.7)";
  ctx.lineWidth = 1.2;
  for (let i = i0; i < i1; i++) {
    const y = h - pack.active[i] * (h - 8) - 4;
    const x = xOf(i, tLeft, tRight, w);
    if (i === i0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  drawNow(ctx, h, tLeft, tRight, w, now);
}

function drawMini(now) {
  const { ctx, w, h } = setupCanvas($("c-mini"));
  ctx.clearRect(0, 0, w, h);
  const n = pack.duration_sec;
  for (const e of pack.events) {
    ctx.fillStyle = e.kind === "obstructive" ? "#ffb020" : "#c084fc";
    ctx.fillRect((e.start / n) * w, 4, Math.max(2, ((e.end - e.start) / n) * w), h - 8);
  }
  for (let t = 0; t < sim.advanced.length; t++) {
    if (!sim.advanced[t]) continue;
    ctx.fillStyle = "rgba(148,163,184,0.45)";
    ctx.fillRect((t / n) * w, 0, w / n + 0.5, h);
  }
  for (const t of coldStartTimes()) {
    const x = (t / n) * w;
    ctx.fillStyle = "#38bdf8";
    ctx.beginPath();
    ctx.moveTo(x - 4, 0);
    ctx.lineTo(x + 4, 0);
    ctx.lineTo(x, 7);
    ctx.closePath();
    ctx.fill();
  }
  const xNow = (now / n) * w;
  ctx.fillStyle = "rgba(7,9,13,0.62)"; // clip map: dim the part not played yet
  ctx.fillRect(xNow, 0, Math.max(0, w - xNow), h);
  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(xNow, 0, 2, h);
}

function render() {
  if (!pack) return;
  const tRight = tNow + winSec * 0.18;
  const tLeft = tRight - winSec;
  if (layoutMode === "overlay") {
    drawOverlay(tLeft, tRight, tNow);
  } else {
    drawAir(tLeft, tRight, tNow);
    drawSpo2(tLeft, tRight, tNow);
    drawModel(tLeft, tRight, tNow);
  }
  drawMini(tNow);
  updateHud();
}

async function maybeActuate() {
  const i = Math.min(sim.actions.length - 1, Math.max(0, Math.floor(tNow)));
  if (i === lastActionSent) return;
  // fire on the first second we cross an ADVANCE/RETRACT
  for (let k = lastActionSent + 1; k <= i; k++) {
    if (sim.actions[k] === ADVANCE) await sendEsp("ADVANCE");
    else if (sim.actions[k] === RETRACT) await sendEsp("RETRACT");
  }
  lastActionSent = i;
}

function tick(ts) {
  if (playing) {
    if (lastTs) {
      tNow += ((ts - lastTs) / 1000) * speed;
      if (tNow >= pack.duration_sec) {
        tNow = pack.duration_sec;
        playing = false;
        $("btn-play").textContent = "Play";
      }
      maybeActuate();
    }
    lastTs = ts;
    render();
  } else {
    lastTs = 0;
  }
  requestAnimationFrame(tick);
}

function coldStartTimes() {
  return pack.events
    .filter((e) => e.is_cluster_first)
    .map((e) => e.start)
    .sort((a, b) => a - b);
}

function targetStartTimes() {
  return pack.events.map((e) => e.start).sort((a, b) => a - b);
}

// Next entry whose pre-roll position is still ahead of us; wraps to the first.
function nextSeekTarget(starts) {
  const ahead = starts.filter((s) => s - PRE_ROLL_SEC > tNow + 0.5);
  const target = ahead.length ? ahead[0] : starts[0];
  return target == null ? null : target;
}

function seekBefore(start) {
  tNow = Math.max(0, start - PRE_ROLL_SEC);
  lastActionSent = Math.floor(tNow) - 1;
  render();
}

// Opening frame: first cold start that has a full pre-roll of history behind it
// and finds the device retracted, so the advance happens in front of the audience.
function burstT() {
  const cold = coldStartTimes();
  const withHistory = cold.filter((t) => t >= PRE_ROLL_SEC + 20);
  const retracted = withHistory.filter(
    (t) => !sim.advanced[Math.floor(t - PRE_ROLL_SEC)],
  );
  const pick = retracted[0] ?? withHistory[0] ?? cold[0];
  if (pick == null) return 0;
  return Math.max(0, pick - PRE_ROLL_SEC);
}

function updateSeekHint() {
  const cold = coldStartTimes();
  if (!cold.length) {
    $("seek-hint").textContent = "cold starts: none in clip";
    return;
  }
  const next = nextSeekTarget(cold);
  const idx = cold.indexOf(next) + 1;
  const wrapped = next - PRE_ROLL_SEC <= tNow + 0.5;
  $("seek-hint").textContent =
    `cold start ${idx}/${cold.length} at ${fmt(next)}` + (wrapped ? " (wrap)" : "");
}

async function connectEsp() {
  if (navigator.serial) {
    try {
      const port = await navigator.serial.requestPort();
      await port.open({ baudRate: 115200 });
      esp.port = port;
      esp.mode = "webserial";
      esp.writer = port.writable.getWriter();
      setEspState("open");
      readEspLoop(port);
      await testEspLink();
      return;
    } catch (e) {
      // user cancelled the picker, or the port is busy: fall through to server
      esp.mode = "none";
      esp.writer = null;
      setEspState("none", String(e.message || e));
    }
  }
  const st = await fetch("/api/esp/status")
    .then((r) => r.json())
    .catch(() => null);
  if (st && st.linked) {
    esp.mode = "server";
    setEspState(st.rx_count ? "live" : "open");
    if (st.last_rx) noteEspRx(st.last_rx);
    await testEspLink();
    return;
  }
  setEspState("none");
  alert(
    "No ESP link.\n\n" +
      "Option 1: Chrome/Edge over http://127.0.0.1 -> Connect ESP -> pick the usbserial port.\n" +
      "Option 2: python server.py --esp auto (server owns the serial port).\n\n" +
      "Close the Arduino IDE Serial Monitor first; it holds the port.",
  );
}

// Sends a harmless help command; the firmware answers, which proves two-way comms.
async function testEspLink() {
  if (esp.mode === "none") {
    setEspState("none");
    return false;
  }
  const before = esp.rxCount;
  if (esp.mode === "server") {
    try {
      const res = await fetch("/api/esp/ping", { method: "POST" }).then((r) => r.json());
      esp.lastTx = "?";
      if (res.reply) {
        noteEspRx(res.reply);
        return true;
      }
      setEspState("err", "no reply from ESP");
      return false;
    } catch (e) {
      setEspState("err", String(e.message || e));
      return false;
    }
  }
  await sendEsp("?");
  for (let i = 0; i < 20; i++) {
    await new Promise((r) => setTimeout(r, 100));
    if (esp.rxCount > before) return true;
  }
  setEspState("err", "no reply from ESP (check baud 115200 / firmware)");
  return false;
}

function updatePolicyUi() {
  const oracle = $("policy-source").value === "oracle";
  for (const id of ["target-oa", "target-hyp", "target-unsure", "p-lag", "p-lead"]) {
    $(id).disabled = !oracle;
  }
  $("oracle-note").textContent = oracle
    ? "Oracle: kind from scored annotations, not a real-time classifier."
    : "Combined model (OA + hypopnea + Unsure): kind cannot be switched.";
}

async function boot() {
  pack = await fetch("/api/pack").then((r) => r.json());
  $("pill-sub").textContent = `MESA ${pack.meta.subject_id}`;
  $("pill-geo").textContent = "A=10s | lead=30s | no wake";
  NORM.pres = robustRange(pack.pres);
  const spo2Range = robustRange(pack.spo2, 0.01, 1.0, 0.05);
  NORM.spo2 = { lo: spo2Range.lo, hi: Math.min(100, spo2Range.hi) };

  const params = new URLSearchParams(location.search);
  const wantedSpeed = Number(params.get("speed"));
  if ([1, 2, 4, 8, 16].includes(wantedSpeed)) {
    speed = wantedSpeed;
    $("speed").value = String(wantedSpeed);
  }
  const wantedLayout = params.get("layout");
  if (wantedLayout === "overlay" || wantedLayout === "stack") {
    layoutMode = wantedLayout;
    $("layout").value = wantedLayout;
    $("stage").className = `stage mode-${wantedLayout}`;
  }
  resim();
  tNow = burstT();
  render();
  requestAnimationFrame(tick);
  if (params.get("play") === "1") {
    playing = true;
    $("btn-play").textContent = "Pause";
  }

  $("btn-play").onclick = () => {
    playing = !playing;
    $("btn-play").textContent = playing ? "Pause" : "Play";
  };
  $("btn-reset").onclick = () => {
    playing = false;
    $("btn-play").textContent = "Play";
    tNow = burstT();
    lastActionSent = Math.floor(tNow) - 1;
    render();
  };
  $("btn-cold").onclick = () => {
    const target = nextSeekTarget(coldStartTimes());
    if (target != null) seekBefore(target);
  };
  $("btn-event").onclick = () => {
    const target = nextSeekTarget(targetStartTimes());
    if (target != null) seekBefore(target);
  };
  $("speed").onchange = (e) => {
    speed = Number(e.target.value);
  };
  $("layout").onchange = (e) => {
    layoutMode = e.target.value;
    $("stage").className = `stage mode-${layoutMode}`;
    $("veil-note").style.display = "none";
    $("veil-note-overlay").style.display = "none";
    render();
  };
  $("win").oninput = (e) => {
    winSec = Number(e.target.value);
    $("win-lab").textContent = `${winSec}s`;
    render();
  };
  $("thr").oninput = () => {
    resim();
    render();
  };
  $("policy-source").onchange = () => {
    updatePolicyUi();
    resim();
    render();
  };
  for (const id of [
    "target-oa",
    "target-hyp",
    "target-unsure",
    "p-advance",
    "p-refractory",
    "p-quiet",
    "p-lag",
    "p-lead",
  ]) {
    $(id).onchange = () => {
      resim();
      render();
    };
  }
  updatePolicyUi();
  $("btn-esp").onclick = connectEsp;
  $("btn-esp-test").onclick = testEspLink;
  $("btn-adv").onclick = () => sendEsp("ADVANCE");
  $("btn-ret").onclick = () => sendEsp("RETRACT");
  $("btn-stop").onclick = () => sendEsp("s");
  window.addEventListener("resize", render);

  paintEspLink();
  const st = await fetch("/api/esp/status")
    .then((r) => r.json())
    .catch(() => null);
  if (st && st.linked) {
    esp.mode = "server";
    setEspState(st.rx_count ? "live" : "open");
    if (st.last_rx) noteEspRx(st.last_rx);
  }
  setInterval(() => {
    if (esp.mode === "server") pollEspServer();
    else paintEspLink();
  }, 2000);
}

boot();
