/* ProactMAD live-replay UI: PSG traces + fire_now + MAD controller + ESP. */

const HOLD = 0, ADVANCE = 1, RETRACT = 2;
const BLOCK_NONE = 0, BLOCK_REFRACTORY = 1, BLOCK_WAKE = 2;
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
    this.positions = [];
    this.blocked = [];
  }
  step(t, prob, wake = false, hold = false) {
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
    const alreadyOut = this.position === ADVANCED || this.position === ADVANCING;
    const highRisk = !wake && prob >= this.threshold;
    if (highRisk || (hold && alreadyOut)) this.quietTimer = 0;
    else this.quietTimer += dt;
    const refractory = t - this.lastAdvanceT < this.refractorySec;
    const canAdvance = this.position === RETRACTED && !refractory && !wake;
    // Why a fire did not move the jaw, so the UI can say it out loud.
    this.blocked.push(
      this.position === RETRACTED && prob >= this.threshold
        ? wake
          ? BLOCK_WAKE
          : refractory
            ? BLOCK_REFRACTORY
            : BLOCK_NONE
        : BLOCK_NONE,
    );
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
    // Exact mechanical state for this second. advancedMask is the control
    // commitment (it goes true the instant travel starts, which is what the
    // A=10 s deadline already accounts for); the jaw is only really forward
    // once the motor has finished, so the UI must not conflate the two.
    this.positions.push(this.position);
    return action;
  }
  simulate(probs, wake, holds) {
    this.reset();
    for (let t = 0; t < probs.length; t++) {
      this.step(t, probs[t], !!wake[t], !!(holds && holds[t]));
    }
    return {
      advanced: this.advancedMask.slice(),
      actions: this.actions.slice(),
      positions: this.positions.slice(),
      blocked: this.blocked.slice(),
      nAdvances: this.nAdvances,
    };
  }
}

function isScored(t) {
  if (!pack) return false;
  if (pack.scored && pack.scored.length) return !!pack.scored[t];
  const v = pack.fire_now[t];
  return v != null && Number.isFinite(v);
}

function headAt(name, t) {
  const arr = pack[name];
  if (!arr) return null;
  const v = arr[t];
  return v == null || !Number.isFinite(v) ? null : v;
}

function fuseHeads(t, thr) {
  if (!isScored(t)) return { want: false, hold: false, mode: "unscored", fire: null, pre: null, active: null };
  const fire = headAt("fire_now", t);
  const pre = headAt("pre_onset", t);
  const active = headAt("active", t);
  const f = fire != null && fire >= thr;
  const p = pre != null && pre >= thr;
  const a = active != null && active >= thr;
  let want = false;
  let hold = false;
  let mode = "quiet";
  if (p && f && !a) { want = true; hold = true; mode = "early-warn"; }
  else if (f && a && !p) { want = true; hold = true; mode = "rescue"; }
  else if (f) { want = true; hold = true; mode = "fire_now"; }
  else if (a && !p) { want = false; hold = true; mode = "detect-only"; }
  else if (p && !f) { want = false; hold = false; mode = "mismatch"; }
  return { want, hold, mode, fire, pre, active };
}

function fusedControllerInput() {
  const thr = Number($("thr").value) / 100;
  const n = pack.duration_sec;
  const probs = new Array(n);
  const holds = new Array(n);
  const modes = new Array(n);
  for (let t = 0; t < n; t++) {
    const f = fuseHeads(t, thr);
    const awake = !!(pack.wake && pack.wake[t]);
    probs[t] = f.want ? f.fire : 0;
    holds[t] = f.hold && !awake;
    modes[t] = awake && f.want ? "wake-gated" : f.mode;
  }
  return { probs, holds, modes };
}

const $ = (id) => document.getElementById(id);

function fmt(t) {
  t = Math.max(0, t);
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const s = Math.floor(t % 60);
  const mm = String(m).padStart(2, "0");
  const ss = String(s).padStart(2, "0");
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

function isNightClip() {
  return !!pack && pack.duration_sec > 3600;
}

// First second inside the credit window [onset - lead, deadline] where the device
// was already advanced (or completing an advance); null if the event was missed.
function coverHit(ev, advanced, A = 10, lead = 30) {
  if (!ev.arousal_linked) return null;
  const deadlineRef = ev.arousal_start != null ? ev.arousal_start : ev.end;
  const deadline = deadlineRef - A;
  const earliest = ev.start - lead;
  const i0 = Math.max(0, Math.floor(earliest));
  const i1 = Math.min(advanced.length, Math.floor(deadline) + 1);
  for (let i = i0; i < i1; i++) if (advanced[i]) return i;
  return null;
}

// Start of the advance run that covered second i, so lead can be measured from
// the moment the jaw was fully forward rather than from the credit window edge.
function runStartAt(advanced, i) {
  let k = i;
  while (k > 0 && advanced[k - 1]) k -= 1;
  return k;
}

function median(values) {
  if (!values.length) return null;
  const v = values.slice().sort((a, b) => a - b);
  const mid = v.length >> 1;
  return v.length % 2 ? v[mid] : (v[mid - 1] + v[mid]) / 2;
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

function drawGrid(ctx, w, h, color = "#e8edf4") {
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
let clipIndex = []; // example library: one entry per presentation story
let ctrl = new MadController();
let sim = { advanced: [], actions: [], positions: [], blocked: [], nAdvances: 0 };
let simVersion = 0;
let controllerProbs = [];
let tNow = 0;
let playing = false;
let lastTs = 0;
let winSec = 60;
let speed = 8;
let speedPinned = false; // once chosen by URL or user, stop auto-picking per clip
let lastActionSent = -1;
let lastPosSent = null; // jaw position the board was last told to hold
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

// Pulse-ox dropout in a whole night reads as 0 or a few percent; those samples
// must not set the display range or be drawn as a real desaturation.
function spo2Valid(v) {
  return v >= 50 && v <= 100;
}

function unitOf(value, range) {
  const u = (value - range.lo) / (range.hi - range.lo);
  return Math.min(1, Math.max(0, u));
}

function yOfUnit(u, h) {
  return h - u * (h - 10) - 5;
}

function drawScaleHint(ctx, h, text) {
  ctx.fillStyle = "rgba(100,116,139,0.9)";
  ctx.font = '10px "IBM Plex Mono", monospace';
  ctx.textAlign = "left";
  ctx.fillText(text, 10, h - 6);
}

// A labelled horizontal reference line at a raw value. Nothing is drawn when the
// value is outside the lane's range, so a lane never shows a baseline it does not
// actually contain (SpO2 windows do not include 0%).
function drawValueAxis(ctx, w, h, range, value, label, opts = {}) {
  const u = (value - range.lo) / (range.hi - range.lo);
  if (!(u >= 0 && u <= 1)) return false;
  const y = yOfUnit(u, h);
  ctx.save();
  ctx.strokeStyle = opts.color || "rgba(15,23,42,0.30)";
  ctx.lineWidth = opts.width || 1;
  if (opts.dash) ctx.setLineDash(opts.dash);
  ctx.beginPath();
  ctx.moveTo(0, y);
  ctx.lineTo(w, y);
  ctx.stroke();
  ctx.setLineDash([]);
  if (label) {
    ctx.fillStyle = opts.labelColor || "rgba(71,85,105,0.95)";
    ctx.font = '9px "IBM Plex Mono", monospace';
    // The bottom-left corner belongs to the scale hint, so a floor label goes right.
    const right = opts.labelSide === "right";
    ctx.textAlign = right ? "right" : "left";
    ctx.textBaseline = "alphabetic";
    ctx.fillText(label, right ? w - 3 : 3, Math.max(9, y - 2));
  }
  ctx.restore();
  return true;
}

function drawUnitSeries(ctx, opts) {
  const { values, fs, range, color, width, tLeft, tRight, w, h } = opts;
  // Live monitor: never draw past the cursor, the future is not known yet.
  const tEnd = Math.min(tRight, opts.tMax ?? tRight);
  const i0 = Math.max(0, Math.floor(tLeft * fs));
  const i1 = Math.min(values.length, Math.floor(tEnd * fs) + 1);
  if (i1 <= i0) return;
  const valid = opts.valid || (() => true);
  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  let pen = false;
  for (let i = i0; i < i1; i++) {
    if (!valid(values[i])) {
      pen = false; // sensor dropout: leave a gap instead of a false floor
      continue;
    }
    const x = xOf(i / fs, tLeft, tRight, w);
    const y = yOfUnit(unitOf(values[i], range), h);
    if (!pen) {
      ctx.moveTo(x, y);
      pen = true;
    } else ctx.lineTo(x, y);
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

// Several target events can share one scored arousal. Group by the annotation
// timestamp so the UI counts and draws arousal episodes, not event-arousal links.
function linkedArousalGroups() {
  const groups = new Map();
  for (const event of policyEvents(pack.events)) {
    if (!event.arousal_linked || event.arousal_start == null) continue;
    const key = Number(event.arousal_start).toFixed(1);
    if (!groups.has(key)) {
      groups.set(key, { start: Number(event.arousal_start), events: [] });
    }
    groups.get(key).events.push(event);
  }
  return [...groups.values()].sort((a, b) => a.start - b.start);
}

function linkedArousalSummary(now = pack.duration_sec) {
  const groups = linkedArousalGroups();
  let reached = 0;
  let covered = 0;
  const lag = $("policy-source").value === "oracle" ? numberParam("p-lag", 10) : 10;
  const lead = $("policy-source").value === "oracle" ? numberParam("p-lead", 30) : 30;
  for (const group of groups) {
    if (group.start > now) continue;
    reached += 1;
    if (group.events.some((event) => coverHit(event, sim.advanced, lag, lead) != null)) {
      covered += 1;
    }
  }
  return {
    totalLinked: groups.length,
    totalAll: pack.arousals.length,
    reached,
    covered,
    missed: reached - covered,
  };
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
  if (esp.state !== "none" && speed > MAX_HW_SPEED) {
    detail += ` | ${speed}x: hardware muted`;
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
  simVersion += 1;
  const thr = Number($("thr").value) / 100;
  const oracle = $("policy-source").value === "oracle";
  ctrl.threshold = thr;
  ctrl.advanceSec = numberParam("p-advance", 10);
  ctrl.refractorySec = numberParam("p-refractory", 60);
  ctrl.quietRetractSec = numberParam("p-quiet", 90);
  const fused = fusedControllerInput();
  controllerProbs = oracle ? oracleFireNow() : fused.probs;
  const holds = oracle ? null : fused.holds;
  sim = ctrl.simulate(controllerProbs, pack.wake, holds);
  buildCumulative();
  $("thr-lab").textContent = thr.toFixed(2);
  $("model-tag").textContent = oracle
    ? "1 Hz heads + oracle controller (teaching)"
    : "1 Hz heads: pre_onset + fire_now + active";
  $("pill-geo").textContent = oracle
    ? `ORACLE A=${numberParam("p-lag", 10)}s lead=${numberParam("p-lead", 30)}s`
    : "3-head A=10s lead=30s";
  $("pill-geo").title = oracle
    ? "Controller input from scored annotations (presentation only), deadline geometry"
    : "pre_onset early-warn, fire_now primary trigger, active hold/rescue. Deployable 1 Hz grid, no hypnogram wake.";
  $("pill-geo").classList.toggle("adv", oracle);
  lastActionSent = -1;
  updateHud();
}

// Running totals so the HUD does not rescan a 10 h night on every frame.
const cum = { advanced: new Int32Array(0), advances: new Int32Array(0) };

function buildCumulative() {
  const n = sim.advanced.length;
  cum.advanced = new Int32Array(n);
  cum.advances = new Int32Array(n);
  let adv = 0;
  let acts = 0;
  for (let i = 0; i < n; i++) {
    adv += sim.advanced[i] ? 1 : 0;
    acts += sim.actions[i] === ADVANCE ? 1 : 0;
    cum.advanced[i] = adv;
    cum.advances[i] = acts;
  }
}

function coverageNow() {
  const evs = policyEvents(pack.events).filter((e) => e.start <= tNow + 1);
  const linked = evs.filter((e) => e.arousal_linked);
  const lag = $("policy-source").value === "oracle" ? numberParam("p-lag", 10) : 10;
  const lead = $("policy-source").value === "oracle" ? numberParam("p-lead", 30) : 30;
  let cov = 0;
  let inherited = 0;
  const leads = [];
  for (const e of linked) {
    const hit = coverHit(e, sim.advanced, lag, lead);
    if (hit == null) continue;
    cov += 1;
    const start = runStartAt(sim.advanced, hit);
    const leadSec = e.start - (start + ctrl.advanceSec);
    // Only a run that started for this event counts as lead; minutes-long holds
    // covering a later event are inherited, not prediction.
    const fresh = e.start - Math.max(lead, 60) - ctrl.advanceSec;
    if (leadSec >= 0 && start >= fresh) leads.push(leadSec);
    else inherited += 1;
  }
  return {
    linked: linked.length,
    cov,
    leadMedian: median(leads),
    nLead: leads.length,
    inherited,
  };
}

function updateHud() {
  const i = Math.min(sim.advanced.length - 1, Math.max(0, Math.floor(tNow)));
  const pos =
    sim.positions && sim.positions[i] != null ? sim.positions[i] : RETRACTED;
  $("mad-state").textContent = POS_NAME[pos];
  $("pill-mad").textContent = advLabel(pos);
  $("pill-mad").classList.toggle("adv", pos === ADVANCED || pos === ADVANCING);
  const fusedNow = fuseHeads(i, ctrl.threshold);
  if (pack.wake && pack.wake[i] && fusedNow.want) fusedNow.mode = "wake-gated";
  const blockedNow = sim.blocked && sim.blocked[i] != null ? sim.blocked[i] : BLOCK_NONE;
  const predictionTime = fmt(i);
  $("m-raw").textContent = fusedNow.fire == null ? "unscored" : fusedNow.fire.toFixed(3);
  $("m-raw-time").textContent = `@ ${predictionTime}`;
  $("m-pre").textContent = fusedNow.pre == null ? "--" : fusedNow.pre.toFixed(3);
  $("m-act").textContent = fusedNow.active == null ? "--" : fusedNow.active.toFixed(3);
  $("m-mode").textContent = fusedNow.mode;
  $("raw-p-value").textContent = fusedNow.fire == null ? "--" : fusedNow.fire.toFixed(3);
  $("raw-p-time").textContent = `${fusedNow.mode}  t=${predictionTime}`;
  $("m-p").textContent = (controllerProbs[i] ?? 0).toFixed(3);
  const advSoFar = cum.advances[i] ?? 0;
  $("m-adv").textContent = `${advSoFar} / ${sim.nAdvances}`;
  const n = sim.advanced.length;
  const frac = n ? cum.advanced[i] / (i + 1) : 0;
  const fracClip = n ? cum.advanced[n - 1] / n : 0;
  $("m-frac").textContent =
    `${Math.round(frac * 100)}% (clip ${Math.round(fracClip * 100)}%)`;
  const c = coverageNow();
  $("m-cov").textContent = `${c.cov} / ${c.linked}`;
  const arousal = linkedArousalSummary(tNow);
  $("m-arousal-total").textContent = `${arousal.totalLinked} / ${arousal.totalAll}`;
  $("m-arousal-covered").textContent = String(arousal.covered);
  $("m-arousal-missed").textContent = String(arousal.missed);
  $("m-lead").textContent =
    c.leadMedian == null
      ? c.inherited
        ? `held over (n=${c.inherited})`
        : "--"
      : `${c.leadMedian >= 0 ? "+" : ""}${c.leadMedian.toFixed(0)} s (n=${c.nLead}` +
        `${c.inherited ? `, ${c.inherited} held` : ""})`;
  $("m-lead").title =
    "Median time the jaw was already fully forward before onset, counting only " +
    "advances started for that event; events covered by an earlier hold are " +
    "reported as held over.";
  $("m-duty").textContent = `${Math.round((1 - fracClip) * 100)}% less jaw time`;
  $("mad-sub").textContent =
    pos === ADVANCED
      ? fusedNow.mode === "rescue"
        ? "Rescue / hold -- fire_now + active"
        : fusedNow.mode === "detect-only"
          ? "Hold -- active only, not a prediction fire"
          : "Hold through burst"
      : pos === ADVANCING
        ? fusedNow.mode === "early-warn"
          ? "Advancing on early-warn (pre_onset + fire_now)"
          : "Motor advancing (A = 10 s)"
        : pos === RETRACTING
          ? "Motor retracting"
          : fusedNow.mode === "unscored"
            ? "No 1 Hz score (lookback or hard artifact)"
            : blockedNow === BLOCK_WAKE
              ? "fire_now high, but scored wake -- gated, jaw stays home"
              : blockedNow === BLOCK_REFRACTORY
                ? "fire_now high, but inside the 60 s refractory -- jaw stays home"
                : fusedNow.mode === "mismatch"
                  ? "pre_onset high, fire_now low -- not actuating"
                  : fusedNow.want
                    ? "fire_now above threshold"
                    : "Monitoring nasal pressure + SpO2";
  const jaw = $("jaw-fill");
  const x = pos === RETRACTED ? 14 : pos === ADVANCED ? 186 : 100;
  jaw.setAttribute("x", String(x));
  jaw.setAttribute("fill", pos === RETRACTED ? "#94a3b8" : "#65a30d");
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
    ctx.fillStyle = e.kind === "obstructive" ? "rgba(217,119,6,0.22)" : "rgba(147,51,234,0.18)";
    ctx.fillRect(x0, 0, Math.max(2, x1 - x0), h);
    ctx.fillStyle = e.kind === "obstructive" ? "#d97706" : "#9333ea";
    ctx.font = "11px IBM Plex Sans";
    ctx.fillText(e.kind === "obstructive" ? "OA" : "HYP", x0 + 4, 16);
    if (e.is_cluster_first) {
      ctx.fillStyle = "#0284c7";
      ctx.beginPath();
      ctx.moveTo(x0 - 5, h);
      ctx.lineTo(x0 + 5, h);
      ctx.lineTo(x0, h - 9);
      ctx.closePath();
      ctx.fill();
    }
  }
  // Show only arousals linked to a target respiratory event. All-arousal count
  // remains in the sidebar for comparison with the original PSG annotation.
  ctx.strokeStyle = "#eab308";
  ctx.fillStyle = "#854d0e";
  ctx.lineWidth = 2.5;
  ctx.setLineDash([4, 3]);
  for (const arousal of linkedArousalGroups()) {
    if (arousal.start < tLeft || arousal.start > Math.min(tRight, now)) continue;
    const x = xOf(arousal.start, tLeft, tRight, w);
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
    ctx.font = "9px IBM Plex Sans";
    ctx.fillText("LINKED AROUSAL", x + 3, h - 5);
  }
  ctx.setLineDash([]);
}

function drawAdvanced(ctx, h, tLeft, tRight, w, now, veilNoteId = null) {
  ctx.fillStyle = "rgba(148,163,184,0.28)";
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
    grd.addColorStop(0, "rgba(241,245,249,0.2)");
    grd.addColorStop(1, "rgba(226,232,240,0.75)");
    ctx.fillStyle = grd;
    ctx.fillRect(xNow, 0, Math.max(0, w - xNow), h);
    note.style.display = "block";
    // The jaw is not forward yet during travel; say so instead of "HOLD".
    note.textContent =
      sim.positions && sim.positions[i] === ADVANCING
        ? "ADVANCING - motor moving, jaw not forward yet"
        : "HOLD - not looking ahead";
  } else {
    note.style.display = "none";
  }
}

function drawNow(ctx, h, tLeft, tRight, w, now) {
  const x = xOf(now, tLeft, tRight, w);
  ctx.strokeStyle = "#0f172a";
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
  // Zero flow: an apnea is the trace collapsing onto this line.
  drawValueAxis(ctx, w, h, NORM.pres, 0, "0");
  drawUnitSeries(ctx, {
    values: pack.pres,
    fs: pack.fs_pres,
    range: NORM.pres,
    color: "#0284c7",
    width: 1.4,
    tLeft,
    tRight,
    tMax: now,
    w,
    h,
  });
  drawScaleHint(ctx, h, `scale ${NORM.pres.lo.toFixed(2)} .. ${NORM.pres.hi.toFixed(2)}   0 = no flow`);
  drawNow(ctx, h, tLeft, tRight, w, now);
}

function drawSpo2(tLeft, tRight, now) {
  const { ctx, w, h } = setupCanvas($("c-spo2"));
  ctx.clearRect(0, 0, w, h);
  drawGrid(ctx, w, h);
  drawEvents(ctx, h, tLeft, tRight, w, now);
  drawAdvanced(ctx, h, tLeft, tRight, w, now);
  // 0% SpO2 is never inside a real display window, so the reference here is the
  // 90% desaturation level; drawValueAxis draws nothing when it is off scale.
  const has90 = drawValueAxis(ctx, w, h, NORM.spo2, 90, "90%", { dash: [4, 3] });
  for (const v of [NORM.spo2.hi, NORM.spo2.lo]) {
    drawValueAxis(ctx, w, h, NORM.spo2, v, `${v.toFixed(1)}%`, {
      color: "rgba(15,23,42,0.14)",
      labelSide: "right",
    });
  }
  drawUnitSeries(ctx, {
    values: pack.spo2,
    fs: pack.fs_decision,
    range: NORM.spo2,
    color: "#e11d48",
    width: 2,
    valid: spo2Valid,
    tLeft,
    tRight,
    tMax: now,
    w,
    h,
  });
  drawScaleHint(
    ctx,
    h,
    `${NORM.spo2.lo.toFixed(1)} .. ${NORM.spo2.hi.toFixed(1)} %` +
      (has90 ? "   dashed = 90%" : "   (90% off scale)"),
  );
  drawNow(ctx, h, tLeft, tRight, w, now);
}

function drawOverlay(tLeft, tRight, now) {
  const { ctx, w, h } = setupCanvas($("c-overlay"));
  ctx.clearRect(0, 0, w, h);
  drawGrid(ctx, w, h);
  drawEvents(ctx, h, tLeft, tRight, w, now);
  drawAdvanced(ctx, h, tLeft, tRight, w, now, "veil-note-overlay");

  const unitRange = { lo: 0, hi: 1 };
  // Zero lines for the two things stacked in this lane: probability 0 at the
  // floor, and the nasal-pressure no-flow level where the traces actually sit.
  drawValueAxis(ctx, w, h, unitRange, 0, "prob 0", {
    color: "rgba(15,23,42,0.28)",
    labelSide: "right",
  });
  drawValueAxis(ctx, w, h, NORM.pres, 0, "Pres 0", {
    color: "rgba(2,132,199,0.35)",
    labelColor: "rgba(2,132,199,0.9)",
  });
  drawValueAxis(ctx, w, h, unitRange, ctrl.threshold, `thr ${ctrl.threshold.toFixed(2)}`, {
    color: "#64748b",
    dash: [3, 3],
  });

  const finite = (v) => v != null && Number.isFinite(v);
  const common = { fs: pack.fs_decision, range: unitRange, tLeft, tRight, tMax: now, w, h, valid: finite };
  const p0 = Math.max(0, Math.floor(tLeft));
  const p1 = Math.min(pack.fire_now.length, Math.floor(Math.min(tRight, now)) + 1);
  if (p1 > p0) {
    ctx.beginPath();
    ctx.fillStyle = "rgba(132,204,22,0.16)";
    let pen = false;
    for (let i = p0; i < p1; i++) {
      if (!finite(pack.fire_now[i])) {
        if (pen) {
          ctx.lineTo(xOf(i - 1, tLeft, tRight, w), h);
          ctx.closePath();
          ctx.fill();
          ctx.beginPath();
          ctx.fillStyle = "rgba(132,204,22,0.16)";
          pen = false;
        }
        continue;
      }
      const x = xOf(i, tLeft, tRight, w);
      const y = yOfUnit(pack.fire_now[i], h);
      if (!pen) {
        ctx.moveTo(x, h);
        ctx.lineTo(x, y);
        pen = true;
      } else {
        ctx.lineTo(x, y);
      }
    }
    if (pen) {
      ctx.lineTo(xOf(p1 - 1, tLeft, tRight, w), h);
      ctx.closePath();
      ctx.fill();
    }
  }
  drawUnitSeries(ctx, {
    values: pack.pres,
    fs: pack.fs_pres,
    range: NORM.pres,
    color: "rgba(2,132,199,0.7)",
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
    color: "#e11d48",
    width: 2,
    valid: spo2Valid,
    tLeft,
    tRight,
    tMax: now,
    w,
    h,
  });
  if (pack.pre_onset) {
    drawUnitSeries(ctx, { values: pack.pre_onset, color: "#ca8a04", width: 1.3, ...common });
  }
  drawUnitSeries(ctx, { values: pack.active, color: "rgba(100,116,139,0.8)", width: 1.1, ...common });
  drawUnitSeries(ctx, { values: pack.fire_now, color: "#4d7c0f", width: 2.2, ...common });
  if ($("policy-source").value === "oracle") {
    drawUnitSeries(ctx, {
      values: controllerProbs,
      fs: pack.fs_decision,
      range: unitRange,
      color: "#ca8a04",
      width: 1.4,
      tLeft,
      tRight,
      tMax: now,
      w,
      h,
    });
  }
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
  drawGrid(ctx, w, h, "#e8edf4");
  drawAdvanced(ctx, h, tLeft, tRight, w, now);
  const unitRange = { lo: 0, hi: 1 };
  drawValueAxis(ctx, w, h, unitRange, 1, "1", { color: "rgba(15,23,42,0.14)", labelSide: "right" });
  drawValueAxis(ctx, w, h, unitRange, 0, "0", { color: "rgba(15,23,42,0.30)", labelSide: "right" });
  drawValueAxis(ctx, w, h, unitRange, ctrl.threshold, `thr ${ctrl.threshold.toFixed(2)}`, {
    color: "#64748b",
    dash: [3, 3],
  });
  const finite = (v) => v != null && Number.isFinite(v);
  const common = { fs: pack.fs_decision, range: unitRange, tLeft, tRight, tMax: now, w, h, valid: finite };
  if (pack.pre_onset) {
    drawUnitSeries(ctx, { values: pack.pre_onset, color: "#ca8a04", width: 1.4, ...common });
  }
  drawUnitSeries(ctx, { values: pack.active, color: "rgba(100,116,139,0.9)", width: 1.2, ...common });
  drawUnitSeries(ctx, { values: pack.fire_now, color: "#4d7c0f", width: 2.0, ...common });
  if (w / (tRight - tLeft) >= 5) {
    ctx.fillStyle = "#4d7c0f";
    const i0 = Math.max(0, Math.floor(tLeft));
    const i1 = Math.min(pack.fire_now.length, Math.floor(Math.min(tRight, now)) + 1);
    for (let i = i0; i < i1; i++) {
      if (!finite(pack.fire_now[i])) continue;
      const x = xOf(i, tLeft, tRight, w);
      const y = yOfUnit(pack.fire_now[i], h);
      ctx.beginPath();
      ctx.arc(x, y, 2.0, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  if ($("policy-source").value === "oracle") {
    drawUnitSeries(ctx, {
      values: controllerProbs,
      color: "#7c3aed",
      width: 1.3,
      ...common,
      valid: () => true,
    });
  }
  drawScaleHint(
    ctx,
    h,
    "gold pre_onset   green fire_now   slate active   0..1 axis right   gap = unscored",
  );
  drawNow(ctx, h, tLeft, tRight, w, now);
}

// The map only changes when the simulation or the canvas does, so cache it: a
// whole night is ~32k seconds and redrawing it every frame is wasted work.
const miniCache = { key: "", canvas: null, dpr: 1 };

function drawMini(now) {
  const { ctx, w, h } = setupCanvas($("c-mini"));
  ctx.clearRect(0, 0, w, h);
  const key = `${pack.meta.subject_id}|${pack.duration_sec}|${simVersion}|${Math.round(w)}x${Math.round(h)}`;
  if (miniCache.key !== key) {
    miniCache.canvas = paintMiniBase(w, h);
    miniCache.key = key;
  }
  // Only the played part of the night exists yet, here as in the traces.
  const xNow = Math.max(0, Math.min(w, (now / pack.duration_sec) * w));
  const d = miniCache.dpr;
  if (xNow > 0) {
    ctx.drawImage(
      miniCache.canvas,
      0,
      0,
      xNow * d,
      h * d,
      0,
      0,
      xNow,
      h,
    );
  }
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(xNow, 0, Math.max(0, w - xNow), h);
  if (isNightClip()) drawHourTicks(ctx, w, h, xNow); // keep the hours readable ahead
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(xNow, 0, 2, h);
}

const HOUR_LABEL_Y = 11;

function drawHourTicks(ctx, w, h, fromX = 0) {
  const n = pack.duration_sec;
  ctx.strokeStyle = "rgba(15,23,42,0.18)";
  ctx.fillStyle = "rgba(100,116,139,0.95)";
  ctx.font = '9px "IBM Plex Mono", monospace';
  ctx.lineWidth = 1;
  for (let t = 3600; t < n; t += 3600) {
    const x = (t / n) * w;
    if (x < fromX) continue;
    ctx.beginPath();
    ctx.moveTo(x, HOUR_LABEL_Y);
    ctx.lineTo(x, h);
    ctx.stroke();
    ctx.fillText(`${t / 3600} h`, x + 3, HOUR_LABEL_Y - 2);
  }
}

function paintMiniBase(w, h) {
  const dpr = Math.max(1, window.devicePixelRatio || 1);
  miniCache.dpr = dpr;
  const off = document.createElement("canvas");
  off.width = Math.max(1, Math.floor(w * dpr));
  off.height = Math.max(1, Math.floor(h * dpr));
  const ctx = off.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const n = pack.duration_sec;
  const night = isNightClip();
  const xAt = (t) => (t / n) * w;

  if (!night) {
    for (const e of pack.events) {
      ctx.fillStyle = e.kind === "obstructive" ? "#d97706" : "#9333ea";
      ctx.fillRect(xAt(e.start), 4, Math.max(1.5, xAt(e.end - e.start)), h - 8);
    }
    ctx.fillStyle = "rgba(100,116,139,0.45)";
    for (const [a, b] of advanceRuns(sim.advanced)) {
      ctx.fillRect(xAt(a), 0, Math.max(1.5, xAt(b - a)), h);
    }
    for (const t of coldStartTimes()) {
      const x = xAt(t);
      ctx.fillStyle = "#0284c7";
      ctx.beginPath();
      ctx.moveTo(x - 4, 0);
      ctx.lineTo(x + 4, 0);
      ctx.lineTo(x, 7);
      ctx.closePath();
      ctx.fill();
    }
    return off;
  }

  // Night map: hours across, events on the upper lane, jaw state on the lower
  // lane, so a whole night reads as "events above, device below".
  const hourY = HOUR_LABEL_Y;
  const evTop = hourY + 1;
  const evH = Math.max(6, h * 0.42 - evTop + hourY);
  const advTop = evTop + evH + 3;
  const advH = Math.max(6, h - advTop - 2);

  ctx.fillStyle = "rgba(148,163,184,0.16)"; // scored wake: the controller is gated
  for (const [a, b] of advanceRuns(pack.wake)) {
    ctx.fillRect(xAt(a), hourY, Math.max(1, xAt(b - a)), h - hourY);
  }
  drawHourTicks(ctx, w, h);
  for (const e of pack.events) {
    ctx.fillStyle = e.kind === "obstructive" ? "#d97706" : "#9333ea";
    ctx.fillRect(xAt(e.start), evTop, Math.max(1.2, xAt(e.end - e.start)), evH);
  }
  ctx.fillStyle = "rgba(71,85,105,0.8)";
  for (const [a, b] of advanceRuns(sim.advanced)) {
    ctx.fillRect(xAt(a), advTop, Math.max(1.2, xAt(b - a)), advH);
  }
  ctx.fillStyle = "rgba(71,85,105,0.55)";
  ctx.fillText("jaw", 4, advTop + advH - 2);
  return off;
}

// Contiguous advanced runs as [start, end) second pairs.
function advanceRuns(mask) {
  const runs = [];
  let run = null;
  for (let t = 0; t <= mask.length; t++) {
    const on = t < mask.length && mask[t];
    if (on && run == null) run = t;
    else if (!on && run != null) {
      runs.push([run, t]);
      run = null;
    }
  }
  return runs;
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

// Above this the jaw cannot physically track the replay, so the board is left
// alone and the screen keeps running.
const MAX_HW_SPEED = 16;

async function maybeActuate() {
  const i = Math.min(sim.actions.length - 1, Math.max(0, Math.floor(tNow)));
  if (i === lastActionSent) return;
  if (speed > MAX_HW_SPEED) {
    lastActionSent = i;
    return;
  }
  // fire on the first second we cross an ADVANCE/RETRACT
  for (let k = lastActionSent + 1; k <= i; k++) {
    if (sim.actions[k] === ADVANCE) await sendEsp("ADVANCE");
    else if (sim.actions[k] === RETRACT) await sendEsp("RETRACT");
    else continue;
    lastPosSent = sim.actions[k] === ADVANCE ? ADVANCED : RETRACTED;
  }
  lastActionSent = i;
}

// After a jump the board is wherever the last command left it, which may not be
// where the replay now says the jaw is. Bring the hardware back in line; a single
// command is safe at any replay speed.
async function syncEspPosition() {
  if (esp.mode === "none") return;
  const i = Math.min(sim.advanced.length - 1, Math.max(0, Math.floor(tNow)));
  const want = i >= 0 && sim.advanced[i] ? ADVANCED : RETRACTED;
  if (want === lastPosSent) return;
  await sendEsp(want === ADVANCED ? "ADVANCE" : "RETRACT");
  lastPosSent = want;
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
  syncEspPosition();
}

// Opening frame: land ~45 s before the first advance so the audience sees the jaw
// move; fall back to the first cold start with enough history behind it.
function burstT() {
  if (isNightClip()) return 0; // a whole night is watched from lights out
  const firstAdvance = sim.actions.findIndex((a, i) => a === ADVANCE && i >= 45);
  if (firstAdvance >= 0) return Math.max(0, firstAdvance - 45);
  const cold = coldStartTimes();
  let preferred = cold;
  if ($("policy-source").value === "oracle") {
    const kinds = selectedKinds();
    if (kinds.size === 1 && kinds.has("obstructive")) {
      const oaCold = pack.events
        .filter((e) => e.is_cluster_first && e.kind === "obstructive")
        .map((e) => e.start)
        .sort((a, b) => a - b);
      if (oaCold.length) preferred = oaCold;
    }
  }
  const withHistory = preferred.filter((t) => t >= PRE_ROLL_SEC + 20);
  const retracted = withHistory.filter(
    (t) => !sim.advanced[Math.floor(t - PRE_ROLL_SEC)],
  );
  const pick = retracted[0] ?? withHistory[0] ?? preferred[0] ?? cold[0];
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
  if (!oracle) {
    $("oracle-note").textContent =
      "Combined model (OA + hypopnea + Unsure): kind cannot be switched.";
    return;
  }
  const kinds = selectedKinds();
  if (kinds.size === 1 && kinds.has("obstructive")) {
    $("oracle-note").textContent =
      "OA-only presentation: advances only on obstructive apnea.";
  } else {
    $("oracle-note").textContent =
      "Oracle: kind from scored annotations, not a real-time classifier.";
  }
}

function clipEntry(id) {
  return clipIndex.find((c) => c.id === id) || null;
}

// Each example ships the controller input it is meant to be shown with.
function applyPolicy(policy) {
  if (!policy) return;
  if (policy.source) $("policy-source").value = policy.source;
  if (Array.isArray(policy.kinds)) {
    $("target-oa").checked = policy.kinds.includes("obstructive");
    $("target-hyp").checked = policy.kinds.includes("hypopnea");
    $("target-unsure").checked = policy.kinds.includes("unsure");
  }
  updatePolicyUi();
}

function applyStageClass() {
  $("stage").className = `stage mode-${layoutMode}${isNightClip() ? " night" : ""}`;
}

function paintStory(entry) {
  if (!entry) {
    $("story-title").textContent = "MESA held-out clip";
    $("story-watch").textContent = "";
    $("story-num").textContent = "";
    return;
  }
  const m = entry.metrics || {};
  $("story-title").textContent = entry.title;
  $("story-watch").textContent = entry.watch || "";
  const bits = [];
  if (entry.duration_sec > 3600) {
    bits.push(`${(entry.duration_sec / 3600).toFixed(1)} h night`);
  }
  if (m.n_covered != null) bits.push(`${m.n_covered}/${m.n_linked} covered`);
  if (m.advances != null) bits.push(`${m.advances} adv`);
  if (m.fraction_advanced != null) {
    bits.push(`${Math.round(m.fraction_advanced * 100)}% vs 100% static`);
  }
  $("story-num").textContent = bits.join(" | ");
}

// Try each URL in turn; an older server process may not have the clip routes yet.
async function fetchFirstJson(urls) {
  let lastErr = null;
  for (const url of urls) {
    try {
      const r = await fetch(url);
      if (!r.ok) {
        lastErr = new Error(`${url} -> ${r.status}`);
        continue;
      }
      return await r.json();
    } catch (e) {
      lastErr = e;
    }
  }
  if (lastErr) console.warn("fetch failed:", lastErr.message || lastErr);
  return null;
}

async function loadClip(id) {
  const entry = clipEntry(id);
  const packId = entry ? entry.pack_id || entry.id : null;
  const urls = packId
    ? [`/api/pack?clip=${encodeURIComponent(packId)}`, `/data/clips/${packId}.json`, "/api/pack"]
    : ["/api/pack", "/data/pack.json"];
  const loaded = await fetchFirstJson(urls);
  if (!loaded) {
    $("story-title").textContent = "Clip failed to load";
    $("story-watch").textContent = "Check the server terminal; data/clips may be missing.";
    return;
  }
  pack = loaded;
  if (entry) $("clip").value = entry.id;
  $("pill-sub").textContent = `MESA ${pack.meta.subject_id}`;
  applyStageClass();
  if (!speedPinned) {
    // 8x is right for an 18 min clip; a whole night needs 60x to watch end to end
    speed = isNightClip() ? 60 : 8;
    $("speed").value = String(speed);
  }
  NORM.pres = robustRange(pack.pres);
  const spo2Clean = pack.spo2.filter(spo2Valid);
  const spo2Range = robustRange(spo2Clean.length ? spo2Clean : pack.spo2, 0.01, 1.0, 0.05);
  NORM.spo2 = { lo: Math.max(50, spo2Range.lo), hi: Math.min(100, spo2Range.hi) };
  applyPolicy(entry ? entry.policy : null);
  paintStory(entry);
  playing = false;
  $("btn-play").textContent = "Play";
  resim();
  tNow = burstT();
  lastActionSent = Math.floor(tNow) - 1;
  render();
  syncEspPosition();
}

async function boot() {
  // /api/clips is the normal path; the static file works with a stale server or
  // any plain file server.
  const index = await fetchFirstJson(["/api/clips", "/data/clips/index.json"]);
  clipIndex = (index && index.clips) || [];
  const sel = $("clip");
  sel.innerHTML = "";
  if (clipIndex.length) {
    for (const c of clipIndex) {
      const opt = document.createElement("option");
      opt.value = c.id;
      opt.textContent = c.title;
      sel.appendChild(opt);
    }
  } else {
    // Stay visible and say why, instead of disappearing from the top bar.
    const opt = document.createElement("option");
    opt.textContent = "library not found - run _build_clips.py";
    sel.appendChild(opt);
    sel.disabled = true;
    sel.title = "data/clips/index.json is missing, or this server predates /api/clips";
  }

  const params = new URLSearchParams(location.search);
  const wantedClip = params.get("clip");
  const startId =
    (wantedClip && clipEntry(wantedClip) && wantedClip) ||
    (index && index.default) ||
    (clipIndex[0] && clipIndex[0].id) ||
    null;
  const wantedSpeed = Number(params.get("speed"));
  if ([1, 2, 4, 8, 16, 30, 60, 120, 240].includes(wantedSpeed)) {
    speed = wantedSpeed;
    speedPinned = true;
    $("speed").value = String(wantedSpeed);
  }
  const wantedLayout = params.get("layout");
  if (wantedLayout === "overlay" || wantedLayout === "stack") {
    layoutMode = wantedLayout;
    $("layout").value = wantedLayout;
  }
  await loadClip(startId);
  const wantedT = Number(params.get("t")); // deep link to a moment in the clip
  if (Number.isFinite(wantedT) && params.get("t") !== null) {
    tNow = Math.min(pack.duration_sec, Math.max(0, wantedT));
    lastActionSent = Math.floor(tNow) - 1;
    render();
  }
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
    syncEspPosition(); // opening frame is jaw home: bring the hardware back too
  };
  $("btn-cold").onclick = () => {
    const target = nextSeekTarget(coldStartTimes());
    if (target != null) seekBefore(target);
  };
  $("btn-event").onclick = () => {
    const target = nextSeekTarget(targetStartTimes());
    if (target != null) seekBefore(target);
  };
  $("clip").onchange = (e) => loadClip(e.target.value);
  $("speed").onchange = (e) => {
    speed = Number(e.target.value);
    speedPinned = true;
    paintEspLink();
  };
  $("layout").onchange = (e) => {
    layoutMode = e.target.value;
    applyStageClass();
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
  $("btn-adv").onclick = () => {
    lastPosSent = ADVANCED;
    sendEsp("ADVANCE");
  };
  $("btn-ret").onclick = () => {
    lastPosSent = RETRACTED;
    sendEsp("RETRACT");
  };
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
