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
let tNow = 0;
let playing = false;
let lastTs = 0;
let winSec = 60;
let speed = 8;
let lastActionSent = -1;
let espWriter = null;
let espFallback = false;

async function sendEsp(cmd) {
  if (espWriter) {
    const enc = new TextEncoder();
    await espWriter.write(enc.encode(cmd + "\n"));
    return;
  }
  if (espFallback) {
    await fetch("/api/esp/cmd", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cmd }),
    }).catch(() => {});
  }
}

function resim() {
  const thr = Number($("thr").value) / 100;
  ctrl.threshold = thr;
  sim = ctrl.simulate(pack.fire_now, pack.wake);
  $("m-thr").textContent = thr.toFixed(2);
  $("thr-lab").textContent = thr.toFixed(2);
  lastActionSent = -1;
  updateHud();
}

function coverageNow() {
  const n = Math.max(1, Math.floor(tNow) + 1);
  const adv = sim.advanced.slice(0, n);
  const evs = pack.events.filter((e) => e.start <= tNow + 1);
  const linked = evs.filter((e) => e.arousal_linked);
  const cov = linked.filter((e) => eventCovered(e, sim.advanced)).length;
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
  const p = pack.fire_now[i] ?? 0;
  $("m-p").textContent = p.toFixed(2);
  $("m-adv").textContent = String(sim.nAdvances);
  const frac = sim.advanced.length
    ? sim.advanced.slice(0, i + 1).reduce((a, b) => a + b, 0) / (i + 1)
    : 0;
  $("m-frac").textContent = `${Math.round(frac * 100)}%`;
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
  const x = pos === RETRACTED ? 28 : pos === ADVANCING ? 90 : pos === ADVANCED ? 170 : 90;
  jaw.setAttribute("x", String(x));
  jaw.setAttribute("fill", pos === RETRACTED ? "#64748b" : "#c4f25a");
  $("clock").textContent = `${fmt(tNow)} / ${fmt(pack.duration_sec)}`;
  $("now-air").textContent = `t = ${fmt(tNow)}`;
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

function drawEvents(ctx, h, tLeft, tRight, w) {
  for (const e of pack.events) {
    if (e.end < tLeft || e.start > tRight) continue;
    const x0 = xOf(e.start, tLeft, tRight, w);
    const x1 = xOf(e.end, tLeft, tRight, w);
    ctx.fillStyle = e.kind === "obstructive" ? "rgba(255,176,32,0.28)" : "rgba(192,132,252,0.28)";
    ctx.fillRect(x0, 0, Math.max(2, x1 - x0), h);
    ctx.fillStyle = e.kind === "obstructive" ? "#ffb020" : "#c084fc";
    ctx.font = "11px IBM Plex Sans";
    ctx.fillText(e.kind === "obstructive" ? "OA" : "HYP", x0 + 4, 16);
  }
  ctx.strokeStyle = "#ffe08a";
  ctx.lineWidth = 2;
  for (const a of pack.arousals) {
    if (a.start < tLeft || a.start > tRight) continue;
    const x = xOf(a.start, tLeft, tRight, w);
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
}

function drawAdvanced(ctx, h, tLeft, tRight, w, now, withVeil = false) {
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
  if (!withVeil) return;
  const i = Math.min(n - 1, Math.max(0, Math.floor(now)));
  const note = $("veil-note");
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
  drawEvents(ctx, h, tLeft, tRight, w);
  drawAdvanced(ctx, h, tLeft, tRight, w, now, true);
  const fs = pack.fs_pres;
  const i0 = Math.max(0, Math.floor(tLeft * fs));
  const i1 = Math.min(pack.pres.length, Math.ceil(tRight * fs));
  let mn = 1e9, mx = -1e9;
  for (let i = i0; i < i1; i++) {
    const v = pack.pres[i];
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  const pad = Math.max(0.05, (mx - mn) * 0.15);
  mn -= pad; mx += pad;
  ctx.beginPath();
  ctx.strokeStyle = "#5ec8ff";
  ctx.lineWidth = 1.4;
  for (let i = i0; i < i1; i++) {
    const t = i / fs;
    const x = xOf(t, tLeft, tRight, w);
    const y = h - ((pack.pres[i] - mn) / (mx - mn)) * (h - 8) - 4;
    if (i === i0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  drawNow(ctx, h, tLeft, tRight, w, now);
}

function drawSpo2(tLeft, tRight, now) {
  const { ctx, w, h } = setupCanvas($("c-spo2"));
  ctx.clearRect(0, 0, w, h);
  drawGrid(ctx, w, h);
  drawEvents(ctx, h, tLeft, tRight, w);
  drawAdvanced(ctx, h, tLeft, tRight, w, now);
  const i0 = Math.max(0, Math.floor(tLeft));
  const i1 = Math.min(pack.spo2.length, Math.ceil(tRight) + 1);
  ctx.beginPath();
  ctx.strokeStyle = "#ff6b8a";
  ctx.lineWidth = 2;
  for (let i = i0; i < i1; i++) {
    const x = xOf(i, tLeft, tRight, w);
    const y = h - ((pack.spo2[i] - 85) / 15) * (h - 10) - 5;
    if (i === i0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
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
  const i1 = Math.min(pack.fire_now.length, Math.ceil(tRight) + 1);
  ctx.beginPath();
  ctx.fillStyle = "rgba(196,242,90,0.18)";
  ctx.moveTo(xOf(i0, tLeft, tRight, w), h);
  for (let i = i0; i < i1; i++) {
    const y = h - pack.fire_now[i] * (h - 8) - 4;
    ctx.lineTo(xOf(i, tLeft, tRight, w), y);
  }
  ctx.lineTo(xOf(i1, tLeft, tRight, w), h);
  ctx.closePath();
  ctx.fill();
  ctx.beginPath();
  ctx.strokeStyle = "#c4f25a";
  ctx.lineWidth = 1.8;
  for (let i = i0; i < i1; i++) {
    const y = h - pack.fire_now[i] * (h - 8) - 4;
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
  ctx.fillStyle = "#f8fafc";
  ctx.fillRect((now / n) * w, 0, 2, h);
}

function render() {
  if (!pack) return;
  const tRight = tNow + winSec * 0.18;
  const tLeft = tRight - winSec;
  drawAir(tLeft, tRight, tNow);
  drawSpo2(tLeft, tRight, tNow);
  drawModel(tLeft, tRight, tNow);
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

function burstT() {
  const first = pack.events.reduce((m, e) => Math.min(m, e.start), 1e9);
  return Math.max(0, first - 45);
}

async function connectEsp() {
  if (navigator.serial) {
    try {
      const port = await navigator.serial.requestPort();
      await port.open({ baudRate: 115200 });
      espWriter = port.writable.getWriter();
      $("pill-esp").textContent = "ESP Web Serial";
      $("pill-esp").classList.add("on");
      return;
    } catch (e) {
      console.warn(e);
    }
  }
  const st = await fetch("/api/esp/status").then((r) => r.json()).catch(() => null);
  if (st && st.linked) {
    espFallback = true;
    $("pill-esp").textContent = `ESP ${st.port || "serial"}`;
    $("pill-esp").classList.add("on");
  } else {
    alert("No ESP. Use Chrome/Edge for Web Serial, or start: python server.py --esp auto");
  }
}

async function boot() {
  pack = await fetch("/api/pack").then((r) => r.json());
  $("pill-sub").textContent = `MESA ${pack.meta.subject_id}`;
  $("pill-geo").textContent = "A=10s | lead=30s | no scored wake";
  tNow = burstT();
  resim();
  render();
  requestAnimationFrame(tick);

  $("btn-play").onclick = () => {
    playing = !playing;
    $("btn-play").textContent = playing ? "Pause" : "Play";
  };
  $("btn-reset").onclick = () => {
    playing = false;
    $("btn-play").textContent = "Play";
    tNow = 0;
    lastActionSent = -1;
    render();
  };
  $("btn-burst").onclick = () => {
    tNow = burstT();
    lastActionSent = Math.floor(tNow) - 1;
    render();
  };
  $("speed").onchange = (e) => {
    speed = Number(e.target.value);
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
  $("btn-esp").onclick = connectEsp;
  window.addEventListener("resize", render);
  fetch("/api/esp/status")
    .then((r) => r.json())
    .then((st) => {
      if (st.linked) {
        espFallback = true;
        $("pill-esp").textContent = `ESP ${st.port || "serial"}`;
        $("pill-esp").classList.add("on");
      }
    })
    .catch(() => {});
}

boot();
