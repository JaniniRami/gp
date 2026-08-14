#!/usr/bin/env python3
"""Build a presentation clip pack from MESA cache + no-wake XGB heads."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy import signal as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mesa_mad.cache import open_night_cache  # noqa: E402
from mesa_mad.config import MesaMadConfig  # noqa: E402
from mesa_mad.controller import ActuatorConfig, BudgetConfig, MadController  # noqa: E402
from mesa_mad.dataset import load_night_bundle  # noqa: E402
from mesa_mad.features import feature_names  # noqa: E402
from mesa_mad.simulate import score_coverage  # noqa: E402

WAKE_DROP = ("cs_t_since_wake", "cs_wake_frac_10m")
NIGHT_KEY = "9d41b407b51be009"
CANDIDATES = ["2297", "3819", "3481", "0495", "1510", "1446"]
CLIP_SEC = 1080.0
DISPLAY_HZ = 32.0  # EDF excerpt rate
PACK_HZ = 16.0  # nasal pressure rate inside pack.json (browser payload)
MIN_COLD_STARTS = 2  # the demo needs several cluster-first events to seek between
OUT = Path(__file__).resolve().parent / "data"


def keep_idx_164() -> list[int]:
    names = feature_names(
        include_instability=True,
        include_flow_morph=True,
        rich_spo2=True,
        coldstart_features=True,
        include_hypnogram_wake_feats=True,  # overnight 166-d; then drop wake cols
        include_breath_features=True,
        include_abs_amplitude=True,
        include_position=True,
        include_mid_scale=True,
        include_slow_scale=True,
    )
    assert len(names) == 166, f"expected 166 overnight names, got {len(names)}"
    drop = set(names.index(w) for w in WAKE_DROP)
    ki = [i for i in range(len(names)) if i not in drop]
    assert len(ki) == 164, len(ki)
    return ki


def downsample(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if abs(fs_in - fs_out) < 1e-9:
        return x.astype(np.float32)
    n_out = int(round(x.size * fs_out / fs_in))
    if n_out < 8:
        return x.astype(np.float32)
    return sps.resample(x, n_out).astype(np.float32)


def score_night(sid: str, fire, active, ki: list[int]):
    pack_path = ROOT / "Data/mesa_runs/xgb_feat_cache" / NIGHT_KEY / "nights" / f"{sid}.npz"
    h5 = ROOT / "Data/mesa_runs/cache" / f"mesa-{sid}.h5"
    if not pack_path.is_file() or not h5.is_file():
        return None
    z = np.load(pack_path, allow_pickle=True)
    X = np.asarray(z["X"], dtype=np.float32)[:, ki]
    idxs = np.asarray(z["idxs"], dtype=np.int64)
    cfg = MesaMadConfig(earliest_lead_sec=30.0, use_clean_pres=True)
    b = load_night_bundle(h5, cfg)
    n = int(b.labels.t_sec.size)
    p_fire = np.zeros(n, dtype=np.float64)
    p_act = np.zeros(n, dtype=np.float64)
    pf = fire.predict_proba(X)[:, 1]
    pa = active.predict_proba(X)[:, 1]
    p_fire[idxs] = pf
    p_act[idxs] = pa
    return b, p_fire, p_act


def run_ctrl(probs: np.ndarray, wake: np.ndarray, thr: float) -> MadController:
    ctrl = MadController(
        act=ActuatorConfig(advance_sec=10.0, retract_sec=10.0, refractory_sec=60.0, quiet_retract_sec=90.0),
        budget=BudgetConfig(enabled=False, init_threshold=thr),
    )
    ctrl.reset(0.0)
    ctrl.threshold = thr
    for t in range(int(probs.size)):
        ctrl.step(float(t), float(probs[t]), wake=bool(wake[t]))
    return ctrl


def window_score(ann, p_fire, wake, t0: int, t1: int, thr: float) -> dict:
    ev = [e for e in ann.targets if t0 <= e.start < t1]
    if len(ev) < 3:
        return {"ok": False, "reason": "few_events"}
    n_oa = sum(1 for e in ev if e.kind == "obstructive")
    n_hyp = sum(1 for e in ev if e.kind in ("hypopnea", "unsure"))
    n_link = sum(1 for e in ev if e.arousal_linked)
    n_cold = sum(1 for e in ev if e.is_cluster_first)
    if n_link < 2:
        return {"ok": False, "reason": "few_linked"}
    if n_cold < MIN_COLD_STARTS:
        return {"ok": False, "reason": "few_cold_starts", "n_cold": n_cold}
    sl = p_fire[t0:t1]
    wk = wake[t0:t1]
    ctrl = run_ctrl(sl, wk, thr)
    mask = np.asarray(ctrl.advanced_mask, dtype=bool)
    # shift event times into clip coordinates
    from mesa_mad.annotations import RespEvent

    local = []
    for e in ev:
        local.append(
            RespEvent(
                start=e.start - t0,
                end=e.end - t0,
                kind=e.kind,
                concept=e.concept,
                arousal_start=None if e.arousal_start is None else e.arousal_start - t0,
                desat_start=e.desat_start,
                is_cluster_first=e.is_cluster_first,
                cluster_id=e.cluster_id,
            )
        )
    outs = score_coverage(local, mask, earliest_lead=30.0, actuation_lag=10.0)
    n_cov = sum(1 for o in outs if o.covered)
    n_pre = sum(1 for o in outs if o.preemptive)
    n_adv = int(ctrl.n_advances)
    frac = float(np.mean(mask)) if mask.size else 0.0
    # want a story: at least one preemptive advance, not advanced the whole time
    if n_adv < 1 or n_pre < 1 or frac > 0.55 or frac < 0.08:
        return {"ok": False, "reason": "weak_story", "n_adv": n_adv, "n_pre": n_pre, "frac": frac}
    if not (5 <= len(ev) <= 22):
        return {"ok": False, "reason": "event_count"}
    # Presentation: several separated bursts, mixed OA+hyp, one advance per burst,
    # and a clip that is not advanced most of the night.
    story = (
        n_pre * 2.0
        + n_cov * 1.2
        + n_oa * 4.0
        + min(n_hyp, 8) * 0.6
        + min(n_cold, 4) * 3.5
        - abs(frac - 0.32) * 8
        - abs(n_adv - n_cold) * 1.5
    )
    return {
        "ok": True,
        "story": story,
        "t0": t0,
        "t1": t1,
        "n_ev": len(ev),
        "n_oa": n_oa,
        "n_hyp": n_hyp,
        "n_link": n_link,
        "n_cold": n_cold,
        "n_cov": n_cov,
        "n_pre": n_pre,
        "n_adv": n_adv,
        "frac": frac,
        "p95": float(np.percentile(sl, 95)),
        "p50": float(np.median(sl)),
    }


def main() -> int:
    ki = keep_idx_164()
    fire = joblib.load(ROOT / "Data/mesa_runs/ablate_no_hypnogram_wake_v1/xgb_fire_now.joblib")
    active = joblib.load(ROOT / "Data/mesa_runs/ablate_no_hypnogram_wake_v1/xgb_active.joblib")
    thr = 0.55
    best = None
    best_payload = None

    for sid in CANDIDATES:
        print(f"scoring {sid} ...", flush=True)
        got = score_night(sid, fire, active, ki)
        if got is None:
            print(f"  skip {sid}")
            continue
        b, p_fire, p_act = got
        ann = b.cache.load_annotations()
        wake = np.asarray(b.labels.wake_mask, dtype=bool)
        n = int(p_fire.size)
        for t0 in range(900, max(901, n - int(CLIP_SEC) - 30), 30):
            t1 = t0 + int(CLIP_SEC)
            s = window_score(ann, p_fire, wake, t0, t1, thr)
            if not s.get("ok"):
                continue
            s["sid"] = sid
            if best is None or s["story"] > best["story"]:
                best = s
                best_payload = (b, p_fire, p_act, ann, wake)
                print(
                    f"  new best {sid} t0={t0} ev={s['n_ev']} oa={s['n_oa']} hyp={s['n_hyp']} "
                    f"cold={s['n_cold']} cov={s['n_cov']}/{s['n_link']} pre={s['n_pre']} "
                    f"adv={s['n_adv']} frac={s['frac']:.2f} story={s['story']:.1f}",
                    flush=True,
                )

    if best is None:
        print("no window found")
        return 1

    sid = best["sid"]
    t0, t1 = int(best["t0"]), int(best["t1"])
    b, p_fire, p_act, ann, wake = best_payload
    cache = open_night_cache(ROOT / "Data/mesa_runs/cache" / f"mesa-{sid}.h5")
    pres = cache.load_pres()
    fs_pres = float(cache.fs_pres)
    spo2 = np.asarray(cache.load_spo2(), dtype=np.float32)

    i0 = int(round(t0 * fs_pres))
    i1 = int(round(t1 * fs_pres))
    clip_pres = downsample(pres[i0:i1], fs_pres, DISPLAY_HZ)
    clip_spo2 = spo2[t0:t1].astype(np.float32)
    clip_fire = p_fire[t0:t1].astype(np.float32)
    clip_act = p_act[t0:t1].astype(np.float32)
    clip_wake = wake[t0:t1].astype(np.uint8)

    ctrl = run_ctrl(p_fire[t0:t1], wake[t0:t1], thr)
    advanced = np.asarray(ctrl.advanced_mask, dtype=np.uint8)
    actions = np.asarray(ctrl.actions, dtype=np.int8)

    def clip_ev(e, kinds_ok):
        if e.start >= t1 or e.end <= t0:
            return None
        if e.kind not in kinds_ok:
            return None
        return {
            "start": float(max(e.start, t0) - t0),
            "end": float(min(e.end, t1) - t0),
            "kind": e.kind,
            "arousal_start": None if e.arousal_start is None or not (t0 <= e.arousal_start < t1) else float(e.arousal_start - t0),
            "arousal_linked": bool(e.arousal_linked),
            "is_cluster_first": bool(e.is_cluster_first),
        }

    events = [x for e in ann.targets if (x := clip_ev(e, {"obstructive", "hypopnea", "unsure"})) is not None]
    arousals = []
    for a, c in ann.arousals:
        if c <= t0 or a >= t1:
            continue
        arousals.append({"start": float(max(a, t0) - t0), "end": float(min(c, t1) - t0)})

    OUT.mkdir(parents=True, exist_ok=True)
    npz_path = OUT / "mesa_clip.npz"
    np.savez_compressed(
        npz_path,
        pres=clip_pres,
        spo2=clip_spo2,
        fire_now=clip_fire,
        active=clip_act,
        wake=clip_wake,
        advanced=advanced,
        actions=actions,
        fs_pres=np.float32(DISPLAY_HZ),
        fs_dec=np.float32(1.0),
    )

    meta = {
        "subject_id": sid,
        "source": "MESA Sleep, NSRR Compumedics",
        "split": "test (subject held-out)",
        "clip_start_sec": t0,
        "clip_end_sec": t1,
        "duration_sec": CLIP_SEC,
        "fs_pres": DISPLAY_HZ,
        "fs_decision": 1.0,
        "geometry": {
            "task": "fire_now deadline-based MAD actuation",
            "actuation_lag_sec": 10.0,
            "earliest_lead_sec": 30.0,
            "refractory_sec": 60.0,
            "quiet_retract_sec": 90.0,
            "events": "obstructive apnea + hypopnea + Unsure (MESA hyp>=30%)",
        },
        "model": {
            "head": "fire_now",
            "pack": "no hypnogram-wake (deployable)",
            "n_features": 164,
            "test_auroc_fire_now": 0.8512,
            "weights": "models/xgb_fire_now.joblib",
        },
        "controller": {
            "threshold": thr,
            "n_advances": int(ctrl.n_advances),
            "fraction_advanced": float(np.mean(advanced)),
            "n_events": len(events),
            "n_oa": best["n_oa"],
            "n_hyp": best["n_hyp"],
            "n_covered_linked": best["n_cov"],
            "n_preemptive": best["n_pre"],
        },
        "disclaimer": (
            "Untreated PSG: coverage is timing (advance completed before arousal deadline), "
            "not proven event or arousal prevention."
        ),
        "events": events,
        "arousals": arousals,
        "window_search": {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in best.items()},
    }
    (OUT / "clip_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    pack_pres = downsample(clip_pres, DISPLAY_HZ, PACK_HZ)
    pack = {
        "meta": {k: v for k, v in meta.items() if k not in ("events", "arousals")},
        "fs_pres": PACK_HZ,
        "fs_decision": 1.0,
        "duration_sec": int(CLIP_SEC),
        "pres": [round(float(v), 4) for v in pack_pres],
        "spo2": [round(float(v), 2) for v in clip_spo2],
        "fire_now": [round(float(v), 4) for v in clip_fire],
        "active": [round(float(v), 4) for v in clip_act],
        "wake": [int(v) for v in clip_wake],
        "advanced_default": [int(v) for v in advanced],
        "actions_default": [int(v) for v in actions],
        "events": events,
        "arousals": arousals,
    }
    pack_path = OUT / "pack.json"
    pack_path.write_text(json.dumps(pack) + "\n")

    edf_path = write_edf(OUT, sid, clip_pres, clip_spo2, t0, t1)

    print("wrote", npz_path, "bytes", npz_path.stat().st_size)
    print("wrote", pack_path, "bytes", pack_path.stat().st_size)
    if edf_path is not None:
        print("wrote", edf_path, "bytes", edf_path.stat().st_size)
    n_cold = sum(1 for e in events if e["is_cluster_first"])
    print(f"cold starts in clip: {n_cold} at {[round(e['start'], 1) for e in events if e['is_cluster_first']]}")
    print("meta", json.dumps({k: meta[k] for k in ("subject_id", "clip_start_sec", "controller")}, indent=2))
    return 0


def write_edf(out_dir: Path, sid: str, pres: np.ndarray, spo2: np.ndarray, t0: int, t1: int):
    """Two-channel EDF excerpt (Pres + SpO2) so the clip is inspectable in any viewer."""
    try:
        import pyedflib
    except ImportError:
        print("pyedflib missing: skipping EDF excerpt", flush=True)
        return None
    path = out_dir / f"mesa-sleep-{sid}_clip.edf"
    w = pyedflib.EdfWriter(str(path), 2, file_type=pyedflib.FILETYPE_EDFPLUS)
    try:
        w.setTechnician("ProactMAD_demo")
        w.setRecordingAdditional(f"MESA-{sid}_excerpt_{t0}-{t1}s")
        pres_lim = round(float(max(1e-3, np.abs(pres).max())), 3)
        w.setSignalHeader(
            0,
            {
                "label": "Pres",
                "dimension": "uV",
                "sample_frequency": DISPLAY_HZ,
                "physical_min": -pres_lim,
                "physical_max": pres_lim,
                "digital_min": -32768,
                "digital_max": 32767,
                "transducer": "nasal pressure",
                "prefilter": "",
            },
        )
        w.setSignalHeader(
            1,
            {
                "label": "SpO2",
                "dimension": "%",
                "sample_frequency": 1.0,
                "physical_min": 50.0,
                "physical_max": 100.0,
                "digital_min": -32768,
                "digital_max": 32767,
                "transducer": "pulse oximetry",
                "prefilter": "",
            },
        )
        w.writeSamples([np.asarray(pres, dtype=np.float64), np.asarray(spo2, dtype=np.float64)])
    finally:
        w.close()
    return path


if __name__ == "__main__":
    raise SystemExit(main())
