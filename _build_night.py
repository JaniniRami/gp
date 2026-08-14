#!/usr/bin/env python3
"""Build a whole-night replay clip from the best-performing held-out MESA night.

The 18 min story clips in `_build_clips.py` prove single points. This one answers
"what does the device do across a whole night": every advance, every retract,
every wake gate, on one timeline.

Ranking is over all held-out nights with the deployable no-wake fire_now head and
the same controller the browser runs. The winner is the night with the best
arousal-linked coverage per unit of jaw time; the cohort distribution is printed
and stored so the clip can be described honestly as the best of N, not as typical.

Outputs
  data/clips/full_night.json   whole-night pack (Pres at 8 Hz to keep it loadable)
  data/clips/index.json        merged: story clips are preserved

Run `_build_clips.py` first (it builds the story clips), then this.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _build_clips as bc  # noqa: E402
from mesa_mad.cache import open_night_cache  # noqa: E402
from mesa_mad.splits import hash_stable_split  # noqa: E402

NIGHT_PACK_HZ = 8.0  # a whole night at 16 Hz is a needlessly large browser payload
WAKE_PAD_SEC = 300  # keep a few minutes of lights-on either side of scored sleep
MIN_NIGHT_SEC = 4 * 3600
MAX_NIGHT_SEC = 11 * 3600

STORY = bc.Story(
    id="full_night",
    title="Whole night, best held-out subject",
    watch=(
        "Every advance and retract across the night: {n_covered}/{n_linked} "
        "arousal-linked events covered with {advances} advances, jaw forward "
        "{pct_advanced}% of the night against 100% for a fixed MAD."
    ),
    policy={"source": "model", "kinds": bc.MODEL_KINDS},
    score=lambda s: None,
    order=-1,
)


def sleep_span(wake: np.ndarray) -> tuple[int, int]:
    asleep = np.flatnonzero(~np.asarray(wake, dtype=bool))
    if asleep.size < 600:
        return 0, int(wake.size)
    t0 = max(0, int(asleep[0]) - WAKE_PAD_SEC)
    t1 = min(int(wake.size), int(asleep[-1]) + WAKE_PAD_SEC)
    return t0, t1


def night_quality(pres: np.ndarray, fs: float) -> dict:
    """Sensor dropout tolerance for a whole night is looser than for an 18 min clip."""
    q = bc.pres_quality(pres, fs)
    q["ok"] = q["flat_frac"] < 0.08 and q["flat_run"] <= 120
    return q


def spo2_quality(spo2: np.ndarray) -> dict:
    """A night whose pulse ox drops out for hours cannot carry the SpO2 lane."""
    valid = (spo2 >= 50.0) & (spo2 <= 100.0)
    gaps, run = [], 0
    for ok in valid:
        if ok:
            if run:
                gaps.append(run)
            run = 0
        else:
            run += 1
    if run:
        gaps.append(run)
    longest = max(gaps) if gaps else 0
    frac = float(np.mean(valid))
    return {"valid_frac": frac, "gap": int(longest), "ok": frac >= 0.97 and longest <= 300}


def night_score(s: dict) -> tuple[float | None, str]:
    """Coverage bought with as little jaw time as possible."""
    # a whole-night replay needs the advance/retract cycle to repeat often enough
    # that the audience sees it happen many times, not a quiet night
    if s["n_linked"] < 25:
        return None, f"few_linked({s['n_linked']})"
    if s["coverage"] < 0.85:
        return None, f"coverage({s['coverage']:.2f})"
    if s["frac"] > 0.40:
        return None, f"frac({s['frac']:.2f})"
    # MESA is hypopnea-dominated; still refuse a night with no obstructive apnea at
    # all, so the OA story can be told on the same recording
    if s["n_oa"] < 3:
        return None, f"few_oa({s['n_oa']})"
    if not s["starts_retracted"]:
        return None, "starts_advanced"
    # MESA nights carry a lot of scored WASO; only reject a span that is mostly wake
    if s["wake_frac"] > 0.55:
        return None, f"wake({s['wake_frac']:.2f})"
    # coverage first, then jaw time, then prefer a night with real sleep, enough
    # events that the audience sees the cycle repeat, and visible desaturation
    score = (
        10.0 * s["coverage"]
        - 6.0 * s["frac"]
        - 2.0 * s["wake_frac"]
        + 0.04 * min(s["n_linked"], 60)
        + 0.05 * min(max(94.0 - s["spo2_min"], 0.0), 15.0)
        + 0.03 * min(s["n_oa"], 20)
    )
    return score, "ok"


def build(night: bc.Night, stats: dict, cohort: dict) -> dict:
    sid, t0, t1 = night.sid, stats["t0"], stats["t1"]
    cache = open_night_cache(bc.H5_DIR / f"mesa-{sid}.h5")
    fs_pres = float(cache.fs_pres)
    pres = cache.load_pres()[int(round(t0 * fs_pres)) : int(round(t1 * fs_pres))]
    pack_pres = bc.downsample(pres, fs_pres, NIGHT_PACK_HZ)

    ctrl = bc.run_ctrl(night.p_fire[t0:t1], night.wake[t0:t1])
    advanced = np.asarray(ctrl.advanced_mask, dtype=np.uint8)
    actions = np.asarray(ctrl.actions, dtype=np.int8)

    ev_local = bc.shift_events(night.events, t0, t1)
    events = [
        {
            "start": float(e.start),
            "end": float(e.end),
            "kind": e.kind,
            "arousal_start": None if e.arousal_start is None else float(e.arousal_start),
            "arousal_linked": bool(e.arousal_linked),
            "is_cluster_first": bool(e.is_cluster_first),
        }
        for e in ev_local
    ]
    ann = night.bundle.cache.load_annotations()
    arousals = [
        {"start": float(max(a, t0) - t0), "end": float(min(c, t1) - t0)}
        for a, c in ann.arousals
        if c > t0 and a < t1
    ]

    m_model = bc.metrics_from(stats, "model")
    watch = bc.fill_watch(STORY, m_model)
    meta = {
        "subject_id": sid,
        "source": "MESA Sleep, NSRR Compumedics",
        "split": "test (subject held-out)",
        "clip_start_sec": t0,
        "clip_end_sec": t1,
        "duration_sec": t1 - t0,
        "fs_pres": NIGHT_PACK_HZ,
        "fs_decision": 1.0,
        "story": {
            "id": STORY.id,
            "title": STORY.title,
            "watch": watch,
            "policy": STORY.policy,
        },
        "geometry": {
            "task": "fire_now deadline-based MAD actuation",
            "actuation_lag_sec": bc.LAG,
            "earliest_lead_sec": bc.LEAD,
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
        "metrics": {"model": m_model, "oracle_oa": None},
        "cohort": cohort,
        "signal_quality": {
            "spo2_valid_frac": stats.get("spo2_valid_frac"),
            "spo2_longest_gap_sec": stats.get("spo2_gap"),
            "wake_frac": round(float(stats["wake_frac"]), 3),
        },
        "controller": {
            "threshold": bc.THR,
            "n_advances": int(ctrl.n_advances),
            "fraction_advanced": float(np.mean(advanced)),
        },
        "disclaimer": (
            "Untreated PSG: coverage is timing (advance completed before the arousal "
            "deadline), not proven event or arousal prevention. Best of "
            f"{cohort.get('n_nights_scored', '?')} held-out nights scored end to end "
            f"({cohort.get('n_nights_ranked', '?')} met the demo gates); this is a "
            "selected night, not a cohort average."
        ),
    }

    pack = {
        "meta": meta,
        "fs_pres": NIGHT_PACK_HZ,
        "fs_decision": 1.0,
        "duration_sec": int(t1 - t0),
        "pres": [round(float(v), 3) for v in pack_pres],
        "spo2": [round(float(v), 1) for v in night.spo2[t0:t1]],
        "fire_now": [round(float(v), 3) for v in night.p_fire[t0:t1]],
        "active": [round(float(v), 3) for v in night.p_act[t0:t1]],
        "wake": [int(v) for v in night.wake[t0:t1].astype(np.uint8)],
        "advanced_default": [int(v) for v in advanced],
        "actions_default": [int(v) for v in actions],
        "events": events,
        "arousals": arousals,
    }
    bc.CLIP_DIR.mkdir(parents=True, exist_ok=True)
    out = bc.CLIP_DIR / "full_night.json"
    out.write_text(json.dumps(pack) + "\n")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)
    return pack


def merge_index(entry: dict) -> None:
    path = bc.CLIP_DIR / "index.json"
    doc = json.loads(path.read_text()) if path.is_file() else {"clips": []}
    clips = [c for c in doc.get("clips", []) if c["id"] != entry["id"]]
    clips.insert(0, entry)  # the night is the headline example
    doc["clips"] = clips
    doc["default"] = entry["id"]
    path.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"index.json now lists {len(clips)} clips, default={doc['default']}", flush=True)


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    ki = bc.keep_idx_164()
    fire = joblib.load(bc.MODEL_DIR / "xgb_fire_now.joblib")
    active = joblib.load(bc.MODEL_DIR / "xgb_active.joblib")

    sids = sorted(p.stem for p in bc.NPZ_DIR.glob("*.npz"))
    test = [
        s
        for s in hash_stable_split(sids).test
        if (bc.H5_DIR / f"mesa-{s}.h5").is_file()
    ]
    if limit:
        test = test[:limit]
    print(f"ranking {len(test)} held-out nights", flush=True)

    rows: list[dict] = []
    for k, sid in enumerate(test, 1):
        try:
            night = bc.load_night(sid, fire, active, ki)
        except Exception as exc:  # noqa: BLE001
            print(f"[{k}/{len(test)}] {sid} load failed: {exc}", flush=True)
            continue
        if night is None or len(night.events) < 10:
            continue
        t0, t1 = sleep_span(night.wake)
        if not (MIN_NIGHT_SEC <= t1 - t0 <= MAX_NIGHT_SEC):
            print(f"[{k}/{len(test)}] {sid}: span {t1 - t0}s skipped", flush=True)
            continue
        st = bc.window_stats(night, t0, t1)
        if not st.get("ok"):
            continue
        sq = spo2_quality(night.spo2[t0:t1])
        st.update(spo2_valid_frac=round(sq["valid_frac"], 3), spo2_gap=sq["gap"])
        sc, why = night_score(st)
        if sc is not None and not sq["ok"]:
            sc, why = None, f"spo2({sq['valid_frac']:.2f},gap={sq['gap']}s)"
        rows.append({**st, "score": sc})
        print(
            f"[{k}/{len(test)}] {sid}: {(t1 - t0) / 3600:.1f} h cov="
            f"{st['n_cov']}/{st['n_linked']} frac={st['frac']:.2f} adv={st['n_adv']} "
            f"wake={st['wake_frac']:.2f} "
            f"score={'-' if sc is None else f'{sc:.2f}'} ({why})",
            flush=True,
        )
        del night

    if not rows:
        print("no nights ranked")
        return 1
    ranked = sorted(
        (r for r in rows if r["score"] is not None), key=lambda r: -r["score"]
    )
    covs = np.array([r["coverage"] for r in rows])
    fracs = np.array([r["frac"] for r in rows])
    cohort = {
        "n_nights_scored": len(rows),
        "n_nights_ranked": len(ranked),
        "median_coverage": round(float(np.median(covs)), 3),
        "median_fraction_advanced": round(float(np.median(fracs)), 3),
        "threshold": bc.THR,
        "note": (
            "Cohort context for the selected night: median over all held-out nights "
            "scored end to end at this threshold."
        ),
    }
    print("cohort:", json.dumps(cohort), flush=True)

    for r in ranked[:10]:
        print(
            f"  cand {r['sid']} cov={r['coverage']:.2f} frac={r['frac']:.2f} "
            f"adv={r['n_adv']} oa={r['n_oa']} spo2min={r['spo2_min']:.0f} "
            f"spo2ok={r['spo2_valid_frac']:.2f} score={r['score']:.2f}",
            flush=True,
        )

    for r in ranked:
        night = bc.load_night(r["sid"], fire, active, ki)
        if night is None:
            continue
        cache = open_night_cache(bc.H5_DIR / f"mesa-{r['sid']}.h5")
        fs = float(cache.fs_pres)
        seg = cache.load_pres()[int(r["t0"] * fs) : int(r["t1"] * fs)]
        q = night_quality(seg, fs)
        if not q["ok"]:
            print(
                f"  {r['sid']} rejected: dead={q['flat_frac']:.3f} run={q['flat_run']}s",
                flush=True,
            )
            del night
            continue
        pack = build(night, r, cohort)
        m = pack["meta"]["metrics"]["model"]
        merge_index(
            {
                "id": STORY.id,
                "pack_id": STORY.id,
                "title": STORY.title,
                "watch": pack["meta"]["story"]["watch"],
                "policy": STORY.policy,
                "subject_id": r["sid"],
                "clip_start_sec": r["t0"],
                "duration_sec": r["t1"] - r["t0"],
                "metrics": m,
                "metrics_oracle_oa": None,
                "cohort": cohort,
            }
        )
        print(
            f"night clip: MESA {r['sid']} {(r['t1'] - r['t0']) / 3600:.1f} h "
            f"cov={m['n_covered']}/{m['n_linked']} adv={m['advances']} "
            f"frac={m['fraction_advanced']} spo2min={m['spo2_min']}",
            flush=True,
        )
        return 0

    print("every ranked night failed the quality check")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
