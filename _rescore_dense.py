#!/usr/bin/env python3
"""Rescore shipped demo packs on a deployable 1 Hz grid.

Fills legacy Unsure/central pads that were stored as fake zeros, adds the
pre_onset head, and re-runs the 3-head fusion controller. Existing clip
windows are kept; only the scores and default MAD trajectory change.
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
import _build_night as bn  # noqa: E402


def story_for_pack(pack: dict) -> bc.Story | None:
    """The Story whose template regenerates this pack's headline sentence."""
    sid = str(pack["meta"]["subject_id"])
    cid = pack["meta"].get("story", {}).get("id")
    if cid == bn.STORY.id:
        return bn.STORY
    if cid == bn.MILD_STORY.id:
        return bn.story_for(sid, bn.MILD_STORY, bn.nsrr_ahi(sid))
    return {s.id: s for s in bc.STORIES}.get(cid)


def patch_pack(path: Path, night: bc.Night) -> tuple[dict, str | None]:
    pack = json.loads(path.read_text())
    dur = int(pack["duration_sec"])
    clip_start = int(pack["meta"]["clip_start_sec"])
    if night.p_fire.size == dur:
        t0, t1 = 0, dur
    else:
        t0, t1 = clip_start, clip_start + dur
    sl = slice(t0, t1)
    scored = night.scored[sl]
    pre, fire, act = night.p_pre[sl], night.p_fire[sl], night.p_act[sl]
    wake = night.wake[sl]
    ctrl = bc.run_ctrl_fused(pre, fire, act, scored, wake)
    pack["fire_now"] = bc.prob_json(fire)
    pack["pre_onset"] = bc.prob_json(pre)
    pack["active"] = bc.prob_json(act)
    pack["scored"] = [int(v) for v in scored.astype(np.uint8)]
    pack["advanced_default"] = [int(v) for v in ctrl.advanced_mask]
    pack["actions_default"] = [int(v) for v in ctrl.actions]
    stats = bc.window_stats(night, t0, t1)
    watch = None
    if stats.get("ok"):
        m = bc.metrics_from(stats, "model")
        pack["meta"]["metrics"]["model"] = m
        story = story_for_pack(pack)
        if story is not None:
            watch = bc.fill_watch(story, m)
            pack["meta"]["story"]["watch"] = watch
        pack["meta"]["controller"] = {
            "threshold": bc.THR,
            "n_advances": int(ctrl.n_advances),
            "fraction_advanced": float(np.mean(ctrl.advanced_mask)),
            "fusion": "fire_now primary; pre_onset early-warn; active hold/rescue",
            "frac_scored": round(float(np.mean(scored)), 4),
        }
    pack["meta"]["model"] = {
        "heads": ["pre_onset", "fire_now", "active"],
        "primary_trigger": "fire_now",
        "pack": "no hypnogram-wake (deployable), 1 Hz every cannula-valid second after 600 s lookback",
        "n_features": 164,
        "test_auroc_fire_now": 0.8512,
        "weights": {
            "pre_onset": "models/xgb_pre_onset.joblib",
            "fire_now": "models/xgb_fire_now.joblib",
            "active": "models/xgb_active.joblib",
        },
        "fusion": (
            "Advance only if fire_now is above threshold (early-warn when pre_onset "
            "also high; rescue when active is also high). active-only holds if already "
            "advanced. pre_onset without fire_now does not actuate."
        ),
        "frac_scored": round(float(np.mean(scored)), 4),
        "n_unscored": int((~scored).sum()),
    }
    path.write_text(json.dumps(pack) + "\n")
    print(
        f"wrote {path.name}: scored {float(np.mean(scored)):.1%}  "
        f"adv={ctrl.n_advances} frac={np.mean(ctrl.advanced_mask):.2f}",
        flush=True,
    )
    return (pack["meta"]["metrics"]["model"] if stats.get("ok") else {}), watch


def _vec(values) -> np.ndarray:
    return np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)


def night_stub_from_pack(pack: dict) -> bc.Night:
    """Clip-relative Night so the controller can be re-run without MESA caches."""
    from mesa_mad.annotations import RespEvent

    events = [
        RespEvent(
            start=e["start"],
            end=e["end"],
            kind=e["kind"],
            concept=e["kind"],
            arousal_start=e.get("arousal_start"),
            is_cluster_first=bool(e.get("is_cluster_first", False)),
        )
        for e in pack["events"]
    ]
    spo2 = pack.get("spo2")
    return bc.Night(
        sid=str(pack["meta"]["subject_id"]),
        bundle=None,
        p_fire=_vec(pack["fire_now"]),
        p_act=_vec(pack["active"]),
        p_pre=_vec(pack["pre_onset"]),
        scored=np.array(pack["scored"], dtype=bool),
        wake=np.array(pack["wake"], dtype=bool),
        events=events,
        spo2=None if spo2 is None else np.asarray(spo2, dtype=np.float32),
    )


def reapply_controller_only() -> int:
    index = json.loads((bc.CLIP_DIR / "index.json").read_text())
    seen: set[Path] = set()
    for clip in index["clips"]:
        cid = clip.get("pack_id") or clip["id"]
        path = bc.CLIP_DIR / f"{cid}.json"
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        pack = json.loads(path.read_text())
        m, watch = patch_pack(path, night_stub_from_pack(pack))
        for row in index["clips"]:
            if row.get("id") != path.stem or not m:
                continue
            row["metrics"] = m
            if watch:
                row["watch"] = watch
    (bc.CLIP_DIR / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return 0


def main() -> int:
    ki = bc.keep_idx_164()
    fire = joblib.load(bc.MODEL_DIR / "xgb_fire_now.joblib")
    active = joblib.load(bc.MODEL_DIR / "xgb_active.joblib")
    pre = joblib.load(bc.MODEL_DIR / "xgb_pre_onset.joblib")
    index = json.loads((bc.CLIP_DIR / "index.json").read_text())
    by_sid: dict[str, list[Path]] = {}
    for clip in index["clips"]:
        cid = clip["pack_id"] if clip.get("pack_id") else clip["id"]
        path = bc.CLIP_DIR / f"{cid}.json"
        if not path.is_file():
            continue
        sid = str(clip.get("subject_id") or json.loads(path.read_text())["meta"]["subject_id"])
        by_sid.setdefault(sid, []).append(path)

    for sid, paths in by_sid.items():
        ranges: list[tuple[int, int]] = []
        unique_paths: list[Path] = []
        seen: set[Path] = set()
        for path in paths:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            unique_paths.append(path)
            pack = json.loads(path.read_text())
            t0 = int(pack["meta"]["clip_start_sec"])
            ranges.append((t0, t0 + int(pack["duration_sec"])))
        print(f"dense-scoring MESA {sid} for {len(unique_paths)} pack(s)", flush=True)
        night = bc.load_night(
            sid, fire, active, ki, pre=pre, dense=True, dense_ranges=ranges
        )
        if night is None:
            print(f"  skip {sid}: load failed", flush=True)
            continue
        print(
            f"  scored {float(np.mean(night.scored)):.1%} of {night.scored.size}s "
            f"(legacy pack holes filled)",
            flush=True,
        )
        for path in unique_paths:
            m, watch = patch_pack(path, night)
            for clip in index["clips"]:
                if clip.get("id") != path.stem or not m:
                    continue
                clip["metrics"] = m
                if watch:
                    clip["watch"] = watch

    (bc.CLIP_DIR / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    default = bc.CLIP_DIR / "full_night.json"
    if default.is_file():
        # keep data/pack.json as the flagship 18 min clip, not the night
        pass
    return 0


if __name__ == "__main__":
    if "--controller-only" in sys.argv:
        raise SystemExit(reapply_controller_only())
    raise SystemExit(main())
