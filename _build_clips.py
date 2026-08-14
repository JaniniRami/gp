#!/usr/bin/env python3
"""Build a library of presentation clips from held-out MESA nights.

One clip per "story": each is an 18 min window from a test-split subject where
the deployable no-wake fire_now head + the MAD controller produce a result that
is worth showing on a projector (advance completed before the arousal deadline,
device retracted most of the night, one advance holding a whole burst, ...).

Outputs
  data/clips/<story_id>.json   full replay pack (same schema as data/pack.json)
  data/clips/<story_id>.edf    2-channel excerpt (Pres + SpO2) for inspection
  data/clips/index.json        list of clips with story text + headline metrics
  data/pack.json               copy of the default clip (legacy single-clip path)

Selection is over the MODEL probabilities, never the oracle: the clips have to
show the model working, not the annotations.
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from scipy import signal as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mesa_mad.annotations import RespEvent  # noqa: E402
from mesa_mad.cache import open_night_cache  # noqa: E402
from mesa_mad.config import MesaMadConfig  # noqa: E402
from mesa_mad.controller import ActuatorConfig, BudgetConfig, MadController  # noqa: E402
from mesa_mad.dataset import load_night_bundle  # noqa: E402
from mesa_mad.features import feature_names  # noqa: E402
from mesa_mad.simulate import score_coverage  # noqa: E402
from mesa_mad.splits import hash_stable_split  # noqa: E402

WAKE_DROP = ("cs_t_since_wake", "cs_wake_frac_10m")
NIGHT_KEY = "9d41b407b51be009"
MODEL_DIR = Path(__file__).resolve().parent / "models"
if not (MODEL_DIR / "xgb_fire_now.joblib").is_file():
    MODEL_DIR = ROOT / "Data/mesa_runs/ablate_no_hypnogram_wake_v1"
NPZ_DIR = ROOT / "Data/mesa_runs/xgb_feat_cache" / NIGHT_KEY / "nights"
H5_DIR = ROOT / "Data/mesa_runs/cache"

CLIP_SEC = 1080
STEP_SEC = 30
DISPLAY_HZ = 32.0  # EDF excerpt rate
PACK_HZ = 16.0  # nasal pressure rate inside the browser payload
THR = 0.55
LAG = 10.0
LEAD = 30.0
TARGET_KINDS = ("obstructive", "hypopnea", "unsure")

OUT = Path(__file__).resolve().parent / "data"
CLIP_DIR = OUT / "clips"


# --------------------------------------------------------------------------- #
# stories
# --------------------------------------------------------------------------- #


def _common_gate(s: dict) -> bool:
    """Every clip opens calm and retracted, so the first advance is visible on screen."""
    return bool(s["starts_retracted"] and s["first_advance_t"] > 45)


def _score_oa_lead(s: dict) -> float | None:
    """A cluster-first event is caught cold: jaw fully forward before its onset.

    Cold starts are the hard case (no burst context to lean on), so the flagship
    clip has to win one of those with a fresh advance, not inside an existing hold.
    """
    # the OA requirement also keeps the OA-only oracle view of this clip meaningful
    if s["n_fresh_cold_oa"] < 1 or s["n_cold"] < 2 or s["n_oa"] < 2:
        return None
    if s["coverage"] < 0.85 or s["frac"] > 0.45 or s["n_linked"] < 3:
        return None
    if s["best_fresh_cold_oa"] < 5.0:
        return None
    return (
        6.0 * s["n_fresh_cold"]
        + 0.30 * min(s["best_fresh_cold_oa"], 45.0)
        + 2.0 * s["n_oa_pre"]
        + 8.0 * s["coverage"]
        - 10.0 * s["frac"]
        + 1.5 * min(s["n_cold"], 4)
    )


def _score_low_duty(s: dict) -> float | None:
    """Every arousal-linked event covered while the jaw sits home most of the clip."""
    if s["n_linked"] < 5 or s["n_cold"] < 2:
        return None
    if s["coverage"] < 0.999 or s["frac"] > 0.30 or s["n_adv"] < 2:
        return None
    if s["n_fresh"] < 1:
        return None
    return (
        12.0
        - 24.0 * s["frac"]
        + 0.6 * s["n_linked"]
        + 1.5 * min(s["n_cold"], 4)
        + 0.8 * min(s["n_oa"], 4)
    )


def _score_burst_hold(s: dict) -> float | None:
    """One advance rides out a whole cluster: few motor actions, many events covered."""
    if s["max_per_advance"] < 4 or s["n_cold"] < 2:
        return None
    if s["coverage"] < 0.85 or s["frac"] > 0.50:
        return None
    return 4.0 * s["max_per_advance"] + 6.0 * s["coverage"] - 5.0 * s["frac"]


def _score_multi_cold(s: dict) -> float | None:
    """Repeatability: several separated cluster-first events, all of them caught."""
    if s["n_cold"] < 3 or s["n_cold_linked"] < 2:
        return None
    if s["n_cold_cov"] < s["n_cold_linked"] or s["coverage"] < 0.85 or s["frac"] > 0.50:
        return None
    # separate advance/retract cycles, not one long hold across the whole clip
    if s["n_fresh_cold"] < 2 or s["n_adv"] < 3 or s["max_per_advance"] > 4:
        return None
    return (
        4.0 * s["n_cold_cov"]
        + 3.0 * s["n_fresh_cold"]
        + 6.0 * s["coverage"]
        - 6.0 * s["frac"]
    )


def _score_deep_desat(s: dict) -> float | None:
    """Severe desaturations: the clinically loud clip, still caught pre-onset."""
    if s["spo2_min"] > 88.0 or s["n_pre"] < 2 or s["n_cold"] < 2:
        return None
    if s["coverage"] < 0.80 or s["frac"] > 0.50 or s["n_fresh"] < 1:
        return None
    return (
        (92.0 - s["spo2_min"])
        + 2.0 * s["n_pre"]
        + 6.0 * s["coverage"]
        - 5.0 * s["frac"]
    )


@dataclass
class Story:
    id: str
    title: str
    watch: str
    policy: dict
    score: object
    order: int


MODEL_KINDS = ["obstructive", "hypopnea", "unsure"]

STORIES: list[Story] = [
    Story(
        id="oa_lead",
        title="Obstructive apnea caught before onset",
        watch=(
            "Jaw fully forward {best_fresh_oa_cold_start_lead_sec:.0f} s before a "
            "cluster-first obstructive apnea; {n_covered}/{n_linked} arousal-linked "
            "events covered with {advances} advances."
        ),
        policy={"source": "model", "kinds": MODEL_KINDS},
        score=_score_oa_lead,
        order=0,
    ),
    Story(
        id="low_duty",
        title="All events covered, jaw home most of the clip",
        watch=(
            "{n_covered}/{n_linked} covered while the device is advanced only "
            "{pct_advanced}% of the clip; a conventional MAD sits forward 100% of the night."
        ),
        policy={"source": "model", "kinds": MODEL_KINDS},
        score=_score_low_duty,
        order=1,
    ),
    Story(
        id="burst_hold",
        title="One advance holds through a whole cluster",
        watch=(
            "A single advance covers {max_events_per_advance} events in one cluster: "
            "the motor fires once and holds instead of chattering per event."
        ),
        policy={"source": "model", "kinds": MODEL_KINDS},
        score=_score_burst_hold,
        order=2,
    ),
    Story(
        id="deep_desat",
        title="Deep desaturations, every cluster covered",
        watch=(
            "SpO2 dips to {spo2_min:.0f}%. Every arousal-linked event "
            "({n_covered}/{n_linked}) is covered before its arousal deadline."
        ),
        policy={"source": "model", "kinds": MODEL_KINDS},
        score=_score_deep_desat,
        order=3,
    ),
    Story(
        id="multi_cold",
        title="Repeats: advance, retract, advance again",
        watch=(
            "{n_fresh_cold_start} cluster-first events caught by a fresh advance over "
            "{advances} advance/retract cycles. Step through them with 'Next cold start'."
        ),
        policy={"source": "model", "kinds": MODEL_KINDS},
        score=_score_multi_cold,
        order=4,
    ),
    Story(
        id="oa_only_oracle",
        title="Same clip, OA-only oracle (teaching)",
        watch=(
            "Same window, actuation driven by the scored obstructive apneas: the ideal "
            "timing the model is being asked to reproduce ({n_covered}/{n_linked})."
        ),
        policy={"source": "oracle", "kinds": ["obstructive"]},
        score=_score_oa_lead,
        order=5,
    ),
]


def fill_watch(story: Story, m: dict) -> str:
    ctx = dict(m)
    ctx["pct_advanced"] = round(m["fraction_advanced"] * 100)
    try:
        return story.watch.format(**ctx)
    except (KeyError, ValueError):
        return story.watch


# --------------------------------------------------------------------------- #
# model + controller helpers
# --------------------------------------------------------------------------- #


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
    drop = {names.index(w) for w in WAKE_DROP}
    ki = [i for i in range(len(names)) if i not in drop]
    assert len(ki) == 164, len(ki)
    return ki


def overnight_cfg() -> MesaMadConfig:
    """Same 166-d overnight flags the demo XGB heads were trained with."""
    return MesaMadConfig(
        earliest_lead_sec=LEAD,
        actuation_lag_sec=LAG,
        use_clean_pres=True,
        include_instability=True,
        include_flow_morph=True,
        rich_spo2=True,
        coldstart_features=True,
        include_hypnogram_wake_feats=True,
        include_breath_features=True,
        include_abs_amplitude=True,
        include_position=True,
        include_mid_scale=True,
        include_slow_scale=True,
    )


def deployable_decision_indices(bundle, cfg: MesaMadConfig) -> np.ndarray:
    """Seconds a real MAD would score: lookback filled + cannula not hard-broken.

    Training loss-masks (central/mixed, legacy Unsure pads) are NOT applied.
    Those neighborhoods still get a 1 Hz decision on-device.
    """
    from mesa_mad.cache import FS_PRES

    lab = bundle.labels
    idxs = np.arange(int(lab.t_sec.size), dtype=np.int64)
    idxs = idxs[lab.t_sec[idxs] >= float(cfg.min_decision_sec)]
    if bundle.pres_quality is not None and bundle.use_clean_pres:
        q = bundle.pres_quality
        qi = np.clip((lab.t_sec[idxs] * FS_PRES).astype(np.int64), 0, q.size - 1)
        idxs = idxs[q[qi] >= 0.3]
    return idxs


def extract_features_at(bundle, cfg: MesaMadConfig, idxs: np.ndarray) -> np.ndarray:
    """Causal overnight features at 1 Hz indices. Empty idxs -> (0, n_feat)."""
    from mesa_mad.cache import FS_PRES
    from mesa_mad.dataset import _slice_ending_at, sample_windows_at
    from mesa_mad.evaluate import _feat_kwargs
    from mesa_mad.features import coldstart_context_at_indices, extract_batch

    idxs = np.asarray(idxs, dtype=np.int64)
    fkw = _feat_kwargs(cfg)
    names = feature_names(
        include_instability=True,
        include_flow_morph=True,
        rich_spo2=True,
        coldstart_features=True,
        include_hypnogram_wake_feats=True,
        include_breath_features=True,
        include_abs_amplitude=True,
        include_position=True,
        include_mid_scale=True,
        include_slow_scale=True,
    )
    if idxs.size == 0:
        return np.zeros((0, len(names)), dtype=np.float32)
    lab = bundle.labels
    Xs: list[np.ndarray] = []
    chunk = 512
    for start in range(0, idxs.size, chunk):
        sl = idxs[start : start + chunk]
        fast_l, mid_l, slow_l, spo2_l, flat_l = [], [], [], [], []
        light_l, mid_light_l, slow_light_l = [], [], []
        for ti in sl:
            w = sample_windows_at(bundle, cfg, float(lab.t_sec[ti]))
            fast_l.append(w["fast"])
            mid_l.append(w["mid"])
            slow_l.append(w["slow"])
            spo2_l.append(w["spo2"])
            flat_l.append(w["mid_flat"])
            t = float(lab.t_sec[ti])
            if bundle.pres_light is not None:
                light_l.append(_slice_ending_at(bundle.pres_light, FS_PRES, t, cfg.fast_sec))
            if bundle.mid_pres_light is not None:
                mid_light_l.append(_slice_ending_at(bundle.mid_pres_light, cfg.mid_hz, t, cfg.mid_sec))
            if bundle.slow_pres_light is not None:
                slow_light_l.append(
                    _slice_ending_at(bundle.slow_pres_light, cfg.slow_hz, t, cfg.slow_sec)
                )
        kw = dict(fkw)
        kw["mid_flat"] = np.stack(flat_l)
        if light_l and len(light_l) == len(fast_l):
            kw["fast_light"] = np.stack(light_l)
        if mid_light_l and len(mid_light_l) == len(fast_l):
            kw["mid_light"] = np.stack(mid_light_l)
        if slow_light_l and len(slow_light_l) == len(fast_l):
            kw["slow_light"] = np.stack(slow_light_l)
        if fkw.get("coldstart_features"):
            kw["coldstart_context"] = coldstart_context_at_indices(bundle, sl, cfg)
        if fkw.get("include_breath_features") and bundle.breath_t is not None:
            from mesa_mad.breath_features import breath_feats_at_times

            kw["breath_traj"] = breath_feats_at_times(
                bundle.breath_t, bundle.breath_feats, lab.t_sec[sl]
            )
        if fkw.get("include_position"):
            from mesa_mad.breath_features import position_feats_at_times

            pf, _ = position_feats_at_times(bundle.pos, lab.t_sec[sl])
            kw["pos_feats"] = pf
        X, _ = extract_batch(
            np.stack(fast_l), np.stack(mid_l), np.stack(slow_l), np.stack(spo2_l), **kw
        )
        Xs.append(X)
    return np.concatenate(Xs, axis=0).astype(np.float32)


def fuse_heads(
    pre: float, fire: float, active: float, scored: bool, thr: float = THR
) -> tuple[bool, bool, str]:
    """Map the three 1 Hz heads onto (want_advance, hold_context, mode).

    fire_now is the only actuation trigger. pre_onset is early-warn context.
    active can hold an already-advanced jaw; it cannot start an advance alone.
    """
    if not scored or not np.isfinite(fire):
        return False, False, "unscored"
    f = float(fire) >= thr
    p = np.isfinite(pre) and float(pre) >= thr
    a = np.isfinite(active) and float(active) >= thr
    if p and f and not a:
        return True, True, "early-warn"
    if f and a and not p:
        return True, True, "rescue"
    if f:
        return True, True, "fire_now"
    if a and not p:
        return False, True, "detect-only"
    if p and not f:
        return False, False, "mismatch"
    return False, False, "quiet"


def run_ctrl_fused(
    pre: np.ndarray,
    fire: np.ndarray,
    active: np.ndarray,
    scored: np.ndarray,
    wake: np.ndarray,
    thr: float = THR,
) -> MadController:
    ctrl = MadController(
        act=ActuatorConfig(
            advance_sec=10.0, retract_sec=10.0, refractory_sec=60.0, quiet_retract_sec=90.0
        ),
        budget=BudgetConfig(enabled=False, init_threshold=thr),
    )
    ctrl.reset(0.0)
    ctrl.threshold = thr
    n = int(fire.size)
    for t in range(n):
        want, hold, _mode = fuse_heads(pre[t], fire[t], active[t], bool(scored[t]), thr)
        awake = bool(wake[t])
        # Scored wake already blocks new advances; it must also be allowed to
        # accumulate quiet and retract. Holding through WASO because a head is
        # high is not a deployable policy.
        ctrl.step(
            float(t),
            float(fire[t]) if want and np.isfinite(fire[t]) else 0.0,
            wake=awake,
            in_burst=hold and not awake,
        )
    return ctrl


def run_ctrl(probs: np.ndarray, wake: np.ndarray, thr: float = THR) -> MadController:
    ctrl = MadController(
        act=ActuatorConfig(
            advance_sec=10.0, retract_sec=10.0, refractory_sec=60.0, quiet_retract_sec=90.0
        ),
        budget=BudgetConfig(enabled=False, init_threshold=thr),
    )
    ctrl.reset(0.0)
    ctrl.threshold = thr
    for t in range(int(probs.size)):
        v = float(probs[t])
        if not np.isfinite(v):
            v = 0.0
        ctrl.step(float(t), v, wake=bool(wake[t]))
    return ctrl


def downsample(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if abs(fs_in - fs_out) < 1e-9:
        return x.astype(np.float32)
    n_out = int(round(x.size * fs_out / fs_in))
    if n_out < 8:
        return x.astype(np.float32)
    return sps.resample(x, n_out).astype(np.float32)


@dataclass
class Night:
    sid: str
    bundle: object
    p_fire: np.ndarray
    p_act: np.ndarray
    p_pre: np.ndarray
    scored: np.ndarray
    wake: np.ndarray
    events: list = field(default_factory=list)
    spo2: np.ndarray | None = None


def load_night(
    sid: str,
    fire,
    active,
    ki: list[int],
    pre=None,
    *,
    dense: bool = False,
    dense_ranges: list[tuple[int, int]] | None = None,
) -> Night | None:
    npz = NPZ_DIR / f"{sid}.npz"
    h5 = H5_DIR / f"mesa-{sid}.h5"
    if not npz.is_file() or not h5.is_file():
        return None
    z = np.load(npz, allow_pickle=True)
    X_pack = np.asarray(z["X"], dtype=np.float32)
    idxs_pack = np.asarray(z["idxs"], dtype=np.int64)
    cfg = overnight_cfg()
    b = load_night_bundle(h5, cfg)
    n = int(b.labels.t_sec.size)
    p_fire = np.full(n, np.nan, dtype=np.float64)
    p_act = np.full(n, np.nan, dtype=np.float64)
    p_pre = np.full(n, np.nan, dtype=np.float64)
    X164 = X_pack[:, ki] if X_pack.shape[1] != len(ki) else X_pack
    p_fire[idxs_pack] = fire.predict_proba(X164)[:, 1]
    p_act[idxs_pack] = active.predict_proba(X164)[:, 1]
    if pre is not None:
        p_pre[idxs_pack] = pre.predict_proba(X164)[:, 1]

    if dense:
        want = deployable_decision_indices(b, cfg)
        if dense_ranges:
            in_clip = np.zeros(n, dtype=bool)
            for a, b_ in dense_ranges:
                in_clip[max(0, int(a)) : min(n, int(b_))] = True
            want = want[in_clip[want]]
        have = np.zeros(n, dtype=bool)
        have[idxs_pack] = True
        missing = want[~have[want]]
        if missing.size:
            print(f"  {sid}: extracting {missing.size} deployable seconds not in the legacy pack", flush=True)
            X_miss = extract_features_at(b, cfg, missing)
            X_miss = X_miss[:, ki] if X_miss.shape[1] != len(ki) else X_miss
            p_fire[missing] = fire.predict_proba(X_miss)[:, 1]
            p_act[missing] = active.predict_proba(X_miss)[:, 1]
            if pre is not None:
                p_pre[missing] = pre.predict_proba(X_miss)[:, 1]
        scored = np.isfinite(p_fire)
    else:
        scored = np.zeros(n, dtype=bool)
        scored[idxs_pack] = True

    ann = b.cache.load_annotations()
    events = [e for e in ann.targets if e.kind in TARGET_KINDS]
    spo2 = np.asarray(b.cache.load_spo2(), dtype=np.float32)
    return Night(
        sid=sid,
        bundle=b,
        p_fire=p_fire,
        p_act=p_act,
        p_pre=p_pre,
        scored=scored,
        wake=np.asarray(b.labels.wake_mask, dtype=bool),
        events=events,
        spo2=spo2,
    )


def prob_json(arr: np.ndarray) -> list:
    out = []
    for v in np.asarray(arr, dtype=np.float64):
        out.append(None if not np.isfinite(v) else round(float(v), 3))
    return out


def shift_events(events: list[RespEvent], t0: int, t1: int) -> list[RespEvent]:
    out = []
    for e in events:
        if e.start < t0 or e.start >= t1:
            continue
        out.append(
            RespEvent(
                start=e.start - t0,
                end=min(e.end, t1) - t0,
                kind=e.kind,
                concept=e.concept,
                arousal_start=(
                    None
                    if e.arousal_start is None or not (t0 <= e.arousal_start < t1)
                    else e.arousal_start - t0
                ),
                desat_start=e.desat_start,
                is_cluster_first=e.is_cluster_first,
                cluster_id=e.cluster_id,
            )
        )
    return out


def advance_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.asarray(mask, dtype=bool).astype(np.int8)
    if not m.size:
        return []
    d = np.diff(np.concatenate(([0], m, [0])))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def window_stats(night: Night, t0: int, t1: int, probs: np.ndarray | None = None) -> dict:
    """Exact stats for one window: fresh controller from t0, like the browser does."""
    ev = shift_events(night.events, t0, t1)
    if len(ev) < 3:
        return {"ok": False}
    sl = night.p_fire[t0:t1] if probs is None else probs
    wk = night.wake[t0:t1]
    if probs is None:
        ctrl = run_ctrl_fused(
            night.p_pre[t0:t1],
            night.p_fire[t0:t1],
            night.p_act[t0:t1],
            night.scored[t0:t1],
            wk,
        )
    else:
        ctrl = run_ctrl(np.asarray(sl, dtype=np.float64), wk)
    mask = np.asarray(ctrl.advanced_mask, dtype=bool)
    outs = score_coverage(ev, mask, earliest_lead=LEAD, actuation_lag=LAG)
    linked = [o for o in outs]
    n_linked = len(linked)
    if n_linked < 3:
        return {"ok": False}
    covered = [o for o in linked if o.covered]
    pre = [o for o in covered if o.preemptive]
    leads = [o.lead_at_complete for o in pre if o.lead_at_complete is not None]
    runs = advance_runs(mask)
    per_run = []
    for a, b_ in runs:
        per_run.append(
            sum(
                1
                for o in covered
                if o.advance_complete_t is not None and a <= o.advance_complete_t < b_
            )
        )

    def run_start(t: float | None) -> int | None:
        if t is None:
            return None
        for a, b_ in runs:
            if a <= t < b_:
                return a
        return None

    # Uncapped timing: seconds between the jaw being fully advanced and the onset.
    # (lead_at_complete saturates at earliest_lead because of the credit window.)
    in_place: list[float] = []
    in_place_oa: list[float] = []
    in_place_cold: list[float] = []
    # "fresh" = the advance run started for this onset (within the last 60 s) and
    # finished before it, so the lead is prediction rather than hold-through.
    fresh: list[float] = []
    fresh_cold: list[float] = []
    fresh_cold_oa: list[float] = []
    onset_kind = {float(e.start): e.kind for e in ev}
    for o in covered:
        a = run_start(o.advance_complete_t)
        if a is None:
            continue
        lead = float(o.onset) - (a + 10.0)
        in_place.append(lead)
        if onset_kind.get(float(o.onset)) == "obstructive":
            in_place_oa.append(lead)
        if o.is_cluster_first:
            in_place_cold.append(lead)
        if 0.0 < lead <= 50.0:
            fresh.append(lead)
            if o.is_cluster_first:
                fresh_cold.append(lead)
                if onset_kind.get(float(o.onset)) == "obstructive":
                    fresh_cold_oa.append(lead)
    oa_leads = [
        o.lead_at_complete
        for o in pre
        if onset_kind.get(float(o.onset)) == "obstructive" and o.lead_at_complete
    ]
    cold_linked = [o for o in linked if o.is_cluster_first]
    spo2 = night.spo2[t0:t1] if night.spo2 is not None else np.array([100.0])
    valid = spo2[(spo2 > 50) & (spo2 <= 100)]
    fire_arr = np.asarray(sl, dtype=np.float64)
    finite = fire_arr[np.isfinite(fire_arr)]
    return {
        "ok": True,
        "sid": night.sid,
        "t0": int(t0),
        "t1": int(t1),
        "n_ev": len(ev),
        "n_oa": sum(1 for e in ev if e.kind == "obstructive"),
        "n_hyp": sum(1 for e in ev if e.kind in ("hypopnea", "unsure")),
        "n_cold": sum(1 for e in ev if e.is_cluster_first),
        "n_cold_linked": len(cold_linked),
        "n_cold_cov": sum(1 for o in cold_linked if o.covered),
        "n_linked": n_linked,
        "n_cov": len(covered),
        "coverage": len(covered) / n_linked,
        "n_pre": len(pre),
        "n_oa_pre": len(oa_leads),
        "best_oa_lead": float(max(oa_leads)) if oa_leads else 0.0,
        "median_lead": float(np.median(leads)) if leads else 0.0,
        "median_in_place": float(np.median(in_place)) if in_place else 0.0,
        "best_in_place": float(max(in_place)) if in_place else 0.0,
        "best_oa_in_place": float(max(in_place_oa)) if in_place_oa else 0.0,
        "best_cold_in_place": float(max(in_place_cold)) if in_place_cold else 0.0,
        "n_cold_in_place": int(sum(1 for v in in_place_cold if v > 5.0)),
        "n_in_place_pre_onset": int(sum(1 for v in in_place if v > 0)),
        "n_fresh": len(fresh),
        "n_fresh_cold": len(fresh_cold),
        "n_fresh_cold_oa": len(fresh_cold_oa),
        "best_fresh_cold": float(max(fresh_cold)) if fresh_cold else 0.0,
        "best_fresh_cold_oa": float(max(fresh_cold_oa)) if fresh_cold_oa else 0.0,
        "median_fresh": float(np.median(fresh)) if fresh else 0.0,
        "starts_retracted": bool(mask.size and not mask[0]),
        "first_advance_t": int(np.argmax(np.asarray(ctrl.actions) == 1))
        if any(a == 1 for a in ctrl.actions)
        else -1,
        "n_adv": int(ctrl.n_advances),
        "frac": float(np.mean(mask)) if mask.size else 0.0,
        "max_per_advance": int(max(per_run)) if per_run else 0,
        "spo2_min": float(np.min(valid)) if valid.size else 100.0,
        "wake_frac": float(np.mean(night.wake[t0:t1])),
        "p95": float(np.percentile(finite, 95)) if finite.size else 0.0,
    }


def cheap_gate(night: Night, starts: np.ndarray, cold: np.ndarray, linked: np.ndarray,
               night_mask_cs: np.ndarray, t0: int, t1: int) -> bool:
    lo = int(np.searchsorted(starts, t0))
    hi = int(np.searchsorted(starts, t1))
    if hi - lo < 3:
        return False
    if int(linked[lo:hi].sum()) < 3:
        return False
    if int(cold[lo:hi].sum()) < 2:
        return False
    frac = float(night_mask_cs[t1] - night_mask_cs[t0]) / float(t1 - t0)
    if frac > 0.55:
        return False
    if float(np.mean(night.wake[t0:t1])) > 0.18:
        return False
    return True


def pres_quality(pres: np.ndarray, fs: float) -> dict:
    """Reject sensor dropout, not apnea.

    Apneic seconds are legitimately low-amplitude, so only a near-dead trace
    (std below 2% of the clip median) counts as a dropout, and a long
    contiguous dead stretch is what actually ruins a projector clip.
    """
    n_sec = int(pres.size // max(1, int(round(fs))))
    if n_sec < 60:
        return {"ok": False, "flat_frac": 1.0, "flat_run": n_sec}
    step = int(round(fs))
    x = np.asarray(pres[: n_sec * step], dtype=np.float64).reshape(n_sec, step)
    sd = x.std(axis=1)
    scale = float(np.median(sd)) if np.median(sd) > 0 else 1.0
    dead = sd < 0.02 * scale
    runs = advance_runs(dead)
    longest = max((b - a for a, b in runs), default=0)
    flat = float(np.mean(dead))
    return {"ok": flat < 0.10 and longest <= 30, "flat_frac": flat, "flat_run": int(longest)}


def oracle_probs(events: list[RespEvent], n: int, kinds: set[str]) -> np.ndarray:
    p = np.zeros(n, dtype=np.float64)
    for e in events:
        if not e.arousal_linked or e.kind not in kinds:
            continue
        ref = e.arousal_start if e.arousal_start is not None else e.end
        a = max(0, int(np.ceil(e.start - LEAD)))
        b = min(n - 1, int(np.floor(ref - LAG)))
        if b >= a:
            p[a : b + 1] = 1.0
    return p


def metrics_from(stats: dict, policy: str) -> dict:
    return {
        "policy": policy,
        "threshold": THR,
        "n_events": stats["n_ev"],
        "n_oa": stats["n_oa"],
        "n_hyp": stats["n_hyp"],
        "n_cold_starts": stats["n_cold"],
        "n_linked": stats["n_linked"],
        "n_covered": stats["n_cov"],
        "coverage": round(stats["coverage"], 4),
        "n_preemptive": stats["n_pre"],
        "median_lead_sec": round(stats["median_lead"], 1),
        "best_oa_lead_sec": round(stats["best_oa_lead"], 1),
        "median_in_place_lead_sec": round(stats["median_in_place"], 1),
        "best_in_place_lead_sec": round(stats["best_in_place"], 1),
        "n_in_place_before_onset": stats["n_in_place_pre_onset"],
        "cold_starts_in_place_before_onset": stats["n_cold_in_place"],
        "best_cold_start_lead_sec": round(stats["best_cold_in_place"], 1),
        "n_fresh_advance_before_onset": stats["n_fresh"],
        "n_fresh_cold_start": stats["n_fresh_cold"],
        "best_fresh_cold_start_lead_sec": round(stats["best_fresh_cold"], 1),
        "best_fresh_oa_cold_start_lead_sec": round(stats["best_fresh_cold_oa"], 1),
        "median_fresh_lead_sec": round(stats["median_fresh"], 1),
        "first_advance_sec": stats["first_advance_t"],
        "advances": stats["n_adv"],
        "advances_per_hour": round(stats["n_adv"] * 3600.0 / (stats["t1"] - stats["t0"]), 2),
        "fraction_advanced": round(stats["frac"], 4),
        "static_mad_fraction_advanced": 1.0,
        "duty_saved_vs_static": round(1.0 - stats["frac"], 4),
        "max_events_per_advance": stats["max_per_advance"],
        "spo2_min": round(stats["spo2_min"], 1),
        "wake_frac": round(stats["wake_frac"], 3),
    }


def build_clip(story: Story, night: Night, stats: dict) -> dict:
    sid, t0, t1 = night.sid, stats["t0"], stats["t1"]
    cache = open_night_cache(H5_DIR / f"mesa-{sid}.h5")
    pres = cache.load_pres()
    fs_pres = float(cache.fs_pres)
    i0, i1 = int(round(t0 * fs_pres)), int(round(t1 * fs_pres))
    clip_pres = downsample(pres[i0:i1], fs_pres, DISPLAY_HZ)
    clip_spo2 = np.asarray(night.spo2[t0:t1], dtype=np.float32)
    clip_fire = night.p_fire[t0:t1]
    clip_act = night.p_act[t0:t1]
    clip_pre = night.p_pre[t0:t1]
    clip_scored = night.scored[t0:t1]
    clip_wake = night.wake[t0:t1].astype(np.uint8)

    ctrl = run_ctrl_fused(clip_pre, clip_fire, clip_act, clip_scored, night.wake[t0:t1])
    advanced = np.asarray(ctrl.advanced_mask, dtype=np.uint8)
    actions = np.asarray(ctrl.actions, dtype=np.int8)

    ev_local = shift_events(night.events, t0, t1)
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

    # headline metrics under the model, plus the OA-only oracle view of the same clip
    m_model = metrics_from(stats, "model")
    o_stats = window_stats(
        night, t0, t1, probs=oracle_probs(ev_local, t1 - t0, {"obstructive"})
    )
    m_oracle = metrics_from(o_stats, "oracle_oa") if o_stats.get("ok") else None
    watch = fill_watch(story, m_model)

    meta = {
        "subject_id": sid,
        "source": "MESA Sleep, NSRR Compumedics",
        "split": "test (subject held-out)",
        "clip_start_sec": t0,
        "clip_end_sec": t1,
        "duration_sec": t1 - t0,
        "fs_pres": PACK_HZ,
        "fs_decision": 1.0,
        "story": {
            "id": story.id,
            "title": story.title,
            "watch": watch,
            "policy": story.policy,
        },
        "geometry": {
            "task": "fire_now deadline-based MAD actuation",
            "actuation_lag_sec": LAG,
            "earliest_lead_sec": LEAD,
            "refractory_sec": 60.0,
            "quiet_retract_sec": 90.0,
            "events": "obstructive apnea + hypopnea + Unsure (MESA hyp>=30%)",
        },
        "model": {
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
        },
        "metrics": {"model": m_model, "oracle_oa": m_oracle},
        "controller": {
            "threshold": THR,
            "n_advances": int(ctrl.n_advances),
            "fraction_advanced": float(np.mean(advanced)),
        },
        "disclaimer": (
            "Untreated PSG: coverage is timing (advance completed before the arousal "
            "deadline), not proven event or arousal prevention. Selected example window."
        ),
    }

    pack = {
        "meta": meta,
        "fs_pres": PACK_HZ,
        "fs_decision": 1.0,
        "duration_sec": int(t1 - t0),
        "pres": [round(float(v), 4) for v in downsample(clip_pres, DISPLAY_HZ, PACK_HZ)],
        "spo2": [round(float(v), 2) for v in clip_spo2],
        "fire_now": prob_json(clip_fire),
        "pre_onset": prob_json(clip_pre),
        "active": prob_json(clip_act),
        "scored": [int(v) for v in clip_scored.astype(np.uint8)],
        "wake": [int(v) for v in clip_wake],
        "advanced_default": [int(v) for v in advanced],
        "actions_default": [int(v) for v in actions],
        "events": events,
        "arousals": arousals,
    }
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    (CLIP_DIR / f"{story.id}.json").write_text(json.dumps(pack) + "\n")
    write_edf(CLIP_DIR / f"{story.id}.edf", sid, clip_pres, clip_spo2, t0, t1)
    return pack


def write_edf(path: Path, sid: str, pres: np.ndarray, spo2: np.ndarray, t0: int, t1: int):
    try:
        import pyedflib
    except ImportError:
        return None
    w = pyedflib.EdfWriter(str(path), 2, file_type=pyedflib.FILETYPE_EDFPLUS)
    try:
        w.setTechnician("ProactMAD_demo")
        w.setRecordingAdditional(f"MESA-{sid}_excerpt_{t0}-{t1}s")
        lim = round(float(max(1e-3, np.abs(pres).max())), 3)
        w.setSignalHeader(
            0,
            {
                "label": "Pres",
                "dimension": "uV",
                "sample_frequency": DISPLAY_HZ,
                "physical_min": -lim,
                "physical_max": lim,
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


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    ki = keep_idx_164()
    fire = joblib.load(MODEL_DIR / "xgb_fire_now.joblib")
    active = joblib.load(MODEL_DIR / "xgb_active.joblib")

    sids = sorted(p.stem for p in NPZ_DIR.glob("*.npz"))
    splits = hash_stable_split(sids)
    test = [s for s in splits.test if (H5_DIR / f"mesa-{s}.h5").is_file()]
    if limit:
        test = test[:limit]
    print(f"scanning {len(test)} held-out nights", flush=True)

    cands: dict[str, list[dict]] = {s.id: [] for s in STORIES}
    for k, sid in enumerate(test, 1):
        try:
            night = load_night(sid, fire, active, ki)
        except Exception as exc:  # noqa: BLE001
            print(f"[{k}/{len(test)}] {sid} load failed: {exc}", flush=True)
            continue
        if night is None:
            continue
        n = int(night.p_fire.size)
        if n < CLIP_SEC + 1200:
            continue
        starts = np.array([e.start for e in night.events], dtype=np.float64)
        if starts.size < 4:
            continue
        order = np.argsort(starts)
        starts = starts[order]
        cold = np.array([night.events[i].is_cluster_first for i in order], dtype=bool)
        linked = np.array([night.events[i].arousal_linked for i in order], dtype=bool)
        night_ctrl = run_ctrl(night.p_fire, night.wake)
        night_mask = np.asarray(night_ctrl.advanced_mask, dtype=bool)
        cs = np.concatenate(([0], np.cumsum(night_mask.astype(np.int64))))

        # per-night shortlist, then exact re-simulation of only those windows
        rough: list[tuple[int, int]] = []
        for t0 in range(600, n - CLIP_SEC - 120, STEP_SEC):
            t1 = t0 + CLIP_SEC
            if cheap_gate(night, starts, cold, linked, cs, t0, t1):
                rough.append((t0, t1))
        if not rough:
            print(f"[{k}/{len(test)}] {sid}: no candidate window", flush=True)
            continue
        found = {s.id: 0 for s in STORIES}
        for t0, t1 in rough:
            st = window_stats(night, t0, t1)
            if not st.get("ok"):
                continue
            if not _common_gate(st):
                continue
            for story in STORIES:
                sc = story.score(st)
                if sc is None:
                    continue
                cands[story.id].append({**st, "score": float(sc)})
                found[story.id] += 1
        hits = {i: c for i, c in found.items() if c}
        print(
            f"[{k}/{len(test)}] {sid}: {len(rough)} windows -> "
            + (", ".join(f"{i}x{c}" for i, c in hits.items()) if hits else "none"),
            flush=True,
        )
        del night

    # keep the best few windows per story, well separated in time
    ranked: dict[str, list[dict]] = {}
    for story in STORIES:
        rows = sorted(cands[story.id], key=lambda r: -r["score"])
        keep: list[dict] = []
        for r in rows:
            if any(
                q["sid"] == r["sid"] and abs(q["t0"] - r["t0"]) < 600 for q in keep
            ):
                continue
            keep.append(r)
            if len(keep) >= 12:
                break
        ranked[story.id] = keep
        print(f"story {story.id}: {len(cands[story.id])} candidates, top={keep[:1]}", flush=True)

    used_sids: set[str] = set()
    index: list[dict] = []
    built: dict[str, dict] = {}
    for story in sorted(STORIES, key=lambda s: s.order):
        # The oracle view is the same window as the flagship clip, replayed under a
        # different controller input, so the two are directly comparable.
        if story.id == "oa_only_oracle" and "oa_lead" in built:
            src = built["oa_lead"]
            index.append(
                {
                    "id": story.id,
                    "pack_id": "oa_lead",
                    "title": story.title,
                    "watch": fill_watch(story, src["metrics_oracle_oa"] or src["metrics"]),
                    "policy": story.policy,
                    "subject_id": src["subject_id"],
                    "clip_start_sec": src["clip_start_sec"],
                    "duration_sec": src["duration_sec"],
                    "metrics": src["metrics_oracle_oa"] or src["metrics"],
                    "metrics_oracle_oa": src["metrics_oracle_oa"],
                }
            )
            print(f"story {story.id}: reuses oa_lead clip (MESA {src['subject_id']})", flush=True)
            continue
        chosen = None
        for allow_reuse in (False, True):
            for r in ranked[story.id]:
                if not allow_reuse and r["sid"] in used_sids:
                    continue
                night = load_night(r["sid"], fire, active, ki)
                if night is None:
                    continue
                cache = open_night_cache(H5_DIR / f"mesa-{r['sid']}.h5")
                fs = float(cache.fs_pres)
                seg = cache.load_pres()[int(r["t0"] * fs) : int(r["t1"] * fs)]
                q = pres_quality(seg, fs)
                if not q["ok"]:
                    print(
                        f"  {story.id}: {r['sid']}@{r['t0']} rejected, "
                        f"dead={q['flat_frac']:.2f} run={q['flat_run']}s"
                    )
                    continue
                chosen = (night, r)
                break
            if chosen:
                break
        if not chosen:
            print(f"story {story.id}: NO CLIP", flush=True)
            continue
        night, r = chosen
        pack = build_clip(story, night, r)
        used_sids.add(r["sid"])
        m = pack["meta"]["metrics"]["model"]
        entry = {
            "id": story.id,
            "pack_id": story.id,
            "title": story.title,
            "watch": pack["meta"]["story"]["watch"],
            "policy": story.policy,
            "subject_id": r["sid"],
            "clip_start_sec": r["t0"],
            "duration_sec": r["t1"] - r["t0"],
            "metrics": m,
            "metrics_oracle_oa": pack["meta"]["metrics"]["oracle_oa"],
        }
        index.append(entry)
        built[story.id] = entry
        print(
            f"story {story.id}: MESA {r['sid']} t0={r['t0']} "
            f"cov={m['n_covered']}/{m['n_linked']} adv={m['advances']} "
            f"frac={m['fraction_advanced']:.2f} lead={m['median_lead_sec']}s "
            f"cold={m['n_cold_starts']} spo2min={m['spo2_min']}",
            flush=True,
        )
        del night

    if not index:
        print("no clips built")
        return 1
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    index_path = CLIP_DIR / "index.json"
    # keep entries this script does not own (e.g. the whole-night clip)
    story_ids = {s.id for s in STORIES}
    kept: list[dict] = []
    if index_path.is_file():
        old = json.loads(index_path.read_text())
        kept = [c for c in old.get("clips", []) if c["id"] not in story_ids]
    clips = kept + index
    index_path.write_text(
        json.dumps({"default": clips[0]["id"], "clips": clips}, indent=2) + "\n"
    )
    shutil.copyfile(CLIP_DIR / f"{index[0]['pack_id']}.json", OUT / "pack.json")
    print(f"wrote {len(index)} clips to {CLIP_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
