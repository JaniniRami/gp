# ProactMAD live demo

Presentation replay of the MESA MAD controller on a **subject-held-out** night clip.

The browser plays nasal pressure + SpO2 as a live scrolling PSG. The **fire_now** head (deployable, no hypnogram-wake features) drives the same 1 Hz control loop used in the paper stack: 10 s advance, 60 s refractory, hold-through-burst, then retract. When the MAD advances, the traces go gray -- the device is already out, so we stop looking for a new fire.

**This is untreated PSG.** Coverage means the advance completed before the arousal deadline. It does not mean the event or arousal was prevented.

## What is in this repo

| Path | What |
|---|---|
| `static/` | Live-replay website |
| `server.py` | Local server (opens the browser) |
| `data/mesa-sleep-1510_clip.edf` | 12 min MESA excerpt (Pres + SpO2) |
| `data/pack.json` | Waveforms, events, model scores for the UI |
| `models/xgb_fire_now.joblib` | Deployable `fire_now` XGB (164 features, test AUROC 0.85) |
| `models/xgb_active.joblib` | Active-event head (shown as the cyan model trace) |
| `firmware/drv8871_mad_control/` | ESP32 + DRV8871 sketch (IN1=27, IN2=26) |

Clip: **MESA 1510**, test split, 7320-8040 s. One advance covers a mixed **OA + hypopnea** burst (8 linked events, all preemptive at threshold 0.55). Geometry: **A = 10 s**, earliest lead **30 s**.

## Run on a Mac (M4)

Chrome or Edge recommended (Web Serial for the ESP). Safari plays the website but cannot talk to USB serial.

```bash
git clone https://github.com/JaniniRami/gp.git
cd gp
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py
```

The site opens at [http://127.0.0.1:8765](http://127.0.0.1:8765).

Presenter path: **Jump to burst** ? **Play** at 8x ? watch `fire_now` cross the threshold ? MAD ADVANCING (10 s) ? gray HOLD through the cluster ? retract when quiet.

### Change the controller policy

The **Controller policy** card lets you change the advance duration,
refractory interval, quiet time before retraction, probability threshold,
target kinds, and deadline geometry.

The default **Combined model (honest)** uses the deployable model exactly as
trained: OA + hypopnea + Unsure. It does not output event kind separately.

For an **OA-only presentation**, choose **Annotation oracle (presentation)**
and leave only **OA** selected. Hypopnea-only, Unsure-only, and any combination
work the same way. Oracle mode uses the scored event kind and is visibly
labeled `ORACLE`; it demonstrates the policy but is not a claim that the
real-time model can identify event kind in advance.

## ESP32 actuator

1. Flash `firmware/drv8871_mad_control/drv8871_mad_control.ino` (ESP32 board, Arduino-ESP32 **3.x**).
2. Wiring: DRV8871 **IN1 ? GPIO 27** (open/advance), **IN2 ? GPIO 26** (close/retract).
3. In the website click **Connect ESP** (Chrome Web Serial), or:

```bash
python server.py --esp auto
```

The control loop sends `ADVANCE` / `RETRACT`. The firmware runs the motor for 10 s then stops (hold).

Manual serial (115200): `o` open, `c` close, `s` stop, `ADVANCE`, `RETRACT`, `0-255` speed.

## Model (honest deployment pack)

- Head: deadline `fire_now` (act in `[onset?30, (arousal|end)?10]`).
- Features: fast nasal pressure 32 s @ 32 Hz + mid/slow + SpO2 + position + flow-morph + breath + signal-only cold-start.
- **Dropped:** `cs_t_since_wake`, `cs_wake_frac_10m` (PSG hypnogram wake is not available on a real MAD).
- Held-out MESA test, 1 Hz grid: fire_now AUROC **0.851**.

The website uses **precomputed 1 Hz scores** from those weights so the M4 stays smooth. Threshold can still be moved live; the JS control loop re-runs instantly.

## Data notice

`data/mesa-sleep-1510_clip.edf` is a short derived excerpt of [MESA Sleep](https://sleepdata.org/datasets/mesa) (NSRR). Use only under your NSRR data-use agreement. Do not treat this public demo clip as a license to redistribute full nights.
