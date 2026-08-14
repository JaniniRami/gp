# ProactMAD live demo

Presentation replay of the MESA MAD controller on a **subject-held-out** night clip.

The browser plays nasal pressure + SpO2 as a live scrolling PSG. The **fire_now** head (deployable, no hypnogram-wake features) drives the same 1 Hz control loop used in the paper stack: 10 s advance, 60 s refractory, hold-through-burst, then retract. When the MAD advances, the traces go gray -- the device is already out, so we stop looking for a new fire.

Nothing to the right of the cursor is drawn: signals, scored events, and arousals appear only once the replay reaches them, so the audience never sees the apnea before the model reacts to it.

**This is untreated PSG.** Coverage means the advance completed before the arousal deadline. It does not mean the event or arousal was prevented.

## What is in this repo

| Path | What |
|---|---|
| `static/` | Live-replay website |
| `server.py` | Local server (opens the browser, optional ESP serial) |
| `data/mesa-sleep-3481_clip.edf` | 18 min MESA excerpt (Pres 32 Hz + SpO2 1 Hz) |
| `data/pack.json` | Waveforms, events, model scores for the UI |
| `models/xgb_fire_now.joblib` | Deployable `fire_now` XGB (164 features, test AUROC 0.85) |
| `models/xgb_active.joblib` | Active-event head (shown as the cyan model trace) |
| `firmware/drv8871_mad_control/` | ESP32 + DRV8871 sketch (IN1=27, IN2=26) |

Clip: **MESA 3481**, test split, 40440-41520 s. 12 target events in **6 bursts**, 3 advances at
threshold 0.55, 41% of the clip advanced, 8/10 arousal-linked events covered (all preemptive).
Geometry: **A = 10 s**, earliest lead **30 s**.

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

Presenter path: **Next cold start** -> **Play** at 8x -> watch `fire_now` cross the threshold -> MAD ADVANCING (10 s) -> gray HOLD through the cluster -> retract when quiet.

## Controls (top bar)

Everything lives in the two rows at the top; the page never scrolls.

| Control | What it does |
|---|---|
| **Play / Pause** | Runs the replay at the selected speed |
| **speed** | 1x live up to 16x |
| **Next cold start** | Seeks to **60 s before the next cluster-first event** and cycles through all of them (wraps at the end) |
| **Next event** | Same 60 s pre-roll, but for every target event including mid-burst ones |
| **Reset** | Back to the opening frame (first cold start with a full pre-roll and the device retracted) |
| **Stacked lanes / Normalized overlay** | Trace layout |
| **window** | Visible time span, 30-120 s |
| **threshold** | `fire_now` decision threshold; the controller re-simulates instantly |

The hint at the right of the control row always names the next cold start (`cold start 4/6 at 10:28`),
and cold starts are marked with cyan triangles in the trace lanes and on the clip map.

URL parameters for a preset opening state: `/?layout=overlay`, `/?play=1`, `/?speed=16`.

### Screen layout

The dashboard is sized to the browser viewport, so everything (traces, clip map,
legend, policy card, control rows) stays on one screen with no scrolling on a
13" MacBook Air.

Every trace is normalized once per clip using robust percentiles (0.5-99.5 for
nasal pressure, SpO2 capped at 100), so the vertical scale never jumps while
the trace scrolls. The real range in use is printed in the bottom-left of each
lane.

- **Stacked lanes** (default): Pres, SpO2, and model heads in their own lanes.
- **Normalized overlay**: all four series on one shared 0-1 axis in a single
  lane, with the threshold line as a common reference.

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
2. Wiring: DRV8871 **IN1 -> GPIO 27** (open/advance), **IN2 -> GPIO 26** (close/retract).
3. Either click **Connect ESP** in the website (Chrome/Edge Web Serial) and pick the
   `usbserial` / `usbmodem` port, or let the server own the port:

```bash
python server.py --esp auto
```

Close the Arduino IDE Serial Monitor first -- only one process can hold the port.

### Link indicator

The chip in the top-right reports the state of the actual link, not just a button press:

| Colour | Meaning |
|---|---|
| Gray, "ESP not connected" | No port open |
| Amber, "ESP port open" | Port open, board has not answered yet |
| Green, "ESP linked" | The board replied; the chip shows the last reply and its age (`rx "OK ADVANCE" 2s ago`) |
| Red, "ESP link error" | Port closed, write failed, or no reply to a link test |

**Test link** sends `?` and waits for the firmware to answer, which proves two-way
communication. The **Advance / Retract / Stop** buttons in the Actuator card jog the
hardware by hand; they are disabled until a link exists. Replies are also mirrored in
the Actuator card (`ESP: rx ... | tx ...`).

The control loop sends `ADVANCE` / `RETRACT` on rising edges during playback. The firmware
runs the motor for 10 s then stops (hold) and answers `OK ADVANCE` / `OK RETRACT` / `OK HOLD`.

Manual serial (115200): `o` open, `c` close, `s` stop, `ADVANCE`, `RETRACT`, `0-255` speed, `?` help.

Server endpoints used by the UI: `GET /api/esp/status` (link state, last reply, age, rx/tx counts),
`POST /api/esp/cmd`, `POST /api/esp/ping`.

## Model (honest deployment pack)

- Head: deadline `fire_now` (act in `[onset-30, (arousal|end)-10]`).
- Features: fast nasal pressure 32 s @ 32 Hz + mid/slow + SpO2 + position + flow-morph + breath + signal-only cold-start.
- **Dropped:** `cs_t_since_wake`, `cs_wake_frac_10m` (PSG hypnogram wake is not available on a real MAD).
- Held-out MESA test, 1 Hz grid: fire_now AUROC **0.851**.

The website uses **precomputed 1 Hz scores** from those weights so the M4 stays smooth. Threshold can still be moved live; the JS control loop re-runs instantly.

Rebuilding the clip (needs the MESA caches, not in this repo): `python _build_pack.py` writes
`data/pack.json`, the EDF excerpt, and `data/clip_meta.json`. It scans candidate held-out nights
and picks a window with several bursts, requiring at least `MIN_COLD_STARTS` cluster-first events.

## Data notice

`data/mesa-sleep-3481_clip.edf` is a short derived excerpt of [MESA Sleep](https://sleepdata.org/datasets/mesa) (NSRR). Use only under your NSRR data-use agreement. Do not treat this public demo clip as a license to redistribute full nights.
