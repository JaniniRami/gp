# ProactMAD live demo

Presentation replay of the MESA MAD controller on **subject-held-out** night clips.

The browser plays nasal pressure + SpO2 as a live scrolling PSG. The **fire_now** head (deployable, no hypnogram-wake features) drives the same 1 Hz control loop used in the paper stack: 10 s advance, 60 s refractory, hold-through-burst, then retract. When the MAD advances, the traces go gray -- the device is already out, so we stop looking for a new fire.

Nothing to the right of the cursor is drawn: signals, scored events, and arousals appear only once the replay reaches them, so the audience never sees the apnea before the model reacts to it.

**This is untreated PSG.** Coverage means the advance completed before the arousal deadline. It does not mean the event or arousal was prevented.

## What is in this repo

| Path | What |
|---|---|
| `static/` | Live-replay website |
| `server.py` | Local server (opens the browser, optional ESP serial) |
| `data/clips/*.json` | Example clip library (waveforms, events, model scores) |
| `data/clips/full_night.json` | Whole-night replay of the best-performing held-out night (severe) |
| `data/clips/mild_night.json` | Whole-night replay of a mild held-out night (clear duty cycle) |
| `data/clips/*.edf` | Matching 18 min excerpts (Pres 32 Hz + SpO2 1 Hz) |
| `data/clips/index.json` | Story text, recommended policy and headline metrics per clip |
| `data/pack.json` | Copy of the flagship 18 min clip (fallback when no clip index is readable) |
| `models/xgb_fire_now.joblib` | Deployable `fire_now` XGB (164 features, test AUROC 0.85) ù primary MAD trigger |
| `models/xgb_pre_onset.joblib` | Pre-onset early-warn head (not the actuation trigger) |
| `models/xgb_active.joblib` | In-event / hold / rescue context head |
| `firmware/drv8871_mad_control/` | ESP32 + DRV8871 sketch (IN1=27, IN2=26) |

## Whole-night replay (default clip)

The demo opens on **Whole night, best held-out subject**: MESA **2934**, a 7.7 h recording
played end to end, so you watch the controller advance and retract all night instead of a
single window.

| Night | Arousal-linked covered | Advances | Night advanced | SpO2 nadir |
|---|---|---|---|---|
| MESA 2934, 7.7 h | 104/104 (100%) | 13 | 67% (fixed MAD = 100%) | 82% |

Severe OSA night: 119 obstructive apneas plus hypopneas, 104 of them arousal-linked.
Every cannula-valid second after the 600 s lookback is scored at 1 Hz (`pre_onset`,
`fire_now`, `active`). At threshold 0.55 this model is above the `fire_now` trigger for
most of sleep, so the jaw is forward through sleep and home in scored wake (~40% of the
recording). That is the deployable controller, not a detection overlay.

The older 40% advanced / 34-advance numbers filled missing seconds with **0**. Those
zeros were legacy Unsure pads, not model predictions. Gaps are now `null` (drawn as a
break in the trace), never a fake zero.

At **60x** the night takes about 8 min of presentation time; 120x and 240x are there if
you only want to show the pattern. Above 16x the ESP is deliberately **not driven** (the
motor cannot track a sped-up night), and the link chip says so.

The night map under the traces is the whole recording at a glance: hour ticks, scored
events on the upper lane (amber OA, purple hypopnea/Unsure), the jaw state on the lower
lane, wake in light gray, and the cursor sweeping left to right. The pattern to point at
is the jaw holding through clusters in sleep and retracting in wake.

This night was picked by `_build_night.py` on the legacy sparse decision grid (coverage
>= 85%, at most 40% advanced under that grid). It is still the best of 168 held-out
nights on that ranking, not a typical night. The stored `meta.cohort` medians (89%
coverage at 40% advanced) are from that sparse ranking, not the dense 1 Hz grid the
player now runs.

## Mild night (clearest view of the duty cycle)

On a severe night the model is above the trigger through most of sleep, so the jaw
stays out and the advance/retract cycle is hard to see. Pick **Whole night, mild
severity** for a night with enough quiet between clusters that each actuation is a
separate, visible decision.

| Night | NSRR AHI | Arousal-linked covered | Advances | Night advanced | SpO? nadir |
|---|---|---|---|---|---|
| MESA 3901, 8.9 h | 5.8 (mild) | 12/14 | 15 (15 retracts) | 19% (fixed MAD = 100%) | 91% |

This is the honest deployable grid: 99.1% of the span scored at 1 Hz, the remaining
0.9% being the 600 s lookback at the start (drawn as a gap, not a zero). Two linked
events are missed ù worth showing rather than hiding.

Caveats to state if asked: this night is **hypopnea-dominated** (only 1 scored
obstructive apnea), so use MESA 2934 for the OA story, and it carries a lot of scored
WASO (55% of the span is wake), which is why the jaw is home so much of the recording.
MESA 3901 was chosen as a severity example, not as the top-ranked night.

## Example library

Pick a clip from the **clip** menu in the top bar. Besides the whole night above, each
one is an 18 min window from a different held-out MESA subject, chosen automatically
because it makes one specific point.
All numbers below are at threshold 0.55, geometry **A = 10 s**, earliest lead **30 s**,
targets OA + hypopnea + Unsure, coverage counted over **arousal-linked** events only.

| Clip | Subject | Covered | Advances | Clip advanced | Point it makes |
|---|---|---|---|---|---|
| Obstructive apnea caught before onset | 3396 | 8/8 | 2 | 94% | Jaw fully forward 38 s before a cluster-first OA |
| All events covered | 5103 | 6/6 | 1 | 97% | Dense cluster: `fire_now` stays high through the window |
| One advance holds through a whole cluster | 5911 | 11/11 | 1 | 57% | 11 events covered by a single motor action |
| Deep desaturations, every cluster covered | 1913 | 6/6 | 1 | 94% | SpO2 nadir 66%, every cluster covered in time |
| Dense cluster, one hold | 1278 | 6/6 | 1 | 98% | Watch the three heads, not a retract cycle |
| Same clip, OA-only oracle (teaching) | 3396 | 8/8 | 2 | 57% | Ideal timing the model is asked to reproduce |

These 18 min windows sit inside event clusters, so after honest 1 Hz scoring `fire_now`
is high for most of the clip and the jaw stays out. **Duty cycle belongs on the whole
night**, not on a cluster excerpt. They are still useful to talk through early-warn vs
rescue vs detect-only on a short timeline.

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

Presenter path for the whole night: **Play** at 60x and let the night map fill in; stop at
any cluster, drop to 8x and talk through one advance.

Presenter path for a story clip: **Next cold start** -> **Play** at 8x -> watch `fire_now`
cross the threshold -> MAD ADVANCING (10 s) -> gray HOLD through the cluster -> retract
when quiet.

## Controls (top bar)

Everything lives in the two rows at the top; the page never scrolls.

| Control | What it does |
|---|---|
| **clip** | Switches example; loads that clip and its recommended controller input |
| **Play / Pause** | Runs the replay at the selected speed |
| **speed** | 1x live up to 240x (60x plays a whole night in about 9 min) |
| **Next cold start** | Seeks to **60 s before the next cluster-first event** and cycles through all of them (wraps at the end) |
| **Next event** | Same 60 s pre-roll, but for every target event including mid-burst ones |
| **Reset** | Back to the opening frame (lights out for the night clip, 45 s before the first advance for a story clip) and **retracts the MAD** if a board is linked |
| **Stacked lanes / Normalized overlay** | Trace layout |
| **window** | Visible time span, 30-120 s |
| **threshold** | `fire_now` decision threshold; the controller re-simulates instantly |

The hint at the right of the control row always names the next cold start (`cold start 4/6 at 10:28`),
and cold starts are marked with cyan triangles in the trace lanes and on the clip map.

The line under the traces names the clip and the point it makes, with that clip's
headline numbers on the right (`7/8 covered | 2 adv | 44% vs 100% static`). The
**This clip** card tracks the same quantities live as the replay advances, including
how far ahead of each onset the jaw was already in place. That lead counts only
advances started for the event in question (within 60 s of onset); events that were
already covered by an earlier hold are reported as `held`, so hold-through is never
sold as prediction.

The green **raw fire_now (1 Hz)** readout shows the unmodified model probability and
the exact replay second (`0.985 @ 46:18`). The model lane is a one-second staircase
with one dot per prediction, rather than a smoothed curve. It always remains the raw
model output: in annotation-oracle mode, the separate controller input is drawn as a
dashed amber trace and does not replace the green model scores.

Only **OSA-linked arousals** are drawn on the traces, as dashed gold lines labeled
`LINKED AROUSAL`. Spontaneous, movement-related and otherwise non-respiratory arousals
remain hidden so they are not mistaken for controller failures. The **This clip** card
keeps the original PSG comparison visible:

- MESA 2934 has **215 total scored arousals**.
- **100 unique arousal episodes** are linked to the 104 target-event links (four
  arousals are shared by two events).
- The MAD is **covered in time for 98** linked arousal episodes and **misses 2**.
- The headline `102/104` remains event-link coverage; `98/100` is unique-arousal
  coverage.

ùCovered in timeù means advancement completed before the arousal deadline. Because the
recording is untreated PSG, it is not evidence that those 98 arousals were actually
avoided.

URL parameters for a preset opening state: `/?clip=burst_hold`, `/?t=660` (jump to a
second in the clip), `/?layout=overlay`, `/?play=1`, `/?speed=16`. They combine, so
`/?clip=oa_lead&t=430&speed=4` opens exactly on the advance you want to talk about, and
`/?clip=full_night&play=1&speed=60` starts the whole night running by itself.

If the **clip** menu is empty or disabled ("library not found"), the page is talking to
an old server process. Stop it, `git pull`, run `python server.py` again, and hard-reload
the browser (Cmd+Shift+R). The startup log lists the clips it found; the UI also falls
back to reading `data/clips/index.json` directly, so a plain file server works too.

### Screen layout

The dashboard is sized to the browser viewport, so everything (traces, clip map,
legend, policy card, control rows) stays on one screen with no scrolling on a
13" MacBook Air.

Every trace is normalized once per clip using robust percentiles (0.5-99.5 for
nasal pressure, SpO2 capped at 100), so the vertical scale never jumps while
the trace scrolls. The real range in use is printed in the bottom-left of each
lane. Pulse-ox dropout (SpO2 below 50%) is left as a gap rather than drawn as a
desaturation, and never sets the range.

Nasal pressure is stored at 32 Hz in the 18 min clips and at 8 Hz in the whole-night
clip (still well above breathing rate) to keep the night to about 3 MB.

- **Stacked lanes** (default): Pres, SpO2, and model heads in their own lanes.
- **Normalized overlay**: all four series on one shared 0-1 axis in a single
  lane, with the threshold line as a common reference.

### Change the controller policy

The **Controller policy** card lets you change the advance duration,
refractory interval, quiet time before retraction, probability threshold,
target kinds, and deadline geometry.

The default **Combined model (honest)** uses the deployable model exactly as
trained: OA + hypopnea + Unsure. It does not output event kind separately.

For an **OA-only presentation**, choose the **Same clip, OA-only oracle** example (or
pick **Annotation oracle** and leave only **OA** selected). Hypopnea-only, Unsure-only, and any combination
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

**Reset** (and switching clip, or seeking with **Next cold start** / **Next event**) puts the
board back where the replay says the jaw is, which for the opening frame means a `RETRACT`,
so the hardware never sits advanced while the screen shows RETRACTED. Nothing is sent when
the board is already in that position, and nothing is sent at page load.

Manual serial (115200): `o` open, `c` close, `s` stop, `ADVANCE`, `RETRACT`, `0-255` speed, `?` help.

Server endpoints used by the UI: `GET /api/esp/status` (link state, last reply, age, rx/tx counts),
`POST /api/esp/cmd`, `POST /api/esp/ping`.

## Model (honest deployment pack)

Three 1 Hz heads, scored every cannula-valid second after the 600 s lookback
(no hypnogram-wake features; gaps are `null`, not 0):

| Head | Meaning at time t | Deployment role |
|---|---|---|
| `pre_onset` | Onset in the next ~30 s (before the event) | Early-warn score |
| `fire_now` | Actuating now still beats the deadline (including after onset) | **Primary MAD trigger** |
| `active` | Already inside an event | Confirm / hold / rescue context ù not the early-warn trigger |

Fusion (threshold 0.55): advance only if `fire_now` is high. `pre_onset` + `fire_now`
is a clean early-warn; `fire_now` + `active` is in-event rescue. `active` alone holds
if the jaw is already out, otherwise it is too late. `pre_onset` without `fire_now`
does not actuate. Quiet seconds retract after the quiet timer.

- Features: fast nasal pressure 32 s @ 32 Hz + mid/slow + SpO2 + position + flow-morph + breath + signal-only cold-start.
- **Dropped:** `cs_t_since_wake`, `cs_wake_frac_10m` (PSG hypnogram wake is not available on a real MAD).
- Held-out MESA test, 1 Hz grid: fire_now AUROC **0.851**.

The website uses **precomputed raw 1 Hz probabilities** from those weights so the M4
stays smooth. It displays each value to three decimal places with its exact decision
second. Unscored seconds (lookback or hard cannula artifact) are drawn as a gap.
Threshold can still be moved live; it changes the controller decision but not
the raw model probability.

Rebuilding the whole-night clip (needs the MESA caches): `python _build_night.py`. It
ranks every held-out night, prints the top candidates and the cohort medians, and writes
`data/clips/full_night.json` plus a merged `index.json`. Run `_build_clips.py` first if you
are rebuilding both; each script preserves the other's entries.

To build a named night for one chosen held-out subject, skipping the ranking (this is how
the mild night was made):

```bash
python _build_night.py --sid 3901              # writes data/clips/mild_night.json
python _build_night.py --sid 3901 --default    # ...and opens the demo on it
```

The subject must be in the held-out test split; the script refuses otherwise. It records
the NSRR AHI and "chosen as a severity example, not the top-ranked night" in the pack's
`meta.cohort`.

To refill a deployable 1 Hz grid on existing clips (fills legacy Unsure pads that
used to be stored as fake zeros, and adds `pre_onset`): `python _rescore_dense.py`.

Rebuilding the library (needs the MESA caches, not in this repo): `python _build_clips.py`
scans every held-out night, scores each 18 min window with the same controller the browser
runs, and keeps the best window per story in `data/clips/`. Windows are selected on the
**model** probabilities, never the oracle, and must open retracted so the first advance is
visible on screen. `python _build_pack.py` is the older single-clip builder.

A cold start (cluster-first event) counts as caught by a *fresh* advance only when the
advance run began within 60 s of the onset and finished before it; leads inherited from an
earlier hold are reported separately, so "caught before onset" never means hold-through.

## Data notice

The files in `data/clips/` are derived excerpts of [MESA Sleep](https://sleepdata.org/datasets/mesa) (NSRR): 18 min windows plus one downsampled whole night (nasal pressure at 8 Hz, SpO2 at 1 Hz, scored events). Use only under your NSRR data-use agreement, and do not treat these demo files as a license to redistribute MESA recordings.
