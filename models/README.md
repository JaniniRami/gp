# Model card — deployable fire_now XGB

Source run: `ablate_no_hypnogram_wake_v1` (MESA MAD product pack).

- `xgb_fire_now.joblib` — headline actuation head
- `xgb_active.joblib` — in-event / active head (UI overlay)
- `xgb_pre_onset.joblib` — pre-onset head (not required for the demo player)
- `feature_names.json` — 164 columns, hypnogram-wake features already removed
- `config.json` — geometry and metrics

Do not report the with-wake AUROC as the product number. Scored wake is not available on-device.
