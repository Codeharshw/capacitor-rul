# Bring your own capacitor data

## Required prepared CSV

One file represents **one physical capacitor**, with a single cumulative observed-cycle coordinate. Required columns:

| Column | Meaning |
|---|---|
| `sensor` | `LCR` or `calculated` |
| `event` | Nonnegative cumulative observed-cycle number |
| `available_event` | Cycle coordinate when this measurement becomes available; >= `event` |
| `capacitance_uf` | Positive finite capacitance, explicitly converted to microfarads |

Pairs may be sparse. A sensor may be absent. Duplicate sensor/event keys are rejected, because silently choosing one can change the EOL reference or reported error. Sort order is handled by the loader. Missing observations should be omitted; zeros and NaNs are not missing-data placeholders.

Do not combine different devices into one series. Preserve `source_file`, temperature, downtime and acquisition settings in a separate private audit table; this first portable runner uses only the required columns. It does not model temperature or changing experimental stress.

## Before converting raw recordings

1. Confirm channel assignments, voltage/current polarity, dividers, shunt conversion and physical units for every acquisition version. A similarly named CSV does not establish the same conversion.
2. Separate charge, discharge, LCR and relay-switching intervals. Detect cycles from a logged counter where trustworthy or validate a voltage-based detector. Count rejected-fit cycles in the exposure axis.
3. Fit accepted discharge tails to `V(t) = A exp(-t/tau)`. Inspect residuals, bounds, gaps and current stability. Derive a resistance candidate from the actual circuit/current observations, not a guessed nominal discharge resistor.
4. If justified, calculate `C_uF = 1e6 * tau_s / R_ohm`. Keep it independent of LCR when comparing sensors. Do not calibrate each session to LCR and then call the resulting target an independent physics measurement.
5. Collapse each settled LCR window to a robust measurement. Record its availability when the full window finishes. Log frequency, excitation, series/parallel equivalent mode, fixture compensation and relay configuration.
6. Retain a documented physical reference and rejected-observation audit. This package intentionally does not guess units or reuse the old setup's voltage thresholds, current scaling or capacitance-screen limits.

The repository starts from prepared capacitance observations because hardware conversion is setup-specific. It does **not** accept an arbitrary DAQ CSV and certify its circuit or capacitance values automatically.

## Bins and inputs

The runner takes medians in fixed 500-event bins. A bin becomes usable only after its upper event boundary and all included measurements are available. Its event is the median measurement event; delayed results are not moved backwards in time. Consequently an unclosed final bin may be unavailable for a live forecast.

Input history is fixed at 10,000 events with a 500-event step. Channels are relative capacitance, measurement age and a missing flag. Gaps remain masked; no future interpolation is used. Origins with fewer than eight historical bins, insufficient stage support, fewer than five recent bins or a last measurement over 1,500 events old are unsupported. `split_audit.csv` reports the actual supported test count; sparse support must be disclosed with accuracy.

Forecast targets are the closest bin median within 250 events of the requested horizon. The effective horizon is exported. Do not resample a coarse experiment into many artificial independent observations.

## Configuration

Start from `examples/full_config.json`, change the references and cutoffs for your data, and keep the edited file with your results.

- `reference_capacitance_uf`: fixed value for each supplied sensor, justified independently of future outcomes. A first-window median can be more stable than a single response, but changing reference changes the EOL definition. Record that choice rather than silently picking a convenient maximum.
- `eol_fraction`: default 0.8. This is the project's criterion, not a universal usability guarantee for every capacitor/application.
- `cutoffs`: distinct issue-block starts, multiples of 500 and >=30k. Blocks must not overlap.
- `horizons`: positive multiples of 500. These control capacitance forecast evaluation.
- `test_block`: 5k by default. Input observations update during the block; weights do not.
- `models`: `plain`, `stage`, and optionally `rc`, `dual`. Baselines always run. RC and dual arms run only for LCR, supervised by independently paired calculated labels. They are not two independent-sensor predictors.
- `rc_weight`: fixed auxiliary weight, default 0.01. The portable runner does not search it. Predeclare it or choose it in a separate earlier validation experiment.
- `seeds`, `max_epochs`, `mc_samples`, `rul_draws`: declare before inspecting test scores. The quickstart is deliberately small; its numbers are not the historical study.
- `rul_limit`: finite reporting horizon, default 60k events. Beyond it, estimates are right-censored.

Keep 10k of earlier outcomes for tuning. Inner training uses labels available by cutoff-10k. Validation origins lie in [cutoff-10k, cutoff-5k), with labels available by cutoff. The chosen epoch count is then used to refit on all labels available by cutoff. An epoch-zero choice preserves recent linear. For paired LCR objectives, the common training registry waits for both labels when an auxiliary pair exists, including in the control arms, to preserve comparable availability.

## Outputs and interpretation

`fold_metrics.csv` has R² within each block, not a full-record fit score. The summary computes sqrt(mean fold MSE), averages fold MAE and reports MSE skill against recent linear. It does not average predictions or pool sensors. A negative skill means the method lost to the baseline.

`rul.csv` reports conditional rate-based first passage. The observed crossing bracket comes from adjacent above/below bin medians and is a retrospective evaluation label only. Already-observed first passage is zero even if the capacitor later recovers. If no bracket is observed, RUL error stays unavailable. Censored medians are scored at the configured boundary for restricted bracket distance; this is not an exact unrestricted lifetime error. Missing/support failures must not be converted to successful zero-RUL predictions.

The RUL inference scale remains 10k events; training very different horizons does not establish accuracy at long lifetimes. Baseline intervals are point-only. Neural intervals are Gaussian moment approximations and are not calibrated by the portable runner.
