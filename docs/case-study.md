# Engineering case study

## Problem

Estimate how capacitance will evolve and how many observed charge/discharge cycles remain before a fixed 20% loss threshold. Two measurement methods were available: an LCR meter and independent discharge-derived capacitance candidates.

## What I built

1. Audited the experiment recordings, units, logger-channel interpretation, excluded experiments and delayed LCR measurements.
2. Reconstructed observed cycle exposure without inventing cycles during gaps. Derived discharge time constants independently of LCR calibration.
3. Compared the two sensors in absolute capacitance, normalized loss, sensor ratio and stage-specific degradation rate.
4. Built chronological forecasting experiments with persistence, average drift and recent-slope controls.
5. Compared standard MC-dropout LSTMs, an RC auxiliary-loss variant, and a separate dual-observation loss experiment.
6. Fitted past-only piecewise stages, tested their incremental value in independent-sensor Bayesian LSTMs, and assessed interval calibration.
7. Evaluated advance RUL predictions with fixed references, observation-gap brackets and censored estimates.
8. Tested adaptive local-rate forecasts and validation-selected neural blends, then consolidated the findings rather than selecting a favorable plot.

## Key engineering decisions

**Forecasting rather than reconstructing a derived target.** A time constant used to calculate a capacitance target must not be supplied from that target's future cycle as a forecast input.

**Availability rather than row order alone.** A fitted discharge interval or LCR window can become available after its nominal event. The model uses information only when it is available.

**Own-sensor evaluation.** LCR and calculated capacitance differ in scale and relative degradation. An arithmetic compromise is not measured ground truth.

**Baseline accountability.** The recent-linear forecast remained stronger than the neural alternatives in aggregate capacitance RMSE. That is a useful result, not a reason to omit the baseline.

**Interval sharpness as well as coverage.** The dual loss could cover both measurements with a very broad band while being a poor LCR predictor. Interval scores penalize that behavior.

**First-passage uncertainty.** Forecasting capacitance 10k events ahead and predicting time to a limit are different tasks. Small capacitance errors near a slowly declining plateau can cause large lifetime errors.

## Outcome

Stage context reduced LSTM RMSE by about 1.36% for LCR and 5.39% for calculated capacitance in the later experiment. It did not beat the recent-linear baseline overall. The original RC constraint worsened held-out LCR error. The direct dual loss moved predictions toward a compromise and inflated variance.

The current record already passed its chosen limits, so current conditional RUL is zero. That historical threshold decision does not establish accurate advance failure prediction. New devices, future observations and independent measurement calibration are needed for stronger reliability claims.

## Skills demonstrated

Time-series data engineering, measurement-model debugging, PyTorch sequence models, controlled ablations, uncertainty evaluation, reproducible experiments and communicating negative results.

## Scope

The current release contains CPU-friendly LSTMs and experimental RC/dual losses. No CNN result or deployment on live hardware is claimed. A live system would need persistent acquisition state, measured label delays, monitoring, independent validation and a retraining policy.
