# Models and loss functions

## Standard approximate Bayesian forecast

The historical original model had two 32-unit LSTM layers, dropout 0.3 and Gaussian mean/log-variance heads. A scaled change target allowed extrapolation outside the absolute capacitance range seen during training. The Gaussian NLL, omitting its additive constant, is

```text
L_NLL = mean(0.5 * [ell + (z - mu)^2 * exp(-ell)])
```

Dropout stays on at inference. Across sampled means mu_t and variances v_t:

```text
predictive_mean = mean_t(mu_t)
total_variance = mean_t(v_t) + variance_t(mu_t)
```

This uses the MC-dropout approximation. It does not implement an explicit variational KL loss or exact Bayesian inference. `mean ± 1.96 SD` is a Gaussian moment prediction band; coverage must be checked.

## RC physics-informed auxiliary objective

The original experiment supervised LCR using NLL and added

```text
C_hat_uF = current_level + TRAIN_mean_delta + TRAIN_SD_delta * mu
L_RC = mean_eligible(0.5 * [(R*C_hat_uF*1e-6 - tau)
                          / max(abs(R)*1e-6*TRAIN_SD_delta, 1e-12)]^2)
L_total = L_NLL + lambda * L_RC
```

For positive R and an inactive numerical floor this is scaled squared error against `C_calc = 1e6*tau/R`. Future calculated values are training labels, never future inference inputs. Original lambda candidates 0.001/0.01/0.1 were selected on earlier validation LCR NLL. Source circuit/current calibration was unresolved, so a smaller RC residual did not establish improved physical truth.

Call this an **experimental RC auxiliary-loss / PINN-inspired model**, not a validated PDE-based or physical aging PINN. It enforces an observation relation and does not predict unknown degradation regime changes.

## Dual-observation objective

Both targets must use a common forecast baseline, training-only offset and scale. For alpha=0.5 when calculated labels are eligible (zero otherwise):

```text
L_dual = mean(0.5 * [ell + ((1-alpha)*(mu-z_LCR)^2
                          + alpha*(mu-z_calc)^2) * exp(-ell)])
```

This is a weighted composite loss with shared variance, not a Gaussian-mixture likelihood. Its error decomposes into

```text
(mu - [(1-alpha)*z_LCR + alpha*z_calc])^2
    + alpha*(1-alpha)*(z_LCR-z_calc)^2
```

The mean favors a compromise; sensor disagreement can inflate the fitted variance. Neither a training envelope nor the loss guarantees the true future capacitance or RUL lies between the sensors. No optional envelope constraint was included in the reported experiment.

## Later stage-context experiment and this package

The later architecture is one 16-unit LSTM layer, dropout 0.25 and a two-output head. Eight extra features describe piecewise stages fitted only from available history. The control receives zeros for those features and has the same parameter count (1,394). A dynamic program fits up to six discontinuous linear stages with minimum eight bins / 5k events and a BIC-like penalty. Statistical stages are not independently verified physical failure mechanisms.

The model learns a correction to a recent-linear rate. With h = effective_horizon/10k, baseline level b, recent rate r, and TRAIN scale s:

```text
C_hat = b + h * (r + s * mu)
target = (C_actual - b - h*r) / s
variance_scaled = h^2 * exp(ell) + (noise_floor/s)^2
```

All scaling and noise-floor estimates use training data. Epoch selection uses earlier validation RMSE, including an epoch-zero baseline fallback; AdamW uses learning rate 0.001, weight decay 0.001, batch size 128, gradient clipping 5, and patience 8.

The portable package uses this later architecture for **all** configurable arms. Its experimental `rc` arm uses the equivalent squared auxiliary target error on the common rate-residual scale, with a fixed coefficient. Its `dual` arm uses the common residual scale and fixed alpha=0.5. They use the same small backbone and train/validation schedule as the controls. They therefore **do not reproduce** the original 32-unit experiments or the historical Step 11 objective-specific centering by default. This is a controlled reusable implementation, with the historical formulas and numbers documented separately.

## RUL

For a fixed reference C0, EOL is first passage below 0.8 C0. A sampled path has one initial level and one degradation rate shared over the whole future path. Under that assumed rate:

```text
RUL = 0                                 if already below the limit
RUL = (level - 0.8*C0) / (-rate) * 10000  if rate < 0
RUL = censored                          otherwise / beyond report horizon
```

Do not independently sample each future time point and stitch them into a trajectory; that creates artificial crossings. The portable runner samples the MC/aleatoric rate mixture and initial noise, then keeps the rate coherent within a path. It does not model future regime switches or calibrate RUL probability coverage.

## References

- Gal & Ghahramani (2016), [Dropout as a Bayesian Approximation](https://proceedings.mlr.press/v48/gal16.html).
- Kendall & Gal (2017), [What Uncertainties Do We Need in Bayesian Deep Learning?](https://arxiv.org/abs/1703.04977).
- [PyTorch LSTM reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html).
- [Time-series validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).

These sources motivate methods; they do not validate this device's measurement model or results.
