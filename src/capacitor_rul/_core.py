"""Numerical core extracted from the completed stage-context experiment.

The 10k history / 500-event step and stage definitions match the research core.
This module contains no experiment-directory or machine-specific dependencies.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch import nn
LOOKBACK=10000
STEP=500
STAGE_PENALTY=2.0
MAX_STAGES=6
MIN_POINTS=8
MIN_SPAN=5000
HORIZONS=(1000,2500,5000,10000)
CONTEXT_NAMES=['stage_minus_recent_rate','previous_minus_recent_rate','last_jump',
 'stage_span_10k','stage_residual_sd','stage_minus_recent_level','has_previous_stage','stage_count_at_cap']


def line(x, y):
    x = np.asarray(x, float) / 10000.
    y = np.asarray(y, float)
    centre = float(x.mean())
    slope = float(np.sum((x-centre)*(y-y.mean())) / max(np.sum((x-centre)**2), 1e-12))
    return float(y.mean()), slope, centre

def piecewise(b, penalty=2.):
    """Exact DP for discontinuous linear segments; BIC-like penalty counts knots.

    At most six stages, at least eight occupied bins and 5k events per stage.
    The independence assumption in BIC is approximate for serial sensor data;
    penalty strength is selected by chronological forecast error, not fit R2.
    """
    x = b.event.to_numpy(float)/10000.
    y = b.value.to_numpy(float)
    n = len(y)
    if n < MIN_POINTS or b.event.iloc[-1]-b.event.iloc[0] < MIN_SPAN:
        raise ValueError('Insufficient history for a stage')
    # Center y to reduce cancellation in SSE prefix sums.
    y0 = y-y.mean()
    pref = [np.r_[0., np.cumsum(a)] for a in (x, y0, x*x, x*y0, y0*y0)]
    cost = np.full((n+1, n+1), np.inf)
    for i in range(n-MIN_POINTS+1):
        ends = np.arange(i+MIN_POINTS, n+1)
        ends = ends[(x[ends-1]-x[i])*10000 >= MIN_SPAN]
        count = ends-i
        sx, sy, sxx, sxy, syy = [a[ends]-a[i] for a in pref]
        xx = sxx-sx*sx/count
        xy = sxy-sx*sy/count
        cost[i, ends] = np.maximum(syy-sy*sy/count-xy*xy/np.maximum(xx, 1e-12), 0.)
    dp = np.full((MAX_STAGES+1, n+1), np.inf)
    prev = np.full(dp.shape, -1, int)
    dp[0, 0] = 0.
    for k in range(1, MAX_STAGES+1):
        for end in range(k*MIN_POINTS, n+1):
            v = dp[k-1, :end]+cost[:end, end]
            start = int(np.argmin(v))
            dp[k, end], prev[k, end] = v[start], start
    scores = [n*np.log(max(dp[k, n]/n, 1e-8)) + penalty*(3*k-1)*np.log(n)
              for k in range(1, MAX_STAGES+1)]
    k = int(np.argmin(scores))+1
    segments = []
    end = n
    while k:
        start = prev[k, end]
        if start < 0: raise ValueError('No valid partition')
        a, slope, centre = line(b.event.iloc[start:end], b.value.iloc[start:end])
        segments.append({'start':int(start), 'end':int(end), 'level':a, 'slope':slope, 'centre':centre})
        end = start
        k -= 1
    return list(reversed(segments))

def weights(c):
    """Equal nominal-horizon weights; one case per origin and horizon."""
    h = c.horizon.to_numpy()
    w = np.zeros(len(c))
    for v in np.unique(h):
        take = h == v
        w[take] = 1. / take.sum()
    return w / w.sum()

class SensorData:
    def __init__(self, b):
        self.b = b.sort_values('event').reset_index(drop=True)
        self.cache = {}

    def origin(self, origin):
        if origin in self.cache:
            return self.cache[origin]
        past = self.b[self.b.available <= origin].reset_index(drop=True)
        if len(past) < 8 or origin - past.last_event.iloc[-1] > 1500:
            self.cache[origin] = None
            return None
        recent = past[past.event >= origin - LOOKBACK]
        if len(recent) < 5:
            self.cache[origin] = None
            return None
        try:
            segments = piecewise(past, STAGE_PENALTY)
        except ValueError:
            self.cache[origin] = None
            return None
        a, rate, centre = line(recent.event, recent.value)
        base = a + rate * (origin / 10000. - centre)
        latest = float(past.value.iloc[-1])
        s = segments[-1]
        stage = past.iloc[s['start']:s['end']]
        stage_level = s['level'] + s['slope'] * (origin / 10000. - s['centre'])
        residual = stage.value - (s['level'] + s['slope'] * (stage.event / 10000. - s['centre']))
        previous_rate, jump = rate, 0.
        if len(segments) > 1:
            prev = segments[-2]
            x = float(stage.event.iloc[0]) / 10000.
            previous_rate = prev['slope']
            jump = s['level'] + s['slope'] * (x-s['centre']) - (prev['level'] + prev['slope'] * (x-prev['centre']))
        ctx = np.array([s['slope']-rate, previous_rate-rate, jump,
                        (origin-stage.event.iloc[0])/10000., float(residual.std(ddof=0)),
                        stage_level-base, float(len(segments)>1), float(len(segments)==6)], np.float32)
        # Each window timestep holds only a bin that had closed by that timestep.
        # Missing periods stay masked. No interpolation through future observations.
        seq = []
        for end in range(origin-LOOKBACK+STEP, origin+1, STEP):
            known = past[past.available <= end]
            if known.empty or end-known.last_event.iloc[-1] > 1500:
                seq.append([0., 1., 1.])
            else:
                r = known.iloc[-1]
                seq.append([float(r.value-latest), float((end-r.last_event)/1500.), 0.])
        ans = {'X':np.array(seq, np.float32), 'context':ctx, 'base':float(base), 'rate':float(rate),
               'persistence':latest, 'piecewise':float(stage_level), 'stage_rate':float(s['slope']),
               'stage_count':len(segments), 'stage_start':float(stage.event.iloc[0]),
               'available_max':float(past.available.max()), 'last_event':float(past.last_event.iloc[-1])}
        self.cache[origin] = ans
        return ans

    def cases(self, horizons=HORIZONS):
        rows = []
        for origin in range(LOOKBACK, int(self.b.available.max())+1, STEP):
            d = self.origin(origin)
            if d is None:
                continue
            for h in horizons:
                distance = abs(self.b.event-(origin+h))
                t = self.b.iloc[int(distance.argmin())]
                if distance.min() > STEP/2:
                    continue
                rows.append({'origin':origin, 'horizon':h, 'target_event':float(t.event),
                             'target_available':float(t.available), 'actual':float(t.value),
                             'effective_horizon':float(t.event-origin),
                             'input_available_max':d['available_max']})
        if not rows:
            raise ValueError('No supported forecast cases: check history length, gaps and requested horizons')
        c = pd.DataFrame(rows)
        assert (c.input_available_max <= c.origin).all()
        assert (c.target_event > c.origin).all() and (c.target_available >= c.target_event).all()
        assert not c.duplicated(['origin','horizon']).any()
        return c

    def arrays(self, c):
        ds = [self.origin(int(o)) for o in c.origin]
        assert all(d is not None for d in ds)
        return (np.stack([d['X'] for d in ds]), np.stack([d['context'] for d in ds]),
                np.array([d['base'] for d in ds]), np.array([d['rate'] for d in ds]))

class StageLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(3, 16, batch_first=True)
        self.dropout = nn.Dropout(.25)
        # Identical parameter count in both arms; context is zero in the control.
        self.head = nn.Linear(16+len(CONTEXT_NAMES), 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, context):
        _, (h, _) = self.lstm(x)
        z = self.head(self.dropout(torch.cat((h[-1], context), dim=1)))
        return z[:,0], z[:,1].clamp(-8., 5.)

def scaler(data, train):
    X, ctx, base, rate = data.arrays(train)
    observed = X[:,:,0][X[:,:,2] == 0]
    residual = (train.actual.to_numpy()-base)/(train.effective_horizon.to_numpy()/10000.)-rate
    w = weights(train)
    return {'sequence_mean':float(observed.mean()), 'sequence_sd':max(float(observed.std()),1.),
            'context_mean':ctx.mean(axis=0).tolist(), 'context_sd':np.maximum(ctx.std(axis=0),1e-3).tolist(),
            'rate_scale':max(float(np.sqrt(np.sum(w*residual**2))),1.),
            'max_label_available':float(train.target_available.max())}

def tensors(data, c, sc, mode):
    X, ctx, base, rate = data.arrays(c)
    X[:,:,0] = np.where(X[:,:,2] == 0, (X[:,:,0]-sc['sequence_mean'])/sc['sequence_sd'], 0.)
    ctx = (ctx-np.array(sc['context_mean']))/np.array(sc['context_sd'])
    if mode == 'plain':
        ctx[:] = 0.
    return torch.tensor(X, dtype=torch.float32), torch.tensor(ctx, dtype=torch.float32), base, rate

def predict(data, fit, c, samples=100, stochastic=True):
    X, ctx, base, rate = tensors(data, c, fit['scaler'], fit['mode'])
    model = fit['model']
    model.train(stochastic)
    means, variances = [], []
    scale = fit['scaler']['rate_scale']
    with torch.no_grad():
        for _ in range(samples):
            m, lv = model(X, ctx)
            means.append(rate + scale*m.numpy())
            variances.append(scale**2 * np.exp(lv.numpy()))
    m, v = np.stack(means), np.stack(variances)
    h = c.effective_horizon.to_numpy()/10000.
    return base+h*m.mean(axis=0), fit['noise']**2+h**2*(v.mean(axis=0)+m.var(axis=0)), (m, v, base)

def fit(data, train, tune, mode, seed, max_epochs, epochs=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sc = scaler(data, train)
    model = StageLSTM()
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
    X, ctx, base, rate = tensors(data, train, sc, mode)
    h = train.effective_horizon.to_numpy()/10000.
    target = torch.tensor((train.actual.to_numpy()-base-h*rate)/sc['rate_scale'], dtype=torch.float32)
    ht = torch.tensor(h, dtype=torch.float32)
    w = torch.tensor(weights(train)*len(train), dtype=torch.float32)
    # Estimate a robust short-scale noise floor from TRAIN observations only.
    b = data.b[data.b.available <= train.target_available.max()]
    delta = b.value.diff()[b.event.diff() <= 1000].dropna().to_numpy()
    noise = max(float(1.4826*np.median(abs(delta-np.median(delta)))/np.sqrt(2)), .35) if len(delta) else .35
    bundle = {'model':model, 'scaler':sc, 'noise':noise, 'mode':mode, 'seed':seed}
    best_score = np.inf
    best_epoch = 0
    best = None
    history = []
    stale = 0
    # Epoch zero is the unmodified recent-linear point forecast. Learning must help
    # on earlier tuning labels before it is permitted to replace that fallback.
    for ep in range((max_epochs if epochs is None else epochs)+1):
        losses = []
        if ep:
            model.train()
            order = rng.permutation(len(train))
            for start in range(0, len(train), 128):
                idx = order[start:start+128]
                m, lv = model(X[idx], ctx[idx])
                var = ht[idx]**2*lv.exp()+(noise/sc['rate_scale'])**2
                loss = (.5*(var.log()+(target[idx]-ht[idx]*m)**2/var)*w[idx]).mean()
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),5.)
                opt.step()
                losses.append(float(loss.detach()))
        score = None
        if tune is not None:
            mu, _, _ = predict(data,bundle,tune,samples=1,stochastic=False)
            score = float(np.sqrt(np.sum(weights(tune)*(mu-tune.actual.to_numpy())**2)))
            if score < best_score-1e-5:
                best_score, best_epoch, stale = score, ep, 0
                best = {k:v.detach().clone() for k,v in model.state_dict().items()}
            else:
                stale += 1
        history.append({'epoch':ep,'train_scaled_NLL':float(np.mean(losses)) if losses else None,
                        'tuning_RMSE':score})
        if epochs is None and stale >= 8:
            break
    if epochs is None:
        model.load_state_dict(best)
    bundle.update(selected_epochs=best_epoch if epochs is None else epochs, history=history,
                  tuning_RMSE=best_score if tune is not None else None, n_train=len(train))
    return bundle

def metric(c):
    w = weights(c)
    y = c.actual.to_numpy()
    e = c.prediction.to_numpy()-y
    var_y = float(np.sum(w*(y-np.sum(w*y))**2))
    mse = float(np.sum(w*e**2))
    out = {'n':len(c), 'n_origins':c.origin.nunique(), 'n_horizons':c.horizon.nunique(),
           'RMSE':np.sqrt(mse), 'MAE':float(np.sum(w*abs(e))), 'bias':float(np.sum(w*e)),
           'R2':1-mse/var_y if var_y > 1e-12 else np.nan}
    if 'variance' in c and c.variance.notna().all():
        v = c.variance.to_numpy()
        sd = np.sqrt(v)
        lo, hi = c.prediction.to_numpy()-1.96*sd, c.prediction.to_numpy()+1.96*sd
        score = hi-lo + 40*np.maximum(lo-y,0)+40*np.maximum(y-hi,0)
        out.update(coverage95=float(np.sum(w*(abs(e)<=1.96*sd))), width95=float(np.sum(w*3.92*sd)),
                   Gaussian_NLL=float(np.sum(w*.5*(np.log(2*np.pi*v)+e**2/v))),
                   interval_score95=float(np.sum(w*score)))
    return out
