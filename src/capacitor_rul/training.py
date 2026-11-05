"""Later small residual-rate architecture; controlled objectives and past-only tuning."""
import numpy as np
import torch
from ._core import StageLSTM, scaler, tensors, predict, weights
from .losses import gaussian_nll, dual_gaussian_nll


def train(data, cases, validation, mode, seed, epochs, selected_epochs=None, rc_weight=.01):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    context = 'stage' if mode=='stage' else 'plain'
    sc = scaler(data,cases)
    model = StageLSTM()
    optimizer = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    X,ctx,base,rate = tensors(data,cases,sc,context)
    h = torch.tensor(cases.effective_horizon.to_numpy()/10000.,dtype=torch.float32)
    target = torch.tensor((cases.actual.to_numpy()-base-h.numpy()*rate)/sc['rate_scale'],dtype=torch.float32)
    auxiliary = torch.tensor((cases.aux_actual.to_numpy()-base-h.numpy()*rate)/sc['rate_scale'],dtype=torch.float32)
    valid = torch.isfinite(auxiliary)
    if mode in ['rc','dual'] and valid.sum()<5:
        raise ValueError(f'{mode} requires at least five independently paired training targets')
    b = data.b[data.b.available<=cases.target_available.max()]
    delta = b.value.diff()[b.event.diff()<=1000].dropna().to_numpy()
    noise = max(float(1.4826*np.median(abs(delta-np.median(delta)))/np.sqrt(2)),.35) if len(delta) else .35
    bundle = dict(model=model,scaler=sc,noise=noise,mode=context,seed=seed,objective=mode)
    w = torch.tensor(weights(cases)*len(cases),dtype=torch.float32)
    best_score,chosen,stale = np.inf,0,0
    best = None
    history = []
    limit = epochs if selected_epochs is None else selected_epochs
    for ep in range(limit+1):
        losses=[]
        if ep:
            model.train()
            order=rng.permutation(len(cases))
            for start in range(0,len(cases),128):
                ix=order[start:start+128]
                mean,lv=model(X[ix],ctx[ix])
                mean=h[ix]*mean
                var=h[ix]**2*lv.exp()+(noise/sc['rate_scale'])**2
                if mode=='dual':
                    per=dual_gaussian_nll(mean,var.log(),target[ix],auxiliary[ix],valid[ix])
                    loss=(per*w[ix]).mean()
                else:
                    loss=(gaussian_nll(mean,var.log(),target[ix])*w[ix]).mean()
                    if mode=='rc' and valid[ix].any():
                        # For positive R, normalized RC-tau equals error against tau/R.
                        use=valid[ix]
                        loss=loss+rc_weight*.5*(mean[use]-auxiliary[ix][use]).square().mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite training objective')
                optimizer.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                optimizer.step();losses.append(float(loss.detach()))
        value=None
        if validation is not None:
            mu,_,_=predict(data,bundle,validation,samples=1,stochastic=False)
            value=float(np.sqrt(np.sum(weights(validation)*(mu-validation.actual.to_numpy())**2)))
            if value<best_score-1e-5:
                best_score,chosen,stale=value,ep,0
                best={k:v.detach().clone() for k,v in model.state_dict().items()}
            else:stale+=1
        history.append({'epoch':ep,'training_NLL_or_composite':np.mean(losses) if losses else None,'validation_RMSE':value})
        if validation is not None and stale>=8:break
    if validation is not None:model.load_state_dict(best)
    bundle.update(selected_epochs=chosen if validation is not None else limit,history=history,n_train=len(cases))
    return bundle
