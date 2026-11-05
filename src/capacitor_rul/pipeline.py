"""Portable chronological backtest. Outputs never select hyperparameters."""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import __version__
from ._core import SensorData, predict, metric
from .data import load_measurements, aggregate, match_auxiliary
from .training import train
from .rul import crossing_bracket,summarize,bracket_distance


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path,value):
    def clean(obj):
        if isinstance(obj,dict):return {k:clean(v) for k,v in obj.items()}
        if isinstance(obj,(list,tuple)):return [clean(x) for x in obj]
        if isinstance(obj,np.generic):obj=obj.item()
        if isinstance(obj,float) and not np.isfinite(obj):return None
        return obj
    Path(path).write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n',encoding='utf-8')


def load_config(path, sensors):
    c=json.loads(Path(path).read_text(encoding='utf-8'))
    for name in ['cutoffs','horizons','reference_capacitance_uf']:
        if name not in c:raise ValueError('Config missing '+name)
    c.setdefault('test_block',5000);c.setdefault('seeds',[42,52,62]);c.setdefault('max_epochs',60)
    c.setdefault('mc_samples',100);c.setdefault('rul_draws',2000);c.setdefault('rul_limit',60000)
    c.setdefault('rc_weight',.01);c.setdefault('eol_fraction',.8)
    c.setdefault('models',['plain','stage']);c.setdefault('threads',2)
    if not c['cutoffs'] or len(set(c['cutoffs']))!=len(c['cutoffs']) or any(x<30000 or x%500 for x in c['cutoffs']):
        raise ValueError('Distinct cutoffs must be multiples of 500 and >=30000')
    if not c['horizons'] or any(x<=0 or x%500 for x in c['horizons']):raise ValueError('Horizons must be positive multiples of 500')
    if not 0<c['eol_fraction']<1:raise ValueError('eol_fraction must be between 0 and 1')
    for s in sensors:
        x=c['reference_capacitance_uf'].get(s)
        if x is None or not np.isfinite(x) or x<=0:raise ValueError(f'Provide a positive, fixed reference for {s}')
    if not set(c['models'])<=set(['plain','stage','rc','dual']):raise ValueError('Unknown model name')
    if any(c[x]<1 for x in ['max_epochs','mc_samples','rul_draws','threads']):raise ValueError('Training/draw/thread counts must be positive')
    if not c['seeds'] or len(set(c['seeds']))!=len(c['seeds']):raise ValueError('Provide distinct seeds')
    if c['test_block']<=0 or c['test_block']%500 or c['rul_limit']<=0 or c['rc_weight']<0:raise ValueError('Invalid block, RUL limit or RC weight')
    ordered=sorted(c['cutoffs'])
    if any(a+c['test_block']>b for a,b in zip(ordered,ordered[1:])):raise ValueError('Test issue blocks must not overlap')
    return c


def run(csv_path,config_path,out):
    raw=load_measurements(csv_path)
    cfg=load_config(config_path,raw.sensor.unique())
    out=Path(out)
    if out.exists():raise ValueError(f'Output exists: {out}. Use a new folder to preserve results.')
    out.mkdir(parents=True)
    torch.set_num_threads(cfg['threads'])
    json_write(out/'config.json',cfg)
    json_write(out/'provenance.json',dict(package_version=__version__,data_sha256=sha256(csv_path),
        config_sha256=sha256(config_path),python=platform.python_version(),torch=torch.__version__,
        numpy=np.__version__,pandas=pd.__version__,new_device_validation=False,
        comparison='portable reimplementation; does not reproduce historical numbers by default'))
    bins={s:aggregate(g) for s,g in raw.groupby('sensor')}
    for s,b in bins.items():b.to_csv(out/f'{s}_bins.csv',index=False)
    predictions,fold_metrics,selections,audits,rul_rows=[],[],[],[],[]
    for sensor,b in bins.items():
        data=SensorData(b)
        cases=data.cases(tuple(cfg['horizons']))
        cases=match_auxiliary(cases,bins.get('calculated') if sensor=='LCR' else None)
        cases.to_csv(out/f'{sensor}_cases.csv',index=False)
        threshold=cfg['reference_capacitance_uf'][sensor]*cfg['eol_fraction']
        observed_lo,observed_hi=crossing_bracket(b,threshold)
        for cut in cfg['cutoffs']:
            inner=cases[cases.label_available<=cut-10000].reset_index(drop=True)
            val=cases[(cases.origin>=cut-10000)&(cases.origin<cut-5000)&(cases.label_available<=cut)].reset_index(drop=True)
            outer=cases[cases.label_available<=cut].reset_index(drop=True)
            test=cases[(cases.origin>=cut)&(cases.origin<cut+cfg['test_block'])].reset_index(drop=True)
            if any(z.empty for z in [inner,val,outer,test]):
                raise ValueError(f'{sensor} cutoff {cut}: insufficient train/validation/test support; adjust config')
            assert inner.label_available.max()<=val.origin.min()
            assert outer.label_available.max()<=test.origin.min()
            audits.append(dict(sensor=sensor,cutoff=cut,inner_n=len(inner),inner_label_max=inner.label_available.max(),
                validation_origin_min=val.origin.min(),validation_label_max=val.label_available.max(),
                outer_label_max=outer.label_available.max(),test_origin_min=test.origin.min(),test_n=len(test)))
            states=[data.origin(int(o)) for o in test.origin]
            for method in ['persistence','recent_linear','piecewise']:
                if method=='persistence':values=[d['persistence'] for d in states]
                elif method=='recent_linear':values=[d['base']+d['rate']*h/10000 for d,h in zip(states,test.effective_horizon)]
                else:values=[d['piecewise']+d['stage_rate']*h/10000 for d,h in zip(states,test.effective_horizon)]
                z=test.assign(prediction=values,sensor=sensor,cutoff=cut,method=method)
                predictions.append(z)
                fold_metrics.append(dict(sensor=sensor,cutoff=cut,method=method,**metric(z)))
            origins=[o for o in range(cut,cut+cfg['test_block'],500) if data.origin(o) is not None]
            issued=pd.DataFrame(dict(origin=origins,horizon=10000,effective_horizon=10000))
            components={}
            for mode in cfg['models']:
                if mode in ['rc','dual'] and sensor!='LCR':continue
                means,variances,rate_means,rate_vars,noises=[],[],[],[],[]
                for seed in cfg['seeds']:
                    tuned=train(data,inner,val,mode,seed,cfg['max_epochs'],rc_weight=cfg['rc_weight'])
                    fitted=train(data,outer,None,mode,seed,cfg['max_epochs'],tuned['selected_epochs'],cfg['rc_weight'])
                    mu,var,_=predict(data,fitted,test,samples=cfg['mc_samples'])
                    means.append(mu);variances.append(var)
                    _,_,(rm,rv,_)=predict(data,fitted,issued,samples=cfg['mc_samples'])
                    rate_means.append(rm);rate_vars.append(rv);noises.extend([fitted['noise']]*len(rm))
                    checkpoint=out/'checkpoints'/f'{sensor}_{cut}_{mode}_{seed}'
                    checkpoint.mkdir(parents=True)
                    torch.save(fitted['model'].state_dict(),checkpoint/'weights.pt')
                    json_write(checkpoint/'metadata.json',{k:v for k,v in fitted.items() if k not in ['model','history']})
                    pd.DataFrame(tuned['history']).to_csv(checkpoint/'validation_history.csv',index=False)
                    selections.append(dict(sensor=sensor,cutoff=cut,method=mode,seed=seed,epochs=tuned['selected_epochs']))
                mu=np.mean(means,axis=0)
                variance=np.mean(variances,axis=0)+np.var(means,axis=0)
                z=test.assign(sensor=sensor,cutoff=cut,method=mode,prediction=mu,variance=variance,
                              lower95=mu-1.96*np.sqrt(variance),upper95=mu+1.96*np.sqrt(variance))
                predictions.append(z);fold_metrics.append(dict(sensor=sensor,cutoff=cut,method=mode,**metric(z)))
                components[mode]=(np.concatenate(rate_means),np.concatenate(rate_vars),np.asarray(noises))
            for j,origin in enumerate(origins):
                state=data.origin(origin)
                already=crossing_bracket(b[b.available<=origin],threshold)[1] is not None
                rng=np.random.default_rng(100000+origin)
                paths={m:(np.array([state['persistence'] if m=='persistence' else state['piecewise'] if m=='piecewise' else state['base']]),
                          np.array([0 if m=='persistence' else state['stage_rate'] if m=='piecewise' else state['rate']]))
                       for m in ['persistence','recent_linear','piecewise']}
                for mode,(rm,rv,noise) in components.items():
                    ix=rng.integers(0,len(rm),size=cfg['rul_draws'])
                    rate=rm[ix,j]+np.sqrt(rv[ix,j])*rng.normal(size=len(ix))
                    level=state['base']+noise[ix]*rng.normal(size=len(ix))
                    paths[mode]=(level,rate)
                for method,(level,rate) in paths.items():
                    row=dict(sensor=sensor,cutoff=cut,origin=origin,method=method,threshold=threshold,
                             reference_capacitance_uf=cfg['reference_capacitance_uf'][sensor],
                             status='already_below_limit' if already else 'conditional_constant_rate',
                             input_available_max=state['available_max'])
                    row.update(summarize(np.array([threshold-1]) if already else level,rate,threshold,cfg['rul_limit']))
                    lo=max(0,observed_lo-origin) if observed_lo is not None else None
                    hi=max(0,observed_hi-origin) if observed_hi is not None else None
                    row.update(observed_RUL_lo=lo,observed_RUL_hi=hi)
                    row['restricted_bracket_error']=bracket_distance(row['RUL_median'],lo,hi,cfg['rul_limit']) if not already else None
                    rul_rows.append(row)
            print(f'{sensor} cutoff {cut:,}: {len(test)} test cases complete',flush=True)
            # Checkpoint tables survive a later fold failure.
            pd.concat(predictions,ignore_index=True).to_csv(out/'predictions.csv',index=False)
            pd.DataFrame(fold_metrics).to_csv(out/'fold_metrics.csv',index=False)
            pd.DataFrame(audits).to_csv(out/'split_audit.csv',index=False)
            pd.DataFrame(selections).to_csv(out/'epoch_selection.csv',index=False)
            pd.DataFrame(rul_rows).to_csv(out/'rul.csv',index=False)
    p=pd.concat(predictions,ignore_index=True);m=pd.DataFrame(fold_metrics);r=pd.DataFrame(rul_rows)
    for _,g in p.groupby(['sensor','cutoff']):
        if g.groupby('method').size().nunique()!=1:raise AssertionError('Unequal test support')
    summary=m.groupby(['sensor','method']).agg(folds=('cutoff','size'),MSE=('RMSE',lambda x:np.mean(x**2)),MAE=('MAE','mean')).reset_index()
    summary['RMSE']=np.sqrt(summary.MSE)
    base=summary[summary.method=='recent_linear'][['sensor','MSE']].rename(columns={'MSE':'recent_MSE'})
    summary=summary.merge(base,on='sensor',validate='many_to_one')
    summary['skill_vs_recent_linear']=1-summary.MSE/summary.recent_MSE
    summary.to_csv(out/'summary.csv',index=False)
    for sensor in bins:
        fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
        z=summary[summary.sensor==sensor].sort_values('RMSE')
        axes[0].barh(z.method,z.RMSE,color='#316e87');axes[0].invert_yaxis()
        axes[0].set(title=f'{sensor}: matched capacitance error',xlabel='RMSE (uF)')
        pre=r[(r.sensor==sensor)&(r.status=='conditional_constant_rate')]
        truth=pre.drop_duplicates(['cutoff','origin']).sort_values('origin')
        for _,g in truth.groupby('cutoff'):
            axes[1].fill_between(g.origin/1000,g.observed_RUL_lo/1000,g.observed_RUL_hi/1000,color='grey',alpha=.25)
        for method,g in pre.groupby('method'):
            line=axes[1].plot(g.origin/1000,g.RUL_median/1000,'o',ms=3,label=method)[0]
            cens=g[g.RUL_median.isna()]
            axes[1].scatter(cens.origin/1000,np.full(len(cens),cfg['rul_limit']/1000),marker='^',color=line.get_color())
        axes[1].set(title='Pre-limit RUL; triangles indicate censoring',xlabel='Issue event (thousands)',ylabel='Remaining events (thousands)')
        if len(pre):axes[1].legend(fontsize=8)
        for ax in axes:ax.grid(alpha=.15)
        fig.savefig(out/f'{sensor}_comparison.png',dpi=160);plt.close(fig)
    (out/'README.txt').write_text('Fresh backtest of your supplied data. See summary.csv, fold_metrics.csv (including R2), predictions.csv and rul.csv.\nRUL assumes a constant sampled future rate; it cannot anticipate unknown changes. Censored medians are not zero.\nBaselines have point RUL only; their zero-width paths are not uncertainty intervals. Unsupported origins are omitted, not invented.\n',encoding='utf-8')
    json_write(out/'manifest.json',{str(f.relative_to(out)):sha256(f) for f in sorted(out.rglob('*')) if f.is_file()})
    (out/'COMPLETE.txt').write_text('COMPLETE\n',encoding='ascii')
    print(summary[['sensor','method','RMSE','MAE','skill_vs_recent_linear']].to_string(index=False),flush=True)
    return out
