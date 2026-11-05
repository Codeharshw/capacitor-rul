"""Conditional first-passage paths; one sampled rate is shared across each path."""
import numpy as np


def crossing_bracket(bins, threshold):
    below=bins[bins.value<threshold]
    if below.empty:return None,None
    first=below.iloc[0]
    preceding=bins[(bins.event<first.event)&(bins.value>=threshold)]
    return (float(preceding.event.iloc[-1]) if len(preceding) else None),float(first.event)


def summarize(level,rate,threshold,limit=60000):
    level,rate=np.broadcast_arrays(np.asarray(level,float),np.asarray(rate,float))
    times=np.full(rate.shape,np.inf)
    times[level<threshold]=0.
    use=(rate<0)&(level>=threshold)
    times[use]=(level[use]-threshold)/(-rate[use])*10000
    times[times>limit]=np.inf
    result={}
    for key,p in [('p025',.025),('median',.5),('p975',.975)]:
        q=float(np.quantile(times,p,method='inverted_cdf'))
        result['RUL_'+key]=q if np.isfinite(q) else None
        result[key+'_right_censored']=not np.isfinite(q)
    result['no_crossing_by_limit']=float(np.isinf(times).mean())
    return result


def bracket_distance(median,lo,hi,limit=60000):
    if lo is None or hi is None:return None
    p=limit if median is None else np.clip(median,0,limit)
    return float(max(np.clip(lo,0,limit)-p,p-np.clip(hi,0,limit),0))
