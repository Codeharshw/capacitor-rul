"""Standard and experimental independent-observation losses in common target units."""
import torch


def gaussian_nll(mean, logvar, target):
    return .5*(logvar+(target-mean).square()*torch.exp(-logvar))


def dual_gaussian_nll(mean, logvar, lcr, calculated, valid, alpha=.5):
    if not 0<=alpha<=1:
        raise ValueError('alpha must be in [0,1]')
    safe = torch.where(valid,calculated,lcr)
    weight = valid.to(mean.dtype)*alpha
    squared = (1-weight)*(mean-lcr).square()+weight*(mean-safe).square()
    return .5*(logvar+squared*torch.exp(-logvar))


def rc_penalty(mean_capacitance_uf, tau_s, resistance_ohm, target_scale_uf):
    """Original RC relation; resistance and scale must be positive, validated inputs."""
    if torch.any(resistance_ohm<=0) or target_scale_uf<=0:
        raise ValueError('Positive resistance and target scale required')
    residual = resistance_ohm*mean_capacitance_uf*1e-6-tau_s
    denominator = (resistance_ohm.abs()*1e-6*target_scale_uf).clamp_min(1e-12)
    return .5*(residual/denominator).square()
