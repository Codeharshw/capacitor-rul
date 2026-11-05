"""Explicit event/availability schema. No unit guessing or sensor calibration."""
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = ['sensor','event','available_event','capacitance_uf']


def load_measurements(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(REQUIRED)-set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}. See docs/data-format.md')
    df = df[REQUIRED].copy()
    if df.empty or df.sensor.isna().any() or not df.sensor.isin(['LCR','calculated']).all():
        raise ValueError('Use sensor=LCR and/or calculated; at least one observation is required')
    for col in REQUIRED[1:]:
        df[col] = pd.to_numeric(df[col], errors='raise')
    if not np.isfinite(df[REQUIRED[1:]].to_numpy()).all():
        raise ValueError('Numeric values must be finite; omit missing measurements instead of filling them')
    if (df.event<0).any() or (df.capacitance_uf<=0).any():
        raise ValueError('Events must be nonnegative; capacitance must be positive and already in uF')
    if (df.available_event<df.event).any():
        raise ValueError('A measurement cannot be available before its measurement event')
    if df.duplicated(['sensor','event']).any():
        raise ValueError('Duplicate sensor/event observations; resolve duplicates before modeling')
    return df.sort_values(['sensor','event']).reset_index(drop=True)


def aggregate(df: pd.DataFrame, width: int = 500) -> pd.DataFrame:
    z = df.assign(bin=np.floor(df.event/width).astype(int))
    b = z.groupby('bin',as_index=False).agg(event=('event','median'),value=('capacitance_uf','median'),
            available=('available_event','max'),last_event=('event','max'),n=('event','size'))
    # A bin is not usable until its event range has closed, including its last label.
    b['available'] = np.maximum(b.available,(b.bin+1)*width)
    return b.sort_values('event').reset_index(drop=True)


def match_auxiliary(cases, calculated_bins, tolerance=250):
    """Auxiliary values are training targets, never forecast inputs."""
    z = cases.copy()
    z['aux_actual'] = np.nan
    z['aux_available'] = np.nan
    if calculated_bins is not None:
        for i,row in z.iterrows():
            distance = abs(calculated_bins.event-row.target_event)
            j = distance.idxmin()
            if distance.loc[j] <= tolerance:
                a = calculated_bins.loc[j]
                z.loc[i,['aux_actual','aux_available']] = [a.value,a.available]
    # Training a paired observation waits for both independent sensor labels.
    z['label_available'] = np.maximum(z.target_available,z.aux_available.fillna(z.target_available))
    return z
