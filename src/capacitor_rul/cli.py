import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .data import load_measurements


def make_demo(out):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(42)
    event=np.arange(0,80500,500)
    # Synthetic example only. Two measurement methods deliberately disagree.
    health=1-.0000018*event-.06/(1+np.exp(-(event-54000)/1200))
    rows=[]
    for sensor,c0 in [('LCR',600.),('calculated',950.)]:
        values=c0*health+rng.normal(0,.4 if sensor=='LCR' else 1.,len(event))
        if sensor=='calculated':values-=.0002*event
        for e,v in zip(event,values):
            rows.append(dict(sensor=sensor,event=int(e),available_event=int(e+100),capacitance_uf=float(v)))
    pd.DataFrame(rows).to_csv(out/'measurements.csv',index=False)
    config=dict(cutoffs=[35000,50000,65000],horizons=[1000,5000],test_block=5000,
                reference_capacitance_uf={'LCR':600.,'calculated':950.},eol_fraction=.8,
                models=['plain','stage'],seeds=[42],max_epochs=4,mc_samples=20,rul_draws=500)
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    print(f'Synthetic demonstration written to {out}. These are not research measurements.')


def main():
    ap=argparse.ArgumentParser(description='Capacitor RUL: separate sensor forecasts, honest benchmarks.')
    sub=ap.add_subparsers(dest='command',required=True)
    demo=sub.add_parser('demo',help='Generate synthetic example data and a small training config')
    demo.add_argument('--out',type=Path,required=True)
    check=sub.add_parser('validate',help='Validate a prepared measurement CSV')
    check.add_argument('csv',type=Path)
    run=sub.add_parser('run',help='Run chronological forecast and RUL evaluation')
    run.add_argument('csv',type=Path);run.add_argument('--config',type=Path,required=True)
    run.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    try:
        if args.command=='demo':make_demo(args.out)
        elif args.command=='validate':
            df=load_measurements(args.csv)
            print(df.groupby('sensor').agg(observations=('event','size'),first_event=('event','min'),last_event=('event','max')).to_string())
        else:
            from .pipeline import run as execute
            execute(args.csv,args.config,args.out)
    except (ValueError,FileNotFoundError,FileExistsError) as e:
        ap.exit(2,f'Error: {e}\n')
