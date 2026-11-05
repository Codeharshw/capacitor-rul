import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
import torch
from capacitor_rul.data import load_measurements,aggregate,match_auxiliary
from capacitor_rul._core import SensorData
from capacitor_rul.losses import gaussian_nll,dual_gaussian_nll,rc_penalty
from capacitor_rul.rul import summarize,crossing_bracket,bracket_distance
from capacitor_rul.cli import make_demo
from capacitor_rul.pipeline import run


class DataIntegrity(unittest.TestCase):
    def test_delayed_bin_is_not_available_early(self):
        raw=pd.DataFrame(dict(event=[10,100,600],available_event=[10,1200,600],capacitance_uf=[600,599,598]))
        b=aggregate(raw)
        self.assertEqual(b.available.tolist(),[1200,1000])

    def test_reject_duplicates_and_future_availability_reversal(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'input.csv'
            d=pd.DataFrame(dict(sensor=['LCR','LCR'],event=[100,100],available_event=[100,100],capacitance_uf=[600,599]))
            d.to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,'Duplicate'):load_measurements(p)
            d=d.iloc[:1].copy();d.available_event=99;d.to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,'before'):load_measurements(p)

    def test_future_sensor_values_cannot_change_origin_features(self):
        e=np.arange(0,50500,500)
        b=pd.DataFrame(dict(event=e,value=600-.001*e,available=e+500,last_event=e))
        a=SensorData(b).origin(30000)
        changed=b.copy();changed.loc[changed.available>30000,'value']+=99999
        c=SensorData(changed).origin(30000)
        self.assertIsNotNone(a)
        np.testing.assert_allclose(a['X'],c['X'])
        np.testing.assert_allclose(a['context'],c['context'])
        self.assertEqual(a['base'],c['base'])

    def test_auxiliary_pair_waits_for_both_targets(self):
        c=pd.DataFrame(dict(target_event=[1000,2000],target_available=[1500,2500]))
        b=pd.DataFrame(dict(event=[1000],value=[950],available=[3000]))
        a=match_auxiliary(c,b)
        self.assertEqual(a.label_available.tolist(),[3000,2500])


class Objectives(unittest.TestCase):
    def test_missing_auxiliary_nan_cannot_poison_gradient(self):
        mean=torch.tensor([1.,2.],requires_grad=True);lv=torch.zeros(2)
        y=torch.tensor([3.,4.]);calc=torch.tensor([float('nan'),8.])
        valid=torch.tensor([False,True])
        value=dual_gaussian_nll(mean,lv,y,calc,valid)
        self.assertTrue(torch.isfinite(value).all())
        self.assertEqual(float(value[0].detach()),float(gaussian_nll(mean,lv,y)[0].detach()))
        value.sum().backward();self.assertTrue(torch.isfinite(mean.grad).all())

    def test_dual_mean_favors_arithmetic_compromise(self):
        mu=torch.tensor([5.],requires_grad=True)
        loss=dual_gaussian_nll(mu,torch.zeros(1),torch.tensor([2.]),torch.tensor([8.]),torch.tensor([True]))
        loss.backward();self.assertAlmostEqual(float(mu.grad),0.)

    def test_rc_equivalent_to_independent_calculated_target(self):
        c=torch.tensor([500.,600.]);r=torch.tensor([220.,100.]);tau=torch.tensor([.12,.07])
        np.testing.assert_allclose(rc_penalty(c,tau,r,20.).numpy(),(.5*((c-tau/r*1e6)/20.)**2).numpy(),rtol=1e-5)


class RULIntegrity(unittest.TestCase):
    def test_non_degrading_rate_is_censored_not_zero(self):
        a=summarize([600,600],[0,1],480)
        self.assertIsNone(a['RUL_median']);self.assertTrue(a['median_right_censored'])

    def test_known_linear_lifetime_and_already_below(self):
        self.assertEqual(summarize([600],[-20],480)['RUL_median'],60000.)
        self.assertEqual(summarize([470],[-20],480)['RUL_median'],0.)
        self.assertEqual(bracket_distance(None,10000,15000),45000.)

    def test_crossing_bracket_is_not_an_exact_crossing(self):
        b=pd.DataFrame(dict(event=[0,100,500],value=[600,500,470]))
        self.assertEqual(crossing_bracket(b,480),(100.,500.))


class EndToEnd(unittest.TestCase):
    def test_all_objectives_and_manifest_on_synthetic_data(self):
        with tempfile.TemporaryDirectory() as t:
            base=Path(t);make_demo(base/'input')
            cfg=json.loads((base/'input/config.json').read_text())
            cfg.update(cutoffs=[35000],horizons=[1000],max_epochs=1,mc_samples=2,rul_draws=20,models=['plain','stage','rc','dual'])
            (base/'input/config.json').write_text(json.dumps(cfg))
            out=run(base/'input/measurements.csv',base/'input/config.json',base/'results')
            self.assertTrue((out/'COMPLETE.txt').is_file())
            p=pd.read_csv(out/'predictions.csv')
            self.assertTrue((p.input_available_max<=p.origin).all())
            self.assertEqual(set(p[p.sensor=='LCR'].method),{'persistence','recent_linear','piecewise','plain','stage','rc','dual'})
            audit=pd.read_csv(out/'split_audit.csv')
            self.assertTrue((audit.inner_label_max<=audit.validation_origin_min).all())
            self.assertTrue((audit.outer_label_max<=audit.test_origin_min).all())
            from capacitor_rul.pipeline import sha256
            for name,expected in json.loads((out/'manifest.json').read_text()).items():
                self.assertEqual(sha256(out/name),expected)


if __name__=='__main__':unittest.main()
