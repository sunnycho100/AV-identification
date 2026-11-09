import unittest
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from data_protocol import splits, features, fit_stats, normalize
from native_benchmark import NativeLSTM, WeightedKNN, training_pool


class NativeBenchmarkTests(unittest.TestCase):
    def test_real_pool_weights_and_source_exclusion(self):
        path=Path(json.loads(Path('experiment.json').read_text())['prepared'])
        m=pd.read_csv(path/'windows.csv')
        for fold in range(3):
            s=splits(m,fold,40)
            for mode in ['core','expanded']:
                ids,w,mass=training_pool(m,s,dict(data_mode=mode,external_mass=.5))
                np.testing.assert_allclose(w.mean(),1)
                for label in [0,1]:
                    np.testing.assert_allclose(w[m.iloc[ids].label.eq(label)].sum()/w.sum(),.5,atol=1e-12)
                self.assertFalse(m.iloc[ids].campaign.eq('NGSIM').any())
                for role in ['val','test','external_test']:
                    self.assertFalse(set(ids)&set(s[role]))
                if mode=='expanded':
                    np.testing.assert_allclose(mass[:len(s['train'])].sum(),.5,atol=2e-7)

    def test_weighted_knn_balancing_and_scale_invariance(self):
        x=np.array([[0.],[1.],[2.],[3.]])
        y=np.array([0,0,0,1]);w=np.array([1.,1.,1.,3.])
        for distance in [False,True]:
            a=WeightedKNN(4,distance).fit(x,y,w).predict_proba(np.array([[1.5]]))
            b=WeightedKNN(4,distance).fit(x,y,w*10).predict_proba(np.array([[1.5]]))
            np.testing.assert_allclose(a,b)
            if not distance: np.testing.assert_allclose(a,[[.5,.5]])
        exact=WeightedKNN(4,True).fit(x,y,w).predict_proba(x[[3]])
        np.testing.assert_allclose(exact,[[0.,1.]])

    def test_lstm_accepts_all_eleven_channels(self):
        torch.manual_seed(42)
        model=NativeLSTM(11,16,2,.2)
        x=torch.randn(3,40,11,requires_grad=True)
        z=model(x)
        self.assertEqual(tuple(z.shape),(3,))
        z.sum().backward()
        self.assertTrue(torch.all(x.grad.abs().sum((0,1))>0))


if __name__=='__main__': unittest.main()
