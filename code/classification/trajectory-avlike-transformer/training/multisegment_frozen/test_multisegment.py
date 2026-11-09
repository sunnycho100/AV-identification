import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from data_protocol import splits
from networks import TrajectoryNet
from multisegment import make_bags,audit_bags,role_bags,train_weights,fused,pooled_predictions,SUBSETS,read_logits

class TestProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg=json.loads(Path('experiment.json').read_text())
        cls.m=pd.read_csv(Path(cls.cfg['prepared'])/'windows.csv',float_precision='round_trip')
        cls.bags=make_bags(cls.m)

    def test_real_bags_and_splits(self):
        audit_bags(self.m,self.bags)
        q=pd.DataFrame(self.bags)
        self.assertEqual(q[(q.campaign=='Vicolungo')&(q.regime=='gap4')].shape[0],107)
        self.assertEqual(q[(q.campaign=='Vicolungo')&(q.regime=='dense')].shape[0],107)
        self.assertEqual(q[(q.campaign=='Vicolungo')&(q.regime=='gap12')].shape[0],44)
        for fold in range(3):
            s=splits(self.m,fold); train=role_bags(self.bags,np.r_[s['train'],s['external'],s['aux']])
            w=train_weights(train); q=pd.DataFrame(train)
            self.assertAlmostEqual(w[q.label.eq(0)].sum(),.5)
            self.assertAlmostEqual(w[q.campaign.eq('Vicolungo')].sum(),.5)
            self.assertFalse(set(q.source_id)&set(self.m.iloc[np.r_[s['val'],s['test'],s['external_test']]].source_id))

    def test_summary_serialization(self):
        keys=('cnn','test','gap4','mean_probability',np.int64(8))
        values=[v.item() if isinstance(v,np.generic) else v for v in keys]
        json.dumps(dict(zip(['model','role','regime','fusion','k'],values)))

    def test_fusion_and_duplicate(self):
        z=np.arange(-4,4)[None,:].astype(float)
        np.testing.assert_allclose(fused(z,1,'mean_probability'),expit(z))
        np.testing.assert_allclose(fused(z,8,'mean_probability'),expit(z).mean(1,keepdims=True))
        np.testing.assert_allclose(fused(z[:,::-1],8,'mean_probability'),fused(z,8,'mean_probability'))
        for k in [1,2,4,8]:
            np.testing.assert_allclose(fused(np.ones((1,8))*.7,k,'mean_probability'),expit(.7))
        self.assertEqual([len(SUBSETS[k]) for k in SUBSETS],[8,28,70,1])
        self.assertEqual(float(fused(z,8,'majority')[0,0]),.5)

    def test_embedding_k1_permutation_and_duplicates(self):
        torch.manual_seed(10); model=TrajectoryNet(11,'multiscale_cnn',32,.2,1).eval()
        x=torch.randn(8,40,11)
        with torch.no_grad(): h=model.encode(x); expected=torch.sigmoid(model(x)).numpy()
        a=pooled_predictions(model,h,[dict(indices=list(range(8)))])
        np.testing.assert_allclose(a[1][0],expected,atol=1e-7)
        b=pooled_predictions(model,h,[dict(indices=list(reversed(range(8))))])
        np.testing.assert_allclose(a[8],b[8],atol=1e-7)
        d=pooled_predictions(model,h,[dict(indices=[0]*8)])
        np.testing.assert_allclose(d[8],d[1][:,:1],atol=1e-7)

    def test_original_prediction_provenance(self):
        for name in ['cnn','transformer','rf']:
            z=read_logits(self.cfg,name,42,0,'test',self.m,{})
            self.assertTrue(np.isfinite(z[splits(self.m,0)['test']]).all())

if __name__=='__main__': unittest.main()
