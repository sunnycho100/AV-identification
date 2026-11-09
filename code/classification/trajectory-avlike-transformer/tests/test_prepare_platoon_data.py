"""Label and observation-budget regressions; python3 -m unittest discover -s tests."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('prepare_platoon_data',Path(__file__).resolve().parents[1]/'data'/'prepare_platoon_data.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class PreparationTests(unittest.TestCase):
    def frame(self,n=200):
        t=np.arange(n)*.1
        return pd.DataFrame(dict(Time=t,Speed1=10.,Speed2=10.,E1=t*10+20,N1=0.,E2=t*10,N2=0.,Driver1='Human',Driver2='ACC'))

    def test_label_follows_follower_not_leader_and_switch_splits(self):
        f=self.frame()
        f.loc[100:,'Driver2']='Human'
        groups,_,_,labels,audit=m.pair_segments(f,1,2)
        self.assertEqual(len(groups),2)
        self.assertEqual([labels[g[0]] for g in groups],[1,0])
        self.assertEqual(audit['known_human_rows'],100)
        for g in groups:
            self.assertEqual(len(set(labels[g])),1)

    def test_unknown_and_time_gap_never_bridged(self):
        f=self.frame()
        f.loc[80,'Driver2']='Unknown'
        f.loc[150:,'Time']+=1
        groups,*_=m.pair_segments(f,1,2)
        for g in groups:
            self.assertNotIn(80,g)
            self.assertTrue(np.all(np.diff(g)==1))
            self.assertTrue(np.allclose(np.diff(f.Time.iloc[g]),.1))
        self.assertGreaterEqual(len(groups),2)

    def test_missing_follower_label_is_not_inferred_from_leader(self):
        f=self.frame().drop(columns='Driver2')
        groups,_,_,_,audit=m.pair_segments(f,1,2)
        self.assertEqual(groups,[])
        self.assertEqual(audit['unknown_label_rows'],len(f))

    def test_four_second_features_ignore_external_frames(self):
        f=self.frame()
        xy=f[['E1','N1','E2','N2']].to_numpy()
        time=f.Time.to_numpy()
        x=m.window_features(xy[40:80][None],time[40:80][None])
        xy[:40]=1e9
        xy[80:]=-1e9
        np.testing.assert_array_equal(x,m.window_features(xy[40:80][None],time[40:80][None]))
        self.assertEqual(x.shape,(1,40,12))
        np.testing.assert_allclose(x[0,:,0],10,atol=1e-5)
        np.testing.assert_allclose(x[0,:,5],20,atol=1e-5)
        np.testing.assert_allclose(x[0,:,1:3],0,atol=1e-4)

    def test_casale_without_altitude_still_converts(self):
        groups,_,_,_,_=m.pair_segments(self.frame(),1,2)
        self.assertEqual(len(groups),1)


if __name__=='__main__':
    unittest.main()
