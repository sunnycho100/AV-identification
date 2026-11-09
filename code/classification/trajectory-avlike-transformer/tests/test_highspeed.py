import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'data'),str(ROOT/'src')]
from prepare_highspeed_data import compute_features, continuous_runs, THRESHOLD
from highspeed_experiment import make_splits, joint_weights, enforce_training_data_policy
from analyze_highspeed import expected_splits_for_run
from prepare_native5_data import local_interpolate


class HighspeedTests(unittest.TestCase):
    def test_si_units_and_derivatives_for_constant_acceleration(self):
        t=np.arange(40)*.1
        # Equal accelerations retain a constant relative speed; gap closes at 1m/s.
        raw=np.stack([25+.5*t,24+.5*t,30-t],axis=-1)[None]
        out=compute_features(raw)
        np.testing.assert_allclose(out[0,:,1],.5,atol=1e-5)
        np.testing.assert_allclose(out[0,:,2],0,atol=1e-5)
        np.testing.assert_allclose(out[0,:,7],1,atol=1e-5)
        np.testing.assert_allclose(out[0,:,10],30-t,atol=1e-5)
        self.assertAlmostEqual(THRESHOLD,22.352)

    def test_windows_cannot_cross_missing_time_mode_or_speed_breaks(self):
        t=np.arange(130)*.1; t[90:]+=.1
        labels=np.ones(130,dtype=int); labels[50:]=0
        valid=np.ones(130,dtype=bool); valid[70]=False
        runs=continuous_runs(t,labels,valid)
        self.assertEqual([len(x) for x in runs],[50,20,19,40])
        self.assertTrue(all(np.allclose(np.diff(t[x]),.1) for x in runs))

    def test_batch_processing_does_not_use_other_windows(self):
        rng=np.random.default_rng(4)
        raw=rng.normal(size=(3,40,3))
        before=compute_features(raw)[0]
        raw[1:]=1e5
        np.testing.assert_array_equal(before,compute_features(raw)[0])

    def manifest(self):
        rows=[]
        for part in ['part2','part3','part4_highway']:
            for y in [0,1]:
                rows.append(dict(campaign='Vicolungo',source_id=f'Vicolungo_{y}_{part}',label=y,start_in_segment=0))
        for file in ['test6','test9','test10']:
            for y in [0,1]:
                rows.append(dict(campaign='CATS',source_id='CATS__test1124_'+file,label=y,start_in_segment=0))
        rows.append(dict(campaign='Vanderbilt',source_id='Vanderbilt_2019',label=1,start_in_segment=0))
        rows.append(dict(campaign='NGSIM',source_id='NGSIM_i80',label=0,start_in_segment=0))
        return pd.DataFrame(rows)

    def test_external_holdout_never_enters_supervision_or_pretraining(self):
        m=self.manifest()
        split=make_splits(m,0)
        held=set(m.iloc[split['external_test']].source_id)
        for role in ['train','external_train','external_ssl']:
            self.assertFalse(held & set(m.iloc[split[role]].source_id))
        for indices in split.values():
            self.assertFalse(m.iloc[indices].campaign.eq('NGSIM').any())

    def test_legacy_audit_keeps_original_ngsim_and_microsim_policy(self):
        m=pd.concat([self.manifest(),pd.DataFrame([dict(campaign='MicroSim',
            source_id='MicroSim_session',label=1,start_in_segment=0)])],ignore_index=True)
        for micro in [False,True]:
            cfg=dict(train_stride=40,include_microsim=micro)
            historical=expected_splits_for_run(m,0,cfg)
            self.assertIn('NGSIM',set(m.iloc[historical['external_ssl']].campaign))
            current=expected_splits_for_run(m,0,dict(cfg,include_ngsim=False))
            campaigns=set(m.iloc[current['external_ssl']].campaign)
            self.assertNotIn('NGSIM',campaigns)
            self.assertEqual('MicroSim' in campaigns,micro)
            for key in historical:
                if key!='external_ssl':
                    np.testing.assert_array_equal(historical[key],current[key])

    def test_new_fits_record_exclusion_and_reject_ngsim_enable(self):
        cfg={}
        enforce_training_data_policy(cfg)
        self.assertIs(cfg['include_ngsim'],False)
        with self.assertRaises(ValueError):
            enforce_training_data_policy(dict(include_ngsim=True))

    def test_external_sampling_preserves_core_class_prior(self):
        m=self.manifest(); split=make_splits(m,1)
        weights=joint_weights(m,split['train'],split['external_train'])
        labels=m.iloc[np.r_[split['train'],split['external_train']]].label.to_numpy()
        self.assertAlmostEqual(weights.sum(),1)
        self.assertAlmostEqual(weights[labels==1].sum(),m.iloc[split['train']].label.mean())
        self.assertAlmostEqual(weights[:len(split['train'])].sum(),.5)

    def test_interpolation_preserves_si_velocity_and_reports_all_donors(self):
        times=np.arange(24)*.2
        values=np.column_stack([np.full(24,90/3.6),times*25])
        grid=.1+np.arange(20)*.2
        result=local_interpolate(times,values,grid)
        np.testing.assert_allclose(result[0][:,0],25)
        np.testing.assert_allclose(result[0][:,1],grid*25)
        self.assertAlmostEqual(result[1],0)
        self.assertAlmostEqual(result[2],4)
        self.assertEqual((result[3],result[4]),(0,20))

    def test_interpolation_rejects_missing_time_and_extrapolation(self):
        times=np.r_[np.arange(10)*.2,np.arange(10,24)*.2+.5]
        values=np.zeros((len(times),2))
        self.assertIsNone(local_interpolate(times,values,np.arange(20)*.2))
        self.assertIsNone(local_interpolate(times,values,np.arange(20)*.2-.1))


if __name__=='__main__':
    unittest.main()
