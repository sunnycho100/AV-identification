"""Contract tests for gap geometry, missing time, identity switches and inference."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np,pandas as pd
import trajectories_to_windows as adapter

class AdapterTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.old=adapter.ROOT;adapter.ROOT=self.root
        td=self.root/'trajectory/outputs/trajectories/AV_SYNTH_pilot_base';td.mkdir(parents=True)
        dd=self.root/'trajectory/outputs/object_detection/camera-data/AV_SYNTH_pilot_base_road';dd.mkdir(parents=True)
        self.td=td;tracks={'1':[],'2':[]};rows={'1':[],'2':[]}
        for f in range(41):
            t=f*.1;boxes=[]
            for tid,x,v in [('1',27*t,27),('2',24+28*t,28)]:
                tracks[tid].append(dict(frame=f,t_s=t,x=x,y=-5,z=-1.73,vx=v,vy=0,speed_mps=v,yaw=0,score=.9))
                rows[tid].append(dict(frame=f,t_s=t,lane=1,direction=1,detected=True,lead_id='2' if tid=='1' else '',speed_mps=v))
                boxes.append(dict(x=x,y=-5,z=-1.73,l=4,w=2,h=1.6,score=.9,class_name='car'))
            (dd/f'{f:03d}_pred.json').write_text(json.dumps(boxes))
        (td/'tracks.json').write_text(json.dumps(dict(tracks=tracks)))
        (td/'instrumented.json').write_text(json.dumps(dict(vehicle_id='1',warnings=[])))
        for tid,r in rows.items():pd.DataFrame(r).to_csv(td/f'vehicle_{tid}.csv',index=False)
    def tearDown(self):adapter.ROOT=self.old;self.temp.cleanup()
    def run_clip(self):return adapter.convert_clip('AV_SYNTH','pilot_base','test')
    def test_bumper_gap_and_target_label(self):
        raw,meta,_,report=self.run_clip();self.assertEqual(len(raw),1)
        np.testing.assert_allclose(raw[0][:,0],27);np.testing.assert_allclose(raw[0][:,1],28)
        np.testing.assert_allclose(raw[0][:,2],20+np.arange(40)*.1,atol=2e-6)
        self.assertEqual(meta[0]['label'],1);self.assertEqual(report['accepted_labeled_windows'],1)
    def test_no_label_for_unidentified_traffic(self):
        (self.td/'instrumented.json').unlink();raw,meta,_,_=self.run_clip()
        self.assertEqual(len(raw),1);self.assertEqual(meta[0]['label'],-1)
    def test_unknown_leader_switch_cannot_be_bridged(self):
        p=self.td/'vehicle_1.csv';df=pd.read_csv(p);df.loc[20,'lead_id']=999;df.to_csv(p,index=False)
        self.assertEqual(len(self.run_clip()[0]),0)
    def test_missing_observations_cannot_be_extrapolated(self):
        p=self.td/'vehicle_1.csv';df=pd.read_csv(p);df.loc[18:22,'detected']=False;df.to_csv(p,index=False)
        self.assertEqual(len(self.run_clip()[0]),0)
    def test_source_speed_gate(self):
        raw=np.tile([20.,20.,25.],(40,1));self.assertIn('speed_below_source_domain',adapter.protocol_reasons(raw))

if __name__=='__main__':unittest.main(verbosity=2)
