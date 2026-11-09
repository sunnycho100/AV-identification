"""Rebuild original labeled sources as exact 3.0-second, all-speed windows."""
import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
BASE=ROOT/'classification/trajectory-avlike-transformer'
OUT=BASE/'data/prepared_3s_20261102'
RAW=BASE/'data/restored_20261102'
sys.path.insert(0,str(BASE/'data'))
import prepare_highspeed_data as P


class Builder(P.Builder):
    def finish(self):
        meta=pd.DataFrame(self.rows);meta['observation_span']=3.0
        raw=np.concatenate(self.raw);assert raw.shape[1:]==(31,3)
        assert np.allclose(meta.raw_end_time-meta.raw_start_time,3.0)
        assert (raw[:,:,:2]>=0).all()
        meta.to_csv(self.output/'windows.csv',index=False)
        np.savez_compressed(self.output/'windows_3s.npz',primitive_windows=raw,y=meta.label.to_numpy(np.int64),dt=.1)
        pd.DataFrame(self.audit).to_csv(self.output/'pair_audit.csv',index=False)
        P.json_write(self.output/'sources.json',self.sources)
        counts=meta.groupby(['campaign','label']).size().rename('windows').reset_index()
        counts.to_csv(self.output/'counts.csv',index=False)
        P.json_write(self.output/'protocol.json',dict(duration_s=3.0,samples=31,hz=10,stride_samples=31,
            minimum_raw_speed_mps=0,source_converter=str((BASE/'data/prepare_highspeed_data.py').relative_to(ROOT)),
            source_converter_sha256=hashlib.sha256((BASE/'data/prepare_highspeed_data.py').read_bytes()).hexdigest(),
            label_policy='Original per-frame follower driver or documented experimental role; no vehicle-brand inference',
            excluded_sources=['NGSIM'],physical_checks='Original gap/speed/acceleration/kinematic residual checks, except no 50mph floor',
            upstream_interpolation='Inherited from released OpenACC and Vanderbilt step2',
            source_manifest='sources.json',counts=counts.to_dict('records')))
        print(counts.to_string(index=False),flush=True)


def main():
    sys.stdout.reconfigure(encoding='utf-8');P.SEQ=31;P.STRIDE=31;P.THRESHOLD=-1e-6
    b=Builder(OUT);P.openacc(b,RAW/'openacc')
    P.cats(b,next((RAW/'external').glob('Filed-Experiment-Data-ACC_Data-*')))
    P.vanderbilt(b,RAW/'external/vanderbilt_step2.csv');b.finish()


if __name__=='__main__':main()
