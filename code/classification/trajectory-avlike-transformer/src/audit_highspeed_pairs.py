"""Verify the paired treatments share initial weights, splits and preprocessing."""
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    base=Path('artifacts/experiments')
    main=base/'20261010_highspeed_v1'; balanced=base/'20261010_highspeed_balanced'
    count=0
    for seed in [42,52,62]:
        for fold in ['part2','part3','part4_highway']:
            reference=main/'core'/f'seed_{seed}'/fold
            for root,variants in [(main,['joint','ssl_core','ssl_external']),(balanced,['core','joint'])]:
                for variant in variants:
                    path=root/variant/f'seed_{seed}'/fold
                    for name in ['feature_stats.json','splits.json']:
                        assert json.loads((reference/name).read_text())==json.loads((path/name).read_text())
                    for evaluation in ['test','external_test']:
                        a=pd.read_csv(reference/f'{evaluation}_predictions.csv')
                        b=pd.read_csv(path/f'{evaluation}_predictions.csv')
                        np.testing.assert_array_equal(a.window_index,b.window_index)
                        np.testing.assert_array_equal(a.initial_logit,b.initial_logit)
                    count+=1
    micro_counts=[]
    for fold in ['part2','part3','part4_highway']:
        paths=[base/f'20261010_native5_{v}_micro'/'ssl_external'/'seed_42'/fold for v in ['without','with']]
        a,b=[json.loads((p/'splits.json').read_text()) for p in paths]
        for key in a:
            if key!='external_ssl':assert a[key]==b[key]
        added=set(b['external_ssl']['window_indices'])-set(a['external_ssl']['window_indices'])
        assert added
        m=pd.read_csv('data/prepared_highspeed_native5_20261010/windows.csv')
        assert m.iloc[sorted(added)].campaign.eq('MicroSim').all()
        for evaluation in ['test','external_test']:
            x,y=[pd.read_csv(p/f'{evaluation}_predictions.csv') for p in paths]
            np.testing.assert_array_equal(x.initial_logit,y.initial_logit)
        micro_counts.append(len(added))
    report=dict(status='passed',neural_pairs_checked=count,microsim_training_windows_by_fold=micro_counts,
        scope='Initial pre-SSL logits, feature statistics and split identity; no claims of identical early-stopped epoch counts.')
    (main/'paired_verification.json').write_text(json.dumps(report,indent=2)); print(report)


if __name__=='__main__':main()
