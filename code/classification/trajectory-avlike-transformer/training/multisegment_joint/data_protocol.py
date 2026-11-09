"""Window-local processing and immutable source splits; NGSIM is never selected."""
import numpy as np
from scipy.signal import savgol_filter

PARTS = ('part2', 'part3', 'part4_highway')


def splits(m, fold, stride=40):
    out = {}
    for role, offset in [('test', 0), ('val', 1), ('train', 2)]:
        mask = m.campaign.eq('Vicolungo') & m.source_id.str.endswith('_'+PARTS[(fold+offset)%3])
        if role == 'train':
            mask &= m.start_in_segment.mod(stride).eq(0)
        out[role] = np.flatnonzero(mask)
    held = m.campaign.eq('CATS') & m.source_id.isin(['CATS__test1124_test9', 'CATS__test1124_test10'])
    out['external_test'] = np.flatnonzero(held)
    out['external'] = np.flatnonzero(m.campaign.isin(['CATS', 'Vanderbilt']) & ~held
                                    & m.start_in_segment.mod(stride).eq(0))
    out['aux'] = np.flatnonzero(m.campaign.isin(['ASta', 'Casale'])
                              & m.start_in_segment.mod(stride).eq(0))
    for ids in out.values():
        assert not m.iloc[ids].campaign.eq('NGSIM').any()
    for train in ['train', 'external', 'aux']:
        for test in ['test', 'val', 'external_test']:
            assert not set(m.iloc[out[train]].source_id) & set(m.iloc[out[test]].source_id)
    return out


def features(raw, mode='legacy11', smoothing=7):
    """raw [N,T,3]: follower speed, leader speed, net gap; no outside frames."""
    a = np.asarray(raw, dtype=np.float64)
    size = min(smoothing, a.shape[1] if a.shape[1] % 2 else a.shape[1]-1)
    sm = savgol_filter(a, size, 2, axis=1, mode='interp')
    vf, vl, gap = sm.transpose(2, 0, 1)
    dv = vl-vf
    thw = np.clip(gap/np.maximum(vf, 1e-6), 0, 10)
    if mode == 'primitives3':
        return sm.astype(np.float32)
    if mode == 'stable5':
        return np.stack([vf, vl, gap, dv, thw], -1).astype(np.float32)
    if mode == 'legacy11':
        af, al = np.gradient(vf, .1, axis=1, edge_order=2), np.gradient(vl, .1, axis=1, edge_order=2)
        jerk = np.gradient(af, .1, axis=1, edge_order=2)
        closing = -dv
        ttc = np.full_like(gap, 30.)
        np.divide(gap, closing, out=ttc, where=closing > 1e-6)
        return np.stack([vf, af, jerk, vl, al, gap, dv, closing, al-af, thw,
                         np.clip(ttc, 0, 30)], -1).astype(np.float32)
    # Derivatives of local polynomials, avoiding two successive noisy differences.
    velocity_deriv = savgol_filter(a[:, :, :2], size, 2, deriv=1, delta=.1, axis=1, mode='interp')
    af, al = velocity_deriv.transpose(2, 0, 1)
    if mode == 'kinematic7':
        out = [vf, vl, gap, dv, af, al, thw]
    elif mode == 'ego3':
        jerk = savgol_filter(a[:, :, 0], size, 2, deriv=2, delta=.1, axis=1, mode='interp')
        out = [vf, af, jerk]
    elif mode == 'centered6':
        out = [vf-vf.mean(1, keepdims=True), vl-vl.mean(1, keepdims=True),
               gap-gap.mean(1, keepdims=True), dv, af, al]
    else:
        raise ValueError(mode)
    return np.stack(out, -1).astype(np.float32)


def fit_stats(x):
    a = x.reshape(-1, x.shape[-1]).astype(np.float64)
    lo, hi = np.quantile(a, [.005, .995], axis=0)
    a = np.clip(a, lo, hi)
    return dict(lower=lo.tolist(), upper=hi.tolist(), mean=a.mean(0).tolist(),
                std=np.maximum(a.std(0), 1e-6).tolist())


def normalize(x, stats):
    return ((np.clip(x, stats['lower'], stats['upper'])-stats['mean'])/stats['std']).astype(np.float32)


def joint_weights(m, core, external, mass=.5):
    """Preserve core class prior, balance external campaigns and recordings."""
    w = np.zeros(len(core)+len(external))
    w[:len(core)] = (1-mass)/len(core)
    extra = m.iloc[external]
    for label in (0, 1):
        cmass = (m.iloc[core].label.to_numpy() == label).mean()*mass
        campaigns = extra[extra.label == label].campaign.unique()
        for campaign in campaigns:
            sources = extra[(extra.label == label) & (extra.campaign == campaign)].source_id.unique()
            for source in sources:
                ids = np.flatnonzero((extra.label.to_numpy() == label) & (extra.source_id.to_numpy() == source))
                w[len(core)+ids] = cmass/len(campaigns)/len(sources)/len(ids)
    assert np.isclose(w.sum(), 1)
    return w.astype(np.float32)


def source_weights(m, ids):
    rows = m.iloc[ids]
    out = np.zeros(len(ids))
    for campaign in rows.campaign.unique():
        sources = rows[rows.campaign == campaign].source_id.unique()
        for source in sources:
            mask = rows.source_id.eq(source).to_numpy()
            out[mask] = 1/rows.campaign.nunique()/len(sources)/mask.sum()
    assert np.isclose(out.sum(), 1)
    return out.astype(np.float32)


def forecast_examples(raw, cfg, stats):
    """Only the 3-second prefix enters the encoder; future only enters the target."""
    prefix = raw[:, :30]
    x = normalize(features(prefix, cfg['features'], cfg['smoothing']), stats)
    horizon = (np.arange(10)+1)*.1
    baseline = np.repeat(prefix[:, -1:, :], 10, axis=1).copy()
    baseline[:, :, 2] += (prefix[:, -1, 1]-prefix[:, -1, 0])[:, None]*horizon
    target = raw[:, 30:]-baseline
    return x, target.astype(np.float32)
