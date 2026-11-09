"""Paired duration diagnostic of the chosen frozen classifier; no fitting."""
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from frozen_classifier import ROOT, BASE, CFG, FrozenClassifier, metrics, save, sha
from todd_experiment_data import featurize
from data_protocol import features
sys.path.insert(0, str(BASE/'data'))
from prepare_platoon_data import read_platoon

OUT = ROOT / CFG['output']


def offset(total, size, placement):
    return {'prefix': 0, 'center': (total-size)//2, 'suffix': total-size}[placement]


def restore_test(model):
    folder = model.path.parent
    split = json.loads((folder/'splits.json').read_text())
    training = set().union(*(set(split[k]['sources']) for k in ['train', 'external', 'aux']))
    held = set(split['test']['sources']); assert not held & training
    assert not held & set(split['val']['sources'])
    meta = pd.read_csv(folder/'test_predictions.csv')
    assert set(meta.source_id) == held
    source_records = json.loads((BASE/'data_records/records/raw_sources.json').read_text())
    archived_hashes = {Path(r['file']).name: r['sha256'] for r in source_records}
    # Match the archived primitive_windows dtype used by training/transformer/run.py.
    raw = np.zeros((len(meta), 40, 3), np.float32); provenance = {}
    for source, group in meta.groupby('source_id'):
        path = BASE/'data/restored_20261102/openacc'/f'{source.split("__", 1)[1]}.csv'
        assert sha(path) == archived_hashes[path.name], 'Raw source differs from archived payload'
        _, df = read_platoon(path); provenance[str(path.relative_to(ROOT))] = sha(path)
        for i, row in group.iterrows():
            leader, follower = map(int, re.search(r'__p(\d+)-(\d+)$', row.pair_id).groups())
            part = df.iloc[int(row.raw_start_row):int(row.raw_end_row)+1]
            assert len(part) == 40
            assert np.isclose(part.Time.iloc[0], row.raw_start_time, rtol=0, atol=1e-5)
            assert np.isclose(part.Time.iloc[-1], row.raw_end_time, rtol=0, atol=1e-5)
            assert np.allclose(np.diff(part.Time), .1, atol=1e-5)
            labels = part[f'Driver{follower}'].astype(str).str.strip().map({'ACC': 1, 'Human': 0})
            assert labels.eq(row.label).all()
            raw[i] = part[[f'Speed{follower}', f'Speed{leader}', f'IVS{leader}']].to_numpy()
    x = features(raw); p = model.predict(x, np.ones_like(x, bool), np.ones(x.shape[:2], bool), clipped=True)
    expected = 1/(1+np.exp(-meta.logit.to_numpy()))
    difference = float(abs(p-expected).max())
    # Archived predictions came from CUDA; this diagnostic runs PyTorch 2.6 CPU.
    # Keep both the observed drift and exact class-agreement check in the report.
    assert difference < 2e-4, f'Archived source probability drift exceeds tolerance: {difference}'
    assert np.array_equal(p >= .5, expected >= .5), 'Archived class decisions changed'
    # Select once using only pair identity and timestamps, before varying duration.
    selected = []
    for _, group in meta.groupby('pair_id', sort=True):
        last = -np.inf
        for i, row in group.sort_values('raw_start_time').iterrows():
            if row.raw_start_time > last + 1e-6:
                selected.append(i); last = row.raw_end_time
    meta = meta.loc[selected].reset_index(drop=True); raw = raw[selected]
    meta.to_csv(OUT/'public_cohort.csv', index=False)
    np.savez_compressed(OUT/'public_raw.npz', primitive_windows=raw, labels=meta.label.to_numpy())
    return meta, raw, dict(raw_sources=provenance, source_disjoint=True, archived_predictions_max_error=difference,
                          archived_probability_tolerance=2e-4, archived_class_decisions_identical=True,
                          original_raw_payload_hashes_verified=True,
                          archived_windows=len(expected), disjoint_windows=len(meta), pairs=int(meta.pair_id.nunique()),
                          sources=sorted(held), metadata_sha256=sha(folder/'test_predictions.csv'),
                          splits_sha256=sha(folder/'splits.json'))


def summarize(rows):
    summaries=[]
    for (cohort, placement, size), g in rows.groupby(['cohort','placement','samples'], sort=False):
        r=dict(cohort=cohort, placement=placement, samples=int(size), duration_s=(int(size)-1)/10,
               **metrics(g.label, g.probability))
        if cohort == 'public':
            pair=g.groupby(['source_id','pair_id']).correct.mean()
            r['pair_source_accuracy']=float(pair.groupby(level=0).mean().mean())
            r['pairs']=int(g.pair_id.nunique()); r['sources']=int(g.source_id.nunique())
        summaries.append(r)
    return pd.DataFrame(summaries)


def run():
    torch.set_num_threads(2); OUT.mkdir(parents=True, exist_ok=True)
    save(OUT/'protocol.json', CFG)  # Frozen before calculating new sensitivity outcomes.
    model=FrozenClassifier(); checks={}; details=[]
    field=np.load(ROOT/CFG['default_dataset']); fm=pd.read_csv(ROOT/CFG['default_metadata'])
    default=model.predict(field['features'],field['feature_mask'],field['valid_tokens'])
    archived=pd.read_csv(ROOT/'experiments/field_todd_variable_20261103/diagnostics/model_input_crossover/predictions.csv')
    archived=archived[(archived.model=='old_source')&(archived.inputs=='new_windows')].set_index('clip').loc[fm['clip']]
    assert np.allclose(default, archived.probability, atol=1e-6); checks['field_diagnostic_reproduced']=True
    changed=field['features'].copy(); changed[~field['feature_mask']]=np.nan
    assert np.array_equal(default,model.predict(changed,field['feature_mask'],field['valid_tokens']))
    checks['masked_values_do_not_change_prediction']=True
    for i in range(len(fm)):
        n=int(field['valid_tokens'][i].sum())
        p=model.predict(field['features'][i:i+1,:n],field['feature_mask'][i:i+1,:n],np.ones((1,n),bool))
        assert abs(p[0]-default[i])<1e-6
    checks['padding_and_actual_length_equivalent']=True
    meta,raw,restored=restore_test(model); checks.update(restored)
    for placement in CFG['sensitivity']['placements']:
        for size in CFG['sensitivity']['public_samples']:
            start=offset(40,size,placement); crop=raw[:,start:start+size]
            x=features(crop)  # Derivatives have no access to omitted samples.
            p=model.predict(x,np.ones_like(x,bool),np.ones(x.shape[:2],bool))
            for i,row in meta.iterrows():
                details.append(dict(cohort='public',placement=placement,samples=size,case=str(row.window_index),
                    source_id=row.source_id,pair_id=row.pair_id,label=int(row.label),probability=float(p[i]),
                    correct=bool((p[i]>=.5)==row.label)))
    ids=np.flatnonzero(field['valid_tokens'].sum(1)>=max(CFG['sensitivity']['field_samples']))
    for placement in CFG['sensitivity']['placements']:
        for size in CFG['sensitivity']['field_samples']:
            xx=[];mm=[]
            for i in ids:
                n=int(field['valid_tokens'][i].sum()); start=offset(n,size,placement)
                raw=field['primitive_windows'][i,start:start+size]; a=field['leader_available'][i,start:start+size]
                x,m,_=featurize(raw[:,0],raw[:,1],raw[:,2],a,np.zeros(size,int),np.where(a,0,-1))
                xx.append(x);mm.append(m)
                if size==n: assert np.allclose(x,field['features'][i,:n],atol=1e-5)
            x=np.stack(xx); p=model.predict(x,np.stack(mm),np.ones(x.shape[:2],bool))
            for j,i in enumerate(ids):
                row=fm.iloc[i]
                details.append(dict(cohort='field',placement=placement,samples=size,case=row['clip'],
                    source_id='Todd_SE',pair_id=row['clip'],label=int(row.label),probability=float(p[j]),
                    correct=bool((p[j]>=.5)==row.label)))
    table=pd.DataFrame(details); summary=summarize(table)
    table.to_csv(OUT/'duration_predictions.csv',index=False); summary.to_csv(OUT/'duration_summary.csv',index=False)
    checks['source_checkpoint_unchanged']=sha(model.path)==CFG['checkpoint_sha256']; assert checks['source_checkpoint_unchanged']
    checks['field_dataset_unchanged']=sha(ROOT/CFG['default_dataset'])
    checks['trained_models']=0; checks['field_common_videos']=fm.iloc[ids]['clip'].tolist()
    checks['status']='PASS'; save(OUT/'validation.json',checks)
    default_metrics=metrics(fm.label,default); keep=fm['clip'].ne('HV_T_EW_2')
    report=dict(default_metrics=default_metrics,excluding_HV_T_EW_2=metrics(fm.label[keep],default[keep]),
                public_restoration=restored,field_common_videos=len(ids),
                sensitivity=summary.replace({np.nan:None}).to_dict('records'),protocol=CFG)
    save(OUT/'summary.json',report)
    write_report(summary,default_metrics,report['excluding_HV_T_EW_2'],checks)
    draw(summary)
    print(summary[['cohort','placement','duration_s','n','correct','accuracy','pair_source_accuracy']].to_string(index=False),flush=True)


def write_report(summary,main,subset,checks):
    primary=summary[summary.placement=='center']; lines=[
        '冻结源分类器默认方案与输入时长诊断（2026-11-03）', '',
        '默认模型保留原128维Transformer权重、源统计量和0.5阈值；不使用Todd标签训练或校准分类器。',
        '此前已执行的Todd分类微调保留为对照，不能声称从未做过微调。上游轨迹重建包含现场适配及人工审核。',
        '这是同类标签的无目标标签微调迁移，不是未见类别识别，也不是整个pipeline完全零现场监督。',
        '默认是在查看Todd开发结果后选定，以下13段并非模型选择之后新增的独立测试集。', '',
        f"当前13段默认结果：{main['correct']}/{main['n']} = {main['accuracy']:.2%}。",
        f"去掉HV_T_EW_2的描述性结果：{subset['correct']}/{subset['n']} = {subset['accuracy']:.2%}；默认仍保留全部13段。", '',
        '时长对照：固定模型、阈值、标准化、样本集合；先裁原始速度/间距，再在窗口内平滑和求导。',
        '主结果使用居中截取；前段/后段截取全部另行报告，不挑最好位置。',
        '公开测试来源是原checkpoint留出的两份Vicolungo part2；按原完整窗口的固定物理门槛选样本，之后不随时长筛选。',
        f"由原{checks['archived_windows']}个重叠窗口中按时间取{checks['disjoint_windows']}个不重叠窗口，来自{checks['pairs']}个记录内车对。",
        '公开数据主指标在车对内平均、来源内平均、再平均两个来源；窗口准确率另列。车辆仍可能跨记录重复。',
        '原名义4秒模型有40点，首末时间跨度3.9秒；2秒和2.5秒分别21点和26点，保持0.1秒时间步长。',
        '路测时长对照只使用有完整2.5秒的同一批11段；其余2段只从这项配对比较排除，不从默认评价排除。', '',
        primary[['cohort','duration_s','n','correct','accuracy','pair_source_accuracy']].to_string(index=False), '',
        '本实验只检验固定长窗口训练模型在缩短输入后的表现，不能证明4秒训练优于2秒训练。',
        '公开数据与路测数据的来源、速度分布、测量质量及前车可用性不同，两个队列的准确率不能直接解释为时长效果。',
        '公开测试也使用当前部署的无裁剪标准化；旧裁剪方式仅用于复现归档概率，复现通过后才计算敏感度。', '',
        '文件：classifier_default.json（配置位于code/configs）；field/（默认预测）；duration_summary.csv；duration_predictions.csv；',
        'public_cohort.csv、public_raw.npz（配对公开样本）；validation.json；duration_sensitivity.png。',
        '复现：code/run.ps1 identify；code/run.ps1 python -Environment classifier scripts/evaluate_frozen_duration.py。']
    (OUT/'RESULTS.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def draw(summary):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),sharey=True)
    for ax,cohort,title in zip(axes,['public','field'],['Public held-out sources','Todd SE: matched 11 videos']):
        for place,style in [('center','-o'),('prefix','--s'),('suffix',':^')]:
            g=summary[(summary.cohort==cohort)&(summary.placement==place)].sort_values('duration_s')
            key='pair_source_accuracy' if cohort=='public' else 'accuracy'
            ax.plot(g.duration_s,100*g[key],style,label=place.capitalize(),linewidth=1.6)
        ax.set(title=title,xlabel='Observed endpoint span (s)',ylim=(0,100)); ax.grid(axis='y',alpha=.2)
        ax.set_xticks([2,2.5,3.9] if cohort=='public' else [2,2.5]); ax.legend(frameon=False)
    axes[0].set_ylabel('Accuracy (%)')
    fig.text(.5,.005,'Fixed source model; no target classifier fitting. Curves use the same cases at every duration.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.045,1,1)); fig.savefig(OUT/'duration_sensitivity.png',dpi=180); plt.close(fig)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');run()
