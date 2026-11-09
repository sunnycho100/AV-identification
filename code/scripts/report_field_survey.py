"""Summarize the completed all-recording survey without promoting proxies to truth."""
from collections import Counter
import numpy as np,pandas as pd
from field_full_survey import ROOT,TR,OUT,CONFIG,read,save,sha

def main():
    cfg=read(CONFIG);clips=cfg['clips'];rows=[read(p) for p in sorted((OUT/'runs').glob('*/evaluation.json'))];assert len(rows)==90
    inference=[read(p) for p in sorted((OUT/'runs').glob('*/inference.json'))];tracking=[read(p) for p in sorted((OUT/'runs').glob('*/tracking.json'))]
    assert len(inference)==len(tracking)==90 and all(r['status']=='COMPLETE' for r in inference+tracking)
    assert not any(r['status']=='EVALUATION_FAILED' for r in rows)
    prep=read(OUT/'preparation_summary.json');assert prep['complete'] and len(prep['rows'])==60 and all(r['status']=='READY' for r in prep['rows'])
    for m in cfg['models'].values():assert sha(ROOT/m['checkpoint'])==m['sha256']
    failures=[];paired=[];validation=[];stability=[]
    for r in rows:
        c,key=r['clip'],r['model'];tag=cfg['models'][key]['tag'];d=OUT/'runs'/f'{c}_{key}';dd=TR/'outputs/object_detection/camera-data'/f'{c}_{tag}';cal=read(dd/'calibration_used.json');n=read(d/'inference.json')['frames']
        assert len(list(dd.glob('*_pred.json')))==n;assert np.isfinite(cal['K']).all() and np.isfinite(cal['lidar2cam']).all()
        tt=read(TR/'data/camera-data'/c/'frame_times.json')['frames'];assert len(tt)==n and all(b['t_s']>=a['t_s'] for a,b in zip(tt,tt[1:]))
        if r['status']=='MARKER_RUN_UNRESOLVED':failure='source_marker_confirmed_but_template_sequence_failed'
        elif not r.get('identity_accepted'):
            if not r.get('marker_detection_frames',0):failure='no_target_car_box_under_projected_marker_gate'
            elif r.get('marker_detection_frames',0)>=.5*r.get('marker_frames',1):failure='detections_present_but_MOT_identity_not_accepted'
            else:failure='sparse_target_detection_or_fragmented_MOT'
        elif not r.get('following_observations',0):failure='target_exported_no_valid_leader_pair'
        elif not r.get('windows_by_duration',{}).get('2.0',0):failure='pair_present_but_no_continuous_2s_window'
        else:failure='candidate_2s_window_available'
        failures.append(dict(clip=c,model=key,core=r['core'],category=failure))
        if r.get('identity_accepted'):
            from visualize_field_baseline import midpoint_residual
            states=read(TR/'outputs/trajectories'/f'{c}_{tag}'/'tracks.json')['tracks'][str(r['identified_target'])];obs=[s for s in states if 'x_det' in s]
            if len(obs)>=3:
                t=np.array([s['t_s'] for s in obs]);raw=np.array([[s['x_det'],s['y_det']] for s in obs]);smooth=np.array([[s['x'],s['y']] for s in obs])
                with np.errstate(divide='ignore',invalid='ignore'):
                    a=midpoint_residual(t,raw);b=midpoint_residual(t,smooth)
                stability.append(dict(clip=c,model=key,observations=len(obs),raw_midpoint_residual_m=a,smoothed_midpoint_residual_m=b,median_position_adjustment_m=float(np.median(np.linalg.norm(raw-smooth,axis=1)))))
        p=OUT/'pairs'/f'{c}_{tag}'/'following_pairs.csv'
        if p.exists():
            df=pd.read_csv(p,dtype={'target_id':str,'leader_id':str});dims=read(p.parent/'dimensions.json')
            if len(df):
                expected=(df.leader_x_m-df.target_x_m)*df.direction-np.array([(dims[x.target_id]['length_m']+dims[x.leader_id]['length_m'])/2 for x in df.itertuples()]);assert np.isfinite(expected).all() and (expected>0).all();assert np.max(np.abs(expected-df.net_gap_m))<1e-6
        validation.append(dict(clip=c,model=key,frames=n,status='PASS'))
    pd.DataFrame(failures).to_csv(OUT/'failure_categories.csv',index=False,encoding='utf-8-sig')
    groups=[]
    for key in cfg['models']:
        rr=[r for r in rows if r['model']==key]
        for site,view in sorted(set((r['site'],r['view']) for r in rr)):
            g=[r for r in rr if r['site']==site and r['view']==view];gps=[r['gps_speed_mae_mps'] for r in g if r.get('gps_speed_mae_mps') is not None]
            groups.append(dict(model=key,site=site,view=view,recordings=len(g),core=g[0]['core'],identity_accepted=sum(bool(r.get('identity_accepted')) for r in g),target_exported=sum(r.get('target_observations',0)>0 for r in g),pair_available=sum(r.get('following_observations',0)>0 for r in g),**{f'window_{dur}s_recordings':sum(r.get('windows_by_duration',{}).get(str(float(dur)),0)>0 for r in g) for dur in [1,2,3,4]},gps_recordings=len(gps),median_recording_speed_mae_mps=float(np.median(gps)) if gps else None))
    pd.DataFrame(groups).to_csv(OUT/'view_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(stability).to_csv(OUT/'stability_summary.csv',index=False,encoding='utf-8-sig')
    save(OUT/'stability_summary.json',dict(rows=stability,note='Median displacement from the time-weighted midpoint of adjacent detection-supported positions, consecutive gaps <=0.2 s. Within-track smoothness diagnostic on the same observed frames; not localization error and not an isolated architectural ablation.'))
    by={(r['clip'],r['model']):r for r in rows}
    for r in rows:
        if r['model']=='original':continue
        b=by[r['clip'],'original'];v=dict(clip=r['clip'],model=r['model'],training_exposed=r['training_exposed'],marker_frames=r.get('marker_frames',0),original_identity=bool(b.get('identity_accepted')),adapted_identity=bool(r.get('identity_accepted')),original_marker_detections=b.get('marker_detection_frames',0),adapted_marker_detections=r.get('marker_detection_frames',0),original_2s_windows=b.get('windows_by_duration',{}).get('2.0',0),adapted_2s_windows=r.get('windows_by_duration',{}).get('2.0',0))
        paths=[OUT/'runs'/f'{r["clip"]}_{k}'/'speed_check.csv' for k in ['original',r['model']]]
        if all(p.exists() for p in paths):
            x,y=map(pd.read_csv,paths);common=x.merge(y,on=['frame','t_s'],suffixes=('_original','_adapted'));v['gps_common_frames']=len(common)
            if len(common):
                np.testing.assert_allclose(common.gps_speed_mps_original,common.gps_speed_mps_adapted);v['original_common_speed_mae_mps']=float(common.absolute_error_mps_original.mean());v['adapted_common_speed_mae_mps']=float(common.absolute_error_mps_adapted.mean())
        paired.append(v)
    pd.DataFrame(paired).to_csv(OUT/'paired_adaptation.csv',index=False,encoding='utf-8-sig')
    mixed=[by[r['clip'],next((k for k in r['models'] if k!='original'),'original')] for r in clips]
    overall=[]
    for name,cohort in [('original',[r for r in rows if r['model']=='original']),('view_adapted_mix',mixed)]:
        for scope in ['all','core','supplementary']:
            g=[r for r in cohort if scope=='all' or r['core']==(scope=='core')]
            overall.append(dict(model=name,scope=scope,recordings=len(g),identity_accepted=sum(bool(r.get('identity_accepted')) for r in g),target_exported=sum(r.get('target_observations',0)>0 for r in g),pair_available=sum(r.get('following_observations',0)>0 for r in g),**{f'window_{dur}s_recordings':sum(r.get('windows_by_duration',{}).get(str(float(dur)),0)>0 for r in g) for dur in [1,2,3,4]}))
    pd.DataFrame(overall).to_csv(OUT/'overall_coverage.csv',index=False,encoding='utf-8-sig')
    summary=dict(recordings=60,core_recordings=43,supplementary_recordings=17,class_counts=dict(Counter(str(r['label']) for r in clips)),decoded_frames=sum(r['frames'] for r in prep['rows']),model_recording_runs=90,model_frame_evaluations=sum(r['frames'] for r in inference),newly_computed_frame_evaluations=sum(r['computed'] for r in inference),cached_frame_evaluations=sum(r['reused'] for r in inference),overall=overall,groups=groups,original_failure_categories=dict(Counter(r['category'] for r in failures if r['model']=='original')),paired_adaptation=paired)
    save(OUT/'survey_summary.json',summary);save(OUT/'validation.json',dict(status='PASS',rows=validation,checkpoint_sha256_unchanged=True,finite_calibrations=True,frame_counts_match=True,timestamps_nondecreasing=True,net_gap_same_coordinate_formula=True,note='Software/data consistency only. Does not validate true identity continuity, lane assignments, metric geometry, or GPS timing.'))
    classification=read(OUT/'classification_summary.json');lines=['全量路测测试结果 | 2026-11-01','',f"60 段视频（AV 30 / HV 30），三个核心视角 43 段，补充视角 17 段。全部完成解码、相机参数、检测和 MOT。共 {summary['decoded_frames']} 个视频帧、90 个模型-视频组合、{summary['model_frame_evaluations']} 次逐帧模型评估（包含经权重与相机矩阵一致性检查后复用的缓存）。",'原模型覆盖全部 60 段；现有 Verona 适配权重覆盖 16 段，Seminole 适配权重覆盖 14 段。此次没有新增训练。','', '按视角汇总：自车匹配 / 自车有效点导出 / 存在前车配对 / 至少一个 2 秒候选窗口，分母均为该组全部视频。']
    for g in groups:lines.append(f"{g['model']:10s} {g['site']}/{g['view']}: {g['identity_accepted']}/{g['recordings']} | {g['target_exported']}/{g['recordings']} | {g['pair_available']}/{g['recordings']} | {g['window_2s_recordings']}/{g['recordings']}")
    lines+=['','固定视角组合：Verona 一律用 adapted_V，Seminole 一律用 adapted_S，其余用 original；不按单段结果择优。以下仍依次为自车匹配 / 自车有效点 / 前车配对 / 2 秒窗口。']
    for g in overall:lines.append(f"{g['model']} {g['scope']}: {g['identity_accepted']}/{g['recordings']} | {g['target_exported']}/{g['recordings']} | {g['pair_available']}/{g['recordings']} | {g['window_2s_recordings']}/{g['recordings']}")
    original=[r for r in rows if r['model']=='original'];lines+=['','原模型的全量管线阶段分布：']+[f'{k}: {v} 段' for k,v in summary['original_failure_categories'].items()]
    lines+=['','冻结分类器：每个架构 15 个历史模型平均概率；固定阈值 0.5；每段视频取第一个满足规则的窗口。以下是自动候选轨迹上的诊断正确数，不是独立地面真值支持的路测泛化精度。']
    for q in classification['results']:
        if q.get('duration_s') not in [1,2] or 'architecture' not in q:continue
        m=q['first_per_recording'];nt=q['subsets']['not_checkpoint_training'];f=q['fusion_two_nonoverlap']
        fusion_text=f"{f.get('correct',0)}/{f['n']}" if f['n'] else '无可融合样本'
        lines.append(f"{q['model']} {q['architecture']} {q['duration_s']}s: 正确 {m['correct']}/{m['n']} ({100*m['accuracy']:.1f}%)；覆盖 {m['n']}/{q['recording_denominator']} 段；排除该检测权重训练视频后 {nt.get('correct',0)}/{nt['n']}；两段不重叠窗口融合 {fusion_text}。")
    lines+=['','解读与限制：','1. 60 段均已从源图确认实验车。单帧种子 + 原半自动模板的短时匹配用于身份关联；模板序列失败单列，不能算成网络漏检。短时匹配通过也不证明整条轨迹没有换车或错误延续。','2. marker 检测覆盖是投影框包含图像标记的诊断，不是独立 2D/3D 标注上的 recall/AP。无匹配可能来自检测、投影几何、道路/距离门限等环节。','3. 自车点导出要求有实际检测支持并完整处于画面内；跟驰点还要求同帧前车、正净间距及一致车道/方向。没有前车的视频保留自车轨迹，不给旧分类器填零。当前 MOT 仅跟 car，卡车等前车是已知缺口。','4. 分类窗口按 10 Hz，1/2/3/4 秒分别报告，2 秒为主；不设速度下限，最多跨 0.2 秒插值；同车、同前车和车道/方向段内生成。远处仍在图内的框不等于可靠几何，当前距离门限不是人工标注的可靠区间。','5. 车身尺寸、车道聚类、前车关联和真实速度/间距仍未独立验证；平滑或连续不能证明精度。GPS 速度对照沿用数据提供的时间，不拟合位置尺度、时间偏移或分别变换两车。','6. 训练/前期调试视频均留在清单并标记；not_checkpoint_training 只排除对应检测权重的训练记录，仍有相同实验车和相机，不是车辆级独立测试。previously_inspected 表示本轮之前进入过管线试验，不包括本轮源图标注。','7. 1/2/3 秒输入与原分类器 4 秒训练时长不同；全窗口准确率受重叠相关性影响。融合只是最早两个不重叠窗口的概率平均。没有按分类对错筛视频。','', '文件索引：','recording_results.csv：每个视频/模型的身份、覆盖、连续时长、窗口数和 GPS 速度诊断。','view_summary.csv：核心/补充视角分组。paired_adaptation.csv：同视频原模型和适配模型对照。','failure_categories.csv：全量阶段结果；diagnostic 分类，不是逐帧故障真值。','classification/<model>/：预测、首窗口结果、NPZ、元数据和融合结果。','pairs/：仅实验车及其直接前车的候选轨迹、配对审核记录。','source_review/、source_frames/、markers/：源图与身份核验；reviews/：每条已匹配轨迹固定六个位置的叠加图。','validation.json：帧数、有限值、时间、权重和净间距公式检查。','', '复现：在 identification 下依次执行（检测使用 .venv-bev，准备使用 .venv-classifier）：','.venv-classifier/Scripts/python.exe scripts/field_full_survey.py prepare','.venv-bev/Scripts/python.exe scripts/field_full_survey.py infer','.venv-bev/Scripts/python.exe scripts/field_full_review.py markers','.venv-bev/Scripts/python.exe scripts/field_full_review.py track','.venv-bev/Scripts/python.exe scripts/evaluate_field_survey.py export','.venv-classifier/Scripts/python.exe scripts/evaluate_field_survey.py classify','.venv-bev/Scripts/python.exe scripts/report_field_survey.py','']
    lines+=['轨迹稳定性：stability_summary.csv 在同一组有检测支持的帧上比较检测位置与平滑位置的邻点中点残差（相邻间隔最多 0.2 秒）。它度量局部抖动，不是独立定位误差，也不是 BEVHeight 架构消融。','目视复核：HV_V_EW_6 的 adapted_V 轨迹 27 在自车出画后仍延续到道路/阴影处（固定叠加图第 308、348、389 帧）。不得将整条已关联轨迹直接当作真实自车轨迹。详见 trajectory_review_notes.json；这是抽样复核记录，未完成全量逐帧轨迹真值标注。','overall_coverage.csv：原模型与固定视角组合在全部/核心/补充范围的覆盖对照。detection_reviews/：60 段统一源图锚点处的原模型及适配检测框对照，由 review_survey_detections.py 生成。']
    (OUT/'RESULTS.txt').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5));g=[g for g in groups if g['model']=='original'];x=np.arange(len(g));labels=[g['site'].split('_')[0]+'/'+g['view'].replace('View_','') for g in g]
    for offset,col,name in [(-.27,'identity_accepted','Image identity matched'),(0,'pair_available','Candidate following pair'),(.27,'window_2s_recordings','Candidate 2 s window')]:axes[0].bar(x+offset,[z[col]/z['recordings'] for z in g],.26,label=name)
    axes[0].set_xticks(x,labels);axes[0].set_ylim(0,1.05);axes[0].set_ylabel('Fraction of all recordings in view');axes[0].set_title('Original model: coverage, not accuracy');axes[0].legend(fontsize=8)
    for key,color in [('adapted_V','#b5533d'),('adapted_S','#23789a')]:
        p=[v for v in paired if v['model']==key and v['marker_frames']];axes[1].scatter([v['original_marker_detections']/v['marker_frames'] for v in p],[v['adapted_marker_detections']/v['marker_frames'] for v in p],label=key,color=color,alpha=.75)
    axes[1].plot([0,1],[0,1],'k--',lw=1);axes[1].set(xlabel='Original marker-gated detection fraction',ylabel='Adapted marker-gated detection fraction',title='Paired diagnostic; not independent recall');axes[1].legend();fig.tight_layout();fig.savefig(OUT/'survey_overview.png',dpi=170);plt.close(fig)
    print(summary)

if __name__=='__main__':main()
