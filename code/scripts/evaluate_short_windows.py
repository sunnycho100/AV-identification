"""Fixed two-second crops of frozen tracks: duration versus actual accuracy.

Reuse prior clip delays, never optimize per crop. All-window diagnostics include
the earlier delay-fitting region; explicitly not an independent test score.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import evaluate_trajectory_quality as Q

ROOT=Q.ROOT
ALIGN=ROOT/'experiments/field_time_alignment_20261102'
OUT=ROOT/'experiments/field_short_windows_20261102'
CASES=['HV_S_EW_6','HV_T_EW_6','HV_V_EW_6']
NAMES={'S_Seminole_Hwy':'Seminole NW','T_Todd_Dr':'Todd SE','V_Verona_Rd':'Verona SE'}


def measure(d):
    duration=float(d.t_s.iloc[-1]-d.t_s.iloc[0])
    v=np.linalg.norm(d[['x','y']].iloc[-1].to_numpy()-d[['x','y']].iloc[0].to_numpy())
    g=np.linalg.norm(d[['gps_e','gps_n']].iloc[-1].to_numpy()-d[['gps_e','gps_n']].iloc[0].to_numpy())
    speed=d.visual_speed_mps.to_numpy()-d.gps_after_speed_mps.to_numpy()
    return dict(start_s=float(d.t_s.iloc[0]),end_s=float(d.t_s.iloc[-1]),duration_s=duration,
        samples=len(d),speed_mae_mps=float(np.mean(abs(speed))),speed_bias_mps=float(np.mean(speed)),
        visual_distance_m=float(v),gps_distance_m=float(g),distance_error_m=float(abs(v-g)),
        distance_error_per_second_mps=float(abs(v-g)/duration),
        relative_distance_error_pct=float(abs(v-g)/g*100) if g>1e-6 else None,
        entirely_after_delay_fit=bool(d.is_test.all()))


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(horizon_s=2.,step_s=.1,required_samples=21,source=str(ALIGN.relative_to(ROOT)),
        scope='All 43 reviewed recordings retained in status list. Windows use the 35 previously estimated fixed delays only.',
        primary_crop='Central 2 s window of the longest continuous supported segment; ties use earliest time. Selection never uses GPS error.',
        other_crops='First and last 2 s of the same longest segment. Enumerate all supported 2 s windows, also outside the longest segment.',
        continuity='Same visual piece and consecutive 10 Hz samples; original observation gaps at most 0.2 s. No bridging unsupported gaps.',
        metrics='Speed MAE; endpoint displacement length error; error divided by duration; error/GPS distance percentage.',
        baseline='All supported samples and first/last endpoints of the existing full track, plus longest supported-segment baseline.',
        timing='Reuse previous clip delay, no per-window re-fit, no speed bias removal or scale fitting.',
        interpretation='Descriptive crop sensitivity on development data, including some earlier delay-fitting samples. Not held-out accuracy. Overlapping windows are not independent recordings.',
        minimum_error_window='Diagnostic lower bound only; never a proposed GPS-based selection rule.',
        image_quality='Reliability and image visibility are distinct; view geometry/clarity are hypotheses, not causally isolated by cropping.')
    Q.save(OUT/'protocol.json',protocol)
    manifest={}
    def use(p):manifest[str(p.relative_to(ROOT))]=Q.sha(p);return p
    use(Path(__file__).resolve());use(ROOT/'scripts/evaluate_trajectory_quality.py')
    stats=pd.read_csv(use(ALIGN/'clip_results.csv'));catalog=pd.read_csv(use(ROOT/'catalogs/field_clips.csv')).set_index('clip_id')
    windows=[];summaries=[];case_series={};checks=0
    for r in stats.itertuples():
        base=dict(clip=r.clip,site=r.site,delay_status=r.status,has_2s=False,delay_s=None if pd.isna(r.delay_s) else r.delay_s)
        if r.status!='ESTIMATED_DELAY':summaries.append(base);continue
        d=pd.read_csv(use(ALIGN/'series'/f'{r.clip}.csv'))
        _,gps=Q.gps_frame(use(ROOT/catalog.loc[r.clip,'gps_path']))
        ref,valid=Q.gps_interp(gps,d.t_s.to_numpy()+r.delay_s);assert valid.all()
        d['gps_e']=ref[:,0];d['gps_n']=ref[:,1]
        assert np.allclose(np.linalg.norm(ref[:,2:],axis=1),d.gps_after_speed_mps,atol=1e-8)
        assert np.isclose(measure(d)['speed_mae_mps'],r.whole_speed_after_mae_mps)
        new=d.piece.ne(d.piece.shift())|d.t_s.diff().gt(.10001)
        d['run']=new.cumsum();runs=[g.reset_index(drop=True) for _,g in d.groupby('run',sort=False)]
        longest=max(runs,key=lambda g:len(g))
        base.update({f'full_{k}':v for k,v in measure(d).items()})
        base['full_supported_runs']=len(runs)
        if len(longest)<21:
            base['short_window_status']='NO_CONTINUOUS_2S';summaries.append(base);continue
        base.update(has_2s=True,short_window_status='AVAILABLE',longest_run=int(longest.run.iloc[0]))
        base.update({f'longest_{k}':v for k,v in measure(longest).items()})
        selected={'first':0,'center':(len(longest)-21)//2,'last':len(longest)-21}
        for label,i in selected.items():base.update({f'{label}_{k}':v for k,v in measure(longest.iloc[i:i+21]).items()})
        local=[]
        for run in runs:
            for i in range(len(run)-20):
                w=run.iloc[i:i+21];m=measure(w)
                assert len(w)==21 and np.allclose(np.diff(w.t_s),.1,atol=1e-8)
                assert w.piece.nunique()==1 and abs(m['duration_s']-2)<1e-8
                local.append(dict(clip=r.clip,site=r.site,run=int(w.run.iloc[0]),**m));checks+=1
        z=pd.DataFrame(local);windows.extend(local)
        base.update(windows_2s=len(z),windows_entirely_after_delay_fit=int(z.entirely_after_delay_fit.sum()),
                    minimum_window_speed_mae_mps=float(z.speed_mae_mps.min()),maximum_window_speed_mae_mps=float(z.speed_mae_mps.max()),
                    mean_window_speed_mae_mps=float(z.speed_mae_mps.mean()),
                    minimum_window_distance_error_m=float(z.distance_error_m.min()),maximum_window_distance_error_m=float(z.distance_error_m.max()))
        summaries.append(base)
        if r.clip in CASES:
            center=longest.iloc[selected['center']:selected['center']+21]
            center.to_csv(OUT/f'{r.clip}_center_2s.csv',index=False);case_series[r.clip]=(d,center)
        checks+=3
    full=pd.DataFrame(summaries);ww=pd.DataFrame(windows)
    full.to_csv(OUT/'clip_summary.csv',index=False);ww.to_csv(OUT/'all_2s_windows.csv',index=False)
    cases=full.set_index('clip').loc[CASES].reset_index();cases.to_csv(OUT/'three_cases.csv',index=False)
    eligible=full[full.has_2s];site=[]
    for name,z in list(eligible.groupby('site'))+[('All',eligible)]:
        site.append(dict(site=name,recordings=len(z),full_speed_mae_mps=float(z.full_speed_mae_mps.mean()),
            longest_speed_mae_mps=float(z.longest_speed_mae_mps.mean()),center_2s_speed_mae_mps=float(z.center_speed_mae_mps.mean()),
            full_distance_error_pct=float(z.full_relative_distance_error_pct.mean()),center_2s_distance_error_pct=float(z.center_relative_distance_error_pct.mean()),
            center_2s_distance_error_m=float(z.center_distance_error_m.mean())))
    Q.save(OUT/'site_summary.json',site)
    # Constant bias shrinks distance error with duration, not per-second error.
    for seconds in [2,4]:
        t=np.arange(round(seconds*10)+1)/10
        synthetic=pd.DataFrame(dict(t_s=t,x=t*15,y=0*t,gps_e=t*20,gps_n=0*t,visual_speed_mps=0*t+15,gps_after_speed_mps=0*t+20,is_test=True))
        m=measure(synthetic);assert np.isclose(m['speed_mae_mps'],5) and np.isclose(m['distance_error_per_second_mps'],5)
        assert np.isclose(m['distance_error_m'],5*seconds)
    for p,h in manifest.items():assert Q.sha(ROOT/p)==h
    Q.save(OUT/'input_manifest.json',manifest)
    Q.save(OUT/'validation.json',dict(status='PASS',windows_checked=checks,source_hashes_unchanged=True,
        all_recordings=len(full),estimated_delays=int((full.delay_status=='ESTIMATED_DELAY').sum()),
        two_second_recordings=len(eligible),overlapping_windows=len(ww),
        held_interval_recordings=int(ww[ww.entirely_after_delay_fit]['clip'].nunique()),
        source_speed_agreement=True,synthetic_duration_bias_check=True,parameters_refitted=False))
    plot(cases,ww,case_series)
    table(cases)
    print(cases[['clip','full_duration_s','full_speed_mae_mps','full_distance_error_m','full_relative_distance_error_pct',
        'center_start_s','center_speed_mae_mps','center_distance_error_m','center_relative_distance_error_pct','minimum_window_speed_mae_mps','maximum_window_speed_mae_mps']].to_string(index=False))
    print(pd.DataFrame(site).to_string(index=False))
    print('Eligible clips',len(eligible),'windows',len(ww),'held interval clips',ww[ww.entirely_after_delay_fit]['clip'].nunique())


def plot(cases,windows,series):
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(3,2,figsize=(13,12));fig.subplots_adjust(left=.085,right=.97,top=.89,bottom=.075,hspace=.6,wspace=.25)
    for row,(r,aa) in enumerate(zip(cases.itertuples(),axs)):
        d,c=series[r.clip];ax=aa[0]
        for i,(_,g) in enumerate(d.groupby('run',sort=False)):
            ax.plot(g.t_s,g.visual_speed_mps,color='#087e9b',lw=2,label='提取速度' if i==0 else None)
        ax.plot(d.t_s,d.gps_after_speed_mps,color='#d46b24',lw=2,label='GPS 参考（固定时偏）')
        ax.axvspan(c.t_s.iloc[0],c.t_s.iloc[-1],color='#e3ecf5',alpha=.65,zorder=-2,label='预定的中间 2 秒')
        ax.set(xlabel='原生视频时间（s）',ylabel='速度（m/s）');ax.legend(fontsize=8,loc='best');ax.grid(alpha=.2)
        ax.set_title(f'{NAMES[r.site]} | {r.clip}\n完整区间 MAE {r.full_speed_mae_mps:.2f} → 中间 2 秒 {r.center_speed_mae_mps:.2f} m/s',loc='left',fontsize=11)
        ax=aa[1];z=windows[windows['clip']==r.clip]
        for i,(_,g) in enumerate(z.groupby('run',sort=False)):
            ax.plot(g.start_s,g.speed_mae_mps,color='#087e9b',lw=2,label='该 2 秒窗速度 MAE' if i==0 else None)
        ax.axhline(r.full_speed_mae_mps,color='#888888',ls='--',lw=1.1,label='完整区间速度 MAE')
        ax.scatter([r.center_start_s],[r.center_speed_mae_mps],color='#d46b24',s=40,zorder=5,label='预定中间窗')
        ax.set(xlabel='2 秒窗口起始时间（s）',ylabel='速度 MAE（m/s）');ax.grid(alpha=.2);ax.legend(fontsize=8,loc='best')
        ax.set_title(f'所有受观测支持的 2 秒窗\n速度 MAE 范围：{r.minimum_window_speed_mae_mps:.2f}–{r.maximum_window_speed_mae_mps:.2f} m/s',loc='left',fontsize=11)
    fig.suptitle('缩短到 2 秒：距离误差变小，速度是否也更准？',fontsize=19,y=.975,fontweight='bold')
    fig.text(.5,.935,'中间窗按最长连续观测段的时间中点确定，不按 GPS 误差挑选；每段沿用同一个时偏。',ha='center',fontsize=11)
    fig.text(.5,.025,'本图为已有轨迹的裁剪诊断（包含部分时偏拟合数据），不是重新训练或独立测试；不连接观测缺口。',ha='center',fontsize=10)
    fig.savefig(OUT/'two_second_comparison.png',dpi=165);fig.savefig(OUT/'two_second_comparison.svg');plt.close(fig)


def table(cases):
    lines=[r'\begin{table}[pos=tbp]',r'\centering',
        r'\caption{Descriptive two-second cropping of the three illustrated HV recordings. The center crop is chosen by the midpoint of the longest supported segment, without consulting GPS errors. The previous per-recording delay is fixed. This diagnostic includes some delay-fitting samples and is not a held-out score.}',
        r'\label{tab:short-window}',r'\small',r'\setlength{\tabcolsep}{5pt}',
        r'\begin{tabular}{lrrrrrrr}',r'\toprule',
        r' & Full span & \multicolumn{2}{c}{Speed MAE (m/s)} & \multicolumn{2}{c}{Distance error (m)} & \multicolumn{2}{c}{Relative error (\%)} \\',
        r'View & (s) & Full & 2 s & Full & 2 s & Full & 2 s \\',r'\midrule']
    for r in cases.itertuples():
        lines.append(f'{NAMES[r.site]} & {r.full_duration_s:.1f} & {r.full_speed_mae_mps:.2f} & {r.center_speed_mae_mps:.2f} & {r.full_distance_error_m:.2f} & {r.center_distance_error_m:.2f} & {r.full_relative_distance_error_pct:.1f} & {r.center_relative_distance_error_pct:.1f} '+r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\par\smallskip',
        r'\begin{minipage}{0.97\linewidth}\footnotesize Distance error compares endpoint displacement lengths, not absolute locations; relative error divides by the GPS displacement length. Full-span speed averages use supported samples only; Todd has gaps in its 10.7-s span. Each 2-s crop requires 21 consecutive 10-Hz samples in one supported segment. Full and cropped rows intentionally use different intervals; lower accumulated distance error alone does not demonstrate better metric reconstruction.\end{minipage}',r'\end{table}']
    (OUT/'short_windows.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':sys.stdout.reconfigure(encoding='utf-8');run()
