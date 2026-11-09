"""Illustrate reviewed ego tracks versus GPS, preserving metric scale and gaps.

The previous fixed delays are reused, never refitted. A proper rotation fitted
on early samples and a common-start translation remove coordinate convention
differences for this illustration only. No absolute localization claim follows.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import evaluate_trajectory_quality as Q

ROOT=Q.ROOT
SOURCE=ROOT/'experiments/field_review_test_20261102'
ALIGN=ROOT/'experiments/field_time_alignment_20261102'
OUT=ROOT/'experiments/field_three_camera_cases_20261102'
CASES=[('HV_S_EW_6','Seminole NW'),('HV_T_EW_6','Todd SE'),('HV_V_EW_6','Verona SE')]
BLUE='#087e9b'; ORANGE='#d46b24'; GRAY='#969ca4'; PURPLE='#9157a5'
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False,
                     'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white',
                     'axes.labelcolor':'#24364a','text.color':'#24364a','savefig.facecolor':'white'})


def rotation_fit(v,g):
    # Row-vector v @ R = g. Only proper rotation: no reflection or rescaling.
    a=float(np.sum(v[:,0]*g[:,0]+v[:,1]*g[:,1]))
    b=float(np.sum(v[:,0]*g[:,1]-v[:,1]*g[:,0]))
    theta=np.arctan2(b,a);c,s=np.cos(theta),np.sin(theta)
    return np.array([[c,s],[-s,c]]),float(theta)


def broken(ax,x,y,pieces,**kwargs):
    # Keep every supported point; never connect across an unsupported interval.
    for i,p in enumerate(np.unique(pieces)):
        mask=pieces==p
        opts=dict(kwargs)
        if i:opts.pop('label',None)
        ax.plot(np.asarray(x)[mask],np.asarray(y)[mask],**opts)


def case(clip,site,stats,catalog,manifest):
    def use(p):manifest[str(p.relative_to(ROOT))]=Q.sha(p);return p
    series=pd.read_csv(use(ALIGN/'series'/f'{clip}.csv'))
    row=stats.loc[clip];tau=float(row.delay_s)
    use(SOURCE/'clips'/clip/'target_trajectory.csv')
    _,g=Q.gps_frame(use(ROOT/catalog.loc[clip,'gps_path']))
    t=series.t_s.to_numpy();fit=series.is_fit.to_numpy(bool);test=series.is_test.to_numpy(bool)
    gps,valid=Q.gps_interp(g,t+tau);assert valid.all()
    before,bvalid=Q.gps_interp(g,t);assert bvalid.all()
    # Original camera coordinates and native ENU differences, in meters.
    v=series[['x','y']].to_numpy();dv=v-v[0];dg=gps[:,:2]-gps[0,:2]
    r,angle=rotation_fit(dv[fit],dg[fit]);vr=dv@r
    # Plot axes follow the GPS heading of the fit interval, fixed thereafter.
    heading=dg[np.flatnonzero(fit)[-1]];heading=heading/np.linalg.norm(heading)
    axes=np.column_stack((heading,[-heading[1],heading[0]]))
    vv=vr@axes;gg=dg@axes;delta=vv-gg
    error=np.linalg.norm(delta,axis=1);ids=series.piece.to_numpy()
    distance_v=np.linalg.norm(dv,axis=1);distance_g=np.linalg.norm(dg,axis=1)
    spd=series.visual_speed_mps.to_numpy();ref=np.linalg.norm(gps[:,2:],axis=1)
    assert np.allclose(ref,series.gps_after_speed_mps,atol=1e-8)
    assert np.allclose(np.linalg.norm(vv,axis=1),distance_v,atol=1e-8)
    assert np.allclose(np.linalg.norm(np.diff(vv,axis=0),axis=1),np.linalg.norm(np.diff(v,axis=0),axis=1),atol=1e-8)
    assert np.allclose(r.T@r,np.eye(2)) and abs(np.linalg.det(r)-1)<1e-8
    assert not np.any(fit&test)
    assert abs(np.mean(abs(spd[test]-ref[test]))-row.test_speed_after_mae_mps)<1e-8
    # Independently confirm closed-form early-angle estimate with SVD.
    u,_,vt=np.linalg.svd(dv[fit].T@dg[fit]);fix=np.diag([1.,np.linalg.det(u@vt)])
    assert np.allclose(r,u@fix@vt,atol=1e-8)
    last=len(t)-1;held=np.flatnonzero(test);peak=int(held[np.argmax(error[held])])
    m=dict(clip=clip,site=site,delay_s=tau,delay_flags=row.identifiability_flags,
           native_start_s=float(t[0]),native_end_s=float(t[-1]),span_s=float(t[-1]-t[0]),
           fit_cutoff_s=float(row.fit_cutoff_s),test_start_s=float(row.test_start_s),
           alignment_rotation_deg=float(np.rad2deg(angle)),scale=1.,
           endpoint_visual_distance_m=float(distance_v[-1]),endpoint_gps_distance_m=float(distance_g[-1]),
           endpoint_distance_deficit_m=float(distance_g[-1]-distance_v[-1]),
           endpoint_along_error_m=float(delta[-1,0]),endpoint_lateral_error_m=float(delta[-1,1]),
           endpoint_vector_residual_m=float(error[-1]),test_max_vector_residual_m=float(error[peak]),
           test_max_residual_native_time_s=float(t[peak]),test_max_residual_relative_time_s=float(t[peak]-t[0]),
           test_mean_shape_residual_m=float(error[test].mean()),
           test_speed_mae_mps=float(row.test_speed_after_mae_mps),
           visual_speed_at_start_mps=float(spd[0]),visual_speed_at_end_mps=float(spd[-1]),
           gps_speed_at_start_mps=float(ref[0]),gps_speed_at_end_mps=float(ref[-1]),
           fit_samples=int(fit.sum()),test_samples=int(test.sum()),supported_pieces=int(len(np.unique(ids))))
    q=pd.DataFrame(dict(video_t_s=t,elapsed_s=t-t[0],gps_query_s=t+tau,piece=ids,is_fit=fit,is_test=test,
                        video_forward_m=vv[:,0],video_left_m=vv[:,1],gps_forward_m=gg[:,0],gps_left_m=gg[:,1],
                        video_distance_from_start_m=distance_v,gps_distance_from_start_m=distance_g,
                        along_error_m=delta[:,0],lateral_error_m=delta[:,1],vector_residual_m=error,
                        video_speed_mps=spd,gps_speed_mps=ref,gps_before_speed_mps=np.linalg.norm(before[:,2:],axis=1)))
    q.to_csv(OUT/f'{clip}_plot_data.csv',index=False)
    Q.save(OUT/f'{clip}_alignment.json',dict(**m,rotation_camera_to_enu=r.tolist(),
        camera_origin=v[0].tolist(),gps_origin_enu=gps[0,:2].tolist(),enu_to_plot_axes=axes.tolist(),
        alignment_scope='Ego illustration only; fixed transform would apply to all same-camera vehicles. Source geometry unchanged.',
        fit_policy='Same first 40 percent fit interval as prior delay; common initial sample anchors translation; no scale fit.'))
    return q,m


def time_axis(ax,q,m):
    t0=m['native_start_s']
    ax.axvspan(0,m['fit_cutoff_s']-t0,color='#dce6f0',alpha=.55,zorder=-4)
    ax.axvline(m['test_start_s']-t0,color='#708090',lw=1,ls=':')
    ax.set_xlabel('距首个有效观测的时间（s）');ax.grid(alpha=.16)
    ax.set_xlim(0,float(q.elapsed_s.max()))


def xy(ax,q,m):
    ax.plot(q.gps_forward_m,q.gps_left_m,color=ORANGE,lw=2.4,label='GPS 参考（时间已校正）')
    broken(ax,q.video_forward_m,q.video_left_m,q.piece,color=BLUE,lw=2.3,label='提取轨迹')
    # Pair the same video instants, with timestamp labels on the GPS path.
    for ix in np.unique(np.linspace(0,len(q)-1,4,dtype=int)):
        z=q.iloc[ix]
        ax.plot([z.gps_forward_m,z.video_forward_m],[z.gps_left_m,z.video_left_m],color='#9ca8b5',lw=.9,ls='--')
        ax.scatter([z.gps_forward_m],[z.gps_left_m],s=23,color=ORANGE,zorder=5)
        ax.scatter([z.video_forward_m],[z.video_left_m],s=23,marker='s',color=BLUE,zorder=5)
        ax.annotate(f'{z.elapsed_s:.1f}s',(z.gps_forward_m,z.gps_left_m),xytext=(0,9),textcoords='offset points',fontsize=8,ha='center')
    ax.scatter([0],[0],marker='o',facecolors='white',edgecolors='#24364a',s=50,zorder=6)
    ax.set_aspect('equal',adjustable='datalim');ax.set_xlabel('沿初始行驶方向的位置（m）');ax.set_ylabel('向左横向位置（m）')
    ax.grid(alpha=.16);ax.margins(x=.08,y=.3)
    ax.set_title('A  空间轨迹：起点与初始方向对齐',loc='left',fontweight='bold',fontsize=11)
    ax.text(.02,.04,f'末端同一时刻相差 {m["endpoint_vector_residual_m"]:.2f} m\n横纵轴等比例；虚线连接同一时刻',transform=ax.transAxes,fontsize=9,
            bbox=dict(boxstyle='round,pad=.35',facecolor='white',edgecolor='#d2dbe4',alpha=.94))


def distance(ax,q,m):
    ax.plot(q.elapsed_s,q.gps_distance_from_start_m,color=ORANGE,lw=2.1,label='GPS 参考')
    broken(ax,q.elapsed_s,q.video_distance_from_start_m,q.piece,color=BLUE,lw=2.1,label='提取轨迹')
    ax.set_title('B  从共同起点走了多远',loc='left',fontweight='bold',fontsize=11)
    ax.set_ylabel('距起点的直线距离（m）');time_axis(ax,q,m)
    ax.text(.04,.92,f'末端 GPS {m["endpoint_gps_distance_m"]:.2f} m / 提取 {m["endpoint_visual_distance_m"]:.2f} m',
            va='top',transform=ax.transAxes,fontsize=9)


def speed(ax,q,m):
    ax.plot(q.elapsed_s,q.gps_before_speed_mps,color=GRAY,lw=1.2,ls='--',label='GPS 校正前')
    ax.plot(q.elapsed_s,q.gps_speed_mps,color=ORANGE,lw=2.1,label='GPS 校正后')
    broken(ax,q.elapsed_s,q.video_speed_mps,q.piece,color=BLUE,lw=2.1,label='提取速度')
    ax.set_title('C  速度：差在量级还是变化趋势',loc='left',fontweight='bold',fontsize=11)
    ax.set_ylabel('速度（m/s）');time_axis(ax,q,m)
    ax.text(.04,.92,f'后段速度 MAE = {m["test_speed_mae_mps"]:.2f} m/s',va='top',transform=ax.transAxes,fontsize=9)
    ax.legend(loc='best',fontsize=8,framealpha=.9)


def residual(ax,q,m):
    broken(ax,q.elapsed_s,q.along_error_m,q.piece,color=BLUE,lw=2,label='纵向误差（负值：落后于 GPS）')
    broken(ax,q.elapsed_s,q.lateral_error_m,q.piece,color=PURPLE,lw=1.8,label='横向误差（左侧为正）')
    ax.axhline(0,color='#777777',lw=.8)
    ax.set_title('D  同一时刻的位置差',loc='left',fontweight='bold',fontsize=11)
    ax.set_ylabel('提取位置 − GPS 位置（m）');time_axis(ax,q,m);ax.legend(fontsize=8,loc='best')
    ax.text(.04,.06,f'后段最大相对位置差：{m["test_max_vector_residual_m"]:.2f} m（{m["test_max_residual_relative_time_s"]:.1f}s）',transform=ax.transAxes,fontsize=9)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest={}
    for p in [Path(__file__).resolve(),ROOT/'scripts/evaluate_trajectory_quality.py',ROOT/'catalogs/field_clips.csv',
              ALIGN/'clip_results.csv',ROOT/'experiments/field_trajectory_quality_20261102/case_selection.json']:
        manifest[str(p.relative_to(ROOT))]=Q.sha(p)
    Q.save(OUT/'protocol.json',dict(cases=[{'clip':c,'view':s} for c,s in CASES],
        case_selection='Reuse the previously selected HV case in each view, keeping driving state consistent. No selection by GPS error in this visualization.',
        time_alignment='Reuse field_time_alignment_20261102 per-recording constant delay; do not refit.',
        spatial_alignment='Start at first supported matched sample; proper rotation fitted only on the previous early fit samples. Scale fixed to 1. Axes follow early GPS direction.',
        reference='Original GPS positions and east/north velocities; reference with uncertain time calibration, not independently validated dense ground truth.',
        support='Reuse 10 Hz samples with at most 0.2 s gaps; broken visual lines show unsupported intervals. Distances are endpoint displacements, not path integrals.',
        caution='Per-clip shape illustration after coordinate-origin/direction removal; not absolute localization accuracy or site-wide georegistration.',
        source_geometry='Only figure coordinates are transformed. Source ego/leader tracks, common geometry, checkpoints and labels unchanged.'))
    stats=pd.read_csv(ALIGN/'clip_results.csv').set_index('clip')
    catalog=pd.read_csv(ROOT/'catalogs/field_clips.csv').set_index('clip_id')
    cases=[case(c,s,stats,catalog,manifest) for c,s in CASES]
    fig,axes=plt.subplots(3,3,figsize=(17,13.4))
    fig.subplots_adjust(left=.065,right=.98,bottom=.09,top=.86,wspace=.24,hspace=.55)
    for i,(q,m) in enumerate(cases):
        for ax,draw in zip(axes[i],[xy,distance,speed]):draw(ax,q,m)
        axes[i,0].text(-.05,1.24,f'{m["site"]}  |  {m["clip"]}  |  固定 GPS 时间偏移 {m["delay_s"]:+.2f}s',
                       transform=axes[i,0].transAxes,fontsize=12,fontweight='bold')
    fig.suptitle('三个摄像头：GPS 参考轨迹与提取轨迹',fontsize=21,y=.98,fontweight='bold')
    fig.text(.5,.945,'保留真实距离尺度；仅消除起点与初始方向差异。展示同一车辆、同一时刻的运动差异。',ha='center',fontsize=11)
    handles=[Line2D([0],[0],color=ORANGE,lw=2,label='GPS 参考（时间校正后）'),Line2D([0],[0],color=BLUE,lw=2,label='提取轨迹')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.035),ncol=2,frameon=False,fontsize=11)
    fig.text(.5,.019,'浅蓝区：拟合固定时延与旋转的前段；竖虚线后：检验区间。视觉曲线缺口不连接。GPS 时延估计仍有不确定性。',ha='center',fontsize=10)
    fig.savefig(OUT/'three_camera_trajectories.png',dpi=170);fig.savefig(OUT/'three_camera_trajectories.svg');plt.close(fig)
    for q,m in cases:
        fig,axs=plt.subplots(2,2,figsize=(14,10))
        fig.subplots_adjust(left=.075,right=.97,bottom=.13,top=.86,wspace=.23,hspace=.36)
        for ax,draw in zip(axs.flat,[xy,distance,speed,residual]):draw(ax,q,m)
        axs[0,0].legend(loc='upper left',fontsize=8)
        fig.suptitle(f'{m["site"]}  |  {m["clip"]}',fontsize=20,y=.965,fontweight='bold')
        fig.text(.5,.915,f'固定时偏 {m["delay_s"]:+.2f}s；起点平移 + 前段拟合的固定旋转；距离比例 = 1.00',ha='center',fontsize=12)
        fig.text(.5,.070,'浅蓝区用于拟合；竖虚线后为检验区间。空间图为相对形状比较，不代表绝对定位精度。',ha='center',fontsize=11)
        fig.text(.5,.039,'GPS 与视觉数据保留原始尺度；蓝线不跨观测缺口。全部时偏均为存在不确定性的探索性估计。',ha='center',fontsize=10)
        fig.savefig(OUT/f'{m["clip"]}.png',dpi=170);fig.savefig(OUT/f'{m["clip"]}.svg');plt.close(fig)
    Q.save(OUT/'case_summary.json',[m for q,m in cases])
    for p,h in manifest.items():assert Q.sha(ROOT/p)==h
    Q.save(OUT/'input_manifest.json',manifest)
    Q.save(OUT/'validation.json',dict(status='PASS',cases=3,source_hashes_unchanged=True,
        rotation_preserves_all_pair_step_lengths=True,rotation_is_proper=True,
        independent_svd_matches_early_rotation=True,speed_matches_previous_test_statistics=True,
        delay_refitted=False,scale_fitted=False,visual_gaps_connected=False))
    print(json.dumps([m for q,m in cases],ensure_ascii=False,indent=2))


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
