"""Prepare two image-review tasks; preserve original relations and user answers."""
import json
import sys
import urllib.request
from pathlib import Path
from todd_three_second_tracks import ROOT,OUT,context,read
from track_review_seeds import contact


def main():
    dest=OUT/'human_review';dest.mkdir(exist_ok=True)
    specs=[
        dict(clip='AV_T_WE_4',frames=[155,160,165,168,169,172,175,183,190],review_range=[155,190],
             purpose='确认第168/169帧的三维位置跳变是否来自错误定位；只根据真实画面判断车辆身份和运动。',
             questions={'155至190帧是否始终能认出同一辆试验车':None,
                        '168至169帧车辆是否连续运动而没有画面跳切':None,
                        '这段是否发生遮挡_换道_或与邻车混淆':None,
                        '最后能有把握认出自车的帧号':None,
                        '哪些帧的框明显不贴合自车_请写帧号和说明':None}),
        dict(clip='HV_T_WE_5',frames=[125,130,134,138,142,146,149,152,157],review_range=[125,157],
             purpose='检查130至149帧检测缺口中的同一辆自车是否仍可辨认，及149帧后的检测是否回到同一辆车。',
             questions={'130帧之前与149帧之后是否为同一辆试验车':None,
                        '131至148帧是否始终能在原图辨认自车':None,
                        '中间是否发生遮挡_换道_或与邻车混淆':None,
                        '最后能有把握认出自车的帧号':None,
                        '若不能全程辨认_请记录看不清的起止帧':None})]
    cases=[]
    for spec in specs:
        c=spec['clip'];r,folder,dd,cal,K,M,times=context(c)
        allrows=read(OUT/'low_score_detections'/c/'image_candidates.json')['rows']
        rows=[v for v in allrows if v['frame'] in spec['frames']]
        assert len(rows)==9
        contact(rows,folder,dest/f'{c}_keyframes.jpg',c)
        cases.append(dict(clip=c,review_range_frames=spec['review_range'],
            review_range_seconds=[times[f] for f in spec['review_range']],keyframes=spec['frames'],
            question=spec['purpose'],contact_sheet=str((dest/f'{c}_keyframes.jpg').relative_to(ROOT)).replace('\\','/'),
            answers=spec['questions'],notes=''))
    template=ROOT/'data/annotations/trajectory_extension_review.json'
    if not template.exists():
        template.write_text(json.dumps(dict(schema='field-ego-extension-review/v1',
            purpose='仅用于自车短轨迹补充的人工审计，不替换前车关系标注，不自动修改三维轨迹',
            instructions='只填写 cases 中 answers 的 null 和 notes。判断题填字符串“是”“否”或“无法判断”；帧号填整数；说明题填文字。null 表示尚未回答，并非否。看不清就写无法判断，不需要保证补足三秒。',
            cases=cases),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    guide='''本轮只审两段自车轨迹，不重审前车关系。

先打开 identification/trajectory_extension_review.json；只填 answers 里的 null 和 notes。
判断题写“是”“否”或“无法判断”，帧号写整数；说明题写文字。未回答保留 null。
不需要修改 data/annotations/leader_review_decisions.json，也不要把本模板导入旧前车审核网页。

看图：同目录 AV_T_WE_4_keyframes.jpg、HV_T_WE_5_keyframes.jpg 可直接在 VS Code 打开。
青色框是自动图像跟踪候选，绿色框是关联三维框；3D:False 只表示没有通过关联的三维检测，不等于画面里没有自车。
这些框均是候选，不是人工真值。请根据车辆外观和连续画面判断。

看连续画面：使用原审计网页。VS Code Ports 转发远程 8765，点击在浏览器中打开。
服务未运行时，在 identification 目录运行：
.\\.venv-bev\\Scripts\\python.exe .\\review_web\\server.py --port 8765
左侧“视频范围”切为“全部60段”（默认优先6段里没有本轮两段），输入完整视频名搜索。
在“跳到帧”输入下面的帧号，点击“跳转”，用上一帧/下一帧或慢放检查。
关闭“显示候选框”看原图；需要时勾选2倍放大。网页框是旧审核候选，与新关键帧图可能不同。
网页顶部R01–R06的问题是旧任务。本轮按本说明和新JSON的问题判断。

AV_T_WE_4：先看155–190帧，尤其168和169帧。
判断是不是同一辆试验车、真实画面中的运动是否连续、有无遮挡/换道；记录错误框出现在哪些帧。
绿色三维框的跳动不能直接当作真实车辆运动。若有疑问，可扩看140–200帧。

HV_T_WE_5：先看125–157帧，重点逐帧看131–148帧。
判断缺口里是否仍能持续认出试验车、149帧后是否仍是同一辆、有无被挡住或换道。
若有疑问，可扩看115–170帧。记录看不清的准确帧区间；不必猜被遮住部分的框。

本轮先做身份/可见性复核，不要求手画每一帧，也不要求你估计米坐标、速度或决定是否可插值。
完成后在VS Code保存新JSON（Ctrl+S），告诉我“补轨审计已保存”。也可直接在聊天按两段视频回答。
旧网页的“保存这段审核/保存到工作区”保存的是前车关系；本轮观看时无需点击。
其余三段暂缓：本轮两段处理完，再决定是否值得做远距离补点或查找完整录像。
'''
    (dest/'HOW_TO_REVIEW.txt').write_text(guide,encoding='utf-8')
    print('Prepared',template)
    try:
        with urllib.request.urlopen('http://127.0.0.1:8765/',timeout=3) as response:
            print('Existing review service HTTP',response.status)
    except Exception as exc:print('Review service not reachable; use documented start command:',type(exc).__name__)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8');main()
