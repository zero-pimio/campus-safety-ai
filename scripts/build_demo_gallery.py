#!/usr/bin/env python3
"""Collect four existing demonstrations into a portable, offline gallery."""
from __future__ import annotations

import hashlib
import html
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runtime/demos/showcase-20260923'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def probe(path):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=codec_name,width,height,duration,level,r_frame_rate:frame=best_effort_timestamp_time',
        '-of', 'json', str(path)], text=True))


def command(*args):
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-xerror', *map(str, args)], check=True)


def make_gallery():
    OUT.mkdir(parents=True, exist_ok=True)
    fight = ROOT / 'runtime/demos/airtlab-101-20260921'
    scene = ROOT / 'runtime/scene-validation-20260921'
    fall = ROOT / 'runtime/checks/fall-web-small-val-real-20260923'
    entries = [
        dict(id='fight', title='打架识别', source=fight/'airtlab-fight-demo.mp4',
             video='fight.mp4', poster='fight-preview.jpg', events=fight/'events.jsonl',
             model=read(fight/'demo-verification.json')['model_version'],
             badge='真实模型分数回放 · 历史演示',
             note='场景打架分数与人物定位。关联日志 1 START / 1 END，EOF 收口；画面未叠加事件阶段，不能把人物框理解为逐人打架标签。',
             missing='待补正常/嬉闹反例和自然结束；历史演示权重不代表当前默认。',
             source_note='AIRTLab violent/cam1/101，验证正例；作者声明研究教育用途。'),
        dict(id='fall', title='跌倒识别', source=OUT/'fall-continuous.mp4',
             video='fall-continuous.mp4', poster='fall-continuous-preview.png', events=fall/'events.jsonl',
             model=read(fall/'run.json')['model_version'],
             badge='最新固定窗口候选 · 真实事件回放',
             note='2.112400 秒触发，3.295967 秒因轨迹变化收口。人仍倒地，END 不代表人员恢复；分数来自实际运行，姿态定位来自另一次真实推理缓存。',
             missing='候选模型未替换默认；待补正常→跌倒→真实恢复的完整录像。',
             source_note='GMDCSA24 Subject3/Fall03，验证素材；作者仓库附 MIT 文本，未另行确认独立视频权利链。'),
        dict(id='intrusion', title='人员入侵', source=scene/'intrusion/demo.mp4',
             video='intrusion.mp4', poster='intrusion-preview.jpg', events=scene/'intrusion/events.jsonl',
             model='YOLO26n + ByteTrack / intrusion-v1',
             badge='真实检测与规则 · 人工示范禁区',
             note='1 START / 1 END，视频 EOF 收口。素材原标签为走路/打招呼，人工设置区域用于演示规则，没有入侵真值。',
             missing='待补进入→持续→离开及短探入反例；不代表校园精度验收。',
             source_note='AIRTLab non-violent/cam1/51；抽帧展示，时间轴保持，研究教育用途。'),
        dict(id='parking', title='车辆违停', source=scene/'parking/annotated.mp4',
             video='parking.mp4', poster='parking-preview.jpg', events=scene/'parking/events.jsonl',
             model='YOLO26n + ByteTrack / parking-v1',
             badge='真实检测运行 · 未触发 · 正例待补',
             note='0 START / 0 END。画面边缘车辆裁切，未满足 30 秒驻留；这不是负例验收通过。为播放器兼容转为 H.264，没有制造告警。',
             missing='待补完整车辆稳定驻留触发、随后驶离的真实正例。',
             source_note='UT Interaction set1/seq1，人物互动素材；抽帧展示，未确认商业使用权限。'),
    ]
    verification = []
    for entry in entries:
        src, dst = entry['source'], OUT/entry['video']
        before = probe(src)
        if src != dst:
            if not dst.exists():
                if entry['id'] == 'parking':
                    command('-i', src, '-map', '0:v:0', '-an', '-c:v', 'libx264', '-crf', '20',
                            '-pix_fmt', 'yuv420p', '-fps_mode', 'passthrough', '-movflags', '+faststart', '-n', dst)
                else:
                    shutil.copyfile(src, dst)
            if entry['id'] != 'parking' and sha(src) != sha(dst):
                raise ValueError(f'Existing copy differs: {dst}')
        after = probe(dst)
        a = [float(f['best_effort_timestamp_time']) for f in before['frames']]
        b = [float(f['best_effort_timestamp_time']) for f in after['frames']]
        error = max(abs(x-y) for x, y in zip(a, b, strict=True))
        if error > .001 or abs(float(before['streams'][0]['duration'])-float(after['streams'][0]['duration'])) > .001:
            raise ValueError('Transcode changed the timeline')
        if after['streams'][0]['codec_name'] != 'h264':
            raise ValueError('Expected H.264 output')
        if after['streams'][0]['level'] > 42:
            raise ValueError('Unexpected H.264 level for this browser demonstration')
        rate_n, rate_d = map(int, after['streams'][0]['r_frame_rate'].split('/'))
        if rate_d <= 0 or rate_n / rate_d > 120:
            raise ValueError('Unreasonable nominal frame rate for this demonstration')
        command('-i', dst, '-map', '0:v:0', '-fps_mode', 'passthrough',
                '-enc_time_base', '1:1000000', '-f', 'null', '-')
        poster = OUT/entry['poster']
        if not poster.exists():
            command('-ss', '2.5', '-i', dst, '-frames:v', '1', '-update', '1', '-n', poster)
        records = [json.loads(line) for line in entry['events'].read_text().splitlines() if line.strip()]
        entry.update(status='stage_demo_available', acceptance_complete=False,
                     sha256=sha(dst), source_video_sha256=sha(src), frame_count=len(b),
                     duration_seconds=float(after['streams'][0]['duration']),
                     starts=sum(r['phase']=='START' for r in records),
                     ends=sum(r['phase']=='END' for r in records),
                     evidence_path=str(entry['events'].relative_to(ROOT)),
                     event_log_sha256=sha(entry['events']), source_path=str(src.relative_to(ROOT)))
        verification.append(dict(id=entry['id'], full_decode=True, max_pts_error_seconds=error,
                                 frames=len(b), sha256=entry['sha256'], codec='h264'))
        del entry['events'], entry['source']
    for module, title in [('vehicle_gate', '车辆闯道闸'), ('pedestrian_access', '人员闯门禁')]:
        entries.append(dict(id=module, title=title, status='design_pending', acceptance_complete=False,
                            video=None, badge='设计已完成 · 运行模块与视频待实现',
                            note='需展示正常授权、未授权完整通行、尾随、迟到/未知及事件结束。模拟授权信号必须全程标注。',
                            missing='尚无真实联动演示，不能用动画代替模型识别视频。'))
    cards = []
    for e in entries:
        def esc(key, entry=e):
            return html.escape(str(entry.get(key, '')))
        if e['video']:
            media = (f'<video controls playsinline preload="metadata" aria-label="{esc("title")}演示" '
                     f'poster="{esc("poster")}" src="{esc("video")}"></video>')
            footer = (f'<div class="actions"><span>{e["duration_seconds"]:.2f} 秒 · {e["frame_count"]} 帧</span>'
                      f'<a href="{esc("video")}" download>下载 MP4 ↗</a></div>'
                      f'<details><summary>素材与版本</summary><p>{esc("source_note")}</p><p>{esc("model")}</p></details>')
        else:
            media = '<div class="pending"><span>下一阶段</span><strong>联动演示待实现</strong><small>设计规格已纳入视频交付要求</small></div>'
            footer = ''
        cards.append(f'<article id="{esc("id")}"><div class="cardhead"><h2>{esc("title")}</h2><span>{esc("badge")}</span></div>'
                     f'{media}<div class="body"><p>{esc("note")}</p><p class="missing">{esc("missing")}</p>{footer}</div></article>')
    style = """
    :root{color-scheme:dark;font-family:-apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif;background:#0b111a;color:#ebf2f8}
    *{box-sizing:border-box}body{margin:0}main{max-width:1320px;margin:auto;padding:56px 32px}header{margin-bottom:30px}
    .eyebrow{color:#74dec9;letter-spacing:3px;font-size:13px}h1{font-size:40px;letter-spacing:-1px;margin:16px 0}
    header p{color:#aabbcb;line-height:1.8;max-width:900px}.stats{display:flex;gap:14px;flex-wrap:wrap;margin-top:24px}
    .stats span{border:1px solid #263748;border-radius:8px;padding:12px 18px;color:#b7c7d7}.stats b{color:#fff;font-size:21px;margin-right:7px}
    .grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px}article{background:#121d2b;border:1px solid #253446;border-radius:16px;overflow:hidden}
    .cardhead{padding:22px 24px 18px}h2{font-size:24px;margin:0 0 10px}.cardhead span{font-size:13px;color:#80d9c7}
    video{width:100%;height:340px;display:block;background:#05090f;object-fit:contain}.body{padding:18px 24px 22px}
    p{line-height:1.8;margin:0 0 12px;color:#c2ceda;font-size:14px}.missing{color:#e5ba7a;font-size:13px}
    .actions{display:flex;justify-content:space-between;gap:10px;align-items:center;border-top:1px solid #293848;padding-top:16px;font-size:13px;color:#98adbe}
    a{color:#89e5d1;text-decoration:none}a:hover{text-decoration:underline}a:focus-visible,summary:focus-visible{outline:2px solid #89e5d1;outline-offset:4px}
    details{margin-top:16px;font-size:12px;color:#9fb0c1}summary{cursor:pointer}details p{font-size:12px;margin-top:10px;overflow-wrap:anywhere}
    .pending{height:200px;background:linear-gradient(120deg,#132c34,#162030);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px;color:#9ab0be}
    .pending span{font-size:12px;letter-spacing:3px}.pending strong{font-size:25px;color:#c3d6df}.pending small{font-size:13px}
    footer{margin-top:30px;padding-top:24px;border-top:1px solid #263748;color:#8fa4b7;font-size:13px;line-height:1.9}
    @media(max-width:850px){main{padding:30px 18px}.grid{grid-template-columns:1fr}h1{font-size:32px}video{height:auto;max-height:420px}.cardhead,.body{padding-left:18px;padding-right:18px}}
    """
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>校园安全 AI · 视频演示</title><style>'+style+'</style><main><header><div class="eyebrow">CAMPUS SAFETY · 2026.09.23</div>'
            '<h1>看见识别过程，也看见当前边界。</h1><p>四段已有阶段演示，可播放、全屏观看和下载。每个模块完成时都必须交付演示视频；当前短片不代表现场、平台或目标硬件已通过完整验收。</p>'
            '<div class="stats"><span><b>4</b>段可播放视频</span><span><b>2</b>类闯卡待实现</span><span>本地播放 · 无外部资源</span></div></header>'
            '<div class="grid">'+''.join(cards)+'</div><footer>点击视频播放按钮观看；所有视频均无自动播放。复制整个目录后打开 index.html 可离线观看。<br>'
            '研究演示素材的来源与使用说明见各卡片及 provenance 目录。<a href="catalog.json" download>下载清单</a> · '
            '<a href="gallery-verification.json" download>媒体核验</a> · <a href="README.txt">观看说明</a></footer></main></html>')
    (OUT/'index.html').write_text(page, encoding='utf-8')
    (OUT/'catalog.json').write_text(json.dumps(dict(date='2026-09-23', modules=entries), ensure_ascii=False, indent=2)+'\n')
    (OUT/'gallery-verification.json').write_text(json.dumps(dict(status='media_verified', videos=verification,
        browser_playback_checked=False, note='Browser inspection is recorded separately by the delivery task.'), indent=2)+'\n')
    (OUT/'README.txt').write_text('校园安全 AI 本地演示\n\n打开 index.html，点击视频即可播放、全屏或下载。整个目录可一起移动。\n'
        '4 段为当前阶段演示，2 类闯卡仍待实现；没有模块由这几段视频被标记为完整验收。\n'
        '打架：历史模型场景分数，EOF 收口。入侵：人工示范禁区，EOF 收口。\n'
        '跌倒：候选非默认，track_changed 结束不是恢复。违停：真实检测未触发，正例待补。\n'
        'AIRTLab：研究教育用途；GMDCSA24：作者库附 MIT 文本，未另外确认独立视频权利链；UT Interaction：商用权利未确认。\n'
        'provenance 包含原运行、事件与来源记录，其中原路径指向开发工作区，不是便携包中的原始素材路径。\n'
        '本包不包含原始训练数据、模型权重或平台访问凭据。\n', encoding='utf-8')
    print(json.dumps(dict(output=str(OUT/'index.html'), videos=len(verification), modules=len(entries)), ensure_ascii=False))


if __name__ == '__main__':
    make_gallery()
