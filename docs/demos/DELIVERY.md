# 各模块演示视频交付与验收

GitHub 读者请先看[演示与本地复现](README.md)。下文记录工作区内已生成的视频与验收情况，`runtime/` 路径均为本地产物；其中选定演示另复制到公开展示目录，不包含原始素材集合、完整缓存或实验权重。

2026-09-23 根据用户要求新增：**每个项目模块完成时必须交付可播放演示视频，并向用户展示。代码、截图、模型权重或 JSON 不能替代视频交付。** 此要求适用于本轮六个模块及之后明确新增的模块。

## 当前可观看的交付

**在线入口：[GitHub Pages 视频展示页](https://zero-pimio.github.io/campus-safety-ai/)**。2026-09-23 新增打架、固定窗口跌倒和人员入侵三段，保留两段 URFD 历史演示；共 5 段。车辆新视频保留本地，两项闯卡为设计状态。公开文件 SHA、时长、帧数与完整解码检查见[媒体核验](../../exports/github-pages-showcase/verification.json)，各素材条款见[展示页说明](../../exports/github-pages-showcase/README.md)。

统一展示页：本地演示页（本地产物：`runtime/demos/showcase-20260923/index.html`）。在文件浏览器打开 index.html 即可播放、全屏观看或下载各 MP4；目录可整体搬移，页面不请求外网资源。文件清单、SHA 和媒体核验见同目录 catalog.json、gallery-verification.json。原始历史演示保留，新页面只做汇集和必要兼容转码。

车辆视觉识别已另做新素材演示页（本地产物：`runtime/demos/parking-recognition-20260923/index.html`），便于单独查看完整车辆与遮挡时的检测表现；旧合集及压缩包保留原样。

| 模块 | 当前视频 | 已展示的实际结果 | 尚缺完整演示验收 |
| --- | --- | --- | --- |
| 打架 | fight.mp4（本地产物：`runtime/demos/showcase-20260923/fight.mp4`） | AIRTLab 验证正例真实模型分数回放；关联日志 1 START/1 END | 历史演示权重，不代表当前默认；画面为场景分数，未叠加事件链；END 是 EOF；需正常/嬉闹反例与自然结束 |
| 跌倒 | fall-continuous.mp4（本地产物：`runtime/demos/showcase-20260923/fall-continuous.mp4`） | 本轮 small 固定窗候选实际运行；2.112400 秒 START、3.295967 秒 END，track_changed | 人仍倒地，END 不代表恢复；缺正常→跌倒→真实恢复完整录像与长时正常难例 |
| 人员入侵 | intrusion.mp4（本地产物：`runtime/demos/showcase-20260923/intrusion.mp4`） | 实际 YOLO/ByteTrack + 人工示范禁区；1 START/1 END，EOF 收口 | 没有入侵真值；需进入→持续→离开及不越界/短探入反例 |
| 车辆违停 / 车辆视觉识别 | 新素材识别视频（本地产物：`runtime/demos/parking-recognition-20260923/parking-recognition.mp4`）；旧演示（本地产物：`runtime/demos/showcase-20260923/parking.mp4`）保留 | 新 PNNL 停车场素材，50 秒、1500 帧真实逐帧 YOLO26n 推理；展示车辆框、置信度和时间；对应停车规则仍为 0 START/0 END | 遮挡时有低分/漏检；车辆框不等于违停；缺稳定驻留告警与真实驶离的完整验收 |
| 车辆闯道闸 | 待实现，无视频 | [设计与演示场景](../event-specs/vehicle-gate-v1.md) 已写明 | 通行/授权模块、真实视频联动、尾随/正常/迟到信号和结束章节 |
| 人员闯门禁 | 待实现，无视频 | [设计与演示场景](../event-specs/pedestrian-access-v1.md) 已写明 | 通行/授权模块、单人/尾随/护送/应急/未知与结束章节 |

以上四段是已有阶段成果，六个模块均没有据此标记“全部验收完成”。历史 URFD 跌倒/走路视频继续保留，不能替代本轮固定窗口告警结果。短片时长也不是每小时误报或长期稳定性实测。

本轮已实际打开本地页面并在内置浏览器播放全部四片：打架/跌倒/入侵到达片尾，违停播放至 16.8 秒后暂停；全部文件另经完整解码。新版跌倒保留 167 帧、5.601633 秒及逐帧 PTS，START/END 显示时间未提前。首版微秒图像时间基导致异常 H.264 名义帧率，已修正为合理 VUI、Level 4.1，首版保存在独立目录；浏览器播放复核通过。详细状态见展示目录 browser-verification.json。

## 车辆识别更换素材后的实测

按用户“视觉识别那一块，再换一个演示视频试试”的要求，使用 [UCF / PNNL 官方素材](https://www.crcv.ucf.edu/data/ParkingLOT/)中的 PNNL Parking Lot 2。车辆全景保留，当前 YOLO26n 权重不变，1500 帧全部实际推理；50 秒录像在本机 CPU 的本次完整流程耗时约 246.9 秒，因此不能把 30 fps 播放帧率当成实时推理速度。

原始结果保存在 `runtime/parking-improvement-20260923/visual-full-rate/`；新演示视频只展示对应帧车辆检测框、类别、置信度和时间，不绘制虚构轨迹、计时或告警。显示门限 0.40 未用于改变检测缓存。视频约 24 秒处，白色小车被行人遮挡时短暂分类为 truck，画面如实显示“货车”；部分帧也低于显示门限。车辆视觉识别仍需改进，不能据此声称精度验收通过。

对应原停车策略实际为 0 START / 0 END。5 Hz 诊断回放定位到短低分清零与恢复后偏移 anchor 再次清零，详见[诊断记录](../audits/2026-09-23/parking-rule-diagnosis.md)。本轮没有修改规则门限或车型输出来制造告警。片中车辆从一开始已停稳，没有真实驶离过程；新素材与来源记录见[研究笔记](../research/parking-video-sources-2026-09-23.md)。

可复用渲染命令（输出文件须为新路径）：

```bash
.venv/bin/python scripts/render_parking_recognition_demo.py \
  --run-dir runtime/parking-improvement-20260923/visual-full-rate \
  --output runtime/demos/parking-next/parking-recognition.mp4
```

新视频媒体核验记录为 `runtime/demos/parking-recognition-20260923/verification.json`，浏览器实际播放情况另记 `browser-verification.json`。原始旧合集及其压缩包未覆盖。

## 每个完成模块的最小交付包

- `demo.mp4`：H.264、兼容播放器的像素格式、可播放且有完整关键过程；可另附分场景视频与有章节的合集。
- `manifest.json`：原素材和输出 SHA、来源/使用说明、人物/相机划分、模型/策略/标定版本、实际运行命令、视频时间依据、事件日志与证据关联。
- `events.jsonl` 和适用的 `observations.jsonl` / `assessments.jsonl`：同一真实运行的记录，不能混用不同参数产生的结果；单独推理的定位辅助须显式说明。
- `verification.json`：源/输出帧数与 PTS、时长、完整解码、事件首次可见时间、关键帧视觉检查、未满足项。
- 使用说明：如何播放、各章节实际演示什么、模拟输入和现场验证状态，以及仍需验收的条件。

页面或最终交付消息实际展示可播放媒体，附可下载的本地文件；仅发送目录名不算展示完成。视频由保存的真实推理结果叠加生成时，明确“真实结果回放”；不能把播放流畅度当目标硬件推理速度。

## 画面与时间要求

1. 同屏展示原始画面、实际检测/姿态或区域、视频时间、模型分数及预热/未知状态、实际事件阶段和结束原因。仅有场景分类时不把人框染成“每个人都在打架”。
2. 分数和事件只能在其 `observedAt` 及之后可见，不能按后验 startedAt 回填早期画面；未判定不能显示“安全”或把未知画成正常。
3. 原速视频保留逐帧 PTS；抽帧视频注明采样间隔，并核对媒体时间。加速/慢放、节选或拼接章节必须标明，不能复制驻留帧凑够触发时长，也不能拼接伪造恢复。
4. END 是分析状态收口；source_lost、track_changed、EOF、动作分数清除等原因分别展示。人倒地但动作分数下降时不能写“人员恢复”。
5. 闯卡演示逐项标注视频检测、授权、闸杆状态的来源。真实录像 + 模拟授权须全程同屏说明；固定检测动画不能称为模型识别演示；模拟控制器通过不代表真实平台联调成功。
6. 片段至少覆盖正常过程、正例触发、持续/等待、结束/重置及关键反例；不足的场景保留“待补”，不通过调低阈值美化结果。

## 实际播放检查

生成后用 ffprobe 读取帧数/时间戳、ffmpeg 完整解码；核对与原输入相同的时间轴或显式采样关系。再查看开头、关键触发前后、END 和结尾，确认中文字体、文本完整、框/分数同步及结束语义。浏览器页面须实际打开并检查视频元数据与播放状态；不能仅凭生成命令退出码声称浏览器已播放。

本轮新跌倒渲染使用现成验证运行，不重新训练或重复保留测试。可复用脚本为 [render_fall_run_demo.py](../../scripts/render_fall_run_demo.py)，同目录核验记录包含完整输入绑定。当前脚本使用本机中文字体，需 Pillow、ffmpeg/ffprobe；从仓库根目录执行，输出必须为新文件：

```bash
.venv/bin/python scripts/render_fall_run_demo.py \
  --run-dir runtime/checks/fall-web-small-val-real-20260923 \
  --pose-cache runtime/training/fall-web-pose-small-conf035-v1/gmdcsa24-subject-3-fall-03.json \
  --checkpoint runtime/training/fall-window-web-small-v1-20260923/best.pt \
  --output runtime/demos/fall-replay-next/fall-continuous.mp4
```

视频验收与工程、精度、平台、目标硬件验收分别记录。只有相应代码、回放结果及本页要求的演示均完成，才能把模块标为该阶段完成；真正现场完成仍需真实控制器/摄像头、独立样本与 RK3588 长稳实测。
