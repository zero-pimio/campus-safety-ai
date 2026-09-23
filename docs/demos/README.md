# 演示与本地复现

这里记录打架、跌倒、人员入侵和车辆识别的阶段演示及生成入口。**识别框、模型分数或一次告警不等于场景精度与部署验收通过。** 车辆闯道闸、人员闯门禁目前仅完成设计。

本页不分发第三方原片、抽帧图片或本地生成的 MP4。请从作者来源取得适合自身用途的素材，再在本地运行；仓库代码许可不扩大素材、模型的许可范围。下方 `runtime/...` 均为命令将生成的本地路径，不是 GitHub 下载地址。

## 当前结果

| 模块 | 已有本地演示事实 | 仍需改进或验证 |
|---|---|---|
| 打架 | AIRTLab 正例的真实场景分数回放；关联历史运行有 1 START / 1 END | 历史演示使用当时权重；END 为 EOF，不能当作自然停止打架；缺校园嬉闹等难负样本。人框不表示每个人都是打架参与者 |
| 跌倒 | 固定窗口候选在 GMDCSA24 Subject 3 / Fall 03 产生 1 START / 1 END；START 约 2.1124 秒，END 约 3.2960 秒 | END 原因为 `track_changed`，人仍倒地，不能描述为恢复；实验权重与姿态缓存未作为公开下载资产发布 |
| 人员入侵 | YOLO + ByteTrack 与人为配置的禁区，在 AIRTLab 行走片段产生 1 START / 1 END | 素材无真实入侵标注；END 为 EOF；进入、持续、离开和正常经过仍需独立验证 |
| 新车辆识别 | PNNL Parking Lot 2：50 秒、1500 帧、逐帧 YOLO26n 真实推理，展示同帧车辆框与置信度 | 对应停车规则仍为 0 START / 0 END；遮挡时漏检、低分及车型跳变仍存在；检测到车不等于违停 |
| 车辆闯道闸、人员闯门禁 | 已有规则边界、输入契约与验收场景设计 | **Design only：无运行实现或实测视频，也未接入真实控制器** |

车辆新演示没有复制旧框来遮掩漏检：显示阈值为 0.40，约 24 秒白车被行人遮挡时，原始分类短暂变成 `truck`，画面如实显示“货车”。视频 30 fps 是原速播放帧率；本次本机 CPU 完整流程约 246.9 秒，不能据此声称实时 30 fps。

跌倒候选的保留人物测试只有 5 段、约 38.4 秒、1 次跌倒，检出 1 次并有 1 次俯卧撑误报。该短片结果不能代替每小时误报、跨摄像头泛化、RK3588 速度或长期稳定性验收。实验细节见[训练记录](../training/project-progress-2026-09-23.md)。

## 运行准备

以下命令从仓库根目录执行，使用 Python 3.11 和已激活的虚拟环境；Bash 示例适用于 macOS/Linux，Windows 可使用相应 Python 环境或 WSL。先安装 `ffmpeg`、`ffprobe` 并加入 PATH。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[vision]" "pillow>=10,<13"
```

按 [Ultralytics 官方 YOLO26 文档](https://docs.ultralytics.com/models/yolo26/)取得所需权重，检测权重放到 `models/yolo26n.pt`。场景入口要求权重文件已经存在，不会代替用户下载。权重版本不同，结果可能不同。

这些命令会创建本地运行目录；再次执行请使用新的输出目录。当前 CLI 参数已按本仓库源码及 `--help` 核对，完整可选依赖见 [pyproject.toml](../../pyproject.toml)。

## 打架：分数推理与叠加

公开可获得的起步路径使用 PP-TSM 官方部署模型，不要求本项目未发布的微调权重。[运行时准备脚本](../../scripts/setup_video_runtime.sh)会安装相关依赖、从官方地址下载 PP-TSM 并校验压缩包 SHA；在 Bash 中显式指定已安装的 Python 3.11：

```bash
PYTHON311="$(command -v python3.11)" bash scripts/setup_video_runtime.sh
source .venv/bin/activate
bash scripts/download_video_smoke_samples.sh

python -m campus_safety_ai.apps.fight_video \
  --video datasets/samples/fight-fi001.mp4 \
  --model-config configs/models/fight-pptsm-v1.toml \
  --output-dir runtime/demo-fight-new \
  --events runtime/demo-fight-new/events.jsonl \
  --outbox runtime/demo-fight-new/outbox.sqlite3 \
  --evidence-dir runtime/demo-fight-new/evidence

python -m campus_safety_ai.apps.fight_visualize \
  --video datasets/samples/fight-fi001.mp4 \
  --observations runtime/demo-fight-new/fight-fi001-observations.jsonl \
  --detector models/yolo26n.pt \
  --output runtime/demo-fight-new/fight-demo.mp4
```

下载脚本的两段冒烟素材来自 [Surveillance Camera Fight Dataset 作者仓库](https://github.com/seymanurakti/fight-detection-surv-dataset)。它们用于检查链路，不保证触发完整事件，也不等同于上表历史 AIRTLab 演示。历史素材的作者来源是 [AIRTLab 数据仓库](https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos)。数据范围和使用边界见[打架数据集说明](../research/fight-datasets.md)。

## 人员入侵：本地视频与示范禁区

从上述 AIRTLab 作者仓库获取 `violence-detection-dataset/non-violent/cam1/51.mp4`，保留目录结构并放在 `datasets/private/airtlab/` 下。以下场景配置仅针对该画面；换摄像头或素材时需要重新画区域。

```bash
python -m campus_safety_ai.apps.scene_video \
  --video datasets/private/airtlab/violence-detection-dataset/non-violent/cam1/51.mp4 \
  --event intrusion \
  --scene-config configs/scenes/airtlab-intrusion-demo.toml \
  --event-config configs/events/intrusion-v1.toml \
  --detector models/yolo26n.pt \
  --camera-id airtlab-walk-demo --source-epoch 1 \
  --tracker bytetrack --frame-stride 3 --device cpu \
  --output-dir runtime/demo-intrusion-new
```

输出目录包含 `annotated.mp4`、`run.json`、`detections.jsonl`、`events.jsonl` 和本地 Outbox。示范区域是人工配置，不表示原录像中的人物实际闯入了禁区。该视频按抽帧间隔播放，不能与逐帧检测性能混为一谈。

## 车辆识别：50 秒新素材

从 [UCF CRCV 官方 Parking Lot 页面](https://www.crcv.ucf.edu/data/ParkingLOT/)下载 **PNNL Parking Lot 2**，保存为 `datasets/private/parking/PNNLParkingLot2.avi`。本次采用文件的 SHA-256 为 `c6c1c7929258622fb068fd9af2020306b8985702c246b515260917a71847d7c5`，实测为 1500 帧、30 fps、50 秒。来源核验与时间戳边界见[素材说明](../research/parking-video-sources-2026-09-23.md)。

```bash
python -m campus_safety_ai.apps.scene_video \
  --video datasets/private/parking/PNNLParkingLot2.avi \
  --event parking \
  --scene-config configs/scenes/pnnl-parking-demo.toml \
  --event-config configs/events/parking-v1.toml \
  --detector models/yolo26n.pt \
  --camera-id pnnl-parking-demo --source-epoch 1 \
  --started-at 2000-01-01T00:00:00Z \
  --tracker bytetrack --frame-stride 1 --confidence 0.1 --device cpu \
  --no-video --output-dir runtime/demo-parking-inference-new

python scripts/render_parking_recognition_demo.py \
  --run-dir runtime/demo-parking-inference-new \
  --output runtime/demo-parking-video-new/parking-recognition.mp4 \
  --font path/to/ChineseFont.ttc
```

将 `--font` 换为本机已有的中文 TTF/OTF/TTC 字体文件，例如 Noto Sans CJK；该参数支持不同操作系统的字体路径。macOS 默认字体可用时可省略 `--font`。渲染器要求运行已完成、`frame_stride=1`、逐源帧检测齐全，并检查源/模型哈希、相机、序号与相对时间；输出 H.264 / yuv420p MP4、`preview.jpg`、`verification.json`。

保留完整画面，逐帧显示实际车辆检测，不添加追踪 ID、停车计时或告警。原片车辆在开头已停稳，未观察到驶离；EOF 只能表示录像结束。人为禁停区不能作为原场地实际违停标签。详见[停车规则诊断](../audits/2026-09-23/parking-rule-diagnosis.md)。

## 跌倒：实验运行的条件与入口

素材来源为 [GMDCSA24 作者仓库](https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos)；历史离线演示另使用 [UR Fall Detection Dataset 作者页面](https://fenix.ur.edu.pl/mkepski/ds/uf.html)中的素材。两种实验不能混作同一个模型或同一套持续告警结果。

**当前公开仓库没有最新实验分类头、完整姿态缓存及原始视频下载包，不能克隆后直接重现同一分数和演示。** 需要先按[训练与数据复核记录](../training/project-progress-2026-09-23.md)准备授权素材、基线模型、登记缓存并训练；只下载 YOLO 姿态模型还不够。相关入口可先查看：

```bash
python -m pip install -e ".[training]"
python scripts/extract_fall_web_pose.py --help
python scripts/train_fall_windows.py --help
python -m campus_safety_ai.apps.fall_video --help
python scripts/render_fall_run_demo.py --help
```

在本地已经具备与姿态检测器、采样和描述子配置相匹配的分类头后，可运行以下命令；其中视频和 checkpoint 是需要自行准备的输入路径：

```bash
python -m campus_safety_ai.apps.fall_video \
  --video path/to/fall-video.mp4 \
  --checkpoint path/to/trained-best.pt \
  --detector models/yolo26s-pose.pt \
  --event-config configs/events/fall-web-experimental-v1.toml \
  --camera-id fall-local-demo --device cpu \
  --output-dir runtime/demo-fall-new
```

该入口生成运行记录、观测、事件及证据；叠加视频另由 [render_fall_run_demo.py](../../scripts/render_fall_run_demo.py)读取匹配的运行、登记姿态缓存、分类头与 manifest 生成。当前跌倒渲染脚本仍使用固定 macOS 字体路径，不应承诺跨平台开箱即用；新车辆渲染脚本的 `--font` 不能误用于它。

## 两类闯卡与后续视频验收

[车辆闯道闸](../event-specs/vehicle-gate-v1.md)、[人员闯门禁](../event-specs/pedestrian-access-v1.md)及[联合设计](../design/access-control-v1.md)目前均为设计文档。后续视频必须说明门闸状态、授权记录来自真实设备还是模拟输入，展示正常通行、异常通行、等待迟到信号及事件结束，不能用设计动画冒充已实现的模型联动。

每个完成模块都需要可播放演示、来源与模型版本、同一次运行的记录以及媒体核验。核验至少包括完整解码、时长/帧数、框与分数同步、开始和结束原因、正常反例；还要保留未完成的场景验收项。任何演示都不替代真实平台联调和 RK3588 长时间实测。
