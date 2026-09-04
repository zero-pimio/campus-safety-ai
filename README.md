# 校园安全 AI（YOLO）项目

这是根据“视觉识别 YOLO 课题”对话创建的防返工工程骨架。首版范围不是同时实现七种行为，而是先稳定下面这条链路：

```text
FramePacket → Detections → Tracks → EventRecord → SQLite Outbox → Destination
```

当前可运行能力：

- 五类稳定契约中的 `FramePacket`、`Detections`、`Track`、`EventRecord`；
- 可配置的简单 IoU / Ultralytics ByteTrack 跟踪 adapter；轨迹超时会主动收口事件；
- 使用真实时间戳的人员非法闯入状态机；
- 归一化多边形禁区、底部中心点判定、进入/离开延时和冷却；
- SQLite 持久 Outbox、幂等入队、分块 flush、JSONL/内存目标；
- 视频源周期结束时的 `finalize` 收口，断流或回放播完不会留下悬挂的 OPEN 事件；
- 固定 `Detections.jsonl` 离线回放与零依赖单元测试；
- 可选的 Ultralytics YOLO 感知适配器。
- 人员打架视频窗口观测契约，以及带迟滞阈值、持续确认、冷却和幂等的 START/END 状态机；
- PP-Human 官方 PP-TSM 打架二分类模型的本地 MP4 推理入口（Apple Silicon CPU 已验证）。
- 同一 `FightClassifier` interface 下的 Paddle、PyTorch checkpoint 和 ONNX Runtime adapter；
- OpenCV 本地视频/RTSP 视频源 adapter、SQLite Outbox、JSONL 目标和截图/采样短片证据；
- TOML 驱动的事件、模型、运行时和投递配置，以及固定 split 的模型评估报告。

当前边界：本地 MP4、三种 PC 模型运行时、ByteTrack adapter 和本地证据已跑通；RTSP adapter 已实现但尚未用真实摄像头长稳验收。OpenRemote/MQTT、RKNN/昇腾板卡、校园数据精度和证据保留策略仍未验证。这些属于后续门禁，不能把本骨架当成已部署系统。

## 立即运行

项目本身只需要 Python 3.11+：

```bash
python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m campus_safety_ai.apps.fake_chain
PYTHONPATH=src python3 -m campus_safety_ai.apps.offline_replay \
  --input tests/golden/intrusion/detections.jsonl \
  --output runtime/replay-events.jsonl
PYTHONPATH=src python3 -m campus_safety_ai.apps.fight_replay \
  --input tests/golden/fight/observations.jsonl \
  --output runtime/fight-events.jsonl
```

演示会在 `runtime/` 创建 SQLite Outbox 和 JSONL 事件文件。该目录不提交 Git。

## 跑本地视频打架识别

Apple Silicon 上不要使用系统默认的 Python 3.14；视频运行时使用项目内 Python 3.11 虚拟环境：

```bash
brew install python@3.11
./scripts/setup_video_runtime.sh
./scripts/download_video_smoke_samples.sh

.venv/bin/campus-safety-fight-video \
  --video datasets/samples/fight-fi001.mp4 \
  --output-dir runtime/fight-video

.venv/bin/campus-safety-fight-visualize \
  --video datasets/samples/fight-fi001.mp4 \
  --observations runtime/fight-video/fight-fi001-observations.jsonl \
  --output runtime/fight-video/fight-fi001-boxed.mp4
```

默认装配由 `configs/runtimes/pc-dev.toml` 驱动。切换到本项目训练 checkpoint 或导出的 ONNX 不改业务代码：

```bash
.venv/bin/campus-safety-fight-video \
  --video datasets/samples/fight-fi001.mp4 \
  --model-config configs/models/fight-trained-v1.toml

.venv/bin/campus-safety-fight-video \
  --video datasets/samples/fight-fi001.mp4 \
  --model-config configs/models/fight-onnx-v1.toml
```

`--video` 也接受 RTSP URL；每次重连必须增加 `--source-epoch`。达到打架阈值的窗口会在 `runtime/fight-video/evidence/` 写入截图、采样短片和 manifest。事件通过 SQLite Outbox 投递到 `runtime/fight-video/events.jsonl`。可用 `--no-evidence`、`--events`、`--outbox` 和 `--evidence-dir` 覆盖本地装配。

换成自己的 MP4 时只需修改 `--video`。命令会输出每个窗口的 `fight_score`，并写入 `*-observations.jsonl` 与 `*-events.jsonl`。官方模型固定使用 8 帧窗口，`fight=1`；适配器始终取 `softmax(logits)[1]`，不会把非打架类的 top-1 置信度误当成打架分数。

可视化视频中的绿色框来自 YOLO26n 的通用 `person` 检测；打架分数来自整幅画面的视频分类，不表示某一个绿色框中的人已被单独判定为打架。达到阈值时整幅画面显示红色边框。

已验证的官方样例结果：`fi001.mp4` 的打架分数约为 `0.768393`，`nofi001.mp4` 约为 `0.182818`。这只证明链路和标签语义正确，不代表校园场景精度已经达标。模型与安装依据见 [本地视频运行时调研](docs/research/video-fight-runtime.md)，数据来源及许可边界见 [打架数据集调研](docs/research/fight-datasets.md)。

## 训练打架分类模型

第一版训练模型是 MobileNetV3-Small + 8 帧 TSN，固定输入为 `1×24×224×224`，便于后续导出 ONNX 并在 RK3588 上转换 RKNN。训练清单会把 AIRTLab 同一动作的双机位视频放入同一 split，避免事件泄漏。

```bash
.venv/bin/pip install -e '.[training]'
.venv/bin/campus-safety-build-fight-manifest \
  --project-root . \
  --output datasets/manifests/fight-v1.csv

.venv/bin/campus-safety-train-fight \
  --project-root . \
  --manifest datasets/manifests/fight-v1.csv \
  --output-dir runtime/training/fight-tsn-v1 \
  --epochs 20 \
  --batch-size 8

.venv/bin/campus-safety-export-fight-onnx \
  --checkpoint runtime/training/fight-tsn-v1/best.pt \
  --output runtime/training/fight-tsn-v1/fight-tsn.onnx

.venv/bin/campus-safety-evaluate-fight \
  --project-root . \
  --manifest datasets/manifests/fight-v1.csv \
  --checkpoint runtime/training/fight-tsn-v1/best.pt \
  --split test \
  --output runtime/training/fight-tsn-v1/test-evaluation.json
```

训练输出包含逐 epoch 的 `metrics.jsonl`、验证 F1 最优的 `best.pt`、ONNX 模型、输入预处理元数据和独立 split 评估报告。当前 `fight-tsn-v1` 的公开数据 test split 为 92 段，实测 F1 `0.803738`（TP 43、TN 28、FP 13、FN 8）；这仍只是公开数据基线，最终阈值与验收必须使用按摄像头或日期隔离的校园视频。

## 安装真实 YOLO 能力

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[vision,dev]'
```

`UltralyticsPerception` 只负责把厂商结果转换成项目契约。事件代码不得直接读取 `ultralytics.Results`、RKNN tensor 或 ACL tensor。

## 当前开发顺序

```text
人员打架 + 非法闯入 + 车辆违停
+ 一款边缘设备 + OpenRemote 告警闭环
```

人员打架是当前第一条实现主线；跌倒、超速、骑车看手机、追逐和 Mage-VL 复核后移。详细门禁见 [docs/ROADMAP.md](docs/ROADMAP.md)，事件语义见 [打架规格](docs/event-specs/fight-v1.md)和[闯入规格](docs/event-specs/intrusion-v1.md)。

项目总体范围、深模块设计、数据与测试体系、16 周执行计划、风险和当前真实状态见 [总项目规划](docs/PROJECT-PLAN.md)；统一领域术语见 [CONTEXT.md](CONTEXT.md)。

已核对 GitHub/Hugging Face 的候选项目。结论不是整仓照搬，而是保留本项目契约，优先借用 PaddleDetection 的事件算法、Supervision 的 ByteTrack/区域组件；只有明确需要 NVR/回看时才把 Frigate 作为 sidecar。完整证据见 [开源项目选型](docs/research/open-source-project-selection.md)。

## 目录

```text
src/campus_safety_ai/
├── contracts/          # 稳定数据契约
├── core/               # 感知、事件分析、可靠投递
├── adapters/           # 视频源、模型、跟踪、证据与发布目标 adapter
└── apps/               # 假链路和离线回放入口
configs/                # 场景、事件、模型、运行时配置
tests/                  # 契约、golden 回放、Outbox 测试
docs/                   # 规格、ADR、路线和研究记录
deployment/             # PC/RKNN/Ascend/OpenRemote 占位说明
datasets/               # 只提交 manifest，不提交隐私视频
```
