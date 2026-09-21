# 校园安全 AI（YOLO）项目

这是根据“视觉识别 YOLO 课题”对话创建的防返工工程骨架。首版范围不是同时实现七种行为，而是先稳定下面这条链路：

```text
FramePacket → Detections → Tracks → EventRecord → SQLite Outbox → Destination
```

当前可运行能力：

- 五类稳定契约 `FramePacket`、`Detections`、`Track`、`BehaviorObservation`、`EventRecord` 全部带校验与序列化对称；
- 可配置的简单 IoU / Ultralytics ByteTrack 跟踪 adapter；轨迹超时会主动收口事件；
- 使用真实时间戳的人员非法闯入状态机；
- 本地视频的 YOLO → 跟踪 → 闯入/车辆驻留规则 → 截图证据 → Outbox 入口；
- 按固定驻留起点计算累计位移的车辆违停规则，支持冷却、漏检中断及源周期收口；
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
- EasyAIoT `/alert/hook` adapter：保留本项目 `EventRecord`，转换为 EasyAIoT 告警字段并通过 SQLite Outbox 投递；鉴权 token 只从环境变量读取。

当前边界：本地 MP4、三种 PC 模型运行时、ByteTrack adapter 和本地证据已跑通；RTSP adapter 已实现但尚未用真实摄像头长稳验收。OpenRemote/MQTT、RKNN/昇腾板卡、校园数据精度和证据保留策略仍未验证。这些属于后续门禁，不能把本骨架当成已部署系统。

## 本轮整改（2026-09-14）

详细任务、优先级、架构、平台整合步骤和验收矩阵见 [整改方案](docs/REMEDIATION-PLAN-2026-09-14.md)，视频逐文件处理记录见 [清理清单](docs/audits/2026-09-14/video-cleanup.json)。

打架视频 CLI 现在逐窗口写观测、提交事件，不再等视频结束后统一交付，也不保留完整结果列表。Python 离线调用仍可使用 `FightVideoPipeline.run()` 汇总有限视频；持续源使用 `stream()`。正常 EOF 会产生必要的 END 记录。EasyAIoT 视频入口已使用独立后台交付：分析先本地入队，网络失败自动退避重试，重启可补传。RTSP 自动重连仍待实施，详见方案。

## 立即运行

项目本身只需要 Python 3.11+：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
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

`--video` 也接受 RTSP URL；每次重连必须增加 `--source-epoch`。达到打架阈值的窗口会在 `runtime/fight-video/evidence/` 写入截图、采样短片和 manifest，START/END 事件通过 `evidenceUris` 保留证据引用。事件通过 SQLite Outbox 投递到 `runtime/fight-video/events.jsonl`。可用 `--no-evidence`、`--events`、`--outbox` 和 `--evidence-dir` 覆盖本地装配。

换成自己的 MP4 时只需修改 `--video`。命令会输出每个窗口的 `fight_score`，并写入 `*-observations.jsonl` 与 `*-events.jsonl`。官方模型固定使用 8 帧窗口，`fight=1`；适配器始终取 `softmax(logits)[1]`，不会把非打架类的 top-1 置信度误当成打架分数。

新版可视化中的青色框来自 YOLO26n 的通用 `person` 检测；打架分数来自整幅画面的视频分类，不能判定具体参与者。达到阈值时场景标题和整幅边框显示红色，人物框保持青色。

已验证的官方样例结果：`fi001.mp4` 的打架分数约为 `0.768393`，`nofi001.mp4` 约为 `0.182818`。这只证明链路和标签语义正确，不代表校园场景精度已经达标。模型与安装依据见 [本地视频运行时调研](docs/research/video-fight-runtime.md)，数据来源及许可边界见 [打架数据集调研](docs/research/fight-datasets.md)。

## 跑本地视频闯入与违停规则

新增入口复用现有 YOLO26n 权重和事件投递。需要 `vision` 依赖，模型文件必须已存在，输出目录必须为空。每段录像必须显式选择对应的场景配置；示例区域仅用于功能回放，不代表原视频中真实存在禁区或禁停规定。

```bash
PYTHONPATH=src .venv/bin/python -m campus_safety_ai.apps.scene_video \
  --video datasets/private/airtlab/violence-detection-dataset/non-violent/cam1/51.mp4 \
  --event intrusion --scene-config configs/scenes/airtlab-intrusion-demo.toml \
  --camera-id airtlab-walk-demo --source-epoch 1 \
  --tracker bytetrack --frame-stride 3 --output-dir runtime/scene-intrusion-demo
```

车辆规则使用 `--event parking --scene-config <包含 parking_zone 的 TOML>`，策略默认读取 `configs/events/parking-v1.toml`：驻留 30 秒、归一化位置变化不超过 0.02；这是图像中的驻留判断，不包含授权停车、拥堵和世界坐标速度。协议见 [违停规格](docs/event-specs/parking-v1.md)。

每次输出 `run.json`、`detections.jsonl`、独立 `outbox.sqlite3` 及 `annotated.mp4`；有事件时生成截图，默认 JSONL 目标还会生成 `events.jsonl`。显式提供 `--platform-config` 时才使用该配置的目标。检测记录可用相同策略及跟踪器重放：

```bash
PYTHONPATH=src .venv/bin/python -m campus_safety_ai.apps.offline_replay \
  --input runtime/scene-intrusion-demo/detections.jsonl \
  --event intrusion --scene-config configs/scenes/airtlab-intrusion-demo.toml \
  --tracker bytetrack --output runtime/scene-intrusion-replay/events.jsonl
```

该新入口仅支持本地恒定帧率录像，时间按原始帧号与文件 FPS 计算；不接受 RTSP。EOF、解码/推理异常及处理中断会尝试关闭已开始的事件，失败状态与证据写入错误留在报告中。实际素材与验证边界见 [本轮记录](docs/audits/2026-09-21/scene-validation.md)。

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
  --output-dir runtime/training/fight-tsn-new \
  --epochs 20 \
  --batch-size 8

.venv/bin/campus-safety-export-fight-onnx \
  --checkpoint runtime/training/fight-tsn-new/best.pt \
  --output runtime/training/fight-tsn-new/fight-tsn.onnx

.venv/bin/campus-safety-evaluate-fight \
  --project-root . \
  --manifest datasets/manifests/fight-v1.csv \
  --checkpoint runtime/training/fight-tsn-new/best.pt \
  --split test \
  --output runtime/training/fight-tsn-new/test-evaluation.json
```

训练输出包含逐 epoch 的 `metrics.jsonl`、验证 F1 最优的 `best.pt` 和含参数/数据指纹/运行状态的 `run.json`；导出与评估命令另生成 ONNX、预处理元数据和 split 报告。非空训练目录与已有评估报告会拒绝覆盖；不传 `--output-dir` 时自动创建独立实验目录。继续优化可用 `--init-checkpoint` 初始化新微调实验，`--patience` 控制早停；这是新实验，不恢复旧优化器状态。示例目录使用后再次训练需换名。

模型与阈值必须仅通过验证集选择。评估 `--split val --select-threshold` 可按 balanced accuracy 选视频分类阈值；最终 `--split test --threshold-report <验证报告>` 会校验模型及清单 SHA 后应用已冻结阈值。该阈值不自动替代在线事件的开始/结束迟滞阈值。评估报告保留逐视频预测、分数据集误报/漏报及含解码的吞吐。

原 `fight-tsn-v1` 的公开数据 test split 为 92 段，F1 `0.803738`（TP 43、TN 28、FP 13、FN 8）；这仍只是公开数据基线，最终阈值与验收必须使用按摄像头或日期隔离的校园视频。

低内存微调可使用 `--batch-size 2 --accumulation-steps 4 --freeze-batch-norm`：累积梯度后更新，固定 BatchNorm 运行统计，名义有效批量为 8。视频采样采用有界顺序解码，较大跳距仍使用 seek。实际对照训练、验证/测试指标和后续验收边界见 [2026-09-21 训练优化报告](docs/training/optimization-2026-09-21.md)。

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

## 接入 EasyAIoT

EasyAIoT 负责设备、视频、节点和运营侧；本项目负责打架/闯入事件的时间语义与可靠投递。使用 [EasyAIoT 平台配置](configs/platform/easyaiot-dev.toml) 时，先确认 EasyAIoT 已为对应设备启用实时告警任务，并将 token 放入环境变量：

```bash
export EASYAIOT_TOKEN="..."
PYTHONPATH=src .venv/bin/python -m campus_safety_ai.apps.fight_video \
  --video datasets/samples/fight-fi001.mp4 \
  --platform-config configs/platform/easyaiot-dev.toml
```

适配器 POST 到 EasyAIoT 的 `/admin-api/video/alert/hook`，发送 `device_id`、`event`、`information`、`image_path`、`record_path`、`task_type` 和 `correlation_id`。`information.eventRecord` 保留完整的 `START/UPDATE/END` 事件；EasyAIoT 不可用时，Outbox 行保持未投递，下一次 flush 可重试。已按检出上游校验 `code=0` 与 `data.status=success`，跳过/抑制不标记交付成功；correlation_id 使用同事件稳定的 36 字符 UUID，它不代表接收端去重。源码版本、契约验证、mini 配置和真实联调步骤见 [EasyAIoT 整合记录](integration/easyaiot/README.md)。真实 EasyAIoT 服务、摄像头长稳、MinIO 路径共享和通知闭环仍需现场验收。

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


## 交付队列状态与恢复补传

EasyAIoT 视频 CLI 自动启用后台发送。SQLite 连接分别由分析线程和交付线程创建、使用和关闭；网络等待不会占用视频消费线程。本地 JSONL 与已有离线调用保持同步语义，Python 调用可用 `background_delivery=True` 启用后台发送。

查看队列（只读，不发送、不升级数据库）：

```bash
.venv/bin/python -m campus_safety_ai.apps.deliver_events \
  --outbox runtime/outbox.sqlite3 --status
```

平台恢复后独立补传（不重新执行视频推理）：

```bash
.venv/bin/python -m campus_safety_ai.apps.deliver_events \
  --platform-config configs/platform/easyaiot-mini-dev.toml
```

将上述 `--outbox` 替换为实际队列路径；也可用 `--platform-config` 从配置读取路径。只读检查无需鉴权 token，不创建或迁移数据库。状态包含待发记录数 `pending`、可发送事件数 `readyEvents`、退避事件数 `deferredEvents`、被同事件前序记录阻塞的数量 `blockedRecords`，以及最早队首的下次尝试时间 `nextAttemptAt`（Unix 秒，0 表示立即可发，空队列为 null）。`head` 保留全局最早待发记录，不能代表所有事件都被它阻塞。

`--retry-base`、`--retry-max` 设置补传退避秒数。默认 1 秒起步、指数递增、上限 60 秒。后台/`drain_due()` 按 `eventId` 隔离重试：同一事件按入队顺序发送，START 失败时其 END 等待，但其他事件可继续。调用方须按 revision 顺序入队；不保证不同事件的全局顺序。每批最多尝试 200 条，损坏 JSON 也会作为该事件的失败记录保留，不终止整个消费者。失败只保存异常类型，避免将凭据写入错误字段。Python 旧同步 `flush()` 保留立即发送、失败抛异常的语义，不执行后台退避策略。

旧数据库增加重试字段或 `event_id` 前，若已有事件会生成邻接的 `.backup-<id>.sqlite3` 备份；旧事件、退避时间和 delivered 状态保留。`event_id` 从原 payload 回填，无法解析的旧记录使迁移回滚并保留备份，需要先检查该记录。退出最多等待约 3 秒，未完成事件继续保留；正在进行的 HTTP 请求可能到其超时才结束，日志会明确提示。

同一 outbox 在本机只允许一个发送器（macOS/Linux 文件锁）；同步 `flush()` / `drain_due()` 和后台发送共享此锁，独立生产者仍可入队。后台有界退出后若请求仍在进行，锁会持续到请求真正结束。补传前停止占用该队列的分析进程，或等待其退出，不混跑旧版本发送器。语义仍是至少一次：接收端事务去重、按事件生命周期合并、永久错误人工处理和积压容量治理尚未实现。
