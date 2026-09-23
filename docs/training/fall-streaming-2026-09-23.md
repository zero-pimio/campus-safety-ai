# 跌倒连续告警与数据准入（2026-09-23）

> 本文保留首轮工程实现时的状态。随后已完成新候选训练/测试、RTSP 重连和运行保护，当前结果以 [最新推进记录](project-progress-2026-09-23.md) 为准；下文“尚未实现”是当时的历史边界。

本轮沿用现有框架，新增固定窗口跌倒处理、事件开始/结束/重置、本地持久化、连续事件评估及困难负例登记审计。现有模型与原划分没有替换，没有重新训练。工程链路可运行，但识别效果、真实完整恢复过程及长期运行仍未完成验收。

## 已实现

`PoseFrame → 固定时长窗口 → 冻结运动分类头 → FallObservation → FallEventAnalysis → EventRecord → SQLite Outbox`。

- 1 秒固定历史窗口、16 个真实时刻采样点，默认至少每 0.1 秒推理一次。窗口、帧间隔及状态都有界；unknown、关联中断及人物更换不跨窗计算运动。
- 连续确认、迟滞清除、冷却、幂等、源周期隔离及 EOF/错误/中断关闭。确认只累计实际观测时长，低分结束表示动作清除，不表示人员安全。
- 新 CLI 支持本地 CFR 视频、已有可核验姿态缓存，以及 RTSP 单次连接。RTSP 读取结束关闭事件，尚无自动重连。
- 每次观测即时写 JSONL，事件先持久化到本地 Outbox；START 生成截图，END 保留证据。默认投递本地 JSONL，平台适配器未在本轮真实联调。
- 评估按真值事件与唯一 START 一对一匹配，输出每小时误报、漏报率、报警延迟、unknown 和无观测覆盖；无正例时召回/漏报为 null。
- 数据 CSV 审计阻断同人物、摄像头、原始录像组、相同路径或 SHA 跨集合；严格模式检查文件实际 SHA。空模板不能冒充完成采集。

配置见 [fall-v1.toml](../../configs/events/fall-v1.toml)，语义与指标分母见 [事件规范](../event-specs/fall-v1.md)，采集和划分见 [困难负例规范](hard-negatives-v1.md)。

## 真实开发回放及失败记录

本轮只读使用旧的 ADL-10、Fall-06 连续帧及冻结 `3ae4b6237f16f6c3…` 姿态头；这些都已参与开发，不能当成新独立测试。结构化结果见 [验证记录](../audits/2026-09-23/fall-stream-validation.json)。

| 输入与处理规则 | 原帧数 | 观测数 | 预热或 unknown 条数 | START / END | END 原因 |
| --- | ---: | ---: | ---: | ---: | --- |
| ADL-10，固定窗口、旧缓存关联规则 | 300 | 82 | 14 | 1 / 1 | motion_cleared |
| ADL-10，统一单人规则后 | 300 | 94 | 41 | 0 / 0 | 无事件 |
| Fall-06，统一单人规则后 | 100 | 34 | 21 | 1 / 1 | track_changed |
| Fall-06 原 PNG 编码为 30 FPS 视频，重新跑 YOLO | 100 | 35 | 21 | 1 / 1 | track_changed |

第一行与后续各行采用相同 `.5 / .3秒 / 1秒 / 2秒` 事件参数。更早的 `.75 / 1秒 / 2秒 / 10秒` 初始运行也有一次走路误报，记录保留在 `runtime/checks/fall-stream-20260923/adl-10/`。事件参数在首次结果产生前已指定，没有为通过该片段而试调。

缓存适配器初版沿用旧“多检测中选一条轨迹”的关联结果，绕过新视频入口的单人限制。独立复核发现 ADL-10 有 14 帧属于这种情况；已用失败测试复现并修正为 unknown，中断后建立新轨迹。上述统一规则后的最终缓存运行分别保存在 `single-person-adl-10/` 和 `single-person-fall-06/`，原缓存文件未改。

ADL-10 的已标注负类开发片段按首末官方时间覆盖 9.968 秒。统一单人规则前后 unknown **时间**分别为 1.669 秒和 3.511 秒，占 16.74% 和 35.22%；观测条数占比与时间占比不同。最终零告警伴随更低可判断覆盖，不能称为分类头精度提高。评估器在这段无正例录像上正确输出 recall=null、miss_rate=null；不能据此计算真实跌倒事件召回。

Fall-06 的事件因轨迹重建而结束，未验证人员恢复。真实 YOLO 视频试跑的截图已解码检查为 640×480，START/END 引用一致，Outbox 无待发送记录。该试跑使用原 PNG 临时编码为 CFR 视频，只验证视频入口链路；官方毫秒时间的检查使用姿态缓存。它不是 RTSP 或 RK3588 性能测试。

“正常→跌倒动作→低分清除→第二次事件”由可重复的工程测试覆盖；两段实际素材不含可用于证明完整人员恢复的独立人工事件标注，因此未宣称真实恢复闭环已经通过。

## 运行命令

以下命令在项目根目录执行，输出目录必须新建或为空，防止覆盖先前证据。

```bash
# 已有连续缓存，保留官方帧时刻；不重跑 YOLO。
.venv/bin/python -m campus_safety_ai.apps.fall_video \
  --pose-record runtime/training/urfd-pose-continuous-v1/fall-06.json \
  --camera-id fall-development --source-epoch 1 \
  --started-at 2026-09-23T00:00:00Z \
  --output-dir runtime/checks/fall-development-new

# 用户的本地恒定帧率视频，重新跑姿态检测。
.venv/bin/python -m campus_safety_ai.apps.fall_video \
  --video /absolute/path/to/video.mp4 \
  --camera-id campus-camera-01 --source-epoch 1 \
  --output-dir runtime/checks/fall-video-new --device cpu

# 已有人工事件真值时，评估实际事件，而不是统计阳性帧比例。
.venv/bin/python -m campus_safety_ai.apps.evaluate_fall \
  --ground-truth /absolute/path/to/annotations.json \
  --events runtime/checks/fall-video-new/events.jsonl \
  --observations runtime/checks/fall-video-new/observations.jsonl \
  --output runtime/checks/fall-video-new/evaluation.json
```

默认使用本机既有 `models/yolo26n-pose.pt`、`runtime/training/fall-pose-dense-v1-20260921/best.pt` 及其冻结协议，缺文件直接失败，不静默下载或换权重。`--video` 也接受 RTSP 地址；该路径仅完成代码接入，未有真实摄像头验收。平台接入需显式传入平台配置，本轮没有发送外部通知。

每次输出 `run.json`、`observations.jsonl`、`events.jsonl`、`delivered-events.jsonl`、`outbox.sqlite3`，有真实视频且触发时另存 `evidence/`。用本地 `events.jsonl` 评估，因为它保留源周期和结束原因。

## 后续工作

本地完整回归为 **306 项测试、288 个子测试通过**；Ruff 与 `git diff --check` 通过。测试包括连续两次事件、unknown/间隔/轨迹/周期重置、解码错误收口、指标分母与一对一匹配、跨集合泄漏和缓存多人规则。完整输出见 [tests.txt](../audits/2026-09-23/tests.txt)。

最高优先级仍是识别：补齐困难负例与新人物/场景数据，并让训练、验证和部署使用相同固定窗口与未知规则。不能把持续运行代码完成等同于识别效果提高。用户随后授权网上拉取公开素材；下载结果单独保留来源清单，不自动混入旧 train/val/test。

已完成的首批是 23 段原视频和 2 组 RGB 原始序列，约 358 MB，见 [网上素材交付](../research/web-hard-negatives-2026-09-23.md)。标注和划分尚待复核，未重训。

本轮仍未完成 RTSP 自动重连、外部平台幂等联调、积压上限、日志/证据清理，以及 RK3588 速度、内存、精度和真实长时间稳定性测量。内存有界不代表磁盘无限运行；日志、证据和 Outbox 的保留策略属于下一轮可靠性任务。
