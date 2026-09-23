# 跌倒流程 CPU 参考测试与 RK3588 待验收项

当前脚本测量真实本地视频经过 YOLO 姿态模型、冻结动作分类头、固定窗口事件策略、截图、SQLite Outbox 和本地 JSONL 交付的耗时与进程峰值 RSS。它只实现 **PyTorch CPU** 后端，不使用 RKNN、NPU、RTSP 或真实平台。它可以在依赖满足的 Linux/RK3588 主机上执行同一 CPU 路径，但不能将该结果标记为 RKNN 性能或板卡验收通过。

脚本：`scripts/benchmark_fall_runtime.py`。测试输入只接受存在的本地视频文件；没有平台配置入口。输出目录必须是新的或空目录，已有报告和输入文件不会覆盖。当前版本还要求系统可运行 `ffprobe`，使用 `OpenCvTimedFileFrames` 按逐帧真实 PTS 回放，不使用平均 FPS 构造时间。

## 执行有限测试

在项目根目录、已配置好项目视觉和训练运行依赖的 Python 环境中执行：

```bash
.venv/bin/python scripts/benchmark_fall_runtime.py \
  --video runtime/checks/fall-stream-20260923/fall06-original-frames-cfr.mp4 \
  --checkpoint runtime/training/fall-pose-dense-v1-20260921/best.pt \
  --detector models/yolo26n-pose.pt \
  --event-config configs/events/fall-v1.toml \
  --output-dir runtime/checks/fall-cpu-reference-new \
  --duration-seconds 20 --warmup-frames 3 --loops 2
```

默认总 wall 时间预算 20 秒，包含输入散列、逐帧 PTS 探测、初始化、warmup 和测量；在帧边界检查预算。一次同步解码、推理或磁盘写入不能被强行中断，所以它不是硬实时超时器。`--loops` 是最大重复次数；源结束或达到时间预算后停止。短视频可以增加 `--loops` 与时间预算以收集更多性能样本，但重复同一录像不会增加独立精度样本。

warmup 使用独立源周期，数据不计入测量帧均值。每次回放使用新的姿态关联状态、窗口/事件状态、递增 `source_epoch` 以及独立事件数据库。分类头权重只加载一次。每次姿态模型初始化耗时单独记录，并计入该测量周期 wall 时间。默认记录真实截图及本地交付；`--no-evidence` 会改变测试范围，比较结果时必须一致。可用 `--torch-threads N` 请求线程数，报告同时记录请求值与后端最终实际值；不能只依据请求值判断资源配置。

如果目标 Linux 主机已有兼容 Python 环境，把命令中的 `.venv/bin/python` 换成该环境解释器，其他参数保持一致。需要带上完整仓库代码、视频、姿态权重、分类头，以及分类头同目录的 `frozen.json` 和 `protocol.json`。脚本会校验冻结选择、训练协议、姿态权重和特征实现的来源一致性，不通过时输出失败报告。目标板卡型号、操作系统、Python/PyTorch/OpenCV 可用构建和 RKNN SDK 尚未提供，本次没有安装 SDK 或尝试转换。

## 如何读取结果

| 文件/字段 | 解释 |
| --- | --- |
| `benchmark.json` | 主机系统/架构/软件版本、后端、视频与模型 SHA-256、配置、warmup、每个源周期、完成或失败状态 |
| `measurement.frames_per_wall_second` | 测量帧数除以测量源周期 wall 总和，包含姿态模型每周期初始化、事件结束和测量日志写入开销 |
| `measurement.video_seconds_per_wall_second` | 处理帧对应的真实 PTS 曝光总时长与 wall 比率；小于 1 表示该输入条件下慢于视频实时速度 |
| `decoded-timing.json` | 首 PTS 规范为 0 的逐帧时间、相邻 PTS 间隔、明确末帧曝光、帧数与累计时长；其 SHA-256 记录在主报告 |
| `video_time_source`、`video_duration_basis` | 帧时刻和曝光时长的具体来源，明确禁止以平均 FPS 近似 VFR |
| `frames_per_total_wall_second` | 测量帧数除以整次测试 wall 时间，额外包括加载、warmup 和报告开销 |
| `peak_rss_bytes` | 整个进程生命周期峰值 RSS；macOS 原始单位是字节，Linux KiB 会转换为字节；包括 warmup，不能当作 NPU 内存或仅模型权重大小 |
| `*-epoch-*/frames.csv` | 每帧真实 `video_seconds`、`frame_video_duration_seconds`，以及解码、姿态推理、窗口与策略、证据与本地交付、完整处理耗时和峰值 RSS |
| `*-epoch-*/observations.jsonl` | 真实行为观测及 unknown 原因 |
| `*-epoch-*/events.jsonl` | 真实 START/END 生命周期及结束原因 |
| `*-epoch-*/outbox.sqlite3`、`delivered-events.jsonl` | 本地持久化交付证据，不代表远端收到告警 |

视频曝光累计规则是：非末帧使用 `[当前 PTS, 下一帧 PTS)` 的间隔，末帧使用 ffprobe 明确给出的 `duration_time` / `pkt_duration_time`；没有明确末帧时长则拒绝运行，不从 FPS 猜测。预算中途停止时仅累计已处理帧的这些已知曝光。尺寸来自实际解码帧，完整回放校验解码帧数与 PTS 数一致。窗口配置优先读取冻结 pose identity 的 `window_policy`。

完整帧耗时从读取该帧前计到本地事件交付结束，不含随后写本行 CSV 的少量开销；该开销包含在周期和整体 wall 时间中。均值、最小、最大值在线累计，不在内存中保留全部视频或所有耗时样本。逐帧 CSV 用于需要时另行计算分位数；长时间测试仍会增长磁盘日志。

缺失输入、来源校验失败或运行错误会抛错，已创建的 `benchmark.json` 保留 `failed` 状态和错误，不产生硬件成功结论。若预算用尽但没有完成任何测量帧，测试同样失败。非法参数或已有非空输出目录在创建报告前拒绝。

## 2026-09-23 旧 nano 本机参考结果

真实执行输出位于 `runtime/checks/fall-runtime-benchmark-20260923-reference/benchmark.json`，保持原报告。这份早期结果使用旧脚本及 CFR 输入，没有当前新增的 PTS sidecar；不对旧报告补造逐帧 PTS 验证记录。这是 macOS Darwin 24.0.0 / arm64 / 8 逻辑 CPU、PyTorch 2.13.0 / Ultralytics 8.4.128 的 **CPU 参考**。没有 RK3588 实机结果。

- 输入 640×480、30 FPS、100 帧 CFR 录像；SHA-256 `46f4fc4e4fe45000112784bc8b5e48ef807cf1bd4b4ed1ebb8f6f10ecba81b06`。
- 分类头 SHA-256 `3ae4b6237f16f6c377f118fc1c0b69e9d069765e8d61f13b351e5403d1f1b368`；姿态模型 SHA-256 `eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9`。
- warmup 3 帧（epoch 1），测量两个独立周期（epoch 2、3），每个周期 100 帧，共 200 测量帧。
- 测量 wall 时间 12.217 秒，**16.37 帧/秒**；视频/wall 比率 **0.546**，当前本机 CPU 路径未达到该 30 FPS 输入的实时速度。
- 平均完整帧耗时 **60.31 ms**，其中姿态推理 **59.28 ms**；整次 wall 时间 13.955 秒，初始加载 1.218 秒。
- 进程峰值 RSS **404,324,352 字节（约 385.6 MiB）**；报告最终 PyTorch 工作线程 7、interop 线程 8。
- 每个测量周期有 35 条观测，其中 21 条为 unknown；各产生 1 个 START 和 1 个 END，本地 pending=0、open=0。**这两个 END 均因 `track_changed` 产生，不是人体恢复证据。** 该录像不提供事件真值评估，重复两次也不能计算独立准确率或长期误报率。

以上报告已通过逐帧 CSV、计数、时间均值、输入散列和跨周期事件 ID 隔离复核。指标会随主机负载、线程数、温度和软件环境变化，不应直接换算成 RK3588 吞吐。

## 2026-09-23 small 候选独立参考结果

严格 PTS 版本真实执行输出位于 `runtime/checks/fall-runtime-small-benchmark-20260923/benchmark.json`，附 `decoded-timing.json` 和同目录 README。主机是 **Apple M1 / macOS Darwin 24.0.0 / arm64 / 8 逻辑 CPU**；仅运行 PyTorch CPU，未使用 RK3588 或 NPU。

使用 `runtime/training/fall-window-web-small-v1-20260923/best.pt`、`models/yolo26s-pose.pt` 和 `configs/events/fall-web-experimental-v1.toml`。分类头 SHA-256 为 `a64b7c46d980ff28c99e763098c28bc051bd8c88f65af82c299187a0d38fa6d5`，姿态权重 SHA-256 为 `a083adb42303728ae14c4bd6bd56d80da46f82fb2564dbd6f31dcc92ea321646`。输入仅为 **val GMD Subject3/Fall03**，没有再次运行独立 test。

```bash
.venv/bin/python scripts/benchmark_fall_runtime.py \
  --video 'datasets/private/web-hard-negatives-20260923/gmdcsa24/videos/Subject 3/Fall/03.mp4' \
  --checkpoint runtime/training/fall-window-web-small-v1-20260923/best.pt \
  --detector models/yolo26s-pose.pt \
  --event-config configs/events/fall-web-experimental-v1.toml \
  --output-dir runtime/checks/fall-runtime-small-benchmark-new \
  --duration-seconds 60 --warmup-frames 3 --loops 1
```

- 独立 warmup 3 帧；正式完整回放 1 遍，167 / 167 帧，1280×720。60 秒是总 wall 上限，实际已完成 1 遍而提前结束，并非运行满 60 秒。
- 真实视频曝光 **5.601633 秒**：末帧相对 PTS 5.568300 秒，加明确末帧曝光 0.033333 秒。没有按平均 FPS 换算。
- 正式循环 wall 时间 **15.761402 秒**，吞吐 **10.5955 帧/秒**，视频/wall 比率 **0.3554**。
- 单帧完整处理平均 **93.534 ms**，其中姿态推理 **91.320 ms**；总 wall **18.888142 秒**。正式循环 wall 包括其姿态模型初始化，不是纯网络前向吞吐。
- 进程生命周期峰值 RSS **382,779,392 字节（365.047 MiB）**；实际 torch intra-op / inter-op 线程为 7 / 8，RSS 包括导入、模型与预热。
- 47 条观测中 19 条 unknown，1 个 START、1 个 END；END 原因是 **`track_changed`，不是恢复**。退出时 pending=0、open=0；只向本地 JSONL/SQLite 写入，未对外发告警。

**与旧 nano 参考的输入、模型、窗口和时间处理方式不同，不能据两组数字直接计算模型性能倍数。** 当前结果只表明这台主机上该 CPU 路径处理这个输入慢于播放速度；不等于 RK3588 性能，也不证明长期内存稳定。

启用本报告的产物校验后，8 项测试和 11 个子测试通过，核对了实际 PTS、每帧曝光、帧数、各阶段累计耗时、源周期、事件关闭与输入 SHA；Ruff 通过。没有修改训练、评估或协议文件。

## 板上仍需执行的验收

1. 明确板卡/内存、系统、SDK、供电、散热、运行后端与 CPU/NPU 使用策略，保留版本和运行配置。
2. 先运行本脚本记录 CPU 基线；若实现 RKNN 后端，分别验证姿态输出、动作头输出和最终事件与冻结参考的一致性，不能仅凭模型导出成功验收。
3. 使用独立的新人物、新摄像头连续录像及事件真值，评估每小时误报、漏报和触发/结束延迟。性能回放不替代精度评估。
4. 用实际分辨率、路数、帧率和截图/告警配置测试速度、CPU/NPU占用、RSS、磁盘增长与温度，再执行数小时或更长稳定性测试，加入源断开、网络故障和积压恢复。
5. 真实平台联调和摄像头长稳单独记录。2026-09-23 本机对 `127.0.0.1:48080`、`:6000` 仅做 TCP 连接探测，两端口均未发现监听；没有发送 HTTP 请求或告警。这不表示其他主机、其他地址或后续时刻的平台状态。

单元检查可以直接运行 `python -m pytest -q tests/test_benchmark_fall_runtime.py`。若要验证实际运行产物，设 `FALL_BENCHMARK_REPORT` 为一份由当前严格 PTS 版本生成、包含至少一个测量周期及 `decoded-timing.json` 的真实 `benchmark.json` 再运行；测试不会模拟硬件成功，也不会自行发送平台消息。
