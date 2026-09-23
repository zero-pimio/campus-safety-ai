# 跌倒持续识别：断流恢复与退出

状态：本地注入解码器的集成测试已验证；尚未完成真实 RTSP 摄像头长时间运行、真实平台联调或 RK3588 实测。

## 启动

下面使用已经配置的 `RTSP_URL` 环境变量。每次运行必须使用新目录，`camera_id` 标识视频源；同一摄像头进程重启后应传入比之前更大的 `source_epoch`，不要重复使用同一源周期。

```bash
.venv/bin/python -m campus_safety_ai.apps.fall_video \
  --video "$RTSP_URL" \
  --camera-id campus-demo \
  --source-epoch 1 \
  --output-dir runtime/fall-camera-demo-001 \
  --reconnect-attempts 5 \
  --reconnect-initial-backoff-seconds 1 \
  --reconnect-max-backoff-seconds 30 \
  --rtsp-open-timeout-ms 5000 \
  --rtsp-read-timeout-ms 5000 \
  --max-pending-records 10000 \
  --max-pending-bytes 67108864
```

默认写本地 JSONL 投递目标。只有显式提供 `--platform-config` 时才选择平台配置；EasyAIoT 使用独立后台投递线程。窗口与模型提取策略必须一致，不能通过随意调整窗口来声称提高精度。

## 源周期与生命周期

- 初次打开失败、读失败或读超时均关闭解码器。RTSP 结束被视为断流；有限本地录像 EOF 正常结束，解码错误退出，两种情况均不会触发 RTSP 重连。
- 断流立即 `finalize("source_disconnected")`，将已开始事件的 END 交给投递层并写事件审计，再等待重连。END 的时间是最后一次有效观测时间，**不表示确认人员恢复**。
- 每次实际重开尝试把 `source_epoch` 加一，即使该次打开失败；新周期序号从 1 开始，使用全新的姿态关联对象。固定窗口、候选与冷却状态在关闭周期时重置，重新暖机后才能形成新的事件。
- 重试预算是整次运行的总次数，不因短暂读到画面而清零。默认初次打开以外最多尝试 5 次，退避为 1、2、4、8、16 秒；超过上限后饱和。`--reconnect-attempts 0` 禁用重连，零秒退避只用于本地测试。
- `Ctrl-C` 与 `SIGTERM` 设置停止标志；等待退避时立即唤醒，读帧期间在当前有界读操作返回后退出。Python API 也可传入 `threading.Event`。模型推理与第三方解码后端仍须自行满足耗时约束；配置超时不是摄像头实测结果。

RTSP 的 `captured_at` 使用整次运行共用的 UTC/单调时钟锚与接收时间，重连不会重新锚定系统墙钟。它不代表摄像头真实曝光时刻或网络延迟；断连区间没有观测，不能计入“已观察正常时长”。本地文件使用 ffprobe 解出的逐帧显示时间戳，支持 VFR，不用平均 FPS 估算。

## 报告与背压

`run.json` 提供 `timestamp_basis`、`last_source_epoch`、`reconnect` 的打开/读失败次数、实际重试次数、恢复读到画面的次数、实际等待秒数和预算耗尽标记。`status` 为 `running`、`reconnecting`、`completed`、`interrupted`、`failed` 或 `blocked`。持续 RTSP 不会以 EOF 成功完成；正常运维停止记为 `interrupted`。

应用错误信息、源异常与运行报告不包含 RTSP 用户名、密码、查询参数；报告保留去凭据与查询参数后的源地址。第三方 FFmpeg/OpenCV 原生日志与命令行进程参数不属于应用 JSON 报告，部署时仍应限制其访问，不应直接把完整命令或后端调试日志对外发布。

默认最多积压 10000 条记录或 64 MiB 待投递 JSON 正文。该限制不包含已投递历史、索引、WAL、观测日志或证据文件。超过限制时原子拒绝新增批次，立即停止识别，状态为 `blocked`。

`starts`/`ends` 是**生成的事件记录数**；不证明平台已送达。`delivery_status` 记录投递退出前的队列快照；后台关闭期间可能继续排空队列，最终状态可通过只读 `deliver_events --status` 查询。`unsubmitted_records` 与 `outstanding_event_ids` 标记提交没有成功返回的记录；完整事件保存在 `unsubmitted-events.jsonl`，不能删除或把它当成已经送达。提交失败可能发生在写库之后，所以恢复应保留幂等键并检查 outbox，不能生成新事件 ID。被拒绝的 END 也保留在该文件；事件审计闭合不等于交付闭合。维护工具在此文件非空时拒绝隔离证据。

```bash
.venv/bin/python -m campus_safety_ai.apps.deliver_events \
  --status --outbox runtime/fall-camera-demo-001/outbox.sqlite3
```

## 可重复的本地验证

```bash
.venv/bin/python -m pytest \
  tests/test_fall_sources.py tests/test_source_retry.py tests/test_fall_video.py -q
```

测试通过注入 fake capture/时钟执行真实 `OpenCvRtspFrames` 和应用主循环，并保留真实滑动窗口、事件生命周期与 SQLite outbox。覆盖断流立即 END、重连后的第二事件、候选不能跨周期、初次打开失败后恢复、预算耗尽、Ctrl-C/外部停止打断等待、打开失败与异常解码器的资源回收、有限文件不重连、处理错误不重试，以及队列满时保留拒绝的 START/END。它验证状态转换与资源回收，不验证模型精度、摄像头网络特性或连续运行时长。
