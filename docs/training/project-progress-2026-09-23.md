# 校园安全项目推进记录（2026-09-23）

本轮已完成公开素材复核、与持续推理共用采样器的新跌倒候选训练、一次保留人物测试，以及断流重连、积压保护、磁盘保护和拒绝事件恢复。新增候选在这批短片上有改善，但仍会把俯卧撑报为跌倒；默认模型没有替换，不能称为校园部署验收通过。打架模型及既有训练/测试划分未修改。

## 数据与实验边界

- GMDCSA24 已下载的 23 段视频按来源人物划分：Subject1/2 训练 12 段、Subject3 验证 6 段、Subject4 测试 5 段；跌倒事件分别为 4/2/1。原视频、官方 CSV、下载来源与 SHA 均保留。
- 2 组 UP-Fall 仅作为外部诊断素材，未进入训练、选型或测试分数。人物/摄像头信息不足的素材没有冒充独立验证。
- 视频时间采用逐帧解码 PTS；UP-Fall 标注采用原 PNG 文件名时间，预览 MP4 的平均 FPS 不作为真值。
- 7 段 GMD 跌倒均逐图复核了确定运动核心及起始/落地不确定区间。稳定躺卧是“跌倒运动的负类”，并非“人已恢复”；没有任何一段被标为完整恢复成功。
- 相机身份未知，Subject2/3 的房间和机位近似，因此只能说明按来源人物留出，不能说明跨相机独立。

冻结清单为 [fall-web-v1.json](../../datasets/manifests/fall-web-v1.json)，SHA256 `543fa4df12d416955da4462fecad02ac339d71e5c3b4a5a279c43425821bb37b`。完整标注依据见 [复核报告](fall-web-annotation-review.md)，素材来源见 [下载记录](../research/web-hard-negatives-2026-09-23.md)。

## 识别改进与一次保留测试

旧 nano 姿态模型将左侧挂衣误检成人，触发严格单人保护并清空窗口；另一些横躺帧完全漏检真人。实际叠加图与逐点置信度见 [诊断报告](fall-pose-coverage-diagnosis.md)。仅提高置信度到 0.35 后已恢复部分有效动作；随后比较官方 YOLO26s-pose，在训练/验证集上重新提取全部真实帧。模型来自 [Ultralytics 官方模型说明](https://docs.ultralytics.com/tasks/pose/)，权重来源及 SHA 存于诊断报告。

训练和实时推理均使用 `FallPipeline`：1 秒历史窗、16 个因果采样点、0.1 秒步长，未知/换人/断流不跨窗插值。正窗口要求包含至少 0.2 秒视觉确认的跌倒运动，负窗口须完全覆盖于已核对负区间，其余排除。最终训练仅有 **430 个有效负窗口、27 个有效正窗口，来自 12 段视频**；高度重叠窗口不能算新增独立视频。

每个姿态方案只比较预定的线性头和 16 隐藏单元头，训练集归一化、按片段/类别加权训练，以验证集选择 epoch、阈值和事件参数。没有用测试分数回调参数。nano 候选验证检出 1/2 次，small 候选检出 2/2 次，均有 2 次误报，因此在查看测试前冻结 small 方案。

| 数据 | 原方案：检出 / 漏报 / 误报 | 新候选：检出 / 漏报 / 误报 | 新候选可判断时间 |
| --- | --- | --- | --- |
| 验证：6 段，33.42 秒，2 次跌倒 | 0 / 2 / 2 | 2 / 0 / 2 | 54.63% |
| 保留人物测试：5 段，38.42 秒，1 次跌倒 | 0 / 1 / 5 | 1 / 0 / 1 | 62.96% |

测试中新候选的 unknown 时间占比为 **36.06%**，原方案为 **60.51%**；两者另有少量未观测尾段。新候选唯一误报来自 Subject4/ADL05 的俯卧撑，真实跌倒报警延迟约 **0.768 秒**。严格事件匹配与额外 0.5 秒容差的结论相同。报告中的误报率分别为新候选 93.69 次/小时、原方案 468.47 次/小时，分母是这 38.42 秒标注时长；这是短片归一化统计，**不是一小时实测，也不是校园长期误报率估计**。

新旧对比各自使用对应的姿态检测器、检测阈值和原有/新事件策略，是整条方案比较，不能把全部改善归因于分类头。测试只有一个人物、一次跌倒，不能宣称稳定的 100% 召回；查看后的测试集今后属于已知评估集，不能再称全新测试。

所选候选目录：`runtime/training/fall-window-web-small-v1-20260923/`；分类头 SHA256 `a64b7c46d980ff28c99e763098c28bc051bd8c88f65af82c299187a0d38fa6d5`。其中 `protocol.json`、`frozen.json`、`validation.json`、`test.json`、`test-evaluation-started.json`、原始窗口与事件回放均保留。比较选型记录位于 `runtime/training/fall-window-selection-20260923.json`。缓存登记外部 SHA，篡改、未登记旧文件和跨集合身份泄漏会被拒绝；测试启动即留下标记，中途失败也不允许静默再测。

## 连续告警与运行可靠性

- 本地视频入口改用逐帧 PTS，真实 VFR 编码测试确认不会用平均 FPS 代替时间。固定窗、连续确认、迟滞清除、冷却、EOF/异常/中断收口均有测试。
- RTSP 增加有限次数指数退避；断流先生成 END，再重连。每次重连递增 source epoch，重建姿态关联、清空窗口，SIGINT/SIGTERM 可中断等待。仅重试源连接/读取故障，不将模型错误当成断流反复重试。
- 默认 Outbox 待发送上限 10,000 条或 64 MiB，整批原子拒绝超限新记录；重复记录不额外占容量。被拒 START/END 的完整原载荷保存在 `unsubmitted-events.jsonl`，不会报告为已送达。
- 默认预留 256 MiB 磁盘空间、单次运行目录 2 GiB 上限，每 30 帧检查。超限/无法检查时停止新推理并尝试收口，已有数据保留。它不是磁盘空间预分配保证。
- 恢复命令默认只检查，`--apply` 才把拒绝记录原子补回本地 Outbox；验证生成审计、载荷一致、连续事件版本和容量，不发网络。原日志保留，写入带 SHA 的回执；之后仍需单独投递。
- 证据保留命令默认 dry-run；仅处理已登记、未改动、已结束且全部送达并过期的本次生成证据，隔离有恢复回执。隔离本身不释放磁盘，日志和 Outbox 幂等历史没有自动删除，仍需运营归档。

细节见 [重连](../operations/rtsp-recovery.md)、[维护](../operations/runtime-maintenance.md)、[拒绝事件恢复](../operations/recover-events.md)。摄像头断流/恢复、两次事件、积压和低磁盘均经可重复故障注入测试；没有把这些测试当作真实摄像头长稳结果。

实际新候选视频入口另重跑验证片段 Subject3/Fall03：167 个真实帧、47 次观测、1 START/1 END，SQLite 无积压，证据引用保留；分数与冻结缓存回放最大差为 `8.94e-8`。END 原因为 `track_changed`，不是人员恢复。产物在 `runtime/checks/fall-web-small-val-real-20260923/`。

## 导出与硬件边界

所选分类头已导出 `onnx/model.onnx`，155 条真实验证特征及零向量经过 CPU ONNX Runtime 与 PyTorch 对照，动态 batch 1/64 均通过：最大 logits 误差 `5.96e-7`，阈值判定完全一致。导出只包括分类头和归一化，不含 YOLO、姿态特征提取或事件状态机，不能称为整条 RKNN 链路已完成。

已经提供整条本地视频流程的 [板上性能测试脚本与说明](../../deployment/fall-runtime-benchmark.md)。旧 nano/CPU 本机短测为 16.37 FPS、峰值 RSS 385.6 MiB（200 帧），不能达到该 30 FPS 视频实时速度；它不是 RK3588 数据。新候选 small 在另一段 167 帧验证视频上的本机 CPU 参考为 **10.60 FPS、峰值 RSS 365.05 MiB**，仍低于输入实时速度；报告在 `runtime/checks/fall-runtime-small-benchmark-20260923/`。两次输入、负载不同，不得直接作倍数对比。

本机 `127.0.0.1:48080`、`:6000` 均无 TCP 监听，未发送平台告警。没有可用 RK3588 板卡、SDK/运行时或真实 RTSP 地址，所以平台幂等联调、板上速度/内存/精度与长期稳定性仍待实测。公开短片也没有补齐校园嬉闹、拥挤遮挡及真实正常→跌倒→恢复过程。

## 复现入口

以下从项目根目录执行；输出目录必须新建。默认入口继续使用原冻结头，使用新候选需同时指定检测器、分类头和事件配置：

```bash
.venv/bin/python -m campus_safety_ai.apps.fall_video \
  --video /absolute/path/to/video.mp4 \
  --checkpoint runtime/training/fall-window-web-small-v1-20260923/best.pt \
  --detector models/yolo26s-pose.pt \
  --event-config configs/events/fall-web-experimental-v1.toml \
  --camera-id campus-camera-01 --device cpu \
  --output-dir runtime/fall-experimental-new

# 仅检查被拒记录；实际恢复需显式增加 --apply，恢复不是发送。
.venv/bin/python -m campus_safety_ai.apps.recover_events \
  --runtime-dir runtime/fall-experimental-new

# 全量工程回归（可选依赖需按项目环境安装）。
.venv/bin/python -m pytest -q
```

训练入口为 `scripts/extract_fall_web_pose.py`、`scripts/train_fall_windows.py`，导出入口为 `scripts/export_fall_window_head.py`。正式保留测试已完成，原目录拒绝重复测试或覆盖。最终检查与结构化结果汇总见 `docs/audits/2026-09-23/project-progress.json` 及 `project-tests.txt`。

最终全量回归为 **499 项测试、320 个子测试通过，无跳过**；包括对本机真实 benchmark 文件的校验。Ruff 和 `git diff --check` 通过；10 条警告来自 Torch tracing/旧 ONNX exporter 弃用提示。新的 CLI 入口已在本机虚拟环境重新注册并通过帮助命令检查。
