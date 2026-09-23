# GMD 验证集姿态覆盖诊断（2026-09-23）

已定位两种独立失败：**YOLO26n 把左侧挂衣误识别成人，导致真实演员的高质量姿态被单人保护规则拒绝；部分横躺帧中真人漏检，模型反而只保留挂衣框。** 因此缓存中的 `associated` 不一定意味着正确人体得到覆盖，单看 unknown 降低会产生误导。

本子任务仅查看 `val` 的 GMD Subject3/Fall03、Fall04，没有运行或查看 Subject4 test 分数，没有修改标签、原 pose 缓存、生产代码或默认模型。完整结果与逐帧叠加图保存于 `runtime/checks/fall-pose-coverage-diagnosis/`。

## 可复现失败

使用实际 `SinglePersonPose.predict`、原 nano 权重、原 `conf=0.2/imgsz=640`、MPS，在已确认下落核心内的 Fall03 第 **51 帧（0 基）/ 1.7121 秒**重放：

```bash
.venv/bin/python runtime/checks/fall-pose-coverage-diagnosis/diagnose.py --assert-core
```

已运行并得到预期失败：

```text
{"sample": "gmdcsa24-subject-3-fall-03", "frame": 51, "t": 1.7121, "valid": false, "reason": "multiple_people"}
AssertionError: Confirmed fall-motion core frame is rejected by actual SinglePersonPose
```

这条命令是诊断保留的失败检查，不应列为“通过测试”。读取原缓存也重现 Fall03 为 103 associated / 64 multiple_people（167 帧）；Fall04 为 99 associated / 23 multiple_people / 41 missing_person（163 帧）。关联状态只反映检测器和保护逻辑，不是真实人体标注。

先提出并区分了三个假设：背景误框、同一演员重复框、横躺导致真人/关键点质量不足。原始框和画面直接支持背景误框；抽查没有看到镜像人物或重复演员框。横躺真人漏检同样发生，但在已经正确检测到演员的下落核心帧，躯干关键点质量很高，不能把主要问题归因于核心关键点缺失。

## 逐帧画面证据

每份 `raw-predictions.json` 保存每个框的原像素 `xyxy`、box confidence 和全部 **17×(x,y,confidence)** 关键点。每张原尺寸 overlay 右侧列出 17 个关键点置信度，并标记左右肩、左右髋四个 CORE 点。另有两张 `all-23-frames.jpg` 联系图，已实际查看两模型全部 23 个抽样帧；以下关键帧另查看原尺寸叠加图。

| 视频/帧（0 基） | PTS / 秒 | nano conf=0.2 的实际结果 | 视觉结论 |
|---|---:|---|---|
| Fall03 / 51 | 1.712100 | 演员框 0.9151；左侧挂衣框 0.3281 | 场景只有一名演员，误框导致 `multiple_people` |
| Fall03 / 59 | 1.984200 | 演员框 0.9112；挂衣框 0.2243 | 下落核心仍被背景假人打断 |
| Fall04 / 54 | 1.808467 | 演员框 0.8702；挂衣框 0.3376 | 真人坐低时躯干仍可靠，但人数保护拒绝 |
| Fall04 / 70 | 2.336333 | **只有挂衣框** 0.2317 | 真人横躺未被检测；单框不是正确覆盖 |
| Fall04 / 78 | 2.608200 | 0 框 | 真人明确可见但漏检，保留 unknown |
| Fall04 / 100 | 3.344400 | **只有挂衣框** 0.2356 | 假人也能获得四个中等 confidence 的“躯干点” |

Fall03/51 的演员框约为 `[580.8,176.7,823.5,609.9]`，衣物框约为 `[0.0,0.0,181.0,543.4]`。演员四个躯干点 confidence 为 **0.9644、0.9927、0.9926、0.9974**；挂衣伪躯干点也有 **0.4693、0.3197、0.6386、0.5237**。仅要求四个点高于 0.25 不能排除挂衣假人。

Fall04/54 的演员四个躯干点为 **0.9977、0.9650、0.9781、0.9276**。这些帧并不需要放松关键点质量条件；主要损失来自 `len(boxes) != 1` 时整个观测失效并清空关联锚点。

横躺失败帧中演员位于图像下方，部分手臂接近或超出下边界；躯干仍可见。图像证据支持“姿态/视角下的检测覆盖不足”，不能据此唯一断言是训练分布、截断、纹理还是具体网络层造成。没有将漏检补成低风险，也没有用衣物框替代真人轨迹。

关键本地叠加图：

- `runtime/checks/fall-pose-coverage-diagnosis/yolo26n-pose/gmdcsa24-subject-3-fall-03-f051.jpg`
- `runtime/checks/fall-pose-coverage-diagnosis/yolo26n-pose/gmdcsa24-subject-3-fall-04-f070.jpg`
- `runtime/checks/fall-pose-coverage-diagnosis/yolo26n-pose/gmdcsa24-subject-3-fall-04-f078.jpg`
- `runtime/checks/fall-pose-coverage-diagnosis/yolo26s-pose/gmdcsa24-subject-3-fall-04-f070.jpg`
- `runtime/checks/fall-pose-coverage-diagnosis/yolo26s-pose/gmdcsa24-subject-3-fall-04-f100.jpg`

## 单变量阈值与模型对照

固定相同 23 个验证帧：Fall03 11 帧、Fall04 12 帧，覆盖下落核心及前后阶段。每帧分别运行 `conf=0.2/0.35/0.5`，其他参数均为 `imgsz=640/device=mps`，未运行行为分类头。总计两个模型各 69 次原帧预测。这是有目的的失败诊断抽样，**不是全量召回率或独立性能基准**。

下表按看图核对后的框内容计数；“真人单框”不等于连续窗口已经可评分，更不等于事件检测成功。

| 模型 / threshold | 真人单框 | 真人+挂衣双框 | 只有挂衣 | 无框 | 合计 |
|---|---:|---:|---:|---:|---:|
| nano / 0.20 | 11 | 9 | 2 | 1 | 23 |
| nano / 0.35 | 20 | 0 | 0 | 3 | 23 |
| nano / 0.50 | 20 | 0 | 0 | 3 | 23 |
| small / 0.20 | 22 | 0 | 0 | 1 | 23 |
| small / 0.35 | 22 | 0 | 0 | 1 | 23 |
| small / 0.50 | 21 | 0 | 0 | 2 | 23 |

提高 nano threshold 在这些抽样帧清除了挂衣框，保留了确认下落核心的真人框；但原本只有衣物假框的帧现在诚实地变成 missing，不能将这种 unknown 增加解释为能力退化，也不能把此前的假人单框当成功。

Small 对 Fall04/70 得到真人框 confidence **0.7411**，四个核心点 **0.9695、0.9705、0.9878、0.9903**；对 Fall04/100 得到真人框 **0.4094**，核心点 **0.9806、0.9687、0.9743、0.9659**。但 Fall04/78 在 small/0.2 下仍然没有框。Small/0.5 又把第 100 帧丢掉，证明一味提高 confidence 并不能解决覆盖。

[Ultralytics 官方姿态文档](https://docs.ultralytics.com/tasks/pose/)确认 YOLO26n/s/m 等姿态模型、17 点顺序及 `result.keypoints.data` 输出。本轮没有把官方 COCO 指标当成本项目精度或 RK3588 性能。

新下载仅 `models/yolo26s-pose.pt`，没有替换 nano：

- 官方原始 URL：`https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-pose.pt`
- 文件大小：24,151,790 字节。
- SHA-256：`a083adb42303728ae14c4bd6bd56d80da46f82fb2564dbd6f31dcc92ea321646`。
- 原 nano SHA-256：`eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9`。
- 本机 Ultralytics：8.4.128；下载凭据：`runtime/checks/fall-pose-coverage-diagnosis/model-download.json`。

## 保守改进方向

1. **先验证 nano/0.35 的完整 train/val 连续覆盖。** 使用新 cache identity、新缓存目录，从头提取真实时间戳序列；保留原 nano/0.2 缓存用于对比。只有完整连续窗口与事件级指标支持时才冻结新检测参数，不凭这 23 帧直接发布。
2. **同时统计真实演员覆盖、错误目标关联、轨迹重置和 unknown。** 检测器单框率不能代替正确人体覆盖；仅降低 `multiple_people` 也不能代替提高识别能力。真实多人仍须按现有边界返回 unknown，不能简单忽略第二框或总取最大框。
3. **保持关键点质量与源周期保护。** 当前确认核心的躯干点可靠，放宽质量阈值既无必要，也可能接纳挂衣骨架。不要针对这一个房间硬编码左侧屏蔽区域；如果以后使用业务区域，应由可审阅的相机配置定义并验证人员进入区域边界的行为。
4. **Small 作为覆盖候选保留。** 如果 nano/0.35 的全量验证仍有运动核心或关键后续状态缺失，再在新的 train/val 缓存中评估 small；模型身份改变意味着重新提取、训练、冻结和设备测量，不能把 small 直接塞给与 nano provenance 绑定的旧行为头。
5. **恢复仍不能由姿态丢失推断。** 横躺无框后保持 unknown/中断语义，不宣称人员已站起。模型变化也没有补出本批素材缺失的“跌倒→站立恢复”真值。

本子任务按要求止于证据诊断和建议，未宣称生产缺陷已修复，也未修改被诊断保护逻辑。最小失败检查保留为基线；若后续实现落地，应让新协议下的同帧及完整事件回放通过，并加入真实多人、衣物假人、横躺漏检的回归证据。

复现 bounded 对照：

```bash
.venv/bin/python runtime/checks/fall-pose-coverage-diagnosis/diagnose.py
.venv/bin/python runtime/checks/fall-pose-coverage-diagnosis/diagnose.py --model models/yolo26s-pose.pt
```
