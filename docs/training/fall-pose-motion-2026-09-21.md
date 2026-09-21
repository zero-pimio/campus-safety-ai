# 跌倒识别：运动特征与连续帧训练（2026-09-21）

本轮实际训练了三个阶段的姿态运动分类器，最终连续帧版本减少了已发现的走路误报：同一段 ADL-10 验证录像的 285 个可判断时点，初版姿态模型有 80 个阳性，新版为 0。Fall-06 仍能识别跌倒，但早期分数有抖动。这是两段开发验证视频上的改进，尚不能据此宣称校园实时识别达标。

## 实际改变

此前 RGB 基线在“首帧重复”对照中仍能分开全部验证标签，存在静态线索风险。本版分类器不读取 RGB 外观，只读取人体关键点的运动；YOLO26n-pose 仍使用官方预训练权重，没有重新训练姿态骨干。

运动描述符包含 32 维：髋部水平/垂直位移速率、躯干旋转、框宽高比变化、相对核心关节形态变化及肩/髋宽变化的统计。坐标按人体框尺度归一化，仅使用原时间序列中相邻且有效的运动对。任意静止姿态重复后特征均为零；整体平移和等比例缩放不改变描述符。相机运动、关键点抖动及透视变化仍可能影响结果，不能称为完全消除了场景依赖。

轨迹关联不确定、身份中断、关键点不足时保留 unknown，不能当作正常。连续取样不会跨越隐藏在两个抽样点之间的轨迹中断。负类的含义是“未观察到跌倒转变”，不是“人员安全”；静止躺地不属于这版动作分类器的独立检测目标。

## 数据与训练

沿用 [URFD 官方数据](https://fenix.ur.edu.pl/mkepski/ds/uf.html) 和固定 42 train / 14 val / 14 test 划分。同一序列不跨集合，但没有跨人物或跨校园独立评估。许可为 CC BY-NC-SA 4.0。

在原每段 16 帧基础上，固定补齐 8 个训练序列的全部 **1302 张连续 RGB 原帧**：ADL-01/02/04/06、Fall-01/04/05/09。另补齐两段验证录像 ADL-10 的 300 帧与 Fall-06 的 100 帧。原帧 CRC/SHA、同步时间、原划分和旧 16 帧对应关系均经过检查。

| 最终训练输入 | 有效行 | 无法判断行 | 独立原视频 |
| --- | ---: | ---: | --- |
| 原完整序列的 16 帧描述符 | 42 | 0 | 42 |
| 原稀疏采样的日常/转变前短片段 | 282 | 40 | 原 42 段的子集 |
| 8 段连续录像的因果前缀 | 334 | 12 | 原 42 段中的 8 段 |

合计 **658 行有效输入、42 个独立训练组**，不是 658 段独立视频。无效行保留审计记录并排除损失计算，不填零伪装成负例。每行等权后再做二分类类别平衡，因此密集视频对训练的影响较大。

密集前缀从观察到第 16 帧开始，每 3 帧取一个终点，在已观察历史内取 16 个等分中心帧。ADL 按原序列类别标为负。原 fall 序列采用官方深度姿态标注的保守时间代理：最后一个实际输入帧早于首个姿态 0 至少 200 ms 才作为负例；达到首个姿态 1 后至少 200 ms 才作为正例；56 个过渡区前缀排除。官方姿态 0 段均为 30 帧，不能把这个标注当作人工精确标注的 RGB 跌倒起点，也没有把 ADL 的躺地姿态改标成跌倒。

标准化仅由原 42 个 train 描述符拟合。每阶段比较线性层与 16 单元隐藏层，AdamW lr=0.01、weight_decay=0.01，最多 300 epoch、验证损失 40 轮不改善早停；增加单个零运动负例约束，权重 0.25。按原 14 段 val 的损失选 epoch，再按验证 balanced accuracy/F1/loss 选候选，阈值必须通过零运动约束。连续诊断参与了后续训练方案设计，因此属于开发证据；本轮没有根据 test 分数调参。

| 最终阶段候选 | 参数 | 实际/选中 epoch | 验证 loss | 所选阈值 | 参数 L2 变化 |
| --- | ---: | --- | ---: | ---: | ---: |
| 线性 | 66 | 286 / 246 | 0.12758464 | 0.1913495 | 2.854404 |
| 16 单元隐藏层，最终选中 | 562 | 225 / 185 | 0.09918131 | 0.5 | 6.516090 |

两个候选的全部分类头参数都发生了实际变化。独立审计确认标准化与原 42 train 的均值/总体标准差逐位一致。

## 结果及失败对照

| 阶段 | 原 14 段 val 正确数 | ADL-10 阳性时点 / 可判断时点 | 结论 |
| --- | ---: | ---: | --- |
| 仅整段姿态运动训练 | 14/14 | 80/285 | 整段高分掩盖了开头误报 |
| 增加稀疏负前缀 | 14/14 | 80/285 | 没有改善该误报，不采用 |
| 增加实际连续帧前缀 | 14/14 | 0/285 | 已改善这段走路录像 |

最终模型在已有 14 段 test 上整段分类也是 14/14：TP=6、TN=8、FP=0、FN=0，loss=0.03417715；14 段均可判断。**这批 test 已被先前模型使用，本次是复用划分的对比，不是新独立留出集。** 最终候选与源码冻结后只执行了本候选的一次 test 阶段。

两段连续检查均只使用当前及过去输入，前 15 帧为预热；预热后没有 unknown。ADL-10 新版最大分数为 0.448654，仍接近 0.5 阈值，不能从这一段推出稳定的通用误报率。Fall-06 新版 85 个可判断时点中 77 个阳性，634 ms 首次阳性，早期正负抖动，934 ms 至片尾持续阳性。这里的时间是素材内分数观测时间，**不是报警延迟**；阳性时点比例也不是事件召回率。系统尚未形成固定滑动窗口、事件结束/重置和稳定告警的实测闭环。

静止重复对照有 28 个，其中 18 可判断、10 unknown；18 个有效对照均不报跌倒，但它们只有一个唯一零向量。这是合成约束检查，不是 18 个新的真实负例。

![相同连续录像的实际预测分数对比](/Users/wanwanzhu/yolo/runtime/checks/urfd-pose-dense-v1/recognition-comparison.png)

## 已保存产物与复现

- [最终分类头权重](/Users/wanwanzhu/yolo/runtime/training/fall-pose-dense-v1-20260921/best.pt)
- [ONNX 分类头](/Users/wanwanzhu/yolo/runtime/training/fall-pose-dense-v1-20260921/model.onnx)
- [冻结方案](/Users/wanwanzhu/yolo/runtime/training/fall-pose-dense-v1-20260921/protocol.json)、[冻结指纹](/Users/wanwanzhu/yolo/runtime/training/fall-pose-dense-v1-20260921/frozen.json)、[复用 test 结果](/Users/wanwanzhu/yolo/runtime/training/fall-pose-dense-v1-20260921/test.json)
- [连续检查](/Users/wanwanzhu/yolo/runtime/checks/urfd-pose-dense-v1/summary.json)、[训练派生数据记录](/Users/wanwanzhu/yolo/runtime/training/fall-pose-dense-v1-20260921/derived-prefixes.json)
- [走路识别视频，300 帧](/Users/wanwanzhu/yolo/runtime/checks/urfd-pose-dense-v1/videos/adl-10-new-model.mp4)、[跌倒识别视频，100 帧](/Users/wanwanzhu/yolo/runtime/checks/urfd-pose-dense-v1/videos/fall-06-new-model.mp4)：完整连续原帧、原速播放，叠加上述逐时点真实预测。绿色表示未见跌倒动作，红色表示检测到跌倒动作，黄色表示预热或无法判断；不把历史分数平滑成更好看的结果。视频每帧 PTS 与官方毫秒时间相符，详见 [视频校验记录](/Users/wanwanzhu/yolo/runtime/checks/urfd-pose-dense-v1/videos/verification.json)。渲染脚本为 `scripts/render_urfd_pose_demo.py`。

ONNX 只包含分类头和标准化，输入为未标准化的 `[1,32]` 运动描述符，输出顺序 `no_fall_transition, fall_transition`；需 softmax 后应用 0.5 阈值。它不包含 YOLO 或特征提取，不能直接输入视频。14 段 val 的 PyTorch/ONNX 最大 logits 差为 `9.536743e-7`，已独立重算一致；未完成 RKNN、板卡或校园视频验收。

工程检查：现有完整回归 247 项测试、224 个子测试通过；随后新增的密集采样边界测试 4 项、21 个子测试通过，覆盖训练组隔离、实际输入帧的标签边界、未来帧隔离及隐藏跟踪中断。相关新源码和测试 Ruff 通过，`git diff --check` 通过。原打架权重、打架划分和 RGB 跌倒权重 SHA 均与本轮开始一致。

```bash
cd /Users/wanwanzhu/yolo
# 数据、预训练姿态权重及初版稀疏姿态缓存已在本机保存。
.venv/bin/python scripts/prepare_urfd_dense_train.py
.venv/bin/python scripts/extract_urfd_dense_train_pose.py
# 每次实验和检查必须使用新的空输出目录。
.venv/bin/python scripts/train_urfd_pose_dense.py \
  --output runtime/training/fall-pose-dense-new
.venv/bin/python scripts/check_urfd_pose_continuous.py \
  --phase check --device cpu \
  --checkpoint runtime/training/fall-pose-dense-new/best.pt \
  --output runtime/checks/fall-pose-dense-new
# 选定并冻结后，才执行一次复用 test；不能把复用结果当成新留出证据。
.venv/bin/python scripts/train_urfd_pose_motion.py --phase test \
  --pose-cache runtime/training/urfd-pose-cache-v2 \
  --output runtime/training/fall-pose-dense-new
```

```text
manifest  5acbbbfa11af84b0b167f23b2e31855a4c127802078fc764586f81de8e4722f6
best.pt   3ae4b6237f16f6c377f118fc1c0b69e9d069765e8d61f13b351e5403d1f1b368
ONNX      532f07a26b5405703c202b0945a7241d1f8155180c66399bee6bd42e2785450c
YOLO pose eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9
```

下一步决定可用性的证据是：增加正常坐下、蹲下、弯腰、主动躺卧等连续负例；冻结人员/摄像头独立的数据划分；标注真实事件起止并统计事件漏报、每小时误报和 unknown 覆盖率。现有框架和权重只能支持继续试验，不能保证真实校园效果。
