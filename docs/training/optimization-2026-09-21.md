# 2026-09-21 训练优化与可行性检查

主体框架已成型，可沿用现有数据、训练、推理、事件状态机及持久化投递分层继续迭代。本轮完成两组真实微调、完整验证/测试、候选 ONNX 导出和本地视频告警冒烟验证；训练侧补齐可追溯性、损失统计、验证阈值、低内存运行及解码优化。

**本轮不替换原模型。** 验证集选出的候选在 92 段 test 上 F1 为 `0.796296`，低于原基线 `0.803738`；漏报仍为 8，误报由 13 增至 14。验证集提高未转化成测试集提高。公开数据结果与本地链路可用性，尚不能证明校园实时告警已达到部署要求。

## 1. 实验依据与数据审计

项目根目录：`/Users/wanwanzhu/yolo`。本轮原始记录统一保存在：

`/Users/wanwanzhu/yolo/runtime/training/optimization-20260921T024422Z/`

| 记录 | 用途 |
| --- | --- |
| [experiment-plan.json](/Users/wanwanzhu/yolo/runtime/training/optimization-20260921T024422Z/experiment-plan.json) | 预先记录候选学习率、选择规则、资源调整和测试使用边界 |
| [data-audit.json](/Users/wanwanzhu/yolo/runtime/training/optimization-20260921T024422Z/data-audit.json) | manifest 数量、分组、路径和文件存在性审计 |
| [environment.json](/Users/wanwanzhu/yolo/runtime/training/optimization-20260921T024422Z/environment.json) | 本机 Python、PyTorch、OpenCV、ONNX 等版本 |
| [baseline-val-calibrated.json](/Users/wanwanzhu/yolo/runtime/training/optimization-20260921T024422Z/baseline-val-calibrated.json) | 完整验证集预测、分数据集指标、所选视频阈值 |
| [baseline-export-verification.json](/Users/wanwanzhu/yolo/runtime/training/optimization-20260921T024422Z/baseline-export-verification.json) | 基线 PyTorch/ONNX 数值比较及本机纯推理耗时 |
| [selection.json](/Users/wanwanzhu/yolo/runtime/training/optimization-20260921T024422Z/selection.json)、[freeze.json](/Users/wanwanzhu/yolo/runtime/training/optimization-20260921T024422Z/selected/freeze.json) | 根据 val 选模，在 test 前冻结模型和阈值 |
| [comparison.json](/Users/wanwanzhu/yolo/runtime/training/optimization-20260921T024422Z/comparison.json)、[test.json](/Users/wanwanzhu/yolo/runtime/training/optimization-20260921T024422Z/selected/test.json) | 最终测试、逐视频预测、分数据集指标及不替换基线的决定 |
| [export-verification.json](/Users/wanwanzhu/yolo/runtime/training/optimization-20260921T024422Z/selected/export-verification.json) | 本轮候选导出数值对齐和 CPU 耗时 |

当前 manifest 共 590 段视频；标签 `0=non_fight`、`1=fight`。AIRTLab 同一表演的 cam1/cam2 使用同一 group，因此片段数大于独立 group 数。

| 数据集 / 标签 | train | val | test |
| --- | ---: | ---: | ---: |
| SCFD / non_fight | 105 | 22 | 23 |
| SCFD / fight | 105 | 22 | 23 |
| AIRTLab / non_fight | 84 | 18 | 18 |
| AIRTLab / fight | 118 | 24 | 28 |
| 视频合计 | **412** | **86** | **92** |
| 独立 group 合计 | **311** | **65** | **69** |

审计未发现缺失或空文件、跨 split 路径、跨 split group；当前 manifest 中也没有重复视频路径。这是路径和已有分组层面的检查，尚未进行视频内容去重，也不等同于跨演员、场景、摄像头或日期的校园留出验证。

复现标识：

- manifest SHA256：`9072e283bf064ab8fdf09d9f129992f6bf780a019fcbb929e64028199f652fc5`
- 初始 checkpoint SHA256：`da569fb56931ecf1fdbabff11b6b3fa902661b6c2d9c62b65b6f8d88032041af`
- 环境：macOS 15.0.1 arm64、Python 3.11.16、PyTorch 2.13.0、TorchVision 0.28.0、NumPy 1.26.4、OpenCV 4.11.0.86、ONNX 1.22.0、ONNX Runtime 1.29.0。

manifest SHA 标识清单内容，不代表已经逐个记录视频文件的内容哈希。既有 test 报告早于本轮实验；本轮承诺是模型和阈值冻结后只评估一次所选候选，不能把已有 test 描述为首次使用的新测试集。

## 2. 已落实的训练与评估改进

| 改进 | 作用及边界 |
| --- | --- |
| 微调前评估 epoch 0 并保存初始权重 | 新训练没有超过初始验证成绩时，仍能保留原始候选；重置优化器，不冒充中断续训 |
| 全量 train/val、固定 split、只用 val 选模 | 本轮不使用小样本上限，不使用 test 选择学习率、epoch 或阈值 |
| 按正确分母累计加权训练损失 | 类别权重仅由 train 的 189 个负类和 223 个正类计算；避免加权批均值按样本数直接汇总造成统计偏差 |
| 验证使用未加权交叉熵 | 与独立 evaluate 的 loss 口径一致，支持 F1 平局时比较验证 loss |
| 同一视频共享随机裁剪位置和翻转 | 避免逐帧独立裁剪产生人为画面跳动；验证保持确定性中心裁剪 |
| 有界顺序解码 | 相邻采样帧距离不超过 128 时顺序读取，较大间隔仍 seek；只缓存上一采样帧，避免缓存整段视频 |
| 小批量与梯度累积 | `batch_size=2`、`accumulation_steps=4`，名义有效 batch 为 8；最后不足一组按实际损失分母归一化 |
| 冻结 BatchNorm 运行统计 | 降低小批量微调对已有运行均值/方差的扰动，BN 仿射参数仍参与学习；该策略与原 batch 8 训练并非完全等价 |
| 训练与评估保留溯源 | 记录 seed、参数、实际样本、初始权重和 manifest SHA；评估保留每个视频的标签、得分、预测、dataset、group 和误分类 |
| 参数与输出保护 | 拒绝无效尺寸、批量、样本上限；训练目录必须为空，评估报告拒绝覆盖；partial 报告不可用于选择或冻结阈值 |

本轮预设两个学习率 `1e-5`、`5e-5`，各最多 6 个 epoch，seed 为 `20260921`，连续 3 个 epoch 无改进时早停。训练器按验证 F1 选 epoch，F1 相同则比较未加权验证 loss；学习率调度依据验证 F1。具体运行参数以各候选的 `run.json` 为准。

初始 batch 8 尝试在 8 GiB M1 上观察到较高内存压力，交换空间约 10003 MiB；在完成任何 epoch 前中断了本轮自己的训练进程，原目录保留。后续两个候选统一改用 microbatch 2、累积 4、冻结 BN，采用相同的数据、初始化和选模规则。第二组使用本轮优化后的解码实现，源码哈希分别保留；此次资源和速度调整不构成模型精度提升证据。

两组微调均按预设规则早停，共完成 7 个 epoch：

| 运行目录 | 学习率 | 完成 epoch | 最佳 epoch | 最佳 val F1 @0.5 | 最佳 val loss | 平均每 epoch 耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `compact-lr-1e-5` | 0.00001 | 3 | 0 | 0.840000 | 0.591288 | 316.80 秒 |
| `compact-lr-5e-5` | 0.00005 | 4 | 1 | 0.854167 | 0.669354 | 96.84 秒 |

第一组未超过初始化成绩，保留 epoch 0；第二组按预设 F1 优先规则选中 epoch 1，尽管其验证 loss 更高。第二组平均 epoch 耗时较短，但两轮并非严格隔离的性能对照，不能把全部差异归因于解码。

独立的 [解码对照记录](/Users/wanwanzhu/yolo/runtime/training/optimization-20260921T024422Z/decode_benchmark-production.json) 覆盖 train/val、两个数据集和两个类别的 8 个样本，输出张量逐位一致；同一次对照中旧实现 6.54 秒、新实现 1.81 秒，约 3.62 倍。此为少量样本单轮测量，不代表所有视频或机器上的固定加速比。固定为单线程的解码尝试更慢，未采用。

## 3. 完整评估结果与冻结方法

### 视频分类结果

| 模型 / split | 样本数 | 阈值 | Accuracy | Balanced accuracy | Precision | Recall | F1 | FP / FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 原始基线 / 本轮 val | 86 | 0.5 | 0.813953 | 0.806522 | 0.777778 | 0.913043 | 0.840000 | 12 / 4 |
| 原始基线 / 本轮 val 选阈值 | 86 | 0.5414354205 | 0.837209 | 0.831522 | 0.807692 | 0.913043 | **0.857143** | 10 / 4 |
| 原始基线 / 历史 test | 92 | 0.5 | 0.771739 | 0.763032 | 0.767857 | 0.843137 | **0.803738** | 13 / 8 |
| 本轮候选 / val | 86 | 0.5 | 0.837209 | 0.833152 | 0.820000 | 0.891304 | 0.854167 | 9 / 5 |
| 本轮候选 / val 选阈值 | 86 | 0.6010497212 | 0.848837 | 0.845652 | 0.836735 | 0.891304 | 0.863158 | 8 / 5 |
| 本轮候选 / test（主结果） | 92 | 0.6010497212 | 0.760870 | 0.750837 | 0.754386 | 0.843137 | **0.796296** | 14 / 8 |
| 本轮候选 / test（固定阈值辅助结果） | 92 | 0.5 | 0.760870 | 0.748446 | 0.745763 | 0.862745 | 0.800000 | 15 / 7 |

历史 test 数据来自 [既有基线报告](/Users/wanwanzhu/yolo/runtime/training/fight-tsn-v1/test-evaluation.json)，不是本轮重新运行的候选测试。验证集选阈值带来的上升发生在用于选择该阈值的数据上，应由冻结后的 test 检查其表现。

基线采用验证阈值后，AIRTLab 的 42 段 val 视频 F1 为 `0.938776`，FP/FN 为 `2/1`；SCFD 的 44 段 val 视频 F1 为 `0.775510`，FP/FN 为 `8/3`。候选最终 test 中，AIRTLab 的 46 段 F1 为 `0.892857`、FP/FN 为 `3/3`；SCFD 的 46 段 F1 为 `0.692308`、FP/FN 为 `11/5`。误分类存在明显的数据集差异，不能只报告合并指标。

主结果使用事先冻结的验证阈值；固定 0.5 的辅助结果也在测试前声明，从同一次 test 推理保存的得分计算，没有重复运行 test 或根据 test 重选阈值。主结果相对基线 F1 差为 `-0.007442`；按 69 个 group 配对 bootstrap 5000 次所得 95% 百分位区间为 `[-0.043259, 0.025637]`，不支持稳定提升的结论。该区间仅描述已使用过的公开测试集，不能证明校园场景泛化。

### 本轮固定选择顺序

1. 候选训练只读取 train 进行梯度更新，val 用于保存 epoch；初始基线也是候选。
2. 比较各候选完整 val 上默认阈值 0.5 的 F1，相同则比较未加权 val loss，选定 checkpoint。
3. 仅在该 checkpoint 的完整 val 上选择视频阈值：优先 balanced accuracy，其次 F1、接近 0.5，最后取更高阈值。
4. 冻结 checkpoint、manifest、验证报告和阈值。使用 `--threshold-report` 评估 test，评估器校验 checkpoint 与 manifest SHA，拒绝 partial 或 test 来源报告。
5. test 输出以后不根据其成绩更换本轮模型或阈值。新增实验应单独记录，不把同一 test 反复优化后的成绩当作全新留出结果。

预测规则为 `fight_score >= threshold`。这一阈值属于视频分类，不能自动覆盖事件状态机的 `start_score`、`end_score` 或持续时间条件。

候选取自 `compact-lr-5e-5/best.pt`，完整复制到 `selected/best.pt`，SHA256 为 `9380baeae11804c5cd55ef1c16c130e6e1139d85236cd09b7ca5618fe9938e79`。2026-09-21 03:24:23 UTC 冻结后完成一次 test 推理，混淆矩阵为 TN=27、FP=14、FN=8、TP=43。`selected` 仅表示按 val 选出的实验候选；默认模型、原权重和事件策略均未替换。

## 4. 复现训练与评估命令

以下命令可在独立的新实验目录复现低内存微调流程，两组训练串行执行。本轮已实际完成相应实验；后续应优先补充困难负样本和校园事件验证数据，无需机械重复相同参数。

```bash
cd /Users/wanwanzhu/yolo
FOLLOWUP_ROOT="/Users/wanwanzhu/yolo/runtime/training/followup-$(date -u +%Y%m%dT%H%M%SZ)"
for RATE in 1e-5 5e-5; do
  .venv/bin/campus-safety-train-fight \
    --project-root /Users/wanwanzhu/yolo \
    --manifest datasets/manifests/fight-v1.csv \
    --init-checkpoint runtime/training/fight-tsn-v1/best.pt \
    --output-dir "$FOLLOWUP_ROOT/lr-$RATE" \
    --epochs 6 --patience 3 --learning-rate "$RATE" \
    --batch-size 2 --accumulation-steps 4 --freeze-batch-norm \
    --frames 8 --image-size 224 --workers 0 --device mps --seed 20260921 \
    || break
done
```

训练期间查看各候选的 `metrics.jsonl` 和 `run.json`，确认状态与完成 epoch；`best_epoch=0` 表示新微调尚未超过原始权重。`--init-checkpoint` 会重置优化器，不恢复之前的优化器状态。调整参数或再次运行时，应使用新的目录。

选定最终 checkpoint 后，设置其实际绝对路径和一个尚未使用的报告目录，再执行以下冻结流程。不要在比较候选时逐个运行 test。

```bash
# 执行前设置 SELECTED_CHECKPOINT 为最终 checkpoint 的绝对路径。
: "${SELECTED_CHECKPOINT:?请先根据完整 val 结果设置最终 checkpoint 路径}"
EVALUATION_ROOT="/Users/wanwanzhu/yolo/runtime/training/frozen-eval-$(date -u +%Y%m%dT%H%M%SZ)"

/Users/wanwanzhu/yolo/.venv/bin/campus-safety-evaluate-fight \
  --project-root /Users/wanwanzhu/yolo \
  --checkpoint "$SELECTED_CHECKPOINT" \
  --split val --select-threshold --batch-size 2 --device mps \
  --output "$EVALUATION_ROOT/selected-val.json"

/Users/wanwanzhu/yolo/.venv/bin/campus-safety-evaluate-fight \
  --project-root /Users/wanwanzhu/yolo \
  --checkpoint "$SELECTED_CHECKPOINT" \
  --split test --threshold-report "$EVALUATION_ROOT/selected-val.json" \
  --batch-size 2 --device mps --output "$EVALUATION_ROOT/selected-test.json"
```

评估器默认 manifest 为项目下的 `datasets/manifests/fight-v1.csv`。不得在最终评估加入 `--max-samples`；报告中的 `partial` 应为 `false`，val/test 数量分别应为 86/92。如果前一步验证失败，不应继续执行测试命令。

## 5. 部署前检查与适用边界

已测基线 ONNX 大小为 3,726,837 字节，固定输入 `[1,24,224,224]`。4 个验证视频覆盖 SCFD/AIRTLab 两类，PyTorch 与 ONNX 最大 logits 绝对误差为 `5.4836e-6`。这是少量样本数值对齐证据，尚未替代完整候选或板端验收。

本轮候选也已独立导出，ONNX checker 通过；同样覆盖两个数据集、两个类别的 4 个真实验证输入，最大 logits 绝对误差 `8.5756e-6`，低于设定的 `1e-4`。导出与实验配置保存在 `selected`，供复核使用，不改变默认模型。

| 耗时记录 | 已确认结果 | 可解释范围 |
| --- | --- | --- |
| 本轮基线 MPS 完整 val | 86 段约 62.67 秒，约 1.37 段/秒 | 包括解码、预处理、传输、模型与 loss，不是纯模型延迟 |
| 基线 ONNX CPU 单线程 | 预热 5 次、重复 30 次；中位数 45.61 ms、P95 53.49 ms | 单个固定验证输入的 `session.run`；不含解码、预处理、网络、证据、事件投递，不代表 RKNN 或在线吞吐 |
| 候选 ONNX CPU 单线程 | 预热 5 次、重复 30 次；中位数 43.56 ms、P95 44.80 ms | 与上行相同范围；单轮本机测量，不据此认定架构性能提升 |

本地拼接视频 `fight-then-normal.mp4` 通过候选 ONNX 跑出 4 个窗口，生成同一 eventId 的 START revision 1 和 END revision 2；SQLite 完整性检查通过、队列无待投递记录。该冒烟使用本地 JSONL 接收且关闭证据生成，证明的是本地推理—事件—outbox 链路，不是实际平台接收、证据上传或事件准确率。记录见 [smoke-check.json](/Users/wanwanzhu/yolo/runtime/training/optimization-20260921T024422Z/selected/smoke-check.json)。

工程验证：`.venv` 完整单元测试 **129 项通过**；系统 Python 核心测试 127 项中跳过 11 项依赖相关测试，其余通过；Ruff 及 `git diff --check` 通过。本轮同时修复 outbox 的事件间重试隔离、同事件顺序、发送者互斥、迁移备份和只读状态查询，未迁移或清理既有运行数据。

部署检查按以下顺序推进：

1. **每个新候选独立导出验证。** 本轮已完成。后续使用独立输出路径执行 `campus-safety-export-fight-onnx --checkpoint <最终checkpoint> --output <新目录/fight-tsn.onnx>`，保留相邻 JSON 元数据并重新检查输入、类别顺序和数值差异。
2. **验证实际短窗。** 训练/离线评估从整段视频的 8 个区间取帧；线上默认每隔 7 帧取一帧，8 帧构成不重叠窗口。30 FPS 时窗口首尾跨度约 1.63 秒；AIRTLab 实查验证片段长 4.2–5 秒，全视频采样跨度与线上不同。应增加带时间段标注的长视频、困难负样本和打架到正常的过渡视频。
3. **验证事件状态机。** 结合当前 `start_score=0.75`、`end_score=0.35`、确认/清除各 2 秒等配置，记录事件 precision/recall、漏报事件数、每摄像头每小时误报数、发现延迟和结束延迟。事件阈值需要事件验证集支持，不能直接搬用视频阈值。
4. **完善并验证真实运行环境。** 接入目标 RTSP 流，补齐和检查断线重连、源周期管理、证据写盘及远端可访问性、告警 outbox、接收端去重与事件合并、确认协议和资源占用；在目标并发数下持续运行。相关部分仍可能需要代码修改，当前本机测试不证明平台已连通或线上稳定。
5. **如部署 RK3588，再独立验收。** 完成 ONNX→RKNN、至少 100 个窗口数值/分类比较、端到端视频和至少 30 分钟稳定运行，再报告板端兼容性；量化前后重新检查误报和漏报。

继续训练可以改善公开数据上的分类能力，但不能保证现场效果。当前结构对 8 帧特征求平均，不显式建模时间顺序；现有片段级标签也不能说明动作发生在哪一秒、由谁参与。校园现场适用性最终需要按摄像头、日期和场景隔离的数据、足量困难负样本及事件级评估。使用数据或部署前还需核对数据授权范围：当前 SCFD 标注为研究用途且第三方视频权利未核实，AIRTLab 标注为研究和教育用途。
