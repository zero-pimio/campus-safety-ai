# 本地场景规则验证素材核验

日期：2026-09-21。仅盘点现有本地文件和归档目录；未下载、未训练、未跑模型。人员素材依据原始 `walk` 动作标签选择，不依据推理分数。后续经授权，仅把 UT set1 的 `seq1.avi` 单成员解到本轮 runtime，并抽取首、中、末三帧查看；未修改原始数据或解压其他成员。

## 可直接使用的人员素材

| 文件（项目根 `/Users/wanwanzhu/yolo` 下） | 文件字节数 | 实读元数据 | manifest 信息 |
| --- | ---: | --- | --- |
| `datasets/private/airtlab/violence-detection-dataset/non-violent/cam1/51.mp4` | 3,069,034 | 1920×1080；30 FPS；149 帧；4.9667 秒 | `airtlab:non-violent:51`；train；label=0；greet,walk |
| `datasets/private/airtlab/violence-detection-dataset/non-violent/cam1/53.mp4` | 3,054,503 | 1920×1080；30 FPS；150 帧；5.0000 秒 | `airtlab:non-violent:53`；train；label=0；walk,greet |

来源为本地 AIRTLab 仓库，remote 为 `https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos.git`。原始动作标签见 `datasets/private/airtlab/violence-detection-dataset/nonviolent-action-classes.csv`；训练清单见 `datasets/manifests/fight-v1.csv:348`、`:350`。AIRTLab README 说明这些是同一室内场景中非专业演员的表演视频；同名 cam1/cam2 是同一表演的不同视角。README 声明研究和教育用途，清单中对应 `research-and-education`。

两段可验证真实视频上的人物检测、人工绘制区域和闯入规则数据流。现有标签仅为片段动作标签，没有人物框/轨迹或禁区、闯入事件真值，且已经属于 fight 训练集，不能用来宣称独立闯入测试集指标或校园现场效果。

## 有限核查的车辆画面

`runtime/scene-validation-20260921/source/seq1.avi` 是 UT-Interaction set1 的原始单成员副本。实读大小 **16,745,420 字节**，720×480，30 FPS，2031 帧，67.7 秒。固定抽取第 0、1015、2030 帧（0、33.8333、67.6667 秒），三帧均可见左上和左下边缘被裁切的部分车体；中间帧同时有两人互动。车辆缺少完整轮廓，未验证是否能被检测器稳定检出。仅能作为真实画面上的人工区域停车规则冒烟候选，不能称为完整车辆样例、违停真值或违停准确率验收。

来源和完整性记录：

- 原归档：`datasets/private/ut-interaction/ut-interaction_set1.zip`；152,414,545 字节；成员 `seq1.avi`，CRC32 `e3ad697e`。
- 归档 SHA256：`172965ce7208704165448c6d8890d9853e9fee96ffc7e06036bb0ea1b6c29b2f`。
- 解出成员 SHA256：`aa86ddf19e07556d768315d271931a3f863f9cf5a3d092f6fc6def4f30398957`。
- 元数据：`runtime/scene-validation-20260921/source/seq1-source-metadata.json`。
- 预览：同目录 `seq1-frame-00000.jpg`、`seq1-frame-01015.jpg`、`seq1-frame-02030.jpg`。
- 解包前检查成员是唯一的 `seq1.avi`、相对路径、无 `..`、无反斜杠及符号链接，并核实目标位于指定 runtime/source 目录；使用单成员流式复制，未调用 `extractall`。

现有 `docs/research/fight-datasets.md:147` 记录其来源为 [UT Austin 挑战页](https://cvrc.ece.utexas.edu/SDHA2010/Human_Interaction.html)，set1 为停车场人际互动视频，标注为人物互动起止区间和空间框。`datasets/private/ut-interaction/ut-interaction_labels_110912.xls` 在本地，但本轮未解析。没有证据表明该数据提供车辆停车、禁停区域或违停标签；也没有据此确认商业授权。本轮未重新访问外部来源。

## 本地素材覆盖范围

按文件名/目录盘点，原始可直接读取视频为 AIRTLab 290 段、SCFD 300 段；现有 runtime 视频为打架演示或拼接/渲染产物。未逐段视觉扫描全部 590 段，因此结论是“未找到有明确车辆停车/违停真值的素材”，并非断言所有视频都没有车。既有拼接演示不能充当真实违停验收。

归档仅检查 ZIP 中央目录：UT 原始 set1/set2 各 10 段 AVI，分段包各 60 段 AVI；bus-violence 包有 1400 段 MP4 和 train/test 列表；AVA 包只有标注文件。CAVIAR 当前保留 XML，无直接视频，BEHAVE 无直接视频。除上述经授权的 `seq1.avi` 外，均未解包。UT 停车场背景、人际动作标签和 bus-violence 的名称均不构成车辆违停标注证据。

## 视频入口与 tracker 注意事项

以下为本地源码与当前安装依赖核验，不是视频跟踪性能结果。

1. `adapters/runtimes/ultralytics.py:22-32` 默认使用实际模型 `result.names`，输出原图坐标，并透传 `camera_id/source_epoch/sequence/captured_at`。`configs/models/detect-v1.toml` 的目标标签列表不是完整 COCO 类号映射；不要按列表位置重映射类别。传入不完整 `label_map` 后遇到其他类会查键失败。
2. 当前 `predict` 未显式传 `conf`。安装版 Ultralytics `engine/model.py:507` 默认阈值 0.25；适配器 ByteTrack 的低阈值是 0.1，因此 0.1–0.25 检测已被前级滤掉，不能把该阶段当作完整低分候选恢复。
3. `ultralytics_bytetrack.py:46-71` 在 `(camera_id, source_epoch)` 改变时重置，并报告旧活动/丢失轨迹。每个源应有独立 tracker 实例；多个摄像头交错调用同实例会不断重置。适配器自身没有顺序号去重保护。
4. 当前安装版 `BYTETracker` 的 `max_frames_lost=args.track_buffer`（`byte_tracker.py:262`），每次 `update` 仅增加一次 `frame_id`（`:268`），按调用次数淘汰（`:488`）。默认 30 指 30 次处理更新，抽帧后不等于 1 秒。场景规则持续时间应采用媒体时间，不能采用推理耗时或假设缓冲固定为墙钟秒数。
5. 适配器按检测标签字符串分配可逆的内部整数类号（`ultralytics_bytetrack.py:74-99`）；上游 `get_dists` 仅计算 IoU 和置信分数（`byte_tracker.py:507-512`），没有类别门控。因此标签映射正确也不保证人物与车辆分别关联。混合场景应考虑目标族过滤或独立追踪，并核查类别跳变。
6. 已有 ByteTrack 测试是合成检测输入的身份/淘汰/epoch 验证，不是本轮真实视频跟踪准确率证据。SimpleIoUTracker 使用媒体时间和同类匹配，与 ByteTrack 的缓冲语义不同，切换实现时应记录配置。
