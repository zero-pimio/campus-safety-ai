# 打架视频分类训练 v1

## 目标

训练视频级 `non_fight/fight` 二分类模型，输出整段 8 帧窗口的打架概率。该模型不负责指出具体参与打架的人；人物参与者定位是后续独立训练任务。

## 数据

- SCFD：150 fight + 150 non-fight。
- AIRTLab：保留全部 non-violent 困难负样本；violent 中只保留含 fight、punch、kick、push、slap 或 choke 的事件，排除只有 gunshot、stab、club 的片段。
- AIRTLab cam1/cam2 中同名视频属于同一表演，使用共同 `group_id`，禁止跨 train/val/test。
- 当前 manifest 共 590 段：train 412、val 86、test 92。

Bus Violence 是车内专项数据，UT/CAVIAR/BEHAVE 主要用于互动或定位验证，暂不混入第一版训练。测试集合在模型和阈值冻结前不参与调参。

## 模型与接口

- Backbone：ImageNet 预训练 MobileNetV3-Small。
- 时间聚合：每帧共享 2D Backbone，8 帧特征求平均后进行二分类。
- 输入：RGB，8 帧，沿通道拼接为 `N,24,224,224`。
- 归一化：ImageNet mean/std。
- 输出：`[non_fight_logit, fight_logit]`；运行时使用 softmax 的第 1 类概率。
- ONNX：固定 batch=1、opset 13，导出后执行 ONNX checker 与 ONNX Runtime 数值对齐。

该结构优先选择 RKNN 常见算子和固定输入；只有在 RK3588 上完成 RKNN 转换、100 个窗口输出比较和 30 分钟稳定运行后，才能称为板卡兼容。

## 验收

训练阶段报告 accuracy、balanced accuracy、precision、recall、F1。最终还必须把窗口分数接回事件状态机，报告事件级 precision/recall/F1、漏报事件数、平均发现延迟和每摄像头每小时误报数。
