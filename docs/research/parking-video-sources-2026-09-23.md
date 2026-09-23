# 车辆驻留诊断素材来源（2026-09-23）

已从 UCF CRCV 官方页面下载 PNNL Parking Lot 2 和 Parking Lot 1，共 38,934,508 字节。优先用前者验证真实连续画面中的 30 秒驻留；两段都不能证明车辆驶离、正常经过车辆不告警，或真实场地的违法停车。

## 官方来源与使用范围

[UCF CRCV：PNNL Parking Lot 数据页面](https://www.crcv.ucf.edu/data/ParkingLOT/)提供视频直接下载，注明数据由 Pacific Northwest National Laboratory 提供，并要求使用时引用相关论文。页面没有登录或申请环节；本次没有接受额外协议。页面未明确列出 SPDX 许可或商业再分发授权，本地保留用于研究与诊断，不据此声称已获得公开再发布授权。

建议引用：Guang Shu, Afshin Dehghan, Omar Oreifej, Emily Hand, Mubarak Shah. *Part-based Multiple-Person Tracking with Partial Occlusion Handling*. CVPR 2012。该页面的跟踪标注针对行人，不提供车辆停车事件或交通违法标签。

## 本地文件与实际测量

目录：`datasets/private/parking-review-20260923/`。以下结果来自下载后 `ffprobe` 与完整逐帧解码，不是仅照抄网页描述。

| 素材 | 本地相对路径 | 连续时长 | 分辨率、帧率 | 实际解码帧数 | 文件大小 |
|---|---|---:|---|---:|---:|
| PNNL 2，首选 | `ucf-pnnl2/PNNLParkingLot2.avi` | 50.000000 秒 | 1920×1080，30/1 fps | 1500 | 24,494,718 字节 |
| PNNL 1，遮挡补充 | `ucf-pnnl1/PNNL_Parking_LOT1.avi` | 34.413793 秒 | 1920×1080，29/1 fps | 998 | 14,439,790 字节 |

PNNL 1 官方页概述为 1000 帧，本次下载文件容器与完整解码均为 998 帧；以本地实测为准。没有补帧、拼接或循环播放来增加驻留时长。

下载直链：[PNNL 2 原始 AVI](https://www.crcv.ucf.edu/data/ParkingLOT/PNNLParkingLot2.avi)、[PNNL 1 原始 AVI](https://www.crcv.ucf.edu/data/ParkingLOT/PNNL_Parking_LOT%281%29.avi)。SHA-256、下载时间、元数据和关键帧索引记录在本地 `sources.json`。

## 视觉核对与适用边界

已检查 PNNL 2 每约 5 秒、PNNL 1 每约 4 秒的接触表，并另存准确帧索引关键帧。白色小车、右侧灰车以及 PNNL 2 前景银车轮廓完整；多人走动造成局部遮挡。所检查画面中车辆从片头就已停稳，未观察到自然驶入或驶离，也未确认存在经过车辆负例。

- 适合：完整车辆持续驻留、行人经过、局部遮挡、轨迹与驻留计时诊断。
- 演示时必须注明：**禁停区由本项目人为配置，数据只展示停车事实，不代表原场地实际违停。**
- 可通过把禁停区放在车外来验证区域过滤；这不等同于移动车辆或合法短停负例。
- 文件结束只允许标记为 `source_ended` 等观察结束原因，不能写成“驶离”或“违停解除”。
- 两段来自同一停车场视角，不构成独立摄像头泛化验证。此次仅诊断使用，不并入模型正式训练/测试集。

## 时间戳核对

两段均为 XVID AVI。`frame-timestamps.json`保存 `ffprobe -show_frames` 原始结果：PNNL 2 有 1500 帧，前 1499 帧的 best-effort 时间为约 1/30…1499/30 秒，末帧缺失该时间；PNNL 1 也只有最后一帧缺失。已有时间严格递增，不能将其描述为完整原始捕获 PTS。

在明确声明 CFR 推导的前提下，可按容器及官方说明一致的帧率使用 `frame_index / fps` 生成从零开始的相对时间轴。不得把任意帧率套到未知采样率的图像序列上以获得 30 秒驻留。需要严格原始逐帧 PTS 的入口应保留其缺失检查，不能为了导入这两个 AVI 而关闭检查。

每个素材目录含 `ffprobe.json`、`frame-timestamps.json`、接触表及 `frame-index-*.jpg`。PNNL 2 精确抽帧索引为 0、450、900、1350、1485；PNNL 1 为 0、300、600、900、990。

## 其他一手来源的筛选结果

- [VIRAT](https://viratdata.org/)地面视频要求 Protection Agreement；本次未下载或接受协议。
- [ETISEO](https://www-sop.inria.fr/orion/ETISEO/download.htm)视频需要提交申请获取密码；本次未申请。
- [CDnet 2012](https://changedetection.net/dataset2012/)提供 `parking` 2500 帧，但[官方说明](https://changedetection.net/datasetOverview/)指出帧率随视频而变，当前未核实该序列的原始时间基准，因此不将其当作已确认时长的驻留视频。
- [Dragon Lake Parking](https://sites.google.com/berkeley.edu/dlp-dataset)完整视频需要申请，且有非商业用途和禁止再分发条款；本次未申请或下载。

尚缺的专项录像：车辆驶入后停稳至少 30 秒再驶离，以及同一画面中正常经过或短停后离开的车辆。此次找到 PNNL 2 后停止广泛检索，仅按后续要求补查同一官方来源的 PNNL 1，未把 EOF 充当自然驶离。
