# 校园安全边缘 AI 开源项目选型

核对日期：2026-08-25

## 结论

没有一个现成仓库同时完整覆盖“非法闯入、车辆违停、跌倒、超速、RKNN/昇腾和 OpenRemote 告警”。最稳妥的路线不是推翻当前 contract-first 骨架，而是保留：

```text
FramePacket -> Detections -> Tracks -> EventRecord -> SQLite Outbox -> Destination
```

然后按模块吸收成熟能力。明确 Top 3 如下：

1. **PaddleDetection PP-Human + PP-Vehicle：最接近现成校园安全算法包。** 官方流水线直接提供人员闯入、跌倒、车辆违停等能力；适合作为事件算法基线和对照实现，不建议整仓接管当前系统。
2. **Frigate：最成熟的摄像头/NVR 工程底座。** 它已有 RTSP、录像、对象跟踪、区域/滞留、估算测速、MQTT 和 RKNN 社区支持；适合作为部署对照或可选 sidecar，但内置能力没有跌倒，且其事件模型不能替代本项目稳定契约。
3. **Supervision：最适合低侵入复用的 Python 组件库。** 可直接复用 ByteTrack、PolygonZone、LineZone、时间驻留和速度估算示例，同时让模型结果仍先转换成项目 `Detections`；它不是摄像头平台，也不提供跌倒模型。

如果只追求“最快做出四项里三项的单机演示”，优先跑通 PaddleDetection；如果追求长期可维护、可换芯片和 OpenRemote 可靠告警，继续当前骨架，优先引入 Supervision 的跟踪/几何组件，并把 PaddleDetection 和 Frigate 作为对照或旁路能力。

## 能力速览

说明：`直接` 表示官方仓库已有对应功能；`组合` 表示可由官方已有组件构成，但仍需本项目业务规则；`研发` 表示只有通用模型或训练框架；`无` 表示未找到官方实现。

| 项目 | 闯入 | 违停 | 跌倒 | 超速 | 主要许可 | 对当前骨架的角色 |
|---|---:|---:|---:|---:|---|---|
| [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) | 直接 | 直接 | 直接 | 无 | Apache-2.0 | 事件算法/模型适配器与对照基线 |
| [Frigate](https://github.com/blakeblackshear/frigate) | 直接 | 组合 | 无 | 直接但仅估算 | MIT，名称与 Logo 除外 | NVR/RTSP/MQTT sidecar 或部署对照 |
| [Supervision](https://github.com/roboflow/supervision) | 组合 | 组合 | 无 | 组合 | MIT | 跟踪、区域、驻留、几何工具库 |
| [Ultralytics](https://github.com/ultralytics/ultralytics) | 组合 | 组合 | 研发 | 组合 | AGPL-3.0 / 商业许可 | PC 训练与感知适配器 |
| [OpenVINO Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo) | 无 | 无 | 无 | 无 | Apache-2.0，模型另查 | Intel 推理参考，不是业务框架 |
| [MMAction2](https://github.com/open-mmlab/mmaction2) + [MMPose](https://github.com/open-mmlab/mmpose) | 无 | 无 | 研发 | 无 | Apache-2.0，模型/数据另查 | 后期跌倒模型研发支线 |
| [NVIDIA DeepStream](https://docs.nvidia.com/metropolis/deepstream/dev-guide/) | 组合 | 研发 | 研发 | 研发 | SDK EULA；部分样例 Apache-2.0 | 仅 NVIDIA 硬件路线的高吞吐替代后端 |
| [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo) | 无 | 无 | 无 | 无 | Apache-2.0，源模型另查 | Rockchip 部署适配与板端验收样例 |
| [Ascend samples](https://github.com/Ascend/samples) | 无 | 无 | 无 | 无 | Apache-2.0，模型/SDK另查 | 昇腾 ACL/CANN 接口参考 |

## Top 1：PaddleDetection PP-Human / PP-Vehicle

### 实际覆盖能力

[PP-Human 官方快速开始](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/deploy/pipeline/docs/tutorials/PPHuman_QUICK_STARTED_en.md)列出检测、跟踪、ReID、属性和行为识别；其中跌倒由“多目标跟踪 + 关键点 + 骨架动作识别”组成，Breaking-In 则由多目标跟踪构成。代码中也有独立的[骨架动作推理实现](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/deploy/pipeline/pphuman/action_infer.py)。

[PP-Vehicle 官方快速开始](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/deploy/pipeline/docs/tutorials/PPVehicle_QUICK_STARTED_en.md)明确包含车辆检测、OC-SORT 跟踪、属性、车牌、违停、压线和逆行；违停依据车辆轨迹、指定区域和停留时间阈值判断。未在官方 PP-Vehicle 能力中找到速度标定或超速事件，因此不能把轨迹跟踪等同于测速。

### 成熟度、许可和边缘边界

主仓当前页显示约 14.2k stars，且 [Releases](https://github.com/PaddlePaddle/PaddleDetection/releases) 已到 v2.9.0（2026-03-19），代码以 [Apache-2.0](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/LICENSE) 发布。PP-Human 的官方速度数据来自 T4/TensorRT 或 Jetson AGX，不是 RKNN 或昇腾板卡实测；这些数字不能外推为目标板多路实时能力。权重和训练数据仍需逐项核对其单独条款。

### 与当前骨架整合

- 新增 `PaddlePerception`，只把 Paddle 检测/关键点输出转换为项目 `Detections` 或后续专门的姿态契约；不得让 Paddle 原生 tensor 跨过 `Perception` 边界。
- 把 PP-Human 的跌倒和 PP-Vehicle 的违停作为离线基线：对同一 ReplayBundle 输出候选事件，再由本项目状态机统一生成 `EventRecord`。
- 不复制其完整 pipeline 主循环、可视化和配置体系，避免与 `EventAnalysis`、Outbox 和 OpenRemote 投递形成第二套事实源。
- G2 仍要在最终 RKNN 或昇腾板卡上完成模型转换、100 图输出比较和连续视频稳定性验证。

## Top 2：Frigate

### 实际覆盖能力

Frigate 官方定位是[本地实时 IP 摄像头 NVR](https://github.com/blakeblackshear/frigate)，具备录像、RTSP 重流、对象检测/跟踪、区域编辑和 MQTT。其[区域文档](https://docs.frigate.video/configuration/zones/)明确支持：

- 以目标框底部中心点判断区域归属；
- 按 `person`、`car` 等对象类别限制区域；
- 以 `required_zones` 限制告警、检测和截图；
- `loitering_time` 和 `inertia`，可分别用于滞留/违停候选和边界抖动过滤；
- 四点地面区域 + 实际边长标定的速度估算、MQTT 速度字段和 `speed_threshold`。

官方也明确警告：速度是估算值，高度依赖机位、区域点和实际距离，不应用于执法。因此它适合课程项目里的“超速提示”，不等于法定测速。Frigate 没有内置跌倒模型；违停也仍需增加禁停时段、停留阈值、白名单等校园语义。

### 成熟度、许可和边缘边界

当前 GitHub 页显示约 33.7k stars、77 个 releases，代码/配置/文档采用 [MIT](https://github.com/blakeblackshear/frigate#license)，但 Frigate 名称和 Logo 不包含在 MIT 授权中。[官方硬件页](https://docs.frigate.video/frigate/hardware/)把 Rockchip/RKNN 标为社区支持，列出 RK3562/3566/3568/3576/3588，并明确只支持有限模型架构、适合 tiny/small 模型。该项目未列出昇腾后端。

### 与当前骨架整合

推荐先作为可选 sidecar，而不是 fork Frigate：

- 让 Frigate 负责 RTSP 重流、录像和回看，本项目读取其重流，保留自己的 `Perception -> EventAnalysis -> Outbox` 主线；
- 或做独立 `FrigateEventSource` 实验适配器，把 MQTT 事件转换为 `FramePacket/Tracks/EventRecord` 所需字段，但必须标记来源，并用 golden replay 检查语义差异；
- OpenRemote 仍只由本项目 `EventDelivery` 可靠投递，避免 Frigate MQTT 和本项目 Outbox 对同一事件重复报警。

这个组合的代价是可能重复解码或重复推理；只有当 NVR/回看成为明确验收项时，才值得进入主部署方案。

## Top 3：Supervision

### 实际覆盖能力

Supervision 是[模型无关的计算机视觉工具库](https://github.com/roboflow/supervision)，支持把 Ultralytics、Transformers、MMDetection 等结果转换为统一 `Detections`。官方提供内置 [ByteTrack](https://supervision.roboflow.com/develop/notebooks/object-tracking/)、PolygonZone、LineZone，以及[区域停留时间完整示例](https://github.com/roboflow/supervision/tree/develop/examples/time_in_zone)。官方教程也提供车辆跟踪、透视变换和速度估算路线，但它没有校园事件状态机、跌倒权重、RTSP 服务或可靠告警投递。

当前 GitHub 页显示约 49.7k stars、约 5,074 commits，仓库含测试，采用 [MIT](https://github.com/roboflow/supervision/blob/develop/LICENSE.md)；[发布页](https://github.com/roboflow/supervision/releases)仍持续发布并修复 PolygonZone、LineZone 和 ByteTrack。示例若使用 Ultralytics，Ultralytics 部分仍受其 AGPL/商业许可约束；Supervision 的 MIT 不会改变上游模型许可。

### 与当前骨架整合

- 首先只替换简单 IoU 跟踪器为 `sv.ByteTrack`，输入/输出仍是项目 `Detections`/`Track`。
- 可参考 PolygonZone/LineZone 的几何实现，但保留当前基于真实时间戳的进入延时、离开延时、冷却和事件幂等状态机；官方 time-in-zone 示例按视频 FPS 计时，不应直接替代真实时间戳语义。
- 超速必须增加摄像机标定、透视映射和轨迹平滑契约，不能用像素/帧直接声称 km/h。

## 其他候选及“不作为主线”的原因

### Ultralytics：保留为感知基线，不作为校园安防框架

Ultralytics 当前仓库非常活跃，[Releases](https://github.com/ultralytics/ultralytics/releases)持续更新；官方支持检测、姿态、跟踪、训练和多种导出，[导出器](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/exporter.py)列出 ONNX、OpenVINO、TensorRT、NCNN、RKNN 等格式。官方 [Hugging Face YOLO11 模型卡](https://huggingface.co/Ultralytics/YOLO11)说明检测权重基于 COCO 80 类，跟踪可用于检测/分割/姿态模型。

但它不直接提供校园闯入、违停、跌倒和超速的业务语义。代码和官方权重为 AGPL-3.0，并另有商业许可；[官方许可说明](https://github.com/ultralytics/ultralytics/blob/main/CONTRIBUTING.md#license)要求在项目不适合开源时考虑 Enterprise License。继续保留现有 `UltralyticsPerception` 是合理的，但在确定项目发布/商用方式前不能忽略许可边界；“支持导出 RKNN”也不等于已在指定芯片、量化配置和视频链路上验证成功。

### OpenVINO Smart Classroom：类别不匹配，而且模型库进入维护模式

[Smart Classroom Demo](https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/smart_classroom_demo/cpp/README.md)识别的是坐、站、举手、写字、转身、趴桌等有限课堂动作，并可做人脸识别；它没有本项目四类事件。[Open Model Zoo](https://github.com/openvinotoolkit/open_model_zoo)已明确标注“作为模型来源进入 maintenance mode”，虽然仓库采用 Apache-2.0，仍不适合作为新系统主线。可只把 OpenVINO Runtime 当作 Intel 设备推理后端。

### MMAction2 + MMPose：研究价值高，交付路径重

[MMAction2](https://github.com/open-mmlab/mmaction2)支持视频分类、时序定位、时空动作检测和 ST-GCN、PoseC3D 等骨架动作识别；[MMPose](https://github.com/open-mmlab/mmpose)提供多人姿态模型。两者均为 Apache-2.0 代码，但通用预训练类别不是校园跌倒成品，且没有禁区、停车区、速度标定、NVR 或告警闭环。其 PyTorch/MMCV/MMEngine 依赖和板端转换成本高，适合 G7 的跌倒精度升级，不适合首学期主线。

### NVIDIA DeepStream：只在已确定 Jetson/NVIDIA 时考虑

DeepStream 的 [nvdsanalytics](https://docs.nvidia.com/metropolis/deepstream/8.0/text/DS_plugin_gst-nvdsanalytics.html)直接支持 ROI、拥挤、方向和过线分析，且依赖检测器与 tracker metadata；它是成熟的多路视频高吞吐方案，但不提供现成跌倒/违停/超速语义，并把系统绑定 NVIDIA GPU/Jetson。当前 [DeepStream 文档](https://docs.nvidia.com/metropolis/deepstream/dev-guide/)列出 SDK 9.1；SDK 下载受 [DeepStream EULA](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Legal.html)约束，尽管部分[参考应用为 Apache-2.0](https://github.com/NVIDIA-AI-IOT/deepstream_reference_apps/blob/master/LICENSE)。若 G0 选 RKNN 或昇腾，它不应进入主线。

### RKNN Model Zoo：部署验证集，不是应用

[RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo)提供 ONNX -> RKNN 转换及 Python/C API 推理样例，支持 RK3562/3566/3568/3576/3588 等，且对 RV1103/1106 标为有限支持；官方要求与对应 RKNPU SDK 版本匹配。仓库为 Apache-2.0，但模型来源另有许可。它不含跟踪、事件状态机、NVR 或告警，应作为 `RknnPerception` 的 G2 验收参考，不能 fork 成业务主仓。

### Ascend samples：昇腾接口样例，不是校园方案

[Ascend samples](https://github.com/Ascend/samples)是 AscendCL 的设备、内存、流、模型加载、图像/视频推理样例集合，采用 Apache-2.0。官方说明需要按具体产品选择匹配的 CANN 版本。它可帮助实现 `AscendPerception`，但没有本项目事件逻辑。旧 [Ascend ModelZoo](https://github.com/Ascend/modelzoo)还明确提示公开数据集和模型可能只允许非商业研究/教育用途，因此每个权重必须单独审查；不能因为仓库代码是 Apache-2.0 就推断所有模型可自由商用。

### Hugging Face 校园/跌倒专用权重：只用于探索，不直接进入主线

主动检索发现了更贴题的权重，但其证据不足以替代成熟框架：

- [visionlab-ai/school-violence-detection-models](https://huggingface.co/visionlab-ai/school-violence-detection-models)是毕业设计性质的 3D CNN 模型，模型卡自报在 RWF-2000 validation 上 F1 0.93，同时明确数据分布不一定符合学校高位 CCTV、会把激烈体育活动误报为暴力，且源码链接需要向作者索取。它也不覆盖本项目首期四项。
- [AXERA-TECH/Fall-axera](https://huggingface.co/AXERA-TECH/Fall-axera)针对 AX650N/AX8850 的 YOLOv7-pose 跌倒模型，模型卡明确说遮挡、跌倒方向、机位和场景都会影响结果，并建议再加 tracking/ST-GCN；它不是 RKNN 或昇腾权重，且为 AGPL-3.0。
- [Ultralytics/YOLO11](https://huggingface.co/Ultralytics/YOLO11)是成熟通用基线，但 COCO 权重没有“跌倒”“闯入”“违停”“超速”类别。

这些权重可以进入隔离的模型评估表，但只有在来源、权重格式、许可证、校园数据 Precision/Recall/F1 和目标板性能都通过门禁后，才能成为 `Perception` 的可选实现。

## OpenRemote 边界

OpenRemote 自身是独立的 IoT/资产/规则平台，其主仓采用 [AGPL-3.0](https://github.com/openremote/openremote/blob/master/LICENSE.txt)，并有持续发布的[官方 releases](https://github.com/openremote/openremote/releases)。它提供 REST 通知接口，例如 [`/notification/alert`](https://docs.openremote.io/docs/rest-api/send-notification/)，也支持通过 Agent/协议连接外部服务。

本项目不应把视觉状态机写进 OpenRemote：`EventAnalysis` 先产生稳定、可重放的 `EventRecord`，SQLite Outbox 保证重试和幂等，`OpenRemoteDestination` 再映射资产属性/告警。这样即使替换 Paddle、Ultralytics、Frigate、RKNN 或 Ascend，平台侧事件语义不变。

## 建议执行顺序

1. 保持当前非法闯入 golden replay 和 Outbox 契约不变。
2. 用 Supervision ByteTrack 替换占位 IoU tracker，并补固定 detections 回归测试。
3. 选取同一段校园授权视频，在 Paddle PP-Human/PP-Vehicle 与当前 Ultralytics 基线上对比闯入、跌倒、违停候选结果。
4. 第三事件若选超速，优先实现四点地面标定 + 真实时间戳轨迹速度，并用 Frigate 的速度估算作为对照；若选跌倒，优先验证 PP-Human，再决定是否投入 MMPose/MMAction2。
5. G0 只选一款板卡：RKNN 走 RKNN Model Zoo；昇腾走 Ascend samples。不得把 PC/Jetson 数字当作目标板结果。
6. 最后接 `OpenRemoteDestination`，用假事件先验收断网重试、重复投递和恢复，再接真实视频。

## 最终选择

- **主仓与长期架构：保留当前 contract-first 项目。**
- **首选现成算法参考：PaddleDetection PP-Human/PP-Vehicle。**
- **首选可复用代码组件：Supervision。**
- **首选完整视频工程对照：Frigate；只有明确需要 NVR/回看时才作为 sidecar。**
- **首选边缘部署参考：由最终板卡二选一，RKNN Model Zoo 或 Ascend samples。**
- **不要直接把 OpenVINO Smart Classroom、MMAction2/MMPose、DeepStream、任意 Hugging Face 专用权重 fork 成主线。**
