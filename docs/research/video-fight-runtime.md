# Apple Silicon 本地视频打架分类运行时调研

调研日期：2026-08-25  
目标环境：Apple Silicon（arm64）、macOS 15.0.1  
目标边界：只解决 `本地 MP4 -> 分段打架分类分数`；摄像头、RTSP、告警联动不在本次范围。

## 结论

推荐使用 **Python 3.11 + PaddlePaddle 3.3.1 CPU + PP-Human 官方 PP-TSM 打架部署模型**，但不要先安装完整 PaddleDetection 依赖栈。当前项目只需要复用 PP-Human 的视频抽帧、预处理和 Paddle Inference 调用，把二分类输出中的 `fight` 概率适配为现有 `BehaviorObservation.score`。

这条路线有三层不同的确定性：

1. **官方支持**：PaddlePaddle 为 macOS arm64 提供 CPU wheel，官方支持 Python 3.9–3.13；macOS 不提供 GPU wheel。[PaddlePaddle macOS PIP 安装文档](https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/macos-pip_en.html)、[官方 wheel 对照表](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/Tables.html)
2. **官方提供**：PaddleDetection release/2.9 仍提供 PP-TSM 打架识别部署权重、配置和 MP4 推理入口。[PP-Human 行为识别教程](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/deploy/pipeline/docs/tutorials/pphuman_action_en.md)、[打架识别示例配置](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/deploy/pipeline/config/examples/infer_cfg_fight_recognition.yml)
3. **本机已实测**：项目 Python 3.11.16 环境中的 PaddlePaddle 3.3.1 CPU 自检通过；官方部署模型已完成真实正负 MP4 的解码、预处理、时序切窗和推理，并在拼接视频上产生同一事件的 `START/END`。

默认 `python3` 是 3.14.3，不在官方支持范围，不能用于这条路线；本机已有 `python3.11`。

## 推荐实现路线

### 1. 独立运行环境

使用项目内独立 Python 3.11 虚拟环境，最小运行依赖如下：

```text
paddlepaddle==3.3.1
numpy==1.26.4
opencv-python-headless==4.11.0.86
pillow>=10,<13
```

选择依据：

- PaddlePaddle 3.3.1 在 PyPI 上有 CPython 3.11、`macosx_11_0_arm64` wheel；本机已实装并通过 `paddle.utils.run_check()`。[PyPI 官方发布页](https://pypi.org/project/paddlepaddle/)
- NumPy 固定为 1.26.4，避免旧静态模型与 NumPy 2.x 的兼容风险。当前项目只复用最小推理逻辑，未安装完整 PaddleDetection；后者 release/2.9 的依赖文件另有 `opencv-python<=4.6.0` 限制。[PaddleDetection requirements.txt](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/requirements.txt)
- 只做视频分类部署推理不需要单独安装 PaddleVideo，也不需要安装目标检测、MOT、pycocotools、lapx 等完整 PP-Human 依赖。

注意：PaddleDetection 的安装文档仍只列到 Python 3.10，且依赖说明明显保留了 Paddle 2.x 时代内容；官方没有发布“PaddleDetection 2.9 + Paddle 3.3.1 + macOS arm64”的完整兼容矩阵。[PaddleDetection 安装文档](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/docs/tutorials/INSTALL.md) 因此应复用最小推理模块并以本机测试为准，不宜直接声称完整 PP-Human 在 Mac 上得到官方保证。

### 2. 官方模型

- 部署模型：[`ppTSM_fight.zip`](https://videotag.bj.bcebos.com/PaddleVideo-release2.3/ppTSM_fight.zip)
- 本次下载大小：87,616,865 bytes
- 本次下载 SHA-256：`a58a13c904b9b0fa9e101663826e6a003ecf5e61e3648dd51666c2d6702fbdfd`
- 压缩包内已有：`ppTSM/model.pdmodel`、`ppTSM/model.pdiparams`、`ppTSM/model.pdiparams.info`

官方文档给出的模型精度是 89.06%，并称其由六个公开数据集组合训练；该数字不能直接代表校园固定监控场景的准确率。官方给出的 2 秒视频 128 ms 速度也不是 Mac 指标，而是文档中的部署基准，不能作为 Apple Silicon CPU 性能承诺。[官方行为识别教程](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/deploy/pipeline/docs/tutorials/pphuman_action_en.md)

### 3. 准确的输入预处理

官方训练/部署配置定义：

- PP-TSM 二分类，`num_classes: 2`
- 每个窗口取 8 帧，`num_seg: 8`、`seg_len: 1`
- 短边缩放到 340，再中心裁剪到 320 × 320
- ImageNet mean：`[0.485, 0.456, 0.406]`
- ImageNet std：`[0.229, 0.224, 0.225]`
- PP-Human 视频流水线默认每 7 帧抽一帧，收齐 8 帧推理一次

来源：[PaddleVideo FightRecognition 配置](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/applications/FightRecognition/pptsm_fight_frames_dense.yaml)、[PP-Human 示例配置](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/deploy/pipeline/config/examples/infer_cfg_fight_recognition.yml)、[官方预处理/推理实现](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/deploy/pipeline/pphuman/video_action_infer.py)

输入到 Paddle 模型的张量应为 `float32`，形状 `(N, T, C, H, W)`；本模型单批输入为 `(1, 8, 3, 320, 320)`。

### 4. 准确的输出语义

模型输出形状为 `(1, 2)` 的 logits。官方 `VideoActionRecognizer.postprocess()` 对 logits 做 softmax，再只返回 top-1 的 `class` 和该类 `score`。[官方推理实现](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/deploy/pipeline/pphuman/video_action_infer.py)

官方数据切分脚本明确给出标签映射：

```text
nofight = 0
fight   = 1
```

来源：[官方数据切分脚本](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/deploy/pipeline/tools/split_fight_train_test_dataset.py)

当前项目不能直接把官方返回的 top-1 `score` 当作“打架分数”，因为当 top-1 为 `nofight` 时，该数值是“不打架置信度”。正确适配应始终计算：

```text
fight_score = softmax(logits)[1]
```

然后将它写入 `BehaviorObservation(score=fight_score, behavior="fighting", ...)`，交由项目现有的连续确认、滞回、清除和冷却状态机生成事件。

### 5. 首轮视频验收

首轮只验收本地 MP4，不接摄像头：

1. 一段明确打架正样例，必须产出逐窗口 `fight_score`，并触发一次 `START`；
2. 一段明确非打架负样例，不能触发 `START`；
3. 一段“打架后恢复正常”的拼接样例，应形成同一事件的 `START` 和 `END`；
4. 一段校园困难负样例（拥抱、追逐、体育运动或嬉闹），记录误报结果，不把单次演示通过等同于模型验收；
5. 输出每个窗口的起止时间、采样帧、原始 logits、`fight_score` 和模型版本，保证可回放。

短视频至少应有 8 帧。官方完整 pipeline 对不足 8 帧的视频可能把 `sample_freq` 算成 0，因此项目适配层应显式拒绝不足 8 帧输入，或将采样间隔下限固定为 1。[官方 pipeline.py](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/deploy/pipeline/pipeline.py)

## 可作为对照的官方完整命令

官方 PP-Human 的完整入口是：

```bash
python deploy/pipeline/pipeline.py \
  --config deploy/pipeline/config/examples/infer_cfg_fight_recognition.yml \
  --video_file=/absolute/path/test.mp4 \
  --device=cpu \
  --output_dir=/absolute/path/output
```

配置会自动下载模型，并在每个 8 帧窗口打印类似：

```text
video_action_res: {'class': 1, 'score': 0.93}
```

这条命令适合做官方行为的对照验证，不推荐作为当前项目的最终架构：它会引入大量与“视频分类分数”无关的 PaddleDetection 依赖，而且只暴露 top-1 分数。[官方 PP-Human 快速开始](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/deploy/pipeline/docs/tutorials/PPHuman_QUICK_STARTED_en.md)、[官方 pipeline.py](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/deploy/pipeline/pipeline.py)

## 不推荐或当前不可行的路线

### 默认 Python 3.14

不可行。PaddlePaddle 官方 macOS wheel 支持范围截止 Python 3.13，没有 CPython 3.14 wheel。[官方 macOS 安装文档](https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/macos-pip_en.html)

### Apple Silicon GPU / MPS 加速

不可行。PaddlePaddle 官方明确说明 macOS 当前只支持 CPU 版本；不存在可直接安装的 macOS GPU wheel。[官方 macOS 安装文档](https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/macos-pip_en.html)

### 直接安装完整 PaddleDetection requirements

不推荐用于首轮。完整依赖包含目标检测、跟踪和评测组件，扩大安装失败面；本次任务仅需要 Paddle Inference、OpenCV、NumPy 和少量预处理代码。[PaddleDetection requirements.txt](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/requirements.txt)

### 单独安装 PaddleVideo 做部署

不需要。PaddleVideo 用于模型训练、评估和导出；PP-Human 已提供静态部署模型和推理预处理实现。PaddleVideo 的安装文档本身仍停留在较老的 Python/Paddle 说明，不适合作为当前 Mac 部署环境基线。[PaddleVideo 安装文档](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/docs/en/install.md)

### 仅靠 YOLO 人框距离、框重叠或肢体接近判断打架

不采用。PP-Human 官方方案明确指出打架强依赖时序，使用视频分类而不是单帧检测/分类或骨骼单人行为识别。[官方行为识别教程](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/deploy/pipeline/docs/tutorials/pphuman_action_en.md)

## 许可证边界

- PaddleDetection 和 PaddleVideo 仓库代码均为 Apache-2.0：[PaddleDetection LICENSE](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/LICENSE)、[PaddleVideo LICENSE](https://github.com/PaddlePaddle/PaddleVideo/blob/develop/LICENSE)。
- `ppTSM_fight.zip` 内没有单独的 LICENSE，官方教程也没有明确写出该部署权重的单独授权条款。
- 模型由六个公开数据集组合训练，代码许可证不能自动覆盖模型权重及训练数据衍生权。当前可用于本地技术原型；商业交付或再分发模型前，应向 Paddle 官方确认权重授权，并分别核查训练数据集条款。

## 本机验证记录

验证环境：arm64、macOS 15.0.1、Python 3.11.16。

已验证：

- `paddlepaddle==3.3.1`、`opencv-python-headless==4.11.0.86`、`numpy==1.26.4` 已安装在项目 `.venv`；
- `paddle.utils.run_check()` 输出 CPU 安装成功；
- 官方模型下载成功，SHA-256 与上文一致；
- Paddle Inference 能加载旧 `.pdmodel/.pdiparams`；
- `(1, 8, 3, 320, 320)` 零张量推理成功并返回 `(1, 2)` logits。
- 官方 SCFD `fi001.mp4` 输出 `fight_score=0.768393`，`nofi001.mp4` 输出 `0.182818`；
- 两段打架后接两段非打架的拼接视频产生 4 个窗口，分数依次为 `0.829649`、`0.881198`、`0.268649`、`0.075504`，并生成同一事件的 `START/END`；
- 14 个单元测试在视频环境通过，其中 2 个覆盖官方模型输入形状和 fight 类概率语义。

尚未验证：

- Mac CPU 的真实吞吐和内存；
- 模型在校园固定监控场景的误报、漏报；
- 摄像头、RTSP、断流重连和边缘设备部署。
