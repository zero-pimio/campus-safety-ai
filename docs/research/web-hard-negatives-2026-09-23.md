# 首批网上困难负例素材（2026-09-23）

> 本文记录下载时的状态；随后 23 段 GMD 已完成复核、分组并用于新候选训练/验证/一次保留测试，2 组 UP-Fall 仍仅诊断。当前状态见 [项目推进记录](../training/project-progress-2026-09-23.md)。下载清单中的未分配/未训练标记是保留的下载快照，当前划分以 `datasets/manifests/fall-web-v1.json` 为准。

已下载 **23 段原始 MP4 和 2 组原始 RGB 帧序列**，原始文件合计 **358,133,117 字节，约 358 MB**。按来源粗标签为 17 段日常活动/取物负例、8 段跌倒对照；两组 RGB 序列另附 MP4 预览，不额外计样本。下载完成不表示完成训练准入，本轮没有用这些文件训练或调整模型。

本机素材根目录：`datasets/private/web-hard-negatives-20260923/`。

- 素材总清单（本地产物：`datasets/private/web-hard-negatives-20260923/inventory.json`）
- GMDCSA24 逐视频清单（本地产物：`datasets/private/web-hard-negatives-20260923/gmdcsa24/inventory.json`）
- UP-Fall 原始序列及预览清单（本地产物：`datasets/private/web-hard-negatives-20260923/up-fall/manifest.json`）
- [独立完整性检查](../audits/2026-09-23/web-data-integrity.json)

## 已下载内容

| 来源 | 有效下载 | 涵盖内容 | 人物/摄像头边界 |
| --- | --- | --- | --- |
| GMDCSA-24 作者仓库 | 16 ADL + 7 Fall，171.1 MB；全部 5100 帧解码成功 | 弯腰捡物、坐下/起立、主动躺卧、坐地、俯卧撑等，以及跌倒对照 | 保留 4 个官方 subject；物理摄像头身份未核实，不能按片段造新 camera ID |
| UP-Fall 官方入口 | 1 组捡物 + 1 组向前跌倒 RGB ZIP，187.0 MB；181+195 张 PNG 解码成功 | 捡物负例和跌倒对照，附全部帧转换的 MP4 | 同一 Subject1 的两个试验、两个原相机视角；跌倒画面另有身份未核实的辅助人员 |

来源是 [GMDCSA-24 作者仓库](https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos)与 [UP-Fall 官方数据页](https://sites.google.com/up.edu.mx/har-up/)。具体下载地址、原标签、引用及许可核对分别见 [GMDCSA 下载记录](gmdcsa24-download-2026-09-23.md)和 [UP-Fall 下载记录](up-fall-download-2026-09-23.md)。

GMDCSA 的 23 个文件与固定提交 `5abac7693229900cf80f722e878fbb119211fc1c` 的 Git blob 完全一致，并保存 SHA-256；根任务再次核对大小与 SHA。没有与既有 `datasets/` 视频发现字节完全相同的重复，23 个新增视频 SHA 均不同。这个检查不识别重编码副本或同场表演，仍需依赖来源组登记。

GMDCSA 另有 1 段视频传输不完整，残片被保留为失败下载且不计入有效素材。TsetFall 必须申请解码密钥，本轮没有获取视频，详见 [受阻记录](tsetfall-download-2026-09-23.md)。SAFER-Activities 的官方数据页要求登录并同意分享联系信息，未下载该数据。[SAFER 官方发布页](https://huggingface.co/datasets/SAFER-Activities/SAFER-Activities)

## 使用前必须补齐的内容

这批是新增开发素材，尚未分配 train/val/test。GMDCSA 的来源组暂按 subject 保守共组；UP-Fall 保留 subject/activity/trial，今后另一机位必须与同一试验共组。不能把 25 段素材写成 25 个独立人物或摄像头，也不能直接声明跨人物/摄像头验证已完成。

已保留作者 CSV 原标注，其中部分时间区间存在倒置或不完整语法，未擅自修正。UP-Fall 当前是试验级活动标签，缺精确事件开始、结束和恢复时刻。下一步先复核标注和所有可见人物，再冻结分组；固定窗口训练必须与运行窗口、姿态关联和 unknown 规则一致。

UP-Fall 的 MP4 为全部原 PNG 按时间戳排序后生成的 CFR 预览，帧间时间是近似；原 ZIP 与逐帧时间戳完整保存。正式事件延迟评估应使用真实原时刻，并核验新 MP4 的逐帧 PTS，不能仅凭平均 FPS 推定准确时间。

GMDCSA 根仓库附 MIT 文本，快照已保存；UP-Fall 官方允许公开研究下载并要求引用，但未找到覆盖原始 RGB 数据的明确标准许可证，其商用/再分发权限未确认。原数据保存在本地 private 目录，没有发布或上传。

**仍缺嬉闹、独立校园长录像及完整“正常→跌倒→恢复”的人工事件标注。** 弯腰和坐起已增加，但不能据此宣称已覆盖所有蹲下/遮挡/多人场景。此批未触碰旧训练集、测试划分或模型权重。

连续告警和事件评估工具已完成本地验证，使用方法见 [工程交付记录](../training/fall-streaming-2026-09-23.md)。
