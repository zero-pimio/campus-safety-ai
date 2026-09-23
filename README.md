# 校园安全 AI · 视频展示

[打开在线视频展示页](https://zero-pimio.github.io/campus-safety-ai/)

提供 5 段真实结果回放，支持选择、播放/暂停、进度、全屏和下载。网页只播放已生成的演示，不运行在线检测。

| 视频 | 时长 / 帧数 | 当前边界 |
| --- | --- | --- |
| 打架识别 | 6.066667 秒 / 182 帧 | 历史候选场景分数；人物框不是逐人打架标签，事件在 EOF 收口 |
| 跌倒持续告警 | 5.601633 秒 / 167 帧 | 新固定窗口候选；轨迹变化导致 END，人仍倒地，不代表恢复 |
| 人员入侵 | 5 秒 / 50 帧 | 真实检测与人工示范禁区；没有入侵真值，事件在 EOF 收口 |
| 历史跌倒识别 | 3.304 秒 / 100 帧 | URFD 历史增长前缀模型，开始阶段有抖动 |
| 历史正常走路 | 9.969 秒 / 300 帧 | 285 个可判断时点没有跌倒误报，不代表长期误报率 |

车辆识别已有 50 秒本地新演示，但素材的派生视频公开再分发范围尚未确认；网页保留来源和复现入口，不上传该 MP4。对应停车规则 0 START / 0 END，不能标为违停验收完成。车辆闯道闸、人员闯门禁仅完成设计，尚无运行视频。

## 素材来源与使用条款

各素材分别适用以下条款，不能把一种许可套用到所有视频。原作者不为本项目效果背书。

- **打架、人员入侵及对应封面**：[AIRTLab 作者数据仓库](https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos)，使用 violent/cam1/101 与 non-violent/cam1/51。作者允许研究与教育用途；本展示为非商业研究演示。保留[用途及引用说明](licenses/AIRTLab-NOTICE.txt)。请引用 M. Bianculli et al., *A dataset for automatic violence detection in videos*, Data in Brief 33 (2020), 106587, [DOI](https://doi.org/10.1016/j.dib.2020.106587)。论文许可证不被当作外部视频的许可证。
- **新跌倒回放及封面**：[GMDCSA24 作者仓库固定版本](https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos/tree/5abac7693229900cf80f722e878fbb119211fc1c)，使用 Subject3/Fall03，作者 Ekram Alam。按作者根仓库 MIT 许可使用，保留[完整版权与许可](licenses/GMDCSA24-MIT.txt)。原数据集 DOI：[10.5281/zenodo.12921216](https://doi.org/10.5281/zenodo.12921216)。
- **历史跌倒、走路及对应封面**：[UR Fall Detection Dataset](https://fenix.ur.edu.pl/mkepski/ds/uf.html)，University of Rzeszów / Michał Kępski，使用 fall-06、adl-10 的 cam0 RGB。本项目添加姿态、模型分数和中文说明，派生视频与封面按 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 分享，保留署名、非商业与相同方式共享要求。请引用 Bogdan Kwolek, Michal Kepski, *Human fall detection on embedded platform using depth maps and wireless accelerometer*, Computer Methods and Programs in Biomedicine 117(3), 2014, 489–501。
- **车辆素材参考**：[UCF CRCV / PNNL Parking Lot](https://www.crcv.ucf.edu/data/ParkingLOT/)。[PNNL 机构通知](https://www.pnnl.gov/important-notices)允许其服务器资料的非商业科研教育分发，但未确认条款与 UCF 托管文件的适用关联，本目录不分发该视频。

所有已发布 MP4 均为保存的实际推理结果回放，没有为展示重训或调整告警阈值。模型、描述及限制见 [catalog.json](catalog.json)；输出 SHA256、帧数、时长、时间戳与完整解码结果见 [verification.json](verification.json)。[media.json](media.json)继续保留两个历史 URFD 文件的旧版元数据。画面播放帧率不等于推理速度，短片也不代表校园精度、RK3588 性能或长期稳定性验收。

## 预览与发布

本目录为纯静态网页；在本目录运行 `python3 -m http.server 8000` 后通过 HTTP 访问。新页面通过 fetch 读取 catalog.json，请勿仅用 file:// 打开。

GitHub Pages 发布源为 `codex/video-showcase` 分支根目录，`.nojekyll` 禁用 Jekyll。此目录在 main 保存源码与同字节媒体，发布时复制到该分支。历史 assets/fall-detection.mp4、assets/walking.mp4 链接保持兼容。
