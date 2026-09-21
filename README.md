# 校园安全 AI · 跌倒识别演示

公开演示网页包含两个真实连续视频中的模型识别回放，可播放、全屏和下载 MP4。

[打开视频展示页](https://zero-pimio.github.io/campus-safety-ai/)

- 跌倒动作：3.304 秒 / 100 帧，早期判断存在抖动。
- 日常走路：9.969 秒 / 300 帧，该段 285 个可判断时点未出现跌倒误报。

两段素材均为开发验证数据，不能作为独立测试准确率或校园实地验收。网页不提供实时检测服务。

## 素材来源与许可

[UR Fall Detection Dataset](https://fenix.ur.edu.pl/mkepski/ds/uf.html)，University of Rzeszów / Michał Kępski。使用 cam0 RGB 序列 fall-06、adl-10。

本项目在原始连续画面上添加人体姿态、模型分数和中文说明，并编码为 MP4；封面为相应视频帧。原始素材和本演示视频、封面图按 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) 分享，限非商业用途，保留署名和相同许可。原作者不为本项目效果背书。

请引用：Bogdan Kwolek, Michal Kepski. *Human fall detection on embedded platform using depth maps and wireless accelerometer*. Computer Methods and Programs in Biomedicine, 117(3), 2014, 489–501.

视频 SHA256、时长、帧数和模型指纹见 [media.json](media.json)。

本目录为纯静态网页。GitHub Pages 发布源为 codex/video-showcase 分支的根目录，`.nojekyll` 禁用 Jekyll 处理。也可运行 `python3 -m http.server 8000` 本地预览。
