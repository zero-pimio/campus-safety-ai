# GMDCSA24 困难负例下载核验（2026-09-23）

本轮实际取得 **23 段：16 段 ADL 困难负例、7 段跌倒对照**，覆盖作者的 4 个 subject，视频共 **171,109,337 字节**。23 段全部完整解码，共 5,100 帧，且文件内容均与固定 Git commit 的原始 blob 一致。原计划的另 1 段传输不完整，未计入成功素材。

使用作者直接公开的 GitHub 视频文件，没有注册、提交申请或下载第三方转载。原始来源固定到 commit `5abac7693229900cf80f722e878fbb119211fc1c`，保留原 CSV 与文件名，不改写源标签。[作者仓库](https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos/tree/5abac7693229900cf80f722e878fbb119211fc1c)

本地目录为 `datasets/private/web-hard-negatives-20260923/gmdcsa24/`。最终实际数量、字节数、逐视频 SHA-256 与解码结果以 `inventory.json` 为准；`download-plan.json` 仅是选择计划，不能当作已下载结果。

## 选择依据

| 作者 subject | ADL 文件号 | 对应官方描述中的困难动作 | Fall 文件号 |
| --- | --- | --- | --- |
| Subject 1 | 01、11、15、16 | 主动躺床、走到椅子坐下、拾地面手机后坐下、坐姿拾手机 | 03、08 |
| Subject 2 | 03、08、09、12 | 主动躺床、身体拉伸触地、从床起身后坐地、主动趴床 | 01、04 |
| Subject 3 | 05、06、07、08 | 弯腰锻炼、锻炼后坐地、坐站锻炼、主动躺地 | 03、04 |
| Subject 4 | 05、10、15、17 | 俯卧撑、行走后主动躺床、坐姿转躺床、拾地面书本 | 03；07 传输失败未纳入 |

选择依据来自原始 [Subject 1 ADL.csv](https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos/blob/5abac7693229900cf80f722e878fbb119211fc1c/Subject%201/ADL.csv)、[Subject 2 ADL.csv](https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos/blob/5abac7693229900cf80f722e878fbb119211fc1c/Subject%202/ADL.csv)、[Subject 3 ADL.csv](https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos/blob/5abac7693229900cf80f722e878fbb119211fc1c/Subject%203/ADL.csv)、[Subject 4 ADL.csv](https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos/blob/5abac7693229900cf80f722e878fbb119211fc1c/Subject%204/ADL.csv)。这里的中文描述用于说明选样理由，不是本轮重新完成的动作逐帧标注；没有把坐站锻炼直接改称蹲下，也没有从本数据虚构嬉闹类别。

## 来源、许可与数据质量

仓库根目录提供 MIT LICENSE，版权行为 2024 Ekram Alam；本地完整保留。这里只记录作者仓库公开许可事实，未另行验证肖像同意文件、独立视频权利链或特定部署使用资格。[原始 LICENSE](https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos/blob/5abac7693229900cf80f722e878fbb119211fc1c/LICENSE)

本次轻量 Git tree 实际列出 160 段视频：ADL 81 段、Fall 79 段。这个数字来自上述固定版本的实际目录；不根据介绍中的类别汇总去更改作者文件归类。树结构保存在 `source-tree-paths.txt`。

原 CSV 存在需复核的问题，例如 Subject 2 ADL/06 的 `Sitting[9.7 to 3]` 起止反向，Subject 4 Fall/09 的一个区间缺少 `to`。这些行保留在原 CSV，未暗自修正；它们也不在本轮选择的视频中。即使所选行没有上述格式问题，粗时间区间仍不能直接视为本项目精确的跌倒开始、恢复结束真值。每个下载样本均标记 `annotation_review_required: true`，原 `Description`、`Classes`、时长与光照等字段原样保存在 `source_annotation`。

## 身份、划分与使用边界

人物 ID 仅采用作者的 Subject 1–4 分组，表示来源所声明的 subject。摄像头物理身份没有逐文件可核验映射，因此一律保存 `camera_id: null`；不能给每个文件分配一个新“未知摄像头”并声称跨摄像头独立。原始连续录像之间是否有裁剪关系也未完全核实，所以 `source_group` 保守地按 subject 共组；这不是宣称每段视频都是独立录像。

本轮 `split` 全部为 null，没有自动填 train/val/test，也没有混入原打架或 URFD 清单。文件全帧可解码只验证下载可读性，不证明动作标注正确、任务精度提高或真实校园泛化。下一步先人工复核事件起止、确认原始录像关系和摄像头身份；正式划分还需满足人物/摄像头/原录像分组隔离。

## 下载与验证记录

- `source-metadata/`：固定版本 README、LICENSE、4 个 subject 的 ADL/Fall 共 8 个原 CSV。
- `metadata-sources.json`：元数据来源 URL、字节数、SHA-256。
- `videos/`：保留作者路径的实际视频。
- `inventory.json`：仅汇总成功下载并完整解码的视频；每段含 SHA-256、字节数、帧尺寸、FPS、声明帧数、实际解码帧数、逐段状态与原标签。
- `downloaded.json`：下载阶段记录，包含失败原因；失败下载不计入成功 inventory。
- `download_selected.py`：固定选样、每视频 16 MB 上限、90 秒传输超时、3 路并发的定向抓取与全帧 OpenCV 验证脚本。24 个传输的理论视频总量上限为 384 MB，未抓取整仓视频。

初次批量 HEAD 元数据预取耗时过长，已中断后改为直接有界 GET；没有因此把未完成请求算作下载。可用的首个样本是 Subject 1/ADL/15.mp4（7,233,079 字节）。本轮未启动训练、未修改模型、未将官方原标注自动转成事件评估真值。

唯一失败文件为 Subject 4/Fall/07.mp4，curl 返回 `Transferred a partial file`，保留 5,868,453 字节 `.mp4.part` 并记录在 `failed_downloads`；没有重试或当作有效视频。成功视频和该残片合计 176,977,790 字节，连同少量元数据及 Git tree 远低于 400 MB 上限。

逐帧解码的总时长估计为 171.1343 秒（每段 `decoded_frames / FPS` 求和）；这是素材时长核验，不是可直接计算每小时误报的真实连续曝光。23 个视频的 SHA-256 已保存；另重算 Git blob SHA-1 并对照固定版本的 tree，23/23 相符，记录为 `matches_pinned_git_blob: true`。
