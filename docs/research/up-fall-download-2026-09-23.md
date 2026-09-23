# UP-Fall 公开 RGB 小样本下载核查（2026-09-23）

已从官方页面列出的 Google Drive 入口完整下载 **2 个原始图像 ZIP，共 187,023,780 字节（约 187 MB）**，低于本来源 300 MB 预算。每张 PNG 均成功解码；另生成两个保留全部帧的 MP4，并逐帧回读验证。它们是补充素材，还没有训练新模型，也没有形成事件级验证结论。

文件位于 `datasets/private/web-hard-negatives-20260923/up-fall/`。`manifest.json` 保留原下载地址、原人物/活动/试验/相机编号、source_group、许可状态和 SHA-256；每个 `.frames.json` 保存全部原 PNG 文件名与时间戳。

## 素材与实测

| 本地文件前缀 | 原始活动 | 指定受试者 / 相机 | PNG 全量解码 | 原时间戳跨度 | ZIP 字节数 |
|---|---|---|---:|---:|---:|
| `Subject1Activity9Trial1Camera1` | 捡起物品，弯腰负例 | Subject1 / Camera1 侧面 | 181 / 181 | 9.924861 秒 | 92,159,673 |
| `Subject1Activity1Trial1Camera2` | 双手支撑向前跌倒 | Subject1 / Camera2 正面 | 195 / 195 | 9.880677 秒 | 94,864,107 |

活动编号以原作者论文 Table 5 为准：Activity 9 是 Picking up an object，Activity 8 才是 Sitting，Activity 10 是 Jumping。相机方位和标称 640×480、18 Hz 来自 Table 6；本批实际 PNG 分辨率均为 640×480，帧间隔以文件名为准。[原作者论文 PDF](https://mdpi-res.com/d_attachment/sensors/sensors-19-01988/article_deploy/sensors-19-01988.pdf?version=1556443196)

本地检查了两份联系图；捡物动作位于较早帧，稀疏的五帧联系图会遗漏它，因此额外保存 `Subject1Activity9Trial1Camera1-contact-dense.jpg`。可见约第 18–36 帧发生弯腰取物，随后站起。跌倒片段可见向前落到软垫并保持俯卧。该检查用于确认素材内容，未把抽样帧当作精确的事件起止标注。

### 可追溯下载地址

- 捡物原官方链接：[Subject1 / Activity9 / Trial1 / Camera1](https://drive.google.com/a/up.edu.mx/uc?id=1GLGUqi8r8Jw6oG1MQq7tTd1goigKtG_f&export=download)；实际公开响应地址为 [Google usercontent](https://drive.usercontent.google.com/download?id=1GLGUqi8r8Jw6oG1MQq7tTd1goigKtG_f&export=download)。
- 跌倒原官方链接：[Subject1 / Activity1 / Trial1 / Camera2](https://drive.google.com/a/up.edu.mx/uc?id=1Lh9EMMbDu1Z8yooJVXc4FeAHrOFuR2NL&export=download)；实际公开响应地址为 [Google usercontent](https://drive.usercontent.google.com/download?id=1Lh9EMMbDu1Z8yooJVXc4FeAHrOFuR2NL&export=download)。

以上文件 ID 从[官方 HAR-UP 页面](https://sites.google.com/up.edu.mx/har-up/)的嵌入式 Subject/Activity/Trial 下载表读取。先用 Range 请求验证真实 ZIP 文件头及总长度，再从已观察到的公开重定向地址下载全量内容。前置 Google 页面中途出现 TLS `UNEXPECTED_EOF`，公开对象地址随后下载成功；没有登录、绕过身份验证或发送访问申请。

### SHA-256

| 文件 | SHA-256 |
|---|---|
| `Subject1Activity9Trial1Camera1.zip` | `f99b453bff0cfdabf79d04627e0cc470d43df58aa4df025c2404b0cfa7398a55` |
| `Subject1Activity9Trial1Camera1.mp4` | `ffc96dd3e489ce72c77439d6fdd9213c73eeae85a62aed34494cdaa267557303` |
| `Subject1Activity1Trial1Camera2.zip` | `b45d7cc85ceb01bb30a182f69055b929218793dbd3eb1ab7d911bf7170d731da` |
| `Subject1Activity1Trial1Camera2.mp4` | `1f9b8e51840718c28632f414b7d7a27bbbcb28bf9c4f90e627ce84a20a7e2b22` |

MP4 将所有原帧按文件名时间戳排序，使用 `(帧数 − 1) / 首末帧跨度` 作为恒定帧率；两份视频分别为约 18.1363 和 19.6343 fps，完整回读分别为 181 和 195 帧。CFR 是便于现有流程读取的时间近似，原 PNG 时间戳和 ZIP 完整保留；文件名没有时区，未擅自补 UTC。

## 数据使用依据

官网明确称数据公开可用，给出按试验的相机图像 ZIP 入口，并要求使用数据时引用原论文。可据此记录为公开研究发布、要求引用。[官方数据说明与下载页](https://sites.google.com/up.edu.mx/har-up/)

**未找到明确覆盖原始 RGB 数据的标准许可证。** 原论文的 CC BY 条款对象是论文；项目代码仓库的 MIT 文本对象是软件和相关文档。两者均不能自动填入本批原始数据的许可字段。[原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC6539235/)、[项目仓库](https://github.com/jpnm561/HAR-UP)、[代码 MIT LICENSE](https://raw.githubusercontent.com/jpnm561/HAR-UP/master/LICENSE)

元数据采用：`download_access=official_public_download`；`research_use_basis=official_public_research_release_with_citation`；`data_license=not_explicitly_stated`；`redistribution_or_commercial_permission=not_verified`。数据保存在本地 private 目录，没有发布原始文件。

引用：Lourdes Martínez-Villaseñor, Hiram Ponce, Jorge Brieva, Ernesto Moya-Albor, José Núñez-Martínez, Carlos Peñafort-Asturiano. *UP-Fall Detection Dataset: A Multimodal Approach*. Sensors 2019, 19(9), 1988. DOI: [10.3390/s19091988](https://doi.org/10.3390/s19091988)。

## 人物、摄像头与分组边界

- 本批只有 **1 名指定受试者 Subject1、2 个原相机视角、2 个试验来源组**。未声称两段来自不同人物，也未声称完成新人物、新摄像头独立验证。
- 人物保留 `up-fall-subject01`；摄像头保留 `up-fall-camera01` 与 `up-fall-camera02`，不能按片段另造 camera ID。
- 来源组为 `up-fall-subject01-activity09-trial01` 和 `up-fall-subject01-activity01-trial01`。同一试验若以后补另一相机，必须仍归同一 source_group。
- 跌倒片段还有可见的辅助人员，其稳定身份未在本记录核实；指定受试者 ID 不等于所有可见人物身份已知。
- 本批 split 为 `unassigned`。没有将同一人物分到训练与测试，也没有把短片或相机视角计作额外独立受试者。
- 原 trial 活动类别是粗标签，尚缺精确跌倒起止/恢复标注；不能据此直接报告事件漏报或每小时误报。后续应先补标注、审查全部可见人物，再纳入分组清单。
