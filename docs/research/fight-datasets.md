# 人员打架视频识别数据集调研

更新日期：2026-08-25

## 结论

当前项目不能把 RWF-2000 当作“马上可以下载的数据集”：其[作者官方仓库](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection)明确写明，视频因隐私要求目前不提供。

面向校园固定监控，建议分四层使用数据：

1. **立即可用的起步集**：Surveillance Camera Fight Dataset（300 段，仓库内直接提供）+ AIRTLab 难负样本；它们适合先跑通短窗口二分类，但规模和场景都不足以承担最终验收。
2. **应尽快申请/下载的主数据**：NTU CCTV-Fights 的 280 个 CCTV 子集用于真实打架及时间边界，UBI-Fights 的固定机位子集用于长视频和正常背景。两者最接近固定 CCTV，但都有来源偏差和许可约束。
3. **跨域辅助而非最终测试**：XD-Violence 用于大规模预训练，UCF-Crime 用于长视频弱监督/异常检测；Hockey Fight、Movies Fight 只保留为历史基准，不应用它们证明校园可用性。
4. **打架参与者定位**：AVA、UT-Interaction、Edinburgh BEHAVE Interactions、CAVIAR 提供人物框、时空区间或人物身份关联，适合训练/验证“具体谁在打架”的定位支路；它们规模小或域偏差大，不能替代真实 CCTV 的整段打架分类主数据。

本轮新增核实 13 个候选（编号 11–23）。其中 Violent-Flows 也常被称为 Crowd Violence，它们是同一个 246 视频数据集，不能重复计数。NWPU Campus、IITB-Corridor、ShanghaiTech Campus、ComplexVAD 和 UBnormal 属于异常检测/困难负样本，不应直接与 PP-TSM 的 fight/non-fight 短片主训练集混为一谈。

最终上线门禁必须使用项目自采、按摄像头隔离的校园回放集。公开视频不能覆盖本校机位高度、校服、课间拥挤、夜间红外、码率和遮挡。

## 数据集逐项核对

### 1. RWF-2000

- **规模与标签**：2,000 个 5 秒片段，约 300,000 帧；1,000 violent、1,000 non-violent；视频级二分类。论文给出的官方划分为 80% 训练、20% 测试，并按原始源视频去重，避免同源片段跨集合。[原始论文](https://arxiv.org/html/1911.05913v3)
- **场景/视角**：作者从公开网络视频中筛出真实公共场所监控画面，包含远距离小目标、低分辨率、黑暗、拥挤、模糊和短暂动作；这比体育和电影数据更接近校园固定 CCTV。
- **音频**：原始论文的比较表将其列为无音频，项目应按纯视觉集处理。
- **下载**：**目前不可从官方仓库下载视频**；仓库只保留预处理、网络和权重代码，并明确说明视频因隐私要求不提供。[作者仓库](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection)
- **许可/限制**：未经 SMIIP Lab 批准不得修改、再分发或商用；不得以损害心理健康或个人隐私的方式使用。
- **建议用途**：若未来从作者获得正式授权，可作为短窗口 CCTV 训练/外部测试集。当前只能把其论文结果作为参考，不能把第三方转载下载视为已获授权的数据来源。
- **主要偏差**：片段从网络视频裁出；violent 定义不只打架，还可能包含抢劫、爆炸、枪击、血腥和袭击；5 秒整段标签无法精确学习 START/END。

### 2. UCF-Crime

- **规模与标签**：1,900 个长且未裁剪的监控视频，共 128 小时；13 类异常加 normal。Fighting 只有 50 个视频，其中论文表格标出 45 个在训练集。训练标签是视频级弱标签；测试集另有时间区间标注。[CVPR 2018 原始论文](https://openaccess.thecvf.com/content_cvpr_2018/html/Sultani_Real-World_Anomaly_Detection_CVPR_2018_paper.html)
- **场景/视角**：真实、长时、固定监控为主，分辨率和环境多样，适合模拟线上长流中的稀疏事件。
- **音频**：原始数据集和方法未把音频作为受支持模态；后续 XD-Violence 原始论文的比较表也将 UCF-Crime 列为无音频。按纯视觉处理。
- **下载**：UCF 官方项目页提供完整 ZIP、分块 Dropbox、测试时间标注、异常检测划分和四折动作识别划分；官方还提示压缩包内 `Anomaly_Train.txt` 有损坏，应使用项目页修正版。[UCF 官方项目页](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/)
- **许可/限制**：官方项目页未给出明确的数据许可证。可以用于论文复现不等于得到商用或再分发权；上线或向客户交付模型前需向 UCF/作者确认。
- **建议用途**：适合弱监督长视频预训练和外部长流测试；不适合独自训练 fight 专用分类器，因为 fight 阳性只有 50 个。
- **主要偏差**：异常类别混杂；视频级训练标签噪声大；来源和压缩可能成为捷径；同一场景或源视频必须分组划分。

### 3. XD-Violence

- **规模与标签**：4,754 个未裁剪视频、217 小时；2,405 violent、2,349 non-violent。官方划分为 3,954 训练、800 测试（500 violent、300 non-violent）；训练为视频级弱标签，测试提供时间标注；violent 视频可有 1–3 个标签，类别含 abuse、car accident、explosion、fighting、riot、shooting。[原始论文](https://arxiv.org/abs/2007.04687)
- **场景/视角**：电影、体育、游戏、新闻、直播、CCTV、手持、行车记录仪等混合来源，并非 CCTV 专集。
- **音频**：明确保留音频，作者同时提供 VGGish 音频特征；这是本清单中最适合研究视听融合的数据集。
- **下载**：作者项目页提供训练/测试视频、测试标注、I3D RGB/Flow 与 VGGish 特征的百度网盘、阿里云盘和 OneDrive 链接；部分旧链接已失效，应以当前项目页为准。[作者项目页](https://roc-ng.github.io/XD-Violence/)
- **许可/限制**：作者项目页没有明确许可证文本。数据包含电影和网络内容，商用、再分发和派生发布均需单独确认。
- **建议用途**：适合大规模表征预训练、弱监督和可选音频分支；校园第一版应保持纯视觉主线，避免依赖真实摄像头通常不稳定或不可采集的音轨。
- **主要偏差**：来源域极杂，电影剪辑、声音和背景可能成为捷径；“violence”远宽于人员打架；测试集分布不是校园固定监控分布。

### 4. Hockey Fight

- **规模与标签**：1,000 个约 1.6–1.96 秒片段，500 fight、500 non-fight，约 360×288；视频级二分类。[原始论文](https://doi.org/10.1007/978-3-642-23678-5_39)
- **场景/视角**：全部来自 NHL 冰球比赛，电视转播相机通常会在打架时拉近，背景、衣着和运动规则高度单一。
- **音频**：原始对比资料将其列为无音频。
- **下载**：作者历史下载页位于 Visilab；页面当前可达性不稳定。[作者实验室项目页](https://visilab.etsii.uclm.es/?page_id=1397)
- **许可/限制**：作者页面未给标准数据许可证；素材来自 NHL 转播，不能把“能下载”理解为可商用或可再分发。
- **建议用途**：只用于训练管线冒烟测试和与旧论文对比，不进入校园最终测试。
- **主要偏差**：电视转播缩放、冰场背景、护具、多人聚集与真实校园监控差异极大，模型很容易学习场景捷径。

### 5. Movies Fight

- **规模与标签**：200 个约 1.6–2 秒片段，100 fight、100 non-fight；视频级二分类。fight 来自动作电影，non-fight 来自其他公开动作数据。[原始论文](https://doi.org/10.1007/978-3-642-23678-5_39)
- **场景/视角**：电影剪辑与部分体育画面，镜头语言、画质和类别来源不一致。
- **音频**：原始对比资料将其列为无音频。
- **下载**：与 Hockey Fight 共用作者历史下载页；当前可达性不稳定。
- **许可/限制**：未给标准许可证，且包含电影素材，版权风险高。
- **建议用途**：只做历史基准或代码冒烟测试，不做训练主集和部署验收。
- **主要偏差**：规模太小，fight 与 non-fight 的来源域不一致；原始 RWF-2000 论文已指出现代深度模型在该集上严重过拟合。

### 6. Surveillance Camera Fight Dataset（Surveillance Fight / SCFD）

- **规模与标签**：300 个 2 秒片段，150 fight、150 non-fight；视频级二分类。fight 样本包含物击、踢、拳击、摔跤，环境覆盖咖啡馆、街道、公交等。[作者官方仓库](https://github.com/seymanurakti/fight-detection-surv-dataset)
- **场景/视角**：作者优先选取无背景运动的监控视频，整体接近固定 CCTV，但样本来自 YouTube 剪辑。
- **音频**：官方 README 未承诺音频或同步质量，当前项目按纯视觉处理。
- **下载**：视频直接存放在作者 GitHub 仓库的 `fight/` 与 `noFight/` 目录，是本次调研中最容易立即启动的真实监控短片集。
- **许可/限制**：仓库根目录声明 MIT；但视频来自 YouTube，README 未说明原视频权利链。MIT 仓库许可不应被扩大解释为第三方视频的商业授权。
- **建议用途**：立即用于第一版短窗口分类、训练脚本和回放闭环；必须另留整源隔离的外部测试，不能把仓库随机拆帧后得到的高准确率当部署证据。
- **主要偏差**：只有 300 段；正负片段可能来自不同源视频；网络剪辑、水印、码率与类别可能相关；短片没有长时间正常背景。

## 更适合固定 CCTV 的补充数据集

### 7. NTU CCTV-Fights

- **规模与标签**：1,000 个真实打架视频、17.68 小时、2,414 个打架实例，提供每个实例精确起止点。其中 CCTV 子集为 280 视频、8.54 小时、747 实例；其余 720 个主要来自手机，也含行车记录仪、无人机/直升机。[NTU ROSE Lab 官方页](https://rose1.ntu.edu.sg/dataset/cctvFights/)
- **场景/视角**：CCTV 子集是真实监控，视频 5 秒到 12 分钟；这是本项目最有价值的真实打架时间定位数据。
- **音频**：官方页没有把音频定义为稳定模态；按纯视觉使用，并在获批下载后用 `ffprobe` 盘点音轨。相关原始比较论文提示该类网络视频中不少为静音或只有背景音乐。
- **下载与许可**：需在 ROSE Lab 注册、提交申请并接受 Release Agreement。仅限教育/研究机构非商业研究；未经书面许可不得再分发、生成派生数据集或商用。
- **建议用途**：申请后只取 280 个 CCTV 子集做真实 fight 微调和外部测试；其余 720 个只做泛化预训练。由于全集都是含 fight 的视频，必须配正常长视频，不能独立训练二分类器。
- **主要偏差**：来自 YouTube 搜索，存在地域、压缩、标题筛选和剪辑偏差；CCTV 只占 28%；缺少同机位日常正常视频。

### 8. UBI-Fights

- **规模与标签**：1,000 个视频、80 小时；216 个含 fight，784 个为正常日常活动；全量帧级标注，统一 640×360、30 FPS。文件名标出 indoor/outdoor、fixed/rotated/movable、RGB/grayscale。[作者官方页](https://socia-lab.di.ubi.pt/EventDetection/)
- **场景/视角**：包含室内、室外、固定、旋转和移动摄像机；可用文件名筛出 `camera=0` 固定机位，但作者页未公布其数量。
- **音频**：官方未声明，不作为训练依赖。
- **下载与许可**：作者页提供直接下载；页面未给标准 SPDX/Creative Commons 数据许可证，商用和再分发前需书面确认。
- **建议用途**：长视频时间边界学习、大量 normal 负样本和固定机位独立域验证；类别不平衡应在采样/损失层处理，而不是删掉大部分正常视频。
- **主要偏差**：网络视频来源不统一；固定机位占比未知；水印、颜色、压缩和来源可能被模型利用。

### 9. AIRTLab Violence Dataset（难负样本补充）

- **规模与标签**：350 个 H.264、1080p、30 FPS 短片；230 violent、120 non-violent，平均 5.63 秒。每个动作由两个机位同步拍摄，因此实质上是 175 个表演的双视角，而不是 350 个独立事件。[作者官方仓库](https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos)
- **场景/视角**：同一房间、2–4 名非专业演员、两个固定机位。violent 含踢、拳、掌击、棍击、刺击和枪击；non-violent 特意包含拥抱、击掌、鼓掌、庆祝和手势。
- **音频**：作者没有把音频定义为数据特征，按纯视觉处理。
- **下载与许可**：GitHub 直接下载；作者声明免费用于研究和教育并要求引用。关联开放论文采用 CC BY 4.0，但仓库没有单独标准 LICENSE 文件时，商用范围仍应确认。
- **建议用途**：优先作为“快速大动作但非打架”的困难负样本，帮助降低误报；同一动作的 cam1/cam2 必须放在同一 split。
- **主要偏差**：完全摆拍、单房间、演员少、画质远好于校园 CCTV；violent 中武器/枪击不等同于普通人员打架。

### 10. Bus Violence（仅当覆盖校车/公交场景）

- **规模与标签**：1,400 个 H.264 片段，700 violence、700 non-violence；三个机位/分辨率，动作由演员在行驶公交中模拟。[作者项目页](https://ciampluca.github.io/bus_violence_dataset/)
- **场景/视角**：车内监控，背景和光照随车辆移动；比建筑固定 CCTV 更接近校车，不适合代表走廊或操场。
- **音频**：官方任务按视觉处理，不把音频作为可靠模态。
- **下载与许可**：Zenodo 官方托管；CC BY-NC 4.0，禁止商业使用。
- **建议用途**：若产品范围包含校车，作为专项微调/外部测试；否则不进入第一阶段。
- **主要偏差**：全部摆拍且只在一辆公交、短时段内采集；多机位同一表演必须按事件分组，不能随机拆分。

## 本轮新增候选（11–23）

### 11. Violent-Flows / Crowd Violence（新增）

- **规模与标签粒度**：246 个短视频，violent/non-violent 视频级二分类；最短 1.04 秒、最长 6.52 秒、平均 3.60 秒。作者页同时把任务描述为分类和 violence outbreak detection，但下载包的核心监督仍是片段级标签。[作者官方页](https://talhassner.github.io/home/projects/violentflows/index.html)
- **场景/视角**：全部来自 YouTube 的真实群体场景，强调人群整体运动；混合手持、新闻和远景画面，不是纯固定 CCTV。
- **下载与许可**：作者页要求填写信息后取得 FTP 密码，当前仍列出 `Movies.zip`；页面没有给标准数据许可证，商用、再分发和派生发布前必须向作者确认。
- **适用任务**：可作为 PP-TSM 的“群体冲突”小型外部测试或辅助训练，不适合参与者定位，因为没有人物框/轨迹。`Crowd Violence` 是该集常用别名，不应再作为独立数据集计数。
- **主要偏差**：只有 246 段、持续时间很短；类别可能被镜头抖动、群体密度和新闻画质区分；群体骚乱与校园两三人打架不是同一任务。

### 12. Real-Life Violence Situations（RLVS，新增）

- **规模与标签粒度**：2,000 个短视频，1,000 violence、1,000 non-violence，视频级二分类；原始论文报告该集用于短片暴力识别。[原始论文](https://doi.org/10.1109/ICICIS46948.2019.9014714)
- **场景/视角**：作者托管页说明 violence 主要是真实街头打架，normal 包含体育、进食、走路等；视频主要来自 YouTube，视角与画质混杂。[作者 Kaggle 托管页](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)
- **下载与许可**：Kaggle 可直接下载，当前元数据显示 `Data files © Original Authors`，不是 CC0/开源授权。转载到 Hugging Face、其他 Kaggle 账号或 Roboflow 的副本不会扩大原许可。
- **适用任务**：规模和二分类形式适合 PP-TSM 起步微调与跨域验证；没有时间边界、人物框或轨迹，不适合直接训练参与者定位。
- **主要偏差**：网络来源、电影/摆拍/真实场景可能混合；positive 与 normal 的来源域可能不同；不能把其随机 clip split 当作校园 CCTV 泛化证据。

### 13. AVA Actions v2.2（新增，定位优先）

- **规模与标签粒度**：430 个电影视频，每个取 15 分钟，在 1 秒间隔的关键帧对人物框标注动作；官方汇总为 80 类、约 162 万动作标签。v2.2 划分为 235 train、64 val、131 test，每行含 `person_box/action_id/person_id`。[Google 官方页](https://sites.research.google/gr/ava/)与[下载格式](https://sites.research.google/gr/ava/download/)
- **与打架相关的标签**：官方 label map 明确含 `fight/hit (a person)`、`kick (a person)`、`push (another person)`、`grab (a person)`，同时含 `hug`、`hand shake`、`martial art` 等困难负类。[官方标签表](https://research.google.com/ava/download/ava_action_list_v2.2.pbtxt)
- **场景/视角**：电影画面，视角、剪辑和构图丰富，但不是固定 CCTV。
- **下载与许可**：Google 提供标注包和 YouTube ID；官方页声明所列数据集采用 CC BY 4.0。原视频可因 YouTube 下架而缺失，且电影素材的底层权利风险应单独评估。
- **适用任务**：本清单中最适合预训练“人物框 → 多标签动作”的参与者定位头，也可用 person_id 做相邻关键帧关联；不建议直接拿来微调当前整幅画面 PP-TSM 二分类器。
- **主要偏差**：电影近景、剪辑、表演强度与固定监控差异大；1 秒稀疏关键帧不是逐帧框，需插值/跟踪并在校园域复核。

### 14. UT-Interaction（新增，定位优先）

- **规模与标签粒度**：20 个约 1 分钟连续视频，6 类人际互动：握手、指向、拥抱、推、踢、拳击；每视频平均约 8 次互动。官方提供互动起止区间和空间包围框，也提供 120 个裁剪片段用于分类。[UT Austin 官方挑战页](https://cvrc.ece.utexas.edu/SDHA2010/Human_Interaction.html)
- **场景/视角**：720×480、30 FPS；10 段停车场、10 段草地，含路人干扰、多人同时互动、树木运动和相机抖动。
- **下载与许可**：官方页直接提供 set1/set2、ground truth 和 segmented 数据；页面要求引用，但没有标准许可证或明确商业授权文本，商用前需确认。
- **适用任务**：非常适合小规模验证“检测人物 → 组成互动对 → 时空定位”的接口，以及用拥抱/握手作为难负样本；样本太小，不应成为 PP-TSM 主训练集。
- **主要偏差**：完全摆拍、只有两个场地、人物约 200 像素高，远大于校园广角 CCTV 中的小目标；六类动作每类有效执行有限。

### 15. Edinburgh BEHAVE Interactions（新增，定位优先）

- **规模与标签粒度**：4 个长 clip、两个视角，25 FPS、640×480；10 类多人互动含 Fight、Chase、Meet、WalkTogether 等。行为标注含群组成员 ID、起止帧和标签；很多但不是全部序列另有人物 VIPER XML 框。[爱丁堡官方数据页](https://groups.inf.ed.ac.uk/vision/DATASETS/BEHAVEDATA/INTERACTIONS/index.html)
- **场景/视角**：研究团队在户外场地摆拍，两视角、多人物和群组关系比单个二分类 clip 更适合验证“谁参与了事件”。
- **下载与许可**：官方页直接提供 4 个 WMV、子片段、JPEG 和部分 ground truth，并写明 researchers can freely use、发表需引用；未给标准 SPDX/Creative Commons 版本，商业使用仍建议书面确认。
- **适用任务**：适合参与者 ID 关联、群组级 fight 定位和 Chase/Fight 区分；不适合作为 PP-TSM 主训练集，且使用前必须盘点哪些子片段确有完整人物框。
- **主要偏差**：数据小、摆拍、背景单一，ground truth 不完整；同一事件的双视角必须放同一个 split。

### 16. CAVIAR（新增，定位优先）

- **规模与标签粒度**：官方页提供多种走路、会面、购物、遗包和打架场景；与本项目最相关的是 4 个 fight 序列：`Fight_RunAway1/2`、`Fight_OneManDown`、`Fight_Chase`，均有人物/群组手工 XML ground truth，部分序列还有头、手、脚、肩等额外点。[CAVIAR 官方数据页](https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/)
- **场景/视角**：384×288、25 FPS，INRIA 大厅固定广角；另一购物中心部分是同步双视角。打架序列为演员摆拍。
- **下载与许可**：MPEG2、JPEG 和 XML 均可从官方页直接下载；页面明确标为 Creative Commons BY-SA，但未在文字中注明具体版本，使用时保存页面与许可快照。
- **适用任务**：四个 fight 序列可做参与者框、群组框和 START/END 事件的黄金回归样例；规模远不足以训练 PP-TSM，最适合作为定位与证据视频单元测试。
- **主要偏差**：2003–2004 年低分辨率素材、场景和演员极少，开头还有场次手势；不能用其准确率代表现代校园摄像头。

### 17. Bullying10K（新增，隐私/定位研究）

- **规模与标签粒度**：10,000 个 2–20 秒 DVS 事件片段、约 120 亿事件、255 GB；6 个暴力动作（拳击、踢、抓头发、勒颈、推、扇耳光）和 4 个友好动作。提供动作分类、时间动作定位和 26 关键点姿态三类基准。[原始论文](https://arxiv.org/abs/2306.11546)
- **场景/视角**：两台 DAVIS346 事件相机、左右双视角，亮/暗两种光照；每段两名演员，动作均为受控摆拍，DVS 只记录亮度变化而不是普通 RGB。
- **下载与许可**：作者在 [Figshare 官方托管页](https://figshare.com/articles/dataset/Bullying10k/19160663)发布，当前页面显示下载约 46.26 GB 的打包版本并标为 CC BY 4.0；论文描述原始全量体积为 255 GB。
- **适用任务**：适合研究隐私友好的动作定位、姿态与高速运动，不可直接喂给当前 RGB PP-TSM；若未来采用事件相机，可作为独立技术路线。
- **主要偏差**：传感器域与普通摄像头完全不同；固定两人摆拍、动作类别有限，不能直接迁移到拥挤校园 RGB 视频。

### 18. NWPU Campus（新增，校园异常/困难负样本）

- **规模与标签粒度**：43 个场景、28 类异常、547 个视频（305 train、242 test）、1,466,073 帧/16.29 小时；训练为正常视频，测试含帧级正常/异常，异常类含 chasing、scuffle、battering、group conflict、falling、protest 等。[作者官方页](https://campusvad.github.io/)
- **场景/视角**：真实校园固定监控，25 FPS；含“同一行为在不同场景可能正常或异常”的 scene-dependent 标签，是本项目最接近部署环境的异常压力测试之一。
- **下载与许可**：作者页提供百度网盘和 Google Drive，共约 76.6 GB；仅限教育/研究机构的非商业学术研究，禁止未经许可再分发、派生新数据集和商业使用。
- **适用任务**：优先作为校园域困难负样本、长视频误报测试和场景依赖验证；可抽取明确 scuffle/group conflict 区间做外部评估，但不能直接把所有 anomaly 当 fight 训练 PP-TSM，也没有参与者框。
- **主要偏差**：多数异常是保护下摆拍；标签语义远宽于打架；只看二分类会把骑车、违停、狗等都错误折叠为 violence。

### 19. IITB-Corridor（新增，校园异常/困难负样本）

- **规模与标签粒度**：483,566 帧的单走廊监控数据；异常/群体行为含 protest、chasing、fighting、sudden running，另含隐藏面部、徘徊、遗包、可疑物、骑车；作者以 frame-level AUC 评估。[IIT Bombay 作者项目页](https://rodrigues-royston.github.io/Multi-timescale_Trajectory_Prediction/)
- **场景/视角**：IIT Bombay 校园走廊固定机位，适合检验课间走廊中的追逐、突然奔跑和球类活动误报。
- **下载与许可**：作者页提供 train/test Google Drive 和密码，并明确仅供研究使用；没有标准许可证，不能默认商用或再分发。
- **适用任务**：用作长视频困难负样本、fight/chasing/protest 区分和检测延迟压力测试；不是短片二分类主数据，也没有作者页明确的人物参与者框。
- **主要偏差**：单场景、机位和人群构成单一；异常类型混杂，可能学到走廊背景或轨迹捷径。

### 20. ShanghaiTech Campus（新增，校园异常/困难负样本）

- **规模与标签粒度**：13 个校园场景、330 个正常训练视频和 107 个测试视频，测试含 130 个异常事件；官方页提供异常事件像素级 ground truth，异常含 chasing 和 brawling。[作者实验室官方页](https://svip-lab.github.io/dataset/campus_dataset.html)
- **场景/视角**：真实校园多固定机位，光照、角度和人物尺度多样，比单场景 Avenue/UCSD 更接近本项目。
- **下载与许可**：官方页提供 Google Drive/OneDrive，但页面未显示明确数据许可证文本；第三方代码仓库的许可证不能替代视频数据许可，商用/再分发前应向作者确认。
- **适用任务**：适合校园域异常定位、长时间正常背景和 brawling/chasing 外部测试；训练集只有 normal，不能直接作为监督式 fight/non-fight 主训练集。像素掩码可用于验证异常区域，但不等价于“参与打架的人物 ID”。
- **主要偏差**：异常类别混杂且阳性只在测试集；若用测试异常训练会破坏标准协议；像素异常掩码与人物框/轨迹语义不同。

### 21. UBnormal（新增，合成定位/困难负样本）

- **规模与标签粒度**：543 个合成视频、29 个虚拟场景、22 类异常；train/val/test 的异常类型按 open-set 方式拆开。对正常和异常对象提供像素级分割、对象类别和异常标注。[原始论文](https://openaccess.thecvf.com/content/CVPR2022/html/Acsintoae_UBnormal_New_Benchmark_for_Supervised_Open-Set_Video_Anomaly_Detection_CVPR_2022_paper.html)与[作者仓库](https://github.com/lilygeorgescu/UBnormal)
- **场景/视角**：Cinema4D 合成的固定监控式场景，30 FPS、最低 720 像素高；不含真实人的隐私问题。
- **下载与许可**：作者仓库提供 Google Drive，明确为 CC BY-NC-ND 4.0；禁止商业使用和发布改编版本。
- **适用任务**：可做异常区域分割、开放集验证和合成困难负样本，不适合直接微调当前真实 RGB fight PP-TSM，也不能替代参与者动作标签。
- **主要偏差**：显著 sim-to-real 域差；“异常”不等于打架；无衍生许可会限制重新标注、重打包或发布派生版本的工作流。

### 22. MIVIA Action Dataset（新增，但不推荐用于打架）

- **规模与标签粒度**：14 名受试者（7 男、7 女），7 类高层动作，每人每类 2 次；Kinect 深度序列，动作含开罐、喝水、睡觉、随机移动、停止、与桌子互动、坐下等。[MIVIA 官方页](https://mivia.unisa.it/datasets/video-analysis-datasets/mivia-action-dataset/)
- **场景/视角**：室内受控 Kinect 深度数据，没有 fight/punch/kick，也不是 CCTV RGB。
- **下载与许可**：作者页提供整包下载并要求引用，但没有标准数据许可证文本。
- **适用任务**：最多用于动作识别代码的深度模态冒烟或少量普通动作负样本研究；不适用于当前 PP-TSM 主线，也不支持打架参与者定位。
- **主要偏差**：传感器、动作语义和场景与项目目标均不匹配。网上把 MIVIA 泛称为“violence dataset”的转载没有作者页证据，列为**不推荐**。

### 23. ComplexVAD（新增，参与对象框/轨迹工具优先）

- **规模与标签粒度**：104 个训练视频、113 个测试视频，共 3,681,438 帧；测试集含 118 个异常、40 类异常。每个异常参与人/物在每帧都有框和 track ID，可做 frame、region 和 track 三级评估。[原始论文](https://openaccess.thecvf.com/content/WACV2025W/ASTAD/papers/Mumcu_ComplexVAD_Detecting_Interaction_Anomalies_in_Video_WACVW_2025_paper.pdf)
- **场景/视角**：University of South Florida 校园道路与人行横道的单固定机位，1920×1080、30 FPS，跨 5 个月早/中/下午采集；人脸经过模糊。异常重点是两个人或人与物之间的交互，例如两人相撞、破坏车辆、遗留包裹。[MERL 官方下载页](https://merl.com/research/downloads/ComplexVAD)
- **下载与许可**：MERL 官方页指向 Zenodo 直接下载；原始论文与补充材料明确为 CC BY-SA 4.0。
- **适用任务**：不是 fight 主数据，但非常适合先把“异常参与对象框 → track → 事件证据”的数据管线、区域/轨迹指标和可视化跑通；比只给像素掩码的异常集更接近项目的参与者定位接口。
- **主要偏差**：单场景且主要不是打架；位置与背景先验很强。即使定位工具在该集通过，也不能说明模型能识别校园打架参与者。

## 推荐组合与划分

### 总推荐顺序（按当前项目价值）

| 顺序 | 数据/组合 | 在当前项目中的角色 | 结论 |
|---|---|---|---|
| 1 | 校园自采、按摄像头隔离的数据 | 最终验收、困难负样本、参与者框/轨迹 | **不可替代**；公开视频只能帮助预训练和外部测试 |
| 2 | NTU CCTV-Fights + UBI-Fights fixed/normal | 真实 CCTV 主阳性、时间边界、长正常背景 | **优先申请/下载**；严格遵守非商用与再分发限制 |
| 3 | Surveillance Camera Fight + AIRTLab hard negatives | 立即跑通 PP-TSM 微调与误报抑制 | **当前最易执行**；规模小且权利链仍需盘点 |
| 4 | AVA → UT-Interaction/BEHAVE/CAVIAR | 参与者动作头预训练 → 小型定位回归 | **定位主线首选公开组合**；最终仍需校园人物框标注 |
| 5 | ComplexVAD | 参与对象逐帧框、track 和区域/轨迹评估工具 | **定位工程工具首选**；没有足够 fight 阳性 |
| 6 | NWPU Campus → IITB-Corridor → ShanghaiTech | 校园长流、场景依赖、追逐/拥挤/体育等压力测试 | **困难负样本与外部测试**；不是 PP-TSM fight 主训练集 |
| 7 | RLVS → Violent-Flows | 网络真实打架/群体冲突的二分类辅助 | 可扩量，但许可和来源偏差使其低于 CCTV 主集 |
| 8 | XD-Violence / UCF-Crime | 大规模表征、弱监督、长流异常压力测试 | 只做辅助，不用它们证明校园打架精度 |
| 9 | Bullying10K / UBnormal | DVS 隐私路线、合成/open-set 定位研究 | 独立研究支线，不直接接当前 RGB PP-TSM |
| 10 | Hockey / Movies / MIVIA Action | 历史基准或代码冒烟 | 不进入部署验收；MIVIA Action 与打架任务不匹配 |

### 第一阶段：现在即可执行

1. 下载 Surveillance Camera Fight Dataset，保留原文件和来源清单。
2. 下载 AIRTLab，只取其困难 non-violent 作为补充，并可少量保留 violent 做跨域验证。
3. 参与者定位先下载 AVA 标注和 UT-Interaction/CAVIAR 小数据；AVA 用于初始化 person-action 头，UT/CAVIAR 只做时空框与事件绑定回归，不把所有人物框染红冒充定位结果。
4. 只先取 ComplexVAD 的标注规范/小样例设计定位接口；完整数据约数十 GB，等定位管线确定后再下载，避免为了“有框”先搬入大数据。
5. RLVS 可作为研究扩量候选，但先将许可标记为 `original-authors-rights`；Violent-Flows 在取得作者下载凭证后只做辅助域。
6. 为每个文件生成 `dataset/source_video/event_group/camera_id/license_status` 元数据；未知许可证标 `research-only-unverified`，不得混进可交付训练产物。
7. 按**原视频或同一表演事件分组**切分 70/15/15；严禁按帧随机切分。训练阶段可从训练视频滑窗，验证/测试只按固定窗口协议生成。

### 第二阶段：申请和扩充

1. 申请 NTU CCTV-Fights；280 个 CCTV 与 720 个非 CCTV 分开建域。
2. 获取 UBI-Fights；按文件名先筛固定机位，并保留其大量 normal 视频。
3. 获取 NWPU Campus、IITB-Corridor 和 ShanghaiTech；保持原始异常检测 split，只用于校园域长流、误报和 brawling/chasing 外部测试，不与短片 fight 二分类训练集直接合并。
4. 用 AVA/UT/BEHAVE/CAVIAR 建立参与者定位基线后，追加项目自采的 `person_track_id + role + start/end + bbox` 标注；公开视频定位集只做预训练/回归。
5. XD-Violence 仅作为大规模预训练；UCF-Crime 仅作为长视频压力测试和异常域辅助。
6. RWF-2000 只有获得作者正式授权文件后才进入数据仓库；不使用 Kaggle、Roboflow 或其他转载包替代授权。

### 最终评估 split

- **训练集**：公开训练域 + 校园自采训练摄像头。
- **验证集**：训练阶段从未见过的源视频；阈值和持续时间只在此调节。
- **公开外部测试**：NTU CCTV 子集或 UBI fixed 子集，按源视频分组且不回流训练。
- **校园验收测试**：至少按 `camera_id` 留出整台摄像头；更严格时再留出整栋楼或整天。测试集在阈值冻结前不可查看结果。
- 同一 YouTube 原视频裁出的多个片段、同一摆拍动作的多机位、镜像/转码/增强版本必须属于同一组。
- 报告至少包含 clip AUROC/AUPRC、事件级 precision/recall/F1、每摄像头每小时误报数、平均发现延迟、漏报事件数；不要只报 balanced accuracy。

## 校园自采困难负样本

校园数据的重点不是再摆拍更多“拳打脚踢”，而是收集最容易误报的正常动作。建议在获得隐私和安全审批后，使用成年人或明确同意的参与者拍摄；不要组织未成年人真实冲突。

- 拥抱、击掌、拍肩、拉手、搀扶、同学追逐嬉闹。
- 篮球/足球抢球、排球扣球、武术社团、舞蹈、广播操、体测和跑步冲线。
- 快速聚集、围观、排队插队但无肢体攻击、老师拉开学生。
- 搬桌椅、挥扫帚/拖把、举伞、甩书包、扔球、推自行车。
- 一人跌倒后多人搀扶，医护处置，保安正常控制或疏散。
- 课间拥挤、食堂交叉遮挡、楼梯上下行、玻璃/镜面反射。
- 夜间红外、背光、雨雪、强压缩、丢帧、网络重连、摄像头轻微震动和画面遮挡。
- 无人空场、树影/旗帜晃动、屏幕播放体育或打架视频，防止“画中画”误报。

采集时同时记录 `camera_id`、位置类型、日期时段、光照、码率/FPS、人数、遮挡等级、动作标签、事件起止、参与者匿名轨迹和授权状态。原视频与标注分权保存；训练导出应去标识化并有保留期限。

## 许可证与使用边界汇总

| 数据集 | 当前取得方式 | 已核实边界 |
|---|---|---|
| Surveillance Camera Fight | GitHub 直接获取 | 仓库 MIT；第三方视频权利链未澄清，商用风险待确认 |
| AIRTLab | GitHub 直接获取 | 作者声明研究/教育免费；建议商用前书面确认 |
| UCF-Crime | UCF 官方下载 | 官方页未给明确数据许可证 |
| XD-Violence | 作者项目页网盘 | 官方页未给明确数据许可证，含电影/网络素材 |
| NTU CCTV-Fights | 注册、申请、签协议 | 仅学术非商业；禁止再分发、派生数据集和商业使用 |
| UBI-Fights | 作者页直接下载 | 无标准许可证文本；商用/再分发前确认 |
| RWF-2000 | 官方当前不提供视频 | 非商用、禁止未经许可修改和再分发；不得绕过官方限制使用转载包 |
| Hockey / Movies | 作者历史页面 | 无标准数据许可证，且含 NHL/电影版权素材 |
| Bus Violence | Zenodo | CC BY-NC 4.0 |
| **Violent-Flows（新增）** | 作者页登记后取得 FTP 密码 | 无标准许可证；YouTube 来源，商用/再分发前确认 |
| **RLVS（新增）** | 原作者 Kaggle 页面直接下载 | `Data files © Original Authors`；不是 CC0，第三方转载不扩大权限 |
| **AVA Actions（新增）** | Google 官方标注包 + YouTube ID | 官方页 CC BY 4.0；底层电影/YouTube 视频可下架且权利需另核 |
| **UT-Interaction（新增）** | UT Austin 官方页直接下载 | 要求引用；无标准许可证或明确商业授权文本 |
| **Edinburgh BEHAVE Interactions（新增）** | 爱丁堡官方页直接下载 | 页面称研究者可自由使用且发表须引用；无标准许可版本，商用前确认 |
| **CAVIAR（新增）** | 爱丁堡官方页直接下载 | Creative Commons BY-SA；官方页文字未注明具体版本 |
| **Bullying10K（新增）** | Figshare 官方托管 | CC BY 4.0 |
| **NWPU Campus（新增）** | 作者页百度网盘/Google Drive | 仅非商业学术研究；禁止未授权再分发、派生数据集和商用 |
| **IITB-Corridor（新增）** | 作者页 Google Drive | 仅研究用途；无标准许可证文本 |
| **ShanghaiTech Campus（新增）** | 作者实验室 Google Drive/OneDrive | 官方数据页未给明确许可；第三方代码许可证不能替代数据许可 |
| **UBnormal（新增）** | 作者仓库 Google Drive | CC BY-NC-ND 4.0，禁止商业使用和发布改编版本 |
| **MIVIA Action（新增，不推荐）** | MIVIA 官方页直接下载 | 要求引用；无标准许可证，而且不含打架动作 |
| **ComplexVAD（新增）** | MERL 官方页指向 Zenodo | CC BY-SA 4.0；可做参与对象框/track 工具，非 fight 主数据 |

许可证结论只用于工程筛选，不构成法律意见。任何训练权重需要进入商业产品前，应保存数据来源、下载日期、许可文本快照与作者批准记录，并由项目法务复核。
