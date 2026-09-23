# 新增跌倒素材标签复核与划分冻结（2026-09-23）

本轮将已下载的 **23 段 GMDCSA24 视频和 2 组 UP-Fall 帧序列**整理成可复现的研究清单 [fall-web-v1.json](../../datasets/manifests/fall-web-v1.json)。保留全部原始视频、原 CSV、下载 inventory 与旧模型，没有重新下载或运行预测。本记录说明数据标签与分组，不代表模型精度已提高。

冻结清单 SHA-256：`543fa4df12d416955da4462fecad02ac339d71e5c3b4a5a279c43425821bb37b`。本地冻结凭据：`runtime/annotations/fall-web-v1/freeze.json`。

## 固定分组与适用边界

| 来源 / split | 作者人物分组 | 视频数 | 跌倒事件数 | 全部帧数 | 观测秒数 |
|---|---|---:|---:|---:|---:|
| GMD train | Subject 1、2 | 12 | 4 | 2,956 | 99.293 |
| GMD val | Subject 3 | 6 | 2 | 997 | 33.416 |
| GMD test | Subject 4 | 5 | 1 | 1,147 | 38.423 |
| UP external_diagnostic | Subject 1，共两次试验 | 2 | 1 | 376 | 19.806 |

GMD 相同作者 subject 共用 `subject_id` 和保守 `source_group`，没有按片段发明独立录像组。train/val/test 在 **GMD 作者声明的人物组内部**不交叉；不是所有历史 URFD 人物都经过身份核对的全项目跨人物证明。Subject 4 的原始图像仅用于固定标签，复核没有运行模型或查看分数。

**物理摄像头身份仍未知。** GMD 的 `camera_id` 保留 null。Subject 2 与 Subject 3 在视觉上使用相同或相似的房间、机位，可能共享相机，因此不能声称跨相机泛化，也不通过既有严格人物/相机准入。原始连续录像关系尚未完全核实，同一 subject 共组只是保守归并。没有为了通过审计改写未知身份。

UP 保留原 Camera1 / Camera2 视角 ID、同一 designated Subject1；跌倒录像还出现身份未知的辅助人员，`additional_visible_persons_unidentified=true`。两组只能作外部诊断，**不得参与训练、调参或选模**，其全部 `training_label` 为 null。原 RGB 数据标准许可仍未明确，保留原清单的公开研究发布依据与许可边界。

这批 GMD 只有 4 个作者人物组、7 次表演跌倒。短片时长不是具有校园代表性的长期连续录像曝光，不可用几个分钟甚至不足一分钟的分母声称可信部署误报率。

## 时刻核验

GMD 逐视频执行 `ffprobe -show_entries frame=best_effort_timestamp_time,pkt_duration_time,duration_time`，提取每个解码帧的 PTS，减去首帧 PTS 使首帧为 0。首帧通常为媒体 PTS 0.033333 秒，帧间隔并不恒定。联系图通过 OpenCV 全量顺序解码，帧数与 PTS 数必须一致；按 **0 基帧索引**配对图像和时刻，不能改用 `frame_index / avg_fps`。

GMD `duration_s` 为末帧相对 PTS 加末帧明确记录的解码持续时间。23 段累计约 171.132 秒，与旧 inventory 的平均 FPS 推算值略有区别；旧 inventory 保持原样。

UP 真值时刻取 ZIP 中 PNG 原文件名时间戳，核对 `.frames.json` 的声明时间与文件名一致、全部文件名与原 ZIP 成员一致，并核验 ZIP SHA-256。预览 MP4 与原帧一一按序对应，只用于查看画面；其恒定帧率不作为真值。UP 的 `duration_s` 仅为首末帧的已观测时间跨度，不推造最后一帧曝光时间，文件名时区也保持未知。

每条样本均含 `video_path`、视频 SHA-256、来源 URL、原路径、人物/相机/来源组、原标签、原 CSV 路径及 SHA、逐帧 `timing` 和 timing sidecar SHA。UP 还保留 ZIP、原始时间 sidecar 的 SHA。

## 视觉复核方法与标签

复核者是 **assistant**，不是独立人工双重审核。实际查看了：7 段 GMD 跌倒各一张全片 0.2 秒联系图与一张事件附近约 0.067 秒联系图；16 段 ADL 各一张 0.6 秒联系图；2 组 UP 各一张 0.25 秒联系图。证据图附帧号、真实相对时刻、抽样帧列表和 SHA，均在 `runtime/annotations/fall-web-v1/<sample_id>/`。没有根据模型分数决定真值。

ADL 维持作者类别为粗负例，视觉抽样支持主动躺床/躺地、弯腰取物、坐下、身体拉伸、俯卧撑等动作；Subject3/ADL07 可见深屈膝反复起立，记录为 `squat_like_exercise`，保留原文“sitting and standing”。没有把原官方名称改成精确逐帧动作真值，也没有虚构嬉闹素材。

区间为半开 `[start_s, end_s)`，完整覆盖已观测录像：

| `label` | `training_label` | 含义 |
|---|---:|---|
| `non_fall` | 0 | ADL 官方粗负类加视觉抽样核对，或跌倒前可确认的正常阶段 |
| `uncertain` | null | 不能精确定界的起始、落地区间 |
| `fall_motion` | 1 | 视觉确认的下落运动核心，7 段均大于 0.2 秒 |
| `post_fall_transition` | null | 落地后的坐姿转躺、翻滚、肢体调整或裁切下难以判断的动作 |
| `post_fall` | 0 | 已稳定支撑姿态，仅对“下落运动”任务为负类；**不是恢复** |

不能把整段 Fall 视频、持续躺地或 uncertain 默认扩成所有窗口阳性。窗口构建必须在单独冻结的训练协议中说明与上述核心区间的交叠规则；标签清单本身不通过增加窗口数量声称独立数据增加。跨 uncertain 的负窗口不应入训；若采用“已确认核心交叠至少 0.2 秒即正例”，须同时保留该规则和窗口实际交叠量。

## 跌倒时序边界

表中时刻为相对真实帧时刻，保留三个小数便于阅读；JSON 包含完整小数与所引用的 0 基帧索引。起始、落地均是 **范围**，不是伪精确单点。

| 样本 | 可能开始范围 / 秒 | 确认下落核心 / 秒 | 落地范围 / 秒 |
|---|---|---|---|
| Subject1/Fall03 | 1.008–2.480 | 2.480–3.152 | 3.152–3.408 |
| Subject1/Fall08 | 0.704–0.896 | 0.896–1.840 | 1.840–2.032 |
| Subject2/Fall01 | 1.200–1.808 | 1.808–2.944 | 2.944–3.344 |
| Subject2/Fall04 | 3.280–3.488 | 3.488–4.192 | 4.192–4.320 |
| Subject3/Fall03 | 1.312–1.648 | 1.648–2.176 | 2.176–2.384 |
| Subject3/Fall04 | 1.008–1.344 | 1.344–1.872 | 1.872–2.080 |
| Subject4/Fall03 | 1.584–1.808 | 1.808–2.288 | 2.288–2.544 |
| UP Subject1 Activity1 Trial1 Camera2，外部诊断 | 3.258–3.505 | 3.505–4.254 | 4.254–4.488 |

Subject1/Fall03 的缓慢预备侧倾不能唯一确定哪一帧“失衡”，因此保留较宽起始不确定区间。Subject2/Fall04 先落地坐下，再慢慢躺倒：坐姿转躺标为 `post_fall_transition`，不造第二次跌倒、不造恢复。Subject4/Fall03 有前景裁切且落地后肢体仍动，整个尾段没有强行给 static 负标签。

每条 `events` 记录包含 `onset_range_s`、`landing_range_s`、`confirmed_motion_range_s`；`start_s` 和 `end_s` 是最早可能开始至最晚可能落地的事件包络。用这个包络做宽容事件匹配时必须披露边界不确定性，报警延迟应报告相对于起始范围的区间。不能拿包络末尾当作人员恢复时刻。

没有任何联系图序列拍到跌倒后的完整站立恢复，所有跌倒 `recovery_observed=false`、`recovery_time_s=null`，尾端按未观察到恢复的截尾录像处理。本批不能验收真实“正常→跌倒→恢复”完整过程。

## 复现与校验

在项目根目录使用已有 `.venv` 中的 OpenCV/Pillow，系统 `ffprobe`：

```bash
.venv/bin/python scripts/prepare_fall_web_annotations.py
.venv/bin/pytest -q tests/test_prepare_fall_web_annotations.py
.venv/bin/ruff check scripts/prepare_fall_web_annotations.py tests/test_prepare_fall_web_annotations.py
```

首次/完整运行从原文件重新生成所有联系图及逐帧时刻。`--evidence-only` 只生成证据，不写冻结 manifest；`--reuse-evidence` 重用已保存联系图，但仍核验媒体 SHA、图像 SHA、原 CSV 内容、重新探测 GMD PTS、UP 原 ZIP 文件名及时间戳一致性。不会联网或覆盖源素材。标注边界常量写在脚本中，修改它们即新标签版本，需要重新冻结清单及训练/测试证据。

本轮 11 项定向测试通过，覆盖变帧率时刻、缺失曝光/非法时间拒绝、原 PNG 时间与文件名一致性、25 条冻结结构、人物组不能跨集、未知相机不能伪造、uncertain 不能静默改标签。Ruff 通过。25 条全量视频 / 原 ZIP SHA 与原清单一致，23 段原 CSV 行保持一致。
