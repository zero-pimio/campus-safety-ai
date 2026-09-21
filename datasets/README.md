# 数据目录规则

这里只提交数据 manifest、来源、许可证、哈希、划分和标注规范，不提交校园隐私视频。真实数据放在忽略的 `datasets/private/` 或受控存储中。

训练/验证/测试必须按摄像头或日期划分；禁止把同一段视频的相邻帧随机分到训练集和测试集。

`scripts/download_video_smoke_samples.sh` 会从 Surveillance Camera Fight Dataset 作者仓库下载一段打架和一段非打架视频到忽略目录 `datasets/samples/`，只用于本地技术冒烟。仓库代码标注为 MIT，但视频来自 YouTube，权利链未被单独澄清；不能据此认定视频可商用或再分发。

## 本地数据（2026-08-25）

完整或已核验下载放在 `datasets/private/`，该目录不会进入 Git：

- `scfd/`：300 段视频。
- `airtlab/`：350 段视频。
- `ut-interaction/`：set1、set2、分段包、标签和辅助包；ZIP 已通过完整性检查。
- `ava-v2.2/`：官方 v2.2 标注、训练/验证 CSV 和动作标签表；不包含底层电影视频。
- `caviar-fights/`：4 个打架视频和对应的逐帧 XML 标注。
- `behave-fights/`：4 个官方打架视频。
- `bus-violence/`：Zenodo 官方完整压缩包；ZIP 已通过完整性检查。

以下数据不能直接批量下载：NTU CCTV-Fights 需要注册并签署协议；RWF-2000 官方已经停止提供；Violent-Flows 需要作者发放 FTP 密码；RLVS 的 Kaggle 官方托管入口当前未提供匿名 API 下载；NWPU、XD-Violence、UCF-Crime、AVA 全视频等大集需要先预留额外磁盘空间。不要通过第三方转载绕过许可或隐私限制。

## 2026-09-14 清理后的实际状态

按当前训练和回放用途，从项目移出 68 段未使用视频：AIRTLab 中被打架动作筛选规则排除的 60 段，以及尚未接入当前流程的 CAVIAR 4 段、BEHAVE 4 段。以上历史下载数量不再代表当前视频数量。来源说明、标注和归档包保留。

当前保留 SCFD 300 段、AIRTLab 290 段、冒烟样例 2 段、合成回放输入 1 段。训练清单重建结果与清理前完全一致，590 个引用均存在。视频仅移至废纸篓独立目录，未永久销毁，磁盘空间尚未释放。

恢复路径、文件哈希和逐文件状态见 `docs/audits/2026-09-14/unused-source-videos-20260914-145853.json`。恢复前检查原路径未被新文件占用。
