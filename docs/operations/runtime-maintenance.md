# 运行积压与证据保留

此工具提供本地容量背压、积压指标和可恢复的证据隔离。它不构成真实摄像头、平台或 RK3588 长时间运行验收。

## 积压容量与恢复

`EventDelivery` 和 `BackgroundDelivery` 接受 `max_pending_records`、`max_pending_bytes` 两个正整数上限。底层 API 默认 `None` 保持现有调用兼容；持续跌倒命令默认上限为 10,000 条和 64 MiB。两个上限任意一个被超过即拒绝整个新增批次，抛出 `OutboxCapacityError`。容量检查和入库位于同一个 SQLite 写事务中，多个生产连接不能通过检查竞态超额入库。

重复幂等键（包括已投递历史记录）不占新增容量。未投递事件、重试次数、事件内顺序保持不变。异常包含当前与新增记录数/字节数；调用方必须停止生产并保留被拒绝记录，不能捕获后继续推理、跳过事件或宣称已正常结束。当前持续跌倒入口把被拒绝记录保存在 `unsubmitted-events.jsonl` 并报告阻塞状态；该文件不是已入库或已交付证明。

查看积压不会创建、迁移数据库，也不向平台发送消息：

```bash
.venv/bin/python -m campus_safety_ai.apps.deliver_events \
  --status --outbox runtime/fall-camera-01/outbox.sqlite3
```

主要指标：

| 字段 | 含义 |
| --- | --- |
| `pending`、`pendingBytes` | 待发送记录数及其 UTF-8 JSON 有效载荷字节数 |
| `oldestPendingAt`、`oldestPendingAgeSeconds` | 已知入库时间的最老待发送记录及其等待时间，不是摄像头视频时间 |
| `unknownPendingAgeRecords` | 旧数据库迁移前无法确定入库时间的积压；有此项时已知最老时间不能代表整个队列年龄 |
| `readyEvents`、`deferredEvents`、`blockedRecords` | 可重试事件、退避事件和同事件内等待前序记录的条数 |
| `totalRecords`、`payloadBytes` | 包含已发送历史的总行数与有效载荷字节数 |
| `databaseBytes`、`databaseAuxiliaryBytes` | 数据库文件与 WAL/SHM 辅助文件占用，不是内存使用量 |

处理流程：先停止生产，恢复平台可用性并排空原 outbox；再根据原始幂等键和事件内版本顺序恢复 `unsubmitted-events.jsonl`。该日志暂不自动回灌，也不能在尚未确认恢复完成前删除或清空。若希望继续采集，应启动新的源周期，不能把中断前后的轨迹直接拼接。恢复命令是否实际发送平台消息取决于显式指定的平台配置。

## 证据保留

只管理一个明确指定的运行目录。数据库和证据目录必须处于该目录内部，不递归寻找其他运行、训练集或源视频。默认运行下面的命令只列出候选路径、关联事件、字节数与跳过原因：

```bash
.venv/bin/python -m campus_safety_ai.apps.maintain_runtime \
  --runtime-root runtime/fall-camera-01 \
  --outbox runtime/fall-camera-01/outbox.sqlite3 \
  --evidence-dir runtime/fall-camera-01/evidence \
  --retention-days 30
```

证据必须同时满足：

1. 由证据写入器在生成后调用 `register_managed_evidence(evidence_root, path)`，在 `.evidence-ownership.jsonl` 留下精确路径、SHA-256、大小与文件身份；禁止用此接口批量登记旧素材或输入源视频。
2. 每个引用该证据的事件具有完整、顺序递增的 START → END/CLOSED 记录，并且全部记录已发送。共享证据只要还有一个活动或未投递事件，就继续保留。
3. 事件结束、最后投递和证据登记三项时间都超过保留期。旧数据没有投递时间时不猜测，保守保留。
4. 当前内容与生成证明一致，为管理目录内的普通文件且没有符号链接或额外硬链接。路径逃逸、远程 URI、变更或缺失文件均不处理。
5. `unsubmitted-events.jsonl` 不含待恢复内容；否则此次证据隔离整体阻塞，防止数据库外记录失去证据。

确认清单后，**先停止该运行的全部视频生产程序与投递程序**，再在同一条命令末尾加 `--apply`。它取得投递独占锁、在写事务内重读事件状态并再次校验文件，随后仅把精确候选移动到 `runtime/fall-camera-01/quarantine/evidence-<唯一标识>/`。若后台投递仍持有锁，操作明确拒绝。发送锁不代替停止只生产、不发送的其他进程。

每次移动前会落盘 `receipt.jsonl` 的 `planned` 记录，移动后记录 `moved`，包含原路径、隔离路径、关联事件、SHA-256 和大小。中途中断后，按回执确认原路径/隔离路径的实际状态；校验隔离文件哈希与大小，且原路径不存在时，可移回回执中的精确原路径。不要覆盖已存在的文件。隔离操作不直接永久删除数据，也不释放磁盘空间；报告的 `bytesFreed` 固定为 0。后续归档、备份和永久删除仍需独立保留策略。

## 明确保留的边界

- 从未登记的旧证据、无事件引用的文件、源视频、训练数据及不相关文件全部保留。
- 不删除任何 outbox 行，已交付幂等键持续存在；因此不会因清理破坏重试顺序或让旧事件重新发送。
- 不裁剪 `events.jsonl`、`observations.jsonl`、`delivered-events.jsonl`、所有权登记及恢复日志，仅统计已知运行日志大小。日志轮转/归档、outbox 历史归档和磁盘配额尚未完成。
- 待发送有效载荷上限不等于总磁盘上限。已交付历史、日志、SQLite 索引/WAL、备份和隔离证据仍会增长。必须结合磁盘使用监控；当前实现不能宣称无限期无人值守运行。
- 本地临时目录测试覆盖重复入库、满队列原子拒绝、并发生产背压、未知年龄、活动/共享/未投递证据保护、路径/链接保护、校验失败、投递锁冲突和移动中断恢复。未在真实数据库或现存证据上执行隔离，也未发送真实平台告警。
