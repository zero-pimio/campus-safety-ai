# EasyAIoT 源码整合记录

目标仓库由用户确认：[ptgosn/easyaiot](https://gitee.com/ptgosn/easyaiot)。固定 commit：`49f00960a2907c0066b2c3f6e36c8cb9c9239320`，默认分支 main。已完整检出该版本的工作树，浅克隆深度 1，不包含完整历史。

独立路径：`/Users/wanwanzhu/easyaiot-integration-20260914`。检出后约占 1.7 GiB，保留上游文件和 Git 历史，未改上游代码。该容量为新引入的外部源码/资源，不属于校园项目旧视频清理释放的空间。

## 已完成的代码整合

1. 校园事件继续通过 `adapters/easyaiot.py` 对接上游，不引入平台 ORM、Java 或前端依赖到核心分析。
2. 响应校验改为 HTTP 成功且 `code == 0`、`data.status == "success"`。返回跳过、抑制、业务失败、空/非法响应时保留 outbox 未交付状态。
3. `correlation_id` 改为标准长度 UUID，由本项目事件 ID 稳定映射；同一事件所有版本保持同值。原始 eventId、revision、idempotencyKey 和完整 EventRecord 保留在 information 中，幂等键仍随请求头发送。
4. 增加本地 mini 模式的配置样例；保留原网关配置，实际部署选择对应入口。
5. 校园视频分析逐窗口交付，避免持续视频等待 EOF 才向平台发送。

## 上游源码对齐证据

所有下述路径相对于上述独立检出目录，以锁定版本为准。

| 事实 | 源码位置 | 对整合的影响 |
| --- | --- | --- |
| VIDEO 注册 `/video/alert` 蓝图，hook 使用 POST | `VIDEO/run.py:862`；`VIDEO/app/blueprints/alert.py:245` | mini 直连为 `/video/alert/hook`，网关前缀另配 |
| api_response 的成功业务码是 0，HTTP 统一为 200 | `VIDEO/app/blueprints/alert.py:48` | 必须解析响应体 |
| skipped/suppressed 同样返回成功业务码 | `VIDEO/app/blueprints/alert.py:257` | 还需验证 data.status |
| 未找到设备已启用的告警任务时 skipped | `VIDEO/app/services/alert_hook_service.py:824` | cameraId 必须映射到真实设备，且绑定启用的任务 |
| 非 mini 形态可能按设备/任务抑制 | `VIDEO/app/services/alert_hook_service.py:816` | 生命周期更新可能被抑制；不能静默认为已落库 |
| mini 可直接落库；其他形态可经 Kafka | `VIDEO/app/services/alert_hook_service.py:661` | hook 成功不代表通知送达或 UI 可见 |
| correlation_id 是 VARCHAR(36) 普通索引 | `VIDEO/models.py:288` | 关联与幂等是不同语义，不放长版本键 |
| create_alert 新建行并 commit | `VIDEO/app/services/alert_service.py:529` | 该直接入库路径没有按项目幂等键做事务去重；不能保证一次且仅一次 |
| information 可以接收字典 | `VIDEO/app/services/alert_service.py:450` | 保留完整项目事件契约的方式可用 |
| 告警列表过滤没有 image_url 的记录 | `VIDEO/app/services/alert_service.py:185` | “接口接受但界面没有”需先核对证据状态 |
| mini 图片使用 VIDEO 所在机器的本地路径 | `VIDEO/app/services/alert_hook_service.py:693` | 远程边缘路径不能直接给平台读取 |
| 现有环境依赖含数据库及多种模型框架 | `VIDEO/requirements-base.txt` | 不把整份上游依赖装进校园项目虚拟环境 |

本轮审计是关键接口与路径审计，不是整个 6,762 文件平台的全面代码审计。

## 不依赖 Docker 的可重复契约验证

从校园项目根目录执行：

```bash
.venv/bin/python integration/easyaiot/verify_contract.py \
  --upstream /Users/wanwanzhu/easyaiot-integration-20260914 \
  --output docs/audits/2026-09-14/easyaiot-contract.json
```

验证器检查 upstream HEAD，读取其真实 `api_response`、`alert_hook` 函数，并通过仅监听 127.0.0.1 随机端口的 HTTP 服务调用。Flask 的 request/jsonify 环境与处理服务是替代实现；测试不启动 Kafka、不访问真实平台数据库、不发送通知。

测试四种真实 handler 返回分支：success 接受，skipped/suppressed/failed 拒绝。另有本项目单元测试覆盖非法 JSON、错误业务码、异常响应字段、过大响应和 outbox 保留。

这验证了客户端与所检出 handler 的响应契约及本地 HTTP 路径，不证明完整平台部署成功、身份验证正确、接收端去重生效或告警页面可见。

## 真实 mini 平台联调操作

当前 Docker daemon 不可连接，完整平台尚未启动。以下是准备好的联调方式，不是已执行结果。

先按上游的 mini 部署流程准备服务；确认 VIDEO 的实际端口、存储路径和权限，不直接把网关配置替换成 mini 配置。平台侧为真实设备创建/关联已启用的实时告警任务。camera-id 传该设备 ID。若有鉴权，先在本地终端环境注入 `EASYAIOT_TOKEN`，不写进配置文件。

在项目根目录使用本地平台配置：

```bash
.venv/bin/campus-safety-fight-video \
  --video runtime/smoke/composite/fight-then-normal.mp4 \
  --model-config configs/models/fight-onnx-v1.toml \
  --platform-config configs/platform/easyaiot-mini-dev.toml \
  --camera-id YOUR_REGISTERED_DEVICE_ID \
  --source-epoch 20260914 \
  --output-dir runtime/easyaiot-mini \
  --evidence-dir /Users/wanwanzhu/yolo/runtime/easyaiot-mini/evidence
```

`YOUR_REGISTERED_DEVICE_ID` 必须替换为平台真实 ID。上例证据路径只适用于 VIDEO 可读同一路径的本机环境；容器要显式共享同一挂载路径，远端平台需要上传流程。此命令真实发送告警，应在隔离开发平台、关闭外部通知的测试设备执行。

实际验收：检查 outbox、平台入库、correlation 聚合、截图/视频可读和重复请求；单看 CLI 无错误不够。平台当前会新建告警行，多个生命周期记录不代表自动合并为一条可处置事件。

## 下一批优先整改

- 已实现交付线程独立连接、分析入队、持久退避及有界退出；下一步补充永久错误分类、人工处理和容量治理。
- 接收端以完整 idempotencyKey 建唯一约束、在同事务内写事件；不要拿客户端防重复入队替代接收去重。
- 统一 eventId 的生命周期更新与 revision 单调性，旧版本不可覆盖新版本。
- 将证据发布状态与平台的 image_url 过滤逻辑对齐；补齐失败可观察性。
- 给校园生命周期事件设计专用抑制策略，避免同一设备时间窗抑制 END。
- 完成 RTSP 重连与源周期管理后，进行真实单路 24 小时验证。

详细工程工作包、前置条件、验收与回退见 `docs/REMEDIATION-PLAN-2026-09-14.md`。
