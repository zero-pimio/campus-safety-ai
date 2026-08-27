# ADR-001：三个深模块和稳定数据契约

状态：Accepted（项目骨架默认）

## 决策

外部主链只暴露 `Perception.detect`、`EventAnalysis.advance`、`EventDelivery.submit`。厂商模型、跟踪器和消息平台的原生类型不得跨模块传播。ByteTrack 首版位于 `EventAnalysis` 内部，出现第二个真实跟踪实现后再提取 seam。

## 原因

这样换 YOLO/芯片只影响感知层，换 MQTT/OpenRemote 只影响投递层；同时避免把每一步都做成难以理解的浅插件。

## 回滚

如果真实集成证明模块边界过粗，通过契约测试和固定 ReplayBundle 抽取内部 seam，不改变外部数据格式。
