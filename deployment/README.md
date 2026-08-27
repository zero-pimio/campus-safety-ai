# 部署状态

- PC：当前只有固定检测结果的本地回放；真实 YOLO 尚未做运行验收。
- Rockchip：规划链路 `PT → ONNX → RKNN → RKNN Runtime`，尚未锁定具体板卡/SDK。
- Ascend：规划链路 `PT → ONNX → ATC → OM → ACL/MindX`，尚未锁定 Atlas/CANN 环境。
- OpenRemote：消息契约和 Outbox 已有，MQTT/OpenRemote adapter 尚未实现。

禁止把“模型转换成功”写成边缘闭环成功。G2/G5 必须分别验证输出一致性、端到端视频、资源和稳定性。

