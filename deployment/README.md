# 部署状态

- PC：固定回放、PP-TSM/PyTorch/ONNX 打架分类、YOLO 人体检测、ByteTrack、本地证据与 SQLite Outbox 已做本机冒烟；RTSP 长稳尚未验证。
- Rockchip：规划链路 `PT → ONNX → RKNN → RKNN Runtime`，尚未锁定具体板卡/SDK。
- Ascend：规划链路 `PT → ONNX → ATC → OM → ACL/MindX`，尚未锁定 Atlas/CANN 环境。
- OpenRemote：消息契约、Destination seam 和 Outbox 已有，当前 adapter 是 JSONL/内存；MQTT/OpenRemote 远程 adapter 尚未实现。

禁止把“模型转换成功”写成边缘闭环成功。G2/G5 必须分别验证输出一致性、端到端视频、资源和稳定性。
