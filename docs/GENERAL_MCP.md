# 0.6.0 通用图像 MCP

MCP服务默认使用同一COCO80通用配置，仍通过stdio暴露恰好三个工具。输入为D盘数据目录下8位JPEG/PNG，大小≤32MiB、像素≤1600万；原图只读、每方法候选仅一次、编号与网页隔离。

| 工具 | 示例 | 结果 |
|---|---|---|
| assess_image_quality | image_path=本地照片绝对路径 | 编号、实际质量指标和热图路径；无需检测器 |
| generate_exposure_candidate | image_id=原图编号, method=gamma_only | 已评估原图的有界候选，不继续增强候选 |
| detect_objects | image_id=编号, target_classes=[人,汽车] | person/car候选框、分数及校准元数据 |

类别仅COCO80，管道/二维码不支持。不提供完整可靠性验收，检测始终是候选。

```powershell
python -m pip install -r requirements-mcp.txt
python serve_mcp.py
```

最后一个命令等待MCP客户端的协议消息，没有网页界面。要观察真实工具发现与调用，在已装检测依赖的环境运行：

```powershell
python -X utf8 -m optical_agent.experiments demo_mcp --image D:\CodexData\optical_agent\你的照片.jpg
```

本机冻结验证发现3工具、8次调用、9/9直接调用对照一致，记录在D盘mcp/runs/demo_v060_frozen.json。首次加载模型较慢；损坏或缺失模型明确报错，不当作没有目标。完全黑/白图适合单独调用曝光工具，候选生成会停止。

客户端连接模板仍为根目录mcp-client.example.json，替换项目路径及解释器。日志写stderr，不污染stdout协议。30分钟过期、20原图上下文、串行推理及脱敏规则继续生效。

[0.5.3历史实现说明](14_MCP_SERVER.md)中的SODD类别、默认图片和成绩仅适用于旧版本；不用于当前入口。
