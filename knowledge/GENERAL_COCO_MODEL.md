# 当前通用检测器与 COCO 类别

官方来源：Torchvision 0.17 的 fasterrcnn_mobilenet_v3_large_320_fpn 模型说明及 COCO 官方数据页。核实日期：2026-10-23；核实级别：官方模型文档与本项目适配代码。该模型使用 MobileNetV3-Large 主干和 Faster R-CNN 检测结构，官方 DEFAULT 权重对应 COCO_V1。

本项目使用完整的官方预训练检测头、预处理及 COCO 类别映射，不把原水下六类检测头加载到通用模型。官方类别索引中背景与空占位项不对外作为目标；用户看到的是 80 类常见物体，框内类别名称保持稳定，官方原始编号可能不连续。

输入先处理方向信息并转换为 RGB；8位像素按官方输入转换方式处理，模型处于评估模式。图像仍保留在本机。COCO 预训练分数不自动等于本项目已校准概率；其他场景泛化和校正收益需要另外评估，官方基准数值不作为本版实测。

来源链接：
https://docs.pytorch.org/vision/0.17/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn.html
https://cocodataset.org/
