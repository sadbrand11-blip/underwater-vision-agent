# 增强作为增广：训练域与推理预处理

来源：Ashraf Saleem 等，Enhancement as Augmentation: Improving Detection in Highly Degraded Underwater Images Through Mixed-Domain Training，WACV Workshops 2026，属于研讨会论文。官方原文：https://openaccess.thecvf.com/content/WACV2026W/WVAQ/papers/Saleem_Enhancement_as_Augmentation_Improving_Detection_in_Highly_Degraded_Underwater_Images_WACVW_2026_paper.pdf 。核实日期：2026-10-03；核实级别：官方 PDF 全文。

## 研究设计

论文分别测试四种增强模型，每次将一种增强器的变体与原图配对用于混合域训练，保持检测器架构与推理流程不变，以研究增强造成的外观分布变化。

## 观感与检测

在论文使用的 USGS round goby 子集上，增强质量指标的高低不能预测检测收益；过度增强可能损害检测。混合域训练与推理前对每张图固定增强是不同的操作。

## 项目适用性

研究支持把增强作为需要验证的训练策略，但不保证对任意水域、目标或检测器有效。本项目0.4.0改进增广候选误报增加，没有通过采用门槛；论文结果不能覆盖这个失败或提高现有可靠性状态。

原文位置：PDF 第1页 Abstract、Introduction；混合域训练实验见正文实验章节。项目观察来自 VISION_RESULTS.md。
