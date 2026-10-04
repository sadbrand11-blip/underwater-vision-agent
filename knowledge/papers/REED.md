# REED：真实极端曝光与恢复证据

来源：Bo Wang 等，From Abyssal Darkness to Blinding Glare: A Benchmark on Extreme Exposure Correction in Real World，ICCV 2025。官方原文：https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_From_Abyssal_Darkness_to_Blinding_Glare_A_Benchmark_on_Extreme_ICCV_2025_paper.pdf 。核实日期：2026-10-03；核实级别：官方 PDF 全文。

## 数据采集

REED 通过相机连拍采集从极暗到极亮的真实图像序列，采用裁剪和改进的 SIFT 对齐，以缓解运动和场景变化带来的错位。真实序列不同于对现有 RGB 图片乘以亮度系数。

## 方法与限制

论文提出 CLIER，并使用亮度归一化、语义指导和迭代曝光修正。论文指出某些情况下仅看亮度不足以判断曝光状态；极端曝光中的亮度与纹理缺失使恢复困难。

## 项目适用性

本项目没有接入 CLIER 或 REED 数据。生成模型推测的纹理不等同于从饱和像素测得的真实细节；不能凭增强观感撤销本项目的质量失败。这是本项目的证据保护原则，不是对论文方法所有输出的判定。

原文位置：PDF 第1页 Abstract、Figure 1；数据采集与对齐见数据集章节。
