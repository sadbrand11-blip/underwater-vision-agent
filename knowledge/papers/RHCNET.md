# RHCNet：水下结构退化与特征校准

来源：Yueying Wang 等，RHCNet: Residual-Guided Hierarchical Calibration Network for Robust Underwater Object Detection，CVPR 2026。官方原文：https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_RHCNet_Residual-Guided_Hierarchical_Calibration_Network_for_Robust_Underwater_Object_Detection_CVPR_2026_paper.pdf 。核实日期：2026-10-03；核实级别：官方 PDF 全文。

## 检测困难

论文针对前景背景相似、结构细节丢失和对比度降低造成的水下检测困难，提出残差引导特征增强 RGFE 和分层特征校准金字塔 HFCP。

## 方法含义

RHCNet 基于 ResNet-50，结合边缘、纹理与多尺度语义特征。这里的 calibration 是特征对齐和校准，不是本项目把检测分数映射为经验精确率的 isotonic 概率校准；二者不能互相替代。

## 项目边界

本项目仍使用 Faster R-CNN MobileNetV3 320，没有实现 RGFE、HFCP 或接入 RHCNet 权重。研究结果来自论文的 DUO、UTDAC 等协议，不能转移为本项目 SODD 或机器人模型的成绩。

原文位置：PDF 第1页 Abstract、Introduction；方法见第3节。calibration 的区分结合原文方法与本项目实现作出。
