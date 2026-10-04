# Sea-Thru：水下成像物理与输入条件

来源：Derya Akkaynak、Tali Treibitz，Sea-Thru: A Method for Removing Water From Underwater Images，CVPR 2019。官方原文：https://openaccess.thecvf.com/content_CVPR_2019/papers/Akkaynak_Sea-Thru_A_Method_for_Removing_Water_From_Underwater_Images_CVPR_2019_paper.pdf 。核实日期：2026-10-03；核实级别：官方 PDF 全文。

## 成像物理

直接信号衰减与后向散射由不同系数控制，不能简单共用一个大气去雾系数。直接信号的衰减系数还依赖目标距离和反射特性，不能只看水体属性或整图亮度。

## 方法条件

Sea-Thru 使用 RGBD 图像及已知距离信息，以暗像素估计后向散射，再利用空间变化的照明估计距离相关衰减。只有单张 RGB 图片时，不具备原方法所需的完整输入。

## 项目边界

本项目只有 RGB 输入及有界 Gamma、CLAHE 等校正，没有深度图，也没有实现 Sea-Thru。引用这项研究可解释水下退化，但不能把现有亮度校正称作物理水体去除。

原文位置：PDF 第1页 Abstract；第2页 Scientific Background 与成像方程。项目边界为实现核对，不是论文的性能结论。
