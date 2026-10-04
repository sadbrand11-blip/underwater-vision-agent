# MPD：人眼观感与机器任务质量

来源：Chunyi Li 等，Image Quality Assessment: From Human to Machine Preference，CVPR 2025。官方原文：https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Image_Quality_Assessment_From_Human_to_Machine_Preference_CVPR_2025_paper.pdf 。核实日期：2026-10-03；核实级别：官方 PDF 全文。

## 研究发现

人眼偏好与机器视觉任务偏好存在差异，观感更好不保证检测、分割或问答结果更好。论文以 MPD 数据库研究这种差异，并发现现有以人眼为中心的质量指标不能准确描述机器偏好。

## 评估原则

机器质量需要指定下游任务、测试模型及评价指标。PSNR、SSIM 或无参考观感分数可以用于诊断，但选择校正方案时还要核对实际检测表现。

## 项目适用性

本项目没有接入 MPD 模型或训练机器质量网络。质量分数、检测框数量及分数上升均不能代替真实框标注下的召回与误报评测。论文证据支持分别评价观感和任务表现，不给本项目识别精度背书。

原文位置：PDF 第1页 Figure 1、Abstract；机器偏好定义见正文任务说明。第二、三节含项目据此采取的工程解释。
