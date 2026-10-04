# UIEB：参考增强与评测边界

来源：Chongyi Li 等，An Underwater Image Enhancement Benchmark Dataset and Beyond。作者项目页标注 TIP 2019；本卡不重新核定期刊卷年。官方页面：https://li-chongyi.github.io/proj_benchmark.html 。核实日期：2026-10-03；核实级别：官方页面全文。

## 数据与用途

UIEB 包含 950 张真实水下图，其中 890 张有对应参考增强图，其余 60 张作为挑战集，没有令人满意的参考结果。Water-Net 是作者在该基准上提供的增强网络基线。

## 参考的含义

参考图用于比较增强结果，不是同场景相机曝光标定真值。本项目据此计算的是与参考增强结果的一致性；这不能单独证明物理颜色恢复或目标检测精度提高。这个用途限制结合官方页面的参考增强定义与本项目评测协议作出。

## 使用限制

作者页面规定仅限学术、非商业用途，禁止重新分发数据集。本项目不把原图或参考图提交到 GitHub。

原文位置：官方页面 Abstract、UIEB Dataset 两节。与项目关系：数据定义和指标解释；本轮没有训练或接入 Water-Net。
