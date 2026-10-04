# Version history

## 0.5.4 — simplification and shared execution mechanics

Compact homepage/workbench, unified experiment commands, shared native/LangGraph mechanics, and stable knowledge sources. Existing vision models and historical results retain their original versions. See [verification](SIMPLIFICATION.md).

# Archived development README

Historical narrative; some private artifacts and launchers are intentionally omitted.

# 水下图像曝光校正与目标识别 Agent

## 0.5.3：本地 MCP 视觉工具

通过官方 MCP SDK 暴露曝光评估、校正候选和目标检测三个工具。双击 **MCP离线调用示例.bat**，即可观察真实的子进程工具发现与调用。服务、图片和记录位于 D 盘；检测输出为候选识别。native 和 LangGraph 入口保留，本轮不调用付费 API。

- [中文操作与原理](../docs/14_MCP_SERVER.md)
- [协议与一致性验证](reports/MCP_RESULTS.md)
- [客户端配置模板](../mcp-client.example.json)

## 0.5.2：可选 LangGraph 等价对照

网页默认仍使用 **native**，新增 **LangGraph** 运行器选项。真实图节点复用当前视觉工具、RAG、长期记忆与可靠性校验；本轮没有付费 API 调用。

- [原理与操作说明](../docs/13_LANGGRAPH_RUNTIME.md)
- [离线等价性结果](reports/LANGGRAPH_RESULTS.md)
- [机器可读摘要](reports/LANGGRAPH_RESULTS.json)

双击 **LangGraph对照启动.bat**，选择离线模式与 LangGraph，创建图片会话后查看实际节点执行记录。切换运行器请新建会话。


## 新增：可对话的 Agent

**0.5.1 简单长期记忆：**新增明确设置的目标/展示偏好和自动保存的真实分析摘要，使用D盘SQLite，重启后仍可读取。历史只辅助规划，不能替代当前图像证据；支持查询、删除、关闭和数据库故障降级。网页顶部打开“长期记忆与偏好”。198项Python测试及18个网页诊断场景通过，两个实际进程验证偏好持久化与新图重新检测；本轮未调用付费API。[操作与优化思路](../docs/12_LONG_TERM_MEMORY.md) · [实际检查与限制](reports/MEMORY_RESULTS.md)。

**0.5.0 本地RAG对照：**增加TF-IDF、BGE语义检索和Hybrid，扩展为14份资料、126片段，完成100题开发/保留评测。C1保留集实际交付Recall@3：35.0% / 45.0% / 47.5%；Hybrid按预设门槛作为默认，MRR@10=0.4875、无答返回0/10、CPU热查询P95约26ms。仍漏掉大量可答证据；这不是LLM回答质量或视觉精度提升。网页顶部新增“知识检索对照”，无需图片即可使用。[操作与优化思路](../docs/11_RAG_OPTIMIZATION.md) · [真实结果](reports/RAG_RESULTS.md) · [逐题诊断](reports/RAG_CASES.csv) · [独立审核及限制](reports/RAG_EXPERIMENT_AUDIT.md)。模型、论文、索引与原记录全部保存D盘，本轮无付费API调用。

双击启动现在打开对话工作台，可先选择离线工具演示。新增独立工具、云端原生工具调用、会话追问、错误恢复、本地知识检索和50条任务评测。视觉测量和可靠性状态由程序产生。

[打开与学习指南](AGENT_GUIDE.md) · [本轮实现与验证](reports/AGENT_UPGRADE_RESULTS.md)

**0.3.0 动态决策：**新增固定目标、可观察计划、两种有界候选、目标类别过滤、观察驱动的重新规划和独立缓存。网页默认动态策略，保留旧调度；可选择正常、模拟偏暗和模拟全黑D盘输入。真实DeepSeek使用100/120次请求、292,537 token；同题开发修复前3/4轮、修复后4/4轮符合，真实视觉演示仅3/6次符合，固定Workflow为6/6。全黑校正误调用被拦截；仍有预算耗尽和停止错误，**本轮未通过新正式验收，也未证明视觉精度提高**。[操作与原理](../docs/08_ADAPTIVE_AGENT.md) · [完整结果与失败](reports/ADAPTIVE_EVALUATION_RESULTS.md) · [独立审计](reports/ADAPTIVE_EXPERIMENT_AUDIT.md)

**0.2.4 正式Agent Eval：**真实DeepSeek＋现有SODD视觉工具，20条预设保留任务重复3次，完成60次案例、78轮问题。242/600次请求，328,096 token；预设oracle案例流程符合率59/60（98.3%），按轮工具路线/必要证据符合率77/78（98.7%），参数规则140/140通过。程序验收通过，独立语义审计为WARN：类别筛选未进入评分，成绩不能解释为完整用户意图满足率。保留1次最终失败和1次成功格式恢复。[正式报告](reports/FORMAL_EVALUATION_RESULTS.md) · [操作与学习示例](../docs/07_FORMAL_EVALUATION.md) · [审计与剩余问题](reports/EXPERIMENT_AUDIT.md)

**0.2.3 独立评测与失败定位：**新增10条开发题、20条保留验收题，使用独立标签核对用户原意、图片、工具和实际证据。真实DeepSeek首轮发现3次依赖顺序错误，修复后相关案例与5张SODD图片任务通过；共93/120次请求。小样本结果不代表正式验收，也不代表检测精度提高。[优化思路与操作](../docs/06_INDEPENDENT_EVALUATION.md) · [真实评测报告与失败对照](reports/INDEPENDENT_EVALUATION_RESULTS.md)

**0.2.2 任务理解与完成校验：**自由中文任务先解析成任务类型与图片指向，再由程序固定必要证据。缺少前后比较证据，不能用曝光结果冒充完成；校正图检测不能交付原图检测。页面展示“理解的任务、必要结果、任务状态、缺项”，与视觉可靠性分别显示。[优化思路与操作](../docs/05_TASK_VALIDATION.md) · [离线验证记录](reports/TASK_VALIDATION_RESULTS.md)

**0.2.1 云端调度修复：**没有实际依据时要求先调用工具；无效结束结果最多纠正两次。分析失败后，原图上方会展开“错误详情”，显示上游HTTP状态、失败位置和处理建议。[修复与验证记录](reports/DEEPSEEK_FIX_RESULTS.md)

离线演示是规则调度，不代表真实LLM能力。真实对话使用本地.env配置的API密钥；本次正式测试保存了真实调用、用量及独立评分记录，测量基线为0.2.3 Agent行为。


这是一个在电脑上运行的水下图像分析原型。上传图像后，它会评估偏暗、过亮和局部信息丢失，做一轮有界曝光校正，在原图和校正图上识别目标，并根据图像质量与两次识别是否一致决定能否自动接受结果。没有相机控制或人工复核队列。

**当前能识别的目标：**螺旋桨 `propeller`、两类管道 `pipe` / `pipe_type2`、红色鳍片 `red_fin`、渔网 `net`、二维码 `qr_codes`。这些类别来自 [SODD](https://zenodo.org/records/10230328)。当前模型不负责识别鱼、珊瑚等其他生物。

## 打开使用

在 Windows 中双击 双击启动.bat (historical artifact not bundled in the public snapshot)，浏览器会打开对话工作台。先选D盘演示输入类型，再点击“使用 D 盘示例图”；保持“离线工具演示”可以查看规则示例。要使用DeepSeek，将运行方式改为“真实大模型对话”。调度策略默认“动态目标与规划”，选择任务后点击“开始分析”；也可以上传自己的水下图像。模型文件会自动加载，无需手动打开 `.pt` 文件。原有完整分析入口在页面顶部，仍可展示热图和完整报告。

如果从 GitHub 下载项目，先安装 [requirements.txt](../requirements.txt) 中的依赖，并让 Git LFS 完整取得 `models/sodd_detector.pt`。数据集图片不在 Git 仓库中；没有示例图时仍可上传自己的水下照片。命令行用法：

```powershell
python app.py
python run_agent.py D:\my_underwater_image.jpg --output D:\CodexData\optical_agent\my_result
```

默认识别模型是 models/sodd_detector.pt (historical artifact not bundled in the public snapshot)。旧钢材与 PCB 模型保留作历史实验，网页不会再用它们分析水下图像。

## 数据集为何这样选

| 数据集 | 当前用途 | 已下载到 D 盘 | 必须注意 |
|---|---|---:|---|
| [MOUD](https://doi.org/10.5281/zenodo.14865039) | 检查真实水下高、中、低照明下的曝光判断和校正 | Scene 1 每档每区域 2 张，共 24 张；另有 4 个标注文件 | 分档来自水下灯光强度，不是相机曝光时间；图像不保证几何对齐。本场景的目标标注只覆盖部分高照明图，不能直接计算低照明识别召回。 |
| [SODD](https://zenodo.org/records/10230328) | 训练与测试六类水下目标框 | 原始图像：训练 160、验证 40、测试 40；配套框标注 | 这是室内水池视频帧，近邻帧可能跨官方划分；现有结果只是小样本初步验证，不代表野外水下泛化。 |

全部数据集下载物位于 `D:\CodexData\optical_agent`，不提交到 Git。下载脚本按需读取官方 ZIP 的选定文件，避免获取 MOUD 每个约 8 GB 的完整场景包：

```powershell
python -m optical_agent.experiments download_underwater_data sodd --root D:\CodexData\optical_agent --train 160 --val 40 --test 40 --workers 2
python -m optical_agent.experiments download_underwater_data moud --root D:\CodexData\optical_agent --per-area 2 --workers 2
```

下载脚本会核对每个 ZIP 文件的大小和 CRC，并写入 D 盘的 `subset_manifest.json`。SODD 的作者提供原始帧和经过亮度变换等处理的增强帧；当前仅选原始帧，新的曝光扰动只在训练划分内生成。MOUD 只用于真实照明质量实验；不能把不同帧直接当成有参考的校正图对。

## 本版实现与结果

- **曝光校正：**局部网格测量暗部、高亮、对比度、边缘、清晰度和噪声；最多做一轮有界 gamma、CLAHE 和轻度降噪。原图与校正候选都保留。完全黑暗或饱和的纹理不会被当作真实恢复。
- **目标识别：**TorchVision Faster R-CNN MobileNetV3 320，从 COCO 权重微调，用 SODD 原始图和训练集内的像素域曝光扰动训练。目标分数用验证图的保序回归（isotonic regression）做经验校准。
- **反馈：**原图与校正图分别检测；若校正导致质量或检测证据下降，可选回原图。严重信息丢失、目标框和不可恢复区域重叠、两次检测冲突、没有足够置信度目标时输出“不可靠”或“质量失败”。
- **初步测量：**40 张 SODD 保留测试图的检测器目标框召回约 67.8%，mAP@0.5 约 72.1%，平均误报 0.625 个/图；只统计 Agent 自动接受的目标时，端到端召回为 55.9%。MOUD 24 张真实照明图在当前阈值下均提示偏暗，最低照明档在校正后仍然存在大面积信息损失。完整条件与限制见 [UNDERWATER_RESULTS.md](reports/UNDERWATER_RESULTS.md)。

当前仅有传统曝光校正基线，没有训练好的水下校正神经网络。SODD 的模拟欠曝、过曝实验仅是像素域压力测试，不能解释为真实相机曝光变化。MOUD 没有对齐参考图，因此没有在它上面计算 PSNR/SSIM。质量阈值仍属工程启发式；`reliable` 只表示通过本原型的内部规则，不代表对任意水域或目标都有可靠识别能力。

## 重现实验

```powershell
python -m optical_agent.experiments train_detector --dataset sodd --root D:\CodexData\optical_agent\sodd\SODD\data --weights path\to\fasterrcnn_mobilenet_v3_large_320_fpn.pth --output models\sodd_detector_raw.pt --epochs 4 --exposure-augment --validate-under
python -m optical_agent.experiments calibrate_detector --checkpoint models\sodd_detector_raw.pt --root D:\CodexData\optical_agent\sodd\SODD\data --output models\sodd_detector.pt --include-under
python -m optical_agent.experiments evaluate_detector --checkpoint models\sodd_detector.pt --root D:\CodexData\optical_agent\sodd\SODD\data --split test --output runs\sodd_test_pilot.json
python -m optical_agent.experiments evaluate_underwater_illumination --root "D:\CodexData\optical_agent\moud\Scene_1\Images" --output runs\moud_illumination_pilot.json
python -m optical_agent.experiments evaluate_agent --dataset sodd --root D:\CodexData\optical_agent\sodd\SODD\data --model models\sodd_detector.pt --split test --output runs\sodd_agent_pilot.json
```

完整实验需增加不同场景、不同相机的真实标注图，并按拍摄序列划分训练和测试。现有 240 张 SODD 图和 24 张 MOUD 图只够验证流程与暴露失败模式。

旧工业缺陷实验结果仍可在 [RESULTS.md](reports/RESULTS.md) 查看，它们与当前水下模型的指标不能混用。
# 0.4.0 视觉实验更新

已增加 UIEB 官方下载与配对评测、跨数据集分组划分、六类设施对照训练和独立水下机器人分支。数据、图像缓存和新训练权重统一保存在 `D:\CodexData\optical_agent`；本轮不调用付费 LLM。

网页“视觉配置”可选择旧默认、实验重训对照或实验改进候选。机器人合并 AUV/ROV 等类别，为实验能力；模型缺失、校准不足和质量失败会明确显示。

- [中文操作及优化思路](../docs/10_VISION_OPTIMIZATION.md)
- [实际视觉实验报告](reports/VISION_RESULTS.md)
- [机器可读摘要](reports/VISION_RESULTS.json)
- [实验审计](reports/VISION_EXPERIMENT_AUDIT.md)

旧 0.2.4 正式 Agent Eval 与 0.3.0 动态调度成绩保持原样。本轮是单种子的视觉实验，不能解释为新的正式 Agent 验收或可靠部署保证。
