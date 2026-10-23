# 通用自然图像数据与独立评测（0.6.0）

本版复用 torchvision 官方 COCO 预训练检测器，不训练视觉模型、不调用付费 API。全部下载、解压、预测缓存、分组清单、校准及结果位于 `D:\CodexData\optical_agent\general_v060`，不进入 Git。

## 下载与冻结

```text
python -m optical_agent.experiments prepare_general_data --download --sice --prepare
```

- COCO 来源为[官方数据页](https://cocodataset.org/#download)，下载 val2017 与 instances_val2017 标注。官方旧域名指向同一个 `images.cocodataset.org` S3 桶；下载器使用该桶的 HTTPS 路径式地址，逐块核对 Content-Range、长度，支持断点。图像包进行完整 SHA256 和 ZIP CRC 检查；标注只通过远程 ZIP 目录读取 instances_val2017 成员并核验该成员 CRC，省去训练/字幕标注下载，不冒充已下载完整标注包。SHA256 是本地完整性记录，不冒充官方发布摘要。
- SICE 仅尝试[作者仓库](https://github.com/csjcai/SICE)列出的 Google Drive 和百度网盘地址。若返回登录、配额或分享页，记录 HTTP 状态、原因和未运行状态，不下载未知镜像，不捏造 PSNR/SSIM。
- 后续取得 SICE 时将每个场景整理为 `sice/input/<场景>/images/*.jpg` 及可选 `reference.png`；新增数据必须新建实验版本，不能改写已有冻结清单。参考图是筛选的增强结果，不是唯一曝光真值，且必须确认对齐后才可进行逐像素参考评测。
- 逐图检查解码、尺寸和框坐标，异常项进入清单的 quarantine。像素哈希、64位 pHash（汉明距离≤4）与完整拍摄序列进行合并，再用种子 `20261023` 按组分为约60%开发、20%校准、20%测试。模拟扰动和参考图始终跟随原图；缺少序列的哈希分组不能保证真正的场景独立。

数据不随代码仓库重新发布。COCO 图片应遵循各自原始许可及[COCO使用条款](https://cocodataset.org/#termsofuse)，不能因项目代码采用 MIT 就把数据视为 MIT；SICE 的使用以作者发布条件为准。

## 评测顺序

```text
python -m optical_agent.experiments evaluate_general_vision --stage development --device cuda
python -m optical_agent.experiments evaluate_general_vision --stage calibration --device cuda
python -m optical_agent.experiments evaluate_general_vision --stage freeze --device cuda
python -m optical_agent.experiments evaluate_general_vision --stage test --device cuda
```

额外评测依赖为 `pycocotools==2.0.10`，安装在 D 盘隔离环境，不升级既有 Torch。原图基准计划覆盖全部有效 COCO val2017；诊断子集每集合按固定哈希取30张，与标签或检测结果无关。每张诊断图比较原图、亮度×0.6、亮度×1.4及横向0.6—1.4变化，并各自比较原样、Gamma、有界局部校正。扰动明确属于模拟，不解释为真实重拍。

- mAP 使用官方 `pycocotools.COCOeval`，原始分数下限0.05；不使用近似伪实现。另报告 IoU0.5、同类别、分数降序的一对一匹配召回和误报/图；重复框算误报，COCO crowd 区域按覆盖关系忽略。
- 原始预测缓存固定关闭校准读取，以保存同一模型的真实原始输出；测试工作点指标再应用冻结的逐类阈值。缓存中的 `calibration_status` 不代表当前网页加载校准后的状态，也不是端到端可靠性验收。
- 校准只读 calibration 集：每类至少30个 GT、10张正样本图、30个预测，阈值网格0.10—0.90步长0.05，在经验精度≥0.80约束下选择最大召回，否则 `insufficient` 并保留0.35候选阈值。分数桶的真实 TP 比例是该样本的经验精度，不能作为逐框正确概率或现场保证。
- 校准文件绑定完整模型SHA、预处理版本、类别版本及清单SHA，范围明确为 `coco_originals`；原图校准不自动适用于校正图，校正候选仍需独立证据与可靠性检查。
- 校正只用 development 诊断决定：候选相对原图平均宏召回增加>0.005，且每种输入误报增加≤0.05/图才可采用，否则默认保留原图。候选框变多、分数增大不算准确率提升。
- 校准和策略冻结后只进行一次保留测试；中断只复用完全相同身份的缓存，不挑选重跑后的最好成绩。模型、质量配置、实现及清单变化会拒绝继续旧实验。

## 结果边界

COCO val2017 是公开基准，本项目的重新分组不意味着模型从未见过这些图像。检测精度、模拟曝光诊断、真实多曝光校正和 Agent 调度必须分别报告。未获得 SICE 时其状态是 `not_run`，不能把 COCO 模拟扰动替代真实多曝光验收。详尽原始分母与逐图结果保存在 D 盘，公开文档只发布脱敏摘要。

## 本轮审查与复算记录

校准身份保护变更后，第一轮开发与校准结果完整归档并重新执行真实推理；从未通过测试集挑选版本。随后审查发现“只有一个有效分数桶时，校准不足类仍可能保留候选阈值”的边界，已加入手算回归并统一回退0.35。

该聚合层修复没有改变任何模型、图片变换或预测函数。本轮采用显式 `--stage reaggregate`：比对四个视觉源码完整SHA、预测与评分函数AST和其余模块结构，只读取原有真实预测，保留旧身份与摘要，不修改缓存中的签名。`aggregation_identity.json` 记录开发/校准数值、校准文件与校正策略完全一致的证明。新增聚合程序随后与全部运行源码一起冻结，保留测试使用当前真实代码签名。此特殊修复入口不允许绕过模型、输入、分组或推理实现变化。

## 2026-10-23 实际结果

COCO 5,000张全部通过解码、尺寸及标注框检查，36,781个标注中有446个crowd。像素与pHash检查未发现链接重复，按种子分为3,000/1,000/1,000张。SICE两份作者Drive文件实际返回HTTP 200的“Quota exceeded”页面；三个百度链接需要浏览器/账号下载，故真实多曝光与PSNR/SSIM标为 **未运行**。

唯一一次保留测试正常退出，覆盖1,000张原图、7,448个非crowd实例（另有96个crowd）。冻结的86个运行源码在测试后全部保持一致。

| 指标 | 真实结果 | 范围 |
|---|---:|---|
| COCO AP@[.50:.95] / AP50 | 24.12% / 39.46% | 官方COCOeval，原始模型分数下限0.05 |
| 固定0.35阈值的微召回 / 误报每图 | 36.36% / 1.525 | 同类别IoU0.5一对一匹配 |
| 冻结逐类工作点的微召回 / 误报每图 | 33.55% / 1.086 | 样本充分类用校准阈值，其余类0.35 |
| 满足校准集规则的类别 | 22/80 | 1,000张校准原图；其余58类不足 |
| 22类在测试集的合计精度 | 81.05% | 仍有10类单类测试精度低于80%，不能宣称每类达标 |
| 本地实验单图处理平均 / P95 | 110.29 / 132.98 ms | RTX3060 Laptop；读图、质量与推理，非LLM端到端延迟 |

开发诊断中两种校正均未通过预先约定的收益门槛，故默认保留原图，显式校正和比较能力继续保留。每个集合只取固定的30张诊断原图；测试的360条“输入变体×方法”结果不等于360张独立图，也不证明真实相机曝光恢复。当前冻结证据共有5,990份唯一预测记录，其中5,790份产生过实际模型前向，未调整候选复用缓存；这些计数不包含已归档的首轮诊断过程。

机器结果：D盘 `public_summary.json` 为不含个人路径或实际图像的精简摘要，`final_summary.json` 提供逐类及诊断指标。原始缓存、`evaluation_identity.json`、`aggregation_identity.json` 与 `freeze.json` 分别保留推理、聚合修复和最终测试身份。没有训练模型或新增付费API调用；本次视觉基准不作为正式LLM Agent Eval成绩。
