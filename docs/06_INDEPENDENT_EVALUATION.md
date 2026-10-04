# 第二步：独立评测、失败定位与调度改进

## 为什么要再做一次独立检查

第一步检查“结果是否满足已经解析的任务”。如果用户要求比较，模型却解析为曝光评估，程序可能正确完成错误的任务。

第二步把用户原意预先标成标准答案。评分程序读取标准答案与实际运行记录，不采用模型自己生成的完成条件；标准答案也不发送给模型。

流程：**预先标注 → 实际执行 → 独立评分 → 查看失败位置 → 针对性修复 → 相同案例复测**。

## 先看一次无需联网的例子

在项目目录运行：

```powershell
python -m optical_agent.experiments demo_independent_eval
```

这是故障注入演示：故意让解析器把比较任务当作曝光评估，随后用正确解析执行同一问题。

| 观察项 | 错误解析 | 正确解析 |
|---|---|---|
| 用户要求 | 比较校正前后识别 | 比较校正前后识别 |
| 实际理解 | 曝光评估 | 前后比较 |
| Action | 调用曝光评估 | 评估、校正、两图检测、比较 |
| Observation | 真实曝光指标 | 真实前后比较证据 |
| 第一层完成校验 | 通过 | 通过 |
| 独立评分 | 失败：没有满足用户原意 | 通过 |

详细记录在 `runs/agent_eval/offline_demo.json`。看 `expected`、`task_contract`、`trace` 中的工具、`evidence`、`independent_score`，依次对应预期、实际理解、Action、Observation 和评分。

## 题目与评分

`eval/agent_tasks_v2.json` 有10条开发案例、20条保留验收案例。部分案例有两轮，共用同一图片会话。旧50条题目仍保留。

每轮标准答案包含允许的任务类型、图片编号、任务状态、必要证据、引用要求和工具范围。澄清与不支持使用不同完成状态；质量失败也可以是正确完成的分析。检测结果允许为空，校正允许无需调整。

评分分别统计：

- 解析覆盖率：成功得到有效任务契约的用户轮数 / 所有已执行用户轮数。
- 任务类型、图片指向正确率：同时报告有效解析分母和全部请求分母。
- 工具选择率：工具范围、期望状态与必要结果满足独立标签的轮数 / 已执行轮数。
- 参数／前提正确率：参数合法且满足任务约束和前提的工具调用 / 全部工具调用。计算失败另计。
- 独立完成率：全部条件满足的轮数，以及所有轮都通过的案例数；另报计划案例分母，把未运行案例保留在报告中。
- 耗时、模型调用、工具调用、token：案例总量含准备轮与追问，报告平均值和P95。token只使用服务确实返回的用量，缺失时显示不可用。

完整比较还要检查两图质量、校正参数和两图检测确实存在。证据存在但未在报告选用也不能完成。图片、数值、引用和可靠性状态必须与本地真实结果一致。

失败分类：任务理解、图片指向、工具选择、参数／前提、计算、报告、API、预算、解析、准备轮、事实不一致。恢复成功的案例仍保留中间错误，不能用最终完成掩盖调用错误。

## 运行和查看真实评测

本轮结果见 [中文报告](reports/INDEPENDENT_EVALUATION_RESULTS.md)。原始脱敏记录保存在 `runs/agent_eval/<运行编号>/`，与网页会话日志分开，且不提交Git。

```powershell
# 离线工程检查，不消耗API，不代表LLM准确率
python -m optical_agent.experiments evaluate_orchestration --task-file eval/agent_tasks_v2.json --backend scripted --split dev --run-id offline-check --phase offline

# 真实评测会产生API费用。须配置本地.env，并为同一阶段保留同一个运行编号。
python -m optical_agent.experiments evaluate_orchestration --task-file eval/agent_tasks_v2.json --backend live --vision fixture --split dev --run-id your-authorized-run --phase before --max-http-attempts 120

# 指定失败案例和同类成功案例复测；仍使用同一运行编号
python -m optical_agent.experiments evaluate_orchestration --task-file eval/agent_tasks_v2.json --backend live --vision fixture --split dev --task-ids dev_02 dev_04 --run-id your-authorized-run --phase fix1 --max-http-attempts 120

# 当前真实图片小批次：SODD测试索引6、7、8、9、10，分别对应五类任务
python -m optical_agent.experiments evaluate_orchestration --task-file eval/agent_tasks_v2.json --backend live --vision sodd --split dev --task-ids dev_01 dev_02 dev_03 dev_04 dev_06 --run-id your-authorized-run --phase after --max-http-attempts 120
```

开发和真实图片使用不同视觉后端，不能互相当作前后改善对照。`paired_comparison` 只比较相同题目、视觉后端、图片内容及重复编号。

同阶段重新运行会跳过已完成的案例，不重复付费；中途尚未保存的案例可能需要重跑，但原请求计数保留。代码改变须选择下一阶段，不能覆盖原阶段记录。

## 请求额度和记录

`http_budget.json` 在每次实际HTTP请求前原子计数。重试、准备轮、错误恢复与复测都共用额度。第121次请求不发出，重启程序不重置计数。保守计数可能包含已预留但进程崩溃时尚未发出的请求；不会因此多发请求。计数文件损坏或上限改变时停止。

已有运行的计数文件丢失也会停止，初始化标记和运行manifest共同防止恢复时从零计数。不要删除计数文件或另换运行编号来继续同一授权批次。

401、403、402立即停止批次；连续两个案例发生连接或服务不可用错误也停止。达到额度保存部分结果和未运行任务。每轮仍最多10次模型调用、12次工具调用、180秒。

`manifest.json` 在联网前冻结整份题目哈希与模型名。各阶段记录代码提交、相关源文件哈希、提示和工具说明哈希；原始模型片段先脱敏。服务有返回模型版本时，记录在模型调用的 `diagnostics.served_model`。

题目标签由本轮依据计划预先编写，尚未经过外部人工复核。发布时仅更正了文件中标签来源的说明，所有案例和评分标签与联网前快照完全一致；旧运行的精确输入在其 `frozen_tasks.json` 中。复用旧运行编号时用该快照作为 `--task-file`，以保持全文哈希一致。

## 这一步能够证明什么

它能发现并改善任务理解、工具调用和报告交付方面的问题。10条开发案例与5张真实图片属于小样本试运行；20条保留验收案例本轮未运行，不能宣布正式验收完成。

固定视觉结果用于隔离调度问题。真实图片使用现有检测模型，没有重新训练。任务完成率提高不能解释为目标识别召回率提高。
