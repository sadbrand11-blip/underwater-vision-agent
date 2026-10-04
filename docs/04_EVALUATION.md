# 第四步：怎样判断 Agent 做对了

视觉模型评测与调度评测是两个问题。前者需要目标框标注，后者需要用户任务与预期证据。当前知识解释只支持引用已有片段，尚不评估自由生成的科研论证。

`eval/agent_tasks.json` 固定50条任务，五类各10条：曝光、检测、校正比较、依据解释、多轮追问。每类前6条开发、后4条验收；验收任务各运行3次。这是小型、模板化任务集，后续需要增加独立措辞与真实用户问题。

运行离线执行器检查：

```powershell
python evaluate_orchestration.py --backend scripted --output runs\orchestration_scripted.json
```

这里模型响应和检测器均为测试夹具，不能用结果证明大模型或视觉模型有效。live_targets_met始终为false。

换成真实SODD检测器，保留规则调度：

```powershell
python evaluate_orchestration.py --backend scripted --vision sodd --output runs\orchestration_sodd_scripted.json
```

这能验证真实视觉工具的接线和CV计数，但仍不能验证云端模型路由。默认验收任务映射到4张来源图的曝光变体，每条重复三次；它不是新的独立场景检测基准。原有40图视觉评测仍使用evaluate_agent.py。

配置密钥后，真实调度验收命令：

```powershell
python evaluate_orchestration.py --backend live --vision sodd --output runs\live_orchestration.json
```

## 验收口径

- 工具选择率按任务统计：调用工具与任务相关，并产生任务要求的证据；不要求固定顺序。
- 参数正确率按工具调用统计：调用被执行器接受，满足类型、图像存在、校正和比较前提；没有调用时为null。
- 任务完成：成功结束，引用所需证据，报告值与实际观察相同；依据解释还必须有引用。
- 识别准确性：另用标注一对一匹配统计召回、误报和拒识，不用“回答完成”替代识别正确。
- 真实LLM目标：选择≥90%、参数≥95%、完成≥85%，未注册执行和规则越权为0；scripted不计作达标。

三个对照系统为固定视觉流程、完整视觉工具调度、独立视觉工具调度。固定流程没有RAG，解释任务单独展示；共同视觉任务单独比较。多轮任务的准备轮必须有比较证据。任务汇总的耗时、模型和工具次数、token及参数错误包含准备轮和末轮；各轮耗时、用量和轨迹也单列保存。

0.2.2 API的 task_status=completed 表示通过**已解析任务**的完成校验；它不保证意图解析正确。评测仍须用独立预期任务核对 task_contract 的类型和图片指向，以及任务要求的证据，不能直接将自校验通过率当作意图准确率。CV统计只纳入报告选用完整比较证据的输出；当前为比较与依据解释任务，和“共同视觉任务完成率”的分母不同。

新增任务理解调用与其 token、耗时计入原有每轮预算和评测用量。新增离线校验示例可运行 `python demo_task_validation.py`，明确为合成响应回放。本轮不运行收费 live 验收，记录见 [任务完成校验验证](../TASK_VALIDATION_RESULTS.md)。

新增误检使用一对一类别/IoU≥0.5匹配后的“每图误检数正增加量”；原历史报告采用旧计数，不覆盖它。token未提供时输出null，不当作零成本。

练习：找一条完成失败的任务，对照预期证据与trace；说明“拒识”为什么仍可能是正确的调度结果，而端到端识别召回仍要计入漏检。
