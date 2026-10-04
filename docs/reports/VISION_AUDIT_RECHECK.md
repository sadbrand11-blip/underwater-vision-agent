# 0.4.0 交付入口只读复查

日期：2026-10-01。

**总判定：WARN。保留首次审核的 WARN，不撤回或改写原始发现。**

**交付修订结论：本轮复查范围内，已报告的入口问题均已关闭；Gamma-only 与完整校正的差异通过明确披露关闭。冻结的单帧数值、模型和九个核心源码未被本轮修订改变。**

review_independence: same-family  
acceptance_status: provisional  
reviewer_model: gpt-6-astra  
reviewer_reasoning: ultra  
reviewer_task: /root/vision_integrity_audit  
reviewer_thread: 01a0f5fb-6226-75b0-a839-292f9ff7d29d

本轮是初审的补充记录，不是新的模型训练、选参、测试集阈值搜索或正式 Agent Eval。主审及两位助手只读检查了实现与证据；主审仅向获授权的审计追踪目录写入报告及助手请求、回复和身份记录。没有读取 .env、调用付费 API 或改写实验文件。

路径：
- R = <LOCAL_WORKSPACE>
- D = D:/CodexData/optical_agent/vision_v040
- P = D:/CodexData/optical_agent

## A. Ground-truth provenance — WARN，原限制保留

本轮没有变更任何 GT、数据划分或冻结输入。重新核对原 freeze.json 中全部 **12,384 个路径**，均与原摘要一致。数据来源、UIEB 配对及 pHash 分组的初审结论保持有效；本轮没有把交付修订当作数据完整性的替代证据。

九个核心源码分别为 quality.py、detector.py、agent.py、metrics.py、train_vision_detectors.py、evaluate_vision_exposure.py、evaluate_vision_detectors.py、optical_agent/vision_data.py、optical_agent/vision_router.py。逐项比较：

1. 当前 R 中源码；
2. D/frozen_source/ 中归档字节；
3. D/frozen_source_manifest.json 的哈希；
4. 原 D/freeze.json 的哈希。

结果为 **9/9 四方一致，逐字节比较也通过**。原冻结源码对应 freeze.json:10–18；归档清单对应 frozen_source_manifest.json:4,8,12,16,20,24,28,32,36。

重要时间边界：归档清单修改时间为 2026-10-01 05:52:59.832928 UTC，晚于原冻结和测试。它是事后保存、与原冻结摘要一致的副本，不能被称为测试前保存的源码副本。

关键哈希：
- freeze.json：d3a8687956d3673bf5feda1e5f98eccbc78beaf5fec692396cbefac283fb01e7
- frozen_source_manifest.json：bbb7e1837e74f8b1ae93a8e7ceb77b9c8aba04f799b165de77b4185c2ab081f9
- manifest.json：5fed9e376d7b115ed3a6319b1d1f1ca8711ca8fec52b1c97d969f469713e9203

场景独立性仍未证明，项目重新划分仍不是原官方测试集；MOUD 的 24 张诊断图片仍没有进入原预冻结。当前报告已在 R/VISION_RESULTS.md:105 和 R/report_vision.py:62,124 明确说明后者。后补说明不能补造历史预注册。

## B. Score normalization and metric validity — WARN，数值规则未变

原始/校正/最终/接受检测指标、曝光指标及采用门槛未改。当前 VISION_RESULTS.json 的模型、曝光、设施采用门槛、压力诊断和训练数据块逐字段与已审原始记录一致。

因此初审中以下判断继续有效：

- 未发现模型依赖分母或虚假归一化；
- 一对一匹配、固定阈值测试和原始 AP 结果成立；
- 校正/最终/接受 AP 候选已经被运行阈值截断，不能与原图 AP 当成相同候选下限的比较；
- 校准为每分支跨类合并的小样本 isotonic，只使用原图，未建立逐类或校正图校准保证；
- 机器人自动接受精确率约 30.95%、召回约 17.33%，不能升级为部署可靠性。

当前说明对应 R/VISION_RESULTS.md:70,84,112；R/report_vision.py:57,61,129。

交付包装器仅修改多帧结论和展示。单帧在 R/optical_agent/vision_delivery.py:14–15 直接调用冻结实现，因此不把本次多帧修复传播成“原单帧成绩提高”。

## C. Actual results and records — WARN，实测一致，历史追溯限制保留

### 数值和报告

当前 VISION_RESULTS.json 的 data/grouping/quarantined/models/training/stress/exposure/facilities 全部与原始清单、曝光摘要和检测摘要一致。原有采用门槛仍失败，未改变默认采用状态。

通过内存拦截全部输出写入，执行报告生成逻辑，得到的 Markdown 和 JSON 与已保存版本完全一致；未实际生成或修改报告文件。CSV 仍为 **1,579 行，624 个检测错误行，1,491 个不可靠行**。

当前原始摘要：
- D/detection_test/summary.json SHA256：9f6d61b964aa952d2e120fed84312ad83b7c175a31b472042c1f562c91f76cd0
- D/exposure/test_summary.json SHA256：9397ddfcbdf489603a078393b3de4e5c28b2c0361f7e1949af14b110bf93fb40

没有初审前 report_vision.py、Markdown、JSON 的独立字节备份或 Git 历史；能够证明的是当前数值与已审原始记录一致，不能声称已逐次审计所有历史编辑。原始摘要自身也不在测试前输入冻结中；文件修改时间不能代替不可改写的历史审计。

本轮还发现默认 CP1252 环境读取报告输入会失败；随后当前 R/report_vision.py:39–42,98 已显式指定 encoding='utf8'。此修订没有改变已保存 Markdown 或 JSON 的字节。

### 测试和网页记录

实际读取到的日志：

- P/tests_v040_targeted_final.log：**31 passed in 6.67s**。
- P/tests_v040_final_delivery.log:4：**163 passed in 153.57s**。
- 最终 P/tests_v040_release.log:4：**163 passed in 154.94s**。

注意：release 路径最初存在旧的 155 项记录，随后被清空并重新运行。主审没有将旧记录或运行中的空日志当作本次通过证据；以上最终数值来自本次完成后的第 4 行。

首次复查审核的六个 API 原始案例位于 D/web_check/，与当时 R/VISION_VERIFICATION.json 一致。封存前又独立审核了最终运行 D/web_checks/v040_delivery_r1_20261001T062037996659Z/ 的六个案例；其本地 summary.json 与当前公开 JSON 字节相同，各案例逐字段相符：

- legacy_exposure：质量 1，检测 0；
- candidate_pipe：检测 1，目标 pipe_type2/pipe；
- candidate_robot：检测 1，目标 underwater_robot；
- joint_targets：检测 2，目标 pipe_type2/pipe/underwater_robot；
- correction_robot：质量 2、校正 1、检测 2、比较 1，返回 unreliable；
- lost_robot：质量 1、比较 1、校正 0、检测 0，返回 quality_failure。

六例均 completed、零 LLM 请求尝试，模式均为 scripted / offline_adaptive_rule_fixture。新增真实校正记录关闭了初审“原四例没有执行任何校正”的证据缺口。

R/docs/10_VISION_OPTIMIZATION.md:89–93 的具体数值也与 correction_robot.json 一致：亮度中位数 0.105882→0.152941，Gamma 0.8，CLAHE 0，选择 candidate_gamma，状态 unreliable。候选数不作为准确率证据。

前一批 D/web_check/ 六例在最终补齐 CLI/框图/全别名断言之前生成，没有被冒称为后续源文件的验证。封存前完成的唯一目录运行标识为 v040_delivery_r1_20261001T062037996659Z，公开记录明确引用最终 release 日志的 163 passed / 154.94s。主审逐项核对了该轮六份原始 JSON、运行修订标识、计数、目标类别、模拟标记和零模型请求记录；同时保留最后几项变更的只读内存探针及 31 项针对性测试证据。

最初四份网页原始记录曾被前一批六例运行写回同一目录，未留下本目录中的不可变原四例副本。它们没有进入原视觉冻结，不影响已冻结视觉成绩。当前 R/verify_vision_web.py:30–31 已改成唯一时间戳的 D/web_checks/<run_id>；:80–81 记录 run_id、raw_records 和测试来源。最终唯一目录实测已经检查，现有 D/web_check/ 六例仍保留。

最终公开验证 JSON 和唯一运行 summary.json 的 SHA256 均为 afb060d20b4f33423552cf78069b0544ece2d41f223a42e60e90a2c9d347ab0e；其引用的 release 日志 SHA256 为 8915e0473f5061bdd00d65ac937a9327adeb7b7ad630f4bda6c5e82873fc9c1d。

脚本检查 API JSON，不验证浏览器实际渲染或真实 LLM 能力。

## D. Dead code and delivery correctness — PASS（限已报修订案例）

首次报告中的问题和中间复查发现均保留。下面记录最终关闭方式及其影响边界：

| 初审/复查发现 | 最终状态 | 证据及独立复现 |
|---|---|---|
| 多帧忽略质量合格但不可靠的冲突帧 | 已关闭当前交付入口 | vision_delivery.py:24–37 检查所有质量有效帧；同一原探针现在 unreliable、接受框 0、一致帧 0。 |
| CLI 仍调用冻结多帧实现 | 已关闭 | run_agent.py:10,28 改用 DeliveryOpticalAgent；通过该导入类执行冲突帧探针同样拒绝。 |
| 拒绝后最终接受框图仍有旧框 | 已关闭 | vision_delivery.py:53–58 重绘最终接受框图；数组等于无接受框选中图，原图/候选框图仍可检查。 |
| 分支 v1→v2 后缓存混合 | 已关闭已报场景 | adaptive_tools.py:56–71 将分支对象、版本、阈值、校准和校准状态加入指纹；:209–211 仅合并当前版本。旧会话拒绝版本替换，手工残留旧版本缓存不会进入结果。 |
| 管道＋机器人及其他已支持类别联合请求被缩减 | 已关闭已报场景 | adaptive_fixture.py:12–19 从全部类别和别名累加并去重。管道＋机器人、螺旋桨＋机器人、全七类、红鳍片、网具、二维码、小写 auv/rov 和 pipe_type2 的相应断言通过；设施/机器人按需要各调用一次。 |
| 旧调度＋实验模型调用所有分支 | 已关闭网页暴露组合 | app.py:205–206 在推理前拒绝该组合；内存 Flask 探针 HTTP 400，模型调用 0。static/conversation.js:3–5 同时限制 UI 组合。 |
| 实验多分支多帧缺乏匹配评测 | 明确约束 | app.py:85–86 限制实验配置为一张原图及其候选；默认多帧使用交付保护。 |
| Gamma-only 与完整校正方法不一致 | 披露缺口已关闭 | 报告 VISION_RESULTS.md:82、操作文档:91–95 明确两条路径不同。方法没有被悄悄改成相同；完整校正的测量不转移给 Gamma-only 网页结果。 |

测试证据：早期独立执行七项针对性回归通过；最新三项修订再次通过对应只读内存探针和数组/类别断言。导入 app 的探针将 load_local_env 替换为内存空操作，以遵守不读取 .env 的限制。

冻结的 agent.py 仍保留历史多帧行为，因为它是原实验来源。修复通过独立交付包装器实施。当前 app 和 CLI 的已核验入口使用包装器；这不是对外部任意代码直接调用旧 OpticalAgent 的保证。

几何对齐仍属于输入约定，没有通过多帧修复变成经过验证的几何配准。单帧可信度、总体部署精度及自然语言任意表达的理解能力也不由上述有限回归测试证明。

## E. Scope and adoption — WARN，披露改善，证据范围未扩大

仍是单种子、小 SODD 选模/校准/测试集和自定义哈希分组。修订未增加独立场景、负设施图片、独立校准集或多种子统计。

原曝光与设施采用门槛继续失败。当前报告正确保留旧默认，机器人仍为实验能力，未声称新的正式 Agent Eval。

新增修复涉及可用性、数据流和证据展示，不构成模型性能提升。API 六例和 163 个软件测试不能换算成识别准确率，也不能替代浏览器视觉验证或真实 LLM 规划验收。

延迟限制已补充于 VISION_RESULTS.md:80,105 和 report_vision.py:64–65，排除模型加载、解码、框图、网页渲染和 LLM；质量评估与曝光局部计时的边界也有说明。

恢复过程在 docs/10_VISION_OPTIMIZATION.md:65–77 明确为历史失败后的恢复，不是逐位重放。给出的 --branch robot --strategy improved --epochs 4 --resume 命令与实现参数一致，但没有执行该命令。缺失的中间源码快照、running 元数据和后补说明的限制均保留，没有改写原始训练记录。

## F. Evaluation classification — PASS

初审分类保持不变：

- UIEB 参考一致性：real_gt（数据集提供增强参考；不是物理曝光真值）。
- SODD/UIIS 原图及校正检测：real_gt。
- 数字亮度 ×0.6 与空候选压力实验：simulation_only，检测指标可继承真实框。
- UIEB challenge/MOUD 无参考质量统计：self_supervised_proxy。
- 本轮 Stub/全黑/多帧冲突探针：simulation_only 的软件功能检查。
- 六例网页记录：实际本地模型的功能烟雾检查；没有新的 GT 准确率评测。
- 没有新增 human_eval，也没有把本系统输出当作 GT。

## 结论与剩余限制

当前限定范围内的已报交付缺陷均已关闭，新增方法和统计限制说明准确。**允许保留现有实验报告，并以 v040_delivery_r1 描述交付修订；不应据此把研究完整性总判定改成无条件 PASS。**

继续保留的 WARN 来自：

1. 真正场景独立性、单种子和有限样本。
2. 跨类原图校准的支持度及校正图校准缺口。
3. MOUD 和部分依赖/追溯材料未预冻结。
4. 初始失败试跑/中间重试的完整源码快照缺失及部分原始状态元数据陈旧。
5. 早期网页记录曾覆盖；后续唯一运行目录只是改善未来追溯，不能补造原四例的不可变记录。
6. 没有浏览器实际渲染或新正式 LLM Agent Eval 的独立验收。

首次报告 001-vision-review.response.md 已从原始主审会话中的 final_answer 逐字提取，未进行措辞更新；其 UTF-8 SHA256 为：
7996fa5e3cb776ffb4b8dabc8ab1f99b75db1f7c6a5b9d1a323afc9eaa9878e1

## 复查源码指纹

以下为主审封存阶段读取的 SHA256，用于识别已复查源文件。没有把这些交付指纹加入或倒填原实验 freeze.json。

| 文件 | SHA256 |
|---|---|
| optical_agent/vision_delivery.py | 71a0130a68229e8f510f7bb6af61e26948b7c8f47a5151467447fe99e73661ee |
| app.py | 6938b4a1f64cfed792ea1cc94d469768268bc5380886e349828526ae29ebd540 |
| optical_agent/adaptive_tools.py | ee24213b97f444fcdc68d229d0b6d88a5a2f12edb93ed3b8b64a26c8b645e9d9 |
| optical_agent/adaptive_fixture.py | e9f0e924f85193a07ce076f47f530ca8c72f2b6f01c802c579a1197f1c08d947 |
| run_agent.py | 09bd60835c7a6c8c307717f0cbf2b348e93cc447bd61cf63af4f9863269bf2b5 |
| static/conversation.js | 623d0e327073ad55a98971c084203379ef605917b05c064744af7cdf0c9c8e9e |
| verify_vision_web.py | 6698e9f3da166fe76a691154e7e35eaf58487560afa7696ecfebc305e14cdb7b |
| tests/test_vision_v040.py | 4eaa4f6fc77bdbee04746d7e186cdcacf3b86cbf493d7fa6efc6ec37548ede5f |
| report_vision.py | 6399606bdb5e0100a91bd663498c8d2dd8c3cad82ec7455171e9bea839630b54 |
| VISION_RESULTS.md | 79b1db4d05b87cbb2f4ec822c90d8d4b6ada05bc3bc7a14ac2f9db7df6e76c69 |
| VISION_RESULTS.json | aea3c20ed997f737db23acc159fa74a848da0c05a5a24f65b37849dc0584fb85 |
| docs/10_VISION_OPTIMIZATION.md | 61eaf1338b5656780d5d326c8b550bc5d49c183180803ae5d2d88f93486b1d06 |

## 追踪文件与助手身份

主审确认两位助手的模型/推理身份来自各自本地会话的 session_meta 与 turn_context，而不是根据自述推断。早期助手回复里的 unknown/inherited 原文照存，其后身份确认单独记录在 meta.json。

- 数据助手：/root/vision_integrity_audit/data_provenance；线程 01a0f5fb-ae56-7601-94f5-39ff07604192；gpt-6-astra，ultra；同家族暂定。
- 运行助手：/root/vision_integrity_audit/runtime_audit；线程 01a0f5fb-cfde-7ea2-a6d4-f528ca9feaa7；gpt-6-astra，ultra；同家族暂定。

本轮主审写入的文件均位于 R/.aris/traces/experiment-audit/2026-10-01_run01/：

- 001-vision-review.response.md：首次主审完整原文。
- 002-delivery-review.response.md：本次完整复查。
- 003-data-provenance.requests.json：数据助手全部直接请求和补充指令。
- 003-data-provenance.response-001.md：数据初审完整回复。
- 003-data-provenance.response-002.md：数据复查完整回复。
- 003-data-provenance.meta.json：模型、推理、任务、线程、回复来源与哈希。
- 004-runtime-audit.requests.json：运行助手全部直接请求和补充指令。
- 004-runtime-audit.response-001.md：初次运行审计完整回复。
- 004-runtime-audit.response-002.md：初次网页记录及最小复现完整回复。
- 004-runtime-audit.response-003.md：交付初次复查及当时剩余问题。
- 004-runtime-audit.response-004.md：最后三项修订的关闭复查。
- 004-runtime-audit.meta.json：模型、推理、任务、线程、回复来源与哈希。

父代理原先创建的 001-vision-review.request.json 和 run.meta.json 原样保留，没有由本审计覆盖。
