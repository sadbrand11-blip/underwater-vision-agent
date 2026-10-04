"""Publish minimal derived summaries; keep raw sessions and request traces private."""
from optical_agent.paths import PROJECT_ROOT
import argparse
import json
from pathlib import Path

from optical_agent.adaptive_eval import save, summarize
from optical_agent.independent_eval import digest


def verify_pins(folder, pins, expected_paths):
    if set(pins)!=set(expected_paths):
        raise ValueError('Historical input path set changed')
    raw={path:(folder/path).read_bytes() for path in pins}
    if any(digest(raw[path])!=expected for path,expected in pins.items()):
        raise ValueError('Historical raw record, freeze or ledger bytes changed')
    return raw


def main():
    root = PROJECT_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    args = parser.parse_args()
    if args.run_id != 'adaptive-20260930-v030':
        parser.error('Historical publisher restricted to adaptive-20260930-v030; new runs need their own report')
    if not __import__('re').fullmatch(r'[A-Za-z0-9_-]{1,64}',args.run_id):
        parser.error('invalid run-id')
    folder = root/'runs/adaptive_eval'/args.run_id
    before_ids=['dev_quality','dev_pipe','dev_conditional','dev_decline','dev_followup','dev_unsupported']
    after_ids=['dev_conditional','dev_decline','dev_followup']
    expected_paths={'before/'+c+'_0.json' for c in before_ids}|{'fix1/'+c+'_0.json' for c in after_ids}
    expected_paths|={'fix1/'+c+'_'+str(i)+'.json' for c in ['demo_normal','demo_dark','demo_lost'] for i in [0,1]}
    expected_paths|={'before/freeze.json','fix1/freeze.json','http_budget.json'}
    pins=json.loads((root/'eval/adaptive_record_hashes.json').read_text(encoding='utf-8'))
    verified=verify_pins(folder,pins['files'],expected_paths)
    budget = json.loads(verified['http_budget.json'])
    before = [json.loads(verified['before/'+c+'_0.json']) for c in sorted(before_ids)]
    after = [json.loads(verified['fix1/'+c+'_0.json']) for c in sorted(after_ids)]
    demo = [json.loads(verified[p]) for p in sorted(verified) if p.startswith('fix1/demo_')]
    if len(before) != 6 or len(after) != 3 or len(demo) != 6:
        parser.error('Incomplete declared pilot; do not publish a complete report')
    expected_sets=[({('dev_quality',0),('dev_pipe',0),('dev_conditional',0),('dev_decline',0),('dev_followup',0),('dev_unsupported',0)},before),
                   ({('dev_conditional',0),('dev_decline',0),('dev_followup',0)},after),
                   ({(c,i) for c in ['demo_normal','demo_dark','demo_lost'] for i in [0,1]},demo)]
    freezes={phase:json.loads(verified[phase+'/freeze.json']) for phase in ['before','fix1']}
    for wanted,records in expected_sets:
        if {(r['case'],r['repeat']) for r in records} != wanted:
            parser.error('Exact case/repeat set changed')
        for record in records:
            phase='before' if record in before else 'fix1'
            if (record['status']!='recorded' or record['frozen_hash']!=digest(freezes[phase])
                    or record['source']!=freezes[phase]['inputs'][record['case']]
                    or len(record['turns'])!=(2 if record['case']=='dev_followup' else 1)
                    or [t['turn'] for t in record['turns']] != list(range(len(record['turns'])))):
                parser.error('Record status/hash/input/turn set changed')
    matched_ids = {r['case'] for r in after}
    matched_before = [r for r in before if r['case'] in matched_ids]
    counts = {'used':budget['used'],'limit':budget['limit'],'remaining':budget['limit']-budget['used']}
    cohorts = {name:summarize(records, len(records), counts) for name,records in
               [('development_initial',before),('matched_before',matched_before),('matched_after',after),('real_vision_demo',demo)]}
    rows = [t for r in before+after+demo for t in r['turns']]
    extra = []
    for t in rows:
        expected = t['expected']
        for event in t['adaptive']['result']['trace']:
            if event['type'] != 'tool':
                continue
            if ((expected.get('no_correction') and event['tool']=='generate_exposure_candidate')
                    or (expected.get('no_detection') and event['tool']=='detect_objects')):
                extra.append({'case':t['case'],'repeat':t['repeat'],'tool':event['tool'],'ok':event['ok']})
    examples = []
    for r in demo + [r for r in after if r['case']=='dev_decline']:
        t = r['turns'][0]
        result = t['adaptive']['result']
        examples.append({'case':r['case'],'repeat':r['repeat'],'source_kind':r['source']['kind'],
            'input_quality':{'exposure_state':r['source']['exposure_state'],'quality_pass':r['source']['quality_pass']},
            'goal':result['goal_contract'],'initial_plan':result['initial_plan'],'plan_revisions':result['plan_revisions'],
            'actions':[{k:v for k,v in e.items() if k in {'tool','arguments','ok','cached','observation_id','error_category'}}
                       for e in result['trace'] if e['type']=='tool'],
            'selected':result['selected_image_id'],'task_status':result['task_status'],'execution_status':result['execution_status'],
            'vision_status':result['vision_status'],'score':t['adaptive']['score'], 'visual_counts':t['adaptive'].get('visual_counts',{}),
            'computation_counts':result['computation_counts']})
    tokens = sum(t['adaptive']['result']['usage']['total_tokens'] for t in rows)
    if (budget['limit']!=120 or budget['used']!=100 or len(budget['attempts'])!=100
            or [x['number'] for x in budget['attempts']]!=list(range(1,101)) or tokens!=292537
            or sum(t['adaptive']['result']['request_attempts'] for t in rows)!=100
            or [sum(t['adaptive']['score']['passed'] for r in cohort for t in r['turns']) for cohort in [before,after,demo]]!=[6,4,3]):
        parser.error('Historical accounting/outcomes changed')
    public = {'version':'0.3.0','run_id':args.run_id,'formal_acceptance':False,
        'cohorts':cohorts, 'budget':counts, 'token_total':tokens,
        'freeze_hashes':{phase:digest(freezes[phase]) for phase in ['before','fix1']},
        'known_label_specific_extra_calls':extra,
        'scope_note':'注册工具率不等于工具选择正确率。评分中的irrelevant_calls只覆盖一般工具范围；信息丢失案例的禁用检测/校正另列。未修复或选择重跑真实视觉失败。'}
    save(root/'eval/adaptive_summary.json',public)
    save(root/'eval/adaptive_examples.json',examples)
    sections = ['# 0.3.0动态决策：真实评测与固定流程对照','',
        f'日期：2026-09-30；运行编号：`{args.run_id}`；模型：`deepseek-flash`；共 **{budget["used"]}/{budget["limit"]} 次HTTP请求、{tokens:,} token**。',
        '', '**本轮没有通过新的正式验收。** 开发修复后4/4轮符合预设任务，但真实视觉演示仅3/6次符合预期。固定Workflow在同一演示6/6次符合。程序拦截了全黑图的校正误调用，没有恢复纹理或越过可靠性规则。',
        '', '## 方法与范围', '',
        '- 6个开发案例共7轮：真实DeepSeek＋明确标注的固定视觉夹具。类别与框分数用于测试调度，不能算作检测准确率。',
        '- 单次开发修复后复测3个相同案例、4轮；首轮记录未覆盖。另用现有SODD检测器运行3种输入，每种2次。',
        '- 正常图为测试索引6；索引7乘0.6与索引8全黑为模拟扰动。三个来源不同，不能用它们估计真实相机重拍增益。',
        '- 独立评分字段不随模型请求发送。开发修复提示包含dev_followup的具体解析示例；修复后4/4是见过开发题的回归，不是独立泛化。标签由开发助手预先编写，未经外部人工复核，不修改原0.2.4正式20题或成绩。',
        '- 固定Workflow共享候选、过滤、比较工具；可校正时固定尝试两种候选。限定中文解析使用规则，无LLM费用。动态在先、固定在后，延迟对照受预热与缓存影响。',
        '- 新运行独立计数；重试/复测共用120上限。达到单任务预算的失败保留，没有用剩余额度挑选重跑后的最好结果。',
        '', '## 分组成绩', '',
        '| 分组 | 案例运行 | 轮次符合 | 案例符合 | 目标与图片符合 | 参数/前提通过 | 修订事件数 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    labels = {'development_initial':'初始开发','matched_before':'同题修复前','matched_after':'同题修复后','real_vision_demo':'真实视觉演示'}
    for name,c in cohorts.items():
        a=c['adaptive']
        sections.append(f'| {labels[name]} | {c["planned_cases"]} | {a["passed"]}/{a["turns"]} | {c["case_completion_rate"]:.1%} | {a["goal_conformance"]:.1%} | {a["parameter_rate"]:.1%} | {a["plan_revisions"]} |')
    sections += ['', '所有分组解析覆盖率100%。未注册工具实际执行、可靠性越权和程序结果篡改均为0。注册工具率100%仅说明调用了注册工具，**不说明调用时机正确**：全黑案例共5次不符合预设禁用条件的调用（1次检测、4次校正尝试）；4次校正均被前提检查拦截，校正实际计算为0。',
        '', '## 真实视觉演示逐次结果', '',
        '| 输入 | 重复 | 任务/执行状态 | 视觉状态 | 采用图片 | 独立符合 |', '|---|---:|---|---|---|---|']
    for r in demo:
        t=r['turns'][0];a=t['adaptive']['result']
        sections.append(f'| {r["case"]} | {r["repeat"]+1} | {a["task_status"]}/{a["execution_status"]} | {a["vision_status"] or "未交付完整结论"} | {a["selected_image_id"]} | {"通过" if t["adaptive"]["score"]["passed"] else "失败"} |')
    sections += ['', '正常图的空管道检测可完成任务，视觉结论为不可靠，不能证明不存在管道。偏暗第一轮采用Gamma候选，亮度中位数0.267→0.424，仍没有可靠的管道证据。偏暗第二轮模型把“不可靠”错误理解为“必须保留原图”，与比较工具采用规则冲突；纠错和额外检索耗尽10次模型/12次工具预算。',
        '', '两次全黑图均测得质量失败；模型仍请求不满足前提的校正，连续两次工具错误后停止。未产生候选，但调度没有完成预设的“交付质量失败可靠性报告”要求，因此都算失败，不能记作成功拒识。',
        '', '## 固定Workflow对照（同一真实视觉演示）','',
        '| 指标 | 动态Agent | 固定Workflow |', '|---|---:|---:|']
    cohort=cohorts['real_vision_demo']
    for field,title in [('passed','任务符合次数'),('latency_ms','耗时ms，均值/P95'),('tool_calls','工具调用，均值/P95'),('total_computations','视觉计算，均值/P95'),('tokens','token，均值/P95')]:
        values=[]
        for system in ['adaptive','fixed_workflow']:
            v=cohort[system][field]
            values.append(f'{v["mean"]:.1f}/{v["p95"]:.1f}' if isinstance(v,dict) else f'{v}/6')
        sections.append(f'| {title} | {values[0]} | {values[1]} |')
    sections += ['', '**本批次没有证明动态Agent比固定流程更快、更便宜或更准确。** 新能力是自由目标、观察驱动的候选选择与可追溯计划修订；小样本显示仍有额外调用和停止错误。注册工具率不能替代正确工具选择率，剩余问题尚未达到85%任务完成或95%参数/前提目标。',
        '', '## 失败与恢复', '', '| 阶段 | 案例 | 失败/不足 | 处理 |', '|---|---|---|---|',
        '| 初始开发 | dev_followup第2轮 | corrected误解析为current，绑定采用图而非最近候选 | 单次提示修复；相同案例复测通过 |',
        '| 初始开发 | dev_decline | 任务结果符合，但保留原图时没有修订记录 | 补充提示及完成条件；复测得到真实模型修订 |',
        '| 真实视觉 | demo_dark重复2 | 图片选择不符合比较采用规则，继而预算耗尽 | 保留失败，没有第二次针对性修复或重跑 |',
        '| 真实视觉 | demo_lost重复1、2 | 在质量失败时尝试校正，未完成质量失败报告 | 前提检查拦截；保留两次失败 |',
        '', '## 一次可观察的重新规划', '',
        '开发退化案例使用明示的视觉夹具：原图管道分数0.9，增强后0.4。真实DeepSeek选择两种候选，真实工具返回退化比较；模型引用该观察，修改计划为保留原图并检查可靠性。它没有提高分数或删除冲突：最终比较结论仍为不可靠。',
        '', '`质量观察 → 两种候选 → 原图/候选检测与评估 → 比较观察 → revise_plan（引用comparison编号） → 原图可靠性工具 → 选用真实比较证据结束`。',
        '', '这个例子证明模型会对观察作出可记录的动作选择，不证明学习到了水下视觉规律。修订事件数不等于有效且执行的重新规划次数：修复后开发组的另一事件属于纯校正任务，却提出未执行的比较工具；当时校验仅检查注册工具，评分未发现此范围缺口。初始计划也不是完整行动清单，部分分支未显式修订。历史成绩保留，不据此声称规划有效率100%。',
        '', '## 视觉计数与结果边界', '',
        '一对一IoU≥0.5匹配使用SODD提供的框。正常和偏暗来源的标注分别是二维码和渔网，不含目标“管道”：已检测的原图/候选管道TP=0、FP=0、FN=0，召回率无有效分母，应理解为不适用。全黑第1轮误执行检测，原图管道FN=2；第2轮未检测，不纳入检测样本。重复与候选来自相同来源，不是独立场景。逐图片计数见机器摘要中的示例，不能据此宣称召回提升。',
        '', '## 可复现与审计', '',
        '运行代码、提示、工具、题目、权重/校准、知识内容、请求配置与来源图片/标注保存哈希。初始和修复阶段各有冻结快照，共用一个请求计数器。原始脱敏记录保留在本地`runs/adaptive_eval/`，不公开密钥、原始图像或请求日志。',
        '', '公开的 [机器摘要](eval/adaptive_summary.json) 与 [必要轨迹片段](eval/adaptive_examples.json) 可核对本文数字。独立审计见 [审计记录](ADAPTIVE_EXPERIMENT_AUDIT.md)；同模型家族审核为暂定意见。操作见 [中文学习指南](docs/08_ADAPTIVE_AGENT.md)。',
        '', '## 审计后的离线补强与验证', '',
        '发布版本补强了修订步骤的任务工具范围、禁止校正且绑定原图时不能采用历史候选、跨批次终止状态持久化。另限制本报告生成器只能发布这个历史运行，核对完整案例/重复/轮次集合、冻结哈希、来源和账目。离线检查通过，未新增API请求，也没有改写历史成绩。补强后的版本尚未重新云端评测；此报告中的云端成绩对应保存的fix1冻结版本。',
        '', '完成131项完整Python回归后，另有跨调用网络停止与历史输入哈希保护测试通过，最新21项动态模块测试通过；9项网页检查通过。本机网页0.3.0健康检查、三种离线路径、缓存复用、图片访问和旧接口验证通过，离线成功不作为LLM能力证据。',
        '', '0.2.4正式结果保持原样。本轮属于开发验证和演示，0.3.0目前是受约束的研究原型，不能用本批结果声称正式验收通过或视觉准确率提升。']
    (root/'docs/reports/ADAPTIVE_EVALUATION_RESULTS.md').write_text('\n'.join(sections)+'\n',encoding='utf-8')
    print(json.dumps({'http_attempts':budget['used'],'tokens':tokens,'cohort_passes':{k:f'{v["adaptive"]["passed"]}/{v["adaptive"]["turns"]}' for k,v in cohorts.items()},'extra_calls':len(extra)},ensure_ascii=False))


if __name__=='__main__':
    main()
