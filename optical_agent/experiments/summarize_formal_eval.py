"""Generate a public Chinese report from the durable formal result (no API calls)."""
from optical_agent.paths import PROJECT_ROOT
import argparse
import json
from pathlib import Path

from optical_agent.formal_eval import QUALIFICATION, formal_metrics
from optical_agent.eval_integrity import verify_formal_artifacts
from optical_agent.independent_eval import write_json


def percent(value):
    return '未提供' if value is None else f'{value * 100:.1f}%'


def summarize(run_dir, root):
    run_dir, root = Path(run_dir), Path(root)
    integrity = verify_formal_artifacts(run_dir, root)
    result = json.loads((run_dir / 'formal_sodd_live.json').read_text(encoding='utf-8'))
    manifest = json.loads((run_dir / 'formal_manifest.json').read_text(encoding='utf-8'))
    cases = [c for c in json.loads((run_dir / 'frozen_tasks.json').read_text(encoding='utf-8'))['cases'] if c['split'] == 'test']
    computed = formal_metrics(result['cases'], cases, backend=result['backend'], vision_backend=result['vision_backend'], repeats=result['repeats'])
    if any(computed[k] != result['metrics'][k] for k in computed):
        raise ValueError('保存的成绩与原始记录重新计算结果不符')
    lookup = {(r['task_id'], r['repetition']): r for r in result['cases']}
    failures = []
    case_results = []
    for case in cases:
        rounds = []
        for repetition in range(3):
            row = lookup.get((case['id'], repetition))
            rounds.append({'round': repetition + 1, 'completed': bool(row and row['completed']),
                           'executed': row is not None, 'journal_status': row.get('journal_status') if row else 'not_started'})
            if row:
                for turn in row['turns']:
                    if turn['score']['failure_categories'] or not turn['score']['task_completed'] or turn['result'].get('format_repairs'):
                        r = turn['result']
                        failures.append({'case': case['id'], 'round': repetition + 1, 'turn': turn['turn_index'] + 1,
                            'message': turn['message'], 'expected_task': turn['expected']['task_types'],
                            'expected_images': turn['expected']['image_ids'], 'actual_contract': r.get('task_contract'),
                            'completed': turn['score']['task_completed'], 'failures': turn['score']['failure_categories'] +
                                (['format_repair'] if r.get('format_repairs') else []),
                            'format_repairs': r.get('format_repairs', 0),
                            'execution_status': r['execution_status'], 'task_status': r['task_status'],
                            'selected_evidence': [e['evidence_id'] for e in r.get('evidence', [])],
                            'tools': [{'tool': t['tool'], 'arguments': t.get('arguments'), 'ok': t.get('ok'),
                                       'error_category': t.get('error_category'), 'cached': t.get('cached')}
                                      for t in r['trace'] if t['type'] == 'tool']})
                if row.get('journal_status') == 'interrupted':
                    failures.append({'case': case['id'], 'round': repetition + 1, 'turn': None,
                                     'completed': False, 'failures': ['interrupted'],
                                     'reason': row.get('interruption_reason')})
        case_results.append({'id': case['id'], 'kind': case['kind'], 'rounds': rounds})
    served = sorted({t.get('diagnostics', {}).get('served_model') for row in result['cases']
                     for turn in row['turns'] for t in turn['result']['trace']
                     if t.get('diagnostics', {}).get('served_model')})
    summary = {'schema_version': 3, 'run_id': run_dir.name, 'qualification': QUALIFICATION,
               'started_at': result['started_at'], 'updated_at': result['updated_at'],
               'model': result['model'], 'served_model_aliases': served, 'baseline_behavior_version': '0.2.3',
               'task_sha256': result['task_sha256'], 'manifest_sha256': result['manifest_sha256'],
               'versions': manifest['bindings']['versions'], 'metrics': result['metrics'],
               'budget': result['budget'], 'stop_reason': result['stop_reason'],
               'vision_metrics': result['vision_metrics'], 'cases': case_results, 'failures': failures,
               'report_integrity': integrity,
               'metric_scope': '预设粗粒度任务oracle的流程符合度；未评价指定类别筛选，不代表完整语义满足率',
               'format_repairs': sum(t['result'].get('format_repairs', 0) for row in result['cases'] for t in row['turns'])}
    m = result['metrics']
    text = ['# 正式 Agent Eval 验收报告', '', f'运行编号：`{run_dir.name}`；{QUALIFICATION}', '',
            f"待测基线：0.2.3 Agent行为；冻结提交 `{manifest['bindings']['versions']['code_commit']}`。",
            f"模型：请求 `{result['model']}`，服务返回别名 `{', '.join(served)}`；无法固定云端内部模型版本。", '',
            '## 1. 评测是否完成，指标是否达标', '',
            '**成绩含义：预设粗粒度任务oracle的流程符合度。评分核对任务类型、图片、工具和必要证据；尚未评价用户指定类别的筛选，不代表完整用户意图满足率。**', '',
            f"评测状态：**{m['evaluation_status']}**；验收状态：**{m['acceptance_status']}**；正式通过：**{m['formal_acceptance']}**。",
            f"已执行 {m['executed_cases']}/60 次案例、{m['executed_turns']}/78 轮问题；停止原因：{result['stop_reason'] or '无'}。", '',
            '| 指标 | 成绩 | 要求 |', '|---|---:|---:|',
            f"| 预设oracle案例流程符合率（计划分母） | {percent(m['planned_case_completion_rate'])} | ≥85% |",
            f"| 按轮的工具路线/必要证据符合率 | {percent(m['tool_selection_accuracy'])} | ≥90% |",
            f"| 参数正确率 | {percent(m['argument_accuracy'])} | ≥95% |",
            f"| 解析覆盖率（已执行问题） | {percent(m['parsing_coverage'])} | 另行报告 |",
            f"| 解析覆盖率（全部78轮计划问题） | {percent(m['planned_parsing_coverage'])} | 另行报告 |",
            f"| 任务类型匹配率（已解析） | {percent(m['intent_accuracy_on_parsed'])} | 另行报告 |",
            f"| 图片指向正确率（已解析） | {percent(m['image_accuracy_on_parsed'])} | 另行报告 |",
            f"| 未注册工具实际执行 / 可靠性越权 / 事实或原图违反 | {m['unregistered_tool_executions']} / {m['reliability_overrides']} / {m['fact_violations']} | 均为0 |", '',
            '正确澄清、说明不支持、质量失败、空检测和无需校正，均按预设标签评分；标签不发送给模型。无工具调用的澄清、拒绝或直接缓存交付也参与按轮工具路线评分。',
            '“事实违反为0”表示报告与工具缓存相符、原图保持一致，检测结果仍可存在实际误报。', '',
            '## 2. 三轮成绩与每题结果', '',
            '| 轮次 | 完成案例 | 工具选择 | 参数正确 |', '|---|---:|---:|---:|']
    for round_id, rm in m['rounds'].items():
        text.append(f"| {round_id} | {percent(rm['planned_case_completion_rate'])} | {percent(rm['tool_selection_accuracy'])} | {percent(rm['argument_accuracy'])} |")
    text += ['', '| 案例 | 类型 | 第1轮 | 第2轮 | 第3轮 |', '|---|---|---|---|---|']
    for row in case_results:
        values = ['通过' if r['completed'] else '失败' if r['executed'] else '未运行' for r in row['rounds']]
        text.append(f"| {row['id']} | {row['kind']} | " + ' | '.join(values) + ' |')
    text += ['', '## 3. 调用用量和耗时', '',
             f"持久化请求计数：**{result['budget']['used']}/{result['budget']['limit']}**，包括重试；记录未归属的预留尝试：{m['unattributed_reserved_attempts']}。",
             f"服务返回已知token：{m['known_token_usage']}；{m['token_usage_turns']}/{m['executed_turns']}轮有用量数据。", '',
             '| 每次案例运行（多轮问题合计） | 平均 | P95 |', '|---|---:|---:|']
    for key in ('model_calls', 'tool_calls', 'tokens', 'latency_ms'):
        stat = m[key]
        text.append(f"| {key} | {stat['mean']} | {stat['p95']} |")
    text += ['', '180秒限制在调用边界检查，不表示强制中断正在执行的视觉计算；未知价格不用于估算费用。', '',
             '## 4. 最终失败与恢复事件表', '', '| 案例 | 轮/问题 | 分类 | 预期任务 → 实际解析 | 预设oracle通过 |', '|---|---|---|---|---|']
    for fail in failures:
        actual = (fail.get('actual_contract') or {}).get('task_type')
        text.append(f"| {fail['case']} | {fail['round']}/{fail['turn']} | {','.join(fail['failures'])} | {fail.get('expected_task')} → {actual} | {fail['completed']} |")
    if not failures:
        text.append('| 无 | — | — | — | — |')
    text += ['', f"最终案例失败{sum(not row['completed'] for row in result['cases'])}次；格式修复尝试{summary['format_repairs']}次。格式错误恢复后可通过预设oracle，费用包含在请求计数中。",
             '失败分类按轮问题计数，可同时属于多个分类；包含已恢复的参数错误。案例未完成与错误次数分别报告。',
             '详细题目、图片指向、实际工具及证据见 `eval/formal_summary.json`；原始脱敏记录仅保存在本地。', '',
             '| 最终失败分类（可重叠） | 轮数 |', '|---|---:|']
    for category, count in m['failures'].items():
        text.append(f'| {category} | {count} |')
    text += ['', '## 5. 视觉结果与证据边界', '',
             '| 实际产生检测结果的图像 | 唯一来源 | TP | FP | FN | 召回 | 精确率 |', '|---|---:|---:|---:|---:|---:|---:|']
    for kind, v in result['vision_metrics'].items():
        text.append(f"| {kind} | {v['unique_sources']} | {v['tp']} | {v['fp']} | {v['fn']} | {percent(v['recall'])} | {percent(v['precision'])} |")
    text += ['', '一对一同类框匹配，IoU≥0.5，阈值沿用冻结模型。重复运行与缓存按来源去重；各行的图像集合不同，不能直接作增强收益比较。',
             'SODD来自视频帧，附近帧可能跨训练和测试集；本次不证明跨水域泛化、曝光恢复收益或新类别识别。', '',
             '## 6. 如何读一次记录', '',
             '先看预设的正确任务和图片，再看模型解析的任务契约；Action是工具及参数，Observation是实算结果。',
             '最后看独立评分，核对实际结果是否满足预设标签。程序任务自校验通过，仍可能因理解错题而独立失败。',
             '本轮保持冻结，未根据保留题失败修改提示；后续优化应使用开发题，重新运行保留题必须标为复测。', '',
             '审计意见由另一个同模型家族的审核Agent给出，属于暂定语义审核；程序验收状态与审核意见分开。', '']
    text += ['## 7. 发布核对与剩余问题', '',
             f"发布前核对：{integrity['bound_files_checked']}个绑定文件、{integrity['images_checked']}张图像、{integrity['turns_checked']}轮oracle评分、{integrity['cv_counts_checked']}份视觉计数、{integrity['http_attempts_checked']}次请求记录。新增云端请求0次。",
             '核对重放已保存的检测观察，重新计算质量、独立评分及GT框匹配；没有重新运行神经网络或新增真实LLM测试。', '',
             '语义缺口：hold_13要求二维码却返回pipe，hold_14要求管道却包含二维码，现有oracle仍可能通过。下一版应在开发题中补充请求类别、允许输出类别及无候选时的说明条件；本次冻结标签及成绩保持原样。',
             '已观察到的最终失败：hold_06第3轮把二维码和管道误判为不支持。已恢复事件：hold_03第2轮因JSON后追加说明触发1次格式修复。',
             '独立审计见 [EXPERIMENT_AUDIT.md](EXPERIMENT_AUDIT.md)。完整语义满足率尚未获得验收；当前通过限定于冻结的粗粒度协议。', '']
    write_json(root / 'eval' / 'formal_summary.json', summary)
    (root / 'docs/reports/FORMAL_EVALUATION_RESULTS.md').write_text('\n'.join(text), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_id')
    args = parser.parse_args()
    root = PROJECT_ROOT
    if not __import__('re').fullmatch(r'[A-Za-z0-9_-]{1,64}', args.run_id):
        parser.error('Invalid run ID')
    summary = summarize(root / 'runs' / 'agent_eval' / args.run_id, root)
    print(json.dumps({'run_id': args.run_id, 'evaluation_status': summary['metrics']['evaluation_status'],
                      'acceptance_status': summary['metrics']['acceptance_status']}, ensure_ascii=True))
