"""Goal -> visible plan -> native actions/observations -> optional replanning."""
import copy
import json
import time
from pathlib import Path

from optical_agent.adaptive_goals import GOAL_SYSTEM, goal_system, parse_goal, validate_plan
from optical_agent.adaptive_tools import AdaptiveState, SPECS, allowed_names, compact, descriptions, execute, render_result, scope_id, validate_finish
from optical_agent.diagnostics import details, redact, response_preview
from optical_agent.llm import LLMError
from optical_agent.memory import MemoryTurn, MEMORY_RULE

from optical_agent.adaptive_prompts import PLAN_SYSTEM, ACTION_SYSTEM
from optical_agent.adaptive_common import initialize_turn, complete_turn, execute_batch, finalize_turn, write_turn_log


class AdaptiveRuntime:
    def __init__(self, client, *, max_model_calls=10, max_tool_calls=12, max_seconds=180,
                 clock=time.monotonic, log_dir=None, memory_store=None):
        self.client = client
        self.max_model_calls, self.max_tool_calls, self.max_seconds = max_model_calls, max_tool_calls, max_seconds
        self.clock, self.log_dir = clock, Path(log_dir) if log_dir else None
        self.memory_store = memory_store

    def run(self, session, message):
        if not isinstance(message, str) or not message.strip() or len(message) > 4000:
            raise ValueError('请输入1至4000字的任务')
        if not session.lock.acquire(blocking=False):
            raise ValueError('会话正在分析，请等待完成')
        try:
            if not hasattr(session, 'adaptive'):
                session.adaptive = AdaptiveState(session.context)
            return self._run(session, message)
        finally:
            session.last_seen = session.clock()
            session.lock.release()

    def _run(self, session, message):
        f = initialize_turn(self, session, message)
        state = f.state
        for step in range(self.max_model_calls):
            f.phase = 'goal_understanding' if f.goal is None else 'initial_planning' if not f.initial_plan else 'tool_dispatch'
            if not complete_turn(self, f, f.phase):
                break
            try:
                if not isinstance(f.reply, dict) or f.response.get('diagnostics', {}).get('finish_reason') == 'length':
                    raise ValueError('模型输出无效或被截断')
                if f.phase != 'tool_dispatch':
                    if f.reply.get('tool_calls'):
                        raise ValueError('理解与初始规划阶段不能执行工具')
                    if f.goal is None:
                        f.goal = parse_goal(f.reply.get('content') or '', state, f.start_image, f.memory.goal_preferences if f.memory else None)
                        state.begin(f.goal)
                        f.trace.append({'type': 'goal_understanding', 'goal_contract': f.goal.as_dict()})
                        if f.goal.disposition != 'ready':
                            f.status = f.goal.disposition
                            state.pending = {'original_message': f.original_message, 'start_image': f.start_image, 'question': f.goal.question} if f.status == 'needs_clarification' else None
                            break
                        state.pending = None
                        if f.memory:
                            f.memory.retrieve(f.goal.as_dict(), f.original_message + '；' + message, json.loads(f.reply['content']).get('target_scope'))
                            f.trace.append({'type': 'memory_retrieval', 'memory_ids': [x['memory_id'] for x in f.memory.context['history_refs']], 'is_current_evidence': False})
                    else:
                        f.initial_plan = validate_plan(json.loads(f.reply.get('content') or ''), allowed_names(f.goal))
                        state.plan = copy.deepcopy(f.initial_plan)
                        f.trace.append({'type': 'initial_plan', 'steps': copy.deepcopy(f.initial_plan), 'executed': False})
                    continue
                batch = f.reply.get('tool_calls')
                if batch:
                    if not isinstance(batch, list):
                        raise ValueError('tool_calls必须为列表')
                    ids = []
                    for item in batch:
                        if not isinstance(item, dict) or item.get('type') != 'function' or (not isinstance(item.get('id'), str)) or (not item['id']) or (not isinstance(item.get('function'), dict)) or (not isinstance(item['function'].get('name'), str)) or (not isinstance(item['function'].get('arguments'), str)):
                            raise ValueError('原生工具调用结构错误')
                        ids.append(item['id'])
                    if len(set(ids)) != len(ids):
                        raise ValueError('工具调用编号重复')
                    f.history.append({'role': 'assistant', 'content': f.reply.get('content'), 'tool_calls': batch})
                    execute_batch(self, f, execute, '连续两次工具错误，已保留实际结果。')
                    if f.errors >= 2:
                        break
                    continue
                proposed = json.loads(f.reply.get('content') or '')
                f.validation = validate_finish(state, proposed)
                f.trace.append({'type': 'task_validation', **f.validation})
                if not f.validation['passed']:
                    raise ValueError('；'.join((x['requirement'] for x in f.validation['missing'])))
                f.selection, f.status = (proposed, 'completed')
                break
            except (ValueError, TypeError, KeyError) as exc:
                can_repair = f.repairs < 2 and step + 1 < self.max_model_calls and (self.clock() - f.started < self.max_seconds)
                f.error_details = dict(f.response.get('diagnostics') or details(f.phase), stage=f.phase, cause=redact(str(exc), f.secrets))
                f.trace.append({'type': 'format_error', 'phase': f.phase, 'will_retry': can_repair, 'details': f.error_details})
                if not can_repair:
                    f.status, f.error = ('task_incomplete', '结果校验或格式纠错未完成：' + redact(str(exc), f.secrets))
                    break
                f.repairs += 1
                target = f.goal_history if f.goal is None else f.plan_history if not f.initial_plan else f.history
                target.extend([{'role': 'assistant', 'content': f.reply.get('content') or json.dumps(f.reply, ensure_ascii=False)}, {'role': 'user', 'content': '程序校验失败：' + str(exc) + '。修正格式或通过工具补齐；固定目标不可改写。'}])
        result = finalize_turn(self, f)
        write_turn_log(self, session, result)
        return result
