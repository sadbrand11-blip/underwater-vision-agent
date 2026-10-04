"""Optional, serial LangGraph counterpart to the unchanged native runtime.

Control flow runs through real StateGraph nodes. Images, credentials and mutable
vision caches stay in the local Turn context, never in exported graph snapshots.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import TypedDict

from optical_agent.adaptive_runtime import AdaptiveRuntime, PLAN_SYSTEM, ACTION_SYSTEM
from optical_agent.adaptive_goals import goal_system, parse_goal, validate_plan
from optical_agent.adaptive_tools import allowed_names, compact, descriptions, execute, scope_id, validate_finish
from optical_agent.diagnostics import details, redact
from optical_agent.llm import LLMError
from optical_agent.memory import MemoryTurn, MEMORY_RULE


class GraphState(TypedDict, total=False):
    next: str
    phase: str
    model_requests: int
    tool_calls: int
    repairs: int
    goal_contract: dict | None
    evidence_ids: list[str]


from optical_agent.adaptive_common import AdaptiveTurn as Turn, initialize_turn, complete_turn, execute_batch, finalize_turn, write_turn_log


class LangGraphRuntime(AdaptiveRuntime):
    """Same constructor/run interface; inherited run owns the session lock."""
    def __init__(self, client, *, max_graph_steps=100, **kwargs):
        super().__init__(client, **kwargs)
        self.max_graph_steps = max_graph_steps

    def _run(self, session, message):
        # Lazy imports keep native usable without any graph dependencies.
        from langgraph.graph import StateGraph, START, END
        from langgraph.errors import GraphRecursionError
        from langsmith import tracing_context

        turn = Turn(session, message, session.adaptive, self.clock())
        builder = StateGraph(GraphState, context_schema=Turn)
        nodes = {'initialize': self._initialize, 'understand_goal': self._understand,
                 'plan': self._plan, 'choose_action': self._choose,
                 'execute_tools': self._tools, 'validate': self._validate,
                 'repair': self._repair, 'output': self._output}
        for name, method in nodes.items():
            def node(graph_state, runtime, name=name, method=method):
                frame = runtime.context
                started = self.clock()
                entry = {'node': name, 'status': 'running'}
                frame.graph_trace.append(entry)
                try:
                    method(frame)
                    entry.update(status='completed', next=frame.next,
                                 reason=self._reason(frame, name))
                except (ValueError, TypeError, KeyError) as exc:
                    if name in {'understand_goal', 'plan', 'choose_action', 'validate'}:
                        self._format_error(frame, exc)
                        entry.update(status='validation_error', next=frame.next, reason=redact(str(exc), frame.secrets))
                    else:
                        entry.update(status='error', reason=type(exc).__name__)
                        raise
                finally:
                    entry['latency_ms'] = round((self.clock() - started) * 1000, 2)
                return {'next': frame.next, 'phase': frame.phase,
                        'model_requests': frame.requests_count, 'tool_calls': frame.calls,
                        'repairs': frame.repairs, 'goal_contract': frame.goal.as_dict() if frame.goal else None,
                        'evidence_ids': list(frame.state.observations)}
            builder.add_node(name, node)
        builder.add_edge(START, 'initialize')
        routes = {'initialize': ['understand_goal'], 'understand_goal': ['plan', 'repair', 'output'],
                  'plan': ['choose_action', 'repair', 'output'],
                  'choose_action': ['execute_tools', 'validate', 'repair', 'output'],
                  'execute_tools': ['choose_action', 'output'], 'validate': ['repair', 'output'],
                  'repair': ['understand_goal', 'plan', 'choose_action']}
        for name in nodes:
            if name == 'output':
                builder.add_edge(name, END)
            else:
                builder.add_conditional_edges(name, lambda s: s['next'], {target: target for target in routes[name]})
        graph = builder.compile()
        # Also overrides ambient tracing settings: no LangSmith requests.
        with tracing_context(enabled=False):
            try:
                graph.invoke({}, config={'recursion_limit': self.max_graph_steps, 'callbacks': []}, context=turn)
            except Exception as exc:
                turn.status = 'graph_step_limit' if isinstance(exc, GraphRecursionError) else 'graph_error'
                turn.error = '图步骤达到上限，保留已完成结果。' if isinstance(exc, GraphRecursionError) else '图执行异常，保留已完成结果。'
                turn.error_details = details('graph_execution', cause=type(exc).__name__)
                turn.graph_trace.append({'node': 'exception_handler', 'status': 'error', 'next': 'output',
                                         'reason': type(exc).__name__, 'latency_ms': 0})
                # Finalization is idempotent; memory is never written twice.
                if turn.finalized:
                    turn.memory = None
                    turn.result = self._finalize(turn)
                else:
                    self._output(turn)
        turn.result['runtime_engine'] = 'langgraph'
        turn.result['graph_trace'] = redact(turn.graph_trace, turn.secrets)
        write_turn_log(self, session, turn.result)
        return turn.result

    def _reason(self, f, name):
        if name == 'validate':
            return '实际证据满足完成条件' if f.selection else '完成条件仍有缺项'
        if name == 'repair':
            return '反馈具体错误，返回原阶段'
        if name == 'execute_tools':
            return '顺序执行工具并返回观察' if f.next != 'output' else f.status
        if f.next == 'output':
            return f.status
        return {'initialize': '读取本轮状态与记忆快照', 'understand_goal': '目标契约已固定',
                'plan': '初始计划已校验', 'choose_action': '模型选择工具' if f.next == 'execute_tools' else '模型提交最终证据'}.get(name, '结果已生成')

    def _initialize(self, f):
        initialize_turn(self, f.session, f.message, frame=f)

    def _model(self, f, phase):
        if not complete_turn(self, f, phase):
            return False
        if not isinstance(f.reply, dict) or f.response.get('diagnostics', {}).get('finish_reason') == 'length':
            raise ValueError('模型回复缺失或被截断')
        if phase != 'tool_dispatch' and f.reply.get('tool_calls'):
            raise ValueError('此阶段不能执行工具')
        return True

    def _understand(self, f):
        if not self._model(f, 'goal_understanding'):
            return
        f.goal = parse_goal(f.reply.get('content') or '', f.state, f.start_image, f.memory.goal_preferences if f.memory else None)
        f.state.begin(f.goal)
        f.trace.append({'type': 'goal_understanding', 'goal_contract': f.goal.as_dict()})
        if f.goal.disposition != 'ready':
            f.status = f.goal.disposition
            f.state.pending = {'original_message': f.original_message, 'start_image': f.start_image, 'question': f.goal.question} if f.status == 'needs_clarification' else None
            f.next = 'output'
            return
        f.state.pending = None
        if f.memory:
            f.memory.retrieve(f.goal.as_dict(), f.original_message + ';' + f.message, json.loads(f.reply['content']).get('target_scope'))
            f.trace.append({'type': 'memory_retrieval', 'memory_ids': [x['memory_id'] for x in f.memory.context['history_refs']], 'is_current_evidence': False})
        f.next = 'plan'

    def _plan(self, f):
        if not self._model(f, 'initial_planning'):
            return
        f.initial_plan = validate_plan(json.loads(f.reply.get('content') or ''), allowed_names(f.goal))
        f.state.plan = copy.deepcopy(f.initial_plan)
        f.trace.append({'type': 'initial_plan', 'steps': copy.deepcopy(f.initial_plan), 'executed': False})
        f.next = 'choose_action'

    def _choose(self, f):
        if not self._model(f, 'tool_dispatch'):
            return
        batch = f.reply.get('tool_calls')
        if batch:
            if not isinstance(batch, list):
                raise ValueError('tool_calls必须为列表')
            ids = []
            for item in batch:
                if (not isinstance(item, dict) or item.get('type') != 'function' or not isinstance(item.get('id'), str)
                    or not item['id'] or not isinstance(item.get('function'), dict)
                    or not isinstance(item['function'].get('name'), str) or not isinstance(item['function'].get('arguments'), str)):
                    raise ValueError('工具调用结构错误')
                ids.append(item['id'])
            if len(set(ids)) != len(ids):
                raise ValueError('工具调用编号重复')
            f.history.append({'role': 'assistant', 'content': f.reply.get('content'), 'tool_calls': batch})
            f.next = 'execute_tools'
        else:
            f.next = 'validate'

    def _tools(self, f):
        execute_batch(self, f, execute, '连续两次工具错误，保留已完成结果。')

    def _validate(self, f):
        proposed = json.loads(f.reply.get('content') or '')
        f.validation = validate_finish(f.state, proposed)
        f.trace.append({'type': 'task_validation', **f.validation})
        if not f.validation['passed']:
            raise ValueError(';'.join(x['requirement'] for x in f.validation['missing']))
        f.selection, f.status, f.next = proposed, 'completed', 'output'

    def _format_error(self, f, exc):
        can_repair = f.repairs < 2 and f.requests_count < self.max_model_calls and self.clock() - f.started < self.max_seconds
        f.error_details = dict(f.response.get('diagnostics') or details(f.phase), stage=f.phase, cause=redact(str(exc), f.secrets))
        f.trace.append({'type': 'format_error', 'phase': f.phase, 'will_retry': can_repair, 'details': f.error_details})
        f.repair_cause = str(exc)
        if not can_repair:
            f.status, f.error, f.next = 'task_incomplete', '任务结果未通过校验：' + redact(str(exc), f.secrets), 'output'
        else:
            f.next = 'repair'

    def _repair(self, f):
        f.repairs += 1
        target = f.goal_history if f.goal is None else f.plan_history if not f.initial_plan else f.history
        target.extend([{'role': 'assistant', 'content': f.reply.get('content') if isinstance(f.reply, dict) and f.reply.get('content') else json.dumps(f.reply, ensure_ascii=False)},
                       {'role': 'user', 'content': '上次结果错误：' + f.repair_cause + '。请修正并满足固定条件；不要虚构证据。'}])
        f.next = 'understand_goal' if f.goal is None else 'plan' if not f.initial_plan else 'choose_action'

    def _output(self, f):
        if f.finalized:
            return
        f.finalized = True
        f.result = self._finalize(f)
        f.next = 'end'

    def _finalize(self, f):
        return finalize_turn(self, f)
