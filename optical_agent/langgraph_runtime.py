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


@dataclass
class Turn:
    session: object
    message: str
    state: object
    started: float
    before_counts: dict = field(default_factory=dict)
    pending: dict | None = None
    start_image: str = 'original'
    original_message: str = ''
    goal: object = None
    initial_plan: list = field(default_factory=list)
    selection: dict | None = None
    validation: dict | None = None
    history: list = field(default_factory=list)
    goal_history: list = field(default_factory=list)
    plan_history: list = field(default_factory=list)
    trace: list = field(default_factory=list)
    graph_trace: list = field(default_factory=list)
    errors: int = 0
    repairs: int = 0
    calls: int = 0
    attempts: int = 0
    requests_count: int = 0
    status: str = 'budget_exceeded'
    error: str | None = None
    error_details: dict | None = None
    usage: dict = field(default_factory=lambda: dict.fromkeys(('prompt_tokens', 'completion_tokens', 'total_tokens'), 0))
    usage_available: bool = True
    secrets: tuple = ()
    memory: object = None
    response: dict = field(default_factory=dict)
    reply: object = None
    phase: str = 'goal_understanding'
    repair_cause: str = ''
    next: str = 'understand_goal'
    result: dict | None = None
    finalized: bool = False


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
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with (self.log_dir / (session.id + '.jsonl')).open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(turn.result, ensure_ascii=False) + '\n')
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
        state = f.state
        f.before_counts = dict(state.counts)
        f.pending = copy.deepcopy(state.pending)
        f.start_image = f.pending['start_image'] if f.pending else state.selected_image
        f.original_message = f.pending['original_message'] if f.pending else f.message
        state.goal, state.plan, state.revisions, state.active_candidates = None, [], [], set()
        f.secrets = (getattr(self.client, 'api_key', None),)
        f.memory = MemoryTurn(self.memory_store, f.session, 'adaptive', f.secrets) if self.memory_store is not None else None
        if f.memory:
            f.trace.append({'type': 'memory_snapshot', 'enabled': f.memory.active, 'available': f.memory.context['available'], 'is_current_evidence': False})
        f.history = copy.deepcopy(state.history[-8:]) + [{'role': 'user', 'content': f.message}]
        f.next = 'understand_goal'

    def _model(self, f, phase):
        f.phase = phase
        remaining = self.max_seconds - (self.clock() - f.started)
        if remaining <= 0 or f.calls >= self.max_tool_calls or f.requests_count >= self.max_model_calls:
            f.next = 'output'
            return False
        state, goal, memory = f.state, f.goal, f.memory
        snapshot = state.snapshot()
        snapshot.update(phase=phase, user_message=f.message, pending=f.pending, task_start_image=f.start_image,
                        remaining_model_calls=self.max_model_calls - f.requests_count,
                        remaining_tool_calls=self.max_tool_calls - f.calls)
        if memory and memory.active:
            snapshot['memory_scope_enabled'] = True
            if goal is not None:
                snapshot['MEMORY_BACKGROUND'] = memory.model_background()
        if phase == 'goal_understanding':
            messages = [{'role': 'system', 'content': goal_system(state, bool(memory and memory.active))},
                        {'role': 'user', 'content': json.dumps(snapshot, ensure_ascii=False)}] + f.goal_history
            tools, choice = None, 'auto'
        elif phase == 'initial_planning':
            snapshot['AVAILABLE_TOOLS'] = sorted(allowed_names(goal) - {'revise_plan'})
            messages = [{'role': 'system', 'content': PLAN_SYSTEM + ('\n' + MEMORY_RULE if memory and memory.active else '')},
                        {'role': 'user', 'content': json.dumps(snapshot, ensure_ascii=False)}] + f.plan_history
            tools, choice = None, 'auto'
        else:
            messages = [{'role': 'system', 'content': ACTION_SYSTEM + ('\n' + MEMORY_RULE if memory and memory.active else '') + '\nSTATE=' + json.dumps(snapshot, ensure_ascii=False)}] + f.history
            tools = [t for t in descriptions() if t['function']['name'] in allowed_names(goal)]
            relevant = [oid for oid, o in state.observations.items() if o['type'] in {'quality', 'correction', 'knowledge'} or oid.endswith(':' + scope_id(goal))]
            if goal.task_type == 'quality':
                relevant = ['quality:' + goal.image_id] if 'quality:' + goal.image_id in state.observations else []
            elif goal.task_type == 'detection' and not goal.require_reliability:
                oid = f'detections:{goal.image_id}:{scope_id(goal)}'
                relevant = [oid] if oid in state.observations else []
            possible = [goal.image_id] if goal.image_id != 'original' else ['original', *state.candidates]
            ready = any(validate_finish(state, {'selected_image_id': image, 'evidence_ids': relevant, 'citation_ids': list(state.citations)})['passed'] for image in possible)
            choice = 'auto' if ready else 'required'
        started = self.clock()
        f.requests_count += 1
        try:
            f.response = self.client.complete(messages, tools, timeout=min(30, remaining), tool_choice=choice)
        except LLMError as exc:
            f.attempts += exc.http_attempts
            f.trace.append({'type': 'model_error', 'phase': phase, 'http_trace': exc.http_trace, 'details': exc.details})
            f.status, f.error, f.error_details, f.next = exc.code, str(exc), exc.details, 'output'
            return False
        f.reply, tokens = f.response['message'], f.response.get('usage') or {}
        f.attempts += f.response.get('http_attempts', 0)
        f.trace.append({'type': 'model', 'phase': phase, 'usage': tokens, 'latency_ms': round((self.clock() - started) * 1000, 2),
                        'http_trace': f.response.get('http_trace', []), 'diagnostics': f.response.get('diagnostics')})
        good = all(type(tokens.get(k)) is int and tokens[k] >= 0 for k in f.usage)
        f.usage_available &= good
        if good:
            for k in f.usage:
                f.usage[k] += tokens[k]
        if self.clock() - f.started >= self.max_seconds:
            f.next = 'output'
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
        for item in f.reply['tool_calls']:
            if f.calls >= self.max_tool_calls or self.clock() - f.started >= self.max_seconds:
                break
            fn = item['function']
            try:
                arguments = json.loads(fn['arguments'])
            except (ValueError, TypeError):
                arguments = None
            observation = execute(f.state, fn['name'], arguments)
            f.calls += 1
            f.trace.append(copy.deepcopy(f.state.events[-1]))
            if observation['ok'] and fn['name'] == 'revise_plan':
                f.trace.append({'type': 'plan_revision', **copy.deepcopy(observation['data']), 'executed': False})
            f.errors = 0 if observation['ok'] else f.errors + 1
            f.history.append({'role': 'tool', 'tool_call_id': item['id'], 'content': json.dumps(redact(compact(observation), f.secrets), ensure_ascii=False)})
            if f.errors >= 2:
                f.status, f.error = 'tool_errors', '连续两次工具错误，保留已完成结果。'
                f.error_details = details('tool_execution', cause=observation.get('error'))
                f.next = 'output'
                return
        f.next = 'choose_action'

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
        from optical_agent.adaptive_tools import render_result
        session, state, started = f.session, f.state, f.started
        before_counts = f.before_counts or dict(state.counts)
        pending, original_message, message = f.pending, f.original_message, f.message
        goal, initial_plan, selection, validation = f.goal, f.initial_plan, f.selection, f.validation
        trace, repairs, calls, attempts, requests_count = f.trace, f.repairs, f.calls, f.attempts, f.requests_count
        status, error, error_details = f.status, f.error, f.error_details
        usage, usage_available, secrets, memory = f.usage, f.usage_available, f.secrets, f.memory
        token_keys = ('prompt_tokens', 'completion_tokens', 'total_tokens')
        if selection is None:
            selected = goal.image_id if goal and goal.image_id in state.images else 'original'
            scope = scope_id(goal) if goal else None
            ids = [oid for oid, o in state.observations.items() if o['type'] in {'quality', 'correction'}
                   or (scope and oid.endswith(':' + scope))]
            selection = {'selected_image_id': selected, 'evidence_ids': ids, 'citation_ids': []}
        if goal is None:
            result = {'answer': '未能理解任务，未生成新的分析结论。', 'evidence': [], 'citations': [],
                      'image_refs': ['original'], 'vision_status': None, 'vision_result': None, 'selected_image_id': 'original',
                      'detections': [], 'target_count': None}
        else:
            result = render_result(state, selection)
        if status in {'needs_clarification', 'unsupported'}:
            result['answer'] = goal.question
            result.update(evidence=[], detections=[], target_count=None, vision_status=None, vision_result=None, image_refs=['original'])
            validation = {'passed': None, 'missing': []}
        elif status != 'completed':
            error = error or '达到运行预算，任务未完成，已保留实际结果。'
            result['answer'] = error + '\n\n' + result['answer']
            validation = {'passed': False, 'missing': (validation or {}).get('missing', [])}
            error_details = error_details or details('runtime_budget', cause=error)
        else:
            state.selected_image = selection['selected_image_id']
            state.history.extend([{'role': 'user', 'content': original_message + ('；补充：' + message if pending else '')},
                                  {'role': 'assistant', 'content': json.dumps(selection, ensure_ascii=False)}])
            state.history = state.history[-8:]
            error_details = None
        usage['available'] = bool(usage_available and trace and self.client.provider == 'cloud')
        if not usage['available']:
            usage.update(dict.fromkeys(token_keys, None))
        result.update(agent_mode='adaptive', execution_status=status, execution_mode=self.client.provider,
            model=self.client.model_name, goal_contract=goal.as_dict() if goal else None, task_contract=goal.as_dict() if goal else None,
            task_status='completed' if status == 'completed' else status if status in {'needs_clarification', 'unsupported'} else 'incomplete',
            task_validation=validation, initial_plan=initial_plan, plan_revisions=copy.deepcopy(state.revisions),
            trace=trace, usage=usage, error=error, error_details=error_details, format_repairs=repairs,
            tool_calls=calls, model_calls=sum(e['type'] == 'model' for e in trace), model_request_attempts=requests_count,
            request_attempts=attempts, latency_ms=round((self.clock() - started) * 1000, 2), session_id=session.id,
            computation_counts={k: state.counts[k] - before_counts[k] for k in state.counts})
        result = redact(result, secrets)
        if memory:
            result = memory.finish(result)
        return result
