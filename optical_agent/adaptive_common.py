"""Shared turn mechanics; each engine retains its own control flow."""
import copy
import json
from dataclasses import dataclass, field

from optical_agent.adaptive_goals import goal_system
from optical_agent.adaptive_tools import allowed_names, compact, descriptions, scope_id, validate_finish
from optical_agent.adaptive_prompts import PLAN_SYSTEM, ACTION_SYSTEM
from optical_agent.diagnostics import details, redact
from optical_agent.llm import LLMError
from optical_agent.memory import MemoryTurn, MEMORY_RULE

@dataclass
class AdaptiveTurn:
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

def initialize_turn(runtime, session, message, frame=None):
    f = frame if frame is not None else AdaptiveTurn(session, message, session.adaptive, runtime.clock())
    state = f.state
    f.before_counts = dict(state.counts)
    f.pending = copy.deepcopy(state.pending)
    f.start_image = f.pending['start_image'] if f.pending else state.selected_image
    f.original_message = f.pending['original_message'] if f.pending else f.message
    state.goal, state.plan, state.revisions, state.active_candidates = None, [], [], set()
    f.secrets = (getattr(runtime.client, 'api_key', None),)
    f.memory = MemoryTurn(runtime.memory_store, f.session, 'adaptive', f.secrets) if runtime.memory_store is not None else None
    if f.memory:
        f.trace.append({'type': 'memory_snapshot', 'enabled': f.memory.active, 'available': f.memory.context['available'], 'is_current_evidence': False})
    f.history = copy.deepcopy(state.history[-8:]) + [{'role': 'user', 'content': f.message}]
    f.next = 'understand_goal'
    return f

def build_request(runtime, f, phase):
    state, goal, memory = f.state, f.goal, f.memory
    snapshot = state.snapshot()
    snapshot.update(phase=phase, user_message=f.message, pending=f.pending, task_start_image=f.start_image,
                    remaining_model_calls=runtime.max_model_calls - f.requests_count,
                    remaining_tool_calls=runtime.max_tool_calls - f.calls)
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
    return messages, tools, choice

def complete_turn(runtime, f, phase):
    f.phase = phase
    remaining = runtime.max_seconds - (runtime.clock() - f.started)
    if remaining <= 0 or f.calls >= runtime.max_tool_calls or f.requests_count >= runtime.max_model_calls:
        f.next = 'output'
        return False
    messages, tools, choice = build_request(runtime, f, phase)
    started = runtime.clock()
    f.requests_count += 1
    try:
        f.response = runtime.client.complete(messages, tools, timeout=min(30, remaining), tool_choice=choice)
    except LLMError as exc:
        f.attempts += exc.http_attempts
        f.trace.append({'type': 'model_error', 'phase': phase, 'http_trace': exc.http_trace, 'details': exc.details})
        f.status, f.error, f.error_details, f.next = exc.code, str(exc), exc.details, 'output'
        return False
    f.reply, tokens = f.response['message'], f.response.get('usage') or {}
    f.attempts += f.response.get('http_attempts', 0)
    f.trace.append({'type': 'model', 'phase': phase, 'usage': tokens, 'latency_ms': round((runtime.clock() - started) * 1000, 2),
                    'http_trace': f.response.get('http_trace', []), 'diagnostics': f.response.get('diagnostics')})
    good = all(type(tokens.get(k)) is int and tokens[k] >= 0 for k in f.usage)
    f.usage_available &= good
    if good:
        for k in f.usage:
            f.usage[k] += tokens[k]
    if runtime.clock() - f.started >= runtime.max_seconds:
        f.next = 'output'
        return False
    return True

def execute_batch(runtime, f, executor, tool_error_message):
    for item in f.reply['tool_calls']:
        if f.calls >= runtime.max_tool_calls or runtime.clock() - f.started >= runtime.max_seconds:
            break
        fn = item['function']
        try:
            arguments = json.loads(fn['arguments'])
        except (ValueError, TypeError):
            arguments = None
        observation = executor(f.state, fn['name'], arguments)
        f.calls += 1
        f.trace.append(copy.deepcopy(f.state.events[-1]))
        if observation['ok'] and fn['name'] == 'revise_plan':
            f.trace.append({'type': 'plan_revision', **copy.deepcopy(observation['data']), 'executed': False})
        f.errors = 0 if observation['ok'] else f.errors + 1
        f.history.append({'role': 'tool', 'tool_call_id': item['id'], 'content': json.dumps(redact(compact(observation), f.secrets), ensure_ascii=False)})
        if f.errors >= 2:
            f.status, f.error = 'tool_errors', tool_error_message
            f.error_details = details('tool_execution', cause=observation.get('error'))
            f.next = 'output'
            return
    f.next = 'choose_action'

def finalize_turn(runtime, f):
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
    usage['available'] = bool(usage_available and trace and runtime.client.provider == 'cloud')
    if not usage['available']:
        usage.update(dict.fromkeys(token_keys, None))
    result.update(agent_mode='adaptive', execution_status=status, execution_mode=runtime.client.provider,
        model=runtime.client.model_name, goal_contract=goal.as_dict() if goal else None, task_contract=goal.as_dict() if goal else None,
        task_status='completed' if status == 'completed' else status if status in {'needs_clarification', 'unsupported'} else 'incomplete',
        task_validation=validation, initial_plan=initial_plan, plan_revisions=copy.deepcopy(state.revisions),
        trace=trace, usage=usage, error=error, error_details=error_details, format_repairs=repairs,
        tool_calls=calls, model_calls=sum(e['type'] == 'model' for e in trace), model_request_attempts=requests_count,
        request_attempts=attempts, latency_ms=round((runtime.clock() - started) * 1000, 2), session_id=session.id,
        computation_counts={k: state.counts[k] - before_counts[k] for k in state.counts})
    result = redact(result, secrets)
    if memory:
        result = memory.finish(result)
    return result

def write_turn_log(runtime, session, result):
    if runtime.log_dir:
        runtime.log_dir.mkdir(parents=True, exist_ok=True)
        with (runtime.log_dir / (session.id + '.jsonl')).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + '\n')
