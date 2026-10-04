"""Deterministic comparison of actual runtime results; timing is reported separately."""
import copy
import json


def without_timings(value, parse_json=False):
    if isinstance(value, dict):
        return {k: without_timings(v, parse_json) for k, v in value.items()
                if k not in {'latency_ms', 'elapsed_ms', 'processing_ms'}}
    if isinstance(value, (list, tuple)):
        return [without_timings(v, parse_json) for v in value]
    if parse_json and isinstance(value, str):
        prefix, body = value.split('\nSTATE=', 1) if '\nSTATE=' in value else ('', value)
        try:
            parsed = json.loads(body)
        except (ValueError, TypeError):
            return value
        normalized = without_timings(parsed, parse_json)
        return [prefix, normalized] if prefix else normalized
    return value

def core(result):
    keys = ('goal_contract', 'task_contract', 'selected_image_id', 'evidence', 'citations', 'image_refs',
            'detections', 'target_count', 'vision_status', 'vision_result', 'task_status', 'task_validation',
            'initial_plan', 'plan_revisions', 'execution_status', 'format_repairs', 'tool_calls', 'model_calls',
            'model_request_attempts', 'request_attempts', 'computation_counts', 'usage')
    values = {k: copy.deepcopy(result[k]) for k in keys}
    # Measurements and tool timings are different concepts; only timings vary.
    values = without_timings(values)
    values['actions'] = [(t['tool'], t['arguments'], t['ok'], t.get('cached', False), t.get('executed'))
                         for t in result['trace'] if t['type'] == 'tool']
    return values
