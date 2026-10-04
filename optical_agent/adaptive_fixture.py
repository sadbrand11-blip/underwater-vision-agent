"""Explicit offline rule fixture, not evidence of LLM planning ability."""
import json
import re

from optical_agent.adaptive_goals import ALIASES, CLASSES


def scripted_goal(message, pending=None):
    if pending:
        message = pending['original_message'] + '；' + message
    targets = []
    inventory = {**ALIASES, **{name:(name,) for name in (*CLASSES,'underwater_robot')},
                 '航行器':('underwater_robot',),'红鳍片':('red_fin',),'网具':('net',)}
    for alias, names in inventory.items():
        found = (re.search(r'(?<![a-z0-9_])'+re.escape(alias.lower())+r'(?![a-z0-9_])',message.lower())
                 if alias.isascii() else alias in message)
        if found:
            targets.extend(names)
    targets = list(dict.fromkeys(targets))
    ref = 'corrected' if any(x in message for x in ('校正后的', '校正图')) else 'current' if '刚才' in message else 'original'
    kind, policy, reliable, question = 'analysis', 'if_needed', True, ''
    if any(x in message for x in ('鱼', '珊瑚')):
        kind = 'unsupported'
    elif '只检查' in message or '只评估' in message:
        kind, policy, reliable = 'quality', 'never', False
    elif '只校正' in message:
        kind, policy, reliable = 'correction', 'always', False
    elif '比较' in message:
        kind, policy = 'comparison', 'always'
    elif '解释' in message or '依据' in message:
        kind = 'explanation'
    elif ('识别' in message or '检测' in message) and '校正' not in message and '可靠' not in message and '曝光' not in message:
        kind, policy, reliable = 'detection', 'never', False
    elif ref == 'corrected' and '识别' in message and '先校正' not in message:
        kind, policy, reliable = 'detection', 'never', False
    if '先校正' in message and kind == 'analysis':
        policy, ref = 'always', 'original'
    return {'task_type': kind, 'image_reference': ref, 'target_classes': targets,
            'correction_policy': policy, 'require_reliability': reliable, 'question': question}


class AdaptiveScriptedClient:
    provider = 'scripted'
    model_name = 'offline_adaptive_rule_fixture'

    def complete(self, messages, tools=None, timeout=30, *, tool_choice='auto'):
        system = messages[0]['content']
        state = json.loads(system.split('\nSTATE=', 1)[1]) if '\nSTATE=' in system else json.loads(messages[1]['content'])
        phase = state['phase']
        if phase == 'goal_understanding':
            content = scripted_goal(state['user_message'], state['pending'])
            if state.get('memory_scope_enabled'):
                full = (state['pending']['original_message'] + '；' if state.get('pending') else '') + state['user_message']
                all_classes = any(x in full for x in ('全部类别', '所有类别', '全部目标', '所有目标'))
                content['target_scope'] = 'all' if all_classes else 'explicit' if content['target_classes'] else 'default'
                if all_classes:
                    content['target_classes'] = []
            return self.reply(content)
        if phase == 'initial_planning':
            names = state['AVAILABLE_TOOLS']
            return self.reply({'steps': [{'tool': names[0], 'purpose': '离线规则演示拟执行动作'}]})
        goal, observations = state['goal'], state['observations']
        ids = list(observations)
        image_id = goal['image_id']
        scope = __import__('hashlib').sha256(json.dumps(tuple(goal['target_classes'])).encode()).hexdigest()[:12]
        kind = goal['task_type']
        if kind == 'quality':
            oid = 'quality:' + image_id
            return self.finish(image_id, [oid]) if oid in ids else self.action('assess_image_quality', image_id=image_id)
        if kind == 'detection' and not goal['require_reliability']:
            oid = f'detections:{image_id}:{scope}'
            return self.finish(image_id, [oid]) if oid in ids else self.action('detect_objects', image_id=image_id)
        if 'quality:original' not in ids:
            return self.action('assess_image_quality', image_id='original')
        q = observations['quality:original']['data']['quality']
        if kind == 'correction':
            if q['quality_pass'] is not True or q['exposure_state'] == 'normal':
                return self.finish('original', ['quality:original'])
            if 'correction:candidate_gamma' not in ids:
                return self.action('generate_exposure_candidate', image_id='original', method='gamma_only')
            return self.finish('candidate_gamma', ['quality:original', 'correction:candidate_gamma'])
        if q['quality_pass'] is not True:
            rid = 'reliability:original:' + scope
            if rid not in ids:
                return self.action('assess_reliability', image_id='original')
            return self.with_knowledge(goal, observations, ['quality:original', rid], 'original')
        needs = goal['correction_policy'] == 'always' or (goal['correction_policy'] == 'if_needed' and q['exposure_state'] != 'normal')
        if needs:
            candidate = 'candidate_gamma'
            if 'correction:' + candidate not in ids:
                return self.action('generate_exposure_candidate', image_id='original', method='gamma_only')
            for image in ('original', candidate):
                if 'quality:' + image not in ids:
                    return self.action('assess_image_quality', image_id=image)
                if f'detections:{image}:{scope}' not in ids:
                    return self.action('detect_objects', image_id=image)
            cmp = 'comparison:' + candidate + ':' + scope
            if cmp not in ids:
                return self.action('compare_candidates', candidate_image_ids=[candidate])
            report = observations[cmp]['data']['reports'][0]
            selected = report['selected_image']
            if report['effect_degraded'] and not any(cmp in r['observation_ids'] for r in state['revisions']):
                return self.action('revise_plan', reason='离线规则示例：候选证据下降，保留原图', observation_ids=[cmp],
                                   steps=[{'tool':'compare_candidates','purpose':'按实际比较结论采用原图'}])
            return self.with_knowledge(goal, observations, ['quality:original', f'detections:{selected}:{scope}', cmp], selected)
        det, rid = 'detections:original:' + scope, 'reliability:original:' + scope
        if det not in ids:
            return self.action('detect_objects', image_id='original')
        if rid not in ids:
            return self.action('assess_reliability', image_id='original')
        return self.with_knowledge(goal, observations, ['quality:original', det, rid], 'original')

    def with_knowledge(self, goal, observations, ids, selected):
        matches = [m for o in observations.values() if o['type'] == 'knowledge' for m in o['data']['matches']]
        if goal['requires_citations'] and not matches:
            return self.action('retrieve_knowledge', query='曝光信息丢失与目标拒识原因')
        return self.finish(selected, ids, [m['citation_id'] for m in matches] if goal['requires_citations'] else [])

    @staticmethod
    def reply(content):
        return {'message': {'content': json.dumps(content, ensure_ascii=False)}, 'usage': {}, 'http_attempts': 0}

    @staticmethod
    def action(name, **args):
        return {'message': {'content': None, 'tool_calls': [{'id': 'fixture_action', 'type': 'function',
                'function': {'name': name, 'arguments': json.dumps(args, ensure_ascii=False)}}]}, 'usage': {}, 'http_attempts': 0}

    @classmethod
    def finish(cls, image, ids, citations=None):
        return cls.reply({'selected_image_id': image, 'evidence_ids': ids, 'citation_ids': citations or []})
