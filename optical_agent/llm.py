"""A small HTTP client. Images and API keys never enter tool observations."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from optical_agent.diagnostics import details, redact, response_preview
from optical_agent.request_budget import RequestBudgetError


class LLMError(Exception):
    def __init__(self, code, message, *, diagnostics=None, http_trace=None, extra_secrets=()):
        super().__init__(redact(message, extra_secrets))
        self.code = code
        self.details = redact(diagnostics or details('model_request'), extra_secrets)
        self.http_trace = redact(http_trace or [], extra_secrets)
        self.http_attempts = len(self.http_trace)


def load_local_env(path):
    """只读取本地配置，不打印密钥，不覆盖进程中已存在的值。"""
    path = Path(path)
    if path.is_file():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            name, value = line.split('=', 1)
            if name.strip() in {'DEEPSEEK_API_KEY', 'LLM_API_KEY', 'LLM_BASE_URL', 'LLM_MODEL'}:
                os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


class CloudClient:
    provider = 'cloud'

    def __init__(self, base_url=None, model=None, api_key=None, *, request_budget=None):
        self.base_url = (base_url or os.getenv('LLM_BASE_URL', 'https://api.deepseek.com')).rstrip('/')
        self.model_name = model or os.getenv('LLM_MODEL', 'deepseek-flash')
        self.api_key = api_key if api_key is not None else (os.getenv('LLM_API_KEY') or os.getenv('DEEPSEEK_API_KEY', ''))
        self.request_budget = request_budget

    @property
    def configured(self):
        return bool(self.api_key)

    def complete(self, messages, tools=None, timeout=30, *, tool_choice='auto'):
        if tool_choice not in {'auto', 'required'}:
            raise ValueError('tool_choice must be auto or required')
        secrets = (self.api_key,)
        http_trace = []
        if not self.configured:
            raise LLMError('missing_api_key', '未配置云端密钥。可先使用离线工具演示，再配置本地 .env。',
                           diagnostics=details('configuration', suggestion='检查项目目录中的 .env 文件并保存密钥。'))
        payload = {'model': self.model_name, 'messages': redact(messages, secrets),
                   'stream': False, 'max_tokens': 2048,
                   'response_format': {'type': 'json_object'}}
        if tools is not None:
            payload.update(tools=tools, tool_choice=tool_choice)
        if urlparse(self.base_url).hostname == 'api.deepseek.com':
            payload['thinking'] = {'type': 'disabled'}
        deadline = time.monotonic() + timeout
        for attempt in range(2):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LLMError('network_error', '模型请求超时，已保留完成的分析。',
                    diagnostics=details('connection', cause='timeout', suggestion='稍后重试，检查网络连接。'),
                    http_trace=http_trace, extra_secrets=secrets)
            attempt_start = time.monotonic()
            if self.request_budget is not None:
                try:
                    self.request_budget.reserve()
                except RequestBudgetError as exc:
                    raise LLMError('request_budget_exceeded', str(exc),
                        diagnostics=details('evaluation_budget', suggestion='查看已保存的部分结果与持久化请求计数。'),
                        http_trace=http_trace, extra_secrets=secrets) from None
            try:
                response = requests.post(self.base_url + '/chat/completions', json=payload,
                    headers={'Authorization': 'Bearer ' + self.api_key}, timeout=remaining)
            except requests.RequestException as exc:
                cause = type(exc).__name__
                hint = ('检查代理程序是否运行及代理地址是否有效。' if isinstance(exc, requests.exceptions.ProxyError)
                        else '检查本机证书配置，不要关闭证书验证。' if isinstance(exc, requests.exceptions.SSLError)
                        else '检查网络连接，稍后重试。')
                diagnostic = details('connection', cause=cause,
                    preview=response_preview(str(exc), secrets), suggestion=hint, extra_secrets=secrets)
                http_trace.append({'attempt': attempt + 1, 'http_status': None, 'cause': cause,
                    'latency_ms': round((time.monotonic() - attempt_start) * 1000, 2)})
                if attempt == 0 and deadline - time.monotonic() > 1:
                    time.sleep(0.5)
                    continue
                raise LLMError('network_error', '云端连接失败，未收到 HTTP 响应。已保留完成的分析。',
                    diagnostics=diagnostic, http_trace=http_trace, extra_secrets=secrets) from None
            http_trace.append({'attempt': attempt + 1, 'http_status': response.status_code,
                'latency_ms': round((time.monotonic() - attempt_start) * 1000, 2)})
            request_id = response.headers.get('x-request-id')
            request_id = response_preview(request_id, secrets, limit=160) if isinstance(request_id, str) else None
            if response.status_code in (401, 403):
                raise LLMError('authentication_error', f'模型服务拒绝访问（HTTP {response.status_code}），请检查本地密钥和账户状态。',
                    diagnostics=details('authentication', http_status=response.status_code, request_id=request_id,
                        suggestion='检查密钥是否有效、是否被撤销，以及账户的访问权限。'),
                    http_trace=http_trace, extra_secrets=secrets)
            if response.status_code == 402:
                raise LLMError('insufficient_balance', '账户余额不足（HTTP 402），请在 DeepSeek 开放平台检查可用余额。',
                    diagnostics=details('account_balance', http_status=402, request_id=request_id,
                        suggestion='查看开放平台的可用余额。'), http_trace=http_trace, extra_secrets=secrets)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == 0 and deadline - time.monotonic() > 1:
                    time.sleep(0.5)
                    continue
                raise LLMError('network_error', f'模型服务暂不可用（HTTP {response.status_code}）。',
                    diagnostics=details('api_request', http_status=response.status_code, request_id=request_id,
                        suggestion='请求受到限制或服务繁忙，稍后重试。'), http_trace=http_trace, extra_secrets=secrets)
            if response.status_code != 200:
                raise LLMError('invalid_response', f'模型请求未成功（HTTP {response.status_code}），请检查服务地址、模型名称和请求参数。',
                    diagnostics=details('api_request', http_status=response.status_code, request_id=request_id,
                        suggestion='检查服务地址、模型名称和请求参数。'), http_trace=http_trace, extra_secrets=secrets)
            try:
                data = response.json()
                choice = data['choices'][0]
                message = choice['message']
                if not isinstance(message, dict):
                    raise ValueError('Invalid message')
                if request_id is None and isinstance(data.get('id'), str):
                    request_id = response_preview(data['id'], secrets, limit=160)
                diagnostic = details('model_response', http_status=200,
                    finish_reason=choice.get('finish_reason'), request_id=request_id,
                    preview=response_preview(message, secrets), extra_secrets=secrets)
                diagnostic['served_model'] = response_preview(data['model'], secrets, limit=160) if isinstance(data.get('model'), str) else None
                return {'message': message, 'usage': data.get('usage', {}),
                    'diagnostics': diagnostic,
                    'http_attempts': len(http_trace), 'http_trace': http_trace}
            except (ValueError, KeyError, IndexError, TypeError):
                raise LLMError('invalid_response', '模型服务返回了无法解析的结果（HTTP 200）。',
                    diagnostics=details('response_decode', http_status=200, request_id=request_id,
                        suggestion='服务已响应，但响应结构无效；查看请求编号与服务配置。'),
                    http_trace=http_trace, extra_secrets=secrets) from None


class ScriptedClient:
    """明确标注的规则演示；不代表 LLM 的工具选择能力。"""

    provider = 'scripted'
    model_name = 'offline_rule_demo_not_an_llm'

    def complete(self, messages, tools=None, timeout=30, *, tool_choice='auto'):
        from optical_agent.tasks import scripted_intent
        if tools is None:
            data = json.loads(next(m['content'] for m in messages if m['role'] == 'user'))
            # 明确为规则夹具；澄清答复不足以独立理解时才合并原问题。
            text = data['message']
            parsed = scripted_intent(text)
            if parsed['task_type'] == 'clarify' and data.get('pending_clarification'):
                explicit_ref = parsed['image_reference']
                parsed = scripted_intent(data['pending_clarification']['original_message'] + '；补充：' + text)
                if explicit_ref != 'unspecified':
                    parsed['image_reference'] = explicit_ref
            return {'message': {'role': 'assistant', 'content': json.dumps(parsed, ensure_ascii=False)}, 'usage': {}, 'http_attempts': 0}
        state = json.loads(messages[0]['content'].split('SESSION_STATE=')[1].split('\n')[0])
        contract = json.loads(messages[0]['content'].split('TASK_CONTRACT=')[1].split('\n')[0])
        available_tools = {t['function']['name'] for t in tools}
        completed = []
        for m in messages:
            if m.get('role') == 'tool':
                observation = json.loads(m['content'])
                if observation.get('ok'):
                    completed.append(observation['tool'])
        image_id = contract['image_id']
        why = contract['task_type'] == 'explanation'
        compare = contract['task_type'] in {'comparison', 'explanation'}
        detection = contract['task_type'] == 'detection'
        if compare:
            if 'analyze_image_full_pipeline' in available_tools:
                actions = [('analyze_image_full_pipeline', {'image_id': 'original'})]
            else:
                actions = [('assess_image_quality', {'image_id': 'original'}),
                    ('correct_image_exposure', {'image_id': 'original'}),
                    ('assess_image_quality', {'image_id': 'corrected'}),
                    ('detect_objects', {'image_id': 'original'}), ('detect_objects', {'image_id': 'corrected'}),
                    ('compare_detection_evidence', {'original_image_id': 'original', 'corrected_image_id': 'corrected'})]
            if why:
                actions.append(('retrieve_knowledge', {'query': '曝光校正为什么失败，信息丢失与拒识原因'}))
            evidence_ids = ['comparison:original']
        elif detection:
            actions = [('detect_objects', {'image_id': image_id})] if 'detect_objects' in available_tools else [('analyze_image_full_pipeline', {'image_id': 'original'})]
            evidence_ids = [f'detections:{image_id}']
        elif contract['task_type'] == 'correction':
            actions = [('assess_image_quality', {'image_id': 'original'}),
                       ('correct_image_exposure', {'image_id': 'original'})] if 'correct_image_exposure' in available_tools else [('analyze_image_full_pipeline', {'image_id': 'original'})]
            evidence_ids = ['correction:original']
        else:
            actions = [('assess_image_quality', {'image_id': image_id})] if 'assess_image_quality' in available_tools else [('analyze_image_full_pipeline', {'image_id': 'original'})]
            evidence_ids = [f'quality:{image_id}']
        if len(completed) < len(actions):
            name, arguments = actions[len(completed)]
            message = {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': f'demo_{len(completed)}', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(arguments)}}]}
        else:
            citations = []
            for m in messages:
                if m.get('role') == 'tool':
                    citations.extend(x['citation_id'] for x in json.loads(m['content']).get('data', {}).get('matches', []))
            message = {'role': 'assistant', 'content': json.dumps({'evidence_ids': evidence_ids, 'citation_ids': citations})}
        return {'message': message, 'usage': {}, 'http_attempts': 0}
