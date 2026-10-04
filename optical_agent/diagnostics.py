"""Small, secret-safe diagnostics for the local UI and execution logs."""

from __future__ import annotations

import json
import os
import re


def redact(value, extra_secrets=()):
    secrets = {secret for secret in (*extra_secrets, os.getenv('DEEPSEEK_API_KEY'),
                                   os.getenv('LLM_API_KEY')) if isinstance(secret, str) and secret}

    def clean(item):
        if isinstance(item, str):
            for secret in sorted(secrets, key=len, reverse=True):
                item = item.replace(secret, '[REDACTED]')
            item = re.sub(r'(?i)Bearer\s+[^\s"\'<>]+', 'Bearer [REDACTED]', item)
            item = re.sub(r'sk-[A-Za-z0-9_-]{10,}', '[REDACTED]', item)
            item = re.sub(r'(https?://)[^/\s@]+@', r'\1[REDACTED]@', item)
            return re.sub(r'(?i)(api[_-]?key|access_token|password|secret)=([^&\s]+)',
                          r'\1=[REDACTED]', item)
        if isinstance(item, dict):
            sensitive = {'authorization', 'api_key', 'apikey', 'password', 'access_token'}
            return {clean(key): '[REDACTED]' if str(key).lower() in sensitive else clean(val)
                    for key, val in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(val) for val in item]
        return item

    return clean(value)


def response_preview(value, extra_secrets=(), limit=1024):
    # Redact before truncating: a cutoff must not leave the start of a key visible.
    safe = redact(value, extra_secrets)
    text = safe if isinstance(safe, str) else json.dumps(safe, ensure_ascii=False)
    return text[:limit] + (' …[已截断]' if len(text) > limit else '')


def details(stage, *, http_status=None, cause=None, finish_reason=None,
            request_id=None, preview=None, suggestion=None, extra_secrets=()):
    return redact({'stage': stage,
                   'http_status': http_status if type(http_status) is int else None,
                   'cause': cause, 'finish_reason': finish_reason, 'request_id': request_id,
                   'response_preview': preview, 'suggestion': suggestion}, extra_secrets)
