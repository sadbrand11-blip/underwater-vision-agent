"""Persist and reserve HTTP attempts before transport, including retries and resumes."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time


class RequestBudgetError(RuntimeError):
    pass


class RequestBudget:
    def __init__(self, path, limit=120, *, must_exist=False):
        if type(limit) is not int or not 1 <= limit <= 600:
            raise ValueError('评测 HTTP 上限必须为 1—600；默认仍为120')
        self.path, self.limit = Path(path), limit
        self.marker = self.path.with_suffix(self.path.suffix + '.initialized')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock():
            if self.path.exists():
                self._read()
            else:
                if must_exist or self.marker.exists():
                    raise RequestBudgetError('已有运行的请求计数文件丢失，停止新增请求；不能从零初始化')
                self._write({'schema_version': 1, 'limit': limit, 'used': 0, 'attempts': []})
            if not self.marker.exists():
                with self.marker.open('w', encoding='utf-8') as handle:
                    handle.write('Request ledger initialized; do not reset.\n')
                    handle.flush()
                    os.fsync(handle.fileno())

    @contextmanager
    def _lock(self):
        # OS releases the lock on crash; never delete a possibly live process's lock.
        with self.path.with_suffix(self.path.suffix + '.lock').open('a+b') as handle:
            handle.seek(0, 2)
            if handle.tell() == 0:
                handle.write(b'0')
                handle.flush()
            deadline = time.monotonic() + 10
            while True:
                try:
                    handle.seek(0)
                    if os.name == 'nt':
                        import msvcrt
                        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise RequestBudgetError('无法锁定请求计数文件，停止新增请求') from None
                    time.sleep(.02)
            try:
                yield
            finally:
                handle.seek(0)
                if os.name == 'nt':
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle, fcntl.LOCK_UN)

    def _read(self):
        try:
            data = json.loads(self.path.read_text(encoding='utf-8'))
            if (data['schema_version'] != 1 or data['limit'] != self.limit
                    or type(data['used']) is not int or not 0 <= data['used'] <= self.limit
                    or not isinstance(data['attempts'], list) or len(data['attempts']) != data['used']
                    or [a['number'] for a in data['attempts']] != list(range(1, data['used'] + 1))):
                raise ValueError('inconsistent ledger')
            return data
        except (ValueError, KeyError, TypeError, OSError):
            raise RequestBudgetError('请求计数文件损坏或上限与原记录不一致，停止新增请求；不能重置计数') from None

    def _write(self, data):
        temporary = self.path.with_suffix(self.path.suffix + '.tmp')
        with temporary.open('w', encoding='utf-8', newline='\n') as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)

    def snapshot(self):
        with self._lock():
            data = self._read()
            return {'limit': data['limit'], 'used': data['used'], 'remaining': data['limit'] - data['used']}

    def reserve(self):
        with self._lock():
            data = self._read()
            if data['used'] >= self.limit:
                raise RequestBudgetError(f'已达到本阶段 {self.limit} 次 HTTP 请求上限，未发出新请求')
            data['used'] += 1
            data['attempts'].append({'number': data['used'], 'reserved_at': datetime.now(timezone.utc).isoformat()})
            self._write(data)
            return data['used']
