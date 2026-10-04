"""In-memory sessions for the single-user local application."""

import threading
import time
import uuid

from optical_agent.tools import ToolContext


class Session:
    def __init__(self, context, clock=time.monotonic):
        self.id = uuid.uuid4().hex
        self.context = context
        self.messages = []
        self.pending_clarification = None
        self.clock = clock
        self.last_seen = clock()
        self.lock = threading.Lock()


class SessionStore:
    def __init__(self, ttl_seconds=1800, clock=time.monotonic, max_sessions=20):
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self.max_sessions = max_sessions
        self.sessions = {}
        self.lock = threading.Lock()

    def create(self, detector, image, retriever=None):
        with self.lock:
            self._purge()
            if len(self.sessions) >= self.max_sessions:
                raise ValueError('会话数量已达上限，请稍后重试或重启本地服务')
            session = Session(ToolContext(detector, image, retriever), self.clock)
            session.last_seen = self.clock()
            self.sessions[session.id] = session
            return session

    def _purge(self):
        expired = [sid for sid, s in self.sessions.items()
                   if self.clock() - s.last_seen > self.ttl_seconds and not s.lock.locked()]
        for sid in expired:
            del self.sessions[sid]

    def get(self, session_id):
        with self.lock:
            self._purge()
            session = self.sessions.get(session_id)
            if session is None:
                raise KeyError('会话不存在或已过期，请重新上传图片')
            session.last_seen = self.clock()
            return session
