import concurrent.futures
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests

from optical_agent.llm import CloudClient, LLMError
from optical_agent.request_budget import RequestBudget, RequestBudgetError


class RequestBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'http_budget.json'

    @patch('optical_agent.llm.requests.post')
    def test_121st_http_request_never_sent_even_after_resume(self, post):
        post.return_value = Mock(status_code=200, headers={}, json=Mock(return_value={
            'choices': [{'message': {'content': '{}'}}]}))
        budget = RequestBudget(self.path, 120)
        client = CloudClient(api_key='fixture-secret', request_budget=budget)
        for _ in range(120):
            client.complete([], None)
        resumed = CloudClient(api_key='fixture-secret', request_budget=RequestBudget(self.path, 120))
        with self.assertRaises(LLMError) as caught:
            resumed.complete([], None)
        self.assertEqual(post.call_count, 120)
        self.assertEqual(caught.exception.code, 'request_budget_exceeded')
        self.assertEqual(caught.exception.http_attempts, 0)
        self.assertEqual(budget.snapshot(), {'limit': 120, 'used': 120, 'remaining': 0})

    @patch('optical_agent.llm.time.sleep')
    @patch('optical_agent.llm.requests.post')
    def test_retry_counts_before_send_and_cannot_bypass_limit(self, post, sleep):
        budget = RequestBudget(self.path, 1)
        def network_failure(*args, **kwargs):
            self.assertEqual(budget.snapshot()['used'], 1)
            raise requests.ConnectionError('offline fixture')
        post.side_effect = network_failure
        with self.assertRaises(LLMError) as caught:
            CloudClient(api_key='fixture-secret', request_budget=budget).complete([], None)
        self.assertEqual(post.call_count, 1)
        self.assertEqual(caught.exception.code, 'request_budget_exceeded')
        self.assertEqual(caught.exception.http_attempts, 1)
        self.assertIsNone(caught.exception.http_trace[0]['http_status'])

    @patch('optical_agent.llm.time.sleep')
    @patch('optical_agent.llm.requests.post')
    def test_service_retry_consumes_two_reservations(self, post, sleep):
        budget = RequestBudget(self.path, 2)
        post.side_effect = [Mock(status_code=503, headers={}), Mock(status_code=200, headers={},
            json=Mock(return_value={'choices': [{'message': {'content': '{}'}}]}))]
        response = CloudClient(api_key='fixture-secret', request_budget=budget).complete([], None)
        self.assertEqual(response['http_attempts'], 2)
        self.assertEqual(budget.snapshot()['used'], 2)

    def test_corrupt_or_changed_cap_fails_closed(self):
        budget = RequestBudget(self.path, 10)
        budget.reserve()
        with self.assertRaises(RequestBudgetError):
            RequestBudget(self.path, 120)
        self.path.write_text('{broken', encoding='utf-8')
        with self.assertRaises(RequestBudgetError):
            budget.reserve()

    @patch('optical_agent.llm.requests.post')
    def test_missing_ledger_cannot_reset_existing_run(self, post):
        budget = RequestBudget(self.path, 120)
        budget.reserve()
        self.path.unlink()
        with self.assertRaises(RequestBudgetError):
            RequestBudget(self.path, 120)
        budget.marker.unlink()
        with self.assertRaises(RequestBudgetError):
            RequestBudget(self.path, 120, must_exist=True)
        self.assertEqual(post.call_count, 0)

    def test_concurrent_reservations_never_exceed_cap(self):
        budget = RequestBudget(self.path, 10)
        def reserve(_):
            try:
                return budget.reserve()
            except RequestBudgetError:
                return None
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(reserve, range(20)))
        self.assertEqual(sorted(n for n in results if n is not None), list(range(1, 11)))
        self.assertEqual(budget.snapshot()['used'], 10)
        self.assertNotIn('secret', self.path.read_text(encoding='utf-8'))
