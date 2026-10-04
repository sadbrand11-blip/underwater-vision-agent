import unittest
from unittest.mock import Mock, patch
import requests

from optical_agent.llm import CloudClient, LLMError


class CloudWireTests(unittest.TestCase):
    @patch('optical_agent.llm.requests.post')
    def test_intent_request_is_json_without_tool_fields(self, post):
        post.return_value = Mock(status_code=200, headers={}, json=Mock(return_value={
            'choices': [{'message': {'content': '{}'}}]}))
        CloudClient(api_key='fixture-key').complete([{'role': 'user', 'content': 'test'}], None)
        payload = post.call_args.kwargs['json']
        self.assertNotIn('tools', payload)
        self.assertNotIn('tool_choice', payload)
        self.assertEqual(payload['response_format'], {'type': 'json_object'})
        self.assertEqual(payload['thinking'], {'type': 'disabled'})

    @patch('optical_agent.llm.requests.post')
    def test_wire_format_and_retry(self,post):
        failed=Mock(status_code=503)
        success=Mock(status_code=200)
        success.json.return_value={'choices':[{'message':{'role':'assistant','content':'{}'}}],
                                   'usage':{'total_tokens':12}}
        post.side_effect=[failed,success]
        result=CloudClient(api_key='fixture-key').complete([{'role':'user','content':'test'}],[])
        self.assertEqual(post.call_count,2)
        payload=post.call_args.kwargs['json']
        self.assertEqual(payload['thinking'],{'type':'disabled'})
        self.assertFalse(payload['stream'])
        self.assertEqual(result['usage']['total_tokens'],12)
        self.assertEqual(result['http_attempts'], 2)
        self.assertEqual([a['http_status'] for a in result['http_trace']], [503, 200])

    @patch('optical_agent.llm.requests.post')
    def test_required_tool_choice_preserves_json_and_non_thinking(self, post):
        post.return_value = Mock(status_code=200, headers={}, json=Mock(return_value={
            'choices': [{'finish_reason': 'tool_calls', 'message': {'tool_calls': []}}]}))
        client = CloudClient(api_key='fixture-key')
        response = client.complete([{'role': 'user', 'content': 'test'}], [], tool_choice='required')
        payload = post.call_args.kwargs['json']
        self.assertEqual(payload['tool_choice'], 'required')
        self.assertEqual(payload['thinking'], {'type': 'disabled'})
        self.assertEqual(payload['response_format'], {'type': 'json_object'})
        self.assertEqual(response['diagnostics']['http_status'], 200)

    @patch('optical_agent.llm.requests.post')
    def test_auth_and_balance_codes_are_preserved_without_retry(self, post):
        for code, stage in ((401, 'authentication'), (403, 'authentication'), (402, 'account_balance')):
            with self.subTest(code=code):
                post.reset_mock()
                post.return_value = Mock(status_code=code, headers={})
                with self.assertRaises(LLMError) as caught:
                    CloudClient(api_key='fixture-key').complete([], [])
                self.assertEqual(post.call_count, 1)
                self.assertEqual(caught.exception.details['http_status'], code)
                self.assertEqual(caught.exception.details['stage'], stage)
                self.assertEqual(caught.exception.http_attempts, 1)
                self.assertIn(f'HTTP {code}', str(caught.exception))

    @patch('optical_agent.llm.time.sleep')
    @patch('optical_agent.llm.requests.post')
    def test_proxy_failure_has_no_http_code_and_safe_details(self, post, sleep):
        secret = 'fixture-private-key-123456789'
        post.side_effect = requests.exceptions.ProxyError('http://name:privatepass@127.0.0.1:9 Bearer ' + secret)
        with self.assertRaises(LLMError) as caught:
            CloudClient(api_key=secret).complete([], [])
        self.assertEqual(post.call_count, 2)
        self.assertEqual(caught.exception.http_attempts, 2)
        self.assertIsNone(caught.exception.details['http_status'])
        self.assertEqual(caught.exception.details['cause'], 'ProxyError')
        self.assertNotIn(secret, str(caught.exception.details))
        self.assertNotIn('privatepass', str(caught.exception.details))

    @patch('optical_agent.llm.requests.post')
    def test_http_200_decode_failure_is_not_an_auth_error(self, post):
        post.return_value = Mock(status_code=200, headers={}, json=Mock(side_effect=ValueError('bad JSON')))
        with self.assertRaises(LLMError) as caught:
            CloudClient(api_key='fixture-key').complete([], [])
        self.assertEqual(caught.exception.details['http_status'], 200)
        self.assertEqual(caught.exception.details['stage'], 'response_decode')
        self.assertEqual(post.call_count, 1)

    @patch('optical_agent.llm.requests.post')
    def test_request_id_is_redacted_before_truncation(self, post):
        secret = 'fixture-private-key-123456789'
        post.return_value = Mock(status_code=200, headers={'x-request-id': 'x' * 150 + secret},
            json=Mock(return_value={'choices': [{'message': {'content': '{}'}}]}))
        response = CloudClient(api_key=secret).complete([], [])
        self.assertNotIn('fixture-pr', response['diagnostics']['request_id'])
        self.assertIn('REDACTED', response['diagnostics']['request_id'])
