import asyncio
import hashlib
import importlib.util
import io
import json
import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from optical_agent.adaptive_goals import CLASSES, Goal
from optical_agent.adaptive_tools import AdaptiveState, execute, pixel_hash
from optical_agent.mcp_adapter import VisionMCPAdapter, UnavailableDetector, MAX_BYTES
from optical_agent.mcp_server import RedactedFormatter
from optical_agent.tools import ToolContext


class Detector:
    dataset, threshold, calibration = 'sodd', .35, None
    classes = ('background', *CLASSES)

    def __init__(self):
        self.calls, self.parallel, self.max_parallel = 0, 0, 0

    def predict(self, image):
        self.calls += 1
        self.parallel += 1
        self.max_parallel = max(self.parallel, self.max_parallel)
        time.sleep(.005)
        self.parallel -= 1
        return [{'class_name': 'pipe', 'class_id': 6, 'score': .8, 'box': [4, 4, 16, 16]},
                {'class_name': 'qr_codes', 'class_id': 5, 'score': .7, 'box': [20, 20, 32, 32]}]


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.detector = Detector()
        self.now = [0.]
        self.adapter = VisionMCPAdapter(lambda: self.detector, data_root=self.root,
            output_root=self.root / 'outputs', clock=lambda: self.now[0])
        self.image = self.picture('normal.png', 100)

    def picture(self, name, value):
        path = self.root / name
        Image.fromarray(np.full((64, 64, 3), value, np.uint8)).save(path)
        return str(path)

    def quality(self, path=None):
        result = self.adapter.call('assess_image_quality', {'image_path': path or self.image})
        self.assertTrue(result['ok'], result)
        return result

    def error(self, tool, args, category):
        result = self.adapter.call(tool, args)
        self.assertFalse(result['ok'], result)
        self.assertEqual(result['error']['category'], category)
        return result

    def test_direct_measurements_and_both_candidates_equal(self):
        q = self.quality(self.picture('dark.png', 35))
        ctx = next(iter(self.adapter.contexts.values()))
        direct = AdaptiveState(ToolContext(self.detector, ctx.state.images['original']))
        direct.begin(Goal('quality', 'original', CLASSES, 'never', False))
        measured = execute(direct, 'assess_image_quality', {'image_id': 'original'})
        self.assertEqual(q['data']['quality'], measured['data']['data']['quality'])
        for method in ('gamma_only', 'local_bounded'):
            args = {'image_id': q['data']['image_id'], 'method': method}
            generated = self.adapter.call('generate_exposure_candidate', args)
            self.assertTrue(generated['ok'], generated)
            direct.begin(Goal('correction', 'original', CLASSES, 'always', False))
            expected = execute(direct, 'generate_exposure_candidate', {'image_id': 'original', 'method': method})['data']['data']
            self.assertEqual(generated['data']['parameters'], expected['parameters'])
            self.assertEqual(generated['data']['pixels_sha256'], expected['pixels_sha256'])
            self.assertEqual(self.adapter.call('generate_exposure_candidate', args)['data']['image_id'], generated['data']['image_id'])
        self.assertEqual(ctx.state.counts['correction'], 2)
        self.assertEqual(len(ctx.state.candidates), 2)

    def test_normal_identity_original_readonly_and_cache(self):
        original = Path(self.image).read_bytes()
        q = self.quality()
        repeated = self.quality()
        self.assertEqual(q['data']['image_id'], repeated['data']['image_id'])
        self.assertTrue(repeated['cached'])
        args = {'image_id': q['data']['image_id'], 'method': 'gamma_only'}
        result = self.adapter.call('generate_exposure_candidate', args)
        self.assertFalse(result['data']['applied'])
        again = self.adapter.call('generate_exposure_candidate', args)
        self.assertTrue(again['cached'])
        ctx = next(iter(self.adapter.contexts.values()))
        self.assertEqual(ctx.state.counts['quality'], 1)
        self.assertEqual(ctx.state.counts['correction'], 1)
        self.assertFalse(ctx.state.images['original'].flags.writeable)
        self.assertEqual(Path(self.image).read_bytes(), original)
        self.error('generate_exposure_candidate', {'image_id': result['data']['image_id'], 'method': 'local_bounded'}, 'prerequisite')

    def test_filter_boxes_count_and_cached_all_class_predictions(self):
        oid = self.quality()['data']['image_id']
        all_boxes = self.adapter.call('detect_objects', {'image_id': oid})
        self.assertEqual(all_boxes['data']['count'], 2)
        pipes = self.adapter.call('detect_objects', {'image_id': oid, 'target_classes': ['管道']})
        self.assertEqual(pipes['data']['target_classes'], ['pipe_type2', 'pipe'])
        self.assertEqual(pipes['data']['count'], 1)
        self.assertEqual(pipes['data']['detections'], all_boxes['data']['detections'][:1])
        self.assertEqual(pipes['data']['status'], 'candidate_only')
        self.assertTrue(pipes['cached'])
        self.assertEqual(self.detector.calls, 1)
        self.assertTrue(Path(pipes['data']['annotated_image_path']).is_file())

    def test_unknown_and_parameter_failures(self):
        self.error('unknown', {}, 'unknown_tool')
        self.error('assess_image_quality', {}, 'parameters')
        self.error('assess_image_quality', {'image_path': self.image, 'image_id': 'foo'}, 'parameters')
        self.error('assess_image_quality', {'image_id': 'original'}, 'unknown_image')
        self.error('assess_image_quality', {'image_path': self.image, 'extra': True}, 'parameters')
        self.error('generate_exposure_candidate', {'image_id': 'foo', 'method': 'chain'}, 'parameters')

    def test_registration_is_not_a_quality_prerequisite(self):
        oid = self.adapter._register(self.image)
        self.error('generate_exposure_candidate', {'image_id': oid, 'method': 'gamma_only'}, 'prerequisite')

    def test_unsupported_alias_and_empty_targets(self):
        oid = self.quality()['data']['image_id']
        for targets in (['鱼'], ['AUV'], ['pipe', 'underwater_robot']):
            self.error('detect_objects', {'image_id': oid, 'target_classes': targets}, 'unsupported_class')
        self.error('detect_objects', {'image_id': oid, 'target_classes': []}, 'parameters')
        self.assertEqual(self.detector.calls, 0)

    def test_input_limits_and_damage(self):
        self.error('assess_image_quality', {'image_path': 'relative.png'}, 'invalid_path')
        outside = self.root.parent / (self.root.name + '_outside.png')
        Image.new('RGB', (16, 16)).save(outside)
        self.addCleanup(outside.unlink)
        self.error('assess_image_quality', {'image_path': str(outside)}, 'invalid_path')
        broken = self.root / 'broken.png'
        broken.write_bytes(b'not an image')
        self.error('assess_image_quality', {'image_path': str(broken)}, 'invalid_image')
        large = self.root / 'large.png'
        with large.open('wb') as f:
            f.truncate(MAX_BYTES + 1)
        self.error('assess_image_quality', {'image_path': str(large)}, 'size_limit')
        huge = self.root / 'huge.png'
        Image.new('1', (4001, 4000)).save(huge)
        self.error('assess_image_quality', {'image_path': str(huge)}, 'pixel_limit')

    def test_resolved_symlink_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as other:
            target = Path(other) / 'outside.png'
            Image.new('RGB', (16, 16)).save(target)
            link = self.root / 'link.png'
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest('OS does not permit creating a symlink')
            self.error('assess_image_quality', {'image_path': str(link)}, 'invalid_path')

    def test_black_white_and_binary_never_restored(self):
        for value in (0, 255):
            q = self.quality(self.picture(f'{value}.png', value))
            self.assertEqual(q['data']['quality_status'], 'quality_failure')
            self.error('generate_exposure_candidate', {'image_id': q['data']['image_id'], 'method': 'gamma_only'}, 'quality_failure')
        binary = np.zeros((64, 64, 3), np.uint8)
        binary[:, 32:] = 255
        path = self.root / 'binary.png'
        Image.fromarray(binary).save(path)
        q = self.quality(str(path))
        self.assertEqual(q['data']['quality_status'], 'quality_unassessable')
        self.error('generate_exposure_candidate', {'image_id': q['data']['image_id'], 'method': 'gamma_only'}, 'quality_failure')

    def test_empty_detection_and_model_missing_are_distinct(self):
        oid = self.quality()['data']['image_id']
        self.detector.predict = lambda image: []
        result = self.adapter.call('detect_objects', {'image_id': oid})
        self.assertTrue(result['ok'])
        self.assertEqual(result['data']['count'], 0)
        self.assertEqual(result['data']['status'], 'candidate_only')
        missing = VisionMCPAdapter(UnavailableDetector, data_root=self.root, output_root=self.root / 'missing')
        q = missing.call('assess_image_quality', {'image_path': self.image})
        result = missing.call('detect_objects', {'image_id': q['data']['image_id']})
        self.assertFalse(result['ok'])
        self.assertEqual(result['error']['category'], 'model_unavailable')
        self.assertNotIn('data', result)

    def test_wrong_model_taxonomy_is_unavailable_not_empty(self):
        self.detector.classes = ('background', 'not_a_sodd_class')
        oid = self.quality()['data']['image_id']
        self.error('detect_objects', {'image_id': oid}, 'model_unavailable')
        self.assertEqual(self.detector.calls, 0)

    def test_expiry_restart_and_capacity(self):
        oid = self.quality()['data']['image_id']
        self.now[0] = 1800
        self.error('assess_image_quality', {'image_id': oid}, 'unknown_image')
        new = self.quality()['data']['image_id']
        self.assertNotEqual(oid, new)
        restarted = VisionMCPAdapter(lambda: self.detector, data_root=self.root, output_root=self.root / 'restart')
        result = restarted.call('detect_objects', {'image_id': new})
        self.assertEqual(result['error']['category'], 'unknown_image')
        self.adapter.max_contexts = 1
        latest = self.quality(self.picture('other.png', 110))['data']['image_id']
        self.assertEqual(len(self.adapter.contexts), 1)
        self.error('assess_image_quality', {'image_id': new}, 'unknown_image')
        self.assertIn(latest, self.adapter.index)

    def test_active_context_protected_during_expiry_and_eviction(self):
        oid = self.quality()['data']['image_id']
        self.adapter.max_contexts = 1
        with self.adapter._lease(oid):
            self.now[0] = 2000
            self.error('assess_image_quality', {'image_path': self.picture('other.png', 110)}, 'capacity')
            self.assertIn(oid, self.adapter.index)
        self.assertTrue(self.adapter.call('assess_image_quality', {'image_id': oid})['ok'])

    def test_concurrent_repeated_registration_and_gpu_serial(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: self.adapter.call('assess_image_quality', {'image_path': self.image}), range(8)))
        self.assertTrue(all(r['ok'] for r in results))
        self.assertEqual(len({r['data']['image_id'] for r in results}), 1)
        ctx = next(iter(self.adapter.contexts.values()))
        self.assertEqual(ctx.state.counts['quality'], 1)
        first = results[0]['data']['image_id']
        second = self.quality(self.picture('other.png', 105))['data']['image_id']
        with ThreadPoolExecutor(max_workers=4) as pool:
            detections = list(pool.map(lambda oid: self.adapter.call('detect_objects', {'image_id': oid}), [first, second, first, second]))
        self.assertTrue(all(r['ok'] for r in detections))
        self.assertEqual(self.detector.calls, 2)
        self.assertEqual(self.detector.max_parallel, 1)
        self.assertTrue(all(c.active == 0 for c in self.adapter.contexts.values()))

    def test_computation_and_storage_failures_release_lock_and_hide_secrets(self):
        oid = self.quality()['data']['image_id']
        secret = 'sk-test-mcp-secret-0123456789'
        def broken(_):
            raise RuntimeError(secret)
        self.detector.predict = broken
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': secret}):
            result = self.adapter.call('detect_objects', {'image_id': oid})
            self.assertFalse(result['ok'])
            self.assertEqual(result['error']['category'], 'computation')
            self.assertNotIn(secret, json.dumps(result))
        ctx = next(iter(self.adapter.contexts.values()))
        self.assertEqual(ctx.active, 0)
        self.assertTrue(ctx.lock.acquire(blocking=False))
        ctx.lock.release()
        with patch.object(self.adapter, '_save', side_effect=OSError(secret)):
            self.error('assess_image_quality', {'image_id': oid}, 'storage')
        self.assertTrue(self.adapter.call('assess_image_quality', {'image_id': oid})['ok'])

    def test_log_formatter_redacts_exception_and_message(self):
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(RedactedFormatter('%(message)s'))
        log = logging.getLogger('mcp_secret_test')
        log.addHandler(handler)
        self.addCleanup(log.removeHandler, handler)
        secret = 'sk-mcp-secret-012345678901'
        with patch.dict(os.environ, {'DEEPSEEK_API_KEY': secret}):
            try:
                raise ValueError(secret)
            except ValueError:
                log.error('Bearer %s', secret, exc_info=True)
        self.assertNotIn(secret, stream.getvalue())
        self.assertIn('[REDACTED]', stream.getvalue())


@unittest.skipUnless(importlib.util.find_spec('mcp'), 'Optional MCP dependency not installed')
class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / 'normal.png'
        Image.new('RGB', (64, 64), (100, 100, 100)).save(self.image)
        self.server_script = Path(__file__).parent / 'mcp_fixture_server.py'

    def test_standard_client_real_subprocess_discovery_errors_and_cache(self):
        async def check():
            from mcp import Client
            from mcp.client.stdio import StdioServerParameters
            params = StdioServerParameters(command=sys.executable, args=['-u', '-B', str(self.server_script), str(self.root)])
            async with Client(params, mode='legacy', cache=None, read_timeout_seconds=30) as client:
                names = [t.name for t in (await client.list_tools()).tools]
                self.assertEqual(set(names), {'assess_image_quality', 'generate_exposure_candidate', 'detect_objects'})
                self.assertEqual(len(names), 3)
                q = (await client.call_tool('assess_image_quality', {'image_path': str(self.image)})).structured_content
                oid = q['data']['image_id']
                self.assertTrue(oid.startswith('img_'))
                q2 = (await client.call_tool('assess_image_quality', {'image_id': oid})).structured_content
                self.assertTrue(q2['cached'])
                d = (await client.call_tool('detect_objects', {'image_id': oid, 'target_classes': ['管道']})).structured_content
                self.assertEqual(d['data']['count'], 1)
                for name, args, category in [
                    ('assess_image_quality', {'image_id': 'unknown'}, 'unknown_image'),
                    ('assess_image_quality', {'image_path': str(self.image), 'image_id': oid}, 'parameters'),
                    ('assess_image_quality', {'image_id': oid, 'extra': True}, 'parameters'),
                    ('detect_objects', {'image_id': oid, 'target_classes': ['鱼']}, 'unsupported_class'),
                    ('detect_objects', {'image_id': oid, 'target_classes': 42}, 'parameters'),
                    ('not_registered', {}, 'unknown_tool')]:
                    result = await client.call_tool(name, args)
                    self.assertTrue(result.is_error, result)
                    self.assertEqual(result.structured_content['error']['category'], category)
        asyncio.run(check())

    def test_raw_stdout_contains_only_jsonrpc_even_with_library_print(self):
        # Inspect the bytes as well as the SDK client: a stray print must not hide
        # behind the client's parser/error handling.
        env = dict(os.environ, PYTHONIOENCODING='utf-8', DEEPSEEK_API_KEY='sk-protocol-test-0123456789')
        proc = subprocess.Popen([sys.executable, '-u', '-B', str(self.server_script), str(self.root)],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding='utf-8', env=env)
        lines, received = [], queue.Queue()
        def reader():
            for line in proc.stdout:
                lines.append(line)
                received.put(line)
        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        def request(request_id, method, params):
            proc.stdin.write(json.dumps({'jsonrpc': '2.0', 'id': request_id, 'method': method, 'params': params}) + '\n')
            proc.stdin.flush()
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                msg = json.loads(received.get(timeout=20))
                if msg.get('id') == request_id:
                    return msg
            self.fail('Missing protocol response')
        try:
            initialized = request(1, 'initialize', {'protocolVersion': '2025-11-25', 'capabilities': {},
                                                   'clientInfo': {'name': 'wire-test', 'version': '1'}})
            self.assertIn('result', initialized)
            proc.stdin.write('{"jsonrpc":"2.0","method":"notifications/initialized"}\n')
            proc.stdin.flush()
            listed = request(2, 'tools/list', {})
            self.assertEqual(len(listed['result']['tools']), 3)
            q = request(3, 'tools/call', {'name': 'assess_image_quality', 'arguments': {'image_path': str(self.image)}})
            oid = q['result']['structuredContent']['data']['image_id']
            detected = request(4, 'tools/call', {'name': 'detect_objects', 'arguments': {'image_id': oid}})
            self.assertEqual(detected['result']['structuredContent']['data']['count'], 2)
        finally:
            proc.stdin.close()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
            thread.join(timeout=5)
            stderr = proc.stderr.read()
            proc.stdout.close()
            proc.stderr.close()
        self.assertEqual(proc.returncode, 0, stderr)
        self.assertIn('FIXTURE_PRINT_MUST_STAY_OFF_PROTOCOL', stderr)
        for line in lines:
            self.assertEqual(json.loads(line)['jsonrpc'], '2.0')
        self.assertNotIn('FIXTURE_PRINT', ''.join(lines))
        self.assertNotIn('sk-protocol-test-0123456789', stderr)


if __name__ == '__main__':
    unittest.main()
